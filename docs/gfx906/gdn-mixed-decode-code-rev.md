# Code review — `gfx906/gdn-mixed-decode` (W1) vs `gfx906/main`

> Branch: `gfx906/gdn-mixed-decode` (off `gfx906/main` @ `67ae6c3f96`)
> Base head: `a0a4334baa gdn: peel non-spec decodes in spec-mixed batches (W1)`
> Diff: 9 commits, 14 files, **+1098 / −80 lines**
> Model under change: hybrid GDN/attention (Qwen3.5/3.6/3.8-AWQ-INT4, 27B gate)
> Reviewers: human + GPT-5 code-review pass on the same diff
> Date: 2026-08-26

This file folds the human review with the GPT review (`/tmp/gdn-mixed-decod-code-rev-gpt.md`),
validating or rejecting each GPT claim against the actual source and tests, and writing
the merged action items.

---

## TL;DR

| # | Severity | Title | Source |
|---|---|---|---|
| 1 | **P1** | CPU GDN backend (`vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`) is now broken for spec + non-spec decode + prefill batches | **GPT** (validated) |
| 2 | **P1** | Kimi-K3 AMD path uses `chunk_indices` from `m.prefill_query_start_loc` (now prefill-only after W1) but passes `cu_seqlens=non_spec_query_start_loc` (full non-spec) — inconsistent cu_seqlens / chunk metadata | **Human** (new finding from validating GPT) |
| 3 | **P2** | 12 ruff lint errors in the new probe / harness files (8× E501 line-too-long, 1× F401 unused `torch`, 1× F841 unused `mask`) | **GPT** (validated) |
| 4 | **P2** | `probe_ngram_cpu_engine.py` mixes the untimed warmup `llm.generate` into `nprop` / `acc/step` / `_bookkeep_times`, so the printed step count, acceptance rate, and bookkeeping top-list are invalid | **GPT** (validated) |
| 5 | **P2** | W1 changes a shared metadata contract (prefill-only metadata fields) but updates only the Qwen GPU path; the change is not gated or versioned | **Human + GPT** (consolidated) |
| 6 | **P3** | Devlog does not follow `docs/gfx906/AGENTS.md` template (`## HYPOTHESIS`, `VERDICT:` header, dated headings) | **Human** (carried over) |
| 7 | **P3** | Scope creep: three L3 probe scripts (`bench_ngram_cpu.py`, `compare_ngram_cpu_gpu.py`, `probe_ngram_cpu_engine.py`) belong on `gfx906/ngram-cpu-d2h`, not on W1 | **Human** (carried over) |
| 8 | **P3** | Harness env vars (`BENCH_MIXED`, `BENCH_NREQS`, `BENCH_SPEC_CONFIG`, `BENCH_CG_MAX`) are undocumented outside the harness source | **Human** (carried over) |
| 9 | **P3** | `non_spec_state_indices_tensor` length / V1-decode-first invariant is implicit, not asserted | **Human** (carried over) |
| 10 | **P3** | Upstream PR #53077 (`num_spec_decodes = 0` reset for empty draft schedule) is not in this branch and not noted | **Human** (carried over) |

**Headline numbers from W1's gates (unchanged by this review):**
- Kernel-spy on 9B probe: 2112 → 96 chunk calls on the `[1 decode + 1 spec]` mixed step (the 2116 removed = 96 real-prefill + 2016 wasted 1-token "prefill" runs).
- 27B mixed 2-req ngram serving A/B: **59.35 vs 55.60 t/s = +6.7 %** (4 samples/arm, bands ±0.3 %).
- Request A (always-spec, untouched path) token-identical across 6 runs, both arms.
- 10/10 metadata tests pass on the branch.

The reviewer's view of the W1 change itself is unchanged: the Qwen ROCm/GPU dispatch is
internally coherent, and the +6.7 % result is real. **The blockers from the review are
around (a) CPU/XPU backend breakage introduced by the metadata-contract change, and (b)
the new probe / harness files being below the repo's stated code-quality bar.**

---

## 1. [P1] CPU GDN backend is broken for spec + decode + prefill batches

### GPT claim (validated)

> In `_cpu_gdn_attention_spec_aware`, it still constructs `mixed_qkv_ns` from **all**
> `non_spec_token_indx` entries (including the peeled decode rows), then calls
> `_spec_aware_nonspec_subset`. That helper uses `prefill_query_start_loc` and
> `prefill_state_indices`, which now cover only the prefill rows.

### Verification

W1 changed the metadata builder so that, when a spec-mixed batch contains non-spec
1-token decodes:

```python
# gdn_attn.py — what the metadata builder now emits for a
# batch ordered as [decode (1 tok), prefill (50 tok), spec (3 tok)]:
num_decodes           = 1
num_prefills          = 1
non_spec_token_indx   = [51 entries]   # 1 decode + 50 prefill tokens
prefill_query_start_loc   = [0, 50]    # ONLY prefill (50 tokens)
prefill_state_indices     = shape (1,) # ONLY prefill row
has_initial_state         = shape (2,) # decode + prefill (full non-spec)
```

The CPU path's caller (lines 350-389 of `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`)
on the W1 branch is unchanged from `gfx906/main`:

```python
# UNCHANGED on W1 — still assumes the pre-W1 contract:
if (num_prefills > 0 or num_decodes > 0) and non_spec_token_indx is not None:
    mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)  # 51 rows
    b_ns = b.index_select(0, non_spec_token_indx)
    a_ns = a.index_select(0, non_spec_token_indx)
    nonspec_out = _spec_aware_nonspec_subset(
        layer, attn_metadata_i, mixed_qkv_ns, b_ns, a_ns, conv_buf, ssm_state, width
    )

# Scatter outputs back:
if nonspec_out is not None:
    core_attn_out.index_copy_(0, non_spec_token_indx, nonspec_out)
```

`_spec_aware_nonspec_subset` (lines 643-714 of the same file) feeds the
pre-W1-contract `prefill_query_start_loc`/`prefill_state_indices` to
`causal_conv1d_fwd_cpu` / `chunk_gated_delta_rule_cpu`:

```python
# Inside _spec_aware_nonspec_subset, unchanged:
conv_out = ops.causal_conv1d_fwd_cpu(
    x=mixed_qkv.transpose(0, 1),                  # (qkv_dim, 51)  — still 51 rows
    query_start_loc=prefill_qsl,                  # [0, 50]        — prefill only
    cache_indices=prefill_state_indices,          # shape (1,)     — prefill only
    has_initial_state=has_initial_state,          # shape (2,)     — full non-spec
    ...
)
...
attn_out, _ = ops.chunk_gated_delta_rule_cpu(
    ...,
    cu_seqlens=prefill_qsl,                       # [0, 50]        — prefill only
    initial_state_indices=prefill_state_indices,  # shape (1,)     — prefill only
)
return attn_out.squeeze(0)                        # shape (50, …)
```

### Failure modes (verified empirically)

I verified the failure mode with a quick sanity test:

```python
>>> out = torch.zeros(100)
>>> idx = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=torch.long)  # 11
>>> src = torch.tensor([100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.])  # 10
>>> out.index_copy_(0, idx, src)
IndexError: index_copy_(): Number of indices (11) should be equal to source.size(dim) (10)
```

In our scenario:
- `non_spec_token_indx` has 51 entries (1 decode + 50 prefill).
- The CPU helper, fed 51 rows but using `prefill_qsl=[0, 50]`, returns 50 rows of output.
- The caller's `index_copy_(0, idx_51, src_50)` raises `IndexError` at the **scatter step**
  (`vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py:388`).

The conv kernel may also fail or produce wrong output upstream of the scatter, depending
on how it interprets the `query_start_loc=[0, 50]` over a `(qkv_dim, 51)` input. I did
not run the C++ conv kernel; the index_copy error is sufficient to confirm the bug.

### Why the existing CPU tests don't catch it

`tests/kernels/mamba/cpu/test_cpu_gdn_ops.py` has two tests that exercise
`_cpu_gdn_attention_spec_aware`:

1. `test_spec_aware_mixed_routing_preserves_token_order` — uses
   `num_prefills=1, num_decodes=0`. The `_spec_aware_nonspec_subset` call is
   monkey-patched out.
2. `test_spec_aware_nonspec_materializes_state_indices` — uses
   `num_prefills=2, num_decodes=0`. The C++ kernels are monkey-patched out.

Neither test runs a `[decode, prefill, spec]` shape end-to-end through the real
helpers, so neither trips the size mismatch. Both still pass on the W1 branch
(verified locally — 10/10 metadata tests, plus the two CPU tests, are unaffected).

### Scope of impact

- The W1 branch is **gfx906 (AMD GPU)**, so the CPU path is **not exercised by the
  branch's own serving gate** (the +6.7 % number is GPU-only). The bug does not invalidate
  the W1 serving A/B.
- However, the CPU code is in the upstream-style tree (`vllm/model_executor/layers/mamba/ops/cpu/`)
  and ships on every release. **CPU GDN serving with ngram spec-decode + concurrent decode/prefill
  is broken by W1.** Anyone running a CPU GDN deployment (server, CI matrix, laptop) hits this.
- An XPU implementation may share the same assumption (GPT flagged it; I verified Kimi-K3 AMD
  is independently broken — see §2). I did not find a separate XPU GDN implementation in this
  tree.

### Recommended fix

The cleanest fix mirrors what `_forward_core` does on the GPU:

```python
def _spec_aware_nonspec_subset(
    layer, attn_metadata_i, mixed_qkv, b, a, conv_buf, ssm_state, width,
):
    # NEW: peel the 1-token non-spec decodes (only present when
    # spec_sequence_masks is not None — V1 decode-first invariant).
    num_decodes = attn_metadata_i.num_decodes
    non_spec_token_indx = attn_metadata_i.non_spec_token_indx
    has_initial_state_full = attn_metadata_i.has_initial_state
    decode_state_indices = attn_metadata_i.non_spec_state_indices_tensor

    decode_out = None
    if num_decodes > 0:
        nd = num_decodes
        # 1-token recurrent update, same as _forward_core 2.2 / 2.3-elif.
        decode_qkv = mixed_qkv[:nd]  # decode-first slice (V1 invariant)
        decode_b = b[:nd]
        decode_a = a[:nd]
        ...
        decode_out, _ = recurrent_gated_delta_rule_cpu(
            q=decode_qkv[..., q_slice], k=..., v=...,
            cu_seqlens=torch.arange(nd + 1, dtype=torch.int32),
            ssm_state_indices=decode_state_indices[:nd],
            initial_state=ssm_state, output_final_state=True, ...,
        )
        mixed_qkv = mixed_qkv[nd:]
        b = b[nd:]
        a = a[nd:]
        # has_initial_state and the decode rows are no longer needed.

    # Existing prefill path, unchanged:
    prefill_state_indices = attn_metadata_i.prefill_state_indices
    prefill_qsl = attn_metadata_i.prefill_query_start_loc
    prefill_has_initial_state = (
        has_initial_state_full[num_decodes:] if has_initial_state_full is not None
        else None
    )
    prefill_out = _spec_aware_nonspec_subset_prefill_only(
        layer, mixed_qkv, b, a, conv_buf, ssm_state,
        prefill_qsl, prefill_state_indices, prefill_has_initial_state, width,
    )

    if decode_out is None:
        return prefill_out
    return torch.cat([decode_out, prefill_out], dim=0)  # decode-first order
```

(If `decode_out` and `prefill_out` end up needing different code paths, factor the existing
helper body into `_spec_aware_nonspec_subset_prefill_only` and add the recurrent-update branch
for the decode rows.) The scatter at line 388 then matches: `non_spec_token_indx` has
`num_decode_tokens + num_prefill_tokens` entries and `nonspec_out` has the same.

Add an integration test (mirror `test_spec_aware_mixed_routing_preserves_token_order` but
with `num_decodes=1, num_prefills=1, num_spec_decodes=1` and the actual kernels unpatched).

**Alternative (cheaper, blocks regression only):** in the W1 metadata builder, when
`spec_sequence_masks is not None`, restore the reclass block but route the W1 win via a
new metadata field (`peeled_decode_*`) consumed only by the GPU `_forward_core`. The CPU
backend keeps the old contract; the GPU keeps the new one. Less performant for the CPU
backend (still pays the chunk tax on 1-token decodes) but no correctness regression.

---

## 2. [P1] Kimi-K3 AMD path has a parallel cu_seqlens / chunk-metadata mismatch

### Finding

While validating GPT's "CPU/XPU also broken" claim, I traced the Kimi-K3 AMD path
(`vllm/models/kimi_k3/amd/kda.py`, line 469+):

```python
split_non_spec = spec_sequence_masks is None and m.num_decodes > 0
if split_non_spec:
    # ... peeled decode path ...
    prefill_query_start_loc = m.prefill_query_start_loc           # prefill-only (after W1)
    prefill_state_indices   = m.prefill_state_indices             # prefill-only (after W1)
    prefill_has_initial_state = m.prefill_has_initial_state       # prefill-only (after W1)
else:
    prefill_query_start_loc = non_spec_query_start_loc            # full non-spec
    prefill_state_indices   = non_spec_state_indices_tensor       # full non-spec
    prefill_has_initial_state = has_initial_state                 # full non-spec

...
(
    core_attn_out_non_spec,
    last_recurrent_state,
) = chunk_kda_with_fused_gate(
    ...,
    cu_seqlens=prefill_query_start_loc,    # LOCAL variable
    chunk_indices=m.chunk_indices,         # ALWAYS m.chunk_indices (W1: prefill-only)
    chunk_offsets=m.chunk_offsets,         # ALWAYS m.chunk_offsets (W1: prefill-only)
)
```

When `spec_sequence_masks is not None and m.num_decodes > 0 and m.num_prefills > 0`:

- The local `prefill_query_start_loc` is set to **`non_spec_query_start_loc` (full non-spec)**.
- But `chunk_indices`/`chunk_offsets` are read from `m.chunk_indices`/`m.chunk_offsets`,
  which are built (by the W1 metadata builder) from **`m.prefill_query_start_loc` (prefill-only)**.

So the cu_seqlens passed to `chunk_kda_with_fused_gate` covers the full non-spec token
sequence, but the chunk indexing tables (chunk_indices / chunk_offsets) are derived from
a prefill-only cu_seqlens. The kernel will read out-of-range chunk indices for the decode
rows or simply produce wrong chunks. This is independent of the CPU backend bug.

### When does it fire?

`spec_sequence_masks is not None and num_decodes > 0 and num_prefills > 0` on the Kimi
KDA backend. Kimi K3 is GDN-only and does not currently ship with spec-decode
(spec_decode support appears absent in `vllm/models/kimi_k3/nvidia/kda_metadata.py` —
no `num_spec_decodes` references), so **this code path is dead today**. It becomes a
latent bug the first time spec-decode is enabled on Kimi K3 — at which point the
W1 contract change bites.

### Recommended fix

Either:

(a) Refactor Kimi K3 to use the same metadata fields (`m.prefill_query_start_loc` etc.)
and split conditionally on `m.num_decodes > 0` regardless of `spec_sequence_masks` —
mirroring the GPU `_forward_core` shape.

(b) Add an explicit `m.spec_sequence_masks is None and m.num_decodes > 0` guard around
the `chunk_indices=m.chunk_indices` calls, forcing the no-spec-only local path so the
cu_seqlens / chunk indices match.

I would lean (a), since Kimi K3 already has the peel logic conditional on the no-spec case
and extending it is straightforward — but this should be filed as a separate change, not
grafted into W1.

---

## 3. [P2] 12 ruff errors in the new probe / harness files

### GPT claim (validated)

```
.venv/bin/python -m ruff check \
    benchmarks/kernels/gfx906/compare_ngram_cpu_gpu.py \
    benchmarks/kernels/gfx906/probe_gdn_mixed.py \
    benchmarks/kernels/gfx906/probe_ngram_cpu_engine.py \
    docs/gfx906/_bench_gfx906.py
…
Found 12 errors.
```

Reproduced locally:

```
E501 Line too long (93 > 88)  compare_ngram_cpu_gpu.py:9   (Usage docstring line)
F841 Local variable `mask` is assigned to but never used   compare_ngram_cpu_gpu.py:97
F401 `torch` imported but unused                            probe_gdn_mixed.py:31
E501 Line too long (90 > 88)  probe_gdn_mixed.py:198        (pool docstring)
E501 Line too long (94 > 88)  probe_ngram_cpu_engine.py:15  (Usage docstring line)
E501 Line too long (90 > 88)  probe_ngram_cpu_engine.py:108
E501 Line too long (91 > 88)  probe_ngram_cpu_engine.py:110
E501 Line too long (98 > 88)  probe_ngram_cpu_engine.py:141
E501 Line too long (98 > 88)  probe_ngram_cpu_engine.py:144
E501 Line too long (94 > 88)  docs/gfx906/_bench_gfx906.py:136  (pool docstring)
```

All 12 are in new probe/harness files; the production code (`gdn_attn.py`,
`qwen_gdn_linear_attn.py`) and the metadata test are clean. Per
`docs/gfx906/AGENTS.md` and the project lint policy, the 88-column rule applies to
all files in the repo. The probe / harness files are evidence in the W1 devlog, so they
should at minimum be in the same shape as the code under test.

`compare_ngram_cpu_gpu.py:97` also has an `F841` (unused `mask = (n_tmp >= min_n)`) that
predates W1 (it was inherited from the L3 branch via commit `8ac610f8a0`, which is an
in-tree copy of `18c235772d`). Worth fixing while in the area.

### Recommended fix

A single `ruff check --fix` pass cleans 10 of the 12 (the line-length fixes are auto-applied
via the project's `pyproject.toml`); the F401 and F841 need manual deletion of one line each.
Five-minute job.

---

## 4. [P2] `probe_ngram_cpu_engine.py` mixes warmup into the acceptance-rate metric

### GPT claim (validated)

> `_propose_times` is populated before the measured generation: the `NgramProposer`
> constructor calls `propose` for JIT warmup, and the script then performs a separate
> untimed `llm.generate` warmup. The final `nprop` calculation sums every recorded call,
> but `len(out_toks)` contains only tokens from the timed generation. … Reset or isolate
> the timing/count buffers immediately before the measured `generate` call, and count
> actual decode steps independently.

### Verification

`vllm/v1/spec_decode/ngram_proposer.py:62-64` — the constructor calls
`self.propose(...)` once for Numba JIT compilation. This is invoked when `LLM(...)`
constructs the proposer during model load.

The probe script (lines 78-79) does an explicit untimed warmup:
```python
llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0.0), use_tqdm=False)
```
This drives the proposer (and the bookkeeping) and so pollutes `_propose_times`
and `_bookkeep_times` before the timed `llm.generate(...)` on line 116.

The summary prints (line 120-122):
```python
nprop = sum(len(v) for v in _propose_times.values())
acc = (len(out_toks) - nprop) / max(1, nprop)
```
`nprop` includes the JIT warmup + the untimed warmup + the timed run, while `len(out_toks)`
is only the timed run. The acceptance-rate number is therefore meaningless.

Same applies to `_bookkeep_times`: the `bookkeep_top` (line 128-129) list includes the
untimed-warmup steps and may even be dominated by them.

### Recommended fix

```python
# Immediately before the timed call:
_propose_times.clear()
_bookkeep_times.clear()

t0 = time.perf_counter()
outs = llm.generate(...)
dt = time.perf_counter() - t0
...
```

And the acceptance-rate formula should be `acc = (len(out_toks) - nprop) / max(1, nprop)`
only if each step issues exactly one `propose` call — which it does for a single-request
batch — but `nprop` must count only the timed steps. With the clear() above, `nprop`
becomes the actual count of timed `propose` calls and the formula is correct.

This script is in scope for "scope creep" (see §7) — it does not belong on this branch —
but the bug is real and should be fixed in whatever branch it lands on.

---

## 5. [P2] W1 changes a shared metadata contract without versioning it

### GPT claim (consolidated)

> The metadata contract was changed globally while at least the CPU GDN implementation
> still assumes the old contract.

### Validation

This is GPT's [P1] rephrased as a contract-level observation. I agree the framing is
correct, but I would land it as a separate finding because it is a *process* issue, not
a code issue per se:

- W1 mutates `prefill_query_start_loc`, `prefill_state_indices`, `prefill_has_initial_state`
  semantics in `GDNAttentionMetadata`. These fields are part of the public surface
  of `gdn_attn.py` and are consumed by:
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (Qwen ROCm, updated ✓)
  - `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py` (CPU, NOT updated — §1)
  - `vllm/models/kimi_k3/amd/kda.py` (Kimi K3 AMD, latently broken — §2)
  - The custom op wiring in `vllm/_xpu_ops.py` (needs audit)

- W1 has no `vLLM_USE_GDN_DECODE_PEEL=…` toggle, no version field on
  `GDNAttentionMetadata`, and no per-backend dispatch table. The Qwen ROCm update is
  silent: it expects the new contract and silently misuses the old one if a stale
  metadata builder is loaded.

- The metadata test (`tests/v1/attention/test_gdn_metadata_builder.py`) was extended to
  cover the new contract. There is no equivalent test for the CPU `_spec_aware_nonspec_subset`
  path that exercises the contract change end-to-end (see §1 "Why the existing CPU tests
  don't catch it").

### Recommended fix

Short-term (W1 ship-blocker): the §1 CPU fix + a CPU integration test.

Medium-term (gfx906/main hardening): introduce a `GDNAttentionMetadata.schema_version`
or, better, a `GDNBackend.decode_peel_supported: bool` on the backend base class, so the
metadata builder can refuse to emit the new contract to backends that have not opted in.
This is the standard fix in vLLM for cross-backend contract changes — see e.g. the
`Mamba2AttentionMetadataBuilder` history.

---

## 6. [P3] Devlog does not follow `docs/gfx906/AGENTS.md` template

### Status

Carried over from the human review. The W1 devlog uses a prose-style format that is the
*de-facto* convention in this tree (see `DEVLOG-tp2-dense.md`, `DEVLOG-spec-decode.md`),
but it does not match the strict template in `docs/gfx906/AGENTS.md` and
`docs/gfx906/_devlog-template.md`:

- No `## HYPOTHESIS` falsifiable one-liner.
- No `**VERDICT:** … · **GATE:** …` header (only `**Status: …**` then a separate `GATE:` line).
- No `## YYYY-MM-DD` dated session headings.
- `## Verdict` heading is at the bottom, not preceding details.

### Recommended fix

Bring the W1 devlog into template shape before any merge into `gfx906/main`. See the
`_devlog-template.md` for the exact skeleton.

---

## 7. [P3] Scope creep: L3 probe scripts committed to W1

### Status

Carried over from the human review. Commit `8ac610f8a0 gfx906/ngram: CPU proposer cost
probes + GPU/CPU draft A/B + harness spec envs` adds three scripts
(`bench_ngram_cpu.py`, `compare_ngram_cpu_gpu.py`, `probe_ngram_cpu_engine.py`) that
target the **L3 (CPU ngram proposer D2H serialization)** investigation, not W1. They are
not referenced by the W1 devlog, the W1 probe scripts (`probe_gdn_mixed.py`), or the W1
bench harness (`_bench_gfx906.py` `BENCH_MIXED` env var). They were in-tree on
`gfx906/ngram-cpu-d2h` as commit `18c235772d` (the L3 branch) — `8ac610f8a0` is a verbatim
copy of `18c235772d` (same diff stats, same author/date metadata) and should not be in
this branch.

### Recommended fix

Drop `8ac610f8a0` from this branch. The L3 scripts live on `gfx906/ngram-cpu-d2h`
where they are actively used; copying them here adds 379 net lines of unrelated code
*and* the `compare_ngram_cpu_gpu.py` / `probe_ngram_cpu_engine.py` files violate the
88-column rule (§3) and have the broken acceptance-rate metric (§4). Removing the
commit resolves two of the three P2s in this review.

---

## 8. [P3] Harness env vars undocumented outside `_bench_gfx906.py`

### Status

Carried over from the human review. `BENCH_MIXED`, `BENCH_NREQS`, `BENCH_SPEC_CONFIG`,
`BENCH_CG_MAX` are documented only in inline comments inside
`docs/gfx906/_bench_gfx906.py`. A reader following the W1 devlog recipe
(`BENCH_NREQS=2 BENCH_MIXED=1 BENCH_SPEC_CONFIG='…'`) needs to read the harness source
to discover the semantics.

### Recommended fix

Add a one-paragraph section to `docs/gfx906/running.md` (or wherever the gfx906 bench
recipes live) covering the env-var surface introduced by W1. Reference the devlog and
the probe scripts from there.

---

## 9. [P3] V1 decode-first invariant is implicit in `_forward_core`

### Status

Carried over from the human review. The W1 code slices `non_spec_state_indices_tensor[:num_decode_tokens]`
(line 1406) and `mixed_qkv_non_spec[:num_decode_tokens]` (line 1518), which are correct
**only because the V1 scheduler places 1-token decodes first**. That invariant is
elsewhere in the vLLM codebase but is not asserted or commented in the GDN layer.

### Recommended fix

Add one assertion + one comment:

```python
# V1 invariant: decode-first ordering inside non_spec rows.
# non_spec_state_indices_tensor[:num_decode_tokens] selects decode rows.
assert num_decode_tokens == num_decodes
assert (non_spec_query_start_loc[:num_decodes + 1]
        == torch.arange(num_decodes + 1, dtype=torch.int32,
                        device=non_spec_query_start_loc.device)).all(), (
    "GDN peel assumes V1 decode-first; non_spec_query_start_loc[:num_decodes+1] "
    "is not a 0..num_decodes ramp"
)
```

Cheap, and converts a latent silent-corruption bug into a hard failure.

---

## 10. [P3] Upstream PR #53077 not pulled into W1

### Status

Carried over from the human review. Upstream commit `6df7adc17f` (PR #53077, 2026-08-20)
adds `num_spec_decodes = 0` in the empty-draft-schedule early-exit. Neither
`gfx906/main` nor `gfx906/gdn-mixed-decode` contains it.

The interaction with W1 is non-trivial: W1's `num_decodes > 0` peels non-spec decodes
under the assumption that `num_spec_decodes` correctly tracks the count. If a schedule
flips to empty-draft mid-step, the pre-PR-#53077 code can carry a stale
`num_spec_decodes > 0` into W1's new dispatch, where it does not currently mishandle it
(verified by reading the metadata builder) but where it *interacts* with the new
`prefill_query_start_loc` rebasing in a way that the existing tests don't cover.

### Recommended fix

Either pull PR #53077 forward onto `gfx906/main` first (then rebase W1), or note the
known gap in the W1 devlog with a one-line cross-link. The latter is enough for the
W1 branch itself; the former is needed before any merge into `gfx906/main`.

---

## Findings the human review listed that GPT did **not** raise (carried forward)

These are in the prior review and not duplicated by GPT. Kept here so the merged action
list is complete:

- **Dead comment trail.** `qwen_gdn_linear_attn.py:1413-1422` and `gdn_attn.py:367-370`
  describe the OLD reclass behavior in present tense. Trim to a one-liner.
- **Test parameter case reshuffling.** `spec_decode_with_real_prefill` and
  `zero_length_padding_with_spec` were reordered to encode V1 decode-first ordering.
  Values are otherwise unchanged. `git blame` traces it to W1, which is correct, but any
  external test-ID pinning needs to re-pin.
- **`causal_conv1d_fn` in mixed prefill+decode spec-mixed batches** uses the FULL
  non-spec `has_initial_state` and `non_spec_state_indices_tensor`. This is the same
  semantics as the no-spec mixed case and is correct, but it is the first time this
  metadata shape has been exercised in production with peeled decodes; worth a small
  test or an explicit comment that the full-vector semantics are intentional.

---

## Findings GPT listed that the human review did **not** raise (new)

- The CPU backend breakage (§1) — material, validated, fixes on the table.
- The Kimi-K3 latent breakage (§2) — material but untriggered today; recommend separate
  change.
- 12 ruff errors in probe / harness files (§3) — easy fix, validated.
- The `probe_ngram_cpu_engine.py` acceptance-rate metric bug (§4) — easy fix, validated.
- The metadata-contract change is unversioned (§5) — process framing of the same root
  cause as §1+§2.

---

## Action items, ranked by impact

1. **[P1] Fix the CPU GDN backend to handle the spec + decode + prefill shape.** Either
   add a decode peel to `_spec_aware_nonspec_subset` or restore the old metadata
   contract via a separate field. Add a CPU integration test that exercises the
   `[decode, prefill, spec]` shape end-to-end. **Without this, W1 breaks CPU GDN
   serving.**
2. **[P1] File a separate Kimi-K3 fix** for the latent cu_seqlens / chunk_indices
   mismatch (§2). Kimi K3 doesn't ship spec-decode today, so this can land after W1.
3. **[P2] Drop commit `8ac610f8a0`** (the three L3 probe scripts) from this branch —
   they belong on `gfx906/ngram-cpu-d2h` (where commit `18c235772d` already holds them).
   This also resolves §3 (4 of 12 lint errors live in those files) and §4 (the broken
   acceptance metric).
4. **[P2] Run `ruff check --fix` on the remaining files** and clean the F401 / F841 by hand.
5. **[P2] Fix the broken acceptance-rate metric** in `probe_ngram_cpu_engine.py` (or, if
   the script is dropped from W1 per item 3, fix it on the ngram-cpu-d2h branch).
6. **[P2] Add `schema_version` or per-backend `decode_peel_supported`** on the metadata
   builder so future contract changes fail loudly instead of silently miscompiling CPU/XPU
   backends.
7. **[P3] Bring the W1 devlog into `docs/gfx906/AGENTS.md` template shape.**
8. **[P3] Add an assertion + comment for the V1 decode-first invariant** in
   `qwen_gdn_linear_attn.py::_forward_core`.
9. **[P3] Document the new harness env vars** in `docs/gfx906/running.md` or equivalent.
10. **[P3] Note PR #53077 dependency** in the devlog, or pull PR #53077 into
    `gfx906/main` before merging W1.
11. **[P3] Re-run a 4-req mixed serving A/B** to anchor production-shape extrapolation
    (carried from human review; not duplicated by GPT).

---

Copyright Kevin Read <me@kevin-read.com>
