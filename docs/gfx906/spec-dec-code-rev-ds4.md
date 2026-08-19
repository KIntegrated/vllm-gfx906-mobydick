# Critical code review — `gfx906/spec-decode`

Reviewed against `gfx906/main` (merge-base `37df3d5b5b`). 12 commits,
~3,773 insertions. Production-code surface:

- `csrc/rocm/dense_gemv_gfx906.cu` (+400: `dense_gemv_m_kernel[M]`,
  `dense_gemv_m_kernel_rt`, `dense_gemv_m4_gfx906`)
- `vllm/model_executor/layers/utils.py` (`_gfx906_spec_gemv_m4`,
  `_gfx906_gemv_long_k` extension)
- `vllm/config/compilation.py` (L5 "cg-small" restore)
- `vllm/gfx906_fa/gfx906_fa_backend.py` / `gfx906_fa_paged.py` (FULL
  cudagraph for spec decode)
- `vllm/v1/spec_decode/llm_base_proposer.py` (Gfx906FAMetadata allowlist)
- `vllm/_custom_ops.py`, `csrc/rocm/ops.h`, `torch_bindings.cpp`
- `tests/model_executor/layers/test_rocm_unquantized_gemm.py` (+128)

The rest is benchmark/probe scripts under `benchmarks/kernels/gfx906/` and
docs/devlog.

---

## Summary verdict

The kernel engineering is competent and well-documented, but the branch is
a **research/perf branch being asked to tell a correctness/serving-validity
story**, and it does not yet hold together for merge. The two clearest
problems are **(1) a likely dead-code "L5" fix** whose claimed serving win
(`0.945x → 1.077x`) is attributed to code that does not execute in the
documented run configuration, and **(2) a new kernel that is not actually
spec-decoded-only**, so the "no-spec regression structurally zero" claim is
narrower than the change. Several performance conclusions are "cause
unattributed", the perf numbers were measured on a **non-default build
(`VLLM_NO_MAX_ILP=1`)** without re-checking on the default build, and one
commit records **GPU faults** under the default flags. Treat the merge as
**needs work / not ready**, with the specific items below.

---

## Blocking / high-severity findings

### F1 — The L5 "cg-small restore" fix is likely unreachable on the real serving path (dead code)

`vllm/config/compilation.py` re-adds capture sizes `1..q-1` **inside the
`if not use_v2_model_runner:` block** (`compilation.py:1484-1496`). But the
messages themselves relied on:

- `vllm/v1/worker/gpu/model_runner.py:582` → `resolve_cudagraph_mode_and_sizes(..., use_v2_model_runner=True, ...)`
- `gfx906` spec decode runs on that v1 `ModelRunner` (MRV2).

So the L5 block is gated off in exactly the configuration the serving A/B
runs use. The HEAD commit credits the `0.945x → 1.077x` serving A/B
improvement to this fix, but on MRV2 the rounding
(`adjust_cudagraph_sizes_for_spec_decode`) is not applied via this path and
the small sizes are not re-added here. Either:

- the measurement is wrong, or the improvement came from something else
  (the "MTP + drafter GEMV dispatch" in the same commit also changes the
  timeline), or
- the gfx906 run actually used MRV1 and the served config is different from
  what the checkpoint under review implies.

**Action:** Confirm which model runner the A/B runs used; if MRV2, either
move the small-size handling into the MRV2 capture code
(`vllm/v1/worker/gpu/cudagraph_utils.py`, where uniform decode sizes are
manufactured at lines ~236) or drop the claim. Right now the `docs/`
devlog records a serving win for code that the tree does not wire up.

### F2 — The new M<=4 kernel is not spec-decode-only; "nospec regression structurally zero" is too strong

`_gfx906_spec_gemv_m4` (`utils.py`) has **no speculative-decoding guard**. It
is called unconditionally from `rocm_unquantized_gemm_impl` whenever gfx906 +
fp16 + contiguous + `2 <= n <= 4` + shape guards hold, for *any* workload —
not just spec draft steps (prefill dictation of small batches, batched
decode of 2–4 equal-len sequences, MTP drafter fc, etc.). The docstring and
the commit message ("M=1 paths untouched — nospec regression structurally
zero") correctly protect M=1 but say nothing about **M=2..4 non-spec** paths,
which now take a shape-tuned custom kernel with loose accuracy.

**Action:** add an explicit A/B + numeric gate for a *non-spec* M=2..4 fp16
batch on gfx906, or gate the dispatch on an actual spec-decode context. The
"structurally zero nospec regression" claim is only about M=1.

### F3 — Perf numbers are on a non-default, possibly hardware-unstable build

Commit `16d7b7254a` records: *"branch is currently `VLLM_NO_MAX_ILP=1` —
the per-file max-ilp flag ... is suspected of two weight-load GPU faults
(hipErrorLaunchFailure → CP preemption failure → BACO reset)".* The serving
A/Bs and the `1.077x`/`1.500x` numbers were taken on that build. The
`AGENTS.md`/devlog describe the **default** build as per-file max-ilp. So the
headline performance is on a degraded build and the default build's stability
was the open question. HEAD does not state the fault was resolved.

**Action:** re-run the gate A/Bs on the default (max-ilp) build once the
fault is root-caused, and state that explicitly in the merge/PR notes.
Perf numbers that depend on `VLLM_NO_MAX_ILP=1` are not directly comparable
to the shipped baseline elsewhere in this repo.

---

## Medium-severity findings

### F4 — `dense_gemv_m_kernel_rt` (M=4) kept on an "unattributed" 1.6x regression, adding permanent divergent paths

The templated-on-M kernel measurably *regressed* M=4 (`507 → 311 GB/s on the
LM head, `docs/gfx906/DEVLOG-spec-decode.md` "cause unattributed"), so M=4
ships a second, runtime-M kernel alongside the templated M=1..3. This doubles
kernel surface and the RT kernel's `xa[4]`/`acc[RPT][4]` arrays are always
sized 4 even for M<4 (the whole reason the templated variant exists). If the
M=4 regression were root-caused, the RT kernel would likely be deleted.

**Action:** record the root cause (register pressure / occupancy vs.
`xl`?) or at minimum add a comment linking a hypothesis; otherwise two
long-lived kernel variants with divergent code is net complexity for a
branch that is explicitly intermediate.

### F5 — Numerical accuracy of the K-split fp16-CAS epilogue is loose and order-nondeterministic

For ksplit>1 the accumulator is quantized to fp16 **per K-chunk** (up to
17 chunks at `K=17408`), then combined with a non-atomic-deterministic CAS
`atomic_add_pk2/pk4_f16` ordering (block-scheduling dependent). The tests
use generous tolerances for exactly this reason:
- `atol=0.3, rtol=2e-2` (M=4, K=5120)
- `atol=0.5, rtol=2e-2` (M=4, K=17408)

For draft tokens this is *acceptable* (the teacher verification discards bad
drafts), but the branch markets "token-identical to baseline" for k=2 — that
is a claim about the *output with the kernel's rounding*, not about
run-to-run determinism of the CAS order. Two successive runs of the same
input can pick different chunk orders and yield slightly different draft
tokens; with the k=2 MTK drafter those feed `fc` → LM head and are only
accepted/rejected downstream. If determinism (or exactness) is ever needed,
the MTP drafter path breaks the "K-split is bench-only / never on model
path" property that the M=1 kernel's comment claims.

**Action:** state explicitly in the devlog that (a) M>=2 spec GEMMs are
order-nondeterministic and (b) the "token-identical" claim does not extend
to bit-exact run-to-run output, or fix the epilogue to accumulate fp32 with
a single fp16 store per ksplit (as the M=1 `dense_gemv_kernel` already does
for the non-split path).

### F6 — `_gfx906_gemv_long_k` broadened from one measured shape to two; RPT rules for M>1 unverified

`_gfx906_gemv_long_k` now returns the custom GEMV for `row==5120` and `k in
(10240, 17408)` — a **behavior change for k=10240** that previously fell to
triton. This is the MTP drafter `fc`, invoked for **M=1** only here (guard in
the caller `m==1`? actually the caller path only reaches it for `n==1`, M=1
GEMV). Confirm that `k==10240` M=1 is the only new user; if a `k==10240`
shape ever appears on a non-draft M=1 decode it will now route to the custom
kernel without a dedicated A/B.

Also: `dense_gemv_m4_gfx906` defaults `rpt=2` unconditionally, ignoring the
M=1 rules (N==256 router → RPT=2; N>=1024 gate_up → RPT=4). Since M>1 shapes
include N=512 columns of gate_up/others, the RPT=4-vs-2 sweep was only done
for a subset of shapes. Minor perf risk for untested N.

### F7 — FA "FULL cudagraph for spec decode" — familiarity / generality of the capture-safe fast path

`gfx906_fa_backend.py` now returns `UNIFORM_BATCH` whenever
`num_speculative_tokens > 0` (regardless of the actual drafter), and
`gfx906_fa_paged.py` adds a `num_tokens == num_seqs * max_seqlen_q` uniform
fast path used for the Q-scatter and out-gather. The uniform fast path is
not spec-only: it fires for any uniform-length batch (including uniform
prefill). The correctness argument ("host ints imply uniform, so no D2H
sync") is sound, but it is a shape-of-the-batch inference, not a
spec-decode marker. Confirm no prefill case has `num_tokens ==
num_seqs * max_seqlen_q` **and** a `need_causal` reduction (>1 query) with
padded rows that would now be read — the comment claims pad rows are never
consumed, but that is kernel-internal and depends on `Sq_pad`/bounds checks.

---

## Lower-severity / hygiene

### F8 — Test coverage only gated on real gfx906; no PR-detection of the MRV2/dead-code issue

`test_rocm_unquantized_gemm_spec_gemv_m4_*` cover dispatch + real kernels,
but there is **no test** for the `compilation.py` L5 block (it runs after
`adjust_cudagraph_sizes_for_spec_decode`, is gated on MRV1 + gfx906, and
mutates capture sizes). The head of the branch's most-cited win has zero
test/CI coverage, which is how F1 slipped through.

### F9 — Documentation seems out of sync with shipped state

- `docs/gfx906/DEVLOG-spec-decode.md:747` notes "shows GPU0 temp/clock N/A
  with 80% zombie VRAM, fresh contexts fail" — hardware state during the
  headline numbers.
- The templated-vs-RT M=4 split is described in comments but "cause
  unattributed" is a permanent open loop (F4).
- Commit messages routinely record "unverified"/"TODO if time" items
  (e.g. k=3 "TODO if time") while asserting headline perf — these should be
  reconciled in the devlog before any merge pitch.

### F10 — Copyright / attribution

New kernel/test files carry the `# SPDX-FileCopyrightText: Copyright Kevin
Read` header consistently. `docs/gfx906/*` became markdown commentary
(external reviewer files `*-rev-glm.md` / `*-rev_claude.md` are kept without
copyright headers by design — that's fine). No issues found; keep it.

---

## Cheap correctness wins I verified as OK

- `dense_gemv_m_kernel` reduction is correct (every thread holds the full
  `RPT*M` accumulator set and shfl-reduces across its k-slice; cross-warp via
  LDS when `KCHUNK>512`).
- `TORCH_CHECK(N % rpt == 0)` and the dispatch's `m % 2 == 0` guarantee the
  packed-CAS RPT tail is fully covered (ragged-tail guards never fire).
- Buried special-case `m==5120, 2048<=k<=2304` keeps the tuned hipBLAS path
  intact (test `test_rocm_unquantized_gemm_spec_gemv_m4_dispatch`).
- `dense_gemv_m4_gfx906` left M=1 to the existing kernel; M=1 regression
  risk is genuinely zero on that route.
- FP32 accumulation in-register; only the K-split cross-chunk combine is fp16
  (F5).

---

## File / line index

| Finding | Location |
|---|---|
| F1 | `vllm/config/compilation.py:1497-1509` (inside MRV1-only block, line 1484) |
| F2 | `vllm/model_executor/layers/utils.py:270-296` (`_gfx906_spec_gemv_m4` has no spec gate); call site `:547-549` |
| F3 | `16d7b7254a` commit msg; `docs/gfx906/DEVLOG-spec-decode.md` |
| F4 | `csrc/rocm/dense_gemv_gfx906.cu:337-...` (`dense_gemv_m_kernel_rt`) |
| F5 | `csrc/rocm/dense_gemv_gfx906.cu` (`atomic_add_pk2/4_f16`); tests `:329-372` (atol 0.3/0.5) |
| F6 | `vllm/model_executor/layers/utils.py:297-320`; `_gfx906_spec_gemv_m4` rpt default |
| F7 | `vllm/gfx906_fa/gfx906_fa_backend.py:118-129`; `gfx906_fa_paged.py:560-576, 645-652` |
| F8 | `tests/model_executor/layers/test_rocm_unquantized_gemm.py` |