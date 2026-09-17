# QSA-FN — Qwen3.8-Flash-Next (Qwen4Exp QSA) on gfx906

> Branch `gfx906/qsa-fn` off `gfx906/v0.29.0` · model `Qwen/Qwen3.8-Flash-Next`
> (`qwen4_exp`) · 2026-09-17 · roadmap [`QSA-FN-*`](ROADMAP.md) · pre-work
> evidence: [`RECON-qwen38-flash-qsa.md`](RECON-qwen38-flash-qsa.md) (the
> bf16-only site inventory, the ISA facts and the kernel timings live there; not
> restated per entry). Newest entry first.

## 2026-09-17 (3) — QSA-FN-2: the tester serve recipe

**VERDICT:** `SHIPPED` (recipe + flag-set validation on the tiny harness); the
real-checkpoint behaviour stays `OPEN` and is the tester's report.

**GATE:** the full recipe flag set loads, captures and serves on the tiny
test rig (no real checkpoint available here).

### What was done

`docs/gfx906/_serve_qsa_flash_gfx906.sh` (`start|wait|stop|report`), carrying
the MI210 production launch with two **required** gfx906 deviations and the
dtype/parser deltas:

- `--dtype float16` (explicit), **no** `--mamba-cache-dtype bfloat16`;
- **`VLLM_USE_V2_MODEL_RUNNER=1`** — this model cannot run on V1 at all (the PLE
  inputs come from the V2 model states); this is the one recipe in the repo that
  must *not* pin V1 pending DFL2-2;
- **`--no-enable-prefix-caching`** — the V2-MAMBA-1 workaround; **retired
  2026-09-17** when that bug was fixed ([`DEVLOG-v2-mamba-align.md`](DEVLOG-v2-mamba-align.md)),
  now a fallback for older builds only;
- `--max-model-len 262144` native RoPE (no YaRN), `--block-size 64`,
  `--max-num-seqs 4`, `--max-num-batched-tokens 4096`, ladder `[4,8,12,16]`
  (= `max_seqs × (k+1)` for MTP k=3), `method:"mtp"` spelling,
  `qwen3_xml` tool parser + `qwen3` reasoning parser, `--enable-expert-parallel`.

Validated on the tiny rig (with `--load-format dummy`): V2 + expert-parallel at
TP=1 + custom-all-reduce off + the `[4,8,12,16]` ladder (4 PIECEWISE + 4 FULL
captured, not collapsed) + MTP k=3 (`SpeculativeConfig(method='mtp',
num_spec_tokens=3)`, drafts being created) + no prefix caching + both parsers:
loads, captures, serves completions / chat / 1321-token prefill / 4-way batch.

**Caveat recorded in the script and the roadmap:** with random weights the chat
message comes back with `content=None` and everything in `reasoning` — that is
the parser behaving normally on garbage, and `content` vs `reasoning` on *real*
output is one of the things the tester must report. Nothing about quality is
implied by this validation.

## 2026-09-17 (2) — QSA-FN-3: the tiny-config harness (and three things it found)

**VERDICT:** `SHIPPED` (the harness runs and gates the fp16 path end-to-end) —
with one deliberate limitation: it cannot measure **quality** (random weights)
and its per-kernel *shares* do not transfer to the real model, so QSA-FN-5's
prefill-share gate stays open.

**GATE:** the model serves on one MI50 and produces tokens in every path —
prefill, B=1..4 decode, graph replay, MTP spec decode, tool parser.

### What was done

Two new files, both committed:

- `docs/gfx906/_qsa_tiny_model.py` — writes a `config.json` that keeps the
  **architecture identical** (all four layer types, PLE with a real ngram table,
  hyperconnection, QSA indexer + sparse attention, MTP) and shrinks only the
  dimensions: hidden 2560→256, 48→4 layers (so the QSA layer fraction stays
  **1/4 = 12/48**, matching the real model), head_dim 256→64, E=512→8,
  MoE inter 640→64, `ngram_vocab_size_base` 20 M→4096 (the table is
  `≈ ngram_heads × base` rows), `max_position_embeddings` 262144→4096.
  `indexer_budget/compress_ratio` **cannot** be shrunk (the config requires the
  ratio to be 512 or 2048) and `vocab_size` must stay 248 320 (the tokenizer's
  size).
- `docs/gfx906/_serve_qsa_tiny_gfx906.sh` — `model|start|wait|stop`; serves it
  with `--load-format dummy` (random weights), V2, plugins off, capture ladder
  `[1..max_num_seqs]`.

Weights are random, so this is an **execution + A/B harness**: it proves the
fp16 QSA/PLE/HC/MoE/GDN path compiles, executes and decodes on this box, and
lets arms be differenced. It says nothing about output quality (log its
`prompt_sha1`s and never quote a t/s as a quality result).

### Baseline, fp16, one MI50, V2, prefix caching OFF

Prefill (TTFT, warmed): 1321 tokens **22 ms** (60.0 k prompt tok/s), 2641
**42 ms** (62.4 k), 3961 **48 ms** (83.1 k). Decode (128 tok/req, median of 3
after a per-shape warmup): **B=1 563 t/s, B=2 1023 t/s, B=4 1994 t/s**.

### The three findings

1. **This model cannot run on the V1 model runner.** `Qwen4ExpModel.forward`
   takes `query_start_loc`/`ngram_context` with `None` defaults and PLE raises
   `PLE inputs were not prepared`; the plumbing that fills them lives in
   `v1/worker/gpu/model_states/mamba_hybrid.py` (`Qwen4ExpModelState`), i.e. the
   **V2** runner. The house recipe pins V1 (`_serve_tp2_gfx906.sh`, pending
   DFL2-2) — for `qwen4_exp` that pin is not an option. Recipe/FN-8 impact.
2. **With V2 + prefix caching ON (→ `mamba_cache_mode='align'`),
   `precopy_mamba_align_fused_kernel` faults on gfx906.** First failing case:
   a 2015-token prefill (`Memory Fault Error … kernel:
   precopy_mamba_align_fused_kernel`, grid `[256, 7, 16]`) → EngineDeadError;
   1343-token prefills pass. That kernel is V2-only — SYV-13 (2026-09-08)
   checked it and concluded it was "not live on ROCm" (true while V1 was
   pinned). It is live now, and it is untested here: its test is CUDA-gated
   (`tests/kernels/mamba/test_precopy_mamba_align.py` skips on ROCm; only 3 of
   195 mamba tests ran). **Workaround for any Qwen4Exp run:
   `--no-enable-prefix-caching`** (mamba cache mode then stays `none`; verified:
   the same 1321/2641/3961-token prefills run clean). Filed as `V2-MAMBA-1` and
   **fixed the same day** — it was not a kernel bug at all but a wrong divisor
   in `add_request` ([`DEVLOG-v2-mamba-align.md`](DEVLOG-v2-mamba-align.md)), so
   the workaround is retired; the two mamba kernel tests named above are
   ROCm-enabled in this branch.
3. **The bf16 arm is independently broken** — and it is *not* this change. With
   `--dtype bfloat16` the engine dies in `rocm_unquantized_gemm_impl` with
   `Matrices A and B must have the same dtype (assuming fp16)`, inside the
   hyperconnection chain during the first compiled forward. In a bf16 run my
   `params_dtype=vllm_config.model_config.dtype` evaluates to exactly the
   literal `torch.bfloat16` it replaced, and the relaxed guards accept bf16
   exactly as before — so the pre-FN-1 tree must fail identically. fp16 is not
   just preferred on gfx906, it is the only arm; the A/B that FN-1 would
   ideally ship (whole-model fp16 vs bf16) is therefore unavailable, leaving
   the 4.39×/1.28× kernel numbers as the evidence.

### Harness hygiene (cost me a wrong first reading)

- **The first request at a new shape includes a one-time Triton-autotune storm**
  (a 20 s stall at B=1/B=2 that looked like a decode collapse). Warm up each
  batch shape and take the median of ≥3 reps; the pre-warm numbers are
  meaningless (B=1 measured 6.5 t/s and 563 t/s in the same process).
- **The venv auto-loads five stale profiler plugins** (`agdn/mtp1/pfk4/syv9/t1
  phase`, installed as `vllm.general_plugins` entry points, armed by leftover
  files in `/local/tmp/mtp1`). vLLM loads every discovered plugin when
  `VLLM_PLUGINS` is unset. A/B'd: with `VLLM_PLUGINS=""` the numbers are the
  same within spread (B=1 466 vs 448, B=4 1616 vs 1600 t/s), so they are inert
  — but the harness pins `VLLM_PLUGINS=` anyway, and any future measurement
  here should too.
- **Not measured: the QSA share of prefill.** It is QSA-FN-5's decision gate and
  the harness cannot supply it (the tiny model's per-layer dims are 10–24× off
  the real ones, so the share is not transferable). Options: measure it on a
  tester's box, or scale a second config to the real shape ratios.

## 2026-09-17 (1) — QSA-FN-1: fp16 enablement

> Branch `gfx906/qsa-fn` off `gfx906/v0.29.0` · model `Qwen/Qwen3.8-Flash-Next`
> (`qwen4_exp`) · 2026-09-17 · roadmap [`QSA-FN-1`](ROADMAP.md) ·
> pre-work evidence: [`RECON-qwen38-flash-qsa.md`](RECON-qwen38-flash-qsa.md)
> (the bf16-only site inventory, the ISA facts and the kernel timings all live
> there; not restated here).

**VERDICT:** `SHIPPED` — code + kernel-level gates, and the fp16 path now
end-to-end via the tiny harness ([entry 2026-09-17 (2)](#2026-09-17-2--qsa-fn-3-the-tiny-config-harness-and-three-things-it-found)).

**GATE:** `tests/models/qwen4_exp/` (both dtype arms, per file), **plus** the
FN-7 existing-model gates — FA suite **104 passed**, PPL probe **10.5472**
(= the recorded value for this build), MoE 35B bench **58.30 t/s** mean
(= parity). The kernel timings are **launch-regime evidence, not the gate**.

---

## HYPOTHESIS

The reported failure is the gfx906 bf16→fp16 auto-fallback meeting QSA's
bf16-only guards, and the CDNA2 patch set does not fix it. If fp16 is admitted
everywhere the QSA path reads or stores 2-byte floats, the path becomes runnable
on gfx906 *and* ~4× cheaper — because gfx906 lowers fp16 `tl.dot` to
`v_dot2_f32_f16` while bf16 is emulated as scalar `v_fmac_f32`.

## What was done

Edit list (all of it; `RECON-qwen38-flash-qsa.md` §1 is the pre-edit inventory):

- `common/qsa_cache.py` — new `QSA_ACTIVATION_DTYPES` / `QSA_KV_CACHE_DTYPES`
  (single source of truth for the guard sets); `QSAStateBackend` dtype lists;
  `_QSAStateCache.bind_kv_cache` now checks `self.dtype`; `_BF16_PER_INT64` →
  `_ELEMS_PER_INT64` (same value 4 — 4 × 2 B = 8 B = one int64, correct for both
  2-byte dtypes); docstrings de-BF16'd. **Kept dtype-general** — this module is
  shared with the NVIDIA implementation, which is untouched.
- `amd/qsa.py` — backend `supported_dtypes` / `supported_kv_cache_dtypes`, the
  Impl's KV-dtype guard, `forward_qsa`'s Q/K/V guard (now also requires
  `key_cache.dtype == query.dtype`), the attention activation guard (**the
  reported error**) and both cache-dtype guards.
- `amd/indexer_qsa.py` — the second copy of the activation guard; the raw and
  compressed index-key caches now take `model_config.dtype` instead of a bf16
  literal.
- `amd/ops/qsa.py` — `qsa_sparse_paged_attention`'s dtype assert.
- `amd/model.py` (×2) and `amd/mtp.py` (×1) — `HyperConnectionConfig(
  params_dtype=…)` now takes the model dtype. **Pulled in from QSA-FN-2**: the HC
  linears must match the activations, so a fp16 run with bf16 HC weights is not a
  coherent fp16 path. The three NVIDIA sites keep their bf16 literal.
- Tests: `test_qsa_amd.py` — the sparse-attention reference test is now
  parametrized `bf16|fp16`, and a new `test_qsa_mqa_paged_matches_reference`
  covers the indexer scoring kernel against a torch reference in both dtypes.
  `test_qsa_reference.py` — the shared state-cache bind test is parametrized, and
  a new `test_qsa_raw_key_cache_packs_int64_rope_positions` pins the int64-MRoPE
  packing in both 2-byte dtypes.

No kernel code changed: the fp16 path rides the existing Triton kernels.

## Evidence — FOR

- `test_qsa_amd.py` **9 → 16 passed** (5 attention shapes × {bf16, fp16}, the
  indexer kernel × {bf16, fp16}, plus the 4 pre-existing tests). Baseline "9
  passed" was re-measured on MI50 before the edit.
- `test_qsa_reference.py` **16 → 19 passed**; `test_config.py` 7, `test_ple.py`
  10 unchanged. `test_qsa_pre_indexer.py` / `test_hc_ops.py` remain CUDA-gated
  (all skipped on ROCm).
- Measured kernel win (**launch-regime**, MI50, one GPU): sparse attention
  **26.5 ms fp16 vs 116.5 ms bf16** (4.39×, interleaved reps, stable to ±0.2 %);
  per-row indexer kernel **5426 µs fp16 vs 6928 µs bf16** (1.28×).
- FN-7 regression gates, this build, same boot: `test_gfx906_fa.py` **104
  passed**; PPL probe (Qwen3.8-27B-AWQ-INT4, fp16, 359 tokens) **10.5472, 0
  top-20 misses** — identical to the value recorded for this build in
  `RECON-triton-1.md`/`CHANGELOG.md` (10.5472 stock 3.8.0 vs 10.5516 fork);
  MoE 35B `_bench_gfx906.py` pp2048/tg256, 4 samples, mclk 1000: **58.31 /
  58.35 / 58.29 / 58.23 t/s** (mean 58.30) against the recorded 57.97 (stock
  3.8.0) / 58.36 (fork) — inside the per-process spread.
- By-product baseline (not a gate; no prior record): Qwen3.5-27B-AWQ PPL
  **14.3750** (359 tokens, 0 misses). Not comparable to the 10.55 band — that is
  a different model (`CHANGELOG.md` 0.29.0-validation entry).

## Evidence — AGAINST / limits

- **No end-to-end run exists.** "All bf16 sites are covered" rests on a grep plus
  the declarative guard sets, not on a served request. A missed guard fails
  *loudly* (same `NotImplementedError`), but an fp16 path that only trips at
  runtime (e.g. a PLE/MoE/HC kernel without fp16 support) would not be caught
  here.
- The indexer's compress/store/selection kernels are not exercised in fp16 at
  all (no AMD-side test existed for the indexer before this change either);
  only the scoring kernel is.
- The config-shape plumbing (`Qwen4ExpQSAAttention(...)`, the indexer's caches,
  the HC weights) is not instantiated in any test — that needs a VllmConfig fixture
  and is exactly what QSA-FN-3 should provide.

## Why it works

`v_dot2_f32_f16` is a real gfx906 instruction and Triton 3.8.0 (`GCN5_1`)
emits it for fp16 dots; there is no bf16 instruction on Vega20, so every bf16
dot falls back to scalar fp32 FMA plus converts. That 4.4× is a codegen fact,
not a tiling effect — which is why the whole win is available for guard edits
alone.

## Interactions / superseded-by

- Depends on TRITON-1 (stock Triton 3.8.0 `GCN5_1` adopted 2026-09-15); on the
  older 3.6 fork the ISA facts should be the same (its `supportsVDot` includes
  `VEGA20`) but were not re-measured here.
- Unblocks QSA-FN-4 (the tiled indexer needs an fp16 arm to be a win at all) and
  is the reason QSA-FN-5's int8 KV is unattractive: fp16 is now the cheap,
  fast baseline on this chip.
- QSA-FN-6 (int8-`tl.dot` IMA) is unaffected — it is a codegen hazard in the
  int8 dot, still off by default.

## Refrigerated residue

- `QSAKeyStateCache`'s int64 packing width is still a hardcoded 4; the generic
  form is `8 // element_size`, needed only if a 1-byte cache dtype ever lands.
- `common/hyperconnection.py`'s `HyperConnectionConfig.params_dtype` default is
  now dead (all six construction sites pass it explicitly) — left alone rather
  than touching the NVIDIA-adjacent default.
- `cache_config.mamba_cache_dtype` for this model: the CDNA recipe pins bf16.
  Not touched here (it is a launch-recipe item, QSA-FN-2).
