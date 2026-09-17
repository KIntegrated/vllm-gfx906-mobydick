# QSA-FN — Qwen3.8-Flash-Next (Qwen4Exp QSA) on gfx906 — fp16 enablement

> Branch `gfx906/qsa-fn` off `gfx906/v0.29.0` · model `Qwen/Qwen3.8-Flash-Next`
> (`qwen4_exp`) · 2026-09-17 · roadmap [`QSA-FN-1`](ROADMAP.md) ·
> pre-work evidence: [`RECON-qwen38-flash-qsa.md`](RECON-qwen38-flash-qsa.md)
> (the bf16-only site inventory, the ISA facts and the kernel timings all live
> there; not restated here).

**VERDICT:** `SHIPPED` for the code + kernel-level gates; the *end-to-end* gate
stays `OPEN`, owned by QSA-FN-3 (no loadable checkpoint exists on this box).

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
