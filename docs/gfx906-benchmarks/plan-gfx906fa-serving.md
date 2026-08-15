# Plan — make the gfx906 CUSTOM (Q8 FA) backend the serving decode path

Status: v1 (2026-08-15). Sub-plan of Phase 3 **P3-3a** (parent:
`plan-decode-phase3.md`). Evidence, bench history and the bug-hunt
narrative live in `DEVLOG-moe-opt.md` §"P3-3".

## 0. Goal and target

The vendored Q8 FlashAttention kernel is measured **72 µs/layer vs Triton
paged attention's 194 µs (2.7×)** but cannot run in serving mode today.
Goal: CUSTOM as the serving decode attention path with prefix caching ON
and cudagraphs as strong as the Triton baseline's.

| config | decode | attention slice/step |
|--------|--------|----------------------|
| serving, Triton, FULL_DECODE_ONLY (baseline) | **44.09 t/s** (22.7 ms e2e) | 10 × 194 µs = 1.94 ms |
| serving, CUSTOM — target M1 (PIECEWISE) | ≥ 46 t/s | 10 × 72 µs + gather ≈ 0.9–1.2 ms |
| eager, CUSTOM LEGACY=0 (works today) | 19.33 t/s | — |

Stop condition: if M1 nets **< +0.3 ms/step** after the buffer-churn fix,
abandon and fall back to P3-3b (Triton KV partitioning).

## 1. What blocks it (root causes, all verified in code/traceback)

**RC1 — crash in serving, LEGACY=0** (`RuntimeError: value_cache blocks
mismatch`, `gfx906_fa_paged.py:373` → TORCH_CHECK in
`csrc/gfx906_fa/gfx906_fa.cpp:431`):
`profile_run` → `_init_minimal_kv_cache_for_profiling` →
`initialize_kv_cache(is_profiling=True)` allocates a **minimal** KV cache;
the dummy forwards run `do_kv_cache_update`, whose `_ensure_q8_sidebuffer`
binds `_k_cache_q8` to that shape; `_cleanup_profiling_kv_cache` frees the
fp16 caches; the real cache is allocated later with a different
`num_blocks`; the first real step passes `value_cache` (real) +
`key_cache_q8` (minimal) → check fires. LEGACY=1 never allocates the side
buffer, which is why the default never crashed.

**RC2 — correctness hole with prefix caching (LEGACY=0)**: COW block copies
(`gpu_model_runner.execute_model` → `copy_kv_cache_blocks_inplace`,
`vllm/v1/worker/utils.py:563`) mirror only the fp16 caches; the Q8 side
buffer keeps stale rows → garbage on prefix hits that fork a partially
shared block. Full-block prefix hits reuse blocks **in place** (no copy) —
that is why the `[cached]` probe passed. Related:
`_zero_block_ids` zeroes fresh fp16 blocks only — **harmless** for Q8
(rows beyond `kv_max` are masked by the kernel; every live row is rewritten
by `reshape_and_cache_q8` before it can be read) — document, no change.

**RC3 — cudagraph mode downgrade**: `Gfx906FAMetadataBuilder
._cudagraph_support = AttentionCGSupport.NEVER` →
`resolve_cudagraph_mode_and_sizes` downgrades the engine to PIECEWISE,
while the Triton baseline serves with FULL_DECODE_ONLY. Even a working
CUSTOM would be measured in a weaker mode (attention eager between graph
pieces: ~30–40 extra launches/step, graph-boundary overhead).

**RC4 — capture debt (only needed for M2/FULL mode)**: forward-time
reallocs with `torch.cuda.empty_cache()` (class-level gather buffers
re-keyed by `Sk_pad` every 32-token boundary; `q_pad_buf` grow), host
branching on `max_seqlen` metadata (`_should_use_direct_paged`,
`ncols1`), dtype conversion `q.float()` per call (cost, not blocker).
All legal eagerly (PIECEWISE), illegal/fragile inside FULL capture.

## 2. Milestones

### M1 — PIECEWISE serving, correct (no capture-safety work needed)

Attention runs eagerly between piecewise graphs → dynamic shapes and
allocations stay legal. Work items:

- **W1 side-buffer lifecycle**: in `do_kv_cache_update`, if
  `_k_cache_q8` is None or `shape[0] != key_cache.shape[0]` → free,
  realloc at the real cache shape, `zero_()`. Runs inside the
  `vllm::unified_kv_cache_update` splitting op → always eager → alloc is
  safe. Keep the C++ TORCH_CHECK as the invariant guard.
- **W2 COW mirror**: registry (kv-cache `data_ptr` → Q8 tensor) filled by
  `_ensure_q8_sidebuffer`; gfx906-only hook at/around
  `copy_kv_cache_blocks_inplace` replays the same (src, dst) block-row
  copy on the Q8 storages (plain uint8 copy, same indices tensor).
  Stopgap if the runner hook is invasive: `GFX906_FA_NO_PREFIX_CACHE`
  env to disable prefix caching for A/B measurement only.
- **W3 gather-buffer hysteresis** (correctness-neutral perf): grow the
  class-level gather buffers in capacity steps (round `Sk` up to, say,
  512 tokens; serve per-step calls from `narrow()` views) so
  realloc+`empty_cache()` stops firing every 32-token boundary. Measure —
  this may be a visible slice of the eager gather tax.
- **W4 tests + probes** (§3): T1 lifecycle, T2 COW, T4 model-level
  partial-block prefix; re-run `[fresh]`/`[cached]` probes
  (`/tmp/bench/_p33_leg0.py` pattern) and DOUBLE_CHECK live gather.

**M1 exit**: serving bench (`BENCH_EAGER=0`, pp=2048/tg=256) runs LEGACY=0
end-to-end; probes correct; record t/s + resolved `cudagraph_mode`
(expected PIECEWISE) vs 44.09 baseline.

### M2 — FULL_DECODE_ONLY (CGSupport.ALWAYS)

Only if M1 nets ≥ +0.3 ms/step or piecewise boundary overhead is clearly
visible in the M1 trace. Work items:

- **W5 static decode shapes**: capacity-allocated gather buffers
  (B_max × Hkv × ceil(max_model_len/32)·32 × …) + `q_pad_buf` pre-alloc
  at max capture sizes; no alloc/`empty_cache` in forward. W1's
  realloc-on-mismatch still fires harmlessly during the pre-capture
  warmup (`_dummy_run` at capture sizes) — never during capture.
- **W6 audit host-branching in the decode path**: `_should_use_direct_paged`
  (B, Sq are static per capture; drop `max_seqlen_q`-dependence for decode),
  `ncols1` (decode Sq=1 → 2, baked), any `max_seqlen_k` use. Kernel loop
  bounds must derive from `kv_max` (device) — already true for the FA
  kernel; verify the gather kernel grid is `kv_max`-driven too (a
  capacity-sized grid wastes work at short seq).
- **W7 metadata**: `build_for_cudagraph_capture` already delegates to
  `build()`; confirm `seq_lens`/`block_table`/`query_start_loc` are the
  runner's persistent buffers (pointer-stable across replays) — they are
  for other backends with ALWAYS support; verify.
- **W8 flip support**: `_cudagraph_support = AttentionCGSupport.ALWAYS`
  behind `GFX906_FA_CG=always` env while validating → engine resolves
  FULL_DECODE_ONLY again. Expect the capture sizes [1,2,4,8] to exercise
  B=2..8 — where **direct-paged** (B≥2, no gather) becomes eligible;
  measure it too.

**M2 exit**: capture/replay correctness (T3) + serving t/s ≥ M1 number;
record mode = FULL_DECODE_ONLY.

### M3 — decision

Keep the best of {Triton FULL_DECODE_ONLY, CUSTOM M1, CUSTOM M2}. If all
CUSTOM variants < +0.3 ms/step over Triton → reopen P3-3b (design sketch
preserved in the devlog; do not re-derive).

## 3. Tests (extend `tests/kernels/attention/test_gfx906_fa.py`)

- **T1 lifecycle**: allocate side buffer at N1 blocks, free fp16 cache,
  allocate at N2 ≠ N1, `do_kv_cache_update` → `forward_paged` passes and
  output matches reference (simulates profile → real).
- **T2 COW**: fill blocks, run the copy util (or its Q8 mirror) with
  (src, dst) lists, re-gather → Q8 rows match freshly re-quantized fp16
  rows.
- **T3 capture/replay**: `torch.cuda.CUDAGraph` capture of
  (gather + `fa.forward`) with capacity buffers and a `kv_max` device
  tensor grown between replays → outputs match eager at each length.
- **T4 model-level partial-block prefix**: two sequential requests
  sharing a prefix whose last shared block is partially filled (forces
  COW) — expected garbage pre-W2, correct post-W2.

## 4. Bench matrix (standard docker recipe; pp=2048/tg=256)

- eager: Triton 19.49 / CUSTOM LEGACY=0 ~19.3 (regression check only).
- serving: Triton 44.09 / CUSTOM M1 / CUSTOM M2 — **record resolved
  cudagraph_mode with every number**.
- prefill: CUSTOM vs Triton pp-only run (prefill uses the same backend;
  must not regress).
- sanity greedy A/B vs Triton path: Q8 K quant → expect ~1e-3 divergence,
  both fluent (same trade llama.cpp makes). Fluency + diff report, not
  token-equality.

## 5. Expected outcome / stop conditions

Math: attention slice 1.94 ms → 10 × 72 µs (FA) + 10 × (fused gather +
q-fp32 ≈ 20–50 µs) ≈ **0.9–1.2 ms** → 22.7 → ~21.5–21.8 ms e2e →
**46–47 t/s** at M1; +0.2–0.4 ms more if M2 removes piecewise boundaries.

Stops: M1 < +0.3 ms/step after W3 → P3-3b; M2 correctness unresolved after
2 time-boxed days → keep best M1/M3 result, document; prefill regresses →
revert backend selection for prefill only (hybrid selection is allowed:
the backend priority list is per-engine, verify whether a per-layer-type
split is feasible before investing).

## 6. Risks

- **Piecewise boundary overhead** may eat the kernel win — measure at M1
  before assuming M2 is needed.
- **Capacity gather buffers**: VRAM is negligible (B × ~4.3 MB at ctx
  2816) but a capacity-sized gather grid does wasted work at short seq —
  keep grids `kv_max`-driven.
- **Class-level buffers + multiple engines in one process** (probes):
  key the registry by device+shape; add a reset hook for tests.
- **The copy hook must not affect non-gfx906 paths** — guard by registry
  emptiness.
- **Numerics**: Q8 K changes logits ~1e-3; greedy divergence from the
  fp16 path is expected and accepted (documented trade), but T1–T4
  byte/parity checks must stay strict.

## 7. Out of scope

- Direct-paged at B=1 (needs B≥2; becomes relevant at M2 capture sizes —
  measure, don't develop).
- fp16-Q kernel variant, sink/alibi/cascade support, TP.
- Changing the Triton backend (P3-3b owns that fallback).
