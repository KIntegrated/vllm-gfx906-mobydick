# Plan — make the gfx906 CUSTOM (Q8 FA) backend the serving decode path

Status: v2 (2026-08-15). Sub-plan of Phase 3 **P3-3a** (parent:
`plan-decode-phase3.md` v6 — **P3-3a currently suspended** pending M0
go/no-go). Evidence, bench history and the bug-hunt narrative live in
`DEVLOG-moe-opt.md` §"P3-3". Review findings that drove v2:
`gfx906fa-serving-plan-rev-claude.md` (merged claude + ds4 + qwen).

## 0. Goal and target

The vendored Q8 FlashAttention kernel is measured **72 µs/layer vs Triton
paged attention's 194 µs (2.7×)** but cannot run in serving mode today.
Goal: CUSTOM as the serving decode attention path with prefix caching ON
and cudagraphs as strong as the Triton baseline's.

| config | decode | attention slice/step |
|--------|--------|----------------------|
| serving, Triton, FULL_DECODE_ONLY (baseline) | **44.09 t/s** (22.7 ms e2e) | 10 × ~194 µs ≈ 1.94 ms ¹ |
| serving, Triton, PIECEWISE (M0 reference) | TBD | TBD |
| serving, CUSTOM M1 (PIECEWISE) | TBD after M0 | 10 × 72 µs + gather TBD |
| serving, CUSTOM M2 (FULL_DECODE_ONLY) | TBD after M1 | TBD |
| eager, CUSTOM LEGACY=0 (works today) | 19.33 t/s | — |

¹ "10 × 194 µs" comes from a P3-0 profile at seq~500. The 44.09 t/s bench
runs at average Sk≈2176. M0 must produce the real attention slice from a
rocprofv3 pass over the 44.09 t/s run itself before these projections are
trusted.

**Why the §5 math is provisional:** the only measured full-path data point
at B=1 is eager parity (CUSTOM 19.33 vs Triton 19.49 t/s). Since
`_DIRECT_PAGED_MIN_BATCH=2` and the serving bench is single-request (B=1),
every serving decode step routes through the gather path — direct-paged
never engages. The eager parity implies the gather+dtype tax is absorbing
essentially the entire 122 µs kernel win at B=1, suggesting gather alone is
~100+ µs/layer, not the 20–50 µs assumed in §5. The M0 gather micro-bench
exists to measure this directly.

**M1 target (revised):** "CUSTOM-PIECEWISE ≥ Triton-PIECEWISE + 0.3 ms/step"
(mode-matched). The 44.09 t/s baseline is context only, not the comparison
point, because CUSTOM runs PIECEWISE while that baseline is FULL_DECODE_ONLY.

Stop condition: if M1 nets **< +0.3 ms/step vs the Triton-PIECEWISE
baseline** after W3, abandon and fall back to P3-3b (Triton KV
partitioning).

## 1. What blocks it (root causes, all verified in code/traceback)

**RC1 — crash in serving, LEGACY=0** (`RuntimeError: value_cache blocks
mismatch`, `gfx906_fa_paged.py:373` → TORCH_CHECK in
`csrc/gfx906_fa/gfx906_fa.cpp:431`):
`profile_cudagraph_memory` (not `profile_run` — see note) →
`_init_minimal_kv_cache_for_profiling` →
`initialize_kv_cache(is_profiling=True)` allocates a **minimal** KV cache;
the dummy forwards run `do_kv_cache_update`, whose `_ensure_q8_sidebuffer`
binds `_k_cache_q8` to that shape; the real cache is allocated later with a
different `num_blocks`; the first real-capture warmup calls
`do_kv_cache_update` with `value_cache` (real shape) + `key_cache_q8`
(minimal shape) → TORCH_CHECK fires. LEGACY=1 never allocates the side
buffer, which is why the default never crashed.

*Note on trigger:* `_init_minimal_kv_cache_for_profiling` is called from
`profile_cudagraph_memory` (`gpu_model_runner.py:6765`), which runs
**after** `profile_run`. During `profile_run` itself, `kv_cache_config`
does not exist yet → `_get_slot_mappings` returns None → no
`do_kv_cache_update` at all. T1 must simulate the
`profile_cudagraph_memory` → real-cache path, not `profile_run`.

**RC2 — correctness hole with prefix caching (LEGACY=0)**: COW block copies
(`gpu_model_runner.execute_model` → `copy_kv_cache_blocks_inplace`,
`vllm/v1/worker/utils.py:563`) mirror only the fp16 caches; the Q8 side
buffer keeps stale rows → garbage on prefix hits that fork a partially
shared block. Full-block prefix hits reuse blocks **in place** (no copy) —
that is why the `[cached]` probe passed. Related:
`_zero_block_ids` zeroes fresh fp16 blocks only — **harmless** for Q8
(rows beyond `kv_max` are masked by the kernel; every live row is rewritten
by `reshape_and_cache_q8` before it can be read) — document, no change.

**Invariant**: correctness of the LEGACY=0 path rests on *every live K row
being written via `do_kv_cache_update` before it is read*. Exceptions to
guard: (a) KV connectors / disaggregated prefill write blocks directly into
the fp16 cache bypassing `do_kv_cache_update` — add a loud log/assert when
a connector is enabled with LEGACY=0; (b) COW copies (RC2/W2).

**RC3 — cudagraph mode downgrade**: `Gfx906FAMetadataBuilder
._cudagraph_support = AttentionCGSupport.NEVER` →
`resolve_cudagraph_mode_and_sizes` downgrades the engine to PIECEWISE,
while the Triton baseline serves with FULL_DECODE_ONLY. Even a working
CUSTOM would be measured in a weaker mode (attention eager between graph
pieces: ~30–40 extra launches/step, graph-boundary overhead). **This makes
M1's serving number incomparable to the 44.09 t/s baseline** — see revised
M1 target in §0 and M1 exit in §2.

**RC4 — capture debt (only needed for M2/FULL mode)**: forward-time
reallocs with `torch.cuda.empty_cache()` (class-level gather buffers
re-keyed by `Sk_pad` every 32-token boundary; `q_pad_buf` grow), host
branching on `max_seqlen` metadata (`_should_use_direct_paged`,
`ncols1`), dtype conversion `q.float()` per call (cost, not blocker).
All legal eagerly (PIECEWISE), illegal/fragile inside FULL capture.

**RC4 additional — two capture landmines not in the original RC4 list:**

- **Warmup→capture Sk_pad mismatch**: pre-capture warmup builds metadata
  with `for_cudagraph_capture=False` → `max_seq_len=1` → `Sk_pad=32`.
  The capture itself uses `for_cudagraph_capture=True` →
  `max_seq_len=max_model_len` → `Sk_pad≈2816`. `_ensure_gather_buffers`
  requires an **exact** shape match; the mismatch triggers
  `torch.cuda.empty_cache()` **inside the CUDA graph capture** — illegal.
  W5 must fix the reuse semantics (see §2/M2), not just pre-allocate.

- **Class-level gather buffers dangle across capture sizes**: in ascending
  capture order (B=1, 2, 4, 8), each capture writes into the class-level
  buffer that exists at that moment. Capturing B=2 reallocates and frees
  the B=1 buffer; the B=1 graph then writes K/V into freed memory on every
  replay. W5 must ensure all captured graphs share one buffer object that is
  never shrunk.

## 2. Milestones

### M0 — pre-work (go/no-go gate; ~half day; do before any M1 code)

Required before any M1 coding per parent plan v6:

1. **Gather micro-bench**: measure `gather_paged_kv_q8` + `q.float()` cost
   per layer at serving shapes (B=1, Sk≈2176–2816, Hkv=8, D=256) in
   isolation. This is the decisive go/no-go gate. If gather > ~80 µs/layer
   the realistic M1 serving gain collapses to ≤0.3 ms and P3-3a stays
   suspended.
2. **Triton-PIECEWISE baseline**: run the standard serving bench
   (`BENCH_EAGER=0`, pp=2048/tg=256) with the Triton backend forced into
   PIECEWISE mode. Record t/s and the resolved `cudagraph_mode`. This is
   the mode-matched comparison point for all M1 numbers.
3. **Baseline attention slice**: rocprofv3 pass over the 44.09 t/s run to
   record the real Triton attention time at Sk≈2176, replacing the
   seq~500 estimate from §0.

**M0 exit / go condition**: gather ≤ ~80 µs/layer → proceed to M1.
If gather > ~80 µs/layer → P3-3a suspended; switch to P3-2.

### M1 — PIECEWISE serving, correct (no capture-safety work needed)

Attention runs eagerly between piecewise graphs → dynamic shapes and
allocations stay legal. Work items:

- **W1 side-buffer lifecycle**: in `do_kv_cache_update`, if `_k_cache_q8`
  is None or `(key_cache.data_ptr(), key_cache.shape, key_cache.device)`
  does not match the stored binding → free, realloc at the real cache
  shape, `zero_()`. (Checking `shape[0]` alone misses coincidentally-equal
  block counts on engine re-init; bind by full identity.)
  Update the W2 registry entry in the same operation (data_ptr changes on
  realloc). Trigger: `profile_cudagraph_memory` → real-capture warmup path,
  not `profile_run` (see RC1 note). Keep the C++ TORCH_CHECK as the
  invariant guard.

- **W2 COW mirror**: the copy utility (`copy_kv_cache_blocks_inplace`,
  `vllm/v1/worker/utils.py:563`) runs in `_update_states`
  (`gpu_model_runner.py:1267–1271`) before the forward. The mirror must be
  in the same call window (a post-forward hook would miss prefix hits that
  affect the current step).

  Registry design (required specifics before writing the hook):
  - **Key**: the fp16 backing storage pointer of the `kv_cache` tensor
    for each FA layer (e.g. `kv_cache.untyped_storage().data_ptr()`), not
    the Q8 side-buffer pointer (which has a separate storage).
  - **Scope**: only gfx906 FA layers. GDN/mamba state blocks also appear
    in `kv_caches` — applying `(src, dst)` block indices to the wrong
    storage is a silent corruption path. Guard by registry emptiness for
    non-registered storages.
  - **KV-sharing**: if two FA layers share one fp16 backing storage
    (`kv_sharing_target_layer_name`), the data_ptr key collides. Key by
    `(data_ptr, layer_name)` or register a per-layer list.
  - **On W1 realloc**: the data_ptr changes → update the registry entry
    in the same `_ensure_q8_sidebuffer` call (see W1).
  - **Copy semantics**: a `view(num_blocks, -1)` uint8 replay of the same
    `(src_block_id, dst_block_id)` pairs is valid — `_k_cache_q8` is
    block-major contiguous with the same `num_blocks` as the fp16 cache
    (verified). Also replay V (fp16) if a V-Q8 side buffer is added later.

  Stopgap if the runner hook is invasive: `GFX906_FA_NO_PREFIX_CACHE` to
  disable prefix caching **for A/B measurement only** — if used, record it
  explicitly next to the bench number. M1 numbers measured with this flag on
  are incomparable to the 44.09 baseline (which ran with prefix caching).

- **W3 gather-buffer hysteresis** (perf, gated on flame-trace evidence):
  grow the class-level gather buffers in capacity steps (round `Sk` up to,
  say, 512 tokens; serve per-step calls from `narrow()` views) so
  realloc+`empty_cache()` stops firing every 32-token boundary. **Do not
  bundle this into the M1 milestone** — gate on measuring whether the
  realloc is visible in a flame trace first. Include as a separate A/B row
  in the §4 bench matrix rather than as a prerequisite.

- **W4 tests + probes** (§3): T1 lifecycle, T2 COW, partial-block probe
  (see M1 exit below); check probes into `tests/` or `tools/` (not `/tmp`);
  DOUBLE_CHECK live gather.

**M1 exit** (all required before declaring M1 done):
1. Serving bench (`BENCH_EAGER=0`, pp=2048/tg=256) runs LEGACY=0
   end-to-end without crash.
2. **Partial-block COW probe correct** (hard gate): two consecutive requests
   where the shared prefix length is **not** a multiple of 16 (e.g.
   520-token first request; second request reusing that prefix → last shared
   block is partially filled → COW fires → Q8 rows must be correct).
   This is the only path that exercises RC2/W2; the standard bench and the
   existing `[cached]` probe both use block-aligned prefixes and never fire
   COW.
3. Resolved `cudagraph_mode` asserted PIECEWISE (assert, not just record).
4. FA-layer backend selection asserted in logs (not silently falling back
   to Triton).
5. Record t/s + mode vs **Triton-PIECEWISE** (M0 reference), with 44.09
   as context. Gate: CUSTOM-PIECEWISE ≥ Triton-PIECEWISE + 0.3 ms/step.
6. Report state: `NO_PREFIX_CACHE` flag on/off, bench entrypoint, sample
   count.

### M2 — FULL_DECODE_ONLY (CGSupport.UNIFORM_SINGLE_TOKEN_DECODE)

Only if M1 nets ≥ +0.3 ms/step vs the Triton-PIECEWISE baseline, or if
piecewise boundary overhead is clearly visible in the M1 trace. Work items:

- **W5 static decode shapes** (correctness requirement, not just perf):
  Replace the exact-shape reuse check in `_ensure_gather_buffers` with a
  ≥-capacity check + `narrow()` view. Pre-allocate at max capture sizes:
  `B_cap = max(max_cudagraph_capture_size, max_num_seqs)`,
  `Sk_cap = ceil(max_model_len/32)·32`. Remove all `torch.cuda.empty_cache()`
  from the forward path. Add `assert not torch.cuda.is_current_stream_capturing()`
  in any remaining realloc path. Ensure all captured graphs (B=1, 2, 4, 8)
  share **one** buffer object that is never shrunk — the B=2 capture must
  not free the B=1 buffer (see RC4 additional). Same treatment for
  `q_pad_buf`.

- **W6 audit host-branching in the decode path**: `_should_use_direct_paged`
  (B, Sq are static per capture; the branch is on values known at capture
  time — bake the result), `ncols1` (decode Sq=1 → ncols1=2, baked), any
  `max_seqlen_k` use. For the gather grid: it is **frozen at capture**
  (kernel launch parameters cannot be per-call in FULL mode). Rely on the
  existing per-workgroup `full_oob` early-out in `gather_paged_kv_q8_kernel_v2`
  (`(block_start_tok >= seq_len) || (phys_block < 0)`) which zeroes V tails
  and skips K for out-of-length blocks — this is the capture-safe substitute
  for a kv_max-driven grid. Verify the early-out covers D=64/128 variants.
  Pin the `block_size == 16` assumption explicitly (direct-paged is
  hard-coded to block_size=16; add an assert so a future block-32 model
  fails loudly rather than silently falling back to gather-only).

- **W7 metadata**: `build_for_cudagraph_capture` already delegates to
  `build()`; confirm `seq_lens`/`block_table`/`query_start_loc` are the
  runner's persistent buffers (pointer-stable across replays). Verified:
  `seq_lens` is a persistent GPU buffer zero-filled for padded rows
  (`gpu_model_runner.py:2241`); `query_start_loc.gpu` and block-table device
  tensors are persistent; padded rows use `NULL_BLOCK_ID` / `-1` / `kv_max=0`
  → inert. Also verify `kv_max` expansion (`seq_lens [B] → [B, grid_x]`
  via `unsqueeze/expand/contiguous` in gfx906_fa.cpp:217) is pointer-stable
  under capture: source is the persistent `seq_lens` buffer, so replay
  re-reads current values — safe, but requires `seq_lens` to be int32
  contiguous (else an extra captured copy).

- **W8 flip support**: `_cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`
  (not `ALWAYS` — `ALWAYS` claims mixed prefill+decode full-cudagraph
  support, which is false: `forward_paged`'s non-decode branch contains a
  host Python loop with `int(cu[s+1]-cu[s])` — a D2H sync inside capture.
  `UNIFORM_SINGLE_TOKEN_DECODE` passes the decode-FULL gate, rejects
  spec-decode and mixed FULL, and accurately describes what is capture-safe.)
  Gate behind `GFX906_FA_CG_MODE=always` env (rename from `GFX906_FA_CG=always`
  to match the `GFX906_FA_FUSED` / `GFX906_FA_DIRECT_PAGED` valued-key
  convention). Assert that the Sq=1 fast path
  (`max_seqlen_q == 1 and num_tokens == num_seqs`) is taken during every
  capture — if not, the Python loop is on the captured stream and must be
  rewritten before capture.

**M2 exit**: 
- T3 passes (see §3 — must cover warmup→capture `max_seq_len` transition
  and multi-size capture+replay, not just single-size capture).
- Serving t/s ≥ M1 number.
- Kernel launch count or inter-kernel gap for the B=1 graph confirms
  reduction vs M1's eager attention (otherwise M2 passed while buying
  nothing).
- Record mode = FULL_DECODE_ONLY.
- Run `BENCH_SAMPLES ≥ 5` for all serving numbers; record σ.

### M3 — decision

Keep the best of {Triton FULL_DECODE_ONLY, CUSTOM M1, CUSTOM M2}. If all
CUSTOM variants < +0.3 ms/step over Triton-PIECEWISE → reopen P3-3b
(design sketch preserved in the devlog; do not re-derive).

## 3. Tests (extend `tests/kernels/attention/test_gfx906_fa.py`)

- **T1 lifecycle**: allocate side buffer at N1 blocks, simulate the
  `profile_cudagraph_memory` → real-cache path (not `profile_run`), then
  `do_kv_cache_update` with real-shape cache → `forward_paged` passes and
  output matches reference. Use N1 and N2 that differ by more than one
  alignment class (not just ±1 block) to ensure the mismatch check fires.

- **T2 COW**: fill blocks, run the Q8 mirror copy with (src, dst) lists,
  re-gather → Q8 rows match freshly re-quantized fp16 rows. Also assert
  that Q8 rows for COW-copied blocks are correct post-copy (the "every live
  row is rewritten before read" invariant does not hold for COW-copied rows
  until W2 mirrors them).

- **T3 capture/replay** (extended from v1): `torch.cuda.CUDAGraph` capture
  — must cover (a) the warmup→capture `max_seq_len` transition
  (Sk_pad=32 warmup → Sk_pad=max_model_len capture) to exercise the RC4
  Sk_pad mismatch landmine; (b) multi-size capture+replay: capture B=1,
  then B=2 (which would free B=1's buffer under the old exact-match design),
  then replay B=1 → output must match eager. A single-size capture cannot
  catch the dangling-buffer bug.

- **T4 partial-block COW probe** (replaces the v1 "model-level" T4 which
  required scheduler machinery): a checked-in probe script (in `tests/` or
  `tools/`, not `/tmp`) that runs two consecutive completions via a live
  engine where the shared prefix is **not** a multiple of 16 tokens
  (e.g. 520-token prompt → second request reuses prefix → COW fires on the
  last partially-filled block). Assert output correctness. This is also the
  hard gate in the M1 exit.

## 4. Bench matrix (standard docker recipe; pp=2048/tg=256)

All numbers must record: resolved `cudagraph_mode`, `GFX906_FA_NO_PREFIX_CACHE`
state, bench entrypoint, `BENCH_SAMPLES` count.

- **M0 gather micro-bench**: `gather_paged_kv_q8` + `q.float()` cost at B=1,
  Sk≈2176–2816 in isolation. Go/no-go gate for P3-3a.
- **M0 Triton-PIECEWISE baseline**: Triton backend forced PIECEWISE,
  pp=2048/tg=256. This is the mode-matched M1 comparison point.
- **M0 attention slice**: rocprofv3 over 44.09 t/s run → real Triton FA
  time at Sk≈2176 (replaces the seq~500 estimate).
- **eager**: Triton 19.49 / CUSTOM LEGACY=0 ~19.33 (regression check only).
- **M1 serving**: CUSTOM PIECEWISE — compare vs Triton-PIECEWISE (M0), not
  vs 44.09.
- **W3 A/B** (if flamed): gather buffer hysteresis on vs off, same serving
  recipe. Separate row, not part of M1 milestone.
- **M2 serving**: CUSTOM FULL_DECODE_ONLY — compare vs M1 + kernel launch
  count delta.
- **prefill**: CUSTOM vs Triton pp-only run. Acceptance bar: CUSTOM prefill
  ≥ Triton prefill − 5%. Record a CUSTOM-vs-Triton prefill baseline before
  M1 (no such number exists in the devlog yet).
- **B≥2 concurrent-decode** (M2 only, if feasible): 4–8 concurrent requests,
  decode-dominated, to exercise direct-paged. Otherwise mark B≥2/direct-paged
  measurement explicitly out of scope for the M3 decision.
- **sanity greedy A/B vs Triton**: Q8 K quant → expect ~1e-3 logit
  divergence. Acceptance: perplexity on a fixed prompt set within 2% of the
  fp16 path (not "fluency" — not measurable).

## 5. Expected outcome / stop conditions

**Projection (provisional until M0 gather micro-bench):**
attention slice 1.94 ms → 10 × 72 µs (FA) + 10 × (gather TBD µs) ≈ TBD ms.
The 46–47 t/s headline from v1 rests on the 20–50 µs gather assumption,
which is unverified and contradicted by the eager parity data. Update
after M0.

**Stop conditions:**
- M0 gather > ~80 µs/layer → P3-3a suspended; proceed with P3-2.
- M1 < +0.3 ms/step vs Triton-PIECEWISE after W3 → P3-3b.
- M2 correctness unresolved within scope bounds (see W5–W8 per-item):
  keep best M1/M3 result, document.
- Prefill regresses beyond −5% vs Triton: if CUSTOM prefill is below the
  bar, real options are (a) accept the regression; (b) optimise CUSTOM's
  prefill path; (c) add an in-forward dispatch fallback inside
  `Gfx906FAImpl.forward` based on a `max_query_len` threshold (dispatching
  to the Triton impl for prefill — a hack but implementable); (d) abandon
  M1. **There is no per-phase backend selection in vLLM** (one backend per
  attention group serves both prefill and decode of every layer in that
  group); "revert backend for prefill only" is not a mechanism that exists.

## 6. Risks

- **Gather tax at B=1 may be fatal**: the M0 micro-bench is the primary
  gate. The eager parity strongly suggests it; the micro-bench confirms or
  refutes it directly.
- **Piecewise boundary overhead** may eat the kernel win — measure at M0
  (Triton-PIECEWISE baseline) before assuming any kernel win will survive
  to the M1 number.
- **Capacity gather buffers in M2**: the buffer-never-shrunk invariant is
  a correctness requirement for multi-size capture, not just a VRAM
  concern. VRAM: `B_cap × Hkv × Sk_cap × (bytes_per_row + D×2)` — at
  B=32, ctx 2816 ≈ 137 MB; at max_model_len=32K ≈ 0.4 GB. Fine on MI50
  32 GB, but state the formula rather than a single datapoint.
- **Class-level buffers + multiple engines in one process** (probes):
  key the registry by `(device, layer_name)` or equivalent; the reset hook
  must refuse while live CUDA graphs exist (resetting under live graphs
  dandles them), or be test-only.
- **The COW copy hook must not affect non-gfx906 paths** — guard by
  registry emptiness (correct design; the scoping requirement in W2 makes
  this automatic).
- **Numerics**: Q8 K changes logits ~1e-3; greedy divergence from the
  fp16 path is expected and accepted (documented trade), but T1–T4
  byte/parity checks must stay strict.
- **Block-size assumption**: direct-paged is hard-coded to block_size=16
  (gfx906_fa.cpp:580). The bench model is 16; add an assert in W8 so a
  future block-32 model fails loudly rather than silently falling back to
  gather-only.

## 7. Out of scope

- Direct-paged at B=1 (needs B≥2; becomes relevant at M2 capture sizes).
- B≥2 concurrent-decode measurement (add to §4 or explicitly exclude
  from M3 decision — do not leave it implicit).
- fp16-Q kernel variant, sink/alibi/cascade support, TP.
- Changing the Triton backend (P3-3b owns that fallback).
- KV connectors / disaggregated prefill with LEGACY=0 (guard with
  log/assert per RC2 invariant note; do not develop compatibility).
