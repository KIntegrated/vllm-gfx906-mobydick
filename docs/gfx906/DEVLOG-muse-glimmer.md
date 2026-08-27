# Muse-Glimmer-30B onboarding — CT-asym W4A16 dense text + iRoPE hybrid attention on gfx906

> Branch `feat/muse-glimmer` off `main` (4a9e24b5c) · model
> `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` · date 2026-08-26 · onboarding +
> sliding-window CUSTOM FA track.

**Model:** 52 text layers, hidden 6656, Hq 32 / Hkv 2 (GQA 16:1), head_dim
128, sliding_window 2048, fp16 KV. iRoPE: 13 full-attention NoPE layers
(indices 3,7,…,47,51) + 39 sliding-window(2048) RoPE layers. Vision tower
50 layers (all unquantized fp16), lm_head + layers 47/51 linears unquantized
(315 ignored tensors, CT `ignore` list). Dense CT-W4A16-asym (group 32,
uint4, int8 zp) → `CompressedTensorsWNA16` → Exllama `gptq_gemm`
(`use_v2_format=True`, raw zp).

**Baseline attention (no code change):** per-layer auto-select gives
sliding group → `ROCM_ATTN` (Triton `unified_attention`), full group →
`CUSTOM` (Q8 FA); ViT → `FLASH_ATTN`. Log: `rocm.py:748`
"Found incompatible backend(s) [CUSTOM, TURBOQUANT] … Overriding with
ROCM_ATTN".

## 2026-08-26 — onboarding: load + smoke + hybrid baseline

**VERDICT:** OPEN (baseline established; window work pending) ·
**GATE:** `_bench_gfx906.py` serving wall-clock, pp2048/tg256, 4 samples,
`BENCH_MAX_SEQS=4`, `BENCH_GPU_UTIL=0.93 BENCH_KV_MEM=4634016400`,
graph mode. (Util-only sizing OOMs the first request in warm-cache runs:
profiling peak ~0.16 GiB lower than cold → KV pool oversized for the
532 MiB inductor prefill buffer — 27B's was 356 MiB, so 0.93/0.95 are
both too tight here; explicit 4.32 GiB KV cap per the engine log's
suggestion, also makes A/B arms pool-identical. Logs:
`/local/tmp/muse/bench_hybrid_graph{,_093,_093_r2}.log`.)

## HYPOTHESIS

If the AWQ checkpoint loads through the existing CT-asym WNA16 (Exllama)
dense path and the per-layer backend auto-select routes sliding layers to
ROCM_ATTN, the model serves coherent output with no code change — i.e.
upstream MuseGlimmer support (PR #51655) + the shipped gfx906 quant path is
sufficient for a working baseline, and the only gap is sliding-window
support in CUSTOM FA (perf, not correctness).

## What was done

- `feat/muse-glimmer` from `main` (upstream `muse_glimmer` model/config/
  processor/reasoning+tool parsers already merged, PR #51655; registry maps
  `MuseGlimmerForConditionalGeneration` → `("muse_glimmer", …)`).
- Download: original target `Vishva007/…-W4A16-AutoRound-GPTQ` died in a
  host crash (group-64 sym GPTQ); switched to
  `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` (group-32 CT-asym W4A16 — same
  kernel family as the shipped Ornith MoE path, but **dense** linears).
- Cache layout incident: my `hf download --cache-dir /data/cache/huggingface`
  put blobs in the NFS cache root, but `~/.cache/huggingface` →
  `/local/cache/huggingface` (local disk) is where transformers resolves
  (its `hub/` snapshot already had the small files from the pre-crash
  attempt). Copied the 5 shards + tokenizer.json into the `/local` snapshot
  (23 GB, now on local disk).
- Smoke (text-only `LLM`, `max_model_len=8192`, gpu_util 0.93, 4 greedy
  prompts via native chat template, thinking ON): loads in 58 s from local
  disk (22.55 GiB weights), coherent + correct answers (Paris; Rayleigh
  scattering; haiku drafting; 17·23=391). Log `/local/tmp/muse/smoke_hybrid.log`.
- Baseline bench: config above, log `/local/tmp/muse/bench_hybrid_graph_093_kv.log`.
  Harness gained a `BENCH_KV_MEM` hook (documented in README/running.md but
  missing from the harness file).

## Evidence — FOR

- Coherent, factually correct greedy output on 4 prompts incl. arithmetic;
  thinking turns emitted with the native ` to=self` protocol the reasoning
  parser expects.
- Backend select as predicted (sliding→ROCM_ATTN, full→CUSTOM); ViT
  FLASH_ATTN.
- **Hybrid baseline (graph mode, the gate):** 17.45 / 17.46 / 17.44 t/s
  (samples 1–3; sample 0 12.40 cold), pp2048/tg256, BENCH_MAX_SEQS=4,
  BENCH_GPU_UTIL=0.93 + BENCH_KV_MEM=4634016400. Qwen3.8-27B reference on
  the same harness: ~25.2 t/s (all-attention CUSTOM; no sliding layers).
  (Re-verified at the matched A/B config: 17.54 t/s.)

## Interactions / superseded-by

- Dense CT-asym Exllama dequant is **not** previously validated on gfx906 —
  the Ornith log (DEVLOG-ornith-wna16.md) covered MoE experts only. Smoke
  coherence is weak evidence for the (q−zp)·scale zero-point math; a
  numerics check vs transformers (logprob delta) is the follow-up if any
  doubt appears.
- Sliding-window CUSTOM FA landed in the next entry (SHIPPED, 1.59×);
  the open follow-up is step C — gather windowing (`kv_start` clip) for
  long-context decode, where the mask-only step still scans [0, seq_len).
  At the 2048-token window it is irrelevant for contexts ≤ window, so it
  only matters for long-ctx serving.

## 2026-08-26 — sliding-window CUSTOM FA (mask-only)

**VERDICT:** SHIPPED (1.59×) · **GATE:** same serving A/B as above,
all-52-layers CUSTOM arm vs the hybrid baseline.

## Gate result (2026-08-27 00:0x, matched config)

Both arms: graph mode, pp2048/tg256, 4 samples, BENCH_MAX_SEQS=4,
BENCH_KV_MEM=805306368 (15k-token pool ≫ 2816 needed),
BENCH_BATCHED_TOKENS=1024 (see the memory forensics below — 2048-chunk
prefill + 4.32 GiB KV OOMs the all-CUSTOM arm on the first request).

| arm | sample 0 (cold) | samples 1–3 (stable) |
|---|---|---|
| hybrid (sliding→ROCM_ATTN pinned, full→CUSTOM) | 12.43 | **17.54 t/s** (17.556/17.542/17.521) |
| all-CUSTOM (52/52 layers, window=2048) | 16.86 | **27.90 t/s** (27.931/27.901/27.874) |

**1.59×** decode wall (tg256). Hybrid stable matches the pre-change
baseline (17.45) — the harness-config delta (chunk size, KV cap) does not
move decode. Backend select verified in both logs (per-group `info` lines;
`_cached_get_attn_backend` dedupes identical configs, so one line per
group): all-CUSTOM shows both groups → CUSTOM, hybrid shows full → CUSTOM
+ sliding → ROCM_ATTN (kind-pin rides the "selected via
--attention-backend" log path). The Qwen3.8-27B all-CUSTOM reference on
the same harness is ~25.2 t/s — Muse 30B now runs ahead of it.

Note: that gate ran B=1 (single repeated request) with prefix caching on
the harness default. Both gaps closed by the 2026-08-27 review-follow-up
entry below (B=4 re-run, prefix cache off, window=0 control arm).

## HYPOTHESIS

If the Q8 FA tile kernel gets a per-row sliding cutoff (mask keys older
than `q_abs_row - W + 1`, re-using the existing `q_abs_offset` inline-
causal machinery, passed for every windowed batch including decode), all
52 Muse layers run on CUSTOM and decode recovers part of the 17.45→25.2
t/s gap the Triton sliding path costs.

## What was done

- `fattn-q8.cuh` / `fattn-q8-paged.cuh`: tile-iter + global kernels take a
  scalar `window`; both causal branches extend to
  `k_pos > q_abs_row || (window > 0 && k_pos < q_abs_row - W + 1)`.
  Fully-masked tiles are safe: `KQ_max` seeds at `-FLT_MAX/2` (finite), so
  an all-`-INF` tile leaves max unchanged and `val=exp(-INF)=0` — no NaN
  guard needed (plain causal could never hit an all-masked tile; a window
  can).
- `gfx906_fa_launcher.cu` (+ the cpp's own extern decls): `window` plumbed
  through `gfx906_fa_launch{,_paged}`.
- `gfx906_fa.cpp`: `forward` / `forward_paged_direct` gain `window=0`
  (pybind args added).
- `gfx906_fa_backend.py`: `Gfx906FAImpl` stores `sliding_window` (was
  `NotImplementedError`); backend class now
  `supports_sliding_window() -> True`.
- `gfx906_fa_paged.py`: `forward_paged(window=0)`;
  `need_causal = max_seqlen_q > 1 or window > 0` (decode rows need
  `q_abs_offset` for the per-row window; the causal check itself is a
  no-op there since all keys ≤ seq_len-1 = the decode row).
- Tests: `test_forward_sliding_window_vs_torch_ref` ×4 (D=128 GQA 16:1
  model shape W=128/64, W>L inert, D=256; decode + per-row prefill vs
  torch ref, plus window-bites / window-inert sanity). 32/32 suite green
  (28 pre-existing untouched).

Mask-only step (B of the plan): FA still scans [0, seq_len) — long-ctx
windowing (gather `kv_start` clip) is the follow-up (step C), gated by the
same A/B plus a long-context curve.

## Memory forensics — all-CUSTOM arm OOMs on the first prefill

Symptom: the all-52-CUSTOM bench OOMs on the 532 MiB inductor prefill
buffer at the first request (hybrid arm OOM'd once too, 23:00:36, and
recovered via empty_cache; all-CUSTOM never recovers). Probe
(`/local/tmp/muse/probe_mem.py`): post-init allocated 24.22 / reserved
26.06 GiB (weights 22.55 + graphs 0.71 + ~1 GiB). The prefill adds:

- **per-layer `q_pad_buf` [B, Hq=32, Sq_pad, D=128] fp32** — decoded
  batches were growing its dim0 to the capture max (B=8) although Sq=1
  never READS it (both branches use `q_pad_decode_buf`): a B=8 capture
  then ballooned every layer's prefill buffer to (8, 32, 2048, 128) =
  268 MB → 14 GiB across 52 layers. **Fixed**: `_ensure_forward_buffers`
  sizes dim0 with `qpad_num_seqs = num_seqs if max_seqlen_q > 1 else 1`
  (decode never reads `q_pad_buf`).
- per-layer C++ `o_bshd` FA output [B, Sq, 32, 128] fp32 = 33.5 MB at
  Sq=2048 → ~1.7 GiB across 52 layers while the prefill graph holds
  them (hybrid's 39 Triton layers emit fp16, ~2× smaller).

Residual budget: at KV cap 0.75 GiB + `BENCH_BATCHED_TOKENS=1024`
(prefill chunks 1024 instead of 2048 → q_pad/o_bshd per layer halve,
inductor workspace smaller) the first prefill should clear. KV pool
size is irrelevant to single-request decode wall (2816 tokens ≪ pool),
so the A/B arms just share the same capped pool. Long-term levers (not
for this gate): fp16 `o_bshd` in the Q8 FA forward (halves the per-layer
output), and/or a worker-shared prefill q_pad arena.

## Evidence — FOR / AGAINST

FOR: 32/32 FA suite (4 new window tests green); gate A/B above (1.59×);
all-CUSTOM smoke coherent + correct (4 prompts, thinking ON, 7.4 s for
4 completions vs 12.7 s hybrid). AGAINST: (none).

## 2026-08-27 — review follow-ups: direct-paged window coverage, B=4 gate, control arm

**VERDICT:** SHIPPED · **GATE:** same A/B at B=4 (BENCH_NREQS=4,
BENCH_PREFIX_CACHE=0 per the local-serving default) + a window=0-on-
CUSTOM control arm, per `muse_glimmer_opt_code_rev_claude.md` #1/#7/#8/#9.

## HYPOTHESIS

If the direct-paged kernel's hand-duplicated window formula is correct
and the window mask is ~free, then (a) a `forward_paged_direct` window
test matches the torch reference, (b) the all-CUSTOM win persists at
B=4 — where the sliding layers actually run the direct-paged kernel
(auto mode, min_batch=2 — not the B=1 gather path the first gate
exercised) — and (c) forcing window=0 (unbounded causal, perf-only) on
all-CUSTOM moves the t/s by ~0.

## What was done

- **Direct-paged window test** (#1): `test_forward_paged_direct_
  sliding_window_vs_torch_ref` ×4 — same shapes as the gather test,
  calling the `forward_paged_direct` binding directly (block_size=16,
  unbind(1) V layout): decode B=1, decode B=2 with different seq lens
  (per-row q_abs_offset/window), and per-row prefill. Suite 36/36.
- **`GFX906_FA_NO_WINDOW=1`** control knob (#9): forces
  `sliding_window=0` for every layer (numerically wrong for windowed
  layers, perf-representative) — the third bench arm.
- **`BENCH_PREFIX_CACHE`** harness hook (#7): defaults to the old
  behavior (on) for comparability; gate re-runs use 0.
- Comments: spec-decode caveat on the q_pad decode heuristic (#10 —
  verify steps DO read q_pad_buf but their Sq_pad is bounded by the spec
  depth, so the 14 GiB pathology stays prefill-only); validated envelope
  on `supports_sliding_window` (#11); LOCKSTEP notes on the duplicated
  cutoff formula in both .cuh files (#2).

## Gate result (B=4, pp2048/tg256, 4 samples, prefix cache OFF, 0.75 GiB
KV, bt1024)

| arm | B=1 (prev) | B=4 (this) |
|---|---|---|
| hybrid (sliding→ROCM_ATTN, full→CUSTOM) | 17.54 | 16.75 (16.78/16.74/16.71) |
| all-CUSTOM, window=2048 | 27.90 | 20.59 (20.73/20.59/20.19) |
| all-CUSTOM, window=0 (control) | — | 20.64 (20.73/20.64/20.64) |

- **1.23×** at B=4 vs 1.59× at B=1 — the win persists, smaller: hybrid
  decodes 4.5% slower at B=4 (Triton per-seq sliding attention batches
  worse) while all-CUSTOM drops 26% (B=4 direct-paged steps cost 5.4×
  the B=1 step time for 4× the work — grid-z/occupancy tuning headroom,
  not a regression: B=1 is unchanged).
- **Window-mask cost ≈ 0**: control 20.64 vs windowed 20.59 (−0.2%,
  within run noise) — at B=4 the win is entirely the kernel family on
  the 39 sliding layers; the per-row window branch is free.
- Logs: `/local/tmp/muse/bench_b4_{hybrid,allcustom,nowindow}.log`.

## Evidence — FOR / AGAINST

FOR: 36/36 FA suite; B=4 gate 1.23×; control arm isolates the window
cost at ~0; both kernel copies now reference-tested. AGAINST: (none).

## Interactions / superseded-by

Supersedes the B=1-only gate framing of the 2026-08-26 entry (numbers
there stand; this entry adds the B=4 + control data). The B=4
step-time tuning this entry flagged is done in the next entry (split-K
+ batch-aware KVSPLIT default, Phase C window clip).

## 2026-08-27 — perf follow-ups: direct-paged split-K + Phase C window clip

**VERDICT:** SHIPPED with ERRATUM (see below — both gates ran on the
GATHER path, not the direct-paged path they were attributed to; the
direct-paged path is unreachable in default serving). · **GATE:** B=4
pp2048 serving A/B (S=1 vs batch-aware default, same session) +
pp4096 clip on/off A/B + kernel micro-bench.

## ERRATUM (2026-08-27, review round 2)

The direct-paged branch of `forward_paged` is gated on
`key_cache_q8 is not None` — which the backend only passes when
`GFX906_FA_LEGACY=0`, and that mode is still broken on this tree
(Q8 side-buffer desync → garbage output; smoke-verified 2026-08-27,
see the review-round-2 entry). Default serving (LEGACY=1) therefore
runs EVERYTHING on the gather path, so:

- The **B=4 split-K gate was a GATHER-path measurement**: arm 1
  (KVSPLIT=1) vs arm 2 (unset → gather default 16). The +2.7% is real
  but is the gather kernel's existing S=16-vs-S=1 knob, not the new
  direct-paged plumbing. The batch-aware `clamp(16/B, 2, 8)` default
  is dead code in default serving (it only applies to the
  direct-paged path).
- The **pp4096 clip A/B (and the later pp8192 A/B) were NULL TESTS**:
  with LEGACY=1 the clip code (direct branch) never executes, so both
  arms ran identical gather-path code. The "below the noise floor"
  reading was a comparison of two identical configurations. Phase C's
  e2e serving benefit has NEVER been measured; the kernel-level −48%
  (L=4352) / −71% (L=8448) micro-bench numbers stand (binding level),
  but they only apply when LEGACY=0 works.
- Unchanged by the erratum: the window-mask gates (B=1 1.59×, B=4
  1.23× — the mask runs in the gather kernel), the q_pad fix (shared
  backend buffer code), and the direct-paged code itself (plumbing is
  real and binding-tested; it just can't be reached from serving yet).

Re-gate Phase C (and the batch-aware KVSPLIT) once the LEGACY=0
Q8-side-buffer desync is fixed.

## HYPOTHESIS

(B=4 anomaly) The direct-paged launch pinned `grid.y=1` (no split-K,
unlike the gather path's KVSPLIT=16), so every block's critical path is
the FULL ~88-iteration KV k-loop — hence B=4 steps cost 5.4× the B=1
step for 4× the work. If the paged kernel's existing strided split-K +
split_combine machinery (unused by the paged host path) is plumbed in
with a batch-aware split count, B=4 decode recovers part of the gap
without a kernel change. (Phase C) If a windowed decode row's k-loop
starts at `max(0, L-W)` instead of 0, the output is bit-identical
(prefix keys are window-masked to -INF anyway) and long-context decode
reads ~W/L of the KV.

## What was done

- **Direct-paged split-K** (host-only): the paged kernel already
  strides the k-loop by `gridDim.y*nbatch_fa` and stores unscaled
  `[rows, y, D]` partials + meta when `gridDim.y>1` — the launcher
  pinned `grid.y=1` and the binding never allocated/merged partials.
  Plumbed `kv_split` through launch_paged + `forward_paged_direct`
  (same OOM guard as the gather path: split=1 for Sq>2).
  `GFX906_FA_KVSPLIT` now returns -1 when unset: gather keeps its
  16 default; direct-paged defaults to `clamp(16/batch, 2, 8)` —
  batch-aware because grid-z = batch×Hq (NC2=1 paged), so the split
  that fills the 60-CU MI50 moves with the batch.
- **Phase C clip** (kernel + plumbing): `kv_start` arg end-to-end
  (binding → launcher → `fattn-q8-paged.cuh`); the k-loop walks
  `[kv_start, L)` (split-K still covers the clipped range exactly —
  slice y visits `kv_start + (y+i·S)·nbatch_fa`). Python passes
  `kv_start = max(0, q_abs + 1 − W)` for windowed decode (Sq=1) only;
  prefill rows keep the full scan (per-row windows; that clip is
  still open). `GFX906_FA_WINDOW_CLIP=0` kill switch for A/B.
- Tests: clip suite ×4 shapes (bit-identity to the masked full scan
  < 1e-6; a functional check with an INERT window (W=L) + real clip
  start proving the scan itself shrinks; unaligned start (W=64 →
  start=448 vs nbatch_fa=128); B=2 per-row). Suite 40/40.

## Gate result

**B=4 split-K** (pp2048/tg256, 4 samples, prefix off, 0.75 GiB KV,
same session/build):

| arm | t/s (samples 1–3) |
|---|---|
| KVSPLIT=1 (old grid.y=1) | 20.05 (20.27/20.07/19.89) |
| batch-aware default (S=4 @ B=4) | **20.59** (20.66/20.56/20.45) |
| KVSPLIT=2 (micro-bench optimum) | 20.55 (20.72/20.48/20.44) |

**+2.7%** e2e; S=4 (the formula) matches S=2 (the micro-bench
optimum) within noise, so the shape-independent formula is kept.
Micro-bench (B=4, Hq=32, D=128, L=2816, window=2048): best split per
batch = 8/8/5/2/2 for B=1/2/3/4/8; `clamp(16/B, 2, 8)` hits the
measured optimum at B∈{1,2,3,8}, within 4% at B=4 (235 vs 226 us).

**Phase C** (pp4096/tg256, B=4, prefix off — window bites 50% of the
decode KV at L=4096–4352):

| arm | t/s (samples 1–3) |
|---|---|
| clip ON (run 1) | 10.35 (10.51/10.26/10.19) |
| clip OFF (run 1) | 10.72 (10.65/10.68/10.76) |
| clip ON (run 2) | 10.54 (10.80/10.47/10.33) |
| clip OFF (run 2) | 10.44 (10.77/10.33/10.24) |

Kernel micro-bench at the serving shape (L=4352): FA 337.8 → 175.7 us
(S=4, −48%) — the clip works at kernel level. E2e the effect is below
the noise floor: the four runs span 10.35–10.72 with clip OFF
interleaved at 10.72 AND 10.44 (between the two clip-ON runs), and
three of the four runs drift ~4% DOWN across their own samples (the
host state fluctuates on the ~5 min timescale; an initial "clip ON
slower" read was the run-1 ordering artifact). FA is only ~8% of the
B=4 step at pp4096 (GEMM-bound), so the expected ~2.9% e2e win isn't
resolvable here. At L=8k+ the same clip saves ~4× the FA work (~11%
of the step) — re-measure on a longer-context config if long-ctx
serving matters. Correctness is exact (bit-identical, 40/40), so clip
ON stays the default; it is also the prerequisite for the gather-path
clip.

## Evidence — FOR / AGAINST

FOR: 40/40 suite (clip bit-identity + functional scan-shrink);
B=4 split-K +2.7% same-session A/B; micro-bench split matrix
(stable across repeats: S=2 225.9/226.3 us across two runs); Phase C
kernel −48% at L=4352; 4-run pp4096 A/B shows no clip-induced loss
(OFF interleaved at both ends and the middle of the ON range).
AGAINST: Phase C e2e WIN not resolvable at pp4096 (below the ~2% run
noise); within-run drift ±4% on 3 of 4 runs (host-state fluctuation —
recorded, not diagnosed).

## Gotchas / notes

- `GFX906_FA_FWD_DEBUG=1` is incompatible with cudagraph capture
  (its `torch.cuda.synchronize()` at the _DBG log points invalidates
  the stream capture — `hipErrorStreamCaptureInvalidated`). Debug
  graph-mode runs without it.
- Build note: the CMake build compiles a hipified copy of the .cu
  staged under `build/temp…/csrc/` — the in-tree `.hip` mirrors are
  NOT built (they predate even the window arg); edit the `.cu`.
- A 16-way clang build concurrent with a 20 GB weight load
  coincided with one `hipErrorLaunchFailure` (first non-OOM-collateral
  event of boot I; retry clean) — see degradation.md. Serialize
  builds and loads if it recurs.
- Open: gather-path (B=1) window clip (needs the persistent gather
  kernel's work list to start per-row at `max(0, L-W)` + compacted
  store + shifted q_abs_offset); prefill-row clip (per-row windows
  [max(0, t-W+1), t] — a 2D per-(row, k) problem, not a per-row
  start); **fix the LEGACY=0 Q8-side-buffer desync** — the only path
  to the direct-paged kernel (split-K, batch-aware KVSPLIT, Phase C
  clip); re-gate Phase C e2e at L≥8k after that.

## 2026-08-27 — review round 2: clip bit-identity fix (P1), NaN guard, checks, null-test exposure

**VERDICT:** SHIPPED (P1 clip floor fix reproduced + verified on
hardware; NaN guard + symmetric checks in; both prior e2e clip gates
exposed as null tests by the LEGACY=1 dispatch reality — see the
erratum in the perf-follow-ups entry). · **GATE:** pre/post-fix
unaligned clip-vs-full diff on hardware + 44/44 suite + LEGACY=0
smoke.

## HYPOTHESIS

(Review P1) The Phase C clip is NOT bit-identical to the masked full
scan when `kv_start` is not a multiple of `nbatch_fa` (the normal
production case, e.g. L=4353/W=2048 → 2305): an unaligned start makes
the first tile partial, repacking which lane holds each surviving
score and re-associating the fp16 VKQ reduction. Flooring the clip
start to the tile boundary — only when the keys the floor gains are
window-masked — restores exactness at the cost of ≤ nbatch_fa−1 extra
keys. (Review P3) A fully-masked row (KQ_sum==0) writes inf*0=NaN on
the non-split (gridDim.y==1) store path in BOTH kernels; guard it like
the split-combine l_star guard.

## What was done

- **Clip floor (P1)**: `fattn-q8-paged.cuh` floors `k0_base` to
  `nbatch_fa` when `k0_base <= max(0, q_abs + 1 - window)` (the
  production clip passes exactly that; the guard was found when an
  unconditional floor broke the inert-window test subcase, where the
  clip emulates a window and the gained keys are NOT masked).
- **NaN guard (P3)**: `scale_out = KQ_sum>0 ? 1/KQ_sum : 0` in the
  non-split store of both `fattn-q8-paged.cuh` and `fattn-q8.cuh`
  (the gather copy had the identical pattern — fixed by inspection).
- **Symmetric checks**: `window > 0` now requires `q_abs_offset` in
  BOTH bindings (previously the window mask silently degraded to full
  attention without the offset, while `kv_start` had the check).
  Verified: both bindings reject the malformed call.
- **Nits**: clip math int32-only (no int64 round-trip);
  `GFX906_FA_NO_WINDOW` truthy-parsed like the sibling knobs + warns
  (was `== "1"`, silent); KVSPLIT "0 ≠ unset" note; README knob table
  (KVSPLIT batch-aware row, WINDOW_CLIP, NO_WINDOW); `BENCH_PREFIX_CACHE`
  default flipped to 0 (AGENTS.md local-serving recipe);
  `supports_sliding_window` envelope note: the clip only fires on the
  direct-paged (B≥2) dispatch.
- **Tests**: +3 unaligned bit-identity shapes (513/128, 1025/256,
  4353/2048 — L deliberately not a multiple of BLOCK) + 1 NaN-guard
  test (B=2, fully-masked row, KVSPLIT=1, no-NaN + zero-row + correct
  sibling) + 1 forward_paged-LEVEL test (review #10: dispatch →
  direct branch → Python clip math → binding, B=2, two different
  seq lens, unaligned starts 385/352 — split-K + window + clip all
  active together) → suite 45/45.

## Gate result

Pre-fix (old .so) vs post-fix, clip vs masked full scan:

| shape (L/W→start) | pre-fix rel | post-fix max-abs |
|---|---|---|
| 513/128→385 | 5.2e-4 | 0.0 |
| 1025/256→769 | 5.4e-4 | 3.0e-8 |
| 4353/2048→2305 | 7.2e-4 | 5.6e-9 |
| 512/128→384 (aligned) | 0.0 | 0.0 |

(Review's independent numbers 4.7e-4/5.3e-4/6.8e-4 — same shapes,
seed noise.) The old suite's "unaligned" shape (512/64→448,
448%128=64) was accidentally bit-exact: the 64-key window fills the
half tile on a chunk boundary — the suite's 40/40 never saw the bug.
(The review's "448 is a multiple of 128" is arithmetic wrong —
448%128=64 — but its conclusion that the old test missed the bug is
right.)

**LEGACY=0 smoke (fresh, 2026-08-27)**: 4-prompt greedy → incoherent
prompt-echo garbage ("to=self<prompt>…" + meta-reasoning) with a
capped 0.375 GiB KV pool (an uncapped pool OOMs on the ~1.5 GiB Q8
side buffer + inductor headroom). The Q8 side-buffer desync is still
live → the direct-paged path (split-K, batch-aware KVSPLIT, Phase C
clip) is unreachable in any working serving config; the pp4096 and
pp8192 clip A/Bs were null tests (both arms gather). Blocking item:
fix the desync before any direct-paged e2e claim.

> **RESOLVED** by the round-3 entry below — and partly misdiagnosed:
> the "incoherent garbage" was the model's normal greedy style (see
> the round-3 erratum); the real defects were side-buffer sizing and,
> in the first alias fix, per-call re-alias zeroing.

## Evidence — FOR / AGAINST

FOR: pre/post hardware repro of the P1 bug + fix (3 shapes); 45/45
(incl. the forward_paged-level direct-branch test); window-check
rejection on both bindings; LEGACY=0 garbage smoke (blocker
established, not assumed). AGAINST: no new e2e serving numbers
(nothing to re-gate until the desync fix); Phase C serving benefit
remains unmeasured at every context.

## Notes

- `_fwdlog` (GFX906_FA_FWD_DEBUG) writes to
  `/tmp/gfx906_fa_debug/fwd-<pid>.log` and silently no-ops if the dir
  doesn't exist — `mkdir -p` before trusting an empty log. The
  gather-branch dispatch log's `path=FUSED/FAST/LEGACY` labels the Q8
  K-buffer SOURCE, not direct-vs-gather; the direct branch logs
  `forward_paged DIRECT_PAGED:`.
- The review's ds4 "40/40 miscount" claim was re-checked:
  `pytest --collect-only` collects exactly 40 (now 46) — the devlog
  counts were literal all along.

---

## 2026-08-27 — round 3: LEGACY=0 Q8 side buffer aliased into the fp16 K half (desync blocker resolved)

**VERDICT:** SHIPPED (LEGACY=0 now serves coherent output at the
default uncapped config with zero extra KV memory; the direct-paged
e2e gates are now real measurements: Phase C clip +3.6% @
pp8192/B=2, B=4 KVSPLIT +1.8%). · **GATE:** 46/46 suite (new
alias-layout test) + LEGACY=0 default smoke + prefix-cache smoke +
LEGACY=1 regression smoke + pp8192 clip A/B (8 samples) + B=4
KVSPLIT S=1/2/4 (12 samples).

## HYPOTHESIS

The "Q8 side-buffer desync" that blocked LEGACY=0 is a misdiagnosis:
the boot-I "incoherent prompt-echo garbage" is this model's *normal*
greedy output style — the persisted logs of the clean hybrid/
all-CUSTOM gate runs are word-identical in style (to=self echo +
meta-reasoning) to the run we declared broken. The real defects:
(1) the Q8 side buffer was a separate lazy `torch.empty` (~17% of KV
bytes) allocated after profiling → outside the pool budget → OOM on
uncapped high-util runs; (2) it could be derived against the
profile/dummy cache geometry (1890 vs 471 blocks observed in a
probe); (3) the first alias fix introduced its own bug: the alias
identity check used `is` on the per-call `kv_cache.unbind(1)` view
(a fresh object every call, never equal) → re-alias + zero-fill of
the whole K half on *every* `do_kv_cache_update` (52×/step) → the
Q8 bytes of all previously written slots wiped after each triton K
write (the Q8 write only restores the current step's slots) → fully
degenerate "to the the the same same…" loops.

## What was done

- **Alias, not allocation** (`gfx906_fa_backend.py`): `_k_cache_q8`
  is a strided uint8 view of the fp16 K half
  (`key_cache.view(uint8)[:, :, :, :bytes_per_row]` — the 88 B Q8
  row fits the 256 B fp16 row). Zero extra KV memory: in LEGACY=0
  the fp16 K half is written but never read, so its bytes double as
  Q8 storage. Page copies (prefix-cache COW) move the Q8 bytes with
  the page → the fail-closed prefix-cache guard in
  `get_cudagraph_support` removed (now a warning). All consumer
  kernels (`reshape_and_cache_q8`, `gather_paged_kv_q8`,
  `forward_paged_direct`) are fully stride-parameterized — verified
  against the real backend tensor, including the 2D-byte head
  stride of the unbind view.
- **Memory-based identity**: re-derive the alias only when
  (data_ptr, shape, strides) change (profile dummy → live pool,
  layout re-slices); never per-call.
- **No zero-fill, ever**: zeroing the Q8 region of a re-derived
  alias clobbers in-use context (re-derivation can happen mid-run
  on live memory). Unwritten slots are never read — attention
  seq_lens only cover written slots — so the fill buys nothing.
- **Test**: +`test_paged_direct_and_gather_on_q8_aliased_into_fp16_khalf`
  — mirrors the backend alias exactly (strided view, production
  write order fp16-K-then-Q8), checks `forward_paged_direct` (B=2,
  window + clip, unaligned starts) and the fused
  `gather_paged_kv_q8` against torch refs → suite 46/46.
- **Docs**: README LEGACY/KVSPLIT/WINDOW_CLIP rows rewritten
  (no more "desyncs — do not use"; per-path safe KVSPLIT values),
  supported-models table gains the Muse-Glimmer row,
  `supports_sliding_window` envelope note updated (W=2048 covered
  by the direct-paged clip tests + e2e, not a gather unit case).

## Gate result

| run | config | result |
|---|---|---|
| suite | `test_gfx906_fa.py` | 46/46 |
| LEGACY=0 default smoke | graph, util 0.93, **uncapped** (95,012 pool), maxlen 8192 | coherent, no OOM; 4 completions in 12.7 s (boot-I: this config OOM'd) |
| LEGACY=0 + prefix cache ON | same; shared prefix, 2 divergent tails (COW fork) | coherent in both branches (COW alias invariant) |
| LEGACY=1 regression smoke | default | coherent (391 correct) |
| Phase C clip A/B (direct-paged) | pp8192/B=2/tg256, prefix off, LEGACY=0, 0.75 GiB KV cap + bt1024 | ON 5.758/5.721/5.710/5.693 → **5.72** vs OFF 5.537/5.518/5.508/5.505 → **5.52** = **+3.6%** |
| B=4 KVSPLIT (direct-paged) | pp2048/tg256, B=4, prefix off, LEGACY=0 | S=1 **20.16** → S=2 **20.50** / S=4 (default) **20.53** (+1.8%); at parity with the gather record 20.59 |

The clip ON/OFF delta is itself the evidence the direct-paged path
executed (the clip exists only there) — the first non-null
direct-paged e2e measurement on this model. B=4: the batch-aware
default (S=4) is already optimal at this batch (S=2 within noise);
the e2e split-K gain is small because FA is ~5–8% of the GEMM-bound
B=4 step. Cross-session drift note: these run in boot J, a different
boot than the gather 20.59 record — compare trends, not absolute
values.

## ERRATUM (round 2)

The round-2 entry's "LEGACY=0 smoke (fresh)" paragraph — "incoherent
prompt-echo garbage … the Q8 side-buffer desync is still live" —
rests on a misdiagnosis: that text is the model's normal greedy
style (boot-I clean-run logs match it word for word). The garbage
seen before this fix ("to the the the same same…") is a *different*
failure — the per-call re-alias zeroing above — not a "desync".
Round 2's "blocking item: fix the desync" is closed by this entry;
"direct-paged unreachable in any working serving config" is
withdrawn (it is reachable and gated).

## Notes / open items

- **TP=2 serving (2026-08-27, boot J)**: the uncapped util-0.82 config
  OOM'd the first real 4096-chunk prefill (`gptq_gemm` `aten::empty`):
  the 9.02 GiB/GPU pool (1.36M tok) + 14.48 GiB steady state left
  <7.2 GiB of the 31.98 GiB physical, and the runtime bt4096 inductor
  prefill buffer exceeded it (profiled peak only 2.73 GiB — the
  warm/cold gap is far larger than the 27B 0.16 GiB case at this
  model size). Serving needs `--kv-cache-memory-bytes 6442450944`
  (6 GiB/GPU ≈ 900k tok, still 3.5× the 256k max). The OOM teardown
  force-killed the TP=2 workers mid-P2P op; the next two relaunches
  failed with `hipErrorLaunchFailure` (the documented TP=2 SIGKILL
  wedge; TP=1 canary healthy between) → session stopped, reboot
  required (degradation.md boot J, 17:40–17:52).
  **Boot K (18:16Z)**: canary 38.8 t/s; 1st launch hit the chronic
  weight-load hang (GPU1 fence timeout → `GPU reset(1)`, 18:23:47);
  retry loaded clean and validated the cap through capture — **KV pool
  904,164 tokens (exactly the 6 GiB), graphs 0.89 GiB** (vs 1.28
  uncapped) — then the process group died silently because the
  launching shell call was operator-interrupted (not HW; no kernel
  events, VRAM released; degradation_details.md 18:23–18:29). Operator
  relaunch (128k max, 6 GiB cap = 848,301–884,644 tok):
  - **bt4096 OOMed the first 4096-chunk prefill again** (18:41:32,
    same 254 MiB last-straw alloc as boot J, free: 0) → the compiled
    prefill transient scales ~linearly with the chunk; bt4096 needs
    >10.6 GiB per-GPU headroom at TP=2 (4 data points, incl. TP=1
    bt1024-OK / bt2048-OOM, fit it; degradation.md 18:41).
  - **bt2048 relaunch: OOM site cleared** — 4106-token prefill
    ttft 10.2 s (401 t/s), 0 OOM, coherent output (18:55).
  - **Grid** (`docs/gfx906/_bench_serve_grid_gfx906.py`, 3 samples,
    19:0x): prefill 542/491/438 t/s @2k/8k/16k; decode B=1 (filler,
    ngram 100 % acceptance — a ceiling) 114.6/79.1/57.0 t/s @2k/8k/
    16k (tg256), 112.0/79.2/56.8 (tg512); B=4 @2k/256 45.3 aggregate
    (~11.3/req); real-prompt checks ~11.5/req. README row updated
    with the working TP=2 example + these numbers.
  - **Open (operator question):** is the >10.6 GiB first-prefill
    transient caused by our custom FA/kernels? Partial evidence says
    no: (a) the 532 MiB inductor-buffer OOM predates the window work
    (boot I 22:30, hybrid config with Triton FA on sliding layers);
    (b) the 27B GDN hybrid serves bt4096/0.82/256k on the same
    all-CUSTOM stack; (c) our FA runs as a *split op* outside the
    inductor segments — the failing alloc is an inductor-segment
    buffer for `gptq_gemm` (Exllama); (d) Q8 side view is an alias
    (0 bytes). Unexplained: geometry only accounts ~1.3× of the
    27B→Muse gap. Definitive probe written
    (`/local/tmp/muse/probe_oom_attribution.py`: memory-snapshot owner
    breakdown at OOM; arms custom / rocm_attn / enforce_eager) —
    needs a free GPU (server teardown).
- LEGACY=0 is validated but stays **experimental**; the default
  remains LEGACY=1 (gather, the validated serving mode). Flipping
  the default is a roadmap decision after a longer bake.
- Direct-paged is B-gated: B=1 decode is gather (no clip,
  KVSPLIT=16); B≥2 is direct-paged. Window clip on the gather path
  (B=1) and per-row prefill clip remain open — recorded in
  `roadmap-more-models.md`.
- `muse_glimmer_opt2_code_rev_qwen.md` open items → roadmap:
  #8 device-side `kv_start` clamp, #10 overflow-free cutoff form
  (4 LOCKSTEP sites), #4 long-context split-K accuracy (L=16k–32k,
  split 8 vs 1) + per-call `o_part` allocation / dead `o_meta`.


## 2026-08-27 — round 4: OOM-attribution probe — the first-prefill transient is OUR FA's prefill path, not inductor

## HYPOTHESIS

The >10.6 GiB/GPU transient that OOMed the first 4096-token prefill
chunk under TP=2 serving (boot J 17:40, boot K 18:41; 254 MiB
last-straw `aten::empty` in a `gptq_gemm` inductor segment, free: 0)
is owned by the **inductor piecewise-compiled prefill graph** (the
failing allocation sits in an inductor segment), not the custom
gfx906 FA backend.

## GATE

Three-arm in-process probe (`/local/tmp/muse/probe_oom_attribution.py`,
TP=1, 0.5 GiB explicit KV cap, PP=4097 so the first prefill chunk is
4096 = the OOM site; peak transient = `max_memory_allocated` delta
around the cold first prefill):

| arm | attention | compile | peak transient | outcome |
|---|---|---|---|---|
| custom | our gfx906 FA | inductor | **3.785 GiB** | OOM (last straw 508 MiB, free: 0) |
| rocm | ROCM_ATTN (Triton FA) | inductor | **1.035 GiB** | survived (4.77 GiB free after) |
| eager | our gfx906 FA | eager | **4.506 GiB** | OOM (0.37 GiB free after) |

## VERDICT

**HYPOTHESIS REJECTED as stated.** Compile held at inductor: swapping
our FA for Triton FA removes **2.75 GiB** of the transient (3.785 →
1.035) — the custom FA's prefill path is the **dominant owner**. FA
held constant: inductor is **0.72 GiB smaller** than eager (3.785 vs
4.506) — inductor's memory planner reuses activation memory; eager
materializes more. The boot J/K “inductor gptq_gemm” OOM frame was
where the memory ran **out**, not the owner: the 508 MiB Exllama fp16
M×N output (508 = exactly 2× the TP=2 254 MiB last straw; N TP-split,
~[4096, 32512] fp16 = 254×128 cols) is the model's own GEMM output
landing last on an exhausted allocator. (Resolves the boot-I “532 MiB
inductor OOM” datum: 532,676,608 B is this same 508 MiB alloc,
misread as 532 MiB.)

**Root cause FOUND (per-layer instrumentation, `attr_tp1_custom15/17`):
the q_pad buffer was per-IMPL, and v1 creates one backend impl per
attention layer.** `_ensure_forward_buffers` grew `self._q_pad_buf`
([num_seqs, Hq, Sq_pad, D] fp32 = **256 MiB** at the 4096-chunk:
metadata pads num_seqs to max_num_seqs=4) on each impl's FIRST
prefill call and the capture-latched retire policy keeps every
generation alive → 52 impls × 256 MiB = **13.3 GiB** of duplicate
buffers (TP=2: 16 heads/GPU → 6.7 GiB/GPU). The per-layer probe
showed exactly +256.0 MiB net at EVERY attention call (monotone
23.389 → 25.690 over 9 calls; OOM at ~call 10), while the C++
binding itself is clean (+64 MiB churn, same out_ptr reused —
`probe_fa_reuse.py`: 52 calls, flat) and the Python wrapper is clean
(+64 MiB `out_flat`, freed). This also explains the boot J/K
signatures: "transient ~linear in the chunk" (buffer ∝ Sq_pad ∝
chunk), bt2048 surviving (64 MiB/impl → 3.35 GiB total), and the
2026-08-26 comment that already noted the "x52 layers = 14 GiB,
first-request OOM" pathology (that fix only stopped the DECODE-time
dim0 growth; the per-impl prefill duplication remained).

**Fix: q_pad buffers are now ClassVar (shared across all impls, one
set per worker — the pattern the gather buffers already use, which
had the identical bug fixed earlier).** `vllm/gfx906_fa/
gfx906_fa_backend.py`: `_q_pad_buf` / `_q_pad_decode_buf` /
`_q_pad_retired` / `_q_pad_captured` → ClassVar; `_ensure_forward_
buffers` → @classmethod (num_heads/head_size now parameters). The
q_pad lifecycle test was rewritten with the class-state
snapshot/restore pattern (the gather-buffer tests' pattern) so the
grow sequence starts from a clean shared buffer.

**Verification status:** unit gate **PASS — 51/51** (incl. the
rewritten `test_q_pad_buffer_survives_capture_then_prefill_grow`
capture→grow→retire→replay sequence). Probe verification (custom
arm re-run: expect one-time 256 MiB grow, then flat, survival at the
0.5 GiB KV cap, transient ≈ model core + 0.26 + churn) is
**DEFERRED POST-REBOOT**: both attempts hit the boot-K burst wedge
(custom18_fixed 21:58, retry custom19 22:06 — 2nd consecutive launch
failure, GPU0 left 24.9 GB zombie VRAM; degradation.md 21:58/22:06,
GPU work stopped). It is the first post-reboot step.

Ruled out along the way: `o_part` KVSPLIT partials (the binding
forces `kv_split=1` for `seq_q>2` — no such alloc at prefill),
per-call binding/wrapper churn (flat; same buffer VA reused), Q8
side view (alias, 0 bytes), gather buffers (already class-level,
+0.0 MiB per call). Dead ends recorded: this torch build's memory-
snapshot stack capture returns only `?:0` unwind frames (no user
frames) and kineto records no device-memory events on this ROCm
stack — the per-layer `memory_allocated()` hook + segment-snapshot
diff is what worked.

## Evidence — the host-crash detour (cost a session; keep for the skill)

Fresh in-process compiles on this box (torch 2.10-dev venv) die in
order, before ever reaching the OOM site:

1. **AOT is ON by default on torch ≥ 2.10** (`use_aot_compile()` in
   `vllm/envs.py`): its out-of-process compile workers die with
   “Could not find an active GPU backend” on every fresh compile.
   Servers are unaffected (compile once on a clean boot, then warm AOT
   cache). Fix: `VLLM_USE_AOT_COMPILE=0`.
2. **Inductor writes `async_compile.wait()` into every generated
   wrapper** (this torch) and its pool defaults to the FORK
   SubprocPool (`TORCHINDUCTOR_WORKER_START=subprocess`) — HSA is not
   fork-safe after the parent initialized HSA → same GPU-backend error
   in the child.
3. **`TORCHINDUCTOR_WORKER_START=spawn` also failed** — but
   differently: the spawned child could not init HSA at all while the
   parent's HSA was healthy (at the failing exec: parent
   `is_available=True, bad_fork=False, devcount=1`, triton backends
   fine, `_is_backend_active("amd")` True in parent). A standalone
   spawn child DID init — state-dependent (boot K had 3 GPU incidents
   by then; degradation.md 20:40). Workaround: replace
   `AsyncCompile.process_pool` with a `ThreadPoolExecutor` (parent
   threads; HSA already up) — also keeps compile transients inside the
   measured process.
4. **`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` required**: the rblock
   variant-compile path (`_dynamic_scale_rblock` → direct
   `triton.compile`) crashes in our triton-gfx906 fork with
   `AttributeError: 'NoneType' object has no attribute '__code__'`
   (`get_jit_fn_file_line` → `JITCallable.fn is None`) — first compile
   fine, the *variant* compile broken.

## Notes / pending

- Post-OOM segment snapshot (`oom_snap_custom_tp1_pp4097.json`):
  31.30 GiB reserved / 26.47 allocated, 589 segments; top blocks
  2×2566 MiB + 256 MiB weight shards. History-armed re-run
  (`attr_tp1_custom11_hist.log`) reproduced the OOM byte-for-byte
  (3.785 GiB / 532,676,608 B last straw) but the snapshot frames
  are empty unwind-only stacks — stack capture is broken in this
  torch build, and kineto records no device-memory events on this
  ROCm stack either. The working instruments: per-layer /
  per-binding `memory_allocated()` hooks + segment-snapshot diff.
- The >10.6 GiB/GPU TP=2 serving OOM (boot J/K) is the same bug at
  16 heads/GPU (6.7 GiB q_pad growth + inductor buffers > 10.6
  headroom). The ClassVar fix should let bt4096 serve under the 6
  GiB KV cap — **re-validate the boot K launch recipe (bt2048
  workaround droppable, 6 GiB cap shrinkable) post-reboot, after the
  probe verification passes** (pending list in degradation_details.md
  22:06).

## 2026-08-27 — round 5: M1 window clip on the gather path (B=1 decode) — implemented + unit-gated, e2e pending

## HYPOTHESIS

Extending the Phase C window clip to the gather path — via an
absolute-position gather layout (gather writes only rows
`[kv_start, seq_len)` at buffer index == absolute position) plus the
FA kernel starting its k-loop at the floored `kv_start` — is
**bit-identical** (no mask changes; the floored start matches the
existing paged Phase C floor block) and cuts the gather + FA work for
B=1 long-context decode (the kernel micro-bench showed −48% FA time
at L=8k/W=2k).

## What was done

- `fattn-q8.cuh`: new `kv_start` param (after `window`);
k0_base floor block **LOCKSTEP-copied** from the paged Phase C floor
  (floor is for bit-identity: an unaligned first tile re-associates
  the fp16 reduction, ~5e-4); both loop starts shifted to
  `k0_base + blockIdx.y*nbatch_fa`.
- `gfx906_fa_gather.cu`: persistent gather takes `kv_start` (int32
  [B], optional); per-seq gather start + 128-row margin
  (`GATHER_CLIP_MARGIN`, ≥ max nbatch_fa=128 so the floored start is
  always materialized); grid-stride kernel → no host row-count change.
- `gfx906_fa_launcher.cu` / `gfx906_fa.cpp`: signatures + bindings
  (`py::arg("kv_start") = nullopt`; int32 [B] validation mirroring
  paged-direct).
- `gfx906_fa_paged.py`: `GFX906_FA_GATHER_CLIP` kill switch (default
  1); `kv_start = max(0, q_abs + 1 − window)` per seq, int32
  contiguous, computed in the persistent sub-path and passed to both
  the persistent gather and `forward`; `_DOUBLE_CHECK` requires
  `kv_start is None` (the double-check kernel's window test assumed
  full-gather positions — keep the invariant explicit).
- M1 v1 = **persistent sub-path only** (B≤16, all Sk = all current
  server traffic); other gather sub-paths pass `kv_start=None`
  (full gather, still correct).

## GATE

Bit-identity unit tests (clip ON vs OFF, direct dispatch forced off
so B=2 really uses gather) + full suite + e2e pp8192/B=1 tg256 A/B
(`GFX906_FA_GATHER_CLIP` 1 vs 0, record recipe).

## VERDICT

**Unit gate PASS** (e2e pending — GPU0 busy with the round-4 probe):
5/5 new tests (Sq=1/6 × B=1/2 at L=4353/W=2048 unaligned starts
2305/2300 + short-ctx inert L=513<W) + full suite 51/51. Kernel-level
sanity: rows `[2305, L)` bit-identical, rows `[0, 2305)` skipped by
gather. E2e A/B queued.

---
Copyright Kevin Read <me@kevin-read.com>
