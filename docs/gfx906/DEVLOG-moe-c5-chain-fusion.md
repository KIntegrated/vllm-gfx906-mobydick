# C5 — fuse the shared-expert chain: the stated mechanism is absent in graphed serving, and the whole kernel-side ceiling is 1 % of the step (NO-GO)

**Branch:** `gfx906/c5` off the 0.30 line `1442385804`.
**Issue:** #21 `C5: fuse the shared-expert chain (150–250 µs)`.
**Model measured:** Ornith-1.5-35B-A3B-AWQ-INT4 — `Qwen3_5MoeForConditionalGeneration`,
hidden 2048, 40 layers, 256 experts/top-8, `moe_intermediate_size` 512,
`shared_expert_intermediate_size` 512, mixed GDN/full attention (the model the
C-series' MoE kernels target).
**Gate:** the issue's own gate is "bit-correctness plus a serving A/B (verify,
do not assume)". Everything below is the **pre-gate** — the cheap measurement
that has to be run before any kernel work, because a fused kernel can only be
worth what the chain costs.

## What the issue assumes

> "The shared expert is already dense fp16. Its w13, activation, and w2
> operations are individually near their measured GEMV/LLMM1 optima; the
> remaining opportunity is a single chain kernel that removes **two launches per
> layer**. Expected benefit is roughly **150–250 µs** after the serving
> critical-path discount."

Two things must both hold, and both are measurable without writing a kernel:

1. **the launches must still cost something in the regime we serve in.** Decode
   steps are graph-captured in this fork (`FULL_AND_PIECEWISE`); C3's log
   directly observed *graph nodes* for MoE-layer memsets, i.e. the MoE path is
   inside the capture. A launch removed from a captured graph is not a saving.
2. **the kernels must be away from the bandwidth floor.** A fused kernel cannot
   beat its own floor, so (in-graph chain time − floor) is a hard ceiling on
   *any* kernel-side win, fusion or not.

## Method

`benchmarks/kernels/gfx906/bench_shared_chain_gfx906.py` (new; kept as the
instrument for any µs-scale chain on this host):

- **dispatcher-faithful per-op calls** — the chain is exactly what
  `Qwen3MoeMLP.forward` runs: `gate_up_proj` → `SiluAndMul` →
  `down_proj` (`torch.ops._C.silu_and_mul` for the activation).
- **eager medians** = block-of-10 CUDA-event medians (launch-inclusive).
- **launch-free per-op times** = difference of 40-copy CUDA-graph captures:
  same-stream capture records stream order as dependencies, so the copies do not
  overlap, and replay has no per-launch CPU cost. This separates "the kernel" from
  "the launch" without a profiler (all HSA tracing paths are dead on this host).
- **clock control**: a memory-bound kernel cannot hold mclk up, so a small
  duty-cycled compute matmul runs on a second stream. mclk is read from
  `/sys/class/drm/card*/device/pp_dpm_mclk` (see Environment notes).

## Which shape set is the decision-relevant one

`ROADMAP.md`'s reference workload is **one MI50**: `Qwen3.5-35B-A3B-AWQ`, 40 MoE
layers, B=1 decode step ≈ **15 ms at 66.5 t/s**. So the **TP=1 (full) shapes are
the ones the chain actually runs in the served config** — that row drives the
verdict. The TP=2 rows are informational: they apply when the MoE model is served
across two cards, and they are where Finding C came from.

## Result — the chain, at both TP shape sets

| shape set | op | kernel | eager µs | graph µs | floor µs |
|---|---|---|---|---|---|
| TP=1 (full, M1-sprint shape) | gate_up 1024×2048 | LLMM1 rpb4 | 13.54 | **6.73** | 5.10 |
| | act 1×1024 | `silu_and_mul` | 9.49 | **2.22** | — |
| | down 2048×512 | dense_gemv kc512 | 14.13 | **4.78** | 2.55 |
| | **chain** | | **37.69** | **13.73** | 7.65 |
| TP=2 (**production per-rank**) | gate_up 512×2048 | LLMM1 rpb4 | 13.79 | **4.35** | 2.55 |
| | act 1×512 | `silu_and_mul` | 9.49 | **2.50** | — |
| | down 2048×256 | LLMM1 rpb4 | 13.59 | **3.80** | 1.28 |
| | **chain** | | **36.59** | **10.64** | 3.83 |

Per step (×40 layers, TP=2): eager **1464 µs**, graphed **426 µs**, floor **153 µs**.
Device under test confirmed HOT (mclk 800 MHz) during the run; the idle peer sat
at 350 MHz.

### Finding A — the launch share is ~1 ms/step, and it exists **only** outside graphs

At TP=2 the same three ops cost **36.59 µs/layer eagerly and 10.64 µs/layer
graph-captured** (TP=1: 37.69 vs 13.73). The difference — **1038 µs/step** — is
launch cost that a fused kernel could remove *in eager mode only*. In the served
configuration those launches are already graph nodes, so the mechanism the issue
names is worth ~0 where it would be measured. This is the quantitative form of
the M1 log's recurring "eager wins, graph loses" pattern, and it is the same
reason C3's in-graph node fold cleared correctness but produced no gain.

### Finding B — what remains is node/kernel time, and at the reference config it is 1.6 % of the step

At the **reference config (TP=1)** the graphed chain runs at **13.73 µs/layer
against a 7.65 µs/layer floor** (1.8×); the entire residual — every µs available
to a better kernel, fused or not — is **6.08 µs/layer = 243 µs/step**, i.e.
**1.6 %** of the 15 ms B=1 step (66.5 t/s). At TP=2 the same ratio is worse in
absolute terms and smaller as a share: 10.64 vs 3.83 µs/layer → 6.82 µs/layer =
**273 µs/step**, 1.0 % of a 26.3 ms step. Inside the residual, at TP=1:

- the **activation** is the most attackable single piece: **2.22 µs in-graph for
  2 KB of payload** — pure node/ramp cost, no bandwidth content (89 µs/step
  across 40 layers; 2.50 µs / 100 µs at TP=2);
- the **down projection** runs 1.9× floor (4.78 vs 2.55 µs) on the shipped
  kc512 GEMV path;
- a barrier-based single-kernel design (one node instead of three) pays a
  grid-wide barrier of ~1–2 µs, eating ~a third of the residual;
- calibration from this same chain: the shipped S3 down-GEMV won **70 µs/step**
  (−15…25 % on one shape) and was accepted as a win — so ~150–240 µs/step is
  above this project's shipping bar, not below it.


### Finding C — the shipped S3/A2 GEMV dispatch is dead at TP=2 (unexpected; not a C5 result)

`Qwen3MoeMLP` does **not** pass `disable_tp`, so the shared expert is
tensor-parallel-sharded: gate_up is column-parallel, down is row-parallel. The
`_llmm1_tiny_m` GEMV rules are keyed on the **full** shapes —
`weight.shape[1] == 2048 and m in {1, 256, ≥2048}`, and
`weight.shape[1] == 512 and m == 2048` — so at TP=2 both miss:

- gate_up per rank is `[512, 2048]` → `m = 512`, rule needs `m ∈ {1, 256, ≥2048}` → **LLMM1**;
- down per rank is `[2048, 256]` → `shape[1] = 256 ≠ 512` → **LLMM1**, and there
  is no other path: `dense_gemv_gfx906` rejects K=256 outright
  (`kchunk must be 512, 1024, 2048 or 4096`).

So the M1 sprint's S3 win ("shared down [2048,512] → kc512/r2 5.6–5.7 µs vs
LLMM1 6.7–7.7") and the A2/A2a dispatches were measured on **TP=1** shapes and
**do not run in our TP=2 serving config**. The sprint's in-model census
(`dense_gemv_kernel<2,512> 39.7/step`) corroborates TP=1. This is a dispatch
audit item in its own right — a TP=1-derived kernel win that silently does not
apply at the TP we serve — and it is filed here rather than folded into C5.

## Verdict: the stated mechanism is refuted; the item is reframed as node/bandwidth fusion (not closed)

**The issue's rationale does not hold.** "Removing two launches per layer" is
worth ~1 ms/step *eagerly* and **0 in the graphed config we serve**: 37.69 µs/layer
eager vs 13.73 µs/layer graphed at TP=1 (**958 µs/step** of launch cost that the
capture already amortizes; 1038 µs/step at TP=2). Implementing C5 for its stated
reason would measure nothing — that is exactly C3's outcome.

**What does hold** is a node/kernel-time residual of **6.08 µs/layer = 243 µs/step
at the reference config = 1.6 % of the 15 ms step** above the 7.65 µs/layer
bandwidth floor, against a project calibration of a *shipped* 70 µs/step win on
this same chain (S3's down-GEMV). So the item is worth pursuing — as **node-count
and bandwidth-path fusion**, not launch elimination.

**Next step, with an early-kill gate.** The cheapest real version is not the
barrier kernel; it is **folding the activation into the gate_up producer**:
`silu_and_mul` costs 2.22 µs in-graph (89 µs/step across 40 layers) for 2 KB of
payload — pure node/ramp cost, no bandwidth. The merged `[gate | up]` output
layout makes this barrier-free: a thread that owns row `i` and row `i + d`
combines the pair in registers, so one kernel produces the activated intermediate
and the standalone activation node disappears from the chain.

- **Gate:** in-graph standalone, `fused(gate_up+act)` must beat `gate_up + silu`
  by **≥ 2 µs/layer (≥ 80 µs/step at TP=1)**; then bit-correctness, then a
  serving A/B. Failing that, the barrier variant is the only remaining idea and
  must clear the same bar as a prototype before any integration.

**Tree state so far:** branch `gfx906/c5` carries the probe + this log only; zero
production code touched.

## Residue — measured pieces that stay on the shelf (reopen gates in-line)

- **Barrier-fused single chain kernel** (3 nodes → 1). Ceiling is the same
  243 µs/step at TP=1; the grid barrier (~1–2 µs/layer) eats a third of it.
  **Reopen gate:** only after the cheap variant lands, and only with a prototype
  that beats `13.73 µs/layer` in-graph by ≥2 µs/layer.
- **`kchunk=256` support in `dense_gemv_gfx906` + a TP=2 dispatch rule**, so S3's
  down-GEMV also fires at TP=2 (see Finding C). **Reopen gate:** part of a TP=2
  dispatch audit — the reference config is TP=1, so this is latent, not active.

## Environment notes (both cost a run)

- **`rocm-smi --showclocks` BLOCKS uninterruptibly while a graph-replay loop
  saturates the GPU** — the child sits in D state, so `subprocess`'s timeout never
  fires and the sampler thread returns nothing. Two runs printed "mclk: no
  samples" before this was found. Sample
  `/sys/class/drm/card*/device/pp_dpm_mclk` instead: a plain file read, with `*`
  marking the active level, and it needs no driver ioctl.
- **A duty-cycled heater is required and must be small.** With no heater mclk sits
  at 350 MHz (cold, ~2.5× inflated); a 512² matmul at ~50 % duty holds 800 MHz
  while leaving the machine nearly free. A big continuous matmul holds clocks but
  occupies most CUs and inflates the timed kernels — two different errors.
- **Never quote an eager per-op number as kernel time.** Identical kernels cost
  13–14 µs eagerly and 4.4 µs inside a graph on this host. The older warning
  "standalone ≠ production" is, quantitatively, a launch-cost artefact of eager
  measurement (plus CU contention).
