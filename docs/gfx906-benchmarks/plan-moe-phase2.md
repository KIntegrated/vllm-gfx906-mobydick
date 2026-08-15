# Plan — MoE phase 2 on gfx906 (branch `gfx906/moe-opt`)

Phase 1 shipped a custom W4A16 MoE GEMM (`csrc/rocm/moe_q_gemm_gfx906.cu` +
`Gfx906WNA16Experts`): end-to-end **3.49 → 18.79 tok/s** (prefill ~450 →
~2140 tok/s, decode 3.72 → 19.7 tok/s). This plan orders the remaining
optimization candidates by expected wall-clock value per unit risk, with a
decision gate after each step.

## Current state (post phase 1)

Profile: pp=512/tg=64, in-process torch.profiler, eager. Self CUDA total
2.0 s; **Self CPU 4.3 s → the eager run is launch-bound**, so GPU-side work
only moves wall-clock where the GPU is actually the bottleneck (prefill, and
decode under cudagraphs/high batch).

Per-decode-step GPU budget (profile totals ÷ 64 decode steps; MoE row uses
the decode-only call subset):

| Item | ms/step | notes |
|------|---------|-------|
| aiter `LLGemm1` dense GEMMs | ~7.2 | 223 calls × 31.7 µs (attention proj, shared experts, router, GDN in/out) |
| paged attention | ~2.9 | 9–10 full-attn layers × 290 µs |
| **MoE GEMM (ours)** | **~2.1** | ~80 calls × ~27 µs; floor ≈ 23 µs/step (12 MB @ 512 GB/s) |
| elementwise pile (copy/mul/add/mean/pow/sigmoid/rsqrt) | ~6–7 | vLLM glue + norm ops |
| `rocm_unquantized_gemm` (Triton matmul) | ~2 | |
| topk_softmax + moe_align | ~1.2 | 40 layers × (18 + 13.5 µs) |
| shared-expert Triton `fused_moe_kernel` | ~0.5 | fp16, generic path |

MoE-specific remaining headroom is therefore: **prefill MoE GEMM** (largest,
GPU-bound), decode MoE parallelism (only visible off-eager), topk+align
fusion, shared-expert routing. The big rows (`LLGemm1`, attention,
elementwise) are paid by the dense 9B model too — out of scope here (see P2-6).

Kernel facts that shape the options (current `moe_q_gemm_gfx906.cu`):
- grid = (token_blocks, N/1024, **K/256**); K-slices already exist and merge
  via packed fp16 CAS atomic-adds into a pre-zeroed output.
- 256 threads × 4 N columns; A tile in LDS (`block_a[BM][256+8]`), B streamed
  from gmem (each weight word read once per expert per K-slice).
- `BLOCK_SIZE_M` chosen at runtime: EM ≤ 32 → 1, ≤ 512 → 4, else 16.
- Micro-bench vs roofline: decode M=1 w13 = 35.5 µs (floor ~16 µs, 8 active
  experts × 1 MB); prefill M=512 w13 = 3063 µs at **~5.6 TFLOPS ≈ 40% of the
  ~13.8 TFLOPS fp16 dot peak** (floor by traffic is only ~523 µs — prefill is
  compute-bound, not bandwidth-bound).

## Ordered candidates

Ordering rule: (1) measure before changing; (2) do GPU-bound wins first
(they pay in the eager bench too); (3) gate decode-side work on the cudagraph
measurement; (4) keep correctness-risky routing changes late and isolated.

### P2-0 — Diagnostics baseline (no behavior change) · effort S · risk none

1. Re-run the micro-bench per (M, gemm) bucket: M ∈ {1, 8, 32, 128, 512,
   2048} × {w13, w2}; record µs/call and achieved TFLOPS / GB/s.
2. `rocprofv3` counter pass on one prefill-sized call (M=512): ALU pipe
   utilization, LDS read throughput, gmem throughput — decide whether the
   40%-of-peak gap is dot-throughput, LDS-A-tile traffic, or atomic epilogue.
3. Register/spill check per `/tmp/gfx906-performance-inspection.md`:
   `llvm-readobj --notes` (or `roc-obj`) on the kernel object for each
   instantiated `BLOCK_SIZE_M`; confirm 256 threads fit without spills at BM=16
   and see what BM=32 would cost.

Exit: one paragraph in the dev log stating which pipe bounds prefill MoE and
which occupancy/spill limits exist. This determines which P2-1 sub-options are
worth trying.

### P2-1 — Prefill MoE tuning · effort M · risk low-medium

Goal: w13 M=512 3063 µs → < ~1.5 ms (≈2×); prefill pp=2048 0.95 s →
~0.6–0.7 s. This is the single largest eager wall-clock win left.

Options, in order of preference given P2-0 findings:
- **a) `BLOCK_SIZE_M` sweep {8, 16, 32}** at M=512/2048 (runtime knob already
  exists; just change the heuristic + re-bench). BM=32 only if P2-0 shows no
  spills.
- **b) M-dependent K-slice size**: currently fixed 256 (grid.z = K/256 → 8
  slices for w13, each output cell gets 8 fp16-CAS contenders). For large M
  try 512 (4 slices) to cut atomic traffic; keep 256 or smaller for small M.
- **c) If LDS-bound**: reduce A-tile re-read amplification — e.g. 128-thread
  blocks with 8 N columns each (same weight traffic, half the threads reading
  the same A rows), or a DPP-based row broadcast per `gfx906-notes.md`.
- **d) If dot-bound**: accept ~40–50% of scalar-dot peak as the gfx906 limit
  for this design and stop (no MFMA on Vega; further gains need a different
  algorithm, out of scope).

Risks:
- *Register spills at BM=32* → silent perf regression. Mitigation: P2-0 spill
  check first; keep 16 as fallback; bench all M buckets before committing the
  new heuristic.
- *Atomic contention changes with slice size* → non-monotonic per-M behavior.
  Mitigation: sweep, don't guess; commit the measured table as the heuristic.
- *Correctness*: low — all variants run through the existing standalone test
  (both source layouts, multiple M/block_m) + full-model sanity generation.

Exit criteria: micro-bench table improved at M ≥ 128 with no regression below;
full-model tg=1 bench shows prefill ≤ ~0.7 s; dev log updated.

### P2-2 — Cudagraph ceiling measurement (no kernel change) · effort S · risk ~none

Add a `BENCH_EAGER` env override to `_bench_gfx906.py` (default 1, unchanged)
and run tg=1/tg=256 with cudagraphs on. Quantifies how much of the 19.7
tok/s decode is launch overhead vs GPU time (profile suggests a ~40–50 tok/s
GPU-bound ceiling).

Risks: near zero — separate measurement, default behavior untouched. Caveat:
cudagraph numbers are **not comparable** to the §1 eager table; label them
"serving-mode" in the README/dev log and don't mix them into the version
comparison. If cudagraph capture fails on gfx906 for this model (e.g. GDN
hybrid layers), that itself is a finding to record.

Exit: one number (decode tok/s with cudagraphs) + a go/no-go for P2-3/P2-4.

### P2-3 — Decode MoE parallelism at small M · effort M · risk medium

Goal: w13 M=1 35.5 µs → ~18–20 µs, w2 33 µs → ~16 µs (decode MoE 2.1 →
~1.1 ms/step). Today only ~8 experts are active → ~64 blocks (w13) / ~32
(w2) on a ~40-CU part; the kernel is latency-bound, not bandwidth-bound
(228 GB/s ≈ 45% of peak).

Options:
- Finer K-slicing for EM ≤ 32 (e.g. BLOCK_KN_SIZE=128 → 2× blocks) — pure
  launcher change.
- 128-thread variant (grid.y doubles: 512 N columns per block) — new template
  instantiation, same algorithm.
- Skip the A-LDS round trip for BM=1 (load `a` straight into registers).

Risks:
- *More K-slices → more fp16-CAS contenders per cell*: contention can eat the
  parallelism gain, and it widens run-to-run non-determinism (CAS ordering is
  already non-deterministic by design; document, don't try to fix here).
- *Occupancy/register shifts* in the 128-thread variant — verify with P2-0's
  tooling.
- *Payoff gating*: in eager mode decode is CPU-launch-bound, so this likely
  won't move the §1 bench number much; it pays under cudagraphs (P2-2) and in
  serving/high-batch. **Do this only if P2-2 shows a meaningful GPU-bound
  residual**; otherwise park it.

Exit: micro-bench M=1/8 improved ≥ 30% with no regression at M ≥ 32; full test
matrix green.

### P2-4 — Fused gfx906 topk + align · effort M · risk high (correctness)

Goal: replace `topk_softmax` (18 µs × 40) + `moe_align_block_size`
(13.5 µs × 40) ≈ 1.2 ms/step with one routing kernel for the fixed shape
E=256, k=8 → target ~0.3 ms/step (~5% of decode step).

Approach: single kernel that reads router logits [M, 256], does softmax +
top-8, and writes `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded`
directly (the block layout is trivial at k=8: sort ≤ 8×M entries).

Risks — **highest in this phase**:
- *Wrong routing = silently garbage output* (no crash, no NaN). Mitigation:
  exhaustive offline test vs the existing Triton path on random logits
  (including ties, all-negative rows, M not divisible by block sizes), diff
  the full sorted-token/expert arrays bit-exactly, plus the standing
  full-model sanity generation.
- *Tie-breaking/ordering semantics*: `moe_align_block_size` output order is
  consumed positionally; must replicate its exact layout (padding sentinel
  values included).
- Scope creep: generalizing beyond E=256/k=8 — resist; specialize, fall back
  to the stock path otherwise.

Exit: bit-exact routing arrays vs reference on the test matrix; full-model
bench unchanged in output, improved in time; dev log notes the fallback gate.

### P2-5 — Shared-expert path · effort S · risk low-medium

The fp16 shared expert currently rides the generic Triton `fused_moe_kernel`
(~0.5 ms/step). A single shared expert is just a dense GEMM (no routing):
route it through the existing unquantized linear path or a direct
`mm`-based call from `Gfx906WNA16Experts`.

Risks: low — numerics are fp16×fp16 either way; still needs the test matrix +
sanity gen. Small gain; do only if P2-1..P2-4 land cleanly and time remains.

### P2-6 — Deferred (not phase 2)

Largest remaining rows are **not MoE-specific** and deserve their own project:
aiter `LLGemm1` dense GEMMs (~7 ms/step; first instrument the 223 call shapes,
then consider a tuned gfx906 path), decode paged attention (~2.9 ms/step;
custom decode kernel), elementwise/norm fusion (~6–7 ms/step; mostly
CPU-launch-side in eager). These help the dense 9B model equally.

## Common protocol (every step)

1. Standalone correctness test (`/tmp/bench/_test_gfx906_moe.py`, both source
   layouts) — must stay ALL PASS.
2. Micro-bench table across all M buckets — no silent regression anywhere.
3. Full-model bench: tg=1 (prefill) and tg=256 (decode) with the standard
   docker recipe (`-e PYTHONPATH=/workspace/vllm`, `vllm-cache` volume).
4. Greedy sanity generation (3 prompts, eyeball coherence).
5. Separate commit per step; update `DEVLOG-moe-opt.md` (results + negative
   findings) and README §1/§4 numbers only for accepted changes.

Decision gates:
- After **P2-1**: if prefill did not improve ≥ 30%, stop tuning the kernel
  and record the dot-throughput limit; move to P2-2.
- After **P2-2**: if cudagraph decode is already ≥ ~40 tok/s, deprioritize
  P2-3/P2-4 (launch overhead was the wall); if it's GPU-bound, proceed in
  listed order.

## Rollback / blast radius

All changes stay behind the existing oracle gate (`GFX906_HIP` selected only
on gfx906 with int4 WNA16 MoE weights); reverting any step is a single commit.
Kernel heuristics default to the phase-1 values on unrecognized shapes, so a
tuning regression cannot break other models/architectures.
