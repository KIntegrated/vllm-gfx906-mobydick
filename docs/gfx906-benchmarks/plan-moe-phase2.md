# Plan — MoE phase 2 on gfx906 (branch `gfx906/moe-opt`)
Copyright Kevin Read <me@kevin-read.com>


Phase 1 shipped a custom W4A16 MoE GEMM (`csrc/rocm/moe_q_gemm_gfx906.cu` +
`Gfx906WNA16Experts`): end-to-end **3.49 → 18.79 tok/s** (prefill ~450 →
~2140 tok/s, decode 3.72 → 19.7 tok/s). This plan orders the remaining
optimization candidates by expected wall-clock value per unit risk, with a
decision gate after each step.

This plan was revised after adversarial review (DS4 + Claude). All critical
and moderate findings were validated; the option ordering and framing were
updated to reflect them. A claim-by-claim merge summary is in
`plan-moe-phase2-merge-notes.md`.

## Current state (post phase 1)

Profile: pp=512/tg=64, in-process torch.profiler, eager. Self CUDA total
2.0 s; **Self CPU 4.3 s → the eager run is launch-bound**, so GPU-side work
only moves wall-clock where the GPU is actually the bottleneck (prefill, and
decode under cudagraphs/high batch).

Hardware (gfx906 SKU table, from ROCm specs + AMD launch material):

| SKU | VRAM | CUs | FP16 datasheet | FP16 measured (dot2) | HBM2 BW | LDS/CU | VGPR file/CU |
|-----|------|-----|----------------|----------------------|---------|--------|--------------|
| MI60 | 32 GB | 64 | 29.5 TFLOPS | — | ≤1 TB/s | 64 KiB | 256 KiB |
| MI50 (32 GB) | 32 GB | 60 | 26.8 TFLOPS | **~20 TFLOPS** | ≤1 TB/s | 64 KiB | 256 KiB |
| MI50 (16 GB) | 16 GB | 60 | 26.8 TFLOPS | — | ≤1 TB/s | 64 KiB | 256 KiB |

Both SKUs: gfx906 ISA, wavefront=64, 40 resident wavefronts/CU max (4 pools ×
10), 32-bank LDS (4-byte bank width).

**Benchmarks in this branch ran on the MI50 32 GB** (60 CU). The devlog header
says "MI60" based on the VRAM amount, but `rocprofv3` agent info reports
`Simd_Count=240 → 60 CUs`, which matches MI50, not MI60 (64 CU). The measured
`v_dot2_f32_f16` peak is **~20 TFLOPS** at ILP≥2 (sclk=930 MHz) — the
datasheet 26.8 TFLOPS is unachievable with dot2 at ILP=1 and the datasheet
29.5 TFLOPS (MI60 figure) was never applicable. Earlier references to "~40-CU
part", "~700 GB/s HBM", "~13.8 TFLOPS", and "29.5 TFLOPS" were all wrong;
effective achieved HBM bandwidth at small M is measured per kernel in the
micro-bench.

Per-decode-step GPU budget (profile totals ÷ 64 decode steps; MoE row uses
the decode-only call subset):

| Item | ms/step | notes |
|------|---------|-------|
| aiter `LLGemm1` dense GEMMs | ~7.2 | 223 calls × 31.7 µs (attention proj, shared experts, router, GDN in/out) |
| paged attention | ~2.9 | 9–10 full-attn layers × 290 µs |
| **MoE GEMM (ours)** | **~2.1** | ~80 calls × ~27 µs; floor ≈ 12 µs/step (12 MB @ ~1 TB/s peak HBM) |
| elementwise pile (copy/mul/add/mean/pow/sigmoid/rsqrt) | ~6–7 | vLLM glue + norm ops |
| `rocm_unquantized_gemm` (Triton matmul) | ~2 | |
| topk_softmax + moe_align | ~1.2 | 40 layers × (18 + 13.5 µs) |
| shared-expert Triton `fused_moe_kernel` | ~0.5 | fp16, generic path; 40 Triton launches |
| **`zero_()` fills (w1_out + output)** | **~0.1–0.2** | 2 launches × 40 layers = 80 extra GPU kernel dispatches in an already CPU-launch-bound path |

MoE-specific remaining headroom: **prefill MoE GEMM** (largest, GPU-bound),
zero-fill launches (structural, pays in eager), topk+align launch-count
reduction, shared-expert Triton removal, decode MoE latency (only visible
off-eager). The big rows (`LLGemm1`, attention, elementwise) are paid by the
dense 9B model too — out of scope here (see P2-6).

Kernel facts that shape the options (current `moe_q_gemm_gfx906.cu`):

- grid = (token_blocks, N/1024, **K/256**); K-slices merge via packed fp16 CAS
  atomic-adds (`atomic_add_pk4_f16`) into a **pre-zeroed** output (the
  pre-zeroing is the source of the 80 extra launches).
- 256 threads × 4 N columns; A tile in LDS (`block_a[BM][256+8]`), loaded with
  **scalar b32 half stores/loads** (1.9–3.9 TB/s measured) — not b128
  (9.5–11.2 TB/s). This is the most likely source of the prefill gap.
- `BLOCK_SIZE_M` chosen at runtime: EM ≤ 32 → 1, ≤ 512 → 4, else 16.
  Switch instantiates {1, 2, 4, **8**, 16}. BM=8 is available but not selected.
  BM=32 is **not instantiated** and will hit `TORCH_CHECK(false)`.
- `block_c[BM][4]` fp32 accumulators per lane: at BM=16 = 64 fp32 VGPRs.
  Dequant constants (z1z16/y1y16 for 4 columns × 2 pairs = 16 half2) add
  ~16 VGPRs → total ~80 VGPRs at BM=16, already near the 128-VGPR limit that
  pins occupancy to 1. BM=32 would require 128 acc VGPRs + dequant → spills
  at any occupancy. BM=32 needs a redesign (n_per_thread=2 or fp32 staging)
  before it is viable.
- fp16 CAS epilogue: each of the 8 K-slices and 8 active experts adds through
  `__float2half_rn` + `__hadd2` — up to 64 fp16-rounding steps per output cell
  at M=512/8-experts/8-slices. Correctness gate currently accepts maxrel ≈ 2%
  because the torch reference is also fp16-approximate. This is a known
  precision debt; fp32 staging (P2-1e) resolves it.
- In-kernel accumulator (`block_c`) is already fp32; fp16 loss enters only at
  the CAS epilogue.
- The RDNA3 parent (`moe_q_gemm_rdna3.cu`) has a `USE_LDS_A` template flag
  that skips the LDS round-trip for BF16/M=1. The gfx906 port always uses LDS.
  For M=1, the LDS stage reads each A element exactly once with no reuse — pure
  overhead.
- Micro-bench vs roofline: decode M=1 w13 = 35.5 µs (latency-bound, not
  bandwidth-bound; measured ~228 GB/s ≈ 23% of ≤1 TB/s peak HBM); prefill
  M=512 w13 = 3063 µs (Phase 1) → 2247 µs (Phase 2 post-tuning) at **~5.9
  TFLOPS ≈ 30% of the ~20 TFLOPS measured dot2 peak** (the datasheet 26.8
  TFLOPS MI50 peak is not achievable with `v_dot2_f32_f16` at ILP=1;
  bandwidth floor at ≤1 TB/s ≈ 380 µs → prefill is issue/compute-bound,
  not bandwidth-bound).

## Ordered candidates

Ordering rule: (1) measure before changing; (2) fix structural launch waste
first (pays in eager); (3) do GPU-bound prefill wins; (4) gate decode-side
work on the cudagraph ceiling measurement; (5) keep correctness-risky routing
changes late and isolated.

### P2-0 — Diagnostics baseline (no behavior change) · effort S · risk none

1. **Micro-bench per (M, gemm) bucket**: M ∈ {1, 8, 32, 128, 512, 2048} ×
   {w13, w2}; record µs/call and achieved TFLOPS / GB/s.

2. **Three-way bottleneck pass** — `rocprofv3` on one prefill-sized call
   (M=512): collect ALU pipe utilization, LDS read throughput, and gmem
   throughput. The P2-0 output must explicitly categorise the prefill
   bottleneck as one of:
   - **(A) LDS-bound** — high gmem throughput, low ALU util, low LDS BW →
     b32 A-staging is the bottleneck; ALT-A (b128 LDS) is the first fix.
   - **(B) Dot-pipe-bound** — high ALU util, LDS BW saturated →
     occupancy tuning (n_per_thread=2 or cut VGPRs) is the lever.
   - **(C) Neither saturated (under-occupied)** — occupancy = 1 is pinning
     the pipe; VGPRs must be cut to raise it. Neither ALU nor LDS is
     the per-se limit — latency hiding is missing.
   Outcomes (A) and (C) both point to the b128 LDS path (ALT-A) as the
   first move; (B) points to the n_per_thread reduction (ALT-C). In all
   three cases, "dot-ISA ceiling → accept 40%" is not the immediate
   conclusion — it is the conclusion *after* occupancy/LDS work fails to
   improve.

3. **VGPR table per BM**: run `llvm-readobj --notes` (or `roc-obj`) on the
   compiled kernel object for each instantiated BM ∈ {1, 2, 4, 8, 16}.
   Record `vgpr_count`, `sgpr_count`, occupancy. Output: a table of
   (BM, VGPR, occ, viable-for-phase-2?) that directly drives P2-1 option
   selection. Note that BM=32 is not instantiated and will almost certainly
   exceed 256 VGPRs without a redesign.

4. **Launch-count baseline**: record the number of GPU kernel dispatches per
   forward pass using `rocprofv3 --hip-trace` (or similar). This is the
   denominator for every launch-reduction step in the plan.

Exit: the three-way bottleneck table + VGPR table + launch baseline, written
to the dev log. The P2-1 option list is pruned based on these results before
any code is written.

### P2-0b — Zero-fill launch elimination · effort S · risk low ← **new, do before P2-1**

Fold `w1_out.zero_()` and `output.zero_()` into the GEMM kernel by having the
first K-slice block for each output tile clear its own cells before the atomic
epilogue (a conditional `if (blockIdx.z == 0) output[...] = 0` before the CAS
loop, or a pre-clear pass in the epilogue). This removes 2 kernel launches × 40
layers = **80 dispatches per forward pass** from the CPU-launch-bound path,
regardless of eager vs cudagraph mode. It also removes the external correctness
dependency on a pre-zeroed workspace.

Risks: low — the clearing must happen before any K-slice writes to that tile.
The natural ordering (gridDim.z > 1, each block owns a K-slice partition) makes
this checkable: the first block in grid.z clears, all others CAS-add. The
`atomic_add_pk4_f16` CAS is already a read-modify-write; the clear is a plain
store that completes before any sibling's atomic. Test by removing the external
`zero_()` calls from `Gfx906WNA16Experts.apply` and running the correctness
suite.

Exit: correctness test ALL PASS with external zero_ calls removed; launch-count
baseline (P2-0) shows −80 dispatches.

### P2-1 — Prefill MoE tuning · effort M · risk low-medium

Goal: w13 M=512 3063 µs → < ~1.5 ms (≈2×); prefill pp=2048 0.95 s → ~0.6–0.7
s. **Important caveat**: the practical roofline is **~20 TFLOPS measured dot2
peak** (MI50, ILP≥2; see hardware table above). Post-tuning the kernel reaches
~5.9 TFLOPS = **~30% of that practical peak** at M=512. Reaching <1.5 ms (2×
from Phase 1 baseline 3027 µs) requires ~11 TFLOPS = ~55% of the measured
peak — achievable only with the persistent-CTA redesign (option e) or by
raising ILP substantially. The tuning work in P2-1a–c (b128 LDS, NPT=2, BM=8)
delivered ~26% improvement and stalled there, consistent with being
issue-bound (scalar-heavy instruction mix) rather than LDS or bandwidth limited.
Option (e) (persistent-CTA) remains the only path to 2×; it is deferred.

Options, **re-ordered** by expected value per P2-0 findings:

- **a) Stage A tile with b128 LDS** (do first, regardless of P2-0 outcome):
  Replace the scalar b32 `block_a[m][t] = av` store and `const half*` reads
  with vectorized `float4`/`half4` units: `global_load_dwordx4` → register,
  `ds_write_b128` → LDS (16-byte aligned, matching the existing +8-half row
  pad), `ds_read_b128` in the dot loop. Per `gfx906-notes.md`: b32 = 1.9–3.9
  TB/s, b128 = 9.5–11.2 TB/s — a 3–5× LDS bandwidth difference on the
  innermost operand. This is a surgical, algorithm-preserving change and is the
  most likely explanation for the ~40% dot-peak gap. Risk: medium (LDS
  alignment arithmetic; verify bank-conflict-free access with b128 before
  committing). Do this before any BM or K-slice sweep.

- **b) BM=8 heuristic trial** (no new code — case 8 already in the switch):
  Add BM=8 to the heuristic for the 32 < EM ≤ 512 range (currently maps to
  BM=4) and re-bench. BM=8 doubles the accumulator count (32 → 64 VGPRs) but
  may stay within occupancy-1 budget if the VGPR table from P2-0 confirms it.
  This is the cheapest BM experiment because it requires only a heuristic
  change. Bench all M buckets before committing.

- **c) Reduce n_per_thread from 4 to 2** (if P2-0 shows under-occupied or
  dot-bound): Cut the z1z16/y1y16 per-column constants from 4 sets to 2 per
  lane, halving the dequant VGPR budget and allowing grid.y to double. At
  occupancy 1 the dot loop is ILP-starved; raising occupancy to 2 provides
  better latency hiding than wider tiles at occupancy 1. This contradicts the
  BM-sweep direction but is the correct occupancy move when VGPRs are the
  pinning factor.

- **d) M-dependent K-slice size** (launcher change *and* kernel change):
  Currently BLOCK_KN_SIZE = THREADS_X = 256 is a `static_assert`. Enlarging
  the K-slice to 512 for large M requires a new template with 512 threads,
  which re-tightens the VGPR cap (check with P2-0 tooling). For large M,
  halving grid.z from 8 to 4 K-slices reduces atomic CAS contenders per output
  cell. This is *not* a free launcher knob; it requires a new kernel template
  and must go through the same VGPR analysis as other BM variants.

- **e) Persistent-CTA B-in-LDS kernel** (algorithmic change; only if (a)–(d)
  fail to reach the goal): For prefill, every N-block/K-slice block re-streams B
  from HBM for that block's A rows. A persistent kernel loads the expert's
  int4+scale B tile (N=256, K=256 → 32 KB, fits in 64 KB LDS) into LDS once
  and loops over all token-rows assigned to that expert. This cuts HBM B
  traffic by up to grid.y × grid.z and is the "keep B resident, stream A"
  shape the gfx906-notes recommend for b128 LDS tiles. This is the *only*
  algorithmic path to the ≈2× goal if dot-peak efficiency cannot exceed ~50%
  with the current streaming design. Gate explicitly on the P2-0 three-way
  table before committing to this scope.

- **f) If genuinely dot-ISA limited at 40–50%**: record the scalar-dot ceiling
  in the dev log, lower the P2-1 goal to whatever the VGPR/LDS work achieved,
  and move to P2-2. Do not invest further in this kernel design; any larger gain
  requires MFMA-class hardware.

Risks:
- *b128 LDS alignment*: the current `block_a[BM][256+8]` pad is 8 halves =
  16 bytes — correct for 16-byte alignment. Row stride = (256+8)×2 = 528 bytes;
  528/16 = 33, a non-power-of-two vec4 count, which avoids LDS bank aliasing per
  `gfx906-notes.md`. Verify empirically before declaring victory.
- *BM=32 is not viable without redesign*: do not add a case-32 branch unless
  the VGPR table shows it fits. The dequant temporaries alone will push total
  VGPRs past 256 at BM=32 without cutting n_per_thread.
- *Atomic contention changes with slice size*: sweep, don't guess; commit the
  measured table as the heuristic. More K-slices also increases fp16 rounding
  steps per output cell — track precision delta separately from timing delta.

Exit: micro-bench table improved at M ≥ 128 with no regression below; full-model
tg=1 bench shows prefill ≤ ~0.7 s (or documents why this target is unreachable
and at what ceiling the kernel actually lands); dev log updated with the
three-way bottleneck outcome.

### P2-2 — Cudagraph ceiling measurement (no kernel change) · effort S · risk ~none

Add a `BENCH_EAGER` env override to `_bench_gfx906.py` (default 1, unchanged)
and run tg=256 with cudagraphs on. Quantifies how much of the 19.7 tok/s decode
is launch overhead vs GPU time (profile suggests a ~40–50 tok/s GPU-bound
ceiling).

**Two branches**:

1. **Capture succeeds**: measure decode tok/s with cudagraphs and record it.
   If ≥ ~40 tok/s (i.e. most of the decode gap is already launch overhead),
   P2-3 is lower priority — the GPU ceiling is already close to the serving-mode
   number. If result is < ~40 tok/s, a meaningful GPU-bound residual exists and
   P2-3 pays under cudagraphs.

2. **Capture fails** (GDN hybrid layers, hipGraphCreate returns error, etc.) —
   **treat this as a first-class outcome, not an edge case**. A capture failure
   means: (a) the ~40 tok/s ceiling estimate is unverifiable; (b) decode remains
   CPU/launch-bound in all serving configurations; (c) P2-3 decode kernel
   improvements give no wall-clock win in any accessible mode; (d) P2-4
   (fused topk+align) and P2-0b (zero-fill fusion) are the highest-value
   launch-count reductions available. Record the failure reason, note the
   expected ceiling as a theoretical upper bound only, and proceed to P2-4
   before P2-3 in that case.

Risks: near zero — separate measurement, default behavior untouched. Cudagraph
numbers are **not comparable** to the §1 eager table; label them "serving-mode"
in the README/dev log and don't mix them into the version comparison.

Exit: one number (decode tok/s with cudagraphs) + explicit pass/fail capture
status + go/no-go for P2-3/P2-4 and their ordering.

### P2-3 — Decode MoE latency at small M · effort M · risk medium

Gate: only proceed if P2-2 shows a GPU-bound residual (cudagraph succeeded and
decode is clearly below the theoretical ceiling). In eager mode, decode is
CPU-launch-bound and improvements here won't move the §1 bench number.

Goal: w13 M=1 35.5 µs → ~18–20 µs, w2 33 µs → ~16 µs (decode MoE 2.1 →
~1.1 ms/step). At M=1 with 8 active experts, grid = (8 blocks, 1, 8) — 64
blocks on a **60-CU MI50**. The kernel is latency-bound: measured ~228 GB/s
≈ 23% of peak HBM bandwidth (≤1 TB/s), with significant fixed overhead
per-call (LDS fill, syncthreads, epilogue CAS).

Options, **re-ordered** by expected value:

- **a) Skip A-LDS for BM=1** (highest value, lead with this): At M=1, the LDS
  A-tile holds exactly 1 row of 256 halves, loaded once, synced once, read once
  per K-step — zero reuse. Load A directly from global memory into a `half2`
  register pair and pass it to `dot22_8_f` without staging via LDS. This removes
  the `__syncthreads`, the LDS write, and the LDS read for every K-step at M=1.
  Implementation: add a `USE_LDS_A = (BLOCK_SIZE_M > 1)` template flag (the
  RDNA3 parent already has this design — port it). For fp16 on gfx906, the dot
  function reads `half2*` from the A pointer; a global-memory A pointer works
  the same as an LDS one.

- **b) DPP row-broadcast for A at M=1** (complement to (a)): After loading A
  from global into registers (option a), use `v_mov_b32_dpp row_bcast:15` to
  broadcast the first-lane A value across all 16 lanes in the row without any
  LDS. Per `gfx906-notes.md`: DPP row broadcast ≈ 1778 Gxchg/s vs LDS ≈
  906 Gxchg/s. This complements option (a) for cases where multiple lanes
  process overlapping A ranges.

- **c) Finer K-slicing for EM ≤ 32** (e.g. BLOCK_KN_SIZE=128 → 2× more
  blocks): increases parallelism but adds fp16-CAS contenders per output cell
  (retry cost) and adds more fp16 rounding steps (precision cost). These are
  two distinct costs, not one; measure both. The MI50 has 60 CU, so 128 blocks
  at M=1 is reasonable. Bench carefully.

- **d) 128-thread variant** (grid.y doubles): new template instantiation, same
  algorithm. Verify VGPR budget with P2-0 tooling before adding.

Risks:
- *CAS non-determinism*: any K-slice split already makes output order
  non-deterministic today. More slices change *which* order, not *whether* it is
  non-deterministic. Document this; don't try to fix it here (see determinism
  gate in Common Protocol).
- *More K-slices → more CAS retries and more fp16 rounding steps*: these are
  distinct costs. Track separately.
- *BM=1 no-LDS template*: must still satisfy the `static_assert
  (BLOCK_KN_SIZE == THREADS_X)` constraint — the LDS *size* changes (zero) but
  the K-slice layout is unchanged.

Exit: micro-bench M=1/8 improved ≥ 30% with no regression at M ≥ 32; full test
matrix green; run-to-run determinism check passes (see Common Protocol).

### P2-4 — Fused gfx906 topk + align · effort M · risk high (correctness)

Goal: replace `topk_softmax` (18 µs × 40) + `moe_align_block_size` (13.5 µs ×
40) ≈ 1.2 ms/step with one routing kernel for the fixed shape E=256, k=8.

**Revised gain estimate**: the 1.2 ms/step is measured GPU time; in the
CPU-launch-bound regime the *wall-clock* saving is primarily the 80 fewer kernel
launches (40 × 2 kernels → 40 × 1), not 75% of the 1.2 ms GPU time. The
realistic end-to-end gain in eager mode is likely 0.1–0.3 ms/step from launch
reduction, not 0.9 ms. Under cudagraphs the full GPU savings would accrue if
capture succeeds; if capture fails (P2-2 branch), this step's primary value is
the launch-count reduction. Calibrate expectations accordingly before starting.

Approach: single kernel reading router logits [M, 256], doing softmax + top-8,
writing `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded` directly (the
block layout is trivial at k=8: sort ≤ 8×M entries).

Risks — **highest in this phase**:
- *Wrong routing = silently garbage output* (no crash, no NaN). Mitigation:
  exhaustive offline test vs the existing Triton path on random logits
  (including ties, all-negative rows, M not divisible by block sizes).
- *Tie-breaking and ordering semantics*: the correctness bar is **bit-exact
  output tensor**, not just correct routing coverage. The downstream
  `Gfx906WNA16Experts.apply` writes output scattered via `token_id` → original
  slot through `atomic_add_pk4_f16`. A fused kernel that produces the same
  expert membership but a different `sorted_token_ids` order will produce a
  different fp16 accumulation order and a different output tensor. The fused
  kernel must replicate not only the padding sentinels but the *ordering within
  equal top-k score sets* — i.e. all of `moe_align_block_size`'s exact output
  layout, not just its membership.
- Scope creep: specialize for E=256/k=8 only; fall back to the stock path for
  all other shapes.

Exit: bit-exact `sorted_token_ids`/`expert_ids` arrays *and* bit-exact output
tensor vs reference on the full test matrix; full-model bench improved in
launch count with no regression in output; dev log notes fallback gate.

### P2-5 — Shared-expert Triton elimination · effort S · risk low

The fp16 shared expert currently dispatches through the generic Triton
`fused_moe_kernel` (~0.5 ms/step GPU, ~40 Triton kernel launches per forward
pass). A single shared expert is a dense GEMM (no routing). Route it through
`torch.mm` — or directly through the aiter `LLGemm1` path that already handles
dense fp16 GEMMs — from `Gfx906WNA16Experts`.

In the CPU-launch-bound regime, the primary gain is 40 fewer Triton dispatches
per forward pass, comparable in launch-count savings to P2-4. The GPU time gain
(0.5 ms/step) is secondary and would only be visible under cudagraphs. This step
is therefore **not** a "do last if time remains" item — it is a launch-reduction
win on the same level as P2-4, with much lower correctness risk. Move it before
P2-4 in the ordering if P2-4's implementation turns out to be blocked.

Risks: low — numerics are fp16×fp16 either way; still needs the test matrix +
sanity gen to confirm the `mm` path produces equivalent output.

Exit: correctness test ALL PASS; launch-count baseline drops by ~40 dispatches;
full-model tok/s unchanged or improved.

### P2-6 — Deferred (not phase 2)

Largest remaining rows are **not MoE-specific** and deserve their own project:
aiter `LLGemm1` dense GEMMs (~7 ms/step; first instrument the 223 call shapes,
then consider a tuned gfx906 path), decode paged attention (~2.9 ms/step;
custom decode kernel), elementwise/norm fusion (~6–7 ms/step; mostly
CPU-launch-side in eager). These help the dense 9B model equally.

## Common protocol (every step)

1. **Standalone correctness test** — move `/tmp/bench/_test_gfx906_moe.py` to
   `tests/kernels/rocm/test_moe_gfx906.py` (or `benchmarks/kernels/gfx906/`)
   and commit it before starting P2-1. The test file must be in version control
   before any high-risk change (especially P2-4) is attempted. Must stay ALL
   PASS (both source layouts, multiple M/block_m).
2. **Run-to-run determinism check** (new): after any change that touches K-slice
   count or epilogue reduce order, run the kernel 5× on the same input and diff
   the sorted output tensors. Non-determinism from CAS ordering is acceptable
   and already present; this check catches *new* sources of variation introduced
   by the change.
3. Micro-bench table across all M buckets — no silent regression anywhere.
4. **Launch-count measurement**: record GPU dispatch count per forward pass (via
   `rocprofv3 --hip-trace` or equivalent); include in the dev log alongside the
   tok/s numbers. Every step that is primarily a launch-reduction step must show
   a measurable drop here.
5. Full-model bench: tg=1 (prefill) and tg=256 (decode) with the standard
   docker recipe (`-e PYTHONPATH=/workspace/vllm`, `vllm-cache` volume).
6. Greedy sanity generation (3 prompts, eyeball coherence).
7. Separate commit per step; update `DEVLOG-moe-opt.md` (results + negative
   findings) and README §1/§4 numbers only for accepted changes.

Decision gates:
- After **P2-0**: prune the P2-1 option list based on the three-way bottleneck
  table and VGPR table. Explicitly decide whether option (e) (persistent-CTA) is
  in scope based on the dot-peak utilisation result. If the P2-1 goal of ~1.5 ms
  is unreachable, lower it to the measured ceiling before starting.
- After **P2-0b**: confirm −80 dispatches. If this doesn't move full-model
  tok/s noticeably, record it as a correctness/structural improvement and
  continue — this step's value is partly removing the external pre-zero
  dependency, not just the launch count.
- After **P2-1**: if prefill did not improve ≥ 30%, record the measured ceiling
  and reason in the dev log; move to P2-2. Do not chase BM=32 without first
  verifying the VGPR budget.
- After **P2-2**: explicit branch on capture pass/fail (see P2-2 above). Record
  the decision in the dev log with the capture result.

## Rollback / blast radius

All changes stay behind the existing oracle gate (`GFX906_HIP` selected only
on gfx906 with int4 WNA16 MoE weights); reverting any step is a single commit.
Kernel heuristics default to the phase-1 values on unrecognized shapes, so a
tuning regression cannot break other models/architectures.
