# Adversarial review — MoE phase 2 plan (`plan-moe-phase2.md`)

Reviewer: Claude Sonnet 4.6 · date: 2026-08-15 · scope: the phase-2 plan on
branch `gfx906/moe-opt`, read against the current kernel
(`csrc/rocm/moe_q_gemm_gfx906.cu`), the RDNA3 source it was ported from, the
DEVLOG, and `gfx906-notes.md`. A prior review from DS4
(`plan-moe-phase2-plan-rev-ds4.md`) is in scope; this review is independent but
notes where the two converge on the same problem.

---

## What the plan gets right

- The priority ordering (prefill GPU-bound first, decode behind a cudagraph
  ceiling gate) is correct given the profiler headline: Self CPU 4.3 s vs Self
  CUDA 2.0 s.
- Single-commit rollback, oracle fallback, per-step exit gates — solid process
  hygiene consistent with the phase-1 devlog.
- Correctly scoping out LLGemm1, paged attention, and elementwise norm ops as
  a separate project (P2-6). The dense 9B model needs those wins too.
- Recognising the prefill is compute-bound (5.6 TFLOPS vs ~90 GB/s effective
  BW at M=512) and the decode is latency/launch-bound — the distinction shapes
  the right tool choices.

---

## Critical findings

### CRIT-1. The 2× prefill target and option (d)'s own ceiling contradict each other

The plan targets M=512 w13: 3063 µs → <1.5 ms (≈2×). At the current 5.6 TFLOPS
= ~40% of fp16 dot-peak, reaching 1.5 ms demands ~11 TFLOPS = ~80% of peak.
Option (d) then says "accept ~40–50% of scalar-dot peak as the gfx906 limit
and stop." These two are mutually exclusive: if (d) is the actual ceiling, the
<1.5 ms goal is unreachable without an algorithmic change. The P2-1 exit gate
("≥30% improvement → else stop") would then correctly terminate at ~2000 µs
instead of 1500 µs, wasting the exploratory budget on options (a)/(b)/(c)
that cannot close the remaining gap.

**Fix:** the P2-0 profiling pass must explicitly answer *what fraction of the
40% gap is from pipe saturation vs LDS stalls vs occupancy*. Only after that
can the plan say whether 2× requires an algorithmic change (different tile
strategy, fp32 staging, persistent kernel) or is achievable by tuning alone.
Option (d) should be written as a decision branch in P2-0, not a terminal
fallback inside P2-1.

### CRIT-2. BM=32 cannot be instantiated — the dispatch switch has no case for it

Plan P2-1(a) proposes sweeping `{8, 16, 32}` and says "runtime knob already
exists; just change the heuristic." The `dispatch_moe_gemm_q4` switch in the
kernel instantiates `{1, 2, 4, 8, 16}` only; `block_size_m=32` hits the
`TORCH_CHECK(false,...)` trap. A new template instantiation and a new `case 32`
are required.

More critically: `block_c[BLOCK_SIZE_M][4]` holds `BM × 4` fp32 accumulators
per lane. At BM=16 that is 64 fp32 = 64 VGPRs just for accumulators; the
dequant temporaries (`dq[4][4]` half2 arrays, the z1z16/y1y16 constants) add
another ~32 VGPRs minimum. Total at BM=16 is already near the 128-VGPR budget
that limits occupancy to 1. BM=32 doubles accumulators to 128 fp32 VGPRs —
combined with the dequant constants, the register file will spill even at
occupancy 1. The plan says "BM=32 only if P2-0 shows no spills" but this is
backward: P2-0 should tell you that BM=32 will always spill without an
accumulator redesign, not give conditional permission for it.

*Note: DS4's review also identifies this (CRIT-2 there). Both reviewers reached
it independently.*

### CRIT-3. The LDS A-tile problem is misdiagnosed as an amplification problem

Option (c) says "if LDS-bound: reduce A-tile re-read amplification — e.g.
128-thread blocks or DPP broadcast." The A amplification is grid.y. For w13
(N=1024, BLOCK_KN=256, 4 N-cols/thread), grid.y = ceil(1024/1024) = **1** —
there is no A re-read at all in w13. The LDS pressure is not amplification; it
is the access *width*. The kernel writes A as `block_a[m][t] = av` (b32 half
store) and reads it as `const half* a_ptr` → scalar b32 loads. The
`gfx906-notes.md` measures b32 LDS at 1.9–3.9 TB/s vs b128 at 9.5–11.2 TB/s
— a 3–5× bandwidth gap on the A operand that appears in every K-loop iteration.

The fix is to stage A in 128-bit units: `global_load_dwordx4` into a `float4`
register, `ds_write_b128` into LDS (with alignment so `block_a` rows are
16-byte aligned and stride is a non-power-of-two vec4 count to avoid bank
conflicts per `gfx906-notes.md`), then `ds_read_b128` in the dot loop. This
changes b32 loads to b128 for the hottest LDS path with no algorithm change and
is safer than BM=32 or 128-thread rework.

### CRIT-4. P2-1's options list has no mention of the two A-loading paths already in the RDNA3 parent

The RDNA3 source (`moe_q_gemm_rdna3.cu`) has a `USE_LDS_A` template parameter
that skips the LDS round-trip for BF16/M=1 (read A directly from global into
registers). The devlog notes this explicitly: "try skipping the A-LDS round
trip for BLOCK_SIZE_M=1 (read A from global directly, as RDNA3 bf16 M=1 does)."
Yet that option appears only in P2-3's bullet list, not in P2-1 where it is
also relevant (the M=1 decode path and the opportunity to remove the LDS path
entirely for small M without any register redesign). The plan treats the RDNA3
parent as a reference for structure but does not carry over its own prior
analysis of this path systematically.

### CRIT-5. P2-0 diagnostic list misses the one metric that gates everything else: VGPRs per instantiation

P2-0 says "register/spill check ... confirm 256 threads fit without spills at
BM=16 and see what BM=32 would cost." That is correct as far as it goes, but
the P2-0 output list lacks the follow-through: it should also check `BM=8` (the
only BM between 4 and 16 that is neither in the current heuristic nor clearly
in danger of spilling), and it should use the spill results to immediately prune
the P2-1 option list. As written, P2-0 ends with "one paragraph in the dev
log" — that is not a structured decision gate, it is a note. Given CRIT-2 shows
BM=32 is almost certainly off the table, the P2-0 output should be a table of
(BM, VGPR count, expected occupancy, pruned/viable) that directly drives P2-1.

### CRIT-6. The pre-zeroing launches are invisible to the plan but visible in the CPU-bound profile

`Gfx906WNA16Experts.apply` calls `w1_out.zero_()` and `output.zero_()` once
each — two full-tensor fills per MoE layer. Across 40 layers that is 80 extra
kernel launches per forward pass in an already CPU-launch-bound path. The plan
mentions the kernel's pre-zeroed C contract several times but never identifies
the zeroing launches as removable cost. With `enforce_eager=True` and CPU
overhead already 2× the GPU time, these launches are part of the 4.3 s CPU
budget even if the GPU executes them quickly.

The fix is to fold the zero-fill into the GEMM: each CTA clears its own output
tile at the start of the epilogue (or the first K-slice block for that tile
owns the clear). This eliminates the external dependency, makes the kernel
self-contained, and removes 80 launches/step regardless of eager vs cudagraph
mode. It also closes a correctness class: a caller that forgets to zero the
workspace gets garbage rather than silently wrong output.

---

## Moderate findings

### MOD-1. P2-2's 40 tok/s threshold is asserted without measurement basis

"If cudagraph decode is already ≥ ~40 tok/s, deprioritize P2-3/P2-4" — but
40 tok/s is computed from `profile → GPU-only time ÷ 64 steps`, not from an
actual cudagraph measurement. The ceiling estimate inherits all the profiler
overhead of the profiled run (profiler itself adds CPU overhead, distorting the
self-CPU number), and it assumes cudagraph capture succeeds. For a model with
GDN hybrid layers the capture may well fail, which the plan notes vaguely but
should treat as the **primary** branch: capture fails → no cudagraph win → P2-3
and P2-4 remain CPU-launch-bound → fused topk+align (P2-4) is the highest-value
single launch-reduction step. A capture-failure decision branch is missing.

### MOD-2. P2-4 correctness gate misses the exact semantics of `moe_sum` ordering

The plan says "bit-exact routing arrays vs reference on the test matrix." The
stronger requirement is that the fused kernel must reproduce the *specific
ordering* inside tied top-k sets, because `moe_sum` in `Gfx906WNA16Experts`
accumulates expert outputs positionally into the output buffer via `atomic_add_
pk4_f16`. A fused topk+align kernel that produces a permuted-but-same-coverage
`sorted_token_ids` can still be numerically correct (same expert set, same
weights) while producing a different fp16 accumulation order — which on gfx906
at fp16 will produce a different floating-point result due to rounding. "Same
routing coverage" is not the same as "bit-exact output tensor"; both must be
tested, separately.

### MOD-3. P2-3's CAS contention framing conflates retry cost with precision degradation

The warning about "more K-slices → more CAS contenders → wider non-determinism"
is directionally correct but mixes two distinct costs:
1. **Retry cost**: more contenders → more `atomicCAS` retries → real GPU latency.
   This is a tuning tradeoff worth measuring.
2. **Precision cost**: each `__hadd2` in the CAS loop rounds at fp16 precision.
   With 8 K-slices and 8 active experts, an output cell accumulates up to 64
   fp16-precision additions at the epilogue alone. The plan's correctness gate
   accepts "maxrel ≈ 2% = fp16 accum noise" — but that tolerance was set against
   a Torch reference that itself uses fp16 accumulation. A stricter reference
   (fp32 partial sums) would show the atomic epilogue degrades output quality at
   large M/large topk. This is a correctness debt the plan inherits from phase 1
   and should at least be documented, not silently accepted.

### MOD-4. P2-1(b) K-slice tuning is not a free launcher knob

"M-dependent K-slice size: change BLOCK_KN_SIZE" — but `BLOCK_KN_SIZE=256` and
`THREADS_X=256` are coupled by `static_assert(BLOCK_KN_SIZE == THREADS_X)`.
Changing the slice size to 512 requires 512 threads, which halves the number of
CTA instances (fewer blocks ∈ grid.z) but re-tightens VGPRs per lane (more
per-lane work → more registers at occupancy 1). The plan presents this as a
launcher knob; it requires a new kernel template with different thread count and
must go through the same VGPR analysis as BM=32.

### MOD-5. P2-5 shared-expert path understates the available speedup

"The fp16 shared expert currently rides the generic Triton `fused_moe_kernel`
(~0.5 ms/step)." This is listed as a small gain, last priority. But the shared
expert is *always active on every token on every layer*: 40 layers × 1
shared-expert GEMM call in the CPU-launch-bound path. Routing the shared expert
through `torch.mm` (or the existing unquantized linear path) eliminates one
Triton kernel launch per layer — 40 fewer Triton launches, same as the impact of
fusing topk+align. At 0.5 ms/step that is a ~1.5% wall-clock reduction in
GPU time, but as a launch-count reduction in the CPU-bound regime it could be
larger and is less risky than P2-4. The plan ranks it lower than it deserves.

---

## Alternate optimization options

### ALT-1. Stage A tile with b128 LDS — the single highest-leverage unused path

Current: `block_a[m][t] = av` (half, b32 store) and `const half* a_ptr` reads
(b32 loads). Per `gfx906-notes.md`: b32 = 1.9–3.9 TB/s, b128 = 9.5–11.2 TB/s.
The A load is in the inner K-loop; the b32 rate is the most likely reason the
kernel sits at ~40% of dot-peak at M=512 with a compute-bound profile.

**Approach**: pad rows to 16-byte alignment (`block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE
+ 8]` is already allocated with +8 pad — but the current usage is b32 scalar
stores, not b128). Stage the LDS fill using `float4`-vectorized paths (4×half2
per lane, `ds_write_b128`), read back with `ds_read_b128`, feed `a_ptr` in
16-byte chunks. Alignment: the `+8 half = +16 bytes` pad means the actual row
stride is `(BLOCK_KN_SIZE + 8) × 2 bytes = 528 bytes`, which is not a multiple
of 64 (LDS bank width is 4 bytes; 528/4=132, which is divisible by 4, so no
collision for b128 vec access on every-4th-lane patterns). Verify in a
microbenchmark before committing.

This is orthogonal to BM tuning and lower risk than BM=32 or a thread-count
change. It belongs at the top of P2-1, not as a sub-item under a misdiagnosed
amplification path.

### ALT-2. Eliminate fp16 CAS atomics with a per-slice fp32 staging buffer

The kernel accumulates K-slices into the shared fp16 output via `atomic_add_
pk4_f16`. This has three costs: (a) CAS retry loops under contention, (b)
fp16 rounding at each add, (c) it forces the output tensor to be pre-zeroed
externally (the two `zero_()` launches in CRIT-6).

An alternative: each K-slice CTA writes its fp32 partials to a per-slice
staging buffer `[K_slices, M*topk, N]` (allocated once as a workspace), then
a small reduce pass (one thread per output cell, or a warp-reduce kernel)
accumulates slices deterministically in fp32 and writes fp16 output once. The
staging buffer is ~N × M × K_slices × 4 bytes; for w13 M=512: 1024 × 512 × 8
× 4 = 16 MB, comparable to what the workspace already holds.

Benefits: removes fp16 per-add rounding, removes CAS retry, makes the reduce
deterministic, removes external zero dependency. Costs: one extra kernel launch
(the reduce), ~16 MB more staging. This is the algorithmic path that makes
BM=32 safer (fewer fp16 atoms) and prefill correctness tighter.

### ALT-3. Raise occupancy by reducing n-per-thread instead of increasing BM

The plan's only BM-scaling path adds M rows per lane, increasing register
pressure. The opposite lever: cut from 4 N-columns/thread to 2, halving
`z1z16`/`y1y16` per-column constants and dequant temporaries without touching
the accumulator count. This frees ~16 VGPRs, potentially allowing occupancy 2
at BM=16, which on a latency-bound dot loop provides better wavefront hiding
than wider tiles at occupancy 1.

The grid.y doubles (2 N-columns/thread → 2× blocks covering the same N), which
increases the absolute launch count but also the parallelism exposed to the
scheduler. Worth measuring against the b128 LDS fix (ALT-1) to understand which
bottleneck binds first.

### ALT-4. M=1 decode: skip LDS, read A directly from HBM into registers

The RDNA3 kernel has `USE_LDS_A = (BM > 1 || is_bf16)` — for M=1 bf16 it skips
the LDS stage and reads A directly from global memory into registers. The gfx906
port always uses LDS (the fp16 `dot22_8_f` reads from `const half* a_ptr` in
`block_a`). For M=1 the LDS stage adds a global load + `__syncthreads` +
LDS write per decode call with no reuse benefit (one token, read exactly once).

For gfx906 fp16 at M=1, A can be read directly into a `half2` register pair
and passed to `dot22_8_f` without modification. This removes the LDS round-trip
from the M=1 decode kernel path: one fewer `__syncthreads`, the A global load
goes directly to VGPR, and the 256-entry LDS allocation shrinks to zero for
that instantiation. Combined with the b128 ALT-1 for prefill M≥8, the kernel
becomes a template on `USE_LDS_A` as the parent already is.

### ALT-5. DPP row-broadcast for A in the M=1 decode path

For M=1, each token's A vector is the same across all N-column threads that
share that token's K-slice block. After loading A into registers (ALT-4), a
`v_mov_b32_dpp row_bcast:15` makes the first lane's value available to all 16
lanes in the row without any LDS. `gfx906-notes.md`: DPP row broadcast ≈
1778–1784 Gxchg/s vs LDS ≈ 906 Gxchg/s for the same operation. This is the
complement to ALT-4: skip LDS entirely, broadcast A via DPP for the M=1
decode hot path where the single-token A vector is reused across all N-column
threads in a wave.

### ALT-6. Persistent kernel for prefill: keep B tile in LDS, loop over token rows

For prefill M≥128, the same expert's weight tile is read fresh from HBM for
every N-block, every K-slice. A persistent-CTA approach loads the expert's B
int4 tile (or a portion of it) into LDS once and loops over all token-rows
assigned to that expert. This cuts HBM B traffic by grid.y × grid.z per expert
(up to 8× for w13), at the cost of LDS budget and kernel complexity.

The gfx906 LDS is 64 KB per CTA. An N=256, K=256 tile of B int4 is
256×256/2 = 32 KB, which fits. The B streaming path already reads 16-byte
aligned int32 chunks (`int4 b_w[4]`) — these can target `ds_write_b128`
directly after global load. This is the largest single algorithmic change
available for prefill and is what would make 2× realistically achievable. The
plan defers it to "out of scope" inside option (d); it should be an explicit
gated branch in P2-1 that only triggers if options (a)/(b)/ALT-1 fail to reach
the target.

### ALT-7. Tune BM=8 before BM=32

The current heuristic: EM ≤ 32 → BM=1, ≤ 512 → BM=4, else BM=16. BM=8 is
instantiated in the switch (case 8 exists) but never selected. The mid-range
shapes (M=64–256 per expert) may benefit from BM=8 as a lower-VGPR-pressure
step between BM=4 and BM=16, giving occupancy 2 at BM=8 vs occupancy 1 at
BM=16. This is cheaper to add (the template already exists) and safer than
BM=32. The P2-1(a) sweep should be `{8 in heuristic, 16 tuned, 32 if VGPR
budget allows}`, not `{8, 16, 32}` treated as equally viable.

### ALT-8. Shared expert via `torch.mm` — eliminate Triton path for one-expert case

The shared expert is a single dense GEMM (no routing). The plan notes routing
it through the unquantized linear path, but doesn't spell out the simplest form:
`torch.mm(a, w13_shared.t())` where `w13_shared` is the fp16 weight matrix.
The MI60's `aiter LLGemm1` already handles dense fp16 GEMMs efficiently; this
moves the shared expert onto the same path as the dense layers rather than
dispatching to a generic Triton kernel. Zero code in the kernel; the benefit
is one fewer Triton dispatch per layer (40 fewer Triton kernel launches per
step), falling directly in the CPU-launch-bound regime.

### ALT-9. Separate DEVLOG micro-bench log from correctness-test fixture

The plan's common protocol mandates "Standalone correctness test (ALL PASS)."
The test file is `/tmp/bench/_test_gfx906_moe.py` — a `/tmp` path not tracked
by git. If the file is lost between sessions, the correctness gate disappears.
This is not an optimization but an infrastructure risk: move the test to
`tests/kernels/` or `benchmarks/kernels/gfx906/` and commit it. For phase 2
with higher-risk changes (P2-4 fused routing, ALT-2 fp32 staging), having the
correctness regression test in version control is the minimum reproducibility bar.

---

## Differences from DS4 review and points not covered there

The DS4 review covers CRIT-1, CRIT-2, CRIT-3 with the same substance; the
findings here converge. The following are additions or emphases not foregrounded
in DS4:

- **CRIT-4**: the RDNA3 parent's `USE_LDS_A` flag was explicitly designed to
  handle the M=1 no-LDS case; the plan discards that design knowledge.
- **CRIT-5**: the P2-0 output must be a decision table, not a paragraph.
- **CRIT-6** (zero launches): noted as ALT-5 in DS4, escalated here to CRIT
  because the CPU-bound headline is the plan's own framing — and 80 extra
  launches/step under that headline should be a first-class finding.
- **MOD-2** (moe_sum ordering): DS4 notes tie-breaking semantics; this review
  adds the explicit point that same-coverage ≠ same-output due to fp16 atomic
  ordering.
- **ALT-7** (BM=8 tuning): the switch already has this case; no new code.
- **ALT-8** (shared expert via torch.mm to LLGemm1): explicitly connects to the
  existing aiter fast path.
- **ALT-9** (test file in /tmp): a correctness infrastructure risk for a phase
  with multiple high-risk changes.

---

## Recommended re-plan (summary)

1. **Rewrite P2-0** to output a (BM, VGPR, occupancy, viable?) table and a
   three-way bottleneck diagnosis (dot-pipe / LDS-bus / occupancy). Make the
   "no saturated pipe → raise occupancy" and "dot-bound + ISA ceiling → 2× is
   only achievable with ALT-6" branches explicit before P2-1 starts.

2. **Rewrite P2-1** with ALT-1 (b128 LDS A) as item (a), BM=8 (already in the
   switch, no new code) as item (b), BM=16 tuned as item (c), and BM=32 behind
   an explicit "only if VGPR budget verified" gate — not as part of a sweep.
   Remove the "LDS amplification" framing from option (c); the actual LDS
   problem is access width, not re-read count.

3. **Add ALT-2** (fp32 staging reduce) as a P2-1(d) option that simultaneously
   solves the zeroing-launch problem (CRIT-6), the precision debt (MOD-3), and
   creates the budget for BM=32 without CAS spill risk.

4. **P2-2**: make cudagraph-capture failure a first-class branch. If capture
   fails, the ceiling estimate is invalid and P2-4 (fused topk+align) becomes
   higher-priority as a launch-count reduction.

5. **P2-3**: lead with the M=1 no-LDS register path (ALT-4) and the DPP
   broadcast option (ALT-5) from `gfx906-notes.md`. The current order
   (K-slice → 128-thread → no-LDS) is from lowest to highest expected gain.

6. **Common protocol**: add the correctness test to version control (ALT-9),
   and add a run-to-run determinism check (diff sorted output across 5 runs)
   as a gate for any change touching slice count or reduce order.

The plan's process discipline is sound. The kernel model and priority order
inside P2-1 and P2-3 need revision based on the actual bottleneck geometry.
