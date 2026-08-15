# Adversarial review — MoE phase 2 plan (`plan-moe-phase2.md`)

Reviewer: DS4 · date: 2025-08-15 · scope: the phase-2 plan on branch
`gfx906/moe-opt`, read against the current kernel
(`csrc/rocm/moe_q_gemm_gfx906.cu`), the RDNA3 source it was ported from
(`csrc/rocm/moe_q_gemm_rdna3.cu`), `DEVLOG-moe-opt.md`, `gfx906-notes.md`,
`/tmp/gfx906-performance-inspection.md`, and `_bench_gfx906.py`.

This is an adversarial review: severity-tagged findings first, then a set of
**alternate optimization options** the plan does not consider. It assumes the
phase-1 win (18.79 tok/s, 5.4×) is real and correctly locked behind the
`GFX906_HIP` oracle gate; it pushes back on the *planning*, not the phase-1
result.

---

## What the plan gets right (fair credit)

- **Ordering heuristic is sound on its face** — GPU-bound work (prefill MoE)
  before CPU-bound work (decode MoE), and gating decode-side work on a
  cudagraph ceiling measurement (P2-2) is the correct call given
  `Self CPU 4.3 s > Self CUDA 2.0 s`. Doing P2-3 in eager would be wasted work.
- **Correctly scopes out the big rows** (LLGemm1, paged attention, elementwise)
  as non-MoE and reusable by the dense model (P2-6). That is disciplined.
- **Measure-then-change, decision gates after each step, single-commit
  rollback, oracle fallback on unrecognized shapes** — all good hygiene, and
  consistent with the phase-1 devlog discipline.
- **Honest about decode CAS non-determinism** (addressed below, but the plan
  at least says "document, don't fix").

The flaws are in the *engineering reasoning*: several cause/cost links inside
the candidates are asserted without being load-bearing, and the plan leaves
its most valuable lever (LDS bandwidth, which `gfx906-notes.md` already
measured at 2–5× spread) as the worst-justified option.

---

## Critical findings

### CRIT-1. P2-1 option (d) makes the plan's own exit target unreachable

The plan's goal is **"w13 M=512 3063 µs → < ~1.5 ms (≈2×)"**. At the measured
`~5.6 TFLOPS ≈ 40% of peak`, hitting 1.5 ms requires `~11.5 TFLOPS ≈ 83%` of
the `v_dot2_f32_f16` peak. But option (d) explicitly says *"accept ~40–50% of
scalar-dot peak as the gfx906 limit ... and stop."*

These two statements contradict. If the dot-throughput ceiling is genuinely
`~40–50%`, then **2× is not achievable** with this algorithm and option (d) is
the terminal answer — at which point the "≈2×" goal and the "≥30% → else stop"
gate in P2-1 are mis-specified and will waste the effort on options (a)/(b)/(c)
that cannot bridge the gap. **Fix**: before committing to the 2× target, P2-0's
`rocprofv3` pass must answer *which* pipe binds. Only one road leads to 2×:
BEING-dot-bound but LDS/occupancy-limited (then fixable), versus a genuine
scalar-dot ISA wall (then not). The plan must put the *decision* of whether an
algorithm change (different accumulation, vectorized A) is in scope *before*
P2-1, not "out of scope" inside option (d). See ALT-1 to ALT-4.

### CRIT-2. The plan asserts BLOCK_SIZE_M=32 but the kernel has no BM=32 case

The plan's P2-1(a) proposes sweeping `{8, 16, 32}` and says "runtime knob
already exists; just change the heuristic." The dispatch switch in
`moe_q_gemm_gfx906.cu` only instantiates **`{1, 2, 4, 8, 16}`** —
`case 32:` does not exist, and any `block_size_m=32` today hits the
`TORCH_CHECK(false,...)` trap. Adding BM=32 means:
- a new template instantiation `launch_moe_gemm_q4<32>`,
- a new `case 32` in the switch,
- **live-range pressure**: `block_c[32][4] = 128 VGPR` of accumulators per
  lane, plus dequant temporaries → almost certainly `vgpr_count > 256`, which
  spills *even at occupancy 1* per §6/§9 of the inspection notes.

So "BM=32 only if P2-0 shows no spills" is inverted logic: for this kernel
**BM=32 will spill at occupancy 1 no matter what P2-0 shows**, unless the
accumulator strategy changes (`n_per_thread=2`, LDS accum, or fp32 staging —
see ALT-3). The plan has not thought through the register arithmetic. **Fix**:
P2-0 spill check must *prove* whether `block_c[BM][4]` fits; if it can't (it
won't at BM=32), BM=32 is **not** "the next BM to try" but a dead end that
requires a redesign the plan must name, not tuck into a sweep.

### CRIT-3. P2-1 option (c) targets the wrong mechanism (A re-read amplification is ~1× for w13)

Option (c) says "if LDS-bound: reduce A-tile re-read amplification — e.g.
128-thread blocks ... or DPP broadcast." But the A-tile re-read amplification
is **grid.y**, and for w13 `N=1024, BLOCK_KN=256, 4 N-cols/thread →
grid.y = ceil(1024/1024) = 1`. A is read from gmem **exactly once** per
weight read in w13. The real LDS problem, per `gfx906-notes.md`, is that A is
staged with **scalar/b32 LDS** (`block_a[m][t] = av` stores, and the inner
`const half* a_ptr` loads), i.e. the `~2–4 TB/s` b32 path, not the
`9.5–11 TB/s` b128 path — a **2–5× LDS bandwidth spread**. The plan frames
(c) as an amplification/DPP problem (which is essentially a no-op for w13) and
entirely misses the option with the most measured headroom: **stage A with
`ds_write/read_b128` (`float4`-jarred loads) and 16-byte-aligned rows**. This
is ALT-1 and belongs at the top of P2-1, not buried under a misdiagnosis.

### CRIT-4. P2-0's stated decision ("which pipe binds prefill") cannot, alone, disambiguate the fix

The plan treats P2-0's three sub-items (micro-bench table, rocprof counter
pass, spill check) as sufficient to "determine which P2-1 sub-options are worth
trying." But **dot-bound vs LDS-bound vs instruction-issue-bound are not
cleanly separable from counters alone** on gfx906, because the ALU pipe and
LDS pipe share the same issue slots and wavefront occupancy:
- low ALU-util * together with* low LDS-BW-util = **under-occupied** (occupancy
  / latency problem), not either-pipe overload — a third outcome P2-0 does not
  enumerate; and it is the outcome most consistent with a 4-deep `block_c[16]`
  register file pinning occupancy to 1.
- The inspection notes (§6) explicitly warn that occupancy choice **trades
  latency-hiding against spill budget** and must be re-measured per regime.

**Fix**: P2-0 must produce a *three*-way (ALU-pipe / LDS-bus / occupancy)
table, not just "which pipe is saturated," and the plan should list its
follow-up for the "nothing saturated" outcome (raise occupancy / cut VGPRs). As
written, P2-0 can hand back "dot-bound," which silently selects option (d) and
ends the phase.

### CRIT-5. Decode max-blocks estimate understates the concurrency gap and the real lever

Plan (P2-3) reasons about "~64 blocks (w13) / ~32 (w2) on a ~40-CU part." Two
problems:
1. **Parts being used / CU count is the quiet part.** Both plan and devlog say
   "~40 CU" without naming the part. MI50/MI60 differ in CU count and both are
   higher than 40 (MI50 = 60, MI60 = 64). If the runs are on a MI50-
   constrained SKU, the plan should say so; the decode parallelism math is
   meaningless otherwise.
2. **The decisive lever for small-M decode is not "more blocks", it's
   per-block waste and launch/setup cost.** At M=1, w13 latches `block_size_m
   = 1`, each block processes one token for one expert; grid.y = 1. The kernel
   is latency-bound, and the biggest fixed costs are: the A LDS round-trip that
   the RDNA3 code *already skips* for `bf16 M=1` (its `if constexpr` —
   `USE_LDS_A = (BM>1) || is_same<sp>`, which SKIPS LDS for fp16? No — for
   half it always uses LDS), and the `token_row < size_m` guard branching per
   step. The plan's own devlog lists "skip the A-LDS round trip for BLOCK_
   SIZE_M=1" as an option but P2-3 buries it under 3 lower-value bullets
   (finer K-slice, 128-thread variant). The plan should **lead** P2-3 with the
   M=1 no-LDS path — and note the fp16 caveat: fp16 `dot22_8_f` reads
   `a_ptr` as `half2`, so a no-LDS register path needs a vectorized/DPP A
   layout (ALT-6), not just "read A from global" like the bf16 M=1 case.

### CRIT-6. P2-3's CAS-contention worry is directionally sound but conflates two different costs

Plan (P2-3) warns finer K-slicing "widens run-to-run non-determinism (CAS
ordering)". That mixes two things:
- **Cost**: more K-slices → more `atomic_add_pk4_f16` CAS contenders per
  output cell → real runtime cost from CAS retries, *and* the CAS loop is
  slower per-op than a plain accumulating write. This is a legit cost to
  trade.
- **Determinism**: *any* K-slice split already makes decode non-deterministic
  (the CAS ordering is undefined *today*, at 8 slices). Adding slices changes
  *which* non-determinism, not *whether* there is one.

More importantly, the plan never questions **the fp16 CAS atomic itself** for
prefill. Every K-slice and every expert adds through a 64-bit half2 pair using
`__hadd2` — which rounds each partial at **every** add. fp16 accumulation
error at M=512/8 experts/8 slices is a realistic precision hazard that the
correctness gate ("maxrel ≈ 2% = fp16 accum noise") currently *swallow*, because
the torch reference is also not exact. See ALT-2 (fp32 staging for prefill) —
it simultaneously removes the precision hazard, the CAS-retry cost, and makes
the reduce **deterministic**, at the cost of one staging buffer.

### CRIT-7. `output.zero_()` / `w1_out.zero_()` are extra launch-bound kernels in an already launch-bound path

`_bench_gfx906.py` runs `enforce_eager=True`, and the devlog's headline is
**"Self CPU 4.3 s vs Self CUDA 2.0 s → CPU-launch-bound"**. In that regime the
plan's concern should extend to the *per-step zeroing*:
`apply()` calls `w1_out.zero_()` and `output.zero_()` (two full-tensor fills =
two extra kernel launches per MoE layer, ×40 layers). That is opposite to the
plan's own thesis that decode fix pays only under cudagraphs. **Fix**: fold the
zeroing into the GEMM kernel (each block clears its own output tile before the
atomic epilogue, guarded so the first-slice block owns it, or per-K-slice
subregion). This removes 2 launches × 40 layers from the CPU-launch-bound path
regardless of eager/cudagraph mode — a *cheap structural win the plan never
sees*. It also removes the correctness dependency on a separate pre-zeroed
workspace tensor.

---

## Moderate findings

### MOD-1. P2-2 gate is under-specified (what number is "≥ ~40 tok/s"?)

The exit says "if cudagraph decode is already ≥ ~40 tok/s, deprioritize
P2-3/P2-4." But 40 tok/s is plucked from a profile-derived guess with no CI.
The cudagraph number will be sensitive to batch/prompt shape; the "serving
mode" label is fine, but a single magic threshold without an error bar or a
call-out that gfx906 may **not** support CUDA graphs on this graph structure
(hybrid GDN layers) is weak. The plan vaguely notes capture may fail — that
failure *is* the most likely outcome and should be a first-class branch (it
basically means P2-3/P2-4 are **still CPU/launch-bound**, and P2-4's fused
topk+align is the highest-value launch-count cut). As written, P2-2 reads like
it expects cudagraphs to "just work" and would then *misread* a capture failure
as "GPU-bound" rather than "cudagraphs unsupported here."

### MOD-2. P2-4 "target ~0.3 ms/step" vs measured ~1.2 ms/step lacks a budget breakdown

The plan wants fused topk+align at ~0.3 ms/step from ~1.2 ms/step, but gives
no analysis of *why* the stock kernels cost 18 µs + 13.5 µs × 40. If those are
dominated by **launch overhead** (eager mode, 80 launches for a ~few-hundred-µs
job), then fusing two kernels into one per layer **saves only the launch
difference**, not 75% of the total — the arithmetic doesn't support 0.9 ms of
savings. P2-4 is the plan's highest-risk item (correctness) attached to the
**least-justified** gain. Either rebuild the kernel so the fused version is
inherently cheaper (it coalesces the logits read and softmax once), or lower
the expectation. As with CRIT-1, the plan states a number without the load
model behind it.

### MOD-3. P2-1(b) K-slice change interacts with the static_assert constraint

`BLOCK_KN_SIZE == THREADS_X` is a `static_assert` (line ~177). Enlarging the
K-slice to 512 for large M requires **changing THREADS_X or the LDS sizing**
(`block_a[BM][BLOCK_KN+8]`, and 512 threads re-tightens the per-lane VGPR cap
per §6). The plan lists (b) as if it's a free launcher knob; it is a kernel
template change with occupancy consequences that must be re-benchmarked the
same way as BM. Worth flagging so the "M-dependent K-slice size" doesn't arrive
as a surprise compile/occupancy failure.

### MOD-4. Tie/order semantics in P2-4 are under-stated

The plan says "must replicate `moe_align_block_size`'s exact layout (padding
sentinel values included)" — but the correctness requirement is **stronger than
byte-equality to the reference**: the downstream `Gfx906WNA16Experts.apply`
consumes `sorted_token_ids`/`expert_ids` *positionally* and writes output
scattered to `[M, topk, N]` via `token_id` → original slot. A fused kernel that
produces a *different but valid* ordering (e.g., sorts ties differently while
keeping the same token/expert coverage) would still be correct for GEMM1/GEMM2
reduction, but **must** produce the identical `moe_sum` accumulation order and
identical padding so the precision and the `TopKWeightAndReduceNoOP` contract
hold. "Bit-exact routing arrays" is the only safe bar; the plan should state
that the *ordering within equal top-k sets* is also part of the contract, not
just membership.

---

## Alternate optimization options (beyond the plan)

These are concrete, mostly *lower-risk/higher-leverage* alternatives the plan
omits, ordered roughly by expected value.

### ALT-1 (TOP). Stage A with `ds_write/read_b128` instead of b32 — the single biggest unused lever

`gfx906-notes.md` measured: b32 LDS `~2–4 TB/s`, b64 `~4.3–8.8`, **b128
`~9.5–11.2 TB/s`**. The kernel's A path is scalar/b32 (`block_a[m][t] = av` and
`const half*` reads). For a prefill tile (`BM=16`, `K=256`) this is the most
likely reason the GPU sits at ~40% of dot-peak despite having compute: LDS A
reads degrade to the 2×-slot b32 path while the ALU waits. Fix is surgical:
stage A in `float4`/`half4` units (`global_load_dwordx4` + `ds_write_b128`,
peel and read as `ds_read_b128`), keep rows 16-byte aligned in the A tile. This
directly attacks CRIT-3 and is *lower risk* than BM=32 or a 128-thread rework.
**Do this before any BM/K-slice sweep.**

### ALT-2. fp32 staging + deterministic segmented reduce for prefill (kills CRIT-2's spill, fixes precision/determinism)

Instead of pushing BM to 32 with 128 fp16 accumulators (which spills) or
keeping fp16 CAS atomics (which round each add and are non-deterministic):
- CTA writes fp32 partials to a small per-(slice) scratch region (or keeps a
  `float block_c[BM][4]` — note it is **already fp32 in the kernel**! the fp16
  rounding happens only at the *epilogue* CAS), then
- a **separate, deterministic reduce** (fixed order) accumulates slices +
  experts into the output.

The in-kernel accumulator is already fp32; the loss and non-determinism enter
only at the `__float2half_rn` + `__hadd2` CAS epilogue. Moving the epilogue to a
deterministic fp32-then-once-round improves prefill precision (justified: the
maxrel≈2% gate is silently masking per-add rounding that won't, on average,
meet a tighter bar) **and** removes the CAS-retry contention. Cost: one staging
buffer + one reduce pass; benefit: correctness the current design doesn't
strictly have. This also *defer-loads* the BM=32 question (fewer fp16 atoms).

### ALT-3. Reduce per-lane work to buy occupancy instead of pushing BM up

`block_c[BM][4]` at BM=16 is 64 fp32 accumulators; the plan's only path is
*more* BM. The cheaper occupancy play is **`n_per_thread=2` or LDS-cached A
with half the threads reading the same B**, i.e., *decrease* per-lane
register/ALU load to raise resident wavefronts (per §6: lower VGPR → more
waves/CU → better latency hiding for a dot-loop that is ILP-starved at occ-1).
This contradicts the direction of (a) but is exactly what the inspection notes
suggest for a compute-bound tile. It should be a co-equal P2-1 candidate.

### ALT-4. Two-region launch: persistent-CTA kernel that stages the active expert's B in LDS and loops over its tokens

For prefill, the same expert serves many tokens; today *every* N-block/K-slice
block re-streams B from HBM for *that block's* A rows. A **persistent-version
kernel** that loads the expert's int4+scale tile into LDS once and loops over
all token-rows assigned to it (with atomic or delayed reduce) would cut HBM
weight traffic and keep the dot pipe fed. This is the classic "keep B resident,
stream A" GEMM shape and is the natural gfx906-notes-mandated design (b128 LDS
B tiles). It is more work than a sweep but is the *algorithmic* change option
(d) currently declares "out of scope" — worth making an explicit, gated P2-1
branch rather than silent scope-cut.

### ALT-5. Fuse the zero-fill into the kernel (structural, launch-bound win)

Covered in CRIT-7. Removing `w1_out.zero_()` and `output.zero_()` (2 launches ×
40 layers) by having the kernel clear its own output tile is orthogonal to
every other option and pays in **eager mode today**. Lowest effort, guaranteed
non-negative; it belongs in the common-protocol/phase even if P2-1..5 all fail.

### ALT-6. DPP row-broadcast for the A operand (data-path, not LDS)

`gfx906-notes.md`: `v_mov_b32_dpp` row-shift ≈ **~2× faster than the LDS
equivalent** (~1778 vs ~906 Gxchg/s). If A is broadcast to the wave that owns a
column block, a DPP broadcast can replace the LDS A round trip for the *decode*
M=1 case and shrink the small-M latency (the actual P2-3 win). This
complements (not substitutes for) ALT-1: use b128 for the prefill staged-A,
DPP for the decode single-token path.

### ALT-7. Measure with a launch-bound-aware budget before P2-1, not after

Because the harness is `enforce_eager=True` and the run is CPU-launch-bound, a
prefill *GEMM* micro-improvement may not move the *eager wall clock* at all if
the CPU launch serialization dominates the 0.95 s. The plan's P2-1 exit says
"full-model tg=1 bench shows prefill ≤ ~0.7 s" — but it never isolates how much
of the 0.95 s is GEMM time vs the 40× fix of *launch/setup* around it. Add a
launch-count drop as an *observable metric* to every step's exit gate, or P2-1
could pass its micro-bench yet fail to move the end-to-end number for reasons
unrelated to the kernel.

### ALT-8. Determinism gate as an explicit common-protocol step, not a P2-3 P side-note

The plan's common protocol has correctness (vs torch reference, ALL PASS) but
**no run-to-run determinism check** for the CAS output. Especially once P2-3
touches slice counts and ALT-2 changes the reduce, a "sort-outputs and diff
across 10 runs, bite-exact" gate would catch CAS-order regressions that don't
show up as numerical vs-reference failures (because the reference is also
order-dependent). Cheap to add; the plan treats non-determinism as a footnote.

---

## Summary / recommended re-plan

The phase-2 plan is well *disciplined* (gates, rollback, oracle, scope) but its
**technical priors are off in the places that matter most**:

1. **Rewrite P2-1 around the measured LDS b32-vs-b128 gap (ALT-1) and the
   occupancy lever (ALT-3/ALT-4), not a BM sweep that CRIT-2 shows cannot hit
   BM=32 and CRIT-1 shows cannot reach the stated 2× target.** Make the
   "no saturated pipe → raise occupancy" outcome explicit in P2-0.
2. **Reconcile the 2× goal with option (d);** either commit to ALT-2/ALT-4 as
   the algorithmic path to 2×, or lower the P2-1 goal to (d)'s honest 40–50%
   ceiling.
3. **Add ALT-1, ALT-5 (zero-fill fusion), ALT-2 (deterministic fp32 epilogue)
   as first-class high-value options** — they are all *lower-risk* than BM=32
   and several pay in the *eager, CPU-launch-bound* regime the plan admits it
   lives in.
4. **Fix the decoding plan's part/CU-count claim and lead with the M=1 no-LDS
   (DPP, ALT-6) path** instead of the finer-K-slice/128-thread bullets.
5. **Treat cudagraph-capture failure as a likely, first-class P2-2 branch**
   (it re-classifies P2-3/P2-4 as CPU-work), and attach a launch-count metric
   to every gate so the CPU-bound regime isn't accidentally tuned for GPU time.

Net: the plan's process is sound; its kernel-model reasoning is not. The two
places it will burn effort are (a) chasing BM=32/spill and a 2× that option (d)
pre-admits is impossible, and (b) misdiagnosing the LDS problem as an
amplification problem when `gfx906-notes.md` already measured the real 2–5×
b32-vs-b128 spread. Fix the model, keep the discipline.