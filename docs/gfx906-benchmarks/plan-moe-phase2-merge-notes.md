# Review merge notes — MoE phase 2 plan

Reviewers: DS4 (`plan-moe-phase2-plan-rev-ds4.md`) and Claude Sonnet 4.6
(`plan-moe-phase2-plan-rev-claude.md`). This document records claim-by-claim
validation and notes what changed in the updated plan as a result.

Verdict codes: **CONFIRMED** (both reviewers agree, or one reviewer confirmed
by direct code inspection), **PARTIALLY CONFIRMED** (directionally right but
overstated or needs qualification), **REJECTED** (claim is incorrect).

---

## Critical findings

### CRIT-1 — 2× prefill goal contradicts option (d)'s 40–50% ISA ceiling

**Verdict: CONFIRMED.** Both reviewers reached this independently.

Reaching <1.5 ms from 3063 µs at 40% of dot-peak requires ~80% of dot-peak.
Option (d) pre-admits the ceiling is 40–50%. These two claims are mutually
exclusive. **Plan fix**: the P2-1 goal is now stated with an explicit caveat
that 2× is contingent on the three-way P2-0 bottleneck result; option (d) is
renamed option (f) and is framed as a terminal outcome reached *after* trying
options (a)–(e), not a silent fallback inside the sweep.

### CRIT-2 — BM=32 is not instantiated and will spill

**Verdict: CONFIRMED.** Both reviewers reached this independently by reading
`dispatch_moe_gemm_q4`.

`case 32` is absent from the switch; any `block_size_m=32` hits
`TORCH_CHECK(false)`. At BM=32, `block_c[32][4]` = 128 fp32 VGPRs plus dequant
constants exceeds the 256-VGPR per-lane budget at occupancy 1. **Plan fix**:
P2-1(a) was "sweep {8, 16, 32}" — removed BM=32 from the sweep; the plan now
says BM=32 requires a redesign (n_per_thread=2 or fp32 staging) and must be
gated on the VGPR table from P2-0. BM=8 (already in the switch) is elevated
to its own option (b).

### CRIT-3 — LDS "amplification" framing is wrong for w13

**Verdict: CONFIRMED.** Both reviewers reached this independently.

For w13: N=1024, BLOCK_KN=256, 4 N-cols/thread → grid.y = 1. There is no A
re-read amplification. The actual LDS problem is b32 access width (1.9–3.9
TB/s) vs b128 (9.5–11.2 TB/s), a 3–5× gap measured in `gfx906-notes.md`.
**Plan fix**: option (c) "reduce A-tile re-read amplification" is replaced by
option (a) "stage A tile with b128 LDS" as the first option regardless of the
P2-0 outcome.

### CRIT-4 — P2-0 output cannot disambiguate fixes without a three-way table

**Verdict: CONFIRMED.** Both reviewers raised this (DS4 as CRIT-4, Claude as
CRIT-5). On gfx906, low ALU-util + low LDS-BW together indicate
under-occupancy, not pipe saturation — a third outcome the original plan did
not enumerate, and the most consistent one with occupancy=1 pinned by 80+
VGPRs. **Plan fix**: P2-0 now explicitly produces a three-way bottleneck table
(LDS-bound / dot-bound / neither-saturated) with named follow-on actions for
each outcome.

### DS4-CRIT-5 / Claude-CRIT-4 — Decode options list buries the highest-value lever

**Verdict: CONFIRMED** for the claim that no-LDS should lead. The DS4 text
contains a hedged self-correction on the RDNA3 `USE_LDS_A` flag ("which SKIPS
LDS for fp16? No — for half it always uses LDS") that is correct per the
gfx906 port (which always uses LDS regardless of BM or dtype). This does not
change the conclusion: for M=1 on gfx906, the LDS round-trip is pure overhead
and should be skipped by a `USE_LDS_A = (BM > 1)` template flag, as the RDNA3
parent already demonstrates. **Plan fix**: P2-3 option list reordered: (a)
skip A-LDS for BM=1 → (b) DPP broadcast → (c) finer K-slicing → (d)
128-thread variant.

**Note on CU count and hardware specs** (DS4-CRIT-5 sub-claim, updated with
wiki data and P2-0 measurement): DS4 flagged "~40 CU" as unsubstantiated. From
the ROCm hardware table and AMD launch material: MI60 = 64 CU, 29.5 TFLOPS
FP16; MI50 = 60 CU, 26.8 TFLOPS FP16; both ≤1 TB/s HBM2 and gfx906 ISA.

The devlog header says "MI60 32 GB", but P2-0's `rocprofv3` agent info measured
`Simd_Count=240 → 60 CUs`, confirming the bench hardware is an **MI50** (not
MI60). The measured `v_dot2_f32_f16` peak is **~20 TFLOPS** (ILP≥2,
sclk=930 MHz) — well below either datasheet figure, because the dot2 issue rate
is limited by scalar instruction mix at ILP=1. Post-tuning the kernel reaches
~5.9 TFLOPS = **~30% of the 20 TFLOPS practical peak**; reaching 2× (→ ~55%
of practical peak) requires the persistent-CTA redesign, which was deferred.

**Plan fix**: SKU table added to Current State with a measured-vs-datasheet
column; hardware corrected to MI50 throughout; "~40-CU part", "~700 GB/s",
"~13.8 TFLOPS", and "29.5 TFLOPS" removed; decode bandwidth utilisation
corrected to ~23% of ≤1 TB/s peak.

### DS4-CRIT-6 / Claude-MOD-3 — CAS cost and determinism are two distinct things

**Verdict: CONFIRMED.** CAS retry cost (real GPU latency) and CAS ordering
(output non-determinism) are different concerns. More K-slices also increases
the number of fp16-rounding steps per output cell — a third distinct cost.
**Plan fix**: P2-3 risks section now lists "CAS retries" and "fp16 rounding
steps per output cell" as separately trackable costs; the old "widens
non-determinism" language is replaced by "changes *which* order, not *whether*
non-determinism exists".

### CRIT-6 (Claude) / DS4-CRIT-7 / DS4-ALT-5 — zero_ launches are removable launch overhead

**Verdict: CONFIRMED.** `Gfx906WNA16Experts.apply` calls `w1_out.zero_()` and
`output.zero_()` — 2 launches × 40 layers = 80 kernel dispatches per forward
pass in a CPU-launch-bound path. Both reviewers identified this. **Plan fix**:
zero-fill fusion is added as **P2-0b** — a new step between P2-0 and P2-1,
effort S, risk low, pays immediately in eager mode without touching the kernel
algorithm.

---

## Moderate findings

### DS4-MOD-1 / Claude-MOD-1 — P2-2 cudagraph capture failure is not a branch

**Verdict: CONFIRMED.** The original plan mentions capture failure as a caveat
but does not specify what to do. A failure is likely with GDN hybrid layers.
**Plan fix**: P2-2 now has two explicit branches — capture-succeeds and
capture-fails — with named consequences for P2-3/P2-4 ordering in each.

### DS4-MOD-2 / Claude-MOD-2 (implicit) — P2-4 gain is not justified

**Verdict: CONFIRMED.** The 1.2 ms/step is measured GPU time; in eager mode
the wall-clock saving is primarily launch-count reduction (80 → 40 dispatches),
not GPU execution time. "~0.3 ms/step" as an end-to-end gain is unsupported.
**Plan fix**: P2-4 gain estimate is revised to "0.1–0.3 ms/step from launch
reduction" in eager mode; full GPU savings only accrue under cudagraphs if
capture succeeds.

### DS4-MOD-3 / Claude-MOD-4 — K-slice change is not a free launcher knob

**Verdict: CONFIRMED.** `BLOCK_KN_SIZE == THREADS_X` is a `static_assert`. A
512-size slice requires 512 threads, a new kernel template, and a VGPR
re-check. **Plan fix**: P2-1 option (d) (formerly option (b)) now notes it is
a kernel template change requiring the same VGPR analysis as other BM variants.

### DS4-MOD-4 / Claude-MOD-2 — P2-4 ordering semantics understated

**Verdict: CONFIRMED.** Bit-exact `sorted_token_ids` is not sufficient; the
downstream `atomic_add_pk4_f16` accumulation order depends on the positional
order of `sorted_token_ids`, so a permuted-but-same-coverage output produces a
different fp16 output tensor. **Plan fix**: P2-4 risk description now explicitly
states "bit-exact output tensor, not just bit-exact routing arrays" and explains
why ordering within tied top-k sets is part of the contract.

### Claude-MOD-5 — P2-5 shared expert is a first-class launch-reduction step

**Verdict: CONFIRMED.** 40 Triton dispatches per forward pass removed = same
launch-count reduction as P2-4, with far lower correctness risk. Ranking it
"do last if time remains" understates its value. **Plan fix**: P2-5 description
now notes it is "on the same level as P2-4 as a launch-count reduction" and may
be moved before P2-4 if P2-4 implementation is blocked.

---

## Alternate options carried into the plan

| DS4 label | Claude label | Action |
|-----------|--------------|--------|
| ALT-1 (b128 LDS for A) | ALT-1 | → P2-1 option (a), promoted to first |
| ALT-2 (fp32 staging) | ALT-2 | → P2-1 option (e) description (persistent-CTA is now option (e); fp32 staging informs the risk section on why BM=32 needs redesign) |
| ALT-3 (n_per_thread=2) | ALT-3 | → P2-1 option (c) |
| ALT-4 (persistent-CTA B-in-LDS) | ALT-6 | → P2-1 option (e) |
| ALT-5 (zero-fill fusion) | CRIT-6/ALT-5 | → P2-0b (new step) |
| ALT-6 (DPP for M=1 decode) | ALT-4/5 | → P2-3 option (b) |
| ALT-7 (launch-count metric) | (implicit) | → Common Protocol item 4 |
| ALT-8 (determinism gate) | ALT-8 (implicit) | → Common Protocol item 2 |
| (none) | ALT-7 (BM=8 in switch) | → P2-1 option (b) |
| (none) | ALT-9 (test file out of /tmp) | → Common Protocol item 1 |

---

## Claims not carried into the plan (rejected or out of scope)

- **DS4-ALT-2 (fp32 staging + deterministic reduce as a first-class P2-1
  option)**: The kernel already uses fp32 accumulators internally; the fp16
  loss is only at the CAS epilogue. A staging buffer adds a workspace + reduce
  kernel, which increases complexity and the launch count (counterproductive in
  the launch-bound regime). The precision debt is documented in the plan's
  Current State section; the fix is deferred unless the CAS error becomes a
  model-quality issue. The zero-fill fusion (P2-0b) already removes the
  external pre-zero dependency; the CAS epilogue itself is left as-is for now.

- **DS4-CRIT-5 parenthetical on RDNA3 USE_LDS_A for half**: DS4 hedges and
  concludes "No — for half it always uses LDS." This is correct for the gfx906
  port, but the point (no-LDS is the right design for M=1) stands. The hedging
  is not a rejection of the claim.
