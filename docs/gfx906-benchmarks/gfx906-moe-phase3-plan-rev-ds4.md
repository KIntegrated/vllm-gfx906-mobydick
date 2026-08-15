# Adversarial review — Phase 3 plan (`plan-decode-phase3.md`)

Reviewer: DS4 · date: 2026-08-15 · scope: `plan-decode-phase3.md` v2 read
against `DEVLOG-moe-opt.md` (P2-3/P2-4/P2-5 graph-mode profile and P2-0
hardware correction), `README.md` (hardware/model facts), and the phase-2
review already produced for this effort.

This is an adversarial review of the *planning*. The plan's discipline —
P3-0 diagnostics gate before candidates, honest 2–3× (not 5.5×) P3-1 range,
quantization-asymmetry caveat, skip-instead-of-force P3-5 — is good and gets
credit below. The failures are in *budget arithmetic and scope coverage*:
one source table, when actually summed and compared to the profile it cites,
does not reconcile to the measured step total, and the single largest non-MoE
GPU consumer after aiter is never mentioned.

---

## What the plan gets right (fair credit — before the critique)

- **Measured, not aspirational, budget with explicit floors.** Dense rows are
  sized from `bytes = N·K·2` against ~1 TB/s, and the plan correctly flags
  P3-0 Q1 as the thing that makes the floors real. The W16 vs Q4 asymmetry vs
  llama.cpp is called out repeatedly and the plan does *not* chase llama.cpp's
  absolute number — correct, and consistent with how the phase-2 review wanted
  option-(d)-style honesty.
- **P3-1's expectation is scoped down from the 5.5× floor** to a defensible
  2–3× ("no M=1 kernel reaches perfect BW") instead of claiming a full 5.5×
  win. That is exactly the kind of load model the phase-2 review hammered the
  earlier plan for lacking.
- **P3-3 being latency/occupancy-bound, not BW-bound** is a correct read
  (~0.5 MB KV/layer at seq~500 vs sub-µs floor), and gating it on a llama.cpp
  gap ≥2× is disciplined, including the "must not regress FA prefill" guard.
- **P3-4 reframed honestly** (inductor already fused; real lever is graph
  breaks; EV 0–1 ms) rather than the naive `pass_config`-flag promise.
- **P3-5 default SKIP** and §7 risk section (don't chase llama's Q4 bytes) are
  mature calls.

The shortcomings are (a) the budget table does not sum to the measured step
and drops two material profile rows, (b) the biggest such dropped row is an
unaddressed 10% of the step, and (c) the itemized targets and the §6 success
summary are arithmetically inconsistent.

---

## Critical findings

### CRIT-1. §2 does not reconcile to the measured step total, and two profile rows are silently dropped

Sum the plan's own §2 budget table row by row:
`6.8 + 1.6 + 1.9 + 1.75 + 1.2 + 1.0 + 2.0 = 16.25 ms`. The plan's stated
step is **20.3 ms** (§1, §6). That leaves **≈4 ms (≈20%) of the step with no
home** in the table, in any P3 candidate, or in the §5 protocol. The plan
acknowledges P3-0 Q0 uncertainty in *layer count* but never checks that the
budget rows themselves sum to a consistent single number.

The source it cites (`DEVLOG-moe-opt.md`, P2-3/P2-4/P2-5 graph-mode profile)
lists, at a minimum: `aiter LLGemm1 7.2 ms`, **`aten::mm 2.2 ms (10%)`**,
paged attention 1.95, MoE 1.77, triton_matmul 1.6, GDN/mamba 1.15,
**`aiter LLMM1 1.2 ms (5%)`**, routing 1.0, **`shared expert 0.55 ms (2.5%)`**,
elementwise ~2.0 → ≈20.6 ms (matching profile's "Self CUDA ≈22.6 ms/step",
"matches the 20.3 ms derived from the bench").

Mapping: the plan merges the devlog's distinct **LLGemm1 (7.2)+LLMM1 (1.2)
= 8.4 ms** into a single `~6.8` row, **drops `aten::mm` (2.2) entirely**, and
**drops the `shared expert` (0.55) row from the budget** (it reappears only
as P3-0 Q4 / §2-proj-table rows). Every percentage and every candidate size
in the plan is computed off a denominator that silently omits ~20% of the
measured work. Until P3-0 rebuilds the table against a single measured step
number and names a home for every profiled row, no candidate's ms/step or %
claim is trustworthy.

**Fix: P3-0 Q0 must be extended from "layer composition" to "full-step budget
reconciliation"** — reproduce the plan's §2 table from one profile window,
show it sums to a single measured ms/step, and map every named kernel
(including aten::mm, LLMM1, shared expert) to a row or an explicitly-scoped
"out of plan" note.

### CRIT-2. `aten::mm` (2.2 ms/step, ~10%) has no candidate, no P3-0 question, no mention

The devlog's graph-mode profile is unambiguous: `aten::mm | 2.2 ms (10%) | 4
calls × 532 µs — large dense GEMMs`. That is the *second-largest non-MoE GPU
consumer on the board* after aiter LLGemm1, and the plan never names it. It is
not in §2's budget table, not in the dense-projection breakdown, and none of
P3-0 Q0–Q5 or P3-1..P3-5 addresses it. It also sits at the same scale as the
LM head (P3-5, 1.42 ms) — the plan's LM-head row and this `aten::mm` row may
be the same GEMM seen twice, or the true head may be the `aten::mm` path; the
plan doesn't know, because it never asked.

Two consequences: (1) the plan's "realistic capture" of 2.5–5 ms is computed
against **16 of 20.3 ms elapsed** — the unaddressed ~4 ms (aten::mm 2.2 +
LLMM1 1.2 + shared 0.55 + the LLGemm1 compression) make the "close ≥50% of the
gap" success claim structurally harder; (2) P3-5's reasoning ("LM head only
1.4 ms/step, ~0.4 ms slack, default SKIP") could be wrong for the wrong GEMM —
if the real head/decoder projection runs through `aten::mm` at 532 µs/call,
the "slack" budget and the skip decision are mis-sized.

**Fix: add P3-0 Q6 — identify the four ×532 µs `aten::mm` calls (layer,
shape, backend) and reconcile them against the LM-head/decoder projection
rows. Decide explicitly whether the 2.2 ms is in or out of scope.** An
out-of-scope 10% item must be said to be out of scope, not omitted silently.

### CRIT-3. §6 success-summary and the per-candidate targets do not add up

- P3-2's own stated target is "dense projections 8.4 → ~5.5 ms/step", a
  **2.9 ms** saving. But §6 credits P3-2 with only **1.5–2 ms** and also lists
  P3-1 (the 1.6→0.7 Triton-fallback row that is *inside* that "8.4") as a
  separate 0.8–1.2 ms line. P3-2's target therefore implicitly *includes*
  P3-1's fix, while §6 counts them twice. A careful reader cannot derive the
  §6 total from the itemized targets without double-counting on either side.
- §6: 2.5–5 ms off 20.3 → 15.3–17.8 ms/step → "58–70 t/s". llama.cpp parity is
  14.2 ms/step. The **upper** bound (5 ms off) lands at ~15.3 ms ≈ 65 t/s,
  i.e. essentially *parity*, contradicting §6's own "parity is NOT guaranteed"
  claim; the **lower** bound (2.5 ms) lands at ~56 t/s, which is only just
  above the "≥50% of gap" bar (50% of 6 ms = 3 ms). So the plan simultaneously
  promises "close ≥50%", admits parity is not expected, and gives a range whose
  top end already reaches parity. The headroom model is not coherent.

**Fix: restate one additive budget from a single denominator** — e.g. give a
per-row ms/step target for each candidate against the *reconciled* 20.3 ms
(and the ~4 ms currently missing), and make the "≥50% of gap" bar a checkable
arithmetic statement, not a prose average.

---

## Moderate findings

### MOD-1. MI60 mislabel persists the repo's own corrected finding

The plan is titled/labeled "gfx906 (**MI60**)" and repeats it in §1. This repo
already corrected this: `README.md` and the P2-0 hardware note state the card
is **MI50, 60 CU** (rocprofv3, Simd_Count=240 → 60 CUs; MI60 = 64 CUs; the
"MI60" came only from VRAM). Impact on the plan's *numbers* is small (M=1 work
is latency-bound, and HBM peak is ~1 TB/s on both), but P3-3 argues that
attention is occupancy-limited and P3-2 sizes parallelism on the same card; if
any reader takes a "64 CU" implication literally (MI50 vs MI60 do differ in CU
count / banks), the reasoning silently shifts. Fix is trivial: s/MI60/MI50/,
or footnote the VRAM-vs-CU discrepancy so it isn't re-litigated every plan.

### MOD-2. P3-2's entry path leans on aiter knobs that may not exist; the proven path is the custom kernel

The two largest dense rows (aiter LLGemm1 ~6.8 ms + the P3-1 reroute target)
both resolve to **aiter LLGemm1**. Yet the whole premise of this fork's MoE
work — and the ISA notes it cites repeatedly (no `v_mfma` on gfx906; Triton
and upstream kernels poor on this arch) — is that upstream aiter/rocBLAS are
weak on gfx906 and the winning move was to write a custom HIP kernel
(`moe_q_gemm_gfx906.cu`, 35–125×). P3-2(a) "try other aiter/rocBLAS
dispatch/splitK" is asserted as the low-risk entry point with **no evidence
those knobs exist or are exposed for this op on ROCm gfx906**. 1.5–2 ms of
"configuration-only" dense savings is being promised off a dependency that may
reject every knob. P3-2(b) (port a minimal M=1 W16A16 kernel) is downgraded to
a fallback precisely where this project's own history says the fallback is
actually the primary.

**Fix: P3-0 should explicitly check aiter's tunability for these op/shape
combinations before §6 commits to 1.5–2 ms from configuration; and P3-2 should
lead with the custom-kernel branch (on the proven gfx906 MoE kernel's design:
b128 weight streaming + `__ockl_fdot2`), with the dispatch-flags attempt gated
as a quick try, not promoted as the plan's main line.**

### MOD-3. P3-1's ceiling is itself the P3-2 problem

P3-1's honest 2–3× range is internally sensible, but its destination —
"route to the same LLGemm1 path as siblings" — is a kernel that is itself
~2–2.5× off the floor on sibling shapes (P3-2's own numbers). So P3-1's gain
is "as good as the *other* mediocre aiter kernels", not "approach the floor";
and if P3-0 Q3 shows the [2048→2048] fallback is declined by aiter for a real
shape/dtype constraint (rather than a selection bug), P3-1 collapses entirely
into P3-2(b). The plan does flag the reject case, but the "cheapest possible
win" framing overstates its likelihood, since the very existence of ~37 such
calls already bypassing aiter is evidence aiter declines them for a reason.

### MOD-4. Shared expert is characterized two incompatible ways across §2 and §6

§2's dense-projection table lists shared `gate_up` + `down` as "two plain fp16
Linears" through LLGemm1 (~0.76 ms/step over 40 layers). P3-0 Q4 separately
asks "what the residual `fused_moe_kernel` (Triton, ~2/step, 0.55 ms) actually
computes," and the devlog lists "shared expert (Triton fused_moe)" as its own
row. If the shared expert is two plain LLGemm1 Linears, it is already inside
the P3-1/P3-2 dense surface; if it is a Triton `fused_moe_kernel`, it is
(unaddressed) MoE work and overlaps the deferred P2-5. The plan carries both
views without reconciling which is true, which double-counts or mis-attributes
the shared-expert time between dense and MoE scope. Q4 is on P3-0 (good), but
Q4's answer must also update §2's rows, not just satisfy curiosity.

### MOD-5. The 1 TB/s HBM floor model ignores L2 residency at M=1

The floor math is `N·K·2 bytes / ~1 TB/s`, treating every dense read as HBM.
At M=1 in decode, small matrices (GDN in_proj 2048×2048 = 8 MB; shared
2048×512 = 2 MB) are re-read every layer and may be substantially L2-Cached
across the 40-layer step (L2 hits would raise the *achievable* floor and
shrink the computed ratios the other way from the plan's assumption). The
plan's "5.5×" and "2.5×" off-floor ratios therefore carry an L2 bias that
cuts both ways and is unreferenced. The P3-0 BW microkernel is global BW, not
the per-matrix L2-hit behavior the dense decode actually sees. Worth one
counter (L2 hit rate per dense matrix) in P3-0 so the floors aren't silently
moved by cache behavior the plan never mentions.

### MOD-6. "Q4 = half the bytes" is an approximation with real teeth that the plan overstates

The quantization caveat is a good instinct, but "its Q4 dense weights are
~half our fp16 bytes, which no kernel work can recover" is stated with more
certainty than the mechanism supports: Q4_K_XL super-blocks keep some 8-bit
outlier rows (effective bit-width > 4), and llama.cpp's decode also quantizes
**activations to Q8_1** so its "fast decode" is not purely a weight-bytes
story — it's an int8-activation × Q4-weight mmq scheme. The part "no kernel
work can recover" is therefore too strong: an activation-quantization decode
is exactly a *kernel* change (the same algorithmic class llama.cpp uses), and
it's a headroom direction the plan never names. P3-0 Q2 should explicitly
record llama.cpp's activation-quant mechanism, not just "which GEMM kernel
(mmq?), attention, norms... and what do they cost" — the *cost* table is not
the mechanism.

---

## Alternate options the plan omits (beyond its surface)

### ALT-1 (TOP). P3-0 reconciles the full-step budget before sizing anything
Extend Q0 from "layer composition" to "full §2 table sums to one measured
ms/step, every profiled row has a home and a scope decision." Until then
every candidate's ms/step crediting is ±20% regardless of the layer-count
uncertainty the plan does acknowledge. This is a diagnostic change that costs
nothing but makes the success criteria checkable — it's the single highest
value add to the plan.

### ALT-2. Put aten::mm / the real head GEMM on the map
Identify the 4×532 µs calls (P3-0 Q6). If they are the decoder/LM-head
projection running through rocBLAS viable trace, that is 2.2 ms of *named*,
*unaddressed* work — larger than P3-1, P3-3, and P3-4 individually. Even a
"skip deliberately" decision (like P3-5) is better than omission.

### ALT-3. Lead P3-2 with the house custom-kernel design (per MOD-2)
This project already has the template (`moe_q_gemm_gfx906.cu`, b128 LDS +
`__ockl_fdot2`, VGPR/floor data to 2048×2048 shapes) for an M=1 W16A16 dense
GEMM. The MoE phase proved upstream kernels are the wrong tree on gfx906. The
plan's "configuration-only first" ordering inverts the project's own evidence.

### ALT-4. Track an activation-quantization decode option (per MOD-6)
llama.cpp's Q8_1 activation × Q4 weight mmq is likely *why* llama.cpp hits
~1.4× on decode despite similarly-tuned per-kernel latency. Making this an
explicit, gated P3 candidate (measure one layer's numeric impact, sanity-diff)
would attack the actual mechanism behind the reference rather than only
"match with fp16."

### ALT-5. Make P3-5 a first-class P3-0 answer, not a silent skip
The LM head's 1.4 ms is 7% of the step. The plan's defensive "default SKIP"
risks a 7% decode gap left on the table that llama.cpp does not have (it
splices/layers its output projection efficiently). At minimum P3-0 Q2/Q6 must
compare the *head/decoder projection* kernel against llama.cpp's, since the
head is the single biggest single GEMM and the plan never cross-checks it
against the reference design except as a "0.4 ms slack, skip" footnote.

---

## Bottom line

`plan-decode-phase3.md` has the right *method* — gate on diagnostics, honest
expectation bands, refuse to chase llama.cpp's Q4 absolute number — but its
**budget is out of arithmetic**: §2 sums to ~16.25 of 20.3 ms, silently drops
`aten::mm` (2.2 ms, 10%) and the shared-expert row, compresses LLGemm1+LLMM1
into an understated 6.8 ms row, and then cites a §6 success range (2.5–5 ms →
"up to ~parity") that is inconsistent with the very per-item targets it just
wrote, at the top end approaching the parity it says not to expect.

Before any candidate is funded, fix the accountability layer (ALT-1): make the
§2 table add to a single measured step, name a scope decision for every
profiled kernel, and reconcile P3-2's target (2.9 ms) with its credited credit
(1.5–2 ms) and with P3-1. Then the plan's discipline — which is genuinely
good — can be trusted to steer the ~6 ms decode gap. As written, the plan's
honest per-item reasoning is undercut by a denominator none of it is
computed against, and the largest unaddressed single kernel is a 10% of the
step it never mentions. Fix the arithmetic and the scope list, keep the
discipline.