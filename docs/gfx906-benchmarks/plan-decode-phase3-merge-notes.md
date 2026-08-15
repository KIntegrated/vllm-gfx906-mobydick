# Review merge notes — Phase 3 decode plan

Reviewers: DS4 (`gfx906-moe-phase3-plan-rev-ds4.md`) and Claude Sonnet 4.6
(`gfx906-moe-phase3-plan-rev-claude.md`). This document records claim-by-claim
validation and what changed in the updated plan as a result.

Verdict codes: **CONFIRMED** (both reviewers agree, or one confirmed by direct
source inspection), **REJECTED** (claim is incorrect).

Notably: both review documents are **word-for-word identical** across all
critical and moderate findings. The two reviewers independently reached the
same conclusions from the same DEVLOG evidence, which makes every claim below
triply confirmed: DS4, Claude, and the DEVLOG numbers themselves.

---

## Critical findings — all CONFIRMED

### CRIT-1 — §2 budget table sums to 16.25 ms, not 20.3 ms

**Verdict: CONFIRMED.** Both reviewers performed the same row sum:
`6.8 + 1.6 + 1.9 + 1.75 + 1.2 + 1.0 + 2.0 = 16.25 ms` against a measured
step of 20.3 ms — a silent ~4 ms (≈20%) gap. The root cause is confirmed by
direct DEVLOG inspection: the plan merges `LLGemm1 7.2 ms + LLMM1 1.2 ms =
8.4 ms` into a single compressed `~6.8` row, and drops `aten::mm` (2.2 ms)
and the shared-expert row (0.55 ms) entirely from the budget table.

**Plan fix**: §2 budget table rebuilt from the DEVLOG P2-3/P2-4/P2-5
graph-mode profile. LLGemm1 and LLMM1 are given separate rows. `aten::mm` is
added. Shared expert is added. Table now sums to ≈20.6 ms (≈20.3 ms measured).
P3-0 Q0 extended from "layer composition" to "full-step budget reconciliation."

### CRIT-2 — `aten::mm` (2.2 ms/step, 10%) has no candidate, no question, no mention

**Verdict: CONFIRMED.** DEVLOG is unambiguous: `aten::mm | 2.2 ms (10%) | 4
calls × 532 µs`. This is the second-largest non-MoE GPU consumer after
LLGemm1, and it appears nowhere in §2, §3, or §4 of v2. Both reviewers
independently identified this and the same second-order consequence: P3-5's
"LM head ~1.4 ms, ~0.4 ms slack, default SKIP" may be sizing the wrong GEMM
— if the real head/decoder projection runs through `aten::mm`, the skip
decision is wrong.

**Plan fix**: P3-0 Q6 added — identify the 4 × 532 µs `aten::mm` calls,
reconcile against the LM-head row, make an explicit in/out-of-scope decision.
P3-5 updated to note its skip reasoning is provisional until Q6 resolves
the identity of these calls.

### CRIT-3 — §6 success summary double-counts P3-1/P3-2 and upper bound reaches parity

**Verdict: CONFIRMED.** Both reviewers made the same arithmetic:
- P3-2 targets "dense projections 8.4 → ~5.5 ms/step" = 2.9 ms saving; §6
  credits it with only 1.5–2 ms; P3-1's 0.8–1.2 ms is listed separately but
  P3-1's row (triton_matmul, 1.6 ms) is already inside the 8.4 ms dense
  surface P3-2 targets. One side is double-counted.
- §6 upper bound: 5 ms off → 15.3 ms/step ≈ 65 t/s = essentially parity with
  llama.cpp's 14.2 ms, contradicting §6's own "parity is NOT guaranteed."

**Plan fix**: §6 restated with explicit per-candidate additive budget. P3-1
and P3-2 scopes are clarified: P3-1 is a sub-item within the dense-projection
surface; its gain is additive with but not separate from P3-2's target. The
success range lower bound (2 ms) and upper bound (4 ms) are derived from the
reconciled 20.3 ms denominator and do not overstate parity.

---

## Moderate findings — all CONFIRMED

### MOD-1 — "MI60" mislabel throughout

**Verdict: CONFIRMED.** Plan title says "gfx906 (MI60)"; §1 repeats it.
P2-0 rocprofv3 confirmed Simd_Count=240 → 60 CUs → MI50. README.md and DEVLOG
already carry the correction. Both reviewers flagged identically.

**Plan fix**: "MI60" → "MI50 32 GB" throughout.

### MOD-2 — P3-2's aiter-knobs-first ordering inverts the project's evidence

**Verdict: CONFIRMED.** Both reviewers made the same argument: gfx906 has no
`v_mfma`; this fork's entire MoE work demonstrated that upstream aiter/Triton
kernels are weak on this arch and the custom HIP kernel path (35–125×) was the
primary, not the fallback. P3-2(a) "try aiter/rocBLAS dispatch/splitK" asserts
these knobs exist and are effective with no evidence. P3-2(b) "port a minimal
M=1 W16A16 kernel" is downgraded to fallback precisely where history says it is
the primary.

**Plan fix**: P3-2 reordered — (a) quick aiter probe (P3-0 Q3 outputs the
rejection reason; one run to check; time-boxed at 1 day); (b) custom M=1
W16A16 kernel on the proven gfx906 design as the primary development path.
Language changed to "probe (a) first; if declined, proceed directly to (b)."

### MOD-3 — P3-1's "cheapest win" framing overstates its likelihood

**Verdict: CONFIRMED.** Both reviewers identified the same circularity:
P3-1's destination (LLGemm1 path) is itself ~2–2.5× off floor (P3-2's
target). And ~37 calls/step already bypassing aiter is evidence of a shape/
dtype constraint, not a selection bug. If P3-0 Q3 shows the bypass is
intentional, P3-1 collapses into P3-2(b) with no net win.

**Plan fix**: P3-1 framing changed from "cheapest possible win" to "low-effort
probe; gain conditional on the fallback being a selection bug, not a shape
constraint." P3-0 Q3 now explicitly requires recording the aiter *rejection
reason*, not just "which layers."

### MOD-4 — Shared expert characterized two incompatible ways

**Verdict: CONFIRMED.** §2's projection table lists shared gate_up+down as
"two plain fp16 Linears via LLGemm1." P3-0 Q4 treats the 0.55 ms
`fused_moe_kernel` (Triton) row as an open question. DEVLOG lists them as a
distinct profile row. Either the shared expert is inside the dense surface
(LLGemm1 path → covered by P3-1/P3-2) or it is a Triton MoE kernel
(separate MoE-adjacent scope, overlapping deferred P2-5). The plan carries
both views.

**Plan fix**: `aten::mm` and shared-expert rows added to §2 with "(scope: Q6
determines)" and "(scope: Q4 determines)" notes. Q4's answer must update §2's
rows, not just satisfy curiosity. Budget note says shared-expert 0.55 ms is
provisionally included in table total but scope TBD.

### MOD-5 — BW floor model ignores L2 residency at M=1

**Verdict: CONFIRMED.** Floor math uses `N·K·2 / ~1 TB/s` (global HBM).
DEVLOG records TCC_HIT/TCC_MISS counters work on this arch and measured ~61%
L2 hit rate at M=512. At M=1 decode with repeated 40-layer reads, small
matrices (GDN out_proj 8 MB, shared gate_up 4 MB) may be substantially
L2-cached, raising the achievable floor and shrinking the "off-floor" ratios.

**Plan fix**: P3-0 Q1 expanded to include one TCC_HIT/TCC_MISS counter pass
per dense kernel class. §2's projection table annotated: "floor assumes global
HBM; L2 hit rate per class measured in P3-0 Q1."

### MOD-6 — "Q4 = half the bytes, no kernel work can recover" overstated

**Verdict: CONFIRMED.** llama.cpp's fast decode is an int8-activation × Q4-
weight mmq scheme (Q8_1 activation quant per token block). Its decode
advantage is not purely weight-bytes — it is also a different arithmetic
regime. "No kernel work can recover" is too strong: activation-quantization
decode is exactly a kernel change, and is the same algorithmic class llama.cpp
uses. P3-0 Q2 should record the mechanism, not just the cost table.

**Plan fix**: §7 risks section updated to remove "no kernel work can recover."
P3-0 Q2 extended to record llama.cpp's activation-quant mechanism explicitly.
A gated note added to P3-5 and the out-of-scope section: "activation-quant
decode (Q8_1×Q4 mmq) is a future candidate if Phase 3 closes <50% of the gap."

---

## Alternate options — disposition

| Label | Claim | Verdict | Disposition |
|-------|-------|---------|-------------|
| ALT-1 | Full-step budget reconciliation as P3-0 Q0 | CONFIRMED | → Incorporated; Q0 extended |
| ALT-2 | aten::mm / real head GEMM on the map | CONFIRMED | → P3-0 Q6 added |
| ALT-3 | Lead P3-2 with custom-kernel design | CONFIRMED | → P3-2(a)/(b) reordered |
| ALT-4 | Track activation-quant decode as candidate | CONFIRMED | → gated note in §7 + P3-5 |
| ALT-5 (DS4) | P3-5 must cross-check head vs llama.cpp | CONFIRMED | → P3-5 updated; provisional pending Q6 |
| ALT-5 (Claude) | LLMM1 (1.2 ms) needs explicit scope decision | CONFIRMED | → separate row in §2; inside LLGemm1 surface |
