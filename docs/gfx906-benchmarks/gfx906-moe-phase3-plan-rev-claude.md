# Adversarial review — Phase 3 plan (`plan-decode-phase3.md`)

Reviewer: Claude Sonnet 4.6 · date: 2026-08-15 · scope: `plan-decode-phase3.md` v2 read
against `DEVLOG-moe-opt.md` (P2-3/P2-4/P2-5 graph-mode profile) and `README.md`.

The plan has the right method — diagnostics gate before candidates, honest 2–3×
expectation bands, explicit quantization-asymmetry caveat, default SKIP on the LM
head. The discipline is genuine and gets credit below. The failure is arithmetic:
the §2 budget table sums to **~16.25 ms of a stated 20.3 ms step**, silently drops
two DEVLOG profile rows, compresses LLGemm1+LLMM1, and the §6 success range is
inconsistent with its own itemized targets. There is also a stale hardware label
and one 10%-of-step kernel never named anywhere in the plan.

---

## What the plan gets right (fair credit before the critique)

- **Measured, not aspirational, floors**: every floor is sized from `N·K·2 / ~1 TB/s`
  and explicitly gated on P3-0 Q1 making that real.
- **P3-1 range scoped down from 5.5× to 2–3×**: "no M=1 kernel reaches perfect BW"
  is the correct load model.
- **P3-3 correctly latency/occupancy-bound**: at seq~500 the KV read per layer is
  ~0.5 MB → sub-µs floor vs 198 µs actual; gating on llama.cpp gap ≥2× is
  disciplined.
- **P3-4 reframed honestly**: inductor already fused; real lever is graph breaks, not
  `pass_config` flags; EV 0–1 ms is stated.
- **P3-5 default SKIP** is a mature call: only ~0.4 ms of slack over the BW floor.
- **Quantization-asymmetry caveat** stated repeatedly; plan does not chase llama.cpp's
  absolute Q4 number.

---

## Critical findings

### CRIT-1. §2 budget does not sum to the measured step — ≈4 ms unaccounted

Summing the plan's own §2 table: `6.8 + 1.6 + 1.9 + 1.75 + 1.2 + 1.0 + 2.0 = 16.25 ms`.
The plan's stated step is **20.3 ms** (§1, §6). Gap: **≈4.05 ms ≈ 20%** with no home in
the table, any candidate, or the protocol.

The DEVLOG graph-mode profile (Self CUDA ≈22.6 ms matching the 20.3 ms bench) lists
at minimum: `aiter LLGemm1 7.2 ms`, **`aten::mm 2.2 ms (10%)`**, paged attn 1.95,
MoE 1.77, `triton_matmul 1.6`, GDN 1.15, **`aiter LLMM1 1.2 ms`**, routing 1.0,
**`shared expert (Triton) 0.55 ms`**, elementwise ~2.0 → ≈20.6 ms total.

What happened:

| Profile row (DEVLOG) | ms/step | Plan §2 | Status |
|----------------------|---------|---------|--------|
| aiter LLGemm1        | 7.2     | merged into "~6.8" | compressed |
| aiter LLMM1          | 1.2     | (dropped) | **missing** |
| aten::mm (4 × 532 µs)| 2.2     | (not in §2) | **missing** |
| paged attention      | 1.95    | ~1.9   | ✓ present |
| gfx906 MoE kernel    | 1.77    | ~1.75  | ✓ present |
| triton_matmul        | 1.6     | ~1.6   | ✓ present |
| GDN/mamba            | 1.15    | ~1.2   | ✓ present |
| routing pipeline     | 1.0     | ~1.0   | ✓ present |
| shared expert        | 0.55    | (not in table) | **missing** |
| elementwise pile     | ~2.0    | ~2.0   | ✓ present |
| **Plan §2 total**    | **16.25** | vs 20.3 ms measured | **−4.05 ms** |

**Fix**: P3-0 Q0 must be extended from "layer composition" to "full-step budget
reconciliation" — reproduce §2 from one profile window, show it sums to a single
measured ms/step, and assign every named kernel to a plan row or an explicit
"out of plan" note. No candidate size is trustworthy until this closes.

---

### CRIT-2. `aten::mm` — 2.2 ms/step, 10% of the step, never mentioned anywhere

The DEVLOG is unambiguous: `aten::mm | 2.2 ms (10%) | 4 calls × 532 µs — large
dense GEMMs`. That is the **second-largest non-MoE GPU consumer** after aiter LLGemm1.
It is absent from §2's budget table, from the dense-projection breakdown, and none of
P3-0 Q0–Q5 or P3-1..P3-5 names it.

Two compounding consequences:
1. §6's "realistic capture" of 2.5–5 ms is computed against 16 of 20.3 ms elapsed —
   the unaddressed ~4 ms makes the "close ≥50% of the gap" success claim structurally
   harder than the plan acknowledges.
2. P3-5's skip reasoning ("LM head only 1.4 ms, ~0.4 ms slack") may be sizing the
   wrong GEMM: if the real LM-head path runs through `aten::mm` at 532 µs/call, the
   skip decision is built on a mislabeled row.

**Fix**: Add P3-0 Q6 — identify the 4 × 532 µs `aten::mm` calls (layer, shape,
backend, which Linear). Reconcile against the LM-head row and dense-projection table.
Make an explicit in/out-of-scope decision. Omission of a 10%-of-step kernel is a
material accounting error.

---

### CRIT-3. §6 success summary and per-candidate targets do not add up

**Double-counting P3-1 inside P3-2**: P3-2's stated target is "dense projections
8.4 → ~5.5 ms/step" = **2.9 ms saving**. But §6 credits P3-2 with only **1.5–2 ms**
and lists P3-1 (the 1.6 → 0.7 ms triton_matmul fix) as a separate **0.8–1.2 ms**
line. P3-1's row is already inside the "8.4 ms dense" surface P3-2 claims to move,
so one of them is double-counted.

**Upper bound reaches the parity it disclaims**:
- §6 range: 2.5–5 ms off 20.3 ms/step
- Upper bound (5 ms): 15.3 ms/step → **~65 t/s ≈ llama.cpp parity** (14.2 ms)
- Lower bound (2.5 ms): 17.8 ms/step → ~56 t/s — only just above the "≥50% of 6 ms"
  bar the plan calls success

The plan simultaneously states "parity is NOT guaranteed" and gives a range whose
top end touches parity. The headroom model is not coherent.

**Fix**: Restate one additive budget from the reconciled denominator. Verify P3-1
and P3-2 don't double-count the triton_matmul row. Compute the §6 range
arithmetically from per-candidate targets — if the upper bound reaches parity, say
so rather than simultaneously promising and disclaiming it.

---

## Moderate findings

### MOD-1. "MI60" label — repo already confirmed this card is MI50 (60 CU)

Plan header: "Phase 3 — non-MoE decode path on gfx906 (**MI60**)". P2-0 rocprofv3
measured `Simd_Count=240 → 60 CUs` → MI50 (MI60 = 64 CU). README.md and the DEVLOG
both carry the correction. P3-3 argues occupancy-limited parallelism at M=1; a reader
taking "MI60" literally and assuming 64 CU silently shifts that reasoning.

**Fix**: `s/MI60/MI50/` throughout, or add the same one-line footnote README.md uses.

---

### MOD-2. P3-2's aiter-knobs-first ordering inverts this project's own evidence

P3-2(a) "try aiter/rocBLAS dispatch/splitK" is the low-risk entry point; P3-2(b)
"port a minimal M=1 W16A16 kernel" is the fallback. But this branch's premise is
that upstream kernels are weak on gfx906 (no `v_mfma`; Triton tl.dot → scalar FMA;
poor codegen for this ISA), and the solution was a custom HIP kernel using
`__ockl_fdot2` + b128 LDS streaming — achieving 35–125× over Triton.

There is also no evidence cited that aiter's splitK or backend dispatch env-vars are
exposed and effective for the relevant op/shape on ROCm gfx906. "Configuration-only"
is asserted, not checked.

**Fix**: P3-0 should explicitly test whether aiter's dispatch knobs move the needle
for these shapes (conclusive, one afternoon). P3-2 should lead with the custom-kernel
branch as the primary line; the aiter-flags probe stays as a quick try. The project's
own history says the custom path is the primary, not the fallback.

---

### MOD-3. P3-1's destination is the P3-2 problem — "cheapest win" overstated

P3-1's gain is "route to the same LLGemm1 path as siblings." But the sibling LLGemm1
kernels are ~2–2.5× off the BW floor (P3-2's numbers). P3-1 delivers "as good as the
other mediocre aiter kernels," not "approach the floor."

More pointedly: ~37+ calls/step already bypass aiter and reach `triton_matmul`. That
aiter declines them is evidence of a real shape/dtype constraint, not a selection bug.
If P3-0 Q3 shows the bypass is intentional, P3-1 collapses entirely into P3-2(b).

**Fix**: P3-0 Q3 must record the aiter rejection *reason*, not just which layers emit
it. The "cheapest possible win" framing should be softened to "low-effort probe;
collapses into P3-2 if declined for cause."

---

### MOD-4. Shared expert characterized two incompatible ways across §2 and §6

§2's dense-projection table lists shared `gate_up` + `down` as "two plain fp16
Linears via LLGemm1" (~0.76 ms). P3-0 Q4 asks "what the residual `fused_moe_kernel`
(Triton, ~2/step, 0.55 ms) computes," and the DEVLOG lists "shared expert (Triton
fused_moe)" as a distinct profile row. If it is two plain LLGemm1 Linears, it is
already inside the P3-1/P3-2 dense surface; if it is a Triton `fused_moe_kernel`, it
is deferred MoE work overlapping P2-5. The plan carries both views without reconciling
them.

**Fix**: Q4's answer in P3-0 must update §2's rows, not just satisfy curiosity. The
shared-expert row must get a home in the reconciled budget table (CRIT-1 fix) with an
explicit scope decision: dense-side or MoE-side.

---

### MOD-5. BW floor model ignores L2 residency at M=1 — ratios carry hidden bias

Every "off-floor" ratio uses `N·K·2 / ~1 TB/s` (global HBM). At M=1 in decode the
same weight matrices are re-read every layer: GDN in_proj is 8 MB, shared gate_up is
4 MB — both small relative to the MI50's L2. The DEVLOG already records ~61% L2 hit
rate at M=512 w13 (TCC_HIT/TCC_MISS counters confirmed to work on this arch). Partial
L2 residency raises the achievable floor and shrinks the computed ratios without the
plan knowing.

**Fix**: One P3-0 counter pass with TCC_HIT/TCC_MISS per dense kernel class in
graph-mode steady state. Either confirms floors are global-BW (ratios correct) or
reveals L2 residency (floors lower, less available gain per candidate than claimed).

---

### MOD-6. "Q4 = half the bytes, no kernel work can recover" overstated

The quantization caveat is correct in spirit but understates the mechanism: llama.cpp's
fast decode is an **int8 activation × Q4 weight mmq scheme** — activations quantized
to Q8_1 per token block. Its decode advantage is not purely weight-bytes; it is also a
different arithmetic regime (INT8×INT4 MADs). "No kernel work can recover" is too
strong: activation-quantization decode is exactly a kernel change — the same
algorithmic class llama.cpp uses.

**Fix**: P3-0 Q2 should record llama.cpp's activation-quant mechanism explicitly (not
just the cost table). Then make an explicit decision: activation-quant decode is a
gated Phase 3 candidate (measure one layer's numeric impact + sanity-diff), or it is
ruled out for named reasons. The current framing may attribute to "irreducible quant
difference" what is actually a recoverable kernel design.

---

## Alternate options the plan omits

**ALT-1 (highest value)**: Extend P3-0 Q0 to full-step budget reconciliation — make
§2 reproduce from one profile window and sum to one measured ms/step, with every
profiled row assigned to a plan row or an explicit "out of plan" note. Zero cost;
makes every subsequent candidate checkable. This is the single highest-value change
to the plan as written.

**ALT-2**: Put `aten::mm` on the map (CRIT-2). Four calls at 532 µs = 2.2 ms. The
plan does not know if this is the LM head, the decoder projection, or something else.
Even a deliberate "skip" (like P3-5) is better than silence for a 10%-of-step kernel.

**ALT-3**: Lead P3-2 with the proven custom-kernel design (MOD-2). `moe_q_gemm_gfx906.cu`
already demonstrates b128 LDS + `__ockl_fdot2` at M=1 shapes; the dense W16A16 case
(no dequant) is strictly simpler. The "configuration-only first" ordering inverts the
project's evidence.

**ALT-4**: Name activation-quantization decode as a gated candidate (MOD-6). llama.cpp's
Q8_1×Q4 mmq decode is likely a meaningful part of its 1.43× lead, separate from
weight-bytes. One layer's numeric impact measurement (sanity-diff: greedy text identical
or within fp tolerance) gives the data to decide before Phase 3 closes.

**ALT-5**: `aiter LLMM1` (1.2 ms, 5%) needs an explicit scope decision. If it is the
same aiter op at a different shape, it belongs inside the P3-1/P3-2 surface. If it is
a shared-expert or small-batch variant, it belongs in the shared-expert reconciliation.
Either way it should not be the compressed stub the §2 merge implies.

---

## Bottom line

`plan-decode-phase3.md` has the right method: gate on diagnostics, honest expectation
bands, refuse to chase llama.cpp's Q4 absolute number, skip-by-default on the LM head.
That discipline is genuine.

The budget is out of arithmetic. §2 sums to ~16.25 of 20.3 ms, drops `aten::mm`
(2.2 ms, 10%) entirely, compresses LLGemm1+LLMM1 into an understated 6.8 ms row, and
leaves the shared-expert row with no home. §6's success range (2.5–5 ms → "58–70 t/s")
cannot be derived from the itemized targets without double-counting P3-1 inside P3-2,
and at its upper bound reaches the parity it says not to expect.

Fix the accounting layer first (ALT-1 + CRIT-1). Add P3-0 Q6 for the `aten::mm`
calls. After that, the plan's existing structure — P3-0 gates, ordered candidates,
honest ranges — can be trusted to steer the ~6 ms decode gap. As written, the
per-item reasoning is sound but the denominator is missing 20% of the step.
