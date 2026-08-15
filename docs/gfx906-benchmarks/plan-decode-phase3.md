# Phase 3 — non-MoE decode path on gfx906 (MI50)

Status: DRAFT v3 — post adversarial review (DS4 + Claude Sonnet 4.6,
2026-08-15). Scope: close the remaining decode gap vs llama.cpp. Prefill is
already 2.6× faster than llama.cpp — out of scope unless a candidate helps
both for free.

Review changes (v2→v3):

- **Hardware label corrected**: MI60 → MI50 32 GB (60 CU) throughout. P2-0
  rocprofv3 confirmed Simd_Count=240 → 60 CUs; the "MI60" label was VRAM-only.
- **§2 budget rebuilt**: LLGemm1 and LLMM1 are now separate rows; `aten::mm`
  (2.2 ms, 10%) added; shared expert (0.55 ms) added. Table now sums to
  ≈20.6 ms, matching the measured 20.3 ms step.
- **P3-0 Q0 extended**: from "layer composition" to "full-step budget
  reconciliation" — every profiled row must have a plan home or explicit
  out-of-scope note.
- **P3-0 Q6 added**: identify the 4 × 532 µs `aten::mm` calls (shape, layer,
  backend). Reconcile against LM-head row. Make explicit in/out-of-scope
  decision.
- **P3-0 Q1 extended**: add TCC_HIT/TCC_MISS counter pass per dense kernel
  class to measure L2 residency at M=1.
- **P3-0 Q2 extended**: record llama.cpp's activation-quant mechanism (Q8_1
  activation quantization), not just its per-kernel cost table.
- **P3-1 reframed**: "cheapest win" → "low-effort probe; collapses into P3-2
  if bypass is a shape/dtype constraint rather than a selection bug." P3-0 Q3
  now requires recording the aiter rejection reason.
- **P3-2 reordered**: (a) quick aiter probe (time-boxed, P3-0 Q3 drives it);
  (b) custom M=1 W16A16 kernel as primary development path — not fallback.
- **P3-5 updated**: skip reasoning now provisional pending P3-0 Q6 identity
  of the `aten::mm` calls; if those are the LM head, slack is different.
- **§6 restated**: additive per-candidate targets from the reconciled 20.3 ms
  denominator; P3-1 is a sub-item of the dense surface (not double-counted
  against P3-2); upper/lower bounds do not claim parity.
- **§7 risks updated**: "no kernel work can recover [the Q4 gap]" removed;
  activation-quant decode noted as a gated future candidate.

Review changes (v1→v2, preserved): P3-0 Q0 first added; P3-1 range softened
to 2–3×; P3-4 reframed (inductor already fused; lever is graph breaks); P3-3
latency-bound nature made explicit; §6 success criteria + quantization caveat.

---

## 1. Where we are

Same model family (Qwen3.5/3.6-35B-A3B, hybrid GDN + full-attn, 256-expert
top-8 MoE), MI50 32 GB (gfx906, 60 CU), single request:

| engine | prefill pp=2048 | decode |
|--------|-----------------|--------|
| llama.cpp (Q4_K_XL) | 807 t/s | **70.3 t/s** (14.2 ms/step) |
| vLLM + cudagraphs (AWQ int4) | ~2140 t/s | ~49 t/s (20.3 ms/step) |

Gap: **~6 ms/step (≈30% of the step)**, entirely in GPU kernel time
(cudagraphs remove the launch overhead; eager is not the target anymore).
Primary metric: **serving mode** (`BENCH_EAGER=0`, `FULL_DECODE_ONLY`),
decode tok/s and ms/step.

---

## 2. Per-step decode budget (measured, graph mode, M=1)

From the DEVLOG P2-3/P2-4/P2-5 graph-mode profile (torch profiler,
`FULL_DECODE_ONLY`, Self CUDA ≈22.6 ms/step matching the 20.3 ms bench).
All rows are from that single profile window; percentages are of 20.3 ms.

| component | ms/step | % | status |
|-----------|---------|---|--------|
| dense projections, aiter LLGemm1 (`LLMM1` op) | **7.2** | 35% | **P3 target** |
| `aten::mm` (4 calls × 532 µs) | **2.2** | 11% | **P3-0 Q6 — scope TBD** |
| `vllm::rocm_unquantized_gemm` / `triton_matmul` | ~1.6 | 8% | **P3 target (P3-1)** |
| paged attention (custom FA), 10 layers × ~198 µs | ~1.9 | 9% | **P3 target (P3-3)** |
| gfx906 MoE routed kernel (Phase 1/2) | ~1.75 | 9% | done |
| GDN recurrence + conv + chunk ops | ~1.15 | 6% | watch |
| aiter LLMM1 (small-batch dense path) | **1.2** | 6% | **P3 target — inside LLGemm1 surface** |
| routing pipeline (topk+align+sort) | ~1.0 | 5% | P2-4 deferred |
| shared expert (Triton `fused_moe_kernel`) | **0.55** | 3% | **P3-0 Q4 — scope TBD** |
| elementwise/norm/copy/zeros pile | ~2.0 | 10% | **P3 target (P3-4)** |
| **table total** | **≈20.6** | ≈101% | matches measured 20.3 ms |

Notes:
- **LLGemm1 (7.2) and LLMM1 (1.2) are separate aiter paths.** Previous v2
  merged them into a single "~6.8" row, understating the surface by 1.6 ms.
- **`aten::mm` (2.2 ms) was absent from v2.** Its identity (which layer,
  which shape) is the first thing P3-0 Q6 must answer. Until then it carries
  a "(scope TBD)" note — it is too large to leave unnamed.
- **Shared expert (0.55 ms)** may be MoE-side (Triton `fused_moe_kernel`,
  overlaps deferred P2-5) or dense-side (inside LLGemm1 surface). Q4 decides.
- **Floor math** (`N·K·2 / ~1 TB/s`) treats every read as global HBM. At M=1
  decode, small matrices re-read 40×/step may see substantial L2 residency —
  DEVLOG measured ~61% L2 hit at M=512 w13. P3-0 Q1 now includes a
  TCC_HIT/TCC_MISS counter pass per dense kernel class; floors will be revised
  if L2 residency is material.

Dense-projection breakdown (M=1, weight-read-bound; bytes = N·K·2 fp16;
floor = bytes / achievable HBM BW measured in P3-0 Q1):

| projection (N,K) | layers | µs/call | floor @1TB/s | ratio |
|------------------|--------|---------|--------------|-------|
| GDN in_proj (12288, 2048) | ~30 | ~80 | ~50 | 1.6× |
| LM head (248320, 2048) | 1 | ~1420 | ~1000 | 1.4× |
| GDN out_proj (2048, 4096) | ~30 | ~41 | ~17 | **2.4×** |
| FA qkv (9216, 2048) | ~10 | ~64 | ~38 | 1.7× |
| shared gate_up (1024, 2048) | ~40 | ~10 | ~4 | **2.5×** |
| shared down (2048, 512) | ~40 | ~9 | ~2 | 4.5× |
| router (256, 2048) | ~40 | ~6 | ~1 | — |
| GDN small proj (64, 2048) | ~30 | ~4 | <1 | — |
| Triton fallback (2048, 2048) | ~37? | ~44 | ~8 | **5.5×** |

All ratios are provisional until P3-0 Q0 reconciles the layer counts and Q1
provides the real achievable BW (and L2 hit rate per class).

---

## 3. Open questions P3-0 must answer (diagnostics, no code changes)

**0. Full-step budget reconciliation** (extended from v2's "layer composition"):
   Reproduce the §2 budget table from one profile window. Confirm it sums to
   a single measured ms/step. Every named kernel must be assigned to a §2 row
   or an explicit "out of plan" note with a stated reason. Particular targets:
   reconcile the ~37 GDN-shaped calls against the assumed 30/10 GDN/FA split;
   confirm LLGemm1 vs LLMM1 row identities; account for `aten::mm`. Until
   this is done, all §2 ratios and §4 candidate sizes are provisional.

**1. Achievable HBM BW on this MI50** (simple sum/copy microkernel, fp16):
   Sets every floor in §2. Record sclk/mclk under load.
   **Also**: one `rocprofv3 --pmc TCC_HIT TCC_MISS` pass per dense kernel class
   (LLGemm1, triton_matmul) in graph-mode steady state. A high L2 hit rate
   raises the achievable floor and shrinks the computed off-floor ratios —
   this determines whether the ratios in §2 are real opportunities or L2 noise.

**2. llama.cpp per-kernel decode table**: `rocprofv3 --hip-trace` around
   `llama-bench -p 0 -n 256` (Q4 model) → aggregate kernel times/step.
   Reference design: which GEMM kernel (mmq?), attention, norms does it use?
   **Also**: record llama.cpp's activation-quant mechanism explicitly — it uses
   Q8_1 activation quantization per token block (int8 × Q4 mmq). Its decode
   advantage is not purely weight-bytes; it is also a different arithmetic
   regime. "Part of llama.cpp's lead is quantization, not kernel quality" is
   true but incomplete; the mechanism determines what fraction is recoverable
   by kernel work alone.

**3. Which vLLM layers emit `rocm_unquantized_gemm`** (~37 calls/step):
   Python probe (inspect Linear layers / backend selection for this arch on
   ROCm). Hypothesis: FA o_proj and/or GDN out-proj variants that aiter
   declines. **Record the aiter rejection reason**, not just which layers — if
   aiter declines for a real shape/dtype constraint (not a selection bug), P3-1
   collapses directly into P3-2(b) and the "quick win" path disappears.

**4. Shared-expert path identity**: the DEVLOG records both "shared gate_up/
   down via LLGemm1" (dense surface) and "shared expert (Triton fused_moe)"
   as a distinct 0.55 ms profile row. These are not compatible. Determine
   which path this model actually uses and update §2's shared-expert row.
   Q4's answer determines whether 0.55 ms is inside the P3-1/P3-2 dense
   scope or deferred MoE-adjacent scope (overlapping P2-5).

**5. Elementwise pile identity** (~2 ms/step): which of the many small
   triton/aten elementwise kernels dominate; whether inductor pass_config
   fusions (`fuse_norm_quant`, `fuse_act_quant`, …) are applicable to this
   model without changing numerics.

**6. `aten::mm` identity** (new): identify the 4 × 532 µs `aten::mm` calls —
   which layer, which shape (N×K), which backend path leads here instead of
   LLGemm1 or the Triton fallback. Hypotheses: FA output projection (o_proj);
   or the decoder's final projection (LM head); or a GDN linear variant.
   The identity matters because: (a) if these are the LM head, P3-5's
   "~0.4 ms slack, default SKIP" is using the wrong row — the real slack may
   be larger or smaller; (b) at 2.2 ms total, this is larger than P3-1, P3-3,
   and P3-4 individually and warrants an explicit in/out-of-scope decision, not
   omission.

---

## 4. Ordered candidates (each: test → bench → commit, per common protocol)

Ordering = measured ms/step × feasibility, from the reconciled §2 table.
Every step is gated on P3-0. Sizes below are from the DEVLOG profile; they
will be revised after P3-0 Q0–Q6 complete.

### P3-1 — Triton `rocm_unquantized_gemm` fallback (~1.6 ms/step)

Low-effort probe: find why these [2048→2048] M=1 GEMMs bypass aiter (P3-0 Q3)
and whether routing them to LLGemm1 is possible. **This is a probe, not a
promised win**: the ~37 calls already bypassing aiter is itself evidence of a
shape/dtype constraint rather than a selection bug. If P3-0 Q3 records an
intentional rejection reason, fold P3-1 directly into P3-2 — there is no
separate easy win here.

Expectation if the bypass IS a bug: 1.6 → ~0.5–0.9 ms/step (2–3×; the 5.5×
floor assumes perfect HBM utilization that no M=1 kernel reaches). Note that
destination (LLGemm1) is itself ~2–2.5× off floor — the ceiling for P3-1's
gain is "as good as the other mediocre aiter kernels," not "approach the floor."

**Gate**: P3-0 Q3 identifies the layers and records the rejection reason.
Change is config/backend-selection only; if that is insufficient, proceed to
P3-2.

### P3-2 — M=1 dense GEMM efficiency (7.2 + 1.2 ms/step LLGemm1+LLMM1 surface, plus 1.6 ms Triton fallback)

The combined dense + LLMM1 + Triton-fallback surface is **≈10.0 ms** (7.2 +
1.2 + 1.6). BW floor across the projection table is ~4.5 ms; realistic capture
at M=1 is ~2–3 ms. Options in priority order:

**(a) Quick aiter probe** (time-boxed at 1 day, driven by P3-0 Q3 output):
env/backend flags, splitK for N≤2048. If these knobs exist and move the
needle by ≥20% on one shape, continue. If aiter declines for a
shape/dtype constraint (Q3 records the reason), stop immediately and
proceed to (b). Do not invest further in (a) past one run per flag variant.

**(b) Custom M=1 W16A16 dense kernel — primary development path**:
`moe_q_gemm_gfx906.cu` already demonstrates b128 LDS weight streaming +
`__ockl_fdot2` dot loop at M=1. Dense W16A16 (no dequant, no packing) is
strictly simpler. The MoE phase proved upstream kernels are the wrong
tree on gfx906 for this class of problem; the custom kernel is the primary
bet, not the fallback. Reference: llama.cpp's mmq single-pass decode kernel
from P3-0 Q2 trace.

Target (combined P3-1 + P3-2, no double-counting): dense surface
~10 ms → ~6.5–7.5 ms, i.e. **~2.5–3.5 ms saving**.
**Gate**: P3-0 BW number + L2 hit rate make the floors real; micro-bench per
shape before touching the model path.

### P3-3 — paged attention decode (~1.9 ms/step, 10 layers × 198 µs)

At seq~500 the KV read per layer is ~0.5 MB — BW floor is sub-microsecond,
so 198 µs is **latency/occupancy-bound**, not BW-bound. Likely cause: poor
parallelism at M=1 (GQA kv_heads=2). Compare against llama.cpp's attention
kernel from the P3-0 Q2 trace; if it is materially faster at batch=1, study
its work-split before writing anything.

**Gate**: P3-0 shows a gap ≥2× vs llama.cpp's attention; otherwise defer.
FA prefill advantage must not regress — bench both phases.

### P3-4 — elementwise/norm pile (~1–2 ms/step)

Inductor has already fused much of this (profile names are
`triton_*_fused_*`). The remaining kernels are mostly OUTSIDE compiled regions
(custom-op boundaries). The real lever is reducing graph breaks / moving work
into compiled regions, not `pass_config` flags. P3-0 Q5 must identify which
pile items live where; then enable applicable fusions and measure.

Numerics-sensitive: correctness test + sanity generation required; any output
diff beyond fp rounding → revert. Expected value uncertain: **0–1 ms/step**.

### P3-5 — LM head (~1.4 ms/step in dense-projection table)

1 GB weight read per step = ~1 ms BW floor; ~0.4 ms of apparent slack.
**Default: SKIP** with the following caveat: P3-0 Q6 may reveal that the real
LM-head path runs through `aten::mm` (4 × 532 µs = 2.1 ms), which would change
the row identity, the slack calculation, and potentially the skip decision.
Revisit P3-5 after Q6 resolves the `aten::mm` identity. If the true head is
the `aten::mm` path, P3-5 becomes a 2.1 ms item with more slack and the skip
reasoning must be re-evaluated.

Options if Q6 changes the picture: study llama.cpp's head/decoder projection
kernel from the P3-0 Q2 trace as the reference. Skip fp8/fp4 lm_head (changes
model numerics) unless numeric impact is measured and acceptable.

### `aten::mm` (2.2 ms, 11%) — pending P3-0 Q6

Too large to leave as a silent omission but identity unknown. After Q6:
- If these are the LM-head path → subsumes P3-5; re-evaluate skip.
- If these are FA o_proj or GDN-variant → add as a candidate inside the
  P3-1/P3-2 dense surface.
- If these are outside this phase's scope → say so explicitly and record why.

### Explicitly out of scope

- Prefill GEMMs (we are 2.6× ahead of llama.cpp there).
- MoE routed kernel (Phase 1/2 done; 8–9% of step, at its issue-bound ceiling).
- P2-4 routing pipeline (~1 ms) — revisit only if P3 lands and gap vs
  llama.cpp is still >2 ms.
- Multi-batch serving (single-request bench; batched decode changes the whole
  budget — separate project).
- **Activation-quantization decode (Q8_1 × Q4 mmq)**: llama.cpp's fast decode
  uses int8 activation quantization, which is part of (not all of) its
  remaining decode lead. This is a kernel-level change that could narrow the
  irreducible gap; it is out of scope for Phase 3 but is a named gated
  candidate for a future phase. Gate: measure one layer's numeric impact
  (sanity-diff greedy text) and the throughput delta before committing.

---

## 5. Common protocol (every step)

1. Correctness: existing `tests/kernels/moe/test_gfx906_moe_gemm.py` stays
   green for any MoE-adjacent change; model-level sanity generation (fixed
   prompt, greedy) must match the pre-change output exactly for config-only
   changes, or stay within fp tolerance + coherent text otherwise.
2. Serving-mode bench: `BENCH_EAGER=0` full run (pp=2048/tg=256) — record
   total tok/s and derived decode ms/step. Also run eager if a change could
   affect prefill.
3. Micro-bench any new/changed kernel per shape before model integration.
4. Separate commit + dev-log entry (positive AND negative results).

---

## 6. Expected outcome (success criteria)

Per-candidate targets against the reconciled 20.3 ms/step denominator:

| candidate | saving (best case) | saving (realistic) | cumulative realistic |
|-----------|-------------------|--------------------|---------------------|
| P3-1 (probe only, if bug) | ~1.1 ms | ~0.5 ms | ~0.5 ms |
| P3-2 (dense kernel, incl. P3-1) | ~3.5 ms | ~2.0 ms | ~2.5 ms |
| P3-3 (attention) | ~1.0 ms | ~0.5 ms | ~3.0 ms |
| P3-4 (elementwise) | ~1.0 ms | ~0.5 ms | ~3.5 ms |
| aten::mm (post-Q6) | TBD | TBD | — |

Note: P3-1 is a sub-item of the dense surface. The P3-2 row above covers the
full dense + Triton-fallback surface; P3-1's gain is not added separately.

**Realistic range: ~2–4 ms off 20.3 ms/step** → **~53–65 t/s decode**.
Parity with llama.cpp's 70.3 t/s is NOT the goal and is not reachable on
this budget: a structural part of llama.cpp's decode lead is its int8-
activation × Q4-weight mmq arithmetic (Q8_1 quant per token block), which
halves both weight and activation traffic vs our fp16. Phase success =
**close ≥50% of the 6 ms gap (≥3 ms) with measured per-kernel evidence and
no prefill regression**; failure to reach parity is an acceptable, documented
outcome.

---

## 7. Risks

- **Quantization asymmetry vs llama.cpp**: its Q4 dense weights are ~half our
  fp16 bytes. Additionally, its fast decode uses int8 activation quantization
  (Q8_1), not just weight quantization — this is a further arithmetic
  advantage that kernel tuning of fp16 paths cannot fully recover. The gap
  that is "irreducible without changing arithmetic" is larger than the weight-
  bytes component alone. Target the kernel-tuning gap; the arithmetic gap is a
  separate, named future candidate (see §4).
- **Hybrid GDN model**: state-update kernels (GDN recurrence, conv1d) are
  Triton and were tuned by upstream for other archs; touching them is high
  risk/low reward here (watch list only).
- **Numerics**: any fusion or backend switch changes fp reduction order;
  greedy-output diffing is the tripwire.
- **Scope creep**: this phase is decode-only, single-request, gfx906. If a
  candidate turns out to need upstream aiter/FA work, stop and re-plan.
- **Provisional numbers**: §2 layer counts/ratios rest on one profiled window
  (P3-0 Q0 must reconcile before any step is sized from them). L2 residency
  at M=1 may shift every floor estimate (P3-0 Q1 TCC counter pass).
- **aiter tunability on gfx906**: P3-2(a) assumes aiter splitK/dispatch knobs
  are exposed and effective for the relevant shapes. They may not be. Time-box
  the probe at 1 day; do not invest in (a) past one run per flag variant.
