# Phase 3 — non-MoE decode path on gfx906 (MI50)

Status: v4 — **P3-0 complete** (2026-08-15); §2/§3 replaced with measured
answers, §4 re-prioritized. Scope: close the remaining decode gap vs
llama.cpp. Prefill is already 2.6× faster than llama.cpp — out of scope
unless a candidate helps both for free.

P3-0 outcomes (details in DEVLOG "P3-0" section):

- Hardware label confirmed **MI50 32 GB** (lspci 1002:66a1; device string
  "MI60 / MI50"). 60 CUs.
- HBM read BW = **798 GB/s**; TCC hit at M=1 dense gemms ~14.5% → floors
  are real, L2 does not rescue small m=1 gemms.
- Layer composition: **30 GDN + 10 FA** (two independent confirmations).
- The old "aten::mm 4×532 µs = 2.2 ms/step" row is **VOID** — warmup/capture
  artifact of the P2-3 window; zero M=1 aten::mm in steady-state decode.
- The entire `triton_matmul` row (1.63 ms/step) is ONE tiny Linear per
  layer: `shared_expert_gate` [1×2048] (m=1 fails LLMM1's m%4==0 → Triton).
- llama.cpp decode = 14.32 ms/step kernel time; its dense weights are Q8_0
  (half our fp16 bytes); MoE ≈ parity with us; **attention is 3–10× faster**;
  GDN is slower than ours.
- Reconciled vLLM budget: ~17.6 ms kernel + ~2.7 ms inter-kernel gap =
  20.3 ms wall.

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

## 2. Per-step decode budget (RECONCILED, graph mode, M=1)

Kernel-level sums over 31 steady-state graph steps (prefill excluded by
construction; DEVLOG P3-0 for method):

| component | ms/step | calls/step | status |
|-----------|---------|------------|--------|
| LLGemm1 dense projections (aiter, incl. shared expert 80 + LM head) | **5.83** | 230 | **P3-2 target** |
| `triton_matmul` = `shared_expert_gate` [1×2048] ×40 layers | **1.63** | 40 | **P3-1 target (precisely scoped)** |
| paged attention (custom FA), 10 layers × ~194 µs | **1.94** | 10 | **P3-3 target** |
| gfx906 MoE routed kernel (Phase 1/2) | 1.75 | ~78 | done |
| routing pipeline (topk+align+count_sort) | 1.06 | 79 | P2-4 deferred |
| GDN decode (recurrent + conv1d) | ~0.5 | 60 | leave alone (faster than llama.cpp) |
| elementwise/norm/copy pile | ~2.3 | ~300 | deprioritized (P3-4) |
| fused_moe_kernel (Triton, residual, 2/step) | 0.39 | ~2 | out of scope |
| other small kernels (GDN/attn per-row ops) | ~2.2 | — | watch |
| **kernel total** | **~17.6** | | |
| inter-kernel gap (wall 20.3 − kernel 17.6) | **~2.7** | | attacked via kernel-count cuts |

Notes:
- The v3 table's "aten::mm 4×532 µs = 2.2 ms/step" row is **VOID**: it was a
  warmup/capture artifact of the P2-3 profile window; steady-state decode has
  zero M=1 aten::mm calls (shape-aware profile + kernel-level reconciliation).
- Shared expert = two plain fp16 Linears per layer via LLMM1 (inside the
  LLGemm1 row); the 0.39 ms Triton `fused_moe_kernel` residual is a separate
  small path, out of scope.
- **Floor math** (`N·K·2 / 798 GB/s`) treats every read as global HBM.
  Confirmed by P3-0: TCC_HIT/(HIT+MISS) ≈ 14.5% for M=1 dense gemms, so L2
  residency is not material and the floors below hold.

Dense-projection breakdown inside LLGemm1 (M=1, fp16 weights;
floor = N·K·2 B / 798 GB/s):

| projection (N,K) | layers | µs/call | floor @798GB/s | ratio |
|------------------|--------|---------|----------------|-------|
| GDN in_proj (12288, 2048) | 30 | ~80 | ~63 | 1.3× |
| LM head (248320, 2048) | 1 | ~1420 | ~1280 | 1.1× |
| GDN out_proj + FA o_proj (2048, 4096 / 2048) | 30+10 | ~33 | ~21–47 | ~1.5–2× |
| FA qkv (9216, 2048) | 10 | ~64 | ~47 | 1.4× |
| shared gate_up (1024, 2048) | 40 | ~10 | ~5 | ~2× |
| shared down (2048, 512) | 40 | ~9 | ~3 | ~3× |
| router (256, 2048) | 40 | ~6 | ~1 | — |
| GDN small proj (64, 2048) | 30 | ~4 | <1 | — |

(µs/call from the eager attribution trace; ±20%. The two big rows — in_proj
and LM head — are already near floor; the mid-size rows carry most of the
remaining LLGemm1 slack.)

---

## 3. P3-0 open questions — ALL ANSWERED (see DEVLOG P3-0)

Resolved: Q0 budget reconciled + layer split 30/10; Q1 BW=798 GB/s, TCC hit
~14.5% on dense gemms; Q2 llama.cpp table captured (Q8_0 dense weights,
Q8_1 activation quant, attention 3–10× faster, MoE parity); Q3 Triton
fallback = shared_expert_gate [1×2048] (no bias anywhere — the v3 bias
hypothesis was wrong; m=1 fails LLMM1's m%4==0); Q4 shared expert = two
plain fp16 Linears via LLMM1 (dense surface); Q5 pile identified (~2.3 ms,
mostly inductor-fused + aten Fill/copyBuffer); Q6 the 2.2 ms aten::mm row is
a profile artifact, void.

(Original question list below, kept for audit.)

### Original P3-0 questions (superseded)

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

### P3-1 — `shared_expert_gate` [1×2048] scalar gemv (~1.63 ms/step) — TOP CANDIDATE

Precisely scoped by P3-0: 40 × rank-1 dot products (sigmoid gate on the
shared-expert output, one per layer), each 41 µs in Triton because m=1 fails
LLMM1's `m % 4 == 0` check in `rocm_unquantized_gemm_impl`. Floor ~5 µs.
Expected saving **~1.4 ms/step** (also removes 40 kernel launches/step,
hitting the inter-kernel gap).

Fix options, order of invasiveness:
(a) dispatch m<4 (n≤16) to `torch.nn.functional.linear` (rocBLAS gemv) —
    zero new code; measure first, rocBLAS skinny gemv may still be ~10–20 µs;
(b) zero-pad the gate weight to [4,2048] at load time + `ops.LLMM1(w, x, 4)`
    + slice — ~5 lines in `qwen3_next.py`, no new kernel;
(c) tiny custom GEMV kernel if (a)/(b) disappoint.
Correctness: greedy-output diff vs current build (gate is numerics-visible
but a rank-1 dot; fp reorder tolerance applies). **Gate**: micro-bench the
chosen option standalone before model integration.

### P3-2 — LLGemm1 dense surface (5.83 ms/step, 230 calls)

Floor across the projection table ≈ 4.6 ms @798 GB/s; the two big rows
(in_proj ~1.3×, LM head ~1.1×) are already near floor — realistic capture is
**~1–1.5 ms/step**, concentrated in the mid-size rows (out_proj/qkv/shared).
Options in priority order:

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

(P3-1's 1.63 ms Triton row is tracked separately above; this item covers
only the LLGemm1 surface.)
**Gate**: floors confirmed (798 GB/s, TCC hit ~14.5%); micro-bench per shape
before touching the model path.

### P3-3 — paged attention decode (1.94 ms/step, 10 layers × ~194 µs) — #2 CANDIDATE

At seq~500 the KV read per layer is ~0.5 MB with **82% L2 hit** — 194 µs is
latency/occupancy-bound, not BW-bound. P3-0 Q2 confirms the gap is real:
llama.cpp's flash_attn_tile (256×256 KV-chunked) does ~15 µs/layer at avg
seq~128 → ~30–60 µs even scaled to our seq~500, i.e. **3–10× faster**.
Likely cause on our side: poor work-split at M=1 with GQA kv_heads=2 (few
work items for 60 CUs). Study the custom FA kernel's grid config and
llama.cpp's tile split before writing anything; expected saving **~1.5–1.7
ms/step** if a comparable split is reachable.

**Gate**: gap confirmed ≥3× (it is); FA prefill advantage must not regress —
bench both phases.

### P3-4 — elementwise/norm pile (~2.3 ms/step) — DEPRIORITIZED

P3-0 Q5: the pile is Fill/zeros 0.37 + copyBuffer 0.32 + rmsnorm variants
~0.6 + act/sigmoid ~0.4 + misc triton ~0.6. llama.cpp spends ~3.5 ms/step on
its equivalent (plus 1.31 ms Q8_1 quant tax we don't pay) — **we are already
at or better than the reference**, so there is no demonstrated win here.
Keep as a watch item; only revisit if P3-1/P3-3 land and the gap vs
llama.cpp still exceeds 2 ms. (If revisited: inductor has already fused most
of it — the lever would be graph breaks, not pass_config.)

### P3-5 — LM head (~1.2 ms/step, inside LLGemm1) — SKIP

Q6 resolved: the LM head runs through LLMM1 (not aten::mm) at ~1420 µs vs
~1280 µs floor @798 GB/s — **1.1× off floor, no meaningful slack**. Skip.
(llama.cpp's head is Q8_0 = half the bytes; that part of its lead is
quantization, not kernel quality.)

### `aten::mm` (2.2 ms, 11%) — RESOLVED: profile artifact, void

Q6: zero M=1 aten::mm calls in steady-state decode (shape-aware graph
profile + kernel-level reconciliation). The P2-3 row came from that window's
warmup/capture region. No action; recorded so nobody re-chases it.

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
| P3-1 (shared_expert_gate) | ~1.4 ms | ~1.2 ms | ~1.2 ms |
| P3-3 (attention work-split) | ~1.7 ms | ~1.0 ms | ~2.2 ms |
| P3-2 (LLGemm1 mid-size rows) | ~1.5 ms | ~1.0 ms | ~3.2 ms |
| P3-4 (elementwise) | — | 0 (deprioritized) | ~3.2 ms |

**Realistic range: ~2.5–4 ms off 20.3 ms/step** → **~57–68 t/s decode**.
Parity with llama.cpp's 70.3 t/s is NOT the goal and is not reachable on
this budget: its dense weights are Q8_0 (half our fp16 bytes) and it uses
Q8_1 activation quantization — a structural arithmetic advantage, not a
kernel-quality one (MoE is at parity; GDN we already win). Phase success =
**close ≥50% of the 6 ms gap (≥3 ms) with measured per-kernel evidence and
no prefill regression**; failure to reach parity is an acceptable,
documented outcome.

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
