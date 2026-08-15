# Phase 3 — non-MoE decode path on gfx906 (MI50)
Copyright Kevin Read <me@kevin-read.com>


Status: v9 — **P3-1 landed** (serving 41.51 → 44.09 t/s); **P3-2(b) kernel
built + micro-benched (−401 µs/step, −7.2% vs LLMM1 rpb=4) + integrated**
(clean A/B pending under the new best config); (a) probe closed. **New
best serving: 52.07 t/s** (CUSTOM Q8 FA + requested-PIECEWISE + GEMV) —
beats the 44.09 Triton-FULL record; the requested-FULL path is broken in
this tree (CGSupport.NEVER downgrade → 22.44 t/s, engine bug). **P3-3a
rescoped** (DEVLOG "Serving-mode backend findings"): fix the downgrade
path, verify CUSTOM serving correctness (COW/multi-batch), keep the
Triton-baseline record for the archive. See §4 ordering and
`plan-gfx906fa-serving.md` (v2; M0 1/3 done). Scope: close the remaining
decode gap vs llama.cpp (~1.35× at 52.07 vs ~70). Prefill is already 2.6×
faster than llama.cpp — out of scope unless a candidate helps both for free.

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

Review changes (v8→v9):

- **New best serving 52.07 t/s** (DEVLOG "Serving-mode backend findings",
  2026-08-15): the fork's gfx906 FA plugin now wins backend selection by
  default (and `VLLM_ATTENTION_BACKEND` was dropped upstream — the knob is
  `attention_config["backend"]`), so the current default request
  (FULL_DECODE_ONLY) downgrades via CGSupport.NEVER and serves at 22.44
  t/s; requesting **PIECEWISE** compiles the model piecewise and gets
  **52.07 t/s** with CUSTOM FA + GEMV. llama.cpp gap: ~1.6× → ~1.35×.
- **P3-3a rescoped** around the finding: (i) fix/guard the
  FULL-request→PIECEWISE-downgrade path (decode degrades toward eager —
  engine bug, candidate: piecewise-compile when the downgrade fires);
  (ii) verify CUSTOM serving correctness under prefix-cache COW and
  multi-batch (probe running); (iii) M1 side-buffer lifecycle items stand.
  The 44.09 Triton-FULL number is archived as the pre-FA record.
- **P3-2(b) A/B**: the first (misconfigured) sequence showed +0.7% under
  the broken config (CPU-launch-bound, GPU win hidden); the clean GEMV
  on/off A/B runs under the 52.07 config.

Review changes (v7→v8):

- **P3-2(b) micro-bench results in** (DEVLOG "P3-2(b)", 2026-08-15):
  v1's K-split (fp16-CAS) hypothesis for the small rows is **falsified** —
  splits are 2.4–4.2× slower than LLMM1 at M=1; the "3.6–14× floor" rows
  are launch/latency-bound, not CU-occupancy-bound, and no GEMM kernel
  closes them (that is kernel-count-reduction territory). v2 (single-pass
  + RPT=2 rows/thread) wins K=2048 rows with N==256 or N≥2048: qkv −23%,
  router −17%, in_proj/LM head −6%; best-per-shape weighted total
  **5203 vs 5604 µs/step = −401 µs/step (−7.2%)** — under the 0.7–1.1 ms
  v7 estimate (the latency-bound reframing caps it).
- **P3-2(b) integrated** at the `_llmm1_tiny_m` choke point with the
  measured shape rule (K==2048 ∧ fp16 ∧ N==256∨N≥2048; RPT=2 via C++
  default), `VLLM_GFX906_DENSE_GEMV=0` kill switch; serving A/B
  (FULL_DECODE_ONLY, mode-matched) running.
- **Ordering**: A/B resolves before P3-3a M0 runs (single GPU, sequential);
  M0 (Triton-PIECEWISE baseline + attention slice) is next in the queue
  either way.

Review changes (v6→v7):

- **Day-1 outcomes recorded** (DEVLOG "Day-1 pre-step pair", 2026-08-15):
  gather micro-bench **21.7 µs/layer @ Sk=2816, B=1** (2.0× HBM floor,
  407 GB/s; Q-fp32 side costs 21.9 µs/layer) → **gate passed, P3-3a
  RESUMED** per the §4 decision tree. Sub-plan M0 item 1 done; remaining
  M0: Triton-PIECEWISE baseline + attention slice. Sub-plan M0 spec's
  "Hkv=8" is wrong for this model (`num_key_value_heads=2`); the bench
  used the real value — at Hkv=8 the gather would be ~4×.
- **P3-2(a) probe STOPPED (structurally a no-op on gfx906)**: aiter triton
  gemm + wvSplitKrc sit inside `if not on_gfx906():`; wvSplitK excluded
  (no matrix cores on Vega20); `VLLM_ROCM_USE_AITER` defaults False;
  aiter triton-gemm whitelist matches no model shape. Rejection reason =
  arch exclusion, not shape/dtype — P3-1's "collapse into P3-2(b)" path
  confirmed. The only knob on the real surface is LLGemm1
  `rows_per_block` (dispatch hardcodes 4): sweep gives +1.4% best
  (rpb=8) weighted per-step — below the ≥20% continue bar; no config
  change warranted.
- **P3-2(b) rescoped from the rpb sweep**: big rows (in_proj/LM head/qkv/
  o_proj) are BW-bound at rpb=4 (0.95–1.29× floor); small rows
  (gate_up/down/router/GDN-small, 150 calls/step) are launch/latency-bound
  (3.6–14× floor; LLGemm1 at M=64 launches 16 blocks, 60 CUs under-filled)
  → the custom kernel must K-split. Realistic capture ≈ 0.7–1.1 ms/step
  (slightly under the v6 1–1.5 ms estimate).

Review changes (v5→v6):

- **Ordering resequenced**: P3-2 promoted to primary dev bet; P3-3a suspended.
  Rationale: eager near-parity (19.33 vs 19.49 t/s) indicates gather+dtype
  tax likely eats most of the 72 µs kernel win at B=1; P3-2's ceiling
  (~1–1.5 ms) is comparable with far lower integration risk. A Day-1
  gather micro-bench (at Sk~2816 B=1) is the explicit go/no-go gate to
  resume P3-3a; it runs alongside the P3-2(a) aiter probe. See §4 ordering
  block for full decision tree.
- **P3-3a measurement fix added**: if resumed, requires a Triton PIECEWISE
  baseline bench before reporting M1 numbers, to deconfound kernel win from
  graph-boundary overhead (CUSTOM runs PIECEWISE, baseline runs
  FULL_DECODE_ONLY).

Review changes (v4→v5):

- **P3-1 recorded DONE**: `_llmm1_tiny_m()` pad-to-4 fix in the dispatch
  layer; serving 41.51 → **44.09 t/s** (+6.2%, ≈1.4 ms/step), eager 18.88 →
  19.49. §2 row updated (residual ~0.29 ms).
- **P3-3 reframed**: the original "partition the Triton kernel" plan was
  overtaken by the discovery that the tree vendors a Q8 FlashAttention
  backend that was dead code (unregistered) + carried a real stride bug —
  both fixed (`7e9e855bab`). Kernel measured **72 µs/layer vs Triton
  194 µs**; eager parity only (launch-bound); serving blocked by side-buffer
  lifecycle (crash), COW prefix copies (correctness), CGSupport.NEVER
  (mode downgrade). Split into **P3-3a** (make CUSTOM serving-viable — new
  sub-plan `plan-gfx906fa-serving.md`) and **P3-3b** (Triton partitioning,
  fallback).
- **P3-2 promoted** to top unblocked candidate while P3-3a is in flight.
- **§1/§6 rebased on measured serving numbers**: 44.09 t/s = 22.7 ms e2e
  step; the 20.3 ms P3-0 figure is the profiled step, not e2e (~2.4 ms of
  host/scheduler/sampling sits outside it).
- **§7 additions**: cudagraph-mode confound (backend choice changes graph
  coverage, not just the kernel); prefix-COW correctness risk for any
  K-quant side buffer.

---

## 1. Where we are

Same model family (Qwen3.5/3.6-35B-A3B, hybrid GDN + full-attn, 256-expert
top-8 MoE), MI50 32 GB (gfx906, 60 CU), single request:

| engine | prefill pp=2048 | decode |
|--------|-----------------|--------|
| llama.cpp (Q4_K_XL) | 807 t/s | **70.3 t/s** (14.2 ms/step) |
| vLLM + cudagraphs, Triton attn, post-P3-1 | ~2140 t/s | **44.09 t/s** (22.7 ms e2e step; ~19.0 ms profiled step) |

Gap: **1.59× (~8.5 ms e2e/step)**. ~2.4 ms of the e2e step is outside the
profiled kernel window (host/scheduler/sampling) — kernel-side targets can
only attack the remaining ~6 ms. Primary metric: **serving mode**
(`BENCH_EAGER=0`), decode tok/s and ms/step. Note: eager best is 19.49 t/s.

---

## 2. Per-step decode budget (RECONCILED, graph mode, M=1)

Kernel-level sums over 31 steady-state graph steps (prefill excluded by
construction; DEVLOG P3-0 for method):

| component | ms/step | calls/step | status |
|-----------|---------|------------|--------|
| LLGemm1 dense projections (aiter, incl. shared expert 80 + LM head) | **5.83** | 230 | **P3-2 target** |
| `triton_matmul` = `shared_expert_gate` [1×2048] ×40 layers | 1.63 → **0.29** | 40 | **P3-1 DONE** (padded LLMM1, 40 × 7.3 µs) |
| paged attention, 10 layers × ~194 µs (Triton) | **1.94** | 10 | **P3-3a target** — in-tree CUSTOM kernel is 72 µs/layer; serving integration pending (`plan-gfx906fa-serving.md`); fallback P3-3b |
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
P3-0 is complete and the sizes below are final for this phase. Live order
(v7): **P3-2(b) is the primary development bet**; **P3-3a resumed** (Day-1
gate passed) as the time-fenced parallel line.

**Day-1 pre-step — DONE (2026-08-15, DEVLOG "Day-1 pre-step pair"):**
- **Gather micro-bench**
  (`benchmarks/kernels/gfx906/bench_gfx906_fa_gather.py`):
  `gather_paged_kv_q8` at B=1, Sk~2816 (Hkv=2, D=256, bs=16, pre-allocated
  out buffers, byte-identical to torch reference): **21.7 µs/layer** (2.0×
  HBM floor, 407 GB/s; 18.6 @ Sk=2048, 25.3 @ Sk=3328) + Q-fp32 side costs
  21.9 µs/layer (q.float 3.9 / q_pad zero 2.7 / q copy 7.0 / out unpack
  8.3) → combined tax 43.6 µs vs 122 µs FA kernel win → **net ~0.78
  ms/step** over 10 layers. **Gate passed (≤ ~80 µs) → P3-3a resumes.**
- **P3-2(a) aiter splitK probe** (audit +
  `bench_llmm1_rows_per_block.py`): **stopped — structurally a no-op on
  gfx906.** aiter triton gemm/wvSplitKrc gated `not on_gfx906()`; wvSplitK
  needs matrix cores (absent on Vega20); `VLLM_ROCM_USE_AITER` default
  False; aiter triton-gemm whitelist matches no model shape. The only knob
  on the real surface — LLGemm1 `rows_per_block` (dispatch hardcodes 4) —
  sweeps to +1.4% best weighted per-step (rpb=8); rpb=2 is net-negative
  (6× regression on shared gate_up, M=1024). Below the ≥20% continue bar.

**After Day-1 (resolved branch: gather ≤ ~80 µs):**
- **P3-3a resumes** per `plan-gfx906fa-serving.md`: M0 remainder (Triton-
  PIECEWISE baseline + attention slice) then M1 (side-buffer lifecycle +
  COW Q8 mirror + partial-block COW probe; gather-buffer hysteresis is a
  separate A/B, not part of M1). Runs as a **time-fenced parallel line
  alongside P3-2**, not a replacement — it suspends again if M1 nets
  < +0.3 ms/step vs the Triton-PIECEWISE baseline (mode-matched).
- **P3-2(b) custom W16A16** — DONE at micro-bench level (−401 µs/step,
  −7.2%; K-split falsified, single-pass RPT=2 wins; see (b) below),
  integrated, serving A/B in flight.

### P3-1 — `shared_expert_gate` [1×2048] scalar gemv — DONE (2026-08-15)

**Landed** (`3e7c4f2252`, devlog `8b2c5ccc05`): `_llmm1_tiny_m()` in
`vllm/model_executor/layers/utils.py` zero-pads the weight to 4 rows →
`ops.LLMM1(w, x, 4)` → slice; both dispatch sites accept `(m % 4 == 0 or
m < 4)`. Micro-bench: Triton 42.8 µs vs LLMM1pad4 7.3 µs (torch linear
281 µs rejected; rocBLAS skinny gemv is terrible here). **Serving 41.51 →
44.09 t/s (+6.2%, ≈1.4 ms/step); eager 18.88 → 19.49.** Greedy A/B: 2/3
prompts identical, one diverges ~token 11 (fp16 reorder on the sigmoid
gate; both fluent — accepted). Residual §2 row now ~0.29 ms.

### P3-2 — LLGemm1 dense surface (5.83 ms/step, 230 calls) — PRIMARY DEV BET (v7)

Floor across the projection table ≈ 4.6 ms @798 GB/s; the two big rows
(in_proj ~1.3×, LM head ~1.1×) are already near floor — realistic capture is
**~1–1.5 ms/step**, concentrated in the mid-size rows (out_proj/qkv/shared).
Options in priority order:

**(a) Quick aiter probe — STOPPED (Day-1, 2026-08-15)**: aiter is
structurally excluded on gfx906 (all aiter gemm paths gated
`not on_gfx906()`; wvSplitK needs matrix cores; `VLLM_ROCM_USE_AITER`
default False; whitelist has no matching shape) — rejection reason: arch
exclusion. The one real knob, LLGemm1 `rows_per_block` (dispatch hardcodes
4), sweeps to +1.4% best weighted per-step (rpb=8; rpb=2 net-negative) —
below the ≥20% continue bar; no config change warranted. See DEVLOG
"Day-1 pre-step pair".

**(b) Custom M=1 W16A16 dense kernel — BUILT + MICRO-BENCHED + INTEGRATED
(A/B in flight)**:
`csrc/rocm/dense_gemv_gfx906.cu`, op `dense_gemv_gfx906`. Row-parallel
(`__ockl_fdot2`, fp32 acc), templates RPT∈{1,2,4} × KCHUNK∈{512,2048,4096};
K-split via packed fp16-CAS into pre-zeroed out. Micro-bench
(`benchmarks/kernels/gfx906/bench_dense_gemv_gfx906.py`), best per shape:
qkv 9216 −23%, router 256 −17%, in_proj 12288 −6%, LM head −6% (0.9× floor);
o_proj (K=4096), shared gate_up (1024), shared down (K=512), GDN small
(64) stay LLMM1 (RPT=2 is pathologically bad at N=1024). **Weighted step
total 5203 vs 5604 µs = −401 µs/step (−7.2%)** — under the 0.7–1.1 ms
estimate: v1 falsified K-split (2.4–4.2× slower at M=1) and the small rows
are launch/latency-bound, not occupancy-bound, so the estimate's top half
was unreachable. Integrated at `_llmm1_tiny_m` (both n==1 sites); rule
K==2048 ∧ fp16 ∧ (N==256∨N≥2048); kill switch `VLLM_GFX906_DENSE_GEMV=0`;
C++ default RPT = measured rule. **Gate** (micro-bench per shape before
touching the model path): passed.

(P3-1's 1.63 ms Triton row is tracked separately above; this item covers
only the LLGemm1 surface.)
**Gate**: floors confirmed (798 GB/s, TCC hit ~14.5%); micro-bench per shape
before touching the model path.

### P3-3 — paged attention decode (1.94 ms/step) — RESCOPED (v9: 52.07 t/s new best; downgrade bug is the target)

The original plan (partition the Triton kernel over KV) was overtaken by
events: the tree already vendors a Q8 FlashAttention backend (llama.cpp
`flash_attn_tile_q8` port, head_size 256 OK) that was **dead code at
runtime** (plugin entry point missing from stale egg-info) and carried a
**real stride bug** (V-cache is a non-contiguous `unbind(1)` view; kernels
derived strides from shapes → read K bytes as V). Both fixed in
`7e9e855bab`; regression tests in
`tests/kernels/attention/test_gfx906_fa.py` build the cache exactly like
the backend so this bug class can't hide again.

Measured state: **FA kernel 72 µs/layer vs Triton 194 µs (2.7×)**; eager
parity only (19.33 vs 19.49 — B=1 eager is launch-bound; the gather+q-fp32
tax eats the kernel win). Serving is blocked by three identified issues:
Q8 side-buffer lifecycle vs profile→real cache realloc (crash: `value_cache
blocks mismatch`), COW prefix-cache copies bypassing the side buffer
(correctness), and `CGSupport.NEVER` downgrading the engine to PIECEWISE
while the Triton baseline serves with FULL_DECODE_ONLY.

- **P3-3a — make CUSTOM serving-viable** (sub-plan:
  `plan-gfx906fa-serving.md`; rescoped v9). Requested-PIECEWISE serving
  already works (52.07 t/s, beating the old 46–48 target); remaining:
  (i) the requested-FULL→downgrade bug (decode degrades toward eager at
  22.44 t/s — piecewise-compile when the downgrade fires, or raise the
  backend's CGSupport if FULL capture is actually safe); (ii) correctness
  under prefix-cache COW + multi-batch (probe); (iii) M1 side-buffer
  lifecycle items (realloc-on-shape-change, COW Q8 mirror, gather-buffer
  hysteresis) as needed by (ii).
- **P3-3b — Triton KV partitioning (fallback)**: original design (grid axis
  3 over KV splits + merge kernel, gated on_gfx906 / sinks-None) stays
  parked until the P3-3a decision.

**Gate**: FA prefill advantage must not regress — bench both phases.

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

Per-candidate targets against the measured 22.7 ms e2e step (44.09 t/s
baseline; ~19.0 ms of it is profiled kernel time):

| candidate | saving (realistic) | cumulative |
|-----------|--------------------|------------|
| P3-1 (LANDED, measured) | **+1.4 ms** | 22.7 ms / **44.09 t/s** ✅ |
| P3-3a CUSTOM serving (M1→M2) | 0.7–1.2 ms | ~21.5–22.0 / ~46–47 |
| P3-2 LLGemm1 mid-size rows | ~1.0 ms | ~20.5–21.0 / ~48–49 |
| P3-3b Triton partitioning | 0.7–1.0 ms | alternative to P3-3a |
| P3-4 elementwise | 0 (deprioritized) | — |

**Realistic remaining: ~1.7–2.2 ms → ~48–49 t/s.**
Parity with llama.cpp's 70.3 t/s is NOT the goal and is not reachable on
this budget: its dense weights are Q8_0 (half our fp16 bytes) and it uses
Q8_1 activation quantization — a structural arithmetic advantage, not a
kernel-quality one (MoE is at parity; GDN we already win). Phase success =
**close ≥50% of the remaining kernel-side gap with measured per-kernel
evidence and no prefill regression**; failure to reach parity is an
acceptable, documented outcome.

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
- **Cudagraph-mode confound**: an attention backend's CGSupport decides
  PIECEWISE vs FULL_DECODE_ONLY; swapping backends changes graph coverage,
  not just the kernel. Always record the resolved `cudagraph_mode` next to
  serving numbers (v4's "~49 t/s" row silently assumed FULL mode).
- **Prefix-COW correctness**: any K-quant side buffer must mirror COW block
  copies (`copy_kv_cache_blocks_inplace`) — a bug class that only fires on
  prefix hits sharing a partially-filled block, invisible to fresh-run
  benches (see `plan-gfx906fa-serving.md` RC2).
