# Phase 3 — non-MoE decode path on gfx906 (MI60)

Status: DRAFT v2 — post adversarial review (evidence from graph-mode +
eager attribution profiles, 2026-08-15). Scope: close the remaining decode
gap vs llama.cpp. Prefill is already 2.6× faster than llama.cpp — out of
scope unless a candidate helps both for free.

Review changes (v1→v2): added P3-0 Q0 (layer-composition reconciliation —
profiled counts don't match the assumed 30/10 GDN/FA split); softened P3-1
expectation to a 2–3× honest range; reframed P3-4 (inductor already fuses;
real lever is graph breaks, expected value 0–1 ms); made P3-3's
latency-bound nature explicit; added §6 success criteria incl. the
quantization-asymmetry caveat.

## 1. Where we are

Same model family (Qwen3.5/3.6-35B-A3B, hybrid GDN + full-attn, 256-expert
top-8 MoE), MI60 32 GB, single request:

| engine | prefill pp=2048 | decode |
|--------|-----------------|--------|
| llama.cpp (Q4_K_XL) | 807 t/s | **70.3 t/s** (14.2 ms/step) |
| vLLM + cudagraphs (AWQ int4) | ~2140 t/s | ~49 t/s (20.3 ms/step) |

Gap: **~6 ms/step (≈30% of the step)**, entirely in GPU kernel time
(cudagraphs remove the launch overhead; eager is not the target anymore).
Primary metric from now on: **serving mode** (`BENCH_EAGER=0`,
`FULL_DECODE_ONLY`), decode tok/s and ms/step.

## 2. Per-step decode budget (measured, graph mode, M=1)

From shape-aware torch profiles + chrome-trace correlation (eager trace for
attribution; graph-mode kernel times):

| component | ms/step | % | status |
|-----------|---------|---|--------|
| dense projections, aiter LLGemm1 (`LLMM1` op) | ~6.8 | 33% | **P3 target** |
| `vllm::rocm_unquantized_gemm` (Triton fallback) [2048→2048] | ~1.6 | 8% | **P3 target** |
| paged attention (custom FA), 10 layers × ~198 µs | ~1.9 | 9% | **P3 target** |
| gfx906 MoE routed kernel (Phase 1/2) | ~1.75 | 8% | done |
| GDN recurrence + conv + chunk ops | ~1.2 | 6% | watch |
| routing pipeline (topk+align+sort) | ~1.0 | 5% | P2-4 deferred |
| elementwise/norm/copy/zeros pile | ~2.0 | 10% | **P3 target** |

Dense-projection breakdown (M=1, weight-read-bound; bytes = N·K·2 fp16):

| projection (N,K) | layers | µs/call | floor @BW | ratio |
|------------------|--------|---------|-----------|-------|
| GDN in_proj (12288, 2048) | ~30 | ~80 | ~50 | 1.6× |
| LM head (248320, 2048) | 1 | ~1420 | ~1000 | 1.4× |
| GDN out_proj (2048, 4096) | ~30 | ~41 | ~17 | **2.4×** |
| FA qkv (9216, 2048) | ~10 | ~64 | ~38 | 1.7× |
| shared gate_up (1024, 2048) | ~40 | ~10 | ~4 | **2.5×** |
| shared down (2048, 512) | ~40 | ~9 | ~2 | 4.5× |
| router (256, 2048) | ~40 | ~6 | ~1 | — |
| GDN small proj (64, 2048) | ~30 | ~4 | <1 | — |
| Triton fallback (2048, 2048) | ~37? | ~44 | ~8 | **5.5×** |

(Floor assumes ~1 TB/s HBM; **P3-0 measures the real achievable BW first** —
if it is lower, every floor scales up and the ratios shrink.)

Total dense-projection time ≈ 8.4 ms/step; bandwidth floor ≈ 4.5 ms →
realistic capture of half the difference ≈ **~2 ms/step**, plus attention and
elementwise work below. A full close to llama.cpp's 14.2 ms/step is NOT the
goal — llama.cpp runs Q4 weights (half the bytes of our fp16 dense layers), so
part of its lead is quantization, not kernel quality.

## 3. Open questions P3-0 must answer (diagnostics, no code changes)

0. **Exact layer composition**: the profiled per-step call counts (~37 for
   GDN-shaped projections) do not match the assumed 30 GDN + 10 FA split.
   Read the model config / HF json (full_attention_interval) and reconcile
   against the trace before any per-layer math is trusted. All §2 ratios are
   provisional until this is settled.
1. **Achievable HBM BW on this MI60** (simple sum/copy microkernel, fp16):
   sets every floor in §2. Also record sclk/mclk under load.
2. **llama.cpp per-kernel decode table**: `rocprofv3 --hip-trace` around
   `llama-bench -p 0 -n 256` (Q4 model) → aggregate kernel times/step. This
   is the reference design: which GEMM kernel (mmq?), attention, norms does
   it use, and what do they cost? Direct comparison against our table.
3. **Which vLLM layers emit `rocm_unquantized_gemm`** (~37 calls/step):
   python probe (inspect the model's Linear layers / backend selection for
   this arch on ROCm). Hypothesis: FA o_proj and/or GDN out-proj variants
   that aiter's LLMM1 path declines.
4. **Shared-expert path**: confirm it is two plain fp16 Linears (gate_up +
   down) via LLGemm1, and what the residual `fused_moe_kernel` (Triton,
   ~2/step) actually computes.
5. **Elementwise pile identity** (~2 ms/step): which of the many small
   triton/aten elementwise kernels dominate; whether inductor pass_config
   fusions (`fuse_norm_quant`, `fuse_act_quant`, … — all currently False)
   are applicable to this model without changing numerics.

## 4. Ordered candidates (each: test → bench → commit, per common protocol)

Ordering = measured ms/step × feasibility. Every step is gated on P3-0.

### P3-1 — Triton `rocm_unquantized_gemm` fallback (~1.6 ms/step, 5.5× off floor)
Cheapest possible win if it is a backend-selection bug: find why these
[2048→2048] M=1 gemms bypass aiter (P3-0 Q3) and route them to the same
LLGemm1 path as their siblings. Expectation (honest range): 1.6 →
~0.5–0.9 ms/step (2–3×; the 5.5× floor assumes perfect BW utilization that
no M=1 kernel reaches). If P3-0 Q3 shows the call count is actually ~10/step
(e.g. only FA o_proj), this item shrinks to ~0.4 ms and may merge into P3-2.
**Gate:** P3-0 Q3 identifies the layers; change is config/backend-selection
only, no new kernel. Reject if the fallback is intentional (shape/dtype
constraint) — then fold into P3-2.

### P3-2 — M=1 dense GEMM efficiency (GDN out_proj 2.4×, shared gemms 2.5–4.5×)
The aiter LLGemm1 kernel is ~2–2.5× off the bandwidth floor on mid-size M=1
gemms. Options in increasing invasiveness:
(a) try other aiter/rocBLAS dispatch (env/backend flags, splitK for N≤2048);
(b) check llama.cpp's mmq-style single-pass kernel as reference and port a
minimal M=1 W16A16 "gather-free" kernel only if (a) fails.
Target: dense projections 8.4 → ~5.5 ms/step. **Gate:** P3-0 BW number makes
the floor real; micro-bench per shape before touching the model path.

### P3-3 — paged attention decode (~1.9 ms/step, 10 layers × 198 µs)
Note: at seq~500 the KV read per layer is ~0.5 MB — bandwidth floor is
sub-microsecond, so 198 µs is **latency/occupancy-bound**, not BW-bound.
Likely cause: poor parallelism at M=1 (GQA kv_heads=2). Compare against
llama.cpp's attention kernel from the P3-0 trace; if it is materially faster
at batch=1, study its work-split before writing anything. **Gate:** P3-0
shows a gap ≥2×; otherwise defer (our FA prefill advantage must not regress —
bench both phases).

### P3-4 — elementwise/norm pile (~1–2 ms/step)
Caveat: inductor has ALREADY fused much of this (profile names are
`triton_*_fused_*`) — the remaining small kernels are mostly OUTSIDE
compiled regions (custom-op boundaries). So the lever is probably reducing
graph breaks / moving work into compiled regions, not just `pass_config`
flags. P3-0 Q5 must say which pile items live where; then enable the
applicable fusions and measure. Numerics-sensitive: correctness test +
sanity generation required; any output diff beyond fp rounding → revert.
Expected value uncertain (0–1 ms/step).

### P3-5 — LM head (1.4 ms/step)
1 GB weight read per step = ~1 ms floor; only ~0.4 ms of slack. Options are
ugly (fp8/fp4 lm_head changes model numerics; speculative head skipping is an
engine feature). **Default: SKIP** unless P3-0 shows the kernel is >2× off
floor or llama.cpp does something structurally different.

### Explicitly out of scope
- Prefill GEMMs (we are 2.6× ahead of llama.cpp there).
- MoE routed kernel (Phase 1/2 done; 8% of step, at its issue-bound ceiling).
- P2-4 routing pipeline (~1 ms) — revisit only if P3-1..P4 land and we still
  trail llama.cpp by >2 ms.
- Multi-batch serving behavior (this bench is single-request; batched decode
  changes the whole budget — separate project).

## 5. Common protocol (every step)

1. Correctness: existing `tests/kernels/moe/test_gfx906_moe_gemm.py` stays
   green for any MoE-adjacent change; model-level sanity generation (fixed
   prompt, greedy) must match the pre-change output exactly for config-only
   changes, or stay within fp tolerance + coherent text otherwise.
2. Serving-mode bench: `BENCH_EAGER=0` full run (pp=2048/tg=256) — record
   total tok/s and derived decode ms/step. Also the eager run if a change
   could affect prefill.
3. Micro-bench any new/changed kernel per shape before model integration.
4. Separate commit + dev-log entry (positive AND negative results).

## 6. Expected outcome (success criteria)

Realistic capture: P3-1 ~0.8–1.2 + P3-2 ~1.5–2 + P3-3 0–1 + P3-4 0–1 =
**~2.5–5 ms/step off the current 20.3 ms** → ~58–70 t/s decode. Parity with
llama.cpp's 70.3 t/s is NOT guaranteed: a meaningful part of its lead is Q4
vs fp16 dense weights (half the bytes), which no kernel work can recover.
Phase success = close ≥50% of the gap with measured per-kernel evidence and
no prefill regression; failure to reach parity is an acceptable, documented
outcome.

## 7. Risks

- **Quantization asymmetry vs llama.cpp**: its Q4 dense weights are ~half our
  fp16 bytes; part of the 30% gap is irreducible without re-quantizing the
  model. The plan's targets (§2 floors) already account for this — do not
  chase llama.cpp's absolute number.
- **Hybrid GDN model**: state-update kernels (GDN recurrence, conv1d) are
  Triton and were tuned by upstream for other archs; touching them is high
  risk/low reward here (watch list only).
- **Numerics**: any fusion or backend switch changes fp reduction order;
  greedy-output diffing is the tripwire.
- **Scope creep**: this phase is decode-only, single-request, gfx906. If a
  candidate turns out to need upstream aiter/FA work, stop and re-plan.
- **Provisional numbers**: §2 layer counts/ratios rest on one profiled
  window with ~±20% count uncertainty (P3-0 Q0 must reconcile before any
  step is sized from them).
