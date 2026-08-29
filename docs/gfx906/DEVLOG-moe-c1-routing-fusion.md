# MoE C1 — M=1 routing pipeline fusion — stage 1 (align+count) SHIPPED

**VERDICT:** SHIPPED (stage 1) · **GATE:** in-process
`_bench_gfx906.py`, Qwen3.5-35B-A3B-AWQ, TP=1 B=1, pp2048/tg256, graph
regime, 4 samples/arm, `VLLM_GFX906_ALIGN_M1` 0 vs 1, same boot (O),
2026-08-29

## HYPOTHESIS

If the ~0.8 ms/step M=1 routing cost is per-kernel graph-node latency
(rather than the raw node floor or kernel work), then replacing the
two-kernel `moe_align_block_size` + `count_and_sort_expert_tokens` pair
with one 1-CTA kernel (120 → 80 nodes/step) saves ~200 µs/step in serving
wall clock, not just in isolated graph replay.

## What was done

- **Structural probe** (`benchmarks/kernels/gfx906/c1_routing_structural_probe.py`):
  the production M=1 routing chain is **3 kernels/layer** (topkGating +
  align 2-block + count_and_sort = 120 graph nodes/step ≈ 772–800 µs,
  M-independent — latency-bound). Node floor (dummy `add_`): 1.3 µs/node.
  Per-node: topk 11.8 µs, align 3.8, sort 3.8.
  S2 re-check in the same regime: the dedicated M=1 topk is 28% faster per
  node in isolated graphs (8.5 vs 11.8 µs) yet **loses −1.03% in serving**
  (56.36 vs 56.95 t/s, 4 samples/arm) — reproducing the sprint's
  −0.95 t/s graph-regime loss. Isolated-graph kernel numbers do not predict
  production-graph cost; they can flip sign. (S2 was a kernel *swap*;
  node count unchanged.)
- **New kernel** `csrc/rocm/moe_align_m1_gfx906.cu`: one 128-thread CTA,
  LDS-atomic counts + 5-step `shfl_down` warp scan over the 256 expert
  counts (exclusive prefix = total − suffix, total ≡ 8 slots), lane-atomic
  placement. No grid work, no D2H, no global atomics → capture-safe.
  Bit-equal to the two-kernel chain (26/26 tests, tie-heavy included;
  within-expert slot order matches the production single-warp atomic order).
- **Call site** `gfx906_w4a16_moe.py`: gated branch (M=1, topk=8, E=256,
  block_size=1, int32, no expert_map), flag `VLLM_GFX906_ALIGN_M1`,
  **default ON** after the gate PASS; `=0` to opt out.

## Evidence FOR

- Isolated graph (launch-regime evidence): fused align **2.0 µs/node vs
  3.8** for the pair (79.9 vs 300.7 µs / 40 layers) → ~224 µs/step
  predicted.
- **GATE — serving A/B (boot O):**

  | run | off | on | Δ |
  |---|---|---|---|
  | 1 | 56.53 t/s | 57.51 t/s | +1.73% |
  | 2 (back-to-back) | 56.66 t/s | 57.33 t/s | +1.18% (207 µs/step) |

  Back-to-back step: 17.649 → 17.442 ms — 207 µs/step, within 8% of the
  isolated prediction; both sessions consistent; within-arm spread ≤0.7%.
- Narrowed suite 524 passed (gfx906 MoE gemm, align-m1, generic align,
  topk) both with flag off and with the default flipped on.

## Evidence AGAINST

- None in serving. The S2 flip is a warning about *kernel-swap*
  mechanisms, not a counterexample to node removal — the back-to-back gate
  shows node removal transfers (207 vs 224 µs/step predicted).

## Notes / next (stage 2, OPEN)

- topk is the remaining 470 µs/step component (11.8 µs/node × 40). Stage 2
  = one fused topk+align+count kernel (120 → 40 nodes): needs the router
  logits (and the topk-skip decision) at the expert `apply` — an optional
  kwarg through `forward_modular` → quant method → kernel → impl, plus a
  shared eligibility predicate at the router (S2's bit-exact topk phase is
  the starting point). Only justified because stage 1 proved the mechanism
  transfers; the S2 flip means stage 2 must be a *node removal*, not a
  topk kernel swap.
- Logs: `/local/tmp/c1_structural_run{1,2,3,4}.log`,
  `/local/tmp/c1_ab_default.log`, `/local/tmp/c1_ab_topkm1.log`,
  `/local/tmp/c1s1_ab_arm{A,B,A2,B2}.log` (boot O; /tmp wiped on reboot).
