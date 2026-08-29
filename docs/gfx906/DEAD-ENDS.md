# gfx906 dead ends & open verdicts — index

The one-pass answer to "did we try X and did it work?" Every row is backed
by a source log (or, if not yet archived to the new format, the legacy
DEVLOG). Pattern: **hypothesis → gate → verdict → commit/revert → comment**.

- `VERDICT: DEAD-END` / `NEUTRAL` = tried, doesn't help (kept for the
  negative evidence).
- `VERDICT: SHIPPED` = landed, helps (indexed here only when the *cost*
  or *reverse* matters — the positive record lives in its topic log).
- Rows look like: `y/N hypthesis → gate → verdict → commit → refs` so one
  line per line is grep-able and sortable.

## DEAD-ENDS / REJECTED

| Hyp | Gate | Verdict | Commit/revert | Comment | Refs |
|-----|------|---------|---------------|---------|------|
| gemm1 V1 full-K single-wave direct store | standalone harness | **REJECTED** (2.2× slower) | — | 64 blocks can't keep 128 KB streams in flight; kills atomics but loses all transfer | G1 |
| gemm1 NPT=2 (z-split best point) | serving graph, +0.08 t/s | **DEAD-END** (neutral) | `reverted`, flag `VLLM_GFX906_MOE_G1NPT2` removed | census said −186 µs/step; wall-clock neutral in *both* eager & graph — 3rd consecutive transfer failure | G1 |
| gemm1 V3 fp32-scratch K-split | — (closed w/o build) | **DEAD-END** | — | adds ~184 µs/step scratch+launch on top of a design whose best point doesn't transfer | G1 |
| gemm1 V4 halve CAS fan-in (NPT=8/z=32) | launch sweep | **DEAD-END** | — | CAS fan-in axis is monotone-worse as count rises (27.8→33.4 z=16→32) | G1 |
| S5 gemm2 V2 M=1 re-tile (lane-based cols) | standalone 1.18× | **NEUTRAL** in model | shipped default-OFF (`VLLM_GFX906_MOE_M2V2`) | standalone win, in-model transfer failed | M1 |
| S2 dedicated M=1 topk kernel | in-model | **NEUTRAL** | shipped default-OFF (flag) | mode-dependent; standalone win, graph-replay loss | M1 |
| P2-0b zero-fill launch elimination | design | **DESCOPED** (racy) | — | design is racy, not a kernel dead-end | M0 |
| fused fp16·fp32 casts / h2d micro-copies | — | attributed as upstream (non-blocking) | — | — | M0 |
| dense W4A16 purpose-built GEMV | serving | **REJECTED** | prototype in /tmp (not in-tree) | exllama gptq_gemm beats it 187–230% floor vs its 87–97%; no benefit | D |
| dense K=5120 GEMV / lm_head GEMV | serving | **NEUTRAL** | — | LLMM1 already at HBM floor (3114 vs 3128 µs); the "lm_head GEMV lever" does not exist | D |
| P3-2(b) K-split GEMV hypothesis (kc=512 splits) | micro-bench | **REJECTED** (wrong) | — | kc=512 2.4–4.2× slower than LLMM1; M=1 is latency/launch-bound, not occupancy | FA/D |
| GEMV V2 (RPT=2+kc=4096) for skinny dense | micro-bench | **NEUTRAL** | kept per-shape | qkv/router win, N=1024 pathological (−533%), based on shape rule | D |
| llama.cpp Q5_K_XL baseline | — | **IMPOSSIBLE** (36 GiB > 32 GB) | — | won't fit; used Q4_K_XL instead (70.3 t/s ref) | M0 |
| FA V2 fused gather (416 WG, barriers) in serving | serving graph | **REJECTED** (V1 wins) | `GFX906_FA_GATHER_V=2` | V2 degrades 7× in serving (285 µs) vs V1 42 — wave-scheduling/low-WG effect | FA |
| salvage LEGACY=0 via a better dot instruction (dp4a/dot2/dot8 swap in the KQ loop) | ISA rate probe + roofline (analytic) | **DEAD-END** (analysis) | — | dp4a/`v_dot4_i32_i8` already the inner loop and measured FULL-RATE (4.44× fp32, 2× packed fp16); B=1 decode is gather-HBM-bound ~2.7× so ALU swaps can't surface; expansion composites 0.17–0.24×. Q4/dot8 = format change, roadmap M6(c) | FA |
| M6 Part A planar Q8 quants/scale repack closes the LEGACY=0 B=1 gap (flip re-open) | microbench hard stop-rule (ISA loader loads ≥2× AND step ≥2 %) | **DEAD-END** (flip question); code **NEUTRAL** — merged as loader hygiene 2026-08-29 (`02d197189f`), flip-question verdict unchanged | same-boot B=1 A/B PASS: slope 36.0→34.4 ns/token (−4.3/−4.8 %), @Sk=2176 83.6→79.1 us, bit-identical (maxerr equal at every Sk), merged suite 74/74 — **contended-boot caveat (post-merge review): same merged .so = 42.0 us @Sk=2048 / 12.86 ns/tok idle; delta directionally supported, merge not perf-dependent** | ISA-verified loader 10→6 loads/tile-row = 1.67× < 2× (plan's ~17/block assumption was wrong — compiler already 4×8-B + 2-B); standalone B=1 step −2.4 % (disjoint bands) both LEGACY modes; bit-identical (64/64). B=1 gap is elsewhere (write path / Q-side / gather traffic) | MG, plan_fa_part_A.md |
| LEGACY=0 B=1 decode gap closes under today's dispatch (Q8-gather) — flip default to LEGACY=0 | same-boot serving A/B (boot O, Qwen3.8-27B TP=2, B=1 pp2048/tg256, 2 samples/arm) | **DEAD-END** (flip question closed 2026-08-29) | branch `feat/fa-legacy0-b1-decode` (unmerged; no code to revert) | A 40.11/40.12 vs B (Q8-gather) 37.61/37.56 (−6.3 %) vs C (direct-paged, M5 era) 37.55/37.54 (−6.4 %) t/s; B≈C in serving despite −36 %/+31 % kernel-level subcomponent deltas → gap is a LEGACY=0-common per-step serving cost, NOT FA/gather (Q8 gather is 22–45 % FASTER per step, growing with Sk); measured append-time Q8 write = +94.6 us/step eager (16 layers) — the ~1.55 ms/step remainder is graph-node/serving-harness interaction (unmeasured, refrigerated: fuse Q8 write into the triton append) | FA-LEG, DEVLOG-fa-legacy0-b1-decode.md |

**Sources:** `M0`=DEVLOG-moe-opt.md · `FA`=DEVLOG-fa-attention.md ·
`D`=DEVLOG-dense-decode.md · `M1`=DEVLOG-moe-m1-sprint.md ·
`G1`=DEVLOG-moe-gemm1-retiling.md · `S`=DEVLOG-spec-decode.md ·
`Q`=DEVLOG-qwen38.md (incident, not a dead-end) ·
`GA`=DEVLOG-gemma4-*.md · `W1`=DEVLOG-gdn-mixed-decode.md ·
`MG`=DEVLOG-muse-glimmer.md.

| Onboard Qwen3-30B-A3B-AWQ as the next generic AWQ MoE candidate (E=128/topk=8/hidden=2048 may fit the M=1 tile) | — (never started) | **SUPERSEDED** (2026-08-29 user decision: not an active goal — the supported Qwen3.5/3.8 line supersedes the model) | — | candidate removed from the onboarding queue during the roadmap reorg; the generic AWQ queue itself stays open for the next compatible checkpoint | ROADMAP (onboarding queue) |

## OPEN / IN-FLIGHT (verdict not yet recorded)

| Hyp | Gate | Status | Refs |
|-----|------|--------|------|
| gemm1 activation-fusion (fold SiLU·mul into epilogue) | serving wall-clock | low transfer expectation (see §interactions in G1) | G1, roadmap C2 |
| C3 zeroing fold into neighbor kernels | serving wall-clock | ~234 µs/step measured, no numerics change — cheap lever | G1, roadmap C3 |
| Spec-decode MTP k=2 | serving graph | **SHIPPED** 39.4 t/s (1.41×; 1.50× no-max build), 1.82 tok/step | S |
| Dense W16A16 long-K GEMV (K=17408 down_proj) | serving | **SHIPPED**, at HBM floor (227.6 vs 795 µs, 101% floor) | D |
| W1 GDN mixed-batch chunk reclass (~20 ms/step per no-draft seq) | 2-req mixed probe + identity + serving A/B | **SHIPPED** 2026-08-26: kernel spy 2016 wasted chunk calls → 0 (9B); spec side token-identical; 27B mixed 2-req ngram serving 59.35 vs 55.60 t/s = **+6.7 %** (4 samples/arm, ±0.3 %) | W1 |

## The recurring lessons (why the negative evidence is the point)

1. **Standalone harness wins do NOT transfer to serving wall-clock.** The
   profiler/launch-regime is pipeline-state dependent even in eager, not
   only in graph contexts. `G1 §4` measured the graph-per-kernel census to
   be impossible here (~7% visibility).
2. **Serving `_bench_gfx906.py` graph A/B is THE gate** for µs-scale
   verdicts; eager A/B can tie even when kernels differ (launch-bound).
3. PPL/prompt_logprobs is an unreliable gate on Gemma-4 (hybrid attention);
   gate on coherent text + logprob A/B (`GA`).
