# MoE decode on gfx906 — future optimization roadmap
Copyright Kevin Read <me@kevin-read.com>

Status: roadmap (2026-08-16) — **not a committed plan.** Phase 3 is
declared done at **64.08 t/s** (gap 1.10× vs llama.cpp 70.3 t/s); this
document catalogs the remaining MoE-side headroom so the next phase can
start from measurements instead of memory. Every item carries an
evidence state: **measured** (number from a trace/probe this repo has),
**derived** (arithmetic from measured numbers), or **hypothesis**
(untouched). House protocol applies to anything that gets built:
micro-bench per shape before the model path, PPL/greedy gates, serving
A/B, separate commits, positive AND negative results in the DEVLOG.

Transferability of this work to other model families (AWQ MoE in
general, Ling-3.0-tiny, DeepSeek-V4-Flash) is assessed in
`roadmap-more-models.md` (same directory).

Model facts (Qwen3.5-35B-A3B-AWQ, this deployment): 40 layers, all MoE
(E=256, topk=8, hidden 2048, expert w13 N=1024×K=2048 fused gate+up,
w2 N=2048×K=512, AWQ w4a16 128-group), shared expert N=512 (dense
**fp16** — the checkpoint's `modules_to_not_convert` excludes
`shared_expert` — through the LLGemm1/LLMM1/GEMV surface), 30 GDN +
10 FA layers, single-request serving, **no spec decode** (the GDN
empty-core-out gate in v15 relies on that). The checkpoint also
excludes `model.layers.0.` — layer 0's routed experts ship fp16 and
run the unquantized Triton path (C4, O1 resolved).

## 1. Where the MoE time goes (B=1 decode step)

Measured from the post-FA-track rocprofv3 trace (4500 ms / 256-step
window, `/tmp/bench/trace_fa/.../719850_results.db`) and the P3-4 eager
torch-profiler attribution (DEVLOG "P3-4", `/tmp/bench/fillprof_result.txt`):

| component | calls/step | µs/step | notes |
|---|---|---|---|
| `moe_gemm_q4_kernel_gfx906<1,4>` (routed gemm1+gemm2) | 77.7 | **1922** (24.73 µs/call) | BM=1, NPT=4; grid.z 64/16-way K-split + fp16 CAS atomics. (The v14 budget table's 2662 µs is window-amortized and includes the 2.7 ms prefill `<8,2>` calls; 1922 is the decode-step figure.) |
| `topkGating<8,256,...>` | 40.0 | **568** (14.19 µs/call) | softmax+top8 over 256 experts, 1 token |
| `moe_align_block_size_kernel` | 39.0 | **299** (7.66 µs/call) | sorts 8 (token,expert) slots into per-expert blocks |
| `count_and_sort_expert_tokens_kernel` | 39.0 | **178** (4.57 µs/call) | ~at the launch floor |
| `w1_out.zero_()` [8,1024] | 38.7 | **121** | required by the atomic K-splits |
| `output.zero_()` [1,2048] | 38.7 | **113** | required; **aliases** w1_out's memory (see §2) |
| MoE-adjacent `copy_` [1,2048] | ~40 | **~158** (half of the 316 µs/80.3 group) | exact call site not yet pinned (P3-4 left the other half = GEMV pad, now fixed) |
| `fused_moe_kernel.kd` (Triton) | 2.0 | **414** (206.8 µs/call) | **layer 0's routed experts** — checkpoint `modules_to_not_convert` leaves `model.layers.0.` fp16 → unquantized TritonExperts oracle (O1 resolved 2026-08-16; C4) |
| `moe_sum_vec_kernel` | 1.0 | 6 | — |
| **total** | ~245 | **≈ 3.80 ms** | |

Weight-bandwidth floor (derived): per layer the 8 routed experts' W4
weights are 8 × (1.0 MB w13 + 0.5 MB w2) = 12 MB; × 40 layers = 480 MB
per step → **602 µs/step at 798 GB/s** (the P3-4 fill cuts did not move
this; the GEMM kernel is ~3.2× it at 1922). Routing floor: ~40 layers ×
3 kernels × ~4.6 µs launch floor = **~550 µs** if kernels stay separate,
**~200 µs** if fused to one per layer. Zeroings floor: **0** under a
direct-store design. Realistic whole-MoE floor ≈ **0.9–1.0 ms/step**.

**Headroom ≈ 2.7–2.9 ms/step — larger than the entire remaining e2e gap
to llama.cpp (~1.4 ms).** That is the case for a MoE-decode phase; it is
also why nothing in §3 is scheduled yet: the plan's own P2-4 gate was
"gap > 2 ms" and the gap is 1.4 ms, so this roadmap only activates if
the 70 t/s target stays the goal.

**2026-08-18 re-anchor** (eager torch-profiler, DEVLOG-moe-m1-sprint):
topkGating 17.9 µs/call (713 µs/step), expert gemm 27.8 µs/call
(2149 µs/step). The table above is the FA-track rocprofv3 snapshot; the
fresh eager numbers are ~10% higher for these two rows (build/date
drift, not a regression). The sprint worked from the fresh table.
Note for micro-kernel work below ~10 µs: per-kernel profiler rows are
pipeline-state dependent (measured 1.8–20 µs for the SAME kernel in
different contexts) — wall-clock serving A/B is the gate.

**Sprint outcome (2026-08-18/19, DEVLOG-moe-m1-sprint):** of the
fresh table's top MoE levers, S2 (M=1 topk) shipped default-OFF
(graph-replay loss), S5 (gemm2 re-tile) shipped default-OFF behind
`VLLM_GFX906_MOE_M1` (+0.60 t/s graph when on), and S3 (shared
down_proj GEMV) shipped **default-ON, provisional** (−70 µs/step
GPU-busy; clean serving re-A/B pending). The 64.08 t/s Phase-3 close
was superseded within the sprint: 67.03 t/s with S5 on (record:
67.39). Open MoE buckets are now the topkGating launch itself
(713 µs/step — only a fusion (C1b/c) can win it) and the zeroing fold
(C3, 234 µs/step); gemm1 (≈1070 µs/step) closed 2026-08-19 with no
wall-clock lever found (DEVLOG-moe-gemm1-retiling.md).

## 2. Structural facts (constraints for any design)

1. **Atomic K-splits force two zeroings.** The gemm kernel tiles K via
   `grid.z` (K/32) and accumulates with packed fp16 CAS; both gemms need
   pre-zeroed outputs (121 + 113 µs/step). P3-4 proved the zeroings
   cannot be reordered or merged: `modular_kernel._allocate_buffers`
   aliases `workspace13` (gemm1 out) and the fused `output` in one
   `common_workspace`, so the gemm2 zero must follow gemm1+activation
   over the same memory. Any buffer-lifecycle change must respect the
   aliasing (prefill PPL 6.69 → 1.09e7 when it was violated).
2. **B=1 gives em=8, all distinct experts.** BM=1 is forced (each
   (token,expert) slot is its own group in `sorted_token_ids`); no
   slot-pairing benefit exists at B=1 (that only arrives with
   multi-batch, a separate project per the Phase 3 scope note).
3. **gfx906 has no int8 fast path.** No MFMA, no int8 matrix cores, no
   DP4A (Vega 20 / GCN5). The fp16 `v_dot2_f32_f16` 16B-dot is the
   cheapest dot available; an int8-activation (Q8_1) dot must expand to
   int32 multiply-accumulate — see C6 for the likely-negative verdict.
4. **Graph replay removes CPU launch cost, not GPU dispatch cost.**
   Under FULL_DECODE_ONLY each kernel still takes ~3–5 µs of GPU-side
   dispatch at these sizes (P3-4 measured the fill pile at median
   4.64 µs). "Fewer kernels" is the only launch-side lever; at 99.5%
   GPU busy, only the critical-path share of each removed kernel is
   actually recovered (P3-4 net was +0.7 t/s against ~0.4 ms removed —
   discount all estimates in §3 by a similar factor).
5. **No spec decode** in this deployment: rejected-token padding is not
   a correctness concern (it is for upstream; see the GDN zeros gate).

## 3. Candidates

Gains are per-decode-step, from §1. "Gate" is the measurement that must
pass before code changes (house protocol).

### C1 — Routing pipeline: small-M specialization + fusion (measured 1045 µs)

`topkGating` is 14.19 µs/call for one token over 256 experts — ~3× the
launch floor with trivial work; `moe_align_block_size` is 7.66 µs to
sort **8 items**; `count_and_sort` is already at floor. Options, in
increasing order of surgery:

- (a) **Small-M fast path in each kernel** (M ≤ 32): skip the
  histogram-over-256-buckets / prefix-sum structure; topk via a single
  wavefront, align via straight insertion of 8 (expert, slot) pairs.
  Target: all three ≤ ~5 µs → −350 to −450 µs/step.
  - **(a)-topk was attempted (S2, 2026-08-18): the dedicated M=1 kernel
    is bit-equal and wins in isolation (12.5 vs 17.3 µs) and eager
    serving (+0.11 t/s) but LOSES in CUDA-graph replay (−0.95 t/s at
    66 t/s). Shipped default-OFF behind `VLLM_GFX906_TOPK_M1`. A
    standalone small-M topk kernel does not survive the gapless regime —
    see the S2 table in DEVLOG-moe-m1-sprint.**
- (b) **Fuse topk+align+count into one kernel per layer** (they already
  form a 3-stage pipeline on 8 elements). Target: 40 × ~6 µs = 240 µs →
  −800 µs/step. Needs one output-layout design (the gemm kernel consumes
  `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded`).
- (c) **S2' — fold topk into the router GEMV epilogue** (the post-S2
  lead): the gate GEMV [256,2048] already reads the token; computing
  softmax+top8 in its epilogue (or a 1-wave follow-up fused with the
  GEMV) removes one of the 40×/step kernel launches entirely, which is
  the only regime-safe way to win the topk budget. Same output-layout
  design as (b) for the align/count side.

Risk: low–medium (standalone kernels, unit-testable in isolation; the
gemm contract is unchanged for (a)). Gate: micro-bench the three kernels
at M=1/8/32/128 with PMC counters (TCC + wavefront occupancy) to confirm
the 14 µs/call is structure, not DRAM latency. (Superseded for the
µs-scale verdicts by the wall-clock A/B gate — see §1 note.)

### C2 — Routed-gemm re-tile at decode sizes (measured 1922 µs vs 602 µs floor)

(2026-08-18 re-anchor: 2149 µs/step = 27.8 µs/call in the fresh eager
probe; this is the sprint's **S5** — the V2 in-block K-parallel GEMV
re-tile, built and shipped same week, see the result below.)

**Result (shipped, `VLLM_GFX906_MOE_M1`, default OFF):** the V2
lane-column re-tile (512t, wave-per-K-slice, LDS reduce, wave-0-only
epilogue) was built and validated. It wins big for **gemm2 only**:
standalone 21.4 → 10.8 µs (1.98×, <512,4,256>), in-model 26.8 → 22.3
µs/call; serving +0.60 t/s graph (66.43 → 67.03), +0.46 eager
(23.50 → 23.96), token-identical. The **gemm1 re-tile is neutral
in-model** (27.5 vs 26.8 µs/call; standalone 1.18× did not transfer)
and was dropped from the dispatch. Two design/ISA facts learned (see
dev log): with lane-based columns every wave holds the same reduced
value for the same cells, so the CAS epilogue must run in exactly one
wave (direct stores hide the 8× duplication); and intra-wavefront
same-address CAS contention is pathological on this stack (lost updates
at 2^11, aperture-violation aborts with multi-cell patterns) — keep
per-lane distinct CAS targets. gemm1 (≈1070 µs/step) stays open; the
V1 full-K single-wave design and the activation-fusion idea are the
next candidates. Post-sprint review fix (2026-08-19): the v2 launcher
guard was tightened (`size_k%256==0` + `groupsize%32==0` — the
kernel consumes 32 k-elements per iteration per wave) and the dead
gemm1 dispatch branch removed; the env flag does not affect captured
CUDA graphs.

**C2-gemm1 close (2026-08-19, `DEVLOG-moe-gemm1-retiling.md`): all decode-size
gemm1 retiling is closed.** V1 (full-K direct store, 64 blocks) rejected in
standalone: 2.2× (2-wave) / 4.3× (half-wave) slower than current 26.9
µs/launch — 64 long 128 KB streams cannot keep enough HBM bytes in flight.
Full (BLOCK_KN, NPT) surface swept: NPT=2 (128 blocks) is the best point,
23.8 µs (−11.6%), but its in-model gain does **not** transfer to wall clock
(eager census −186 µs/step; serving A/B neutral in graph +0.08 t/s and eager
−0.03 t/s, 4 samples each) — the third consecutive gemm1 retiling (S5-V2,
NPT sweep, V1) to fail transfer. V3 closed unbuilt (added cost on a
non-transferring CAS design); V4 closed by sweep monotonicity (finer
K-splits / more CAS is worse). Measurement finding: under CUDA-graph replay
(FULL and FULL_AND_PIECEWISE) the torch profiler sees ~5 of 80 gemm
calls/step — per-kernel A/B under graphs is infeasible on this stack,
confirming wall-clock A/B as the gate. No dispatch change shipped; the
harness keeps the sweep/v1 kernels as the Phase-0 tool. Remaining C2
substance: activation fusion (transfer expectation now low) and C3 (zeroing
fold, 234 µs/step measured, no numerics change) as the cheap lever.

### C2-V — Validation experiments for the C2 close (proposed 2026-08-19, not run)

Review of the C2 close identified regime and power gaps in the
rejection evidence. Each is cheap to close before the close-out is
treated as permanent:

- **(v1) A/B power floor.** The serving gate ran 4 samples/arm with an
  off-arm spread of 0.17 t/s; the expected eager effect (~186 µs of a
  ~15 ms step, ~1.2%) sits *at* that resolution — "neutral" means
  "below detection", not "no effect". Re-run the NPT=2 A/B (flag is
  in the dev log, one-line dispatch change) with ≥16 samples/arm,
  interleaved (off/on/off/on) to cancel build/date drift, and record
  the resolved CI. Only a CI excluding ~0.3% closes the question.
- **(v2) Parallel-request regime — the main untested axis.** The gate
  is `_bench_gfx906.py`, i.e. a *single request* (batch 1, capture
  size 8). All three "failed transfer" verdicts (S5-V2 gemm1, S2
  topk, NPT sweep) were rendered in that regime only. Under concurrent
  requests: (a) the step is busier, so per-kernel savings are far more
  likely to hit wall-clock; (b) decode batch >1 moves MoE shapes
  toward the grouped-GEMM (BM≥8) path, which the entire (BLOCK_KN,
  NPT) sweep never measured. Experiment: a multi-request bench
  (N=4/8/32 concurrent generations, e.g. `llm.generate` with a batch
  of prompts; `BENCH_MAX_SEQS` already exists) re-run off/on for
  `VLLM_GFX906_MOE_M1` and the NPT=2 trial flag, both regimes. Any
  positive ≥0.5% reopens C2-gemm1 and the S5 gemm1 branch.
- **(v3) V1 block-count axis was never swept.** V1 was rejected at
  exactly 64 blocks (2 waves / half-wave variants only) — its loss
  mechanism (too few long HBM streams) is also the axis it was never
  varied on. A V1 derivative with N-split (e.g. 128–256 blocks each
  streaming half/quarter of K, direct store retained, still no CAS,
  still kills both zeroings — i.e. C3 subsumed) was not built.
  Experiment: extend `moe_m1_harness.cu` with `moe_gemm_q4_v1n<N>`
  (N ∈ {128, 256, 512}) — a day of harness work, standalone gate
  only, before any model-path interest.
- **(v4) S5-era standalone numbers were produced by a broken-check
  harness.** The HARNESS-FAIL self-review fix means every harness run
  since S5 "passed" with correctness checks that could not fail.
  Timing rows are probably unaffected, but re-run the harness PASS
  flow once on the current build to confirm the kept S5/S2 reference
  numbers (21.4/10.8 µs gemm2 V2, 12.5 µs topk M=1) reproduce under
  the fixed checks.

State (2026-08-22): **(v2) running** on `gfx906/moe-c2v` (dev log
`DEVLOG-moe-c2v.md`). Dispatch-gate audit corrected the scope: both
existing re-tile candidates are M=1-only (`MOE_M1` gemm2 v2 tile
gated on `size_m == output_topk`; the reverted NPT=2 trial was the
BM=1 gemm1 path), so the batch axis is the *never-measured* BM=4
grouped path (N=4/8/32 characterization; the NPT=2 arm can only fire
at N=4) and **TP=2 M=1 is a first-class arm** (per-rank N halved —
new tiling axis; first TP=2 35B-MoE run on this box, smoke-gated).
TP=1-only scoping was overruled: a TP=2 win would reopen the branch
regardless of TP=1. (v1) still unrun; (v1)+(v2) remain prerequisites
before the C2 close is cited as evidence in any future scope
decision; (v3) is the remaining unbuilt design axis; (v4) is
bookkeeping hygiene.

The BM=1/NPT=4 tiling launches 4096 blocks (gemm1: 8 slots × 8 n-tiles ×
64 K-splits, 32 threads each) per layer; each (slot, n-column) cell is
CAS'd by 64 blocks. Variants to micro-bench (isolated bench, per
house rule, before touching the model path):

- V1: **full-K direct store** (grid.z=1): 64 blocks (gemm1), each a
  1-wavefront 2048-K loop. Kills atomics + both zeroings. Risk:
  under-occupancy (64 wavefronts on 60 CUs).
- V2: **in-block K-parallel GEMV tiling** (the classic shape):
  256-thread blocks, 8 wavefronts each take a 256-K slice, LDS
  cross-warp reduce, **one direct store per cell**. Grid ≈ 64 blocks
  (gemm1). Same kill as V1 with 8× the threads per block. This is the
  leading design; it restructures the k-loop (currently fixed 32-wide,
  `BLOCK_KN == THREADS_X` static assert) and the dequant prefetch.
- V3: **2–4-way K-split into fp32 scratch + tiny add**: keeps CAS-free
  stores, adds scratch traffic (2 MB/layer, ~2.5 µs) + one add kernel
  (40 × 4.6 µs). Only worth it if V1/V2 lose on occupancy.
- V4: keep the split, **halve the CAS fan-in** (NPT=8, z=32) — cheap
  ablation to quantify the atomic cost itself.

Target: gemm → 700–900 µs/step (1.2–1.5× floor; the floor is not
reachable at 64–512 blocks/CU) **plus** the 234 µs zeroing kill →
**−1.2 to −1.5 ms/step**. Risk: medium (kernel rewrite of a vendor-ported
but in-repo file, `csrc/rocm/moe_q_gemm_gfx906.cu`; the 12-test suite in
`tests/kernels/moe/test_gfx906_moe_gemm.py` + PPL gate cover it).

### C3 — Fold the zeroings into neighbor kernels (measured 234 µs)

If C2 lands, this is subsumed. If C2 stalls: the `w1_out.zero_()` can be
issued from the tail of the align/topk kernel (disjoint buffers, same
stream, capture-safe) and the `output.zero_()` from the activation
kernel (disjoint from `act_out`; verified in P3-4 that the activation
never writes the common buffer). Saves one launch each ≈ −190 µs/step
(after the critical-path discount of §2.4, realistically −100–150).
Risk: low (no numerics change) but it is cross-kernel surgery in the
align + activation paths.

### C4 — Layer-0 fp16 routed experts (measured 414 µs) — identity RESOLVED

`fused_moe_kernel.kd` (2×/step, 206.8 µs/call) is **layer 0's routed
MoE**: the AWQ checkpoint's `modules_to_not_convert` lists
`model.layers.0.`, so that layer's 256 experts ship fp16 and the
quant-method oracle routes them to the unquantized Triton path
(`unquantized.py: Using TritonExperts MoE backend` in the load log;
`int_wna16.py: Using Gfx906WNA16Experts` covers layers 1–39).
Attribution done 2026-08-16 with the P3-4 method (kernel External id →
`vllm::moe_forward_shared` cpu_op → enclosing layer frame; 114/114
ops in `Qwen3NextSparseMoeBlock_0`, eager trace
`/tmp/bench/fillprof/`). Options: **(a) leave** — 414 µs is ~2.4% of
the 17.5 ms step; **(b) re-quantize layer 0 to AWQ at load** —
calibration-free per-group W4A16 of the fp16 weights for the excluded
modules inside the AWQ loader; removes the only Triton dependency in
the decode path and shrinks layer 0's weight bytes 4×, but changes
layer 0 numerics → PPL gate mandatory, and it is a quant-method
change (upstream-class, nontrivial). An fp16-dense replacement
(gather + per-expert aiter GEMM, like the shared expert) was
estimated at no better than ~200–300 µs for 8 active experts × 2
legs — not a clear win. Verdict: **(b) only if the 70 t/s target is
live**; otherwise (a).

### C5 — Shared-expert chain fusion (derived ~300–400 µs)

The shared expert (dense **fp16**, see header) is three sequential
small GEMMs/acts (w13 [1024,2048] → SiLU·mul → w2 [2048,512]) at B=1;
P3-2(b) put each leg at its GEMM-kernel optimum ("3.6–14× floor rows,
launch/latency bound, no GEMM kernel closes them") — but that
adjudicated the legs separately. Fusing the **chain** (w13+act+w2 in
one kernel with an in-block or grid-sync barrier) removes 2 launches ×
40 layers. At B=1 the intermediate is [1,1024] fp16 = 2 KB —
trivially shareable inside a block or via one barrier. Effort: high
(new fp16 kernel + fused SiLU·mul — simpler than a W4A16 version
since there is no dequant); risk: medium; expected −150 to −250 µs
after the §2.4 discount.

**Sprint partial result (S3, 2026-08-18, shipped default-ON
provisional):** the per-leg GEMV question is answered — the w2 leg
[2048,512] moved to `dense_gemv_gfx906` (kc512/RPT=2, 5.6-5.7 vs
6.7-7.7 µs LLMM1; −70 µs/step GPU-busy in-model; kill switch
`VLLM_GFX906_DOWN_GEMV=0`), while the w13 gate_up leg [1024,2048] is
a measured GEMV **loss** (8.0 vs 7.3 µs — no lever, stays LLMM1).
The remaining C5 substance is therefore only the chain fusion (2
launches/layer); its expected value shrank accordingly. The default-ON
call is provisional pending a clean serving re-A/B (see the S3
section of DEVLOG-moe-m1-sprint).

### C6 — Q8_1 activation quant (llama.cpp's decode mechanism) — likely NO on gfx906

llama.cpp quantizes activations to Q8_1 and dots int8×int4 (mmq).
Adopting it here means: (a) a per-layer activation-quant kernel (80 ×
~4 µs ≈ 320 µs **added**), and (b) replacing `v_dot2_f32_f16` (2 fp16
products/instruction, free fp32 accumulate) with int8 products expanded
through int32 MACs — on a part with **no DP4A and no int8 matrix cores**
(§2.3), that is ~2–3× the instruction count per 8 elements. The
activation-byte savings are irrelevant at M=1 (4 KB rows). llama.cpp's
MoE advantage on this box is structural (kernel shape, launch count),
not format: its Q4_K weights are ~12.5% **heavier** in bytes than our
packed AWQ W4 (1.125 MB vs 1.0 MB per expert per leg, 4.5 vs 4.0
bits/elem) yet it still wins overall — the weight format is not its
fast path. **Verdict to record without building: expected net-negative
on gfx906.** Reopen only if C2 lands and the gemm is still > 2× the
weight floor.

### C7 — MoE-block persistent/cooperative kernel (the endgame, ~1 ms launch floor)

One cooperative kernel per layer doing topk → align → gemm1 → act →
gemm2 with grid-wide barriers removes the whole 7-kernel pipeline's
launch floor (~300–400 µs critical-path) and lets the K-splits live in
LDS/L2 without CAS. Feasibility is unproven on gfx906 (HIP cooperative
launch support on Vega 20 must be verified; resident-grid capacity for
the ~4096-block work must be checked). Highest effort, highest risk;
only after C1+C2, and only if the gap target still stands.

### C8 — L2-residency probe for expert weights (informational)

12 MB/layer of W4 weights vs MI50's L2 (a few MB — verify with
`rocagent`/PMC before citing): if weights stream cold from HBM every
step (no cross-step reuse; each layer routes to different experts), the
602 µs floor is HBM-bound and C2's target should be set from HBM
bandwidth measurements, not from the current kernel's L2 behavior. The
P3-0 Q1 TCC_HIT/TCC_MISS mechanism already exists for dense kernels;
repoint it at `moe_gemm_q4`.

### C9 — Multi-stream overlap of shared vs routed (engine-level)

Within a layer the shared-expert chain is independent of the routed
chain; a two-stream (fork/join) capture could overlap ~60 µs of shared
work behind ~49 µs of routed per layer. vLLM captures single-stream;
this is engine surgery, not MoE surgery. Parked — revisit only with a
multi-batch project where the overlap window is larger.

## 4. Recommended sequencing (if the phase is ever started)

1. **Phase 0 — characterization (days, no model-path changes):** C8
   (TCC on the gemm) + C1 gate micro-bench (routing kernels at
   M=1/8/32/128) + C2 ablation bench (V1–V4 at gemm1/gemm2 shapes).
   *(Partially done by the 2026-08-18 sprint re-anchor: the fresh
   in-model table replaced the stale §1 premise; C2's V2 variant was
   benched and shipped (default-OFF); C1's (a)-topk was tried and
   rejected for the gapless regime. C8 remains open.)*
   (C4's identity question was resolved 2026-08-16 — layer 0's fp16
   routed experts; see §6.) Output: a DEVLOG table deciding which of
   C1/C2 is real.
2. **Phase 4a — C1 + C2** (the ~2–2.5 ms pair): routing small-M path or
   fusion, then the gemm re-tile with the zeroing kill riding along.
   Gates per item: isolated micro-bench, 12/12 MoE tests, PPL probe
   (accept the MoE-atomic run noise band, ~0.003 abs), MB greedy heads,
   serving A/B (2 samples each, sequential, §2.4-discounted expectation).
3. **Phase 4b — C5** (shared-expert chain fusion) if 4a nets < +1 ms.
4. **C7** only if the 70 t/s target is still the goal after 4a+4b.
5. **C6 stays rejected on paper** until C2 evidence says otherwise.

Stop conditions: Phase 0 shows the routing/gemm time is DRAM-latency
bound (not structure) → close as "at the floor" with the TCC evidence;
any item netting < +0.3 ms/step measured in serving A/B → reject and
record (P3-4 precedent: 350 µs removed → +0.7 t/s, ~2× discount).

## 5. Open questions

- **O1 (resolved 2026-08-16):** the 2×/step `fused_moe_kernel.kd`
  Triton calls (206.8 µs each) are layer 0's routed experts — the
  checkpoint's `modules_to_not_convert` excludes `model.layers.0.`, so
  they ship fp16 and take the unquantized Triton oracle. See C4.
- **O2:** exact call site of the ~40/step MoE-adjacent `copy_` [1,2048]
  (158 µs/step).
- **O3:** llama.cpp's per-component kernel budget on the same box
  (we have its e2e 14.2 ms/step but never a kernel-level trace) —
  needed to compare "where does it spend 1.4 ms less than us".
- **O4:** MI50 L2 size + expert-weight residency (C8).
- **O5:** does `topkGating`'s 14 µs include an HBM round trip
  (256-expert logits row = 1 KB, likely L2) or pure structure? (C1 gate)

## 6. Phase-2 (MoE) open-items cross-reference

Phase 2 (MoE prefill tuning; see `DEVLOG-moe-opt.md` "PHASE 2") closed
with these items open; where each
lands in this roadmap:

- **P2-4 fused topk+align** (≈1 ms/step routing, high correctness
  risk) → **C1** (routing small-M path or fusion).
- **P2-5 "shared-expert Triton elimination"** (~0.55 ms/step as
  profiled in P2) → premise corrected 2026-08-16 with the O1
  resolution: the P2 profile's "shared expert (Triton fused_moe)" line
  was **layer 0's routed experts** (fp16 unquantized), not the shared
  experts. The shared experts were **already dense fp16** all along
  (`modules_to_not_convert` excludes `shared_expert`) — there was no
  Triton to eliminate. P2-5's residual substance splits into **C4**
  (the layer-0 Triton, 414 µs) and **C5** (shared-expert chain fusion,
  now fp16-fp16 rather than W4A16 — simpler).
- **P2-1(e) persistent-CTA B-in-LDS prefill gemm** (the only path to
  the ~2× prefill gemm goal; a/b/c landed for +26% and stalled at
  ~5.9 TFLOPS ≈ 30% of the practical dot2 peak) → **out of scope for
  this decode roadmap** (prefill). Parked here so it is not lost; the
  DEVLOG "P2-1" section holds the measurements.
- **P2-3 decode-MoE small-M latency** (skipped in P2 on the premise
  "MoE is a small fraction of the step") → premise superseded by the
  Phase 3 budget (§1: MoE ≈ 3.8 ms of a 17.5 ms step); this document
  is the rescope of P2-3.
- **P2-6** (dense GEMMs, decode paged attention, elementwise/norm
  fusion — non-MoE) → became Phase 3's scope (P3-1/2/4); no MoE
  residue.

## 7. Non-MoE items (dense/general; parked from the dense takeover, 2026-08-17)

- **N1 — silence the expected AutoAWQMoEMarlin fallback warning
  (cleanup, expected-on-gfx906).** `auto_awq.py`
  `get_quant_method` (RoutedExperts branch): `check_moe_marlin_supports_layer`
  fails for this checkpoint, so every MoE layer logs
  `Layer '...mlp.experts' is not supported by AutoAWQMoEMarlin. Falling
  back to Moe WNA16 kernels.` — one `logger.warning_once` per layer
  prefix, ~39 lines per engine start. On gfx906 this fallback is
  **expected and is the intended fast path** (Moe WNA16 → the custom
  gfx906 W4A16 kernel); the warning mis-signals a problem. Proposed
  fix: platform gate — on gfx906 (where Marlin W4 MoE is unavailable)
  emit a single info/debug line; keep the per-layer warning on other
  platforms where the fallback may be a genuine surprise. Log-only
  change, but still run the PPL + serving sanity gate (behavior must
  not change). Evidence state: **measured** (site pinned;
  `vllm/model_executor/layers/quantization/auto_awq.py`).
- **N2 — FA B>1 decode direct kernel store (derived ~192 KB/layer at
  B=8).** After the 2026-08-17 native-BSHD output work, B=1 decode has
  zero output-path copies; B>1 decode still pays one reshape copy per
  layer (the [B,Hq,D] row block extracted from BSHD is 192 KB at B=8).
  The real fix is a decode-specialized kernel store: when the real
  seq_q == 1, allocate [B,Hq,D] (no Sq dim), store only the j==0 column
  into (b,h) rows, and make kv_split partials [B,Hq,kv_split,D]. The
  kernel needs a `decode_single` flag — the launcher sees Sq_pad=2,
  not the real seq_q, so it cannot infer this today. Only matters for
  batched decode (the production bench is single-request); B=1 is
  already copy-free. Evidence state: **derived** (per-call probe +
  stride analysis in the DEVLOG "FA decode per-layer copy pile").
- **N4 — TP=2 dense decode: `max_model_len` decode tax — RESOLVED
  (2026-08-22, merged to `gfx906/main`; see `DEVLOG-masked-fa.md`).**
  Mechanism (confirmed 2026-08-21, S8): FULL-cudagraph capture bakes
  `Sk_pad = pad32(max_model_len)` into `GFX906_FA`'s gather launch dims
  (`gpu_model_runner.py:2390` `for_cudagraph_capture` branch), so the
  two-kernel `> 65535` fallback gathered/quantized/zeroed
  max_model_len-wide rows every replay regardless of live context
  (eager A/B at matched ~1.5k context: no gap; graph A/B: −25% at
  262k). **Fix**: `gather_paged_kv_quant_persistent` — one grid-stride
  fused gather+quantize kernel with a fixed capture-time grid and work
  bounded by the live `seq_lens` tensor (the masked-early-exit route,
  `plan_masked_fa.md` §2.2); the capture-time Sk bound + fallback
  design was superseded before implementation (its CORRECTION: a
  single bound cliffs long-running conversations — the exact workload
  262k exists for). Gates: NaN-tail, bit-exact capture/replay B=1..4,
  PPL identical, TP=2 serving A/B: 131k 22.4→40.9 t/s (+83%),
  262k 15.9→40.9 t/s (+157%), P1 tax 0.07% (noise) vs P0 −28.8%.
  `GFX906_FA_PERSIST` default ON. Full record:
  `DEVLOG-masked-fa.md`; diagnosis: `tp_decode_investigation.md`
  RESOLUTION, `DEVLOG-tp2-dense.md` S8. Evidence state: **fixed and
  gated** (serving A/B).
- **N3 — GDN [3,1,32] state-bookkeeping copies (measured 32/step,
  ~180 µs/step eager).** The timeline probe
  (`/tmp/bench/dense_ewp_timeline.py`) attributes the 32 [3,1,32]
  copies/step (~180 µs, launch-latency-bound) to upstream vLLM
  mamba/GDN state management around `_causal_conv1d_update` +
  `fused_recurrent_gated_delta_rule` — not FA, not model code. A fix
  means reducing the count in upstream state handling or folding it
  into the GDN core custom op; deferred because it is upstream code,
  small, and the production graph path amortizes the launch cost.
  Evidence state: **measured** (attribution); fix = **hypothesis**.

## 8. Upstream candidates (platform-generic fixes to offer vllm-project/vllm)

Three landed changes in this branch are not gfx906-specific. If they are
ever offered upstream (rebased onto `vllm-project/vllm` main, per vLLM's
contribution policy: human submitter who can defend the change, tests,
AI-assistance disclosure), these are the items:

- **U1 — fastsafetensors GDS fallback** —
  `vllm/model_executor/model_loader/weight_utils.py` (`128e948baf`).
  The GDS→non-GDS fallback only catches `RuntimeError`, but
  fastsafetensors raises a bare `Exception` when GDS reads fail
  (cuFileRead errno 22 on unsupported systems) — engine death instead
  of fallback. Broadening the catch (keeping the `"gds" in str(e)` +
  not-yet-yielded guards) gives 2.6× faster loads where GDS works and
  a working boot where it doesn't. One-line diff; evidence state:
  **measured** (41 s vs 117 s load, DEVLOG "Local-venv bench
  environment").
- **U2 — hipify in-source build guard** — `cmake/hipify.py` (part of
  `225448d93f`). `shutil.copytree(project_dir, output_dir)` raises
  `SameFileError` when both are the same directory (in-source rebuilds
  on Py3.12); guard with an `abspath` compare. Two-line diff; evidence
  state: **measured** (repro'd the crash, DEVLOG "Build-system note:
  hipify.py in-source guard").
- **U3 — GemmaRMSNorm fused-kernel dispatch** —
  `vllm/model_executor/layers/layernorm.py` (`19c1d41cf5`,
  `70ec1d0e79`). `forward_cuda` delegates to `forward_native`, whose
  fp32 `(1 + w)` breaks the fused `vllm_c` rms-norm kernels'
  `weight.dtype == x.dtype` requirement → decomposed elementwise
  fallback (~131 extra launches/step on Qwen3.5-family hybrids in
  eager mode). Gemma's `(1 + w)` factorization is a plain scaled RMS
  norm with `w' = 1 + w` in the input dtype; dispatching with that
  (plus a per-instance `(1 + w)` cache keyed on
  data_ptr/dtype/device/_version) restores the fused path. PPL-gated
  on two models. Evidence state: **measured** (unit maxdiff 0.002,
  PPL in band both models). Caveat for upstream reception: the
  benefit is eager-path only — inductor fuses the decomposition in
  compiled mode.

## 9. Code-review items (parked 2026-08-17, resolved 2026-08-18)

The pre-merge code review of the branch (`gfx906_qwen_impr_code_rev_qwen.md`
in `docs/gfx906/`) resolved its
four severity findings in `01499157a8` (symmetric W4A16 repack crash +
wrong zp fill, GDN platform gate, oracle GPTQ exclusion, `top_k` guard).
The remainder was parked here and is now fully resolved (see the
2026-08-18 DEVLOG section for evidence and gates). IDs are R*,
severities as P2/P3/P4 per the review. (The earlier combined review
file that also lived at the repo root has been deleted; the resolved
record is here and in the DEVLOG. A second review pass over the
moe-m1-sprint commits is §9.4.)

### 9.1 Edge-path correctness (fix before enabling the affected mode)

- **R1 — direct-paged fp16 Q buffer vs fp32 C API (P2) — RESOLVED.**
  `forward_paged`'s direct branch now requires/uses the fp32
  `q_pad_buf` (with an fp32 fallback allocation) and, for Sq=1,
  the dedicated `q_pad_decode_buf` (zero-copy `[:num_seqs]` prefix,
  matching the legacy branch) instead of a per-layer `.contiguous()`
  copy. The in-place fp16→fp32 row stores are unchanged.
- **R2 — LEGACY=0 + prefix caching (P3) — RESOLVED.**
  `get_cudagraph_support` now fails closed with a `RuntimeError`
  (the combination corrupts attention output) while keeping the
  experimental-mode warning for the no-prefix-cache case.
- **R3 — unbounded `_ensure_gather_buffers` retired growth (P3) —
  RESOLVED (bounded keep-alive, not capacity buffers).** The retired
  list is now capped at `_gather_retired_max = 4` pairs (oldest
  evicted). A true grow-only capacity buffer + exact-size view was
  rejected: the gather kernels address their output from shapes, not
  strides, so a non-contiguous view would corrupt silently (a stride-
  based output addressing change is the real fix if LEGACY=0 ever
  becomes first-class). The buffers are now passed on BOTH the
  LEGACY=0 fused-Q8 and the LEGACY=1 fused-gather+quantize paths
  (see R8).
- **R4 — MoE kernel caller contract (P3) — RESOLVED.** The host entry
  now checks `K == qweight_rows*8`, `groups > 0 and K % groups == 0`,
  `N % 8 == 0`, `scales_N == N`, `zeros_N*8 == N` (the kernel derives
  group boundaries as `K / groups` — violations were silent garbage).
  The oracle additionally gates `intermediate_size_per_partition % 8`
  and `hidden_dim % group_size` for GFX906_HIP.

### 9.2 Cross-platform / build hygiene (fix before the fork is built
or run on non-gfx906 hardware)

- **R5 — plugin entry point traceback on non-gfx906 startup (P2) —
  RESOLVED.** `vllm/gfx906_fa/__init__.py` and `gfx906_fa_paged.py`
  import the extension tolerantly, and `register()` is a no-op off
  ROCm-gfx906 (platform check inside the function, so the module
  imports cleanly and the plugin loader finds a no-op `register`
  instead of an ImportError traceback). The backend is never
  registered off gfx906, so it cannot be selected there.
- **R6 — setup.py vs CMake target mismatch on arch auto-detect (P2) —
  RESOLVED (CMake side).** The standard build relies on empty
  `PYTORCH_ROCM_ARCH` + device auto-detect (the env recipe sets no
  arch), so narrowing `_targets_gfx906()` to explicit `"gfx906" in
  rocm_arch` would break the standard recipe. Instead CMakeLists.txt
  now defines a no-op `add_custom_target(_gfx906_fa_C)` + install
  component (with a WARNING) when the lang is HIP but gfx906 is not
  among the arches, so `--target`/`--component` succeed without
  producing a module (the Python import-tolerance from R5 then keeps
  the backend absent).

### 9.3 Cleanup / debt (bundle with the next touch of the file)

- **R7 — ncols1 tile ladder duplicated (P3) — RESOLVED.** Four code
  copies (two Python, two C++) consolidated: `fa_pick_ncols1()` in
  `gfx906_fa.cpp` (both C++ call sites) and `_pick_ncols1()` in
  `gfx906_fa_paged.py` (imported by the backend); each definition
  carries a cross-language "keep in sync" pointer to its mirror.
  (Full cross-language unification would require passing Sq_pad /
  ncols1 across the pybind boundary — not worth the API churn.)
- **R8 — `gather_paged_kv_quantized` fresh alloc per call (P3) —
  RESOLVED.** It now takes the same `k_out`/`v_out` optional
  grow-buffers as its `gather_paged_kv_q8` sibling (`use_or_alloc`,
  exact-shape match); the backend passes the class-level gather
  buffers on both the LEGACY=1 fused-quant and the LEGACY=0 fused-Q8
  paths (previously LEGACY=1 passed `None` → a fresh 24-200+ MiB
  K+V pair per layer per step on long contexts). The fp16 fallback
  (`GFX906_FA_FUSED_QUANT=0`) reuses only the V buffer (its K output
  is fp16, not the uint8 q8 layout).
- **R9 — MoE `apply()` aliasing order dependency (P3) — RESOLVED.**
  Documented in `gfx906_w4a16_moe.py`: `workspace_shapes` now notes
  that `modular_kernel._allocate_buffers` aliases workspace13 and
  fused_out onto one storage and that the gemm1 → activation →
  `output.zero_()` → gemm2 order is load-bearing; the gemm2 call site
  carries the matching short comment.
- **R10 — stale docs/comments (P4) — RESOLVED.** Fixed: `kchunk` 
  docstrings now include 1024 (`csrc/rocm/ops.h` + `vllm/_custom_ops.
  py`); the `forward_paged_direct` pybind help states the native BSHD
  output and fp32 input; the launcher header's MVP block rewritten
  (mask/KV_max/direct-paged implemented, no host transpose); the MoE
  kernel header's grid formula now carries N_PER_THREAD; the 
  `utils.py` `/tmp/bench/…` comment points at the DEVLOG; the dead
  `mask_buf=None` parameter removed from `forward_paged` and the
  backend call site; the `forward_paged` query docstring corrected
  (fp16, cast into the fp32 q_pad).
- **R11 — lint debt in vendored/bench files (P4) — RESOLVED.**
  `vllm/gfx906_fa/*.py`, the gfx906 bench scripts, and the two gfx906
  test files are now ruff-clean (UP/I/F541 auto-fixes + E501 wraps +
  the B023-adjacent F841/SIM108 manual fixes). `benchmarks/kernels/
  gfx906/*` gets a `per-file-ignores` entry for B023 + the false-
  positive F821s (deliberate timeit-closure captures, verified
  harmless). Pre-existing E501/ruff-format debt in `utils.py` (all on
  lines the branch did not touch) is left as upstream debt.
- **R12 — debug env knobs (P4) — RESOLVED.** New `GFX906_FA_DEBUG=1`
  master switch enables all six off-by-default debug hooks at once
  (`FWD_DEBUG`, `DOUBLE_CHECK`, `DUMP` — default dir
  `/tmp/gfx906_fa_debug` — `NO_BUF_REUSE`, `TORCH_GATHER`,
  `ZERO_KTAIL`); the individual knobs remain for finer control. The
  three ON-by-default functional switches (`FUSED`, `FUSED_QUANT`,
  `QPAD_EMPTY`) are NOT debug hooks and are untouched.


### 9.4 moe-m1-sprint review (2026-08-19) — RESOLVED

Two independent code reviews of the four sprint commits produced six
findings (review records deleted after fixing; the DEVLOG
"post-sprint code review fixes" section holds the full account):

- **Latent v2 tile-guard hole (P1-class)** — `size_k%64==0` admitted
  shapes with `slice_k < 32` (reads past `end_k`) and there was no
  `groupsize%32==0` check. Fixed in `979e72c925` (guard is now
  `size_n%256==0 && size_k%256==0 && size_k<=2048 && groupsize%32==0`)
  together with the dead gemm1 dispatch branch. Verified: 25/25
  `test_gfx906_moe_gemm.py`.
- **S3 default-ON evidence rigor** — decision rested on a
  thermally-flagged serving A/B; marked provisional with reopen
  conditions (`e06f484c0e`). The clean re-A/B is an open action.
- **Stale S2 devlog description** — described a dropped draft variant;
  rewritten to the shipped kernel (`b7ec0306ee`).
- **S2 dispatch assumptions unpinned + missing M-gate** — docstring now
  pins softmax/no-bias/no-padding/full-range/rsf=1; `gating_output.
  shape[0]==1` added; end-to-end dispatch test added (`8e7935edd4`).
- **Tooling placement** — `tmp_tp_probe/` harnesses →
  `benchmarks/kernels/gfx906/harness/`; probe scripts →
  `benchmarks/kernels/gfx906/` (`b3e7139aaa`).
- Deliberately deferred: the stronger high-precision accumulation
  check for S5 (required only before any default-ON flip) and the S5
  test's wide `0.3·max+0.05` off-vs-on gate (same condition).

## Appendix A — `moe_gemm_q4_kernel_gfx906` facts (csrc/rocm/moe_q_gemm_gfx906.cu)

- Template `<BM, NPT>` with instantiations `<1,4>`, `<4,4>`, `<8,2>`;
  BM chosen by `em = M×topk`: ≤32 → 1, ≤512 → 4, else 8 (decode B=1:
  em=8 → BM=1).
- `BLOCK_KN_SIZE = THREADS_X = 32` (static assert); block covers
  `NPT×32` output columns for `BM` slots of **one expert** (expert from
  `expert_ids[blockIdx.x]`; slots from `sorted_token_ids`).
- Grid: x = token-blocks, y = N/(32·NPT), z = K/32. K is **not** looped
  per block beyond its 32-wide slice — grid.z tiles K and the epilogue
  CAS-accumulates (`atomic_add_pk4_f16` / `pk2`, packed 64/32-bit CAS
  loops) into the pre-zeroed output. **Neither gemm can direct-store in
  the current tiling.**
- gemm1 (`output_topk=0`): out row = slot id, [em, N]; gemm2
  (`output_topk=topk`): out row = token id, [M, K], with the topk
  router weight fused into the epilogue (no separate reduce kernel;
  `moe_sum` is unused on this path).
- Weights: packed int32 `[E, K/8, N]` (exllama shuffle) + fp16 scales
  `[E, groups, N]` + packed zeros `[E, groups, N/8]`; per-column dequant
  constants (`z1z16/y1y16`) refreshed at group boundaries (128-group
  AWQ); `zero_offset=0` for AWQ (GPTQ-v1 needs +1).
- Activations stage through LDS (`block_a[BM][32+8]`, 16B-padded for
  `ds_read_b128`); double-buffered weight prefetch was tried and
  rejected (256 VGPRs + spills at BM=16, 5× slower — DEVLOG P2-1).

## Appendix B — bench / probe recipes

- Serving A/B (local venv, sequential): the standard recipe in
  `README.md` §Bench recipes (`_b.py`, pp=2048/tg=256,
  util 0.95, fastsafetensors, FULL_DECODE_ONLY); check `uptime` first.
- MoE kernel tests: `tests/kernels/moe/test_gfx906_moe_gemm.py`
  (25 tests, covers both source layouts + the M=1 v2 flag path);
  top-k router: `tests/kernels/moe/test_fused_topk.py` gfx906 tests
  (bit-equality + dispatch path).
- Standalone A/B harnesses (M=1 MoE gemm, topk): `benchmarks/kernels/
  gfx906/harness/` — build/run standalone, outside the tree.
- PPL probe: `benchmarks/kernels/gfx906/ppl_probe.py` (recreated
  12-prompt set 2026-08-18 — absolute values not comparable to the
  6.69-era set; prefill logprobs; the AWQ
  gemm2 atomics make it non-deterministic at the ~0.003-abs level across
  runs of identical code — compare against that band).
- Isolated micro-benches live in `benchmarks/kernels/gfx906/`
  (gemv, fa, gather, llmm1, moe gemm, moe topk) plus the greedy /
  kernel-prof / PPL probes (moved from docs 2026-08-19).
- Kernel trace: `rocprofv3 --kernel-trace` → SQLite (`agg_db.py`);
  remember the ~10–15% per-dispatch inflation under the tracer and the
  untrustworthy grid-axis columns (use timestamps/durations only).
- Eager torch-profiler attribution (fill/copy method):
  the eager torch-profiler correlation method documented in
  `DEVLOG-moe-opt.md` "P3-4" (probe script since removed) — the same technique
  answers O1/O2/O5.

## Appendix C — file map

- `csrc/rocm/moe_q_gemm_gfx906.cu` — the routed gemm (ours, Phase 1/2;
  includes the M=1 v2 lane-column gemm2 re-tile, default-OFF).
- `csrc/rocm/moe_topk_gfx906.cu` — the M=1 top-k router (S2,
  default-OFF behind `VLLM_GFX906_TOPK_M1`).
- `vllm/model_executor/layers/fused_moe/experts/gfx906_w4a16_moe.py` —
  the expert class (apply: zero_ → gemm1 → activation → gemm2;
  workspace lifecycle via `_resize_cache`).
- `vllm/model_executor/layers/fused_moe/modular_kernel.py` —
  `_allocate_buffers` (**common_workspace aliasing**, §2.1).
- `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` +
  `csrc/moe/*` (v1) — routing kernels (C1 targets).
- `vllm/model_executor/layers/utils.py` — `_llmm1_tiny_m` (dense-side
  dispatch incl. the shared expert; P3-2(b)/P3-4 GEMV m==1).
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — GDN
  (context only; its zeros gate is the §2.5 spec-decode precedent).

## Appendix D — pitfalls (from P3-4 and earlier)

- **Alias the buffers and the PPL detonates** (workspace13 == output;
  §2.1). Any "reorder the zero" idea must be re-derived against
  `_allocate_buffers` first.
- **Atomic K-splits** rule out direct stores in the current tiling
  (C2 must re-tile, not patch the epilogue).
- **No int8 fast path** on gfx906 (§2.3) — Q8_1-activation proposals
  should be rejected at review without new hardware evidence (C6).
- **BM=1 at B=1** — any tiling that assumes ≥2 same-expert slots per
  block is dead at B=1 (multi-batch only).
- **Critical-path discount** (§2.4): at 99.5% GPU busy, kernel-time
  savings convert to e2e at roughly half the arithmetic sum (P3-4:
  ~0.4 ms removed → +0.7 t/s of the possible ~1.6).
- **rocprofv3** grid-axis columns untrustworthy; kernel-trace inflates
  per-dispatch times 10–15%; use timestamps/durations.
- **PPL run-to-run noise** (~0.003 abs) from the fp16 atomic-add order
  — do not chase it as a regression signal.
