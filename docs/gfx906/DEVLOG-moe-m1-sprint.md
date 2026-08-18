# DEVLOG: MoE M=1 expert-kernel sprint (week of 2026-08-18)

Copyright Kevin Read <me@kevin-read.com>

Branch: `gfx906/moe-m1-sprint` (off `gfx906/main` @ 53f5a79429).
Plan: the 3-4 day set proposed from `moe-decode-roadmap.md` §A/§B:
A1a/c/g (cheap expert-kernel wins) → A1b (vectorized loads) → A2'/A3
(gate GEMV status / RMSNorm-gate fusion) → A4 (shared-expert GEMV
dispatch) → B1/B2 (GDN chunk, FA page A/B).

Stop rule (agreed before starting): if the day-1 cluster lands < 0.8 ms
combined, re-profile before continuing — the roadmap may be stale.

## Day 1 (2026-08-18): re-anchoring — the roadmap is stale

**Stop rule triggered immediately.** The roadmap's A1 premise ("w13 16.1 ms
+ w2 9.7 ms = 27.2 ms/step, 59% of the step") does not match the current
kernel:

- Per-kernel micro-bench (`bench_moe_gemm_gfx906.py`, random weights,
  M=1, bm=1): **w13 42.0 µs/call, w2 33.0 µs/call**. w13 traffic at M=1 is
  8 experts × 1 MB = 8 MB → **≈190 GB/s ≈ the DRAM floor** (the harness's
  `bw` column assumes the full 256-expert table and is misleading at M=1;
  8 MB/42 µs is the right number). w2: 8 × 0.5 MB / 33 µs ≈ 121 GB/s
  (76% of floor).
- ×40 layers ⇒ expert GEMMs ≈ **3.0 ms/step**, not 27.2 ms. The roadmap
  numbers predate the Phase 3 kernel work (NPT split, single-stage
  prefetch, NC2=2, BM bucketing).
- A1a's target ("gate gather ... 2D grid (bx, by), 'by = stride-1
  coalesced write' comment") does not exist anywhere in the tree — no
  `gate_k`/`gate_gather` kernel, no such comment. The BM=1 prologue loads
  the single token row 1D-coalesced into LDS. A1a is a phantom item.
- A2 (expert gate GEMV): the [1, 2048] shared_expert_gate was already
  converted in P3-1 (DEVLOG-moe-opt.md); the expert *router* [256, 2048]
  shape is already GEMV-eligible in `_llmm1_tiny_m` (m==256 branch).
  A2 needs a profiler check, not a kernel.

So the first deliverable is a fresh in-model kernel table to re-rank the
whole sprint. New probe: `benchmarks/kernels/gfx906/kernel_prof_probe.py` (the old
`fillprof_probe.py` was deleted in the docs consolidation 0defa92551 and
/tmp/bench was wiped by the reboot). Probe gotcha hit: the v1 engine runs
in a separate process by default → torch profiler sees zero kernels;
needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` (the probe now refuses to run
silently without it).

### Fresh in-model table (eager, 256 decode steps, GPU-busy 17.59 ms/step)

Top decode rows (prefill-only rows excluded; µs/step):

| µs/step | cnt/step | µs/call | kernel |
|---|---|---|---|
| 3,775 | 80.4 | 47.0 | dense_gemv<2, 2048> |
| 2,149 | 77.4 | 27.8 | moe_gemm_q4<1, 4> (experts, both passes) |
| 2,095 | 148.8 | 14.1 | LLGemm1<Half, 4> (LLMM1) |
| 713 | 39.8 | 17.9 | topkGating<8, 256, ...> (MoE router) |
| 586 | 79.7 | 7.4 | fused_add_rms_norm |
| 493 | 9.9 | 49.7 | flash_attn_tile_q8<256,256,2,8> (FA decode) |
| 414 | 29.8 | 13.9 | GDN packed_decode recurrent |
| 382 | 79.7 | 4.8 | act_and_mul (silu-mul) |
| 390+331 | ~80 | ~5-10 | elementwise/casts (unattributed) |
| 372 | 38.9 | 9.6 | moe_align_block_size |
| 308 | 9.96 | 30.9 | FA gather_paged_kv_q8 |
| 303 | 49.8 | 6.1 | sigmoid |
| 269 | 59.0 | 4.6 | copyBuffer |

Key readings:

- The MoE expert kernel is **27.8 µs/call in-model** (55.6 µs/layer
  for w13+w2 combined) vs 42+33 µs in the cold micro-bench — in-model
  L2 state is friendlier. Expert GEMMs ≈ 2.15 ms/step, not 27.2 ms.
- The roadmap's A1 ("59 % of the step") is dead; the real top-3 buckets
  are dense_gemv 3.78 + experts 2.15 + LLMM1 2.09 = 8.0 ms/step (46 %).
- dense_gemv<2,2048> at 47 µs/call × 80/step is the single biggest
  bucket and the identity of its two per-layer shapes is being pinned
  down with a shape-counting spy (/tmp/shape_spy.py).
- Gap to llama.cpp parity (70.3 t/s = 14.22 ms/step) from the current
  67.39 t/s record (14.84 ms/step) is only **~0.6 ms/step** — the sprint
  needs to find ~0.6 ms, not 5.

### Re-ranked sprint (post-profiling)

1. **S1** — attribute + optimize the dense_gemv<2,2048> bucket
   (3.78 ms; 47 µs/call is far off the DRAM floor for any of the
   candidate shapes; e.g. router [256, 2048] = 1 MB should be ~10-15 µs).
2. **S2** — MoE router + topk pair (713 µs topkGating + the gate's share
   of S1) — A2/A3 territory, now measured not assumed.
3. **S3** — LLMM1 2.09 ms bucket: attribute shapes, check which are
   GEMV-eligible (the K<=8192 n==1 gate) and whether the K=2048 family
   should join the dense_gemv dispatch (A4 generalizes here).
4. **S4** — expert-kernel A1 leftovers that survived re-anchoring
   (A1c zp L1, w2 121 GB/s → floor if easy).
5. B1/B2 (GDN chunk, FA page) demoted to stretch — GDN is 414 µs/step
   total recurrent + conv (150), FA decode 493+308+134 ≈ 0.94 ms; both
   smaller than the old roadmap claimed.

Items S1-S3 are dispatch/kernel-local, bit-exact-verifiable, and each
has a micro-bench gate — the same risk profile as the original proposal,
just aimed at the real top buckets.

### Shape spy + GEMV calibration → final re-ranking (S2 is the play)

Shape spy (/tmp/shape_spy.py, monkey-patched op counters, 32-step
eager gen) attributes every per-layer linear of the 35B model. The
dense_gemv<2,2048> bucket = router [256,2048]×40 + GDN in_proj
[12288,2048]×30 + FA qkv [9216,2048]×10 + lm_head [248320,2048]×1;
the LLMM1 bucket = GDN/FA out_proj [2048,4096]×40 + shared gate_up
[1024,2048]×40 + shared down [2048,512]×40 + GDN in_proj_ba
[64,2048]×30.

Bandwidth correction: MI50 HBM ≈ 1 TB/s nominal, ~800 GB/s effective
floor (the K=17408 GEMV docstring: 178 MB in 223.8 µs = 795 GB/s =
"100.2 % of floor"). My 160 GB/s assumption was the consumer Vega
number — wrong by 5×. Recomputed, the big GEMV shapes are AT FLOOR:

| shape (m×k) | x/step | floor µs | best µs/call | cfg | verdict |
|---|---|---|---|---|---|
| 12288×2048 GDN in_proj | 30 | 63.1 | 60.9 | GEMV kc2048 | 1.0× — no lever |
| 248320×2048 lm_head | 1 | 1274.6 | 1127.4 | GEMV kc2048 | 0.9× — no lever |
| 9216×2048 FA qkv | 10 | 47.3 | 46.8 | GEMV kc2048 | 1.0× — no lever |
| 2048×4096 o_proj | 40 | 21.0 | 21.4 | LLMM1 rpb4 | 1.0× — no lever |
| 1024×2048 shared gate_up | 40 | 5.3 | 7.3 | LLMM1 rpb4 | 1.4× — small lever |
| 2048×512 shared down | 40 | 2.6 | 6.7 | LLMM1 rpb4 | 2.5× — lever |
| 256×2048 router | 40 | 1.3 | 4.6 | GEMV kc2048/r2 | 3.5× — lever |
| 64×2048 GDN in_proj_ba | 30 | 0.3 | 4.6 | GEMV kc2048/r2 | 14× but launch-bound, GEMV≈LLMM1 |

**The real top-2 non-floor costs (per step):**

1. **topkGating<8,256,4,16,64>: 17.9 µs × 40 = 713 µs/step** —
   top-8-of-256 logit selection, pure launch/shape overhead; a
   dedicated fused router+topk kernel (1 CTA, 256 threads, one global
   read of the 1 MB gate, in-CTA top-8) should land ~3-5 µs →
   **~550-600 µs/step = the whole 0.6 ms gap to llama.cpp parity.**
2. **MoE expert kernel at M=1: 64 CTAs total (8 rows × K/256), each
   running a 32-iter latency chain** — cold micro-bench w13 42 µs
   (190 GB/s of 800) vs in-model 27.8 µs/call; a better-parallelized
   M=1 expert GEMV (finer K-split / different partition) is worth
   ~0.5-1.0 ms/step. (This is the A1 cluster, re-diagnosed:
   occupancy, not load width.)

Secondary: shared down K=512 GEMV (~160 µs), shared gate_up (~80 µs),
router-alone (~130 µs, mostly subsumed by the fused kernel).

**Final sprint order:** (1) S2 fused router+topk — closes parity;
(2) S5 M=1 expert-kernel parallelism; (3) S3 shared-down K=512 GEMV;
(4) S4 shared gate_up; B1/B2 demoted to stretch.

## S2 — dedicated M=1 topk kernel (implementation)

**Generic topkGating semantics extracted** (topk_softmax_kernels.cu,
SCORING_SOFTMAX, no bias/padding): per-lane (VPT=8) `max()` (=fmaxf,
NaN-ignoring) → width-32 xor(16..1) butterfly; `expf(x - row_max)` with
sequential per-lane sum in expert order → width-32 xor(16..1) butterfly
sum; `p = e * (1.f/row_sum)`; `isnan||isinf → 0.f`; 8× (local argmax
strict `>` i-ascending, xor-butterfly argmax with lowest-expert-index
tie-break, blank winner to -10000.f, `selected_sum += winner_p` in k
order); renorm: `scale = 1.f/denom` (denom≤0→1.f), `out[k] *= scale`.

**Kernel** (`csrc/rocm/moe_topk_gfx906.cu`, as shipped): one 64-lane
CTA; lanes 32-63 exit immediately (`if (t >= LANES) return;`) and the
32 active lanes each own 8 consecutive experts (the generic's exact
TPR=32/VPT=8 partition, one 16B load each), reduced with the generic's
exact width-32 `VLLM_SHFL_XOR_SYNC_WIDTH` butterflies. (An earlier
draft ran all 64 lanes with 32 holding 8×-inf dummy values and LDS
scans for the cross-lane reductions; that variant was dropped — see
the table below — and this description previously described it.)

**Bit-equality argument** (finite inputs):
- max: exact; the width-32 xor(16,8,4,2,1) butterfly is the generic's
  own reduction, reproduced instruction-for-instruction.
- sum: the same xor(16,8,4,2,1) tree, round-for-round identical
  operands and order.
- argmax: strict total order (value desc, expert asc) → any
  reduction order gives the identical winner; blanking is local to the
  owner lane.
- per-lane pieces (expf, local sum order, recip multiply, NaN clamp,
  selected_sum k-order accumulation, renorm scale) are byte-identical
  operations on byte-identical operands.

**gfx906 ISA finding (measured)**: every `__shfl_xor` pattern — width 32
*or* 64 — lowers through the E32 extended-register file (width 32:
558 v_readlane + 357 v_writelane + 526 v_mov_b32_e32; width 64:
155 ds_bpermute + 183 v_mov_b32_e32). `v_perm_b32` only appears for
constant-lane shuffles. Three variants measured (torch-profiler GPU
self-time, M=1/E=256/k=8 fp16):

| variant | GPU self-time | notes |
|---|---|---|
| generic topkGating<8,256,4,16,64> | 17.3–17.5 µs | 4-warp block, row machinery |
| width-32 cut (this file, SHIPPED) | **12.5 µs** | 1 wave, 32 active lanes, generic's exact shuffles |
| width-64 + 32 dummy -inf lanes | (not measured — rejected) | 2× per-lane work; shuffles no cheaper (ds_bpermute) |
| LDS scans + forced s_barrier | 35.5 µs | 22 barriers + 8×32 ds_read2 scan; 2.8× SLOWER |

So the shuffle path wins despite the ugly lowering — one wave with no
row machinery is the real win over the generic. The LDS variant was
verified correct (3000-trial stress harness `benchmarks/kernels/gfx906/harness/topk_harness.cu`,
bit-equal to a CPU reference incl. tie/sparse cases) but is dead weight;
keep the harness for future top-k experiments. Side finding: the backend
elides `__syncthreads()` for single-wave CTAs (per-wave LDS FIFO
ordering); the stress harness passed with elision, which is why the
forced-barrier cost was isolated as the suspect in the 35.5 µs number.

**Dispatch** (fused_topk_router.py): `VLLM_GFX906_TOPK_M1` env (default
OFF — see the S2 result section below), gates = on_gfx906 + M==1 +
E==256 + k==8 + fp16 gating + int32 outputs + contiguous + softmax +
no bias + no padding.

**Build gotcha (this week)**: the in-tree `.deps/triton_kernels-subbuild`
is root-owned (docker-era) and any CMake reconfigure dies writing its
CMakeLists there. Workaround already in running.md (`FETCHCONTENT_BASE_DIR=/tmp/vllm-deps`);
added `TRITON_KERNELS_SRC_DIR=$PWD/.deps/triton_kernels-src/python/triton_kernels/triton_kernels`
to skip the ROCm/triton git clone after reboots wipe /tmp.

## S2 — result: mode-dependent, shipped default-OFF

Bit-equality gates: pytest `test_gfx906_m1_topk_bit_equal_to_generic`
4/4 (random×6 + all-equal + all-zero + 64-way tie, renorm on/off),
micro-bench bit-equal PASS.

Serving A/B (pp=2048 tg=256, 4 samples each, same day, this build):

| regime | generic (TOPK=0) | new kernel (TOPK=1) | Δ |
|---|---|---|---|
| isolated GPU self-time | 17.3 µs/call | 12.5 µs/call (64t) | −4.8 ✓ |
| eager serving | 23.98 t/s | 24.09 t/s | **+0.11** ✓ |
| CUDA-graph serving | 66.31 t/s | 65.32 (64t) / 65.38 (256t) | **−0.95** ✗ |

The same kernel gains in eager (CPU-gapped pipeline) and loses in
graph replay (gapless pipeline). Thread count (64 vs 256) does not
change either number. The in-model profiler table is NOT reliable for
this micro-kernel: the 256t build read 19.9–20.1 µs/call in-model vs
7.2 µs solo vs 1.8 µs interleaved-solo in the torch profiler — the
activity records for small kernels depend on surrounding pipeline
state. Wall-clock A/B is the only trustworthy gate for µs-scale
kernels; treat per-kernel tables with suspicion below ~10 µs.

Hypothesis (unverified): the width-32 shuffle lowering's E32-file
footprint is cheap when CPU launch gaps let the wave set up lazily,
but is exposed in graph replay's back-to-back dispatch. The LDS
variant's true graph-mode number is unknown (its 35.5 µs came from the
unreliable in-model profiler).

**Decision: `VLLM_GFX906_TOPK_M1` defaults to OFF** (opt-in for
eager-only use). Kernel, op, binding, test, and bench ship as the
experimental record; the generic topkGating remains the default path.
The 713 µs/step topk budget stays open: a real fix needs a design
that is fast in the gapless regime (candidate: fuse topk into the
router GEMV epilogue so there is no separate kernel at all — moved to
the roadmap as S2').

## S5 — M=1 expert kernel re-tile: gemm2 shipped default-OFF

### Design (V2-B, lane-based columns)

`moe_gemm_q4_v2_kernel_gfx906<THREADS,NPT,SLICE>` in
`csrc/rocm/moe_q_gemm_gfx906.cu`: one 512-thread CTA (8 waves) covers
`64*NPT` output columns; columns are **lane-based** (`n = offset_n +
tl*NPT`, all waves see the same column set) so each wave can own a
disjoint K-slice (`offset_k = w*size_k/8`); per-wave fp32 partials
`[wave][64][NPT]` reduce through LDS (two `__syncthreads`), then
**only wave 0 runs the epilogue** — direct store (gemm1) or packed
CAS into the pre-zeroed token row (gemm2, all 8 slot x-blocks share
the row).

Design bug caught in harness: with lane-based columns all 8 waves hold
the *same* reduced value for the *same* cells; if every wave runs the
CAS epilogue the value is added 8× (measured exactly 8.02× in the
single-slot case). Direct stores hide it (idempotent) — which is why
the gemm1 path looked correct while gemm2 blew up. Wave-0-only
epilogue fixes both.

### Harness bugs found while validating (all in benchmarks/kernels/gfx906/harness/moe_m1_harness.cu)

1. **Weight load**: `uint2 v = *(const uint2*)(b_ptr + j*size_n)` only
   loads 2 of the 4 NPT=4 columns — `b_w[j][2..3]` were stale. Needs
   `uint4`. Symptom: r[2]/r[3] (upper half2 of each 4-col group) off by
   ~0.8–1.2 vs CPU ref in every mode.
2. **Direct store dropped r[2..3]** for NPT=4 (only one half2 stored).
   The "gemm1 512t4col direct-store: max err 0.2511" pass was a
   **stale-buffer artifact**: col2/3 were left over from the
   current-kernel run on the same buffer. Fixed the store, re-validated.
3. Test bookkeeping: an npp sweep left `d_npp=8` for later
   "single-slot" runs (compared 8-slot output to a 1-slot reference →
   deterministic 14.6 "error"); one test block launched the current
   kernel with a 2-D grid (missing grid.z → partial K sum → 12.1).

### Machine fact: intra-wave CAS contention is pathological

Standalone probes with ≥2 lanes of one wavefront CASing the *same*
address show lost updates (deterministic saturation at 2^11 updates for
32- and 64-bit CAS) and, with several cells,
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` queue aborts on ROCm 7.14
gfx906. Hardware `atomicAdd`, plain stores, single-thread CAS loops,
and the production regime (each lane a distinct cell; ≤16-CTA fan-in
per cell) are all fine — the greedy probe matches baseline to the byte
throughout. **Rule: never let multiple lanes of a wavefront CAS the
same address; production/V2 kernels keep per-lane distinct cells.**
(Also: `global atomic_cmpswap_x2` retry logic in the ISA updates
`old := packed+old` rather than the CAS-returned value; harmless in the
production contention regime, avoid relying on it under heavy
same-address spin.)

### Standalone A/B (harness, launch regime, M=1, 35B-A3B shapes)

| K (gemm) | config | µs/launch | vs current |
|---|---|---|---|
| 2048 (gemm1) | current <1,4> | 32.00 | — |
| 2048 | v2 <512,2,256> | 27.11 | 1.18× |
| 2048 | v2 <512,4,256> | 28.14 | 1.14× |
| 512 (gemm2) | current <1,4> | 21.36 | — |
| 512 | **v2 <512,4,256>** | **10.80** | **1.98×** |
| 512 | v2 <512,2,256> | 11.39 | 1.88× |

Correctness: current vs cpu-ref 0.25 (dequant noise); v2 vs current
0.02 (gemm1) / 0.65 (gemm2, fp16 CAS-chain noise); single-slot npp=1
0.21/0.10; **production gemm2 convention** (top_k=1, per-slot a-rows,
8× CAS into token row) all three kernels agree within 0.29/0.73 abs.

### Production integration + in-model results

Dispatch in `dispatch_moe_gemm_q4`: `block_size_m == 1`,
`output_topk > 0` (gemm2 only) and `size_m == output_topk` (gemm2
a=[EM,N2], EM==topk ⟺ M==1), behind `VLLM_GFX906_MOE_M1` (default
**OFF**). The launcher guard requires `size_n%256==0`, `size_k%256==0`,
`size_k<=2048` and `groupsize%32==0`: the v2 kernel consumes 32
k-elements per iteration per wave and refreshes scale/zeros only at
32-aligned group boundaries, so a size_k not divisible by 256 would
give slice_k<32 (reads past end_k) and a non-32-multiple groupsize
would skip group refreshes (review R1; the earlier `size_k%64==0`
guard admitted e.g. size_k=64). Note the env flag only affects
launches dispatched after the flip — it has no effect on an
already-captured CUDA graph. Caller zeroing unchanged (gemm1 zeroing
now redundant but harmless; gemm2 zeroing still required for CAS).

In-model (kernel_prof_probe, 256 decode steps, µs/call rows —
direction only): gemm2 26.8 → 22.3 (−17%); gemm1 v2 <512,2,256>
27.5 vs 26.8 (neutral, standalone 1.18× did not transfer — kept the
established <1,4> for gemm1).

Wall-clock A/B (pp=2048 tg=256, 4 samples, same day, this build):

| regime | OFF | ON (gemm2 v2) | Δ |
|---|---|---|---|
| CUDA-graph serving | 66.43 t/s | **67.03 t/s** | **+0.60 (+0.9%)** |
| eager serving | 23.50 t/s | **23.96 t/s** | **+0.46 (+2.0%)** |
| greedy (12×128 tokens) | `868ad09e…` | `868ad09e…` | identical |

Both-V2 (gemm1+gemm2) variant: graph 66.81 / eager 24.00 — worse or
equal in both, confirming the gemm1 re-tile is not worth keeping.

**Decision: ship gemm2-only M=1 re-tile, `VLLM_GFX906_MOE_M1` default
OFF** (opt-in; positive and token-identical in both serving modes).
The gemm1 budget (≈1070 µs/step in the re-anchor table) stays open:
its re-tile premise (standalone 1.18×) does not transfer in-model;
next idea is a 256-thread 4-col tile or folding gemm1 into the
activation epilogue — roadmap C2 follow-up.

## S3 — shared-expert down_proj GEMV (shipped default-ON)

The re-anchor table flagged shared down [2048,512] x40/step at 2.5x the
HBM floor (6.7 us LLMM1 rpb4 vs 2.6 us) and shared gate_up [1024,2048]
at 1.4x. The dense GEMV kernel (csrc/rocm/dense_gemv_gfx906.cu) already
supported both shapes (K%8==0, kchunk 512|1024|2048|4096); S3 was a
dispatch extension plus one RPT rule entry.

**Micro-bench (benchmarks/kernels/gfx906/bench_dense_gemv_gfx906.py):**

| shape | LLMM1 rpb4 | best GEMV | verdict |
|---|---|---|---|
| shared down [2048,512] | 6.7-7.7 us | **kc512/r2 5.6-5.7 us** | ship (−15..25%) |
| shared gate_up [1024,2048] | 7.3 us | kc2048/r4 8.0 us (r2: 46.4!) | **no lever — stays LLMM1** |

The gate_up lever from the re-anchor table does not exist: the GEMV is
8% slower at RPT=4 (default rule) and 6x slower at RPT=2. LLMM1 rpb4
is the right kernel there. The down-shape win is real but the 2.5x-floor
gap does not fully materialize: the winning GEMV is itself 2.2x floor.

**Integration:**
- `_llmm1_tiny_m` (vllm/model_executor/layers/utils.py): K=512, m=2048
  → `dense_gemv_gfx906(w, x, 512)`, under the existing
  VLLM_GFX906_DENSE_GEMV gate plus its own kill switch
  `VLLM_GFX906_DOWN_GEMV=0`.
- kernel default RPT rule: kchunk==512 && N==2048 → RPT=2 (measured;
  RPT=4 is the old fallback at 8.0-8.2 us).

**Gates (2026-08-18; NOTE: thermal noise today — fans reported not
ideal, serving numbers directional only):**
- standalone: GEMV kc512/auto(=r2) 5.64 us vs LLMM1 6.71-7.68 us
  (the first-in-process "auto" sample reads ~6.1 us: GPU warmup bias,
  not a config difference — re-run after warmup matches r2 exactly).
- in-model (kernel_prof_probe): dense_gemv_kernel<2,512> 39.7/step at
  8.4 us/call (L2-cold inflation vs 5.64 solo, same pattern as S5);
  GPU-busy 17593 → 17523 us/step (−70 us).
- PPL (benchmarks/kernels/gfx906/ppl_probe.py, recreated 12-prompt set, 359 tokens,
  0 top-20 misses — absolute values NOT comparable to the 6.69-era
  probe whose prompt set was lost in the /tmp wipe): OFF 16.009±0.01
  vs ON 15.993±0.01 = −0.16%, well inside the 2% bar.
- greedy probe 12x128: **token-identical to baseline** (868ad09e...)
  with the GEMV on — the standalone 0.031 maxdiff stays below the
  argmax margin in-model.
- serving graph (BENCH_EAGER=0, 2 runs x 4 samples each):
  65.88/65.87 (OFF) vs 66.07/65.98 (ON) = +0.19/+0.11 t/s, all four
  runs positive; expect a clean re-measure when thermals cooperate
  (predicted ≈ +0.3 t/s from the −40 us/step micro-bench delta).

**Decision: default ON — PROVISIONAL** (kill switch
VLLM_GFX906_DOWN_GEMV=0). Expected in-model gain ~40 us/step (≈0.25%
of the 17.5 ms GPU-busy step) — the smallest lever remaining in the
re-anchored table; the big open buckets are now the expert gemm1
(≈1070 us/step, C2 follow-up) and the topkGating 713 us/step (S2'
router-GEMV fusion).

Provisional because (post-sprint code review, R4): this is the
branch's only default-ON numerics change, it is not bit-equal to
LLMM1, and the decision rests on a serving A/B the same section flags
as thermally noisy. The uncontaminated evidence is: micro-bench
−15..25% on the shape, in-model GPU-busy −70 us/step, greedy
token-identical, PPL −0.16%, and four thermally-noisy serving samples
that are all positive. Reopen conditions: a clean serving A/B (fans
confirmed, both directions, 2 runs each) that goes negative, or a
greedy-margin analysis showing the 0.031 standalone maxdiff is not
comfortably below the smallest in-model winning-logit margin. The
uniform rule going forward (also applied retroactively here): a
default-ON numerics change requires a clean wall-clock A/B — a
per-kernel number may motivate, but not justify, flipping a default.

## Post-sprint code review fixes (2026-08-19)

Two independent reviews of the branch (moe-m1-sprint-code-rev-glm5.md,
merged with moe-m1-sprint-code-rev-ds4.md) produced six findings; the
actionable ones were fixed in five commits:

- **R1+R2** (`979e72c925`): the v2 M=1 launcher guard admitted shapes
  the kernel cannot tile (`size_k%64==0` allowed `slice_k<32` → reads
  past `end_k`; no `groupsize%32==0` check → missed group refreshes).
  Now `size_n%256==0 && size_k%256==0 && size_k<=2048 &&
  groupsize%32==0`. Also dropped the dead gemm1 branch of the M=1
  dispatch (computed for `output_topk==0` but the launch was gated on
  `output_topk>0` only) and documented that the env flag does not
  affect captured CUDA graphs.
- **R5** (`b7ec0306ee`): this devlog's S2 section described an earlier
  draft (-inf dummy lanes + LDS scans); rewritten to describe the
  shipped width-32 shuffle kernel. The dispatch line wrongly claimed
  the env default was ON.
- **R4** (`e06f484c0e`): S3's default-ON decision marked provisional
  with reopen conditions (its serving A/B was thermally flagged; see
  the S3 section).
- **R3/R6** (`b3e7139aaa`): `tmp_tp_probe/` harnesses moved to
  `benchmarks/kernels/gfx906/harness/`, probe scripts to
  `benchmarks/kernels/gfx906/`; path references updated.
- **R5-hardening** (`8e7935edd4`): S2 dispatch docstring pins the
  bit-equality assumptions (softmax/no-bias/no-padding/full-range/
  rsf=1), Python gate now also checks `gating_output.shape[0]==1`,
  and a new end-to-end dispatch test covers `vllm_topk_softmax` with
  the env set (M=1 bit-equal, M>1 guarded off).

Not done (deliberately): the S5 `diff < 0.3·max+0.05` test gate stays
as-is — S5 is default-OFF and the review agreed the stronger
higher-precision accumulation check should only be required before any
future default-ON flip. The clean S3 serving re-A/B (fans confirmed)
is deferred until thermals cooperate; the provisional flag in the S3
section is the record of that.

Verification: incremental rebuild + full
`tests/kernels/moe/test_gfx906_moe_gemm.py` (25 passed) and
`tests/kernels/moe/test_fused_topk.py -k gfx906` (6 passed).
