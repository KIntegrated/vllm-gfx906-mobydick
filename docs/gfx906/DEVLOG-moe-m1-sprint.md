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
whole sprint. New probe: `docs/gfx906/kernel_prof_probe.py` (the old
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
