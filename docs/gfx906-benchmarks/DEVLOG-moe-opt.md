# Devlog — Qwen 3.5 quantized MoE decode/prefill on gfx906

Branch: `gfx906/moe-opt` (from `gfx906/fa-integration`, i.e. fork main + custom FA).

## Problem statement

End-to-end bench (`_bench_gfx906.py`, pp=2048/tg=256, MI60 32 GB, eager):

| Version | dense 9B-AWQ tok/s | MoE 35B-A3B-AWQ tok/s |
|---------|--------------------|-----------------------|
| 0.23.0  | 27.47              | (unsupported)         |
| 0.26.0  | 28.03              | 12.16                 |
| main    | 32.31              | **3.49 (−71%)**       |

MoE is abysmal in both, and main regressed it further. Both prefill and decode
need work; the MoE path is the prime suspect for the regression.

## Todos

- [x] Start branch `gfx906/moe-opt`
- [x] Capture vLLM startup logs for MoE on 0.26 vs main — backend identified
      (0.26: legacy monolithic Triton; main: modular pipeline + TritonWNA16Experts,
      both running `fused_moe_kernel_gptq_awq`)
- [x] Profile decode + prefill on main: MoE GEMM = 91.2% of GPU time
      (3.495 ms/call at decode) — the whole regression is one kernel
- [x] Port RDNA3 exllama-style MoE kernel to gfx906 (`moe_q_gemm_gfx906.cu`),
      wire through modular pipeline + oracle; standalone correctness test ALL PASS
      (both source layouts); micro-bench 35–125× vs Triton
- [x] Full model: **3.49 → 18.79 tok/s** (+54% vs 0.26); prefill 4.7×, decode 5.3×;
      generation sanity check passed; re-profiled (MoE now 15% of GPU)
- [x] Update README.md + this devlog
- [ ] Phase 2 (optional, see candidates below): prefill block_m tuning,
      decode latency (smaller A tile path / 128-thread blocks), LLGemm1 study

## Notes / findings

### Model facts (QuantTrio/Qwen3.5-35B-A3B-AWQ)

- 40 layers: 30 linear-attn + 10 full-attn (every 4th). hidden=2048, head_dim=256.
- **256 routed experts/layer, top-k=8**, moe_intermediate=512; shared expert (fp16)
  + layer 0 + attn + linear_attn are NOT quantized (`modules_to_not_convert`).
- AWQ int4, group_size=128, zero_point=true.
- Active routed params/token ≈ 8 × 3.15M = 25.2M → ~12.7 MB int4+scale per
  layer-token → **~507 MB/token over 40 layers**.
- Roofline at ~700 GB/s effective HBM: **>1000 tok/s decode**. Measured 12 (0.26)
  / 3.5 (main) tok/s → ~100x off roofline. This is a kernel problem, not a
  bandwidth wall.

### Dispatch path for AWQ MoE on ROCm (both versions)

`AutoAWQConfig.get_quant_method(RoutedExperts)` → `check_moe_marlin_supports_layer`
→ **always False on ROCm** (`if current_platform.is_rocm(): return False` in
marlin_utils.py) → warning "Layer ... not supported by AutoAWQMoEMarlin. Falling
back to Moe WNA16 kernels" (seen 39× = all quantized MoE layers) →
`MoeWNA16Config(...).get_quant_method` → `MoeWNA16Method`.

### What each version actually runs

- **0.26**: `MoeWNA16Method.apply` → legacy monolithic `fused_experts` Triton
  kernel (one W4A16 dequant-in-kernel pass; align+gemm1+act+gemm2+sum fused-ish).
  Weights stay int4 packed (22.4 GiB total). = 12.16 tok/s.
- **main**: `MoeWNA16Method.apply` → new modular `FusedMoEKernel`:
  `MoEPrepareAndFinalizeNoDPEPModular` + `TritonWNA16Experts` (separate
  align / gemm1 / silu-mul / gemm2 / sum kernels) = **3.49 tok/s** (−71%).
  (Oracle auto-select: FLASHINFER_TRTLLM out (zp), MARLIN/BATCHED_MARLIN out
  (MoeWNA16 layout), TRITON in → TritonWNA16Experts; HUMMING out (CUDA-only pkg).)
- **Dense AWQ on gfx906** (for contrast): `AutoAWQLinearMethod.apply` has a
  dedicated gfx906 path using custom exllama-derived `ops.gptq_gemm`
  (csrc/libtorch_stable/quantization/gptq/q_gemm.cu, "Optimized for GFX906")
  + `gptq_shuffle_awq_qweight` repack. This fast path has **no MoE equivalent**.

### Hypotheses (to confirm with profiling)

1. Triton W4A16 dequant GEMM codegen is poor on gfx906 (wavefront=64; config
   table `try_get_optimal_moe_config` is CUDA-tuned).
2. 256 experts × tiny per-expert GEMMs: launch overhead + CTA early-exit cost
   in eager mode (bench runs enforce_eager=True, no cudagraphs).
3. main's modular pipeline adds kernel launches + HBM round-trips vs 0.26's
   monolithic kernel → the −71% regression.
4. Possibly: int4 unpack/zp math inside Triton kernel is slow on gfx906 ISA.

### PROFILING (main image, torch profiler in-process, pp=512 tg=64) — CONFIRMED

`fused_moe_kernel_gptq_awq.kd`: **17.45s of 19.13s GPU time = 91.2%**, 4992 calls,
**3.495 ms/call avg**. That's the whole game. Rest: attention 0.8%, GDN linear-
attn 0.3%, topk_softmax 0.24%, moe_align 0.18%, silu_and_mul 0.16%, aiter
LLMM1 (shared-expert-ish) ~2%, aten::mm ~0.7%.

Per decode step: ~80 WNA16 GEMM calls × 3.5ms ≈ 280ms → matches 2.5–3.5 tok/s.
Weight traffic per w13 call ≈ 8 active experts × (1MB int4 + 40KB scales/zp)
≈ 8.3 MB → at 3.5ms that's ~2.4 GB/s ≈ **3% of achievable HBM bandwidth**.

### gfx906 ISA notes (from repo-root `gfx906-notes.md`)

- **No `v_mfma*` on gfx906** (Vega 20, no matrix cores) → Triton `tl.dot`
  lowers to scalar fp16 FMA with poor codegen; exllama's manual
  `__ockl_fdot2` (→ `v_dot2_f32_f16`) is the right tool. dot4c/dot8c NOT
  available (only dot4/dot8 forms).
- DPP (`v_mov_b32_dpp`) ~2x faster than LDS for row-local shuffles;
  `ds_bpermute_b32` for arbitrary in-wave exchange.
- **b128 LDS ops fastest** (9.5–11 TB/s measured) vs b32 (~2–4 TB/s);
  prefer `global_load_dwordx4` (16B) for contiguous packed data.
- LDS bank padding: +1 vec4/row for column-style access (ld=32 → 1865 GB/s,
  ld=33 → 3974 GB/s). Not needed for pure broadcast reads.
- Scheduling: issue multiple independent loads before first use; avoid
  immediate waits after each load (compiler emits staged s_waitcnt).

### Plan / task list

A. Custom HIP grouped GEMM for W4A16 MoE on gfx906 (exllama-style, like the
   dense `gptq_gemm` fast path): 16B vectorized weight loads, scales/zp in
   registers (group=128 → update every 32 k-steps of 8), A staged in LDS,
   fp16 dots via `__ockl_fdot2`. Target decode w13 call ~50–100µs (vs 3.5ms).
B. Wire it in as a new FusedMoEExperts subclass selected on gfx906 for W4A16
   (both MoeWNA16 fallback and AutoAWQ paths), keep Triton elsewhere.
C. Re-bench decode + prefill; then look at secondary costs (aten::mm ~2ms/step,
   LLMM1 ~7ms/step) if they dominate after A.

### KEY FINDING: fork already has this design for RDNA3

`csrc/rocm/moe_q_gemm_rdna3.cu` implements exactly the planned kernel for
gfx1100: sorted_token_ids/expert_ids + [E,K/8,N] shuffled weights +
[E,G,N] scales + [E,G,N/8] packed zeros, 256 threads × 4 N-cols,
BLOCK_KN=256 K-splits, pre-zeroed C + packed fp16 CAS atomic-add epilogue,
topk-weight mul + fused moe_sum (output_topk). Wired via
`compressed_tensors_moe_wna16_rdna3.py` (bypasses modular pipeline; **drops
shared experts — Qwen needs them, so we won't copy that part**).
It is arch-guarded to `__gfx1100__` (empty stub elsewhere) and uses
`__builtin_amdgcn_fdot2`; gfx906 dense path uses `__ockl_fdot2`.

**Revised plan: port that kernel to gfx906** (fp16-only, `__ockl_fdot2`,
runtime zero_offset for AWQ=0) + go through the standard modular pipeline
(FusedMoEExpertsModular + oracle backend) so shared experts work.

Task list (Phase 1):
1. [x] Verify layouts (see "Layout gotchas" below — the first attempt was
   wrong in two independent ways).
2. [x] `csrc/rocm/moe_q_gemm_gfx906.cu`: port of moe_q_gemm_rdna3.cu,
   fp16-only, `__ockl_fdot2`, zero_offset runtime arg, BLOCK_SIZE_M ∈
   {1,2,4,8,16}.
3. [x] Register `moe_gptq_gemm_gfx906` in csrc/rocm/torch_bindings.cpp
   (no arch guard; __ockl_fdot2 is portable) + wrapper/fake in _custom_ops.py.
4. [x] Weight repack in **torch/Python** at load time (per-expert loop):
   AWQ K-first int32 [E,K,N/8] → exllama-shuffled [E,K/8,N] int32; scales
   [E,G,N] and zeros [E,G,N/8] pass through unchanged (already in kernel
   layout). GPTQ-sym (no zp): synthesize z=8 (not selected in Phase 1).
5. [x] `Gfx906WNA16Experts(FusedMoEExpertsModular)` in
   fused_moe/experts/gfx906_w4a16_moe.py: moe_problem_size override (repacked
   layout), workspace_shapes ((M·topk,N),(M·topk,N/2),(M,K)),
   finalize_weight_and_reduce_impl=NoOP, apply = align + gemm1(zeroed ws) +
   activation + gemm2(fused sum into output, zeroed).
6. [x] Oracle int_wna16.py: new WNA16MoEBackend.GFX906_HIP first in priority
   on gfx906 when op exists; `_process_weights_gfx906` repack branch.
7. [x] Build via docker 7.14 recipe (RUNNING-IMAGES.md); standalone numerical
   test vs torch reference (ALL PASS, AWQ layout, both gemm passes, M=1..64);
   micro-bench vs Triton at Qwen shapes (below).
8. [x] Full model bench: **3.49 → 18.79 tok/s** (5.4×); generation sanity
   check passed; re-profiled (MoE now 15% of GPU, run CPU-launch-bound);
   README §4 + dev log updated.

### STATUS: kernel built + verified + 35–125x faster than Triton (2024-xx)

Micro-bench, Qwen shapes (E=256, topk=8, w13 N=1024/K=2048, w2 N=2048/K=512),
per-call µs, AWQ K-first int32 inputs for both:

| M | Triton w13 | gfx906 w13 | Triton w2 | gfx906 w2 | speedup (w13+w2) |
|---|-----------|-----------|----------|----------|------------------|
| 1   | 3667 µs | **35.5 µs** |  564 µs | **33.0 µs** | 62x |
| 8   | 25036 µs | **135 µs** | 1923 µs | **80 µs** | 125x |
| 32  | 67255 µs | **355 µs** | 4849 µs | **249 µs** | 119x |
| 128 | 169662 µs | **1894 µs** | 13475 µs | **965 µs** | 64x |
| 512 | 169723 µs | **3063 µs** | 13567 µs | **1618 µs** | 39x |

Decode w13: 3.5 ms → 35.5 µs (99x). Effective decode bandwidth ≈
8 active experts × ~1MB / 35.5µs ≈ 225 GB/s (latency-bound, not BW-bound;
MI60 HBM ≈ 500+ GB/s). Prefill M=512 is only ~90 GB/s (block_m=16 re-read
amplification + fp16 atomics) — Phase 2 target.

Correctness: `/tmp/bench/_test_gfx906_moe.py` ALL PASS (maxrel ≈ 2% = fp16
accum noise) for M ∈ {1,2,4,8,32,64}, block_m ∈ {1,4,16}, K=2048/512 and
K=768/N=1536 (partial grid), both gemm passes incl. fused topk-weight+reduce.

### Layout gotchas (learned the hard way)

1. **Exllama shuffle is even/odd interleaved, not natural order.**
   `dequant_4bit_8_fp16` masks 0x000F000F / 0x00F000F0 select bits
   [3:0]+[19:16] and [7:4]+[23:20] — i.e. the *lower* half holds k0,k2,k4,k6
   and the *upper* half k1,k3,k5,k7 (see `shuffle_4bit_8` in qdq_4.cuh /
   qdq_4_rdna3.cuh: "77775555 33331111 66664444 22220000"). My first repack
   packed the upper half in natural order (k1,k3,k5,k7) — kernel silently
   produced k4↔k1, k6↔k3 garbage. A self-consistent (wrong-on-both-sides)
   torch test passed until I extracted nibbles with the *kernel's* masks.
2. **AWQ MoE on main is K-first int32, not MoeWNA16 N-first uint8.**
   `AutoAWQMoEMethod.create_weights` allocates w13_qweight [E, K, N/8] int32
   (pack_factor=8; word m holds n=8m..8m+7), scales [E, G_K, N], qzeros
   [E, G_K, N/8] int32. The MoeWNA16 [E,N,K/2] uint8 layout is the legacy
   0.26 method only. Consequence: scales and zero points are **already in
   kernel layout** (zp word c holds columns 8c..8c+7 ascending — identical to
   AWQ's packing) → repack = weights only, per-expert loop to keep the
   [K,N] unpack temp small (~16 MB vs 2 GB vectorized).
3. **Nibble-interleave reshapes**: for z [E, N/2, G], stack(lo,hi, dim=2)
   then reshape(E,N,G) is correct; stacking at the last dim and reshaping
   merges (G,2) not (N/2,2). Same trap in my test reference — twice.
4. **Docker import shadowing**: the image has vllm installed in
   dist-packages; `pip install -e .` does not remove it. Scripts run from a
   cwd other than the repo pick up the stale copy → always set
   `-e PYTHONPATH=/workspace/vllm`. Also clear `vllm/**/__pycache__` after
   editing (stale pycs caused a phantom AttributeError).

### FULL MODEL RESULTS (real Qwen3.5-35B-A3B-AWQ, pp=2048/tg=256, eager)

| Path | tok/s |
|------|-------|
| 0.26 (legacy monolithic Triton) | 12.16 |
| main (TritonWNA16Experts) | 3.49 |
| **main + gfx906 kernel** | **18.79** |

- Backend selected: `Using 'GFX906_HIP' WNA16 MoE backend` +
  `Using Gfx906WNA16Experts`; load-time repack ≈ 65 s.
- Generation sanity check passed (greedy text/code coherent).
- Post-fix profile (pp=512/tg=64, in-process torch.profiler):
  - MoE GEMMs: **301 ms = 14.95% of GPU** (was 91.2%). Decode portion:
    ~4914 calls × 27.8 µs → ~2.1 ms/step for both gemms; prefill portion
    78 calls × 2.11 ms (M=512, block_m=16).
  - **Self CPU 4.3 s vs Self CUDA 2.0 s → CPU-launch-bound in eager mode.**
  - Top remaining per-decode-step GPU costs: aiter `LLGemm1` dense GEMMs
    ~7 ms/step (14554 calls × 31.7 µs), `kernel_paged_attention_2d`
    ~2.9 ms/step (10 full-attn layers × 290 µs), elementwise pile
    (copy_/mul/add/mean/pow/sigmoid/rsqrt) ~5–7 ms/step, topk_softmax
    0.7 ms, moe_align 0.5 ms.
- Prefill/decode split (tg=1 vs tg=256 runs):

| Path | prefill pp=2048 | decode |
|------|-----------------|--------|
| main pre-fix | ~450 tok/s (4.5 s) | 3.72 tok/s |
| + gfx906 kernel | **~2140 tok/s (0.95 s)** (4.7×) | **19.7 tok/s** (5.3×) |

- The −71% regression is fixed and we beat 0.26 by +54%; the remaining gap
  to roofline (~60+ tok/s decode) is eager-mode launch overhead +
  LLGemm1/elementwise, not the MoE kernel anymore.
- Pre-fix comparison runs: `git worktree add` of the parent commit; the
  worktree has no built `.so` artifacts — copy them from the image's
  `/usr/local/lib/python3.12/dist-packages/vllm/*.so` (built at the same base
  commit). Symlinking my *new* tree's .so into the old tree fails: the old
  Python expects ops (e.g. `torch.ops._C.silu_and_mul`) that moved in main.

### Layout gotcha #5 (backend gate)

- First full-model run still picked TRITON: my experts gate required
  `weight_key == kInt4StaticGroupScale`, but **on ROCm AWQ MoE goes through
  `MoeWNA16Config`/`MoeWNA16Method`** (Marlin check is always False on ROCm,
  so `AutoAWQConfig.get_quant_method(RoutedExperts)` falls back to
  `MoeWNA16Config.from_config(...).get_quant_method`) — that path passes the
  group-scale keys, while `AutoAWQMoEMethod` (Marlin-only) passes plain
  `kInt4Static`. The gate now accepts all four int4 keys. Note rejection
  reasons from `is_supported_config` are logged at DEBUG only.
- MoeWNA16 layout (N-first uint8) is what this model actually uses; the
  AutoAWQ K-first int32 path exists too and is handled by the same repack
  dispatcher. Both verified in the standalone test.

## PHASE 2 (plan: `plan-moe-phase2.md`, adversarially reviewed)

### P2-0 — diagnostics baseline (done)

**Hardware correction** (rocprofv3 agent info, this MI60): Simd_Count=240 →
**60 CUs** (the plan's SKU table said 64), Max_Waves_Per_Simd=10 (40 waves/CU),
LDS 64 KB/CU. Device id 90006 = gfx906.

**Micro-bench per bucket** (`/tmp/bench/_p2_diag.py`, % of 29.5 TFLOPS fp16 peak):

| M | bm | w13 µs | w13 %peak | w2 µs | w2 %peak |
|---|----|--------|-----------|-------|----------|
| 1   | 1  | 41.0   | 2.8%      | 29.5  | 1.9%     |
| 8   | 4  | 132.6  | 6.9%      | 79.0  | 5.8%     |
| 32  | 4  | 358.6  | 10.1%     | 233.5 | 7.8%     |
| 128 | 16 | 1880   | 7.7%      | 963   | 7.6%     |
| 512 | 16 | 3027   | **19.2%** | 1613  | 18.1%    |
| 2048| 16 | 9593   | **24.3%** | 5032  | 23.1%    |

(TFLOPS% rises with M and plateaus ~24%; note the dip at M=128 vs M=32 is
bm=4→16 transition noise/atomic overhead.)

**VGPR table** (llvm-readobj on the gfx906 code object extracted from
`_rocm_C.abi3.so`; host LLVM tools work fine — no docker needed):

| BM | vgpr | sgpr | spills | occ (blocks/CU, VGPR-limited) |
|----|------|------|--------|-------------------------------|
| 1  | 74   | 48   | 0      | 13                            |
| 2  | 93   | 48   | 0      | 11                            |
| 4  | 95   | 48   | 0      | 10                            |
| 8  | 129  | 48   | 0      | 7                             |
| 16 | 166  | 48   | 0      | 6                             |

**The plan's "occupancy=1 pinned by ~80 VGPRs" hypothesis is REFUTED**: no
spills at any BM, occupancy 6–13 blocks/CU (24–52 waves). Latency hiding is
present; outcome (C) "under-occupied" does not apply.

**Three-way bottleneck pass** (rocprofv3 --pmc on all buckets; gfx906 counter
names differ from CDNA: SQ_INSTS_LDS / SQ_WAIT_INST_LDS / SQ_THREAD_CYCLES_VALU
/ TCC_EA_RDREQ_32B — the CDNA-style names silently return 0):
- `SQ_LDS_BANK_CONFLICT = 0` in every bucket (the +8-half row pad works).
- `SQ_WAIT_INST_LDS / SQ_INSTS_LDS ≤ 0.6%` — LDS waits are a negligible
  fraction of LDS instructions.
- VALU pipe is dominant by instruction mix (~256 v_dot2 per thread per k-step
  at BM=16 vs ~48 dequant ALU); absolute utilization % is ambiguous across
  rocprofv3's multi-pass collection, so classified as **outcome (B) leaning**
  (dot-pipe-bound at large M).
- Per the plan's decision rule: (A)/(C) → b128 first; (B) → n_per_thread=2.
  Since bank conflicts and LDS waits are negligible but b32 A-staging still
  emits 4× the LDS instructions of b128, **do P2-1(a) b128 A-staging first**
  (surgical), re-measure, then decide on n_per_thread=2.

**P2-0 decision gate** (prunes P2-1):
- (a) b128 LDS A-staging: **in scope, first**.
- (b) BM=8 heuristic trial: in scope (129 VGPRs, no spills, occ 7 — viable).
- (c) n_per_thread=2: conditional on (a)+(b) result.
- (d) K-slice 512: deferred (new template + VGPR recheck; only if atomics
  prove a problem).
- (e) persistent-CTA: **out of scope for now** — occupancy is fine; revisit
  only if (a)–(c) stall below ~30% of peak.

**Launch-count baseline** (rocprofv3 --hip-trace, steady-state 50 ms windows ≈
one decode step at 19.7 tok/s): **~1,500 hipLaunchKernel dispatches per
decode step**. This is the denominator for P2-0b (−80), P2-4 (−40) and
P2-5 (−40).

**Tooling note**: fatbin bundle parsing by hand is a rabbit hole; the fast
path is `llvm-objcopy --dump-section .hip_fatbin=` + scan for \x7fELF headers
+ `llvm-readobj --notes` on the extracted gfx906 object — all with host LLVM
(`/opt/rocm-*/lib/llvm/bin`), no docker needed.

### Design notes (exllama dense kernel study)

- Dense `gemm_half_q_half_gptq_4bit_kernel`: block 256 threads, each thread
  owns 4 N columns; grid = (N/1024, M/m_count, K/256). A staged in LDS
  [m_count][256] half; weights streamed from HBM as int4 (16B) loads —
  requires exllama-shuffled layout w[qk=K/8][N] uint32 (8 k per uint32 per n)
  → one-time repack at load. Scales/zp in registers, updated only at group
  boundaries (group=128 = 4 inner steps). Dots via `__ockl_fdot2`.
- **Dense path dequants to fp16 + cuBLAS Hgemm when M > 32**
  (`MAX_Q_GEMM_ROWS=32`) — the quantized kernel is decode-only there too.
- Register pressure: accumulators = m_count × 4 floats/thread (4 n cols).
  m_count=16 → 64 acc VGPRs (fine, occ≥2); m_count=64 → 256 → impossible.
  ⇒ Phase 1: custom kernel for the DECODE regime (M·topk small, r_e ≤ ~16,
  m_count ≤ 16-32), keep Triton for prefill. Phase 2: prefill strategy
  (Triton retune and/or m=64 variant with n_per_thread=2 or LDS accum).
- vLLM contract to match (from TritonWNA16Experts):
  `moe_align_block_size(topk_ids, BLOCK_M, E, expert_map)` →
  sorted_token_ids [EM_padded] (values = orig row t·topk+iex, sentinel
  = num_tokens·topk), expert_ids [num_blocks] (-1 empty), num_tokens_post_padded.
  C is written SCATTERED to [M, topk, N] at the original slot (no padding in
  output) → workspace (M, topk, max(N,K)) suffices. GEMM2 multiplies by
  topk_weights (MUL_ROUTED_WEIGHT), then moe_sum over topk → [M, K].
- Weight layouts today (MoeWNA16): w13 [E, N=1024, K/2=1024] uint8 (K-packed
  bytes, N-strided rows — unfriendly for coalesced loads), scales [E, N,
  K/gs] fp16, qzeros [E, N/2, K/gs] uint8 (2 n per byte). Need repack to
  exllama layout + transposed scales/zp at load time.
- llama.cpp reference: mm_ids_helper (per-expert compact rows + expert_bounds)
  + Q8_1 activation quant + generic grouped mmq. Their dedup-scatter quantizes
  each token once when it feeds multiple experts.

### P2-0b — zero-fill launch elimination: **DESCOPED (design is racy)**

The plan proposed folding `w1_out.zero_()`/`output.zero_()` into the GEMM
kernel via `if (blockIdx.z == 0) clear-tile-before-CAS`. **This is a data
race**: K-slice blocks run concurrently with no ordering guarantee between
grid.z values — a z>0 block's atomic-add can land *before* the z=0 block's
clear-store, silently dropping that partial sum. The plan's "a plain store
that completes before any sibling's atomic" assumption is false on AMD (and
any GPU); the adversarial review missed it. Correctness tests would pass
thousands of times while remaining wrong (tiny race window).

Safe alternatives considered and rejected for now:
- **Spin-wait flag per tile** (z>0 spins until z=0 sets a gmem flag): relies
  on undocumented FIFO work-dispatch + fair wave-scheduling to avoid
  deadlock; hang risk in production MoE is unacceptable.
- **Monotonic per-element epoch CAS + acq_rel fences** (first-writer plain
  stores, others add; no restore needed since every element is touched every
  call): correct, but adds a 4B atomic per output element — likely *slower*
  than the memset at prefill sizes; only viable decode-gated. Complexity not
  justified for ~80/1500 launches (~2-3% of step time).
- **Zero-on-read in the consumer**: works for gemm1 only (activation reads
  every element), but `silu_and_mul` is a shared CUDA op — modifying it to
  zero its input is cross-cutting. Saves only 40 launches.

Decision: descope P2-0b; proceed to P2-1 (far higher upside). Launch-count
reduction stays in scope via P2-4 (fused topk+align, −40) and P2-5 (shared
expert, −40); the epoch-CAS protocol is a fallback if CPU launch overhead is
still material after those.

### P2-1 — prefill MoE tuning (in progress)

**Corrected roofline (measured, not datasheet):**
- `v_dot2_f32_f16` pipe peak on this MI60: **~20 TFLOPS** (standalone micro-kernel,
  ILP≥2, sclk=930 MHz). The plan's "29.5 TFLOPS fp16 peak" is unreachable with
  dot2 (it assumes a different instruction mix). At ILP=1 the same pipe only
  does ~8.5 TF → latency/dependency sensitive.
- Kernel at M=512 w13: 5.9 TF = **~29% of the practical dot ceiling** (was
  estimated as ~40% of a wrong peak).

**True occupancy table** (`hipOccupancyMaxActiveBlocksPerMultiprocessor` — the
P2-0 "occupancy" column was mislabeled; these are measured):

| variant | VGPR | spills | blocks/CU | waves/CU (per SIMD) |
|---------|------|--------|-----------|---------------------|
| <1,4>   | 74   | 0      | 3         | 12                  |
| <2,4>   | 93   | 0      | 2         | 8                   |
| <4,4>   | 95   | 0      | 2         | 8                   |
| <8,4>   | 129  | 0      | 1         | **4 (1/SIMD)**      |
| <16,4>  | 166  | 0      | 1         | **4 (1/SIMD)**      |
| <8,2>   | ~70  | 0      | 3         | 12                  |
| <16,2>  | 94   | 0      | 2         | 8 (2/SIMD)          |

BM≥8 with NPT=4 runs at **1 wave/SIMD**: zero inter-wave latency hiding.

**Experiments (M=512 w13 µs, baseline <16,4> = 3027):**
1. **b128 LDS reads in dot loop** (committed `9521993915`): 3027→3049 (noise),
   M=1 41.0→35.7. Neutral at prefill — expected, since P2-0 showed zero bank
   conflicts and <1% LDS waits. Kept (lower LDS instruction pressure).
2. **N_PER_THREAD=2** (template param + `VLLM_GFX906_MOE_NPT` override; default
   2 for BM≥8): M=512 3027→**2917 (+3.7%)**, M=128 1880→1811, M=2048
   9593→9326, w2 similar. Correctness 12/12 both NPT settings. Occupancy
   doubles at BM=16 (4→8 waves/CU). Modest gain → not purely latency-bound.
3. **Double-buffered weight prefetch** (two attempts): **FAILED — reverted.**
   - swap-based: <16,4> 256 VGPRs + 214 spills → M=512 3049→17790 µs (5.8x
     slower); even spill-free <4,4> (167 VGPR) got slower (occupancy halved).
   - unrolled-by-2 ping-pong (no swap, no runtime indexing): still 256 VGPRs +
     109 spills at <16,2>. The compiler keeps both chunk buffers live across
     the whole consume phase; hand structures can't beat its liveness analysis.
   - Conclusion: software prefetch is register-infeasible in this kernel shape
     at BM=16. (Would need NPT=2 + BM≤8 or a redesign.)

**ISA-level findings** (llvm-objdump on extracted code objects; host LLVM
tools, no docker needed):
- K-loop body (single-stage <16,2>): ~890 instructions, **512 v_dot2 = 57.5%**;
  scalar (s_*) instructions ~45% of the whole kernel — but S and V share one
  issue port per SIMD, so the kernel is **issue-bound**, not dot-pipe-bound:
  dots alone could only reach ~9.6 TF (48% of dot peak) at 100% issue rate.
- Each iteration: 4× `global_load_dwordx2` then `s_waitcnt vmcnt(3)` — the
  weight chunk's HBM latency is exposed every iteration (single-stage).
- **No PC sampling on gfx906** (`rocprofv3-avail list --spm`: "No spm counters
  supported" — CDNA2+ only). TCC_HIT/TCC_MISS counters work: M=512 w13 shows
  ~61% L2 hit rate on the (contaminated, spilling) build.
- Disassembly detour note: `hipcc -c` of a device-only TU puts the GPU object
  in a second embedded ELF (scan for \x7fELF); loads disassemble as
  `global_load_*`/`buffer_load_*`, not `v_load*`.

**Status/go-forward:** best config so far = single-stage + NPT=2 (BM≥8).
The remaining gap to the plan's <1.5 ms goal (~2x) is NOT reachable by
occupancy or prefetch tweaks in this kernel shape; candidates left:
- BM=8 + NPT=2 (+ maybe DBUF, which fits registers at BM=8) — next experiment.
- Plan option (e) persistent-CTA B-in-LDS redesign if that also stalls.
- Otherwise record the scalar-dot ceiling per plan option (f) and move to P2-2
  (cudagraph measurement), where the decode-side win likely is.

**4. BM=8 for prefill (heuristic `em > 512 -> BM=8`, NPT=2): BIG WIN.**
   <8,2> = 3 blocks/CU (12 waves, 3/SIMD) — better latency hiding than <16,2>:
   - M=128 w13: 1811 -> **933 us (-48%)**, w2 922 -> 634 (-31%)
   - M=512 w13: 2917 -> **2247 us (-23%)**, w2 1530 -> 1201 (-21%)
   - M=2048 w13: 9326 -> **7614 us (-18%)**, w2 4857 -> 3956 (-19%)
   - BM=8 LOSES below em~1024 (padding waste: M=8 139->181us, M=32 370->561)
     -> mid bucket stays BM=4. New heuristic: <=32 -> 1, <=512 -> 4, else 8.
   - Full model (pp=2048/tg=256): 18.79 -> 18.88 tok/s (+0.5%) — prefill is
     only ~7% of total bench time; decode (unchanged BM=1/4 paths) dominates.
   - Cumulative vs Phase-1 baseline at M=512 w13: 3027 -> 2247 us (-26%).

**P2-1 status:** the plan's <1.5 ms goal for M=512 is not reachable in this
kernel shape (issue-bound at ~57% dot share; double-buffering register-
infeasible at BM=16; further gains need the persistent-CTA redesign, option
(e) — deferred). Moving to P2-2 (cudagraph ceiling): decode is 93% of bench
time and CPU-launch-bound, so that is where the remaining end-to-end upside is.

### P2-2 — cudagraph ceiling measurement (done)

**Setup gotchas (hybrid GDN model + gfx906):**
- Default `FULL_AND_PIECEWISE` capture OOMs the KV cache (graphs up to size
  512). Use `cudagraph_mode=FULL_DECODE_ONLY` + `max_cudagraph_capture_size=8`.
- Cudagraph capture requires `max_num_seqs <= Mamba cache blocks` (91 here);
  set `max_num_seqs=32` for the single-request bench.
- Capture itself: 4 decode graphs (sizes 1,2,4,8) in 16 s, +0.11 GiB. Works
  fine on this model/platform — no capture fallback.

**Result (pp=2048/tg=256, single request):**

| mode | total tok/s | elapsed | decode-only est. |
|------|-------------|---------|------------------|
| eager (default) | 18.88 | 13.56 s | ~19.7 tok/s (50.8 ms/step) |
| **cudagraph FULL_DECODE_ONLY** | **41.51** | **6.17 s** | **~49 tok/s (~20.3 ms/step)** |

2.2x end-to-end from configuration alone — confirms the Phase-1 diagnosis
(eager decode is CPU-launch-bound: ~1500 dispatches/step). Prefill is
unchanged (chunked prefill stays eager in this mode).

**Go/no-go:** cudagraph works and decode is now GPU-bound at ~20 ms/step.
P2-3 (decode MoE latency) is worth doing only if MoE is a large share of the
20 ms — profile next. P2-4's launch-count argument largely evaporates in
graph mode (launches are captured); only its GPU-time argument remains.

**Bench script note:** `_bench_gfx906.py` gained `BENCH_EAGER=0` for serving
mode (untracked user file; not committed). Serving-mode numbers must stay
labeled separately from the §1 eager table.

### P2-3/P2-4/P2-5 — go/no-go from graph-mode decode profile (done)

torch profiler, FULL_DECODE_ONLY graphs, pp=512+tg=64. Self CUDA ≈ 22.6 ms/step
(matches the 20.3 ms/step derived from the bench). Per-step breakdown:

| component | ms/step | % | notes |
|-----------|---------|---|-------|
| aiter LLGemm1 (dense projections) | 7.2 | 32% | 224 calls × 32 µs — non-MoE, upstream aiter territory |
| aten::mm | 2.2 | 10% | 4 calls × 532 µs — large dense GEMMs |
| paged attention (FA) | 1.95 | 9% | 10 layers × 198 µs |
| **gfx906 MoE kernel** | **1.77** | **8%** | ~76 calls × 23 µs (BM=1 decode) |
| triton_matmul_kernel | 1.6 | 7% | 39 calls × 41 µs |
| GDN/mamba (chunk+recurrent+conv) | ~1.15 | 5% | hybrid-attention layers |
| aiter LLMM1 | 1.2 | 5% | |
| routing: topkGating + align + count_sort | ~1.0 | 4.5% | P2-4 target |
| shared expert (Triton fused_moe) | 0.55 | 2.5% | P2-5 target |
| elementwise/copy/zeros pile | ~2 | 9% | |

**Decisions:**
- **P2-3 (decode MoE latency): SKIP.** MoE is 8% of the step; even halving it
  is ~+4% e2e. The plan's own gate ("proceed only if decode is clearly below
  ceiling with GPU-bound residual") points the effort elsewhere: the residual
  is dominated by non-MoE kernels (LLGemm1/aten::mm/attn/GDN) that are a
  separate aiter/rocBLAS-on-gfx906 project, not MoE work.
- **P2-4 (fused topk+align): DEFERRED.** ~1 ms/step (4.5%) is the largest
  remaining MoE-adjacent item, but in graph mode only GPU time matters and
  it's the highest-correctness-risk item on the list. Not worth it now;
  revisit if someone takes up dense-GEMM tuning and wants the last few %.
- **P2-5 (shared expert Triton elimination): DEFERRED.** ~0.55 ms/step (2.5%).

### Phase 2 summary

| step | outcome |
|------|---------|
| P2-0 diagnostics | dot pipe peak ~20 TF measured; true occupancy table; issue-bound at large M; launch baseline ~1500/step |
| P2-0b zero-fill fusion | **descoped** — design is racy (documented) |
| P2-1a b128 LDS | neutral, kept |
| P2-1b NPT=2 | +3.7% prefill (occupancy 4→8 waves/CU at BM=16) |
| P2-1c BM=8 prefill heuristic | **−23% M=512 w13** (cumulative −26% vs Phase 1); double-buffering failed (spills) |
| P2-2 cudagraph ceiling | **41.5 tok/s serving mode (2.2× eager)**; capture works on hybrid GDN model with max_num_seqs≤91 |
| P2-3/4/5 | skip/defer — MoE is 8% of graph-mode step; residual is non-MoE kernels |

End-to-end: **3.49 → 18.88 tok/s eager (5.4×), 41.5 tok/s serving mode
(11.9×)**. Remaining per-step time is dominated by dense/GDN kernels outside
this project's MoE scope.
