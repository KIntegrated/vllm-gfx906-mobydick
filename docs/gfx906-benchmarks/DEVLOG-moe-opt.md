# Devlog — Qwen 3.5 quantized MoE decode/prefill on gfx906

Branch: `gfx906/moe-opt` (from `gfx906/fa-integration`, i.e. fork main + custom FA).

## Problem statement

End-to-end bench (`_bench_gfx906.py`, pp=2048/tg=256, MI50 32 GB, eager):

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

## PROBE PITFALLS — read before ANY monkeypatch/counter/measurement probe

> Learned the hard way twice (Phase 3, P3-0 Q3/Q4). vLLM V1 makes naive
> in-process probes silently wrong or loudly crashing. Checklist:

1. **The model does NOT live in your process by default.** V1 runs EngineCore
   in a separate **spawn**ed process (`SyncMPClient`). Wrapping
   `module.forward` / counting op calls in the main process then measures
   NOTHING — the probe exits 0 and prints all-zero tables. A silent lie.
   - Fix: run with `-e VLLM_ENABLE_V1_MULTIPROCESSING=0` (→ `InprocClient`,
     model in-process) BEFORE wrapping anything.
2. **Every probe script needs `if __name__ == "__main__": main()`** — spawn
   re-imports the main module in the child; without the guard you get
   `RuntimeError: ... bootstrapping phase` at `LLM(...)` construction.
3. **In-process mode changes memory accounting** (no separate process
   overhead, but same GPU). With this model's `max_model_len=262144`,
   `gpu_memory_utilization=0.9` OOMs on KV cache in-proc. Probes that only
   need short generations: set `max_model_len=4096` (or whatever the bench
   script uses) — never copy a config you haven't checked for seq-len.
4. **Never trust a probe that prints zeros.** After wrapping, assert at
   least one counter is non-zero (and matches an expected order of magnitude,
   e.g. ~40 router gemms/step here) before believing any result. Zero counts
   = broken instrumentation, not "the op never ran".
5. **Profiling ≠ probing.** torch-profiler traces (chrome JSON) still work in
   MP mode and give kernel→op correlation for EAGER runs; use them when you
   can't go in-process. In cudagraph mode the correlation breaks (kernels are
   launched from `hipGraphLaunch`, no per-op parent) — attribute via
   `hipGraphLaunch` boundaries instead of op nesting.
6. **Counter keys built in a loop need default-arg binding.**
   `def fwd(...): counts[key]...` where `key` is the loop variable makes
   EVERY wrapper increment the LAST key (late binding) — one bucket gets all
   calls, table looks "plausible" but is wrong. Use
   `def make_fwd(orig, _k=key): def fwd(*a, **kw): counts[_k]...`.
7. Cross-ref: layout gotchas #4 (spawn guard) below; P3-0 Q3/Q4 entries in
   this devlog show a probe done right (in-proc + guard + max_model_len fix
   + closure binding).

## Notes / findings

### Model facts (QuantTrio/Qwen3.5-35B-A3B-AWQ)

- 40 layers: 30 linear-attn + 10 full-attn (every 4th). hidden=2048, head_dim=256.
- **256 routed experts/layer, top-k=8**, moe_intermediate=512; shared expert (fp16)
  + layer 0 + attn + linear_attn are NOT quantized (`modules_to_not_convert`).
- AWQ int4, group_size=128, zero_point=true.
- Active routed params/token ≈ 8 × 3.15M = 25.2M → ~12.7 MB int4+scale per
  layer-token → **~507 MB/token over 40 layers**.
- Roofline at ≤1 TB/s theoretical HBM (MI50): **>1000 tok/s decode** — the
  "~700 GB/s" figure used here originally was wrong (see P2-0 hardware note).
  Measured 12 (0.26) / 3.5 (main) tok/s → ~100× off roofline. This is a
  kernel problem, not a bandwidth wall.

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

### gfx906 ISA notes (from [`docs/gfx906/`](../gfx906/))

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
MI50 peak HBM ≤1 TB/s theoretical). Prefill M=512 at ~5.6 TFLOPS — Phase 2
target. (Note: early estimates used wrong BW/peak figures; see P2-0 section.)

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

**Hardware correction** (rocprofv3 agent info): Simd_Count=240 → **60 CUs**
→ this is an **MI50**, not MI60 (MI60 = 64 CU). Max_Waves_Per_Simd=10
(40 waves/CU), LDS 64 KB/CU. Device id 90006 = gfx906. The devlog header's
"MI60" label was based on VRAM; the CU count proves MI50. The measured
`v_dot2_f32_f16` peak is ~20 TFLOPS (see P2-1), not the datasheet 26.8 or
29.5 TFLOPS. The %peak column below uses 29.5 (the figure available at P2-0
time); re-read against the corrected 20 TFLOPS peak — actual utilisation at
M=512 is ~30%, not ~19%.

**Micro-bench per bucket** (`/tmp/bench/_p2_diag.py`, %peak vs 29.5 TFLOPS
datasheet — divide by 0.68 for % of measured 20 TFLOPS practical peak):

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
- `v_dot2_f32_f16` pipe peak on this MI50: **~20 TFLOPS** (standalone micro-kernel,
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

### llama.cpp baseline attempt (blocked by VRAM)

- `nemotron-gfx906` build tree: gfx906 (`AMDGPU_TARGETS=gfx906:xnack-`),
  had no `llama-bench` — built it from the existing tree (link-only, fast).
- **Segfault root cause (host toolchain gotcha):** the binary resolved
  `libamdhip64.so.7` fine (identical backported file in /opt/rocm-6.4.3 and
  /opt/rocm-7.14) but `libhsa-runtime64.so.1` from **6.4.3 (v1.15)** via the
  ldconfig cache — ABI mismatch with the 7.13 HIP runtime → SIGSEGV in
  `hipStreamCreateWithFlags`. Fix: run with
  `LD_LIBRARY_PATH=/opt/rocm-7.14/lib` (hsa-runtime v1.21).
- **Model does not fit:** Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf is **36 GiB** >
  32 GB VRAM. Full offload OOMs (llama.cpp wants one ~35.9 GiB device buffer).
  Partial offload (`-ngl 24`) works but is CPU-bound: tg8 = 1.14 t/s — not a
  meaningful comparison vs vLLM's full-GPU 41.5 t/s serving mode.
- Verdict: the specified baseline file cannot produce a fair GPU number on
  this box. A fitting quant (Q4_K_S/M, ~20 GB) would need a ~20 GB download;
  or drop the llama.cpp reference point entirely.

### llama.cpp baseline (Q4, full offload) — done

Downloaded `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (20.8 GiB; Qwen3.6 uses the
same `qwen35moe` GGUF arch as 3.5, so no llama.cpp rebuild was needed).
The earlier Q5_K_XL attempt (36 GiB) could not fit the 32 GB card at all.

`llama-bench -ngl 99 -p 2048 -n 256 -r 2`, MI60, build 704485942:

| engine | prefill pp=2048 | decode |
|--------|-----------------|--------|
| llama.cpp (Q4_K_XL) | 806.5 t/s | **70.3 t/s** (14.2 ms/step) |
| vLLM eager (AWQ int4) | ~2140 t/s | 19.7 t/s (50.8 ms/step) |
| vLLM + cudagraph (AWQ int4) | ~2140 t/s | **~49 t/s** (20.3 ms/step) |

Caveats: different model generation (3.6 vs 3.5, same 34.66B A3B family and
GGUF arch) and quant (UD-Q4_K vs AWQ int4 — similar class). Good enough as a
reference point.

**Read-out:** vLLM wins prefill 2.6x; llama.cpp still wins decode 1.43x even
with cudagraphs (6 ms/step gap ≈ 30% of the step). Per the graph-mode
profile, our MoE kernel is only ~8% of the step — the gap lives in the
dense/GDN/attention paths (aiter LLGemm1 32%, aten::mm 10%, paged attn 9%,
triton_matmul 7%). That is exactly what a Phase 3 (non-MoE kernel tuning)
would target; llama.cpp's mmid/mmq + fused decode path is the thing to study.

## PHASE 3 (plan: `plan-decode-phase3.md`, v3 after external adversarial review)

Goal: close the ~6 ms/step decode gap vs llama.cpp in non-MoE kernels.
Primary metric: serving mode (`BENCH_EAGER=0`), decode tok/s + ms/step.

### P3-0 — diagnostics (done, 2026-08-15)

All Q0–Q6 answered. Evidence: shape-aware graph-mode torch profile
(`trace_cgsh.json.gz`, pp=512/tg=32), eager attribution trace
(`trace_cgeager.json.gz`), in-proc module probe (`_probe_layers.py`),
HBM BW microkernel, `rocprofv3 --pmc TCC_HIT,TCC_MISS` graph run, and a
llama.cpp `rocprofv3 --kernel-trace` decode run (Q4 model, -p 0 -n 256).

**Hardware label**: lspci says **MI50 32 GB** (Vega 20, 1002:66a1); device
string "AMD Instinct MI60 / MI50". 60 CUs confirmed. Using "MI50" henceforth.

**Q1 — HBM BW + L2**: achievable read BW = **798 GB/s** (8 GiB double4
stream, grid=2×CUs best). `TCC_HIT/(HIT+MISS)` at M=1 decode: LLGemm1
**14.5%**, triton_matmul 12.6%, paged attn 82% (KV is L2-resident), MoE
decode kernel 34%. → dense-gemm floors at 798 GB/s are real; L2 does not
rescue the small m=1 gemms.

**Q0/Q6 — reconciled per-step decode budget** (kernel-level sums over 31
graph steps; prefill excluded by construction):

| component | ms/step | calls/step | notes |
|-----------|---------|------------|-------|
| LLGemm1 dense projections (aiter) | **5.83** | 230 | incl. shared expert (80) + LM head |
| triton_matmul = **shared_expert_gate** [1,2048] | **1.63** | 40 | see Q3 below — the whole row is ONE tiny Linear per layer |
| paged attention (custom FA `attn_fwd`) | **1.94** | 10 | ~194 µs/layer, M=1, seq~500 |
| gfx906 MoE routed kernel (Phase 1/2) | 1.75 | ~78 | done |
| routing (topkGating+align+count_sort) | 1.06 | 79 | P2-4 deferred |
| GDN decode (recurrent + conv1d) | ~0.5 | 60 | faster than llama.cpp's — leave alone |
| elementwise/norm/copy pile | ~2.3 | ~300 | Fill 0.37, copyBuffer 0.32, rmsnorms ~0.6, act/sigmoid ~0.4, misc triton ~0.6 |
| fused_moe_kernel (Triton, residual) | 0.39 | ~2 | identity unclear, small — out of scope |
| other small kernels | ~2.2 | — | per/row ops around GDN+attn |
| **kernel total** | **~17.6** | | |
| inter-kernel gap (wall 20.3 − kernel 17.6) | **~2.7** | | in-graph launch/dep stalls; llama.cpp has ~0 (14.32 kernel = 14.2 wall) |

**Negative result (important):** the P2-3 table's "aten::mm 4×532 µs =
2.2 ms/step (11%)" row **does not exist in steady-state decode**. It was a
warmup/capture-region artifact of that profile window (the shape-aware
profile has zero M=1 `aten::mm` rows; all decode mms go through LLMM1 or the
Triton fallback). The §2 "scope TBD" row is void — 2.2 ms never existed per
step. Lesson: windows that include prefill/capture must be separated by
timestamp before dividing by step count.

**Layer composition (Q0):** **30 GDN + 10 full-attn** layers, confirmed two
independent ways: llama.cpp kernel counts (gated_delta_net 30/step,
flash_attn_tile 10/step) and the vLLM probe (in_proj 30/step, qkv 10/step).

**Q3 — Triton fallback identity (the big one):** in-proc probe of all 270
Linear modules (per decode step):

| /step | N×K | class | path | what it is |
|-------|------|-------|------|------------|
| 40 | 2048×4096 | RowParallel | LLMM1 | GDN out_proj + FA o_proj (all layers) |
| 40 | 2048×512 | RowParallel | LLMM1 | shared expert down |
| 40 | 1024×2048 | MergedColumn | LLMM1 | shared expert gate_up |
| 40 | 256×2048 | Replicated | LLMM1 | router |
| **40** | **1×2048** | **Replicated** | **TRITON** | **`shared_expert_gate` — scalar gate on shared-expert output** |
| 30 | 12288×2048 | MergedColumn | LLMM1 | GDN in_proj (qkvz) |
| 30 | 64×2048 | MergedColumn | LLMM1 | GDN A/B small proj |
| 10 | 9216×2048 | QKVParallel | LLMM1 | FA qkv |

Total 270/step = 230 LLGemm1 + 40 Triton — matches kernel counts exactly.
The **entire** 1.63 ms/step `triton_matmul` row is the `shared_expert_gate`
(`qwen3_next.py: Qwen3NextSparseMoeBlock.shared_expert_gate =
ReplicatedLinear(hidden→1, bias=False)`): a rank-1 dot product that costs
41 µs/call in Triton because `rocm_unquantized_gemm_impl` sends m=1 to the
Triton branch (LLMM1 requires `m % 4 == 0`). **Hypothesis "biased out_projs
fall back to Triton" was WRONG — no module has bias.** Fix surface: one
dispatch branch or a tiny GEMV; floor ~5 µs/call → **~1.4 ms/step
recoverable**. This is now P3-1, precisely scoped.

**Q4 — shared expert:** two plain fp16 Linears per layer (gate_up + down)
via LLMM1 — inside the dense surface, NOT MoE-side. The 2/step Triton
`fused_moe_kernel` residual is a separate small path (0.39 ms), left out of
scope.

**Q2 — llama.cpp decode reference (per step, ÷257 steps, 14.32 ms total):**

| component | llama.cpp ms/step | vLLM ms/step | read-out |
|-----------|-------------------|--------------|----------|
| dense gemms (Q8_0 weights!) | 3.85 (211 calls) | 5.83 LLGemm1 (fp16) + 1.2 LM head | llama.cpp's dense weights are Q8_0 = half our fp16 bytes — most of its lead here is quantization, as predicted |
| MoE experts (Q4/Q5_K gather-gemv, ~80 fused calls) | 2.57 | 1.75 kernel + 1.06 routing | roughly at parity; no big win available |
| shared expert (F16 gemvs) | ~1.0 (in f32-gemv row) | ~0.75 (in LLGemm1) | comparable |
| **attention** | **0.19** (flash_attn_tile 10×15.2 µs @ avg seq~128) | **1.94** (10×194 µs @ seq~500) | **~3–10x gap even after KV-length scaling — biggest kernel-level difference; P3-3 promoted to #2** |
| GDN recurrence+conv | ~2.0 | ~0.5 | vLLM already faster — do not touch |
| activation quant (Q8_1, 291 calls) | 1.31 | 0 | llama.cpp pays this tax; we don't |
| topk routing | 0.51 | 1.06 | comparable |
| norms/elementwise | ~3.5 | ~2.3 | comparable |

GGUF tensor map (via `~/.bin/gguf-parser --raw`): dense projections + LM
head = Q8_0; MoE experts = Q4_K/Q5_K (gate/up) + Q5_K/Q6_K (down); shared
expert + norms = F16. Note: this llama.cpp build reads **GGUF v3** with the
new type enum (STRING=8, ARRAY=9) — hand-rolled parsers using the old enum
silently misparse; use gguf-parser or the repo's own tooling.

llama.cpp arithmetic regime: Q8_1-quantized activations × quant weights
(mmq/mmvq). Its decode lead = weight bytes (Q8_0/Q4 vs our fp16/int4-dense)
+ attention kernel quality, NOT MoE. The int8-activation option stays a
named future candidate (out of Phase 3 scope per plan §4).

**Revised priorities (supersedes plan v3 §4 ordering):**
1. **P3-1: shared_expert_gate fix — ~1.4 ms/step.** One [1,2048] gemv ×40.
   Candidate fixes in order of invasiveness: (a) dispatch m<4 to torch
   F.linear (rocBLAS) and measure; (b) zero-pad weight to [4,2048] at load
   + `ops.LLMM1(..., 4)` + slice; (c) tiny custom GEMV. All low-risk;
   correctness = greedy-output diff.
2. **P3-3: paged attention decode — ~1.5–1.7 ms/step.** 194 µs/layer at M=1
   is latency/occupancy-bound (KV read ~0.5 MB, 82% L2 hit). Study the
   custom FA kernel's work-split for GQA kv_heads=2 vs llama.cpp's
   flash_attn_tile 256×256 KV-chunked design before writing anything.
3. **P3-2: remaining LLGemm1 surface — ~1–1.5 ms/step.** 5.83 ms vs ~4.6 ms
   floor @798 GB/s (L2 hit only 14.5%). Custom M=1 W16A16 kernel remains the
   primary development path (plan v3 decision); aiter knobs time-boxed.
4. Inter-kernel gap (~2.7 ms/step): attacked indirectly by cutting kernel
   count (P3-1 removes 40 launches/step). No direct lever yet.
5. P3-4 elementwise: deprioritized (we're already at/better than llama.cpp).

### P3-1 — `shared_expert_gate` tiny-m gemv fix (done, 2026-08-15)

**Micro-bench** (`/tmp/bench/_p31_memb.py`, n=1, k=2048 fp16):

| option | µs | notes |
|--------|-----|-------|
| Triton triton_matmul (before) | 42.8 | matches 41 µs in-model |
| torch F.linear (rocBLAS gemv) | 281.1 | rocBLAS skinny gemv is terrible — rejected |
| **LLMM1 + zero-pad to 4 rows** | **7.3** | chosen; 5.9× vs Triton |
| `(x*w).sum(-1)` | 12.6 | workable but worse than LLMM1 |

**Change**: `_llmm1_tiny_m()` in `vllm/model_executor/layers/utils.py` —
zero-pads weight rows to a multiple of 4, calls `ops.LLMM1(w, x, 4)`, slices
the output. Both LLMM1 dispatch sites now accept `(m % 4 == 0 or m < 4)`.
Generic (helps any tiny-m decode Linear), not model-specific.

**Correctness**: unit tests added to
`tests/model_executor/layers/test_rocm_unquantized_gemm.py` (mock dispatch
for m=1/2/3 + real-kernel m=1 path). NOTE: that file has 8 pre-existing
failures on the base commit in this env (mock/platform-setup dependent,
not caused by P3-1); passes go 1 → 5. Greedy A/B (`/tmp/bench/_p31_ab.py`,
3 prompts × 64 toks): prompts 0/1 token-identical; prompt 2 diverges at
~token 11 with both continuations fluent — fp16 reorder sensitivity of the
sigmoid gate (measured logit delta ~1e-3), accepted.

**Results** (pp=2048/tg=256, single request):

| mode | before | after | Δ |
|------|--------|-------|---|
| eager | 18.88 t/s | **19.49 t/s** | +3.2% |
| serving (cudagraph) | 41.51 t/s | **44.09 t/s** | +6.2% ≈ 1.6 ms/step |

Serving step time 24.3 → 22.7 ms. Cumulative from the original 3.49:
eager 5.6×, serving 12.6×. llama.cpp gap (serving): 1.70× → **1.59×**.

### Probe/measurement notes (this phase)

- `rocprofv3 --hip-trace` does NOT include kernel dispatches; use
  `--kernel-trace`. CSV output needs `-f csv -d <dir>` (writes to
  `<dir>/<host>-<nn>/`). `--pmc TCC_HIT,TCC_MISS --kernel-trace` together
  give per-dispatch L2 hit/miss joined with kernel names.
- llama.cpp host runs: `LD_LIBRARY_PATH=/opt/rocm-7.14/lib` (hsa-runtime ABI)
  AND it works under rocprofv3 fine.
- vLLM in-proc probes: see "PROBE PITFALLS" section above (spawn, __main__
  guard, max_model_len KV OOM, closure late-binding).
- Eager probe runs fell back to Triton paged attention ("Cannot use ROCm
  custom paged attention kernel") — config-dependent; do not take timings
  from probe runs, only call attribution.

### P3-3 — paged attention: the CUSTOM (Q8 FA) backend saga (2026-08-15)

**Starting point**: vLLM attention 1.94 ms/step (10 FA layers × 194 µs) vs
llama.cpp 0.19 ms — P3-3 was planned as "partition the Triton kernel".
While starting that, discovered the repo **already vendors a custom Q8
FlashAttention backend** (`vllm/gfx906_fa/` + `csrc/gfx906_fa/`, llama.cpp's
`flash_attn_tile_q8`, head_size 256 supported) integrated in `67d2a813a2`
as the gfx906 default — **but it was dead code at runtime**: the
`vllm.general_plugins` entry point is absent from the tree's stale
`vllm.egg-info` AND the image's dist-info, so `CUSTOM.is_overridden()` was
always False and everything fell to Triton. Nobody noticed because the
fallback is silent.

Fix: `vllm/platforms/rocm.py` `_get_backend_priorities` now registers the
plugin explicitly on gfx906 when not already registered (idempotent;
entry-point installs still win). Verified priorities become
`[CUSTOM, ROCM_ATTN, TRITON_ATTN, TURBOQUANT]`.

**Then the real bug hunt.** With the backend live, `GFX906_FA_LEGACY=0`
(the fast path: Q8 side-buffer + fused HIP gather) produced garbage from
the first token ('!!!!!...'), while LEGACY=1 and FUSED=0 worked. Isolation
ladder (each step verified before moving on):

1. `reshape_and_cache_q8` vs `quantize_q8_0` on same data: **byte-identical**
   (D=256 fine).
2. Synthetic gather test vs torch `_gather_kv_q8`: byte-identical,
   V tail zeroed (Sk_pad, varied seq_lens).
3. Synthetic end-to-end (gather → `fa.forward` vs fp32 SDPA, decode AND
   prefill shapes, tails, B=1/B=2): all correct (rel err ~2.5e-3 = Q8 noise).
   NOTE: my first two "references" were wrong (einsum axis bugs — GQA
   broadcast + softmax over the wrong axis); the *pairwise* A/B/C identity
   checks are what kept the investigation honest.
4. In-model double-gather compare (`GFX906_FA_DOUBLE_CHECK=1`): **K identical,
   V corrupted with NaN** — synthetic had passed because my test caches were
   contiguous.

**Root cause**: `value_cache` in the backend is `kv_cache.unbind(1)` of
`[num_blocks, 2, block_size, Hkv, D]` — non-contiguous, block stride 2×.
`gather_paged_kv_q8` (and `forward_paged_direct`) **computed strides from
shapes, ignoring the real tensor strides** → the kernel read K-cache bytes
as V → NaN/garbage V. Only block 0 happened to look sane. Same bug class in
`reshape_and_cache_q8` (latent — its side buffer is contiguous today).
Fixed all three sites to use `tensor.stride(i)` (+ element size for fp16 V,
the launcher wants bytes) with contiguity TORCH_CHECKs on the last dim.

After the fix, live double-check reports `K=True V=True`, and the probe
generates the exact correct greedy output in both LEGACY modes.

**Benchmarks** (pp=2048/tg=256, single request):

| config | eager t/s | notes |
|--------|-----------|-------|
| Triton paged + P3-1 | **19.49** | best eager |
| CUSTOM LEGACY=1 (fp16 gather + per-step Q8 quant) | 18.49 | 2.1 ms/step gather/quant tax; FA kernel itself 194→72 µs/layer (2.7×) |
| CUSTOM LEGACY=0 FUSED (fixed) | 19.33 | fused gather removes quant tax |
| CUSTOM LEGACY=0 DIRECT forced | 19.21 | block-table indirection tax at B=1 |
| serving (cudagraph) + CUSTOM | **crashes** | `value_cache blocks mismatch` during piecewise capture — CGSupport.NEVER does not protect the torch.compile path; deeper integration issue, deferred |
| serving (cudagraph) + Triton (P3-1) | **44.09** | current best decode |

Also fixed along the way: `_bench_gfx906.py` counted tokens by re-encoding
the output *text* — garbage output re-encodes to fewer tokens (the
mysterious "32 tokens" was 256 real tokens of garbage, 19.05 t/s). Now
counts `token_ids`.

**Net P3-3 outcome so far**: attention itself can be 2.7× faster
(72 µs/layer), but at B=1 eager the win is eaten by the gather/conversion
tax and eager is CPU-launch-bound anyway; serving mode — where decode time
actually matters — cannot use CUSTOM until cudagraph capture is fixed
(capture calls attention with a different/aliasing kv_cache view; the
side-buffer alloc also assumes the first cache shape). Learnings:
- **Stride bugs hide from synthetic tests that build contiguous caches** —
  always mirror the real allocation path (`unbind` views) in tests.
- Silent registration fallbacks make dead backends invisible; assert the
  backend you expect in logs.
- The bench's text-re-encode token counting is wrong on degenerate output.
- llama.cpp-style Q8 K quant changes logits ~1e-3 — greedy outputs diverge
  from fp16 runs after ~10-25 tokens (both fluent); same trade llama.cpp
  makes.
- A/B pairwise identity checks (two implementations on same inputs) are
  more reliable than building a mathematical reference from scratch.

Next candidates: (a) cudagraph-safe CUSTOM (fix capture view handling or
pad the side buffer at capture sizes), (b) prefill uses CUSTOM (it wins
there per the vendored docs; our serving prefill could improve), (c) back
to the original P3-3 Triton partitioning for the serving path.

### Day-1 pre-step pair — gather micro-bench + P3-2(a) probe (2026-08-15)

Plan v6's two Day-1 gates, run in the same session. Scripts:
`benchmarks/kernels/gfx906/bench_gfx906_fa_gather.py` and
`bench_llmm1_rows_per_block.py` (7.14 image, repo source-mounted,
HIP_VISIBLE_DEVICES=0).

**Gather micro-bench (P3-3a go/no-go) — GO.**
`gather_paged_kv_q8` at B=1, Hkv=2, D=256, bs=16, pre-allocated out
buffers (serving steady state); correctness vs torch reference
byte-identical incl. V-tail zeroing:

| Sk | µs/layer | GB/s | ×floor(798 GB/s) |
|----|----------|------|------------------|
| 2048 | 18.6 | 345 | 2.3× |
| 2816 | **21.7** | 407 | 2.0× |
| 3328 | 25.3 | 412 | 1.9× |

Q-fp32 side costs per FA layer (Sq=1): q.float 3.9 + q_pad.zero_ 2.7 +
q copy 7.0 + out unpack 8.3 = **21.9 µs/layer** (launch-bound small
copies). Combined tax **43.6 µs/layer** vs 122 µs/layer FA kernel win
(194−72) → net **~0.78 ms/step** over 10 layers — inside the plan's
M1 expectation (0.7–1.2 ms → 46–48 t/s). **P3-3a RESUMES** per plan §4,
as the time-fenced parallel line; the Triton PIECEWISE baseline bench
(deconfound vs FULL_DECODE_ONLY) is required before reporting M1
numbers. Note for M2: the Q-side 21.9 µs is as large as the gather
itself — zero+copy+unpack are the natural next trim.

**P3-2(a) aiter probe — STOPPED (structurally a no-op on gfx906).**
Code audit of `rocm_unquantized_gemm_impl` (vllm/model_executor/layers/
utils.py):
- `VLLM_ROCM_USE_AITER` defaults **False** (envs.py).
- Every aiter gemm path (aiter triton `gemm_a16w16`, `wvSplitKrc`) is
  inside `if not on_gfx906():` — unreachable on MI50.
- `ops.wvSplitK` explicitly excluded on gfx906 ("matrix cores not
  supported", GCN5/Vega20).
- aiter triton-gemm whitelist is GPT-OSS shapes (m==5120/k==2880, …);
  none match this model.
→ aiter rejection reason recorded: **arch exclusion, not shape/dtype
selection** — the P3-1-era "collapse into P3-2(b)" path confirmed.

**LLGemm1 `rows_per_block` sweep** (the one real knob; dispatch
hardcodes 4):

| shape (M×K) | ×/step | rpb2 | rpb4 (cur) | rpb8 | rpb16 | floor |
|-------------|--------|------|------|------|-------|-------|
| 12288×2048 in_proj | 30 | 57.8 | 64.7 | 60.1 | 61.0 | 63.1 |
| 248320×2048 LM head | 1 | 1134.6 | 1209.8 | 1244.2 | 1178.6 | 1274.6 |
| 9216×2048 qkv | 10 | 43.9 | 60.9 | 49.7 | 51.0 | 47.3 |
| 2048×4096 o_proj | 40 | 21.5 | 23.0 | 25.3 | 28.9 | 21.0 |
| 1024×2048 shared gate_up | 40 | **47.7** | 7.6 | 8.8 | 9.9 | 5.3 |
| 2048×512 shared down | 40 | 5.5 | 7.1 | 6.8 | 7.8 | 2.6 |
| 256×2048 router | 40 | 4.7 | 5.0 | 5.1 | 5.9 | 1.3 |
| 64×2048 GDN small | 30 | 4.6 | 4.6 | 4.6 | 5.7 | 0.3 |

Weighted per-step: rpb4 5604 µs (current), rpb8 5523 µs (+1.4% best),
rpb2 6626 µs, rpb16 5793 µs. No shape moves ≥20% vs current → probe
stop condition met; config retune alone is not worth a change.
Anomaly: rpb=2 is 6× slower than rpb=4 on shared gate_up (M=1024)
while winning 6–11% on the big shapes — block-count tail/occupancy
effect (512 blocks vs 256 on 60 CUs); rpb=2 net negative.

**Scoping for P3-2(b)** (custom M=1 W16A16):
- Big rows (in_proj/LM head/qkv/o_proj) are **BW-bound at rpb=4**:
  0.95–1.29× floor. Realistic capture there ≈ 0.2–0.5 ms/step (qkv
  1.29× is the main one).
- Small rows (gate_up/down/router/GDN-small, 150 calls/step) are
  **launch/latency-bound**: 3.6–14× floor. LLGemm1 at M=64 is 16 blocks
  × 256 threads — it does not fill 60 CUs; a K-split (MoE-kernel-style)
  M=1 kernel attacks this. Ceiling ≈ 0.5–0.6 ms/step.
- Total realistic P3-2(b) capture ≈ **0.7–1.1 ms/step**, consistent
  with (slightly under) the plan's 1–1.5 ms estimate.
