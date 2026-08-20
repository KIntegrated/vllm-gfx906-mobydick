# Dev log — Qwen 3.5 quantized MoE decode/prefill on gfx906

Copyright Kevin Read <me@kevin-read.com>

> Split from the original DEVLOG-moe-opt.md (2026-08, topic
> consolidation): this file is the MoE kernel train (W4A16 grouped
> GEMM, prefill tuning, the fill/copy pile, layer-0 attribution, and
> the branch merge/archival). The custom FA backend moved to
> DEVLOG-fa-attention.md; the dense 27B trail to DEVLOG-dense-decode.md.

**VERDICT (top-level):** MoE decode 3.49 → 67.39 t/s on the flagship
(19.3×); prefill ~2140 t/s. See the phase summaries and the devlog's
headline table. Individual experiment verdicts are labelled inline
(`SHIPPED` / `DESCOPED` / `DEAD-END` / `NEUTRAL`).

> **Archive note (2026-08-17):** the plan and code-review documents this log
> references were consolidated into `README.md` (this directory) and removed
> at merge-prep time; their final statuses are recorded in this log and in
> the README performance history. One-off probe scripts referenced below were
> likewise removed. This file is the historical development record.


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

### gfx906 ISA notes (from [kernel notes](latency-hiding.md), [lds-layout](lds-layout.md))

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


---

## P3-4: the fill/copy pile — attributed, three fixes LANDED (2026-08-16)

The fresh post-FA-track trace showed 1178 µs/step in small fills + D2D
copies (246.8 launches/step, median 4.64 µs — 100% launch-latency-bound,
so the only lever is removing launches, not bytes). Attributed via an
eager-mode torch profiler run (`docs/gfx906/fillprof_probe.py`,
correlation-id join of cpu_op → kernel → python stack; the GPU-queue-lag
makes kernel-timestamp-based stack attribution unreliable, CPU-op-timestamp
windows are the reliable one) — 114 decode steps of the standard bench:

| shape | n/step | µs/step | origin |
|---|---|---|---|
| fill [4,2048] | 39.7 | 125 | `F.pad(weight,(0,0,0,4-m))` in `_llmm1_tiny_m` (utils.py) — re-pads the *constant* shared-expert-gate weight [1,2048] every layer every step (LLMM1 needs rows%4==0) |
| fill [8,1024] | 39.7 | 121 | `w1_out.zero_()` before MoE gemm1 (atomic K-split accumulation) |
| fill [1,2048] | 38.7 | 113 | `output.zero_()` before MoE gemm2 (aliases w1_out's memory — see REJECTED below) |
| fill [1,32,128] | 30+ | ~94 avg | GDN `core_attn_out = torch.zeros(...)` in qwen_gdn_linear_attn.py (upstream PR #28182, spec-decode invariant) |
| copy [1,2048] | 80.3 | 316 | ~40 MoE apply copies + ~40 LLMM1 pad copies (F.pad above) |
| copy [1,4096] | 10 | 36 | FA output fp32→fp16 cast (C++ requires fp32 q/o) |
| fill [1,16,2,256] | 10 | 33 | FA `q_pad.zero_()` in forward_paged (LEGACY) |
| runner H2D micro-copies | ~17 | ~60 | zero_block_ids / commit_block_table / _prepare_inputs (upstream) |

### Fixes landed (all bit-exact or provably output-identical)

1. **shared_expert_gate [1,K] → GEMV RPT=1** (utils.py `_llmm1_tiny_m`):
   the GEMV dispatch condition gains `m == 1` (auto RPT=1 for N=1).
   Kills the F.pad fill + pad copy per layer per step. Micro-bench
   (`/tmp/bench/bench_gate_gemv.py`): 22.2 → 4.7 µs/call (4.7×) and
   **bit-equal** to the pad+LLMM1 path at N=1, K=2048 (both 0.0 vs an
   fp32 reference). Call sites of `_llmm1_tiny_m` guarantee n==1 and
   bias is None, so the GEMV preconditions hold.
2. **FA q_pad zero_ skip on the decode fast path**
   (gfx906_fa_paged.py, LEGACY branch): for `max_seqlen_q==1 and
   num_tokens==num_seqs` the pad rows are never consumed — the tile
   kernel computes q rows independently (no cross-row reduction), Python
   keeps row 0 only, and the q8_0 quantization clamps NaN/Inf garbage to
   int8 range (fmaxf NaN semantics) so garbage pad rows stay finite even
   through KVSPLIT combine. Prefill / multi-token decode keep the zero_.
   15/15 test_gfx906_fa.py (incl. the cudagraph capture test) pass.
3. **GDN core_attn_out torch.empty on the non-spec packed-decode fast
   path** (qwen_gdn_linear_attn.py `forward_cuda`), env-gated
   `GFX906_GDN_EMPTY_CORE_OUT=1` (default 0 keeps the upstream
   PR #28182 zeros). Safe because
   `fused_recurrent_gated_delta_rule_packed_decode_kernel` stores
   unconditionally for every (token, head, dim) cell — including an
   explicit zero store on the invalid-state branch — so no byte of the
   output buffer survives from the allocation. The gate mirrors
   `_forward_core`'s fast-path condition exactly.

### Fix REJECTED: MoE zero_ reorder (aliasing trap)

Reordering `output.zero_()` to overlap with `w1_out.zero_()` corrupted
prefill (PPL 6.69 → 1.09e7). modular_kernel.py `_allocate_buffers`
**reuses one `common_workspace` for both `workspace13` (gemm1 out) and
the fused `output`** ("Reuse workspace13 for the output since there is
only one chunk") — they alias, so the gemm2 zeroing must happen after
gemm1+activation have finished using the same memory. Reverted.
(Both gemms also K-split via grid.z atomics, so direct-store epilogues
are off the table for either — the two zeroings are genuinely required
and cannot be merged.)

### Deferred (measured, not worth it / needs upstream)

- FA fp16 q in / fp16 out (~80 µs/step, 2 casts × 10 layers): needs
  templatizing `flash_attn_tile_q8` + the Q-quantizer + the C++
  launcher; the quantization would stay bit-equal (fp16→fp32 is exact)
  but it touches the vendor kernel for ~0.5%. Noted, not done.
- MoE `output.zero_()` / `w1_out.zero_()` themselves: required by the
  atomic K-splits (see above).
- runner H2D micro-copies (~60 µs/step): upstream v1 worker code.
- GDN fill on the prefill/spec path: upstream invariant, left intact.

### Gates

- PPL probe (prefill logprobs): baseline 6.6862 vs fixes 6.6889 —
  inside run-to-run MoE-atomic noise (the AWQ gemm2 atomic-add order is
  not deterministic across runs; same-magnitude deltas appear between
  two runs of identical code).
- MB greedy probe (B=2 decode, FULL_DECODE_ONLY): heads identical; one
  token-0 flip BETWEEN TWO RUNS OF THE SAME (fixed) BUILD — the known
  engine multi-batch near-tie non-determinism, not a regression signal.
- Unit: 15/15 test_gfx906_fa.py, 12/12 test_gfx906_moe_gemm.py.
- Ruff: no new errors (utils.py 12→12, gfx906_fa_paged.py 31→31, the
  other two files clean).

### Serving A/B (local venv, util 0.95, fastsafetensors, FULL_DECODE_ONLY,
pp=2048/tg=256, 2 samples each, sequential)

- baseline (all fixes stashed): 63.175 / 63.53
- fixes (1)+(2)+(3, env on): **64.102 / 64.062 → record 64.08 t/s**
  (fixed-run spread 0.04 vs baseline 0.36; min-fixed > max-baseline).

Net +0.7 t/s (+1.15%) vs the 63.56 record. Smaller than the ~350
µs/step the isolated per-op numbers predict: at 99.5% GPU busy, part of
the removed fills was already partially overlapped by async compute, so
only the critical-path share of each launch is recovered. llama.cpp gap:
70.3/64.08 ≈ 1.10×.

Phase 3 is declared done here. The MoE-side residual (routed gemm
1.92 ms, routing 1.05 ms, gemm zeroings 0.23 ms, plus a 0.41 ms
uncharacterized Triton residual ≈ 3.8 ms/step vs ~1.0 ms floor) is
catalogued for a possible future phase in
`plan-moe-decode-future.md`.

## O1 resolved: the Triton residual is layer 0 (2026-08-16)

The 2×/step `fused_moe_kernel.kd` (206.8 µs/call, 414 µs/step) from the
post-FA-track trace is **layer 0's routed MoE**, not the shared expert
and not MTP.

Method (P3-4 pattern, eager torch-profiler trace
`/tmp/bench/fillprof/`): kernel External id → correlated cpu_op =
`vllm::moe_forward_shared` (dims [1,2048]×3) → enclosing
`python_function` frame. All 114 Triton-launching ops over 114 steps
sit inside `Qwen3NextSparseMoeBlock_0`; zero in layers 1–39. A
model-load log probe confirms the split:
`unquantized.py: Using TritonExperts MoE backend` (layer 0) +
`int_wna16.py: Using Gfx906WNA16Experts` (layers 1–39).

Root cause: the AWQ checkpoint's `quantization_config.
modules_to_not_convert` = `["visual", "linear_attn", "self_attn",
"shared_expert", "mlp.gate", "model.layers.0.", "mtp"]`. So
- **layer 0's 256 routed experts ship fp16** → unquantized oracle →
  Triton `fused_moe` (the 414 µs),
- **all shared experts are dense fp16** (not W4A16) → they already run
  the LLGemm1/LLMM1/GEMV dense surface — the P2-5 premise "shared
  expert dispatches through the generic Triton fused_moe_kernel
  (~40 launches/pass)" was a misattribution of this same layer-0
  signal; there was never a Triton shared expert.

Roadmap updated (`plan-moe-decode-future.md`): O1 resolved, C4
rewritten (layer-0 options: leave / re-quantize to AWQ at load —
(b) only if the 70 t/s target is live), C5 corrected to fp16-fp16
(no dequant), header + §1 table fixed, new §6 cross-references the
open Phase-2 items (P2-4→C1, P2-5→C4+C5, P2-1(e) persistent-CTA =
prefill, out of scope, parked in §6, P2-3 = this roadmap's rescope,
P2-6 → Phase 3).


---

## 2026-08-17 — Pre-merge review fixes (combined review, 4 findings)

`gfx906_qwen_impr_code_rev_claude.md` (root; combined review of the
`moe-opt` branch) carried four verified correctness findings. All fixed:

1. **`_repack_w4a16_wna16_layout` UnboundLocalError on symmetric
   checkpoints** (`qzeros is None`): the `zr = zf.view(...)` / `zp = ...`
   recompute ran unconditionally after the `if/else`, referencing `zf`
   which only exists in the `else` branch; the symmetric `zp` fill was
   dead code. Fixed by moving the recompute under `else:`.
   **Second latent bug found while fixing**: the symmetric fill was
   `torch.full(..., 8, dtype=int32)` — a packed zp word needs 8 in
   *every nibble* (`0x88888888`); the value `8` only set nibble 0, so
   columns 1-7 would have used zero point 0 (verified by probe: column 0
   matched the `(q-8)*s` reference, the rest did not). Now filled as
   uint32 `0x88888888` viewed as int32 (overflows signed). Same fix in
   `_repack_w4a16_awq_kfirst_layout` (it never crashed there because the
   branch had no fall-through, but it would have produced the same
   wrong-zp output).
2. **GDN `core_attn_out` zero-fill skip was not platform-gated**:
   `_GDN_EMPTY_CORE_OUT` (default on) applied the `torch.empty` swap on
   every platform; the row-rewrite proof only holds for the gfx906
   deployment path. Added `on_gfx906()` to the condition (module-level
   constant lookup, negligible in forward).
3. **Oracle accepted GPTQ-style zero-point checkpoints for `GFX906_HIP`**:
   only `may_have_zp` was checked; the kernel/repack implement
   AWQ-encoded zps only (`zero_offset=0`). Note from verification: this
   vLLM's `AutoGPTQConfig` only constructs *symmetric* configs (TYPE_MAP
   is (4,True)/(8,True)), so those were already excluded by the
   `may_have_zp` check; the reachable asymmetric-zp path is
   `QuantizationArgs` (compressed-tensors) — today that hit a loud
   `ValueError` in the repack's layout detection, not the silent
   mis-encoding the review hypothesized, but the oracle now excludes
   both `(AutoGPTQConfig, QuantizationArgs)` explicitly.
4. **`moe_gptq_gemm_gfx906` missing `top_k > 0` guard**: on-device
   integer divide by zero is UB on gfx906. Added `TORCH_CHECK(top_k > 0)`
   to the host-side checks; verified the guard fires with a clean error.

Tests added:
- `tests/kernels/moe/test_gfx906_moe_gemm.py`: `awq_kfirst_sym` /
  `wna16_sym` layouts (qzeros=None, implicit zp 8) across all six
  shapes — 12 new cases; the `wna16_sym` case crashed with
  UnboundLocalError pre-fix and produced wrong output with the
  first-draft `8` fill.
- `tests/quantization/test_moe_wna16.py`: GFX906_HIP rejection cases
  (symmetric GPTQ no-zp; compressed-tensors asymmetric-zp) + acceptance
  cases (AutoAWQ, MoeWNA16 with zp).
- M=1 cases in the MoE GEMM test use tol 1e-1 (denominator over only
  TOPK rows makes fp16 accumulation noise ~2x noisier; a 5e-2 threshold
  flaked once in ~50 runs at worst-measured 3.9e-2).

Gates: full suite 72 passed / 2 skipped (4 suites); PPL MoE **6.6827**
in band (recent 6.6817-6.6832). Dense PPL/serving not re-run: on
gfx906 all four fixes are inert for the dense model (the GDN gate
evaluates to the same value it did before; the other three are
MoE-load-path only).

---

## 2026-08-18 — Parked review items R1–R12 resolved

All 12 items parked in roadmap §9 (2026-08-17) are now resolved. Most
are edge-path / hygiene fixes; three touch the default LEGACY=1 decode
path (R8 buffer reuse, R7 ladder consolidation, R11 lint) — all gated
below.

### Edge-path correctness

- **R1 (P2)** — `forward_paged` direct branch now uses the fp32
  `q_pad_buf` (fp32 fallback allocation) instead of the fp16 fallback
  that `forward_paged_direct`'s `TORCH_CHECK` rejects; the Sq=1 case
  additionally uses the dedicated `q_pad_decode_buf` (zero-copy
  `[:num_seqs]` prefix) so B≥2 decode does not pay a per-layer
  `.contiguous()` copy. LEGACY=0-only path (dormant in default config).
- **R2 (P3)** — `get_cudagraph_support` fails closed (RuntimeError) on
  LEGACY=0 + prefix caching instead of logging and continuing.
- **R3 (P3)** — `_ensure_gather_buffers` retired list is now bounded
  (`_gather_retired_max = 4` pairs, oldest evicted). The review's
  suggested "grow-only capacity buffer + exact-size view" was rejected
  after inspection: the gather kernels address their output from
  SHAPES, not strides (gfx906_fa_gather.cu), so a non-contiguous view
  corrupts silently; a real capacity buffer needs stride-based output
  addressing in C++. Bounding trades worst-case memory for a stale
  graph possibly touching an evicted buffer — only reachable in the
  LEGACY=0 + capture combination that is already inconsistent (RC2).
- **R4 (P3)** — `moe_gptq_gemm_gfx906` host checks now include
  `K == qweight_rows*8`, `groups > 0 and K % groups == 0` (the kernel
  computes `groupsize = K / groups`), `N % 8 == 0`, `scales_N == N`,
  `zeros_N*8 == N` — all were silent-garbage violations. Oracle
  `int_wna16.py` gains the matching GFX906_HIP shape gate
  (intermediate_size_per_partition % 8, hidden_dim % group_size).

### Cross-platform / build

- **R5 (P2)** — `vllm/gfx906_fa/{__init__,gfx906_fa_paged}.py` import
  `_gfx906_fa_C` tolerantly; `register()` is a no-op off ROCm-gfx906
  (platform check inside the function), so the general_plugins entry
  point no longer tracebacks on every non-gfx906 startup and the
  backend cannot be registered (hence selected) elsewhere.
- **R6 (P2)** — resolved on the CMake side: the standard recipe builds
  with empty `PYTORCH_ROCM_ARCH` + device auto-detect, so narrowing
  `_targets_gfx906()` to explicit arches would break the standard
  build. CMakeLists.txt now defines a no-op
  `add_custom_target(_gfx906_fa_C)` + `install(CODE)` component (with a
  WARNING) when lang=HIP and gfx906 is absent from the arches.
  Verified: CMake reconfigures cleanly on this gfx906 machine (the
  gfx906 branch still wins); the no-op branch is structural.

### Cleanup / debt

- **R7 (P3)** — the ncols1 ladder had FOUR code copies (2 Python,
  2 C++; the review said three). Consolidated to `fa_pick_ncols1()`
  (gfx906_fa.cpp, both C++ sites) + `_pick_ncols1()`
  (gfx906_fa_paged.py, imported by the backend), each with a
  cross-language keep-in-sync pointer.
- **R8 (P3)** — `gather_paged_kv_quantized` gained the same
  `k_out`/`v_out` grow-buffer params as `gather_paged_kv_q8`
  (`use_or_alloc`, exact-shape match); the backend now passes the
  class-level gather buffers on BOTH paths (previously LEGACY=1 — the
  DEFAULT — passed None, allocating a fresh 24-200+ MiB K+V pair per
  layer per step on long contexts). The FUSED_QUANT=0 fp16 fallback
  reuses only V (its K output is fp16, not the q8 byte layout).
- **R9 (P3)** — the workspace13/fused_out aliasing (modular_kernel
  `common_workspace`) and its load-bearing order (gemm1 → activation →
  `output.zero_()` → gemm2) are now documented in
  `gfx906_w4a16_moe.py` (workspace_shapes + gemm2 call site).
- **R10 (P4)** — kchunk docstrings include 1024 (ops.h +
  _custom_ops.py); `forward_paged_direct` pybind help states the
  native BSHD output + fp32 input; launcher header MVP block rewritten
  (mask/KV_max/direct-paged are implemented, no host transpose); MoE
  kernel header grid formula carries N_PER_THREAD (N/1024 was only
  true for NPT=4); utils.py `/tmp/bench` comment → DEVLOG; dead
  `mask_buf=None` param removed (R10-partial from the earlier session
  is now complete); `forward_paged` query docstring corrected (fp16).
- **R11 (P4)** — `vllm/gfx906_fa/`, the gfx906 bench scripts and the
  two gfx906 test files are ruff-clean (UP/I/F541 auto-fixes + E501
  wraps + F841/SIM108 manual). `benchmarks/kernels/gfx906/*` gets a
  per-file-ignores entry for B023 + the 6 false-positive F821s
  (deliberate timeit-closure captures). utils.py's 11 E501s are all
  pre-existing upstream lines (none on branch-touched lines) — left.
- **R12 (P4)** — new `GFX906_FA_DEBUG=1` master switch enables all six
  off-by-default debug hooks at once (FWD_DEBUG, DOUBLE_CHECK, DUMP
  with default dir /tmp/gfx906_fa_debug, NO_BUF_REUSE, TORCH_GATHER,
  ZERO_KTAIL); individual knobs kept. The three ON-by-default
  functional switches (FUSED, FUSED_QUANT, QPAD_EMPTY) are not debug
  hooks and are untouched.

### Gates

- Rebuilt both extensions incrementally (`moe_q_gemm_gfx906.hip`,
  `gfx906_fa.cpp`, launcher; verified new strings in the .so's).
- 4 suites: **74 passed / 2 skipped** (72 + 2 new oracle shape-gate
  cases). The oracle acceptance test now uses realistic Qwen3.5 MoE
  shapes (the dummy 1x1 config trips the new shape gate by design).
- PPL MoE: **6.6825** (band 6.6817–6.6832; prior 6.6827) — the R8
  buffer-reuse change is on the default LEGACY=1 path, so the PPL gate
  covers it end-to-end.
- Serving benches NOT re-run: R8 changes allocation source, not the
  kernel work (same bytes moved, same kernel); R1/R2/R3/R4/R5/R6/R12
  are dormant or inert in the default serving configuration. The
  dense model is unaffected except via the shared FA buffers (same
  shapes/sizes as before; R3 only bounds the already-small retired
  list).

---

## 2026-08-18 — Merged into gfx906/main

`gfx906/moe-opt` merged into `gfx906/main` by fast-forward (main was a
direct ancestor; 71 commits, 52 files, +13818/-21, tip `1691d1dd29`).
Linear history kept (fork convention: merge commits only for upstream
merges). All gates current at the merge tip: 4 suites 74 passed /
2 skipped, PPL MoE 6.6825 (band 6.6817-6.6832), serving records 25.60
t/s dense / 67.39 t/s MoE (see the serving sections above).

---


---

## 2026-08-18 — Upstream merge: vllm-project/main into gfx906/main

Merged `upstream/main` (158 commits since merge-base `015660da91`) into
`gfx906/main` (220 ahead). 661 files, +31783/−7248. Four content
conflicts, all resolved by hand:

1. **`vllm/config/attention.py`** — `IndexerKVDType` union: upstream's
   `"auto"` + logger, kept alongside our fp16/fp32 spellings (Minimax M3
   gfx906 work, `0ccc37a118`/`a0c1d17893`). Upstream's default change
   (`"bf16"` → `"auto"`) and the `use_fp4_indexer_cache` deprecation
   alias come along as-is.
2. **`auto_awq.py`** — took upstream's deletion of
   `is_awq_marlin_compatible` (dead at the merge-base; our gfx906 guard
   on it was moot). The real gfx906 AWQ routing (MoeWNA16 fallback,
   `get_supported_act_dtypes`, `get_min_capability`) is untouched and
   verified present in the merged file.
3. **`v1/attention/ops/rocm_aiter_mla_sparse.py`** — kept our
   env-gated fp16 logits gate at the top of `rocm_mqa_logits` (gfx906
   has no FP8); accepted upstream's removal of the stale `TODO(ganyi)`
   workaround comment and the new gfx942 flydsl path (not taken on
   gfx906 — `_ON_GFX942` guard).
4. **`v1/attention/backends/mla/rocm_aiter_mla_sparse.py`** —
   `get_supported_kernel_block_sizes`: base `[1, 64]` → we ran
   `[1, 32, 64]`, upstream generalizes to `[1, MultipleOf(16)]` (a
   superset of ours; our fp16 logits path is block-size agnostic).
   Took upstream.

Upstream content of note: XPU work, Rust frontend/gRPC, spec-decode MTP
fusions, `[ROCm]` Triton W4A16 transpose bugfix (`1d3a8b9e22`, new
tests), gfx942 FlyDSL fp8 MQA logits, DSV4 Triton sparse-MLA decode
(gfx950), quant dead-code removal (`7d7b6f26f4`). Zero csrc/rocm
changes from the merge (our whole kernel stack untouched);
`libtorch_stable` changes → `_C_stable_libtorch` rebuild; `_rocm_C`
recompiled for the changed `ops.h` header; `_C` unchanged (its source
`csrc/torch_bindings.cpp` untouched by the merge).

### Validation (post-merge, local venv, MI50)

- Build: `setup.py build_ext --inplace` exit 0; all changed targets
  rebuilt and reinstalled.
- M3 config tests (`tests/models/minimax_m3/`): **byte-identical
  pre/post** — `test_config.py` 3F/4P both sides (the 3 failures are
  pre-existing: they assert an `fp16→float16` normalizer in
  `AttentionConfig` that was never implemented; the commit that added
  them says "test: partially ok"). `test_fp32_kv_config.py` has a
  pre-existing collection error (imports
  `_use_fp16_dot_for_fp32_inputs`, absent from our tree). No merge
  regression.
- `test_auto_awq.py`: identical 1F/8P both sides (the failure needs
  Qwen2-1.5B-Instruct-AWQ weights, not in the offline cache).
- gfx906 suites: `test_gfx906_moe_gemm.py` + `test_fused_topk.py` +
  `test_moe_wna16.py` = 47 passed / 720 skipped; GEMV + FA + upstream
  `test_triton_w4a16.py` = 52 passed / 2 skipped.
- Greedy probe (35B, 12×128): **token-identical to baseline**
  `ALL 868ad09ee35c493d83043655ffccecff4fd61f1379d9d6c6adde2dfa967aad2c`.
- Serving graph bench (4 samples, 2048pp/256tg): **66.65 / 66.35 /
  66.66 / 66.59 t/s** (mean 66.56) — mid-session band of the pre-merge
  range 65.87–67.03 (S3 default-on, S2/S5 off).
- Ruff on the 4 conflict files: 0 new findings; the 24 present
  (20×E501, 2×F401, 1×I001, 1×SIM102) are pre-existing on the branch
  (identical in the pre-merge worktree) and left untouched.

### Serving benches post-merge (dense + MoE)

| model | config | post-merge t/s | pre-merge reference |
|---|---|---|---|
| MoE 35B-A3B-AWQ | graph, 4 samples | 66.65 / 66.35 / 66.66 / 66.59 (mean **66.56**) | session band 65.87–67.03 (same day, pre-merge); record 67.39 (max-ilp build day) |
| Dense 27B-AWQ | graph, 4 samples, **max_num_seqs=4** | 25.34 / 25.05 / 25.29 / 25.30 (mean **25.25**) | record 25.14 / 25.60 (2 samples) |

**Verdict: no regression in either model at production configs.**

**Finding — dense 27B graph mode OOMs at max_num_seqs=32 (post-merge):**
first 1568-token prefill chunk, 340 MiB inductor static-buffer
allocation, `free: 0` (the known 340/600 MiB signature from the 27B load
test). At 32 seqs the GDN mamba-state pool ('align' mode, block size
784 chosen for mamba-page alignment — logic pre-dates the merge) plus
weights 19.77 GiB + KV 7.13 GiB leave no headroom for the compiled
prefill buffers; at 4 seqs the pool shrinks by ~2 GiB and it runs clean
(KV 7.39 GiB / 43,366 tok). The record-era harness state is ambiguous
(the file was consolidated the same day as the 25.14/25.60 records), so
it is not proven the merge changed 32-seq behavior — but the post-merge
compile config registers many more `splitting_ops` (deepseek_v4,
MLA kv-update, sparse indexer) which plausibly grows the inductor
static footprint. **Production dense config is 4–8 seqs** (load-test
section); 32-seq graph dense is not a deployment config on this
32 GB card. `BENCH_MAX_SEQS` knob added to `_bench_gfx906.py` (default
32 unchanged) with this documented in-line.

Open (pre-existing, not merge-related): the M3 fp16/fp32 config
normalizer gap and the `test_fp32_kv_config.py` import — the M3 gfx906
support shipped as "partially ok" and was never finished.

