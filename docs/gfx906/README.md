# gfx906 (MI50/MI60) optimization hub
Copyright Kevin Read <me@kevin-read.com>

This fork optimizes vLLM for AMD **gfx906 (Vega 20)** — MI50/MI60, 60 CUs,
32 GB HBM2 (~800 GB/s), **no MFMA, no int8 matrix cores**. All numbers are
measured on a single MI50 with ROCm 7.14, torch 2.13, single request,
`pp=2048`/`tg=256`, `cudagraph_mode=FULL_DECODE_ONLY`.

## Fork heritage

- This repository is the gfx906 vLLM port
  [**ai-infos/vllm-gfx906-mobydick**](https://github.com/ai-infos/vllm-gfx906-mobydick),
  itself based on [**nlzy/vllm-gfx906**](https://github.com/nlzy/vllm-gfx906),
  the original gfx906 port of vLLM.
- The custom Q8 FlashAttention kernels come from
  [**cassettesgoboom/gfx906-fa-vllm**](https://github.com/cassettesgoboom/gfx906-fa-vllm),
  vendored early in this fork's history and substantially extended for the
  decode path (see §What changed).

Test models:

- **MoE:** `QuantTrio/Qwen3.5-35B-A3B-AWQ` (40 layers, 256 experts × 8
  active, hybrid: 30 GDN linear-attn + 10 full-attention)
- **Dense:** `Qwen3.5-27B-AWQ` (64 layers: 48 GDN + 16 full-attention,
  Hq=24/Hkv=4/D=256)

Reference point: llama.cpp (Q4_K_XL GGUF, full offload) — **70.3 t/s decode,
806.5 t/s prefill** on the same hardware.

## Headline results

| workload | fork base (`gfx906/main`) | now | Δ | reference |
|---|---|---|---|---|
| MoE serving decode | 3.49 t/s | **67.39 t/s** | 19.3× | llama.cpp 70.3 (1.04× gap) |
| MoE prefill (pp=2048) | ~450 t/s | **~2140 t/s** | 4.7× | llama.cpp 806.5 (2.7× ahead) |
| Dense serving decode | 18.89 t/s | **25.60 t/s** | +35% | — |
| MoE concurrent decode (N=8) | 166.9 t/s (W4 off) | **191.0 t/s** | +14.5% | W4 skinny fp16 M≤16 (`VLLM_GFX906_SKINNY_M16`, flag on, soak-verified; `DEVLOG-fp16-skinny.md`) |

Correctness gates: PPL on a fixed 442-token probe — MoE band 6.6817–6.6942,
dense band 6.6993–6.7197. Kernel test suites: 28/28 FA, 43/43 MoE GEMM (2026-08-24).

## Model support status (single MI50, MI60 numbers similar)

All numbers: serving decode t/s, graph mode, pp=2048/tg=256, single
request (4 samples) unless noted. Recipes: §Bench recipes +
`DEVLOG-spec-decode.md` (spec-decode arms).

| model | status | decode t/s | prefill t/s | notes |
|---|---|---|---|---|
| **Qwen3.5-35B-A3B-AWQ** (MoE) | **flagship, optimized** | **67.39** (record; band 65.3–67.0; final-build restamp 66.1, 2026-08-24) | **~2140** | full custom stack (W4A16 MoE GEMM + Q8 FA); 19.3× over fork base; llama.cpp parity on decode, 2.7× ahead on prefill |
| ↳ same, **N=8 concurrent decode** | W4 (`VLLM_GFX906_SKINNY_M16=1`, soak-verified) | 191.0 (baseline 166.9; **+14.5 %**; soak 189.9 ± 0.4; final-build restamp 192.9/194.0, 2026-08-24) | — | first N=8 record for this model (off arm = C2-V t1n8 steady); skinny fp16 M=5..16 kernel, M-dependent gate |
| ↳ same, **MTP k=2 spec decode** | recommended spec config (35B, W2) | 88.6 (1.16× vs 76.7 greedy graph; 1.83× vs eager 44.9) | — | **final-build restamp 2026-08-24** (the pre-W4 re-measure debt): 78.7 % acceptance, 1.57 tok/step; record 89.9 (1.18× vs 76.2; 80.4 % / 1.61 tok/step, pre-W4 build) |
| **Qwen3.5-27B-AWQ** (dense) | **well supported, optimized** | **25.60** (official-harness record; 27.99 no-spec on the current max-ilp split build, in-process metric) | ~257 (chunked) | GEMV + CUSTOM FA + max-ilp; serving needs `--gpu-memory-utilization 0.93` |
| ↳ same, **MTP k=2 spec decode** | recommended spec config | **39.4** (1.41×; 1.50× no-max build) | ~250 (neutral) | 90.9% draft acceptance, 1.82 tok/step; `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` |
| ↳ same, ngram-3 | works, weak | 28.0–28.9 (1.0–1.09×) | — | agentic prompts only break even; MTP preferred |
| **Gemma-4-26B-A4B-it-AWQ-4bit** (MoE) | **well supported, optimized** | **67.79** | — | no-zero-point W4A16 expert kernel (`gfx906/gemma4-moe-nzp` work, 1.79× over Triton); chat template required (thinking model); PPL/prompt_logprobs unreliable on this model — gate on coherent text + logprob A/B |
| **cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4** (MoE VLM) | **supported (new 2026-08-25, `gfx906/moe-ct-asym-zp`, unmerged)** | **65.03** (A/B mean; band 64.995–65.079; decode-only 81.1) | TTFT 0.77 s @2048 | first **asymmetric** (stored int8 zp) CT W4A16 checkpoint: oracle gate + pass-through zp repack, no kernel change; 18.6× over the Triton arm (3.50) — but the Triton W4A16 `has_zp` branch is pathologically slow on gfx906 (267 ms/tok, both zp layouts) — `DEVLOG-ornith-wna16.md`; PPL 16.67 gfx vs 16.45 triton (fp16-noise band); class-parity with the flagship 67.39 |
| **Qwen3.8-27B-AWQ-INT4** (dense) | **fully functional (TP=1 + TP=2)** | MTP k=2 TP=2 (2026-08-24 final): **59.2** @2k / 44.9 @8k / 25.2 @32k / 16.6 @64k · greedy TP=2: 40.8/38.1/30.5/24.1 · 28.62 @4k TP=1 (record) · 104.2 (N=8, W4 on) | — | `--dtype float16` required (auto-bf16→fp16 fallback landed); **live-ctx tax: FA gather/attention O(Sk) — MTP < greedy beyond ~20k ctx** (agentic ~60k: 16.6 MTP / 24.1 greedy t/s); MTP k=2 41.41 TP=1 (record, 2026-08-23; lifecycle fix byte-identical in graph mode); N=8 needs `--gpu-memory-utilization 0.90` (64 layers, FA KV 655 KB/token); TP=2 needs the official amdgpu DKMS driver + trimmed capture `[1,2,3,4]`; 445k-token KV pool — **256k context validated** (FA gather fix 2026-08-24, `oom-256k-prefill.md` §9); non-deterministic at temp=0 (token-identity gates unusable); records: `DEVLOG-tp2-dense.md` S1–S9, `DEVLOG-masked-fa.md`, `DEVLOG-qwen38.md` |
| **Qwen3.6-27B / 3.6-35B-A3B** (fp16) | **not supported** | — | — | 52/67 GB fp16 checkpoints do not fit a 32 GB card; 3.6 GGUF only used as a llama.cpp reference point |
| small AWQ models (e.g. Qwen3.5-9B-AWQ, 0.8B) | supported | — | 590–1483 (9B, eager) | fine on ≤0.85 util; FA prefill benchmarks in top-level README |


## Contents of this directory

| file | what it is |
|---|---|
| `README.md` | this hub: changes, numbers, recipes, knobs |
| `DEAD-ENDS.md` | one-pass index: hypothesis → gate → verdict → commit for what was tried (grep-able) |
| `_devlog-template.md` | the `VERDICT:`/`HYPOTHESIS:`/`GATE:` devlog convention + worked example |
| `DEVLOG-moe-opt.md` | MoE kernel record (W4A16 GEMM, tuning, pile, merges) |
| `DEVLOG-fa-attention.md` | custom Q8 FA / decode backend record |
| `DEVLOG-dense-decode.md` | Qwen3.5-27B dense decode record |
| `DEVLOG-boot-failure.md` | 2026-08-23 weight-load `hipErrorLaunchFailure` hunt (OPEN: minimal torch repro, GTT-exhaustion theory) |
| `moe-decode-roadmap.md` | future MoE-decode candidates (roadmap, not a committed plan) |
| `spec-decode-roadmap.md` | speculative-decode on gfx906: n-gram probe results + phase plan (ngram/suffix/MTP rails) |
| `running.md` | how to run/build/bench: local venv (canonical) + docker images |
| `_bench_gfx906.py` | end-to-end pp/tg serving bench harness (BENCH_* env knobs) |
| `_pp_bench.py` | prefill/decode split harness |
| `latency-hiding.md`, `lds-layout.md`, `dequant-instructions.md` | measured gfx906 ISA facts (kernel-writing guide, linked from AGENTS.md) |

## What changed vs `gfx906/main`

### 1. Custom Q8 FlashAttention backend (`CUSTOM`)
`vllm/gfx906_fa/`, `csrc/gfx906_fa/` — vendored from
`cassettesgoboom/gfx906-fa-vllm`, built into the wheel when gfx906 is a
target arch (`CMakeLists.txt`, `setup.py`), and the **default** attention
backend on gfx906 (`vllm/platforms/rocm.py`). Escape hatch:
`--attention-backend ROCM_ATTN`.

- Q8_0 KV quantization in-kernel (LEGACY path) or fused during the paged-KV
  gather (`GFX906_FA_FUSED_QUANT`, default on, bit-equal).
- **B=1 decode parallelism**: GQA head-packing (NC2) + KV split with
  split-combine — 245 → 58.3 µs/layer @Sk=2176 (4.2×). Fixed three vendor
  bugs en route (null-mask deref, NC2×prefill OOB guard, OOB-tail masking).
- **NC2 fail-closed**: only NC2∈{1,2,8} are instantiated; invalid explicit
  values error, default 8 auto-downgrades (8→2 when ratio%2==0, 8→1 for MHA).
- kv_split clamped to 1 for prefill (seq_q>2).
- Native **BSHD** output (no transpose copy); decode per-layer copy pile cut
  7→2 (dedicated decode q-pad buffer, fused fp16→fp32 casts, deferred
  `cu_seqlens_q.to(long)`).
- Direct-paged decode path for B≥2/Sq≤16 (`GFX906_FA_DIRECT_PAGED*`).
- LEGACY=1 decode verified FULL-capture-safe; CGSupport default is
  `UNIFORM_SINGLE_TOKEN_DECODE` (this flip alone: 22.44 → 52.90 t/s MoE).

### 2. Custom W4A16 MoE grouped GEMM
`csrc/rocm/moe_q_gemm_gfx906.cu` + `vllm/.../fused_moe/experts/gfx906_w4a16_moe.py`
+ oracle entry in `fused_moe/oracle/int_wna16.py`. Fixes the upstream modular
pipeline's −71% MoE regression (3.49 t/s) on gfx906. AWQ int4, 128-group,
load-time repack (~65 s). Handles both MoeWNA16 (N-first uint8) and AutoAWQ
(K-first int32) layouts.

### 3. Dense M≤16 W16A16 GEMV family
`csrc/rocm/dense_gemv_gfx906.cu`, dispatched from
`vllm/model_executor/layers/utils.py` (`_llmm1_tiny_m`, `_gfx906_gemv_long_k`,
`_gfx906_spec_gemv_m4`; kill switch `VLLM_GFX906_DENSE_GEMV=0`):

- m<4 decode GEMMs → padded LLMM1 / GEMV (−401 µs/step MoE).
- m==1 rows → GEMV RPT=1 (kills the per-step `F.pad` of the shared-expert
  gate weight; 4.7× isolated, bit-equal).
- Long-K n==1 GEMV (K=17408 down_proj, KCHUNK=1024/RPT=2, fp16 atomic
  K-split): 227.6 µs = 100% of HBM floor vs 794 µs triton_matmul.
- Verified at the HBM floor on all dense fp16 shapes (K=5120 lm_head probe:
  neutral — LLMM1 already at floor there).
- **W4 (2026-08-23): M=5..16 skinny rail** — weight-row-parallel kernel
  (RPT=1, exact-M template, grid (N, ksplit), fp16 `atomicAdd` K-split
  epilogue), behind `VLLM_GFX906_SKINNY_M16` (default off at merge).
  x-L2-re-read bound at M·B/1.6 TB/s; M-dependent gate (M≤7 all sizes;
  M=8 ≤32 MB; M≥9 ≤10 MB) — big shapes stay on triton. Serving: +14.5 %
  35B / +6.1 % 27B at N=8; flag-on 30-rep soak passed. `DEVLOG-fp16-skinny.md`.

### 4. Other landed fixes
- **GemmaRMSNorm fused-kernel dispatch** (`layernorm.py`): Gemma's `(1+w)`
  factorization dispatches the fused RMS-norm kernel with `w' = 1+w` in the
  input dtype instead of an fp32 decomposition.
- **GDN `core_attn_out` zero-fill removed** on the packed-decode fast path
  (`GFX906_GDN_EMPTY_CORE_OUT`, default on; the Triton kernel stores
  unconditionally).
- **fastsafetensors GDS fallback** (`model_loader/weight_utils.py`): catch
  bare `Exception` (GDS-unsupported raises non-`RuntimeError`) — 2.6× faster
  loads, was engine-death before.
- **hipify in-source guard** (`cmake/hipify.py`): same-dir copytree crash on
  Py3.12 in-source rebuilds.
- **ROCm platform** (`platforms/rocm.py`): CUSTOM backend default +
  registration; device-name derivation from GCN arch (amdsmi returns 0
  handles after torch import on ROCm 7.14).
- Fill/copy pile reductions (P3-4): three bit-exact launch removals
  (+1.15%), attributed the rest (MoE gemm zeroings are required by grid.z
  atomic K-splits; runner H2D micro-copies are upstream).

## Performance history (serving, pp=2048/tg=256)

### MoE — Qwen3.5-35B-A3B-AWQ

| milestone | t/s | commit |
|---|---|---|
| fork base (upstream modular pipeline, Triton WNA16) | 3.49 | — |
| custom W4A16 MoE kernel (eager 18.88; prefill 2140) | 41.5 (graphs) | `85eacaeed9`…`f770b9f446` |
| P3-1 tiny-m gemv routing | 44.09 | `3e7c4f2252` |
| P3-3a CUSTOM FA FULL-decode capture default | 52.90 | `2cd52b6f4a` |
| fused fp16 KV gather (Route B stage 1) | 57.09 | `01526dfc69` |
| FA kernel track: NC2 packing + KV split (B=1 parallelism) | 62.8 | `e8b3293554` |
| fused gather-and-quantize | 63.56 | `225448d93f` |
| fill/copy pile fixes (P3-4) | 64.08 | `9bdd9f4639` |
| kv_split prefill clamp + NC2 fail-closed; NC2=2 + GDN flip | 65.36 | `b4873459f8`, `1a895e8a01` |
| GemmaRMSNorm fused dispatch; down_proj GEMV | 66.36 | `19c1d41cf5`, `2cd5b4cafa` |
| FA decode copy pile 7→2 (BSHD native output) | 67.02 | `d63b3ab464` |
| max-ilp scheduler per-file (W4/FA/skinny) | **67.39** | `c6247f729e` |

### Dense — Qwen3.5-27B-AWQ

| milestone | t/s | decode-only t/s |
|---|---|---|
| baseline (Triton FA, GEMV off) | 18.89 | 22.55 |
| CUSTOM FA (NC2=1 fallback) | 23.15 | 28.10 |
| NC2=2 for ratio-6 GQA | 23.55 | 28.69 |
| down_proj K=17408 GEMV | 23.85 | — |
| FA copy pile 7→2 | 24.06 | — |
| max-ilp scheduler per-file (W4/FA/skinny) | **25.60** | `c6247f729e` |

### Concurrent decode (N=8, graph, Δ-metric A/B — W4, 2026-08-23)

| model | shape | W4 off | W4 on | Δ |
|---|---|---|---|---|
| Qwen3.5-35B-A3B-AWQ (MoE) | pp=2048/tg=256 | 166.9 | **191.0** (soak 189.9 ± 0.4; final-build restamp 192.9/194.0) | **+14.5 %** |
| Qwen3.8-27B-AWQ-INT4 (dense) | pp=1024/tg=160 (KV cap, util 0.90) | 98.2 | **104.2** | **+6.1 %** |

`VLLM_GFX906_SKINNY_M16=1` (default off); 27B N=4 control flat (−0.6 %, flag
inert). Kernel, gate rationale, and soak: `DEVLOG-fp16-skinny.md`.

## Bench recipes

Canonical environment is the **local editable `.venv`** (docker images are
legacy; both documented in `running.md`). MoE:

```bash
source ~/env-rocm-7.14-gfx906.sh
HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
BENCH_EAGER=0 BENCH_PP=2048 BENCH_TG=256 BENCH_MAXLEN=3328 BENCH_GPU_UTIL=0.95 \
BENCH_CG_MODE=FULL_DECODE_ONLY BENCH_LOAD_FORMAT=fastsafetensors BENCH_SAMPLES=2 \
.venv/bin/python /tmp/bench/_b.py /local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ
```

Dense 27B (NFS model; no fastsafetensors; smaller KV):
`BENCH_GPU_UTIL=0.92 BENCH_KV_MEM=6442450944 BENCH_MAXSEQS=8
BENCH_BATCHED_TOKENS=512 BENCH_TEXT_ONLY=1` with model path
`/data/models/qwen/Qwen3.5-27B-AWQ`.

Rules: serving benches run sequentially; check `uptime` first (background CPU
contention invalidates numbers); `BENCH_ATTN_BACKEND` must NOT be set
(it forces Triton). PPL gate: `docs/gfx906/` probe convention (442-token
fixed set; bands above). PPL run-to-run noise: MoE ±0.3%, dense ±2% —
multi-batch greedy token identity is NOT a valid gate.

## Tests

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_gfx906_fa.py -v      # 15
.venv/bin/python -m pytest tests/kernels/moe/test_gfx906_moe_gemm.py -v      # 12
.venv/bin/python -m pytest tests/model_executor/layers/test_rocm_unquantized_gemm.py -v
```

All three suites are green on real gfx906 hardware. In the last file the
arch-simulation tests patch every platform predicate (incl. `on_gfx906`)
and 2 wvSplitK real-kernel tests skip on gfx906 by design (wvSplitK
targets matrix cores; gfx906 has none).

## Environment knobs

| env | default | effect |
|---|---|---|
| `GFX906_FA_LEGACY` | 1 | fp16 KV cache + in-kernel Q8 quantize; `0` = Q8 side buffer (desyncs on warmup/COW — do not use) |
| `GFX906_FA_FUSED_QUANT` | 1 | fuse quantize into the decode KV gather (bit-equal); `0` kill switch |
| `GFX906_FA_NC2` | 8 (auto-downgrade) | GQA heads packed per KV block; instantiated {1,2,8}; invalid explicit value = error |
| `GFX906_FA_KVSPLIT` | 16 | decode KV-split factor (B=1 parallelism); `1` disables |
| `GFX906_FA_CG` | UNIFORM_SINGLE_TOKEN_DECODE | FA cudagraph-support mode |
| `GFX906_FA_DIRECT_PAGED` / `_MIN_BATCH` / `_MAX_SQ` | auto / 2 / 16 | direct-paged decode path gating |
| `VLLM_GFX906_DENSE_GEMV` | 1 | M=1 dense GEMV dispatch; `0` kill switch |
| `GFX906_GDN_EMPTY_CORE_OUT` | 1 | skip the dead GDN core_attn_out zero-fill |
| `GFX906_FA_GATHER_V` | auto | gather-kernel V handling variant |
| debug: `GFX906_FA_DOUBLE_CHECK`, `_DUMP`, `_FWD_DEBUG`, `_NO_BUF_REUSE`, `_QPAD_EMPTY`, `_TORCH_GATHER`, `_ZERO_KTAIL`, `GFX906_FA_FUSED` | off | validation/debug paths |

## Known issues / limitations

- **CPU stuck-threads in TP=2 serving** (host-level, 2026-08-24): 2
  threads per worker freeze at 100% user CPU in the HSA P2P-IPC handshake
  (`libc __poll` / `libhsa IPCClientImport`) within ~15-20 min of start;
  reboot does not clear it; serving unaffected so far. Full write-up +
  options (NCCL_P2P_DISABLE A/B, AMD escalation): `cpu-stuck-threads.md`.
- **rocprofv3 finalization race**: dense-model traces consistently fail
  (ring buffer invalid at exit; EngineCore teardown races HSA). Use
  three-anchor inference or eager torch-profiler attribution instead.
- **LEGACY=0** (Q8 side buffer) lags the fp16 cache during warmup/COW/
  graph-replay writes → keep the LEGACY=1 default.
- **Layer 0's routed experts ship fp16** (checkpoint
  `modules_to_not_convert`) → Triton `fused_moe`, 414 µs/step MoE. Options
  catalogued in `moe-decode-roadmap.md` (C4).
- **GDN/mamba state copy pile**: ~180 µs/step of `[3,1,32]` copies —
  upstream state bookkeeping, deferred.
- **B>1 decode**: one 192 KB reshape copy per FA layer remains (zero-copy
  needs a decode-specialized kernel store; only matters for batched decode).
- **Eager decode is launch-bound**: eager A/B of kernel improvements can tie
  even when the kernels differ; always gate in graph/serving mode.
- **256k-context prefill OOM on Qwen3.8-27B — RESOLVED 2026-08-24
  (branch `gfx906/fa-gather-lifecycle`; history: 7 OOMs on 2026-08-23,
  full mechanism + verbatim evidence in `oom-256k-prefill.md`, fix +
  validation in `DEVLOG-fa-attention.md` "Gather-buffer lifecycle fix").**
  The unprofiled long-context transient was the custom-FA
  `_gather_retired` keep-alive dict: pre-fix, every chunked-prefill chunk
  with a larger max-seq-len reallocated the gather buffers at exact Sk and
  retired the previous capture-flagged generation (measured 7.79 GB / 152
  generations by the 60k-token OOM point — vs ~1.94 GiB headroom; the AWQ
  `temp_dq` scratch was the allocation that landed on the remains). Fixed:
  capacity-width grow-only buffers + per-generation capture flag
  (`GFX906_FA_GATHER_EXACT=1` restores the old policy). The 250k run-4
  prefill now completes with the needle retrieved (148 tok/s incl.
  prefill), decode A/B flat; **the 131k ceiling is lifted** on this model.
  Not a serving-time leak (W4 soak flat at 98-99 %). KV sizing facts:
  Qwen3.8-27B 64 KB/token (TP=1) / 32 KB/rank (TP=2); Qwen3.5-27B 20 KB;
  Qwen3.5-35B 10 KB.
- Dense GEMM dispatch (exllama gptq + LLMM1) is at its measured optimum —
  a purpose-built W4A16 dense GEMV is the top remaining dense lever
  (roadmap item).
