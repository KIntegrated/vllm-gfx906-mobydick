# gfx906 (MI50/MI60) optimization hub
Copyright Kevin Read <me@kevin-read.com>

This fork optimizes vLLM for AMD **gfx906 (Vega 20)** — MI50/MI60, 60 CUs,
32 GB HBM2 (~800 GB/s), **no MFMA, no int8 matrix cores**. All numbers are
measured on a single MI50 with ROCm 7.14, torch 2.13, single request,
`pp=2048`/`tg=256`, `cudagraph_mode=FULL_DECODE_ONLY`.

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

Correctness gates: PPL on a fixed 442-token probe — MoE band 6.6817–6.6942,
dense band 6.6993–6.7197. Kernel test suites: 15/15 FA, 12/12 MoE GEMM.

## Contents of this directory

| file | what it is |
|---|---|
| `README.md` | this hub: changes, numbers, recipes, knobs |
| `DEVLOG-moe-opt.md` | full development record (history/archive) |
| `moe-decode-roadmap.md` | future MoE-decode candidates (roadmap, not a committed plan) |
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

### 3. Dense M=1 W16A16 GEMV
`csrc/rocm/dense_gemv_gfx906.cu`, dispatched from
`vllm/model_executor/layers/utils.py` (`_llmm1_tiny_m`, `_gfx906_gemv_long_k`;
kill switch `VLLM_GFX906_DENSE_GEMV=0`):

- m<4 decode GEMMs → padded LLMM1 / GEMV (−401 µs/step MoE).
- m==1 rows → GEMV RPT=1 (kills the per-step `F.pad` of the shared-expert
  gate weight; 4.7× isolated, bit-equal).
- Long-K n==1 GEMV (K=17408 down_proj, KCHUNK=1024/RPT=2, fp16 atomic
  K-split): 227.6 µs = 100% of HBM floor vs 794 µs triton_matmul.
- Verified at the HBM floor on all dense fp16 shapes (K=5120 lm_head probe:
  neutral — LLMM1 already at floor there).

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

The last file has 9 pre-existing failures on real gfx906 hardware: its
platform-mock tests monkeypatch `on_gfx1x`/`on_gfx9`/etc. but not
`on_gfx906`, which is True on this machine (identical failures with and
without our changes).

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
- Dense GEMM dispatch (exllama gptq + LLMM1) is at its measured optimum —
  a purpose-built W4A16 dense GEMV is the top remaining dense lever
  (roadmap item).
