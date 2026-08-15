# gfx906 (MI50/MI60) benchmark results

Benchmarks run on real gfx906 hardware collected across the versions this fork
tracks (`0.23.0`, our `0.26.0` port, and current `main`), plus the prefill/
decode split that isolates the custom FlashAttention (`CUSTOM`) backend.

Hardware: single AMD MI60 (32 GB, gfx906) unless noted. ROCm 7.14 image
(`mixa3607/vllm-gfx906`: `0.27.99rc0-rocm-7.14-kintegrated`), torch 2.13.
Models cached under `/data/cache/huggingface`; all runs offline
(`HF_HUB_OFFLINE=1`), `HIP_VISIBLE_DEVICES=0`,
`VLLM_WORKER_MULTIPROC_METHOD=fork`.

---

## 1. End-to-end across versions (dense + MoE AWQ)

Method: `_bench_gfx906.py`, pp = 2048 prefill tokens, tg = 256 generated,
gpu_memory_utilization = 0.85, max_model_len = 3328, greedy (temperature 0.0),
1 warmup pass, single sample. tokens/s = tg / elapsed.

### Dense — `QuantTrio/Qwen3.5-9B-AWQ`

| Version | tok/s | Δ vs 0.23 |
|---------|-------|-----------|
| 0.23.0  | 27.47 | —         |
| 0.26.0  | 28.03 | +2.0%     |
| main    | 32.31 | +17.6%    |

### MoE — `QuantTrio/Qwen3.5-35B-A3B-AWQ`

| Version | tok/s | Δ vs 0.26 | notes |
|---------|-------|-----------|-------|
| 0.23.0  | —     | —         | Unsupported: `RoutedExperts` object has no attribute `tp_size` |
| 0.26.0  | 12.16 | —         | Baseline (legacy monolithic Triton W4A16) |
| main    | 3.49  | **−71%**  | MoE regression on main (modular pipeline + TritonWNA16Experts) |
| main + gfx906 MoE kernel (`gfx906/moe-opt`) | **18.88** | **+55%** | Custom HIP W4A16 grouped GEMM, see §4 |

Prefill/decode split (same harness; prefill = tg=1 run, decode derived as
(tg=256 − tg=1) / 255):

| Path | prefill pp=2048 | decode tok/s |
|------|-----------------|--------------|
| main (pre-fix) | ~450 tok/s (4.5 s) | 3.72 |
| main + gfx906 MoE kernel | **~2140 tok/s (0.95 s)** — 4.7× | **19.7** — 5.3× |

**Serving mode (cudagraphs, `BENCH_EAGER=0`) — not comparable to the eager
table above:** with `FULL_DECODE_ONLY` capture the same config reaches
**41.5 tok/s** end-to-end (decode ~49 tok/s, ~20 ms/step). Eager decode is
CPU-launch-bound (~1500 dispatches/step); graphs remove that. Capture needs
`max_num_seqs <= Mamba cache blocks` on this hybrid GDN model (see dev log
P2-2).

Summary: on the **dense** model the forward port is strictly non-regressing
(0.26 ≈ 0.23, main +18% faster). On the **MoE** model, main regressed badly
vs our 0.26 port (−71%); the custom gfx906 W4A16 MoE kernel (§4) fixes both
the regression and the pre-existing slowness: **3.49 → 18.79 tok/s** in eager
mode, above the 0.26 baseline, with ~5× gains in both prefill and decode.

---

## 4. Custom gfx906 W4A16 MoE kernel (branch `gfx906/moe-opt`)

The WNA16 MoE GEMM was 91% of GPU time on main (`fused_moe_kernel_gptq_awq`,
3.5 ms/call at decode). A custom HIP kernel (`csrc/rocm/moe_q_gemm_gfx906.cu`,
exllama-style: exllama-shuffled int4 weights `[E,K/8,N]`, 256 threads × 4 N
columns, `__ockl_fdot2` dots, K-split fp16 CAS-atomic epilogue with fused
top-k weight + moe_sum) replaces it via the standard modular pipeline
(`Gfx906WNA16Experts` + oracle backend `GFX906_HIP`; weights repacked in torch
at load time, shared experts untouched).

Micro-bench at Qwen shapes (E=256, top-k=8), per-call µs:

| M | Triton w13 | gfx906 w13 | speedup (w13+w2) |
|---|-----------|-----------|------------------|
| 1   | 3667 µs | **35.5 µs** | 62× |
| 8   | 25036 µs | **135 µs** | 125× |
| 32  | 67255 µs | **355 µs** | 119× |
| 512 | 169723 µs | **3063 µs** | 39× |

Post-fix profile (pp=512/tg=64): MoE GEMMs are 15% of GPU time (was 91%);
the run is now CPU-launch-bound in eager mode (Self CPU 4.3 s vs Self CUDA
2.0 s). Largest remaining per-decode-step GPU costs: aiter `LLGemm1` dense
GEMMs (~7 ms), paged attention (~2.9 ms), MoE GEMMs (~2.1 ms). Next targets:
cudagraphs (eager-only bench above), prefill tuning of the new kernel,
`LLGemm1`.

Dev log: [`DEVLOG-moe-opt.md`](DEVLOG-moe-opt.md).

---

## 2. Prefill/decode split — custom `CUSTOM` FA vs stock backend (main)

Method: `_pp_bench.py`, dense `QuantTrio/Qwen3.5-9B-AWQ`, single MI60, eager,
prefix caching on, pp/tgen two-phase timing. `prefill_tps` = fresh-pp TTFT;
`decode_tps` = prefix-cached tg only. tg = 64, gpu_util = 0.7.

### Prefill throughput (tok/s) — the phase the FA backend accelerates

| pp | `CUSTOM` | stock `ROCM_ATTN` | Δ |
| --- | --- | --- | --- |
| 256  | 590  | 575  | +2.6% |
| 512  | 757  | 764  | −0.9% |
| 1024 | 1483 | 1427 | +3.9% |
| 2048 | 1399 | 1288 | **+8.6%** |

### Decode throughput (tok/s) — essentially unchanged

| pp | `CUSTOM` | stock `ROCM_ATTN` |
| --- | --- | --- |
| 256  | 29.0 | 30.3 |
| 512  | 24.7 | 25.5 |
| 1024 | 26.1 | 26.8 |
| 2048 | 25.5 | 26.1 |

Interpretation: `CUSTOM` gains grow with prefill length (up to +8.6% at
pp=2048), and are roughly neutral at small pp. Decode is unchanged, as
expected — the custom FS kernels target prefill, not decode. Note Qwen3.5 is
an *any-attention hybrid* with only a few full-attention layers, so these
understate the FA gain versus a full-attention model.

---

## 3. Upstream gfx906-fa-vllm — full-attention model at long context

From the upstream repo's README (`https://github.com/cassettesgoboom/
gfx906-fa-vllm`), `MiniMax-M2.7-AWQ-4bit`, 8× MI50, TP = 8, BS = 1. This shows
the `CUSTOM` backend's real uplift on a full-attention model at long context,
which the hybrid Qwen3.5 numbers above do not capture.

| ctx  | `CUSTOM` TG (tok/s) | Δ vs stock `TRITON_ATTN` |
| ---  | ---                 | ---                       |
| 1K   | 27.7                | —                         |
| 32K  | 7.7                 | +6%                       |
| 100K | 3.9                 | **+32%**                  |
| 130K | 3.0                 | **+29%**                  |

On a full-attention model at long context the custom kernels deliver roughly
**+20–40%** overall throughput and stay functional where the stock Triton
kernels stall.

---

## Scripts

- `../_bench_gfx906.py` — end-to-end pp/tg bench (used for §1).
- `../_pp_bench.py` — prefill/decode split (used for §2).
- Logs: `/tmp/bench/full_run.log` (end-to-end runs), `/tmp/bench/*.log`.