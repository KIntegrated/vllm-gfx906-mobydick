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
| 0.26.0  | 12.16 | —         | Baseline |
| main    | 3.49  | **−71%**  | MoE regression on main |

Summary: on the **dense** model the forward port is strictly non-regressing
(0.26 ≈ 0.23, main +18% faster). On the **MoE** model, main regresses badly
vs our 0.26 port (−71%), which is a target for investigation.

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