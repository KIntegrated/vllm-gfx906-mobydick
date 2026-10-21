# gfx906 fork — 0.32.0 release snapshot (2026-10-21)

**What this is.** The fork's release on the upstream **0.32.0** base: branch
`gfx906/v0.32.0`, cut from the 0.31.0-final `main` (`3b5f4d4cd5`) and merged with
upstream `v0.32.0` (`5df63a7203`) as merge commit **`1e514c6fd0`** — 822 fork
commits / 840 upstream commits, **59 conflicted files**. Per-file resolution
record: `MERGE-0.32.0-review.md`; theme log: `CHANGELOG.md` 2026-10-18..21.
Promoted to `main` 2026-10-21 at `5f41d480b9`.

## Gates (all measured on this box, this bootstrap)

| gate | result |
|---|---|
| build | `rc=0`; `_C_stable_libtorch*` loads with all four fork ops (schemas verified) |
| static sweep | no conflict markers, no `F821`/`F811` across `vllm/ tests/ csrc/` |
| PPL MoE 35B | **15.9555** (band 15.94–16.02), 0 top-20 misses |
| PPL dense 27B | **14.3750** (its own baseline, exact) |
| PPL Lightning mixed INT4/INT8 | **26.9922** (band 26.96–27.02) |
| `tests/platforms/` | 35 passed (15 `test_cpu.py` fail on this host **identically with changes reverted** — pre-existing) |
| `tests/quantization/test_moe_wna16.py` | 51 passed, 9 skipped (CUDA/XPU-only) |
| `tests/config/test_qwen4_exp_family_match.py` | 43 passed |
| Lightning serving (TP=2, EP, MTP k=3) | `Application startup complete`, `GFX906_HIP` WNA16 MoE backend, V2 model runner, packed block 2096→2672, KV **3,042,304 tokens**, graphs PIECEWISE 2/2 + FULL 2/2 |
| 35B serving (TP=2, MTP k=3) | `Application startup complete`, packed `+30.6 %`, KV **1,065,870 tokens** (32.5× @32k), graphs 7 s / 0.42 GiB, 2/2 requests HTTP 200 |

## What the merge needed beyond conflict resolution

Five marker-less breakages (the expensive class — a clean `grep '<<<<<<<'`
proves nothing):

1. `csrc/cache_kernels.cu` lost its `cuda_utils` include.
2. Upstream's `csrc/cuda_utils.h` silently dropped `cuda_utils::ceil_div` **and**
   `HOST_DEVICE_INLINE` → restored with a `VLLM_CEIL_DIV_QUALIFIER`
   (`__host__ __device__` only under `__CUDACC__`/`__HIPCC__`).
3. Ported op registrations aborted the extension (`aoti_torch_library_def API
   call failed`) until moved to the **top** of the `_C` fragment — diagnosed by
   tagging every def/impl with a line-numbered `fprintf`.
4. `speculative.py` / `config/vllm.py` imported the fork's Qwen4Exp helpers from
   a module upstream deleted → restored dependency-free in
   `vllm/models/qwen4_exp/family.py`.
5. The QSA module was unimportable (the installed transformers predates
   `Qwen4ExpTextConfig`).

Plus two post-merge model fixes: the `ckpt_*_proj_name=None` scheme check
(`96db9355c3`, #68) and the moe_wna16 qzeros loader under expert parallelism
(`5f41d480b9`, #69 — EP reports `tp_size=1` while the checkpoint ids are global;
both now handled).

## Breaking changes the base brings (fork-relevant)

- **`csrc/libtorch_stable/*` → `csrc/*` flattening.** 187 fork files affected;
  the fork's **4 op registrations** were re-ported into `csrc/ops.h` /
  `csrc/torch_bindings.cpp`. `csrc/moe/moe_wna16.cu` stays **unconditional**
  (upstream made it CUDA-only; the gfx906 WNA16 path needs it on ROCm).
- **QSA width validator removed** → #37's LDS guard is now load-bearing: a large
  indexer budget reaches `top_k_per_row_decode` and fails closed at
  `topK ≤ 8192` instead of corrupting memory (#67 added the config-time check).
- **MRV2 is the ROCm default** and the model-specific MRV1 exceptions were
  removed (#56360) — V1 still exists as an escape hatch, verified with
  Muse-Glimmer + DFlash assistant (#65).
- **Removed upstream:** cascade attention, `mamba_cache_mode="all"`, the
  torch.compile SP/AsyncTP passes, and a set of deprecated APIs.
- **`max_tokens` is an upper bound, not a reservation (#57035)** — benchmark
  clients must not be compared against 0.31 t/s numbers naively (#66).
- **transformers ≥ 5.16.1, < 5.20.0**; ROCm deps AITER 0.1.24.post1 / MoRI
  1.2.3.post1 taken from upstream (our usage of both is limited).

## Packed KV on this base — unchanged verdict, one real bug fixed

The `auto` gate had been **silently off under MTP spec decode** (our default
recipe): `_align_hybrid_block_size` returned early when the attention page already
equalled the mamba page, leaving `mamba_page_size_padded` `None`, so the aliased
arm read `(1072, None)` against the packed arm's `(1408, 1,103,872)` and the
prediction short-circuited to **+0.0 %**. Fixed in `7fcd4205b1`; boot now logs
`enabled … +28.4 % (block 1072 -> 1408)`.

Served decision gate on this base (TP=2, MTP k=3, util 0.85, 32k): **890,070 →
1,057,678 tokens (+18.8 %)** against +28.4 % predicted — the prediction remains an
upper bound, same factor as the 0.31 pair. Verdicts re-anchored: 35B 8k +33.6 %
(no spec) / +32.6 % (MTP), 32k +28.4 %; Lightning 8k +0.4 % (off), 16k +14.8 %
(on). Details: `DEVLOG-packed-kv.md`.

## Not validated here (name them, do not claim them)

QSA / Qwen3.8-Flash-Next (no servable checkpoint on this box), DeepSeek-V4/V4.1
and Kimi-K3 AMD paths, AITER MegaMoE, RDNA (`on_gfx1x`) items, PP/PCP/DCP and
KV-connector changes, SM90/SM100-only kernels, XPU/CPU backends, the Rust
frontend.

## Follow-ups opened by this base

#61 (adopt upstream int4/int8 per-token-head KV), #62 (Mamba prefill
checkpoints), #63 (mine ROCM_SEGMENTED_ATTN — tuner/cache infrastructure; the
per-shape `nc2`/`kvsplit` record module landed as slice 1, `51ce08adbe`), #64
(INT4 `ultraquant_4bit` KV), #66 (pins + `max_tokens` audit), and the
packed-cell-as-cache-format design folded into #61/#64. Closed on this base:
#60, #65, #67, #68, #69.
