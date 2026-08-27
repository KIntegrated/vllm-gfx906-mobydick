# Muse-Glimmer-30B onboarding — CT-asym W4A16 dense text + iRoPE hybrid attention on gfx906

> Branch `feat/muse-glimmer` off `main` (4a9e24b5c) · model
> `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` · date 2026-08-26 · onboarding +
> sliding-window CUSTOM FA track.

**Model:** 52 text layers, hidden 6656, Hq 32 / Hkv 2 (GQA 16:1), head_dim
128, sliding_window 2048, fp16 KV. iRoPE: 13 full-attention NoPE layers
(indices 3,7,…,47,51) + 39 sliding-window(2048) RoPE layers. Vision tower
50 layers (all unquantized fp16), lm_head + layers 47/51 linears unquantized
(315 ignored tensors, CT `ignore` list). Dense CT-W4A16-asym (group 32,
uint4, int8 zp) → `CompressedTensorsWNA16` → Exllama `gptq_gemm`
(`use_v2_format=True`, raw zp).

**Baseline attention (no code change):** per-layer auto-select gives
sliding group → `ROCM_ATTN` (Triton `unified_attention`), full group →
`CUSTOM` (Q8 FA); ViT → `FLASH_ATTN`. Log: `rocm.py:748`
"Found incompatible backend(s) [CUSTOM, TURBOQUANT] … Overriding with
ROCM_ATTN".

## 2026-08-26 — onboarding: load + smoke + hybrid baseline

**VERDICT:** OPEN (baseline established; window work pending) ·
**GATE:** `_bench_gfx906.py` serving wall-clock, pp2048/tg256, 4 samples,
`BENCH_MAX_SEQS=4`, `BENCH_GPU_UTIL=0.93 BENCH_KV_MEM=4634016400`,
graph mode. (Util-only sizing OOMs the first request in warm-cache runs:
profiling peak ~0.16 GiB lower than cold → KV pool oversized for the
532 MiB inductor prefill buffer — 27B's was 356 MiB, so 0.93/0.95 are
both too tight here; explicit 4.32 GiB KV cap per the engine log's
suggestion, also makes A/B arms pool-identical. Logs:
`/local/tmp/muse/bench_hybrid_graph{,_093,_093_r2}.log`.)

## HYPOTHESIS

If the AWQ checkpoint loads through the existing CT-asym WNA16 (Exllama)
dense path and the per-layer backend auto-select routes sliding layers to
ROCM_ATTN, the model serves coherent output with no code change — i.e.
upstream MuseGlimmer support (PR #51655) + the shipped gfx906 quant path is
sufficient for a working baseline, and the only gap is sliding-window
support in CUSTOM FA (perf, not correctness).

## What was done

- `feat/muse-glimmer` from `main` (upstream `muse_glimmer` model/config/
  processor/reasoning+tool parsers already merged, PR #51655; registry maps
  `MuseGlimmerForConditionalGeneration` → `("muse_glimmer", …)`).
- Download: original target `Vishva007/…-W4A16-AutoRound-GPTQ` died in a
  host crash (group-64 sym GPTQ); switched to
  `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` (group-32 CT-asym W4A16 — same
  kernel family as the shipped Ornith MoE path, but **dense** linears).
- Cache layout incident: my `hf download --cache-dir /data/cache/huggingface`
  put blobs in the NFS cache root, but `~/.cache/huggingface` →
  `/local/cache/huggingface` (local disk) is where transformers resolves
  (its `hub/` snapshot already had the small files from the pre-crash
  attempt). Copied the 5 shards + tokenizer.json into the `/local` snapshot
  (23 GB, now on local disk).
- Smoke (text-only `LLM`, `max_model_len=8192`, gpu_util 0.93, 4 greedy
  prompts via native chat template, thinking ON): loads in 58 s from local
  disk (22.55 GiB weights), coherent + correct answers (Paris; Rayleigh
  scattering; haiku drafting; 17·23=391). Log `/local/tmp/muse/smoke_hybrid.log`.
- Baseline bench: config above, log `/local/tmp/muse/bench_hybrid_graph_093_kv.log`.
  Harness gained a `BENCH_KV_MEM` hook (documented in README/running.md but
  missing from the harness file).

## Evidence — FOR

- Coherent, factually correct greedy output on 4 prompts incl. arithmetic;
  thinking turns emitted with the native ` to=self` protocol the reasoning
  parser expects.
- Backend select as predicted (sliding→ROCM_ATTN, full→CUSTOM); ViT
  FLASH_ATTN.
- **Hybrid baseline (graph mode, the gate):** 17.45 / 17.46 / 17.44 t/s
  (samples 1–3; sample 0 12.40 cold), pp2048/tg256, BENCH_MAX_SEQS=4,
  BENCH_GPU_UTIL=0.93 + BENCH_KV_MEM=4634016400. Qwen3.8-27B reference on
  the same harness: ~25.2 t/s (all-attention CUSTOM; no sliding layers).
  (Re-verified at the matched A/B config: 17.54 t/s.)

## Interactions / superseded-by

- Dense CT-asym Exllama dequant is **not** previously validated on gfx906 —
  the Ornith log (DEVLOG-ornith-wna16.md) covered MoE experts only. Smoke
  coherence is weak evidence for the (q−zp)·scale zero-point math; a
  numerics check vs transformers (logprob delta) is the follow-up if any
  doubt appears.
- Sliding-window CUSTOM FA landed in the next entry (SHIPPED, 1.59×);
  the open follow-up is step C — gather windowing (`kv_start` clip) for
  long-context decode, where the mask-only step still scans [0, seq_len).
  At the 2048-token window it is irrelevant for contexts ≤ window, so it
  only matters for long-ctx serving.

## 2026-08-26 — sliding-window CUSTOM FA (mask-only)

**VERDICT:** SHIPPED (1.59×) · **GATE:** same serving A/B as above,
all-52-layers CUSTOM arm vs the hybrid baseline.

## Gate result (2026-08-27 00:0x, matched config)

Both arms: graph mode, pp2048/tg256, 4 samples, BENCH_MAX_SEQS=4,
BENCH_KV_MEM=805306368 (15k-token pool ≫ 2816 needed),
BENCH_BATCHED_TOKENS=1024 (see the memory forensics below — 2048-chunk
prefill + 4.32 GiB KV OOMs the all-CUSTOM arm on the first request).

| arm | sample 0 (cold) | samples 1–3 (stable) |
|---|---|---|
| hybrid (sliding→ROCM_ATTN pinned, full→CUSTOM) | 12.43 | **17.54 t/s** (17.556/17.542/17.521) |
| all-CUSTOM (52/52 layers, window=2048) | 16.86 | **27.90 t/s** (27.931/27.901/27.874) |

**1.59×** decode wall (tg256). Hybrid stable matches the pre-change
baseline (17.45) — the harness-config delta (chunk size, KV cap) does not
move decode. Backend select verified in both logs (per-group `info` lines;
`_cached_get_attn_backend` dedupes identical configs, so one line per
group): all-CUSTOM shows both groups → CUSTOM, hybrid shows full → CUSTOM
+ sliding → ROCM_ATTN (kind-pin rides the "selected via
--attention-backend" log path). The Qwen3.8-27B all-CUSTOM reference on
the same harness is ~25.2 t/s — Muse 30B now runs ahead of it.

## HYPOTHESIS

If the Q8 FA tile kernel gets a per-row sliding cutoff (mask keys older
than `q_abs_row - W + 1`, re-using the existing `q_abs_offset` inline-
causal machinery, passed for every windowed batch including decode), all
52 Muse layers run on CUSTOM and decode recovers part of the 17.45→25.2
t/s gap the Triton sliding path costs.

## What was done

- `fattn-q8.cuh` / `fattn-q8-paged.cuh`: tile-iter + global kernels take a
  scalar `window`; both causal branches extend to
  `k_pos > q_abs_row || (window > 0 && k_pos < q_abs_row - W + 1)`.
  Fully-masked tiles are safe: `KQ_max` seeds at `-FLT_MAX/2` (finite), so
  an all-`-INF` tile leaves max unchanged and `val=exp(-INF)=0` — no NaN
  guard needed (plain causal could never hit an all-masked tile; a window
  can).
- `gfx906_fa_launcher.cu` (+ the cpp's own extern decls): `window` plumbed
  through `gfx906_fa_launch{,_paged}`.
- `gfx906_fa.cpp`: `forward` / `forward_paged_direct` gain `window=0`
  (pybind args added).
- `gfx906_fa_backend.py`: `Gfx906FAImpl` stores `sliding_window` (was
  `NotImplementedError`); backend class now
  `supports_sliding_window() -> True`.
- `gfx906_fa_paged.py`: `forward_paged(window=0)`;
  `need_causal = max_seqlen_q > 1 or window > 0` (decode rows need
  `q_abs_offset` for the per-row window; the causal check itself is a
  no-op there since all keys ≤ seq_len-1 = the decode row).
- Tests: `test_forward_sliding_window_vs_torch_ref` ×4 (D=128 GQA 16:1
  model shape W=128/64, W>L inert, D=256; decode + per-row prefill vs
  torch ref, plus window-bites / window-inert sanity). 32/32 suite green
  (28 pre-existing untouched).

Mask-only step (B of the plan): FA still scans [0, seq_len) — long-ctx
windowing (gather `kv_start` clip) is the follow-up (step C), gated by the
same A/B plus a long-context curve.

## Memory forensics — all-CUSTOM arm OOMs on the first prefill

Symptom: the all-52-CUSTOM bench OOMs on the 532 MiB inductor prefill
buffer at the first request (hybrid arm OOM'd once too, 23:00:36, and
recovered via empty_cache; all-CUSTOM never recovers). Probe
(`/local/tmp/muse/probe_mem.py`): post-init allocated 24.22 / reserved
26.06 GiB (weights 22.55 + graphs 0.71 + ~1 GiB). The prefill adds:

- **per-layer `q_pad_buf` [B, Hq=32, Sq_pad, D=128] fp32** — decoded
  batches were growing its dim0 to the capture max (B=8) although Sq=1
  never READS it (both branches use `q_pad_decode_buf`): a B=8 capture
  then ballooned every layer's prefill buffer to (8, 32, 2048, 128) =
  268 MB → 14 GiB across 52 layers. **Fixed**: `_ensure_forward_buffers`
  sizes dim0 with `qpad_num_seqs = num_seqs if max_seqlen_q > 1 else 1`
  (decode never reads `q_pad_buf`).
- per-layer C++ `o_bshd` FA output [B, Sq, 32, 128] fp32 = 33.5 MB at
  Sq=2048 → ~1.7 GiB across 52 layers while the prefill graph holds
  them (hybrid's 39 Triton layers emit fp16, ~2× smaller).

Residual budget: at KV cap 0.75 GiB + `BENCH_BATCHED_TOKENS=1024`
(prefill chunks 1024 instead of 2048 → q_pad/o_bshd per layer halve,
inductor workspace smaller) the first prefill should clear. KV pool
size is irrelevant to single-request decode wall (2816 tokens ≪ pool),
so the A/B arms just share the same capped pool. Long-term levers (not
for this gate): fp16 `o_bshd` in the Q8 FA forward (halves the per-layer
output), and/or a worker-shared prefill q_pad arena.

## Evidence — FOR / AGAINST

FOR: 32/32 FA suite (4 new window tests green); gate A/B above (1.59×);
all-CUSTOM smoke coherent + correct (4 prompts, thinking ON, 7.4 s for
4 completions vs 12.7 s hybrid). AGAINST: (none).

---
Copyright Kevin Read <me@kevin-read.com>
