# #44 — Qwen4Exp expert streaming: where it stands

**VERDICT (2026-10-26, recon + two measurements): the port's *core* is already in
our tree, stock expert offload is unusable at B=1 on this box (-85 % UVA, -99 %
prefetch), and the only real blocker left is the checkpoint (120 GiB, not local).
The remaining delta is the hot-set + CPU-side expert compute, which has no ROCm
equivalent.**
**GATE:** two boots, no code (staging steps 1-2 of the issue).

## What the issue asked, and what changed since it was filed

The issue (2026-10-06) proposed adapting
`DominikBucko/qwen38-flash-next-2x3090`'s host-resident expert design. Since then
**vLLM 0.32 grew the same offloader machinery**, so the "port 4,000 lines of
overlay" framing is obsolete:

| their component | in our tree today |
|---|---|
| `offloader/uva.py` | `vllm/model_executor/offloader/uva.py` + `--cpu-offload-gb` |
| `offloader/prefetch.py` | `prefetch.py` + `--offload-backend prefetch --offload-group-size/--offload-num-in-group/--offload-prefetch-step` |
| expert-only targeting | `--cpu-offload-params experts` (segment matching; the config's own example is `mlp.experts.w2_weight`) |
| hot-set selection (84 experts/layer resident) | **absent** |
| CPU-side expert compute (`cpu_moe.cpp`, `bridge_cuda.cu`) | **absent, and CUDA/C++ — no ROCm path** |

So the port reduces to the two absent rows, and the second is the hard one.

## Measurement: what stock expert offload costs at B=1

`Qwen3.5-35B-A3B-AWQ`, pp256/tg128, B=1, util 0.90, mclk 1000, three samples each,
fresh process per arm (see `/local/tmp/offload_probe.py`):

| arm | t/s (3 samples) | vs base |
|---|---|---|
| base | 81.05 / 81.02 / 80.98 | — |
| `--cpu-offload-gb 6 --cpu-offload-params experts` (UVA) | 12.32 / 12.31 / 12.34 | **-85 %** |
| `--offload-backend prefetch --offload-group-size 8 --offload-num-in-group 2 --offload-params experts` | 0.74 / 0.74 / 0.74 | **-99 %** |

Both arms loaded and ran (UVA logged `Total CPU offloaded parameters: 6.04` GiB of
experts), so this is behaviour, not a misconfiguration. UVA is zero-copy: every
expert access crosses PCIe with no prefetch, which is why it collapses. The
prefetch arm is a further ~17x slower — most likely because the group of 2
offloaded layers per 8 is touched *every* step, so its H2D cannot be hidden at
B=1; that hypothesis is untested and is the first thing to check if this axis is
pursued.

**Consequence for the design:** "offload the experts" as a wholesale strategy is
dead on this box. Only a **hot-set** (most experts resident, misses streamed) can
be viable, which is precisely the absent row — so the issue's original instinct
was right, and the staging order should change: the hot-set is the *first* thing
to prototype, not the last.

## Correction to the issue's premise: the link is x16, not x8

The issue says "our two MI50s have usable P2P over x8 PCIe (the link width is
deliberate)". Measured from sysfs:

```
GPU 04:00.0: 16x @ 16.0 GT/s  (max 16x @ 16.0 GT/s)   # PCIe 4.0 x16 ≈ 32 GB/s
GPU 07:00.0: 16x @ 16.0 GT/s  (max 16x @ 16.0 GT/s)
```

That is **4x the bandwidth the issue budgeted**, and it moves the feasibility
arithmetic in the design's favour: streaming a 120 GiB checkpoint's cold experts
per prefill chunk is now governed by ~25 GB/s, not ~6 GB/s.

## The blocker: the checkpoint

| repo | files | size | access |
|---|---|---|---|
| `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` (their published weights) | 88 | **120.1 GiB** | public, ungated |
| `Qwen/Qwen3.8-Flash-Next` (bf16 base) | 144 | **335.3 GiB** | public, ungated |

Local state: **refs-only stub, no weights**. Disk has 374 GiB free on `/data`, so
the 120 GiB W4A16 fits (note it is ~2x the "~66 GiB" the issue assumed).

## Revised staging

1. **Get the checkpoint** (120 GiB W4A16). Nothing below is measurable without it.
2. **Hot-set prototype, prefill-only** — keep N experts/layer resident, stream the
   rest per chunk over the x16 link; measure on a single long prefill. This is
   where the design either works or does not.
3. Only if (2) clears: consider CPU-side expert compute (the CUDA-only part, no
   ROCm equivalent — likely the real stop sign).
4. Stock `--cpu-offload-*` is **not** a stepping stone: measured -85 %/-99 %.
