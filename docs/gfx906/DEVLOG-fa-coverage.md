# FA coverage — where gfx906 does *not* get our FA, and why

## 2026-09-16 — enumeration: the fallback map, ranked

**VERDICT:** recon complete, four actionable classes identified · **GATE:** none yet (this entry is a
map; the gate belongs to whatever we change — FA suite plus a PPL/serving check for a predicate fix).

## Why

Twice in one day a model silently lost the MI50-tuned FA and nobody was told why: the DFlash2 drafter
(`non-causal attention not supported` -> ROCM_ATTN, which cannot even be CUDA-graph captured) and
Gemma-4 (`TRITON_ATTN`). The reason strings existed the whole time but were logged at `debug_once`.
With them promoted to INFO (`b06a09e978`), the coverage question becomes enumerable off-GPU — the
selector is a pure function of the attention config — so this entry answers it for the synthetic
space and for every checkpoint in the local zoo, including TP widths we cannot run locally (1/2/4/8)
and vision towers (a separate selector).

Tool: `tools/fa_coverage.py` (CPU-only, no model loads, no GPU lottery). It drives
`RocmPlatform.get_valid_backends` for text attention and `vit_unsupported_reason` for the ViT, so it
reports exactly what the engine would choose and why not CUSTOM.

## What the enumeration found

**Synthetic text matrix** (192 configs over head size x block size x sliding x non-causal x sinks x
attention type; 174 do not get CUSTOM):

| reason | rows | real-world class | our exposure |
|---|---|---|---|
| `attention sinks not supported` | 72 | GPT-OSS-style sink models | none in the zoo yet; real models exist |
| `head_size not supported` | 61 | text models with head_dim not in {64,128,256} — e.g. 96/112 | hypothetical as text, see the asymmetry below |
| `attention type encoder not supported` | 36 | encoder / encoder-decoder attention | we serve decoder-only |
| `non-causal attention not supported` | 18 | DFlash2 drafters, lookahead/bidirectional spec decode | DFlash2 parked; FA-NONCAUSAL design exists |

**Vision tower** (30 head-dim x dtype combinations; 12 fall back): `torch.bfloat16` (10) — the Q8 ViT
kernel is fp16/fp32 only — and `head_size > 256` (2). Everything else, including 72/80/96/112, is
served because the ViT path **pads** to the next instantiated kernel dim.

**Real checkpoints** (78 config reads across `/local/models`, `/data/models` and both HF caches, at
tp=1/2/4/8): 13 reads hit a non-CUSTOM config, all small text models with head_dim 40 (4 reads ->
TRITON_ATTN) or head_dim 32 (9 reads -> ROCM_ATTN). Those are the BERT/BGE-shaped embedding and
reranker checkpoints, which we run under `llama-server`, not vLLM — so the practical impact today is
nil, but they are the honest map of what a vLLM bring-up of them would cost.

## The asymmetry worth acting on

`supports_head_size` on the text backend accepts **exactly** `(64, 128, 256)`; the ViT path accepts
anything `<= 256` and pads via `_pad_head_dim`. So a *vision* tower with head_dim 96 is served by our
kernel (at the padded cost — that is VIT-2's subject), while a *text* model with head_dim 96 is
rejected outright and runs ROCM_ATTN. Mirroring the ViT's padding rule in the text path turns a
silent ~3-10x step cost into a bounded arithmetic overhead, and the paging is ours to declare
(`get_kv_cache_shape`), so the padded layout is expressible.

## Ranked next steps

1. **The guard** (durable, cheap): make a non-CUSTOM selection for a gfx906 text attention layer a
   loud, once-per-engine *warning* naming the reason, plus an opt-in fail-closed switch. This is what
   turns the next silent fallback into a log line — it is the piece that pays forward on every future
   model onboarding.
2. **Text head-dim padding** (mirror `_pad_head_dim`): removes the whole `head_size not supported`
   class. Gate: FA suite + PPL on a padded dim + a serving A/B.
3. **Sinks** (72 rows, the largest class): needs a kernel feature (an extra learned logit per head),
   so it is real work — worth it only when a sink model (GPT-OSS-style) is actually wanted.
4. **Non-causal** (18 rows): already designed and *experimentally refuted as an acceptance fix* for
   DFlash2 (`DEVLOG-fa-noncausal.md`); implement only if DFlash2 is revived for performance.
5. **Encoder attention / enc-dec** and **bf16 ViT**: leave documented; both are out of our serving
   envelope.

## Reproduce

```
.venv/bin/python tools/fa_coverage.py                      # matrix + every local checkpoint
.venv/bin/python tools/fa_coverage.py --skip-matrix        # checkpoints only
.venv/bin/python tools/fa_coverage.py --models /data/models --tp 1,2,4,8
```
