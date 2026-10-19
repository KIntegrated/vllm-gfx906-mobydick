# Packed KV layout (E2) for hybrid models — page geometry and the align-mode ceiling

**VERDICT:** OPEN · **GATE:** the geometry dry run (`docs/gfx906/_hybrid_kv_ratio.py`)
reproduces the boot logs exactly; the *pool* gate (served KV-token A/B on an
uncapped pool) is still missing for the Lightning class.
**Refs:** LAB-9 (PR #43), LAB-10 (#58), `_hybrid_kv_ratio.py`.

## HYPOTHESIS

If the mamba state is ~90 % of a hybrid model's align-mode KV page, then
shrinking the attention cell by 22 % can buy at most ~1-2 % capacity, and the
packed layout should stay opt-in for every hybrid class (LAB-10's premise).

## What was done

Added `docs/gfx906/_hybrid_kv_ratio.py`: a CPU-only dry run of
`_align_hybrid_block_size` for both KV layouts that prints attention bytes/token,
the mamba state page, the derived block size, the unified page and the
mamba:attention split of one block. `--expect` fails the run unless the derived
block sizes match the boot logs, so the tool cannot drift silently.

```bash
env HF_HUB_OFFLINE=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  .venv/bin/python docs/gfx906/_hybrid_kv_ratio.py <model> --expect <alias>,<packed>
```

Run on both hybrid classes we serve, `--mamba-cache-mode align`, fp16,
`max_model_len` 8192:

| class | attn B/token | mamba B/block | block | page/block | mamba share | pool-limited gain |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (30 GDN + 10 attn of 40) | 2048 → **1568** (−23.4 %) | 2,146,304 | 1056 → **1376** | 2,162,688 → 2,157,568 | 99.24 % → 99.48 % | **+30.6 %** |
| Nemotron-3.5-Lightning-30B-A3B (52 layers) | 1024 → **800** (−21.9 %) | 2,134,016 | 2096 → **2672** | 2,146,304 → 2,137,600 | 99.43 % → 99.83 % | **+28.0 %** |

Both rows reproduce the recorded boot logs, including the mamba padding the
aligner reports (35B 0.76 % → 0.52 %; Lightning 0.58 % → 0.17 %).

## Evidence FOR the premise (mamba dominates the page)

In align mode the aligner *pins* the unified page to the attention page and pads
mamba up to it, so mamba is **99.2–99.8 %** of every block's bytes in both arms —
the mamba:attention byte ratio of a block is ~99:1, exactly as feared. Any
"save a fraction of the page" reasoning therefore predicts ~0.2–0.8 %.

## Evidence AGAINST the premise (the win is block granularity, not page bytes)

The capacity that matters is **tokens per mamba state block**, and the aligner
grows the block until the attention page covers the (block-independent) mamba
page. A smaller attention cell therefore packs more tokens into the same mamba
state: 1056 → 1376 (+30 %) and 2096 → 2672 (+28 %), i.e. the gain equals the cell
saving minus the (smaller) padding, not the mamba share. Measured on the served
35B: **142,987 → 174,762 KV tokens (+22.2 %)**, consistent with the +30.6 %
page-limited figure once the non-KV VRAM is subtracted. So for align mode the
LAB-10 ceiling is **the cell saving (22–23 %)**, not 1–2 %.

## The one number that does not fit, and why it does not settle the question

Lightning's served A/B gave **526,336 → 528,384 (+0.4 %)** with the block moving
2096 → 2672. Both arms report the same `Available KV cache memory: 9.25 GiB`,
and the reported token counts are ≈ 64.25 / 64.5 × 8192 — i.e. both look capped
by request capacity (~64 concurrent 8k sequences), not by VRAM. A page-limited
pool cannot move the block size 28 % and gain 0.4 %, so the Lightning arm was
**capped**, and its +0.4 % says nothing about the packed ceiling.

Open step (why this stays OPEN): one served run per arm on the Lightning class
with the cap lifted (`--max-num-seqs` / `--max-model-len` set so VRAM binds),
logging `num_gpu_blocks` and the per-group page sizes. Only then is the LAB-9
auto rule ("packed off when the block size moves") testable: the 35B suggests it
is over-conservative where the block move *is* the win.

## Note on the recorded block sizes

`ROADMAP.md` recorded "35B 1056 → **2112**". A *fresh* dry run gives **1376**;
2112 is the second-pass value when the aligner starts from an
already-raised `block_size` (1056), which is the same wart the LAB-9 notes
describe ("reports different would-be values (2672 vs 4192, 1376 vs 2112)").
The boot logs agree with the fresh values (1376 / 2672), so the roadmap record is
corrected to match.
