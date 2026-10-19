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

## Decision rule (2026-10-19, corrected same day by the block-level probe)

`auto` must key on **usable** capacity, not raw pool tokens. A request holds
`ceil(max_model_len / block_size)` blocks, so usable tokens are
`pool / page * max_model_len / blocks_per_req`; the block rounding decides whether
a smaller cell is usable at all. The rule is therefore

```
gain = (page_off * blocks_per_req_off) / (page_on * blocks_per_req_on) - 1
```

keep packed when `gain >= 3 %` (`_gfx906_packed_capacity_gain`,
`RocmPlatform._GFX906_PACKED_MIN_GAIN`, unit test
`tests/platforms/test_gfx906_packed_auto.py`).

### Step 2: the block-level probe (in-process, `num_blocks` + per-group pages)

Nemotron-3.5-Lightning-30B-A3B, `--mamba-cache-mode align`, fp16, max_model_len
8192, util 0.70, identical 2.02 GiB pool in both arms:

| arm | block | page/block (5 groups) | num_blocks | pool tokens | **usable at 8k** |
|---|---|---|---|---|---|
| off | 2096 | 10,731,520 | 202 | 423,392 | 413,696 (+0.4 %) |
| on | 2672 | 10,688,000 | 203 | **542,416 (+28.1 %)** | 415,744 |

The raw pool really does gain the predicted +28 %, but `ceil(8192/2672) = 4 =
ceil(8192/2096)`, so a request holds 4 blocks either way and the *usable* gain is
+0.4 % — exactly the served datum this issue started from. So that +0.4 % was
**not** a capping artifact (my earlier reading); it is the correct usable
capacity, and the raw-pool figure is what is misleading.

Live verification of the corrected rule, one boot per class:

```
Lightning-30B-A3B, max_model_len 8192:
  packed KV layout left off under gfx906_fa_packed=auto:
  predicted KV capacity +0.4 % (block 2096 -> 2672).
Qwen3.5-35B-A3B, max_model_len 8192:
  packed KV layout enabled under gfx906_fa_packed=auto:
  predicted KV capacity +33.6 % (block 1056 -> 1376).
```

The verdict is per (model, max_model_len), not per model class: Lightning at
16384 predicts +14.8 % (8 blocks/request -> 7) and would be enabled, while the
35B wins at both lengths. The 35B's served A/B (+22.2 %) is the measured
counterpart of its prediction.

## Note on the recorded block sizes

`ROADMAP.md` recorded "35B 1056 → **2112**". A *fresh* dry run gives **1376**;
2112 is the second-pass value when the aligner starts from an
already-raised `block_size` (1056), which is the same wart the LAB-9 notes
describe ("reports different would-be values (2672 vs 4192, 1376 vs 2112)").
The boot logs agree with the fresh values (1376 / 2672), so the roadmap record is
corrected to match.

## 2026-10-19 — 16k served check: off-arm measured, on-arm did not engage (open)

Served `Nemotron-3.5-Lightning-30B-A3B`, `--mamba-cache-mode align`, max_model_len
**16384**, util 0.85:

- `--gfx906-fa-packed off`: block 2096, `Available KV cache memory: 6.68 GiB`,
  **`GPU KV cache size: 570,368 tokens`** (34.81x concurrency).
- `--gfx906-fa-packed on`: **did not engage** — block stayed 2096, `6.68 GiB`,
  **570,368 tokens**, and there is **no** `packed KV layout enabled` /
  `KV cell` line in the log. The API process logged
  `Failed to get total memory via amdsmi, falling back to torch.cuda`, i.e. this
  was an amdsmi-degraded boot.

So the 16k served confirmation of the predicted +14.8 % is **not** obtained, and
the attempt turned up a separate defect worth its own issue: an explicit
`--gfx906-fa-packed on` can silently fall back to the alias layout (fail-open) on
a boot where amdsmi misbehaves, with no warning at the packed layer. The 8k
prediction was validated at block level (202 -> 203 blocks, page sums
10,731,520 -> 10,688,000, usable +0.4 % matching the served datum), so the
formula stands; only the 16k *served* datum is missing.

Next: retry the 16k on-arm on a healthy boot (amdsmi normal), and treat the
silent fallback as a bug (log a warning whenever the requested mode is not the
resolved one).

## 2026-10-19 — fail-open fixed: an explicit `on` that cannot be applied now warns

Root cause of the 16k on-arm falling back silently: the platform publishes
`gfx906_fa_packed` from `check_and_update_config`, but that publish is gated on
`on_gfx906()`. A degraded boot (the API process logged `Failed to get total memory
via amdsmi`) fails the device query in the process that builds the config, so the
publish is skipped, `_PACKED_CONFIG_MODE` keeps its `"off"` default, and
`customize_spec` takes the alias branch without logging anything — the explicit
flag is dropped.

Fix (`f7a3d1d…`): `check_and_update_config` now warns once when the requested
mode is `on` and gfx906 is not detected in this process, naming the cause and the
remedy. `auto` stays silent (it is a preference, not a request). Test:
`tests/platforms/test_gfx906_packed_auto.py::test_explicit_packed_on_warns_when_gfx906_is_not_detected`
(monkeypatched `on_gfx906` -> False, asserts the warning).

Still open on this path: retry the 16k on-arm on a healthy boot to close LAB-10's
last served datum.

## 2026-10-19 — 16k datum obtained; the real bug is config-field propagation (no reboot needed)

Retried on the same boot. `--gfx906-fa-packed on` again served the **alias** layout
(block 2096) and the new detection warning did **not** fire, so `on_gfx906()` was
healthy — the earlier "degraded boot" explanation was wrong. Re-run with the
**env var** `GFX906_FA_PACKED=1` on the identical command line: packed engages
(`KV cell 400 B`, block **2672**). So:

> `CacheConfig.gfx906_fa_packed` (`--gfx906-fa-packed on`) does not reach the
> process that resolves the KV layout, while the env override does. The
> config-field path is silently ineffective for the explicit `on` (and is the
> documented interface).

16k served A/B (Lightning, align, max_model_len 16384, util 0.85, same 6.68 GiB):

| arm | block | logged usable tokens | vs off |
|---|---|---|---|
| off | 2096 | 570,368 | — |
| on (env) | 2672 | **610,577** | **+7.05 %** |

So the 16k prediction (+14.8 %) is an **upper bound**: the real gain is +7.1 %,
consistent with the block-rounding theory in direction (8k +0.4 % → 16k +7.1 %)
but over-stated in magnitude because usable tokens quantise into whole
`max_model_len` requests per group. The 3 % threshold keeps both verdicts right
(8k off, 16k on).

Open, in order: (1) fix the config-field propagation (the warning added today does
not cover it — it fires only when detection fails); (2) LAB-10 can close once (1)
lands, using the env-var datum above as the served confirmation.

## 2026-10-19 — propagation fixed: `--gfx906-fa-packed on` now reaches the aligner

Root cause (not a degraded boot): `CacheConfig.gfx906_fa_packed` is published to the
backend only from `check_and_update_config`, i.e. where the config is *built*. The
process that resolves the KV layout does so in `update_block_size_for_backend()`,
and a worker whose config arrives without re-running that hook leaves the backend's
`_PACKED_CONFIG_MODE` at its `"off"` default — so an explicit `on` was silently
served as the alias layout, while `GFX906_FA_PACKED=1` (read from the environment
inside the backend) worked.

Fix: the publish was factored into `RocmPlatform._gfx906_publish_packed_mode()` and
`RocmPlatform` now overrides `update_block_size_for_backend()` to call it before the
base phase 1/2/3 pass. Verified live: `--gfx906-fa-packed on` at
`max_model_len 16384` now keeps `KV cell 400 B` and picks block **2672** (was
2096). Tests: `test_worker_side_block_size_pass_republishes_the_config_field`
plus the existing decision tests (17 passed across the two platform files).

With this, LAB-10 can close on the served 16k datum already measured
(610,577 vs 570,368 usable tokens, +7.05 %), noting that predictions are upper
bounds.
