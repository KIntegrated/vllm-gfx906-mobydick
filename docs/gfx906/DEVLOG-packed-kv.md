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

keep packed when `gain >= 5 %` (`_gfx906_packed_capacity_gain`,
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
`max_model_len` requests per group. The 5 % threshold keeps both verdicts right
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

## 2026-10-19 — VERDICT: packed KV breaks at TP>1; fixed by deriving the layout from the pool

**GATE: `packed` x `TP=2` on Qwen3.5-35B-A3B-AWQ previously died 100 % of the
time with `RuntimeError: D must be multiple of 32` (in `gather_paged_kv_q8`).
Now it serves: 125.56 t/s vs 126.19 t/s for the alias layout (B=4, 4 requests,
pp2048/tg256, util 0.85, 32768 ctx, same boot, mclk 1000). No measurable cost.**

### Mechanism (measured, not inferred)

The attention impls are constructed **while weights load**, i.e. *before* the KV
layout is resolved; the packed mode is published from the platform's
`update_block_size_for_backend`, which runs later. In a TP worker that left
`self._packed = False` while the **pool** was packed, so `forward()` took the
alias branch and split the packed cell along the fused axis:

| | cell | K | V |
|---|---|---|---|
| packed pool (D=256) | 784 B = 392 fp16 elems | 272 B q8 | 512 B fp16 |
| alias split of that cell | — | 256 elems | **136 elems** |

`V` width 136 => `D=136`, and `136 % 32 = 8` => `TORCH_CHECK(D % 32 == 0)`.
Confirmation: a debug print in the packed branch produced **zero** lines in the
failing run (the branch was never entered), while the pool log said `KV cell
784 B` — pool packed, views alias.

TP=1 never failed because the single process publishes the mode before it builds
any impl.

### Fix

The pool's own cell width is authoritative and identical on every rank, so the
layout is derived from it (`_pool_cell_is_packed`: cell bytes ==
`_packed_cell_bytes(head_size)`, 784 B for D=256). `self._packed` now only gates
the construction-time validations (head padding, `head_size % 32`); the view
cache (`_packed_src`, `_v_cache`) is initialised unconditionally, because a
late-bound decision needs the state regardless of the flag. A disagreement logs
once; an unrecognised cell width asserts instead of mis-splitting.

**TP=4/TP=8:** no TP arithmetic enters the decision — TP only changes KV heads
per rank (2 at TP=1, then 1 through sharding or replication), and the choice is
byte-arithmetic on the cell. Covered by
`test_layout_choice_is_independent_of_tp`, which parametrises 1/2/4/8 KV heads per
rank. Not executed on hardware: this box has 2 MI50s.

### Numbers (B=4 unless stated)

| arm | t/s (mclk 1000) | note |
|---|---|---|
| TP=2 alias | 126.19 | B=4 |
| TP=2 packed | 125.56 | B=4, crash gone (**-0.5 %**, in the arm-to-arm spread) |
| TP=1/TP=2 alias, B=1 | 64.12 / — | TP=1 packed B=1: 57.5 |
| TP=1 alias, B=4 | **88.87** | mean of A1 (3 samples) + A2 (2 samples) |
| TP=1 packed, B=4 | **89.24** | **+0.41 %**, A->P->A bracketed |

The TP=1 B=4 pair is the hot re-run of a first pair that read **idle mclk 350**
(87.0/88.9 t/s) - the DVFS trap this repo documents. `mclk_median_mhz` in the
harness JSON is the field that catches it (350 = idle, 1000 = loaded); a 350
sample is discarded, not averaged. So packed is within noise of alias at B=4 and
the win at TP=2 is capacity/robustness, not throughput.

## 2026-10-21 — re-anchored on the 0.32 base, and the `auto` gate fixed for spec decode

**VERDICT: packed still pays on 0.32 (+18.8 % served capacity at 32k) and every
recorded verdict reproduces — but `auto` had been silently OFF under spec decode,
because the aligner left the unified page unset when no padding was needed.**

### The bug

`Platform._align_hybrid_block_size` returned early when the attention page already
equalled the mamba page, so `cache_config.mamba_page_size_padded` stayed `None`.
The packed-KV `auto` gate dry-runs the aligner twice and treats a missing page as
"no prediction", so with MTP k=3 on the 35B the two arms read
`(1072, None)` vs `(1408, 1,103,872)` and the gain short-circuited to **+0.0 %**
— `packed KV layout left off under gfx906_fa_packed=auto`. Since MTP k=3 is our
default recipe, packed was effectively never on. Fixed by recording the unified
page on every path (`7fcd4205b1`); the boot now logs `enabled … +28.4 %`.

Found by adversarially reviewing the plan for this item: the first diagnosis
("the page is None in both arms") was asserted from the formula's short-circuit
and was wrong in an informative way — the asymmetry was the whole bug.

### Re-anchored geometry (config-only dry run, `_hybrid_kv_ratio.py` rules)

| model | max_model_len | spec | off block | on block | prediction | verdict |
|---|---|---|---|---|---|---|
| 35B | 8192 | – | 1056 | 1376 | **+33.6 %** | on (matches the recorded +33.6 %) |
| 35B | 8192 | MTP k=3 | 1072 | 1408 | **+32.6 %** | on (new) |
| 35B | 32768 | MTP k=3 | 1072 | 1408 | **+28.4 %** | on |
| Lightning | 8192 | – | 2096 | 2672 | **+0.4 %** | off (matches +0.4 %) |
| Lightning | 16384 | – | 2096 | 2672 | **+14.8 %** | on (matches +14.8 %) |

The bootstrap difference is the **draft KV group entering the same alignment**:
without a drafter the 35B aligns at 1056 → 1376, with MTP k=3 at 1072 → 1408
(+16 tokens of block, and the padded mamba page moves 2,162,688 → 2,195,456).
Both geometries must be quoted with a packed verdict — the 0.31 numbers are the
non-spec ones.

### Served capacity on 0.32 (decision gate)

Same knobs (TP=2, MTP k=3, util 0.85, `max_model_len 32768`), forced on vs off:

| arm | served KV pool | concurrency @32k |
|---|---|---|
| `--gfx906-fa-packed off` | 890,070 tokens | 27.16× |
| `--gfx906-fa-packed on` | **1,057,678 tokens** | 32.2× |
| | **+18.8 %** | |

`+18.8 %` served against `+28.4 %` predicted — the same over-statement factor as
the 0.31 pair (+22.2 % served / +33.6 % predicted), so the prediction-is-an-upper-
bound caveat stands unchanged. **Keep the packed layout.**

### Consequence for the remaining work

The packed cell is a **cache format** (K q8 bytes | V fp16 in one cell), not a
`KVCacheLayout` (which is tensor axis order, e.g. `LBNHC`). Its upstream analogue
is `ultraquant_4bit` / `int4_per_token_head` — `CacheDType`s — so the "first-class
format" design belongs with P32-2/P32-5 (#61/#64), and P32-1's remaining scope is
the `auto` decision itself, which is now correct.
