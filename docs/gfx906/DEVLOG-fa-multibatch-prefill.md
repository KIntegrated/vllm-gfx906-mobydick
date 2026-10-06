# FA multi-batch prefill tax — pad-tile walk (FIX-H2) + host-cu_seqlens (M3)

**VERDICT:** SHIPPED (FIX-H2 −41% validated; M3 kept wall-neutral) ·
**GATE:** serving A/B @4×122880 (mtp3b4, TP=2, Qwen3.8-27B-AWQ-INT4)
**Branch:** gfx906/fa-decode-fp16 (2026-09-10 → 2026-09-12)
**Full detail:** `git log bd306a996e..6f215f7e7b -- docs/gfx906/ttft-prefill-stall.md`
(pre-restructure prose, §13.5–§13.24) + `docs/gfx906/prefill-multibatch-tax.md`

## 2026-09-10 — HYPOTHESIS: the O(live-context) prefill tax is the FA
kv_max pad-tile expansion

If decode sequences in a mixed batch expand the FA grid to kv_max, every
prefill chunk walks ALL live contexts instead of its own → step cost
grows with Σ live contexts, not the chunk's own attention.

## What was done

Counter/step-model probes (D1a/D1c holder+probe, E1 chunk-size A/B, E2
concurrency-halving), engine trace analysis, root-cause localization to
`gfx906_fa.cpp` kv_max grid construction; fix = clamp pad tiles to
kv_max=0 (mathematically exact: only skips -INF-masked work; decode
capture path guarded by grid_x=1).

## Evidence FOR (gates labeled)

- Serving A/B @4×122880 (the gate): 75.3 → **44.8 min (−41%)**; prefill
  agg 108.6 → 182.9 t/s; D1c probe 14.85 → 7.16 s. Commit `bdcbd8ec3a`.
  Arm identity: BOTH walls are the serving stack (MBT-1 pre-fix
  2026-09-11 vs the mtp3b4 campaign post-fix) — the offline same-shape
  BASE arm never completed (wedges #65/#66), so the offline FIX wall
  (2377.6 s) has no same-stack baseline.
- rep_frac8 fingerprint identical pre/post (exactness check).
- Campaign deliverable on the fix build: mtp3b4 41.1/40.8 min per rep,
  199–201 t/s prefill agg, acceptance 4.00 (`abc9d8ded9`).

## Evidence AGAINST / residual

- D1c residual remains: probe-steps-with-holder-live 7.16 s vs 5.3 s
  control (+1.75 s per 2 steps @67k live).
- 4×32768 M3 A/B: null (414.9 vs 416.5 s) — residual is context-scaled.
- KVSPLIT shape-aware note (co-review F3): the 512-MiB byte budget —
  not a Sq threshold — is what bounds prefill splitting. At B=1
  prefill (Sq=1024, Hq=12/rank, D=256) the partial buffer is 403 MB
  at y=32 < 512 MiB → prefill DOES split at B≤2 (403 MB vs 201 MB
  transient at y=16); at B=4 it is 1.6 GB → forced y=1. The doubling
  of the o_part transient at B≤2 is against a 0.93-util serving config
  with OOM history — revisit the budget if B≤2 long-context prefill
  OOMs appear.

## 2026-09-12 — HYPOTHESIS: the residual is per-seq D2H syncs in
forward_paged's variable-Q branches (M3)

If `int(cu_seqlens_q[s+1]-cu_seqlens_q[s])` device reads per seq per
layer block the host, mixed steps stall ∝ queued GPU work ∝ live
contexts. Trace evidence (torch-profiler CPU-side, offline path):
**10,959 aten::item = 112.5 s of a 120-s window (93%)**, 7,956 inside
forward_paged spans (10.7/call), p50 3 µs vs p90 69 ms (bimodal
instant-vs-backlog-drain).

## What was done (M3)

`CommonAttentionMetadata.query_start_loc_cpu` threaded through
Gfx906FAMetadata → `forward_paged(cu_seqlens_q_host=...)` → all 4
variable-Q branch sites use host ints. Gate `GFX906_FA_HOST_QSEQ=0`
reverts. Commit `6c97dc6492`.

## Verdict (M3): NEUTRAL on wall, kept default-ON

- 4×32768 same-boot A/B: 414.9 vs 416.5 s (null).
- D1c with M3: 7.047 vs 7.16 s (neutral).
- 4×122880 offline: FIX arm 2377.6 s — best known for the shape, but
  stack-confounded (offline vs serving); same-stack BASE blocked by
  init wedges (degradation #65/#66). The aten::item syncs were the
  host PARKING while the GPU worked (healthy pipelining), not lost
  time. Kept: removes ~8 pointless D2H syncs per layer call, reads
  already-available host metadata.

## Interactions / open residual

- **Batch-level collapse finding (2026-09-12, counter decomposition —
  /metrics polling, RAM-safe):** with a 2k probe admitted alongside a
  67k mtp3 decoder, the holder's decode fell 40 → ~1.6 t/s and the
  probe's two 1024-token chunks did not complete inside its 7.098-s
  ttft (admission was instant — not a scheduler hold). The residual is
  a **batch-level collapse under mixed prefill+long-ctx-decode
  admission** (~25–60× throughput drop), NOT a per-layer tax. Mechanism
  OPEN: scheduling priority (chunk vs verify steps), eager/piecewise
  path for mixed shapes, or the draft step under mixed admission.
  Next discriminator: engine-stats step timestamps or an offline D1c
  equivalent (torch profiler BANNED — see DEVLOG-profiling-tooling.md).
- KVSPLIT: TWO related changes — dfed62f133 (gather-path shape-aware
  default 32/16; gate +1.4–2.3% @64/96/120k same-boot A/B) and R3
  `fa_paged_kv_split_default` (paged-direct alignment; own gate
  "n/a under LEGACY=1" — the production path is unchanged, the change
  protects the LEGACY=0 flip). Do not cite the gather numbers for R3.
  Both live in the kv_max/tile-construction path; the serving campaign
  ran on the combined build.

## 2026-10-05 — MBT-1r re-anchor: the published 64k B=1 point does NOT reproduce tonight

**VERDICT: INCONCLUSIVE — the published point stands; tonight's reading is
recorded as a candidate regression (box or code), not as a new anchor.**
Recipe: the 2026-08-29 one replayed through `_serve_tp2_gfx906.sh` (maxlen
262144) + `_bench_serve_grid_gfx906.py '[[65536,128]]' 2` — Qwen3.8-27B-AWQ-INT4,
TP=2, **spec-free**, prefix caching OFF, `GFX906_FA_LEGACY` at its default, on
`gfx906/v0.30.0` @ `a12f960a6d`, canary-gated first (38.7 t/s).

| datum | published 2026-08-29 | post-0.30.0 note 2026-09-24 | re-anchor 2026-10-05 |
|---|---|---|---|
| prefill t/s @64k | 364.8 (365.1 / 364.5) | 354 | **307.1 / 308.9** |
| TTFT s | 179.49 / 179.78 | 185.3 | **206.68 / 205.28** |
| decode t/s @64k | — (out=1, EOS on filler) | — | 21.50 / 21.47 |
| canary t/s | 38.9 | — | 38.7 |

Two samples 0.6 % apart, so the −15.6 % vs the record (−13.2 % vs the 09-24
note) is not sampling noise. Candidates, all still open:

- **Instrument — NOT a difference for the 08-29 comparison (measured).** `syv9phase` was
  installed 2026-09-03, i.e. *after* the 08-29 record, and tonight's arm logged 0
  `SYV9PLUGIN` lines → both arms are plugin-free V1. What the plugin does change is
  *reproducibility*: the 08-29 recipe can no longer boot as recorded on a box where it is
  installed, because its gate does `open(<arm cfg>)` per hooked forward and dies in
  `torch._dynamo` under this V1+fullgraph arm — the two earlier attempts of this same job
  both failed exactly there (`phase1-rootcause.md`). The 09-24 note at 354 t/s is the
  confounded datum instead: V2 runner + MTP k=3 + the plugin live.
- **Flag drift:** the harness leaves `GFX906_FA_LEGACY` at its default, which was **1** on
  08-29 and has been **0** since 2026-09-16 (KVLAYOUT-1, see `README.md`), so the record
  differs from tonight's arm in that switch too. The LEGACY=0 bake was prefill-neutral, and
  the 354 note was explicitly LEGACY=0 — recorded, not dismissed.
- **Prompt drift:** the published 64k row ended `out=1 (EOS on filler)`; tonight the same
  nominal pp65536 filler generated all 128 tokens. Same token count, different completion
  behaviour — the two arms are not provably the same input.
- **DVFS / mclk (leading box-side candidate, with a same-night control):** sampled mid-bench from
  sysfs (`rocm-smi` blocks under load), `pp_dpm_mclk` **toggles 800 ↔ 1000 MHz** on both cards
  during the 64k prefill. The control is the campaign's own traces: all four campaign-1 arms the
  same night sampled the same reader every 5 s across their load windows
  (`/local/tmp/qsa12/mclk_{A1,A2,C,D}.txt`, ~140 samples) and show **1000 MHz held under load,
  350 at idle, and not one 800 MHz sample**. The 64k prefill bench is the only load tonight that
  showed 800. So the down-shift is load-pattern dependent — a sustained full-bandwidth 65k prefill
  pulls the memory clock down, a 32k campaign arm does not — and it is invisible to the
  short-prompt canary. No root on this box, so mclk cannot be pinned to price it exactly.
- **Gate implication:** tonight is a concrete counter-example to "canary healthy ⇒ perf work is
  comparable" for *long-context prefill*: the canary is a short-prompt decode probe (no sustained
  memory traffic), and it read its healthy 38.7 t/s while the 64k prefill point sat 15 % low. A
  prefill-based gate is what this class of degradation needs.
- **Code:** the tree has moved substantially since the 08-29 record and the 09-24 note
  (0.30.0 base, the PLE/GDN train, the fork-switch registration). Not excluded — and with
  the instrument difference retired, this and the box state are the only two survivors.
- **Not implicated:** the wedge family — `journalctl -k` shows no `GPU reset` /
  `wedged` in the window, and the boot was preceded by a 38.7 t/s canary on the
  same cards.

Next, cheapest first: (1) sample `pp_dpm_mclk` across a whole bench (not a mid-point
grab) to price the 800-MHz share of the wall — the one candidate measurable without
touching the tree; (2) a same-boot A/B against the pre-0.30.0 tip for the code side;
(3) only then decide whether the README row gets re-anchored.

Raw: `/local/tmp/mbt1r/bench_27b_64k_noplugin.out`,
`/local/tmp/lcbench_mbt1r-np_server.log` (ready in ~695 s),
`/local/tmp/mbt1r/phase1-rootcause.md`.

---

## 2026-10-06 — the 800 MHz mclk share, priced: ~8.4 % of the 64k prefill wall, about half the gap

**VERDICT: OPEN** (the re-anchor question, one candidate down) · **GATE:** serving
wall-clock, the 2026-08-29 recipe replayed on `gfx906/v0.30.0` @ `e1bffdd742`,
one point per client invocation (`_bench_serve_grid_gfx906.py '[[<pp>,128]]' 1`),
**32k bracketing the 64k pair in the same session**, canary-gated first
(38.6 t/s), `VLLM_PLUGINS=` so the syv9phase instrument cannot kill the V1
fullgraph compile (see the 2026-10-05 entry / `phase1-rootcause.md`).

### HYPOTHESIS

If the DPPM 800 MHz memory-clock down-shift is what moved the published 64k point
from 364.8 to ~308.9 t/s, then (a) a continuous trace of a whole 64k prefill
shows a large 800 MHz share of the loaded wall, and (b) a 32k point in the same
session — the control the 2026-10-05 entry read as 800-free — shows little.

### What was done

One **1 Hz** `pp_dpm_mclk` trace over the entire session, read from
`/sys/class/drm/card*/device/pp_dpm_mclk` (sysfs, never `rocm-smi --showclocks`,
which blocks under load), one point per client invocation so each sample carries
its own absolute start epoch, split prefill/decode at the client's own `ttft_s`,
and the 800 MHz share taken **of loaded time** (350 MHz idle excluded).
Tools and raw traces: `/local/tmp/mclk/` (`driver3.sh`, `sample.sh`,
`report.py`, `pricing.py`, `trace_session.txt`, `windows.txt`). Order: 32k, 64k,
64k, 32k.

### Evidence FOR

| point | prefill t/s | TTFT s | decode t/s | 800 MHz share of loaded prefill (card1/card2) | factor |
|---|---|---|---|---|---|
| ctrl 32k (a) | 353.2 | 87.33 | 25.45 | 30.2 % / 25.0 % | 0.940 / 0.950 |
| 64k (a) | 308.9 | 205.39 | 21.51 | 43.6 % / 40.5 % | 0.913 / 0.919 |
| 64k (b) | 308.9 | 205.38 | 21.51 | 41.5 % / 42.0 % | 0.917 / 0.916 |
| ctrl 32k (b) | 354.6 | 86.95 | 25.44 | 29.1 % / 23.8 % | 0.942 / 0.952 |

`factor = 1 − 0.2 × share(800)` — the bandwidth-bound wall scaled by 800/1000 for
the down-shifted share. Whole session: 669 s wall, 662 samples, 605.7 s loaded
per card, 800 MHz for **36.9 / 34.6 % of loaded time**.

* **The down-shift is real, reproducible and now priced.** Both 64k samples,
  both cards: factor **0.913–0.919**. Holding 1000 MHz across the 64k prefill
  would recover ~**8.4 %** of its wall.
* The gap to the record is −15.3 % (308.9 vs 364.8), so mclk accounts for
  **about half** of it: 308.9 / 0.916 = **~337 t/s**, still ~7.6 % under the
  record. The tree/box A/B (below) remains the live candidate.
* Stability: the 32k pair agrees to 0.4 % (353.2 / 354.6) and the 64k pair to
  0.03 % — the effect is not sampling noise, and the session did not drift.

### Evidence AGAINST / corrections to the previous entry

* **(b) is falsified as stated.** The 32k control also ran 24–30 % of its loaded
  prefill at 800 MHz. "A 32k arm does not down-shift" was an artifact of the
  instrument: the campaign-1 traces sampled every **5 s** (~140 samples) and were
  taken on a *different model* (Whittle/qwen4_exp, not the 27B). At 1 Hz the
  down-shift is there at 32k too, just smaller (factor 0.945 vs 0.916). The
  load-pattern claim survives as a **magnitude**, not as an on/off.
* The 0.8 factor is an upper-ish bound: it assumes the whole wall is
  bandwidth-bound and that only mclk moved. The 64k over 32k drop is otherwise
  physically expected (attention cost grows with context), so the intrinsic
  32k→64k step is not attributable to the clock.
* Decode follows the same shape (25.45 → 21.51 t/s from 32k to 64k), consistent
  with the KV side being bandwidth-bound too.
* Coincidence guard: today's 32k point (353.2 / 354.6) is numerically close to
  the 2026-09-24 note's 354 t/s, but that note reads a **64k** point. Not the
  same datum; do not pair them.
* Not implicated: the wedge family — `journalctl -k` from 03:40 has no
  `GPU reset` / `wedged`, the box was canary-healthy at 38.6 t/s before and back
  to 350 MHz idle after (VRAM 10 / 18 MiB).

### Why it matters for the gate

The house DVFS gate reads mclk through `_bench_gfx906.py::_MclkSampler`, which
polls `rocm-smi --showclocks` and reports the **median** (of the max across
cards). A 40 % 800 MHz share collapses to a median of 1000, so the gate passes a
run that today's trace shows is materially down-shifted. The gate needs the
*share*, sampled from sysfs per card — the instrument is fixed in the same
commit as this entry.

### Interactions / open

* Remaining candidate: a same-boot A/B against the pre-0.30.0 tip (the tree has
  moved a lot since the 08-29 record: 0.30.0 base, the PLE/GDN train, the
  fork-switch registration).
* Residual open question from 2026-10-05 (prompt drift: the record's filler
  ended `out=1` (EOS), tonight's generated 128 tokens) is untouched by this run.
* The README's published row stays untouched until the A/B runs.
