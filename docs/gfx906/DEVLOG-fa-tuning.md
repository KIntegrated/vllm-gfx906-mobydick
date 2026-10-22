# DEVLOG — gfx906 FA launch-config records (P32-4/#63)

**VERDICT: the hand-tuned launch rules are right in 3 of 4 measured shapes; at
B=2/Sq=8 the KV split is one step too high (+3.2 % available), and the split
optimum keys on batch x Sq, not Sq alone. The measurement also confirms the two
rules that looked arbitrary: the GQA downgrade 8 -> 2 is worth 1.7-1.85x at
decode, and the prefill clamp really does collapse nc2 to 1.**

## Why this exists

The FA launch parameters (`nc2` head packing, KV split, gather clip) were
process-wide env defaults with hand-measured provenance in
`csrc/gfx906_fa/gfx906_fa.cpp`. Upstream's ROCm segmented-attention work (#59132)
solves the same problem with a per-shape record cache; #63 mines that idea for
our kernels. Slices:

1. `vllm/gfx906_fa/tuning.py` — identity digest, atomic records, TP identity
   cross-check, and `static_config` as a mirror of the C++ rules (`51ce08adbe`).
2. The mirror pinned against the C++ via no-GPU-work probes, which immediately
   found two wrong rules in the mirror (`kvsplit` batch-aware direct rule;
   `nc2` GQA downgrade + prefill clamp), then per-call overrides so the Python
   side passes the config for real (`22aa898cbb`, `3f1f4d0ee2`).
3. Cache-only warmup: a boot records the static rule per shape and rewrites
   nothing when warm (`f39c6dd054`).
4. `docs/gfx906/_fa_tune.py` — measure the candidates per shape.

## Slice 4 measurement (Qwen3.5-27B geometry: Hq=24, Hkv=4, D=256, Sk=65536)

Method: `docs/gfx906/_fa_tune.py --reps 40 --preload-seconds 30`. The preload
matters — a 3-6 ms kernel burst never lifts mclk off its 350 MHz idle state, and
`rocm-smi --showclocks` **blocks under load and reports DPM levels rather than
the live clock** (it read a flat 350 MHz through a 30 s GEMM preload). Sampling
`/sys/class/drm/card*/device/pp_dpm_mclk` every 0.25 s, per
`dvfs-mi50.md`, gives the truth: **median 1000 MHz, 100 % of loaded samples
>= 900 MHz** in every measured window.

| shape | static rule | measured best | rule vs best |
|---|---|---|---|
| decode B=1 Sq=1 | nc2=2, y=16 (1.112 ms) | same | **0.0 %** |
| decode B=4 Sq=2 | nc2=2, y=16 (4.016 ms) | same | **0.0 %** |
| verify B=1 Sq=8 | nc2=1, y=32 (2.925 ms) | same | **0.0 %** |
| verify B=2 Sq=8 | nc2=1, y=32 (5.788 ms) | y=16 (**5.602 ms**) | **+3.2 %** |

Reproduced twice at 40 and 60 reps, and the y=16 rows agree to 0.05 % across all
three nc2 values, so the effect is the split and not noise.

## What the numbers confirm about the existing rules

- **GQA downgrade 8 -> 2 is load-bearing.** With gqa=6: nc2=1 costs 1.880 ms vs
  1.112 ms at nc2=2 (Sq=1) and 7.414 vs 4.016 ms (Sq=2, B=4) — 1.7-1.85x. The
  rule already picks 2; this is the measurement behind a rule that looked like a
  safety fallback.
- **The prefill clamp is real.** At Sq=8 every nc2 value times identically
  (2.925/2.926 ms), i.e. the launcher's `seq_q > 2 -> nc2=1` collapses them all,
  exactly as documented.

## What it changes (proposal, not applied)

The split rule is `32 if seq_q >= 4 else 16`. The two verify rows show the
optimum moving with **batch x Sq**: B=1 (product 8) prefers 32, B=2 (product 16)
prefers 16. So the rule should key on `batch * seq_q`, with the boundary
somewhere between 8 and 16 — but that is **two data points**, on one model
geometry, at one context length. Changing a rule that affects every model on the
strength of two points would repeat the mistake this dev log exists to avoid, so:

- the measurement is recorded, the rule is untouched;
- the follow-up is a sweep of (B, Sq) around the boundary (B in 1..8 x Sq in
  {4, 8}) on both the 27B (Hq=24/Hkv=4) and the 35B MoE geometry, with the same
  mclk gate, before `kvsplit = f(batch * seq_q)` replaces `f(seq_q)`.

## Gates

PPL is unchanged throughout the plumbing (dense-27B 14.3750 exact after each
slice: mirror fix, overrides, warmup), which is what "behaviour-preserving" has
to mean for a change that only moves who chooses a launch parameter.

## Slice 4b — the (batch, seq_q) sweep, and why a formula is the wrong target

`_fa_tune.py --sweep` over four geometries (27B/35B at TP=1 and per-rank TP=2;
the GQA ratio is preserved under TP, the grid is not) × B ∈ {1,2,4,8} ×
Sq ∈ {2,4,8}, Sk=65536. Each shape gets its own preload and its own mclk gate;
**48/48 windows valid** (median 1000 MHz), candidates timed interleaved with a
rotating order.

The first sweep had **23 of 42 cold-clock windows** (350/800 MHz) and was thrown
away — the tool now discards and retries rather than averaging, and prints the
retry count per cell.

Mean gap of a rule vs the measured best over the 48 cells:

| rule | mean gap |
|---|---|
| `B*Sq <= 8 -> 32`, `<= 24 -> 16`, else `8` | **+1.98 %** |
| **current `32 if Sq >= 4 else 16`** | **+5.63 %** |
| always 8 | +23.30 % |

So the current rule leaves ~5.6 % on the table and a `batch x seq_q` staircase
recovers about two thirds of it. The trend is monotone and principled — the
split trades parallelism for combine traffic, and past B*Sq ≈ 8 the scan is
already wide enough — which is why a staircase fits at all.

**But a staircase is not the right deliverable.** The per-cell data shows the
optimum also depends on heads per rank at fixed B*Sq: at B=1/Sq=2, 27B TP1
(Hq=24) reads 1.103/1.042/1.062 for y=8/16/32 — a flat 5 % spread — while 35B
TP2 (Hq=8) reads 1.462/0.740/0.405, a 3.6x spread where the splitting decision
dominates. No single-key formula covers both, which is exactly the case the
per-shape record table exists for.

Slice 4b therefore lands the *consumer* instead of another formula: `resolve()`
returns a recorded config when the tuner has written one for that
`(identity, shape)` and the static rule otherwise, and the gather call site uses
it. Behaviour is unchanged on any machine that has not run the tuner (verified:
dense-27B PPL 14.3750 with no records present).

Remaining before a rule or record set ships: (a) `--write-records` so a sweep
persists its winners; (b) the context-length axis — every cell here is Sk=65536,
and the split balance moves with context; (c) a numerical check that a changed
`kvsplit` does not shift PPL beyond the probe's spread, since the split changes
the reduction order in `gfx906_fa_split_combine`.

## Slice 4c — context axis, and the numerical gate

Same grid at Sk=16384 vs Sk=65536 (27B TP1, 35B TP2), 48/48 windows valid,
`--write-records --min-gain 1.0` → **14 + 14 records written** into the opt-in
cache.

**The optimum moves with context too.** 27B TP1 B=8/Sq=2: the rule (16) is
exactly optimal at Sk=16384 (1.912 ms, gap 0.0 %) and 2.5 % off at Sk=65536
(y=8 7.743 vs 7.940). With four axes — heads-per-rank, batch, Sq, Sk — a single
formula is the wrong vehicle; the record key already carries all four, which is
what makes the per-shape table the right answer rather than a stopgap.

**Largest gaps are 35B TP2 decode.** B=1/Sq=2: the rule's y=16 runs 0.741 ms
against y=32's 0.405 ms — **45 % slower**, at both context lengths. That is the
concrete cost of a batch-blind, head-blind split rule on a low-Hq MoE rank.

**Numerical gate (the one I owed).** Changing `kvsplit` does move the raw
attention output: on synthetic inputs (B=2, Sq=8, Sk=16384, all-zero K → uniform
attention) the max |delta| vs y=32 is 9.3e-4 at y=1, 2.5e-4 at y=8, 1.7e-4 at
y=16 — far above fp32 rounding, because that input makes the output a
near-cancelling mean of 16k values, i.e. the worst case for a reordered sum.
The gate that matters is the model: dense-27B PPL is **14.3750 at both
`GFX906_FA_KVSPLIT=8` and `=32`** (identical to four decimals, 0 top-20
misses). So a split change is quality-neutral and does not need a PPL band
re-anchor.

Records stay **opt-in** (`VLLM_CACHE_ROOT` pointing at the tuner's cache): the
45 % decode win is measured in a synthetic harness, and enabling it for serving
should be its own decision with a serving A/B, not a side effect of a sweep.
