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
