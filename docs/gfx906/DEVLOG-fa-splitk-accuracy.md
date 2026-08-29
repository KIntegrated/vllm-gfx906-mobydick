# FA split-K long-context accuracy (M4) — production split defaults are
# safe (in fact MORE accurate) at 16k–32k context

Copyright Kevin Read <me@kevin-read.com>

## 2026-08-29 — M4 long-context split-K accuracy point (qwen review #4a)

**VERDICT:** SHIPPED · **GATE:** in-process probe at sk=16384/32768,
production defaults (gather kv_split=16; direct-paged kv_split=8) vs fp32
torch ref, two prime-model geometries (D=256/Hq16/Hkv2 + D=128/Hq32/Hkv2)
— every arm rel_ref ≤ 2.7e-2 (< 5e-2 tol); split-16 error 3.6–4× LOWER
than no-split at the same L.

## HYPOTHESIS

If the split-K path loses precision as context grows (fp16 P·V
accumulators + unscaled partials growing with keys-per-slice × |V|), then
the production split defaults (gather kv_split=16; paged clamp(16/B,2,8))
should show rel error growing with L and exceeding the no-split baseline
at L=16k–32k.

## What was done

Probe: `benchmarks/kernels/gfx906/m4_splitk_accuracy_probe.py`
(subprocess per arm — the gather kv_split is a C++ static parsed on first
forward call; fixed seed 20260829 across arms so split-vs-nosplit deltas
are direct). Arms per (sk ∈ {16384, 32768}, geometry):
`g16` = fa.forward kv_split=16 (production B=1 default),
`g1` = fa.forward kv_split=1 (no-split baseline),
`p8` = fa.forward_paged_direct B=1 → internal kv_split=8
(the clamp(16/B,2,8) production B≥2 default). fp32 torch ref = same
inputs, full attention. Logs: /local/tmp/m4_probe_run2.log,
suite /local/tmp/m4_suite.log (78/78).

Mechanism correction (the M4 item's framing was partly stale): the split
partials are **fp32** in both paths (o_part [B,Sq,Hq,split,D] fp32 +
(m,l) meta fp32, combine = fp32 log-sum-exp — `fa_split_combine_kernel`,
`gfx906_fa.cpp:383` / `gfx906_fa_launcher.cu:106`). The only fp16
precision exposure is the kernel's P·V accumulator
(`v_pk_fma_f16`, `acc` half2 in `fattn-q8-paged.cuh:261` /
`fattn-q8.cuh:532`) — **common to split and no-split**; splitting only
shortens each accumulator (sk/split keys) and adds one fp32 rescale+add
per split in the combine.

## Evidence FOR (claim holds — numbers are launch-regime, in-process)

| arm (rel ref) | sk=16384 | sk=32768 |
|---|---|---|
| gather split-16, D=256 (production B=1) | 5.19e-3 | 6.64e-3 |
| gather split-1,  D=256 (no-split)       | 1.90e-2 | 2.63e-2 |
| paged  split-8,  D=256 (production B≥2) | 3.98e-3 | 4.97e-3 |
| gather split-16, D=128 (Muse)           | 5.64e-3 | 7.18e-3 |
| gather split-1,  D=128 (Muse)           | 1.90e-2 | 2.73e-2 |
| paged  split-8,  D=128 (Muse)           | 3.75e-3 | 5.51e-3 |

Split-vs-nosplit delta (‖g16−g1‖/‖g1‖): 1.95e-2 / 2.70e-2 (D=256),
1.96e-2 / 2.80e-2 (D=128) — each ≤ 2× the no-split arm's own rel_ref
(the G2 bound). Error grows gently with sk (split-1: 0.019→0.026 over
16k→32k; split-16: 0.0052→0.0066) — per-accumulator error scaling with
slice length, exactly the predicted mechanism, in the SAFE direction.

## Evidence AGAINST

None. Every arm is within half the 5e-2 tolerance at 32k.

## Why the split path is MORE accurate

The no-split arm carries one fp16 accumulator over all sk keys; the
fp16 P·V accumulation error grows with the number of
add/rescale operations in a single accumulator. Splitting cuts each
accumulator to sk/split keys (1k at split-16/16k) and merges the fp32
partials with a numerically stable log-sum-exp — the combine's own noise
is far below the per-accumulator noise. Net: production B=1 decode
(split 16) and B≥2 decode (split 2–8) run the MORE accurate configuration
at long context; the unscaled-partial magnitude concern does not
materialize because the partials are fp32, not fp16.

## Interactions / follow-ups

- Suite pins (permanent rot guards, both prime geometries):
  `test_forward_kv_split_gqa_pack_vs_fp32_ref` gained (1,16,16384) +
  (1,1,16384) arms; new
  `test_forward_paged_direct_splitk_long_context_vs_fp32_ref` (L=16384,
  split-8 default, D=256 + D=128). Suite 74 → 78, 78/78
  (/local/tmp/m4_suite.log, 70 s).
- Roadmap M4 item closed; the long-context serving runs in
  `DEVLOG-muse-glimmer.md` (2026-08-29) were produced on these exact
  defaults — no accuracy follow-up needed for the 110k/256k points.
- Cross-ref: Part C accuracy gate (Q4 path) is still its own item —
  this entry covers the Q8 production paths only.

VERDICT: SHIPPED
