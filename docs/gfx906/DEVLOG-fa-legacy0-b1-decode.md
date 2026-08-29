# FA LEGACY=0 B=1 decode gap — kernel-level localization + serving
# adjudication

Copyright Kevin Read <me@kevin-read.com>

## 2026-08-29 — B=1 LEGACY=1-vs-0 decode gap (roadmap item #1)

**VERDICT:** OPEN (serving gate pending — host reboot required, see
below) · **GATE:** same-boot TP=2 serving A/B, Qwen3.8-27B B=1
pp2048/tg256, arms A (LEGACY=1), B (LEGACY=0), C (LEGACY=0 +
DIRECT_PAGED=1 + DIRECT_PAGED_Q8=1).

## HYPOTHESIS

If LEGACY=0 B=1 loses to LEGACY=1 (the M5-era 2.5–3.7 % serving gap,
never same-boot adjudicated: 107.2 boot M vs 111.5 boot L), the loss
sits in a named suspect — quantize write path, FA Q-side, or gather
traffic — and the kernel-level decomposition will assign it.

## What was done

1. Dispatch audit (no GPU): with today's defaults (`GFX906_FA_LEGACY=1`,
   `GFX906_FA_DIRECT_PAGED=auto` min_batch=2,
   `GFX906_FA_DIRECT_PAGED_Q8=0` since M6 Part B), the direct-paged
   branch is **never taken** — LEGACY=0 B=1 runs the fused-Q8 gather
   (`gather_paged_kv_q8`, reads the pre-quantized aliased side buffer)
   + the same FA gather kernel as LEGACY=1, which instead runs
   `gather_paged_kv_quant_persistent` (fp16 K read + in-kernel
   quantize). The M5-era gap was measured when LEGACY=0 B=1 ran
   direct-paged FA (`forward_paged_direct`, internal kv_split=8).
2. Kernel-level probe (GPU0, in-process, eager — launch-regime
   evidence): `benchmarks/kernels/gfx906/legacy0_b1_step_probe.py`,
   B=1, Sk ∈ {2048, 16384, 32768}, both prime geometries; log
   `/local/tmp/b1_step_probe_run1.log`.
3. Serving bake (TP=2, Qwen3.8-27B, maxlen 32768,
   `_serve_tp2_gfx906.sh` + new `EXTRA_SERVE_ENV` passthrough +
   `_bench_serve_grid_gfx906.py [[2048,256]] 2`): arm A done, arm B
   aborted by host wedges (below).

## Evidence FOR (the framing is superseded — launch-regime)

Per-step decomposition (us; A = LEGACY=1 gather+quant+FA, B = LEGACY=0
Q8-gather+FA, C = LEGACY=0-era direct-paged FA):

| Sk | D=256 A / B / C | B−A | C−A | D=128 A / B / C | B−A | C−A |
|---|---|---|---|---|---|---|
| 2048 | 92.1 / 58.5 / 120.9 | −36.4 % | +31.4 % | 64.9 / 50.7 / 72.0 | −21.9 % | +10.9 % |
| 16384 | 496.3 / 273.0 / 664.9 | −45.0 % | +34.0 % | 414.6 / 301.7 / 446.2 | −27.2 % | +7.6 % |
| 32768 | 970.7 / 533.5 / 1307.2 | −45.0 % | +34.7 % | 815.6 / 588.9 / 878.3 | −27.8 % | +7.7 % |

The FA term is identical in A and B (same kernel, same compact
buffers) — the entire A−B delta is the gather kernel: 50.9/307.5/605.5
us (A) vs 17.4/84.1/168.3 us (B) at D=256. The M5-era gap maps onto
C: direct-paged B=1 is +7.6…+34.7 % slower than A, growing with Sk
(block_table indirection + strided aliased-Q8 reads in the FA kernel
vs compact reads — the wrapper header's old "B=1: gather faster by
~3-6 %" A/B was measured at short Sk, where the penalty is smallest).

## Evidence AGAINST / blockers

- The serving gate cannot complete on this boot: both arm B attempts
  wedged at weight load (`hipErrorLaunchFailure`, both ranks, chronic
  two-card load family; 13:40:36 + 13:42:40) — **2 consecutive
  failures = BURST → host reboot required** (recorded in
  `degradation.md` + `degradation_details.md`).
- Arm A control (LEGACY=1, same boot): B=1 **39.76 / 40.12 t/s**
  (pp2048/tg256; B=4 41.15/41.04 aggregate) — `/local/tmp/b1ab_armA.log`.
  Matches the production no-spec decode band (≈39.7 t/s TP=2 record).

## Why the item's premise changed

Part B (2026-08-28) rerouted LEGACY=0 B≥2 through the fused-Q8 gather;
B=1 follows the same Q8 gather under today's defaults (the direct
branch additionally requires `DIRECT_PAGED_Q8=1`). So "LEGACY=0 B=1 is
2.5–3.7 % slower" (M5 era, direct-paged) is no longer the production
comparison — today's comparison (B vs A) is a 22–45 % kernel-level
WIN for LEGACY=0 that the LEGACY=1 default does not collect. If the
serving gate confirms the transfer, this flips from "close the gap" to
"a default-flip candidate for the roadmap" (with the append-time
quantize cost as the known counterweight).

## Interactions / next steps (post-reboot)

1. Canary (healthy band 38–47 t/s) — if slow, REBOOT AGAIN before the
   bake.
2. Arm B: `EXTRA_SERVE_ENV="GFX906_FA_LEGACY=0"` — bench
   `[[2048,256]] 2` vs arm A's 39.76/40.12.
3. Arm C: `EXTRA_SERVE_ENV="GFX906_FA_LEGACY=0 GFX906_FA_DIRECT_PAGED=1
   GFX906_FA_DIRECT_PAGED_Q8=1"` — adjudicates the M5-era 107.2-vs-111.5
   numbers and the wrapper-header 3–6 % claim on the current build.
4. Teardown between arms: SIGTERM + VRAM-release wait (TP=2 rule).
5. Verdict + roadmap/CHANGELOG records; branch stays unmerged for
   review either way.

VERDICT: OPEN
