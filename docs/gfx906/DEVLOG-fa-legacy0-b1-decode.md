# FA LEGACY=0 B=1 decode gap — kernel-level localization + serving
# adjudication

## 2026-08-29 — B=1 LEGACY=1-vs-0 decode gap (roadmap item #1)

**VERDICT:** DEAD-END (flip question closed: LEGACY=1 stays the
default) · **GATE:** same-boot (boot O) TP=2 serving A/B,
Qwen3.8-27B B=1 pp2048/tg256, 2 samples/arm — A 40.11/40.11 vs
B 37.61/37.56 (−6.3 %) vs C 37.55/37.54 (−6.4 %) t/s.

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
   `_bench_serve_grid_gfx906.py [[2048,256]] 2`), boot O (canary
   39.3 t/s; boot N arm A 39.76/40.12 kept as cross-boot anchor):
   all three arms ran clean on boot O after two intermittent GPU1
   qcm-fence load wedges (15:51 arm A 1st try, 16:09 arm C 1st try —
   recorded in degradation.md/_details; bare two-card RCCL probe +
   clean runs between break the chain per the boot-L 12:52
   precedent).
4. Append-path cost probe (`legacy0_append_cost_probe.py`, eager,
   B=1 D=256 Hkv=4): the LEGACY=0 per-layer append adds
   `reshape_and_cache_q8` (6.6 us; +16.4 % on the 36.1 us
   triton write), ×16 full-attn layers = **+94.6 us/step eager**.
   Log: /local/tmp/b1ab_*_bootO.log, /local/tmp/b1_step_probe_run1.log.

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

## Evidence AGAINST (the gate fired — launch-regime numbers did not
## transfer)

Same-boot (boot O) serving, Qwen3.8-27B TP=2, B=1 pp2048/tg256,
2 samples/arm (B=4 aggregate in parens):

| arm | path | t/s | vs A |
|---|---|---|---|
| A | LEGACY=1 (production default) | 40.11 / 40.12 (41.15/41.04) | — |
| B | LEGACY=0 (Q8-gather dispatch) | 37.61 / 37.56 (38.24/38.14) | **−6.3 %** |
| C | LEGACY=0 + direct-paged (M5 era) | 37.55 / 37.54 (38.20/38.12) | **−6.4 %** |

Sample spread ≤0.1 % per arm; cross-boot A consistency: boot N
39.76/40.12 vs boot O 40.11/40.11. The kernel probe's B win (−36 % on
the gather+FA subcomponent at 2k) did NOT transfer — the serving step
at 2k context is ~25 ms, of which the subcomponent is ~92 us (0.4 %).

## Why LEGACY=0 loses (mechanism, bounded)

- B and C differ hugely in the FA/gather subcomponent (−36 %/+31 % vs
  A in the kernel probe) yet land within 0.2 % of each other in
  serving → the serving gap is NOT in the FA kernel or the gather.
  It is in what both LEGACY=0 arms share per step.
- Measured share: the append-time Q8 side-buffer write
  (`reshape_and_cache_q8` + slot cast, 16 full-attn layers/step) is
  +94.6 us/step eager — real but an order of magnitude below the
  ~1.55 ms/step serving delta.
- The unexplained remainder is a serving-harness interaction specific
  to LEGACY=0's per-step path: the ~16–32 extra captured graph nodes
  per decode step (Q8 writes + slot casts) add graph-replay node
  overhead that is invisible in eager timing, ± TP=2 sync-placement
  effects. Not further decomposed (a step trace on this stack has
  the documented wall-alignment caveat; eager TP=2 is not a valid
  isolator — it collapses ~3× from launch overhead).
- The steady-state READ path win (Q8 gather 22–45 % below fp16
  gather+quantize per step, growing with Sk) is real but swamped at
  2k context; even at 32k it is ~440 us/step vs the ~1.55 ms
  fixed LEGACY=0 cost — no crossover at reachable context lengths on
  this step-time shape.

## Interactions / refrigerated residue

- M5's verdict ("LEGACY=0 LOSES, default stays 1") is CONFIRMED by a
  proper same-boot adjudication; the M5-era 2.5–3.7 % gap does not
  reproduce at 6.3–6.4 % on the current build (different era —
  direction unchanged). The wrapper-header "B=1: direct loses 3–6 %"
  note is superseded by the kernel probe numbers (+8–35 % at
  2k–32k, growing with Sk) for future reference.
- Refrigerated: fusing the Q8 write INTO
  `triton_reshape_and_cache_flash` (one kernel writes fp16 K + Q8
  bytes) would cut the append delta to ~0 and the graph nodes in
  half — but the node-overhead remainder would still stand, so this
  alone would not close a 6 % gap; revisit only with a graph-node
  overhead measurement.

VERDICT: DEAD-END (flip question closed; records: DEAD-ENDS.md,
CHANGELOG; branch stays unmerged for review)
