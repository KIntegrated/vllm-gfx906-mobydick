# C2-gemm1 retiling: V1 full-K direct store + z-split sweep (2026-08-19)

Branch `gfx906/moe-gemm1-v1` off `gfx906/main`. Model: Qwen3.5-35B-A3B-AWQ
(local NVMe). Item: `moe-decode-roadmap.md` C2 — "gemm1 (≈1070 µs/step) stays
open; the V1 full-K single-wave design and the activation-fusion idea are the
next candidates."

**Verdict up front: all retiling of the decode-size gemm1 is closed.** V1 is
rejected (2.2–4.3× slower standalone), the full (BLOCK_KN, NPT) tiling surface
was mapped (best point 12% off current in the launch regime), and the best
point's in-model gain **does not transfer to serving wall-clock in either
eager or graph mode**. No dispatch change shipped; the harness extensions and
the measurement findings are the deliverables.

## 1. V1 (full-K per block, direct store) — REJECTED

Roadmap design: 64 blocks for gemm1 (8 slots × 8 n-tiles), each a single
wavefront looping the full K=2048; kills atomics + both zeroings.

Two variants added to `benchmarks/kernels/gfx906/harness/moe_m1_harness.cu`
(`moe_gemm_q4_v1<THREADS, NPT, K_T>`), gemm1 shape (N=1024, K=2048, 8 slots),
launch regime, 2000 launches (µs/launch, band over all post-fix runs):

| variant | blocks × waves | µs/launch | vs current 26.9 |
|---|---|---|---|
| current (256t, z=8 CAS, no zero) | 64 × 4 | 26.9–27.1 | — |
| v1b (128t = 2 waves, NPT=1, full-K, direct) | 64 × 2 | 60.4–60.5 | **2.2× slower** |
| v1a (32t = half wavefront, NPT=4, full-K, direct) | 64 × ½ | 117.2–117.4 | **4.3× slower** |

Both pass correctness (max err 0.2511 = current's fp16 noise; the first v1
build had an `a_off` bug — it used `8*j` instead of `k + 8*j` for the
activation slice, i.e. every 32-K step re-dotted with a[0..32); fixed before
any timing was taken).

Mechanism: 64 blocks each streaming one long 128 KB weight sequence cannot
keep enough bytes in flight on the MI50 (rough estimate: ~128–512 KB in
flight vs ~480 KB needed at ~300 GB/s × ~600 ns HBM latency). The current
kernel's 8-way K-split keeps 64 independent 16 KB streams turning over.
Note v1a is a half-wavefront (32-thread) block on this wave-64 part — an
inefficient config regardless; v1b is the fair test and still loses.

## 2. z-split / NPT surface sweep — NPT=2 is the launch-regime best point

Templated copy of the in-tree kernel (`moe_gemm_q4_sweep<THREADS_X, NPT>`,
`THREADS_X == BLOCK_KN`) in the same harness, all 8 configs correctness-
checked against the CPU reference:

| config | blocks (gemm1) | CAS fan-in | µs/launch (3-run medians) |
|---|---|---|---|
| **256t NPT=2** | **128** | 8 | **23.78 / 23.81 / 23.90** |
| 256t NPT=4 (in-tree) | 64 | 8 | 26.88 / 26.95 / 27.08 |
| 128t NPT=2 | 512 | 16 | 25.07 / 25.14 / 25.14 |
| 128t NPT=4 | 256 | 16 | 27.84 / 27.94 / 27.88 |
| 64t NPT=2 | 2048 | 32 | 25.13 (single run) |
| 64t NPT=4 | 1024 | 32 | 33.39 (single run) |
| 512t NPT=2 | 64 | 4 | 29.16 (single run) |
| 512t NPT=4 | 32 | 4 | 37.31 (single run) |
| v2 512t 2col (S5) | 64 | 8 | ~27.1 |
| v2 512t 4col (S5) | 64 | 8 | ~28.1 |

NPT=2 beats NPT=4 at every BLOCK_KN (more blocks = more independent streams);
256t/NPT=2 is the best point, −3.1 µs (−11.6%) vs the in-tree config.

**This also closes V3/V4 without building them:** V3 (2–4-way K-split into
fp32 scratch + add kernel) adds ~184 µs/step of scratch+launch cost on top of
a CAS design whose best point doesn't transfer (§3) → negative expectation.
V4 (halve CAS fan-in, NPT=8/z=32): the fan-in axis is monotone-worse as the
CAS count rises (27.8 → 33.4 from z=16 → z=32 at NPT=4), so fewer CAS is not
where the gain is.

## 3. In-model trial of NPT=2 — the gain does not transfer

Trial (reverted after A/B): `case 1:` in `dispatch_moe_gemm_q4` launched
`<1,2>` for gemm1 (output_topk == 0) under `VLLM_GFX906_MOE_G1NPT2` (read
per call, default off). Gates:

1. Unit test (2 new, since removed with the revert): flag flips the output
   (tile taken) + both paths within 1e-1 relative of the fp32 reference,
   layouts awq_kfirst + wna16_sym. Suite: 27/27 with the change, 25/25 after
   revert. (Known pre-existing flake: M8-bm4-N1024-awq_kfirst_sym at the 5e-2
   boundary, passes 3/3 in isolation.)
2. PPL probe (fixed 12-prompt, off vs on, same build): 16.0081 vs 16.0011
   (noise; PPL is prefill logprobs = BM≥8 path, untouched by the flag).
3. In-model eager kernel census (`kernel_prof_probe.py`, 256-step profile):
   gemm1 `<1,4>` 29.8 µs/call (38.7/step, 1155 µs/step) → `<1,2>` 25.0
   µs/call (968 µs/step): **−4.8 µs/call, −186 µs/step**. gemm2 `<1,4>`
   unchanged 26.9 µs/call (by design).
4. **Serving A/B, graph mode** (the gate; `docs/gfx906/_bench_gfx906.py`,
   pp=2048 tg=256, 4 samples, `BENCH_MAX_SEQS=32`, GPU util 0.95):
   off 66.614/66.542/66.534/66.442 (mean 66.53) vs on 66.602/66.611/66.586/
   66.640 (mean 66.61) → **+0.08 t/s, neutral** (well inside the off arm's
   own 0.17 spread).
5. **Serving A/B, eager mode** (`BENCH_EAGER=1`, same config): off mean
   23.504 vs on mean 23.477 → **−0.03 t/s, neutral**.

The census's −186 µs/step never appears in wall-clock in *either* regime.
This is the third consecutive decode-size gemm1 retiling to fail transfer
(S5 V2: standalone 1.18×, in-model neutral; S2 topk: standalone win, graph
replay loss), and it extends the roadmap's §1 warning — per-kernel profiler
rows are pipeline-state dependent even in eager, not only in graph contexts.

## 4. Measurement finding: graph-regime per-kernel census is impossible here

Attempted to measure `<1,2>` vs `<1,4>` kernel time *inside* graph replay
with the torch profiler (throwaway probe = `kernel_prof_probe.py` with
`enforce_eager=False`):

- `FULL_DECODE_ONLY` (the bench's serving config): profiler sees **5.0 gemm
  calls/step of 80** and 1019 µs/step GPU-busy of ~15 ms — ~7% visibility.
- `FULL_AND_PIECEWISE`, `max_cudagraph_capture_size=8`: 4.5–2.4 calls/step,
  858–883 µs/step — equally blind.
- (Default `FULL_AND_PIECEWISE` with `max_num_seqs=32` OOMs the FA
  `_q_pad_buf` during capture of the 40–64-seq sizes — 544 MiB at 30.8 GiB
  used; the bench's `FULL_DECODE_ONLY` + capture-size-8 config is what fits.)

So: on this stack, CUDA-graph replay is invisible to the torch profiler at
per-kernel granularity in both FULL and FULL_AND_PIECEWISE modes. This
confirms, empirically, the roadmap's standing rule that **wall-clock serving
A/B is the gate** for µs-scale verdicts, and retires per-kernel
profiler-based in-model A/B under graphs as a method.

## 5. State and follow-ups

- Reverted: dispatch flag + unit tests (zero measured benefit; vLLM
  no-busywork rule). Rebuilt; 25/25 suite.
- Kept: harness `sweep` + `v1` kernels (the Phase-0 tool that produced this
  verdict) and two harness fixes found in self-review:
  - the `v2-nozero-dirty` run is an intentional negative control but set
    `bad`, and the gemm2 checks used an absolute 0.35 threshold that the
    8-slot fp16 accumulation (≈0.65 abs on ~20-magnitude data) always
    exceeds — the harness printed `HARNESS FAIL` on every main-flow run
    since S5. Both fixed (negative control now asserts the buffer WAS dirty,
    rel 0.97; gemm2 gated at 5e-2 relative like the in-repo suite): the
    harness now reports `HARNESS PASS`.
- C2's remaining substance is the **activation-fusion** idea (fold
  SiLU·mul into gemm1's epilogue, kill the activation round-trip). Given
  §3, its wall-clock transfer expectation is now low — a kernel-structure
  change, not a tiling change, is what would be needed to move the
  wall-clock, and the serving step's critical path has proven insensitive
  to gemm1 kernel time in three independent experiments.
- C3 (fold the two zeroings into neighbor kernels, ~234 µs/step measured,
  no numerics change) remains the cheap C2-adjacent lever if a future
  branch takes it.
- gemm2 NPT=2 at the legacy BM=1 path (and gemm2 in general) was not
  measured on the sweep — the in-tree dispatch routes M=1 gemm2 through the
  V2 tile when `VLLM_GFX906_MOE_M1` is set; the legacy path is not used in
  the serving records.

## Bench log locations

- standalone: `/tmp/v1_ab.log` etc. (wiped on reboot; numbers above are the
  record), binary recipe: `hipcc -O3 -o /tmp/moe_m1_harness
  benchmarks/kernels/gfx906/harness/moe_m1_harness.cu` after
  `source ~/env-rocm-7.14-gfx906.sh`.
- A/B logs (this session): `/tmp/ab_g1npt2_{off,on}.log` (graph),
  `/tmp/ab_eager_{off,on}.log` (eager), `/tmp/prof_{off,on}.log` (eager
  census), `/tmp/gprof_{off,on}.log` (graph census attempts).

Copyright Kevin Read <me@kevin-read.com>
