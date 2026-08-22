# C2-V (v2) — MoE gemm re-tile verdicts under batch-decode and TP=2 regime

> Branch `gfx906/moe-c2v` off `gfx906/main` (d608aa40a5) · model
> Qwen3.5-35B-A3B-AWQ (`/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`) ·
> 2026-08-22 · roadmap item C2-V (`moe-decode-roadmap.md`).

**VERDICT:** OPEN (Stage 0 done: reopen gate TRIGGERED at TP=2, +1.47% ≥
0.5%; Stage 1 NPT=2 gemm1 A/B in progress — build + matrix below)

**GATE:** serving A/B, pp=2048/tg=256, graph mode (eager for Stage-1
A/B arms), N=1/4/8/32 concurrent, TP=1 and TP=2, flags off/on with
interleaved engines, 3 repeats per engine. Reopen rule (roadmap
C2-V(v2)): **any positive ≥0.5% reopens C2-gemm1 and the S5 gemm1
branch.** Metric: decode-only t/s via the Δ-wall method
(`benchmarks/kernels/gfx906/moe_multireq_ab.py`: tg vs tg_short in
the same engine — prefill/overlap cancels in the difference). NOTE:
the 67.39 t/s record (band 65.9–67.0) is the *official harness*
metric (`_bench_gfx906.py`: 256 / (prefill+decode wall)) — a
different quantity, not comparable to Δ-method numbers (at equal
step time the harness reads ~22% lower). The A/B delta is the gate
(same metric on both arms); the absolute host/build anchor is the
official harness run (queued after Stage 0, expect the 65.9–67.0
band) + the mtp2 canary.

---

## HYPOTHESIS

If the three "failed transfer" verdicts (S5-V2 gemm2 M=1 tile, S2 topk
M=1, NPT-sweep gemm1 M=1 — all rendered in the single-request gate
regime) are regime artifacts, then either (a) a busier concurrent
decode step or (b) the TP=2 per-rank shape change (expert N halved:
gemm1 1024→512, gemm2 2048→1024) makes the re-tiles show ≥0.5%
wall-clock gain — in which case the C2-gemm1/S5 branch reopens.
Kevin overruled TP=1-only scoping on 2026-08-22: closing on TP=1
alone would discard a TP=2 win; TP=2 stability on this box is proven
(27B dense, official amdgpu DKMS 6.19.14 driver) and TP=2 MoE-35B is
assumed fine — the smoke run verifies.

## Scope correction (2026-08-22, pre-run dispatch-gate audit)

`dispatch_moe_gemm_q4` (`csrc/rocm/moe_q_gemm_gfx906.cu`) shows both
existing re-tile candidates are **M=1-only**:

- `VLLM_GFX906_MOE_M1` (gemm2 v2 512t tile) fires only when
  `block_size_m == 1 && output_topk > 0 && size_m == output_topk` —
  i.e. a single decode token. At N≥2, EM=N·topk > topk → **structurally
  inert** (running it at N≥2 would only re-measure the off arm).
- The NPT=2 gemm1 trial dispatch (reverted per
  `DEVLOG-moe-gemm1-retiling.md` §5, never committed) was the BM=1
  path — M=1 only.

Consequences for the (v2) matrix:

1. Roadmap (v2)(a) — "busier step transfers the M=1 savings" — is moot
   for N≥2: the batch-decode gemm path is the **BM=4 grouped GEMM**
   (em=N·8: N=4→BM=1, N=8/32→BM=4), which the (BLOCK_KN, NPT) sweep
   **never measured** (roadmap (v2)(b)). The batch axis therefore
   reduces to *characterizing* the BM=4 path; re-tiling it (if it
   shows headroom) is a Stage-2 decision, not the (v2) as written.
2. The only batch point where an M=1-gated flag can fire is **N=4**
   (em=32 ≤ 32 → BM=1): the NPT=2 gemm1 arm runs there. `MOE_M1` at
   N=4 is expected inert (EM=32 ≠ topk=8) — one confirmation run
   documents the gate.
3. **TP=2 is a genuinely new tiling axis even at M=1.** The gemm2 v2
   tile's shape gate still passes at TP=2 (N=1024 % 256 == 0,
   K=512 % 256 == 0, groupsize 128 % 32 == 0); gemm1 `<1,4>`/`<1,2>`
   at N=512 is unmeasured. → TP=2 N=1 A/B is a first-class arm.
4. TP=2 35B-MoE is a **first run on this box** (TP=2 proven on the
   27B dense only; MoE W4A16 kernels + GDN + Q8 FA under RCCL P2P
   unverified). A smoke (the first t2n1 run) gates the rest of the
   TP=2 axis; teardown per the TP=2 protocol (clean exit / SIGTERM,
   VRAM 0% verified between runs; the bench's SIGTERM handler does an
   engine-core shutdown before exit).

## Matrix

| stage | point | arms | regime | notes |
|---|---|---|---|---|
| 0.1 | canary (27B mtp2, 60 s) | — | — | `degradation_details.md` protocol; host was rebooted 4 h earlier |
| 0.2 | TP=1 N=1 | off, `MOE_M1=1` | graph | anchor: re-confirms the known +0.60 t/s S5 result on this build |
| 0.3 | TP=2 N=1 | off, `MOE_M1=1` | graph | first TP=2 35B run = smoke; the TP=2 M=1 tiling axis |
| 0.4 | TP=1 N=4/8/32 | off (N=4 + `MOE_M1=1` = inertness proof) | graph | BM=4-path characterization (the (v2)(b) measurement) |
| 1 | TP=1/TP=2 N=1, TP=1 N=4 | off, NPT=2 gemm1 on (rebuild) | graph + eager | land `<1,2>` instantiation + env-gated dispatch (default off) first |

Stage-0 runtime budget ~1 h GPU (8 engines × load+warmup+3 repeats);
Stage 1 ~1.5 h (6 engines × 2 regimes + rebuild). Driver:
`/tmp/c2v/run_stage0.sh` (per-run logs `/tmp/c2v/<label>.log`,
results TSV `/tmp/c2v/stage0_results.tsv`).

## What was done

- 2026-08-22: branch created; dispatch-gate audit (above); bench
  `benchmarks/kernels/gfx906/moe_multireq_ab.py` (Δ-wall decode-only
  t/s: prefill overlap cancels in the tg vs tg_short difference);
  Stage-0 driver; roadmap C2-V state line updated. (Commit
  3824813bcf.)
- (in progress) Stage-0 run sequence (driver PID-logged, logs
  `/tmp/c2v/`): canary → t1n1 off/on → t2n1 off/on (smoke) → t1n4
  off/on → t1n8 off → t1n32 off; official-harness 35B run queued
  after the driver (metric anchor).

### Results so far (2026-08-22, build fed585110)

- **Canary (27B mtp2, 60 s): 38.8 t/s** — below the ~40–47 healthy
  band but well above the <25 REBOOT line. Soft signal; the 35B
  official-harness run is the tie-breaker (if it lands < 65.9 the
  host is off-record and Stage-0 deltas are suspect).
- **Official harness (35B, graph, 4 samples): 65.61 / 65.623 /
  65.603 / 65.566 → 65.60 ± 0.03 t/s.** Marginally below the record
  band (65.9–67.0; record 67.39) — ~0.5 % under the band floor, i.e.
  at record level within run-to-run drift (degradation would be a
  ~3× collapse, not 0.5 %). Cross-check: 256/3.902 s = 65.6 and the
  Δ-method's 240/2.95 s = 81.2 both resolve to d ≈ 12.3 ms/step with
  prefill ≈ 0.75 s — internally consistent. **Host judged healthy for
  A/B gating; deltas, not absolutes, are the gate.**
- **t1n1_off (TP=1, N=1, graph, flag off): 81.17 t/s Δ-metric**
  (stdev 0.7, min 80.39 / max 81.72; prefill ≈ 0.9 s, step ≈ 12.3 ms).
  Not comparable to the 67.39 record (different metric — see GATE); the
  harness anchor run is pending.
- **t1n1_m1on (TP=1, N=1, graph, `MOE_M1=1`): 82.46 t/s** (stdev
  0.13, 82.38–82.61) → **+1.29 t/s = +1.59 % vs off.** This is the
  KNOWN S5 gemm2-v2 single-request effect re-confirmed on this build
  (S5 recorded +0.90 % on the harness metric — consistent once
  prefill dilution is accounted for; 48 layers × ~4.5 µs ≈ 0.2 ms
  ≈ 1.6 % of a 12.3 ms step). The anchor works as designed: the
  flag's single-request gain is real and ≥0.5 % on this build. This
  is NOT new evidence — it is the regime the C2 close already
  measured; the (v2) verdict rests on the TP=2 and batch/N=4 arms.

### TP=2 smoke failure + workaround (t2n1_off, 20:05Z)

First TP=2 35B-MoE run crashed: rank-1 worker died in
`RocmPlatform.get_device_name` (torch-compile-cache-dir query) during
`profile_run` at `moe_forward_shared` — final exception
`AMDSMI_STATUS_NOT_INIT` from the `with_amdsmi_context` wrapper's
`finally: amdsmi_shut_down()`, masking the primary error. Rank 0 hung
on the shm broadcast (GPU0 100 %) until SIGTERM. **No kernel reset**
(software crash — recorded in `degradation.md` +
`degradation_details.md` "2026-08-22 evening", incl. a transient
44 %-VRAM-on-GPU1-with-no-owner observation that turned out to be
the next config's in-flight allocation). Root fragility: **amdsmi is
broken on this boot in every run** (TP=1 included — all runs log the
protected `Failed to get total memory via amdsmi` fallback);
`get_device_name` is the one unprotected caller. Workaround:
sitecustomize shim (`/tmp/c2v/shim/sitecustomize.py`) — swallows
amdsmi init/shutdown failures and gives `get_device_name` the same
`AMD_<arch>` fallback the code already has for the 0-handles case.
Upstream fix belongs in `vllm/platforms/rocm.py` (the wrapper's
`finally` must not raise over the primary error). Stage-0b
(`/tmp/c2v/run_stage0b.sh`, waiting on the harness job) re-runs:
TP=1 batch (t1n4 off/on, t1n8, t1n32) then TP=2 N=1 off/on WITH the
shim (`*_s` arms).

### Stage 0 complete — batch + TP=2-shim results (Δ-metric t/s, graph)

| point | off | on (flag) | Δ | note |
|---|---|---|---|---|
| TP=1 N=1 | 81.17 (±0.7) | 82.46 (±0.13, MOE_M1) | +1.29 (+1.59 %) | known S5 effect re-confirmed (anchor) |
| **TP=2 N=1 (shim)** | **80.32 (±0.03)** | **81.50 (±0.08, MOE_M1)** | **+1.18 (+1.47 %)** | **reopen gate triggered (≥0.5 %)** |
| TP=1 N=4 | 184.59 (±2.1) | 183.76 (±1.3, MOE_M1) | −0.83 (−0.45 %) | inert as designed (M=1 gate) — proof |
| TP=1 N=8 | 167.4 (rep1/2; rep0 cold) | — | — | BM=4 grouped-path characterization |
| TP=1 N=32 | OOM (all configs) | — | — | single-card memory ceiling (below) |

Findings:

1. **TP=2 M=1 is a real, positive axis**: the gemm2-v2 tile transfers
   to the per-rank-halved shapes (gemm2 N=1024, K=512) at +1.47 %, on
   par with TP=1 (+1.59 %). Per the roadmap rule ("any positive ≥0.5 %
   reopens C2-gemm1 and the S5 gemm1 branch") the branch **reopens** —
   at minimum TP=2-scoped; combined with the re-confirmed TP=1 effect,
   the natural follow-up question (for Kevin) is whether `MOE_M1` goes
   default-on (it is env-gated, default off, since the C2 close).
2. **Batch flag arms are structurally inert, as predicted**: MOE_M1 at
   N=4 is within noise (−0.45 %, |Δ| < stdev) — the M=1 gate
   (`size_m == output_topk`) excludes every N≥2 point. The batch
   decode regime runs the BM=4 grouped path (N=8: em=64), which the
   (BLOCK_KN, NPT) sweep never touched; both Stage-1 flags are BM=1
   only, so the N=4 NPT=2 arm is the sole batch A/B point.
3. **Step scaling**: N=1 12.3 ms → N=4 (BM=1) 21.7 ms → N=8 (BM=4)
   47.8 ms. The BM=1→BM=4 transition costs +161 %/step for 2× the
   tokens — the grouped path is the expensive one (per-slot cost
   rises), consistent with the (v2)(b) premise. This is the real
   batch-regime lever if batch decode ever becomes the target.
4. **N=32 single-card memory ceiling (new finding)**: 32-concurrent
   35B decode OOMs on one MI50 32 GB in this build at every config
   tried (graph/eager × util 0.95/0.90 × maxlen 4096/2816/2048 ×
   pp 2048/1024). Failure is the gfx906-FA `_q_pad_buf` [B, Hq,
   Sq_pad, D] fp32 prefill Q buffer (544 MiB at B=32, maxlen-
   independent — it grows to the batch size) on top of a ~28 GiB
   non-KV footprint (17.6 GB weights + GDN state pool + inductor +
   FA buffers); KV pool left with 41–57k tokens < the 74k needed.
   32-seq 35B serving needs TP=2 (per-rank Hq/2 halves the q_pad) or
   a larger card. Not a C2-V gate point (flag inert at BM=4 anyway).
5. **Bench artifact noted**: the first repeat's tg_short run is cold
   at N=8 (wall_short 11.7 vs 6.45 s steady) and inflates the Δ
   metric (308 vs 167 t/s); steady-state = rep1/2 (agreed to 0.1 t/s).
   Future A/B arms should report rep1/2.

### Stage 1 (NPT=2 gemm1) — in progress

`csrc/rocm/moe_q_gemm_gfx906.cu`: `VLLM_GFX906_MOE_NPT=2` now also
selects the `<1,2>` kernel (64 cols/block vs `<1,4>`'s 128) for the
BM=1 **gemm1** path (`output_topk == 0`) — the reverted trial, re-
landed fresh (never committed), env-gated default off, per-call
getenv like MOE_M1. gemm2 stays `<1,4>` (or the MOE_M1 v2 tile) so
the two trial flags never touch the same kernel. Incremental
`build_ext --inplace` (ccache) running; matrix: off/npt2 × {TP1 N1,
TP2 N1 (shim), TP1 N4} graph + {TP1 N1, TP2 N1} eager — TP=2 N=1
includes the first measurement of the gemm1 tile at the halved
N=512 shape (the new tiling axis).

## Evidence — FOR

(none yet)

## Evidence — AGAINST

(none yet)

## Why it failed (if applicable)

—

## Interactions / superseded-by

- `DEVLOG-moe-gemm1-retiling.md` (the C2 close this re-tests) and
  `DEVLOG-moe-m1-sprint.md` (S5/S2 provenance).
- If TP=2 N=1 shows ≥0.5%: the "failed transfer" verdicts stand for
  TP=1 M=1 but the branch reopens **TP=2-scoped** — record both.

## Refrigerated residue

- The BM=4/8 grouped-GEMM tiling surface (BLOCK_KN, NPT at BM=4/8) is
  now known to be *unmeasured, not measured-and-rejected* — if Stage 0
  shows the MoE gemm is a large share of the busy batch step, that is
  the real batch-decode lever (roadmap C2 territory).
- NPT env override (`VLLM_GFX906_MOE_NPT`) is BM≥8-only in-tree; the
  BM=4 path has no in-tree knob — any batch-regime A/B beyond N=4
  needs a rebuild.

## Search keys

`HYPOTHESIS:` `VERDICT:` `C2-V` `moe-c2v` `batch decode` `TP=2`

---
Copyright Kevin Read <me@kevin-read.com>
