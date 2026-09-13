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
