# Diagnosis (read-only) — MTP2 regression on `gfx906/fa-masked-gather`

Copyright Kevin Read <me@kevin-read.com>

**Scope:** root-cause the newly-OOPENed mtp2 regression recorded in the re-baseline
devlog entry (24.9 t/s steady vs the pre-merge mtp2 record 39.9, with
plain-greedy unaffected at 40.9). This investigation was done **read-only** — no
rebuilds, no GPU runs, no code edits. Everything below is from source cross-referencing
and the up-to-date git history on this branch (`79b9cf3732`).

---

## The regression as recorded

Re-baseline gate (TP=2, util 0.93, max_num_seqs 4, capture sizes [1,2,3,4],
`--speculative-config {mtp, n=2}`, 1091-token prompt, tg 256):

| arm | steady t/s | Δ vs P0 |
|---|---|---|
| 131k P0 → P1 | 16.50 → 24.90 | +51% |
| 262k P0 → P1 | 12.54 → 24.91 | +98% |

- N4 tax-removal **does** transfer (P0 tax −24%, P1 residual 0.0%) — that part is
  healthy and consistent with S8.
- **New OPEN:** mtp2 steady 24.9 < plain-greedy P1 40.9 on the *same* harness.
  Acceptance is healthy (mean acceptance length 3.00), so the pathology is step time:
  **~120 ms/verify step vs ~24 ms plain decode step (~5×)**.
- **Ruled out as the cause:** the N4 change (PERSIST=0 reproduces the regression),
  and the mtp2 kernel/framework work (proven identical to the 39.37 build; see F1/F2).

---

## What I verified (source + git)

### F1 — The entire spec-decode worker/verify path is byte-identical to the 39.37-record build

`git diff gfx906/spec-decode..HEAD -- vllm/v1/worker/` → **no output**. The gfx906
spec-decode tip `bcfe978720` is an **ancestor** of HEAD, and nothing in
`vllm/v1/worker/gpu/gpu_model_runner.py`, `…/spec_decode/*`, `…/cudagraph_utils.py`,
or the MTP/Mlp speculators changed between the record build and HEAD. The only non-gfx906_fa
VLLM deltas are in `config/*`, multimodal/vision parsing, MOE WNA16 oracle, and platform
(RoCm) plumbing — none touch the spec-decode execution path.

### F2 — The exact mtp2 kernel numbers for the *current* build configuration are known and good

Commit `cfe09d8611` ("split q_gemm 4-bit kernel on M for max-ilp") — present in HEAD —
documents the serving A/B on this **split** build (which is the current default,
no `VLLM_NO_MAX_ILP`):

```
  baseline 26.44 | 28.56 | 27.99  t/s
  ngram3    28.92 | 27.80 | 28.03  t/s
  mtp2      39.74 | 36.67 | 39.37  t/s   (no-max | full max | split)
```

So on this very build configuration the mtp2 kernel path measures **39.37 t/s** — the
q_gemm M-split (the thing most people would blame for an mtp2 regression) was *already*
tuned to keep mtp2 within CI of the no-max record. This rules out the obvious build-
config/ILP suspect. The MAX_ILP CMake section, the `.cu`/`.hip` split, and the
kill-switch are all present and consistent with that validated state.

### F3 — The N4 persistent-gather frontend change is clean for mtp2 and PERSIST=0 absolves it

The N4 dispatch (`gfx906_fa_paged.py: forward_paged`) routes the *target-model gather
and FA layers* through the persistent kernel when `GFX906_FA_PERSIST` is ON. The spec
decode *draft/verify* path through the target runner inherits this the same way plain greedy
does. Since plain-greedy P1 is **40.9** (healthy) and mtp2 **PERSIST=0** is also
24.9, the N4 kernel is not the cause. The `num_seqs ≤ 16` cap in the persistent
kernel also isn't hit at capture sizes [1,2,3,4].

### F4 — The FULL-graph coverage decision for spec-decode is intact in HEAD

The load-bearing commit `4e40e3eee2` (enable FULL cudagraphs for spec-decode steps)
survives: `get_cudagraph_support` returns `UNIFORM_BATCH` when
`vllm_config.num_speculative_tokens > 0`, and the capture-safe uniform fast paths
(`num_tokens == num_seqs * max_seqlen_q`) are present in `forward_paged`. So the "spec
demotes to PIECEWISE (~3×)" trap the original commit fixed is not re-reflected in source.

---

## Assessment of root-cause candidates (given F1–F4)

Because the tracked spec-decode code is provably identical to a 39.37 build, **no tracked
source diff can explain a 39→25 collapse on its own.** The regression must come from one or more
of the following. I rank them in confidence order.

### A. [HIGH conf.] Historical-record vs re-baseline measurement are **not apples-to-apples**

The strongest and most defensible explanation. The S8/S7 mtp2 records this is compared
against are internally inconsistent about their own operating point:

- S7's table lists TP=1 mtp2 = 39.74 and TP=2 mtp2 = 39.7, but the
  post-commit-3 note explicitly disclaims S5's 39.7: *"S5's 39.7 was at
  max_seq_len 4096 per tp2-bench-final.log"*. Those two statements point at **different
  configs** (one 131k, one 4096).
- S8's 39.9/29.9 (131k/262k mtp2) were produced by a **different harness**
  (`tp2_serve_bench2.py`, batch=1, pp2k/tg256) and a pre-merge (v0.27.2) build.
  The re-baseline here uses the candidate harness and a post-merge build.
- **A ~5× verify step was never a documented property of the record runs**; there is no
  recorded "verify step time ≈ plain step time" for the S5/S7/S8 mtp2 cells. The only
  *documented* mtp2 verify-step comparison in-tree is the n-gram/spec work in
  DEVLOG-spec-decode, at different shapes.

So the "120 ms/verify" observation is real for the re-baseline harness but is **uncalibrated
against the record harness**. Before any debugging, this is the first thing to close: run the
record harness (S5 recipe) on the current build. If mtp2 there is ~39 again, the
"regression" is harness-relative, not absolute. This is the cheapest discriminating test.

### B. [MED conf.] Upstream merge changed a **shared/host-side** execution detail the gfx906 diff misses

The `vllm/v1/worker/` path is identical because gfx906/spec-decode **is** the 
ancestor — but gfx906/main merged **upstream v0.27→v0.28** (`fc777b87dd` + RC2
commits) into the *pre-spec-decode* la
history. The spec-decode work then sits on top. So an upstream behavior change to a dependency
the gfx906 code calls (e.g. scheduler padding, batch descriptor token counts, `max_query_len`
under spec decode, MTP draft KV handling, the v1 gpu model runner *outside* the diff window,
rejection-sampler FPGA/floating determinism, or the adaptive-verification threshold) could
silently make the non-fused multi-step decode path much slower without touching the files gfx906
customized. In particular:

- `use_fused_multi_step_decode` defaults **False** for the gfx906 FA backend (it does not
  implement `supports_draft_decode_metadata_update`, so `unsupported_backends` is non-empty)
  *and* `advance_draft_positions=True` (standard MTP) — so the draft path rebuilds
  attention/gdn metadata between every speculative step. If an upstream change made that metadata
  rebuild (or the per-step KV/token bookkeeping) heavier under the merge, the unfused
  per-step overhead grows one-for-one with the ~3.0 acceptance length. This is a plausible,
  testable mechanism for "verify step much heavier than plain decode."
- The hybrid (GDN/mamba) structure of qwen3.5-27B means the draft target forward touches
  GDN linear-attention state, which the upstream merge has actively changed (many in-mergee
  commits: "GDN", "Mamba", "KV offload / decode bench connector"). An upstream GDN-decode
  change absorbed during the merge is a live suspect for the mtp2-specific slowdown.

### C. [MED-LOW conf.] Build environment / binary staleness or GPU/driver state

The `.so` binaries are dated 08-21/08-22 and match the current sources, so a stale-build
explanation is weak. But the documented per-session flakes (`.about 1/3` TP=2 inits
wedge GPU1; BACO recovery) and cross-session variance mean a single 24.9 steady value taken
right after a flakey session is one sample, not a bound. Worth reconfirming across clean
sessions and clean driver state before treating 24.9 as the parity number.

### D. [LOW conf.] N4 persistent kernel interacts with mtp2 at capacity

Ruled out by PERSIST=0 reproducing it. Kept only for completeness; do not chase.

---

## Recommended next steps (read-only-verifiable / cheap experiments)

1. **Calibration of the record harness (highest priority).** Re-run the exact S5/S7 mtp2
   recipe on the current build — the `tp2_serve_bench2.py`/S5 config at 131k, plus
   the S8 131k/262k cells. If mtp2 ≈ 39 there, the re-baseline harness is the
   discrepancy, not the model path. This also gives a clean verify-step-time on both harnesses
   so the "120 ms" claim can be compared directly.
2. **Profile/isolate the verify step time** on the re-baseline harness (the ~120 ms/verify
   metric): capture a per-stage split (draft forward, GDN state update, FA verify, recompile
   rejection) — a torch-profiler correlation like the P3-4 technique, or the
   `spec_prof_probe.py`/`spec_step_probe.py` already in-tree. This directly answers whether
   the 5× is one heavyweight op or accumulated per-step rebuilds.
3. **A/B the draft-step path:** if the profile shows per-step metadata rebuild dominating, test
   forcing `use_fused_multi_step_decode=True` or try the `FULL_AND_PIECEWISE` vs
   `FULL_DECODE_ONLY` cudagraph mode to see if the fused draft path recovers the gap. This
   isolates the upstream-merge-metadata-rebuild hypothesis (B) without changing tracked code.
4. **Confirm `get_cudagraph_support` actually fires UNIFORM_BATCH at runtime** for the qwen
   mtp2 + TP=2 config (a one-line env-gated log). If it returns
   `UNIFORM_SINGLE_TOKEN_DECODE` or the spec config isn't visible at backend-init, the F4
   FULL-graph guarantee silently breaks and produces exactly this symptom. Cheap and decisive.
5. **Re-confirm across clean sessions/driver state** (B/C) before locking a number.

---

## Bottom line

**Not a code regression in the gfx906 spec-decode work, and not the N4 change.** The mtp2
kernel/framework path is byte-identical to the validated 39.37 (cfe09d8611) build, and
PERSIST=0 absolves the FA change. The 24.9-vs-39.9 gap most credibly reflects either
(a) two non-comparable harnesses/configs (record S5=39.7 was at least inconsistently
documented — one statement says 131k, another says maxlen 4096), or (b) an upstream
v0.27→v0.28 change to the *unfused* draft/metadata-rebuild path (via GDN/hybrid or
shared worker code) rather than the gfx906-tuned kernels. The decisive cheap tests are steps 1
(record-harness rerun on current build) and 4 (confirm runtime UNIFORM_BATCH); step 2
(verify-step profile) localizes the cost precisely. Unless one of those reproduces a genuine
in-code path difference, mtp2 should not be declared "regressed" — it should be re-referenced
against a matched harness and a clean-session sample.
