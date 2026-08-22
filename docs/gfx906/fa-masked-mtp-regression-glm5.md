# MTP regression diagnosis (read-only) — mtp2 on the current build vs the S5/S8-era records

Copyright Kevin Read <me@kevin-read.com>

Reviewer: GLM-5 (agent), 2026-08-22. Scope: the OPEN item from
`DEVLOG-masked-fa.md` post-commit 3 — mtp2 TP=2 steady 24.9 t/s (P1) /
16.5 (P0) vs the pre-merge record 39.9 t/s @131k (S8, acceptance 2.49),
same server recipe. **Read-only session: no servers launched, no code
changed.** Evidence sources: the four mtp2 A/B server logs
(`/tmp/mtp2_ab/`), the surviving S5/S6/S8-era server logs in `/tmp/`
(`vllm-tp2-*.log`, `tp2-eager-*.log`, `r3-v1-262k*.log`), the N4-gate
plain-greedy logs (`/tmp/fa_ab/`), git history, and the vLLM source in
this tree.

**Rev 2 (same day): folds in and adjudicates the sibling review
`fa-masked-mtp-regression-ds4.md`** — every claim marked
VALIDATED/PARTLY/REJECTED in the adjudication section; two corrections
to my own rev-1 text resulted (see "Retractions").

**Rev 3 (same day, later): folds in
`docs/gfx906/fa-masked-mtp-regression-qwen.md`** — which brought NEW
EMPIRICAL DATA (a current-build TP=1 mtp2 rerun at 36.75 t/s and a
per-kernel profiler table; artifacts verified at `/tmp/mtp_prof_probe.py`,
`/tmp/mtp_prof_mtp2{,b}.log`, `/tmp/mtp_prof_plain.log`). Its per-claim
adjudication is the second table below; its headline decomposition is
PARTLY RIGHT but its "TP=1 is fine" leg is REJECTED on per-step
arithmetic (see "Reframing"), which changes what the investigation
should chase first.

---

## The anomaly (restated precisely)

| run | build | config | client t/s (steady) | acceptance | ms/step |
|---|---|---|---|---|---|
| S8 ctl (Aug 21 17:25) | `v0.27.2rc1.dev439+g69f615b98.d20260819` | 131k, mtp2, graph | 39.9 (devlog) | 2.49-2.51 | ~62.5 |
| S8 262k (Aug 21) | same | 262k, mtp2, graph | 29.9 (devlog) | ~2.49 | ~83 |
| mtp2 A/B P0 131k (Aug 22) | `v0.28.0rc2.dev318+gfed585110.d20260821` | 131k, mtp2, graph | 16.50 | **3.00** | 182 |
| mtp2 A/B P0 262k (Aug 22) | same | 262k | 12.54 | 3.00 | 239 |
| mtp2 A/B P1 131k (Aug 22) | same | 131k | 24.90 | 3.00 | 120 |
| mtp2 A/B P1 262k (Aug 22) | same | 262k | 24.91 | 3.00 | 120 |

Steps/s regressed ~2.9× (P0) / ~2× (P1) vs the record. Plain-greedy
decode the same night on the same build ran 40.86 t/s (24.4 ms/step) —
**parity with the era's plain-greedy** — so the platform and the
non-spec step path are healthy; only the mtp2 step is slow.

## What I verified IDENTICAL between the record run and mine (exhaustive)

1. **Server args**: diffed the full `non-default args` lines of
   `vllm-tp2-131k-ctl.log` vs `server_m131072_p0.log` — identical
   except `served_model_name` (cosmetic). Same model snapshot, tp=2,
   util 0.93, max_num_seqs 4, maxlen 131072, capture [1,2,3,4], mtp
   n=2.
2. **Resolved configs**: same `CUDAGraphMode.FULL_AND_PIECEWISE`, same
   `max_num_scheduled_tokens=2048` spec warning (3 occurrences each),
   same Mamba `align` cache mode, same Triton/FLA GDN prefill kernel
   selection.
3. **Capture topology**: both capture "mixed prefill-decode, PIECEWISE
   3/3" + "decode, FULL 1/1". Mechanism traced in source: with
   `uniform_decode_query_len = 1+n = 3`,
   `adjust_cudagraph_sizes_for_spec_decode` rounds [1,2,3,4] → [3]
   (round_up(4)=6 > max 4 is dropped), the gfx906
   `VLLM_GFX906_SPEC_CG_SMALL` patch restores [1,2] for PIECEWISE →
   [1,2,3]; FULL decode keys filter `x ≥ 3` → just [3]. **Identical in
   both eras** — the 1/1 FULL graph is not a merge artifact. Runtime
   dispatch for a 1-seq verify (num_tokens=3, num_reqs=1) matches the
   captured key (`_is_uniform_decode` + `_create_padded_batch_descriptor`
   checked line-by-line) — FULL replay should occur in both builds.
4. **Draft setup**: identical log lines — `Qwen3_5MTP` resolved,
   drafter loaded, embedding + lm_head shared with target, same
   `num_speculative_tokens > 1 … multiple forward` warning present in
   both.
5. **Engine source code**: `git diff 69f615b98..HEAD` on
   `vllm/v1/spec_decode/`, `gpu_model_runner.py`,
   `cudagraph_dispatcher.py`, rejection sampler = **EMPTY**. The
   v0.28.0rc1 merge (fc777b87dd, Aug 20) touched 32 files — none in the
   decode step path (multimodal/vision/kv_offload-tiers/2-line import
   fixes). The gfx906 commits between are MoE-no-zp (Gemma-4 MoE — this
   is a dense model), tp2-debug instrumentation (added then REVERTED),
   and the N4 gather (P0 arm runs the old path; internal P0/P1 A/B is
   clean). The `gfx906/spec-decode` branch is fully merged (no commits
   missing from HEAD); the q_gemm M-split (cfe09d8611) is in both
   builds.
6. **Platform**: plain-greedy 40.86 t/s on the current build+night
   (fa_ab P1) ≈ era plain-greedy — no global degradation, clocks, RCCL
   or driver regression affecting the non-spec path.

**Conclusion from 1-6: nothing diffable differs.** The regression is
NOT in: capture-count collapse, the upstream merge's engine code, the
N4 kernel, unmerged spec-decode work, kernel-build flags (M-split in
both), or the platform.

## Evidence that complicates the record numbers themselves

- **All surviving record-era server logs cap far below their client
  numbers.** `vllm-tp2-mtp2.log` (the S5 mtp2 39.7 server): peak
  accepted/window 15.1 t/s, peak generation-throughput window 26.4.
  `vllm-tp2-final-mtp3.log` (S6 38.6 era): peak 16.2 / 25.3.
  My current-build P0: 19.2 accepted / 28.9 generation — *higher* than
  the record-era logs' peaks.
- **Caveat (calibrated)**: windowed metrics under-report true decode
  rate at these duty cycles — the trusted plain-greedy 40.86 t/s run's
  own log peaks at only 29.8 windowed (prompt phases dilute 10s
  windows). So the era logs' ≤16 accepted peaks are *consistent with*
  a ~39 t/s decode phase at tg=128 duty (≈12.7 expected windowed).
  They neither prove nor refute the records — but they provide no
  positive confirmation either. The raw client outputs of the S5/S6/S8
  sessions are not on disk (session transcripts pruned).
- **The record-era binary also crawled late that night**:
  `r3-v1-262k-c.log` (Aug 21 21:45, v0.27.2 build, mtp3, 262k) peaked
  at 1.6 generation t/s / 12 accepted per window, acceptance 3.00 —
  same crawl-rate regime as my runs, on the OLD binary. The merge is
  not a clean separator.

## Anomalies any follow-up must explain

1. **Acceptance exactly 3.00, per-position rate 1.000** in ALL four of
   my arms (and in the late-Aug-21 r3 run on the old binary), vs
   2.49/0.86 in the S5/S6 records. Plausibly prompt-driven (the A/B
   client's ×10-repeated paragraph is ultra-predictable; the era used
   ocean-essay prompts) — but an exactly-perfect 100% at 190+-token
   scale is also the signature of a draft≡target equivalence (draft
   generated by the full target rather than the 1-layer MTP head would
   both always-accept and triple the per-step forward cost — matching
   the observed ~3× plain-step time). Uniform across P0/P1, so the
   internal N4 A/B stands either way.
2. **Generation throughput = 1.5× accepted** (28.9 vs 19.2) in my run;
   similar ratio in ctl (21.7/13.1). Counter semantics, consistent
   across eras — noted so nobody chases it.
3. **Step-time arithmetic**: plain 24.4 ms + M=3 GEMM premium ~13 ms
   (per the SPEC_CG_SMALL devlog note) + 2× MTP-head draft + rejection
   ≈ 45-60 ms expected; P1 observes 120 ms — ~70 ms unexplained,
   roughly one extra full-model forward per step.

## Retractions (my rev-1 errors, surfaced by the fold)

1. **Rev-1's devlog-side note "S5's 39.7 was at max_seq_len 4096 per
`tp2-bench-final.log`" is WRONG and is hereby retracted.**
`tp2-bench-final.log` carries `speculative_config=None` — it is the
no-spec S5-era bring-up run at 4096, not an mtp2 record. The actual S5
mtp2 server log (`/tmp/vllm-tp2-mtp2.log`) shows `tensor_parallel_size=2
+ max_model_len 131072 + mtp n=2`: **S5's 39.7 was TP=2 @131k.** (This
retraction also dissolves the "one doc says 131k, another says 4096"
inconsistency that ds4's finding A leaned on — that inconsistency was
my own bad citation propagated.) The devlog line itself still needs a
correction commit (bookkeeping below).
2. Rev-1 compared my runs to the S8 mtp2 records without separating the
TWO record families: the TP=1 @`max_model_len 2816` spec-decode-branch
A/B (39.74/39.37) vs the TP=2 @131k S5/S8 records (39.7/39.9). The
fold makes that split explicit (adjudication F2).

## Reframing (rev 3) — the excess has TWO components; qwen's headline is acceptance-confounded

Qwen's verdict table reads "TP=1 mtp2 ≈ 39.4 record → −7% noise; TP=2
lost ≈12 t/s that TP=1 did not" — concluding the pathology is
TP=2 × MTP-specific. That comparison is confounded by acceptance:
their TP=1 run had 3.00 tokens/step vs the record's 1.819. **Per-step**
(the prompt-independent, shape-driven quantity — GEMMs dominate and are
context-independent at B=1):

| | record-era | current | Δ |
|---|---|---|---|
| TP=1 mtp2 ms/step | 45.8 (39.74 t/s ÷ 1.819) | 81.6 (36.75 ÷ 3.00) | **+35.8** |
| TP=2 mtp2 ms/step | 62.4 (39.9 ÷ 2.49) | 120.5 (24.90 ÷ 3.00) | **+58.1** |
| plain TP=1 ms/step | ~39.5 | 24.5 | −15 (faster) |
| plain TP=2 ms/step | ~28.6 | 24.47 | −4 (faster) |
| **mtp2/plain multiplier, TP=1** | **1.16×** | **3.33×** | — |
| **mtp2/plain multiplier, TP=2** | **2.19×** | **4.92×** | — |

So the total excess ≈ +58 ms at TP=2 decomposes into a **TP-common
component (~+36 ms/step, present at TP=1)** and a **TP=2-extra
component (~+22 ms/step)**. Qwen's S1 (TP=2-specific) is real but is
only ~⅓ of the story; their "TP=1 = 82 ms ≈ expected" built the
expectation from the CURRENT plain step (24.5×3=73.5) instead of
comparing to the record's actual mtp2 step (45.8 = 1.16× its plain).
A 3-token verify on a weight-bound GEMM model should sit near
1.2–1.5× plain per the record and first principles — not 3.3×.
**Consequence: regardless of record provenance (H1), the current build
has a real absolute pathology in the mtp2 step at BOTH TP sizes**, and
the TP=1 in-process harness (cheap, no server, no TP complexities) is
the fastest place to chase the common component.

## Cross-review adjudication — `fa-masked-mtp-regression-ds4.md`

| ds4 claim | Verdict | Evidence |
|---|---|---|
| F1: spec-decode worker/verify path byte-identical to the record build; `bcfe978720` ancestor of HEAD; `git diff gfx906/spec-decode..HEAD -- vllm/v1/worker/` empty | **VALIDATED** (both git checks reproduced) — with a framing correction: "the 39.37-record build" and the S5/S8 (39.7/39.9) record build are **different configs** (see F2); tracked-source identity holds for both, so the no-tracked-diff conclusion stands |
| F2: "on this very build configuration the mtp2 kernel path measures 39.37 t/s" (cfe09d8611 A/B) → build-config suspect ruled out | **PARTLY — parity claim REJECTED, narrow claim VALID** | The cfe09d8611 3-arm A/B (mtp2 39.74 no-max / 36.67 full-max / 39.37 split) is the **TP=1, `--max-model-len 2816`, agentic 512-tok prompts, acceptance 90.95 % / 1.819 tok-step** record (`DEVLOG-spec-decode.md` ~lines 900, 1065-1082; 39.37 = 1.407× over the TP=1 baseline 27.99, matching AGENTS.md's spec-decode-branch band). It is NOT a TP=2 @131k number and cannot speak to TP=2 parity. Narrow claim survives: the q_gemm M-split does not regress mtp2 at TP=1 |
| F3: N4 change absolved (PERSIST=0 reproduces the regression; B≤16 not hit at capture [1,2,3,4]) | **VALIDATED** | Same as my "verified identical" section; agreed |
| F4: FULL-graph spec coverage intact (4e40e3eee2 survives; UNIFORM_BATCH under spec) | **VALIDATED** | Commit exists (2026-08-18); `get_cudagraph_support` returns UNIFORM_BATCH when `num_speculative_tokens > 0`. ds4's cited fast-path expression `num_tokens == num_seqs * max_seqlen_q` is a paraphrase — the code has the Sq=1 form (`max_seqlen_q == 1 and num_tokens == num_seqs`) plus the general padded path; immaterial |
| A [ds4 HIGH]: record vs re-baseline not apples-to-apples; S5 39.7 config inconsistently documented (131k vs 4096); run the record harness first | **PARTLY VALIDATED** | VALID: clients differ (S5/S8 = `tp2_serve_bench2.py` ocean-essay pp2k; re-baseline = `tp2_fa_ab_client.py` 1091-tok paragraph), acceptance differs (2.49 vs 3.00), and the record-harness rerun is the cheapest first discriminator (now experiment 1a). REJECTED: the 131k-vs-4096 "inconsistency" is my own mis-citation (Retraction 1) — S5 mtp2 was @131k TP=2; and harness difference alone cannot produce the gap: both clients define t/s identically ((n-1)/(t_last−t_first)), and the record era's *longer* prompt (2048 vs 1091 tokens) would bias step time the other way. Server-side 62.5→120 ms/step at similar-or-better acceptance needs a step-path explanation, not a metric one |
| B [ds4 MED]: upstream v0.27→v0.28 changed the unfused draft/metadata-rebuild path (GDN/mamba, scheduler, `use_fused_multi_step_decode`); "spec-decode work sits on top of the merge" | **MOSTLY REJECTED** | Mechanism kernel verified: `use_fused_multi_step_decode` requires `supports_draft_decode_metadata_update` on every attention group (`speculator.py:100-115`), which Gfx906FABackend does not define → per-draft-step metadata rebuilds are real *if* MTP drafts route there — but this is a **constant across both builds** (speculator.py byte-identical per F1), so it cannot be the delta. The claimed upstream carriers are absent: the merge's entire `vllm/` delta is `kv_offload/tiering/base.py` (4 lines), multimodal parsing, model files, and a multimodal-only rename in `config/vllm.py` (full diff read) — **no scheduler, GDN, or decode-path change exists in the merge**. Lineage correction: the spec-decode work does NOT sit on top of the merge — bcfe978720 (Aug 19 11:48) is an ancestor of the S8 build (Aug 19 21:01) which is an ancestor of the merge (Aug 20); verified with merge-base. Anything B needs lives outside git → collapses into my H1/H3 |
| C [ds4 MED-LOW]: binary staleness / single-sample 24.9 | **PARTLY VALID** | Their side (current binaries match sources) — agreed, consistent with my plain-greedy parity. But 24.9 is not a single sample (4 A/B arms steady 24.90/24.91, plus the Aug-21 r3 old-binary crawl in the same regime), and ds4 never examined the *record* binary's provenance — which is exactly my H1 (dirty Aug-19 build, never rebuilt, state unrecoverable) |
| D: N4-at-capacity — rejected | **VALIDATED (agree)** | — |
| Their steps 1-5 | folded | Step 1 (record-harness rerun) = my old exp 2, promoted to co-first (1a); step 2 (verify-step profile) = my exp 3, adopting their pointer to in-tree `spec_prof_probe.py`/`spec_step_probe.py`; step 3 (force `use_fused_multi_step_decode`) kept **conditional** on B (mostly rejected) — only if the profile shows metadata-rebuild dominance; step 4 (runtime UNIFORM_BATCH confirm) adopted as an add-on to exp 3; step 5 = C hygiene, folded into exp 1 protocol |

Net effect of the fold on my ranking: **H1 (record-binary provenance)
is strengthened** — ds4's B was the last code-side escape hatch and it
has no diffable carrier; H2 (harness/prompt) is narrowed to the
acceptance difference and *cannot* explain step time; H3 (current-
binary runtime pathology) unchanged, with ds4's speculator mechanism
absorbed as a *constant-cost* candidate inside the unexplained ~70 ms
(it may explain why mtp2 is ~3× plain generally, not why it doubled vs
the record).

## Cross-review adjudication — `docs/gfx906/fa-masked-mtp-regression-qwen.md` (rev 3)

| qwen claim | Verdict | Evidence |
|---|---|---|
| F1: current-build TP=1 mtp2 in-process = 36.75 t/s steady (first window 23.4 = Triton JIT warmup spikes of spec-path kernels) | **VALIDATED** | Artifacts read: `mtp_prof_mtp2.log` (27-step window, 218.6 ms/step nominal = the contaminated window they describe), `mtp_prof_mtp2b.log` (248 tok / 83 steps; 248/36.75 s ⇒ 81.6 ms/step steady ✓) |
| F2: step cost 120.5 (TP=2) vs plain 24.5; 3-token linear expectation 73.5+draft ≈ 78–84; excess ≈40–45 ms TP=2-only since TP=1 = 82 ≈ expected | **ARITHMETIC VALIDATED; "TP=1 ≈ expected" REJECTED** | My own A/B numbers reproduce 120.5/24.47 exactly. But the expectation is mis-built: record-era TP=1 mtp2 step = 45.8 ms = 1.16× its plain (see Reframing). TP=1 carries +36 ms excess too |
| F3: GPU-kernel-bound at TP=1 (total self-CUDA ≈ wall; no CPU gap); op-level under graph replay = ranking only, kernel leaves trustworthy | **VALIDATED with noted discrepancy** | mtp_prof_mtp2b: total self-CUDA 115.8 ms/step nominal vs 81.6 wall — includes the profiler window's warmup share + profiler overhead; their caveat is correctly stated |
| F4: kernel table (gptq 4-bit M=3 33.6; rocBLAS 16.8; LL 15.0; gemv 11.7; FA 1.6; GDN ~4.5; **gather_persistent 0.44, 16 calls/step** — N4 fine) | **VALIDATED** | Reproduced from `mtp_prof_mtp2b.log` line-by-line (33.613 / 10.004+6.779 / 7.915+7.141 / 6.804+4.943 / 1.585 / 0.443). The 16-calls-not-48 observation is important — draft passes do NOT re-gather |
| F5: graph coverage intact (FULL 1/1 largest=3 captured; UNIFORM_BATCH) | **VALIDATED** | Matches my capture-section verification on both builds |
| F6: merge-diff archaeology (all GDN/MTP changes gated INACTIVE for this model; drafter = 1-layer predictor, 2 draft forwards) | **VALIDATED (consistent with my stronger form: the merge contains NO decode-path change at all)** | GDN fallback line confirmed in the f7/mtp2 boot logs |
| Headline "pathology is TP=2 × MTP-specific; TP=1 fine" | **PARTLY VALIDATED — TP=2 half stands (~+22 ms extra); TP=1 half REJECTED** | Acceptance-confounded (3.00 vs 1.819); per-step both TP sizes regressed |
| S1 suspects (drafter EAGER at TP=2 / spec-path CPU × 2 workers / RCCL) | **PLAUSIBLE, UNRESOLVED — with the constant-across-builds caveat** | Verified: mtp2 serving logs show NO drafter graph capture (only the target runner's PIECEWISE 3/3 + FULL 1/1) — and the S8-era ctl log shows the SAME two sections, so drafter-eager existed in the record build too. S1 mechanisms are engine-code constants: they can explain the ABSOLUTE excess (why 4.9× plain now) but not the record→now delta unless via H1 |
| S2 (mtp2 P0−P1 delta 61.3 ms ≈ 3× plain's 20.2) | **VALIDATED as an open anomaly** | Arithmetic reproduces (181.8−120.5 vs 44.7−24.5). Standalone two-kernel gather at 131k ≈ 18 ms/step matches the PLAIN delta; the extra ~41 ms in spec context is unexplained |
| S3 (M=3 GEMM efficiency / combo_kernels tiles) | **OPEN — data pending** | Plain TP=1 profile died rc=134 (`mtp_prof_plain.log` ends in `c10::AcceleratorError`); probe 2 rerun outstanding. combo_kernels is NOT new (present in the S8-era config dump) |
| Probes 1–4 | **ADOPTED** | Probe 4 partially answered (no drafter capture, both builds); rest folded into the revised experiment list |

## Ranked hypotheses (rev 3 — restructured around the two components)

- **H-common (new, primary target): a current-build pathology in the
  mtp2 step common to TP=1 and TP=2 (~+36 ms/step vs the
  spec-decode-branch record).** The validated profiler table shows the
  step is GEMM/GEMV-bound (~77–95 ms of ~116 nominal). Candidates: M=3
  dispatch regression (S3 — though cfe09d8611's split build measured
  39.37 TP=1 with the same M=3 kernels), an extra forward per step (my
  anomaly 1: acceptance exactly 3.00 + ~70 ms ≈ one extra plain-ish
  forward), or drafter-eager cost. Chased fastest at TP=1 in-process.
- **H-tp2 (~+22 ms/step on top): TP=2-specific draft/verify overhead**
  (qwen S1: drafter eager × 2 workers, RCCL per-pass) — constant in
  engine code across builds; explains absolute slowness, not the
  record→now delta, unless via H1.
- **H1 (record-binary dirty state) — REPURPOSED, still load-bearing:**
  it can no longer explain away the whole anomaly (the current 3.3×
  plain multiplier at TP=1 is pathological on its own), but it remains
  the only route by which engine-code-constant mechanisms could differ
  between record and now. The clean-rebuild A/B is now a *confirmation*
  experiment, not the primary chase.
- **H2 (harness/prompt) — FURTHER WEAKENED:** qwen's fox-prompt TP=1 run
  hit acceptance 3.00 AND the full excess (81.6 ms/step) — the same
  regime as my serving runs — so prompt choice does not modulate step
  cost. Only the acceptance counters differ.
- **H3:** merged into H-common/H-tp2.
- **Rejected** (unchanged): capture-size collapse, upstream merge engine
  code, N4 gather kernel, MoE/VL gfx906 deltas, unmerged spec-decode
  branch, global platform degradation.

## Trace analysis (rev 4) — exp 4 DONE: the TP=2-extra component is ENGINE-CADENCE overhead, not GPU work

TP=2 mtp2 @131k P1, chrome traces via `--profiler-config` +
`/start_profile` (`/tmp/mtp2_trace/`; rank0 analyzed; ~71 decode steps,
1091-token prefill + 200-token gen; client 20.79 t/s = 144 ms/step,
engine windowed metrics agree):

- **Clock-domain caveat (important for reuse):** the kernel/GPU-event
  timestamps come from rocprofiler-sdk and are NOT wall-aligned — the
  kernel window reads 5.45s vs the client's 9.62s decode wall. GPU-domain
  spans/ratios are meaningful; wall comparisons must use cpu_op/
  cuda_runtime/python events and engine metrics.
- **In-step GPU is dense and healthy:** GPU busy 96% of the kernel
  window; annotated GPU step spans mean 41.4 ms; decode-only kernel
  families (per step, GPU-domain): gptq-M3 **17.6** (vs TP=1's 33.6 —
  halved weight traffic ✓), rccl 8.5 over **130 collective
  kernels/step**, elementwise/copy/triton 4.6 over **942 calls/step**
  (the eager draft passes' launch storm), FA 2.7, dense_gemv 2.7,
  GDN ~1.2, gather 0.35. The 64-call rocBLAS MT-monsters (11/8 ms each)
  are PREFILL chunks, not decode.
- **The missing ~95 ms/step is around the worker's execute, not in it:**
  CPU-domain (wall): execute_model spans median 35.6 ms; worker
  inter-step gaps 6.6 ms; but a dedicated worker thread spent **3813 ms
  in `hipEventSynchronize` (272 calls, ~14 ms avg)** ≈ **57 ms/step
  waiting for the next step's inputs** — the EngineCore-side scheduler
  cadence. Visible spec CPU on the worker: `propose_draft_token_ids`
  ≈6.6 ms/step (incl. ~1 ms/step of Triton `get_tensor_specialization`
  re-checking), `_calc_spec_decode_metadata` 1.1, rejection sampler 0.8.
  Remainder (~30-40 ms/step) is EngineCore scheduler-side spec
  bookkeeping in an untraced process.
- **Mechanism verdict:** the mtp2 GPU step at TP=2 costs ≈45-50 ms
  (consistent with the record-era 62.4 total minus its own overhead) —
  **the GPU side did not regress; the engine-level per-step overhead
  around it exploded to ~95 ms at TP=2 serving.** Plain decode's
  lighter scheduler work fits inside its 24.5 ms cadence, which is why
  plain TP=2 shows no penalty.
- **Fix candidate tested next (exp 5): `--async-scheduling`** — the
  AsyncScheduler exists precisely to overlap this scheduler-side work
  with GPU execution. Default is OFF (`async_scheduling: bool | None =
  None` → sync) in BOTH the record build and HEAD (checked
  `69f615b98:vllm/config/scheduler.py`) — so it is not the record→now
  delta, but it may fix the absolute pathology and beat the record.
  **RESULT (exp 5, run after the trace): NO EFFECT** — 24.93 t/s steady
  vs sync's 24.90 (`async_scheduling: True` confirmed in the server
  args; acceptance 3.00; windowed bursts to 43 but steady cadence
  unchanged). Consistent with the overhead being a per-step
  GPU→CPU→GPU **data-dependency chain** in the spec path (draft
  logits → CPU propose/reject bookkeeping → next launch), which
  scheduler overlap cannot hide. `stream_interval` (batched output
  processing) remains an untested cheap candidate for the
  EngineCore-side share.

## REV 5 (2026-08-22 post-reboot) — THE REGRESSION WAS HOST-STATE DEGRADATION. CENTRAL CLAIM RETRACTED.

A host reboot (12:39:47) changed nothing in the software stack but
**restored mtp2 TP=2 serving to 74.9 t/s (40.1 ms/step, acceptance
3.00, token-true usage-based client)** — 3× the degraded boot's 24.9.
Consequences for everything above:

1. **"mtp2 TP=2 regressed vs the S8 record" — RETRACTED.** The
   current build BEATS the record era: 40.1 ms/step vs the record's
   62.4 (which itself paid ~18 ms of N4 gather tax the record era
   couldn't avoid). The whole record→now gap was host state.
2. **The trace analysis (rev 4) described the DEGRADED host**, not the
   code: the 57 ms/step `hipEventSynchronize` stalls were the
   degradation signature. On the healthy host the same cadence costs
   ~0. Keep the mechanism knowledge (spec decode is the canary for
   sync-latency inflation), discard the "engine overhead exploded"
   conclusion. `--async-scheduling`/`--stream-interval` no-ops were
   no-ops because there was nothing code-side to fix.
3. **The dirty-binary hypothesis (H1) is now UNNECESSARY** — no code
   or binary difference was ever needed to explain anything. The
   clean-rebuild A/B of `69f615b98` (worktree was staged at
   `/local/git/vllm-s8-rebuild`) is CANCELLED as a diagnostic; run it
   only if a fresh-boot mtp2 number ever disagrees with 74.9 again.
4. **Client caveat that prolonged the confusion:** SSE-chunk counting
   under-reports tokens ~3× when acceptance ≈ 3 (chunks ≈ steps). The
   usage-based client (`stream_options.include_usage`) is mandatory
   for spec-decode t/s from now on. Yesterday's "chunk-rate" numbers
   (incl. mtp1 30.2 / mtp3 22.0) decode as healthy once multiplied by
   acceptance: mtp1 ≈ 61.9 t/s @2.00 (33 ms/step), mtp3 ≈ 88.1 t/s
   @4.00 (45.4 ms/step).
5. **Healthy-host post-reboot serving matrix (131k, TP=2, PERSIST=1):**
   plain 40.9 t/s · mtp1 61.9 · mtp2 74.9 · mtp3 88.1 — mtp2/mtp3 now
   clearly BEAT plain greedy (1.83×/2.15×). MTP on TP=2 is the right
   operating point again; S8 re-baselines should use these.
6. Host-degradation mechanism, kernel signatures, canary probe, and the
   recording protocol: `docs/gfx906/degradation.md` +
   `degradation_details.md`. The final full-wedge (14:06, GPU0 PSP
   −62) killed the 4-arm re-baseline mid-run; the P0/P1 tax arms need
   one clean re-run for the devlog record (P1 numbers above stand from
   the stream-interval A/B boots).

## Empirical results (GPU session, 2026-08-22 — after rev 3)

Experiments 1 + 2 (partial) run; numbers are wall-clock t/s with engine
acceptance from log stats (ms/step = 1000/(t/s ÷ acceptance)); in-process
TP=1, `max_num_seqs 4`, util 0.95, capture [1,2,3,4], greedy 512-token
windows after a 16-token warmup:

| cell | t/s | accept | ms/step | notes |
|---|---|---|---|---|
| agentic ×3 prompts, maxlen 2816 (record config) | 49.80 | 2.72 | **54.6** | record: 39.74 @ 1.819 = 45.8 → **TP=1-common excess ≈ +9-17 ms**, not +36 |
| agentic ×3, maxlen 32768 | 34.03 | 2.42 | 71.2 | acceptance moved (GDN greedy nondeterminism, documented); see S4 note below |
| fox ×1, maxlen 2816 | 45.99 | 2.88 | 62.6 | single-prompt ramp inflates |
| fox ×1, maxlen 32768 | 45.75 | 2.88 | **62.9** | **maxlen exonerated at TP=1 B=1** (62.6 vs 62.9) |
| profiler kernel table, fox @2816 vs @32768 (qwen's 32768 table) | — | — | — | **GPU work is maxlen-independent**: gptq M=3 34.2 vs 33.6, rocBLAS 16.6 vs 16.8, LL 14.3 vs 15.0, gemv 11.6 vs 11.7, FA 1.6 vs 1.6, gather 0.45 vs 0.44; totals 114.7 vs 115.8 nominal (incl. warmup) |
| plain M=1 kernel table (fox @32768; probe 2 rerun after rc=134) | — | — | — | gptq `m1mi<1>` 23.1 vs M=3 33.6 (1.45× = expected premium); **dense_gemv family +12 ms and rocBLAS +11 ms appear only under mtp2 = the drafter's fp16 forwards**; FA 0.44; GDN decode small. Plain TP=1 wall ≈ 40.4 ≈ record-era 39.5 — plain unchanged |

Conclusions drawn:
- **Revised decomposition:** TP=1-common excess is modest (~+10-17
  ms/step vs the record; possibly just M=3 GEMM premium + 2 draft
  passes — the M=1 plain table is still needed to say). The earlier
  +36 ms figure trusted qwen's fuzzy 81.6 ms fox number — REVISED.
- **The TP=2-extra component (~+45-58 ms/step) is now the dominant
  open question** — consistent with qwen S1 (drafter eager × 2 workers,
  RCCL per-pass; no drafter graphs in ANY build's capture log).
- **S2 (P0−P1 spec delta 61.3 vs plain 20.2 ms) — HYPOTHESIS:**
  plain-P0 pays 16 Sk_pad-wide two-kernel gathers (≈18 ms ≈ 20.2 ✓);
  spec-P0 additionally pays the draft passes' FA gathers through the
  same old path (~2× more → ≈54-61 ms total ≈ 61.3 ✓). At P1 the
  draft gathers are live-bounded (qwen counted 16 persistent-gather
  calls/step — the draft's attention evidently does not use the
  persistent kernel; its path is unidentified — trace will show).
- **B=3 observation (S4):** at `max_num_seqs 4`/capture≤4, a 3-seq
  verify step is 9 tokens > max FULL capture size → runs PIECEWISE or
  eager; the agentic cells (B=3) are therefore a different execution
  regime from the B=1 cells — do not cross-compare their ms/step with
  B=1 cells. (May explain part of 54.6 vs 71.2 beyond acceptance.)
- Run-to-run greedy divergence (GDN nondeterminism) moves acceptance
  between boots (2.72 vs 2.42 same prompts) — acceptance-normalized
  per-step is the only stable comparator.
- **TP=1-common component CLOSED (revised):** with clean same-harness
  numbers, mtp2/plain at TP=1 = 62.6/40.4 = 1.55× (record 1.16×); the
  +17 ms vs record is the drafter's two fp16 forwards + M=3 premium,
  all visible as ordinary kernels in the M=1-vs-M=3 table diff — no
  pathology. The record's 1.16× was itself extraordinarily good
  (days of branch-specific tuning; m4-templating etc.).
- **Platform blockage:** the mtp1 TP=2 arm (exp 3) could not boot —
  3 consecutive `hipErrorLaunchFailure` at weight load (the documented
  S4-residual TP=2 wedge; BACO reset needs root). Escalation note in
  the devlog. mtp1 remains unmeasured.

In flight at doc-update time: mtp1 TP=2 serving arm (draft-pass-count
scaling); next: TP=2 chrome trace via `start_profile`.

## Recommended experiments (rev 3 order — status in brackets)

1. **Record-config TP=1 rerun, in-process (~6 min)** [DONE — see
   Empirical results]: mtp2,
   `max_model_len 2816`, agentic-style 512-token prompts (the
   spec-decode-branch record config). Per-step stays ~82 ms ⇒ the
   TP=1-common excess is config/prompt-independent and real; drops
   toward ~46 ms ⇒ config-coupled.
2. **Plain TP=1 profile rerun (qwen probe 2; prior attempt rc=134):**
   M=1 kernel table to diff against the M=3 table (S3), and to check
   whether an extra forward's worth of kernels appears in mtp2
   (anomaly 1).
3. **mtp1 TP=2 serving arm (qwen probe 3):** 2 tokens/step, one draft
   pass — separates per-draft-pass overhead (S1) from verify-size
   effects for the TP=2-extra component.
4. **TP=2 mtp2 chrome trace via `start_profile` (qwen probe 1):**
   attribute the ~+22 ms TP=2-extra — draft-pass launch gaps, RCCL,
   CPU serialization between workers.
5. **mtp2 P0-side profile (S2):** why the two-kernel gather costs ~3×
   more in spec context than plain.
6. **Clean-rebuild A/B of `69f615b98` (confirmation, H1):** second
   line — only if 1–4 leave the record→now delta unexplained after the
   absolute pathology is understood.
7. Record-harness calibration (ocean-prompt serving) — folded into the
   3/4 server boots if convenient.

## Bookkeeping (for the next session, not done — read-only)

- `DEVLOG-masked-fa.md` post-commit 3's "post-upstream-merge build"
  attribution is **wrong as stated**: the v0.28.0rc1 merge changed no
  step-path code; the record binary is merely *pre-merge-dated*. The
  OPEN item should be reworded to "mtp2 records from the Aug-19 dirty
  binary are unreproducible on any clean build tested; root cause
  open" pending experiment 1.
- **Also correct the same devlog entry's parenthetical "S5's 39.7 was
  at max_seq_len 4096 per tp2-bench-final.log"** — wrong log (that run
  has `speculative_config=None`); the S5 mtp2 server log shows TP=2 @
  131072 (Retraction 1).
- The S5/S6/S7/S8 mtp2/mtp3 absolute records should carry an asterisk
  until experiment 1 lands: they derive from one never-rebuilt dirty
  binary whose raw client outputs no longer exist. Do NOT extend that
  asterisk to the TP=1 @2816 spec-decode-branch records (39.74/39.37)
  — different config, different session, reproduced post-reboot per
  DEVLOG-spec-decode.md.
