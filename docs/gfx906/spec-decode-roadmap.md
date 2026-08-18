# gfx906 Speculative Decoding Roadmap

> Status: **Phase 0 pending, revised post-review.** n-gram probe
> complete 2026-08-18 (0.68× — see below); the two prerequisite FA
> fixes are in-tree on this branch. Baseline refs: dense 27B-AWQ
> **27.46 t/s** (server, 4 seqs, graph), MoE 35B-A3B-AWQ **66.56 t/s**
> (same session). Adversarial review (two independent passes,
> merged) found the 1.19× ceiling in §1 is an optimistic upper bound
> with an unreconciled unit-accounting discrepancy, Phase 1 is scoped
> as a kernel *extension* when it's closer to a new kernel against
> vendored third-party code with an unaddressed state-rollback
> requirement, Phase 2's "preferred" option is architecturally bigger
> than framed, and the suffix arm in Phase 0 is not the config-only
> probe it's described as. All incorporated below; see
> `spec-decode-roadmap-plan-rev_claude.md` for the full review.

## 1. Where we are

n-gram spec decode (k=3, min2/max5) on dense Qwen3.5-27B-AWQ, 3
multi-turn agentic-coding prompts (tool-call JSON + code context),
512 tok, greedy:

| arm | t/s | notes |
|---|---|---|
| baseline | **27.46** | server path (LLM-harness record 25.25) |
| ngram k=3, pre-fix | 22.99 | artifact: engine demoted to all-PIECEWISE |
| ngram k=3, post-fix | **18.73** | 0.68×; drafts on ~40% of steps, ~36% accepted |

**Fixed in-tree** (commit `4e40e3e`): `Gfx906FABackend` now declares
`UNIFORM_BATCH` CG support when spec tokens are configured, and
`forward_paged` has capture-safe uniform fast paths (the per-seq
`int(cu_seqlens_q[...])` D2H syncs are illegal under capture). Spec
steps run on FULL graphs. **Without these, every spec method on this
backend runs all decode steps piecewise (~100 ms/step).**

**Model facts that shape the plan** (Qwen3.5-27B text_config):

- 64 layers = **48 GDN linear_attention + 16 full_attention**
  (`full_attention_interval=4`). GDN dominates the spec-step overhead.
- **No MTP head** (no `num_nextn_predict_layers`) — `method="mtp"`,
  EAGLE, draft models are unavailable for this checkpoint. The
  no-draft-model family (ngram / ngram_gpu / suffix) is all we can
  serve on the 27B today; the 35B MoE is the same family, check
  before assuming.

**The three remaining cost blocks** (torch.profiler A/B + controlled
probes; full numbers in DEVLOG-moe-opt.md "n-gram spec decode probe"):

| block | cost | affected steps | shared by |
|---|---|---|---|
| B1: GDN spec path uses **chunk (prefill) kernels** — `ChunkGatedDeltaRuleFunction` 415 µs CUDA-total per layer per multi-token step (child kernels inside), vs ~20 µs/layer for the packed-decode M=1 fast path | **≈ 20 ms/draft step** (415 µs × 48 layers) | every draft step | **all** spec methods |
| B2: **no-draft 1-token steps** don't qualify for the uniform-4q FULL graph → PIECEWISE | ~53–64 ms vs 36.5 ms | the ~60% of ngram/suffix steps without a draft | ngram, suffix (MTP/EAGLE always draft → immune) |
| B3: spec bookkeeping (`copy_` Δ134 ms + `index_*` 181 ms per 128-tok run → ~10 ms/draft step; `precopy_mamba_align_fused_kernel` inside) + CPU ngram proposer D2H serialization | ~10 ms/draft step (+~5 ms proposer) | every draft step | all methods (B3b is ngram-CPU only) |

**B1 reconciliation (2026-08-18, from
`/tmp/spec_prof_{nospec,spec}.log`; full derivation in
`DEVLOG-spec-decode.md`)**: the review flagged that "~800 µs/layer" vs
"20–30 ms/step" vs the 38.4 ms naive product didn't reconcile. They
did not — **"800 µs" double-counted** the `ChunkGatedDeltaRuleFunction`
CUDA total (415.2 µs/call, which already includes the 605 ms of child
kernels within its 637.7 ms) *plus* the child kernels separately.
Correct: 415.2 µs/layer × 48 = **19.9 ms per draft step** (plus
~0.7 ms/step `fused_sigmoid_gating`). Step counts from the same
run: 3312 packed_decode / 48 = 69 one-token steps + 1536 chunk /
48 = 32 multi-token steps = 101 steps; 128 = 69 + 32×(1+a) →
**a = 0.84, r = 0.32** for that prompt set (server 3-prompt probe:
r ≈ 0.40, a ≈ 1.08 — prompt-dependent). B3 deltas:
`copy_` 158.6−24.8 = 133.8 ms + `index_select` 87.5 + `index_put_`
93.5 = 181 ms, over 32 draft steps → **~10 ms/draft step**.

**Per-draft-step budget (reconciled)**:
```
measured draft step (server, k=3)  ≈ 82 ms
  = base 4-token compute (GEMM M=4 + FA q4 + proposer sync) ≈ 47
  + B1 GDN chunk                                            20
  + B3 copy/index                                           10
  + slack ~5
```
Post-P1 (chunk → fused M=4 ≈ 48 × 4 × 20 µs ≈ 4 ms, **no credit
taken for B3**):
**draft step ≈ 82 − 20 + 4 ≈ 66 ms**.

| scenario | T_draft | T_nodraft | r, a | t/s | vs 1.15× gate |
|---|---|---|---|---|---|
| post-P1 only (no-draft still piecewise 64 ms) | 66 | 64 | .40/1.1 | 22.2 | 0.81× |
| post-P1+P2 (no-draft → 36.5 ms FULL q1) | 66 | 36.5 | .40/1.1 | 29.8 | **1.09×** |
| post-P1+P2 + B3 −5 ms (kernel owns align-slot writes) | 61 | 36.5 | .40/1.1 | 31.3 | 1.14× |
| post-P1+P2+B3 with a = 1.3 (better drafter) | 61 | 36.5 | .40/1.3 | 33.4 | **1.22×** |

**Implications.** At today's draft quality (a ≈ 1.1) the P1+P2 stack
lands at ~1.09–1.14× — at or below the gate. **Draft quality `a` is
the swing variable**, not step cost alone: the P1+P2+B3 configuration
passes at a ≈ 1.3. Phase 0's suffix acceptance probe is therefore a
go/no-go input, not just data. **Stop rule (kept from review):** if
the Phase-1-measured draft-step cost exceeds ~60 ms, stop after
Phase 1 and record the ceiling. The 66 ms prediction sits just over
the rule — the only in-P1 lever back under it is the B3 overlap
below, which is testable from the Phase-1 profiler run.

**B3 scope — folded into Phase 1 with reasoning, per the review's
either/or.** The ~10 ms/draft B3 is dominated by `index_put_`/
`index_select` + `precopy_mamba_align_fused_kernel` — the
`mamba_cache_mode=align` block-slot mechanism that copies per-request
state between slots after rejection. If the Phase-1 fused M-token
kernel **writes per-token state directly into those align slots**
(which the kernel must do anyway for rollback — §3), the precopy
kernel and part of the index path disappear *as a consequence of the
kernel's output contract*, not as separate work. The `copy_` half
(KV×4 writes, input staging) is largely irreducible and stays in the
model. Phase 1's exit criterion therefore includes a profiler check:
align-slot precopy/index_put time must drop measurably; if it does
not, re-scope B3 as its own work item before Phase 2.

**Re-derive this ceiling after Phase 0** with the measured r/a per
arm. Note the min_n=1 arm (item 3) trades `a` for `r`: with the
reconciled costs, higher r at lower a is neutral-to-negative
(r=0.6, a=0.5 → ~25 t/s), so the arm's purpose is to measure the
a(r) trade, not to bank a win.

## 2. Phase 0 — cheap data, no kernel work (½ day)

Config-only A/Bs on the existing 3-prompt bench
(`benchmarks/kernels/gfx906/spec_ngram_dense.py`):

1. `method="ngram_gpu"` — GPU vectorized proposer, removes the CPU
   D2H serialization (B3b). Config change, **but not drop-in
   equivalent to `ngram`**: it's a reimplementation
   (unfold/argmax match selection), and tie-breaking on repeated
   n-grams need not match the CPU proposer's. Add an explicit
   draft-token-equality + acceptance-rate comparison between the two
   arms as an acceptance criterion — if drafts differ, any k/min_n
   tuning found on one proposer does not transfer to the other, and
   "removes B3b only" is the wrong mental model.
2. `method="suffix"` — in-tree `SuffixDecodingProposer` (suffix tree
   over prompt + cached responses); likely better drafts on code.
   Knobs: `suffix_decoding_max_tree_depth` (24), `..._max_spec_factor`.
   **Not a free config swap — three structural issues:**
   - Requires `arctic-inference==0.1.1` (`_validate_suffix_decoding`
     raises `ImportError` otherwise); unverified on ROCm, so confirm
     the wheel installs before budgeting this as "an afternoon."
   - Suffix speculates a **dynamic number of tokens per request per
     step** (by design — see `suffix_decoding.py`). Non-uniform steps
     fail `_is_uniform_decode`, so **every suffix step runs
     PIECEWISE regardless of the FA fix**, and neither Phase 1's
     uniform-`1 < M ≤ 8` kernel dispatch nor Phase 2's q1/q(1+k)
     routing does anything for it. If suffix wins this A/B, it is
     *not* a candidate for the Phase 1/2 gains as scoped — either pin
     `num_speculative_tokens` to a small fixed value for this probe
     (accept PIECEWISE, treat the run as a draft-quality-only
     measurement) or add non-uniform small-M dispatch as a named,
     budgeted work item before treating suffix as a serving
     candidate.
   - Left unset, `num_speculative_tokens` defaults to
     `suffix_decoding_max_tree_depth` = 24, i.e.
     `uniform_decode_query_len = 25` — outside Phase 1's `M ≤ 8`
     kernel range and well past the Phase 2 §4 memory budget (sized
     for k=3/4). Set `num_speculative_tokens` explicitly for this
     probe; do not run with suffix defaults.
3. ngram `prompt_lookup_min=1` — raises the draft rate r (fewer B2
   steps); watch acceptance drop. **If r moves here, re-derive the
   §1 ceiling math with the new r before scoping Phase 1** — the
   1.19× case is only valid at r=0.40.
4. k sweep {1,2,3,4} on the winner of 1–3.

**Gate:** any combo whose **mean exceeds the baseline's 95% CI lower
bound** (not a flat 27.5 t/s) → ship the config, stop. The 27.46
baseline itself spans 26.81–27.96 t/s across 3 runs (~4% noise) — a
flat 27.5 t/s threshold sits inside that band and would ship on
run-to-run noise, not signal. Add a repeat-count knob to
`spec_ngram_dense.py` (currently single-run per arm on 3 prompts) so
the gate has enough samples to compute a real CI. (Expected: none
pass — B1/B2 are structural — but it's still cheap, roughly an
afternoon plus the repeat runs.)

## 3. Phase 1 — GDN small-M spec kernel (B1, the shared lever)

The GDN state recurrence `h_t = A_t·h_{t-1} + B_t·x_t` is inherently
sequential in t, so "M>1 decode" is **not** a parallelism problem:
the local fused GDN decode kernel (M=1, ~20 µs/layer) extended with an
inner token loop is the right shape, and it should stay within a
factor of M of the M=1 cost.

**Scope correction: this is a new kernel, not a parameter extension.**
`fused_recurrent_gated_delta_rule_packed_decode` lives in
`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py` —
vendored upstream FLA code, not gfx906-owned (gfx906's only footprint
in this path is the zero-fill-skip micro-opt in
`qwen_gdn_linear_attn.py`). It is M=1 by hardcoded shape and grid, not
by an easily-relaxed loop bound: `mixed_qkv` is asserted 2D
`[B, qkv_dim]`, `out` is asserted shape `(B, 1, HV, V)` with the `1`
literal, and the Triton grid indexes only `(i_v, i_hv, i_n)` —
sequence index, no token-within-sequence axis. Making this M-capable
means adding a token axis to the grid or an in-kernel loop with
correct sequential ordering (the gate/mask math must respect
intra-sequence order across the M axis, unlike the trivially-parallel
batch axis), re-deriving strides for a 3D `mixed_qkv`/`out`, and doing
this either as a gfx906-only fork of vendored code or as an
upstream-shaped contribution to `fused_recurrent.py`. Budget this as a
new kernel authored from the M=1 op as a spec (2–3× the effort implied
by "extend with an inner token loop"), and decide up front whether it
lands gfx906-local (fork, ongoing merge-conflict cost against
upstream FLA changes) or upstream (review cycle, but no drift). The
"should stay within a factor of M" cost claim also isn't guaranteed —
a sequential in-kernel loop changes the memory-traffic/compute-per-
launch ratio and could hit different occupancy/register-pressure
limits than M× replication implies; measure, don't assume.

**GDN state rollback on rejection needs a design note before the
kernel is written.** Spec decode needs GDN state *at the last accepted
token*, not at the end of all M drafted tokens — when some drafts are
rejected, state must reflect the accepted prefix, not the full draft.
Upstream already has a mechanism for this: `mamba_cache_mode="align"`
allocates `(1 + num_speculative_blocks)` state slots per sequence
(`vllm/v1/worker/mamba_utils.py`), and a fused postprocess kernel
(`postprocess_mamba_align_gpu` / `run_fused_postprocess`) copies
per-request state between slots after the rejection sampler resolves
`num_accepted_tokens`. The chunk-kernel path presumably populates
these slots at chunk boundaries today; **the new fused M-token kernel
must write per-token state into these same block-indexed slots as it
goes**, not just return a single final state. This is scoped
integration work against an existing mechanism, not a new
state-management design — but it changes the kernel's output
contract (write M intermediate states, not one), and needs to be
speced before implementation starts, not discovered during it.

- **Kernel**: extend the gfx906 fused GDN decode kernel to
  `1 < M ≤ 8` (uniform per-seq token counts; spec steps are uniform
  by construction). Keep the local micro-opts' invariants: non-spec
  keeps the zero-fill skip; the spec path keeps the fill (upstream
  invariant — pad rows feed causal-masked compute). Per-token state
  writes go to the `mamba_cache_mode=align` slot scheme above, not
  just a final-state output.
- **Dispatch** (Python, `qwen_gdn_linear_attn.py`): `on_gfx906()` and
  uniform `1 < num_tokens ≤ 8` → small-M kernel; else upstream chunk
  path. Env kill-switch `VLLM_GFX906_GDN_SPEC_SMALL_M` (default on),
  matching project convention.
- **Numerics gate**: the M=4 result must match **per-token-boundary
  state**, not just the final state after M tokens — the rollback
  mechanism above needs per-slot correctness, not only end-of-step
  correctness. Note "bit-equal to four sequential M=1 calls" is a
  stronger and possibly *wrong* target as stated: the kernel
  accumulates internally in fp32 and only casts to the output/state
  dtype on store, so a fused M-token loop that keeps state resident
  across iterations skips the M−1 extra quantization round-trips that
  four *standalone* M=1 calls would each incur (each reloading `h0`
  from its stored dtype). Define the actual bar as "matches fp32
  reference math to \<tolerance\>, at every token boundary, in both
  output and state" and treat comparison against four sequential calls
  as an informational round-trip-drift characterization, not the pass
  condition. Unit test both; end-to-end via greedy probe + PPL A/B.
- **Expected**: ~20 ms/draft step back (B1) on every draft step, all
  models, all spec methods, plus the B3 overlap if the align-slot
  writes land (−~3–5 ms more — §1). Largest single lever; do this
  before spending anything on B2. **Exit criterion**: measure the
  actual post-fix draft-step cost and re-run the §1
  ceiling/stop-rule check before starting Phase 2 — if draft-step
  cost exceeds ~60 ms, stop here. The profiler run must also show
  the align-slot precopy/`index_put` time dropped (B3-overlap check);
  if it didn't, B3 re-scopes as its own work item.

## 4. Phase 2 — q1 FULL graphs for no-draft steps (B2)

1-token spec steps today dispatch to piecewise because
`_is_uniform_decode` requires `max_num_scheduled_tokens == 1 + k`.

**Scope correction: "Preferred" is a bigger change than the framing
implies, on two independent axes.**

1. `uniform_decode_query_len` (`gpu_model_runner.py`) is a **single
   scalar instance attribute**, set once at init as `1 +
   num_spec_tokens`, and `_is_uniform_decode` is a scalar equality
   check against it — not membership in a set of valid uniform
   lengths. There is exactly one "uniform" query length the runner
   knows about at a time; with spec configured, that's `1+k`, not
   `1`. Supporting both families means `uniform_decode_query_len`
   becomes a set, which fans out through `_is_uniform_decode`, the
   `CudagraphDispatcher`'s capture-size derivation
   (`initialize_cudagraph_keys` currently takes one
   `uniform_decode_query_len` and would need to run twice and merge
   keys without collision), and the capture-size "multiple of" logic
   in `vllm/config/compilation.py` (written for one stride, not two
   interleaved strides). This is core-vLLM surgery across at least
   three files, and any backend with spec decode enabled goes through
   the same dispatch path — real risk of affecting non-gfx906
   backends, not a gfx906-local change.
2. Even solving (1), **the resulting q1-with-spec-metadata graph is
   not the same captured region as the existing no-spec q1 decode
   graph.** Rejection-sampler output plumbing, draft-state
   save/restore hooks, and (per Phase 1 above) GDN align-slot
   bookkeeping are all in or around the captured region when spec is
   configured. "Treat a uniform 1-token spec step as a q1-uniform
   decode" doesn't reuse the no-spec q1 graph shape — it's a new,
   spec-aware q1 graph that happens to share a token count with the
   no-spec one. The FA backend's uniform fast-path host-identity
   check (`num_tokens == num_seqs * max_seqlen_q`) and sampler output
   shapes need a spec-aware q1 variant too.

Given both, do a short spike confirming the dual-uniform-family
plumbing is viable and roughly how invasive it is (and whether it's
acceptable as an upstream-shaped change or must be gfx906-forked)
before calling this "Preferred" over the PAD alternative — the PAD
approach touches one file (scheduler accounting in
`gpu_model_runner.py`) versus this option's multi-file, cross-backend
footprint, so the two are not the comparable-weight choice the
current wording suggests.

- **Preferred** (pending the spike above): capture **both** uniform
  graph families when spec is configured — q1-uniform
  [1..max_num_seqs] and q(1+k)-uniform [1+k .. max_num_seqs(1+k)] —
  and treat a uniform 1-token spec step as a q1-uniform decode (extend
  `_is_uniform_decode` + the capture size derivation; the dispatcher's
  `BatchDescriptor` keying already separates uniform vs not). Core
  vLLM change, inert when no spec config is set — **but "inert" needs
  its own regression test**: add a dispatch-key assertion or explicit
  test that no-spec q1 decode still dispatches to the unmodified
  no-spec graph. A silent misroute here wouldn't necessarily show up
  as an obvious throughput regression on the Phase 3 gate (§5), since
  correctness could be preserved with only a latency difference.
- **Alternative** (heavier per this doc's original framing, park):
  schedule PAD tokens to pad no-draft steps to 1+k
  (scheduler/`num_scheduled_tokens` accounting surgery, line ~2327 of
  `gpu_model_runner.py`). Re-evaluate against "Preferred" once the
  spike above sizes the real cost of the dual-family approach — this
  alternative may turn out to be the lighter option.
- **Budget**: +~0.3–0.5 GiB graph memory for the q1 set; verify
  against the 4-seq dense budget (KV 7.4 GiB at 0.95) and the MoE
  budget before capture. This covers memory only, not the engineering
  cost above.
- **Expected**: no-draft 53–64 ms → ~37 ms; on ngram/suffix this
  touches the *majority* of steps (not suffix at default settings —
  see §2 item 2; suffix's dynamic draft length never produces a
  uniform 1-token step in the first place).

## 5. Phase 3 — tune + adoption gate

- Re-run the agentic bench per arm: method × k × min_n (ngram) /
  tree_depth (suffix); add one **low-repetition control prompt**
  (e.g. novel-identifier listing) to measure the floor, not just the
  ceiling.
- **Adoption gate**: ≥1.15× vs 27.46 on the agentic set, PPL-neutral
  (spec-vs-nospec is S3-class fp drift by construction — gate on PPL
  + coherence, not token identity), no-spec baselines within the
  observed noise band of 27.46 / 66.56 (**not** literal bit-for-bit
  t/s parity — both baselines carry ~4% run-to-run spread at n=3
  prompts single-run per arm, confirmed across two independent
  reviews; a "zero regression" reading against a point estimate would
  fail on any run). State the criterion as mean + CI, matching the
  Phase 0 gate fix in §2.
- On pass: recommended `--speculative-config` in
  `docs/gfx906/running.md`. On fail: record the measured ceiling and
  park — the FA fixes stay (any future drafter needs them).

## 6. Other spec-decode approaches — applicability on this machine

The cost blocks are **method-agnostic rails**; the rails fix once and
every drafter rides them:

| method | available on 27B? | expected | notes |
|---|---|---|---|
| ngram (CPU) | yes | 0.68× today → ~1.19× post P1+P2 (upper bound — see §1 sensitivity table) | prompt-repetition-bound; r≈0.4 on agentic code |
| ngram_gpu | yes (test) | removes B3b, **draft equivalence to ngram unverified** (§2 item 1) | same match logic as ngram, async H2D + GPU unfold/argmax; tie-breaking may differ |
| suffix | yes (test), **PIECEWISE-only until non-uniform dispatch exists** (§2 item 2) | ≥ ngram drafts expected on the drafts it produces; does not benefit from Phase 1/2 as scoped | suffix tree, in-tree proposer; dynamic per-request draft length, not covered by the uniform-M rails; requires `arctic-inference==0.1.1` |
| MTP | **no head in checkpoint** | directionally better than ngram (r=1.0 removes B2 entirely) — **exact multiplier unverified, no derivation available without a checkpoint to test; do not treat "1.1–1.3×" as measured** | draft = one extra layer fwd (~1–2 ms) — the user's original intuition (cheap draft on a weight-bound GPU) applies maximally; r=1.0 so B2 is irrelevant, only B1 matters. Re-evaluate if an MTP-capable Qwen3.5 checkpoint lands |
| EAGLE | no head on disk | n/a | would need a trained head; same rails |
| draft model | none on disk | n/a | a 1B draft reads ~2 GB/step at MI50 bandwidth — the ~2 ms figure assumes close to theoretical peak effective bandwidth and is optimistic; only worth it at high acceptance; same rails |

MoE 35B: 40 layers, `full_attention_interval=4` → 30 GDN + 10 full
attention (mirrors the dense 27B's 64→48/16 split). Baseline step
~15 ms means B1 (48→30 GDN layers) is a *larger* fraction of the spec
step there; Phase 1 is a hard prerequisite for any spec method on the
MoE, and the absolute payoff is bigger (+~10–15 t/s at 1.2×) — but
that 1.2x carries the same upper-bound caveat as the dense-27B ceiling
in §1, not yet a measured number for this model.

## 7. Artifacts

- FA fixes: `vllm/gfx906_fa/gfx906_fa_backend.py` (UNIFORM_BATCH when
  spec configured), `vllm/gfx906_fa/gfx906_fa_paged.py`
  (capture-safe uniform scatter/gather). FA suite 15/15.
- Bench/probe scripts (moved from /tmp): `benchmarks/kernels/gfx906/
  spec_ngram_dense.py` (serving A/B + acceptance),
  `spec_step_probe.py` (step-cost probes), `spec_prof_probe.py`
  (torch.profiler A/B for B1/B3 attribution). DEVLOG references
  `/tmp/bench/spec_step_probe2.py`; confirm this is a clean rename to
  the in-tree `spec_step_probe.py` and not a stale/divergent
  reference before relying on it.
- Numbers: DEVLOG-moe-opt.md "n-gram spec decode probe".
- Review: `spec-decode-roadmap-plan-rev_claude.md` (adversarial
  review, merges an independent GLM pass) — verified findings above
  are folded into this doc; see that file for the full derivation of
  each correction.
