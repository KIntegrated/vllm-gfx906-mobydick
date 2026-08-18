# Adversarial Review: spec-decode-roadmap.md (merged)

Merges my initial review with `spec-decode-roadmap-plan-rev-glm.md`.
Every GLM claim below was independently re-checked against the current
tree (`b011bfd515`) before being accepted, tempered, or rejected — see
each item's "Verification" line. Findings are combined and re-ranked
by how much they change the roadmap's cost/benefit case, duplicates
merged into one writeup, my original numbering dropped in favor of
topic order.

## Accepted from GLM, verified independently

### A1. Suffix arm is not "config-only" (GLM F2) — ACCEPT, all three sub-claims verified

- **External dependency**: `vllm/config/speculative.py:1149-1153`,
  `_validate_suffix_decoding()` raises `ImportError` unless
  `arctic-inference==0.1.1` is installed. Confirmed in-tree. On ROCm
  this wheel's availability/build story is unknown — Phase 0's "it's
  an afternoon" estimate implicitly assumes this installs cleanly,
  which is untested.
- **Dynamic (non-uniform) draft lengths**: `suffix_decoding.py:47`,
  docstring literally says "will speculate a dynamic number of tokens
  for each request every decoding step." Confirmed non-uniform steps
  fail `_is_uniform_decode` (scalar equality check, see my original
  finding on that function) → every suffix step is PIECEWISE,
  independent of the FA fix. Phase 1's kernel dispatch as specified
  (`uniform 1 < num_tokens ≤ 8`) and Phase 2's q1/q(1+k) routing both
  assume uniform batches; neither lever fires for suffix's ragged
  steps. **This means if Phase 0 picks suffix as the winning method,
  Phases 1 and 2 as scoped do not apply to it at all** — a real gap
  in the roadmap's method-agnostic-rails framing (§6's table
  currently implies suffix rides the same two levers as ngram).
- **Default k=24**: confirmed at `speculative.py:1158`,
  `self.num_speculative_tokens = self.suffix_decoding_max_tree_depth`
  and the field default is 24 (line 195). So
  `uniform_decode_query_len = 25` under suffix defaults — outside
  Phase 1's `M ≤ 8` kernel range, and blows well past the §4 memory
  budget (computed for k=3/4, not 25).

**Verdict: accept in full.** This is a real hole — the roadmap's Phase
0 item 2 should either pin `num_speculative_tokens` explicitly to a
small value and note the resulting configuration accepts PIECEWISE
(making it a pure draft-quality probe, not a candidate for Phase 1/2
gains), or the roadmap needs a named third work item for non-uniform
small-M dispatch before suffix can be a real contender.

### A2. GDN state rollback on rejection is unaddressed in §3 (GLM F3) — ACCEPT, with a correction to GLM's severity framing

**Verification:** grepped `qwen_gdn_linear_attn.py` and
`vllm/v1/worker/mamba_utils.py` for how state rollback under spec
rejection currently works. Found a real, already-existing mechanism:
`mamba_cache_mode == "align"`, `num_speculative_blocks`
(`mamba_utils.py:1248`), and a fused postprocess kernel
(`postprocess_mamba_align_gpu`, `run_fused_postprocess`) that copies
per-request accepted-token state between block slots after the
rejection sampler resolves `num_accepted_tokens`. State is allocated
at `(1 + num_speculative_blocks)` granularity per sequence
(`mamba_utils.py:1243,1284,1296`) — i.e., upstream already checkpoints
at sub-final-token granularity, indexed by block slot, independent of
which kernel populates a given slot.

This **partially contradicts** GLM's framing that "the proposed fused
inner-loop kernel, as specified, produces only the final state" is
the core problem and that "the chunk kernel family... gets rollback
states for free" implies the fused kernel is uniquely disadvantaged.
The real requirement is narrower and more concrete than GLM states:
the M-token fused kernel needs to **write intermediate per-token
states into the existing block-indexed align-slots** (one per drafted
token, up to `num_speculative_blocks`), not invent a new snapshot
scheme. That's still real, non-trivial work — the current chunk-kernel
path presumably already does this (chunk boundaries constitute natural
checkpoints), and a hand-written fused-recurrent-loop kernel would need
to add explicit per-iteration writes to those same slots, which is
exactly the "per-token state snapshots" cost GLM flags (state pool
~72 MB/seq if done naively) — but the mechanism to plug into already
exists, so this is scoped integration work, not an open research
question the roadmap has to solve from scratch.

**Verdict: accept, but soften.** Recommend the roadmap add a
paragraph to §3 stating: "the fused M-token kernel must write
per-token GDN state into the existing `mamba_cache_mode=align`
block-slot scheme (`num_speculative_blocks` allocation,
`postprocess_mamba_align_gpu`) rather than only the final state — this
is additional kernel-side plumbing, not a new state-management design,
but it does mean the numerics gate target should be 'state at each
token boundary matches slot-by-slot', not just 'final state bit-equal
after M tokens.'" This also directly sharpens my own finding #5 (bit-
equality ambiguity) — the gate needs to cover intermediate states, a
requirement I flagged the ambiguity of but didn't identify the
concrete mechanism for.

### A3. Ceiling math is best-case-stacked and the B1 arithmetic doesn't reconcile (GLM F1) — ACCEPT

**Verification:** re-derived GLM's table from the roadmap's own T1=36.5
ms, r=0.40, a=1.1 inputs — arithmetic checks out (55 ms → 1.19×, 62 ms
→ 1.13×, 70 ms → 1.08×), confirming the ceiling is not robust to a
~15-25% miss on the post-P1 draft-step estimate.

Also independently checked the B1 unit-accounting claim: roadmap §1
table states "~800 µs/layer" and "~20-30 ms/step (48 layers)" in the
same row. `docs/gfx906/DEVLOG-moe-opt.md:2539` gives "~800 µs/layer/step
(~18-25 ms/step)" for the same block. 800 µs × 48 layers = 38.4 ms,
which matches **neither** the roadmap's 20-30 ms nor the DEVLOG's
18-25 ms — both are roughly half the naive per-layer × layer-count
product. Confirmed this is a real unreconciled discrepancy in the
source numbers, not a misreading on GLM's part (I checked both docs
directly). Possible explanations the roadmap doesn't state: only a
subset of the 48 GDN layers are actually on the hot path per step,
the 800µs figure is amortized across something other than "per step,"
or one of the two docs has a stale number from an earlier profiling
pass.

**Verdict: accept in full.** This also compounds my own original
finding #4 (B3 silently folded into the ceiling math without its own
phase) — between GLM's stacking critique and my B3-omission finding,
the 1.19× figure has at least two independent reasons to be treated as
an upper bound, not an expected outcome. Recommend both: (1) add the
sensitivity table GLM proposes with an explicit stop-rule ("if
post-P1 measured draft-step cost > 60 ms, stop and record — do not
proceed to Phase 2"), and (2) resolve the 800µs/layer vs 20-30ms/step
vs 18-25ms/step discrepancy before it's used to greenlight kernel work
— trace it back to the profiler run that produced it.

### A4. Phase 0 gate (≥27.5 t/s) is inside measurement noise (GLM F4) — ACCEPT, duplicate of my finding #3, merged

GLM independently reached the same conclusion I did (baseline spread
26.81-27.96, ~4%, gate at 27.5 sits inside that band). No new
verification needed beyond what I already did. Merging into one
recommendation: widen the Phase 0 gate to clear the observed noise
band (e.g. ≥28.5-29 t/s) or require repeat runs with a CI-based
criterion, **and** add a repeat-count knob to `spec_ngram_dense.py`
(mine) — GLM's phrasing ("mean + CI lower bound ≥ baseline mean") is
a cleaner statement of the statistical criterion than my original
"outside the noise band" framing; adopting GLM's wording.

### A5. ngram_gpu draft-equivalence is asserted, not established (GLM F5) — ACCEPT

**Verification:** did not independently re-derive tie-breaking
behavior in `ngram_proposer_gpu.py`, but the claim is low-cost to
check and the risk GLM identifies is real and specific: an
unfold/argmax reimplementation is not guaranteed to match the CPU
proposer's match-selection order on ties, and the roadmap's Phase 0
item 1 ("removes B3b... pure config change") implicitly assumes
identical drafts, which would make any k/min_n tuning done on one
proposer transfer to the other. This is cheap to falsify (compare
draft-token equality + acceptance stats between arms, which Phase 0's
bench already runs both arms through). **Verdict: accept** — add an
explicit draft-equality check as a Phase 0 acceptance criterion, not
just a throughput comparison.

### A6. Phase 2 "preferred" is harder than the one-line framing suggests (GLM F6) — ACCEPT, merges with and extends my finding #2

GLM's version and my original finding #2 attack the same claim from
different angles: I focused on `uniform_decode_query_len` being a
scalar (architectural: the dispatcher/config-derivation layer doesn't
support two simultaneous uniform families). GLM focuses on a different
and additive problem: even for a *single* q1-with-spec-metadata
graph, the captured region differs from the plain no-spec q1 decode
graph (rejection-sampler output plumbing, draft-state save/restore
hooks, and — per A2 above — GDN align-slot bookkeeping are all in or
around the captured region), so "treat a uniform 1-token spec step as
a q1-uniform decode" isn't reusing an existing graph shape, it's a new
graph shape that happens to share the token count 1 with the no-spec
graph. Both problems are real and compound: even after solving the
scalar-vs-set `uniform_decode_query_len` issue I raised, the resulting
q1-spec graph still needs its own capture path distinct from
no-spec q1, which is GLM's point.

GLM's regression-test ask is also a good addition I didn't make: "a
silent dispatch change [to the no-spec q1 path] is exactly the
regression the Phase 3 gate forbids" — the Phase 3 adoption gate (§5)
checks throughput regression on the no-spec baselines, but a cudagraph
key derivation bug could silently route no-spec q1 decodes to the
wrong (spec-shaped) graph without changing measured throughput in an
obviously-detectable way if the graphs happen to produce correct-but-
different-latency output. **Verdict: accept**, merge into my finding
#2's recommendation: add "no-spec q1 decode dispatches to the
unmodified no-spec graph, verified by a dispatch-key assertion or
explicit test, not just an aggregate throughput check" as an explicit
Phase 2 exit criterion.

### A7. Minor nits (GLM F7) — MOSTLY ACCEPT

- **1B draft-model bandwidth estimate (§6) is optimistic** (near
  theoretical peak MI50 bandwidth) — accept, low stakes since this row
  is already "none on disk, n/a" in the roadmap's own table; doesn't
  change any decision. Worth a one-word "optimistic" flag if the doc
  is revised anyway.
  - **Reject GLM's "~2.8-3ms" replacement figure** — GLM did not show
    the bandwidth assumption behind that number and I did not
    independently verify MI50 realistic effective bandwidth, so I'm
    not carrying a specific alternative number into the merged doc,
    only the "optimistic, flag it" observation.
- **Phase 3 k-sweep memory scaling** ("k=4 grows capture sizes ~30%
  over the k=3 estimate in §4") — accept the general point (§4's
  memory budget is computed for one k value and Phase 3 sweeps k,
  so the budget should be restated as a function of k), did not
  re-derive the specific 30% figure but the direction is obviously
  correct given q(1+k) graph family sizing scales with k.
- **MoE layer count (48→30) should be stated explicitly in the
  roadmap, not just derivable** — accept, cheap documentation fix,
  consistent with my own finding #6 pattern (unstated derivations).
- **`/tmp/bench/spec_step_probe2.py` vs in-tree `spec_step_probe.py`
  naming drift** — accept as a low-cost verification-hygiene ask; did
  not chase down whether it's a pure rename, but it's a one-line check
  before the doc is considered final.

## My original findings not raised by GLM (retained)

1. **Phase 1 mis-scoped as "extend" when it's a new kernel against
   vendored third-party code** — GLM's F3 (state rollback) approaches
   Phase 1 risk from the rejection/state angle; neither GLM finding
   addresses that the kernel itself
   (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`)
   is upstream-vendored, not gfx906-owned, and is M=1 by hardcoded
   shape/grid, not by a loop bound that's simple to parameterize.
   Retained in full — see original review body below for the kernel-
   level detail (grid indexing, hardcoded `out.shape == (B,1,HV,V)`).
2. **MTP/EAGLE §6 row asserts 1.1-1.3× with no shown derivation**,
   unlike §1's fully-worked ceiling math. GLM did not address §6's
   MTP row. Retained.
3. **Adoption gate (§5) wording ambiguity**: "zero regression" vs.
   the ~4% noise band established by both reviews — literal reading
   fails on any run. Retained; note this is now double-confirmed by
   two independent noise-band analyses (mine and GLM's F4), which
   strengthens the case for fixing the gate's wording precisely, not
   just noting it as a nit.

## What survives attack (from GLM, independently consistent with my read)

- B1/B2/B3 attribution and the FA `UNIFORM_BATCH` prerequisite fix
  (commit `4e40e3e`) — verified against DEVLOG and the diff in my
  original pass; GLM reaches the same conclusion.
- No-MTP-head constraint — not independently re-verified against both
  `config.json`s in this merge pass, but uncontested by either review
  and not load-bearing for any finding above.
- Phase ordering (B1 before B2) — both reviews agree this is correct;
  no new analysis needed, GLM's reasoning (B1 is the shared lever, B2
  only pays on no-draft-heavy methods) matches my own §1 reading.
- The 0.68× negative result and diagnosis — both reviews treat this as
  solid; independently verified against DEVLOG in my original pass.

## Consolidated recommendation list (supersedes both individual docs)

1. **Suffix arm (A1, was GLM F2):** pin `num_speculative_tokens`
   explicitly for the Phase 0 suffix probe, document it will run
   PIECEWISE and won't benefit from Phase 1/2 as scoped, and either
   scope a third "non-uniform small-M" work item or drop suffix as a
   Phase-1/2 beneficiary candidate.
2. **GDN state rollback (A2, was GLM F3, corrected):** add a §3
   paragraph on writing per-token state into the existing
   `mamba_cache_mode=align` block-slot mechanism; update the numerics
   gate to cover per-token-boundary state, not just the final state.
3. **Ceiling math (A3, was GLM F1 + my #4):** add the sensitivity
   table and an explicit post-P1 stop rule; resolve the 800µs/layer
   vs 20-30ms vs 18-25ms discrepancy before using it to justify Phase
   1; fold B3 into a scoped phase or state explicitly why it's assumed
   free.
4. **Phase 0 gate noise floor (A4, was GLM F4 + my #3):** restate as
   mean + CI-lower-bound ≥ baseline mean; add repeat runs to the bench
   script.
5. **ngram_gpu draft equivalence (A5, was GLM F5):** add an explicit
   draft-token-equality check to Phase 0, not just a throughput
   comparison.
6. **Phase 1 kernel scope (my #1, not in GLM):** re-scope from
   "extend" to "new kernel derived from upstream FLA reference,"
   budget 2-3× the implied effort, decide gfx906-local fork vs.
   upstream contribution.
7. **Phase 2 architecture + regression test (A6, was GLM F6 + my #2):**
   acknowledge the scalar `uniform_decode_query_len` architecture
   limit AND the distinct-captured-region problem; add an explicit
   "no-spec q1 dispatch unchanged" regression test as a Phase 2 exit
   criterion, not just an aggregate throughput check.
8. **§6 MTP derivation (my #2, not in GLM):** show the arithmetic
   behind 1.1-1.3× or reframe as directional-only.
9. **§5 gate wording (my #6/#3 + GLM F4, merged):** replace "zero
   regression" with a noise-band-aware criterion; both reviews'
   independent noise measurements make this a solid double-confirm.
10. **Minor doc hygiene (A7):** flag optimistic draft-model bandwidth
    number, restate §4 memory budget as a function of k, state MoE
    layer count explicitly, verify bench script rename is clean.
