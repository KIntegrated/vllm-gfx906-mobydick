# Adversarial review: `spec-decode-roadmap.md`

> Reviewer: GLM (agent). Scope: attack the proposal's claims, math, and
> phasing. Sources checked: DEVLOG-moe-opt.md §"n-gram spec decode probe",
> `vllm/config/speculative.py`, `vllm/v1/spec_decode/suffix_decoding.py`,
> `vllm/v1/spec_decode/ngram_proposer_gpu.py`,
> `vllm/v1/worker/gpu_model_runner.py` (`_is_uniform_decode`,
> `uniform_decode_query_len`), both model `config.json`s, and the in-tree
> bench scripts.

Copyright Kevin Read <me@kevin-read.com>

## Verdict

The problem diagnosis (B1/B2/B3 attribution, the FA prerequisite fix, the
"no MTP head" constraint) is solid and internally consistent. The plan's
weaknesses are: (a) the headline 1.19× / 1.15×-gate margin rests on the
**optimistic end of every cost estimate simultaneously** and collapses
below the gate at mid-range estimates; (b) the **suffix arm of Phase 0 is
not the cheap config A/B the roadmap claims** — it pulls in an external
dependency, dynamic (non-uniform) draft lengths that break both the Phase 1
kernel's uniform-M precondition and the Phase 2 graph routing, and a
default k of 24 that is outside the proposed kernel's M ≤ 8 range; (c) the
GDN state-rollback requirement on rejection is never addressed and is the
most likely place for Phase 1 to grow real complexity and memory cost.

## Findings, ranked

### F1 (High) — The ceiling math is best-case-stacked; the adoption gate has almost no margin

Recompute the roadmap's own numbers with its own inputs
(T1 = 36.5 ms, r = 0.40, a = 1.1 → 1.44 tok/step):

| draft-step cost after P1 | mean step | throughput | speedup | vs 1.15× gate |
|---|---|---|---|---|
| 55 ms (roadmap claim) | 43.9 ms | 32.8 t/s | 1.19× | pass |
| 62 ms (mid estimate) | 46.3 ms | 31.1 t/s | 1.13× | **fail** |
| 70 ms | 48.4 ms | 29.8 t/s | 1.08× | **fail** |

Where does 55 ms come from? The devlog says draft steps are ~80–110 ms
against a ~45 ms ideal, with B1 = ~18–25 ms (roadmap table says 20–30).
Only 80 − 25 = 55 works — i.e. the claim takes the *fastest* observed
draft step, the *largest* B1 attribution, and assumes B3 contributes
nothing beyond what's left. Any of those being off by 20% sinks the
project below its own gate. Also note the roadmap's B1 figure
(0.8 µs? no — ~800 µs/layer × 48 layers ≈ **38 ms**) doesn't reconcile
with its own "~20–30 ms/step" or the devlog's 18–25 ms; the per-call vs
per-layer vs per-step accounting in the B1 row is muddled and should be
cleaned up before it's used to justify a kernel project.

**Ask:** a sensitivity row in §1, and a decision rule for the sub-gate
outcome (e.g. "if post-P1 draft step > 60 ms, stop after P1 and record").

### F2 (High) — The suffix arm is misrepresented as "config-only"; it structurally conflicts with Phases 1–2

Three separate problems, all verifiable in-tree:

1. **External dependency**: `_validate_suffix_decoding()` raises
   `ImportError` unless `arctic-inference==0.1.1` is installed
   (`vllm/config/speculative.py`). Not an in-tree freebie; on ROCm this
   wheel is unverified. Phase 0's "it's an afternoon" claim needs this
   install step to actually succeed first.
2. **Dynamic draft lengths**: `suffix_decoding.py` explicitly
   "will speculate a dynamic number of tokens for each request every
   decoding step." Non-uniform steps are exactly what
   `_is_uniform_decode` rejects → **every** suffix step runs PIECEWISE.
   Phase 2 as written (route uniform 1-token steps to a q1 graph) does
   nothing for suffix's ragged multi-token steps, and Phase 1's kernel
   dispatch requires *uniform* `1 < num_tokens ≤ 8`. So if suffix wins
   Phase 0, the roadmap's two big levers **do not deliver its gains** —
   the plan would need a third work item (masked/non-uniform small-M path
   + non-uniform graph handling) that appears nowhere.
3. **Default k = 24**: when `num_speculative_tokens` is unset, suffix
   defaults it to `suffix_decoding_max_tree_depth` = 24, so
   `uniform_decode_query_len` = 25 — outside the Phase 1 kernel's M ≤ 8,
   and the q(1+k) capture family becomes [25 .. 25·max_num_seqs], blowing
   past the §4 memory budget estimate. Anyone running the Phase 0 arm
   with defaults is benchmarking a configuration the roadmap's own
   infrastructure cannot support.

**Fix:** either scope the suffix arm honestly (set
`num_speculative_tokens ≤ 4` explicitly, accept PIECEWISE, treat it as a
draft-quality probe only), or promote the non-uniform small-M path to a
named, gated work item.

### F3 (High) — Phase 1 ignores GDN state rollback on rejection

The proposal specifies bit-equality of the M-step result vs sequential
M=1 calls and the final-state output. But spec decode needs the GDN state
**at the last accepted token**, not at the end of the M drafted tokens:
when 2 of 4 drafts are rejected, the recurrence state must be restored to
the boundary (or recomputed). Where do those intermediate states come
from?

- The chunk (prefill) kernel family the spec path currently uses returns
  per-chunk states — the current pipeline gets rollback states for free;
  the proposed fused inner-loop kernel, as specified, produces only the
  final state.
- Options are all costly: per-token state snapshots (the state pool is
  ~72 MB/seq — ×M is a non-starter), recompute-on-rejection (serial M=1
  re-runs eat the win), or the rejection-aware trick of keeping
  checkpoints only at accepted-prefix boundaries (requires knowing
  acceptance *inside* the model call, or deferring state commit).
- The existing "state save/restore" in B3 bookkeeping suggests upstream
  already does save/restore somewhere — the proposal must say how its
  kernel slots into that mechanism before any numerics gate makes sense.

This is the most likely source of schedule slip in Phase 1 and it is
entirely unmentioned. **Ask:** one paragraph in §3 on state
checkpoint/rollback semantics, and whether the bit-equality test target
should be "final state after M tokens" or "state at each token boundary".

### F4 (Medium) — Phase 0's ship gate (≥ 27.5 t/s) is inside measurement noise

Baseline arms across the devlog: 25.25 (LLM harness), 26.81 / 27.96 /
27.62 (per-prompt), 27.46 (mean) — spread > 1 t/s ≈ 4%, on n = 3 prompts
× 512 tokens, greedy. A gate at 27.5 (0.15% above the mean) will flip on
run-to-run noise. Either widen the prompt set / token count for gate
runs, or state the gate as "mean + CI lower bound ≥ baseline mean".

### F5 (Medium) — "ngram_gpu: same drafts as ngram" is asserted, not established

The GPU proposer (`ngram_proposer_gpu.py`) is a reimplementation
(unfold/argmax, "picks the longest valid match" across n-gram lengths);
tie-breaking and match-selection order need not match the CPU proposer.
Cheap to verify: compare draft-token equality and acceptance stats
between the two arms in the Phase 0 bench. If acceptance differs, the
"removes B3b only" framing is wrong and k/min_n tuning doesn't transfer.

### F6 (Medium) — Phase 2 "preferred" change is harder than one line of `_is_uniform_decode`

Routing a uniform 1-token spec step to a q1 FULL graph is not just
extending the predicate: the q1-spec graph is *not* the no-spec decode
graph (spec metadata, rejection-sampler output plumbing, draft-state
save/restore hooks are in the captured region or its prologue). The
capture-set derivation, sampler shapes, and the FA backend's uniform
fast-path host-identity check (`num_tokens == num_seqs * max_seqlen_q`)
all need a spec-aware q1 variant. Budget line covers memory only — no
budget for the engineering or the "inert when no spec configured" claim,
which needs its own regression test (the no-spec baselines are the
project's record numbers; a silent dispatch change there is exactly the
regression the Phase 3 gate forbids).

### F7 (Low) — Assorted nits

- §6 "1B draft reads ~2 GB/step ≈ 2 ms" assumes ~1 TB/s effective on
  MI50 — near theoretical peak; realistic is ~2.8–3 ms, plus launch
  overhead. Doesn't change the conclusion, but the number is optimistic.
- §1 says ngram/suffix no-draft steps are "~60%"; devlog consistent. OK.
- Phase 3 k sweep {1,2,3,4}: at k = 4 the q(1+k) graph capture sizes grow
  ~30% over the k = 3 memory estimate in §4 — restate the budget as a
  function of k, especially for the memory-constrained dense 27B
  (BENCH_MAX_SEQS = 4 constraint).
- The MoE GDN-layer claim "48→30" is correct for the 35B (40 layers,
  interval 4 → 30 GDN), but the roadmap never states the MoE layer count;
  add it so the "B1 is a bigger fraction there" claim is checkable.
- §7 lists the bench scripts as moved in-tree; confirm the devlog's
  `/tmp/bench/spec_step_probe2.py` vs in-tree `spec_step_probe.py` naming
  drift is just a rename, not a stale reference.

## What survives attack

- B1/B2/B3 attribution and the FA UNIFORM_BATCH prerequisite (commit
  `4e40e3e`) — consistent across roadmap, devlog, and code.
- No-MTP-head constraint for both checkpoints — verified in both
  `config.json`s (no `num_nextn_predict_layers`; 64/40 layers, interval 4
  → 48/16 and 30/10 GDN/full split).
- Ordering: Phase 1 (B1) before Phase 2 (B2) is right — B1 is the shared
  lever and B2 only pays on no-draft-heavy methods.
- The 0.68× negative result and its diagnosis are trustworthy and well
  documented.

## Recommended edits before committing to Phase 1

1. Add the sensitivity table (F1) and a stop rule tied to measured
   post-P1 draft-step cost.
2. Rescope or re-gate the suffix arm (F2); explicitly set
   `num_speculative_tokens` and acknowledge PIECEWISE + non-uniform-M gap.
3. Add the state-rollback design note to §3 (F3); it may change the
   kernel's output contract (per-boundary states) and thus the numerics
   test.
4. Restate the Phase 0 gate with a noise floor (F4).
5. Add a "no-spec regression" test obligation to Phase 2 (F6).
