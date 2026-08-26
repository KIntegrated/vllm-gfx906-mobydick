# W1 — multi-request GDN chunk reclass (spec-mixed batches)

> Branch `gfx906/gdn-mixed-decode` off `gfx906/main` @ `67ae6c3f96` ·
> model Qwen3.5-9B (probes) / Qwen3.8-27B-AWQ-INT4 (gate) ·
> 2026-08-26 · roadmap item W1 (`spec-decode-roadmap.md`).

**VERDICT:** SHIPPED · **GATE:** serving wall-clock A/B — 27B mixed
2-request ngram serving, graph, 0.82, max_seqs 4, pp2048/tg256, 4
samples/arm, `BENCH_NREQS=2 BENCH_MIXED=1` on `_bench_gfx906.py`:
**59.35 vs 55.60 t/s = +6.7 %** (bands ±0.3 %). Numerics identity
(request A token-identical) and the kernel-path spy are co-gates;
unit tests are a correctness floor, not the gate.

## HYPOTHESIS

If spec-mixed batches peel the 1-token non-spec decodes to the per-seq
recurrent kernel (instead of reclassifying them into 1-token "prefills"
that pay the chunk kernel), then the ~415 µs/layer chunk cost
disappears from mixed steps and serving throughput on a mixed ngram
workload improves — without changing any spec-path output.

## Problem

In spec-decode mixed batches (some requests drafting, some not), the
GDN metadata builder reclassified every non-spec 1-token decode as a
1-token "prefill" (`#34845` fix, `gdn_attn.py`). Each such sequence
then paid the **chunk kernel** (`chunk_gated_delta_rule`, ~415
µs/layer at 27B shapes) for a single token, instead of the per-seq
recurrent kernel (`fused_sigmoid_gating_delta_rule_update`,
~20–32 µs/layer) — ≈ ~20 ms/step *per reclassified sequence*, on all
30–48 GDN layers. No-spec mixed batches already peel decodes to the
recurrent kernel; only the spec-mixed case paid the chunk tax.

## Change (Option A — extend the no-spec peel to spec-mixed batches)

- `vllm/v1/attention/backends/gdn_attn.py`
  - removed the reclass block (`num_prefills += num_decodes; ...`) and
    the `assert not (num_decodes > 0 and num_spec_decodes > 0)`;
  - extended both prefill-only-metadata conditions from
    `spec_sequence_masks is None and num_decodes > 0` to
    `num_decodes > 0`, so in spec-mixed batches the chunk metadata
    (`prefill_query_start_loc` / `prefill_state_indices` /
    `prefill_has_initial_state`) covers the real prefills only
    (rebased off the decode-first front slice).
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
  (`_forward_core`)
  - `split_non_spec` no longer requires `spec_sequence_masks is None`;
  - peeled decode rows: `a_non_spec_decode`/`b_non_spec_decode`
    (`index_select(non_spec_token_indx[:num_decode_tokens])` in
    spec-mixed batches; `[:num_decode_tokens]` slice in no-spec);
  - conv: the 1.2-elif decode-only branch slices
    `non_spec_state_indices_tensor[:num_decode_tokens]` (was
    `[:num_actual_tokens]` — equivalent in no-spec decode-only,
    correct in spec decode-only; `num_decode_tokens` binding moved
    above the conv block);
  - 2.2 (split) and 2.3-elif (spec+decode-only) now run
    `fused_sigmoid_gating_delta_rule_update` on the peeled decode rows
    (no `num_accepted_tokens` — plain 1-token recurrent update, the
    same kernel family the spec rows and the no-spec mixed path use);
  - 2.3-if (prefill) unchanged in form: chunk on the prefill-only
    metadata, then decode/prefill outputs stitched in decode-first
    order for the `index_copy` merge.
- `tests/v1/attention/test_gdn_metadata_builder.py` — updated the
  #34845 expectations (decodes are kept, prefill metadata covers real
  prefills only) and added two peel-specific cases. **10/10 pass.**

### Why it is cudagraph-safe

Mixed batches have non-uniform per-seq token counts, so
`_is_uniform_decode` (gpu_model_runner) is False → FULL decode-graph
keys never match → mixed steps run **eager** with freshly built
metadata. The FULL-graph request-indexed tensors are untouched (their
condition still keys on `num_spec_decodes == batch_size`-style
uniformity). No stale-buffer replay is possible.

### Blast radius

- OLMo / Kimi GDN variants consume the full non-spec metadata
  (`non_spec_query_start_loc`), not the prefill-only fields —
  unaffected (Kimi gains the per-token path in no-spec mixed batches
  for free).
- The MTP-fused decode path (`_can_use_fused_gdn_mtp_decode`) requires
  `num_decodes == 0` — correctly skipped for mixed batches, unchanged.
- Pure-spec, pure-no-spec-decode, and no-spec-mixed batches are
  bit-identical in dispatch to before (the no-spec conditions were
  already `num_decodes > 0` in effect; the slice→index_select change
  only applies when `spec_sequence_masks is not None`).

## Validation — session 1 (2026-08-26, boot H; canary 38.9 t/s healthy)

Probe: `probe_gdn_mixed.py` (9B eager, ngram n=5; A = repetitive
filler, temp 0 → always drafts; B variant per gate; kernel + per-step
composition spies, scheduler-level ground truth).

1. **Kernel routing** (9B, B = random-gibberish prompt, temp 1 → B
   drafts rarely → 84 mixed steps, the maximum-fraction regime):

   | | BEFORE (main) | AFTER (W1) |
   |---|---|---|
   | mixed-step composition | `comp=(1,0,1)` — B reclassified prefill | `comp=(0,1,1)` — B peeled decode |
   | chunk calls | 2112 (96 real prefill + **2016 1-token B "prefills"**) | **96** (prefill only) |
   | fused_seq non-spec | 0 | **2016** (84 steps × 24 layers) |
   | t/s | 60.7 | 60.2–64.9 |

   The reclass pathology is exactly reproduced before and eliminated
   after, at the kernel level. (The 9B t/s delta is within run-to-run
   noise — the 1-token chunk-vs-fused gap is several times smaller on
   9B than 27B; the 27B number below is the gate.)

2. **Identity / numerics** (9B, B = diverse sentence pool ending on a
   novel mid-sentence, temp 0 → deterministic; 40 mixed steps):
   - **Request A (always-spec, untouched path): token-identical
     across all 6 runs, both arms** (hash `5c2d0e91a350`) — the free
     identity gate passes.
   - Request B: full sequences differ between arms, but the
     divergence is **pre-existing fp16 non-determinism, not W1**: the
     *unmodified* main arm itself produces two different B sequences
     across runs (`65096979406d` / `ed6fcbb38122`), and **both arms
     reach exactly the same reachable set** {`6509…`, `ed6f…`}; the
     first 80 chars are identical in every pairing and all sequences
     are coherent prose. Token-identity gating for B is unusable on
     this machine (the same temp=0 non-determinism documented for the
     27B/35B records) — the gate falls back to coherence + A-identity
     + the reachable-set argument, all green.

3. **Serving A/B (the gate)** — 27B Qwen3.8-AWQ-INT4, graph, 0.82,
   max_seqs 4, ngram n=5, `BENCH_NREQS=2 BENCH_MIXED=1` (request 0 =
   2048-token repetitive filler, request 1 = 190-token diverse pool
   with novel ending → mixed batches on most decode steps), 4
   samples/arm:

   | | AFTER (W1) | BEFORE (reclass) |
   |---|---|---|
   | samples (t/s) | 59.49 / 59.26 / 59.28 / 59.35 | 55.65 / 55.42 / 55.77 / 55.53 |
   | mean | **59.35** | **55.60** |

   **+6.7 %**, bands ±0.3 % both arms. Below the roadmap's ~20
   ms/step naive estimate because B (190-token pool on 27B) still
   drafts a substantial share of steps (echoing the pool), capping
   the mixed-step fraction; production agentic batches with several
   non-drafting requests alongside drafting ones get more.

## Session 2 — code-review fixes (2026-08-26, boot H)

Per `docs/gfx906/gdn-mixed-decode-code-rev.md` (merged human + GPT
review of this branch). The review's view of the W1 dispatch itself
was unchanged (coherent, +6.7 % real); the blockers were around the
contract change's blast radius and branch hygiene. All 10 findings
addressed:

- **[P1] CPU GDN backend** (finding 1) — the contract change broke
  `_spec_aware_nonspec_subset`: it fed the full non-spec token range
  (decodes + prefills) to the chunked path with prefill-only
  cu_seqlens, so any spec batch containing a non-spec decode crashed
  (index-size mismatch) or mis-routed. Fixed per the review's
  Option A: the 1-token decodes peel to the per-seq recurrent update
  (factored as `_wide_buffer_nonspec_decode`, shared with
  `_spec_aware_nonspec`) and the prefill tail goes to the chunked
  path with prefill-only metadata (`ae87ce7c25`).
- **[P1] Kimi K3 AMD** (finding 2) — same contract mismatch latent in
  `kimi_k3/amd/kda.py` (full non-spec cu_seqlens + prefill-only
  `m.chunk_indices`). Fixed per the review's option (a): the peel
  keys off `m.num_decodes > 0` alone, so the spec case takes the
  already-consistent no-spec path (`b15b88cc1f`, standalone commit).
  Kimi K3 does not ship spec-decode today and no Kimi checkpoint is
  local, so this dead path is **not runtime-validated** (compile- and
  read-checked only); drop the commit if a minimal W1 diff is wanted.
- **[P2] Scope creep** (finding 7) — the three L3 ngram probe scripts
  (verbatim copy of `18c235772d` via `8ac610f8a0`) removed from this
  branch (−379 lines; canonically on `gfx906/ngram-cpu-d2h`). The
  harness env-var support from that commit is kept — W1's own gate
  recipe uses `BENCH_NREQS`/`BENCH_SPEC_CONFIG`/`BENCH_CG_MAX`
  (`2aa0a92ae8`).
- **[P2] Lint** (finding 3) — ruff clean on all W1-touched files
  (F401/F841 by hand, E501 wraps; string concatenations
  byte-identical, incl. the probe's novel mid-sentence ending that
  keeps request B non-spec) (`9b0b58bbc6`).
- **[P2] Contract made explicit** (findings 5, 9 + carried items,
  `233e8f202b`) — `GDNAttentionMetadata` docstring states the
  contract; the builder asserts the decode-first ramp on the CPU-side
  `non_spec_query_start_loc_cpu` (once per step, **no device sync** —
  the ramp check lives in the builder rather than per-layer in
  `_forward_core`, an intentional adaptation of the review's sketch
  to keep the decode hot path sync-free); `_forward_core` asserts
  `num_decode_tokens == num_decodes` (int compare, free) and the 1.2
  conv comment records that its full-non-spec coverage is intentional;
  stale W1-history comment trails trimmed.
- **[P3] Devlog template** (finding 6) — this restructure.
- **[P3] PR #53077** (finding 10) — **checked, already present**:
  both `gfx906/main` and this branch zero `num_spec_decodes` in the
  empty-draft early-exit of `GDNAttentionMetadataBuilder.build()`
  (same semantics as upstream `6df7adc17f`), so the stale-count
  interaction the review flagged cannot occur. No action needed.
- **[P3] Harness env vars** (finding 8) — documented in
  `running.md` §3.

Evidence (session 2):

- New contract test `tests/kernels/mamba/cpu/test_cpu_gdn_nonspec_peel.py`
  (torch-reference leaf ops; runs on any platform — the C++ CPU ops
  exist only in CPU builds): 3 shapes — [decode, prefill, spec],
  [decode, spec], [prefill, spec]. **3/3 pass on the fix; 2/3 fail on
  the pre-fix code with the exact index-size mismatch** (the shapes
  with a non-spec decode) — a true regression test for finding 1.
  State advancement (SSM + conv rolling buffer, untouched slots) and
  scatter placement are asserted, not just the core outputs.
  Kernel-vs-reference accuracy stays covered by
  `test_cpu_gdn_ops.py` on CPU builds.
- GDN metadata builder tests 10/10 (the new builder assert is silent
  on all existing shapes); ruff clean; py_compile clean on the Kimi
  file.
- **Perf sanity (same-day A/B, both arms on today's machine state)** —
  27B mixed 2-request ngram, same recipe as the gate, 4 samples:
  post-fix **57.50 / 57.37 / 57.19 / 57.05 (mean 57.28, ±0.2 %)** vs
  pre-review-fix (`b31ed39e05`) **57.23 / 57.10 / 57.14 / 57.19
  (mean 57.17, ±0.2 %)** → **+0.2 %, neutral**. Both bands sit ~3.3 %
  below the session-1 record (59.35) uniformly — host drift over the
  boot, not a code effect (no wedge/hang event; spec path not
  collapsed). `/tmp/bench27_w1rev_after_sanity*.log`,
  `/tmp/bench27_w1rev_before_sanity4.log`.

Open (session 2):

- `schema_version` / per-backend `decode_peel_supported` on the
  metadata (finding 6, medium-term): **deferred** — after the CPU and
  Kimi fixes there are no stale consumers in-tree, the docstring +
  builder assert make future drift loud, and a capability table is
  process overhead until a third consumer appears. Revisit then.
- 4-request mixed serving A/B (human-review carryover): still
  pending; the 2-request shape is the recorded gate.

**VERDICT (session 2): SHIPPED.** Both P1s fixed and regression-tested
(CPU) / guarded (Kimi), P2s closed, P3s closed or explicitly deferred;
no perf regression (same-day A/B neutral).

## Verdict (overall)

**VERDICT: SHIPPED.** The reclass was real (2016 wasted 1-token chunk
kernel calls in the 9B probe alone), the peel removes it at the
kernel level, changes nothing on the spec path (A token-identical),
adds no new output modes for the peeled side, and is +6.7 % on the
27B mixed serving A/B. Session 2 closed the review's blast-radius
findings (CPU + Kimi) and the branch-hygiene items. GATE: serving
wall-clock A/B — passed.

## Files

- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`
  (session 2: non-spec decode peel, review P1)
- `vllm/models/kimi_k3/amd/kda.py` (session 2: spec-mixed peel,
  review P1, standalone commit)
- `tests/v1/attention/test_gdn_metadata_builder.py`
- `tests/kernels/mamba/cpu/test_cpu_gdn_nonspec_peel.py` (new,
  session 2: contract regression test)
- `benchmarks/kernels/gfx906/probe_gdn_mixed.py` (new)
- `docs/gfx906/_bench_gfx906.py` (`BENCH_MIXED` mixed-prompt mode +
  cherry-picked `18c235772d` harness envs)
- `docs/gfx906/running.md` (session 2: harness env-var surface)
- removed (session 2): `benchmarks/kernels/gfx906/{bench_ngram_cpu,
  compare_ngram_cpu_gpu,probe_ngram_cpu_engine}.py` (L3 scope creep)
