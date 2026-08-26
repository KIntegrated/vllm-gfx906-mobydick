# DEVLOG: W1 — multi-request GDN chunk reclass (mixed spec batches)

Status: **SHIPPED** (2026-08-26, branch `gfx906/gdn-mixed-decode`) —
all gates green, incl. the serving wall-clock A/B: 27B mixed 2-request
ngram serving **59.35 vs 55.60 t/s = +6.7 %** (4 samples/arm, bands
±0.3 %).

GATE: serving wall-clock A/B on a 2-request mixed probe + numerics
identity + kernel-path spy (house protocol). Unit tests are a
correctness floor, not the gate.

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

## Validation (2026-08-26, boot H; canary 38.9 t/s healthy)

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

## Verdict

**VERDICT: SHIPPED.** The reclass was real (2016 wasted 1-token chunk
kernel calls in the 9B probe alone), the peel removes it at the
kernel level, changes nothing on the spec path (A token-identical),
adds no new output modes for the peeled side, and is +6.7 % on the
27B mixed serving A/B. GATE: serving wall-clock A/B — passed.

## Files

- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- `tests/v1/attention/test_gdn_metadata_builder.py`
- `benchmarks/kernels/gfx906/probe_gdn_mixed.py` (new)
- `docs/gfx906/_bench_gfx906.py` (`BENCH_MIXED` mixed-prompt mode +
  cherry-picked `18c235772d` harness envs)
