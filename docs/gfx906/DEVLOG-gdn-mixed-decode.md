# DEVLOG: W1 — multi-request GDN chunk reclass (mixed spec batches)

Status: **OPEN** — code complete, unit gates green, GPU before/after +
PPL gates pending (boot-G burst wedge 2026-08-26 ~00:58 — reboot
required; see `degradation.md`).

GATE: serving wall-clock A/B on a 2-request mixed probe + PPL delta +
kernel-path spy (house protocol). Unit tests are a correctness floor,
not the gate.

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

## Validation plan (pending GPU)

1. **2-request mixed probe** (`benchmarks/kernels/gfx906/probe_gdn_mixed.py`,
   9B eager, ngram n=5): request A repetitive (always drafts), B
   non-repetitive (rarely drafts). Spy on GDN leaf kernels + per-step
   batch composition. Before: B's rows in the chunk kernel
   (`comp=(prefills≥1, decodes=0, spec=1)`); after:
   `comp=(prefills, decodes≥1, spec=1)` with B in
   `fused_sigmoid_gating_delta_rule_update` (no `num_accepted_tokens`),
   chunk only for real prefills. Expect the mixed-step t/s gap to
   close (chunk ~415 µs vs fused ~20–32 µs per GDN layer × 24–48
   layers).
2. **PPL gate**: same mixed 2-request run through `ppl_probe.py`
   before/after — expect ΔPPL ≤ ~0.2 (cross-kernel fp16 noise band,
   per the MoE A/B precedent). Exact token identity is NOT achievable
   (near-tie argmax flips between the chunk and recurrent kernels);
   request A (the drafting side, untouched path) SHOULD be
   token-identical — that is a free identity gate.
3. **27B serving A/B** (production shape): the t/s gate.

## Known risk

The peeled decode rows change B's GDN state numerics slightly
(chunk vs recurrent fp16 accumulation) → B's tokens may flip at
near-ties; A's tokens should not change. If the PPL gate exceeds
0.2 or coherence degrades, the peel is wrong somewhere in the
conv/state-index plumbing — bisect 1.2-elif vs 2.3-elif.

## Files

- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- `tests/v1/attention/test_gdn_metadata_builder.py`
- `benchmarks/kernels/gfx906/probe_gdn_mixed.py` (new)
