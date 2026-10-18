# Upstream report — DFlash drafter never stores the first `num_context` context-KV rows on prefill steps

**Status:** fix implemented and validated in the fork (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`);
ready to file upstream. Duplicate check (2026-10-18): no matching issue/PR in `vllm-project/vllm`
(the only nearby one is #51581, a different DFlash-quantized-kv bug).

## Symptom

Speculative decoding with a DFlash drafter (tested: Nemotron-Bolt 74B + LiLiCorr k=7; also
Qwen3.8-27B + DFlash2 k=7) collapses once the prompt exceeds ~4k tokens: acceptance falls to
**exactly one token per step** (only the bonus token is accepted), intermittently — the same prompt
gives `1.0, 1.0, 5.65` accepted-per-step across three greedy reps. Below ~4k it is stable at
5.5–6.8. Attention backend, sink support and prefill chunk size make no difference.

## Root cause

`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`, `propose()`:

```python
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    num_context = self._num_graph_context_tokens(batch_desc.num_reqs)   # rows of one verify block
    ...
    elif num_target_tokens <= num_context:
        self._context_slot_mappings[:, num_target_tokens:num_context].fill_(PAD_SLOT_ID)
    else:
        # Prefill context beyond the graph's rows is stored before replay.
        self._precompute_context_kv(num_context, num_target_tokens)
```

Rows `[0, num_context)` are left to the captured graph (introduced by #57632, "Capture the context
K/V precompute in the draft CUDA graph"). On a **prefill** step the graph is not what runs, so those
rows are never stored, and the drafter then attends over stale K/V at those positions. With a longer
context the stale rows are read alongside a much larger attention window and dominate the drafts.

## Evidence

1. Slot log on a 1608-token prompt (the outside-graph write): `write n=1600 slots=1600
   slot_min=151160 slot_max=152759 pos_min=8 pos_max=1607` — the store starts at position 8.
2. Poison test: filling every draft `kv_cache` with NaN once per request before its prefill drops
   acceptance to **0 at both 1.6k and 6.4k**, i.e. the drafter depends on slots the request never
   writes.
3. With the fix: 6.4k-token prompt acceptance `1.0, 1.0, 5.65` → **`3.75, 5.94, 3.80`** (3 greedy
   reps) and 1.6k `2.97` → `4.29, 4.23`.

## Fix

Store from row 0 instead of `num_context`:

```python
self._precompute_context_kv(0, num_target_tokens)
```

The store is idempotent (rows `[0, num_context)` are written with the same hidden states, positions
and slot mappings the graph would use), costs `num_context` extra KV rows per prefill step, and makes
the FULL and non-FULL paths consistent.

## Notes for the upstream report

- Found on 2x MI50 (gfx906, ROCm 7.14, fp16 drafts), but the code path is device-agnostic: it is a
  stale-KV read, not a kernel issue. A CUDA reproduction should show the same collapse.
- Suggest confirming #57632 as the introducer (before it, the non-FULL path stored from row 0).
- The fork fix is `1` line + comment; a regression test would need a spec-decode accuracy run at a
  prompt longer than one `max_num_batched_tokens` (acceptance must not drop to 1.0).
