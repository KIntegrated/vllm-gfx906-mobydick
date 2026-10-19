# Upstream report — DFlash drafter never stores the first `num_context` context-KV rows on prefill steps

**Status:** fix implemented and validated in the fork (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`);
ready to file upstream. Duplicate check **re-run 2026-10-19** against `upstream/main` `dde9aca41c`:
no matching issue/PR (see [Duplicate check](#duplicate-check-2026-10-19) for the near-neighbours).
The bug is **still live in `upstream/main` `dde9aca41c`** — see
[Re-verification against current main](#re-verification-against-current-main-2026-10-19), which also
carries the rebased one-line patch (`/local/tmp/u6-dflash-context-kv-row0.patch`) and the
ready-to-post issue body.

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

## Re-verification against current main (2026-10-19)

`upstream/main` `dde9aca41c` (fetched 2026-10-19 03:35 PDT) has **refactored this code without changing
the row ownership**, so the hole survives and the fix has the same shape:

| site | current main | writes |
|---|---|---|
| `_prepare_graph_forward()` (captured, `prepare_forward=` passed to `DFlashCudaGraphManager.capture`, speculator.py:195-203) | `self._precompute_context_kv(0, num_context)` | `[0, num_context)` — **only inside the graph** |
| `_prepare_graph_replay()` (eager, speculator.py:219-237, called from `propose()`'s `cg_mode == FULL` branch) | `self._precompute_context_kv(num_context, num_target_tokens)` | `[num_context, num_target_tokens)` |
| `propose()` non-FULL branch (speculator.py:585) | `self._precompute_context_kv(0, num_target_tokens, dummy_run)` | `[0, num_target_tokens)` — complete |

So the FULL path still assumes the captured graph stores rows `[0, num_context)` on every step that
takes it, including the prefill steps where the measured hole was. Patch against current main
(`/local/tmp/u6-dflash-context-kv-row0.patch`, worktree `/local/tmp/wt-u6-dflash`):

```diff
--- a/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
+++ b/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
@@ -233,8 +233,11 @@ class DFlashSpeculator(DraftModelSpeculator):
                 PAD_SLOT_ID
             )
         else:
-            # Prefill context beyond the graph's rows is stored before replay.
-            self._precompute_context_kv(num_context, num_target_tokens)
+            # Store from row 0: on a prefill step the captured context step does
+            # not run, so rows [0, num_context) would otherwise never be stored
+            # and the drafter would attend over stale K/V. The rows the graph
+            # does write are identical here, so the overlap is idempotent.
+            self._precompute_context_kv(0, num_target_tokens)
```

**Caveat, do not overstate:** the before/after acceptance numbers below were measured on the
**pre-refactor** shape (the fork's base still carries the old `propose()`). The refactor split the
same two stores out of `propose()` but left them owning the same rows, so the hole should persist —
that is an inference from code, not a re-measurement. The reporter should either re-run the 6.4k
acceptance check on current main or file the mechanism + the fix with that caveat stated.

## Duplicate check (2026-10-19)

No upstream issue or PR reports this hole. Near-neighbours, all different:

| item | why it is not this |
|---|---|
| #47930 (DFlash/DSpark acceptance collapse, APC on) | long-context drafter throughput (≥128k, APC / restored KV); this bug triggers on a plain **prefill** at 1.6-6.4k with no cache involvement |
| #61261 / #60320 (`draft_context_window`) | bounds the drafter's attention window instead of storing the missing rows; measured separately at 128k+ |
| #60174 (DFlash2/DSpark + APC corrupt output) | prefix-cache corruption, not missing context-KV rows |
| #57032 (drafter KV group not identified) | prefix-cache reuse disabled for mamba groups |
| #51581 / #51620 | quantized-drafter context-KV projection |

The introducer is confirmed: #57632 "[DFlash] Capture the context K/V precompute in the draft CUDA
graph", merged 2026-09-24 as `04730e8270` — before it the eager path stored from row 0.

## Ready-to-post issue body

> **Title:** `[Bug][DFlash] Drafter never stores the first num_context context-KV rows on prefill steps — acceptance collapses to 1.0 above ~4k`
>
> **TL;DR** With a DFlash drafter, acceptance collapses to **exactly one token per step** once the
> prompt exceeds ~4k tokens, intermittently. `_prepare_graph_replay()` stores only
> `[num_context, num_target_tokens)` and assumes the captured context step wrote
> `[0, num_context)`; on a prefill step those rows are never stored, so the drafter attends over
> stale K/V. Fix: store from row 0 (idempotent with the rows the graph writes).
>
> **Repro (2× MI50, gfx906, ROCm 7.14, fp16 drafts, greedy):** Nemotron-Bolt 74B + LiLiCorr k=7,
> TP=2, production flags, a 6.4k distinct-source agent prompt reproduces `1.0, 1.0, 5.65`
> accepted/step across three reps; 1.6k sits at 2.97. With the one-line fix: `3.75, 5.94, 3.80` and
> `4.29, 4.23`. Poisons evidence: filling every draft `kv_cache` with NaN once per request before its
> prefill drops acceptance to 0 at both lengths, and the eager store's slot log on a 1608-token
> prompt starts at `pos_min=8`.
>
> **Introduced by:** #57632 (`04730e8270`). **Still present in** `upstream/main` `dde9aca41c`
> (`_prepare_graph_replay`, speculator.py:237), where the captured `_prepare_graph_forward` owns
> `[0, num_context)` and the eager replay path owns `[num_context, num_target_tokens)`.
>
> **Fix:** `self._precompute_context_kv(num_context, num_target_tokens)` →
> `self._precompute_context_kv(0, num_target_tokens)` in `_prepare_graph_replay()`. Costs
> `num_context` duplicate KV rows per prefill step. Output is never wrong (the target verifies), so
> this is invisible to correctness evals and shows up only as lost throughput.
>
> **Caveat:** the measurements above are from the pre-refactor shape; the row ownership is unchanged
> in the refactor, so the hole is inferred to persist, not re-measured on `dde9aca41c`.

## Notes for the upstream report

- Found on 2x MI50 (gfx906, ROCm 7.14, fp16 drafts), but the code path is device-agnostic: it is a
  stale-KV read, not a kernel issue. A CUDA reproduction should show the same collapse.
- The fork fix is `1` line + comment; a regression test would need a spec-decode accuracy run at a
  prompt longer than one `max_num_batched_tokens` (acceptance must not drop to 1.0).
