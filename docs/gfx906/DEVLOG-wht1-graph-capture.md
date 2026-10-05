# WHT-1 graph capture: a one-line family-match bug was the whole blocker

Date: 2026-10-05. Item: **#36 WHT-1**, arm "graph capture" (follows
`DEVLOG-wht1-whittle-load.md`). Branch: `gfx906/wht1-graph-capture`.

## Result

Graph capture works on this model, and decode at B=1 goes from 5.61 t/s to
**21.15 t/s** — same probe, same flags, one variable changed (`--enforce-eager`
removed; the split bug below fixed).

| | eager (control) | graphed |
| --- | --- | --- |
| rep 1 (includes warmup) | 2.75 t/s | 4.25 t/s |
| **rep 2** | **5.61 t/s** | **21.15 t/s** |
| KV cache | 2.0 GiB / 117,964 tok | 1.07 GiB / 62,914 tok (3.84x at 16k) |
| output | coherent, identical | coherent, **identical** |

The KV difference is the cudagraph pool reserving device memory up front, not a
capability loss.

## Root cause: `model_type` was matched exactly, and this checkpoint spells it differently

The failure signature was a capture-time abort on both TP workers:

```
File ".../vllm/models/qwen4_exp/amd/ple_layer.py", line 1237, in ...
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
Search for `hipErrorStreamCaptureUnsupported' ...
```

Line 1237 is `pinned_ids.copy_(ngram_ids, non_blocking=False)` — a **blocking
device->host copy**. The PLE n-gram table is host-resident (mmap'd shards), so the
ids must land on the host before the CPU gathers rows; the comment above it
explains that a blocking copy is required for correctness, and the H2D of the
result stays async. That makes the op structurally uncapturable.

The fork already knows this. `vllm/config/vllm.py` appends
`"vllm::qwen4_exp_amd_ple_ngram_embedding"` to `splitting_ops` for Qwen4Exp
models and downgrades FULL capture to PIECEWISE with a warning. The guard was:

```python
getattr(self.model_config.hf_config, "model_type", None) == "qwen4_exp"
```

Whittle's `config.json` says **`qwen4_exp_text`**, so the guard was False, the op
was never split, and the capture swallowed a host round-trip. The fork's own
engram table (`vllm/config/engram.py`) avoids this trap by keying on the
architecture `Qwen4ExpForCausalLM`; this line keyed on a per-checkpoint spelling.

The fix landed as commit `e3b6542393` on branch **`gfx906/wht1-cudagraph-guard`**,
written by the terminal agent that picked this thread up concurrently (same root cause
found independently, same 13:16 minute). Its implementation is the better one: a reusable
`is_qwen4_exp_config()` helper in `vllm/models/qwen4_exp/config.py`, consulted for both
`hf_config` and `hf_text_config`, so a multimodal wrapper config is matched too. The
earlier draft of this document described a narrower fix (architecture prefix first,
`model_type` prefix as fallback) that never landed — this measurement was taken against
the working tree while that draft was in place, and the two are semantically equivalent
for this checkpoint, but the helper is what is in the branch.

Regression cover on that branch: `tests/config/test_qwen4_exp_family_match.py` (the
helper) and `tests/config/test_qwen4_exp_ple_splitting_ops.py` (this arm's, asserting the
op lands in `splitting_ops` and does not accumulate). The two overlap — worth keeping one
knowingly rather than by accident.

## The three failures on the way there, in order

1. **Default compile** — `ConstraintViolationError: Constraints violated
   (L['query_start_loc'].size()[0])` in `determine_available_memory`. Not the
   capture ladder: vLLM's own default power-of-two ladder fails identically.
   `--enforce-eager` sidestepped it at the cost of graphs.
2. **`-cc.dynamic_shapes_config.type=unbacked`** (the remedy documented in
   `docs/design/debug_vllm_compile.md` for exactly this error) got past the
   profiler and died in the GDN layer instead: `Could not guard on data-dependent
   expression Ne(128*u0, 0)`, at
   `qwen_gdn_linear_attn.py:956` — `z.reshape(z.size(0), -1, self.head_v_dim)`.
   A `-1` reshape against a symbolic extent forces dynamo to reason about the
   shape product. **Not applied**: see "ruled out" below.
3. **`-cc.dynamic_shapes_config.type=backed_size_oblivious`** cleared dynamo and
   the profiler both, and reached capture — where it hit the `model_type` bug.
   With the bug fixed, this is the mode that serves.

## Ruled out

| Hypothesis | Verdict |
| --- | --- |
| Capture ladder too small | No — the default ladder reproduces failure 1 exactly |
| The GDN `-1` reshape must be rewritten first | Not needed for this arm — it is a latent dynamo hazard in *unbacked* mode only; `backed_size_oblivious` never reaches it. Worth fixing on its own merits if the unbacked mode is ever wanted. |
| A capture-unsafe op in the attention/GDN kernels | No — the abort names `ple_layer.py:1237`, the host-read PLE lookup |
| The PLE "16 of 16 ids were outside the table; folded onto row 0" warning is a serving defect | No — 2 occurrences, both during `Graph capturing finished`, none during inference; the same warning appears in the eager run. It is the capture-time dummy batch over a pinned buffer that is not written yet, exactly as the message says. `VLLM_GFX906_PLE_STRICT=1` turns the fold into a raise if it ever needs proving again. |
| The fix changes numerics | No — the output text is byte-identical to the eager run on both probes; splitting only moves capture boundaries, and the regression test pins the config-level effect |

## Open

- **Is `backed_size_oblivious` still required now that the split is applied?**
  The winning combination has two changes; which one carries failure 1 is
  untested. A cheap bisect (split fix + default dynamic shapes) would say.
- **Envelope arm**: largest `max_model_len`/MBT at util 0.85, TP=2. Note the
  graph pool now reserves VRAM up front, so re-measure the KV budget alongside.
- **MTP is absent from the checkpoint** (`mtp_num_hidden_layers: 1` with zero MTP
  tensors in the index), so there is no speculative-decode multiplier to add; the
  remaining throughput work is the envelope and batch shape, not a drafter.
- Quality gates once the envelope settles. Reported to the uploader: the
  root-packed PLE table and the MTP config/checkpoint inconsistency.
