# WHT-1: the PLE n-gram lookup must be split out of the captured graph

Date: 2026-10-05. Item: **#36 WHT-1**. Branch: `gfx906/wht1-cudagraph-guard`
(off `gfx906/v0.30.0` @ `eeed25a9a6`). Follows
`DEVLOG-wht1-whittle-load.md`, whose graph-capture arm this closes.

## Result

Graph capture works for Whittle-Qwen-3.8-35B-A3B on 2× MI50, and it is worth
**≈3.8× on a short probe** (26-token sample, so read it as a signpost, not a
number) — at the cost of **KV cache 2.0 GiB → 1.07 GiB**, which is the price of
the graph pool. The blocker was not a kernel: it was a **string mismatch** in the
guard that keeps a host-side lookup out of the captured region.

| | eager (control) | with capture |
|---|---|---|
| engine init | ok | ok, after the guard fires |
| capture | `--enforce-eager` | **PIECEWISE 3/3 in 27 s, 0.66 GiB** (2nd round 4 s / 0.26 GiB) |
| throughput (26 tok, rep2) | 5.61 t/s | **21.15 t/s** |
| coherence | `4, 2+2+2=6, 2+2+2+2=8. So 4.` | same string |
| KV cache | 2.0 GiB (7.2× @16k) | **1.07 GiB** (~3.8× @16k) |

Repro: `/local/tmp/wht1/smoke_load2.sh`, logs `/local/tmp/wht1/smoke2/`.
Flags are the load devlog's, minus `--enforce-eager`, plus
`-cc.dynamic_shapes_config.type=backed_size_oblivious` (the only dynamic-shapes
mode that reaches capture — see the ladder below).

## Symptom

Engine init died in the memory-profiling pass with

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
Search for `hipErrorStreamCaptureUnsupported' ...
```

at `Capturing CUDA graphs (PIECEWISE): 0/3` — i.e. inside the first captured
piece, after `Initial profiling/warmup run took 304.64 s`.

The three dynamic-shapes modes each die differently, which is what made this look
like a capture-ladder problem:

| mode | failure |
|---|---|
| default (`backed`) | `ConstraintViolationError: Constraints violated (L['query_start_loc'].size()[0])` in `determine_available_memory` |
| `unbacked` | `Could not guard on data-dependent expression Ne(128*u0, 0)` (the GDN layer) |
| `backed_size_oblivious` | reaches capture, then `hipErrorStreamCaptureUnsupported` |

## Triage

The engine-level `RuntimeError` says nothing about *which* op. The **innermost
Python frame of the worker traceback** does:

```
File .../inductor_cache/si/csijgypgq67kcbsmrz7w2p4fvlsep4xt3omhwpsuvyblv6iml4cr.py:3356, in call
  torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding.default(buf31, buf32, 'model.layers.1.ple')
File vllm/models/qwen4_exp/amd/ple_layer.py:1237, in qwen4_exp_amd_ple_ngram_embedding
  pinned_ids.copy_(ngram_ids, non_blocking=False)
```

`vllm::qwen4_exp_amd_ple_ngram_embedding` (registered at
`amd/ple_layer.py:1271`) gathers over the **host-resident** n-gram table: the
`non_blocking=False` D2H into a pinned buffer is deliberate — the comment above
it records that the async form let the host read stale/uninitialised ids
("index out of range in self" engine kills at 00:26 / 00:31). A blocking D2H is
exactly what HIP refuses inside a capture. The op can therefore never live inside
a captured region; it has to run in an eager region between captured pieces —
i.e. it must be a splitting op.

## Root cause

`vllm/config/vllm.py` already had that guard (append the op to `splitting_ops`,
and downgrade `cudagraph_mode` when full capture would wrap the whole forward),
but it matched `hf_config.model_type == "qwen4_exp"`. This checkpoint declares
`model_type: "qwen4_exp_text"` (`architectures: ["Qwen4ExpForCausalLM"]`), so the
block was skipped — and the failure matched its own comment ("Downgrade loudly
rather than let the engine die with hipErrorStreamCaptureUnsupported") exactly.

Three independent observables from the failed run:

| expected if the guard had fired | observed in the failed run |
|---|---|
| `splitting_ops` contains `…ple_ngram_embedding` | absent (it does contain `qwen4_exp_ple_short_conv`) |
| `cudagraph_mode` downgraded to `PIECEWISE` | stayed `FULL_AND_PIECEWISE (2,1)` |
| the downgrade warning is logged | absent |

The fork already knew both spellings elsewhere (`config/speculative.py` matched
`{"qwen4_exp", "qwen4_exp_text"}`; `transformers_utils/config.py` maps both), so
this was drift between three hand-maintained lists, not a misunderstanding.

## Ruled out (before the traceback named the op)

| hypothesis | evidence | verdict |
|---|---|---|
| Host sync in the QSA path | the only `.item()` in `qwen4_exp/` is `common/qsa_cache.py:684`, and it reads **CPU** tensors (`query_start_loc_cpu`) — no device sync | no |
| Cross-stream / prefetch streams | no `Stream`/`Event`/`wait_stream` in `qwen4_exp/amd/**` (the AMD path; the `nvidia/ngram_embedding.py` prefetch streams do not run on gfx906) | no |
| Capture-illegal HIP API in the fork's C++ | the `cudaMalloc`/`cudaMemcpy`/`hipMalloc` hits are in `custom_all_reduce`/`quickreduce`, both out of play (`--disable-custom-all-reduce`) | no |
| EP/MoE all-to-all host syncs | the `fused_moe` `.item()`/`.tolist()` hits are init/`expert_map` or `routed_experts_capturer` (off by default) | no |
| Unwarmed Triton JIT inside capture | **not** the cause here, but a real gap: `model_executor/warmup/qwen4_exp_qsa_warmup.py` looks up `vllm.models.qwen4_exp.**nvidia**.*` in `sys.modules` and returns early on AMD, so the QSA/indexer warmup never runs on this box. Open item — see Next. | open |

## Fix

Match the family, not one spelling, in **one** place —
`vllm/models/qwen4_exp/config.py`:

- `QWEN4_EXP_MODEL_TYPES` is **derived from the config classes** that declare a
  `model_type` (`qwen4_exp`, `qwen4_exp_text`, `qwen4_exp`), plus the drafter's
  rewrite `qwen4_exp_mtp`. A future config class extends the family automatically.
- `is_qwen4_exp_model_type(model_type)` — for bare strings.
- `is_qwen4_exp_config(hf_config)` — for a resolved HF config: `model_type` first,
  `architectures` (`Qwen4Exp*`) as the fallback.

Call sites: `vllm/config/vllm.py` (the compilation guard; checks the wrapper
config *and* `hf_text_config`) and `vllm/config/speculative.py` (the MTP rewrite).
Both used to carry their own literal.

Tests (CPU-only):

- `tests/config/test_qwen4_exp_ple_splitting_ops.py` — the guard's contract: the
  PLE op lands in `splitting_ops` for **both** spellings, and repeated
  `VllmConfig` builds do not duplicate the entry.
- `tests/config/test_qwen4_exp_family_match.py` — the predicate: every declared
  spelling accepted (asserted against the classes, so a new class must be
  covered), `qwen4_exp_mtp` accepted, `qwen3_5` / `qwen4_exp_extra` / `""` /
  non-strings rejected, architecture fallback, missing-`model_type` tolerance.

`ruff check` clean; `import vllm.config.vllm` unchanged at ~13.5 s (the arch
config module pulls 5 light modules and does not import `vllm.config`, so there
is no cycle).

## Verification (2026-10-05, 13:01–13:14)

```
13:01:42 WARNING [vllm.py:1937] Qwen4Exp's host-resident PLE n-gram lookup cannot run inside a
                           full cudagraph (blocking D2H mid-graph); downgrading cudagraph_mode to PIECEWISE.
13:12:42 Graph capturing finished in 27 secs, took 0.66 GiB     (PIECEWISE 3/3)
13:13:06 Graph capturing finished in 4 secs, took 0.26 GiB      (CUDA graph pool 0.26 GiB actual)
13:13:28 throughput 26 tok in 6.1s = 4.25 t/s (rep1, warmup); 26 tok in 1.2s = 21.15 t/s (rep2)
         coherence unchanged
13:13:36 teardown; post-load canary per protocol
```

## Next

1. **Proper throughput measurement.** 26-token probes are a signpost; the arm
   needs the same shape as the eager control's A/B (and the tester's TP=4 numbers
   are not comparable). Never quote these as kernel time.
2. **KV headroom is the new constraint.** Capture costs `2.0 → 1.07 GiB` of KV
   (7.2× → ~3.8× concurrency at 16k). The envelope arm (largest `max_model_len`
   /MBT at util 0.85) now has to be re-measured *with graphs on*, and MBT is the
   lever for the index buffer (`MBT × (token_topk + ratio − 1) × 4 B × 40`).
3. **AMD QSA/PLE warmup gap** (open item above): the warmup module is
   nvidia-only, so first-call JIT on this box is unwarmed. Confirm whether it
   costs measurable latency at first token, then either wire the AMD modules in
   or record it as harmless.
4. **Quality gates** once perf settles — as before, nothing in this change
   touches numerics: the width clamp is lossless, the name remap relocates
   weights, and the family match only decides which ops get split.
