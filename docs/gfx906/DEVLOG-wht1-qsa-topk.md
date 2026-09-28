# WHT-1 / QSA — the selection-width gate: upstream's `512 or 2048` rule, and what the kernel really takes

Ticket: [#36](../../issues/36) · Branch: `gfx906/wht1-qsa-topk` · Status **2026-09-28:
fix complete and unit-verified; the model-level exercise waits on the bf16 base.**

## The blocker this removes

`Qwen4ExpConfig._validate_qsa_config` refused to construct a config unless
`indexer_budget // indexer_compress_ratio ∈ {512, 2048}`. The published
Whittle-Qwen-3.8-35B-A3B sets `indexer_budget 262144` with `indexer_compress_ratio 4`,
i.e. a selection width of **65536** -- every block its 262144-token context can hold, so
the "sparse" selection is a no-op by construction:

```
ValueError: QSA requires indexer_budget / indexer_compress_ratio to be 512 or 2048, got 65536
```

That fired for the GGUF arm *and* would have fired for the bf16 arm, so it blocked every
route to running the model.

## What the rule actually is

- It is **upstream's**, not ours: `vllm/models/qwen4_exp/config.py` on vllm master carries
  the same `block_topk not in (512, 2048)` rejection. It describes the widths upstream's
  *tuned* top-k kernels are exercised at (the released Qwen3.8-Flash-Next family uses
  2048), not a geometry invariant.
- The ROCm decode kernel takes the width as a **runtime** argument:
  `csrc/libtorch_stable/sampler.cu:810` `top_k_per_row_decode(..., int64_t topK)`, with no
  template or check on `topK`. The templated `k ∈ {512,1024,2048}` sets live in the
  CUDA-only `persistent_topk`/`cooperative_topk` kernels, which this path does not use.

So the config check was strictly stronger than the runtime it was protecting. That said,
the kernel is **not** safe at arbitrary widths -- see the ceiling below.

## Measured: where the kernel stops being usable (gfx906, 2026-09-28)

Probe `/local/tmp/wht1/topk_case.py`, one process per case, random fp32 logits, all blocks
visible, kernel output vs `_reference_block_ranks` compared as sets per row:

| columns | rows | width | result |
|---|---|---|---|
| 1024 | 4 | 4 | AGREE |
| 65536 | 4 | 512 | AGREE |
| 65536 | 4 | 2048 | AGREE |
| 65536 | 4 | 4096 | AGREE |
| 65536 | 4 | 8192 | AGREE |
| 65536 | 32 | 8192 | AGREE |
| 8192 | 4 | 8192 | AGREE |
| 65536 | 4 | 12288 | **KERNEL-FAILS** `illegal memory access` |
| 65536 | 4 | 16384 | **KERNEL-FAILS** `illegal memory access` |
| 65536 | 4/32 | 32768, 65536, 131072 | process dies inside the kernel |

The failure mode is a **memory-safety violation**, not a clean limit: the ceiling is
"verified working" (8192, three configurations), not "documented maximum". Worth reporting
upstream (see below).

**Method lesson.** The first version of the probe ran all cases in one process and blamed
my own reference code for the crash: a HIP error is *sticky*, so the kernel's
`illegal memory access` surfaced at the next unrelated tensor op. One case per process
turned a false attribution into a clean boundary.

## The fix

Two files.

1. `vllm/models/qwen4_exp/config.py` -- the width allowlist is gone; the invariants that
   the operators really require stay (all fields present and positive,
   `indexer_kv_heads == 1`, budget divisible by the compression ratio, `indexer_head_dim`
   covering the rotary dimension) plus a positive-width check. New:
   `_QSA_TUNED_TOPK_WIDTHS`, `qsa_uses_tuned_topk_width()`, and the properties
   `qsa_block_topk` / `qsa_uses_tuned_topk`, so callers can see the width instead of
   re-deriving it.
2. `vllm/models/qwen4_exp/amd/ops/qsa.py` -- the selection call site is now
   `_select_qsa_block_ranks()`: `top_k_per_row_decode` for widths up to
   `_TOPK_KERNEL_MAX_WIDTH` (8192, measured above, which also covers the 512/1024/2048 the
   config used to allow -- no behaviour change for any width that was servable before),
   and `_reference_block_ranks()` above it, with a `warning_once`. The reference mirrors
   the `torch` backend of `vllm/model_executor/layers/indexer_topk.py`: mask columns at or
   past each row's visible end, take the top `block_topk`, -1-fill the rest, clamping a
   width wider than the cache -- the same output contract `top_k_per_row_decode` honours,
   which the probe above verifies by agreement at every width the kernel serves.

## Verification

- **The real model now loads through vLLM's own config path**
  (`get_config('/local/models/whittle-serve')` -> `Qwen4ExpTextConfig`, `block_topk`
  65536, `qsa_uses_tuned_topk` False, 40 layers, ctx 262144, `ple_layer_ids [2]`), where
  it previously raised the `ValueError` above.
- `tests/models/qwen4_exp/test_config.py` **21 passed** (+5 new: accepted widths incl. the
  Whittle geometry, the absent-config case, and four broken geometries still rejected).
- `tests/models/qwen4_exp/test_qsa_amd.py` **26 passed** (+4 new: the reference path's
  output contract, cache-wider clamping, the empty-cache case, and the dispatch boundary at
  2/512/2048/8192 -> kernel vs 8193/65536 -> reference). The pre-existing
  `test_qsa_selection_uses_portable_topk_on_rocm` (width 2) stays green, which is the check
  that no previously-servable width changed path.
- Siblings: `test_qsa_pre_indexer.py`, `test_ple.py`, `test_hc_ops.py`,
  `test_qsa_reference.py` green (19 passed / 52 skipped alone). Known pre-existing wart:
  collecting `test_qsa_reference.py` together with `test_config.py` in one process raises
  `RuntimeError: Tried to r...` during collection -- reproduced on the pristine tree
  (stashed my four files), so it is not from this train.
- `ruff check` + `ruff format --check` clean over `vllm/models/qwen4_exp/` and
  `tests/models/qwen4_exp/`.

## What is NOT verified yet

- The reference path **at production scale**: `torch.topk` over `(rows, columns)` is a sort,
  so a prefill chunk with thousands of rows and 65536 columns is genuinely expensive. No
  timing claim is made here; decode rows are 1-few, where it is trivial. Gate for the
  follow-up: once the model runs, compare prefill ms/token against the llm.cpp reference and
  decide whether a Triton radix-select for wide widths is warranted.
- The model end-to-end, which needs the bf16 base (`WHT-1` remains blocked on that).

## Upstream finding worth reporting

`_C.top_k_per_row_decode` corrupts memory (illegal memory access) for selection widths
above 8192 -- reachable by any config whose indexer budget exceeds ~8k blocks, which is
exactly what upstream's config check silently prevents. Filed as **UPR-3** with this
evidence; the reference path here is the local workaround.

## Tree state

`gfx906/wht1-qsa-topk` off `gfx906/v0.30.0`: `vllm/models/qwen4_exp/config.py`,
`vllm/models/qwen4_exp/amd/ops/qsa.py`, `tests/models/qwen4_exp/test_config.py`,
`tests/models/qwen4_exp/test_qsa_amd.py`. No probe code in-tree
(`/local/tmp/wht1/topk_case.py` is scratch). Not merged to the line yet: the fix is
unit-verified but has not been exercised by a real model load on the box, and that is the
natural first end-to-end check once the bf16 weights land.
