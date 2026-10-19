# U30-2 — persistent top-k on gfx906's 64 KiB LDS: which path the build takes

**VERDICT:** `CENSUS DONE — the low-LDS fallback triggers on every call, by
construction; MEASUREMENT BLOCKED on a locally servable QSA model` ·
**GATE:** host-side branch census (no GPU run needed) ·
**Refs:** upstream `#54110`, `lds-layout.md`, `DEVLOG-wht1-qsa-topk.md`

## What the kernel family is

"Persistent top-k" in this tree is the **DeepSeek-V3 sparse-attention indexer**
top-k (`csrc/libtorch_stable/topk.cu`, `persistent_topk.cuh`, `sampled_topk.cuh`,
`filtered_topk.cuh`) — the same family as the `top_k_per_row_decode` op that
fails closed above width 8192 (UPR-3, `DEVLOG-wht1-qsa-topk.md`). Both are
LDS-constrained on gfx906 for the same reason: gfx906 has the smallest LDS of
the fork's targets.

## The census (from `launch_persistent_topk`'s gates)

Three paths, chosen from `device_prop->sharedMemPerBlockOptin`:

| gate | path | LDS requirement | gfx906 (64 KiB) |
|---|---|---|---|
| `rows > 64 && max_seq_len >= kMinSampledLength` **and optin >= 144 KiB** | `sampled_topk` | 128 KiB dynamic (`FILTERED_TOPK_SMEM_DYNAMIC`) + static fallback storage | **not taken** |
| `rows > 32` **and optin >= 128 KiB** | `FilteredTopKRaggedTransform` | 2 × 16K indices × 4 B = 128 KiB | **not taken** |
| else | `persistent_topk_kernel` (chunked, cooperative) | adapts: `min(optin, kSmemMedium)` for rows ≤ 4, `min(optin, 48 KiB)` for rows ≤ 8, else `optin` | **always taken** |

Constants: `kFixedSmemLarge` = 2080 B, `kSmemMedium` = 35,968 B, `kThreadsPerBlock`
= 1024, `kMinSampledLength` = 98,304 (TopK 512) / 65,536 (otherwise). Device
values read from PyTorch on this box: `shared_memory_per_block` 65,536,
`shared_memory_per_multiprocessor` 65,536, `shared_memory_per_block_optin` not
exposed by this torch build (None) — the C++ `sharedMemPerBlockOptin` is the same
64 KiB, so `optin >= 128 KiB` is false and the fallback is **unconditional**, not
data-dependent.

## What the fallback costs, structurally

`available_for_ordered = effective_max_smem - kFixedSmemLarge`, and a row of
`stride` columns is split into `ceil(stride / (available/4))` chunks, one CTA
group each:

| rows | effective LDS | max chunk elements | 65,536-column row |
|---|---|---|---|
| ≤ 4 | 35,968 B | 8,472 | **8 chunks** |
| ≤ 8 | 49,152 B | 11,768 | **6 chunks** |
| > 8 | 65,536 B | 15,864 | **5 chunks** |

A 128 KiB device (e.g. MI300) instead takes `FilteredTopK` for rows > 32 with a
16K-index buffer, and a 144 KiB device takes the single-pass `sampled_topk`. So
gfx906 pays a chunked cooperative scan (5–8 CTA groups per row at 64k columns)
where the big-LDS paths are single-pass — the structural cost is real and
device-inherent, not a bug.

## Why the measurement is blocked

The affected op is the **QSA indexer** (`Qwen3.8-Flash-Next` family). Those
checkpoints are not servable on this box, so "is the fallback's cost measurable
in a serving step" cannot be answered here. Do **not** substitute a synthetic
micro-benchmark and quote it as a model-level result — the indexer's time share
is what decides, and that needs the model.

**Reopen gate:** when a QSA checkpoint is servable locally, measure the indexer
top-k share of a 64k-context decode step (ms/step attribution, `mclk` 1000) and
only then decide whether to carry a gfx906-specialised path (e.g. a 48 KiB
single-pass variant reusing the `kSmemMedium` layout for larger row counts).

## Related

- UPR-3 (#37): the *other* indexer top-k op now fails closed above topK 8192
  instead of faulting, with the same root cause (`2*topK` int32 of dynamic LDS).
- `lds-layout.md` for gfx906's 64 KiB per-CU LDS budget and the layouts that fit.
