# FA-COVER-3 — is the ViT window silently ignored by CUSTOM? (#71)

**VERDICT:** no. The window is not a kernel argument anywhere in vLLM: the ViT
encodes it by **repacking the image into window-sized sequences and passing
different `cu_seqlens`** to windowed blocks. CUSTOM honours the segment
boundaries — window-sized splits show exactly the same error as any other split,
i.e. no cross-segment leakage. The issue's premise was a misreading of "the
selector takes no window argument".
**GATE:** numerical parity on multi-segment varlen at window-sized lengths.
**Refs:** #71, `vllm/model_executor/models/qwen2_5_vl.py:1108-1131`,
`DEVLOG-fa-coverage.md`.

## Why the premise was wrong

`MMEncoderAttention.__init__` has **no** window/sliding-window parameter — not
for our backend, for any backend. `get_vit_attn_backend(head_size, dtype)` cannot
pass one because the layer cannot receive one. Windowing is expressed one level
up: `Qwen2_5_VLVisionTransformer` reorders hidden states by `window_index` and
hands each block either `cu_seqlens_window` (windowed blocks) or `cu_seqlens`
(the layers listed in `fullatt_block_indexes`). The attention kernel therefore
never needs to know about windows — it needs to honour varlen segment boundaries,
which is the property that must be tested.

## The test

`MMEncoderAttention`, fp16, H=16, fp32 reference (`ref_attention` on
float-cast q/k/v), CUSTOM selected on every row:

| segment split | role | max abs diff |
|---|---|---|
| `[112, 112, 112]` | Qwen2.5-VL window size | 0.0114 |
| `[128, 96]` | uneven multi-image | 0.0116 |
| `[256, 128, 64]` | uneven, long | 0.0147 |
| `[16, 16]` | short | 0.0200 |
| `[2, 3, 4]` | very short (upstream test's stress case) | 0.0255 |

Window-sized segments are **not** special: 0.0114 against 0.0116 for an
arbitrary uneven split, and 0.0147 for a long one. If the kernel were attending
across segment boundaries the window row would be an outlier; it is not.

So the answer to #71 is "the window is honoured". What the table *does* show is a
separate accuracy characteristic — see below — and that is not a windowing bug.

## Two findings en route

**1. Fixed: `RocmPlatform` crashed on a `None` model config** (`rocm.py:1429`).
The packed-KV `auto` path called `model_config.is_hybrid` unguarded, so any
caller whose `VllmConfig` has no model config — the upstream ViT attention suite,
for one — died with `AttributeError: 'NoneType' object has no attribute
'is_hybrid'`. That turned 20 tests in
`tests/kernels/attention/test_mha_attn.py` from errors into real results: the
varlen subset now runs 12 passed / 4 failed / 16 skipped instead of 20 errors.

**2. Open: CUSTOM ViT varlen accuracy, ~1.1-1.9e-2 absolute** at fp16 against an
fp32 reference on O(1) values, worst at the very short `[2,3,4]` split (0.0255).
The upstream test's tolerance is `atol=1e-2` (and it applies no tolerance
relaxation for CUSTOM), which is why those cases fail. **This is not yet
attributable to our kernel**: the only non-CUSTOM fixture I could measure was
`FLASH_ATTN` at **bfloat16** (0.0025-0.0038), and a like-for-like fp16 override
did not take effect through `MultiModalConfig` once the selector had cached. Until
that A/B exists the honest statement is "CUSTOM fp16 varlen error is ~1e-2 at GQA
16; the comparison that would call it a defect is missing". Filed separately.
