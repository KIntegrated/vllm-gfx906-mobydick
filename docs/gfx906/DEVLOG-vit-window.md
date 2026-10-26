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

## Follow-up: the accuracy question, answered (FA-VIT-ACC / #72)

**VERDICT:** not a defect — the CUSTOM ViT path **quantises K to q8_0**
(`gfx906_fa_mm_encoder.py:269`), so ~1e-2 absolute error is the design cost of
VIT-1's speedup, and it is 20-25x the fp16 backends' error. Recorded per backend
in the test that was silently comparing all three to a 1e-2 bound.

Fresh-process A/B, fp16, fp32 reference (the selector caches per
(head, dtype, device), which is why patching `MultiModalConfig` in-process had no
effect and why each arm needs its own process):

| shape (lens, Hkv, D) | CUSTOM | FLASH_ATTN | TORCH_SDPA |
|---|---|---|---|
| [112,112,112], 16, 64 | **0.0185** | 0.00048 | 0.00048 |
| [112,112,112], 1, 64 | **0.0114** | 0.00045 | 0.00045 |
| [112,112,112], 1, 256 | **0.0128** | 0.00048 | 0.00048 |
| [128,96], 1, 96 | **0.0092** | 0.00040 | 0.00040 |
| [2,3,4], 1, 64 | **0.0255** | 0.0011 | 0.0010 |

So: 20-25x, uniform across shapes, and **independent of segment split** — the
error is the K quantisation, not the varlen/window handling (which #71 cleared).

Two decisions fell out:

1. **Per-backend tolerance in the test**, with the numbers in a comment
   (`atol/rtol=3e-2` for CUSTOM, `1e-2` otherwise). A single global bound was
   asserting an fp16-exactness property that a q8-K kernel cannot have — the
   right gate for VIT-1 is end-to-end VLM output parity on real models, which is
   what it shipped on.
2. **The escape hatch is already there**: `--mm-encoder-attn-backend FLASH_ATTN`
   selects an fp16 ViT backend for accuracy-sensitive towers. Documented in the
   README decision table.

Also fixed while here: `test_mha_attn_platform[hip]` asserted FLASH_ATTN on ROCm,
which gfx906 legitimately violates (it selects CUSTOM); the expectation is now
platform-aware. `tests/kernels/attention/test_mha_attn.py` went from **20 errors
+ 5 failures** to **36 passed / 16 skipped**.

Not established: whether ~2.6e-2 absolute ViT error is acceptable for towers we
have not served. The bound is measured and the opt-out is named; a per-model gate
would need that model in the zoo.

## Can a user mix them? Yes — verified

`--mm-encoder-attn-backend` and `--attention-backend` are read by different
selectors (`get_vit_attn_backend` vs the decoder selector), so the split is legal:

```
mm_override=None           -> mm_encoder_attn_backend=None                  ViT=CUSTOM      text_backend=None (default)
mm_override='FLASH_ATTN'   -> mm_encoder_attn_backend=FLASH_ATTN            ViT=FLASH_ATTN  text_backend=None (default)
```

and a real boot of Qwen3.5-4B with `--mm-encoder-attn-backend FLASH_ATTN`
(`--enforce-eager`, 2026-10-26) logged
`GFX906_FA_PACKED=1: packed KV layout (K q8 | V fp16)` — that line comes from the
text FA backend, so the decoder stayed on CUSTOM with the packed cell while the
tower was on fp16 flash-attn.

Documented in the README decision table (with the copy-paste command) and in
`running.md`. The reverse direction is likewise supported: pinning the decoder to
`TRITON_ATTN` + an int8 KV format leaves the ViT on CUSTOM unless the mm flag says
otherwise.
