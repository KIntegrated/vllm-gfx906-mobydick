# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Custom-FA path for the Qwen3.5-family ViT (VIT-1).

The Qwen3.5/3.8 vision tower runs **bidirectional, cache-free, ragged**
(`cu_seqlens`) fp16 attention with `head_dim = 72` (16 heads, hidden 1152),
prefill-only, on the critical path of every image-bearing prompt. Upstream
serves it through flash-attn; on gfx906 that is the Triton-AMD path, which costs
a per-boot JIT compile + graph-capture stall and is not MI50-tuned.

Why the dense entry fits: `gfx906_fa.forward` is the non-paged launcher (plain
contiguous `K`/`V`, no block table) and the kernel masks *either* via a
materialised mask *or* the inline-causal `q_abs_offset` — with **neither** it
computes **full bidirectional attention**, which is exactly the ViT case.

Why head_dim is padded: the launcher dispatches on
`head_dim in {64, 128, 256}` and requires `head_size % 32 == 0`, so 72 is
zero-padded to **128**. The padding is exact:

* padded `Q` dims contribute 0 to the QK dot,
* padded `K` dims quantise to zero q8_0 blocks (0 contribution),
* padded `V` dims contribute 0 to P·V,
* the padding is in the **head** dim, so the softmax denominator is unchanged
  (the per-sequence KV bound is `kv_max`, not a shorter pad).

Ragged handling: `MMEncoderAttention` passes `[B, S, H, D]` tensors padded to a
common `S` plus `cu_seqlens`; we bound the KV scan with `kv_max = [B]`, so the
padded KV rows are never read, and padded *query* rows only produce output we
discard (query rows are independent by construction).
"""

from __future__ import annotations

import os

import torch

from vllm import _gfx906_fa_C as gfx906_fa


def _pad_head_dim(head_size: int) -> int | None:
    """Smallest instantiated kernel head dim that fits (None if none does)."""
    for hd in (64, 128, 256):
        if head_size <= hd:
            return hd
    return None


def vit_enabled() -> bool:
    """Kill switch: `GFX906_FA_VIT=0` restores the upstream flash-attn ViT path."""
    return os.environ.get("GFX906_FA_VIT", "1") == "1"


def vit_auto_enabled() -> bool:
    """Whether the ViT path is selected *automatically* on gfx906.

    Off by default until the ViT screens pass (VIT-1): an explicit
    `--mm-encoder-attn-backend custom` already works, this only gates the
    default flip. `GFX906_FA_VIT=0` overrides everything.
    """
    return vit_enabled() and os.environ.get("GFX906_FA_VIT_AUTO", "0") == "1"


def vit_supported(head_size: int, dtype: torch.dtype) -> bool:
    if not vit_enabled():
        return False
    if dtype not in (torch.float16, torch.float32):
        return False
    return _pad_head_dim(head_size) is not None


def forward_vit(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: torch.Tensor | None,
    scale: float,
    head_size: int,
) -> torch.Tensor:
    """Bidirectional ragged attention through the gfx906 dense FA entry.

    Args:
        query/key/value: `[B, S, H|Hkv, D]` fp16 (upstream ViT layout).
        cu_seqlens: int32 `[B+1]` real token counts per batch item (S is the
            padded common length), or None for one batch of `B * S` tokens.
        max_seqlen: int32 `[1]` upstream FA convention (informational here).
        scale: the model's own softmax scale, computed for the real head size.
        head_size: real (unpadded) head dim.

    Returns:
        `[B, S, H, D]` in `query`'s dtype.
    """
    assert query.dtype == torch.float16, "ViT FA expects fp16 Q/K/V"
    b, sq, heads, d_in = query.shape
    sk = key.shape[1]
    hkv = key.shape[2]
    assert d_in == head_size, (d_in, head_size)
    assert value.shape == (b, sk, hkv, head_size)

    pad = _pad_head_dim(head_size)
    assert pad is not None, f"head_size {head_size} is not pad-able"

    if cu_seqlens is None:
        kv_max = torch.full((b,), sk, dtype=torch.int32, device=query.device)
    else:
        cu = cu_seqlens.to(device=query.device, dtype=torch.int32)
        assert cu.numel() == b + 1, (cu.numel(), b)
        kv_max = (cu[1:] - cu[:-1]).contiguous()

    def to_kernel(x: torch.Tensor) -> torch.Tensor:
        # [B, S, H, D] -> [B, H, S, D], zero-padded to the kernel head dim.
        x = x.transpose(1, 2)
        if pad == head_size:
            return x.contiguous()
        out = x.new_zeros((x.shape[0], x.shape[1], x.shape[2], pad))
        out[..., :head_size] = x
        return out

    q = to_kernel(query).float().contiguous()
    v = to_kernel(value).contiguous()
    k_q8 = gfx906_fa.quantize_q8_0(to_kernel(key).contiguous())

    # mask=None + q_abs_offset=None => full bidirectional attention; kv_max keeps
    # the scan inside the real KV rows. NOTE: `forward` takes Q as [B, H, S, D]
    # but returns the BSHD-native output [B, S, H, D] (the layout the LLM path
    # consumes), i.e. no output transpose is needed.
    out = gfx906_fa.forward(q, k_q8, v, float(scale), kv_max)
    return out[..., :head_size].to(query.dtype).contiguous()
