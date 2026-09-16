# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Head-dim padding helpers for the gfx906 FA text path (FA-COVER-1 step 2).

Pins the padding map and the gap: text head dims outside {64,128,256} are still
rejected, while the ViT path already pads them. When the write path learns to pad,
the `supports_head_size` assertion below is the line to update.
"""

import pytest
import torch

try:
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FABackend,
        _pad_head_dim,
        _padded_head_size,
    )
except ImportError as exc:  # pragma: no cover - needs the built gfx906 extension
    pytest.skip(f"gfx906 FA backend unavailable: {exc}", allow_module_level=True)


@pytest.mark.parametrize(
    "head_size,want",
    [
        (32, 64),
        (64, 64),
        (72, 128),
        (80, 128),
        (96, 128),
        (112, 128),
        (128, 128),
        (160, 256),
        (256, 256),
        (288, None),
        (1024, None),
    ],
)
def test_pad_map(head_size, want):
    assert _pad_head_dim(head_size) == want


def test_padding_default_is_on_after_the_phi3_gate(monkeypatch):
    """Default ON since Phi-3: 96 -> CUSTOM at 36.41 vs 28.62 t/s."""
    monkeypatch.delenv("GFX906_FA_PAD", raising=False)
    assert _padded_head_size(128) == 128  # instantiated dims are unaffected
    assert _padded_head_size(96) == 128


def test_kill_switch_restores_exact_dims_only(monkeypatch):
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert _padded_head_size(128) == 128
    assert _padded_head_size(96) is None
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    assert _padded_head_size(96) == 128
    assert _padded_head_size(288) is None


def test_supports_head_size_serves_pad_able_dims_by_default(monkeypatch):
    """Pad-able dims are servable now; dims past 256 still are not."""
    monkeypatch.delenv("GFX906_FA_PAD", raising=False)
    # pad-able: instantiated, below 64 (32 -> 64), and 65..256 (96 -> 128)
    for supported in (32, 40, 64, 72, 80, 96, 112, 128, 160, 256):
        assert Gfx906FABackend.supports_head_size(supported), supported
    # only dims beyond the largest instantiated kernel dim are out of reach
    for unsupported in (257, 288, 512):
        assert not Gfx906FABackend.supports_head_size(unsupported), unsupported
    # kill switch: back to the instantiated dims only
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert Gfx906FABackend.supports_head_size(128)
    assert not Gfx906FABackend.supports_head_size(96)

def test_customize_spec_widens_both_halves(monkeypatch):
    """The spec must widen BOTH halves (this was Phi-3's 224-row bug)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=32, head_size=96, head_size_v=96,
        dtype=torch.float16, kv_quant_mode=None,
    )
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    widened = Gfx906FABackend.customize_spec(spec)
    assert widened.head_size == 128
    # The fused row is head_size + head_size_v: widening only one leaves
    # padded+real (128 + 96 = 224), the third width the allocator built on Phi-3.
    assert widened.head_size_v == 128
    # other fields untouched
    assert widened.block_size == 16 and widened.num_kv_heads == 32
    # kill switch: unchanged
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert Gfx906FABackend.customize_spec(spec).head_size == 96
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    # a dim that cannot be padded is left alone even when opted in
    big = FullAttentionSpec(
        block_size=16, num_kv_heads=32, head_size=512, dtype=torch.float16,
        kv_quant_mode=None,
    )
    assert Gfx906FABackend.customize_spec(big).head_size == 512

@pytest.mark.parametrize("real_d", [72, 80, 96, 112])
def test_spec_and_shape_agree_on_the_padded_row(real_d, monkeypatch):
    """The spec's page size and the declared shape must describe the same row.

    The invariant the Phi-3 gate broke: vLLM sizes a page from the spec
    (block * num_kv_heads * (head_size + head_size_v) * dtype) and builds the tensor
    get_kv_cache_shape, so a mismatch makes the allocator invent a third row width.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    monkeypatch.setenv("GFX906_FA_PAD", "1")
    spec = Gfx906FABackend.customize_spec(
        FullAttentionSpec(
            block_size=16, num_kv_heads=32, head_size=real_d, head_size_v=real_d,
            dtype=torch.float16, kv_quant_mode=None,
        )
    )
    shape = Gfx906FABackend.get_kv_cache_shape(
        2, spec.block_size, spec.num_kv_heads, real_d
    )
    row_elements = shape[1] * shape[-1]  # dim 1 pairs K/V; the last is the padded row
    assert row_elements == spec.head_size + spec.head_size_v, (shape, spec)
    assert (
        spec.page_size_bytes
        == spec.block_size * spec.num_kv_heads * row_elements * 2
    )
