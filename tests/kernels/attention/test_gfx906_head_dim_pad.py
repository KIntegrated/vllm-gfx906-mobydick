# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Head-dim padding helpers for the gfx906 FA text path (FA-COVER-1 step 2).

Pins the padding map and the gap: text head dims outside {64,128,256} are still
rejected, while the ViT path already pads them. When the write path learns to pad,
the `supports_head_size` assertion below is the line to update.
"""

import pytest

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


def test_padding_is_opt_in(monkeypatch):
    """Unset means off: the KV layout must carry the pad before a dim can be served."""
    monkeypatch.delenv("GFX906_FA_PAD", raising=False)
    assert _padded_head_size(128) == 128  # instantiated dims are unaffected
    assert _padded_head_size(96) is None  # not served until the write path lands


def test_kill_switch_restores_exact_dims_only(monkeypatch):
    monkeypatch.setenv("GFX906_FA_PAD", "0")
    assert _padded_head_size(128) == 128
    assert _padded_head_size(96) is None
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    assert _padded_head_size(96) == 128
    assert _padded_head_size(288) is None


def test_supports_head_size_is_still_restrictive(monkeypatch):
    """The gap this topic closes: 96 pads in the ViT path, not in the text path."""
    assert Gfx906FABackend.supports_head_size(64)
    assert Gfx906FABackend.supports_head_size(128)
    assert Gfx906FABackend.supports_head_size(256)
    for unsupported in (32, 72, 80, 96, 112, 160, 288):
        assert not Gfx906FABackend.supports_head_size(unsupported), unsupported
    # and the helpers describe what would be served once padding is opted in
    assert _padded_head_size(96) is None  # opt-in default
    monkeypatch.setenv("GFX906_FA_PAD", "1")
    assert _padded_head_size(96) == 128
