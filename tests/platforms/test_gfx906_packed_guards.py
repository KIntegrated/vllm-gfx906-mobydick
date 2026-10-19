# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The shapes the packed KV cell cannot express fall back instead of failing a boot.

`gfx906_fa_packed=auto` is the default, so these guards are what makes the default
safe: under `auto` an inexpressible shape selects the alias layout with a logged
reason, while an explicit `on` raises (fail loud - the user asked for it).

The mode is driven through `CacheConfig` (`set_packed_config_mode`), not the env
var: `GFX906_FA_PACKED` only understands 0/1, so setting it to `auto` selects off
and the packed branch never runs (`_packed_choice`).
"""

import pytest
import torch

from vllm.gfx906_fa import gfx906_fa_backend as fa
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _spec(**kw):
    fields = dict(block_size=16, num_kv_heads=2, head_size=128, dtype=torch.float16)
    fields.update(kw)
    return FullAttentionSpec(**fields)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("GFX906_FA_PACKED", raising=False)
    fa.reset_packed_mode()
    fa.set_packed_dry_run(False)
    fa.set_packed_config_mode("auto")
    yield
    fa.reset_packed_mode()
    fa.set_packed_config_mode("off")


def test_guard_asymmetric_kv_head_dims_falls_back():
    """The packed cell assumes head_size_v == head_size (one padded half each)."""
    fa.Gfx906FABackend.customize_spec(_spec(head_size_v=64))
    assert fa._PACKED_MODE == "off"


def test_guard_head_padding_falls_back():
    """A head dim the alias path would pad (72 -> 96) cannot be a packed cell."""
    fa.Gfx906FABackend.customize_spec(_spec(head_size=72))
    assert fa._PACKED_MODE == "off"


def test_guard_padded_pages_falls_back():
    """skip-quant reconciles pages via page_size_padded; the packed cell cannot."""
    fa.Gfx906FABackend.customize_spec(_spec(page_size_padded=4096))
    assert fa._PACKED_MODE == "off"


def test_guard_explicit_on_raises_instead_of_falling_back(monkeypatch):
    """An explicit `on` must fail loud rather than quietly serving alias."""
    monkeypatch.delenv("GFX906_FA_PACKED", raising=False)
    fa.reset_packed_mode()
    fa.set_packed_config_mode("on")
    with pytest.raises(NotImplementedError):
        fa.Gfx906FABackend.customize_spec(_spec(head_size_v=64))
