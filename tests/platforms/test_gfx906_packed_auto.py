# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The gfx906 packed-KV `auto` decision must key on predicted capacity.

The hybrid aligner re-derives the scheduler block size from the (smaller) packed
attention page, so "the block size moved" says nothing about whether the layout
helps. Capacity is ``block_size / page_bytes``: the mamba state is allocated per
block, so a smaller attention cell packs more tokens into the same mamba state.

Geometry below is measured (``docs/gfx906/_hybrid_kv_ratio.py``, which
reproduces the boot logs of the LAB-9 A/B).
"""

import pytest

from vllm.platforms.rocm import RocmPlatform

gain = RocmPlatform._gfx906_packed_capacity_gain


def test_capacity_gain_accepts_a_model_where_packing_moves_the_block_size():
    """Qwen3.5-35B-A3B: 1056 -> 1376 tokens/block while the page is unchanged.

    The served A/B measured +22.2 % KV tokens here, so this is the case that the
    old "block size moved -> keep the alias layout" rule got wrong.
    """
    off = (1056, 2_162_688)
    on = (1376, 2_157_568)
    assert gain(off, on) == pytest.approx(0.306, abs=0.002)
    assert gain(off, on) >= RocmPlatform._GFX906_PACKED_MIN_GAIN


def test_capacity_gain_of_the_capped_lightning_geometry():
    """Nemotron-3.5-Lightning-30B-A3B: 2096 -> 2672 tokens/block.

    Prediction (~+28 %) and the recorded served datum (+0.4 %) disagree because
    that arm was capped by request capacity, not by VRAM — the decision follows
    the prediction and the escape hatch is an explicit ``off``.
    """
    off = (2096, 2_146_304)
    on = (2672, 2_137_600)
    assert gain(off, on) == pytest.approx(0.280, abs=0.002)


def test_capacity_gain_rejects_a_layout_that_wastes_the_page():
    """A packed cell that also grows the block's page is a net loss."""
    off = (16, 131_072)
    on = (16, 147_456)  # same block count, 12.5 % bigger page
    assert gain(off, on) < 0
    assert gain(off, on) < RocmPlatform._GFX906_PACKED_MIN_GAIN


def test_capacity_gain_is_zero_without_a_unified_page():
    """No page (aligner returned early) must not enable the layout."""
    assert gain((16, None), (16, None)) == 0.0
    assert gain((16, 131_072), (16, None)) == 0.0
    assert gain((16, None), (16, 131_072)) == 0.0
