# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The packed/alias layout decision must come from the pool geometry.

`self._packed` is captured when the attention impl is constructed. In a TP
worker the impls are built while loading weights, *before* the KV layout is
resolved, so that flag can lag the pool the engine allocated. Splitting a packed
pool along the fused axis then hands the Q8 gather a V width of
`cell - head_size` (392 - 256 = 136 elems for D=256), which the C++ entry point
rejects with "D must be multiple of 32" - the TP=2 packed failure measured
2026-10-19. The cell width is authoritative on every rank, so the layout is
derived from it instead.
"""

import pytest
import torch

from vllm.gfx906_fa.gfx906_fa_backend import (
    Gfx906FAImpl,
    _packed_cell_bytes,
    _pool_cell_bytes,
    _pool_cell_is_fused_alias,
    _pool_cell_is_packed,
)

HEAD = 256


def _pool(cell_elems: int, dtype=torch.float16) -> torch.Tensor:
    """A per-layer pool view: [num_blocks, Hkv, block_size, cell_elems]."""
    return torch.zeros(4, 1, 8, cell_elems, dtype=dtype)


def _impl(head_size: int = HEAD, packed_flag: bool = True) -> Gfx906FAImpl:
    obj = object.__new__(Gfx906FAImpl)
    obj.head_size = head_size
    obj.padded_head_size = head_size
    obj._packed = packed_flag
    obj._packed_layout_warned = False
    return obj


def test_packed_cell_is_784_bytes_for_d256():
    assert _packed_cell_bytes(HEAD) == 784


def test_geometry_identifies_the_packed_pool():
    assert _pool_cell_is_packed(_pool(392), HEAD)
    assert not _pool_cell_is_fused_alias(_pool(392), HEAD)


def test_geometry_identifies_the_fused_alias_pool():
    assert _pool_cell_is_fused_alias(_pool(2 * HEAD), HEAD)
    assert not _pool_cell_is_packed(_pool(2 * HEAD), HEAD)


@pytest.mark.parametrize("tp_kv_heads", [2, 1])
def test_layout_follows_the_pool_not_the_construction_flag(tp_kv_heads):
    """TP=1 keeps 2 KV heads per rank, TP>=2 leaves 1 - the decision is the same.

    This is the TP=2 case: the flag says alias (impl built too early) while the
    pool is packed. The pool must win.
    """
    impl = _impl(packed_flag=False)
    assert impl._use_packed_views(_pool(392))


def test_alias_pool_wins_even_when_the_flag_says_packed():
    impl = _impl(packed_flag=True)
    assert not impl._use_packed_views(_pool(2 * HEAD))


def test_unrecognised_cell_width_fails_loud():
    impl = _impl()
    with pytest.raises(AssertionError, match="unrecognised KV pool cell"):
        impl._use_packed_views(_pool(200))


@pytest.mark.parametrize("kv_heads_per_rank", [1, 2, 4, 8])
def test_layout_choice_is_independent_of_tp(kv_heads_per_rank):
    """TP only changes KV heads per rank; the choice must not depend on it.

    kv_heads_per_rank covers TP=1 (2 heads for a 2-KV-head model), TP>=2 (1 after
    sharding or replication), and wider models. No TP arithmetic may enter the
    layout decision, which is why TP=4/TP=8 need no separate code path - only
    this invariant.
    """
    pool = torch.zeros(4, kv_heads_per_rank, 8, 392, dtype=torch.float16)
    impl = _impl(packed_flag=False)
    assert impl._use_packed_views(pool)
    assert _pool_cell_bytes(pool) == _packed_cell_bytes(HEAD)
