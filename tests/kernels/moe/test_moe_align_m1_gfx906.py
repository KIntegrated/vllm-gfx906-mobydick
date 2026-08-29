# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Tests for the gfx906 M=1 fused align+sort kernel (C1 stage 1).

Run `pytest tests/kernels/moe/test_moe_align_m1_gfx906.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.gfx906_w4a16_moe import (
    _moe_align_block_size_fused_m1,
    _use_fused_align_m1,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx906
else:

    def on_gfx906() -> bool:
        return False


def _has_op() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_align_block_size_m1_gfx906"
    )


def _prod_chain(topk_ids, dev):
    """Production two-kernel chain at the wrapper's buffer sizes."""
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids, 1, 256, None
    )
    assert sorted_ids.shape == (8,)
    assert expert_ids.shape == (8,)
    return sorted_ids, expert_ids, ntp


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906() and _has_op()),
    reason="gfx906 M=1 fused align kernel test (C1 stage 1)")
@pytest.mark.parametrize("seed", range(24))
def test_m1_fused_align_bit_equal_to_production(seed):
    """The single-CTA kernel must reproduce the two-kernel generic chain
    bit-for-bit (sorted_token_ids / expert_ids / num_tokens_post_pad) for
    random and tie-heavy topk_ids. Within-expert slot order is the
    issuing-lane order in both (single warp)."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    cases = [torch.randint(0, 256, (1, 8), generator=g, device=dev)]
    if seed % 4 == 0:
        # tie-heavy: 8 slots over 2 experts
        cases.append(
            torch.randint(0, 2, (1, 8), generator=g, device=dev).int())
    if seed % 4 == 1:
        # all one expert
        cases.append(torch.full((1, 8), seed % 256, device=dev,
                                dtype=torch.int32))
    if seed % 4 == 2:
        # 4 pairs
        cases.append(torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3]], device=dev, dtype=torch.int32))
    if seed % 4 == 3:
        # 8 distinct
        cases.append(
            torch.arange(8, device=dev, dtype=torch.int32).reshape(1, 8))

    for topk_ids in cases:
        topk_ids = topk_ids.int()
        ref_s, ref_e, ref_n = _prod_chain(topk_ids, dev)
        fus_s, fus_e, fus_n = _moe_align_block_size_fused_m1(topk_ids, 256)
        assert torch.equal(fus_n, ref_n), (
            f"ntp diverge: {fus_n.tolist()} vs {ref_n.tolist()}")
        assert torch.equal(fus_s, ref_s), (
            f"sorted diverge: {fus_s.tolist()} vs {ref_s.tolist()} "
            f"topk_ids={topk_ids.tolist()}")
        assert torch.equal(fus_e, ref_e), (
            f"expert_ids diverge: {fus_e.tolist()} vs {ref_e.tolist()} "
            f"topk_ids={topk_ids.tolist()}")


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906()),
    reason="gfx906 M=1 fused align dispatch test (C1 stage 1)")
def test_m1_fused_align_dispatch_gate(monkeypatch):
    """The gate must fire only for the exact decode shape (M=1, topk=8,
    E=256, bsm=1, no expert_map, int32, flag on)."""
    monkeypatch.setenv("VLLM_GFX906_ALIGN_M1", "1")
    dev = "cuda"
    em = torch.zeros(4, device=dev, dtype=torch.int32)
    cases = [
        # (topk_ids, bsm, E, expert_map, expected)
        (torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None,
         True),
        (torch.zeros(4, 8, device=dev, dtype=torch.int32), 1, 256, None,
         False),  # M>1
        (torch.zeros(1, 6, device=dev, dtype=torch.int32), 1, 256, None,
         False),  # topk!=8
        (torch.zeros(1, 8, device=dev, dtype=torch.int32), 4, 256, None,
         False),  # bsm!=1
        (torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 128, None,
         False),  # E!=256
        (torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, em,
         False),  # expert_map
        (torch.zeros(1, 8, device=dev, dtype=torch.int64), 1, 256, None,
         False),  # dtype
    ]
    for topk_ids, bsm, ne, emap, want in cases:
        got = _use_fused_align_m1(topk_ids, bsm, ne, emap)
        assert got is want, (
            f"gate={got} want={want} for "
            f"M={topk_ids.shape[0]} topk={topk_ids.shape[1]} E={ne} "
            f"bsm={bsm} map={emap is not None} dtype={topk_ids.dtype}")
    # Default is ON (serving A/B PASS); the opt-out flag disables it.
    monkeypatch.delenv("VLLM_GFX906_ALIGN_M1")
    assert _use_fused_align_m1(
        torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None)
    monkeypatch.setenv("VLLM_GFX906_ALIGN_M1", "0")
    assert not _use_fused_align_m1(
        torch.zeros(1, 8, device=dev, dtype=torch.int32), 1, 256, None)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906() and _has_op()),
    reason="gfx906 M=1 fused align graph test (C1 stage 1)")
def test_m1_fused_align_graph_capturable():
    """The kernel is capture-safe (single CTA, no D2H, fixed grid) and
    replay-stable: capturing the 40-layer chain and replaying must keep
    producing bit-correct outputs."""
    dev = "cuda"
    topk_ids = torch.randint(0, 256, (1, 8), device=dev).int()
    s = torch.empty(8, device=dev, dtype=torch.int32)
    e = torch.empty(8, device=dev, dtype=torch.int32)
    n = torch.empty(1, device=dev, dtype=torch.int32)

    def chain():
        for _ in range(40):
            torch.ops._rocm_C.moe_align_block_size_m1_gfx906(
                topk_ids, 256, 1, s, e, n)

    ref_s, ref_e, ref_n = _prod_chain(topk_ids, dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(s, ref_s)
    assert torch.equal(e, ref_e)
    assert torch.equal(n, ref_n)
