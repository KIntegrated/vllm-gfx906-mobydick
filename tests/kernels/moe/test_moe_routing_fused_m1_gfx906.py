# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Tests for the gfx906 M=1 fused routing kernel (C1 stage 2).

Run `pytest tests/kernels/moe/test_moe_routing_fused_m1_gfx906.py`.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx906
else:

    def on_gfx906() -> bool:
        return False


def _has_op() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_routing_fused_m1_gfx906"
    )


def _prod_chain(gating, renormalize, dev):
    """Production chain: generic topk + two-kernel align."""
    tw = torch.empty(1, 8, dtype=torch.float32, device=dev)
    ti = torch.empty(1, 8, dtype=torch.int32, device=dev)
    tei = torch.empty(1, 8, dtype=torch.int32, device=dev)
    ops.topk_softmax(tw, ti, tei, gating, renormalize)
    s, e, n = moe_align_block_size(ti, 1, 256, None)
    return tw, ti, tei, s, e, n


def _fused(gating, renormalize, dev):
    tw = torch.empty(1, 8, dtype=torch.float32, device=dev)
    ti = torch.empty(1, 8, dtype=torch.int32, device=dev)
    tei = torch.empty(1, 8, dtype=torch.int32, device=dev)
    s = torch.empty(8, device=dev, dtype=torch.int32)
    e = torch.empty(8, device=dev, dtype=torch.int32)
    n = torch.empty(1, device=dev, dtype=torch.int32)
    torch.ops._rocm_C.moe_routing_fused_m1_gfx906(
        gating, tw, ti, tei, s, e, n, renormalize)
    return tw, ti, tei, s, e, n


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906() and _has_op()),
    reason="gfx906 M=1 fused routing kernel test (C1 stage 2)")
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("seed", range(12))
def test_routing_fused_bit_equal_to_production(seed, renormalize):
    """The single-CTA kernel must reproduce the generic topk + two-kernel
    align chain bit-for-bit (topk_weights/ids/token_expert_ids +
    sorted_token_ids/expert_ids/ntp) for random and tie-heavy logits."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    cases = [(torch.randn(1, 256, generator=g, device=dev) * 0.1).half()]
    if seed % 3 == 0:
        cases.append(torch.full((1, 256), 0.25, device=dev,
                                dtype=torch.half))
    if seed % 3 == 1:
        tied = torch.zeros(1, 256, device=dev, dtype=torch.half)
        tied[0, ::4] = 1.0
        cases.append(tied)
    if seed % 3 == 2:
        cases.append(torch.zeros(1, 256, device=dev, dtype=torch.half))

    for gating in cases:
        ref = _prod_chain(gating, renormalize, dev)
        fus = _fused(gating, renormalize, dev)
        for name, r, f in zip(
            ("topk_weights", "topk_ids", "token_expert_ids",
             "sorted_token_ids", "expert_ids", "ntp"),
            ref, fus,
        ):
            assert torch.equal(f, r), (
                f"{name} diverge (renormalize={renormalize}): "
                f"{f.tolist()} vs {r.tolist()} "
                f"gating[:8]={gating[0, :8].tolist()}")


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906() and _has_op()),
    reason="gfx906 M=1 fused routing graph test (C1 stage 2)")
def test_routing_fused_graph_capturable():
    """Capture-safe: 40-layer chain capture + replay stays bit-correct."""
    dev = "cuda"
    gating = torch.randn(1, 256, device=dev).half()
    tw = torch.empty(1, 8, dtype=torch.float32, device=dev)
    ti = torch.empty(1, 8, dtype=torch.int32, device=dev)
    tei = torch.empty(1, 8, dtype=torch.int32, device=dev)
    s = torch.empty(8, device=dev, dtype=torch.int32)
    e = torch.empty(8, device=dev, dtype=torch.int32)
    n = torch.empty(1, device=dev, dtype=torch.int32)

    def chain():
        for _ in range(40):
            torch.ops._rocm_C.moe_routing_fused_m1_gfx906(
                gating, tw, ti, tei, s, e, n, True)

    ref = _prod_chain(gating, True, dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain()
    g.replay()
    torch.cuda.synchronize()
    for name, r, f in zip(("tw", "ti", "tei", "s", "e", "n"), ref,
                          (tw, ti, tei, s, e, n)):
        assert torch.equal(f, r), f"{name} diverge after graph replay"


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906()),
    reason="gfx906 M=1 fused routing dispatch test (C1 stage 2)")
def test_routing_fused_router_dispatch(monkeypatch):
    """FusedTopKRouter._compute_routing fused mode: meta set + outputs
    bit-equal to the generic path; gate rejects non-decode shapes."""
    dev = "cuda"
    monkeypatch.setenv("VLLM_GFX906_ROUTING_FUSE_M1", "1")
    router = FusedTopKRouter(top_k=8, global_num_experts=256,
                             renormalize=True)
    logits = torch.randn(1, 256, device=dev).half()
    hs = torch.randn(1, 64, device=dev).half()

    tw, ti = router._compute_routing(hs, logits, None)
    assert router._fused_align_meta is not None
    s, e, n = router._fused_align_meta
    ref = _prod_chain(logits, True, dev)
    assert torch.equal(tw, ref[0])
    assert torch.equal(ti, ref[1])
    assert torch.equal(s, ref[3])
    assert torch.equal(e, ref[4])
    assert torch.equal(n, ref[5])
    assert s.shape == (8,) and e.shape == (8,)

    # non-fused path clears the meta
    logits4 = torch.randn(4, 256, device=dev).half()
    hs4 = torch.randn(4, 64, device=dev).half()
    router._compute_routing(hs4, logits4, None)
    assert router._fused_align_meta is None
    monkeypatch.delenv("VLLM_GFX906_ROUTING_FUSE_M1")

    # gate: wrong shape / scoring / dtype -> not eligible
    cases = [
        (torch.randn(4, 256, device=dev).half(), None, False),  # M>1
        (torch.randn(1, 128, device=dev).half(), None, False),  # E!=256
        (torch.randn(1, 256, device=dev).float(), None,
         False),  # fp32
    ]
    for g, idx, want in cases:
        got = router._use_routing_fused_m1(g, idx)
        assert got is want, f"gate={got} want={want} M={g.shape[0]} "
    assert not router._use_routing_fused_m1(logits, torch.int64)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906()),
    reason="gfx906 M=1 fused routing gate shape test (C1 stage 2)")
def test_routing_fused_gate_wrong_expert_count(monkeypatch):
    """E!=256 (e.g. 128-expert models) must stay on the generic path."""
    dev = "cuda"
    monkeypatch.setenv("VLLM_GFX906_ROUTING_FUSE_M1", "1")
    router = FusedTopKRouter(top_k=8, global_num_experts=128)
    logits = torch.randn(1, 128, device=dev).half()
    assert not router._use_routing_fused_m1(logits, None)
    monkeypatch.delenv("VLLM_GFX906_ROUTING_FUSE_M1")
