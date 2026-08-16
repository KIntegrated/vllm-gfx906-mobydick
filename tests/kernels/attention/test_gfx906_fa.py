# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""gfx906 custom FlashAttention (vllm/gfx906_fa) regression tests.

The gather/direct kernels take byte strides computed in C++. A past bug
computed them from tensor SHAPES; the backend passes value_cache from
kv_cache.unbind(1) of [num_blocks, 2, block, Hkv, D] — non-contiguous with
2x block stride — so the kernel read K-cache bytes as V. These tests mirror
that allocation path exactly.
"""

import math

import pytest
import torch

from vllm import _gfx906_fa_C as fa
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx906()),
    reason="gfx906 FA extension kernels",
)

BLOCK, HKV, HQ, D = 16, 2, 16, 256
BYTES = (D // 32) * 34


def _make_paged_cache(num_blocks: int, dev: str):
    """Mirror Gfx906FABackend: one kv_cache tensor, unbind(1) -> K, V views."""
    kc = torch.zeros(num_blocks, BLOCK, HKV, BYTES, dtype=torch.uint8, device=dev)
    kv = torch.zeros(num_blocks, 2, BLOCK, HKV, D, dtype=torch.float16, device=dev)
    key_cache_q8 = kc
    _, value_cache = kv.unbind(1)
    assert not value_cache.is_contiguous()
    return key_cache_q8, value_cache, kv


def _write_v(kv: torch.Tensor, V: torch.Tensor):
    """Write token-major V rows into the V half of the [N,2,B,H,D] cache."""
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, HKV, D)[: V.shape[0]].copy_(V)
    kv[:, 1].copy_(staging)


def _fill(kv_flat_rows: torch.Tensor, kc: torch.Tensor, slot: torch.Tensor):
    fa.reshape_and_cache_q8(kv_flat_rows, slot, kc)


def test_fused_gather_matches_torch_gather_on_unbind_cache():
    dev = "cuda"
    torch.manual_seed(1)
    num_blocks = 40
    kc, vc, kv = _make_paged_cache(num_blocks, dev)
    n_rows = num_blocks * BLOCK
    K = torch.randn(n_rows, HKV, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(n_rows, HKV, D, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(n_rows, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    _write_v(kv, V)

    B, seq_lens, max_len = 2, [100, 300], 300
    n_blocks = (max_len + BLOCK - 1) // BLOCK
    bt = torch.arange(0, B * n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(B, n_blocks).contiguous()
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    Sk_pad = (max_len + 31) // 32 * 32

    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv_q8

    k_ref, v_ref = _gather_kv_q8(kc, vc, bt, sl, max_len)
    k_f, v_f = fa.gather_paged_kv_q8(kc, vc, bt, sl, Sk_pad)
    for b, L in enumerate(seq_lens):
        # Only the valid region must match: the fused kernel leaves K tail
        # rows unwritten (kernel cuts them via kv_max); the torch path
        # gathers real cache rows there instead.
        assert torch.equal(k_f[b, :, :L], k_ref[b, :, :L])
        assert torch.equal(v_f[b, :, :L], v_ref[b, :, :L])
        assert bool((v_f[b, :, L:] == 0).all().item())


def test_cudagraph_capture_replay_legacy_decode_path():
    """M2 gate: the LEGACY (inline-quant) decode path must be FULL-capture-safe.

    Captures the exact serving composite (`forward_paged` with
    key_cache_q8=None, i.e. fp16 K cache + inline K quant) and covers the
    sub-plan T3 landmines for this path: (a) warmup at a small max_seqlen_k
    followed by capture at capacity (buffer-realloc class); (b) multi-size
    capture (B=1 then B=2) with a B=1 replay afterwards (dangling-buffer
    class); (c) the live-metadata invariant — seq_lens is re-read at replay,
    so growing Sk and filling the new K/V rows must make the replayed output
    match eager at the new length.
    """
    dev = "cuda"
    torch.manual_seed(3)
    max_len = 512
    n_blocks = (max_len + BLOCK - 1) // BLOCK
    kc, vc, kv = _make_paged_cache(n_blocks + 4, dev)
    scale = 1.0 / math.sqrt(D)

    # LEGACY path: K lives in an fp16 cache (contiguous here; the backend's
    # unbind(1) K view has the same per-element layout the C++ strides expect),
    # V in the strided unbind view as in serving (exercised via _write_v).
    k16 = torch.zeros(n_blocks + 4, BLOCK, HKV, D, dtype=torch.float16,
                      device=dev)
    K = torch.randn(max_len, HKV, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(max_len, HKV, D, device=dev, dtype=torch.float16) * 0.5
    _write_v(kv, V[:100])
    k16.view(-1, HKV, D)[:100].copy_(K[:100])

    from vllm.gfx906_fa.gfx906_fa_paged import forward_paged

    # Shared q_pad buffer across both graphs, as the backend's lazy-grown
    # class buffer would be at capture capacity.
    q_pad = torch.zeros(2, HQ, 2, D, dtype=torch.float32, device=dev)

    def fwd(q, bt_, sl_, cu_, msk):
        return forward_paged(
            q, k16, vc, bt_, sl_, cu_,
            max_seqlen_q=1, max_seqlen_k=msk, scale=scale,
            key_cache_q8=None, q_pad_buf=q_pad,
        )

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([100], dtype=torch.int32, device=dev)
    cu1 = torch.arange(2, dtype=torch.int32, device=dev)
    q1 = torch.randn(1, HQ, D, device=dev, dtype=torch.float32) * 0.5

    # (a) warmup at small max_seqlen_k, then capture at capacity
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fwd(q1, bt, sl, cu1, 128)
    torch.cuda.current_stream().wait_stream(s)
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        out1 = fwd(q1, bt, sl, cu1, max_len)
    g1.replay()
    torch.cuda.synchronize()
    ref1 = fwd(q1, bt, sl, cu1, max_len)
    assert ((out1 - ref1).norm() / ref1.norm()).item() < 2e-2

    # (b) capture B=2 after B=1, then replay B=1 (dangling-buffer check)
    # Both rows share the same 32 blocks (arange(n_blocks).view(2, -1) would
    # be (2, 16) — wrong column count).
    bt2 = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1).expand(2, -1).contiguous()
    sl2 = torch.tensor([100, 150], dtype=torch.int32, device=dev)
    cu2 = torch.arange(3, dtype=torch.int32, device=dev)
    q2 = torch.randn(2, HQ, D, device=dev, dtype=torch.float32) * 0.5
    with torch.cuda.stream(s):
        for _ in range(2):
            fwd(q2, bt2, sl2, cu2, 256)
    torch.cuda.current_stream().wait_stream(s)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        out2 = fwd(q2, bt2, sl2, cu2, max_len)
    g2.replay()
    torch.cuda.synchronize()
    ref2 = fwd(q2, bt2, sl2, cu2, max_len)
    assert (
        (out2[1] - ref2[1]).norm() / ref2[1].norm()
    ).item() < 2e-2  # row 1 at Sk=150 exercises the longer row
    g1.replay()
    torch.cuda.synchronize()
    assert ((out1 - ref1).norm() / ref1.norm()).item() < 2e-2

    # (c) live seq_lens: grow Sk 100 -> 200, fill K/V rows, replay g1
    k16.view(-1, HKV, D)[100:200].copy_(K[100:200])
    _write_v(kv, V[:200])
    sl.fill_(200)
    g1.replay()
    torch.cuda.synchronize()
    ref200 = fwd(q1, bt, sl, cu1, max_len)
    assert ((out1 - ref200).norm() / ref200.norm()).item() < 2e-2


def test_fused_fp16_gather_matches_torch_gather():
    """LEGACY-path fused gather (gather_paged_kv_fp16) must match the torch
    _gather_kv reference in the valid region; V tail zeroed; K tail
    unmasked (FA kernel cuts via kv_max)."""
    dev = "cuda"
    torch.manual_seed(4)
    L = 500  # not a multiple of 32: exercises Sk_pad tail handling
    n_blocks = (L + BLOCK - 1) // BLOCK
    kc, vc, kv = _make_paged_cache(n_blocks, dev)
    k16 = torch.randn(n_blocks, BLOCK, HKV, D, device=dev,
                      dtype=torch.float16) * 0.5
    V = torch.randn(L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    _write_v(kv, V)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    Sk_pad = (L + 31) // 32 * 32

    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv

    k_ref, v_ref = _gather_kv(k16, vc, bt, sl, L)
    k_f, v_f = fa.gather_paged_kv_fp16(k16, vc, bt, sl, Sk_pad)
    assert k_f.shape == (1, HKV, Sk_pad, D) and v_f.shape == (1, HKV, Sk_pad, D)
    assert torch.equal(k_f[0, :, :L], k_ref[0, :, :L])
    assert torch.equal(v_f[0, :, :L], v_ref[0, :, :L])
    assert bool((v_f[0, :, L:] == 0).all())


def test_forward_decode_prefill_vs_sdpa_on_unbind_cache():
    dev = "cuda"
    torch.manual_seed(2)
    L = 512
    n_blocks = L // BLOCK
    kc, vc, kv = _make_paged_cache(n_blocks, dev)
    K = torch.randn(L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    _write_v(kv, V)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(D)
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, (L + 31) // 32 * 32)
    g = HQ // HKV
    k, v = K.float(), V.float()

    # decode: one query, no causal
    q = torch.randn(1, HQ, 1, D, device=dev, dtype=torch.float32) * 0.5
    out = fa.forward(q, k_q8, v_b, scale, kv_max=sl)[0, :, 0]  # [HQ, D]
    qg = q[0, :, 0].view(HKV, g, D)
    s = torch.einsum("gjd,lgd->gjl", qg, k) * scale
    ref = torch.einsum("gjl,lgd->gjd", torch.softmax(s, -1), v).reshape(HQ, D)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2

    # prefill: full causal chunk
    qf = torch.randn(1, HQ, L, D, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([0], dtype=torch.int32, device=dev)
    outf = fa.forward(qf, k_q8, v_b, scale, kv_max=sl, q_abs_offset=q_abs)[0]
    outf = outf.permute(1, 0, 2)  # [L, HQ, D]
    qtok = qf[0].permute(1, 0, 2).float()  # [L, HQ, D]
    for t in (1, 63, L - 1):
        qg = qtok[t].view(HKV, g, D)
        s = torch.einsum("gjd,lgd->gjl", qg, k[: t + 1]) * scale
        ref = torch.einsum(
            "gjl,lgd->gjd", torch.softmax(s, -1), v[: t + 1]
        ).reshape(HQ, D)
        assert ((outf[t] - ref).norm() / ref.norm()).item() < 5e-2
