# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
