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
import sys

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


@pytest.mark.parametrize("B,seq_lens", [(2, [100, 300]), (1, [3328]), (1, [33])])
def test_fused_gather_quantized_bit_equal_to_gather_then_quantize(B, seq_lens):
    """Stage-2 fused gather+quantize must be bit-equal to the two-kernel
    sequence (gather_paged_kv_fp16 + quantize_q8_0): same quantization
    helper, same arithmetic. Guards against future drift between the
    fused kernel and the reference quantizer."""
    dev = "cuda"
    torch.manual_seed(3)
    max_len = max(seq_lens)
    n_blocks_needed = (max_len + BLOCK - 1) // BLOCK
    num_blocks = B * n_blocks_needed + 4
    # Same allocation as Gfx906FABackend LEGACY path: one [N,2,B,H,D]
    # tensor, unbind(1) -> non-contiguous K/V views (2x block stride).
    kv = torch.zeros(num_blocks, 2, BLOCK, HKV, D,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    assert not key_cache.is_contiguous() and not value_cache.is_contiguous()
    K = torch.randn(num_blocks, BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(num_blocks, BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    kv[:, 0].copy_(K)
    kv[:, 1].copy_(V)

    n_blocks = (max_len + BLOCK - 1) // BLOCK
    bt = torch.arange(0, B * n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(B, n_blocks).contiguous()
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    Sk_pad = (max_len + 31) // 32 * 32

    k_two, v_two = fa.gather_paged_kv_fp16(
        key_cache, value_cache, bt, sl, Sk_pad)
    k_ref = fa.quantize_q8_0(k_two)
    k_one, v_one = fa.gather_paged_kv_quantized(
        key_cache, value_cache, bt, sl, Sk_pad)

    assert k_one.shape == k_ref.shape
    assert k_one.dtype == torch.uint8
    for b, L in enumerate(seq_lens):
        # Valid region: bit-exact (K quantized, V copied).
        assert torch.equal(k_one[b, :, :L], k_ref[b, :, :L])
        assert torch.equal(v_one[b, :, :L], v_two[b, :, :L])
        # Tail: V zeroed, K may be garbage (FA kernel cuts via kv_max).
        assert bool((v_one[b, :, L:] == 0).all().item())


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
    bt2 = torch.arange(n_blocks, dtype=torch.int32,
                       device=dev).view(1, -1).expand(2, -1).contiguous()
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


def test_persistent_gather_capture_replay_large_sk():
    """N4 fix gate: the persistent fused gather+quantize
    (GFX906_FA_PERSIST) must be FULL-capture-safe at capacity Sk_pad
    (262144, above the old 65535 two-kernel boundary) and its replayed
    end-to-end FA output must match the two-kernel fallback at every
    live seq_len in the sweep — the launch dim is frozen at Sk_pad while
    seq_lens is re-read at replay, exactly like the serving FULL graph.
    Buffer contents beyond seq_len may legitimately differ (tail-write
    removal, gated by the NaN-tail test); end-to-end output must not.
    """
    dev = "cuda"
    torch.manual_seed(5)
    sk_pad = 262144
    n_blocks = sk_pad // BLOCK + 4
    _, value_cache, kv = _make_paged_cache(n_blocks, dev)
    scale = 1.0 / math.sqrt(D)

    k16 = torch.zeros(n_blocks, BLOCK, HKV, D, dtype=torch.float16,
                      device=dev)
    k16.normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    from vllm.gfx906_fa import gfx906_fa_paged as fpp
    from vllm.gfx906_fa.gfx906_fa_paged import forward_paged

    q = torch.randn(1, HQ, D, device=dev, dtype=torch.float32) * 0.5
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, -1)
    sl = torch.tensor([100], dtype=torch.int32, device=dev)
    cu = torch.arange(2, dtype=torch.int32, device=dev)
    q_pad = torch.zeros(1, HQ, 1, D, dtype=torch.float32, device=dev)

    def fwd(persist, msk):
        fpp._PERSISTENT = persist
        return forward_paged(
            q, k16, value_cache, bt, sl, cu,
            max_seqlen_q=1, max_seqlen_k=msk, scale=scale,
            key_cache_q8=None, q_pad_buf=q_pad,
        )

    orig = fpp._PERSISTENT
    try:
        # warmup the persistent path at small Sk, then capture at capacity
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(2):
                fwd(True, 128)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fwd(True, sk_pad)
    finally:
        fpp._PERSISTENT = orig

    def rel_err(a, b):
        return ((a - b).norm() / b.norm()).item()

    sweep = [32, 100, 2048, 65504, 65536, 65600, 131072, sk_pad - 32,
             sk_pad]
    for v in sweep:
        sl.fill_(v)
        g.replay()
        torch.cuda.synchronize()
        ref_two = fwd(False, sk_pad)          # two-kernel fallback
        ref_persist = fwd(True, sk_pad)       # eager persistent
        e_two = rel_err(out, ref_two)
        e_persist = rel_err(out, ref_persist)
        assert e_persist == 0.0, (
            f"replay vs eager persistent not bit-exact at sl={v}: "
            f"rel={e_persist}")
        assert e_two < 2e-2, (
            f"replay vs two-kernel fallback diverges at sl={v}: rel={e_two}")


def test_persistent_dispatch_fallback_large_batch():
    """P0 regression guard (fa-masked-gather review): with PERSIST default
    ON, a batch above the kernel's 16-seq bound must fall back to the
    fused/two-kernel paths (old behavior) instead of hitting the C++
    TORCH_CHECK(num_seqs <= 16) — which would crash engine start for any
    default max_num_seqs (> 16)."""
    dev = "cuda"
    torch.manual_seed(7)
    b = 17
    max_len = 512
    n_blocks = b * (max_len // BLOCK)
    kv = torch.zeros(n_blocks, 2, BLOCK, HKV, D,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    kv[:, 0].normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    from vllm.gfx906_fa import gfx906_fa_paged as fpp
    from vllm.gfx906_fa.gfx906_fa_paged import forward_paged

    scale = 1.0 / math.sqrt(D)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(b, max_len // BLOCK).contiguous()
    sl = torch.tensor([33, 100, 200, 333, 512] + [400] * (b - 5),
                      dtype=torch.int32, device=dev)
    cu = torch.arange(b + 1, dtype=torch.int32, device=dev)
    q = torch.randn(b, HQ, D, device=dev, dtype=torch.float32) * 0.5
    q_pad = torch.zeros(b, HQ, 1, D, device=dev, dtype=torch.float32)

    def fwd(persist):
        fpp._PERSISTENT = persist
        return forward_paged(
            q, key_cache, value_cache, bt, sl, cu,
            max_seqlen_q=1, max_seqlen_k=max_len, scale=scale,
            key_cache_q8=None, q_pad_buf=q_pad,
        )

    orig = fpp._PERSISTENT
    try:
        out_persist = fwd(True)   # must not raise (B=17 > 16)
        out_ref = fwd(False)
    finally:
        fpp._PERSISTENT = orig
    assert torch.equal(out_persist, out_ref)


def test_persistent_gather_bit_equal_to_fused_at_batch_bound():
    """B=16 (the kernel's register-prefix bound), ragged seq_lens, small
    Sk: the persistent kernel must be bit-equal to the fused kernel
    (gather_paged_kv_quantized) in-range. The capture probe covers
    B=1..4 at full 262k live; this covers the prefix bound itself."""
    dev = "cuda"
    torch.manual_seed(11)
    b, sk = 16, 1024
    max_len = 1000
    n_blocks = b * (sk // BLOCK)
    kv = torch.zeros(n_blocks, 2, BLOCK, HKV, D,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    kv[:, 0].normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(b, sk // BLOCK).contiguous()
    sl = torch.tensor(
        [1, 33, 100, 300, 512, 777, 1000] + [640] * (b - 7),
        dtype=torch.int32, device=dev)

    kb = torch.empty(b, HKV, sk, BYTES, dtype=torch.uint8, device=dev)
    vb = torch.empty(b, HKV, sk, D, dtype=torch.float16, device=dev)
    k_fused, v_fused = fa.gather_paged_kv_quantized(
        key_cache, value_cache, bt, sl, sk)
    k_p, v_p = fa.gather_paged_kv_quant_persistent(
        key_cache, value_cache, bt, sl, sk, k_out=kb, v_out=vb)
    for s_ in range(b):
        L = int(sl[s_])
        assert torch.equal(k_p[s_, :, :L], k_fused[s_, :, :L]), f"K s={s_}"
        assert torch.equal(v_p[s_, :, :L], v_fused[s_, :, :L]), f"V s={s_}"


def test_persistent_gather_d128_matches_fused():
    """D=128 (other advertised head size; different FA tile config):
    persistent kernel bit-equal to the fused kernel in-range. Kernel is
    D-generic (V uint4 D/8 lanes, K blocks_per_row=D/32); this pins it
    before default-ON widens past the D=256 model family
    (fa-masked-gather-code-rev-qwen P1-2)."""
    dev = "cuda"
    torch.manual_seed(12)
    d = 128
    bpr = (d // 32) * 34
    b, sk = 3, 1024
    n_blocks = b * (sk // BLOCK)
    kv = torch.zeros(n_blocks, 2, BLOCK, HKV, d,
                     dtype=torch.float16, device=dev)
    key_cache, value_cache = kv.unbind(1)
    kv[:, 0].normal_(0, 0.5)
    kv[:, 1].normal_(0, 0.5)

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev)
    bt = bt.view(b, sk // BLOCK).contiguous()
    sl = torch.tensor([1, 300, 1000], dtype=torch.int32, device=dev)

    k_fused, v_fused = fa.gather_paged_kv_quantized(
        key_cache, value_cache, bt, sl, sk)
    k_p, v_p = fa.gather_paged_kv_quant_persistent(
        key_cache, value_cache, bt, sl, sk)
    assert k_p.shape == (b, HKV, sk, bpr)
    for s_ in range(b):
        L = int(sl[s_])
        assert torch.equal(k_p[s_, :, :L], k_fused[s_, :, :L]), f"K s={s_}"
        assert torch.equal(v_p[s_, :, :L], v_fused[s_, :, :L]), f"V s={s_}"


def test_fused_fp16_gather_matches_torch_gather():
    """LEGACY-path fused gather (gather_paged_kv_fp16) must match the torch
    _gather_kv reference in the valid region; V tail zeroed; K tail
    unmasked (FA kernel cuts via kv_max). Covers B=1 (Sk not a multiple
    of 32 — Sk_pad tail handling) and B=2 (per-row lengths and disjoint
    block ranges)."""
    dev = "cuda"
    torch.manual_seed(4)
    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv

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

    k_ref, v_ref = _gather_kv(k16, vc, bt, sl, L)
    k_f, v_f = fa.gather_paged_kv_fp16(k16, vc, bt, sl, Sk_pad)
    assert k_f.shape == (1, HKV, Sk_pad, D) and v_f.shape == (1, HKV, Sk_pad, D)
    assert torch.equal(k_f[0, :, :L], k_ref[0, :, :L])
    assert torch.equal(v_f[0, :, :L], v_ref[0, :, :L])
    assert bool((v_f[0, :, L:] == 0).all())

    # B=2: row 1 uses physical blocks disjoint from row 0's; different
    # lengths; unbind(1) strided views as in serving.
    L1, L2 = 300, 500
    n1 = (L1 + BLOCK - 1) // BLOCK
    n2 = (L2 + BLOCK - 1) // BLOCK
    width = n1 + n2
    kv2 = torch.randn(width, 2, BLOCK, HKV, D, device=dev,
                      dtype=torch.float16) * 0.5
    k16_2, vc2 = kv2.unbind(1)
    bt2 = torch.zeros(2, width, dtype=torch.int32, device=dev)
    bt2[0, :n1] = torch.arange(n1, dtype=torch.int32, device=dev)
    bt2[1, :n2] = torch.arange(n1, n1 + n2, dtype=torch.int32, device=dev)
    sl2 = torch.tensor([L1, L2], dtype=torch.int32, device=dev)
    k_ref2, v_ref2 = _gather_kv(k16_2, vc2, bt2, sl2, L2)
    k_f2, v_f2 = fa.gather_paged_kv_fp16(k16_2, vc2, bt2, sl2, Sk_pad)
    assert torch.equal(k_f2[0, :, :L1], k_ref2[0, :, :L1])
    assert torch.equal(v_f2[0, :, :L1], v_ref2[0, :, :L1])
    assert torch.equal(k_f2[1, :, :L2], k_ref2[1, :, :L2])
    assert torch.equal(v_f2[1, :, :L2], v_ref2[1, :, :L2])
    assert bool((v_f2[0, :, L1:] == 0).all())
    assert bool((v_f2[1, :, L2:] == 0).all())


def test_q_pad_buffer_survives_capture_then_prefill_grow():
    """Review F1: a captured graph bakes in the VA of the q_pad buffer
    that was current at capture time. An eager prefill with a larger
    Sq_pad afterwards grows that buffer; the old one must be retired
    (kept alive) rather than freed-then-realloc'd, and the replayed
    decode must stay numerically correct. Drives the real Gfx906FAImpl
    (not hand-fed buffers) in the hazardous production order: small
    decode → capture → large prefill → decode replay.
    """
    dev = "cuda"
    torch.manual_seed(7)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    assert impl._legacy  # this test targets the default serving path

    n_blocks = 16  # 256 tokens
    _, vc, kv = _make_paged_cache(n_blocks, dev)
    k16 = kv[:, 0]
    K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                    dtype=torch.float16) * 0.5
    k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
    _write_v(kv, V)

    def meta(num_tokens, sq, sk, bt_, sl_, cu_):
        return Gfx906FAMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=sq,
            max_seq_len=sk,
            query_start_loc=cu_,
            seq_lens=sl_,
            block_table=bt_,
            slot_mapping=torch.empty(0, dtype=torch.int64, device=dev),
        )

    layer = None  # impl.forward does not touch the layer object
    s = torch.cuda.Stream()

    # (1) small decode (eager): allocates the small q_pad (Sq_pad=2)
    bt_d = torch.arange((100 + BLOCK - 1) // BLOCK, dtype=torch.int32,
                        device=dev).view(1, -1)
    sl_d = torch.tensor([100], dtype=torch.int32, device=dev)
    cu_d = torch.arange(2, dtype=torch.int32, device=dev)
    q_d = torch.randn(1, HQ, D, device=dev, dtype=torch.float16) * 0.5
    out_d = torch.zeros(1, HQ, D, device=dev, dtype=torch.float16)
    m_d = meta(1, 1, 100, bt_d, sl_d, cu_d)
    with torch.cuda.stream(s):
        for _ in range(2):
            impl.forward(layer, q_d, q_d, q_d, kv, m_d, output=out_d)
    torch.cuda.current_stream().wait_stream(s)
    ref_d = out_d.clone()
    small_buf = impl._q_pad_buf
    assert small_buf.shape[2] == 2

    # (2) capture the decode graph (bakes small_buf's VA in)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        impl.forward(layer, q_d, q_d, q_d, kv, m_d, output=out_d)
    assert impl._q_pad_captured
    assert impl._q_pad_buf is small_buf

    # (3) eager prefill with larger Sq_pad → grow branch
    q_p = torch.randn(64, HQ, D, device=dev, dtype=torch.float16) * 0.5
    out_p = torch.zeros(64, HQ, D, device=dev, dtype=torch.float16)
    m_p = meta(64, 64, 64,
               torch.arange(4, dtype=torch.int32, device=dev).view(1, -1),
               torch.tensor([64], dtype=torch.int32, device=dev),
               torch.tensor([0, 64], dtype=torch.int32, device=dev))
    impl.forward(layer, q_p, q_p, q_p, kv, m_p, output=out_p)
    assert impl._q_pad_buf is not small_buf
    assert impl._q_pad_buf.shape[2] == 64
    assert bool(torch.isfinite(out_p.float()).all())

    # (4) the captured buffer was retired, not freed
    assert any(t is small_buf for t in impl._q_pad_retired)
    assert small_buf.data_ptr() != impl._q_pad_buf.data_ptr()

    # (5) replay: the graph writes q_pad through the retired-but-alive VA
    out_d.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert ((out_d - ref_d).norm() / ref_d.norm()).item() < 2e-2


def test_gather_buffers_lifecycle_postfix():
    """plan-gfx906-fa-fix.md §5 — pins the POST-FIX gather-buffer
    contract (GFX906_FA_GATHER_EXACT=0) by driving the real
    Gfx906FAImpl (class-level buffers; snapshot/restore so nothing
    leaks into other tests):
    (1) Sk shrink / ping-pong (130 -> 100 -> 130): no realloc (Sk is a
        grow-only capacity), retire set unchanged;
    (4) descending capture sweep (B=2 then B=1, as get_capture_descs
        sorts): ONE generation, smaller-B reuses the base VA as a
        leading-dim slice, end-of-capture retired <= 1;
    (2) a captured generation is retired (kept alive) when replaced —
        driven directly: capture at width 128, then grow past it to 160;
    (5) the per-generation flag is reset, not OR'd: the eager
        generation (flag False) that follows a retired captured one is
        itself FREED on replacement;
    (3) eager growth (never captured) frees the old generation —
        allocator-level evidence: the delta is the new-minus-old
        excess, not the new generation on top of the old.
    """
    dev = "cuda"
    torch.manual_seed(11)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    assert impl._legacy  # this test targets the default serving path
    cls = type(impl)

    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = False
    cls._gather_retired_warned = False
    try:
        # 96 blocks so B=8 x 12-block tables stay in range (190 tokens
        # need 12 blocks of 16).
        n_blocks = 96
        _, vc, kv = _make_paged_cache(n_blocks, dev)
        k16 = kv[:, 0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v(kv, V)

        def meta(b, sq, sk, nblk_per_seq):
            return Gfx906FAMetadata(
                num_actual_tokens=b * sq,
                max_query_len=sq,
                max_seq_len=sk,
                query_start_loc=torch.arange(
                    0, b * sq + 1, sq, dtype=torch.int32, device=dev),
                seq_lens=torch.full((b,), sk, dtype=torch.int32,
                                    device=dev),
                block_table=torch.arange(
                    b * nblk_per_seq, dtype=torch.int32,
                    device=dev).view(b, nblk_per_seq),
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        layer = None
        s = torch.cuda.Stream()

        def decode(b, sk, nblk, q, out, m=None):
            impl.forward(layer, q, q, q, kv,
                         m or meta(b, 1, sk, nblk), output=out)

        q1 = torch.randn(1, HQ, D, device=dev, dtype=torch.float16) * 0.5
        q2 = torch.randn(2, HQ, D, device=dev, dtype=torch.float16) * 0.5
        q4 = torch.randn(4, HQ, D, device=dev, dtype=torch.float16) * 0.5
        q8 = torch.randn(8, HQ, D, device=dev, dtype=torch.float16) * 0.5
        o1 = torch.zeros(1, HQ, D, device=dev, dtype=torch.float16)
        o2 = torch.zeros(2, HQ, D, device=dev, dtype=torch.float16)
        o4 = torch.zeros(4, HQ, D, device=dev, dtype=torch.float16)
        o8 = torch.zeros(8, HQ, D, device=dev, dtype=torch.float16)

        with torch.cuda.stream(s):
            # (1) eager growth to Sk_pad(130)=160, then ping-pong down
            # and up: capacity reuse, zero reallocs, empty retire set.
            decode(1, 130, 9, q1, o1)
            g1 = cls._k_gather_buf
            assert g1.shape[2] == 160
            decode(1, 100, 7, q1, o1)
            # sk=100 reference: this is the input the g1g graph replays.
            ref1 = o1.clone()
            assert cls._k_gather_buf is g1 and not cls._gather_retired
            decode(1, 130, 9, q1, o1)
            assert cls._k_gather_buf is g1 and not cls._gather_retired
            # Eager B-grow: gen1 never captured -> freed, not retired.
            # Exact-need sizing for freeable generations: the width is
            # NOT carried across (that would pin the B x Sk high-water
            # product — see test_gather_freeable_generation_exact_need).
            decode(2, 100, 7, q2, o2)
            ref2 = o2.clone()
            assert cls._k_gather_buf is not g1
            assert g1.data_ptr() not in cls._gather_retired
            g2 = cls._k_gather_buf
            assert g2.shape[:3] == (2, HKV, 128)
            assert not cls._gather_buf_captured
        torch.cuda.current_stream().wait_stream(s)

        # (4) descending capture sweep on the pre-allocated buffer (the
        # engine bakes the existing VA; no capture-time allocs). B=2
        # first (largest-first), then B=1 via the leading-dim slice.
        m2_cap = meta(2, 1, 100, 7)
        g2g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2g):
            decode(2, 100, 7, q2, o2, m2_cap)
        assert cls._k_gather_buf is g2
        assert cls._gather_buf_captured  # latched in the reuse path
        k_slice = cls._ensure_gather_buffers(1, HKV, 100, D,
                                             kv.device)[0]
        assert k_slice.data_ptr() == g2.data_ptr()
        assert cls._k_gather_buf is g2
        m1_cap = meta(1, 1, 100, 7)
        g1g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1g):
            decode(1, 100, 7, q1, o1, m1_cap)
        # End-of-capture: the sweep retired nothing (largest-first).
        assert len(cls._gather_retired) <= 1

        # (2) grow past the captured width (160): the captured generation
        # is retired (kept alive); the replacement starts uncaptured.
        with torch.cuda.stream(s):
            decode(2, 190, 12, q2, o2)
        torch.cuda.current_stream().wait_stream(s)
        g3 = cls._k_gather_buf
        assert g3.shape[2] == 192 and g3 is not g2
        assert g2.data_ptr() in cls._gather_retired
        assert not cls._gather_buf_captured  # reset, not OR'd

        # Replaying the B=2 graph hits the retired-but-alive base VA.
        o2.zero_()
        g2g.replay()
        torch.cuda.synchronize()
        assert ((o2 - ref2).norm() / ref2.norm()).item() < 2e-2
        o1.zero_()
        g1g.replay()
        torch.cuda.synchronize()
        assert ((o1 - ref1).norm() / ref1.norm()).item() < 2e-2

        # (5) capture gen3 (flag latches True); a B-grow retires it.
        # The following eager gen4 (flag False) is then FREED on its own
        # B-grow — the flag did not leak across the replacement.
        m2_cap190 = meta(2, 1, 190, 12)
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(2, 190, 12, q2, o2, m2_cap190)
        assert cls._gather_buf_captured and cls._k_gather_buf is g3
        with torch.cuda.stream(s):
            decode(4, 190, 12, q4, o4)  # B-grow: retires g3 (captured)
        torch.cuda.current_stream().wait_stream(s)
        g4 = cls._k_gather_buf
        assert g4.shape[0] == 4 and g3.data_ptr() in cls._gather_retired
        assert not cls._gather_buf_captured
        with torch.cuda.stream(s):
            decode(8, 190, 12, q8, o8)  # B-grow: g4 (eager) freed
        torch.cuda.current_stream().wait_stream(s)
        assert g4.data_ptr() not in cls._gather_retired
        assert g3.data_ptr() in cls._gather_retired
        kept = {g.data_ptr() for g, _ in cls._gather_retired.values()}
        assert kept == {g2.data_ptr(), g3.data_ptr()}

        # (3) eager growth frees: allocator-level evidence. Fresh state.
        cls._k_gather_buf = cls._v_gather_buf = None
        cls._gather_retired = {}
        cls._gather_captured = False
        cls._gather_buf_captured = False
        with torch.cuda.stream(s):
            decode(1, 100, 7, q1, o1)
        torch.cuda.current_stream().wait_stream(s)
        # Record the VA + size but DO NOT hold the tensors — a live
        # Python reference would keep the generation alive and defeat
        # the allocator-level check (plan §5 item 3).
        gen_a_ptr = cls._k_gather_buf.data_ptr()
        gen_a_bytes = cls._k_gather_buf.numel() + cls._v_gather_buf.numel() * 2
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()
        with torch.cuda.stream(s):
            decode(1, 130, 9, q1, o1)  # grow 128 -> 160
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - alloc_before
        # If the old generation were kept alive the delta would be the
        # full new generation; freed, it is only the 160-vs-128 excess.
        assert gen_a_ptr not in cls._gather_retired
        assert delta < gen_a_bytes, (delta, gen_a_bytes)
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_persistent_gather_fa_wide_buffer_poisoned_tail():
    """plan §5 — width >> live is safe end-to-end on the persistent
    path: the gather's work is live-bounded (device-side seq_lens) and
    the FA kernel cuts each sequence at kv_max, never at Sk. The stale
    region [seq_len, width) of the class buffer is poisoned with huge
    q8 bytes / fp16 NaN (exactly the data that would leak through if
    the tail mask were ever broken); the FA output must be bit-equal
    to the exact-width reference. Mixed seq_lens [37, 1000] exercise
    the per-seq margin clamp at num_seqs > 1; a second uniform
    multi-token (MTP-draft-shaped) forward reuses the same wide buffer.
    """
    dev = "cuda"
    torch.manual_seed(13)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured)
    try:
        B, seq_lens, width = 2, [37, 1000], 4096
        nblk = (seq_lens[-1] + BLOCK - 1) // BLOCK  # 63
        n_blocks = B * nblk + 4
        _, vc, kv = _make_paged_cache(n_blocks, dev)
        k16 = kv[:, 0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v(kv, V)

        def meta(b, sq, sk, sl_, bt_):
            return Gfx906FAMetadata(
                num_actual_tokens=b * sq,
                max_query_len=sq,
                max_seq_len=sk,
                query_start_loc=torch.arange(
                    0, b * sq + 1, sq, dtype=torch.int32, device=dev),
                seq_lens=sl_,
                block_table=bt_,
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        bt = torch.arange(B * nblk, dtype=torch.int32,
                          device=dev).view(B, nblk)
        layer = None
        q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out = torch.zeros(B, HQ, D, device=dev, dtype=torch.float16)

        def reset_class():
            cls._k_gather_buf = cls._v_gather_buf = None
            cls._gather_retired = {}
            cls._gather_captured = False
            cls._gather_buf_captured = False

        # Reference: exact-width class buffer (fresh allocation).
        reset_class()
        sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        m = meta(B, 1, seq_lens[-1], sl, bt)
        impl.forward(layer, q, q, q, kv, m, output=out)
        ref = out.clone()
        assert cls._k_gather_buf.shape[2] == 1024  # Sk_pad(1000)

        # Wide buffer with the stale region poisoned.
        bpr = (D // 32) * 34
        kw = torch.empty(B, HKV, width, bpr, dtype=torch.uint8,
                         device=dev)
        vw = torch.empty(B, HKV, width, D, dtype=torch.float16,
                         device=dev)
        for s_i, L in enumerate(seq_lens):
            kw[s_i, :, L:, :].fill_(0x7F)  # huge q8 scale/data bytes
            vw[s_i, :, L:, :].fill_(float("nan"))
        cls._k_gather_buf, cls._v_gather_buf = kw, vw
        cls._gather_buf_captured = False
        out.zero_()
        impl.forward(layer, q, q, q, kv, m, output=out)
        assert torch.equal(out, ref), (
            "poisoned wide buffer changed the FA output — tail mask "
            "leak or width-bound work on the persistent path")

        # MTP-draft-shaped second forward: multi-token queries, longer
        # seq_lens, same wide buffer (capacity reuse, no realloc).
        sl2 = torch.tensor([40, 1003], dtype=torch.int32, device=dev)
        qd = torch.randn(2 * 3, HQ, D, device=dev,
                         dtype=torch.float16) * 0.5
        outd = torch.zeros(2 * 3, HQ, D, device=dev, dtype=torch.float16)
        m2 = meta(2, 3, 1003, sl2, bt)
        reset_class()
        impl.forward(layer, qd, qd, qd, kv, m2, output=outd)
        refd = outd.clone()
        assert cls._k_gather_buf.shape[2] == 1024
        for s_i, L in enumerate([40, 1003]):
            kw[s_i, :, L:, :].fill_(0x7F)
            vw[s_i, :, L:, :].fill_(float("nan"))
        cls._k_gather_buf, cls._v_gather_buf = kw, vw
        cls._gather_buf_captured = False
        outd.zero_()
        impl.forward(layer, qd, qd, qd, kv, m2, output=outd)
        assert torch.equal(outd, refd), (
            "poisoned wide buffer changed the draft-shaped FA output")
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured) = saved


def test_wide_buffer_b17_fused_quant_no_leak():
    """plan §2.2c/§5 — num_seqs > _PERSIST_MAX_SEQS with a WIDE class
    buffer: the fused-quant path keeps the exact contract, so it must
    still work (no crash, finite output, wide buffer untouched) and
    must not accumulate its per-call C++ allocations across steps
    (they are freed after each call — no reuse, but no leak either).
    """
    dev = "cuda"
    torch.manual_seed(17)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    B = 17
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured)
    try:
        sk = 64
        nblk = (sk + BLOCK - 1) // BLOCK  # 4
        n_blocks = B * nblk + 4
        _, vc, kv = _make_paged_cache(n_blocks, dev)
        k16 = kv[:, 0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v(kv, V)

        m = Gfx906FAMetadata(
            num_actual_tokens=B,
            max_query_len=1,
            max_seq_len=sk,
            query_start_loc=torch.arange(B + 1, dtype=torch.int32,
                                         device=dev),
            seq_lens=torch.full((B,), sk, dtype=torch.int32, device=dev),
            block_table=torch.arange(B * nblk, dtype=torch.int32,
                                     device=dev).view(B, nblk),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=dev))
        bpr = (D // 32) * 34
        cls._k_gather_buf = torch.empty(B, HKV, 1024, bpr,
                                        dtype=torch.uint8, device=dev)
        cls._v_gather_buf = torch.empty(B, HKV, 1024, D,
                                        dtype=torch.float16, device=dev)
        cls._gather_retired = {}
        cls._gather_captured = False
        cls._gather_buf_captured = False

        q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out = torch.zeros(B, HQ, D, device=dev, dtype=torch.float16)
        impl.forward(None, q, q, q, kv, m, output=out)
        assert bool(torch.isfinite(out.float()).all())
        # The wide buffer was refused by the exact-contract site
        # (no reuse, no retire, no consumption).
        assert cls._k_gather_buf.shape[2] == 1024
        assert not cls._gather_retired

        # No accumulation across steps: per-call C++ allocs are freed.
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(3):
            impl.forward(None, q, q, q, kv, m, output=out)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - before < 16 * 2**20
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured) = saved


def test_gather_exact_killswitch_restores_old_policy():
    """plan §5 — GFX906_FA_GATHER_EXACT=1 (cls._gather_exact here; the
    in-service A/B sets the env var, read at import by both files)
    restores the pre-fix behavior: exact-Sk realloc and the sticky
    _gather_captured latch, so an Sk ping-pong after capture retires
    EVERY generation — the unbounded growth this fix removes. Pinned
    so the kill switch stays a true A/B arm.
    Drop this test together with the switch itself at the next
    gather-lifecycle change (plan §6 lifecycle note).
    """
    dev = "cuda"
    torch.manual_seed(19)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = True
    try:
        n_blocks = 32
        _, vc, kv = _make_paged_cache(n_blocks, dev)
        k16 = kv[:, 0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v(kv, V)

        def m(sk):
            return Gfx906FAMetadata(
                num_actual_tokens=1, max_query_len=1, max_seq_len=sk,
                query_start_loc=torch.arange(2, dtype=torch.int32,
                                             device=dev),
                seq_lens=torch.tensor([sk], dtype=torch.int32,
                                      device=dev),
                block_table=torch.arange(9, dtype=torch.int32,
                                         device=dev).view(1, 9),
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev))

        q1 = torch.randn(1, HQ, D, device=dev, dtype=torch.float16) * 0.5
        o1 = torch.zeros(1, HQ, D, device=dev, dtype=torch.float16)
        # Eager forward: allocs gen1 [1, HKV, 128] and grows q_pad.
        impl.forward(None, q1, q1, q1, kv, m(100), output=o1)
        gen1 = cls._k_gather_buf
        assert gen1.shape[2] == 128
        # Capture reuses gen1 and latches the sticky latch. Metadata is
        # pre-built (capture-time allocations/copies are illegal).
        m100 = m(100)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            impl.forward(None, q1, q1, q1, kv, m100, output=o1)
        assert cls._gather_captured
        # Sk ping-pong: every replacement retires (sticky latch) — the
        # pre-fix unbounded behavior, kept byte-for-byte by the switch.
        cls._ensure_gather_buffers(1, HKV, 130, D, kv.device)
        assert gen1.data_ptr() in cls._gather_retired
        gen2 = cls._k_gather_buf
        cls._ensure_gather_buffers(1, HKV, 100, D, kv.device)
        assert gen2.data_ptr() in cls._gather_retired
        assert len(cls._gather_retired) == 2
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_gather_freeable_generation_exact_need():
    """plan §2.4 follow-up — a never-captured (freeable) generation is
    replaced at EXACT need, not grow-only max() per axis: 32-seq
    short-context decode followed by one long prefill must NOT leave a
    [32, wide] standing buffer (the B-highwater x Sk-highwater product,
    ~13 GB/rank at 256k on the arm-B geometry). Realloc frequency is
    unchanged by construction (a replacement happens exactly when the
    current buffer no longer fits); only the new allocation's shape is
    pinned here.
    """
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(29)
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FAImpl

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = False
    cls._gather_retired_warned = False
    bpr = (D // 32) * 34
    try:
        k, _ = cls._ensure_gather_buffers(32, HKV, 100, D, dev)
        assert k.shape == (32, HKV, 128, bpr)
        # Sk grow at B=1: freeable -> exact need, NOT [32, 1024].
        k, _ = cls._ensure_gather_buffers(1, HKV, 1000, D, dev)
        assert k.shape == (1, HKV, 1024, bpr)
        assert not cls._gather_retired
        # B grow at small Sk: likewise exact, NOT [32, 1024].
        k, _ = cls._ensure_gather_buffers(32, HKV, 100, D, dev)
        assert k.shape == (32, HKV, 128, bpr)
        assert not cls._gather_retired
        # Within-capacity requests never realloc (the fit path).
        k2, _ = cls._ensure_gather_buffers(8, HKV, 100, D, dev)
        assert k2.shape == (8, HKV, 128, bpr)
        assert k2.data_ptr() == k.data_ptr()
        assert not cls._gather_retired
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_gather_multi_retire_warns(monkeypatch):
    """plan §2.2b guard — retiring MORE than one capture-baked gather
    generation warns, one-shot. The original guard was dead code: it
    required `not capturing` while checking a flag that had just been
    set to `capturing`, so it could never fire. Two capture-then-B-grow
    cycles drive two retires; the third retire must not warn again.
    """
    import vllm.gfx906_fa.gfx906_fa_backend as backend_mod

    class _LoggerStub:
        def __init__(self):
            self.warnings = []

        def warning(self, msg, *args):
            self.warnings.append(msg % args if args else msg)

    stub = _LoggerStub()
    monkeypatch.setattr(backend_mod, "logger", stub)

    dev = "cuda"
    torch.manual_seed(31)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured,
             cls._gather_exact, cls._gather_retired_warned)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
    cls._gather_buf_captured = False
    cls._gather_exact = False
    cls._gather_retired_warned = False
    try:
        n_blocks = 128  # 16 seqs x 7 blocks
        _, vc, kv = _make_paged_cache(n_blocks, dev)
        k16 = kv[:, 0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v(kv, V)

        def meta(b, sk, nblk):
            return Gfx906FAMetadata(
                num_actual_tokens=b,
                max_query_len=1,
                max_seq_len=sk,
                query_start_loc=torch.arange(
                    0, b + 1, dtype=torch.int32, device=dev),
                seq_lens=torch.full((b,), sk, dtype=torch.int32,
                                    device=dev),
                block_table=torch.arange(
                    b * nblk, dtype=torch.int32, device=dev).view(b, nblk),
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        layer = None
        s = torch.cuda.Stream()

        def decode(b, q, out, m=None):
            impl.forward(layer, q, q, q, kv, m or meta(b, 100, 7),
                         output=out)

        qs = {b: torch.randn(b, HQ, D, device=dev,
                             dtype=torch.float16) * 0.5 for b in (2, 4, 8, 16)}
        os_ = {b: torch.zeros(b, HQ, D, device=dev,
                              dtype=torch.float16) for b in (2, 4, 8, 16)}

        # Cycle 1: eager gen1, capture bakes it, B-grow retires it.
        with torch.cuda.stream(s):
            decode(2, qs[2], os_[2])
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(2, qs[2], os_[2], meta(2, 100, 7))
        assert cls._gather_buf_captured
        with torch.cuda.stream(s):
            decode(4, qs[4], os_[4])
        torch.cuda.current_stream().wait_stream(s)
        assert len(cls._gather_retired) == 1 and not stub.warnings

        # Cycle 2: second capture bakes gen2, B-grow retires it -> len 2.
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(4, qs[4], os_[4], meta(4, 100, 7))
        assert cls._gather_buf_captured
        with torch.cuda.stream(s):
            decode(8, qs[8], os_[8])
        torch.cuda.current_stream().wait_stream(s)
        assert len(cls._gather_retired) == 2
        assert len(stub.warnings) == 1
        assert "capture-baked gather" in stub.warnings[0]

        # Cycle 3: one-shot — a third retire must not warn again.
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            decode(8, qs[8], os_[8], meta(8, 100, 7))
        with torch.cuda.stream(s):
            decode(16, qs[16], os_[16])
        torch.cuda.current_stream().wait_stream(s)
        assert len(cls._gather_retired) == 3
        assert len(stub.warnings) == 1
        assert bool(torch.isfinite(os_[16].float()).all())
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured,
         cls._gather_exact, cls._gather_retired_warned) = saved


def test_gather_mixed_width_buffers_not_reused():
    """plan §2.2c follow-up — the persistent branch's k/v capacity
    reuse requires EQUAL widths. A hand-set class buffer pair with
    unequal K/V widths (impossible via _ensure_gather_buffers, which
    allocates the pair at one width) must NOT be half-reused: without
    the width check, Sk = K's width would pass V by a width mismatch to
    the C++ exact-match check and silently drop V to a per-call
    allocation. The forward must fall back whole (bitwise-identical
    output, class buffers untouched).
    """
    dev = "cuda"
    torch.manual_seed(37)
    from vllm.gfx906_fa.gfx906_fa_backend import (
        Gfx906FAImpl,
        Gfx906FAMetadata,
    )

    impl = Gfx906FAImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="float16",
    )
    cls = type(impl)
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured, cls._gather_buf_captured)
    try:
        B, sk = 2, 100
        nblk = 7
        n_blocks = B * nblk + 4
        _, vc, kv = _make_paged_cache(n_blocks, dev)
        k16 = kv[:, 0]
        K = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        V = torch.randn(n_blocks * BLOCK, HKV, D, device=dev,
                        dtype=torch.float16) * 0.5
        k16.copy_(K.view(n_blocks, BLOCK, HKV, D))
        _write_v(kv, V)
        m = Gfx906FAMetadata(
            num_actual_tokens=B,
            max_query_len=1,
            max_seq_len=sk,
            query_start_loc=torch.arange(
                0, B + 1, dtype=torch.int32, device=dev),
            seq_lens=torch.full((B,), sk, dtype=torch.int32, device=dev),
            block_table=torch.arange(
                B * nblk, dtype=torch.int32, device=dev).view(B, nblk),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=dev),
        )
        q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
        out = torch.zeros(B, HQ, D, device=dev, dtype=torch.float16)

        # Reference: fresh exact-width class buffer.
        cls._k_gather_buf = cls._v_gather_buf = None
        cls._gather_retired = {}
        cls._gather_captured = False
        cls._gather_buf_captured = False
        impl.forward(None, q, q, q, kv, m, output=out)
        ref = out.clone()
        assert cls._k_gather_buf.shape[2] == 128  # Sk_pad(100)

        # Unequal-width pair (K wide, V exact): must be refused whole.
        bpr = (D // 32) * 34
        cls._k_gather_buf = torch.empty(B, HKV, 1024, bpr,
                                        dtype=torch.uint8, device=dev)
        cls._v_gather_buf = torch.empty(B, HKV, 128, D,
                                        dtype=torch.float16, device=dev)
        cls._gather_retired = {}
        cls._gather_buf_captured = False
        out.zero_()
        impl.forward(None, q, q, q, kv, m, output=out)
        assert torch.equal(out, ref), (
            "mixed-width class buffers changed the FA output — the "
            "persistent branch half-reused an unequal-width pair")
        # The pair was refused, not consumed or replaced.
        assert cls._k_gather_buf.shape[2] == 1024
        assert cls._v_gather_buf.shape[2] == 128
        assert not cls._gather_retired
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured, cls._gather_buf_captured) = saved


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
    out = fa.forward(q, k_q8, v_b, scale, kv_max=sl)[0, 0]  # [HQ, D] (BSHD)
    qg = q[0, :, 0].view(HKV, g, D)
    s = torch.einsum("gjd,lgd->gjl", qg, k) * scale
    ref = torch.einsum("gjl,lgd->gjd", torch.softmax(s, -1), v).reshape(HQ, D)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2

    # prefill: full causal chunk
    qf = torch.randn(1, HQ, L, D, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([0], dtype=torch.int32, device=dev)
    outf = fa.forward(qf, k_q8, v_b, scale, kv_max=sl, q_abs_offset=q_abs)[0]
    # native BSHD: [0] is already [L, HQ, D]
    assert outf.shape == (L, HQ, D)
    qtok = qf[0].permute(1, 0, 2).float()  # [L, HQ, D]
    for t in (1, 63, L - 1):
        qg = qtok[t].view(HKV, g, D)
        s = torch.einsum("gjd,lgd->gjl", qg, k[: t + 1]) * scale
        ref = torch.einsum(
            "gjl,lgd->gjd", torch.softmax(s, -1), v[: t + 1]
        ).reshape(HQ, D)
        assert ((outf[t] - ref).norm() / ref.norm()).item() < 5e-2


# The NC2/KVSPLIT knobs are parsed once per process (C++ statics on the
# first forward call), so each config runs in a fresh subprocess.
_SPLIT_CHECK_SRC = """
import math
import sys
import torch

nc2 = int(sys.argv[1])
ysplit = int(sys.argv[2])
sk = int(sys.argv[3])
kv_max = int(sys.argv[4])

torch.manual_seed(0)
dev = "cuda"
HQ, HKV, D = int(sys.argv[5]), int(sys.argv[6]), 256

from vllm import _gfx906_fa_C as fa

k16 = torch.randn(1, HKV, sk, D, device=dev, dtype=torch.float16) * 0.5
v16 = torch.randn(1, HKV, sk, D, device=dev, dtype=torch.float16) * 0.5
q32 = torch.randn(1, HQ, 1, D, device=dev, dtype=torch.float32) * 0.5
k_q8 = fa.quantize_q8_0(k16)
sl = torch.tensor([kv_max], dtype=torch.int32, device=dev)
out = fa.forward(q32, k_q8, v16, 1.0 / math.sqrt(D), kv_max=sl)[0, 0]  # BSHD

g = HQ // HKV
k, v = k16[0].float(), v16[0].float()  # [HKV, sk, D]
qg = q32[0, :, 0].view(HKV, g, D)
s = torch.einsum("gjd,gld->gjl", qg, k[:, :kv_max]) * (1.0 / math.sqrt(D))
ref = torch.einsum(
    "gjl,gld->gjd", torch.softmax(s, -1), v[:, :kv_max]
).reshape(HQ, D)
rel = ((out - ref).norm() / ref.norm()).item()
print(f"nc2={nc2} ys={ysplit} sk={sk} kv_max={kv_max} rel={rel:.2e}")
sys.exit(0 if rel < 5e-2 else 1)
"""


@pytest.mark.parametrize("nc2, ys, sk, kv_max, hq, hkv", [
    (1, 1, 512, 512, 16, 2),    # legacy path, no split (sanity in-subprocess)
    (1, 4, 512, 512, 16, 2),    # KV-split only
    (1, 4, 512, 481, 16, 2),    # KV-split with empty trailing splits (kv_max<sk)
    (8, 1, 512, 512, 16, 2),    # GQA head-packing only (no combine)
    (8, 16, 512, 512, 16, 2),   # serving config: GQA pack + KV-split
    (8, 16, 123, 123, 16, 2),   # short Sk: more splits than KV tiles
    (8, 16, 512, 481, 16, 2),   # serving config + empty trailing splits
    # heads_q=6 under default nc2=8 (e.g. Qwen3.5-27B at TP=4): the bare
    # heads_q%nc2 guard used to abort before the 8->2 downgrade; must now
    # downgrade to nc2=2 (ratio 6 % 2 == 0) and produce correct output.
    (8, 16, 512, 512, 6, 1),
    # heads_q=6 per-shard ratio with actual GQA (Hq=6/Hkv=3, ratio 2).
    (8, 16, 512, 512, 6, 3),
])
def test_forward_kv_split_gqa_pack_vs_fp32_ref(nc2, ys, sk, kv_max, hq, hkv):
    import os
    import subprocess

    env = {
        **os.environ,
        "GFX906_FA_NC2": str(nc2),
        "GFX906_FA_KVSPLIT": str(ys),
    }
    r = subprocess.run(
        [sys.executable, "-c", _SPLIT_CHECK_SRC, str(nc2), str(ys),
         str(sk), str(kv_max), str(hq), str(hkv)],
        env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr[-2000:]}"


# ---------------------------------------------------------------------------
# Sliding-window masking (window arg, Muse Glimmer iRoPE track).
# Unmasked keys per query row r: [max(0, r - W + 1), r] (window + causal).
# q_abs_offset is required for the window to apply (the backend always
# passes it for windowed batches, decode included).
# ---------------------------------------------------------------------------

def _windowed_ref(q_row, K, V, scale, r, W):
    """Torch reference for one query row at absolute position r.

    q_row: [Hq, D]; K/V: [HKV, L, D] fp32. Window W in tokens (W=None ->
    plain causal).
    """
    lo = max(0, r - W + 1) if W else 0
    hi = r + 1
    hkv = K.shape[1]
    g = q_row.shape[0] // hkv
    qg = q_row.view(hkv, g, -1)
    s = torch.einsum("gjd,lgd->gjl", qg, K[lo:hi]) * scale
    o = torch.einsum("gjl,lgd->gjd", torch.softmax(s, -1), V[lo:hi])
    return o.reshape(q_row.shape[0], -1)


@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),   # Muse Glimmer shape (Hq 32 / Hkv 2, D 128)
    (128, 32, 2, 512, 64),    # smaller window
    (128, 32, 2, 96, 128),    # W > L: window must be inert
    (256, 16, 2, 512, 128),   # D=256 path
])
def test_forward_sliding_window_vs_torch_ref(d, hq, hkv, L, W):
    dev = "cuda"
    torch.manual_seed(3)
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    sk_pad = (L + 31) // 32 * 32
    k_q8, v_b = fa.gather_paged_kv_q8(kc, vc, bt, sl, sk_pad)
    Kf, Vf = K.float(), V.float()

    # decode (Sq=1): q_abs_offset = L-1; window clips to [L-W, L-1]
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    out = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                     q_abs_offset=q_abs, window=W)[0, 0]
    ref = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2
    if L > W:
        # window must actually bite: differs from no-window attention
        out_nw = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                            q_abs_offset=q_abs)[0, 0]
        assert (out - out_nw).norm().item() > 1e-3
    else:
        # W >= L: windowed output matches no-window (inert)
        out_nw = fa.forward(q, k_q8, v_b, scale, kv_max=sl,
                            q_abs_offset=q_abs)[0, 0]
        assert ((out - out_nw).norm() / out.norm()).item() < 5e-2

    # prefill (Sq=L): per-row causal + window
    qf = torch.randn(1, hq, L, d, device=dev, dtype=torch.float32) * 0.5
    q_abs0 = torch.tensor([0], dtype=torch.int32, device=dev)
    outf = fa.forward(qf, k_q8, v_b, scale, kv_max=sl,
                      q_abs_offset=q_abs0, window=W)[0]
    assert outf.shape == (L, hq, d)
    qtok = qf[0].permute(1, 0, 2).float()  # [L, hq, d]
    rows = sorted({t for t in (0, W - 1, W, L - W, L - 1) if 0 <= t < L})
    for t in rows:
        ref = _windowed_ref(qtok[t], Kf, Vf, scale, t, W)
        assert ((outf[t] - ref).norm() / ref.norm()).item() < 5e-2, \
            f"row {t}"


# ---------------------------------------------------------------------------
# Same window semantics, DIRECT-PAGED kernel (forward_paged_direct,
# fattn-q8-paged.cuh) — the kernel the serving gate actually runs:
# _should_use_direct_paged is "auto" (min_batch=2), so every decode batch
# >= 2 (incl. the BENCH_MAX_SEQS=4 gate config) routes here, and the
# window formula is hand-duplicated in this file. Both files must match
# the torch reference.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),   # Muse Glimmer shape (Hq 32 / Hkv 2, D 128)
    (128, 32, 2, 512, 64),    # smaller window
    (128, 32, 2, 96, 128),    # W > L: window must be inert
    (256, 16, 2, 512, 128),   # D=256 path
])
def test_forward_paged_direct_sliding_window_vs_torch_ref(d, hq, hkv, L, W):
    dev = "cuda"
    torch.manual_seed(4)
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]  # production layout: unbind(1), non-contiguous

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    Kf, Vf = K.float(), V.float()

    # decode B=1 (Sq=1): q_abs_offset = L-1; window clips to [L-W, L-1]
    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    out = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W)[0, 0]
    ref = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out - ref).norm() / ref.norm()).item() < 5e-2
    if L > W:
        # window must actually bite: differs from no-window attention
        out_nw = fa.forward_paged_direct(
            q, kc, vc, bt, sl, scale, None, q_abs)[0, 0]
        assert (out - out_nw).norm().item() > 1e-3
    else:
        # W >= L: windowed output matches no-window (inert)
        out_nw = fa.forward_paged_direct(
            q, kc, vc, bt, sl, scale, None, q_abs)[0, 0]
        assert ((out - out_nw).norm() / out.norm()).item() < 5e-2

    # decode B=2, different lengths (the production direct mode): per-row
    # q_abs_offset/window must hold per batch element. seq1 reuses the
    # same physical blocks with a shorter kv_max.
    L2 = L * 3 // 4
    bt2 = bt.repeat(2, 1)
    sl2 = torch.tensor([L, L2], dtype=torch.int32, device=dev)
    q_abs2 = torch.tensor([L - 1, L2 - 1], dtype=torch.int32, device=dev)
    q2 = torch.randn(2, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    out2 = fa.forward_paged_direct(
        q2, kc, vc, bt2, sl2, scale, None, q_abs2, W)
    assert out2.shape == (2, 1, hq, d)
    ref0 = _windowed_ref(q2[0, :, 0], Kf, Vf, scale, L - 1, W)
    ref1 = _windowed_ref(q2[1, :, 0], Kf, Vf, scale, L2 - 1, W)
    assert ((out2[0, 0] - ref0).norm() / ref0.norm()).item() < 5e-2
    assert ((out2[1, 0] - ref1).norm() / ref1.norm()).item() < 5e-2

    # prefill (Sq=L): per-row causal + window
    qf = torch.randn(1, hq, L, d, device=dev, dtype=torch.float32) * 0.5
    q_abs0 = torch.tensor([0], dtype=torch.int32, device=dev)
    outf = fa.forward_paged_direct(
        qf, kc, vc, bt, sl, scale, None, q_abs0, W)[0]
    assert outf.shape == (L, hq, d)
    qtok = qf[0].permute(1, 0, 2).float()  # [L, hq, d]
    rows = sorted({t for t in (0, W - 1, W, L - W, L - 1) if 0 <= t < L})
    for t in rows:
        ref = _windowed_ref(qtok[t], Kf, Vf, scale, t, W)
        assert ((outf[t] - ref).norm() / ref.norm()).item() < 5e-2, \
            f"row {t}"


# ---------------------------------------------------------------------------
# Phase C: per-row kv_start clip (forward_paged_direct, decode). The k-loop
# walks [kv_start, L) instead of [0, L); bit-identical to the full scan
# when kv_start >= q_abs + 1 - window (prefix is window-masked anyway).
# The functional check below uses an INERT mask (window=L) plus a real
# kv_start to prove the scan itself shrinks, not just the mask.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, hq, hkv, L, W", [
    (128, 32, 2, 512, 128),
    (128, 32, 2, 512, 64),    # unaligned clip start (448 vs nbatch_fa=128)
    (128, 32, 2, 96, 128),    # W > L: clip inert (kv_start = 0)
    (256, 16, 2, 512, 128),
])
def test_forward_paged_direct_sliding_window_clip_vs_torch_ref(d, hq, hkv, L, W):
    dev = "cuda"
    torch.manual_seed(4)
    n_blocks = L // BLOCK
    kc = torch.zeros(n_blocks, BLOCK, hkv, (d // 32) * 34,
                     dtype=torch.uint8, device=dev)
    kv = torch.zeros(n_blocks, 2, BLOCK, hkv, d,
                     dtype=torch.float16, device=dev)
    K = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    V = torch.randn(L, hkv, d, device=dev, dtype=torch.float16) * 0.5
    slot = torch.arange(L, dtype=torch.int64, device=dev)
    fa.reshape_and_cache_q8(K, slot, kc)
    staging = torch.zeros_like(kv[:, 1])
    staging.view(-1, hkv, d)[:L].copy_(V)
    kv[:, 1].copy_(staging)
    vc = kv.unbind(1)[1]

    bt = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(1, n_blocks)
    sl = torch.tensor([L], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(d)
    Kf, Vf = K.float(), V.float()

    q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    q_abs = torch.tensor([L - 1], dtype=torch.int32, device=dev)
    kv_start = torch.tensor([max(0, L - W)], dtype=torch.int32, device=dev)

    out_full = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W)[0, 0]
    out_clip = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, W, kv_start)[0, 0]
    ref = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W)
    assert ((out_clip - ref).norm() / ref.norm()).item() < 5e-2
    # clip must be (near-)bit-identical to the masked full scan: the
    # skipped keys were exactly -INF in the softmax
    assert ((out_clip - out_full).norm() / out_full.norm()).item() < 1e-6

    # Functional: INERT window (W=L, mask does nothing) + real clip start
    # -> the scan itself must shrink to the last W2 keys.
    W2 = min(96, L - 1)
    kv_start2 = torch.tensor([L - W2], dtype=torch.int32, device=dev)
    out2 = fa.forward_paged_direct(
        q, kc, vc, bt, sl, scale, None, q_abs, L, kv_start2)[0, 0]
    ref2 = _windowed_ref(q[0, :, 0], Kf, Vf, scale, L - 1, W2)
    assert ((out2 - ref2).norm() / ref2.norm()).item() < 5e-2
    if L > W2:
        assert (out2 - out_full).norm().item() > 1e-3  # clip actually bit

    # B=2, different lengths: per-row clip
    L2 = L * 3 // 4
    bt2 = bt.repeat(2, 1)
    sl2 = torch.tensor([L, L2], dtype=torch.int32, device=dev)
    q_abs2 = torch.tensor([L - 1, L2 - 1], dtype=torch.int32, device=dev)
    kv_start2_2 = torch.tensor([max(0, L - W), max(0, L2 - W)],
                               dtype=torch.int32, device=dev)
    q2 = torch.randn(2, hq, 1, d, device=dev, dtype=torch.float32) * 0.5
    outb = fa.forward_paged_direct(
        q2, kc, vc, bt2, sl2, scale, None, q_abs2, W, kv_start2_2)
    ref0 = _windowed_ref(q2[0, :, 0], Kf, Vf, scale, L - 1, W)
    ref1 = _windowed_ref(q2[1, :, 0], Kf, Vf, scale, L2 - 1, W)
    assert ((outb[0, 0] - ref0).norm() / ref0.norm()).item() < 5e-2
    assert ((outb[1, 0] - ref1).norm() / ref1.norm()).item() < 5e-2
