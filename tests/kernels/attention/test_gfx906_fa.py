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


def test_gather_buffers_capture_sweep_keepalive():
    """2026-08-19 init Memory Fault: FULL-graph capture used to bake a
    separate gather-buffer generation per captured batch size, and the
    bounded keep-alive evicted early-captured generations that replaying
    graphs still referenced by VA. Two properties are pinned here by
    driving the real Gfx906FAImpl._ensure_gather_buffers:
    (a) a smaller-B capture request reuses the current buffer as a
    leading-dim slice (same base VA, no new generation per size), and
    (b) generations whose VA was baked into a captured graph are never
    freed, even when later same-shape generations are retired (the
    retire dict is keyed by data_ptr, so same-shape entries cannot
    collide and overwrite each other).
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
    assert impl._legacy
    cls = type(impl)

    # The gather buffers are ClassVars shared across impls: snapshot and
    # restore so this test cannot leak generations into other tests.
    saved = (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
             cls._gather_captured)
    cls._k_gather_buf = cls._v_gather_buf = None
    cls._gather_retired = {}
    cls._gather_captured = False
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

        def meta(b, sq, sk, nblk_per_seq):
            return Gfx906FAMetadata(
                num_actual_tokens=b * sq,
                max_query_len=sq,
                max_seq_len=sk,
                query_start_loc=torch.arange(b + 1, dtype=torch.int32,
                                             device=dev),
                seq_lens=torch.full((b,), sk, dtype=torch.int32,
                                    device=dev),
                block_table=torch.arange(
                    b * nblk_per_seq, dtype=torch.int32,
                    device=dev).view(b, nblk_per_seq),
                slot_mapping=torch.empty(0, dtype=torch.int64,
                                         device=dev),
            )

        layer = None  # impl.forward does not touch the layer object
        s = torch.cuda.Stream()

        def decode(b, sk, nblk, q, out, m=None):
            impl.forward(layer, q, q, q, kv,
                         m or meta(b, 1, sk, nblk), output=out)

        q1 = torch.randn(1, HQ, D, device=dev, dtype=torch.float16) * 0.5
        q2 = torch.randn(2, HQ, D, device=dev, dtype=torch.float16) * 0.5
        o1 = torch.zeros(1, HQ, D, device=dev, dtype=torch.float16)
        o2 = torch.zeros(2, HQ, D, device=dev, dtype=torch.float16)

        with torch.cuda.stream(s):
            decode(1, 100, 7, q1, o1)  # gen1 [1,2,128,bpr] (Sk_pad(100)=128)
            ref1 = o1.clone()
            decode(2, 100, 7, q2, o2)  # gen2 [2,2,128,bpr]; gen1 never
            ref2 = o2.clone()          #   captured -> correctly freed
        torch.cuda.current_stream().wait_stream(s)
        gen2 = cls._k_gather_buf
        assert gen2.shape[:2] == (2, HKV) and gen2.shape[2] == 128

        # Capture with pre-built metadata (capture-time allocations would
        # land in the graph pool; the engine pins static tensors instead).
        m2_cap = meta(2, 1, 100, 7)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            decode(2, 100, 7, q2, o2, m2_cap)
        assert cls._gather_captured  # latched during capture
        assert cls._k_gather_buf is gen2

        # (a) smaller-B request reuses the same base VA (slice, no new
        # generation) — the old code allocated one per captured size.
        # Pass the indexed device (query.device in production) — the
        # equality check is exact, an index-less device would force a
        # spurious realloc.
        k_slice = cls._ensure_gather_buffers(1, HKV, 100, D,
                                             kv.device)[0]
        assert k_slice.data_ptr() == gen2.data_ptr()
        assert cls._k_gather_buf is gen2

        m1_cap = meta(1, 1, 100, 7)
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            decode(1, 100, 7, q1, o1, m1_cap)

        # Post-capture churn: B grow, then Sk ping-pong that recreates
        # SAME-SHAPED generations (the shape-key collision hazard). Each
        # step retires the previous generation; all of them must survive.
        q4 = torch.randn(4, HQ, D, device=dev, dtype=torch.float16) * 0.5
        o4 = torch.zeros(4, HQ, D, device=dev, dtype=torch.float16)
        retired = []

        def churn(b, sk, nblk, q, out):
            before = cls._k_gather_buf
            decode(b, sk, nblk, q, out)
            if cls._k_gather_buf is not before:
                retired.append(before)
            assert bool(torch.isfinite(out.float()).all())

        churn(4, 100, 7, q4, o4)    # retires gen2 (captured!)
        churn(1, 130, 9, q1, o1)    # [1,160]
        churn(1, 100, 7, q1, o1)    # [1,128]
        churn(1, 130, 9, q1, o1)    # retires [1,128] (gen2's sibling shape)
        churn(2, 100, 7, q2, o2)    # [2,128] — same shape as captured gen2
        churn(2, 130, 9, q2, o2)    # retires the [2,128] twin above

        # (b) the captured generation and every retired twin are all
        # still referenced — nothing baked into g1/g2 was freed.
        kept = [pair[0] for pair in cls._gather_retired.values()]
        assert any(g is gen2 for g in kept)
        assert all(any(g is k for k in kept) for g in retired)
        assert len(cls._gather_retired) == len(set(
            g.data_ptr() for g in kept)) == len(retired)

        # Replaying both graphs hits the retired-but-alive base VA.
        o1.zero_()
        o2.zero_()
        g2.replay()
        g1.replay()
        torch.cuda.synchronize()
        assert ((o2 - ref2).norm() / ref2.norm()).item() < 2e-2
        assert ((o1 - ref1).norm() / ref1.norm()).item() < 2e-2
    finally:
        (cls._k_gather_buf, cls._v_gather_buf, cls._gather_retired,
         cls._gather_captured) = saved


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
