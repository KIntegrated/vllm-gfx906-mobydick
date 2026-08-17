# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Correctness test for the gfx906 WNA16 MoE grouped-GEMM kernel.

Moved from /tmp/bench/_test_gfx906_moe.py into the repo (Common Protocol item 1
of plan-moe-phase2.md). Exercises _rocm_C.moe_gptq_gemm_gfx906 end to end:
repack via the oracle, moe_align_block_size routing, both gemm passes (w13
scatter + w2 fused topk-weight/reduce), against a naive per-expert torch
reference.

Run (on a gfx906 host / with the ROCm image):
    pytest tests/kernels/moe/test_gfx906_moe_gemm.py -v
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _repack_w4a16_gfx906_expert,
)

# small expert count is fine: the kernel is per-expert for routing
E = 16
TOPK = 4
GS = 128
DBLL = "gfx"  # (unused; kept for readability in parametrize ids)


def _make_layer(N, K, layout):
    """Random W4A16 weights in the requested source layout."""
    G = K // GS
    sh = 1 << (4 * torch.arange(8, device=torch.device("cuda")))
    q = torch.randint(0, 16, (E, K, N), device=torch.device("cuda"), dtype=torch.int32)
    z = torch.randint(0, 16, (E, G, N), device=torch.device("cuda"), dtype=torch.int32)
    s = torch.rand(E, G, N, device=torch.device("cuda"), dtype=torch.float16) * 0.05 + 0.01
    if layout == "awq_kfirst":
        # AutoAWQ K-first int32: word m holds n=8m..8m+7 (low nibble first)
        w = (q.view(E, K, N // 8, 8) * sh).sum(-1).to(torch.int32)
        zw = (z.view(E, G, N // 8, 8) * sh).sum(-1).to(torch.int32)
        return w, s, zw, q, z
    # MoeWNA16 N-first uint8: byte j holds k=2j low / 2j+1 high;
    # zp byte i holds n=2i low / 2i+1 high
    qn = q.permute(0, 2, 1).reshape(E, N, K // 2, 2)
    w = (qn[..., 0] | (qn[..., 1] << 4)).to(torch.uint8)
    sc = s.permute(0, 2, 1).contiguous()
    zn = z.permute(0, 2, 1).reshape(E, N // 2, 2, G)
    zw = (zn[..., 0, :] | (zn[..., 1, :] << 4)).to(torch.uint8)
    return w, sc, zw, q, z.permute(0, 2, 1)


def _dequant_ref(w, s, z, q):
    """[E,N,K] fp32 dequantized weights: (q - z[g(k), n]) * s[g(k), n]."""
    E_, K, N = q.shape
    g = torch.arange(K, device=q.device) // GS  # [K]
    if s.shape[1] == N:  # MoeWNA16 scales/zp: [E, N, G]
        st = s.transpose(1, 2)[:, g, :]  # [E, K, N]
        zt = z.transpose(1, 2)[:, g, :]  # [E, K, N]
        return ((q.long() - zt.long()) * st.float()).permute(0, 2, 1)
    return ((q.long() - z[:, g].long()) * s[:, g].float()).permute(0, 2, 1)


def _run_case(M, N13, K13, N2, K2, block_m, layout):
    from vllm import _custom_ops as ops

    w13, s13, z13, q13, zz13 = _make_layer(N13, K13, layout)
    w2, s2, z2, q2, zz2 = _make_layer(N2, K2, layout)
    wq13, sc13, zp13 = _repack_w4a16_gfx906_expert(w13, s13, z13)
    wq2, sc2, zp2 = _repack_w4a16_gfx906_expert(w2, s2, z2)

    x = torch.randn(M, K13, dtype=torch.float16) * 0.5
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32)
    topk_w = torch.rand(M, TOPK, dtype=torch.float16)
    dev = "cuda"
    x = x.to(dev)
    topk_ids = topk_ids.to(dev)
    topk_w = topk_w.to(dev)

    sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, block_m, E)

    # ---- gemm1 (w13): [M, K13] -> [M*TOPK, N13] scatter ----
    c1 = torch.zeros(M * TOPK, N13, device=dev, dtype=torch.float16)
    empty_tw = torch.empty(0, dtype=torch.float32, device=dev)
    ops.moe_gptq_gemm_gfx906(
        x, c1, wq13, sc13, zp13, empty_tw, sorted_ids, expert_ids, ntp,
        TOPK, block_m, False, 0, 0)

    ref_rows = torch.zeros(M * TOPK, N13, device=dev, dtype=torch.float32)
    for e in range(E):
        sel = (topk_ids == e)
        if not sel.any():
            continue
        rows = sel.nonzero()
        tok = rows[:, 0]
        wdeq = _dequant_ref(w13, s13, zz13, q13)[e]
        ref_rows[tok * TOPK + rows[:, 1]] = x[tok].float() @ wdeq.t()
    err1 = (c1.float() - ref_rows).abs().max().item()
    rel1 = err1 / ref_rows.abs().max().item()

    # ---- activation: silu_and_mul on c1 -> [M*TOPK, N13/2] ----
    inter = (torch.nn.functional.silu(c1[:, : N13 // 2].float()) *
             c1[:, N13 // 2:].float()).half().contiguous()

    # ---- gemm2 (w2): [M*TOPK, K2] -> [M, N2] fused weight + reduce ----
    assert N13 // 2 == K2, "test shape mismatch"
    out = torch.zeros(M, N2, device=dev, dtype=torch.float16)
    ops.moe_gptq_gemm_gfx906(
        inter, out, wq2, sc2, zp2, topk_w.view(-1).float(), sorted_ids, expert_ids,
        ntp, 1, block_m, True, TOPK, 0)

    wdeq2 = _dequant_ref(w2, s2, zz2, q2)
    ref_out = torch.zeros(M, N2, device=dev, dtype=torch.float32)
    for e in range(E):
        sel = (topk_ids == e)
        if not sel.any():
            continue
        rows = sel.nonzero()
        tok = rows[:, 0]
        h = inter[rows[:, 0] * TOPK + rows[:, 1]].float() @ wdeq2[e].t()
        h *= topk_w[tok, rows[:, 1]].float().unsqueeze(1)
        for i in range(len(tok)):
            ref_out[tok[i]] += h[i]
    err2 = (out.float() - ref_out).abs().max().item()
    rel2 = err2 / ref_out.abs().max().item()

    assert rel1 < 5e-2, f"gemm1 too far from reference: maxrel={rel1:.2e}"
    assert rel2 < 5e-2, f"gemm2 too far from reference: maxrel={rel2:.2e}"
    return rel1, rel2


# (M, N13, K13, N2, K2, block_m): Qwen-like shapes scaled to E=16 experts
_CASES = [
    (1, 1024, 2048, 1024, 512, 1),
    (4, 1024, 2048, 1024, 512, 1),
    (8, 1024, 2048, 1024, 512, 4),
    (32, 1024, 2048, 1024, 512, 4),
    (64, 1024, 2048, 1024, 512, 16),
    (2, 1536, 768, 1536, 768, 1),  # K not mult of 256; N=1536 (gridY partial)
]


def _ids(c):
    return f"M{c[0]}-bm{c[5]}-N{c[1]}"


@pytest.mark.parametrize("layout", ["awq_kfirst", "wna16"])
@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_gfx906_moe_gemm(case, layout):
    _run_case(*case, layout)


if __name__ == "__main__":
    print("run via pytest")