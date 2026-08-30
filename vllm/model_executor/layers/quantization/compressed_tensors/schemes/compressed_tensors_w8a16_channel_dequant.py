# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""W8A16 channel-wise dense layers (compressed-tensors) on gfx906.

Nemotron-3.5-Lightning-30B-class checkpoints store the mamba/attention
dense projections and lm_head as symmetric bias-128 int8 with
per-output-channel scales (packed 4 int8 per int32 word along K,
little-endian ascending). The only stock ROCm mixed-precision kernel
for this layout (ConchLinearKernel) costs ~3.8 ms per M=1 GEMV on MI50
(75% of Nemotron decode GPU time across the 46 dense projections), so
this scheme has two paths:

int8 in-kernel (default on gfx906 + fp16; NH-2):
    the packed bytes are kept as a uint8 view of the packed storage —
    no fp16 materialization, 1 byte/weight. The Triton GEMV (M=1) and
    GEMM (M>1) kernels dequant in-register ((w - 128) * scale). The
    GEMM's tile dequant is bit-identical to the old load-time dequant,
    so M>1 numerics are the dequant path's numerics. Halves the dense
    weight traffic vs the dequant path (1.32 GB/step -> 2.65 GB/step
    on Nemotron) and retires the dequant VRAM. Kill switch:
    VLLM_GFX906_W8A16_INT8=0.

load-time dequant (fallback, and all non-gfx906 ROCm):
    the packed int8 is dequantized to fp16 once at load and the
    optimized gfx906 unquantized GEMV family runs on it (2x the int8
    weight bytes in VRAM; prefill is a plain GEMM).

Both paths use the same convention: w = (byte - 128) * scale.
"""

import os

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)

__all__ = ["CompressedTensorsW8A16ChannelDequant"]

_DEQUANT_CHUNK_ROWS = 8192
_ENV_INT8 = "VLLM_GFX906_W8A16_INT8"


@triton.jit
def k_w8a16_gemv(
    x_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    N,
    K,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """M=1 GEMV over bias-128 int8 rows: out[n] = s[n] * sum_k (w[n,k]-128) x[k].

    w is uint8 [N, K]; s is fp16 [N]; out is fp16 [N]. For SPLIT > 1 the
    caller zero-initializes out and partials are atomic-added.
    """
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    rmask = rows < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    per = K // SPLIT
    for k0 in range(pk * per, (pk + 1) * per, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < (pk + 1) * per
        x = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + rows[:, None] * K + ks[None, :],
            mask=rmask[:, None] & kmask[None, :],
            other=0,
        )
        acc += tl.sum((w.to(tl.float32) - 128.0) * x[None, :], axis=1)
    s = tl.load(s_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    acc = acc * s
    if SPLIT == 1:
        tl.store(o_ptr + rows, acc.to(o_ptr.dtype.element_ty), mask=rmask)
    else:
        tl.atomic_add(o_ptr + rows, acc.to(o_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def k_w8a16_gemm(
    x_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """M>1 W8A16 GEMM: dequant each weight tile in-register to fp16
    (bit-identical to the load-time dequant: f32 (w-128)*s -> f16) and
    tl.dot with fp32 accumulation. Requires K % BK == 0 (checked by the
    launcher; K is a multiple of the pack factor 4).
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + ks[None, :], mask=mmask[:, None], other=0.0
        )
        w = tl.load(w_ptr + rn[None, :] * K + ks[:, None], mask=nmask[None, :], other=0)
        s = tl.load(s_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        w16 = ((w.to(tl.float32) - 128.0) * s[None, :]).to(tl.float16)
        acc = tl.dot(x, tl.trans(w16), acc)
    tl.store(
        o_ptr + rm[:, None] * N + rn[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=mmask[:, None] & nmask[None, :],
    )


def _gemv_geometry(n: int, k: int) -> tuple[int, int, int]:
    """(BN, BK, SPLIT) for the M=1 GEMV: enough (row, k-split) programs to
    fill the 60 CUs, and BK must divide the per-split K slice exactly (an
    unaligned chunk silently skips the remainder — a correctness bug, not
    a crash). Splitting is rejected when the resulting slice cannot take a
    BK >= 32 (avoids degenerate 16-wide tiles on odd K)."""
    bn = 32 if n >= 4096 else 16
    split = 1
    while split < 8 and k % (split * 2) == 0:
        per = k // (split * 2)
        bk_c = 512
        while bk_c > 32 and (bk_c > per or per % bk_c):
            bk_c //= 2
        if per % bk_c:
            break
        split *= 2
        if triton.cdiv(n, bn) * split >= 240:
            break
    per = k // split
    bk = 512
    while bk > 16 and (bk > per or per % bk):
        bk //= 2
    assert per % bk == 0, f"N={n} K={k}: no aligned BK for split={split}"
    return bn, bk, split


def _gemm_geometry(m: int, k: int) -> tuple[int, int, int]:
    """(BM, BN, BK) for the M>1 GEMM: K is a multiple of 4, so a BK in
    {64, 32, 16} always divides it (16 only for exotic K)."""
    bm = 16 if m <= 16 else 64
    for bk in (64, 32, 16):
        if k % bk == 0:
            return bm, 64, bk
    raise ValueError(f"W8A16 GEMM: K={k} not a multiple of 16")


def w8a16_gemv(
    w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """M=1: x [1, K] fp16 -> out [N] fp16. w uint8 [N, K] (the packed
    storage viewed as bytes), scale fp16 [N]."""
    n, k = w.shape
    bn, bk, split = _gemv_geometry(n, k)
    dev = w.device
    out = torch.zeros(n, dtype=scale.dtype, device=dev) if split > 1 else torch.empty(
        n, dtype=scale.dtype, device=dev
    )
    k_w8a16_gemv[(triton.cdiv(n, bn), split)](
        x, w, scale, out, n, k, BN=bn, BK=bk, SPLIT=split, num_warps=8
    )
    return out


def w8a16_gemm(
    w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """M>1: x [M, K] fp16 -> out [M, N] fp16."""
    n, k = w.shape
    m = x.shape[0]
    bm, bn, bk = _gemm_geometry(m, k)
    out = torch.empty(m, n, dtype=scale.dtype, device=w.device)
    k_w8a16_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, w, scale, out, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=4
    )
    return out


class CompressedTensorsW8A16ChannelDequant(CompressedTensorsScheme):
    """Symmetric pack-quantized int8 channel-wise weights.

    On gfx906 with fp16 params the packed bytes are served in-kernel
    (NH-2, see module docstring); elsewhere they are dequantized to
    ``params_dtype`` in ``process_weights_after_loading`` and the
    unquantized GEMV family runs on the result.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        # Selection is platform-gated (gfx906 only); mirror the WNA16 gate
        # for the _check_scheme_supported path.
        return 75

    def __init__(self, layer_name: str | None = None):
        self.layer_name = layer_name

    def _int8_path(self, params_dtype: torch.dtype) -> bool:
        from vllm.platforms.rocm import on_gfx906

        return (
            on_gfx906()
            and params_dtype == torch.float16
            and os.environ.get(_ENV_INT8, "1") != "0"
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        layer.serve_int8 = self._int8_path(params_dtype)
        if not hasattr(layer, "has_bias"):
            layer.has_bias = False

        pack_factor = 4  # 32 / num_bits, num_bits == 8
        if input_size_per_partition % pack_factor != 0:
            raise ValueError(
                "W8A16 channel dequant requires input_size_per_partition "
                f"divisible by 4 (got {input_size_per_partition})"
            )

        # [N, K/4] int32, 4 int8 values per word along K (little-endian,
        # ascending k) — the compressed-tensors pack-quantized layout.
        packed_input_dim = input_size_per_partition // pack_factor
        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                packed_input_dim,
                dtype=torch.int32,
            ),
        )

        # Channel scales are per output row [N, 1]: column-parallel layers
        # load their row slice, row-parallel layers keep all rows (matches
        # the WNA16 channel convention — scales follow the output dim only).
        weight_scale = ChannelQuantScaleParameter(
            output_dim=0,
            weight_loader=weight_loader,
            data=torch.empty(output_size_per_partition, 1, dtype=params_dtype),
        )

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_packed = layer.weight_packed.data
        scale = layer.weight_scale.data
        n, k_words = w_packed.shape
        k = k_words * 4
        dtype = scale.dtype

        if scale.shape != (n, 1):
            raise ValueError(
                f"W8A16 channel dequant expects scale shape [{n}, 1], "
                f"got {tuple(scale.shape)}"
            )

        if getattr(layer, "serve_int8", False):
            # uint8 view of the packed storage: [N, K/4, 4] -> [N, K],
            # no copy (byte b of word i is k = 4*i + b, ascending).
            w_u8 = w_packed.view(torch.uint8).reshape(n, k)
            layer.weight_i8 = torch.nn.Parameter(w_u8, requires_grad=False)
            # weight_packed is the storage; keep it alive. weight_shape is
            # load metadata only.
            del layer.weight_shape
            return

        # uint8 view: [N, K] with ascending k within each int32 word
        # (little-endian bytes). Symmetric int8 uses the bias-128
        # convention: w = (q - 128) * scale.
        w_u8 = w_packed.view(torch.uint8)
        dequant = torch.empty(n, k, dtype=dtype, device=w_packed.device)
        for row0 in range(0, n, _DEQUANT_CHUNK_ROWS):
            row1 = min(row0 + _DEQUANT_CHUNK_ROWS, n)
            chunk = w_u8[row0:row1].to(torch.float32)
            chunk -= 128.0
            chunk *= scale[row0:row1].to(torch.float32)
            dequant[row0:row1] = chunk.to(dtype)

        layer.weight = torch.nn.Parameter(dequant, requires_grad=False)
        # Free the packed tensors: the dequantized weight replaces them.
        del layer.weight_packed
        del layer.weight_scale
        del layer.weight_shape

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if getattr(layer, "serve_int8", False):
            if bias is not None:
                raise ValueError(
                    f"{self.layer_name}: the gfx906 W8A16 int8 in-kernel "
                    "path does not support bias"
                )
            if x.dtype != torch.float16:
                raise ValueError(
                    f"{self.layer_name}: the gfx906 W8A16 int8 in-kernel "
                    f"path requires fp16 activations (got {x.dtype})"
                )
            x2d = x.reshape(-1, x.shape[-1])
            if not x2d.is_contiguous():
                x2d = x2d.contiguous()
            w = layer.weight_i8.data
            s = layer.weight_scale.data
            if x2d.shape[0] == 1:
                out = w8a16_gemv(w, s, x2d[0].contiguous())
            else:
                out = w8a16_gemm(w, s, x2d)
            return out.view(*x.shape[:-1], out.shape[-1])

        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
