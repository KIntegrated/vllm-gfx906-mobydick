# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""gfx906 fallback for compressed-tensors W8A16 channel-wise dense layers.

The only stock ROCm mixed-precision kernel that implements channel-wise
8-bit (ConchLinearKernel) costs ~3.8 ms per M=1 GEMV on MI50 (measured on
Nemotron-3.5-Lightning-30B: 75% of decode GPU time across the 46
mamba/attention dense projections). Instead, dequantize the packed int8
weights to plain fp16 at load time and run the optimized gfx906
unquantized GEMV family (dense_gemv_gfx906 / LLMM1 / long-K GEMV).

Cost: 2x the int8 weight bytes in VRAM (~+1.3 GiB for a 30B-class
checkpoint's 71 dense tensors incl. lm_head); the packed tensors are
freed after dequant. Prefill pays nothing (plain GEMM), decode pays
2x weight traffic vs a hypothetical int8 GEMV, which is still ~50x
faster than the Conch path it replaces.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A16ChannelDequant"]

_DEQUANT_CHUNK_ROWS = 8192


class CompressedTensorsW8A16ChannelDequant(CompressedTensorsScheme):
    """Symmetric pack-quantized int8 channel-wise weights, dequantized
    to ``params_dtype`` in ``process_weights_after_loading``.

    Selected only on gfx906 for configs the MP-linear kernel set cannot
    serve competitively; see the module docstring.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        # Selection is platform-gated (gfx906 only); mirror the WNA16 gate
        # for the _check_scheme_supported path.
        return 75

    def __init__(self, layer_name: str | None = None):
        self.layer_name = layer_name

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
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
