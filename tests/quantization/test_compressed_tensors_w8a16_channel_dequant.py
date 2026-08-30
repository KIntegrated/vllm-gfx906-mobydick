# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Unit test for the gfx906 W8A16 channel dequant scheme.

Verifies the pack-quantized int8 -> fp16 dequant is bit-exact against a
plain-tensor reference (bias-128 symmetric convention, 4 int8 per int32
word along K, little-endian/ascending), that the packed tensors are
freed after processing, and that a wrong scale shape fails closed.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A16ChannelDequant,
)


class _Layer(torch.nn.Module):
    pass


def _make_layer(n: int, k: int, device: str, dtype: torch.dtype):
    scheme = CompressedTensorsW8A16ChannelDequant(layer_name="test.w8ch")
    layer = _Layer()
    scheme.create_weights(
        layer,
        output_size=n,
        input_size=k,
        output_partition_sizes=[n],
        input_size_per_partition=k,
        params_dtype=dtype,
        weight_loader=lambda *a, **kw: None,
    )
    return scheme, layer


def test_w8a16_channel_dequant_bit_exact(dist_init):
    torch.manual_seed(0)
    n, k = 256, 768
    scheme, layer = _make_layer(n, k, "cpu", torch.float16)

    q = torch.randint(0, 256, (n, k), dtype=torch.int32)
    scale = (torch.rand(n, 1) * 0.05 + 0.005).half()
    packed = (
        (q[:, 0::4] & 0xFF)
        | ((q[:, 1::4] & 0xFF) << 8)
        | ((q[:, 2::4] & 0xFF) << 16)
        | ((q[:, 3::4] & 0xFF) << 24)
    )
    layer.weight_packed = torch.nn.Parameter(
        (packed & 0xFFFFFFFF).to(torch.int32).contiguous(), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

    scheme.process_weights_after_loading(layer)

    ref = ((q - 128).float() * scale.float()).half()
    assert layer.weight.data.shape == (n, k)
    assert torch.equal(layer.weight.data, ref)
    # packed tensors must be freed
    for name in ("weight_packed", "weight_scale", "weight_shape"):
        assert name not in dict(layer.named_parameters())


def test_w8a16_channel_dequant_rejects_bad_scale_shape(dist_init):
    torch.manual_seed(0)
    n, k = 64, 256
    scheme, layer = _make_layer(n, k, "cpu", torch.float16)

    layer.weight_packed = torch.nn.Parameter(
        torch.zeros(n, k // 4, dtype=torch.int32), requires_grad=False
    )
    # group-shaped scales are not channel-wise: must fail closed
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros(n, k // 64, dtype=torch.float16), requires_grad=False
    )
    with pytest.raises(ValueError, match=r"\[64, 1\]"):
        scheme.process_weights_after_loading(layer)


def test_w8a16_channel_dequant_rejects_bad_k():
    n, k = 32, 254  # not divisible by the pack factor
    with pytest.raises(ValueError, match="divisible by 4"):
        _make_layer(n, k, "cpu", torch.float16)
