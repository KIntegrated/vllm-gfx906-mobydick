# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Unit tests for the gfx906 W8A16 channel scheme (dequant + int8 in-kernel
paths).

Verifies the pack-quantized int8 -> fp16 dequant is bit-exact against a
plain-tensor reference (bias-128 symmetric convention, 4 int8 per int32
word along K, little-endian/ascending), that the packed tensors are
freed after processing on the dequant path, that the int8 path keeps a
no-copy uint8 view with the same convention, and that wrong inputs fail
closed.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A16ChannelDequant,
)

_ENV_INT8 = "VLLM_GFX906_W8A16_INT8"


class _Layer(torch.nn.Module):
    pass


def _make_layer(
    n: int,
    k: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch | None = None,
    gfx906: bool = True,
    env: str | None = None,
):
    if monkeypatch is not None:
        monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: gfx906)
        if env is not None:
            monkeypatch.setenv(_ENV_INT8, env)
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


def _packed_layer(n: int, k: int):
    """Fill a created layer with a known packed pattern + channel scales."""
    torch.manual_seed(0)
    scheme, layer = _make_layer(n, k, torch.float16)
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
    return scheme, layer, q, scale


def test_w8a16_channel_dequant_bit_exact(dist_init, monkeypatch):
    monkeypatch.setenv(_ENV_INT8, "0")
    scheme, layer, q, scale = _packed_layer(256, 768)

    scheme.process_weights_after_loading(layer)

    ref = ((q - 128).float() * scale.float()).half()
    assert layer.weight.data.shape == (256, 768)
    assert torch.equal(layer.weight.data, ref)
    # packed tensors must be freed
    for name in ("weight_packed", "weight_scale", "weight_shape"):
        assert name not in dict(layer.named_parameters())


def test_w8a16_channel_int8_view(dist_init, monkeypatch):
    scheme, layer, q, scale = _packed_layer(256, 768)
    assert layer.serve_int8

    scheme.process_weights_after_loading(layer)

    w = layer.weight_i8.data
    assert w.dtype == torch.uint8
    assert w.shape == (256, 768)
    # no-copy view of the packed storage
    assert w.data_ptr() == layer.weight_packed.data.data_ptr()
    assert w.is_contiguous()
    # byte convention: byte b of word j is (word >> 8b) & 0xFF
    words = layer.weight_packed.data
    for j in range(0, 768 // 4, 192):
        for b in range(4):
            expect = (words[:, j] >> (8 * b)) & 0xFF
            assert torch.equal(w[:, 4 * j + b].long(), expect.long())
    # matches the dequant-path numerics under the bias-128 convention
    ref = ((w.long() - 128).float() * scale.float())
    dequant = ((q - 128).float() * scale.float())
    assert torch.equal(ref, dequant)
    # weight_scale is kept; weight_shape is load metadata, freed
    assert "weight_scale" in dict(layer.named_parameters())
    assert "weight_shape" not in dict(layer.named_parameters())


def test_w8a16_channel_dequant_fallback_when_env_off(dist_init, monkeypatch):
    monkeypatch.setenv(_ENV_INT8, "0")
    scheme, layer, q, scale = _packed_layer(128, 512)
    assert not layer.serve_int8

    scheme.process_weights_after_loading(layer)

    assert layer.weight.data.shape == (128, 512)
    assert not hasattr(layer, "weight_i8")


def test_w8a16_channel_int8_path_requires_fp16(dist_init, monkeypatch):
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: True)
    scheme, layer = _make_layer(64, 256, torch.bfloat16)
    assert not layer.serve_int8  # bf16 models keep the dequant path


def test_w8a16_channel_dequant_rejects_bad_scale_shape(dist_init, monkeypatch):
    monkeypatch.setenv(_ENV_INT8, "0")
    torch.manual_seed(0)
    n, k = 64, 256
    scheme, layer = _make_layer(n, k, torch.float16)

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
        _make_layer(n, k, torch.float16)
