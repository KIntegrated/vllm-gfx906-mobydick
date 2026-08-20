# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _backend_incompatibility_reason,
    _convert_moe_wna16_humming_tensors,
    convert_to_wna16_moe_kernel_format,
    map_wna16_backend,
)
from vllm.model_executor.layers.quantization import moe_wna16
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)


def test_map_wna16_backend_supports_triton():
    assert map_wna16_backend("triton") == WNA16MoEBackend.TRITON


@pytest.mark.parametrize(
    ("backend", "quant_config", "may_have_zp", "may_have_bias", "expected"),
    [
        (
            WNA16MoEBackend.TRITON,
            AutoAWQConfig(4, 128, True, False),
            True,
            False,
            "AutoAWQ weight layout",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, True, True, False, {}, {}),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            True,
            "bias",
        ),
        (
            WNA16MoEBackend.MARLIN,
            MoeWNA16Config(
                linear_quant_method="gptq",
                weight_bits=4,
                group_size=128,
                has_zp=False,
                lm_head_quantized=False,
                modules_to_not_convert=None,
                full_config={},
            ),
            False,
            False,
            "MoeWNA16 checkpoint layout",
        ),
        (
            WNA16MoEBackend.GFX906_HIP,
            # Symmetric GPTQ has no stored zero points.
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            False,
            "zero points are required",
        ),
        (
            WNA16MoEBackend.GFX906_HIP,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
                actorder=None,
            ),
            True,
            False,
            "GPTQ-style zero-point",
        ),
    ],
)
def test_wna16_oracle_rejects_incompatible_quant_structures(
    backend, quant_config, may_have_zp, may_have_bias, expected
):
    from tests.kernels.moe.utils import make_dummy_moe_config

    moe_config = make_dummy_moe_config()

    reason = _backend_incompatibility_reason(
        backend=backend,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=may_have_zp,
        may_have_bias=may_have_bias,
        allow_tile_padding=True,
    )

    assert reason is not None
    assert expected in reason


@pytest.mark.parametrize(
    "quant_config",
    [
        AutoAWQConfig(4, 128, True, False),
        MoeWNA16Config(
            linear_quant_method="awq",
            weight_bits=4,
            group_size=128,
            has_zp=True,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        ),
    ],
)
def test_gfx906_hip_oracle_accepts_awq_style_zero_points(quant_config):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Realistic Qwen3.5-A3B MoE shapes: the gfx906 kernel's shape gate
    # (intermediate % 8, hidden % group_size) rejects the dummy 1x1 config.
    moe_config = make_dummy_moe_config(
        hidden_dim=2048, intermediate_size=1024)

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=True,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is None


@pytest.mark.parametrize(
    "hidden_dim, intermediate_size, group_size, expected",
    [
        (2048, 10, 128, "intermediate size must be a multiple of 8"),
        (200, 1024, 128, "hidden size must be divisible by the group size"),
    ],
)
def test_gfx906_hip_oracle_shape_gate(hidden_dim, intermediate_size,
                                      group_size, expected):
    from tests.kernels.moe.utils import make_dummy_moe_config

    moe_config = make_dummy_moe_config(
        hidden_dim=hidden_dim, intermediate_size=intermediate_size)
    quant_config = AutoAWQConfig(4, group_size, True, False)

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=True,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is not None
    assert expected in reason


@pytest.mark.parametrize(
    ("quant_config", "expected"),
    [
        # W8A16: the kernel is W4A16 only.
        (
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
            ),
            "4-bit weights",
        ),
        # Dynamic scales: the kernel consumes static per-group scales.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=True,
                group_size=128,
            ),
            "static (non-dynamic) scales",
        ),
        # Group size outside the validated 32/128 set.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=64,
            ),
            "group size 32 or 128",
        ),
        # Channel strategy: no [E, G, N] group scales for the kernel.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=True,
                dynamic=False,
            ),
            "group strategy",
        ),
        # g_idx activation ordering: weights are stored in original
        # column order and need a runtime reordering the kernel lacks.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            "g_idx activation ordering",
        ),
        # DYNAMIC is an alias of GROUP with the same runtime contract.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.DYNAMIC,
            ),
            "g_idx activation ordering",
        ),
        # WEIGHT is format-identical to no activation ordering: the
        # repack consumes the stored weights in natural order, so the
        # gate must not reject it.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.WEIGHT,
            ),
            None,
        ),
    ],
)
def test_gfx906_hip_oracle_symmetric_no_zp_contract_gate(quant_config, expected):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Qwen3.5-A3B-shaped config so the shape gate passes and the no-zp
    # gate is what fires.
    moe_config = make_dummy_moe_config(hidden_dim=2048, intermediate_size=1024)

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    if expected is None:
        assert reason is None
        return
    assert reason is not None
    assert expected in reason


@pytest.mark.parametrize("group_size", [32, 128])
def test_gfx906_hip_oracle_accepts_symmetric_no_zp(group_size):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Gemma-4-26B-A4B-shaped config (group-32 symmetric no-zp): the
    # shipped no-zp path (180f030ee3) must keep passing the gate.
    moe_config = make_dummy_moe_config(hidden_dim=2048, intermediate_size=704)
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=group_size,
    )

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is None


def test_compressed_tensors_weights_are_transposed_for_triton():
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
    )
    w13 = torch.arange(16, dtype=torch.int32).reshape(1, 2, 8)
    w2 = torch.arange(12, dtype=torch.int32).reshape(1, 2, 6)
    w13_scale = torch.arange(32, dtype=torch.float16).reshape(1, 4, 8)
    w2_scale = torch.arange(18, dtype=torch.float16).reshape(1, 3, 6)

    converted = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=quant_config,
        input_dtype=None,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    assert converted is not None
    assert torch.equal(converted[0], w13.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[1], w2.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[2], w13_scale.transpose(1, 2).contiguous())
    assert torch.equal(converted[3], w2_scale.transpose(1, 2).contiguous())


def test_moe_wna16_setup_forwards_selected_backend(monkeypatch):
    method = object.__new__(MoeWNA16Method)
    method.experts_cls = object
    method.wna16_backend = WNA16MoEBackend.HUMMING
    method.moe = object()
    quant_config = object()
    method.get_fused_moe_quant_config = lambda layer: quant_config
    layer = SimpleNamespace(_expert_routing_tables=lambda: (None, None, None))
    captured = {}
    kernel = object()

    def fake_make_wna16_moe_kernel(**kwargs):
        captured.update(kwargs)
        return kernel

    monkeypatch.setattr(moe_wna16, "make_wna16_moe_kernel", fake_make_wna16_moe_kernel)

    method._setup_kernel(layer)

    assert method.moe_kernel is kernel
    assert captured["backend"] == WNA16MoEBackend.HUMMING


def test_moe_wna16_humming_adapter_repacks_uint8_tensors():
    qweight = torch.arange(32, dtype=torch.uint8).reshape(1, 4, 8)
    scales = torch.arange(16, dtype=torch.float16).reshape(1, 4, 4)
    qzeros = torch.arange(16, dtype=torch.uint8).reshape(1, 8, 2)

    converted = _convert_moe_wna16_humming_tensors(
        {"qweight": qweight, "scales": scales, "qzeros": qzeros},
        has_zero_point=True,
    )

    assert torch.equal(converted["weight"], qweight.view(torch.int32))
    assert converted["weight"].shape == (1, 4, 2)
    assert torch.equal(converted["weight_scale"], scales)
    expected_qzeros = (
        qzeros.transpose(-1, -2)
        .contiguous()
        .view(torch.int32)
        .transpose(-1, -2)
        .contiguous()
    )
    assert torch.equal(converted["zero_point"], expected_qzeros)
    assert converted["zero_point"].shape == (1, 2, 2)


def test_moe_wna16_uses_humming_quant_config(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import humming_utils

    method = object.__new__(MoeWNA16Method)
    method.wna16_backend = WNA16MoEBackend.HUMMING
    layer = object()
    quant_config = object()
    monkeypatch.setattr(
        humming_utils,
        "get_humming_moe_quant_config",
        lambda actual_layer, *args, **kwargs: (
            quant_config if actual_layer is layer else None
        ),
    )

    assert method.get_fused_moe_quant_config(layer) is quant_config
