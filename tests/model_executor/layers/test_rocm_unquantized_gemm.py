# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    pytest.skip(
        "ROCm skinny GEMM tests are not supported on CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers import utils


def test_rocm_unquantized_gemm_gfx1x_wvsplitk_path(monkeypatch):
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_skinny_activation_contiguous(monkeypatch):
    x = torch.randn(64, 4, dtype=torch.float16).t()
    weight = torch.randn(128, 64, dtype=torch.float16)
    assert x.shape == (4, 64)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    x_view = wvsplitk_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_llmm1_activation_contiguous(monkeypatch):
    x = torch.randn(1, 128, dtype=torch.float16)[:, ::2]
    weight = torch.randn(4, 64, dtype=torch.float16)
    assert x.shape == (1, 64)
    assert x.stride() == (128, 2)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    x_view = llmm1_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_rejects_unsupported_skinny_layouts(
    monkeypatch, noncontiguous_operand
):
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel(monkeypatch, dtype):
    x = torch.randn(64, 4, device="cuda", dtype=dtype).t()
    weight = torch.randn(128, 64, device="cuda", dtype=dtype)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back(monkeypatch):
    # wvSplitK skinny GEMM handles n in [1, 5] (see PR #40687); n > 5 must
    # fall back to torch.nn.functional.linear.
    x = torch.randn(6, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_rocm_unquantized_gemm_tiny_m_llmm1_padded(monkeypatch, m):
    # m < 4 (e.g. Qwen3-Next shared_expert_gate [1, K]): LLMM1 requires
    # M % 4 == 0, so the dispatch zero-pads the weight and slices the out.
    x = torch.randn(1, 2048, dtype=torch.float16)
    weight = torch.randn(m, 2048, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    w_padded = llmm1_mock.call_args.args[0]
    assert w_padded.shape == (4, 2048)
    assert torch.equal(w_padded[:m], weight)
    assert out.shape == (1, m)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
def test_rocm_unquantized_gemm_tiny_m_real_kernel(monkeypatch):
    # Real LLMM1 path for m=1 (shared_expert_gate shape): padded weight must
    # produce the same result as a plain linear on one token.
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    weight = torch.randn(1, 2048, device="cuda", dtype=torch.float16)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    assert out.shape == (1, 1)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m,expect_gemv", [(256, True), (2048, True),
                                           (1024, False)])
def test_rocm_unquantized_gemm_tiny_m_gemv_dispatch(monkeypatch, m,
                                                    expect_gemv):
    # gfx906 custom W16A16 GEMV (P3-2b) dispatch rule: K=2048 rows with
    # N==256 (router) or N>=2048 (in_proj/qkv/LM head) take the GEMV; other
    # N (e.g. 1024 gate_up) stay on LLMM1. Mock-based, arch-independent.
    monkeypatch.delenv("VLLM_GFX906_DENSE_GEMV", raising=False)
    x = torch.randn(1, 2048, dtype=torch.float16)
    weight = torch.randn(m, 2048, dtype=torch.float16)

    gemv_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    llmm1_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "dense_gemv_gfx906", gemv_mock)
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: True)

    out = utils._llmm1_tiny_m(weight, x)
    ref = torch.nn.functional.linear(x, weight)

    assert gemv_mock.called is expect_gemv
    assert llmm1_mock.called is (not expect_gemv)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_tiny_m_gemv_never_off_gfx906(monkeypatch):
    # Regression guard: the custom GEMV is measured only on gfx906 and must
    # never route onto other ROCm targets, even for GEMV-eligible shapes.
    monkeypatch.delenv("VLLM_GFX906_DENSE_GEMV", raising=False)
    x = torch.randn(1, 2048, dtype=torch.float16)
    weight = torch.randn(256, 2048, dtype=torch.float16)

    gemv_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    llmm1_mock = MagicMock(side_effect=lambda w, xv, _: xv @ w.t())
    monkeypatch.setattr(utils.ops, "dense_gemv_gfx906", gemv_mock)
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx906", lambda: False)

    out = utils._llmm1_tiny_m(weight, x)
    ref = torch.nn.functional.linear(x, weight)

    gemv_mock.assert_not_called()
    llmm1_mock.assert_called_once()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="ROCm-only kernel test")
@pytest.mark.parametrize("m", [256, 2048])
def test_rocm_unquantized_gemm_dense_gemv_real_kernel(monkeypatch, m):
    # Numeric gate for the gfx906 custom GEMV on its default model path
    # (K=2048, kchunk=2048 single-pass, RPT=2): must match F.linear at
    # fp16 precision. Catches row-mapping / RPT dispatch regressions.
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_gfx906 is measured only on gfx906")
    torch.manual_seed(0)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    weight = torch.randn(m, 2048, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_gfx906(weight, x, 2048)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (1, m)
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=0.15, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="ROCm-only kernel test")
@pytest.mark.parametrize("m", [256, 2048])
def test_rocm_unquantized_gemm_dense_gemv_ksplit_real_kernel(monkeypatch, m):
    # Numeric gate for the K-split (atomic packed-CAS) epilogue: kchunk=512
    # over K=2048 → ksplit=4, RPT=4 (64-bit pk4 CAS). The model path never
    # takes this branch (kchunk is hardcoded to K); this test keeps the
    # bench-only path honest.
    from vllm.platforms.rocm import on_gfx906

    if not on_gfx906():
        pytest.skip("dense_gemv_gfx906 is measured only on gfx906")
    torch.manual_seed(1)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    weight = torch.randn(m, 2048, device="cuda", dtype=torch.float16)

    out = utils.ops.dense_gemv_gfx906(weight, x, 512)
    ref = torch.nn.functional.linear(x, weight)

    assert out.shape == (1, m)
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=0.15, rtol=2e-2)


def test_rocm_unquantized_gemm_gfx950_wvsplitkrc_path(monkeypatch):
    x = torch.randn(1024, 16, dtype=torch.float16).t()
    weight = torch.randn(256, 1024, dtype=torch.float16)
    assert x.stride() == (1, 16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: True)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitkrc_mock = MagicMock(side_effect=lambda x_view, w, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitKrc", wvsplitkrc_mock)
    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitkrc_mock.assert_called_once()
    wvsplitk_mock.assert_not_called()
    x_view = wvsplitkrc_mock.call_args.args[0]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
