# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Utility methods for model layers."""

from collections.abc import Callable

import os

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import tl
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}

def get_autotune_config():
    return [
        # Decode/MTP uses this path for skinny GEMMs (M <= 16).  On gfx906,
        # smaller N tiles expose more work when TP makes each shard narrow,
        # while larger K/N tiles still win for wider projections.
        triton.Config(
            {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=1,
            num_warps=4,
        ),
        # Keep the previous schedule in the search space. Some non-gfx906
        # environments can still prefer deeper pipelining.
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3,
            num_warps=2,
        ),
    ]

def get_heuristics():
    return {
        # gfx906 matrix instructions naturally operate on 16-row tiles. This
        # path is only selected for M <= 16, so use the full tile instead of
        # emitting separate 8-row variants for batches 5..8.
        "BLOCK_SIZE_M": lambda args: 16
    }

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.heuristics(values=get_heuristics())
@triton.jit
def triton_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.T.
    A has shape (M, K), B has shape (N, K) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16) # acc in fp32 back to fp16

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions" # NOTE(gfx906): b.shape inv
    assert a.dtype == b.dtype, "Matrices A and B must have the same dtype (assuming fp16)"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape # NOTE(gfx906): b.shape inv
    launch_kwargs = {}
    launch_kwargs["waves_per_eu"] = 1 # best for gfx906

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    triton_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(1), b.stride(0),  # NOTE(gfx906): b.stride inv
        c.stride(0), c.stride(1),  #
        **launch_kwargs,
    )
    return c


def _llmm1_tiny_m(weight: torch.Tensor, x_view: torch.Tensor) -> torch.Tensor:
    """ops.LLMM1 requires weight rows % 4 == 0; zero-pad tiny m and slice
    the result back.

    gfx906: the custom row-parallel W16A16 GEMV (dense_gemv_gfx906) beats
    LLMM1 rpb=4 on K=2048 rows with N==256 (router, -17%) or N>=2048
    (in_proj/qkv/LM-head, -6..-23%); see
    benchmarks/kernels/gfx906/bench_dense_gemv_gfx906.py. Varying shapes
    (o_proj K=4096, gate_up 1024, shared down K=512, N=64) stay on LLMM1.
    The GEMV is measured only on gfx906 (MI50) and is gated to it; other
    ROCm targets fall through to LLMM1.

    m==1 (the Qwen3-Next shared-expert gate [1, K]) also goes to the GEMV
    (RPT=1): the LLMM1 route zero-pads the *constant* weight to [4, K]
    every call (a fill + copy per layer per step); GEMV RPT=1 is 4.7x
    faster in isolation and bit-equal at N=1, K=2048
    (bench /tmp/bench/bench_gate_gemv.py).
    """
    m = weight.shape[0]
    from vllm.platforms.rocm import on_gfx906

    if (
        on_gfx906()
        and os.environ.get("VLLM_GFX906_DENSE_GEMV", "1") != "0"
        and weight.dtype == torch.float16
        and x_view.dtype == torch.float16
        and weight.is_contiguous()
        and weight.shape[1] == 2048
        and (m == 1 or m == 256 or m >= 2048)
    ):
        return ops.dense_gemv_gfx906(weight, x_view, 2048)
    if m % 4 == 0:
        return ops.LLMM1(weight, x_view, 4)
    out = ops.LLMM1(torch.nn.functional.pad(weight, (0, 0, 0, 4 - m)), x_view, 4)
    return out[:, :m]


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import (
        on_gfx1x,
        on_gfx9,
        on_gfx906,
        on_gfx950,
        on_gfx1250,
    )

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = 0
    if not on_gfx906():
        cu_count = num_compute_units()

        # Next ^2 of n
        N_p2 = 1 << (n - 1).bit_length()
        # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
        # and each working on a 512-shard of K, how many CUs would we need?
        rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
        # How many of 4 waves in a group can work on same 16 Ms at same time?
        # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
        GrpsShrB = min(N_p2 // 16, 4)
        # Given the above, how many CUs would we need?
        CuNeeded = rndup_cus * GrpsShrB
        # Deterministic reduction stores one float workspace value per K shard.
        fits_wvsplitkrc = (
            N_p2 * m * ((k + 512 - 1) // 512)
        ) <= 128 * 1024 * 12  # deterministic
        fits_wvsplitkrc &= CuNeeded <= cu_count

        skinny_operands_compatible = weight.is_contiguous() and (
            bias is None or bias.is_contiguous()
        )

        use_skinny_reduce_counting = (
            envs.VLLM_ROCM_USE_SKINNY_GEMM
            and on_gfx950()
            and x.dtype in [torch.float16, torch.bfloat16]
            and x.dim() == 2
            and (
                10 <= n <= 128
                and k % 8 == 0
                and k > 512
                and m % 16 == 0
                and fits_wvsplitkrc
                and skinny_operands_compatible
            )
        )

        if use_skinny_reduce_counting:
            x_view = x.reshape(-1, x.size(-1)).contiguous()
            return ops.wvSplitKrc(x_view, weight, cu_count, bias)

        # gfx1250's aiter gemm_a16w16 uses the gluon backend, which requires
        # K % 256 == 0 (it walks K with fixed-size descriptors and won't pad a
        # partial last tile). Some whitelisted shapes have K=2880 (e.g. gpt-oss-120b
        # hidden), so skip aiter there and fall back to the torch GEMM path below.
        if use_aiter_triton_gemm(n, m, k, x.dtype) and not (on_gfx1250() and k % 256 != 0):
            from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

            return gemm_a16w16(x, weight, bias)

    use_skinny = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and (on_gfx9() or on_gfx1x() or on_gfx906())
        and x.dtype in [torch.float16, torch.bfloat16]
        and k % 8 == 0
        and (weight.is_contiguous() and (bias is None or bias.is_contiguous()))
    )

    if use_skinny:
        # The skinny kernels assume contiguous K elements. A shape-preserving
        # reshape can retain a transposed activation's non-contiguous strides.
        x_view = x.reshape(-1, x.size(-1)).contiguous()
        # wvSplitK targets CDNA/RDNA matrix cores and is not supported on
        # gfx906 (GCN5/Vega20); exclude it here so small batches fall through
        # to the Triton skinny-GEMM path below, matching the pre-main behavior.
        if m > 8 and 0 < n <= 5 and not on_gfx906():
            cu_count = num_compute_units()
            out = ops.wvSplitK(weight, x_view, cu_count, bias)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        elif (m % 4 == 0 or m < 4) and n == 1 and k <= 8192 and bias is None:
            out = _llmm1_tiny_m(weight, x_view)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    x_view = x.reshape(-1, x.size(-1))
    # Prefer skinny GEMV kernel
    if (
        (m % 4 == 0 or m < 4)
        and n == 1
        and k <= 8192
        and bias is None
    ):
        out = _llmm1_tiny_m(weight, x_view)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    elif m > 8 and 0 < n <= 4 and (on_gfx9() or on_gfx1x()):
        out = ops.wvSplitK(weight, x_view, cu_count, bias)  # matrix cores not supported by gfx906 so excluded here
        return out.reshape(*x.shape[:-1], weight.shape[0])
    # low batch size, use triton matmul
    elif n <= 16 and bias is None:
        # gfx906 / MI50:
        # For Qwen3.6 TP=8 MLP down projection, the shape is typically:
        #   x:      [n, 2176]
        #   weight: [5120, 2176]
        #   out:    [n, 5120]
        #
        # Focused benchmark showed torch/hipBLAS is faster than this Triton
        # skinny GEMM for m=5120 and k in roughly 2048..2304, for n=2..16.
        #
        # But for k >= 2560, Triton becomes much faster again, so do not
        # disable Triton for all m=5120 row projections.
        if on_gfx906() and n > 1 and m == 5120 and 2048 <= k <= 2304:
            return torch.nn.functional.linear(x, weight, bias)

        return triton_matmul(x if x.is_contiguous() else x.contiguous(), weight)

    # otherwise, use native torch
    return torch.nn.functional.linear(x, weight, bias)
def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


# Above this weight size, oneDNN's onednn_mm consistently matches or beats
# the SGL AMX kernel once M grows past decode-sized batches, and is within
# noise of it at decode-sized M -- so larger weights default to oneDNN
# rather than SGL. 1 MiB comfortably covers MoE router/gate weights (e.g.
# (2048, 128) .. (2880, 32) bf16/fp16, 180-720 KiB) while staying well below
# any dense qkv/o_proj/gate_up/down/lm_head projection in practice. This
# threshold is derived from bf16/fp16 unquantized dense-GEMM benchmarks only,
# so it does not apply to the int8 scaled_mm path below.
_CPU_SGL_GEMM_MAX_WEIGHT_BYTES = 1 * 1024 * 1024


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    if not torch.cpu._is_amx_tile_supported() or dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.int8,
    ):
        return False
    if dtype == torch.float16 and not torch.cpu._is_amx_fp16_supported():
        # AMX-BF16/INT8 (amx_tile) and AMX-FP16 are separate CPU ISA
        # extensions -- e.g. Sapphire/Emerald Rapids expose the former but
        # not the latter -- and can_use_brgemm<at::Half> (gemm.h) always
        # attempts brgemm for fp16 regardless of M, so this needs its own
        # capability check rather than piggybacking on amx_tile.
        return False
    if dtype == torch.int8:
        # int8_scaled_mm_with_quant requires the packed weight to stay int8
        # (gemm_int8.cpp); convert_weight_packed's N < TILE_N fallback
        # returns a float32 tensor instead (gemm.cpp), which would trip
        # that check, so N must be a full TILE_N tile here.
        return k % 32 == 0 and n % 16 == 0
    if n * k * dtype.itemsize > _CPU_SGL_GEMM_MAX_WEIGHT_BYTES:
        return False
    if n < 16:
        # convert_weight_packed transposes to fp32 instead of VNNI-packing
        # when N < TILE_N (gemm.cpp), and weight_packed_linear detects that
        # (via the packed weight's dtype) and routes to its fp32/brgemm
        # fallback kernel -- no N/K alignment required in that regime.
        return True
    return k % 32 == 0 and n % 16 == 0


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    # Skip CPU GEMM dispatch for non-2D weights (e.g. MoE 3D expert weights).
    # These layers are handled by their own specialized methods.
    if layer.weight.ndim != 2:
        # this is not a linear layer
        # For now it should be a causal_conv1d op or MoE 3D expert weights
        if torch.cpu._is_amx_tile_supported() and hasattr(
            ops, "causal_conv1d_weight_pack"
        ):
            # prepack conv weight
            unpacked = (
                layer.weight.view(
                    layer.weight.size(0),
                    layer.weight.size(2),
                )
                .contiguous()
                .clone()
            )
            # Stash the un-packed (dim, width) weight so the speculative-decode
            # GDN path (which uses torch conv, not the AMX kernel) can use it.
            layer._cpu_unpacked_conv_weight = unpacked
            layer.weight.data = ops.causal_conv1d_weight_pack(unpacked)
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            is_prepacked,
        )
        return

    # Small weights (e.g. MoE router/gate projections, where N is the expert
    # count rather than a hidden-size-scaled dimension) never reach oneDNN's
    # compute-bound regime, no matter how large the batch gets: SGL's lower
    # per-call dispatch overhead wins consistently across the full measured
    # M range. Larger dense projections (qkv/o_proj/gate_up/down/lm_head)
    # cross over to favoring oneDNN once batch size grows past decode-sized
    # M, so they keep using oneDNN below.
    if check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
            x, packed_weight, bias_f32 if bias is not None else None, True
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using sgl-kernel weight_packed_linear"
        )
        return

    if (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            logger.debug_once("CPU unquantized GEMM dispatch: using oneDNN onednn_mm")
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )
    logger.debug_once(
        "CPU unquantized GEMM dispatch: using torch.nn.functional.linear (fallback)"
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm

