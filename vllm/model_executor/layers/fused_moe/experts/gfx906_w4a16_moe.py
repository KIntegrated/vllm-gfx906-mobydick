# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 MoE experts using the fused gfx906 HIP kernel (moe_gptq_gemm_gfx906).

Single HIP kernel launch per GEMM that handles expert routing + W4A16
dequant + dot product with atomic output accumulation.  The w2 pass fuses
the top-k weight application and the moe_sum reduction into the atomic
epilogue (``output_topk``), so no separate reduce kernel is needed.

Weight format (repacked at load time by the WNA16 oracle, per expert):
  - Packed int32 ``[E, K/8, N]`` with exllama shuffle
  - Scales ``[E, groups, N]`` fp16
  - Zero points ``[E, groups, N/8]`` packed int32 (8 nibbles per word)
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx906
else:

    def on_gfx906() -> bool:
        return False


def _has_gfx906_moe_op() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_gptq_gemm_gfx906"
    )


class Gfx906WNA16Experts(FusedMoEExpertsModular):
    """W4A16 MoE experts using the fused gfx906 HIP kernel."""

    # AWQ zero points are stored verbatim; GPTQ-v1 style zeros need +1.
    zero_offset = 0

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm() and on_gfx906() and _has_gfx906_moe_op()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        # The kernel always produces N output columns; the activation step
        # handles non-gated activations via apply_moe_activation.
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt4Static,
            kInt4Static32,
            kInt4Static32GroupScale,
            kInt4StaticGroupScale,
        )

        # MoeWNA16 (AWQ fallback on ROCm) uses the group-scale keys;
        # AutoAWQMoEMethod (Marlin path) uses the plain keys. Group size is
        # carried in the scales shape and handled at runtime by the kernel.
        return weight_key in (
            kInt4Static,
            kInt4Static32,
            kInt4StaticGroupScale,
            kInt4Static32GroupScale,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # w1 is the repacked [E, K/8, N] int32 layout; N is the last dim.
        E = w1.shape[0]
        N = w1.shape[2]
        K = a1.size(-1)
        M = a1.size(0)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace13: gemm1 output [M*topk, N] (zeroed before use)
        # workspace2:  activation output [M*topk, N/2]
        # fused_out:   final reduced output [M, K] (zeroed before use)
        return (
            (M * topk, N),
            (M * topk, self.adjust_N_for_activation(N, activation)),
            (M, K),
        )

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduceNoOP:
        # The w2 kernel applies router weights and reduces over top-k itself.
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        E, M, N, K, topk = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts == -1:
            global_num_experts = E

        assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
        assert hidden_states.dtype == torch.float16, (
            "gfx906 W4A16 MoE kernel requires fp16 activations, "
            f"got {hidden_states.dtype}"
        )

        em = M * topk
        if em <= 32:
            block_size_m = 1
        elif em <= 512:
            block_size_m = 4
        else:
            block_size_m = 16

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(
                topk_ids, block_size_m, global_num_experts, expert_map
            )
        )

        if not hasattr(self, "_empty_topk_w"):
            self._empty_topk_w = torch.empty(0, dtype=torch.float32,
                                             device=hidden_states.device)
        if apply_router_weight_on_input:
            w1_tw = topk_weights.view(-1).float()
            w2_tw = self._empty_topk_w
        else:
            w1_tw = self._empty_topk_w
            w2_tw = topk_weights.view(-1).float()

        # --- gemm1: [M, K] -> [M*topk, N] (atomic into zeroed workspace) ---
        w1_out = _resize_cache(workspace13, (em, N))
        w1_out.zero_()
        ops.moe_gptq_gemm_gfx906(
            hidden_states,
            w1_out,
            w1,
            self.quant_config.w1_scale,
            self.quant_config.w1_zp,
            w1_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk,
            block_size_m,
            apply_router_weight_on_input,
            0,
            self.zero_offset,
        )

        # --- activation: [M*topk, N] -> [M*topk, N/2] ---
        act_out = _resize_cache(
            workspace2, (em, self.adjust_N_for_activation(N, activation))
        )
        self.activation(activation, act_out, w1_out)

        # --- gemm2: [M*topk, N/2] -> [M, K] (fused weight + reduce) ---
        output.zero_()
        ops.moe_gptq_gemm_gfx906(
            act_out,
            output,
            w2,
            self.quant_config.w2_scale,
            self.quant_config.w2_zp,
            w2_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            topk,
            self.zero_offset,
        )
