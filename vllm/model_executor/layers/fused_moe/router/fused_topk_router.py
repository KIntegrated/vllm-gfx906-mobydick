# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

if torch.cuda.is_available() and torch.version.hip is not None:
    from vllm.platforms.rocm import on_gfx906
else:

    def on_gfx906() -> bool:
        return False


def _has_gfx906_m1_topk() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_topk_softmax_m1_gfx906"
    )


def _has_gfx906_routing_fused_m1_op() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_routing_fused_m1_gfx906"
    )


def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        return is_padding[:num_tokens] if is_padding is not None else None
    return None


def _gfx906_m1_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> bool:
    """M=1 decode fast path (S2): dedicated gfx906 top-k kernel, bit-equal
    to ops.topk_softmax for the exact shape it serves. Returns True when
    handled.

    Default OFF: the kernel wins in isolation (12.5 vs 17.3 us GPU
    self-time) and in eager serving (+0.11 t/s) but loses in CUDA-graph
    replay (-0.95 t/s at 66 t/s) — see the DEVLOG-moe-m1-sprint S2
    table. Opt in with VLLM_GFX906_TOPK_M1=1 for eager-only deployments.

    Bit-equality assumptions: ops.topk_softmax must be called with the
    same neutral configuration this dispatch checks — softmax scoring,
    no bias, no padding, full expert range (no TP expert sharding) —
    and the caller passes routed_scaling_factor=1 upstream. A caller
    that adds correction-bias or expert-range routing must be excluded
    here or the outputs silently diverge from the generic path.
    """
    if (
        on_gfx906()
        and _has_gfx906_m1_topk()
        and os.environ.get("VLLM_GFX906_TOPK_M1", "0") == "1"
        and gating_output.shape[0] == 1
        and topk_indices.shape[0] == 1
        and gating_output.shape[1] == 256
        and topk_indices.shape[1] == 8
        and gating_output.dtype == torch.float16
        and topk_indices.dtype == torch.int32
        and gating_output.is_contiguous()
        and _get_padding_mask(topk_indices.shape[0]) is None
    ):
        ops.moe_topk_softmax_m1_gfx906(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
        return True
    return False


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    if _gfx906_m1_topk_softmax(
        topk_weights, topk_indices, token_expert_indices, gating_output,
        renormalize,
    ):
        return topk_weights, topk_indices

    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        # C1 stage 2: (sorted_token_ids, expert_ids, num_tokens_post_pad)
        # produced by the fused routing kernel for the M=1 decode shape.
        # Invariant: (re)written on EVERY routing call (None on the
        # non-fused path) and read+cleared ONLY by
        # MoERunner._apply_quant_method immediately after select_experts;
        # no other site may read it. A routing call without a following
        # expert execution would leave dead state.
        self._fused_align_meta: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _use_routing_fused_m1(
        self,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> bool:
        """C1 stage 2: single-CTA topk+align+count for the M=1 decode
        shape (see docs/gfx906/DEVLOG-moe-c1-routing-fusion.md).

        The gate must hold at BOTH this site and the post-routing steps in
        _select_experts (EPLB mapping and indices-dtype conversion must be
        identity, or the align metadata would be computed on different
        ids than the expert sees). Dynamo tracing is excluded: the MoE
        body is one opaque custom op, so tracing runs the unquantized
        method (no meta support); at graph capture/replay time
        is_compiling() is False and the fused branch fires as intended.
        """
        return (
            os.environ.get("VLLM_GFX906_ROUTING_FUSE_M1", "0") == "1"
            and not torch._dynamo.is_compiling()
            and on_gfx906()
            and _has_gfx906_routing_fused_m1_op()
            and self.scoring_func == "softmax"
            and self.top_k == 8
            and self.global_num_experts == 256
            and self.eplb_state is None
            and indices_type in (None, torch.int32)
            and router_logits.shape[0] == 1
            and router_logits.shape[1] == 256
            and router_logits.dtype == torch.float16
            and router_logits.is_contiguous()
            and _get_padding_mask(1) is None
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        if self._use_routing_fused_m1(router_logits, indices_type):
            dev = router_logits.device
            topk_weights = torch.empty(
                1, self.top_k, dtype=torch.float32, device=dev
            )
            topk_ids = torch.empty(
                1, self.top_k, dtype=torch.int32, device=dev
            )
            token_expert_indices = torch.empty(
                1, self.top_k, dtype=torch.int32, device=dev
            )
            # wrapper-convention align buffer sizes for M=1 (see the
            # moe_align_block_size wrapper: numel + E*(1-1) = 8, and numel
            # < E keeps it at 8; expert_ids size = cdiv(8, 1) = 8)
            sorted_token_ids = torch.empty(
                self.top_k, dtype=torch.int32, device=dev
            )
            expert_ids = torch.empty(
                self.top_k, dtype=torch.int32, device=dev
            )
            num_tokens_post_pad = torch.empty(
                1, dtype=torch.int32, device=dev
            )
            torch.ops._rocm_C.moe_routing_fused_m1_gfx906(
                router_logits, topk_weights, topk_ids, token_expert_indices,
                sorted_token_ids, expert_ids, num_tokens_post_pad,
                self.renormalize,
            )
            self._fused_align_meta = (
                sorted_token_ids, expert_ids, num_tokens_post_pad
            )
            return topk_weights, topk_ids

        self._fused_align_meta = None
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
