# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class SpeculatorCudaGraphManager(CudaGraphManager):
    """CudaGraphManager for draft prefill and decode.

    Builds fresh dummy inputs and attention metadata for every warmup and
    capture pass so that the contents of the shared persistent buffers
    (e.g. query_start_loc, seq_lens, FA3 scheduler metadata) always match
    the batch descriptor being captured. Reusing metadata built during an
    earlier capture would execute kernels with stale buffer contents.
    """

    # The dyn table (`num_speculative_tokens_per_batch_size`) describes the
    # target model's decode-time verify depth; it is meaningless for
    # draft/prefill managers. The base `_init_candidates` instead derives it
    # from `decode_query_len - num_speculative_tokens` and expands the dyn
    # table; this class's decode instances are built by
    # `autoregressive/speculator.py:144` with an explicit `decode_query_len=1`,
    # which gives `1 - 3 = -2`, so `decode_query_lens = [1, -1]` (negative
    # values). `round_up(n, -1) // -1` then yields a negative num_reqs,
    # tripping `assert 0 < num_reqs <= num_tokens` in `InputBatch.make_dummy`.
    # This class therefore opts out of the dyn table
    # (`_use_dynamic_schedule=False`); the base defaults to True.
    def __init__(self, *args, honor_dynamic_sd: bool = False, **kwargs):
        # It must be set before super().__init__(): the base __init__ calls
        # self._init_candidates() at around line 145, and setting it afterwards
        # leaves the attribute undefined, so getattr(..., True) returns True and
        # the crash is unchanged (verified by 3/3 cold-start failures with a
        # traceback proving the patch was active).
        self._use_dynamic_schedule = honor_dynamic_sd
        super().__init__(*args, **kwargs)

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            # Defensive: satisfy the InputBatch.make_dummy contract
            # (`assert 0 < num_reqs <= num_tokens`); clamp explicitly rather than crash.
            num_reqs = max(1, min(num_reqs, num_tokens))
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
            )

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)
