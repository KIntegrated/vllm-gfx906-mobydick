# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd.ple_prefetch import (
    PleRowPrefetcher,
    plan_lookahead_chunks,
    prefetch_enabled,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        # Opt-in read-ahead for the PLE ngram table (see ple_prefetch.py).
        # Built on first use, not here: the model is not loaded yet at this
        # point, and the prefetcher needs the embedding's shard pointers.
        self._ple_prefetcher: PleRowPrefetcher | None = None
        self._ple_prefetch_wanted = prefetch_enabled()
        self._ple_host_tokens = None

    # ------------------------------------------------------------- read-ahead
    def _ple_prefetch_modules(self):
        """Locate the ngram embedding module without hardcoding layer indices.

        PLE lives at ``ple_layer_ids`` (layer 2 for this checkpoint), and the
        attribute path is a model-implementation detail. Searching by type keeps
        this correct if the model is rearranged, and returns None -- disabling
        the prefetch -- instead of raising if it moves.
        """
        from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpNGramEmbedding

        for module in self.model.modules():
            if isinstance(module, Qwen4ExpNGramEmbedding):
                return module
        return None

    def _build_ple_prefetcher(self) -> PleRowPrefetcher | None:
        ngram = self._ple_prefetch_modules()
        if ngram is None:
            logger.info_once(
                "PLE read-ahead requested but no ngram embedding module was "
                "found; running without it."
            )
            self._ple_prefetch_wanted = False
            return None
        # A lookahead larger than one scheduled chunk is pure waste: the next
        # chunk cannot be bigger than the batch token budget.
        max_tokens = int(
            getattr(
                self.vllm_config.scheduler_config,
                "max_num_batched_tokens",
                8192,
            )
        )
        return PleRowPrefetcher(
            ngram.ngram_embedding,
            (
                ngram.layer_multipliers,
                ngram.ngram_heads_vocab_sizes,
                ngram.ngram_heads_offsets,
            ),
            type(ngram)._compute_ngram_ids,
            ngram_size=int(ngram.ngram_size),
            heads_per_ngram=int(ngram.heads_per_ngram),
            eos_token_id=int(ngram.eos_token_id),
            max_tokens=max_tokens,
        )

    def _prefetch_next_ple_chunk(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        """Ask the kernel to prefetch the next prefill chunk's embedding rows.

        Every input is already on the host as numpy, so this costs no device
        synchronisation: ``num_computed_tokens_np`` is the optimistic CPU mirror
        and ``prefill_len_np`` the scheduled prompt length. A stale mirror only
        warms pages the chunk does not need, which is wasted bandwidth, never a
        wrong number -- that asymmetry is why an upper-bound mirror is safe here.
        """
        pf = self._ple_prefetcher
        if pf is not None and pf.disabled_reason is not None:
            # Permanently off: stop planning too, this runs on every step and
            # the whole point is to spend no host CPU on the critical path.
            self._ple_prefetcher = None
            self._ple_prefetch_wanted = False
            return
        if pf is None and not self._ple_prefetch_wanted:
            return
        # CUDA graph capture replays this hook with dummy batches, once per
        # capture size. Hinting those pages would stream garbage off NVMe
        # during boot, which is when the box can least afford it. Checked every
        # call: capture happens long after the first real step.
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        if pf is None:
            try:
                self._ple_prefetcher = self._build_ple_prefetcher()
            except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
                logger.warning(
                    "PLE read-ahead could not be set up (%s: %s); running "
                    "without it. Outputs are unaffected.",
                    type(exc).__name__,
                    exc,
                )
                self._ple_prefetch_wanted = False
                return
        pf = self._ple_prefetcher
        if pf is None:
            # The builder declined (no ngram module): nothing to plan against.
            return
        try:
            if self._ple_host_tokens is None:
                buf = getattr(
                    getattr(req_states, "all_token_ids", None), "_uva_buf", None
                )
                host = getattr(buf, "np", None)
                if host is None or host.ndim != 2:
                    # all_token_ids is normally a UVA-backed host array; on a
                    # platform where it is real GPU memory there is no cheap
                    # host view, and a D2H copy per step would defeat the point.
                    pf.disable("req_states.all_token_ids has no host view")
                    return
                self._ple_host_tokens = host
            n = input_batch.num_reqs
            for req, start, end, ctx in plan_lookahead_chunks(
                host_tokens=self._ple_host_tokens,
                idx_mapping=input_batch.idx_mapping_np[:n],
                scheduled=input_batch.num_scheduled_tokens[:n],
                computed=input_batch.num_computed_tokens_np[:n],
                prefill_len=input_batch.prefill_len_np[:n],
                context_len=self.ngram_context_len,
                eos_token_id=self.ngram_eos_token_id,
            ):
                pf.prefetch_token_chunk(self._ple_host_tokens, req, start, end, ctx)
        except Exception as exc:  # noqa: BLE001 - never reach the model path
            pf.disable(f"lookahead planning failed ({type(exc).__name__}: {exc})")

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        # Issued last: the current chunk's inputs are already prepared, so the
        # hints overlap the GPU work that follows instead of delaying it.
        self._prefetch_next_ple_chunk(input_batch, req_states)
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])

        ngram_context = self.ngram_context[:num_reqs]
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
