# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


def get_pp_safe_draft_load_config(load_config: LoadConfig) -> LoadConfig:
    """Avoid collectives that include PP ranks without a draft model."""
    if get_pp_group().world_size > 1 and load_config.load_format == "fastsafetensors":
        logger.warning_once(
            "fastsafetensors cannot load a draft model instantiated on only "
            "one pipeline stage; falling back to the standard safetensors "
            "loader for the draft model. The target model still uses "
            "fastsafetensors."
        )
        return replace(load_config, load_format="auto")
    return load_config


class DraftTokensHandler:
    """Carries draft tokens from the end of one step to the scheduler.

    Two independent channels live here, and keeping them separate is the whole
    point of the GFX906 additions:

    * ``draft_tokens_np`` is a **scheduler** channel. It exists only so the
      scheduler can validate drafts against a grammar, and upstream publishes it
      *only* when the batch has structured-output requests; otherwise
      ``get_draft_tokens()`` reports ``[[-1] * num_draft]``. Changing that would
      change scheduling, so nothing here does.
    * ``draft_tokens_host`` (``host_draft_staging=True``) is a **worker-local**
      channel: a pinned mirror of the same device tensor, read by the PLE n-gram
      lookup of the *next* step. It exists because under MTP three of every four
      rows of a decode step are drafted tokens and the host otherwise has no
      copy of them at all -- see ``docs/d1-host-ids-design.md`` for the trace
      that established that, and for why partial coverage buys nothing (the
      drain is one wait per step regardless of how many rows were host-known).

    The copy rides ``copy_stream`` and is waited on as late as the consumer can
    manage (:meth:`wait_host_draft_tokens`), never here: staging is issued while
    the drafter's own work is still queued, so blocking now would turn a host
    that currently runs ahead of the drafter into one that waits for it.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        host_draft_staging: bool = False,
    ):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

        # GFX906 / VLLM_GFX906_PLE_HOSTIDS. Off by default, and off unless the
        # caller also proved sync scheduling: this channel's value comes from
        # the end-of-step host sync that sync scheduling gives for free, so it
        # is not a policy preference but a precondition.
        self.host_draft_staging = host_draft_staging
        self.draft_tokens_host: torch.Tensor | None = None
        self.draft_tokens_host_np: np.ndarray | None = None
        self.host_draft_req_idx: np.ndarray | None = None
        # Created unconditionally rather than only when staging is on, so the
        # attribute is never None: the record/synchronize sites stay branch-free
        # and mypy keeps the type as Event rather than Event | None. When staging
        # is off it is simply never recorded, and synchronizing an event that was
        # never recorded is a no-op.
        self.host_draft_event = torch.cuda.Event(blocking=True)
        # Bumped every time a block is staged. A consumer must refuse a block
        # whose gen it has already consumed, which is the only way to tell "the
        # drafts of the step I am about to run" from "the drafts of some earlier
        # step, whose row mapping happens to still match because the batch did
        # not change". Mapping agreement alone cannot distinguish those two, and
        # the second one is a plausible ngram id built from a stale context.
        self.host_draft_gen = 0
        self._host_staging_announced = False

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]

        if self.host_draft_staging:
            # Deliberately *before* the early return below: having the ids for a
            # batch with no structured outputs is the entire purpose of the
            # knob, and that is exactly the batch upstream declines to copy.
            self._stage_draft_tokens(input_batch, draft_tokens)

        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def _host_staging_buffer(
        self, rows: int, cols: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Pinned host ``[rows, cols]`` mirror of the draft buffer, grown if needed.

        Allocated once and reused: a per-step pinned allocation on the decode
        path would be a fresh cudaHostAlloc in the middle of the step, which is
        the kind of host work this whole effort is trying to remove. The dtype
        follows the source tensor instead of casting, because a cast would be a
        device kernel per step to achieve nothing.
        """
        buf = self.draft_tokens_host
        if (
            buf is not None
            and buf.shape[0] >= rows
            and buf.shape[1] >= cols
            and buf.dtype == dtype
        ):
            return buf
        rows = max(rows, buf.shape[0] if buf is not None else 0)
        cols = max(cols, buf.shape[1] if buf is not None else 0)
        buf = torch.zeros(rows, cols, dtype=dtype).pin_memory()
        self.draft_tokens_host = buf
        self.draft_tokens_host_np = buf.numpy()
        return buf

    def _stage_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        rows, cols = draft_tokens.shape[0], draft_tokens.shape[1]
        buf = self._host_staging_buffer(rows, cols, draft_tokens.dtype)
        dst = buf[:rows, :cols]
        if draft_tokens.is_cuda:
            current_stream = torch.cuda.current_stream(self.device)
            self.copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.copy_stream):
                dst.copy_(draft_tokens, non_blocking=True)
                # draft_tokens is a temporary device allocation created on the
                # main stream by ``req_states.draft_tokens[idx_mapping]`` and is
                # read here on copy_stream; without record_stream the caching
                # allocator may hand its memory back before the copy runs.
                draft_tokens.record_stream(self.copy_stream)
                self.host_draft_event.record()
        else:
            # Tests, and any caller that hands us a CPU tensor: the copy is
            # already done, so there is no event to wait for later.
            dst.copy_(draft_tokens)
        # Row i of the staged block belongs to the request whose persistent
        # req_state slot is host_draft_req_idx[i]. The mapping is published
        # together with the values on purpose: reading the values by batch
        # position while assuming some other mapping is exactly the off-by-one
        # that yields a plausible ngram id for the wrong request.
        self.host_draft_req_idx = input_batch.idx_mapping_np[:rows]
        self.host_draft_gen += 1
        if not self._host_staging_announced:
            self._host_staging_announced = True
            logger.info(
                "GFX906_PLE_HOSTIDS: draft staging active, %d rows x %d into a "
                "reused pinned buffer on the draft-copy stream",
                rows,
                cols,
            )

    def wait_host_draft_tokens(self):
        """Block until this step's staged drafts have landed.

        Returns ``(values, row_index, gen)``, or ``(None, None, gen)`` if staging
        is off or nothing has been staged yet. ``values[:n, :k]`` is valid for
        the step that staged it, ``row_index[:n]`` names the req_state slot of
        each row, and ``gen`` identifies the staging event. Both arrays are views
        of a buffer that the *next* step overwrites, so a caller must read what
        it needs before returning.

        ``gen`` exists because ``row_index`` alone cannot prove freshness: two
        consecutive decode steps over an unchanged batch produce an identical
        mapping with different values, and consuming the older one is silent. A
        consumer must remember the gen it last consumed and fall back to the
        device path when it has not advanced.

        Call this as late as the computation allows: the wait covers the
        drafter's queued work, and every host action performed before it
        (scheduler bookkeeping, index building) is time the drafter used.
        """
        if not self.host_draft_staging or self.draft_tokens_host_np is None:
            return None, None, self.host_draft_gen
        # Synchronizing an event that was never recorded is a no-op, which is
        # what makes the CPU-staged (test) path correct here too.
        self.host_draft_event.synchronize()
        return self.draft_tokens_host_np, self.host_draft_req_idx, self.host_draft_gen

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
