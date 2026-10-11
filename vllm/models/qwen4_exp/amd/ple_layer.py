# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

import math
import os
import time
import weakref
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.amd import ple_hostids, ple_prof
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

_HOST_STAGEABLE: "weakref.WeakSet[Qwen4ExpNGramEmbedding]" = weakref.WeakSet()
"""PLE embedding modules that opted into host-side id staging, by env at init.

Weak on purpose: engines come and go inside one process (tests, the offline
runner), and a strong registry would keep the 99 GiB shard mappings and their
pinned buffers alive after the engine that owned them was gone. The producer
(:meth:`Qwen4ExpModelState._stage_host_ple_ids`) iterates this set per step; an
empty set -- the default, because ``VLLM_GFX906_PLE_HOSTIDS`` is unset -- means
the per-step cost is one ``len()`` and nothing else.
"""


def host_stageable_layers() -> list:
    """The PLE embedding modules that asked for host-staged ids."""
    return list(_HOST_STAGEABLE)


# from ..common.ple import PLEVocabParallelEmbedding


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


def canonical_ple_shard_rows(
    num_shards: int, shard_row_capacity: int, vocab_size: int
) -> list[int]:
    """The row count vLLM's own splitter would give each shard: one uniform stride
    with a short last shard.

    Used to *describe* the expectation -- the mapping itself is taken from the
    shards that were actually loaded -- and to warn when a checkpoint's split
    disagrees with it.
    """
    return [
        max(0, min(shard_row_capacity, vocab_size - index * shard_row_capacity))
        for index in range(num_shards)
    ]


def build_ple_shard_layout(
    rows: Sequence[int],
    *,
    num_shards: int,
    shard_row_capacity: int,
    vocab_size: int | None,
) -> tuple[list[int], list[int]]:
    """Turn the row count of each loaded shard into id -> (shard, row) boundaries.

    The boundaries have to come from the checkpoint, not from the runtime's own
    arithmetic. `shard_row_capacity` is computed before any weight is loaded --
    from the vocab layout this module builds itself, with a per-head prime and a
    padded total -- while the model's actual id space is whatever the checkpoint's
    `ngram_heads_offsets` / `ngram_heads_vocab_sizes` say once those buffers have
    been loaded over it. When the two disagree, a uniform-stride lookup reads the
    wrong row for every id past the first boundary, and ids in the gap between
    what the stride can address and the real id space index off the end of a
    shard: `Whittle-Qwen-3.8-35B-A3B` splits 7,812,500 x 4 + 7,790,000 rows
    (39,040,000 ids, matching its own flat 8 x 4,880,000 layout) against a stride
    of 7,808,128, which crashed the engine on the last 17,488 ids and mis-read
    every id >= 7,808,128 -- see issue #40.

    `vocab_size` is the model's id space, or None when it is not known yet (a
    stub load, or a checkpoint that never delivered the layout buffers): then the
    shards still define the row -> id mapping and only the consistency check is
    skipped. A `vocab_size` that does not match the rows is a hard error rather
    than a warning, because serving it means serving wrong embeddings silently.
    """
    if num_shards <= 0:
        raise ValueError(f"num shards must be positive, got {num_shards}")
    if len(rows) != num_shards:
        raise ValueError(
            f"expected {num_shards} PLE ngram shards, got {len(rows)}: {list(rows)}"
        )
    for index, count in enumerate(rows):
        if count <= 0:
            raise ValueError(f"PLE ngram shard {index} holds {count} rows")

    total = sum(rows)
    if vocab_size:
        if total < vocab_size:
            raise ValueError(
                f"PLE ngram shards hold {total} rows but the model's ngram id space "
                f"is {vocab_size} (shards: {list(rows)}). The checkpoint's split and "
                "its vocab layout disagree, so a lookup could not be mapped to a row "
                "that the checkpoint meant to store there; refusing to serve."
            )
        if total > vocab_size:
            # A surplus is not a disagreement, it is vocab padding, and the two
            # quantities are not the same kind of number: the shards are stored at
            # the full padded capacity (`shard_row_capacity` is
            # `ceil(padded_vocab_size / parts)`, so 128 of them hold
            # `padded_vocab_size` rows) while `vocab_size` here is
            # `max(ngram_heads_offsets + ngram_heads_vocab_sizes)` -- the end of the
            # last head, unpadded. They differ by whatever `padded_vocab_size` rounds
            # the vocab up to, and the extra rows are simply never addressed: every
            # id is < `vocab_size` <= `total`, and both lookup paths (the stride at
            # :714 and the searchsorted boundaries at :720) put such an id in a shard
            # that has the row. An exact `total == vocab_size` check refused healthy
            # checkpoints -- 128 x 2,500,012 = 320,001,536 rows against an id space
            # of 320,001,446 is 90 rows of padding, not 90 rows of wrong data. Only a
            # *shortfall* can strand an id past the end of the rows, and that still
            # refuses above.
            logger.warning(
                "PLE ngram shards hold %d rows for an ngram id space of %d: %d rows "
                "past the last head's end. That is vocab padding (the shards are built "
                "to the padded vocab, the id space is the end of the last head) and no "
                "id addresses it. Serving the checkpoint's own boundaries.",
                total,
                vocab_size,
                total - vocab_size,
            )
        canonical = canonical_ple_shard_rows(num_shards, shard_row_capacity, vocab_size)
        if list(rows) != canonical:
            logger.warning(
                "PLE ngram shards are not vLLM's canonical split (%d shards, id space "
                "%d, stride %d per shard) but per-shard row counts %s totalling %d ids. "
                "Serving the checkpoint's own boundaries, which is correct for the rows "
                "on disk; a re-split would restore the uniform layout.",
                num_shards,
                vocab_size,
                shard_row_capacity,
                list(rows),
                total,
            )
    else:
        logger.warning(
            "PLE ngram id space is unknown (layout buffers not loaded); taking the "
            "row boundaries from the %d loaded shards (%s rows) without checking them "
            "against the model's vocab layout.",
            num_shards,
            list(rows),
        )

    starts: list[int] = []
    offset = 0
    for count in rows:
        starts.append(offset)
        offset += count
    return starts, [int(count) for count in rows]




_PAGE_SIZE = 4096
_MADV_RANDOM = 1


def _advise_random(tensor: torch.Tensor) -> tuple[bool, int, str]:
    """Tell the kernel that ``tensor``'s pages will be read in random order.

    Returns ``(ok, advised_bytes, detail)``, where ``detail`` explains the
    failure. Never raises: the caller only logs. Sizes are returned in bytes
    and formatted at the log site, so a small test tensor cannot round to 0.

    The ngram table is ~99 GiB of bf16 rows (160 columns = 320 B/row) gathered
    at random out of an ``mmap``'d safetensors mapping, which is far larger
    than this node's page cache: every lookup is a cold random fault. On a
    fault the kernel reads a whole read-ahead window (128 KiB by default) and
    the next touch is in a different shard, so the window is thrown away. On
    2026-09-25 that cost **1.03 MB of NVMe per prefill token at ~28x
    amplification**, and the GPUs sat at 0 % busy for ~18 of every 29 sampled
    seconds of a cold prefill -- prefill was I/O-serialized, not compute-bound
    (docs/prefill-io-wall.md).

    ``MADV_RANDOM`` sets ``VM_RAND_READ`` on the mapping, which makes
    ``do_sync_mmap_readahead()`` fetch the single faulted page and skips
    fault-around. Measured on a real shard, cold cache, 300 scattered touches:
    **32.3 MiB -> 1.0 MiB read (27.6x -> 0.9x), 43 ms -> 24 ms**, with the
    fault count unchanged.

    It is the per-file form of shrinking the read-ahead window, and it is the
    only form that works here: ``file_ra_state.ra_pages`` is latched when the
    file is opened, so setting ``/sys/block/*/queue/read_ahead_kb`` after the
    engine has started does nothing to an already-mmap'd table.

    This is advisory and cannot corrupt anything -- it changes only the kernel's
    read-ahead heuristic, never the bytes returned. A failure is not fatal and
    is reported once by the caller.
    """
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    if ptr == 0 or nbytes == 0:
        return False, 0, "tensor has no storage"
    start = ptr & ~(_PAGE_SIZE - 1)
    length = (ptr - start) + nbytes
    rc = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), _MADV_RANDOM)
    if rc != 0:
        errno = ctypes.get_errno()
        return False, 0, f"madvise failed: errno {errno} ({os.strerror(errno)})"
    return True, length, f"{length} bytes"


# VLLM_GFX906_PLE_FASTGATHER=1 selects ``MmapShardedNGramEmbedding._gather_sorted``
# instead of the original per-shard-mask loop. Measured reason (T3, 2026-09-29,
# docs/t3-ple-prof-results.md): the gather is a *fixed* cost per lookup, not a
# per-token one -- 0.0579 ms/row at decode sizes vs 0.0073 at prefill, i.e. 3.46 ms
# to read 72 rows -- because the loop runs ``num_shards`` (128) iterations and about
# five torch CPU dispatches per iteration, while a decode batch touches 55.7 of the
# 128 shards. Cached like the profiler's flag because it is read once per lookup, on
# a path whose whole purpose is to spend less time per lookup.
_FASTGATHER: bool | None = None


def _fastgather_enabled() -> bool:
    global _FASTGATHER
    if _FASTGATHER is None:
        _FASTGATHER = os.environ.get("VLLM_GFX906_PLE_FASTGATHER", "0") == "1"
        if _FASTGATHER:
            logger.info(
                "PLE ngram lookup: sorted-slice gather enabled "
                "(VLLM_GFX906_PLE_FASTGATHER=1). The loop still runs num_shards "
                "iterations, but a shard this batch does not touch now costs one "
                "integer compare instead of five torch CPU dispatches, and rows "
                "inside a shard are read in ascending order. Output is "
                "row-for-row identical to the default path."
            )
    return _FASTGATHER


def _reset_fastgather_cache() -> None:
    """Test-only: forget the cached ``VLLM_GFX906_PLE_FASTGATHER`` value."""
    global _FASTGATHER
    _FASTGATHER = None


# VLLM_GFX906_PLE_CAPTURE_PROBE=1 is the **E1 Phase 0 ceiling probe**
# (docs/e1-full-graph-plan.md §3), not a performance option. A CUDA graph
# replays an op's memcpys and replays *nothing* for its host work, so a PLE op
# captured into a full decode graph serves the capture-time embedding forever,
# silently -- which this box did for hours on 2026-09-20
# (docs/mtp-cudagraph-profile-crash.md §13.2). And with the blocking D2H below
# it does not even capture: hipErrorStreamCaptureUnsupported (§13.1). To price
# full-graph decode *before* paying for a graph-safe PLE, this knob makes the
# op capturable by omitting both the D2H and the gather while a capture is in
# progress: the graph then records only the H2D from ``_pinned_output`` (zeroed
# at allocation), which is the one node a real graph-safe PLE would contain.
# Outside capture the op is byte-for-byte the default path. **Decode output in
# a probe boot is invalid by construction: PLE contributes zeros.**
_CAPTURE_PROBE: bool | None = None


def _capture_probe_enabled() -> bool:
    global _CAPTURE_PROBE
    if _CAPTURE_PROBE is None:
        _CAPTURE_PROBE = os.environ.get("VLLM_GFX906_PLE_CAPTURE_PROBE", "0") == "1"
        if _CAPTURE_PROBE:
            logger.warning(
                "PLE CAPTURE PROBE ACTIVE (VLLM_GFX906_PLE_CAPTURE_PROBE=1): "
                "during CUDA graph capture the n-gram D2H and the host shard "
                "gather are SKIPPED and PLE contributes ZEROS, so replayed PLE "
                "rows are frozen. Output from this process is NOT VALID. "
                "Diagnostic only -- measures the ceiling of full-graph decode "
                "(docs/e1-full-graph-plan.md)."
            )
    return _CAPTURE_PROBE


def _reset_capture_probe_cache() -> None:
    """Test-only: forget the cached ``VLLM_GFX906_PLE_CAPTURE_PROBE`` value."""
    global _CAPTURE_PROBE
    _CAPTURE_PROBE = None


# ---------------------------------------------------------------------------
# E1 Phase 1 (docs/e1-full-graph-plan.md sec 3e): making a FULL graph capturable
# with host-staged PLE rows, and saying out loud which branch each capture shape
# took while doing it.
#
# Phase 0 was read by INFERENCE -- from later comparison counts I claimed which
# capture shapes had taken the graph-safe branch, and the inference was wrong
# once already (it wrote `divmod(num_tokens, num_reqs)` into the plan when the
# real mechanism is `num_reqs = min(num_tokens, max_num_reqs)`,
# cudagraph_utils.py:280). So the branch that decides whether a FULL graph
# contains a capturable H2D now reports what it did, per capture shape.
# ---------------------------------------------------------------------------

# Bounded label set, so the summary line cannot be widened by an unforeseen case.
CAPTURE_STAGE_LABELS = ("staged", "staged-mismatch", "dummy", "fell-through")
# capture shape (n) -> {label: count}. Cleared by the flush, which is also what
# makes the flush run once per capture round rather than once per process.
_CAPTURE_STAGE: dict[int, dict[str, int]] = {}


def _host_stage_active(ple_embedding: Any) -> bool:
    """Is the host going to stage this layer's rows on real decode steps?

    Mode 1 only, deliberately. Mode 2 (differential) keeps the device path fully
    in charge and retains the blocking D2H by design -- a graph that replays an
    H2D of host-staged rows has no device path left to disagree with, so
    ``GRAPHMODE=full`` and ``VLLM_GFX906_PLE_HOSTIDS=2`` are mutually exclusive.
    That is a consequence worth stating before someone boots it and reads the
    HIP error as a regression: the differential arm has to run on a piecewise
    boot (which is how boot #14 got its 22,637 steps), and what a FULL boot
    verifies instead is the generation tripwire.

    ``getattr(..., True)`` for the disabled flag so a stub embedding that never
    participated in staging -- the probe and prof tests use those -- falls
    through to the device path instead of taking this branch.
    """
    return (
        getattr(ple_embedding, "_hostids_disabled", True) is False
        and getattr(ple_embedding, "_pinned_output", None) is not None
        and ple_hostids.hostids_mode() == 1
    )


def _attest_capture_stage(ple_embedding: Any, n: int, how: str) -> None:
    """Record how the graph got its PLE H2D for an ``n``-token capture shape."""
    counts = _CAPTURE_STAGE.setdefault(n, {})
    counts[how] = counts.get(how, 0) + 1
    if how == "fell-through" and counts[how] == 1:
        # The next statement in the caller is the blocking D2H, which is the op
        # hipErrorStreamCaptureUnsupported fires on. Saying so BEFORE the HIP
        # error is the difference between a log that explains a dead boot and a
        # log that merely shows one.
        logger.warning(
            "PLE STAGE@CAPTURE: n=%d is falling through to the blocking "
            "device->host copy while a capture is in progress ('%s'). That copy "
            "is not capturable, so if this capture dies with "
            "hipErrorStreamCaptureUnsupported, this is why: host staging is off "
            "or was disabled for this layer. VLLM_GFX906_PLE_HOSTIDS=1 is a "
            "prerequisite for GRAPHMODE=full, not an optional extra "
            "(docs/e1-full-graph-plan.md sec 3e).",
            n,
            getattr(ple_embedding, "layer_name", "?"),
        )


def _flush_capture_stage_summary() -> None:
    """One INFO line per capture shape, on the first step that is not a capture.

    Deferred rather than immediate because the op runs once per PLE layer per
    capture iteration: logging in the branch would be a line per layer per shape
    per round, which is noise that hides the one thing the line is for. Clearing
    the dict is what ends the round, so a second capture round (piecewise then
    full, or a retry) gets its own summary.
    """
    for n in sorted(_CAPTURE_STAGE):
        counts = _CAPTURE_STAGE[n]
        body = " ".join(
            f"{label}={counts.get(label, 0)}" for label in CAPTURE_STAGE_LABELS
        )
        captured = counts.get("staged", 0) + counts.get("dummy", 0)
        layers = max(counts.values()) if counts else 0
        logger.info(
            "PLE STAGE@CAPTURE: capture shape n=%d -> %s. %d of %d PLE layer "
            "call(s) at this shape got their embedding rows from the host-staged "
            "pinned buffer (non-blocking H2D, capturable); %d went the device "
            "path (blocking D2H, NOT capturable). 'dummy' = capture of a padded "
            "batch no producer staged, whose values are irrelevant and which the "
            "host overwrites before every replay.",
            n,
            body,
            captured,
            layers,
            counts.get("fell-through", 0) + counts.get("staged-mismatch", 0),
        )
    _CAPTURE_STAGE.clear()


class MmapShardedNGramEmbedding(nn.Module):
    """CPU-resident PLE ngram embedding backed directly by mmap'd safetensors
    shard tensors, with no TP sharding and no copying. Every rank maps the
    same on-disk shard files; the OS page cache backs all of them with the
    same physical pages, so the full table's resident-RAM cost is paid once
    across the whole node, not once per worker process."""

    def __init__(
        self,
        num_shards: int,
        shard_row_capacity: int,
        embedding_dim: int,
        dummy_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_shards = num_shards
        self.shard_row_capacity = shard_row_capacity
        self.embedding_dim = embedding_dim
        # Dummy-weight loads (``--load-format dummy``) never deliver shard
        # tensors: the shards are read straight from the checkpoint's
        # safetensors files and are not parameters, so nothing generates them.
        # Serving zero rows for a missing shard keeps the path exercisable
        # without a checkpoint (the tiny-config harness); a real load still
        # raises.
        self._dummy_weights = dummy_weights
        self.params_dtype: torch.dtype | None = None
        self._shards: list[torch.Tensor | None] = [None] * num_shards
        self._warned_bad_ids = False
        # Set by finalize_shard_layout() from the shards actually loaded. Until
        # then (and for dummy weights) the configured uniform capacity is used.
        self._shard_starts: list[int] | None = None
        self._shard_rows: list[int] | None = None
        self._madvise_reported = False
        self._advised_bytes = 0

    def set_shard(self, shard_index: int, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            raise ValueError(
                f"PLE ngram shard {shard_index} must be loaded on CPU, "
                f"got device {tensor.device}"
            )
        if self.params_dtype is None:
            self.params_dtype = tensor.dtype
        elif tensor.dtype != self.params_dtype:
            raise ValueError(
                f"PLE ngram shard {shard_index} dtype {tensor.dtype} does not "
                f"match previously loaded shards' dtype {self.params_dtype}"
            )
        self._shards[shard_index] = tensor
        self._advise_shard(shard_index, tensor)

    def _advise_shard(self, shard_index: int, tensor: torch.Tensor) -> None:
        """Apply ``MADV_RANDOM`` to one shard if asked to. Off by default.

        ``VLLM_GFX906_PLE_MADV_RANDOM=1`` opts in; with the variable unset the
        mapping is left exactly as the kernel configured it, which is the
        behaviour every existing deployment expects. Opting in cannot change
        model output -- read-ahead affects how many bytes the kernel fetches,
        not which bytes -- so the knob exists to make the A/B reversible from
        the launch environment rather than because the change is risky.
        """
        if os.environ.get("VLLM_GFX906_PLE_MADV_RANDOM", "0") != "1":
            return
        ok, advised, detail = _advise_random(tensor)
        if not ok:
            if not self._madvise_reported:
                self._madvise_reported = True
                logger.warning(
                    "PLE ngram table: MADV_RANDOM could not be applied to shard "
                    "%d (%s). The read path keeps working with the kernel's "
                    "default read-ahead window; see docs/prefill-io-wall.md.",
                    shard_index,
                    detail,
                )
            return
        self._advised_bytes += advised
        if not self._madvise_reported:
            # First shard only: 128 shards x 4 ranks would otherwise be 512 lines.
            self._madvise_reported = True
            logger.info(
                "PLE ngram table: MADV_RANDOM is enabled "
                "(VLLM_GFX906_PLE_MADV_RANDOM=1). The kernel will fetch one page "
                "per fault instead of a read-ahead window, which on this box is "
                "the difference between ~28x read amplification and ~1x on a "
                "table larger than the page cache."
            )

    # Ids outside [0, num_shards * shard_row_capacity) have two known producers,
    # both legitimate, and neither is corruption of the table:
    #
    #   * the MTP drafter's "no sample here" marker. `sample_idx_mapping` is
    #     pre-filled with PADDING_SENTINEL (spec_decode/dflash/speculator.py:145)
    #     and reaches this lookup on every drafter warmup/capture.
    #   * a pinned host id buffer that nothing has written yet. On 2026-09-23 a
    #     graph-capture-time call arrived holding the poison pattern
    #     0xff80ff80ff80ff80 repeated, which killed three boots with
    #     "PLE ngram id out of range" -- see the 2026-09-23 entry in
    #     docs/gfx906/degradation.md.
    #
    # Both are folded onto row 0 rather than rejected or clamped: rejecting them
    # refuses a boot whose inputs are fine by the time they matter (this op is a
    # splitting op, so it runs eagerly and the captured/replayed value is read
    # later, from a buffer the producer has filled by then), and clamping the
    # whole tensor would send large positive garbage to the *last* row, which is
    # a real embedding and therefore a plausible wrong answer. Row 0 costs one
    # wrong row for slots whose result is discarded anyway.
    PADDING_SENTINEL = -1

    @property
    def table_rows(self) -> int:
        """Number of addressable ids: the loaded shards' rows once the layout is
        known, otherwise the configured uniform split's total."""
        if self._shard_rows is not None:
            return sum(self._shard_rows)
        return self.num_shards * self.shard_row_capacity

    def finalize_shard_layout(self, vocab_size: int | None) -> bool:
        """Fix the id -> (shard, row) map from the shards that were actually loaded.

        Called at the end of every `load_weights` call, so it is also the point at
        which a chunked delivery becomes complete. It needs the checkpoint's
        `ngram_heads_offsets` / `ngram_heads_vocab_sizes` buffers to have replaced
        the runtime's own layout -- only then is the model's id space known -- and
        the shards themselves, which are the only record of where each id's row
        lives. See `build_ple_shard_layout` for what goes wrong otherwise.

        Returns True once a layout was derived from every shard. Until then the
        configured uniform split stands, and a shard that never arrives still
        raises on use rather than being papered over.
        """
        if any(tensor is None for tensor in self._shards):
            if not self._dummy_weights:
                logger.debug(
                    "PLE ngram layout not finalised yet: %d of %d shards loaded",
                    sum(t is not None for t in self._shards),
                    self.num_shards,
                )
            return False
        rows = [int(tensor.shape[0]) for tensor in self._shards if tensor is not None]
        self._shard_starts, self._shard_rows = build_ple_shard_layout(
            rows,
            num_shards=self.num_shards,
            shard_row_capacity=self.shard_row_capacity,
            vocab_size=vocab_size,
        )
        return True

    # Ids outside the table's rows have two known producers,
    # both legitimate, and neither is corruption of the table:
    #
    #   * the MTP drafter's "no sample here" marker. `sample_idx_mapping` is
    #     pre-filled with PADDING_SENTINEL (spec_decode/dflash/speculator.py:145)
    #     and reaches this lookup on every drafter warmup/capture.
    #   * a pinned host id buffer that nothing has written yet. On 2026-09-23 a
    #     graph-capture-time call arrived holding the poison pattern
    #     0xff80ff80ff80ff80 repeated, which killed three boots with
    #     "PLE ngram id out of range" -- see the 2026-09-23 entry in
    #     docs/gfx906/degradation.md.
    #
    # Both are folded onto row 0 rather than rejected or clamped: rejecting them
    # refuses a boot whose inputs are fine by the time they matter (this op is a
    # splitting op, so it runs eagerly and the captured/replayed value is read
    # later, from a buffer the producer has filled by then), and clamping the
    # whole tensor would send large positive garbage to the *last* row, which is
    # a real embedding and therefore a plausible wrong answer. Row 0 costs one
    # wrong row for slots whose result is discarded anyway.
    PADDING_SENTINEL = -1

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type != "cpu":
            raise ValueError("MmapShardedNGramEmbedding requires CPU ids")
        original_shape = ids.shape
        flat_ids = ids.reshape(-1).long()
        table_rows = self.table_rows
        if flat_ids.numel() > 0:
            # One range test over the whole tensor, host-side (ids are CPU by
            # the check above), instead of one per shard inside the loop below.
            low, high = int(flat_ids.min()), int(flat_ids.max())
            if low < 0 or high >= table_rows:
                bad = (flat_ids < 0) | (flat_ids >= table_rows)
                if os.environ.get("VLLM_GFX906_PLE_STRICT", "0") == "1":
                    geometry = (
                        f"{self.num_shards} shards, rows {self._shard_rows}"
                        if self._shard_rows is not None
                        else f"{self.num_shards} shards x {self.shard_row_capacity}"
                    )
                    raise ValueError(
                        f"PLE ngram id out of range: ids span [{low}, {high}] "
                        f"but the table holds {table_rows} rows ({geometry}), "
                        f"{int(bad.sum())} of {flat_ids.numel()} ids affected "
                        "(VLLM_GFX906_PLE_STRICT=1)"
                    )
                if not self._warned_bad_ids:
                    self._warned_bad_ids = True
                    logger.warning(
                        "PLE ngram lookup: %d of %d ids were outside [0, %d) "
                        "(span [%d, %d]); folded onto row 0. A few during boot "
                        "or graph capture are expected -- the pinned id buffer "
                        "is not always written first, and the drafter passes "
                        "%d for 'no sample here' -- but a steady stream means "
                        "the id producer is broken. Set "
                        "VLLM_GFX906_PLE_STRICT=1 to raise instead of folding.",
                        int(bad.sum()),
                        flat_ids.numel(),
                        table_rows,
                        low,
                        high,
                        self.PADDING_SENTINEL,
                    )
                flat_ids = flat_ids.masked_fill(bad, 0)
        # One place answers "which shard, which row" -- for both gather forms and
        # for the profiling count below. v0.30 computed a shard index only for the
        # uniform case and let its gather loop derive masks from the checkpoint
        # boundaries; the sorted gather needs the same answer under either layout,
        # and a dispatch that passed names which were only ever bound in one of
        # the two branches is a NameError waiting for the first non-uniform
        # checkpoint rather than a bug anyone would see in review.
        shard_idx, local_idx = self._shard_of(flat_ids)
        if ple_prof.prof_enabled():
            # Distinct shards this lookup actually reads. The gather loop
            # runs a fixed num_shards iterations and `continue`s on an empty
            # mask, so this -- not num_shards -- is the count of real work,
            # and it is the quantity a look-ahead prefetch would reduce.
            self._last_shards_touched = int(torch.unique(shard_idx).numel())
        out = flat_ids.new_empty(
            (flat_ids.numel(), self.embedding_dim), dtype=self.params_dtype
        )
        # The gather below can only write rows whose shard index is in
        # range(num_shards). An id outside the table matches no mask, so its row
        # would keep whatever new_empty() found at that address: a silently wrong
        # embedding instead of an error. That is why the range test above folds
        # every out-of-range id onto row 0 first -- every row this function
        # returns is a row that was actually read.
        #
        # Where each id's row lives is decided by the shard boundaries
        # (finalize_shard_layout), not by a uniform stride: a checkpoint may split
        # its rows however it likes, and the runtime's stride is computed before
        # the checkpoint's vocab layout is known. Using the stride against a
        # different split both mis-maps ids and indexes off the end of a shard.
        if _fastgather_enabled():
            self._gather_sorted(flat_ids, out)
        else:
            self._gather_masked(flat_ids, shard_idx, local_idx, out)
        return out.reshape(*original_shape, self.embedding_dim)

    def _gather_masked(
        self,
        flat_ids: torch.Tensor,
        shard_idx: torch.Tensor,
        local_idx: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Gather one row per id, testing every shard against a boolean mask.

        Fixed-length loop over every shard, every call -- no data-dependent
        iteration count. Necessary for CUDA graph capture safety: a Python
        loop whose length depends on which shards this specific batch
        happens to touch can execute a different number of steps during
        warmup/capture than during replay. That invariant belongs to the
        gather as a whole, so ``_gather_sorted`` keeps it for the same price.

        The cost of this form is that "is this shard touched" is itself a
        tensor operation: ``shard_idx == shard`` allocates and fills a mask, and
        ``mask.any()`` reduces it, for all ``num_shards`` shards whether or not
        the batch touches them. That is ~5 torch CPU dispatches x 128 shards per
        lookup, which T3 measured as 3.46 ms of a ~65 ms decode step.
        """
        starts, rows = self._shard_starts, self._shard_rows
        for shard in range(self.num_shards):
            if starts is None:
                mask = shard_idx == shard
                local = local_idx[mask]
            else:
                start = starts[shard]
                mask = (flat_ids >= start) & (flat_ids < start + rows[shard])
                local = flat_ids[mask] - start
            if not mask.any():
                continue
            tensor = self._shards[shard]
            if tensor is None:
                if self._dummy_weights:
                    out[mask] = 0.0
                    continue
                raise RuntimeError(f"PLE ngram shard {shard} was never loaded")
            out[mask] = tensor.index_select(0, local)

    def _shard_of(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(shard index, row within that shard) for `ids`, under the live layout.

        The uniform configured stride while no checkpoint layout has been
        recorded, and the boundaries `finalize_shard_layout` read off the
        checkpoint after that. Both gather forms ask this one method, so they
        cannot disagree about where an id lives -- which is the whole reason
        the fastgather knob is allowed to change *how* rows are read and not
        *which* row a given id gets.

        Ascending ids therefore visit shards in ascending order under either
        layout, because a shard's ids are a contiguous range in both. That is
        the precondition `_gather_sorted` relies on.
        """
        starts, rows = self._shard_starts, self._shard_rows
        if starts is None:
            shard = torch.div(ids, self.shard_row_capacity, rounding_mode="floor")
            return shard, ids - shard * self.shard_row_capacity
        starts_t, ends_t = self._layout_tensors(starts, rows)
        # torch.searchsorted(sorted_sequence, input) picks the first shard whose
        # exclusive end is greater than the id, which is the shard that owns it:
        # the id that starts a shard is inside that shard, not the previous one.
        # (There is no Tensor.searchsorted method reachable here -- the first draft
        # called ids.searchsorted() and died with AttributeError on the first
        # non-uniform lookup, which is also what two of #40's own tests caught.)
        shard = torch.searchsorted(ends_t, ids, right=True)
        return shard, ids - starts_t[shard]

    def _layout_tensors(self, starts, rows):
        """Cached int64 tensors for the checkpoint layout, rebuilt if it changes.

        Built once per layout rather than per lookup: this runs on every PLE
        layer of every step. The cache key is the identity of the two lists,
        which `finalize_shard_layout` writes exactly once per engine.
        """
        key = (id(starts), id(rows))
        if getattr(self, "_layout_key", None) != key:
            self._layout_starts_t = torch.tensor(starts, dtype=torch.long)
            self._layout_ends_t = torch.tensor(
                [s + r for s, r in zip(starts, rows)], dtype=torch.long
            )
            self._layout_key = key
        return self._layout_starts_t, self._layout_ends_t

    def _gather_sorted(self, flat_ids: torch.Tensor, out: torch.Tensor) -> None:
        """Gather the same rows, with untouched shards costing one int compare.

        Sorting the ids makes each shard's rows a contiguous span of the
        destination, so "is this shard touched" becomes ``per_shard[shard] ==
        0`` -- a Python integer compare against a 128-element list read out of
        the host tensor once -- and the actual reads become ``index_select``
        over a contiguous slice instead of a boolean-masked gather. The loop
        length stays ``self.num_shards``, which is the CUDA-graph invariant
        ``_gather_masked`` documents; only the per-iteration cost changes.

        Two properties that make this safe to swap in behind the knob:

        * **Bitwise-identical output.** A gather is a per-row copy, so a
          permutation of the reads cannot change a value. ``torch.sort`` is not
          required to be stable, and it does not matter here: two equal ids
          read the same table row, so whichever duplicate lands in which output
          slot, both slots get that row's bytes.
        * **Ascending reads inside a shard.** Sorting by id sorts by shard and
          then by row, so the mmap page touches of one shard go in address
          order instead of in batch order. T3 showed gather time tracks page
          faults near-linearly (306k minor faults -> 342 ms, 132k -> 173 ms),
          so locality is worth having on its own.

        The row within a shard comes from ``_shard_of``, and the sorted order is
        applied to it with the same permutation as the ids (``local[order]``)
        rather than recomputed from a stride. This is the point where the port
        had to change my pre-#40 code rather than keep it: a
        ``remainder(sorted_ids, shard_row_capacity)`` here would silently hand
        back the wrong row for any checkpoint whose shards are not equally long,
        which is the bug #40 exists to fix.

        ``flat_ids`` must already be folded into the table's id space by
        ``forward``: ``torch.bincount`` rejects negatives, so calling this
        directly with unfolded ids is a programming error rather than a wrong
        answer.
        """
        n = flat_ids.numel()
        if n == 0:
            return
        shard_idx, local = self._shard_of(flat_ids)
        counts = torch.bincount(shard_idx, minlength=self.num_shards)
        _sorted_ids, order = torch.sort(flat_ids)
        local_sorted = local[order]
        # One host read of num_shards integers. This op is a splitting op and
        # runs eagerly (see qwen4_exp_amd_ple_ngram_embedding), and the loop
        # this replaces already read a value-dependent ``mask.any()`` once per
        # shard, so this is not a new class of host-visible read.
        per_shard = counts.tolist()
        gathered = out.new_empty((n, self.embedding_dim))
        pos = 0
        for shard in range(self.num_shards):
            count = per_shard[shard]
            if count == 0:
                continue
            tensor = self._shards[shard]
            if tensor is None:
                raise RuntimeError(f"PLE ngram shard {shard} was never loaded")
            span = slice(pos, pos + count)
            gathered[span] = tensor.index_select(0, local_sorted[span])
            pos += count
        if pos != n:
            # Would mean ids reached the gather that belong to no shard, which
            # is the silent-wrong-row failure forward()'s range test exists to
            # prevent. Fail loudly instead of copying uninitialized rows.
            raise RuntimeError(
                f"PLE sorted gather accounted for {pos} of {n} ids; the id "
                "range guard in forward() did not run"
            )
        out.index_copy_(0, order, gathered)

def _bounded_ids(arr, cap: int = 16, compact: bool = False) -> str:
    """Render an ids array for a log line, with any truncation visible.

    A refusal that names only the first differing pair proves the buffer is not the
    host's, but not *which row* is foreign -- and under a drafter "which row, which
    pass" is the only question left (boot #11's whole contribution was a refusal that
    could not be attributed that far).  Rows are contiguous groups of
    ``heads_per_ngram`` in the flat array, so the array is the map.  The cap and the
    explicit ``+N more`` marker are here because a log line on this path must not be
    able to grow with the batch, and a dump that silently stops at a limit reads as a
    complete one.

    ``compact`` only drops the space after each comma, so a width-bearing field like
    ``qsl(width=4)=[0,4,4,4]`` reads as one token.  The cap and the marker are
    deliberately identical in both spacings: one convention, not two.
    """
    flat = [int(v) for v in np.asarray(arr).reshape(-1).tolist()]
    sep = "," if compact else ", "
    if len(flat) <= cap:
        return "[" + sep.join(str(v) for v in flat) + "]"
    shown = sep.join(str(v) for v in flat[:cap])
    return f"[{shown}, ... +{len(flat) - cap} more]"


_PACKED_DUMP_ROWS = 16
_PACKED_DUMP_CELLS = 16
_QSL_DUMP = 32
"""Caps for the differential's log lines.

Row count, cells per row, and query boundaries per vector. Every one of them is a
constant, which is what makes "this line cannot grow with traffic" a property of the
code rather than a hope about the workload: the workspace is ``max_num_reqs`` by
``max_total_tokens`` wide, so an uncapped print of it is a multi-kilobyte line on a
prefill step.
"""


def _shape_str(values: Any) -> str | None:
    """``(3,192)`` for a tensor or an array, or None if there is no shape to name.

    Reads the shape metadata only, so it costs no device access -- which matters
    because a geometry field that needs a sync is a field that changes what the
    instrument measures.
    """
    shape = getattr(values, "shape", None)
    if shape is None:
        return None
    return "(" + ",".join(str(int(d)) for d in shape) + ")"


def _host_view(values: Any) -> Any:
    """A numpy view of a tensor or an array, or None for nothing to read.

    The differential reads tensor *references* to the runner's own buffers, which
    may equally well be numpy in a probe or a test fixture, so the same two-line
    dance the existing legs do is factored out here rather than copied a fourth
    time.
    """
    if values is None:
        return None
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _render_vector(name: str, values: Any, cap: int = _QSL_DUMP) -> str:
    """``qsl(width=4)=[0,4,4,4]`` for a query-boundary vector, or a *named* absence.

    The width is part of the field name rather than a footnote: the whole P1.14
    finding is that two qsl views of different widths were compared as if they were
    one object, and a value without its width is how that got past the instrument
    for twelve boots.
    """
    if values is None:
        return f"{name}(n/a(no buffer was handed over))"
    flat = np.asarray(values).reshape(-1)
    return f"{name}(width={flat.size})={_bounded_ids(flat, cap=cap, compact=True)}"


def _fill_extent(values: Any, sentinel: int) -> int:
    """Count the leading cells of a packed row that are not the fill sentinel.

    ``_compute_ngram_ids`` opens with ``packed.fill_(eos_token_id)`` and scatters
    exactly one cell per token it was handed, so the first cell still holding the
    fill value is the first cell the scatter never reached -- a measurement of the
    extent the fill worked from, taken off the scratch itself.

    It is a *lower bound*: a real token that happens to equal ``eos_token_id``
    truncates the count early. On this model the fill value is the vocabulary's
    EOS and decode rows are draft tokens, so the case is rare but not impossible,
    and an instrument that overstates what the scratch proves is worse than one
    that says "at most".
    """
    row = np.asarray(values).reshape(-1)
    hit = np.flatnonzero(row == sentinel)
    return int(hit[0]) if hit.size else int(row.size)


def _bounded_rows(
    values: Any,
    cap: int = _PACKED_DUMP_ROWS,
    cells: int = _PACKED_DUMP_CELLS,
) -> str:
    """Render the rows of a 2D workspace, with both caps made visible.

    Same ``+N more`` convention as :func:`_bounded_ids`, applied one level up: a
    workspace print that stops at a limit has to say it stopped, or the reader
    takes a partial view of the scratch for the whole thing.
    """
    n = int(values.shape[0])
    rows = [
        _bounded_ids(values[r, :], cap=cells, compact=True) for r in range(min(n, cap))
    ]
    if n > len(rows):
        return "[" + ", ".join(rows) + f", ... +{n - len(rows)} more]"
    return "[" + ", ".join(rows) + "]"


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        layer_name: str,
        runtime_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.runtime_dtype = runtime_dtype
        self.layer_name = layer_name
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self._pinned_ngram_ids = torch.empty(
            (max_total_tokens, self.ngram_heads), dtype=torch.long, device="cpu"
        ).pin_memory()
        self._pinned_output = torch.zeros(
            (max_total_tokens, self.embedding_dim),
            dtype=self.runtime_dtype,
            device="cpu",
        ).pin_memory()
        # Host-side id staging (VLLM_GFX906_PLE_HOSTIDS). Row counts, not
        # booleans: a count only means "this step, these rows" if it equals the
        # row count the consumer is about to write, so the flag is checked
        # against `n` rather than trusted on its own. -1 = nothing staged.
        self._host_stage_rows = -1
        self._host_check_rows = -1
        self._host_check_ids: torch.Tensor | None = None
        self._host_hash_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._hostids_disabled = False
        self._host_stage_steps = 0
        self._host_id_mismatches = 0
        self._host_id_matches = 0
        # Differential-mode provenance, for the mismatch dump only. A row that
        # disagrees is worth nothing unless the dump also shows the three tokens
        # that produced it and where the host got them, because the arithmetic
        # itself is now known to match the device (tools/gfx906/cpu-diff-ngram.py).
        self._host_check_inputs: tuple[Any, ...] | None = None
        self._host_check_draft: tuple[np.ndarray, np.ndarray, int] | None = None
        # Input-agreement gate (P1.10b). ``_host_input_checks`` counts the
        # comparisons that went through because the device's own copy of the
        # inputs matched what the host mirrored; ``_host_input_skips`` counts the
        # ones refused because they did not, or because the question could not be
        # asked. Refusals are counted separately from mismatches on purpose: a
        # comparison the harness cannot attribute says nothing about the mirror,
        # and a number that mixes the two would let the harness's blind spots be
        # read as evidence against the host path.
        #
        # There is deliberately no P1.9 here. The snapshot copies that used to sit
        # in ``forward`` are gone: forward IS the compiled region, so a Python
        # guard there is evaluated at trace time and the artifact that a stable
        # AOT cache key keeps loading decides what runs -- the key of this model
        # did not change for eight days across dozens of source edits, so the
        # copies were compiled out of the boot that was supposed to read them. The
        # 23:42 boot is the one time they did get compiled in (drafter-off produced
        # a fresh key) and it died of an illegal device access at the graph-memory
        # profiling teardown; a pre-existing crash class hits the same path, but a
        # diagnostic has no business being the reason a serving boot dies.
        self._host_input_checks = 0
        self._host_input_skips = 0
        # Steps mode 1 refused to stage. Separate from ``_host_input_skips``
        # because the two refusals mean different things about served output.
        self._host_stage_refusals = 0
        # Steps where the device's ngram context was not handed over, so the gate
        # compared ids and query_start_loc and the third leg stayed dark. Counted
        # because a leg nobody checks reads as a leg that passed: this number is
        # printed with every refusal and with the self-disable.
        self._host_ctx_unchecked = 0
        # The op that compares ids is a separate custom op and receives only the
        # ids, so forward hands the device's inputs over through this attribute.
        self._host_check_dev_inputs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        # P1.14: the qsl view the compute is actually handed, captured in the
        # worker next to the staging call. It is NOT ``input_batch.query_start_loc``
        # (which the input gate compares against the host's): ``prepare_inputs``
        # stages ``ple_query_start_loc[: num_reqs_padded + 1]`` into forward, a
        # different buffer at the padded width, so until now nothing in the
        # harness ever read the array the fill math consumed. Overwritten by the
        # same act that sets ``_host_check_rows``, so it cannot outlive the step it
        # describes any more than the staged rows can.
        self._host_consumed_qsl: Any = None
        # Generation tripwire (P1.5). ``_host_stage_gen`` counts staged steps and
        # is copied into pinned memory in the same act that writes the rows, so a
        # value in the pinned slot means "rows for that step are in the buffer
        # next to it". ``_gen_device`` is what the graph writes; ``_pinned_gen_read``
        # is where the host reads it back without waiting. All four 8-byte buffers
        # exist even with the knob off -- they are nothing, and a None-check per
        # step is worse than 8 bytes.
        self._host_stage_gen = 0
        self._pinned_gen = torch.zeros(1, dtype=torch.long, device="cpu").pin_memory()
        self._pinned_gen_read = torch.zeros(
            1, dtype=torch.long, device="cpu"
        ).pin_memory()
        self._gen_device: torch.Tensor | None = None
        self._gen_event = torch.cuda.Event()
        self._gen_pending_expected = 0
        self._gen_check_countdown = ple_hostids.hostids_check_interval()
        self._gen_stuck = 0
        self._gen_checks_ok = 0
        self._gen_trips = 0
        self._staged_branch_captured = False
        if ple_hostids.hostids_mode() != 0:
            _HOST_STAGEABLE.add(self)
            logger.info(
                "PLE HOSTIDS: embedding '%s' registered for host-staged ids (mode %s)",
                layer_name,
                ple_hostids.hostids_mode(),
            )

        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        shard_row_capacity = (
            padded_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        self.ngram_embedding = MmapShardedNGramEmbedding(
            dummy_weights=(
                get_current_vllm_config().load_config.load_format == "dummy"
            ),
            num_shards=self.split_ngram_parts,
            shard_row_capacity=shard_row_capacity,
            embedding_dim=self.head_dim,
            # params_dtype=torch.get_default_dtype(),
        )

        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padded_buffer",
            torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    @staticmethod
    def _compute_ngram_ids(
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        *,
        multipliers: torch.Tensor,
        vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        packed: torch.Tensor,
        positions: torch.Tensor,
        eos_token_id: int,
        ngram_size: int,
        heads_per_ngram: int,
    ) -> torch.Tensor:
        """Map a packed token batch to ngram embedding ids.

        Pure tensor math with the workspaces and hash buffers passed in, so the
        device path and the host-side lookahead prefetch (ple_prefetch.py) run
        *the same code* on different devices. A duplicate of this arithmetic in
        the prefetcher would silently warm the wrong pages, and wasted NVMe
        traffic is the one failure mode a read-ahead cannot show in its output.
        """
        num_reqs = query_start_loc.numel() - 1
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        packed.fill_(eos_token_id)
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = Qwen4ExpNGramEmbedding._shift_precompute(
            context, eos_token_id
        )
        shifted = [context]
        for shift in range(1, ngram_size):
            shifted.append(
                Qwen4ExpNGramEmbedding._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    eos_token_id,
                )
            )
        adjusted_columns = columns + ngram_size - 1
        id_blocks = []
        for ngram in range(2, ngram_size + 1):
            start = (ngram - 2) * heads_per_ngram
            end = start + heads_per_ngram
            mixed = shifted[0] * multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[index] * multipliers[index])
            sizes = vocab_sizes[start:end]
            offsets = head_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        ngram_ids = torch.cat(id_blocks, dim=-1)
        return ngram_ids

    # ------------------------------------------------------------------
    # Host-side id staging (E1 Phase 1, VLLM_GFX906_PLE_HOSTIDS)
    # ------------------------------------------------------------------
    #
    # A captured FULL decode graph replays the H2D out of ``_pinned_output`` and
    # nothing else: the blocking D2H of the ids and the host gather that produced
    # them ran once during capture and are not graph nodes. For the replay to be
    # correct, the host must therefore fill ``_pinned_output`` itself every step,
    # from ids it obtained without asking the device for anything. That is what
    # these methods do; :meth:`Qwen4ExpModelState._stage_host_ple_ids` is the
    # producer, and it runs every step even under graph replay.
    #
    # Single-buffer reuse is safe here for the same reason today's path is safe:
    # the H2D is enqueued on the compute stream ahead of this step's sampling,
    # and sync scheduling makes the host wait for sampled tokens before the next
    # step is prepared. Under async scheduling that ordering disappears, which is
    # exactly why the config guard refuses HOSTIDS together with it.

    def _host_hash_buffers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU views of the three hash-layout buffers, snapshotted once.

        They are registered on the CPU and end up on the device with the model,
        so a per-step ``.cpu()`` would be a synchronising D2H on a path whose
        entire purpose is to remove host waits. Snapshot on first use instead:
        ``load_weights`` has run by then and nothing mutates them afterwards
        (the cache is dropped if it runs again).
        """
        if self._host_hash_cache is None:
            self._host_hash_cache = (
                self.layer_multipliers.detach().cpu().numpy(),
                self.ngram_heads_vocab_sizes.detach().cpu().numpy(),
                self.ngram_heads_offsets.detach().cpu().numpy(),
            )
        return self._host_hash_cache

    def stage_host_step(
        self,
        input_ids: np.ndarray,
        query_start_loc: np.ndarray,
        ngram_context: np.ndarray,
        *,
        gather: bool,
        device_inputs: tuple[Any, ...] | None = None,
        consumed_qsl: Any = None,
    ) -> bool:
        """Turn this step's host-computed row ids into staged embedding rows.

        ``gather=False`` is differential mode: the ids are kept in a side buffer
        for :meth:`compare_host_ids` against the device's own answer, and the
        device path is left completely in charge.

        ``consumed_qsl`` is differential-only too, and it is the *forward* input:
        the exact ``query_start_loc`` view ``prepare_inputs`` hands the model this
        step, as opposed to the runner's array the gate compares against. See
        :meth:`_consumed_report`.
        """
        if self._hostids_disabled:
            return False
        multipliers, vocab_sizes, head_offsets = self._host_hash_buffers()
        ids = ple_hostids.compute_ngram_ids_host(
            input_ids,
            query_start_loc,
            ngram_context,
            multipliers=multipliers,
            vocab_sizes=vocab_sizes,
            head_offsets=head_offsets,
            eos_token_id=self.eos_token_id,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
        )
        rows = int(ids.shape[0])
        if rows > self._pinned_ngram_ids.shape[0]:
            return False
        if not gather:
            if self._host_check_ids is None:
                # Never DMA'd: the device's ids arrive through the normal D2H.
                self._host_check_ids = torch.empty_like(self._pinned_ngram_ids)
            self._host_check_ids[:rows].numpy()[:] = ids
            self._host_check_rows = rows
            # References, not copies: ``build_decode_step_inputs`` allocates these
            # fresh per step, and the compare happens inside the very op of this
            # step, before anything else can recycle them.
            # P1.12 adds a fourth member: the compute's own packed workspace.
            # It is written by nobody else, so reading it at op time is the only
            # view of the cells the hash actually consumed, as opposed to the
            # arrays the caller passed it.
            self._host_check_inputs = (
                input_ids,
                query_start_loc,
                ngram_context,
                self.padded_buffer,
            )
            # The device half of the same step's inputs, captured HERE rather than
            # in forward(). The 14:54 boot proved the forward-time stash is dead
            # code on a compiled path: dynamo runs the Python body at trace time
            # and the op impl on every step afterwards, so the attribute the
            # compare read was whatever tracing left behind (None after the first
            # compare reset it). Staging is plain Python in the worker, it sees
            # the batch this step will consume, and it is the only place both
            # sides of the comparison are in scope at once. References, not
            # copies: these are the runner's persistent buffers, and the compare
            # runs inside this same step.
            if device_inputs is not None:
                self._host_check_dev_inputs = device_inputs
            self._host_consumed_qsl = consumed_qsl
            return True
        dst = self._pinned_ngram_ids[:rows]
        dst.numpy()[:] = ids
        result = self.ngram_embedding(dst).flatten(-2)
        self._pinned_output[:rows].copy_(result.to(dtype=self.runtime_dtype))
        self._host_stage_rows = rows
        self._host_stage_steps += 1
        # Same-act rule: the generation counter goes into pinned memory after the
        # rows it describes are already in the buffer beside it, never before. A
        # counter that leads its rows would let a replay look fresh while reading
        # the previous step's embedding rows -- the exact failure this counter is
        # installed to detect. Differential mode takes the early return above and
        # never advances the counter, so the tripwire stays inert there by
        # construction rather than by a mode check.
        self._host_stage_gen += 1
        self._pinned_gen[0] = self._host_stage_gen
        if self._host_stage_steps == 1:
            logger.info(
                "PLE HOSTIDS: staged %d rows of host-computed ids for '%s' "
                "(first step). Device D2H is skipped for these steps.",
                rows,
                self.layer_name,
            )
        return True

    def take_host_stage(self, rows: int) -> bool:
        """Consume the staged flag. True only if it covers exactly ``rows``.

        Kept as the tested description of the rule; the op reads the attribute
        directly (getattr) so stub embeddings used by the probe/prof tests, which
        never set it, keep taking the device path instead of raising inside a
        capture. Consumed unconditionally when present: a flag left over from a
        step that never reached this op must not be allowed to skip a later
        step's D2H, which would silently feed that step the previous step's rows.
        """
        staged = self._host_stage_rows
        self._host_stage_rows = -1
        return staged == rows

    def forget_host_stage(self) -> None:
        """Drop any staged flag left by a step whose op body never ran.

        ``take_host_stage`` can only consume a flag by running, and under a captured
        graph the op body does not run: a step that staged successfully therefore
        leaves ``_host_stage_rows`` set indefinitely. That is harmless while every
        later step also replays, and dangerous the moment one does not -- the
        dispatch veto sends a refused step eager precisely because it must NOT
        replay, and an eager op body that finds the previous step's flag, with the
        previous step's row count (same batch shape, so it matches), skips its D2H
        and reads rows that belong to a different step. This is the failure
        :meth:`take_host_stage` names in its own docstring; the veto is the first
        mechanism that can actually reach it, so the state clears the flag at the
        top of every staging attempt and this is how.
        """
        self._host_stage_rows = -1

    def _alloc_gen_scratch(self) -> None:
        """Create the device mirror of ``_pinned_gen`` on the weights' device."""
        self._gen_device = torch.zeros(
            1, dtype=torch.long, device=self.layer_multipliers.device
        )

    def _gen_scratch(self, device: torch.device) -> torch.Tensor:
        """The device scratch the graph copies the generation into.

        Falls back to lazy creation for a module that never ran ``load_weights``
        (a stand-in in a test), but refuses to allocate inside a capture instead
        of letting the allocation fail in whatever way the backend chooses.
        """
        scratch = self._gen_device
        if scratch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PLE generation scratch was never allocated before graph "
                    "capture; the generation tripwire cannot be recorded into a "
                    "graph retroactively. Load weights before capturing."
                )
            scratch = torch.zeros(1, dtype=torch.long, device=device)
            self._gen_device = scratch
        return scratch

    def check_host_gen(self) -> str:
        """Sample what the graph delivered, and refuse if it is not fresh.

        Called once per step from the model state, before this step stages
        anything, and independent of whether this step can be staged at all -- a
        step the host refused is exactly the step whose replay is worth looking
        at. Never synchronises: the read is issued non-blocking on step N, its
        event is polled with ``query()`` on a later step, and a read that has not
        landed is left in flight rather than waited on. A tripwire that adds a
        host wait to the step would be priced in the same currency as the bug it
        catches.

        The expected value is the generation staged as of *before* this step
        stages, because the copy issued below is stream-ordered after the previous
        step's replay and before this step's. Computing it after staging instead
        would compare the device against a value it cannot possibly have seen yet
        and fire on every healthy step -- which is a red control here, not a
        hypothetical.
        """
        if self._hostids_disabled or self._gen_device is None:
            return ""
        interval = ple_hostids.hostids_check_interval()
        expected = self._gen_pending_expected
        if expected:
            if not self._gen_event.query():
                self._gen_stuck += 1
                if self._gen_stuck > 4 * interval:
                    self._fire_gen(
                        f"generation read did not land in {self._gen_stuck} steps "
                        "(the device scratch is not being written, or the stream is "
                        "wedged)"
                    )
                return "in-flight"
            self._gen_stuck = 0
            self._gen_pending_expected = 0
            read = int(self._pinned_gen_read[0])
            verdict = ple_hostids.gen_verdict(read, expected)
            if verdict == "ok":
                self._gen_checks_ok += 1
                return verdict
            self._fire_gen(
                f"the device reported generation {read} for a step the host staged "
                f"as generation {expected}"
            )
            return verdict
        if self._host_stage_gen == 0:
            # Nothing has been staged yet, so there is nothing the graph could
            # have delivered. Reporting this as "ok" would make the first N steps
            # of every boot look verified.
            return "no-staging"
        self._gen_check_countdown -= 1
        if self._gen_check_countdown > 0:
            return ""
        self._gen_check_countdown = interval
        self._gen_pending_expected = self._host_stage_gen
        self._pinned_gen_read.copy_(self._gen_device, non_blocking=True)
        self._gen_event.record()
        return "issued"

    def _fire_gen(self, why: str) -> None:
        """The handoff is not fresh. What happens next depends on reachability.

        If no graph has captured the staged branch, the device path genuinely runs
        every step, so disabling the host path is a real fallback and the boot
        stays correct. If a graph *has* captured it, the op body no longer runs at
        replay: there is no device path to fall back to, and every further step
        would serve embedding rows belonging to a different step. Plausible text is
        the worst possible output of a lookup table, so this raises instead.

        A raise on a hot path is a behaviour change, and this one is deliberately
        narrow: it is unreachable unless the staged branch was captured, which is
        unreachable unless the knob is on and a FULL graph was built around it.
        """
        self._gen_trips += 1
        if self._staged_branch_captured:
            raise RuntimeError(
                f"PLE HOSTIDS generation tripwire for '{self.layer_name}': {why}. A "
                "captured graph is replaying host-staged PLE rows and the handoff is "
                "not fresh, so every step from here would embed the wrong step's "
                "tokens. The device path is not reachable from inside a replay, so "
                "continuing would serve frozen embeddings -- the 2026-09-20 failure "
                "mode. Stopping the engine is the only safe action."
            )
        self.disable_hostids(f"generation tripwire: {why}")

    def note_draft_provenance(
        self, values: np.ndarray | None, row_index: np.ndarray | None, gen: int
    ) -> None:
        """Remember which staged draft block this step's ids were built from.

        Differential mode only, and only so a mismatch can name its inputs: the
        dumps that have been wrong so far described the *output* rows. Column 1 of
        the mirror is the current suspect, so the dump prints the whole staged
        block, the slot-to-row map, and the generation the host consumed.
        """
        self._host_check_draft = (values, row_index, gen)

    def _id_dump(
        self,
        rows: int,
        host: np.ndarray,
        dev: np.ndarray,
        device_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> str:
        """Everything a mismatch needs, on one line, mode 2 only.

        The synchronising D2H of the device's inputs is the reason this runs on
        the mismatch path and nowhere else -- ``compare_host_ids`` then disables
        the host path for the life of the process, so the wait costs one step.
        """
        parts = []
        inputs = self._host_check_inputs
        if device_inputs is None and inputs is None:
            parts.append("inputs=none(both sides missing)")
        elif device_inputs is None or inputs is None:
            # Naming the missing side is the point: the 14:54 boot printed the
            # draft half and silently dropped the input half, which read like a
            # dump that had everything.
            parts.append(
                "inputs missing on the "
                + ("device" if device_inputs is None else "host")
                + " side"
            )
        if device_inputs is not None and inputs is not None:
            d_ids, d_qsl, d_ctx = device_inputs
            # Positional, not unpacked: P1.12 appended the compute's packed
            # workspace as a fourth member, and a 3-tuple unpack would make every
            # staged mismatch raise inside the diagnostic instead of printing it.
            h_ids, h_qsl, h_ctx = inputs[0], inputs[1], inputs[2]

            def _np(t: Any) -> Any:
                # A None member means "this side has no device counterpart worth
                # reading" -- which since P1.11 only happens when the integration
                # did not hand the window over, never because reading it was skipped.
                if t is None:
                    return None
                return t.detach().cpu().numpy()

            dev_ids = _np(d_ids)
            dev_qsl = _np(d_qsl)
            dev_ctx = _np(d_ctx)
            parts.append(
                f"inputs host={h_ids[:rows].tolist()} "
                + (
                    f"device={dev_ids[:rows].tolist()}"
                    if dev_ids is not None
                    else "device=n/a"
                )
            )
            # The runner's query_start_loc is padded to the batch capacity, and
            # that padding is itself a suspect: a padded boundary makes the
            # compute split the token stream into requests the host never saw.
            # Cutting it to the host's length hid exactly that, so print the
            # whole thing -- it is at most max_num_seqs+1 numbers.
            parts.append(
                f"qsl host={h_qsl.tolist()} "
                + (
                    f"device={dev_qsl.tolist()}"
                    if dev_qsl is not None
                    else "device=n/a"
                )
            )
            # The window is per REQUEST, not per token: [num_reqs, ngram_size - 1].
            # Slicing it by token-row index would be a category error, so print the
            # whole thing -- at most max_num_seqs x 2 numbers -- and name the side
            # that is missing instead of letting a bare "n/a" read as "agreed".
            host_ctx_txt = (
                "n/a(no host window staged)" if h_ctx is None else str(h_ctx.tolist())
            )
            dev_ctx_txt = "n/a(no device window handed over)"
            if dev_ctx is not None:
                dev_ctx_txt = str(
                    dev_ctx.tolist()
                    if h_ctx is None
                    else dev_ctx[: h_ctx.shape[0]].tolist()
                )
            parts.append(f"ctx host={host_ctx_txt} device={dev_ctx_txt}")
        draft = self._host_check_draft
        if draft is not None:
            values, row_index, gen = draft
            if values is None:
                parts.append("draft=none")
            else:
                parts.append(f"draft gen={gen} rows={values.shape}")
                parts.append(
                    f"draft block={np.asarray(values).reshape(-1)[:16].tolist()}"
                )
                parts.append(
                    "draft row_idx="
                    + str(
                        np.asarray(row_index)[:8].tolist()
                        if row_index is not None
                        else "?"
                    )
                )
        where = np.argwhere(host != dev)
        bad = sorted({int(r) for r, _ in where})
        parts.append(f"rows differing={bad}")
        parts.append(self._packed_dump(self._host_check_inputs, bad))
        # What the compute itself consumed (P1.9). The section above reads the
        # runner's buffers from Python at compare time; this one is a copy taken
        # inside the graph, so it is the only view of the arrays the hash
        # actually read. Saying whether the two agree is the whole point: if they
        # differ, an inputs-identical mismatch is explained, and if they agree,
        # the input side is closed and the arithmetic is the suspect again.
        parts.append(f"ids rows={int(dev.shape[0]) if dev.ndim == 2 else dev.shape[0]}")
        # An ids mismatch only reaches this dump through the input-agreement gate,
        # which is what makes it worth the sync: the device demonstrably read the
        # tokens the host mirrored, so the arithmetic is the only thing left to
        # explain the disagreement. Say so, because a dump that reads as complete
        # when the attribution step is what changed is how a harness bug gets
        # charged to the mirror for the third time.
        parts.append("input gate: agreed (device read the staged tokens)")
        for r in bad[:2]:
            # Print the HEADS THAT ACTUALLY DIFFER, not the first four. The 14:54
            # boot truncated every row to [:4], so its second bad row looked
            # identical on both sides in the very line meant to explain it.
            cols = sorted({int(c) for c in where[where[:, 0] == r][:, 1]})
            shown = cols[:4]
            parts.append(f"row {r} differs in {len(cols)} head(s) {shown}")
            parts.append(f"host row{r}={host[r, shown].tolist()}")
            parts.append(f"dev  row{r}={dev[r, shown].tolist()}")
        return "; ".join(parts)

    def _consumed_report(self, ids_len: int, stage_rows: int, dev_inputs: Any) -> str:
        """One line per compared step, naming the extent the fill consumed (P1.14).

        Why the scratch is the witness. ``_compute_ngram_ids`` opens with
        ``packed.fill_(eos_token_id)``, and ``positions`` is
        ``positions_buffer[:num_tokens]`` -- a registered ``arange`` buffer that is
        never written per step -- so a scatter over ``ids_len`` tokens writes cells
        ``0 .. qsl[1] - 1`` of row 0 and nothing beyond them. Therefore
        ``extent_row0 == consumed_qsl[1]`` whenever the fill and the buffer it read
        agree, and the equality is measured off the device's own scratch rather
        than off any array the harness believes in.

        Why the two qsl vectors are printed side by side. Boot #12's row 0 was
        ``[17383, 653, EOS, EOS]`` at ``ids_len=4``: the fill had written two cells
        for a four-token step, which is what a qsl whose first boundary is 2 looks
        like (with ``arange`` positions, cells 0 and 1 go to request 0 and cells 2
        and 3 land in *other* rows). The gate printed ``qsl device=[0,4]`` and
        concluded the inputs agreed, because it reads ``input_batch.query_start_loc``
        -- the runner's array, sliced to the real ``num_reqs`` -- while forward is
        handed ``ple_query_start_loc[: num_reqs_padded + 1]``, a different buffer at
        the capture-padded width. Two objects, two widths, compared as one. Printing
        both with their widths is the entire change; the line says which of them the
        fill listened to.

        Cost, stated because the house rule is that a check must not be priced in
        the currency of the bug it catches. Three reads per *compared* step: two
        boundary vectors of at most ``max_num_reqs + 1`` integers, and row 0 of the
        scratch cut to ``ids_len`` cells -- strictly smaller than the ids buffer
        this mode copies a few statements earlier in the same op body. That copy is
        blocking, so the stream wait is already paid before any of these are
        issued: they add launches, not waits. Nothing here runs when the knob is
        off or at 1 (mode 1 never sets ``_host_check_rows``, so
        :meth:`compare_host_ids` is never entered), and the no-sync test counts
        ``Tensor.cpu`` and ``Tensor.item`` calls across a whole device-path lookup
        to keep that honest.

        Rate: one line per compared step, no sampling. The failure is single-step
        and intermittent -- two boots in eleven -- so any sampling rule can step
        over the only step that shows it, and a diagnostic that can miss the event
        it exists to catch is not a diagnostic. What is bounded instead is the
        line: three integers, one shape, and two vectors capped at ``_QSL_DUMP``
        numbers, so its length does not depend on the batch or the traffic. Mode 2
        is a correctness boot that keeps the blocking D2H by design, so a per-step
        line costs nothing that mode is not already paying.

        Anything that fails to read is named as an absence. An ``n/a`` that looks
        like a value is the failure mode this harness has been rebuilt around.
        """
        fields = [f"PLE DIFF: ids_len={ids_len}"]
        try:
            consumed = _host_view(getattr(self, "_host_consumed_qsl", None))
            gate = (
                _host_view(dev_inputs[1])
                if dev_inputs is not None and len(dev_inputs) > 1
                else None
            )
            inputs = self._host_check_inputs
            packed = (
                inputs[3]
                if inputs is not None and len(inputs) > 3
                else getattr(self, "padded_buffer", None)
            )
            fields.append(_render_vector("qsl", consumed))
            shape = _shape_str(packed)
            fields.append(
                f"packed.shape={shape if shape is not None else 'n/a(no workspace)'}"
            )
            extent = "n/a(no workspace)"
            if packed is not None and getattr(packed, "ndim", 0) == 2 and ids_len > 0:
                row0 = _host_view(packed[0, :ids_len])
                if row0 is not None:
                    extent = str(_fill_extent(row0, self.eos_token_id))
            fields.append(f"extent_row0={extent}")
            fields.append(_render_vector("gate_qsl", gate))
            if stage_rows != ids_len:
                # The silent case: compare_host_ids used to return here without
                # saying anything, so a compute that produced a different number of
                # rows than the host staged was invisible by construction.
                fields.append(f"stage_rows={stage_rows}")
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill a step
            return (
                "PLE DIFF: n/a(reading the consumed extent raised "
                f"{type(exc).__name__})"
            )
        return " ".join(fields)

    def _packed_dump(self, inputs: Any, bad_rows: list[int]) -> str:
        """What the hash consumed inside itself, not what the caller handed it.

        Every other leg of this dump reads an array the *caller* owns: the ids and
        query boundaries the runner filled, the window the model state filled. The
        q=3 boot (2026-10-02 11:53) closed all three -- ids, qsl and window
        identical on both sides -- and a faithful CPU replay of
        :meth:`_compute_ngram_ids` on those exact inputs reproduced the HOST ids
        to the digit, not the device's. So the shipped arithmetic is not producing
        the device's answer from the arrays the caller can see, and what is left is
        the one array nobody else writes: ``padded_buffer``, the compute's own
        packed workspace, filled from ``input_ids`` inside the compiled piece.

        This prints its first cells and, for each differing row, the exact cell
        each ngram order reads as its left context -- the same
        ``context = cat([ngram_context, packed])`` arithmetic the compute uses, with
        ``adjusted_columns = column + ngram_size - 1``. If that cell holds the
        token, the divergence is in the compiled arithmetic. If it holds something
        else, the compute consumed an ids tensor that is not the one the op was
        handed, and the search moves to the buffer, not the math.

        Compare-only, and it costs one small sync on a path that disables itself
        after the first mismatch. A failure here must never kill a step.

        Geometry (P1.14): the print covers ``pk.shape[0]`` rows and names
        ``packed.shape`` itself. The first version derived the row count from the
        compare-time qsl, so every dump ever taken printed exactly one row of a
        workspace that had three, and reported no sign of the difference.

        ``packed`` may be a tensor (the layer's own ``padded_buffer``, which is what
        the worker stages) or a plain array (what the on-path probe stages), so it
        is read the same way ``ngram_context`` is. The first version of this leg
        called ``.detach()`` unconditionally, the probe's numpy fixture raised
        ``AttributeError``, and the leg printed ``packed=n/a`` while looking like a
        leg that ran -- which is precisely the failure the probe exists to catch.
        """
        if not bad_rows:
            return "packed=n/a(no differing row)"
        packed = inputs[3] if inputs is not None and len(inputs) > 3 else None
        if packed is None:
            return "packed=n/a(stager handed no workspace)"
        try:
            qsl = np.asarray(inputs[1]).reshape(-1)
            ctx = inputs[2]
            ctx_np = (
                None
                if ctx is None
                else (
                    ctx.detach().cpu().numpy()
                    if hasattr(ctx, "detach")
                    else np.asarray(ctx)
                )
            )
            pk = (
                packed.detach().cpu().numpy()
                if hasattr(packed, "detach")
                else np.asarray(packed)
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill a step
            return f"packed=n/a(reading it raised {type(exc).__name__})"
        ctx_len = self.ngram_size - 1
        n_reqs = int(min(pk.shape[0], max(1, qsl.shape[0] - 1)))
        # P1.14: print the workspace's OWN geometry. Until now this line sized
        # itself with ``qsl.shape[0] - 1`` -- the request count of the *compare-time*
        # qsl, i.e. the host's -- so a three-request workspace under a one-request
        # host step printed exactly one row, and every row the host never staged was
        # invisible by construction. Those are precisely the rows a second pass in
        # the same step writes, which is the case this dump is for.
        out = [f"packed.shape={_shape_str(pk)}", "packed device=" + _bounded_rows(pk)]
        for r in bad_rows[:2]:
            req = int(np.clip(np.searchsorted(qsl, r, side="right") - 1, 0, n_reqs - 1))
            col = int(r - qsl[req])
            for ngram in range(2, self.ngram_size + 1):
                idx = col + ctx_len - (ngram - 1)
                if idx < 0:
                    out.append(f"row{r}ngram{ngram}=<before the request>")
                elif idx < ctx_len:
                    val = "?" if ctx_np is None else int(ctx_np[req, idx])
                    out.append(f"row{r}ngram{ngram}=ctx[{req},{idx}]={val}")
                else:
                    out.append(
                        f"row{r}ngram{ngram}=packed[{req},{idx - ctx_len}]"
                         f"={int(pk[req, idx - ctx_len])}"
                    )
        return "; ".join(out)

    def _inputs_agree(
        self,
        device_inputs: tuple[Any, ...] | None,
        host_inputs: tuple[Any, ...] | None = None,
        staging: bool = False,
    ) -> bool | None:
        """Did the device read the tokens the host mirrored?

        True makes the comparison that follows attributable. False means the
        arrays the compute hashed are not the arrays the host built, so a mismatch
        would be a description of the harness and not of the mirror. None means the
        question could not be asked at all, which is treated exactly like False:
        a comparison nobody can attribute must not be reported as a verdict in
        either direction, and it must certainly not disable the host path.

        The device half arrives as tensor *references* handed over by the stager in
        the worker -- the runner's own buffers, which the compute read out of. The
        D2H below is blocking, and legal: the op impl is the split point, so this is
        eager code running after the compiled piece on the same stream. It costs one
        small sync per compared step, in differential mode only, which is a mode
        that exists in order to pay for measurements.

        The third leg is the ngram context, and it is compared (P1.11). The claim
        this function used to make -- that the worker has no device-side
        counterpart of the window to hand back -- was wrong: ``Qwen4ExpModelState``
        *owns* that buffer (``model_state.py:62``), fills it in
        ``_prepare_ngram_context`` and hands a view of it to the forward at
        ``model_state.py:522``. Staging runs after the fill and nothing rewrites it
        before the op, so the reference handed over at staging time names the exact
        cells the compute read, and copying it back here reads those cells.

        The device buffer is padded to ``max_num_reqs`` just like the ids buffer,
        so only the host's rows are compared. When no device context arrives the
        ids and qsl verdict still stands, but the leg is counted in
        ``_host_ctx_unchecked`` and said out loud: a leg nobody checks reads as a
        leg that passed.

        ``host_inputs`` lets the *serving* path ask the same question. It defaulted
        to nothing while this was a comparison-only gate, and the serving path had
        no copy of it -- which is the hole :meth:`stage_inputs_agree` closes. Pass
        the triple the stager built (ids, query boundaries, window) and the host
        half is read from there instead of from ``_host_check_inputs``, which only
        differential mode populates. ``staging`` routes the refusal to
        :meth:`_note_stage_refusal` so a step the serving path declined is not
        logged as "a comparison was skipped": one says the device served it, the
        other says the harness declined to judge. Same numbers, different claim,
        and this box has been misled by diagnostics that blurred the two.
        """
        host = self._host_check_inputs if host_inputs is None else host_inputs
        note = self._note_stage_refusal if staging else self._note_input_refusal
        if host is None or device_inputs is None:
            note(
                "inputs missing on the "
                + ("host" if host is None else "device")
                + " side"
            )
            return None
        d_ids, d_qsl = device_inputs[0], device_inputs[1]
        d_ctx = device_inputs[2] if len(device_inputs) > 2 else None
        if d_ids is None or d_qsl is None:
            note("device inputs incomplete (no ids or no qsl)")
            return None
        h_ids = np.asarray(host[0]).reshape(-1)
        h_qsl = np.asarray(host[1]).reshape(-1)
        h_ctx = None if host[2] is None else np.asarray(host[2])
        dev_ctx: np.ndarray | None = None
        try:
            dev_ids = d_ids.detach().cpu().numpy().reshape(-1)
            dev_qsl = d_qsl.detach().cpu().numpy().reshape(-1)
            if d_ctx is not None:
                dev_ctx = d_ctx.detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill a step
            note(f"reading the device inputs raised {type(exc).__name__}")
            return None
        if dev_ids.shape[0] < h_ids.shape[0]:
            # The runner's array is the padded one, so it is at least as long as the
            # host's rows whenever both describe this step. Shorter means the two
            # sides are not describing the same work at all, and comparing a prefix
            # of that would be a green light invented by the slicing.
            note(
                f"device ids shorter than the staged rows "
                f"({dev_ids.shape[0]} < {h_ids.shape[0]})"
            )
            return False
        n = int(h_ids.shape[0])
        m = int(min(h_qsl.shape[0], dev_qsl.shape[0]))
        ids_diff = dev_ids[:n] != h_ids[:n]
        qsl_diff = dev_qsl[:m] != h_qsl[:m]
        if not ids_diff.any() and not qsl_diff.any():
            # P1.11: ids and query boundaries matching is not yet "the device read
            # what the host mirrored". A row's higher-order ngrams extend into the
            # per-request window, so a window that differs by one cell makes the
            # host's ids a function of a different prompt -- and that is the one
            # input the host mirror builds independently of the runner's arrays.
            verdict, why = self._ctx_leg(h_ctx, dev_ctx)
            if verdict is False:
                note(why)
                return False
            if verdict is None:
                self._host_ctx_unchecked += 1
                if self._host_ctx_unchecked == 1:
                    logger.info(
                        "PLE HOSTIDS: the ngram context is NOT compared on '%s' (%s). "
                        "ids and query_start_loc are still compared, so this gate is "
                        "not blind -- but a step counted here cannot attribute a "
                        "mismatch to the window. (%d steps so far)",
                        self.layer_name,
                        why,
                        self._host_ctx_unchecked,
                    )
            self._host_input_checks += 1
            return True
        if ids_diff.any():
            first = int(np.argmax(ids_diff))
            detail = (
                f"input_ids differ at {first} of {n}: device={int(dev_ids[first])} "
                f"host={int(h_ids[first])} "
                f"({int(ids_diff.sum())} of {n} positions) "
                f"device_ids={_bounded_ids(dev_ids)} "
                f"host_ids={_bounded_ids(h_ids)}"
            )
        else:
            first = int(np.argmax(qsl_diff))
            detail = (
                f"query_start_loc differs at {first} of {m}: "
                f"device={int(dev_qsl[first])} host={int(h_qsl[first])} "
                f"(device={dev_qsl[:m].tolist()} host={h_qsl[:m].tolist()})"
            )
        note(detail)
        return False

    def _ctx_leg(
        self,
        host_ctx: np.ndarray | None,
        dev_ctx: np.ndarray | None,
    ) -> tuple[bool | None, str]:
        """Compare the window the device built to the window the host mirrored.

        Returns ``(True, why)`` when they agree, ``(False, why)`` when they do not
        (which makes the comparison unattributable, so the gate refuses), and
        ``(None, why)`` when one side was never handed over.

        Both sides are ``[num_reqs, ngram_size - 1]`` and they are two independent
        implementations of the same window: the host's comes out of
        ``HostTokenHistory`` in ``build_decode_step_inputs``, the device's is
        ``_prepare_ngram_context`` gathering ``req_states.all_token_ids.gpu`` at
        ``req_states.num_computed_tokens.gpu``. Boot #8's inversion put the single
        wrong term exactly in the cell this window feeds, which is why the leg is
        worth a copy.

        The device side arrives as a numpy array because the caller already paid
        the blocking copy inside the op, next to the two it was paying anyway.
        """
        if host_ctx is None or host_ctx.ndim != 2:
            return None, f"the host step has no 2D ngram context (got {host_ctx})"
        if dev_ctx is None:
            return None, "the worker handed over no device ngram_context"
        if dev_ctx.ndim != 2:
            return False, (
                f"device ngram_context is {dev_ctx.ndim}D, expected 2D "
                "[num_reqs, context_len]"
            )
        if dev_ctx.shape[1] != host_ctx.shape[1]:
            return False, (
                f"ngram_context is {dev_ctx.shape[1]} cells wide on the device and "
                f"{host_ctx.shape[1]} on the host"
            )
        if host_ctx.shape[0] > dev_ctx.shape[0]:
            # Same rule as the ids buffer: a device window with fewer requests than
            # the host staged does not describe this step, and slicing it to the
            # host's length would compare a prefix of something else.
            return False, (
                f"device ngram_context describes {dev_ctx.shape[0]} requests, the "
                f"host step has {host_ctx.shape[0]}"
            )
        head = dev_ctx[: host_ctx.shape[0]]
        if (host_ctx != head).any():
            where = int(np.flatnonzero((host_ctx != head).ravel())[0])
            req, cell = divmod(where, host_ctx.shape[1])
            return False, (
                f"ngram_context differs at request {int(req)} cell {int(cell)}: "
                f"host={int(host_ctx[req][cell])} "
                f"device={int(head[req][cell])} "
                f"(host={host_ctx.tolist()} device={head.tolist()})"
            )
        return True, "ngram_context agrees"

    def _note_input_refusal(self, why: str) -> None:
        """Count a refused comparison, and say so once and then every 32nd."""
        self._host_input_skips += 1
        if self._host_input_skips == 1 or self._host_input_skips % 32 == 0:
            logger.info(
                "PLE HOSTIDS: SKIPPED a comparison for '%s' -- %s. This says "
                "nothing about the host mirror: the device read different inputs "
                "than the ones staged, so there is no comparison to make. "
                "(%d skipped; %d steps went uncompared because the worker handed "
                "over no device window.)",
                self.layer_name,
                why,
                self._host_input_skips,
                self._host_ctx_unchecked,
            )

    def stage_inputs_agree(
        self,
        host_inputs: tuple[Any, ...],
        device_inputs: tuple[Any, ...] | None,
    ) -> bool:
        """May this step be staged? Mode 1's copy of the input-agreement gate.

        ``_inputs_agree`` existed as a gate on a *comparison*, and its only caller
        was :meth:`compare_host_ids`, which differential mode alone reaches. The
        serving path -- ``stage_host_step(gather=True)`` -- therefore staged whatever
        the mirror handed it, and boot #14 showed what the mirror sometimes hands
        out: ``host=4161103993993705554`` where a token id belongs (8 refused
        comparisons across 4 workers in 45 minutes, ~2 per worker). Differential mode
        refused that step and the device served it. Mode 1 would have gathered
        embedding rows out of a nonexistent token id and handed the model a
        plausible answer built from garbage -- the same silent-wrong-answer class as
        a stale replay, arriving by a different route.

        The rule is the differential's rule, unchanged, including the one that
        matters most: a question that cannot be asked (``None``) is answered like a
        ``False``. Refusing is cheap and being wrong is not.

        A refused step stages nothing, so ``take_host_stage`` finds no flag, the op
        falls through to the blocking D2H, and the device serves the step. That is
        the safe direction under piecewise graphs. Under a captured graph it is not:
        the op body does not run at replay, so the step ships the previous step's
        rows. The caller counts that case and warns about it
        (``Qwen4ExpModelState._note_host_refusal``), which is why a gate that used to
        be a diagnostic is now load-bearing on the FULL-graph question.

        Cost: three small blocking D2H copies per staged step, in the worker, before
        the forward. Synchronous scheduling is already mandatory for this knob
        (``ple_hostids_async_conflict``), so it is not a new class of stall; the E1
        boot measures what it costs against the +15 % the capture branch buys.
        """
        return (
            self._inputs_agree(device_inputs, host_inputs=host_inputs, staging=True)
            is True
        )

    def _note_stage_refusal(self, why: str) -> None:
        """Count a step the serving path refused to stage: once, then every 32nd."""
        self._host_stage_refusals += 1
        if self._host_stage_refusals == 1 or self._host_stage_refusals % 32 == 0:
            logger.info(
                "PLE HOSTIDS: REFUSED to stage '%s' -- %s. The device path serves "
                "this step; the pinned rows were not touched. (%d refused; %d steps "
                "went with the ngram window unchecked.)",
                self.layer_name,
                why,
                self._host_stage_refusals,
                self._host_ctx_unchecked,
            )

    def compare_host_ids(
        self,
        device_ids: torch.Tensor,
        device_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Differential mode: the device's ids are now in ``device_ids``.

        This is the gate that turns the position arithmetic in
        ``build_decode_step_inputs`` from a reading of ``_post_update_kernel``
        into a measurement. One mismatch is enough to distrust the host path for
        the life of the process: a *shifting* disagreement (the optimistic
        ``num_computed_tokens_np`` mirror leading the device by a token) would
        otherwise pass some steps and fail others, which is the worst possible
        failure mode for a lookup table whose output looks like text either way.
        """
        rows = self._host_check_rows
        self._host_check_rows = -1
        captured = self._host_check_dev_inputs
        dev_inputs = device_inputs if device_inputs is not None else captured
        self._host_check_dev_inputs = None
        if rows < 0:
            return
        # P1.14: the consumed-extent line, printed BEFORE both early returns and
        # before the input gate. Order is the point. The two returns below are
        # silent, and the first one is exactly the case the extent exists to
        # describe -- a compute that handed over a different number of rows than
        # the host staged -- so an instrument that printed after either would
        # report on the steps that already make sense and stay quiet on the ones
        # that do not. The gate is skipped for the same reason: when it refuses,
        # the comparison never happens, and that is the step worth reading.
        if ple_hostids.hostids_mode() == 2 and not self._hostids_disabled:
            # The method guards its own reads; this guard is for the one failure it
            # cannot see from the inside -- the report not existing or blowing up on
            # whatever ``self`` turns out to be. A diagnostic that kills a step is
            # worse than one that prints nothing, which is the rule P1.9's snapshot
            # copies paid for with a boot.
            try:
                line = self._consumed_report(device_ids.shape[0], rows, dev_inputs)
            except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill a step
                line = f"PLE DIFF: n/a(the report raised {type(exc).__name__})"
            logger.info("%s", line)
        if rows != device_ids.shape[0]:
            return
        # Input-agreement gate (P1.10b): ask the attribution question directly
        # instead of inferring it from how often the op has been entered.
        #
        # The P1.9 gate counted entries into this op and required the stage to sit
        # immediately before one. That does not measure what it claimed under
        # piecewise graphs: the main pass replays without entering the op at all,
        # so only the drafter passes bump the counter, and "gap == 1" was a
        # statement about eager entries rather than about steps.
        #
        # What the comparison actually needs is narrower and checkable: the arrays
        # the compute hashed must be the arrays the host mirrored. If they differ,
        # whatever the ids say, the two halves describe different work -- which is
        # exactly the shape boot #6's dump had, down to row 1 agreeing with the
        # staged token stream and row 2 inverting (by CRT, exactly) to a different
        # token at the same position. With num_speculative_tokens=k the PLE op runs
        # k+1 times per step and each pass rewrites only its own rows of one static
        # buffer, so under a drafter the device's copy is a patchwork and this gate
        # refuses. A refusal is not a verdict about the mirror; it is the harness
        # declining to guess.
        if self._inputs_agree(dev_inputs) is not True:
            return
        assert self._host_check_ids is not None
        host = self._host_check_ids[:rows].numpy()
        dev = device_ids.numpy()
        if bool(np.array_equal(host, dev)):
            self._host_id_matches += 1
            if self._host_id_matches == 1:
                logger.info(
                    "PLE HOSTIDS: host and device ids agree for '%s' (%d rows) -- "
                    "first matching step.",
                    self.layer_name,
                    rows,
                )
            return
        self._host_id_mismatches += 1
        diff = np.argwhere(host != dev)[:4]
        detail = ", ".join(
            f"[{r},{c}] host={int(host[r, c])} device={int(dev[r, c])}" for r, c in diff
        )
        self.disable_hostids(
            f"first mismatch at {rows} rows: {detail}; "
            + self._id_dump(rows, host, dev, dev_inputs)
        )

    def disable_hostids(self, why: str) -> None:
        if self._hostids_disabled:
            return
        self._hostids_disabled = True
        logger.warning(
            "PLE HOSTIDS: DISABLED for '%s' after %d staged step(s) -- %s. "
            "The device path handles every step from here on.",
            self.layer_name,
            self._host_stage_steps,
            why,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )

        ngram_ids = self._compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            multipliers=self.layer_multipliers,
            vocab_sizes=self.ngram_heads_vocab_sizes,
            head_offsets=self.ngram_heads_offsets,
            packed=self.padded_buffer[:num_reqs],
            positions=self.positions_buffer[:num_tokens],
            eos_token_id=self.eos_token_id,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
        )
        # Nothing from the differential harness lives here any more (P1.10b). This
        # body is the compiled region: under torch.compile it runs at TRACE time
        # only, so an attribute stashed here holds a trace-time value while the op
        # impl that reads it runs eager on every later step. The provenance the
        # comparison needs is captured by the stager in the worker, which is plain
        # Python and sees the batch this step will actually consume.
        output = ngram_ids.new_empty(
            (ngram_ids.shape[0], self.embedding_dim),
            dtype=self.runtime_dtype,
            #            dtype=self.ngram_embedding.params_dtype,
        )
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                if name in (
                    "layer_multipliers",
                    "ngram_heads_vocab_sizes",
                    "ngram_heads_offsets",
                ):
                    # The host id producer caches the CPU view of these three.
                    # They are load-time constants, but caching must not survive
                    # a reload of the very weights it reads.
                    self._host_hash_cache = None
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                if loaded_weight.shape[1] != embedding.embedding_dim:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected embedding_dim {embedding.embedding_dim}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                # Store the mmap-backed tensor by reference. No .copy_() or
                # device move: this keeps the ~100GB table page-cache-backed
                # and shared across every worker process on the node instead
                # of privately materialized once per process.
                embedding.set_shard(shard_index, loaded_weight.to("cpu"))
                loaded.add(f"ngram_embedding.shard_{shard_index}")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        # Every shard and both layout buffers are in by now, so this is the first
        # point where the checkpoint's id space and its row split can be compared.
        # A checkpoint whose split does not add up to its own id space is refused
        # here rather than mis-read at serving time.
        layout = self.ngram_heads_offsets + self.ngram_heads_vocab_sizes
        vocab_size = int(layout.max()) if layout.numel() else 0
        self.ngram_embedding.finalize_shard_layout(
            vocab_size if vocab_size > 0 else None
        )

        # The device mirror of the generation counter is allocated here rather
        # than lazily because the first execution of the staged branch may be
        # inside a CUDA graph capture: an allocation inside a capture either
        # fails or lands in that graph's private pool, where the other graphs and
        # the host's read-back would not see it.
        self._alloc_gen_scratch()
        return loaded


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs,
            f"{prefix}.ple_embedding",
            prefix,
            runtime_dtype=model_config.dtype,
        )
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _apply_norm(
        self, norm: Qwen4ExpPLEGroupedNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        # Profiling / CUDA graph capture only; conv state is not updated.
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = self.conv1d(inputs_t)[..., : inputs_t.size(-1)]
        return F.silu(output).squeeze(0).transpose(0, 1)

    def _short_conv_dilated_decode_batched(
        self,
        x_d: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_d: torch.Tensor,
        has_initial_states_d: torch.Tensor | None,
    ) -> torch.Tensor:
        state_indices = state_indices_tensor_d.to(
            device=conv_state.device, dtype=torch.int64
        )
        # FULL cudagraph padded decode rows use NULL_BLOCK_ID. Remap them to
        # slot 0 for a safe gather, then zero output and skip write-back.
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        if has_initial_states_d is None:
            has_initial_state = valid_state
        else:
            if has_initial_states_d.numel() < state_indices_tensor_d.numel():
                raise ValueError(
                    "has_initial_states_d size mismatch: "
                    f"got {has_initial_states_d.numel()}, "
                    f"need >= {state_indices_tensor_d.numel()}."
                )
            has_initial_state = has_initial_states_d[
                : state_indices_tensor_d.numel()
            ].to(device=conv_state.device, dtype=torch.bool)
            has_initial_state = has_initial_state & valid_state

        cached_state = conv_state.index_select(0, state_indices)
        state = cached_state[..., : self.conv_state_len].to(x_d.dtype)
        if self.conv_state_len > 0:
            initial_state = torch.where(
                has_initial_state.view(-1, 1, 1),
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
        else:
            history = x_d.unsqueeze(-1)

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        ).squeeze(-1)
        output = F.silu(conv_output)
        output = output * valid_state.view(-1, 1).to(output.dtype)

        if self.conv_state_len > 0:
            next_state = history[..., -self.conv_state_len :]
            # Padded rows are remapped to the reserved null slot. Preserve its
            # existing value while writing the new states for valid rows.
            existing_base_state = cached_state[..., : self.conv_state_len]
            safe_next_state = torch.where(
                valid_state.view(-1, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            cached_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_prefill_batched(
        self,
        x_p: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_p: torch.Tensor,
        num_prefills: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> torch.Tensor:
        # ``non_spec_query_start_loc`` covers the non-spec (decode + prefill)
        # requests and equals ``query_start_loc`` when spec-decode is inactive.
        non_spec_query_start_loc = metadata.non_spec_query_start_loc
        if non_spec_query_start_loc is None:
            raise ValueError("query_start_loc is required for prefill short-conv")
        query_start_loc_p = (
            non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
        )
        # The metadata builder guarantees that the prefill query offsets start
        # at 0 and end at num_prefill_tokens. Avoid reading those values here,
        # since doing so would force a device-to-host synchronization.
        has_initial_states_p = metadata.has_initial_states_p
        if has_initial_states_p is None:
            raise ValueError("has_initial_states_p is required for prefill short-conv")

        output = torch.empty_like(x_p)
        q_starts = query_start_loc_p.to(torch.int64)
        if state_indices_tensor_p.numel() < num_prefills:
            raise ValueError(
                "state_indices_tensor_p size mismatch: "
                f"got {state_indices_tensor_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if has_initial_states_p.numel() < num_prefills:
            raise ValueError(
                "has_initial_states_p size mismatch: "
                f"got {has_initial_states_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if num_prefills == 0 or x_p.numel() == 0:
            return output
        lengths = q_starts[1:] - q_starts[:-1]
        # Use the CPU-computed packing width from the metadata builder instead
        # of synchronizing on lengths.max().
        max_len = metadata.max_prefill_query_len
        if max_len <= 0:
            return output

        hidden_size = x_p.shape[1]
        positions = torch.arange(
            num_prefill_tokens, device=x_p.device, dtype=torch.int64
        )
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        col_indices = positions - q_starts[req_indices]

        packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
        packed_tokens[req_indices, col_indices] = x_p
        packed_tokens = packed_tokens.transpose(1, 2).contiguous()

        state_indices = state_indices_tensor_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        has_initial = has_initial_states_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.bool
        )
        if self.conv_state_len > 0:
            if conv_state.shape[0] == 0:
                state = conv_state.new_zeros(
                    (num_prefills, hidden_size, self.conv_state_len),
                    dtype=x_p.dtype,
                )
            else:
                state = conv_state.index_select(0, state_indices)[
                    ..., : self.conv_state_len
                ].to(x_p.dtype)
            use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
            initial_state = torch.where(
                use_initial_mask,
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, packed_tokens), dim=-1)
        else:
            history = packed_tokens

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
        valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
        valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
            num_prefills, 1
        )
        conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
        output.copy_(conv_output[req_indices, col_indices])

        if self.conv_state_len > 0 and conv_state.shape[0] > 0:
            state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
                num_prefills, 1, 1
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=history.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            next_state = history.gather(
                dim=2,
                index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
            )
            # Write back without a host synchronization. Valid, non-empty rows
            # receive their new state; padding and zero-length rows keep the
            # current cache value.
            existing_state = conv_state.index_select(0, state_indices)
            existing_base_state = existing_state[..., : self.conv_state_len]
            update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
            safe_next_state = torch.where(
                update_mask.view(num_prefills, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            existing_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, existing_state)
        return output

    def _short_conv_dilated_spec_batched(
        self,
        x_spec: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_query_len: int,
    ) -> torch.Tensor:
        """Dilated short-conv for speculative-decode (MTP) requests.

        Each spec request feeds multiple (draft + 1) query tokens. The conv
        outputs are computed causally after rolling back the previous draft
        state by ``num_accepted_tokens - 1``. The current candidate inputs stay
        in the extended cache for the next forward, matching
        ``causal_conv1d_update``.

        ``spec_query_len`` (== num_speculative_tokens + 1) is the maximum query
        length and is a Python int, so no host synchronization is needed; this
        keeps the path safe for full CUDA-graph capture/replay where the buffers
        are padded at the request level.
        """
        num_reqs = spec_state_indices_tensor.numel()
        hidden_size = x_spec.size(-1)
        # Use a fixed packing width instead of synchronizing on lengths.max().
        max_len = spec_query_len
        # Full CUDA graphs can pad these buffers. Only the first num_reqs
        # accepted-token counts belong to actual speculative requests.
        num_accepted_tokens = num_accepted_tokens[:num_reqs]
        q_starts = spec_query_start_loc[: num_reqs + 1].to(torch.int64)
        # Keep the number of real speculative tokens on the device.
        total_real_tokens = q_starts[num_reqs]

        state_indices = spec_state_indices_tensor.to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        positions = torch.arange(
            x_spec.size(0), device=x_spec.device, dtype=torch.int64
        )
        # Route graph-padded token rows to the discarded dummy request so that
        # they cannot overwrite real packed data.
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        valid_tokens = (positions < total_real_tokens) & (req_indices < num_reqs)
        clamped_req_indices = req_indices.clamp_max(max(num_reqs - 1, 0))
        col_indices = (positions - q_starts[clamped_req_indices]).clamp_(0, max_len - 1)
        pack_req_indices = torch.where(
            valid_tokens,
            clamped_req_indices,
            torch.full_like(req_indices, num_reqs),
        )
        pack_col_indices = torch.where(
            valid_tokens, col_indices, torch.zeros_like(col_indices)
        )

        # The last request row is the dummy sink for graph padding.
        packed = x_spec.new_zeros((num_reqs + 1, max_len, hidden_size))
        packed[pack_req_indices, pack_col_indices] = x_spec
        packed = packed.transpose(1, 2).contiguous()

        if self.conv_state_len > 0:
            cached_state = conv_state.index_select(0, state_indices)
            rollback_offsets = num_accepted_tokens.to(
                device=conv_state.device, dtype=torch.int64
            ).sub(1)
            rollback_offsets = torch.where(
                valid_state,
                rollback_offsets.clamp_(0, max_len - 1),
                torch.zeros_like(rollback_offsets),
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=conv_state.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            rollback_indices = rollback_offsets.view(-1, 1, 1) + state_offsets
            state = cached_state.gather(
                2, rollback_indices.expand(-1, hidden_size, -1)
            ).to(x_spec.dtype)
            state = torch.where(
                valid_state.view(num_reqs, 1, 1),
                state,
                torch.zeros_like(state),
            )
            # Append a zeroed dummy-row state to match the [num_reqs + 1] pack.
            dummy_state = state.new_zeros((1, hidden_size, self.conv_state_len))
            state_full = torch.cat((state, dummy_state), dim=0)
            history = torch.cat((state_full, packed), dim=-1)
        else:
            history = packed

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        output = conv_output[pack_req_indices, pack_col_indices]
        output = output * valid_tokens.view(-1, 1).to(output.dtype)

        # Keep all current candidate inputs in the extended state. On the next
        # target forward, ``num_accepted_tokens - 1`` selects the rollback
        # window before processing the newly scheduled tokens.
        if self.conv_state_len > 0:
            state_capacity = self.conv_state_len + max_len - 1
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache cannot retain speculative tokens: "
                    f"got {conv_state.size(-1)}, need {state_capacity}."
                )
            candidate_state = history[:num_reqs, :, 1 : state_capacity + 1]
            query_lengths = q_starts[1:] - q_starts[:-1]
            state_positions = torch.arange(
                state_capacity, device=history.device, dtype=torch.int64
            ).view(1, 1, state_capacity)
            update_lengths = (self.conv_state_len + query_lengths - 1).view(
                num_reqs, 1, 1
            )
            update_mask = valid_state.view(num_reqs, 1, 1) & (
                state_positions < update_lengths
            )
            existing_state = cached_state[..., :state_capacity]
            next_state = torch.where(
                update_mask,
                candidate_state.to(conv_state.dtype),
                existing_state,
            )
            cached_state[..., :state_capacity] = next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        x = inputs[: metadata.num_actual_tokens]

        # Split spec / non-spec tokens.
        if has_spec:
            if has_prefill or has_decode:
                assert metadata.spec_token_indx is not None
                assert metadata.non_spec_token_indx is not None
                x_spec = x.index_select(0, metadata.spec_token_indx.long())
                x_non_spec = x.index_select(0, metadata.non_spec_token_indx.long())
            else:
                x_spec = x
                x_non_spec = None
        else:
            x_spec = None
            x_non_spec = x

        spec_output = None
        # 1. Run the multi-query speculative-decode part.
        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.spec_query_start_loc is not None
            assert metadata.num_accepted_tokens is not None
            spec_output = self._short_conv_dilated_spec_batched(
                x_spec=x_spec,
                conv_state=conv_state,
                conv_weights=conv_weights,
                spec_state_indices_tensor=metadata.spec_state_indices_tensor[
                    : metadata.num_spec_decodes
                ],
                spec_query_start_loc=metadata.spec_query_start_loc,
                num_accepted_tokens=metadata.num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
            )

        # 2. Run regular decode and prefill requests.
        conv_out_non_spec = None
        state_indices_tensor = metadata.state_indices_tensor
        if x_non_spec is not None:
            assert state_indices_tensor is not None
            if has_prefill:
                state_indices_tensor_d, state_indices_tensor_p = torch.split(
                    state_indices_tensor,
                    [num_decodes, num_prefills],
                    dim=0,
                )
                x_d, x_p = torch.split(
                    x_non_spec,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                non_spec_parts: list[torch.Tensor] = []
                if has_decode:
                    non_spec_parts.append(
                        self._short_conv_dilated_decode_batched(
                            x_d=x_d,
                            conv_state=conv_state,
                            conv_weights=conv_weights,
                            state_indices_tensor_d=state_indices_tensor_d,
                            has_initial_states_d=metadata.has_initial_states_d,
                        )
                    )
                non_spec_parts.append(
                    self._short_conv_dilated_prefill_batched(
                        x_p=x_p,
                        metadata=metadata,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        state_indices_tensor_p=state_indices_tensor_p,
                        num_prefills=num_prefills,
                        num_decode_tokens=num_decode_tokens,
                        num_prefill_tokens=num_prefill_tokens,
                    )
                )
                conv_out_non_spec = torch.vstack(non_spec_parts)
            else:
                conv_out_non_spec = self._short_conv_dilated_decode_batched(
                    x_d=x_non_spec,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices_tensor[: x_non_spec.size(0)],
                    has_initial_states_d=metadata.has_initial_states_d,
                )

        # 3. Merge both parts back into the original token order.
        if has_spec and conv_out_non_spec is not None:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            assert spec_output is not None
            output = x.new_empty((metadata.num_actual_tokens, x.size(-1)))
            output.index_copy_(0, metadata.spec_token_indx, spec_output)
            output.index_copy_(0, metadata.non_spec_token_indx, conv_out_non_spec)
            return output
        elif has_spec:
            assert spec_output is not None
            return spec_output
        if conv_out_non_spec is None:
            return x
        return conv_out_non_spec

    def _short_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return self._short_conv_fallback(inputs)

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            # MRV2 omits Mamba-family metadata during profile warmup.
            return self._short_conv_fallback(inputs)
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        return self._short_conv_dilated_dispatch(
            inputs,
            layer_attn_metadata,
            conv_state,
            conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = torch.zeros_like(normalized)
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            normalized,
            conv_output,
            self.prefix,
        )
        return gated_value.flatten(-2) + conv_output


def qwen4_exp_amd_ple_ngram_embedding(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    ple_embedding = layer.ple_embedding
    n = ngram_ids.shape[0]
    if _CAPTURE_STAGE and not torch.cuda.is_current_stream_capturing():
        # Capture is over -- this is a live step -- and at least one shape was
        # recorded. Report what each capture shape did before real work buries it.
        # Gated on the dict being non-empty so a boot that never captured anything
        # pays a dict check and no CUDA call.
        _flush_capture_stage_summary()
    # E1 Phase 0 ceiling probe: while a capture is in progress, record only the
    # H2D and return. See _capture_probe_enabled() for why a captured gather is
    # worse than a skipped one, and why the consequence (frozen PLE rows) makes
    # this a measurement instrument and never a shipping path.
    if _capture_probe_enabled() and torch.cuda.is_current_stream_capturing():
        output.copy_(ple_embedding._pinned_output[:n], non_blocking=True)
        return
    # E1 Phase 1 (VLLM_GFX906_PLE_HOSTIDS): the host already computed this step's
    # ids, gathered them out of the mmap'd shards, and left the result in
    # _pinned_output -- so the blocking D2H below, the wait this whole line of work
    # exists to remove, and the host gather are both skipped and only the H2D runs.
    # Row-count agreement is a necessary condition and the generation tripwire
    # (P1.5) is what makes it sufficient. Read by getattr, not by method call, so
    # a stub embedding that never stages falls through to the device path.
    staged = getattr(ple_embedding, "_host_stage_rows", -1)
    if staged != -1:
        ple_embedding._host_stage_rows = -1
        if staged == n:
            output.copy_(ple_embedding._pinned_output[:n], non_blocking=True)
            # P1.5 generation tripwire: one 8-byte copy *inside* the graph, so what
            # comes back later is the generation the replay delivered and not the
            # one capture left behind. Without this node the tripwire could only
            # ever report what capture did, which is precisely the 2026-09-20
            # failure it exists to catch.
            scratch = ple_embedding._gen_scratch(output.device)
            scratch.copy_(ple_embedding._pinned_gen, non_blocking=True)
            if torch.cuda.is_current_stream_capturing():
                # Host-side bookkeeping during capture: records that the staged
                # branch -- and therefore this copy -- is in the graph, which is
                # what makes a tripwire fire fatal rather than a disable. Not a
                # graph node, and deliberately so.
                ple_embedding._staged_branch_captured = True
                _attest_capture_stage(ple_embedding, n, "staged")
            return
        if torch.cuda.is_current_stream_capturing():
            # A row count that does not match the shape being captured cannot be
            # honored, and the fall-through below is fatal here. Worth a label of
            # its own in the summary: it means the producer staged something for a
            # different extent than the graph was built for, which is a padding
            # bug, not a "nothing staged" case.
            _attest_capture_stage(ple_embedding, n, "staged-mismatch")
    # E1 Phase 1: a capture is in progress and nothing staged this step.
    #
    # The FULL ("mixed") capture branch picks its request count with
    # `num_reqs = min(num_tokens, max_num_reqs)` (cudagraph_utils.py:280-281), so
    # with max_num_seqs=3 and capture sizes [4, 8, 12] the 4- and 8-token dummies
    # are 3-request batches whose rows are NOT a uniform 1 + num_speculative_tokens
    # per request. The producer refuses those -- they are not a decode step anyone
    # will ever replay -- and without this clause the op fell through to the
    # blocking D2H and the capture died, which is why HOSTIDS=1 + GRAPHMODE=full
    # was not bootable before this clause existed.
    #
    # What capture actually needs is a fixed-extent, non-blocking H2D out of the
    # pinned staging buffer. It does not need these dummies' VALUES: no replay
    # reads what capture left there, because the host writes the real rows into
    # the same pinned address before every replay. So the rows are zeroed first.
    # That is not tidiness. If a step ever replays this graph without the host
    # having staged it, zeros mean PLE contributes nothing -- which the
    # degeneration smoke test catches structurally -- instead of serving the
    # previous step's embedding rows, which read as entirely plausible text. The
    # second is the 2026-09-20 failure mode; this line turns it from silent into
    # detectable.
    if torch.cuda.is_current_stream_capturing() and _host_stage_active(ple_embedding):
        staged_rows = ple_embedding._pinned_output[:n]
        staged_rows.zero_()  # host memset on pinned memory: not a graph node
        output.copy_(staged_rows, non_blocking=True)
        scratch = ple_embedding._gen_scratch(output.device)
        scratch.copy_(ple_embedding._pinned_gen, non_blocking=True)
        ple_embedding._staged_branch_captured = True
        _attest_capture_stage(ple_embedding, n, "dummy")
        return
    # VLLM_GFX906_PLE_PROF=1 brackets the four host-visible steps below (see
    # vllm/models/qwen4_exp/amd/ple_prof.py for what each one means and why
    # graph capture is excluded). With the knob unset the statements executed
    # are exactly the ones that ran before the brackets existed.
    if torch.cuda.is_current_stream_capturing():
        _attest_capture_stage(ple_embedding, n, "fell-through")
    prof = ple_prof.prof_enabled() and not torch.cuda.is_current_stream_capturing()
    if prof:
        faults_before = ple_prof.page_faults()
        t_drain = time.perf_counter()
    pinned_ids = ple_embedding._pinned_ngram_ids[:n]
    # The mmap'd shard lookup below runs on the host, so the device->host copy
    # has to have landed before the host reads the buffer.  non_blocking=True
    # only makes the copy stream-ordered; it does not wait for it.  The host
    # would otherwise read ids that are stale, or -- on the first step after
    # init, since _pinned_ngram_ids is torch.empty(...).pin_memory() and is
    # never zeroed -- outright uninitialized.  Those values then index the
    # shard table outside its row range (see MmapShardedNGramEmbedding.forward),
    # which is what killed the engine at 00:26 / 00:31 with 'index out of range
    # in self'.  The H2D copy of the result at the end stays async: it is
    # stream-ordered against whatever consumes `output` and needs no host wait.
    pinned_ids.copy_(ngram_ids, non_blocking=False)
    # Differential mode (VLLM_GFX906_PLE_HOSTIDS=2): the device's own ids are now
    # in host memory, which is the only thing they can be compared against. In
    # that mode nothing above was skipped, so the cost when the knob is off or at
    # 1 is one attribute read.
    if getattr(ple_embedding, "_host_check_rows", -1) >= 0:
        ple_embedding.compare_host_ids(
            pinned_ids,
            getattr(ple_embedding, "_host_check_dev_inputs", None),
        )
    if prof:
        t_gather = time.perf_counter()
    result = ple_embedding.ngram_embedding(pinned_ids).flatten(-2)
    if prof:
        t_memcpy = time.perf_counter()
    pinned_out = ple_embedding._pinned_output[:n]
    pinned_out.copy_(result.to(dtype=output.dtype))
    if prof:
        t_h2d = time.perf_counter()
    output.copy_(pinned_out, non_blocking=True)
    if prof:
        faults_after = ple_prof.page_faults()
        t_end = time.perf_counter()
        # `rows` is table rows READ (one per ngram id, head_dim wide), not
        # output tokens; `bytes_staged` is what crosses the H2D link.
        shards = getattr(ple_embedding.ngram_embedding, "_last_shards_touched", 0)
        ple_prof.window().record(
            drain_s=t_gather - t_drain,
            gather_s=t_memcpy - t_gather,
            memcpy_s=t_h2d - t_memcpy,
            h2d_s=t_end - t_h2d,
            rows=int(ngram_ids.numel()),
            bytes_staged=int(output.numel() * output.element_size()),
            shards_touched=shards,
            minflt_delta=faults_after[0] - faults_before[0],
            majflt_delta=faults_after[1] - faults_before[1],
        )


# def qwen4_exp_amd_ple_ngram_embedding(
#     ngram_ids: torch.Tensor,
#     output: torch.Tensor,
#     layer_name: str,
# ) -> None:
#     """Run the large PLE embedding lookup outside Inductor's FX graph.

#     Keeping the embedding weight in ``static_forward_context`` prevents AOT
#     compile-time autotuning from materializing a synthetic copy of the weight.
#     """
#     layer = get_forward_context().no_compile_layers[layer_name]
#     if not isinstance(layer, Qwen4ExpPLELayer):
#         raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
#     result = layer.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
#     output.copy_(result)


def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = layer._short_conv(inputs)
    output[: result.shape[0]].copy_(result)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_embedding",
    op_func=qwen4_exp_amd_ple_ngram_embedding,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["output"],
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
