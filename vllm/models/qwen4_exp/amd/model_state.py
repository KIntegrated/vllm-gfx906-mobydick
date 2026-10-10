# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

import dataclasses
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd import ple_hostids
from vllm.models.qwen4_exp.amd.ple_hostids import (
    HostIdsRefused,
    HostTokenHistory,
    build_decode_step_inputs,
)
from vllm.models.qwen4_exp.amd.ple_prefetch import (
    PleRowPrefetcher,
    plan_lookahead_chunks,
    prefetch_enabled,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


def _ms_summary(samples) -> str:
    """``n=.. p50=.. p90=.. max=..`` over a small latency sample set.

    Module-level, not a method: the PLE tests drive the coverage and resync paths
    with stub states that never run ``__init__``, and a diagnostic formatter that
    lives on the class turns every one of those stubs into an AttributeError.
    """
    if not samples:
        return "n=0"
    s = sorted(samples)

    def q(p: float) -> float:
        return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]

    return f"n={len(s)} p50={q(0.5):.2f} p90={q(0.9):.2f} max={s[-1]:.2f}"


_PLE_HB_RAMP = (1, 16, 128, 512)
"""Coverage lines printed below the steady cadence, then the cadence takes over."""

_PLE_HB_EVERY = 1024
"""Steady-state interval: roughly once a minute at B=1 decode."""


def _ple_heartbeat_due(count: int, every: int = _PLE_HB_EVERY) -> bool:
    """Whether a coverage line is due at this counter value.

    A ramp below the steady cadence -- 1, 16, 128, 512 -- then the cadence: 1024,
    2048, 3072 and so on. Four extra lines per worker per boot, and after that the
    cadence a reader has already learned.

    Why a ramp at all: the steady 1024 interval is right for a boot that serves for
    hours and useless as an audit trail for any boot that does not, because a counter
    that never reaches its own cadence prints nothing at all -- and *nothing* is not
    the same claim as *zero*. Boot ``vllm.log.20261007-234401`` staged, served a
    four-case probe, and printed no coverage line after 23:51:01; boot
    ``vllm.log.20261008-210206`` declined by policy and printed none after 21:08:55.
    Both were true statements about a live engine, and neither could be read as one:
    the log could not distinguish "staged every step" from "never staged", which is
    the exact difference the PLE-at-decode A/B turns on (task #53 -- its verdict had
    to be downgraded from a measurement to a lean for this reason alone).

    The ramp makes the first line arrive with the first staged step, so "the arm
    staged" becomes something a short boot proves rather than something inferred
    from its configuration. Module-level, not a method, for the same reason as
    :func:`_ms_summary`: the PLE tests drive these paths with stub states that never
    run ``__init__``.
    """
    return count in _PLE_HB_RAMP or (count >= every and count % every == 0)


def _ple_resync_timing_enabled() -> bool:
    """Whether to time the mirror resync. On unless told otherwise.

    Read once at import, like ``PLE_HOST_STAGE_UNDER_FULL``. Timing costs two
    ``perf_counter_ns`` calls per resync and one per slot read, which is nothing
    next to a device sync; the knob exists so the instrument can be turned off
    without editing the file a serving boot runs from.
    """
    return os.environ.get("VLLM_GFX906_PLE_RESYNC_TIMING", "1").strip().lower() not in (
        "0",
        "off",
        "no",
    )


def _ple_resync_cap(name: str, default: int) -> int:
    """A resync budget from the environment, falling back to ``default``.

    Read once at import, like every other knob here. A missing value is the
    default; so is anything that is not a non-negative integer, and that includes
    a negative one: ``0`` means *uncapped* below, so a typo parsed as a negative
    number clamped to 0 would silently remove a cap somebody meant to keep. The
    typo gets logged, because a knob that gates device syncs failing quietly is
    how boot #27's disagreement budget came to be read as evidence about the
    device when it was describing the checker.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        logger.warning(
            "PLE HOSTIDS: ignoring %s=%r (not a non-negative integer); using %d.",
            name,
            raw,
            default,
        )
        return default
    return value


class PLEHostDoubtUnderFullGraph(RuntimeError):
    """The host stopped trusting its own PLE ids on a step no device path can serve.

    E1 Phase 1 turns this into a real choice rather than a style one. Under a
    captured FULL decode graph the PLE op body does not run at replay, so "disable
    host staging" does not fall back to the device -- it leaves the graph replaying
    the previous step's pinned rows, forever and silently. Boot #28 did exactly
    that for 8,141 steps and still answered a smoke test whose PLE was hard-zeroed
    perfectly, which is what makes plausible-and-wrong the worst output a lookup
    table can produce.

    So a doubt under FULL *can* raise, and this type exists so the raise survives
    the ``except Exception`` that protects the repair path: a safety decision a broad
    handler can swallow is not a safety decision.

    It no longer does by default. :attr:`PLE_DOUBT_UNDER_FULL` ships as ``"zero"``,
    which is neither of the two things this type was written to choose between: the arm
    stands down, the rows it was staging are zeroed once, and the captured graph replays
    zeros -- so the boot keeps its speed and serves NO PLE at decode instead of the
    previous step's PLE. The justification this docstring used to give for ``"degrade"``
    ("under a captured graph the refused steps are stale whichever setting is in force")
    died on 2026-10-07 with the dispatch veto, and the arithmetic died with it: see
    :attr:`PLE_DOUBT_UNDER_FULL` for what each setting costs and why the ranking is not
    close. Three boots in a row died of ``"stop-boot"`` while the real bug was a corrupt
    host mirror, so it stays one environment variable away -- which is where it belongs,
    and is the right call again once the producer stops mis-owning slots.
    """


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
        # Host-side id staging (VLLM_GFX906_PLE_HOSTIDS). The draft mirror is set
        # once by the model runner, never per step; the layer list and refusal
        # counters exist so "the host path quietly never engaged" is a log line
        # rather than a guess.
        self._host_drafts: object | None = None
        self._ple_host_layers: list | None = None
        # Layers watched by the generation tripwire. Kept separate from
        # ``_ple_host_layers`` because disabling the host path must not disarm the
        # instrument: once a graph has captured the staged branch, a disabled host
        # path is the frozen-embedding state, and the tripwire is the only thing
        # that can see it.
        self._ple_host_gen_layers: list = []
        self._ple_host_refusals: dict[str, int] = {}
        self._ple_host_refusal_logged: set[str] = set()
        self._ple_host_steps = 0
        # Calls to _stage_host_ple_ids, staged or not. The per-step repair bound keys
        # on this and not on ``_ple_host_steps``, which only counts steps that staged
        # and would therefore let a run of refused steps look like one step.
        self._ple_host_calls = 0
        # E1 Phase 1: refusals that happened while a graph had the staged branch
        # captured, and the sticky flag that says a graph does. Sticky because a
        # capture cannot be undone, and because checking it per refusal should not
        # have to walk the layer list every time.
        self._ple_host_captured = False
        self._ple_host_stale_steps = 0
        # Steps declined by PLE_HOST_STAGE_UNDER_FULL rather than refused. Kept apart
        # from the refusal counts on purpose: "we did not try" and "we tried and lost"
        # are different facts, and collapsing them would make the coverage line
        # unreadable in exactly the way stale_steps used to be.
        self._ple_host_policy_off = 0
        self._ple_policy_off_logged = False
        # Per-step latch, so a step that refuses twice is still one stale step.
        self._ple_stale_counted_step = False
        # Dispatch veto. ``_ple_veto_armed`` is set by the runner when it has decided
        # in advance that a refused FULL step will be dispatched eager instead of
        # replayed; ``_ple_veto_steps`` counts the FULL steps the veto ledger caught
        # (real refusals only -- a decline stays in the stale ledger, because the veto
        # does not act on it), which are NOT stale steps because the device path
        # computes this step's rows, and ``_ple_veto_applied`` counts the downgrades
        # that actually landed. Both are reported: an armed runner whose downgrades
        # never land would otherwise print a boot of correct steps it never served,
        # which is the one failure mode of this mechanism that is invisible from the
        # outside. applied <= ledgered; the ledger also carries FULL-step refusals the
        # runner never asked about (capture, warmup), which is the only gap to expect.
        self._ple_veto_armed = False
        self._ple_veto_steps = 0
        self._ple_veto_applied = 0
        # Choice #3 (2026-10-08): the latch that says this arm died on a boot whose
        # graph already holds the staged branch, so a step it cannot stage must be
        # reported as *attempted and lost* rather than *not trying*. The difference
        # is what the dispatch veto acts on: ``False`` downgrades the step to eager,
        # where the op body computes this step's rows on the device; ``None`` lets it
        # replay the previous step's rows. Set once by :meth:`disable_host_staging`
        # under ``PLE_DOUBT_UNDER_FULL == "eager"`` and never cleared -- a capture
        # cannot be undone, so the reason outlives every later step.
        self._ple_disabled_eager = False
        # The other way out of the same trap, and the shipped one: the arm stood down
        # and the pinned rows were zeroed at the same moment, so a replay serves zeros
        # instead of a previous step's rows. Mutually exclusive with the latch above --
        # :meth:`disable_host_staging` sets exactly one of them -- and like it, set once
        # and never cleared, because a capture cannot be undone.
        self._ple_disabled_zero = False
        # Heartbeat cadence for a disabled arm, in staging *calls*. See
        # :meth:`_note_disabled_heartbeat`.
        self._ple_disabled_hb_at = 0
        # This step's staging decision: True staged, False attempted and refused,
        # None not attempted (knob off, arm declined, or the arm cannot stage). Only
        # ``False`` vetoes -- see :meth:`ple_rows_refused` for why the difference is
        # the whole safety story.
        self._ple_step_rows: bool | None = None
        # The runner's early hook caches this step's PLE inputs against the identity
        # of the ``InputBatch`` object so ``prepare_inputs`` does not do the work --
        # and the repair -- twice. Strong reference on purpose: the identity check is
        # only sound while the cached batch is alive.
        self._ple_prepared: tuple[Any, dict[str, Any], bool | None] | None = None
        # Boot #26's lesson. The only writers of a mirror slot's length were
        # ``seed`` at add and ``invalidate`` at remove, so a commit lost at the
        # epilogue left the slot short for the rest of the request's life and the
        # exact-length check refused every step after it. These bound and describe
        # the repair; see :meth:`_ple_resync_mirror`.
        self._ple_resyncs = 0
        # Window markers for the demand alarm: the counters as of the last time it
        # checked, so the rate it reports covers that window and not the boot.
        self._ple_resyncs_alarm_at = 0
        self._ple_resyncs_alarm_calls = 0
        self._ple_resync_disagree = 0
        self._ple_resync_refused: set[str] = set()
        self._ple_resync_slots: dict[int, int] = {}
        # The ``_ple_host_calls`` value this worker last paid a sync on, for the
        # per-step bound in PLE_RESYNC_PER_CALL. -1 so the first repair is allowed.
        self._ple_resync_last_call = -1
        # S: the wall-clock cost of one mirror resync, which the whole
        # correctness-vs-staleness decision is priced against and which has never
        # been measured. The caps above were set by judgment. Nothing measurable
        # from outside the process reaches it: the engine holds 99 % of VRAM, a
        # second HIP context will not fit, and on Vega two resident contexts
        # time-slice the GPU, so a probe would stall a serving engine for its
        # whole life. The two terms only exist here:
        #   sync = wake + drain, what ``torch.cuda.synchronize`` waits for; the
        #          wake half is measured (~10 us, load-independent) and the shim
        #          is exonerated, so this is the term that can matter;
        #   read = the blocking pageable D2H of the rows themselves, per slot.
        # Samples are kept rather than averaged because n is tens per boot and
        # the shape (a few step-sized outliers vs a flat ~1 ms) IS the answer.
        # Break-even against a 42.5 ms step: S <= 23 ms is a 5 % penalty at the
        # ~9 % steady-state sync rate; S <= 5.7 ms at the ~37 % latched rate.
        self._ple_resync_sync_ms: list[float] = []
        self._ple_resync_read_ms: list[float] = []
        self._ple_resync_timing = _ple_resync_timing_enabled()
        # Which request each slot's mirror describes, and the disagreement ledger
        # the disable decision reads. The owner has to be known before two token
        # sequences can be compared at all: see the cross-check in
        # :meth:`_ple_resync_mirror_inner` for what boot #27 found out.
        self._ple_host_slot_req: dict[int, str] = {}
        self._ple_resync_disagree_slots: dict[int, int] = {}
        self._ple_resync_owner_skip = 0
        # Resyncs where the mirror claimed MORE cells than the device had
        # committed. Not a disagreement -- there is nothing in common to compare --
        # but a distinct condition, and boot #28 proved it is the common one. See
        # :meth:`_ple_resync_mirror_inner`.
        self._ple_resync_mirror_ahead = 0
        # Resyncs where the window did not match the row at the positions the
        # mirror's length claims, but matches it once the comparison is shifted --
        # a lagged commit, not a bad read. The 2026-10-05 crashes counted these as
        # disagreements and stopped the boot twice. See :meth:`_ple_offset_explains`.
        self._ple_resync_mirror_offset = 0
        # Re-detections of a comparison this slot has already produced. A stuck
        # slot is one event; it must not cost a strike per step. See
        # :meth:`_note_resync_disagreement`.
        self._ple_resync_disagree_repeat = 0
        self._ple_disagree_sig: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        # Consecutive steps each slot has refused on its current signature.
        self._ple_resync_disagree_stuck: dict[int, int] = {}
        # The CUDA-graph mode of the step being prepared, set by
        # :meth:`_stage_host_ple_ids` from the runner's own batch descriptor. This
        # is the per-step answer to a question the sticky ``_ple_host_captured``
        # flag cannot answer, and every staleness claim now has to name it.
        self._ple_step_cg_mode: str | None = None
        # Indirection so a test can stand in for the device sync without a
        # device. Production calls the real one.
        self._ple_resync_sync = torch.cuda.synchronize
        # The draft-block generation this model state last consumed. Freshness is
        # the one property of the mirror that no shape check can infer.
        self._last_draft_gen = -1
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
        # P1.6: the host mirror of committed tokens, plus the request -> slot map
        # that lets ``remove_request`` clean the mirror's queue. Built here rather
        # than lazily because seeding happens at add time, before any step has
        # consulted the knob -- and an unseeded window is a step that cannot
        # stage. The size comes from the ngram config, so the mirror is correct
        # whether or not ids are ever staged on the host.
        self._ple_host_history = HostTokenHistory(
            self.max_num_reqs, self.ngram_context_len + 1
        )
        self._ple_host_slots: dict[str, int] = {}

    # ------------------------------------------------------------- lifecycle
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Seed with exactly the list ``req_states.add_request`` staged into its
        # UVA array -- the runner passes the same ``prefill_token_ids`` to both.
        # That list is host-written, so it is the one part of a request's history
        # the host can trust without asking the device; everything after it
        # arrives through ``HostTokenHistory.stage_step``.
        self._ple_host_slots[new_req_data.req_id] = req_index
        # The mirror's owner, recorded at the only moment it is knowable for sure.
        # A slot's row outlives its request (``invalidate`` clears the mirror, the
        # device's ``all_token_ids`` row keeps the previous occupant's tokens
        # forever), so "which request do these cells belong to" has to be carried
        # rather than inferred.
        self._ple_host_slot_req[req_index] = new_req_data.req_id
        self._ple_host_history.seed(req_index, new_req_data.prefill_token_ids or ())

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        # The runner calls this *before* ``req_states.remove_request``, so the
        # slot is still resolvable here. Cleaning the mirror is not tidiness: an
        # in-flight commit belonging to a finished request would otherwise land on
        # whoever reuses the slot next and corrupt its token count.
        req_index = self._ple_host_slots.pop(req_id, None)
        if req_index is not None:
            self._ple_host_history.invalidate(req_index)
            # A slot is reused by the next request, and the resync budget, the
            # mirrored owner and the disagreement ledger all belong to a request,
            # not to a slot number. Carrying any of them across would let the next
            # request inherit a spent budget -- or, worse, a comparison against a
            # window it never owned.
            self._ple_resync_slots.pop(req_index, None)
            self._ple_resync_disagree_slots.pop(req_index, None)
            self._ple_disagree_sig.pop(req_index, None)
            self._ple_resync_disagree_stuck.pop(req_index, None)
            self._ple_host_slot_req.pop(req_index, None)

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

    def _stage_host_ple_ids(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        *,
        consumed_qsl: Any = None,
        cudagraph_mode: CUDAGraphMode | None = None,
    ) -> bool | None:
        """Compute this step's PLE ids on the host, if the step is one we can.

        Returns the step's staging decision, which is what the runner's dispatch
        veto acts on: ``True`` the rows are in the pinned buffer this step's replay
        will deliver, ``False`` attempted and refused (so a captured FULL graph must
        not be replayed for it), and ``None`` this arm is not trying to stage at all
        -- knob off, policy declined, or staging disabled, where vetoing would send
        every step of the boot eager to buy nothing.

        Runs every step, including steps that replay a captured CUDA graph —
        which is the whole reason the host has to do this work itself: a captured
        FULL decode graph replays the H2D out of ``_pinned_output`` and contains
        neither the D2H of the ids nor the gather that produced them.

        ``consumed_qsl`` is the ``query_start_loc`` view ``prepare_inputs`` is
        handing the forward for this step -- the boundary array the ngram fill
        actually reads, at the capture-padded width. Differential mode forwards it
        to the layer so its per-step line can report what the compute consumed
        instead of what the gate compared (P1.14); with the knob off, or in mode 1,
        it is read by nobody.

        Every failure mode returns without touching the model inputs, so the
        device path stays the default answer to anything unexpected. Refusals are
        counted per reason and the first of each is logged: coverage is a claim
        that needs evidence, and "the knob was set" is not that evidence.
        """
        # Set before anything can refuse, so the first refusal of a boot already
        # knows which graph mode its step was dispatched in.
        self._ple_stale_counted_step = False
        self._ple_step_cg_mode = getattr(cudagraph_mode, "name", None)
        if cudagraph_mode is not None and self._ple_step_cg_mode is None:
            self._ple_step_cg_mode = str(cudagraph_mode)
        # Identity of THIS call, which is deliberately not ``_ple_host_steps": that
        # counter advances only when a step stages, so two refused steps in a row
        # share a value -- and the repair's per-step bound needs a number that moves
        # whether or not the step succeeded. ``getattr`` because the tests drive this
        # path with stub states that never run ``__init__``.
        self._ple_host_calls = getattr(self, "_ple_host_calls", 0) + 1
        self._forget_host_stage_flags()
        mode = ple_hostids.hostids_mode()
        if mode == 0 or not self.uses_ngram_embedding:
            return None
        if (
            self.PLE_HOST_STAGE_UNDER_FULL == "off"
            and self._ple_step_full()
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Deliberately above the layer resolution and the check_host_gen
            # tripwire, not below them: the tripwire raises when a captured graph
            # holds the staged branch and the handoff went stale, and under this
            # policy that describes every step of the boot. Saying "we declined"
            # once is the whole log output this path ever produces here.
            #
            # ``not ...is_current_stream_capturing()`` is load-bearing, and
            # ``test_a_capture_time_step_is_staged_even_with_the_policy_off`` is
            # what enforces it (the two older capture tests --
            # ``test_the_stage_gate_stays_out_of_graph_capture`` and
            # ``test_resync_never_runs_inside_a_capture`` -- cover the neighbouring
            # case of a capture with no descriptor threaded, so they stay green under
            # this mutation; the red one is a FULL step during capture). The staged
            # branch is the only branch of the PLE op that a stream capture can hold:
            # the device path reads three device arrays synchronously, which is
            # ``hipErrorStreamCaptureUnsupported`` and kills the engine at startup.
            # Declining during capture would send the capture down that path, so a
            # FULL boot would die at the first capture instead of serving zeros. It
            # costs the policy nothing: the capture-time rows are zeroed on purpose
            # (``ple_layer.py`` ``staged_rows.zero_()``, "host memset on pinned
            # memory: not a graph node"), which is exactly the contribution being
            # declined here and the reason "PLE contributes zero" is arithmetic
            # rather than inference.
            #
            # The heartbeat at the bottom of this branch is not a contradiction of the
            # paragraph above about printing once. Boot 20261008-210206 -- the first
            # boot served under this policy -- printed its 16 coverage lines at 21:08:55
            # and then NOTHING for the rest of the evening, because the serving
            # heartbeat is keyed to ``_ple_host_steps`` and no step stages under this
            # policy. A reader could not tell "PLE declined and the arm is healthy"
            # from "the engine stopped", which is the same missing-counter mistake the
            # disabled heartbeat was written for on boot 204346.
            self._ple_host_policy_off += 1
            # ``refused=False``: this is a DECLINE, so ``_stage_host_ple_ids`` returns
            # None, so the dispatch veto leaves the descriptor at FULL, so the step
            # replays. Charging it to the veto ledger would read ``stale_steps=0
            # veto=N/0`` on a boot where every FULL step replays zeroed rows -- the
            # exact direction a ledger must not be wrong in.
            self._note_stale_step(refused=False)
            self._note_disabled_heartbeat(
                "the ZEROED rows the capture recorded -- PLE contributes nothing to "
                "these steps (PLE_HOST_STAGE_UNDER_FULL=off, by choice, not by a trip)",
                prefix="STAGING DECLINED BY POLICY",
            )
            if not self._ple_policy_off_logged:
                self._ple_policy_off_logged = True
                logger.info(
                    "PLE HOSTIDS: host staging declined under FULL graphs by policy "
                    "(PLE_HOST_STAGE_UNDER_FULL=off); this step replays the captured "
                    "graph's zeroed rows, so PLE contributes nothing to it. The arm IS "
                    "registered -- absence of any other PLE HOSTIDS line on this boot "
                    "means declined, NOT absent (compare boot #35, where the knob was "
                    "never set and there was no arm at all)."
                )
            return None
        if self._ple_host_layers is None:
            from vllm.models.qwen4_exp.amd.ple_layer import host_stageable_layers

            layers = [
                layer
                for layer in host_stageable_layers()
                if layer.ngram_size - 1 == self.ngram_context_len
            ]
            if not layers:
                # The env var is set but no PLE module registered (import order,
                # or the knob was set for a model without PLE). Say so once.
                self._note_host_refusal("no-registered-layers", refused=False)
                return None
            self._ple_host_layers = layers
            self._ple_host_gen_layers = list(layers)
        # Tripwire first, and above the refusal paths it exists to observe: a step
        # the host could not stage is exactly the step whose replay is worth
        # looking at. It raises when a graph captured the staged branch and the
        # handoff went stale, which is the one case where "fall back to the device
        # path" is not something a replay can do.
        for layer in self._ple_host_gen_layers:
            layer.check_host_gen()
        if not self._ple_host_layers:
            # ``disable_host_staging`` empties the layer list, after which this
            # method's job -- hand rows to a layer -- cannot be done at all. It
            # used to fall through anyway: the build ran, the mirror kept
            # tracking, and ``_ple_host_steps`` advanced through a staging loop
            # that iterated zero times. That is how boot #27 reported 5,120 staged
            # steps in the same log line as a mirror that had received 1,140
            # commits and a layer that had been handed no rows since 02:30:55. A
            # counter that survives the thing it counts cannot be the shipping
            # signal, so say what the step is: a refusal, because staging is off.
            #
            # Which refusal, though, is a decision about the step's fate, not a
            # label. A disabled arm answers ``None`` ("not trying"), and
            # :meth:`ple_veto_descriptor` abstains on ``None`` by design -- sending a
            # whole boot eager to buy nothing is its own outage -- so the step replays
            # the previous step's rows. That is right for a boot that captured the
            # zeroed dummy branch, and wrong for one that captured the staged branch:
            # there ``None`` is a permanent stale serve, and it is silent because
            # ``staged_steps`` stops advancing and takes the heartbeat with it. When
            # the capture held the staged branch and the policy is "eager", the
            # honest answer is the one the veto acts on.
            self._note_disabled_heartbeat()
            if getattr(self, "_ple_disabled_zero", False):
                # The zero route: this step replays pinned rows that were zeroed when
                # the arm was disabled, so PLE contributes nothing to it. That is a
                # DECLINE (``None``), not a refusal -- the veto exists to prevent a
                # replay of the PREVIOUS step's rows, which is not what is happening
                # here, and downgrading would cost a factor of 3.1 per step to buy back
                # an arm that was disabled for being broken. ``refused=False`` puts the
                # step in the stale ledger, whose meaning on this route is "replayed
                # zeroed rows": the ledger over-reports rather than hide, and a reader
                # who sees it rising is reading the right number of PLE-less steps.
                self._note_host_refusal("staging-disabled-zero", refused=False)
                return None
            if getattr(self, "_ple_disabled_eager", False):
                # ``refused=True`` (the default) is what puts the step in the veto
                # ledger, which is the correct ledger: the step is not stale, the
                # device path computes its rows.
                self._note_host_refusal("staging-disabled-eager")
                return False
            self._note_host_refusal("staging-disabled", refused=False)
            return None
        if self._host_drafts is None:
            self._note_host_refusal("no-draft-source", refused=False)
            return None
        # Apply every queued commit whose device->host copy has landed. This is
        # the only place the queue is drained, and it never blocks: a commit still
        # in flight makes this step refuse below rather than wait, because waiting
        # here is the per-step host sync the whole design exists to remove.
        hist = self._ple_host_history
        # ``getattr`` because the suite drives this path with stub histories that
        # do not carry the counters -- same reason the heartbeat helpers are
        # module-level and the post_disable field reads through getattr.
        dropped_before = getattr(hist, "dropped", 0)
        hist.drain()
        if getattr(hist, "dropped", 0) > dropped_before:
            # Task #56 moved the count-range check to where the counts are valid,
            # which is inside ``drain`` -- so a row whose accepted count is out of
            # range is now discovered here, after the copy landed, instead of at
            # the epilogue. Same accounting as the stage-time drop it replaces:
            # the step already dispatched and ran, neither ledger is charged, and
            # the next step pays a repair. It has to be NOTED though, or the one
            # thing that makes a mirror hole visible in a serving log -- the
            # per-reason cadence line -- would go quiet exactly when the check
            # started running somewhere nobody logs from.
            self._note_host_refusal(
                "mirror-drop-at-drain", hist.last_drop, ledger=False
            )
        if hist.pending:
            self._note_host_refusal("history-pending")
            return False

        num_reqs = input_batch.num_reqs
        try:
            values, row_idx, gen = self._host_drafts.wait_host_draft_tokens()  # type: ignore[attr-defined]  # noqa: E501
            # Freshness, not presence. Two consecutive decode steps over an
            # unchanged batch produce an IDENTICAL slot->row map with different
            # values, so a map that lines up proves nothing about which step the
            # numbers came from. A generation that did not advance since the last
            # consumption means the epilogue staged nothing in between, and these
            # are the previous step's drafts -- the signature the 14:10 boot
            # showed: the rows whose tokens come from history and draft column 0
            # agreed, the row that needs column 1 did not.
            if values is not None and gen == self._last_draft_gen:
                raise HostIdsRefused("draft-gen-stale", f"gen={gen}")
        except HostIdsRefused as exc:
            self._note_host_refusal(exc.reason, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
            self.disable_host_staging(
                f"host draft mirror read failed ({type(exc).__name__}: {exc})"
            )
            return None

        # Two attempts, not a loop, and the second only ever follows a mirror
        # repair. ``history-length`` is the one refusal worth retrying because it
        # is the only one that describes the mirror rather than this step's shape,
        # and a short mirror is exactly what a lost commit leaves behind. Every
        # other reason is a property of the batch, which no host-side repair
        # changes, so retrying it would only cost a build.
        built: tuple[Any, ...] | None = None
        for attempt in (0, 1):
            try:
                built = build_decode_step_inputs(
                    input_batch.idx_mapping_np[:num_reqs],
                    input_batch.num_scheduled_tokens[:num_reqs],
                    req_states.num_computed_tokens_np,
                    self._ple_host_history.window,
                    self._ple_host_history.length,
                    values,
                    row_idx,
                    ngram_context_len=self.ngram_context_len,
                    eos_token_id=self.ngram_eos_token_id,
                )
            except HostIdsRefused as exc:
                if (
                    attempt
                    or exc.reason != "history-length"
                    or not self._ple_resync_mirror(req_states, input_batch, str(exc))
                ):
                    self._note_host_refusal(exc.reason, str(exc))
                    return False
            except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
                self.disable_host_staging(
                    f"host id build failed ({type(exc).__name__}: {exc})"
                )
                return None
            else:
                break
        assert built is not None
        input_ids, query_start_loc, ngram_context = built

        gather = mode == 1
        # The device half of this step's inputs, captured in the worker where both
        # sides are in scope. forward() cannot do it: torch.compile runs that body at
        # trace time, so anything stashed there is a trace-time value. The runner's
        # own arrays are padded, so the gate slices them to the host's lengths.
        #
        # The window is handed over as a reference to the state's own buffer, not as
        # a copy, and that is the whole reason it can be compared at all:
        # prepare_inputs() fills ``ngram_context`` via ``_prepare_ngram_context`` and
        # *then* calls this staging function, so by the time the op reads the buffer
        # back it holds this step's window -- the same cells the forward was handed a
        # view of. An earlier version of this file claimed the worker had no
        # device-side counterpart for the window and passed None; boot #8's inversion
        # put the one wrong term exactly in the cell that window feeds, so the claim
        # went looking for its evidence and failed.
        #
        # This block used to live inside ``if not gather``, because only the
        # differential had a question to ask with it. Mode 1 now asks the same
        # question before it serves (see ``stage_inputs_agree``), so the triple is
        # built for both modes.
        device_inputs: tuple[Any, ...] | None = None
        d_ids = getattr(input_batch, "input_ids", None)
        d_qsl = getattr(input_batch, "query_start_loc", None)
        d_ctx = getattr(self, "ngram_context", None)
        if d_ids is not None and d_qsl is not None:
            device_inputs = (d_ids, d_qsl, d_ctx)
        if not gather:
            # Differential mode: keep the provenance so a mismatch can name the
            # block its draft columns came from instead of describing its output.
            for layer in self._ple_host_layers:
                layer.note_draft_provenance(values, row_idx, gen)
        elif not torch.cuda.is_current_stream_capturing():
            # Mode 1: attribute the step BEFORE any layer is staged, so a refusal
            # cannot leave one layer holding rows the next layer does not. A step
            # that fails the gate stages nothing and is served by the device path.
            #
            # The capture test is not decoration. A blocking D2H inside a capture
            # aborts it (hipErrorStreamCaptureUnsupported) and that kills the boot,
            # and during a capture there is no step to attribute anyway: replay
            # reads no host arrays.
            host_inputs = (input_ids, query_start_loc, ngram_context)
            for layer in self._ple_host_layers:
                check = getattr(layer, "stage_inputs_agree", None)
                if check is None:
                    # Stub embeddings (probe/prof tests) never gathered and never
                    # staged real rows; do not make them a reason to refuse.
                    continue
                if not check(host_inputs, device_inputs):
                    self._note_host_refusal("stage-inputs-disagree")
                    return False
        for layer in self._ple_host_layers:
            try:
                accepted = layer.stage_host_step(
                    input_ids,
                    query_start_loc,
                    ngram_context,
                    gather=gather,
                    device_inputs=device_inputs,
                    consumed_qsl=consumed_qsl,
                )
            except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
                self.disable_host_staging(
                    f"host id staging failed ({type(exc).__name__}: {exc})"
                )
                return None
            if not accepted:
                # The layer took the ids and wrote nothing: its own differential is
                # off, or the step is wider than its pinned buffer. Either way
                # these rows are not in the buffer this step's replay reads, which
                # is the definition of a step that did not stage. The return value
                # used to be dropped on the floor and the step counted as staged.
                self._note_host_refusal("layer-declined-staging")
                return False
        self._last_draft_gen = gen
        self._ple_host_steps += 1
        if self._ple_host_steps == 1:
            logger.info(
                "PLE HOSTIDS: host step inputs built for %d request(s), %d row(s) "
                "(mode %s: %s)",
                num_reqs,
                int(input_ids.size),
                mode,
                "host ids authoritative, device D2H skipped"
                if gather
                else "differential: device path in charge, ids compared every step",
            )
        # Separate ``if``, not the ``elif`` above: step 1 is on the ramp, so the
        # first staged step prints its inputs line AND a coverage line. That pair is
        # the earliest proof in the log that the arm works, and it is the reason the
        # ramp exists -- see :func:`_ple_heartbeat_due`.
        if _ple_heartbeat_due(self._ple_host_steps):
            # A serving log should not print per step, and a boot that only prints
            # on failure cannot tell a healthy arm from an idle one. Boot #26 said
            # "576 refusals" and never said how few steps staged; this line is the
            # denominator. Boot 234401 is the other half of the lesson: a boot that
            # stages fewer than the cadence printed no denominator at all.
            logger.info(
                "PLE HOSTIDS: staged %d decode step(s) so far. %s",
                self._ple_host_steps,
                self._ple_host_coverage(),
            )
        return True

    def _note_disabled_heartbeat(
        self, mode: str | None = None, *, prefix: str = "STAGING DISABLED"
    ) -> None:
        """Keep a heartbeat alive on an arm that is not staging.

        The serving heartbeat is keyed to ``_ple_host_steps``, which is the count of
        steps that STAGED -- so it goes permanently silent at the exact moment the arm
        dies, which is the one state where a reader most needs to know the engine is
        still serving. Boot ``vllm.log.20261007-204346`` is the proof: it disabled
        itself at 23:01:56 and ran 14,574 more steps with no heartbeat line at all,
        so the absence of the line read as "no traffic" when it meant "the arm is
        dead". That is a missing counter, and a missing counter is not a measurement
        of zero.

        Keyed to ``_ple_host_calls`` instead, which advances on every call whether or
        not the step staged, so it keeps moving while the engine serves. It is
        deliberately a *call* counter: on a disabled arm the calls are the steps.
        Same cadence as the live heartbeat, ramp included, so the two lines a reader
        learns are one rhythm and a declined boot says something within its first
        steps instead of at 1024 -- boot 210206 printed nothing for its whole
        evening because 1024 was unreachable for it.
        """
        calls = int(getattr(self, "_ple_host_calls", 0))
        if not calls or not _ple_heartbeat_due(calls):
            return
        # One line per call boundary, not one per call that happens to land on it.
        last = int(getattr(self, "_ple_disabled_hb_at", 0))
        if last == calls:
            return
        self._ple_disabled_hb_at = calls
        mode = mode or (
            "EAGER on the device path via the dispatch veto"
            if getattr(self, "_ple_disabled_eager", False)
            else "the ZEROED pinned rows -- PLE contributes nothing to these steps"
            if getattr(self, "_ple_disabled_zero", False)
            else "replaying the last staged rows"
        )
        logger.warning(
            "PLE HOSTIDS: %s -- %d staging call(s) since, served %s. %s",
            prefix,
            calls,
            mode,
            self._ple_host_coverage(),
        )

    def _ple_step_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        cudagraph_mode: CUDAGraphMode | None,
    ) -> tuple[dict[str, Any], bool | None]:
        """The PLE half of ``prepare_inputs``, callable on its own, once per step.

        Split out so the dispatch veto can ask the staging question before
        :meth:`prepare_attn` has committed the step to a graph mode. Returns the
        model-inputs fragment and the staging decision.

        Host ids first: the pinned rows they fill are what this step's H2D (or,
        under a captured graph, the replayed H2D) delivers to the device, so they
        have to be in place before the forward is enqueued. ``consumed_qsl`` is the
        same object the model is about to be handed -- not the runner's
        ``input_batch.query_start_loc`` the differential's input gate compares
        against, which is a different buffer at a different width (P1.14).
        """
        query_start_loc = self._refresh_ple_query_start_loc(input_batch)
        model_inputs = {
            "query_start_loc": query_start_loc,
            "ngram_context": self._prepare_ngram_context(input_batch, req_states),
        }
        decision = self._stage_host_ple_ids(
            input_batch,
            req_states,
            consumed_qsl=query_start_loc,
            cudagraph_mode=cudagraph_mode,
        )
        return model_inputs, decision

    def ple_prepare_step(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        *,
        cudagraph_mode: CUDAGraphMode | None = None,
    ) -> bool | None:
        """Prepare this step's PLE inputs, stage its rows, and report the decision.

        The runner calls this between its own ``prepare_inputs`` and
        ``model_state.prepare_attn`` for steps dispatched as FULL. It has to live
        there and not later, for two reasons that are each load-bearing on their
        own: ``prepare_attn`` builds metadata whose request and token counts are
        mode-dependent (``mamba_hybrid.py`` takes ``num_reqs_after_padding`` under
        FULL and ``num_reqs`` otherwise), so metadata built for a replay cannot be
        handed to the eager forward the veto needs; and ``prepare_attn`` cannot
        simply be called a second time, because it ends in
        ``recoverssm.record_step`` and re-derives the mamba align indices -- a
        second call is a second recorded step.

        The inputs are cached against the identity of ``input_batch``, which the
        same step passes on to :meth:`prepare_inputs`, so the fill and any repair
        happen once. The cached reference is strong, so the ``is`` test cannot
        alias onto a recycled address; a different batch falls back to doing the
        work, which is the safe direction (a redundant fill, never a skipped one).
        """
        model_inputs, decision = self._ple_step_inputs(
            input_batch, req_states, cudagraph_mode
        )
        self._ple_prepared = (input_batch, model_inputs, decision)
        self._ple_step_rows = decision
        return decision

    def _forget_host_stage_flags(self) -> None:
        """Invalidate every layer's "rows are staged for this step" flag.

        Under a captured graph the op body never runs, so a step that staged
        successfully leaves ``_host_stage_rows`` set and unconsumed indefinitely. That
        is harmless while every later step also replays. It is not harmless the moment
        a step runs eager -- which is what the dispatch veto does on a refusal, and
        what every step does after staging is disabled -- because the op body then
        finds a flag from an earlier step, and the row count agrees since it is the
        same batch shape, so :meth:`take_host_stage` says yes and the D2H is skipped.
        The guard designed to catch exactly that is defeated by a flag that is merely
        old, which is why this runs unconditionally rather than on the paths where it
        seems useful.

        Two callers, both on the way to an eager step: the top of
        :meth:`_stage_host_ple_ids`, above every early return, and the top of
        :meth:`disable_host_staging`, which is about to drop the layer list out from
        under the first caller.
        """
        for layer in self._ple_host_layers or ():
            forget = getattr(layer, "forget_host_stage", None)
            if forget is not None:
                forget()

    def _take_prepared_ple_inputs(self, input_batch):
        """Consume the PLE inputs the dispatch hook prepared for THIS batch, if any.

        Returns the fragment ``ple_prepare_step`` built, or ``None`` when nothing was
        prepared for this exact batch -- which is the ordinary case whenever the veto
        is not armed, and the correct case for a dummy batch or a step where the
        runner's ``InputBatch`` is not the object the hook was handed.

        The cache is cleared on every path, including the miss. A latch that survived
        a step which never reached :meth:`prepare_inputs` would make the NEXT step
        skip its staging and reuse rows from two steps back, and the coverage line
        would show a staged step that never happened.
        """
        prepared = self._ple_prepared
        self._ple_prepared = None
        if prepared is None or prepared[0] is not input_batch:
            return None
        return prepared[1]

    def ple_veto_descriptor(self, batch_desc, input_batch, req_states):
        """Return the descriptor this step should actually be dispatched with.

        Called by the runner between its input copy and ``prepare_attn`` -- the only
        point where a downgrade is still cheap, because ``prepare_attn`` builds
        metadata whose request and token counts depend on the graph mode
        (``mamba_hybrid.py`` takes ``num_reqs_after_padding`` under FULL and
        ``num_reqs`` otherwise) and cannot be run twice: it ends in
        ``recoverssm.record_step`` and re-derives the mamba align indices, so a
        second call records a second step.

        The logic lives here rather than in ``execute_model`` because that file is
        shared by every model and says so in its own header; a captured graph's
        staged-branch hazard is Qwen4Exp's business alone. Only the decision to arm
        stays in the runner, because that depends on the dispatch path (data
        parallelism).

        Returns ``batch_desc`` unchanged unless this is a FULL step whose rows the
        host could not stage, in which case it returns the same descriptor with
        ``cg_mode=NONE``: the eager forward runs the op body, which takes the device
        branch because no staged rows were accepted. ``CudagraphManager.dispatch``
        already hands out NONE for token counts it never captured, so a NONE step is
        ordinary traffic on this box, not a special case invented here -- it is what
        every prefill step of this boot is.
        """
        if not self._ple_veto_armed or batch_desc.cg_mode != CUDAGraphMode.FULL:
            # PIECEWISE and NONE steps run the op body, which IS the device path, so
            # a refusal on those already costs nothing and there is nothing to veto.
            return batch_desc
        decision = self.ple_prepare_step(
            input_batch, req_states, cudagraph_mode=batch_desc.cg_mode
        )
        if decision is not False:
            # True: the rows are staged and this step's replay delivers them.
            # None: the arm is not trying at all (knob off, policy declined, staging
            # disabled) -- replay is the only thing this step can do, and vetoing on
            # that would send the entire boot eager to buy nothing.
            return batch_desc
        # A captured FULL graph holds the staged PLE branch, so replaying it here
        # serves the PREVIOUS step's n-gram rows: 1.98 accepted tokens per step
        # measured, against 4.00 for correct and 3.31 for zeroed. The step pays a
        # replay's worth of launch overhead instead, and its PLE is right for this
        # step. On the boot that sized this, 0.2 % to 3.2 % of decode steps.
        self.note_ple_veto_applied()
        return dataclasses.replace(batch_desc, cg_mode=CUDAGraphMode.NONE)

    def ple_rows_refused(self) -> bool:
        """Did THIS step attempt host staging and lose?

        Three-state on purpose. ``None`` means the arm is not trying -- knob off,
        policy declined, or staging disabled -- and treating that as a refusal would
        send every step of the boot eager to buy nothing, which is a different
        performance cliff wearing this one's clothes.
        """
        return self._ple_step_rows is False

    def arm_ple_dispatch_veto(self, armed: bool) -> None:
        """Called once by the runner: will refused FULL steps be dispatched eager?

        Only the runner knows, because the answer depends on things a model state
        must not assume about: data parallelism (a downgrade decided locally desyncs
        the mode the ranks agreed on collectively) and whether this build can reach
        the eager branch at all. The state needs the answer BEFORE the refusal, not
        after it, because the refusal is where a step is charged to a ledger, and a
        step the runner has committed to rescuing is not a stale step.
        """
        self._ple_veto_armed = bool(armed)

    def note_ple_veto_applied(self) -> None:
        """Record a downgrade the runner actually performed.

        Counted separately from the refusals the veto caught so the two can be
        compared: a boot reporting ``veto=512/0`` has an armed runner that never
        downgraded anything, which would otherwise read as hundreds of correctly
        served steps. The self-check is the point of keeping them apart.
        """
        self._ple_veto_applied += 1
        if self._ple_veto_applied == 1:
            logger.warning(
                "PLE HOSTIDS: dispatch veto fired for the first time -- a step whose "
                "PLE rows could not be staged runs EAGER on the device path instead "
                "of replaying a captured graph that would have served the PREVIOUS "
                "step's rows. This is the correct-and-up path: the step costs more "
                "than a replay and its PLE is right. Read veto=<ledgered>/<applied> "
                "on the coverage line -- applied is the downgrades that landed, and "
                "the only gap to expect is capture/warmup refusals -- and "
                "stale_steps, which should now stay near zero apart from declines. "
                "Coverage: %s",
                self._ple_host_coverage(),
            )

    def set_host_draft_source(self, handler: object) -> None:
        """Wire in the worker's staged draft-token mirror (P1.1).

        Called once at init by the model runner, and only when staging is enabled,
        so the per-step path costs one attribute read. Without it the host has
        rows 1..q-1 of no step and refuses every batch.
        """
        self._host_drafts = handler

    def _policy_off_now(self) -> bool:
        """Is the arm quiesced for the step in flight?

        Reads the cg mode that ``_stage_host_ple_ids`` recorded at the top of this
        step, so the two halves of the arm agree about which step they are talking
        about instead of each guessing. Used to stand the mirror commit down as well:
        its only consumer is the staging path, so committing while declined is
        bookkeeping for a reader that does not exist, plus the ``stage-tokens-rejected``
        refusals that bookkeeping generates.
        """
        return (
            self.PLE_HOST_STAGE_UNDER_FULL == "off" and self._ple_step_full()
        )

    def stage_host_tokens(
        self,
        slots: object,
        sampled: object,
        num_sampled: object,
        done: object,
    ) -> bool:
        """Queue one step's accepted tokens for the host PLE mirror (P1.6).

        Called by the model runner's epilogue with the arrays the async output
        copy produced, plus its copy event: nothing here reads ``sampled``, so
        the data may still be in flight. The consumer applies it in
        ``_stage_host_ple_ids`` once ``done.query()`` says so, and refuses the
        step until then -- which is what keeps this knob free of the per-step
        host sync it exists to remove.

        Returns whether the commit was queued. Failures are counted, never
        raised: a knob that can kill a serving step is worse than one that
        reports no coverage.
        """
        if ple_hostids.hostids_mode() == 0:
            # The epilogue runs on every step of every boot; with the knob off
            # this must cost nothing and, above all, must not log.
            return False
        if self._policy_off_now():
            # Quiesced with the staging path, for the reason in _policy_off_now.
            # Not counted as a refusal: nothing was attempted.
            return False
        try:
            staged = self._ple_host_history.stage_step(
                slots,
                sampled,
                num_sampled,
                done,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
            # ``ledger=False``: the epilogue, after this step dispatched and ran.
            self._note_host_refusal(
                f"stage-tokens-{type(exc).__name__}", ledger=False
            )
            return False
        if not staged:
            # The detail names the check that lost the commit. It matters more
            # here than anywhere: a lost commit used to cost the request's whole
            # remaining coverage, and boot #26 had 67 per worker with no way to
            # tell a padded sampler row from a full queue.
            # ``ledger=False``: a dropped commit costs the NEXT step a repair, it does
            # not make THIS step stale, and it is not something the veto can act on.
            self._note_host_refusal(
                "stage-tokens-rejected",
                self._ple_host_history.last_drop,
                ledger=False,
            )
        return staged

    PLE_HOST_STAGE_UNDER_FULL = os.environ.get(
        "VLLM_GFX906_PLE_HOST_STAGE_UNDER_FULL", "off"
    ).strip().lower()
    """Attempt host PLE staging at all on a step that replays a captured FULL graph.

    ``"off"`` (the default since 2026-10-05) does not attempt it: the step is counted
    as a stale step and nothing else on this path runs -- no build, no mirror
    bookkeeping, no per-refusal ``logger.warning`` on the critical path, and not the
    ``check_host_gen`` tripwire, which would otherwise fire on every step of every
    boot because its whole premise ("a handoff we tried to keep fresh went stale")
    no longer holds when the handoff is declined by policy. Anything else, including
    an unrecognised value, means ``"stage"``: a typo must restore the old behaviour,
    never silently disable something.

    Why off is the default, stated with the number that decided it. Boot #36
    (``vllm.log.20261005-190309``, fresh, correct environment, agent idle) reported
    ``staged_steps=20 stale_steps=736 refused=751 resync=9``: the arm staged **2.6 %
    of decode steps**, and a single 512-token stream with no co-tenant refused ~190 of
    its ~197 steps. The coverage is not intermittent and cannot be fixed by raising
    the cap, because a refusal skips staging while staging is the only thing that
    advances the mirror:

        mirror behind -> refuse -> mirror not advanced -> still behind -> refuse ...

    so a request spends its first two steps (``PLE_RESYNC_PER_SLOT``) and decodes the
    rest stale. Refusing on purpose removes the pretence and the log noise; it changes
    nothing about what the device computes.

    What it costs, because this is a model-output decision and not a performance one:
    the captured graph's staged branch is a copy out of pinned rows that are zeroed
    before capture, so **PLE contributes zero to every FULL-graph step**. The
    acceptance test is a greedy-fingerprint diff against a boot that stages
    (``scripts/greedy-fingerprint.py``), never a throughput number -- a boot with PLE
    hard-zeroed answers coherently and with a healthy acceptance rate, which is exactly
    why coherence is not evidence here (boot #16).

    Scope: FULL-graph steps only. Prefill, mixed, and piecewise steps still stage
    exactly as before, so a ``GRAPHMODE=piecewise`` boot is unaffected. While this is
    ``"off"``, :attr:`PLE_DOUBT_UNDER_FULL` is inert on those steps for the same reason
    the tripwire is: nothing reaches it.

    One exception, and it is the difference between a server that serves and one that
    never comes up: **capture-time steps still stage.** The staged branch is the only
    branch of the PLE op a stream capture can hold -- the device path reads three
    device arrays synchronously, which is ``hipErrorStreamCaptureUnsupported`` -- so
    declining during capture would abort the capture and kill the boot at startup.
    It costs the policy nothing, because the rows staged during capture are zeroed
    before the graph closes. Enforced by
    ``test_a_capture_time_step_is_staged_even_with_the_policy_off``, which is the
    FULL-during-capture case; the two older capture tests cover a capture with no
    descriptor and do not catch this mutation.
    """

    PLE_DOUBT_UNDER_FULL = os.environ.get(
        "VLLM_GFX906_PLE_DOUBT_UNDER_FULL", "zero"
    ).strip().lower()
    """What a host-side doubt does once a FULL decode graph is in play.
    Four answers. The box has measured three of them, and the measurements are the
    reason the default has changed twice in two days.

    ``"zero"`` (the shipped default) disables the arm and zeroes the pinned rows ONCE,
    at the moment of the trip -- the same operation capture already performs on its
    dummy batch. The graph replays zeros for the rest of the boot: full speed, no stale
    row, and the step keeps its FULL descriptor because there is nothing for the veto
    to act on. Cost: PLE contributes nothing to a graphed decode step. Soft estimate,
    cross-boot and therefore flagged: acceptance 2.58 with zeroed rows (boot #37)
    against 2.61 staged (boot 20261007-234401), same step cost.

    ``"eager``" (the default for one day, 2026-10-08) reports every later FULL step as
    *attempted and lost*, so :attr:`PLE_HOST_VETO` drops the **whole step** to
    ``CUDAGraphMode.NONE`` and the op body computes this step's rows on the device.
    Correct rows, and the right thing to want, except that the veto prices the entire
    forward and this box is launch-bound (~400 eager kernel launches a step). Boot
    20261008-083018 measured it: **15.88 tok/s single-stream against 52.05 on the
    staged boot -- a factor of 3.1 in step cost** -- with acceptance 2.49 against 2.61.
    The **+15 %** this docstring used to quote is the cost of the device *gather inside
    a captured graph* (E1 Phase 0): a different mechanism with a different price, and
    shipping on that number is why this default lasted one day.

    ``"degrade"`` disables the arm and carries on, which under a captured staged branch
    means replaying the previous step's pinned rows forever. That is what boot
    20261007-204346 did for 14,574 steps while its heartbeat stayed silent.

    ``"stop-boot"`` raises :class:`PLEHostDoubtUnderFullGraph` and takes the engine
    down.

    **Why zero and not eager, in the arithmetic the old docstring got wrong.** Throughput
    is acceptance divided by step cost, so the regimes compare as 2.58/1.0 zeroed,
    1.98/1.0 stale (1.98 is the measured stale-rows acceptance), and 2.49/3.1 = 0.80
    eager. Serving correct rows eagerly after a trip therefore **loses to serving stale
    rows** on throughput at any acceptance this model can reach: to beat stale it would
    need correct-rows acceptance above 3.1 x 1.98 = 6.2, and the ceiling is
    ``1 + num_speculative_tokens`` = 4. ``"eager"`` is not "correct but slow"; it is
    correct and slower than the thing it was meant to avoid. Zeroed rows are the only
    option that is both fast and honest about what it does not have.

    What each mode charges stays visible, which is the point of the split. Under
    ``eager`` refusals land on ``veto=`` and ``stale_steps=`` **must stop rising**; under
    ``degrade`` and ``zero`` they land on ``stale_steps=``, where on a ``zero`` boot that
    number means "steps that replayed zeroed rows", not "steps that replayed a previous
    step's rows". ``post_disable=`` on the coverage line says which of the three a reader
    is looking at, and ``scripts/ple-window-state.py`` turns it into a route.

    ``stop-boot`` stays one environment variable away, which is where it belongs: it was
    never a correctness policy once the disagreements turned out to be real corruption in
    the mirror's committed content (boot #32, 2026-10-05 15:38: ``mirrored [1,
    21474836484, 866]``, where 21474836484 is ``(5 << 32) + 4``); with a live producer
    bug it is an outage, and it took the box down three times running.
    """

    def disable_host_staging(self, why: str) -> None:
        if not self._ple_host_layers:
            return
        # Forget the staged flags BEFORE the layers leave this object. Both exits
        # below set ``_ple_host_layers = []``, which means the invalidation at the top
        # of :meth:`_stage_host_ple_ids` can never reach them again -- and every step
        # from here runs eager, so an op body would find the last good stage's flag,
        # see a row count that matches (same batch shape), skip its D2H and read rows
        # that belong to a previous step. The message below says the device path
        # handles every step from here on; this is what makes that sentence true
        # rather than nearly true.
        self._forget_host_stage_flags()
        captured = self._full_graph_captured_this_boot()
        if captured and self.PLE_DOUBT_UNDER_FULL == "stop-boot":
            # Deliberately not per-step. Once a FULL graph exists in this boot, the
            # steps that replay it are coming whether this one does or not, so
            # "the current step is PIECEWISE" is no protection.
            raise PLEHostDoubtUnderFullGraph(
                f"PLE host staging lost its own guarantee while a FULL decode "
                f"graph holds the staged branch: {why} There is no device path to "
                f"fall back to under a captured graph, so continuing would serve "
                f"stale PLE rows for the rest of the boot -- boot #28 ran 8141 such "
                f"steps. Stopping the boot instead, because this process was "
                f"started with Qwen4ExpModelState.PLE_DOUBT_UNDER_FULL = "
                f"'stop-boot'; 'zero' is the shipped default and the stale steps "
                f"are counted either way (stale_steps= on the coverage line). Note "
                f"that 'degrade' does NOT restore a device path -- the rows stay "
                f"stale -- it only stops this check from being the thing that takes "
                f"the engine down. 'eager' is the one that does restore it, by "
                f"reporting every later FULL step as refused so the dispatch veto "
                f"runs it on the device path."
            )
        if captured:
            if self.PLE_DOUBT_UNDER_FULL == "zero":
                # Zero the pinned rows ONCE, here, and let the graph replay zeros for
                # the rest of the boot. This is the operation capture already performs
                # on its dummy batch (see the capture-time zeroing clause in
                # ple_layer.py), for the same reason: a replay that no host filled must
                # read either the previous step's rows -- plausible text, the 2026-09-20
                # failure mode -- or nothing. Zero is "nothing", and the degeneration
                # smoke test catches nothing structurally.
                #
                # Whole extent, not ``[:rows]``: the H2D the graph replays copies the
                # extent it was captured with (the dummy batch's), and real steps are
                # narrower or wider than that.
                #
                # ``_pinned_gen`` is deliberately NOT zeroed. The generation tripwire
                # compares the gen the graph copies against the gen the host last
                # expected; freezing it makes that comparison agree forever, which is
                # true here (the graph really does hold those rows) and means the
                # tripwire is VACUOUS on this route. ``post_disable=zero`` and the
                # disabled heartbeat are what a reader has to go on instead, which is
                # why both exist.
                zeroed = 0
                for layer in self._ple_host_layers:
                    buf = getattr(layer, "_pinned_output", None)
                    if buf is None:
                        continue
                    # Host memset on pinned memory: not a graph node, and the buffers
                    # are ~(max_total_tokens x embedding_dim) fp16 = ~21 MB per staged
                    # layer, so "once per boot" is the only time this is affordable.
                    buf.zero_()
                    zeroed += 1
                self._ple_disabled_zero = True
                logger.warning(
                    "PLE HOSTIDS: host staging disabled after %d step(s) -- %s. A "
                    "graph has captured the staged branch, so a replay of these steps "
                    "would serve the PREVIOUS step's pinned rows. PLE_DOUBT_UNDER_FULL"
                    "=zero: zeroed %d pinned buffer(s) (%s) and every FULL step from "
                    "here replays ZEROS, so PLE contributes nothing to them. That is "
                    "the same content capture-time zeroing gives the dummy batch, at "
                    "full graph speed: no per-step cost, and no stale row. Cost is "
                    "acceptance -- 2.58 zeroed (boot #37) against 2.61 staged (boot "
                    "20261007-234401), cross-boot and therefore soft. stale_steps from "
                    "here counts steps that replayed ZEROED rows, not previous steps' "
                    "rows, and the generation tripwire is vacuous on this route (the "
                    "frozen gen agrees with itself forever). Refusals so far: %s",
                    self._ple_host_steps,
                    why,
                    zeroed,
                    ",".join(str(getattr(l, "ngram_size", "?")) for l in
                             self._ple_host_layers) or "none",
                    dict(self._ple_host_refusals),
                )
                self._ple_host_layers = []
                return
            if self.PLE_DOUBT_UNDER_FULL == "eager":
                # The replay is the thing being avoided, so say what replaces it.
                # A step is reported as *attempted and lost* from here on, which is
                # the tri-state answer that makes ple_veto_descriptor hand the
                # descriptor to CUDAGraphMode.NONE; the op body then runs, finds no
                # ``_host_stage_rows`` one-shot (nothing stages any more, so nothing
                # sets it) and falls through to the device gather -- this step's
                # rows, computed fresh. That chain is why this branch sets a flag
                # rather than just logging: with ``return None`` the veto abstains --
                # correctly, on its own rules, because sending a boot that is not
                # trying anything eager would cost the whole boot to buy nothing --
                # and the boot replays stale for the rest of its life at the price
                # the measured stale regime costs (1.98 acceptance against 2.61 on
                # this engine).
                self._ple_disabled_eager = True
                logger.warning(
                    "PLE HOSTIDS: host staging disabled after %d step(s) -- %s. A "
                    "graph has captured the staged branch, so a replay of these "
                    "steps would serve the PREVIOUS step's pinned rows. "
                    "PLE_DOUBT_UNDER_FULL=eager: every FULL step from here is "
                    "reported as refused and the dispatch veto runs it EAGER on the "
                    "device path -- correct rows, at the cost of taking the WHOLE "
                    "forward out of the graph: 3.1x per step measured on this box "
                    "(15.88 tok/s against 52.05 staged, boot 20261008-083018), which "
                    "is MORE than serving stale rows costs and why \"zero\" is the "
                    "shipped default. "
                    "From this line stale_steps must stop rising and veto= must rise "
                    "with the refusals; if both move, the veto is not running. The "
                    "generation tripwire stays armed. Refusals so far: %s",
                    self._ple_host_steps,
                    why,
                    dict(self._ple_host_refusals),
                )
                self._ple_host_layers = []
                return
            # Do not claim a fallback that a replay cannot perform. The op body
            # stopped running for these steps, so the graph keeps replaying
            # whatever is in the pinned buffer and the generation tripwire is what
            # says so out loud.
            logger.warning(
                "PLE HOSTIDS: host staging disabled after %d step(s) -- %s. A graph "
                "has captured the staged branch, so the device path does NOT handle "
                "these steps: the pinned rows will keep replaying. The generation "
                "tripwire stays armed. Refusals so far: %s",
                self._ple_host_steps,
                why,
                dict(self._ple_host_refusals),
            )
            self._ple_host_layers = []
            return
        logger.warning(
            "PLE HOSTIDS: host staging disabled after %d step(s) -- %s. The "
            "device path handles every step from here on. Refusals so far: %s",
            self._ple_host_steps,
            why,
            dict(self._ple_host_refusals),
        )
        self._ple_host_layers = []

    PLE_HOST_VETO = (
        os.environ.get("VLLM_GFX906_PLE_HOST_VETO", "1").strip().lower() != "off"
    )
    """Dispatch a refused FULL-graph step eager instead of replaying it.

    The one option that is correct *and* up, and the thing
    :attr:`PLE_HOST_STAGE_UNDER_FULL` calls for in its own docstring: until the
    runner refuses to dispatch a FULL descriptor for a step whose rows cannot be
    staged, a refused step under a captured graph serves the PREVIOUS step's PLE
    rows, which is worse than serving none (measured on acceptance length: 4.00
    correct, 3.31 zeroed, 1.98 stale).

    It is cheap because the residual is small and the fallback is not new machinery.
    Only FULL-dispatched refusals can be stale at all -- ``not-pure-decode`` lands on
    PIECEWISE steps and ``no-drafts`` on NONE steps, where the op body already runs
    the device path -- so on the boot that measured this, the veto would have fired
    on 18 of 8667 steps on one worker and 277 on the worst: 0.2 % to 3.2 %. And
    ``CUDAGraphManager.dispatch`` already answers ``CUDAGraphMode.NONE`` for shapes
    it never captured, which is the same descriptor this produces on purpose.

    ``off`` restores today's behaviour exactly, which is what the A/B needs.
    """

    PLE_RESYNC_PER_SLOT = _ple_resync_cap("VLLM_GFX906_PLE_RESYNC_PER_SLOT", 0)
    """Repairs one request slot may get before it is left to the device path.

    **0 = uncapped, which is the default since 2026-10-07.** It used to be 2, on
    the reasoning that a repair costs a device sync, a sync costs milliseconds, and
    a request that needs a second one has a systematic problem not worth paying for
    again. All three halves of that were wrong, and the cap was measured doing the
    opposite of what it was for:

    * Cost. Boot ``vllm.log.20261006-223026``, 64 repairs per worker, 256 samples:
      ``S`` p50 **0.08 ms**, p90 0.10, **max 0.23**. Against a 42.5 ms step that is
      0.19 % *at the worst sample*, and a repair is 0.08 ms end to end because the
      host sits ahead of the device at this point in the path -- the sync lands 40
      us above its 7.9 us idle floor. The row read is flat with context length (45
      us at 32 B, 99 us at 256 KiB), so a 128K request is not the exception either.
    * Demand. The same boot refused **2 497** ``history-length`` steps against
      9 918: under MTP an unaccepted draft leaves ``positions != lengths - 1`` on
      about a quarter of steps, so needing a repair is the steady state, not the
      pathological case the constant was written for.
    * Effect of the cap. Two repairs latched the slot for the life of its request
      (the counter clears only in ``free()``), so a 200-step request served stale
      rows on ~96 % of its steps. Acceptance length there: **1.98**, against 3.31
      with those rows zeroed (boot #37) and 4.00 with them correct (boot #36).

    So this is now an emergency brake, kept and env-tunable because "a repair is
    free" is a claim about *this* box at *this* step size: the same probe measured
    13.2 ms for a ``synchronize()`` with one matmul in flight, which is what a drain
    costs if the host ever falls behind. ``VLLM_GFX906_PLE_RESYNC_PER_SLOT=12``
    reinstate a throttle without editing the file a serving boot runs from. The
    per-call bound in :attr:`PLE_RESYNC_PER_CALL` is what actually limits a step.

    Distrust is not governed here. A mirror that cannot be vouched for is charged
    in :attr:`PLE_DISAGREE_PER_SLOT`, which fires only once
    :meth:`_ple_offset_explains` has failed to explain the mismatch as a shift --
    the right ledger, and the one that was never the problem.
    """

    PLE_RESYNC_TOTAL = _ple_resync_cap("VLLM_GFX906_PLE_RESYNC_TOTAL", 0)
    """Repairs per worker per boot. 0 means uncapped, which is the shipped default.

    Both numbers this replaced were priced against a cost that does not exist, and
    the second one was sized from a measurement that could not have come out any
    other way. 64 assumed a millisecond sync; S is 0.12 ms p50 / 0.23 ms p90 over
    8000 samples. 2000 was sized as "20 000 steps' worth at the observed 25 %
    demand", and that 25 % came from a boot whose demand was suppressed by the 64
    being measured -- a censored figure wearing a measured one's clothes. The real
    demand, from ``vllm.log.20261007-070900`` with nothing capped: 2000 repairs paid
    plus 3441 steps refused for want of a repair, 5441 of 8667 decode steps, 63 %.
    The allowance was gone in twenty minutes of light probe traffic, after which one
    2000-token request ran 766 chunks and accrued 768 stale steps per worker: every
    step of it.

    That is why there is no cap now. A budget that degrades to stale rows when it
    runs out is the silent failure a budget is supposed to prevent, and the thing it
    was protecting against costs 0.12 ms. The bound that carries the risk is
    :attr:`PLE_RESYNC_PER_CALL` -- one sync per step, so the worst case is 0.12 ms x
    one per step = 0.19 % of a 42.5 ms step at any demand. What a cap should have
    been is an instrument, so the demand rate is announced instead of acted on: see
    :attr:`PLE_RESYNC_ALARM_RATE`.
    """

    PLE_RESYNC_ALARM_RATE = 0.25
    """Fraction of steps needing a repair before the arm says so out loud.

    Checked every :attr:`PLE_RESYNC_ALARM_EVERY` repairs, never per step. Crossing it
    is not an emergency and does not change behaviour: at the measured S a repair on
    every step costs 0.54 % of a step. It is the number that has to be fixed to make
    the sync disappear rather than cheaper, so it must be visible without someone
    counting log lines afterwards.
    """

    PLE_RESYNC_ALARM_EVERY = 1024
    """Repairs between demand-rate checks. ~Once a minute at B=1 decode."""

    PLE_RESYNC_PER_CALL = 1
    """Device syncs allowed within one call to :meth:`_stage_host_ple_ids`.

    The one thing that still has to be bounded now the repair itself is not:
    0.08 ms when the host is ahead of the device, 13.2 ms when it is not. Today the
    sole caller reaches the repair at most once -- its retry loop is ``for attempt in
    (0, 1)`` and the second attempt's ``if attempt`` short-circuits before
    :meth:`_ple_resync_mirror` -- so this cannot fire. It is here so that stays a
    property of the code rather than a coincidence of one call site: a future caller
    that resynced per layer would multiply a rounding error into a third of a boot.
    """

    PLE_RESYNC_MIRROR_AHEAD_UNLIMITED = True
    """Mirror-ahead resyncs are not counted against the disagreement budget, because
    after :meth:`_ple_resync_mirror_inner` stopped reading them as disagreements
    there is nothing left for the budget to mean. Kept as a named constant so the
    decision is visible next to :attr:`PLE_DISAGREE_TOTAL`, which is the counter
    boot #28 fired on by mistake.
    """

    PLE_DISAGREE_TOTAL = 4
    """*Distinct* disagreements per worker before the knob goes off for the boot.

    Four, because four different slots disagreeing is not one bad read, and the
    thing a disable is for -- and can only be for -- is a host that cannot vouch
    for its own reads. A single disagreement is a refused step, which under a
    captured graph costs that request the same coverage a disable would, without
    taking every other slot in the boot with it.

    "Distinct" is load-bearing, and the two crashes of 2026-10-05 are why it got
    written down: one stuck slot re-produced the identical comparison on four
    consecutive steps, that was read as four disagreements, and the disable it
    triggered raised through a TP worker and took the engine down. Counting steps
    of one persistent condition as strikes is not a threshold, it is a timer.
    """

    PLE_DISAGREE_PER_SLOT = 2
    """Two on one request is a mirror and a row that are not going to agree, so
    stop paying a device sync to find that out again. "Two" means two *different*
    comparisons; the same comparison re-reported by a stuck slot is measured in
    steps instead -- see :attr:`PLE_DISAGREE_STUCK_STEPS`."""

    PLE_DISAGREE_STUCK_STEPS = 32
    """Consecutive steps one slot may refuse on the SAME comparison before that
    counts as a boot problem.

    A stuck slot re-produces one identical comparison per step, and the two crashes
    of 2026-10-05 happened because those were charged as strikes: four steps, about
    a tenth of a second of decode, and an engine down. Persistence is still a real
    signal -- 32 refusals in a row means this request's mirror is never going to
    agree, and under a captured graph every one of those steps is stale -- so it
    gets a bound. It just has to be measured in steps, on the scale where "the
    mirror is permanently wrong" and "the checker re-asked a question" are different
    numbers, and not on the scale where they are the same number.
    """

    PLE_RESYNC_MAX_OFFSET = 4
    """Largest window shift, in cells, accepted as an *explanation* for a mirror
    that does not match the device row at the positions its own length claims.

    The offset that killed two boots on 2026-10-05 was 3, exactly
    ``num_speculative_tokens``: a dropped commit leaves the window holding real
    tokens that sit slightly elsewhere in the row. Four leaves a cell of margin and
    stays far below the window width, so a window containing something that is not
    a token id at all -- boot #14's ``host=4161103993993705554`` -- cannot be
    excused by wandering the ring looking for a match.
    """

    def _ple_resync_mirror(
        self, req_states: Any, input_batch: InputBatch, detail: str
    ) -> bool:
        """Rebuild the slots whose token count disagrees with the mirror.

        Returns True if every mismatching slot was rebuilt, so the caller may
        retry the build. False leaves the refusal exactly where it was.

        Where the tokens come from. ``req_states.all_token_ids`` is UVA-backed
        (``states.py`` picks that to save GPU memory), so its cells are
        host-*addressable* -- which is the whole trap: prompt cells are host-written
        at add time while generated cells are written by ``_post_update_kernel`` on
        the device, unordered before a host read in ``prepare_inputs``. A read that
        raced was caught on 2026-10-01 returning ``4161103993993705554`` where a
        token id belongs. So this path pays a device sync first: it is the only
        thing that makes the cells being read the cells the device committed. That
        sync is why the repair is capped.

        Why the repair is not self-certifying. Seeding a slot from a row and then
        taking its length from the same row would leave the exact-length check that
        triggered this unable to notice a bad read. So the rebuilt row is
        cross-checked against the cells the *old* window still claims to hold --
        ``num_sampled`` commits on one side, the device's own token buffer on the
        other.

        That comparison is only meaningful between a mirror and a row that describe
        the **same request**, and boot #27 is what not knowing that cost: at
        02:30:55, three seconds before its first real step, it compared a mirror
        holding ``[4, 5, 10]`` against a row holding ``[4, 5, 19959]`` on slot 2 --
        two different requests in one slot, because ``prepare_dummy_inputs`` never
        goes through ``add_request`` and ``invalidate`` clears the mirror but never
        the device's row. So the overlap is compared only when this slot's current
        owner, from the runner's own ``index_to_req_id``, is the request the mirror
        was seeded for. Anything else is recorded as an owner skip and the row is
        taken on trust, which is what a post-sync read of the device's own token
        buffer is for.

        And a disagreement turns this step down; it does not turn the knob off.
        Under a captured graph there is no fallback for the step either way, so
        disabling traded one request for every slot in the boot. The knob goes off
        only once disagreements repeat -- see :meth:`_note_resync_disagreement`.

        Slots only arrive here having already failed the count check, so seeding
        one cannot turn a working mirror into a broken one.
        """
        try:
            return self._ple_resync_mirror_inner(req_states, input_batch, detail)
        except PLEHostDoubtUnderFullGraph:
            # A stop-the-boot decision must not be swallowed here. This handler
            # exists so a failed *repair* cannot crash a step it had already handed
            # to the device -- a much narrower claim than "no party may abort this
            # boot", and the only reason the broad handler below is safe to have.
            raise
        except Exception as exc:  # noqa: BLE001 - a knob must not kill a boot
            # The refusal path is exactly the wrong place to raise: this runs
            # inside an ``except`` that already decided the step is the device's,
            # and a repair that throws would turn a coverage bug into a crash.
            self._ple_resync_refused.add(f"{type(exc).__name__}: {exc}")
            return False

    def _ple_resync_mirror_inner(
        self, req_states: Any, input_batch: InputBatch, detail: str
    ) -> bool:
        hist = self._ple_host_history
        if torch.cuda.is_current_stream_capturing():
            # A sync inside a capture aborts it (hipErrorStreamCaptureUnsupported)
            # and that kills the boot. Nothing is lost by skipping: a capture is
            # not a served step, and replay reads no host arrays.
            self._ple_resync_refused.add("capturing")
            return False
        # 0 means uncapped. See PLE_RESYNC_TOTAL for the two caps this replaced and
        # the stale steps each of them caused.
        if self.PLE_RESYNC_TOTAL and self._ple_resyncs >= self.PLE_RESYNC_TOTAL:
            self._ple_resync_refused.add("total cap")
            return False
        # At most PLE_RESYNC_PER_CALL syncs per step, whichever way this method gets
        # called. Counted on ``_ple_host_calls``, not ``_ple_host_steps``: refused
        # steps do not advance the latter, so two refused steps in a row would
        # otherwise look like one. ``getattr`` for the stub states the tests drive.
        call = int(getattr(self, "_ple_host_calls", 0))
        if call - getattr(self, "_ple_resync_last_call", -1) < self.PLE_RESYNC_PER_CALL:
            self._ple_resync_refused.add("step cap")
            return False

        slots = np.asarray(
            input_batch.idx_mapping_np[: input_batch.num_reqs], dtype=np.int64
        )
        if slots.size == 0:
            return False
        positions = np.asarray(req_states.num_computed_tokens_np, dtype=np.int64)[slots]
        lengths = np.asarray(hist.length, dtype=np.int64)[slots]
        bad = np.nonzero(positions != lengths - 1)[0]
        if bad.size == 0:
            # The refusal described a mismatch this step's mapping does not
            # contain, so there is nothing here to rebuild.
            self._ple_resync_refused.add("no mismatching slot")
            return False
        row_width = int(req_states.all_token_ids.gpu.shape[1])
        # Who owns each slot according to the runner, which is the only party that
        # knows. Nothing here may assume the mirror and the row describe the same
        # request until this says so.
        owners = getattr(req_states, "index_to_req_id", {})
        plan: list[tuple[int, int, int, str | None, bool]] = []
        for i in bad:
            slot = int(slots[i])
            want = int(positions[i]) + 1
            # 0 means uncapped; see PLE_RESYNC_PER_SLOT for what that cost analysis
            # looked like, and why the count below is kept even when nothing is
            # gated on it (it is the only per-slot repair statistic in the log).
            if (
                self.PLE_RESYNC_PER_SLOT
                and self._ple_resync_slots.get(slot, 0) >= self.PLE_RESYNC_PER_SLOT
            ):
                self._ple_resync_refused.add("per-slot cap")
                return False
            if want < 1 or want > row_width:
                self._ple_resync_refused.add("position beyond row")
                return False
            owner = owners.get(slot)
            same_request = owner is not None and owner == self._ple_host_slot_req.get(
                slot
            )
            if not same_request:
                # Either the runner has no request in this slot (a capture or
                # warmup dummy) or the mirror describes a request that has left.
                # Both are normal, and neither is evidence about the device.
                self._ple_resync_owner_skip += 1
            plan.append((slot, want, int(hist.length[slot]), owner, same_request))

        # The one device sync on this path, before any cell is read.
        # ``getattr`` because the tests drive this method with stub states that
        # never run ``__init__``; on a real model state the flag being True implies
        # the sample lists exist too, so only the flag needs the fallback.
        timing = getattr(self, "_ple_resync_timing", False)
        _t = time.perf_counter_ns() if timing else 0
        self._ple_resync_sync()
        self._ple_resyncs += 1
        # Marked after the sync succeeded, so a repair that threw before syncing
        # does not spend this step's allowance.
        self._ple_resync_last_call = call
        if timing:
            self._ple_resync_sync_ms.append((time.perf_counter_ns() - _t) / 1e6)
        rows = req_states.all_token_ids.gpu
        read_samples = self._ple_resync_read_ms if timing else None
        for slot, want, old_len, owner, same_request in plan:
            # Timed per slot, not per resync: the loop can return early on a
            # disagreement, and a sample taken before that return is the one we
            # most want (it is the slot that just cost a sync and paid nothing).
            _tr = time.perf_counter_ns() if timing else 0
            row = rows[slot, :want].to("cpu", non_blocking=False).numpy()
            if read_samples is not None and len(read_samples) < 8192:
                read_samples.append((time.perf_counter_ns() - _tr) / 1e6)
            # The comparison covers only cells BOTH sides claim to hold. ``want``
            # bounds it from the row side, and that bound is the whole bug: ``row``
            # is ``want`` cells long, so with a mirror AHEAD of the device
            # (``old_len > want`` -- a rolled-back speculative step, or a chunk
            # re-issued after a refusal) ``old_len - overlap`` lands past the end of
            # ``row`` and numpy's slicing rules hand back the EMPTY slice. Then
            # ``np.array_equal`` between a length-0 array and a length-``overlap``
            # one is False by construction, so the check manufactured a disagreement
            # on every resync of that shape -- which is what killed boot #27 at
            # 02:30:55 and disabled boot #28 at 03:49:06. Both were read as evidence
            # about the device. Neither was.
            overlap = min(hist.width, old_len, want) if same_request else 0
            if same_request and old_len > want:
                # The mirror holds cells the row does not, so the two describe
                # different positions and no comparison is meaningful. Adopt the
                # row, and count it separately: a boot that does this constantly is
                # asking about commit ordering, which is not the question a
                # disagreement answers.
                self._ple_resync_mirror_ahead += 1
                overlap = 0
            if overlap:
                mirrored = hist.window[slot, hist.width - overlap :]
                row_cells = row[old_len - overlap : old_len]
                if not np.array_equal(row_cells, mirrored):
                    # The window does not match the row at the positions the
                    # mirror's length claims. Before that counts as evidence about
                    # the host's reads, ask whether a SHIFT explains it. The crash
                    # of 2026-10-05 ran on ``positions [97209] != committed-1
                    # [97206]`` -- the device exactly ``num_speculative_tokens``
                    # ahead -- and a window that merely lags matches the row once
                    # the comparison is moved by that offset. A lag is the
                    # dropped-commit coverage bug, already counted in
                    # ``mirror(dropped=)``, and the resync repairs it by seeding
                    # from the row. It is not a host that cannot vouch for its own
                    # reads, and reading it as one stopped the boot twice.
                    shift = self._ple_offset_explains(mirrored, row, old_len)
                    if shift is None:
                        self._note_resync_disagreement(
                            slot, mirrored, row_cells, overlap
                        )
                        return False
                    self._ple_resync_mirror_offset += 1
                    if (
                        self._ple_resync_mirror_offset == 1
                        or self._ple_resync_mirror_offset % 16 == 0
                    ):
                        logger.warning(
                            "PLE HOSTIDS: resync on slot %d found the mirror's "
                            "window matching the device's own row shifted by %d "
                            "cell(s) rather than disagreeing -- a lagged commit, "
                            "not a bad read, so the row is adopted and this is not "
                            "charged to the disagreement budget. %d such resync(s) "
                            "this boot; the two crashes on 2026-10-05 charged "
                            "exactly these as disagreements.",
                            slot,
                            shift,
                            self._ple_resync_mirror_offset,
                        )
            hist.seed(slot, row)
            if owner is not None:
                # The rebuilt window belongs to whoever is in the slot now.
                self._ple_host_slot_req[slot] = owner
            self._ple_resync_slots[slot] = self._ple_resync_slots.get(slot, 0) + 1
        self._ple_resync_demand_alarm()
        if self._ple_resyncs == 1 or self._ple_resyncs % 16 == 0:
            logger.warning(
                "PLE HOSTIDS: resynced %d mirror slot(s) from the device token "
                "row after a count mismatch (%s). Slots so far: %s. One device "
                "sync per repair; budget %s per worker, %s per slot. "
                "S so far: sync %s ms, read %s ms.",
                bad.size,
                detail,
                dict(self._ple_resync_slots),
                self.PLE_RESYNC_TOTAL or "uncapped",
                self.PLE_RESYNC_PER_SLOT or "uncapped",
                _ms_summary(getattr(self, "_ple_resync_sync_ms", ())),
                _ms_summary(getattr(self, "_ple_resync_read_ms", ())),
            )
        return True

    def _ple_resync_demand_alarm(self) -> None:
        """Say out loud when repairs have become the normal case.

        This exists because the cap it replaced was wrong in a specific, repeatable
        way: running out of budget did not stop anything, it converted a coverage bug
        into a quality bug and moved the evidence from ``per-slot cap`` to ``total
        cap`` while the steps kept replaying. A rate is the honest instrument. It is
        reported and NOT acted on, because at the measured cost of a repair there is
        nothing worth stopping: one repair per step is 0.54 % of a step.

        The number this is pointing at is the count check itself. The deltas behind
        these refusals are small and symmetric -- measured on one boot, every single
        one within +/-3 cells, 49 % with the mirror ahead and 51 % behind -- which is
        the shape of ``num_speculative_tokens``, not the shape of lost commits
        (``mirror(dropped=)`` was 4 for the whole boot). The mirror is being advanced
        by a count that is not the device's committed count. Fixing *that* removes
        the sync; capping the sync only refuses to pay for correctness.
        """
        n = self._ple_resyncs
        calls = int(getattr(self, "_ple_host_calls", 0))
        every = self.PLE_RESYNC_ALARM_EVERY
        # The checkpoint AND the rate are windowed over the last
        # ``PLE_RESYNC_ALARM_EVERY`` repairs, not over everything since boot. The
        # life-to-date version of this test shipped on 2026-10-07 and was measured
        # failing on live traffic: one decode run needed 712 repairs across 1 024
        # staging calls (69 % of steps) and the alarm said nothing, because the same
        # boot's cumulative ratio was 1 503 / 11 264 = 13 %. A ratio instrument has to
        # be measured over the interval of the burst it exists to catch; a long quiet
        # boot dilutes a burst inside it forever.
        since_n = n - int(getattr(self, "_ple_resyncs_alarm_at", 0))
        if since_n < every:
            return
        since_calls = max(
            calls - int(getattr(self, "_ple_resyncs_alarm_calls", 0)), 1
        )
        self._ple_resyncs_alarm_at = n
        self._ple_resyncs_alarm_calls = calls
        rate = since_n / since_calls
        if rate < self.PLE_RESYNC_ALARM_RATE:
            return
        logger.warning(
            "PLE HOSTIDS: %d mirror repairs over the last %d staging calls -- %.0f %% "
            "of steps needed one in that window, above the %.0f %% alarm. (Life to date "
            "it is %d/%d = %.0f %%, which is exactly why the window is the unit: the "
            "cumulative ratio is what let a 69 %% burst stay silent.) Nothing is being "
            "capped: at the measured repair cost (%s ms sync + %s ms read) even a "
            "repair on every step is about 0.5 %% of a step. This is a pointer at the "
            "count check, not an emergency -- the mirror is being advanced by a count "
            "that is not the device's committed count, and the fix that removes the "
            "sync is to advance it by the accepted count instead. "
            "See docs/measure-resync-S.md.",
            since_n,
            since_calls,
            100 * rate,
            100 * self.PLE_RESYNC_ALARM_RATE,
            n,
            calls,
            100 * n / max(calls, 1),
            _ms_summary(getattr(self, "_ple_resync_sync_ms", ())),
            _ms_summary(getattr(self, "_ple_resync_read_ms", ())),
        )

    def _ple_offset_explains(
        self, mirrored: np.ndarray, row: np.ndarray, old_len: int
    ) -> int | None:
        """Return a shift that makes ``mirrored`` match ``row``, or None.

        ``mirrored`` is the last ``overlap`` cells the host mirror holds; ``row`` is
        the device's own token row for this request, cells ``0 .. want-1``. The
        mirror claims those cells sit at ``old_len - overlap .. old_len - 1``. This
        asks whether they in fact sit one cell earlier, or two, or three -- in
        either direction -- which is the shape a dropped commit leaves behind.

        Nearest shifts are tried first, and only shifts whose slice lies inside the
        row are considered, so nothing here can read past the committed tokens into
        the uncommitted tail. Returning a shift is therefore a narrow claim: the
        window's contents are real tokens of this request, merely offset. A window
        holding a value that is not a token id anywhere in the row matches at no
        shift, and that is the case the disagreement budget exists for -- boot #14's
        ``host=4161103993993705554`` is exactly that, and still stops the boot.
        """
        overlap = int(mirrored.shape[0])
        if overlap < 2:
            # One cell matches at too many shifts to mean anything.
            return None
        n = int(row.shape[0])
        lo = -(old_len - overlap)
        hi = n - old_len
        for step in range(1, self.PLE_RESYNC_MAX_OFFSET + 1):
            for shift in (-step, step):
                if not lo <= shift <= hi:
                    continue
                start = old_len - overlap + shift
                if np.array_equal(row[start : start + overlap], mirrored):
                    return shift
        return None

    def _note_stale_step(self, refused: bool = True) -> None:
        """Record that THIS step ran, or will run, on the previous step's PLE rows.

        One per step, from either route: a refusal that sends the step down a path
        the captured graph never takes, or a declined resync. Counting per slot would
        make the number mean "how many slots were unhappy" instead of "how many steps
        were stale", and the second is the shipping question.

        ``refused`` says whether this step's staging decision is a REFUSAL (``False``
        from the producer, the one case the dispatch veto acts on) or a DECLINE
        (``None``: policy off, no registered layers, staging disabled, no draft
        source). The veto leaves a declined step at FULL, so it replays, and a
        declined step must be in the stale ledger even with the veto armed. The rule
        is deliberately one-directional: when the fate is not known, the step is
        charged as stale, so the ledger over-reports the defect rather than hiding it.
        """
        if not self._ple_step_full():
            return
        if self._ple_stale_counted_step:
            return
        self._ple_stale_counted_step = True
        if self._ple_veto_armed and refused:
            # A refusal on a runner that has committed to dispatching the step eager
            # is a step that gets its rows from the device path. It is not a stale
            # step and must not be counted as one, or the ledger would report the
            # defect the veto just closed. Refusals are still counted per reason, so
            # "why did this step leave the fast path" stays answerable.
            self._ple_veto_steps += 1
            return
        self._ple_host_stale_steps += 1

    def _note_resync_disagreement(
        self, slot: int, mirrored: np.ndarray, row_cells: np.ndarray, overlap: int
    ) -> None:
        """Record one cross-check failure, and decide whether it is a boot problem.

        Takes the two slices that were compared, not the arrays they came from: the
        row is ``want`` cells long and the window is the full ring, so a helper that
        re-derived the slice from the lengths would print the wrong cells -- and a
        diagnostic that names cells nobody compared is worse than no diagnostic.

        Refuses the step. It does not refuse the boot unless the pattern says the
        host's own reads are the thing that is broken -- and the pattern has to be
        *distinct* disagreements, not the same stuck slot re-reported once per step.
        """
        detail = (
            f"PLE host mirror resync disagreed with the device's own token row on "
            f"slot {slot}: mirrored {mirrored.tolist()} vs row "
            f"{row_cells.tolist()} over the last {overlap} cell(s)."
        )
        # A stuck slot re-detects the identical comparison on every step, and the
        # strikes must not accumulate over it. On 2026-10-05 the same
        # ``(mirrored, row)`` pair was produced four times in a row on one slot, it
        # read as four disagreements, and the disable it triggered raised through a
        # TP worker and took the engine down -- twice. A repeat refuses the step
        # like any other disagreement and keeps its own count.
        sig = (tuple(int(v) for v in mirrored.tolist()), tuple(int(v) for v in row_cells.tolist()))
        if self._ple_disagree_sig.get(slot) == sig:
            # The same comparison again. Not a new strike: on 2026-10-05 the same
            # ``(mirrored, row)`` pair was produced four times in a row on one slot,
            # it read as four disagreements, and the disable it triggered raised
            # through a TP worker and took the engine down -- twice. Persistence is
            # still bounded, just in steps rather than in strikes.
            self._ple_resync_disagree_repeat += 1
            stuck = self._ple_resync_disagree_stuck.get(slot, 0) + 1
            self._ple_resync_disagree_stuck[slot] = stuck
            if stuck >= self.PLE_DISAGREE_STUCK_STEPS:
                self.disable_host_staging(
                    f"{detail} This exact comparison has refused slot {slot} for "
                    f"{stuck} consecutive steps: the mirror for this request is "
                    "never going to agree with the row, and under a captured graph "
                    "every one of those steps replays stale PLE rows."
                )
                return
            if (
                self._ple_resync_disagree_repeat == 1
                or stuck % 16 == 0
            ):
                logger.warning(
                    "PLE HOSTIDS: resync declined slot %d again on the SAME "
                    "comparison -- %s Counted once, not again: %d step(s) this slot "
                    "has been stuck on it, budget %d. One stuck slot is not a "
                    "strike per step; charging it that way crashed the box twice on "
                    "2026-10-05.",
                    slot,
                    detail,
                    stuck,
                    self.PLE_DISAGREE_STUCK_STEPS,
                )
            return
        self._ple_disagree_sig[slot] = sig
        # A different comparison on the same slot is new information, so the stuck
        # streak restarts and the per-slot strike counter moves.
        self._ple_resync_disagree_stuck[slot] = 1
        # The step this decline refuses is a stale step, and nothing else in the boot
        # said so. The refusal path already counted these; the decline path did not,
        # which is why boot #32's three FULL declines -- each of which replayed the
        # previous step's rows -- appeared nowhere in the staleness ledger that
        # decides whether FULL + HOSTIDS=1 is shippable.
        self._note_stale_step()
        self._ple_resync_disagree += 1
        per_slot = self._ple_resync_disagree_slots.get(slot, 0) + 1
        self._ple_resync_disagree_slots[slot] = per_slot
        if (
            self._ple_resync_disagree >= self.PLE_DISAGREE_TOTAL
            or per_slot >= self.PLE_DISAGREE_PER_SLOT
        ):
            self.disable_host_staging(
                f"{detail} Repeated disagreement: the host cannot vouch for its own "
                "reads, so it stops serving ids built from them rather than "
                "repeating this step after step."
            )
            return
        # "The step goes to the device path" was written before E1 and is false on
        # a step dispatching a captured FULL graph, where the op body never runs at
        # replay and there is nothing to fall back to. All four declines in boot #29
        # were FULL steps, so the line described stale output as handled.
        logger.warning(
            "PLE HOSTIDS: resync declined slot %d on a step dispatching cg_mode=%s "
            "-- %s %s Declining rather than turning the knob off: %d/%d distinct "
            "disagreements this boot, %d on this slot.",
            slot,
            self._ple_step_cg_mode,
            detail,
            (
                "There is NO device path for this step: the captured graph replays "
                "the previous step's PLE rows, so this step's PLE is stale."
                if self._ple_step_full()
                else "The step goes to the device path."
            ),
            self._ple_resync_disagree,
            self.PLE_DISAGREE_TOTAL,
            per_slot,
        )

    def _ple_host_layer_steps(self) -> int:
        """Steps on which a layer actually put rows in its pinned buffer.

        Read at the sink, from the layer's own counter, because every other number
        in this file is upstream of the thing that matters. Three boots in a row
        were read from a counter that sat between the build and the layer and so
        kept rising after the layer stopped being called: boot #26's
        ``staged_steps`` never existed, and boot #27's reached 5,120 while this one
        stayed at 1 -- the capture dummy.
        """
        return sum(
            int(getattr(layer, "_host_stage_steps", 0))
            for layer in self._ple_host_gen_layers
        )

    def _ple_host_coverage(self) -> str:
        """What the host path actually did, for the log lines that need a denominator.

        Boot #26 reported 576 refusals per worker and no successes in sight, and
        the number that would have made that line legible -- how many steps
        staged, how many commits went missing and how -- was only printed when the
        knob disabled itself.
        """
        hist = self._ple_host_history
        layers = self._ple_host_gen_layers
        # Two ways this boot is not staging, and they look identical from the
        # outside: the worker turned the knob off (which empties the layer list),
        # or every layer stopped taking steps (its own hostids went off, and
        # ``stage_host_step`` will decline from here on). Reporting the first and
        # reading the second as "on but idle" is how boot #27's 5,120 staged steps
        # got read as a live knob by the person who wrote the counter.
        knob_off = bool(layers) and not self._ple_host_layers
        layers_off = bool(layers) and all(
            getattr(layer, "_hostids_disabled", False) for layer in layers
        )
        return (
            f"staged_steps={self._ple_host_steps} "
            f"stale_steps={self._ple_host_stale_steps} "
            f"veto={self._ple_veto_steps}/{self._ple_veto_applied} "
            f"policy_off={self._ple_host_policy_off} "
            f"refused={sum(self._ple_host_refusals.values())} "
            f"layers_staged={self._ple_host_layer_steps()} "
            f"disabled={'yes' if knob_off or layers_off else 'no'} "
            # What a disabled arm actually serves. Under "eager" the refusals land on
            # veto= and stale_steps stops rising; under "degrade" and "zero" they land
            # on stale_steps, and those two are NOT the same thing -- one replays the
            # previous step's rows, the other replays zeros. A reader who cannot tell
            # them apart from the coverage line cannot tell a fixed boot from a broken
            # one, and scripts/ple-window-state.py routes on this field.
            f"post_disable="
            f"{'zero' if getattr(self, '_ple_disabled_zero', False) else 'eager' if getattr(self, '_ple_disabled_eager', False) else 'replay'} "
            f"mirror(staged={hist.staged} dropped={hist.dropped} "
            f"sliced={hist.sliced} flushes={hist.flushes} pending={hist.pending}) "
            f"resync={self._ple_resyncs} "
            f"mirror_ahead={self._ple_resync_mirror_ahead} "
            f"mirror_offset={self._ple_resync_mirror_offset} "
            f"cg_mode={self._ple_step_cg_mode} "
            f"disagree={self._ple_resync_disagree}/{self.PLE_DISAGREE_TOTAL} "
            f"disagree_repeat={self._ple_resync_disagree_repeat} "
            f"stuck={max(self._ple_resync_disagree_stuck.values(), default=0)} "
            f"owner_skip={self._ple_resync_owner_skip} "
            f"resync_refused={sorted(self._ple_resync_refused)} "
            # The two numbers the correctness decision has been waiting on. S is
            # the sync+read cost of a repair, printed against the step it would be
            # charged to (see the accumulators in __init__). refusal_mix is the
            # end-of-boot reason distribution, which the first-of-kind-only log
            # never gave: boot #36 raised 1,229 refusals and knew the reason for
            # 22 of them, so the sync RATE -- the term that decides whether S is
            # affordable -- was unmeasurable.
            f"S_sync_ms({_ms_summary(getattr(self, '_ple_resync_sync_ms', ()))}) "
            f"S_read_ms({_ms_summary(getattr(self, '_ple_resync_read_ms', ()))}) "
            f"refusal_mix="
            f"{dict(sorted(self._ple_host_refusals.items(), key=lambda kv: -kv[1]))}"
        )

    def log_staging_coverage_at_shutdown(self, reason: str = "worker-shutdown") -> None:
        """Print the FINAL coverage line, once, when the worker is torn down.

        The serving heartbeat is a cadence and a cadence is part of its claim: a
        boot whose traffic never reaches 1024 staged steps prints NOTHING about
        what the arm did, and silence supports both "staged every step" and "boot
        too short to report". Boot ``vllm.log.20261009-161850``'s predecessor is
        the measured case -- a ~300-step boot whose last denominator was the ramp
        line at 128, so the last 172 steps were unaccounted and the numbers a
        reader wanted (final staged, final dropped, final refusal mix) existed
        only in a process that was about to exit.

        This is that final state, in the log, and it is unconditional: it prints
        at ``staged_steps=0`` too, which is the only way an option-A boot (policy
        ``off``, so nothing stages and the serving heartbeat never fires) shows
        its own refusal mix at all.

        Why a real teardown hook and not ``atexit``: this runs in the TP worker
        subprocess, where ``atexit`` callbacks fire after the interpreter has
        begun finalizing -- logger may be gone, module globals may be None, and
        the ordering relative to ``destroy_model_parallel()`` is undefined. The
        caller is :meth:`vllm.v1.worker.gpu.model_runner.GPUModelRunner.shutdown`,
        reached from ``Worker.shutdown()`` in the ``finally`` block of
        ``worker_main`` (multiproc_executor.py), which is how SIGINT and SIGTERM
        both arrive for a worker -- i.e. exactly the operator's
        ``kill -INT -$(pgid)`` restart path. SIGKILL is the one case that cannot
        report, and no in-process hook can.

        Two guards, both load-bearing:

        * idempotent, because ``Worker.shutdown()`` is reachable twice in one
          process (the executor's RPC path and ``worker_main``'s ``finally``), and
          a doubled FINAL line in a boot log is a false second data point;
        * exception-swallowing, because the caller is a teardown path that other
          cleanup depends on: a coverage line must never be the reason a worker
          fails to release its GPU resources. The failure is reported, at warning
          level and without a traceback-hungry re-raise, and teardown continues.
        """
        if getattr(self, "_ple_coverage_logged_at_shutdown", False):
            return
        self._ple_coverage_logged_at_shutdown = True
        try:
            logger.info(
                "PLE HOSTIDS: FINAL coverage at %s. %s",
                reason,
                self._ple_host_coverage(),
            )
        except Exception as exc:  # pragma: no cover - teardown path
            logger.warning(
                "PLE HOSTIDS: could not print the final coverage line (%s: %s); "
                "the counters were dropped rather than risk blocking teardown",
                type(exc).__name__,
                exc,
            )

    def _full_graph_captured_this_boot(self) -> bool:
        """True once a FULL graph has captured the staged branch. Boot-global.

        The op sets ``_staged_branch_captured`` during capture only, so this is
        False for the whole of a piecewise boot.

        It is sound for the question *can a replay of the staged branch happen at
        all in this boot* -- which is what :meth:`disable_host_staging` needs, and
        capture cannot be undone. It is NOT a per-step predicate, and using it as
        one is what made boot #28's 51 ``not-pure-decode`` refusals look like 51
        stale steps: under ``FULL_AND_PIECEWISE`` (the v1 default, and what #28
        ran) this returns True for every step after the first capture, including
        the PIECEWISE ones, where the device PLE path runs normally and a refusal
        merely costs an optimisation. Per-step questions go to
        :meth:`_ple_step_full`.
        """
        if self._ple_host_captured:
            return True
        if any(
            getattr(layer, "_staged_branch_captured", False)
            for layer in self._ple_host_gen_layers
        ):
            self._ple_host_captured = True
        return self._ple_host_captured

    def _ple_step_full(self) -> bool:
        """Is the step being prepared one that replays a FULL decode graph?"""
        if self._ple_step_cg_mode is None:
            # Nothing threaded a descriptor (a caller outside the runner's step
            # loop). Fall back to the boot-global answer and claim the cliff rather
            # than miss it: this fallback can over-report stale steps, which is
            # visible and fixable, where under-reporting them is neither.
            return self._full_graph_captured_this_boot()
        # ``prepare_inputs`` stores ``cudagraph_mode.name``, the string "FULL", so
        # the plain comparison is the one production hits. It normalises anyway: an
        # enum stored here instead would answer False, and the failure would be
        # silent -- the stale-step ledger and the doubt policy would both go quiet
        # while a captured graph froze PLE rows. That is the same silent-False
        # hazard A2 was written against.
        return (
            getattr(self._ple_step_cg_mode, "name", self._ple_step_cg_mode)
            == "FULL"
        )

    def _note_host_refusal(
        self,
        reason: str,
        detail: str = "",
        *,
        refused: bool = True,
        ledger: bool = True,
    ) -> None:
        """Record one staging refusal, and say in the log what happens to the step.

        ``refused=False`` marks a DECLINE (the producer returns ``None``, so the
        dispatch veto does not act and the step keeps its FULL descriptor). It changes
        two things below: the sentence, and which ledger the step lands in.

        ``ledger=False`` marks an event that is not about this step's fate at all. The
        epilogue mirror commit (``stage_host_tokens``) runs AFTER the step has already
        been dispatched and run, so its failures belong to the mirror's bookkeeping:
        they are counted per reason and logged, but they enter NEITHER step ledger.
        Without the flag they landed in ``veto_steps``, because they are noted while
        ``_ple_step_cg_mode`` still reads ``FULL`` from this step's own producer call,
        and a boot whose veto never fired printed ``veto=12/1`` as though it had.
        """
        self._ple_host_refusals[reason] = self._ple_host_refusals.get(reason, 0) + 1
        if reason not in self._ple_host_refusal_logged:
            self._ple_host_refusal_logged.add(reason)
            # ``detail`` is what the exception knew and the reason token does not:
            # for ``history-length`` it is ``positions [...] != committed-1 [...]``,
            # which is the difference between knowing the mirror is short and
            # knowing by how much and for which slot. Boot #26 raised 411 of these
            # and could not answer that question because only the reason was
            # logged. Once per reason, not per step: the detail carries per-step
            # arrays and a serving log should not repeat them.
            logger.info(
                "PLE HOSTIDS: first refusal (%s) on a step dispatching cg_mode=%s%s; "
                "%s. Counts so far: %s. Coverage: %s",
                reason,
                self._ple_step_cg_mode,
                f" -- {detail}" if detail else "",
                (
                    # This line is what a reader trusts when nothing else is on the
                    # box, and until 2026-10-07 it asserted a replay in every armed
                    # boot while the coverage field 60 characters later read
                    # ``stale_steps=0``: the same line contradicting itself. With the
                    # veto armed and this a real refusal the step does NOT replay.
                    "this step already ran; its accepted tokens were not queued for "
                    "the host mirror, so the mirror is short by this step and the "
                    "next one pays a repair to catch up"
                    if not ledger
                    else "the dispatch veto runs this step EAGER (cg_mode=NONE) and its "
                    "PLE comes from the device path"
                    if self._ple_veto_armed and refused and self._ple_step_full()
                    else "the pinned rows were zeroed when the arm was disabled, so "
                    "this step replays ZEROS and PLE contributes nothing to it"
                    if self._ple_step_full() and getattr(self, "_ple_disabled_zero", False)
                    else "no device path exists for this step, so the captured graph "
                    "replays the previous PLE rows"
                    if self._ple_step_full()
                    else "device path used for that step"
                ),
                dict(self._ple_host_refusals),
                self._ple_host_coverage(),
            )
        if not ledger:
            # Not a step-fate event, whatever mode this step ran in: it dispatched and
            # ran already. Charge nothing, but keep a per-reason cadence so a flood of
            # dropped commits is still visible in a serving log -- boot #26 raised 67
            # per worker with nothing in the log to distinguish them.
            dropped = self._ple_host_refusals[reason]
            if dropped == 1 or dropped % 32 == 0:
                logger.warning(
                    "PLE HOSTIDS: mirror commit dropped (%s)%s -- this step already "
                    "dispatched and ran, so neither ledger is charged; the mirror is "
                    "short by this step and the next one pays a repair. dropped=%d so "
                    "far. Counts: %s. Coverage: %s",
                    reason,
                    f" -- {detail}" if detail else "",
                    dropped,
                    dict(self._ple_host_refusals),
                    self._ple_host_coverage(),
                )
            return
        if not self._ple_step_full():
            return
        # E1 Phase 1, and the sharpest edge on shipping FULL. "Device path used
        # for that step" is FALSE under a captured graph: the op body does not run
        # at replay, so a refused step replays the PREVIOUS step's pinned rows.
        # Plausible text is the worst output a lookup table can produce, and the
        # generation tripwire cannot see a sustained refusal (nothing advances the
        # counter, so nothing looks stale). So the count is the thing that decides
        # whether FULL + HOSTIDS=1 is shippable as a default or needs the runner to
        # pick a non-FULL descriptor for unstaged steps: if a FULL boot leaves this
        # at zero, the cliff was never reached; if it is non-zero, FULL needs that
        # fallback and this line is the evidence for it.
        self._note_stale_step(refused=refused)
        # Which ledger this step landed in decides what the sentence may claim. With
        # the dispatch veto armed the runner takes this step eager and the device
        # path computes its rows, so printing "there is NO device path ... ran on
        # stale PLE" would be a lie in the log, and a log that lies about the one
        # thing a mechanism was built to prevent is worse than no log. The count also
        # has to come from the same ledger: gating on ``stale_steps`` while the veto
        # keeps it at zero would evaluate ``0 % 32 == 0`` and log per step.
        vetoed = self._ple_veto_armed and refused
        count = (
            self._ple_veto_steps if vetoed else self._ple_host_stale_steps
        )
        if count == 1 or count % 32 == 0:
            logger.warning(
                "PLE HOSTIDS: refusal (%s) on a step dispatching cg_mode=FULL -- %s "
                "%s=%d so far. Counts: %s. Coverage: %s",
                reason,
                (
                    "the dispatch veto runs it EAGER on the device path, so its PLE "
                    "is computed for this step rather than replayed;"
                    if vetoed
                    else "a captured graph holds the staged PLE branch and there is "
                    "NO device path, so the graph replays the previous step's "
                    "embedding rows and this step ran on stale PLE;"
                ),
                "veto" if vetoed else "stale",
                count,
                dict(self._ple_host_refusals),
                self._ple_host_coverage(),
            )

    def _refresh_ple_query_start_loc(self, input_batch: InputBatch) -> torch.Tensor:
        """Make the whole static PLE boundary array describe THIS step.

        Why the whole buffer and not the padded prefix. A captured full-decode graph
        indexes this array with the request width it was **captured** at, not the
        width of the step being replayed, so refreshing only
        ``[: num_reqs_padded + 1]`` leaves the tail holding some older, wider step's
        boundaries. Boot #13 caught the consequence: a one-request, four-token step
        whose ngram fill read an array beginning ``[0, 2, 2, ...]`` -- the padded form
        of an earlier two-token step, whose tail is ``num_tokens`` by the runner's own
        convention (``input_batch.py:159``) -- and therefore scattered positions 2 and
        3 into **row 2** of the workspace instead of row 0. The scratch pins how narrow
        that is: of the non-decreasing arrays the fill could have held, only four-entry
        ones starting ``[0, 2, 2]`` reproduce the captured workspace cell for cell
        (entry 3 is unconstrained by the observation; the runner would have written
        ``2`` there). A three-entry ``[0, 2, 2]`` does not -- the first analysis said
        that until ``request_indices.clamp_`` was included in the reproduction.

        The runner already keeps its own copy unambiguous for every width by
        writing ``num_tokens`` into the tail, so the fix is to hand over that whole
        array instead of a prefix of it. When the source is a short
        ``[num_reqs + 1]`` view -- adaptive verification rebuilds
        ``query_start_loc`` at ``model_runner.py:1521`` -- the remaining cells get
        the same ``num_tokens`` the runner would have written there, rather than
        whatever the previous step left.

        Cost is the same one copy that call already was: this buffer is
        ``max_num_reqs + 1`` int32 (16 bytes at the capture width this box runs at),
        and the short-source fill is a host-side branch that does not fire when the
        runner handed over its full buffer.

        Returns the view forward is handed -- the same ``[: num_reqs_padded + 1]``
        prefix as before, from the same slice, so the differential's
        ``consumed_qsl`` capture and the graph still describe one object.
        """
        src = input_batch.query_start_loc
        dst = self.ple_query_start_loc
        copied = min(src.shape[0], dst.shape[0])
        dst[:copied].copy_(src[:copied])
        if copied < dst.shape[0]:
            dst[copied:].fill_(input_batch.num_tokens)
        return dst[: input_batch.num_reqs_after_padding + 1]

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        cudagraph_mode: CUDAGraphMode | None = None,
    ) -> dict[str, Any]:
        """``cudagraph_mode`` is the runner's ``batch_desc.cg_mode`` for THIS step.

        The runner knows it ~170 lines before this method runs and already hands
        ``batch_desc`` to its own ``prepare_inputs``, so threading it costs one
        argument, and :meth:`prepare_attn` on the same interface has taken the same
        value all along. It is what lets a staleness claim name the graph mode it is
        talking about instead of inferring one from a boot-global flag.
        """
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        # The dispatch veto asks for the PLE half of this step's inputs early --
        # before prepare_attn -- and leaves the result here under the identity of
        # this exact batch.
        prepared = self._take_prepared_ple_inputs(input_batch)
        if prepared is not None:
            model_inputs.update(prepared)
        else:
            extra, decision = self._ple_step_inputs(
                input_batch, req_states, cudagraph_mode
            )
            self._ple_step_rows = decision
            model_inputs.update(extra)
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
