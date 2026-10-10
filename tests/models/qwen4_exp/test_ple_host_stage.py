# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-staged PLE ids: the producer, the consumer, and the differential gate.

E1 Phase 1 (P1.3 + P1.4). A captured FULL decode graph replays the H2D out of
``_pinned_output`` and contains neither the blocking device->host copy of the
n-gram ids nor the host gather that produced them, so the host has to fill that
buffer itself, every step, from ids it obtained without asking the device. These
tests cover the three new behaviours that claim makes true:

* ``stage_host_step`` -- compute ids on the host, gather, publish the row count;
* the custom op skipping its D2H and gather when the row count agrees;
* ``compare_host_ids`` -- differential mode, the only thing that can validate the
  position arithmetic against the device's own answer.

Construction note: a real ``Qwen4ExpNGramEmbedding`` wants a model config, a
VllmConfig and mmap'd shards totalling tens of GiB, and a real
``Qwen4ExpModelState`` wants an engine. Both are exercised here as unbound
methods against a stand-in ``self`` that carries exactly the attributes the
methods touch. That is not a shortcut around the code under test -- the method
bodies are the real ones (assigned from the class), so a change in them shows up
here; it only removes the machinery that cannot go wrong in a different way.
Every attribute the fake carries is one the production ``__init__`` sets.
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import re
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.models.qwen4_exp.amd import model_state as model_state_module
from vllm.models.qwen4_exp.amd import ple_hostids
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd.ple_hostids import HostTokenHistory
from vllm.models.qwen4_exp.amd.ple_layer import (  # noqa: F401
    Qwen4ExpNGramEmbedding,
    qwen4_exp_amd_ple_ngram_embedding,
)

NGRAM_SIZE = 3
HEADS_PER_NGRAM = 1
HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM  # 2
EMBED_DIM = 8
HEAD_DIM = EMBED_DIM // HEADS  # 4
ROWS = 6
CAPACITY = 8
SENTINEL = -12345
EOS = 0
CTX = NGRAM_SIZE - 1


def _layout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fixed, plausible hash layout: 3 multipliers, one range per head."""
    multipliers = np.array([-12345, 67891, -3333], dtype=np.int64)
    sizes = np.array([1000003, 2000003], dtype=np.int64)
    offsets = (np.cumsum(sizes) - sizes).astype(np.int64)
    return multipliers, sizes, offsets


class FakeEvent:
    """Stand-in for ``torch.cuda.Event`` whose readiness the test owns.

    ``synchronize`` raises on purpose: "the tripwire never waits" is then asserted
    by every test that uses the event, not only by the one written to check it.
    """

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.records = 0
        self.queries = 0

    def record(self) -> None:
        self.records += 1

    def query(self) -> bool:
        self.queries += 1
        return self.ready

    def synchronize(self) -> None:  # pragma: no cover - must never run
        raise AssertionError("the generation tripwire synchronised on the device")


class StagingEmbedding:
    """Stand-in ``self`` for the real staging methods of the embedding module.

    ``ngram_embedding`` is a recording gather so "did the host gather run" is
    observable, and the pinned buffers are ordinary (unpinned) CPU tensors with
    the production shapes and dtypes; pinning is a DMA property and nothing here
    does DMA.
    """

    # The real implementations, not copies.
    _host_hash_buffers = Qwen4ExpNGramEmbedding._host_hash_buffers
    _stage_step_impl = Qwen4ExpNGramEmbedding.stage_host_step
    take_host_stage = Qwen4ExpNGramEmbedding.take_host_stage

    #: Whether a staged step also puts its generation where the op would have put
    #: it. See :meth:`stage_host_step` below.
    _mirror_op_gen = True

    def stage_host_step(self, *args: Any, **kwargs: Any) -> bool:
        """The real staging method, plus the one thing only the custom op does.

        The staged branch of the op copies this step's generation into the device
        scratch on every staged step, eager and replayed alike
        (``ple_layer.py:2806``), and the generation tripwire reads that scratch back
        later. A test that drives ``stage_host_step`` never runs the op, so the
        scratch would stay at zero and the tripwire would correctly report a handoff
        that never landed -- a true statement about the fake, arriving as a layer
        disabled in the middle of a test about something else. Boot #27 is what that
        looks like from the outside: a ``staged_steps`` counter that climbed while
        the layer underneath it had stopped taking steps.

        A test that wants the wire to report something -- stale, ahead, in flight --
        says so itself with :func:`_replay`, which turns this mirror off.
        """
        accepted = self._stage_step_impl(*args, **kwargs)
        if accepted and self._mirror_op_gen and self._gen_device is not None:
            self._gen_device.copy_(self._pinned_gen)
        return accepted
    compare_host_ids = Qwen4ExpNGramEmbedding.compare_host_ids
    # The dispatch veto's invalidation. Borrowed rather than restated because the
    # whole point of the hazard test below is that the flag the VETO leaves behind is
    # the one the real op would have consumed, so the clearing has to be the real
    # clearing.
    forget_host_stage = Qwen4ExpNGramEmbedding.forget_host_stage
    note_draft_provenance = Qwen4ExpNGramEmbedding.note_draft_provenance
    _id_dump = Qwen4ExpNGramEmbedding._id_dump
    disable_hostids = Qwen4ExpNGramEmbedding.disable_hostids
    _alloc_gen_scratch = Qwen4ExpNGramEmbedding._alloc_gen_scratch
    _gen_scratch = Qwen4ExpNGramEmbedding._gen_scratch
    check_host_gen = Qwen4ExpNGramEmbedding.check_host_gen
    _fire_gen = Qwen4ExpNGramEmbedding._fire_gen
    _inputs_agree = Qwen4ExpNGramEmbedding._inputs_agree
    stage_inputs_agree = Qwen4ExpNGramEmbedding.stage_inputs_agree
    _note_stage_refusal = Qwen4ExpNGramEmbedding._note_stage_refusal
    _ctx_leg = Qwen4ExpNGramEmbedding._ctx_leg
    _packed_dump = Qwen4ExpNGramEmbedding._packed_dump
    _consumed_report = Qwen4ExpNGramEmbedding._consumed_report
    _note_input_refusal = Qwen4ExpNGramEmbedding._note_input_refusal

    def __init__(self, rows: int = ROWS, capacity: int = CAPACITY) -> None:
        multipliers, sizes, offsets = _layout()
        self.lookup_calls = 0
        self.ngram_size = NGRAM_SIZE
        self.heads_per_ngram = HEADS_PER_NGRAM
        self.eos_token_id = EOS
        self.layer_name = "ple"
        self.runtime_dtype = torch.float16
        self.layer_multipliers = torch.from_numpy(multipliers)
        self.ngram_heads_vocab_sizes = torch.from_numpy(sizes)
        self.ngram_heads_offsets = torch.from_numpy(offsets)
        self._pinned_ngram_ids = torch.full(
            (capacity, HEADS), SENTINEL, dtype=torch.long
        )
        self._pinned_output = torch.zeros(
            (capacity, EMBED_DIM), dtype=self.runtime_dtype
        )
        self._host_stage_rows = -1
        self._host_check_rows = -1
        self._host_check_ids = None
        self._host_hash_cache = None
        self._hostids_disabled = False
        self._host_stage_steps = 0
        self._host_id_mismatches = 0
        self._host_id_matches = 0
        self._host_check_inputs: tuple[Any, ...] | None = None
        # The compute's own packed workspace, as the real layer holds it: written
        # only by _compute_ngram_ids, which P1.12 reads at dump time.
        self.padded_buffer = torch.full((4, 64), EOS, dtype=torch.long)
        self._host_check_draft: tuple[Any, Any, int] | None = None
        self._host_check_dev_inputs: Any = None
        # P1.14: the qsl view the forward is handed. The stand-in has to carry it
        # because the per-step report reads it, and a stand-in whose log lines
        # cannot be rendered is a stand-in whose tests are not testing the line.
        self._host_consumed_qsl: Any = None
        self._rows = rows
        # Generation tripwire state, mirroring the production __init__. The event
        # is a FakeEvent so `query()` is controllable and a synchronize() is a
        # test failure rather than a wait.
        self._host_stage_gen = 0
        self._pinned_gen = torch.zeros(1, dtype=torch.long)
        self._pinned_gen_read = torch.zeros(1, dtype=torch.long)
        self._gen_device: torch.Tensor | None = torch.zeros(1, dtype=torch.long)
        self._gen_event = FakeEvent()
        self._gen_pending_expected = 0
        self._gen_check_countdown = 1
        self._gen_stuck = 0
        self._gen_checks_ok = 0
        self._gen_trips = 0
        self._staged_branch_captured = False
        # Input-agreement gate (P1.10b) counters, mirroring the production __init__.
        # Nothing else from P1.9 survives: the op-entry stamps and the input snapshot
        # were both observers of a region the AOT cache decides how to run.
        self._host_input_checks = 0
        self._host_input_skips = 0
        self._host_stage_refusals = 0
        # P1.11: steps where the device window was never handed over, so the third
        # leg stayed dark. Mirrors the production __init__ for the same reason the
        # other counters do -- the gate prints it, so a stand-in without it is a
        # stand-in whose log lines cannot be rendered.
        self._host_ctx_unchecked = 0

    @property
    def ngram_embedding(self):  # noqa: ANN201
        def _gather(ids: torch.Tensor) -> torch.Tensor:
            self.lookup_calls += 1
            # Deterministic function of the ids, so a wrong buffer is visible.
            out = torch.zeros((ids.shape[0], HEAD_DIM, HEADS), dtype=torch.float32)
            out[...] = ids.abs().sum(dim=-1).reshape(-1, 1, 1)
            return out

        return _gather


def _step_inputs(num_reqs: int = 2, q: int = 3) -> tuple:
    """Pure-decode step: ``q`` rows per request, all query_len == q."""
    input_ids = np.arange(num_reqs * q, dtype=np.int64) + 100
    query_start_loc = np.arange(0, num_reqs * q + 1, q, dtype=np.int64)
    ngram_context = np.tile(
        (np.arange(num_reqs, dtype=np.int64) + 7).reshape(-1, 1), (1, HEADS)
    )
    return input_ids, query_start_loc, ngram_context


def _expected_ids(input_ids, query_start_loc, ngram_context) -> np.ndarray:
    multipliers, sizes, offsets = _layout()
    return ple_hostids.compute_ngram_ids_host(
        input_ids,
        query_start_loc,
        ngram_context,
        multipliers=multipliers,
        vocab_sizes=sizes,
        head_offsets=offsets,
        eos_token_id=EOS,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
    )


# --------------------------------------------------------------------------
# stage_host_step: the producer's per-layer half
# --------------------------------------------------------------------------


def test_stage_host_step_fills_ids_and_gathered_rows() -> None:
    emb = StagingEmbedding()
    inp, qsl, ctx = _step_inputs()
    assert emb.stage_host_step(inp, qsl, ctx, gather=True) is True

    ids = _expected_ids(inp, qsl, ctx)
    assert emb._host_stage_rows == ROWS
    # The ids really went into the buffer the graph's H2D does not read but the
    # gather does -- proving the pinned write happened, not just the flag.
    assert np.array_equal(emb._pinned_ngram_ids[:ROWS].numpy(), ids)
    assert emb.lookup_calls == 1, "the host gather did not run"
    # Row content is the gather's output in the runtime dtype, per row.
    expect = (
        torch.from_numpy(ids)
        .abs()
        .sum(dim=-1, keepdim=True)
        .expand(-1, EMBED_DIM)
        .to(torch.float16)
    )
    assert torch.equal(emb._pinned_output[:ROWS], expect)
    assert emb._host_stage_steps == 1


def test_stage_host_step_check_mode_leaves_the_device_path_untouched() -> None:
    emb = StagingEmbedding()
    inp, qsl, ctx = _step_inputs()
    assert emb.stage_host_step(inp, qsl, ctx, gather=False) is True

    assert emb._host_check_rows == ROWS
    assert emb._host_stage_rows == -1, "check mode must not arm the skip"
    assert emb.lookup_calls == 0, "check mode must not gather"
    assert torch.all(emb._pinned_ngram_ids == SENTINEL), (
        "check mode wrote the device's id buffer"
    )
    assert torch.all(emb._pinned_output == 0)
    assert emb._host_check_ids is not None
    assert np.array_equal(
        emb._host_check_ids[:ROWS].numpy(), _expected_ids(inp, qsl, ctx)
    )


def test_disabled_embedding_stages_nothing() -> None:
    emb = StagingEmbedding()
    emb._hostids_disabled = True
    inp, qsl, ctx = _step_inputs()
    assert emb.stage_host_step(inp, qsl, ctx, gather=True) is False
    assert emb._host_stage_rows == -1
    assert emb.lookup_calls == 0


def test_more_rows_than_the_pinned_buffer_refuses() -> None:
    """A padded batch cannot be staged: the buffer is max_num_batched_tokens."""
    emb = StagingEmbedding(capacity=4)
    inp, qsl, ctx = _step_inputs(num_reqs=4, q=3)  # 12 rows > 4
    assert emb.stage_host_step(inp, qsl, ctx, gather=True) is False
    assert emb._host_stage_rows == -1
    assert emb.lookup_calls == 0, "refused late: it gathered anyway"


def test_hash_buffers_are_read_once() -> None:
    """A per-step ``.cpu()`` would be a synchronising D2H on the hot path."""
    emb = StagingEmbedding()
    first = emb._host_hash_buffers()
    assert emb._host_hash_cache is not None
    assert emb._host_hash_buffers() is first, "snapshot is not cached"
    # And invalidation (what load_weights does) makes it re-read.
    emb._host_hash_cache = None
    second = emb._host_hash_buffers()
    assert second is not first
    assert np.array_equal(second[0], first[0])


def test_load_weights_drops_the_hash_cache() -> None:
    """The cache must not outlive a reload of the weights it read.

    Checked against source because load_weights needs real checkpoint tensors:
    what can rot is the *absence* of the invalidation, and a missing line is
    exactly what a source assertion is good at.
    """
    src = (
        pathlib.Path(__file__).resolve().parents[3]
        / "vllm/models/qwen4_exp/amd/ple_layer.py"
    ).read_text()
    assert "def load_weights" in src
    after = src[src.index("def load_weights") :]
    assert "self._host_hash_cache = None" in after, (
        "load_weights no longer drops the host hash-cache snapshot"
    )


# --------------------------------------------------------------------------
# take_host_stage / compare_host_ids: the rules the op relies on
# --------------------------------------------------------------------------


def test_take_host_stage_requires_exact_row_agreement() -> None:
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    assert emb.take_host_stage(ROWS) is True
    # Consumed: a second step must not inherit the previous step's answer.
    assert emb._host_stage_rows == -1
    assert emb.take_host_stage(ROWS) is False

    emb._host_stage_rows = ROWS
    assert emb.take_host_stage(ROWS - 1) is False, (
        "row count disagreed and the flag was still trusted"
    )
    assert emb._host_stage_rows == -1, "a stale flag survived"


def test_compare_disables_on_the_first_mismatch() -> None:
    emb = StagingEmbedding()
    ids = np.full((ROWS, HEADS), 42, dtype=np.int64)
    emb._host_check_ids = torch.from_numpy(ids.copy())
    emb._host_check_rows = ROWS
    device = torch.from_numpy(np.full((ROWS, HEADS), 43, dtype=np.int64))
    # Input agreement (P1.10b): these tests hand the compare an ids buffer directly,
    # so they have to state the tokens the compute read as well. Without that the
    # gate refuses, which is what it exists to do, and the test would be asserting
    # nothing about mismatch handling.
    _agree_inputs(emb)

    emb.compare_host_ids(device, emb._host_check_dev_inputs)

    assert emb._hostids_disabled is True
    assert emb._host_id_mismatches == 1
    assert emb._host_check_rows == -1, "the check was not consumed"


def test_compare_passes_and_is_consumed_on_agreement() -> None:
    emb = StagingEmbedding()
    ids = np.full((ROWS, HEADS), 42, dtype=np.int64)
    emb._host_check_ids = torch.from_numpy(ids.copy())
    emb._host_check_rows = ROWS
    # Input agreement (P1.10b), as in the mismatch test above.
    _agree_inputs(emb)
    emb.compare_host_ids(torch.from_numpy(ids.copy()), emb._host_check_dev_inputs)
    assert emb._host_input_checks == 1
    assert emb._hostids_disabled is False
    assert emb._host_id_mismatches == 0
    assert emb._host_check_rows == -1


def test_compare_ignores_a_row_count_disagreement() -> None:
    """A check left over from a step that never reached the op is not a mismatch.

    Row counts differ legitimately: the op runs per forward with the padded-or-
    real token count of that forward, and the producer staged for a different
    step. Declaring that a mismatch would disable the knob on the first padded
    batch, which is a much more common event than an arithmetic error.
    """
    emb = StagingEmbedding()
    emb._host_check_ids = torch.zeros((CAPACITY, HEADS), dtype=torch.long)
    emb._host_check_rows = ROWS
    emb.compare_host_ids(torch.zeros((ROWS - 1, HEADS), dtype=torch.long))
    assert emb._hostids_disabled is False
    assert emb._host_id_mismatches == 0
    assert emb._host_check_rows == -1
    # The row-count gate runs first, so the input gate never speaks and no inputs are
    # even read: a step the host did not build costs the harness nothing.
    assert emb._host_input_skips == 0
    assert emb._host_input_checks == 0


def test_disable_is_logged_once_and_idempotent() -> None:
    emb = StagingEmbedding()
    emb.disable_hostids("first reason")
    emb.disable_hostids("second reason")
    assert emb._hostids_disabled is True


# --------------------------------------------------------------------------
# the custom op: skip the D2H, keep the H2D
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _op_env(emb, monkeypatch, *, capturing: bool = False):
    layer = ple_layer_module.Qwen4ExpPLELayer.__new__(ple_layer_module.Qwen4ExpPLELayer)
    layer.ple_embedding = emb  # type: ignore[attr-defined]
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    monkeypatch.delenv("VLLM_GFX906_PLE_CAPTURE_PROBE", raising=False)
    monkeypatch.delenv("VLLM_GFX906_PLE_PROF", raising=False)
    real = torch.cuda.is_current_stream_capturing
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    try:
        yield emb
    finally:
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", real)


def _run_op(n: int = ROWS) -> torch.Tensor:
    ids = torch.arange(n * HEADS, dtype=torch.long).reshape(n, HEADS)
    output = torch.full((n, EMBED_DIM), float("nan"), dtype=torch.float16)
    qwen4_exp_amd_ple_ngram_embedding(ids, output, "ple")
    return output


def test_op_skips_d2h_and_gather_when_the_host_staged_this_step(monkeypatch):
    emb = StagingEmbedding()
    staged = torch.full((ROWS, EMBED_DIM), 3.5, dtype=torch.float16)
    emb._pinned_output[:ROWS] = staged
    emb._host_stage_rows = ROWS

    with _op_env(emb, monkeypatch) as emb:
        out = _run_op()

    assert emb.lookup_calls == 0, "the host gather ran although the host staged"
    # No D2H: the id buffer still holds the sentinel it was built with, so the
    # device's ids never landed. This is the wait the whole change removes.
    assert torch.all(emb._pinned_ngram_ids == SENTINEL), "the D2H ran"
    assert torch.equal(out, staged), "the H2D did not deliver the staged rows"
    assert emb._host_stage_rows == -1


def test_op_runs_the_device_path_when_row_counts_disagree(monkeypatch):
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS - 1

    with _op_env(emb, monkeypatch) as emb:
        out = _run_op()

    assert emb.lookup_calls == 1, "the device path was skipped on a disagreement"
    assert not torch.all(out == 0)
    assert emb._host_stage_rows == -1, "a rejected flag survived to the next step"


def test_a_rejected_flag_does_not_skip_the_next_step(monkeypatch):
    """The failure mode being guarded: stage once, skip forever."""
    emb = StagingEmbedding()
    emb._host_stage_rows = 1  # staged for a different step than the op sees

    with _op_env(emb, monkeypatch) as emb:
        _run_op()
        _run_op()

    assert emb.lookup_calls == 2, "a stale flag skipped a later step's gather"


def test_op_compares_after_the_d2h_in_check_mode(monkeypatch):
    emb = StagingEmbedding()
    ids = torch.arange(ROWS * HEADS, dtype=torch.long).reshape(ROWS, HEADS)
    emb._host_check_ids = ids.clone()
    emb._host_check_rows = ROWS
    # The op hands the compare whatever ``_host_check_dev_inputs`` holds, so the
    # agreeing case has to be staged for this test to be about the comparison rather
    # than about the refusal (which the tests below cover on their own).
    _agree_inputs(emb)

    with _op_env(emb, monkeypatch) as emb:
        _run_op()

    assert emb.lookup_calls == 1, "check mode must not change what runs"
    assert emb._hostids_disabled is False, "a matching check disabled the knob"
    assert emb._host_id_mismatches == 0
    assert emb._host_input_checks == 1, "the op never got as far as the comparison"


def test_op_disables_after_a_check_mismatch(monkeypatch):
    emb = StagingEmbedding()
    emb._host_check_ids = torch.full((ROWS, HEADS), -7, dtype=torch.long)
    emb._host_check_rows = ROWS
    # Input agreement (P1.10b): see the two compare tests above.
    _agree_inputs(emb)

    with _op_env(emb, monkeypatch) as emb:
        _run_op()

    assert emb.lookup_calls == 1, "the device path must still produce the answer"
    assert emb._hostids_disabled is True
    assert emb._host_id_mismatches == 1


def test_host_stage_wins_over_check_mode(monkeypatch):
    """Mode 1 stages and skips; nothing was compared, so nothing can disable."""
    emb = StagingEmbedding()
    emb._pinned_output[:ROWS] = 2.0
    emb._host_stage_rows = ROWS
    emb._host_check_rows = ROWS
    emb._host_check_ids = torch.zeros((ROWS, HEADS), dtype=torch.long)

    with _op_env(emb, monkeypatch) as emb:
        _run_op()

    assert emb.lookup_calls == 0
    assert emb._hostids_disabled is False, "the skipped path compared nothing"


# --------------------------------------------------------------------------
# the producer in Qwen4ExpModelState: what decides coverage
# --------------------------------------------------------------------------


class FakeModelState:
    """Stand-in ``self`` for the real producer methods."""

    _stage_host_ple_ids = model_state_module.Qwen4ExpModelState._stage_host_ple_ids
    set_host_draft_source = model_state_module.Qwen4ExpModelState.set_host_draft_source
    stage_host_tokens = model_state_module.Qwen4ExpModelState.stage_host_tokens
    disable_host_staging = model_state_module.Qwen4ExpModelState.disable_host_staging
    _note_host_refusal = model_state_module.Qwen4ExpModelState._note_host_refusal
    _full_graph_captured_this_boot = (
        model_state_module.Qwen4ExpModelState._full_graph_captured_this_boot
    )
    _ple_step_full = model_state_module.Qwen4ExpModelState._ple_step_full
    _ple_host_coverage = model_state_module.Qwen4ExpModelState._ple_host_coverage
    # The teardown emitter is borrowed like the rest: the stub is a stand-in self,
    # and the FINAL coverage line has to be reachable through it or the suite
    # cannot test what a worker prints on its way out.
    log_staging_coverage_at_shutdown = (
        model_state_module.Qwen4ExpModelState.log_staging_coverage_at_shutdown
    )
    _ple_resync_mirror = model_state_module.Qwen4ExpModelState._ple_resync_mirror
    _ple_resync_mirror_inner = (
        model_state_module.Qwen4ExpModelState._ple_resync_mirror_inner
    )
    _note_resync_disagreement = (
        model_state_module.Qwen4ExpModelState._note_resync_disagreement
    )
    # F1: the offset test is part of the resync path, so the stub has to carry it
    # for the same reason it carries the resync itself.
    _ple_offset_explains = model_state_module.Qwen4ExpModelState._ple_offset_explains
    # The demand alarm runs inside the repair's own try/except, which turns any
    # exception into a refused repair and records the exception text in
    # ``resync_refused``. That is deliberate in production (a repair must never be
    # able to break a boot) and it is exactly why the stub has to borrow the alarm
    # rather than omit it: without it every repair in this file fails with
    # ``AttributeError: no attribute '_ple_resync_demand_alarm'`` attributed to the
    # repair, which is a confusing way to discover a missing borrow and was.
    _ple_resync_demand_alarm = (
        model_state_module.Qwen4ExpModelState._ple_resync_demand_alarm
    )
    # The veto path builds the step's model inputs for real, because that is the
    # saving it exists to make (prepare_inputs must not redo the work), so the stub
    # needs the refresh the producer hands to the stager. ``_prepare_ngram_context``
    # is the one method deliberately NOT borrowed -- see its stub below.
    _refresh_ple_query_start_loc = (
        model_state_module.Qwen4ExpModelState._refresh_ple_query_start_loc
    )
    # Dispatch veto, borrowed for the same reason as the rest: the ledger split in
    # ``_note_stale_step`` and the descriptor decision in ``ple_veto_descriptor`` are
    # production behaviour the tests below assert on.
    ple_veto_descriptor = model_state_module.Qwen4ExpModelState.ple_veto_descriptor
    ple_prepare_step = model_state_module.Qwen4ExpModelState.ple_prepare_step
    _ple_step_inputs = model_state_module.Qwen4ExpModelState._ple_step_inputs
    ple_rows_refused = model_state_module.Qwen4ExpModelState.ple_rows_refused
    _take_prepared_ple_inputs = (
        model_state_module.Qwen4ExpModelState._take_prepared_ple_inputs
    )
    arm_ple_dispatch_veto = model_state_module.Qwen4ExpModelState.arm_ple_dispatch_veto
    note_ple_veto_applied = model_state_module.Qwen4ExpModelState.note_ple_veto_applied
    # Part of the staging path itself now, so it has to be borrowed for the same
    # reason the staging method is: without it every test in this file fails inside
    # ``_stage_host_ple_ids`` on a missing attribute.
    _forget_host_stage_flags = (
        model_state_module.Qwen4ExpModelState._forget_host_stage_flags
    )
    _note_stale_step = model_state_module.Qwen4ExpModelState._note_stale_step
    # The disabled-arm heartbeat is part of the staging path (it runs inside the
    # disabled branch of ``_stage_host_ple_ids``), so it is borrowed for the same
    # reason ``_note_stale_step`` is: omitting it would make every post-disable test
    # here die on an AttributeError instead of asserting anything.
    _note_disabled_heartbeat = (
        model_state_module.Qwen4ExpModelState._note_disabled_heartbeat
    )
    _policy_off_now = model_state_module.Qwen4ExpModelState._policy_off_now
    _ple_host_layer_steps = model_state_module.Qwen4ExpModelState._ple_host_layer_steps
    # The budgets are read off the real class rather than restated here, so a
    # cap changed in production changes what these tests assert. The two resync
    # budgets are the exception and they are the exception on purpose: a cap
    # MECHANISM cannot be tested against "uncapped", which is what both ship as now
    # (per-slot 0, total 0 -- see test_repairs_are_uncapped_by_default for the
    # measurements that removed them). Mechanism tests set a small budget on the
    # instance; the shipped-value test is what pins them.
    PLE_RESYNC_PER_SLOT = model_state_module.Qwen4ExpModelState.PLE_RESYNC_PER_SLOT
    PLE_RESYNC_TOTAL = model_state_module.Qwen4ExpModelState.PLE_RESYNC_TOTAL
    PLE_RESYNC_PER_CALL = model_state_module.Qwen4ExpModelState.PLE_RESYNC_PER_CALL
    PLE_RESYNC_ALARM_RATE = model_state_module.Qwen4ExpModelState.PLE_RESYNC_ALARM_RATE
    PLE_RESYNC_ALARM_EVERY = (
        model_state_module.Qwen4ExpModelState.PLE_RESYNC_ALARM_EVERY
    )
    PLE_HOST_VETO = model_state_module.Qwen4ExpModelState.PLE_HOST_VETO
    PLE_DISAGREE_TOTAL = model_state_module.Qwen4ExpModelState.PLE_DISAGREE_TOTAL
    PLE_DISAGREE_PER_SLOT = model_state_module.Qwen4ExpModelState.PLE_DISAGREE_PER_SLOT
    PLE_RESYNC_MAX_OFFSET = model_state_module.Qwen4ExpModelState.PLE_RESYNC_MAX_OFFSET
    PLE_DISAGREE_STUCK_STEPS = (
        model_state_module.Qwen4ExpModelState.PLE_DISAGREE_STUCK_STEPS
    )
    PLE_DOUBT_UNDER_FULL = model_state_module.Qwen4ExpModelState.PLE_DOUBT_UNDER_FULL
    # Option A, read off the real class for the same reason the budgets are: a test
    # that restates the default keeps passing after the default changes.
    PLE_HOST_STAGE_UNDER_FULL = (
        model_state_module.Qwen4ExpModelState.PLE_HOST_STAGE_UNDER_FULL
    )

    def __init__(self, slots: int = 8) -> None:
        self.uses_ngram_embedding = True
        self.ngram_context_len = CTX
        self.ngram_eos_token_id = EOS
        self._host_drafts = None
        self._ple_host_layers: list | None = None
        self._ple_host_gen_layers: list = []
        self._ple_host_refusals: dict[str, int] = {}
        self._ple_host_refusal_logged: set[str] = set()
        self._ple_host_steps = 0
        # E1 Phase 1: the captured-graph refusal tracking the real class does. The
        # stub borrows ``_note_host_refusal``, so it has to carry the state that
        # method reads -- which is the point of borrowing rather than copying.
        self._ple_host_captured = False
        self._ple_host_stale_steps = 0
        self._ple_host_policy_off = 0
        self._ple_policy_off_logged = False
        self._ple_stale_counted_step = False
        self._last_draft_gen = -1
        self._ple_host_tokens = None
        # The real class builds this in __init__ and seeds it at add time, so the
        # stub has to as well: an unseeded window is a refused step, and these
        # tests are about which steps get through.
        self._ple_host_history = HostTokenHistory(slots, CTX + 1)
        for slot in range(slots):
            self._ple_host_history.seed(slot, _slot_tokens(slot)[: _POSITION + 1])
        # Repair state, because ``_stage_host_ple_ids`` now reaches for it on a
        # count mismatch. The stub has no device, so the sync is a counter: the
        # repair's correctness does not depend on it, only its ordering does, and
        # the resync tests assert that it was called.
        self._ple_resyncs = 0
        self._ple_resyncs_alarm_at = 0
        self._ple_resyncs_alarm_calls = 0
        self._ple_resync_disagree = 0
        self._ple_resync_refused: set[str] = set()
        self._ple_resync_slots: dict[int, int] = {}
        # Boot #27's lesson, carried as state: a cross-check is only meaningful
        # against the request that owns the slot, so the stub has to know owners,
        # and it has to match the mapping ``_batch_and_states`` puts in
        # ``req_states.index_to_req_id``.
        self._ple_host_slot_req = {slot: _req_id(slot) for slot in range(slots)}
        self._ple_resync_disagree_slots: dict[int, int] = {}
        self._ple_resync_owner_skip = 0
        # A1: mirror-ahead resyncs are their own condition now, not disagreements.
        self._ple_resync_mirror_ahead = 0
        # F1/F2 (the 2026-10-05 crashes): the offset-explained count, the repeat
        # count, and the per-slot signature that turns a stuck slot into one event
        # instead of a strike per step. NOTE for whoever adds state to the real
        # class next: this stub hand-copies production attributes, so new state on
        # ``Qwen4ExpModelState`` is invisible here until it is added, and the
        # failure it produces is an ``AttributeError`` from deep inside
        # ``_ple_host_coverage`` rather than anything that names the missing field.
        self._ple_resync_mirror_offset = 0
        self._ple_resync_disagree_repeat = 0
        self._ple_disagree_sig: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        self._ple_resync_disagree_stuck: dict[int, int] = {}
        # A2: the per-step graph mode the runner threads in. ``None`` means "no
        # descriptor", which makes ``_ple_step_full`` fall back to the boot-global
        # flag -- so every pre-A2 test that sets ``_ple_host_captured`` keeps
        # asserting what it asserted before. Per-step tests set this to "FULL".
        self._ple_step_cg_mode = None
        # Dispatch veto (2026-10-07). The real class sets these in __init__; the stub
        # has to carry them too or ``_ple_host_coverage`` raises AttributeError from
        # inside a log-format call, which is the worst possible place to discover a
        # missing field. ``_ple_veto_armed`` is what ``_note_stale_step`` reads to
        # decide which ledger a FULL refusal belongs in, and ``_ple_step_rows`` is
        # this step's staging decision (True staged / False refused / None not
        # attempted) -- the third state is what keeps "arm is off" from reading as
        # "every step refused".
        self._ple_veto_armed = False
        self._ple_veto_steps = 0
        self._ple_veto_applied = 0
        # Choice #3 (2026-10-08): the post-disable route latch and its heartbeat
        # marker. Borrowed behaviour reads both, so the stub has to carry both --
        # ``disable_host_staging`` writes the first under the "eager" policy and
        # ``_note_disabled_heartbeat`` compares against the second.
        self._ple_disabled_eager = False
        # The "zero" policy's latch (same day, chosen over eager after eager measured
        # 3.1x). Borrowed behaviour reads it with getattr, but the stub carries it
        # anyway: a stub that is missing an initial field cannot tell a test that the
        # real class forgot to initialise it from one where the test forgot to set it.
        self._ple_disabled_zero = False
        self._ple_disabled_hb_at = 0
        self._ple_step_rows: bool | None = None
        self._ple_prepared = None
        # The buffer :meth:`_refresh_ple_query_start_loc` writes into, which the veto
        # path needs because it prepares the step's inputs the way production does.
        # Width is the capture-padded width the real one is allocated at; the stub's
        # batches are far smaller.
        self.ple_query_start_loc = torch.zeros(16, dtype=torch.int32)
        self.ngram_context_calls = 0
        self._ple_syncs = 0
        self._ple_resync_sync = self._count_sync
        self.slots = slots

    def _prepare_ngram_context(self, input_batch, req_states):
        """Sentinel, counted.

        The only method in this file the stub does not borrow, and the only one whose
        body could not change a result here: it fills a device tensor the tests never
        read back, and borrowing it would drag in ``ngram_context``,
        ``ngram_context_offsets`` and a GPU ``all_token_ids`` for assertions about
        which STEPS staged. The counter is what the veto tests use, to show the early
        hook did the preparation once rather than leaving it for ``prepare_inputs``.
        """
        self.ngram_context_calls += 1
        return "ngram-context"

    def begin_step(self, cg_mode=None) -> None:
        """Stand in for the per-step state ``prepare_inputs`` sets.

        Production calls ``prepare_inputs`` once per step: that is what records the
        graph mode and clears the one-per-step stale latch. A test that drives a
        refusal path directly skips it, and then two refusals on two different steps
        is indistinguishable from two refusals on one step -- which is exactly the
        distinction ``stale_steps=`` is supposed to make. So cross the boundary
        explicitly rather than poking the private flag.
        """
        self._ple_step_cg_mode = cg_mode
        self._ple_stale_counted_step = False

    def _count_sync(self) -> None:
        self._ple_syncs += 1

    def staged_rows(self) -> list[int]:
        return [layer._host_stage_rows for layer in self._ple_host_layers or []]


class FakeDrafts:
    def __init__(self, values: np.ndarray, row_idx: np.ndarray) -> None:
        self._result = (values, row_idx, 1)

    def wait_host_draft_tokens(self):
        return self._result

    def bump_generation(self, gen: int) -> None:
        """Simulate an epilogue that staged a fresh block."""
        values, row_idx, _ = self._result
        self._result = (values.copy(), row_idx.copy(), gen)


def _req_id(slot: int) -> str:
    """The request id a slot carries, shared by the stub's owner map and the batch.

    The resync cross-check compares a mirror against a device row, and that is
    only a question when both describe the same request -- so the fixture has to
    name requests, not just slots.
    """
    return f"req-{slot}"


def _slot_tokens(slot: int, width: int = 64) -> np.ndarray:
    """The token history a slot carries, shared by the batch and the stub."""
    return np.arange(width, dtype=np.int64) + 1000 * (slot + 1)


_POSITION = 5  # every request in _batch_and_states sits here


def _step_device_ids(
    slots, tokens: np.ndarray, num_reqs: int, q: int, position: int | None = None
) -> list[int]:
    """The runner's ``input_ids`` for the step ``_batch_and_states`` describes.

    Row 0 of each request is the newest committed token from the host mirror; the
    rest are the staged drafts. Same construction the tests spell out by hand, in
    one place so the fixture and the assertions cannot drift apart.
    """
    position = _POSITION if position is None else position
    drafts = _drafts(num_reqs, q).wait_host_draft_tokens()[0]
    ids: list[int] = []
    for order, slot in enumerate(slots):
        ids.append(int(tokens[slot, position]))
        ids.extend(int(v) for v in drafts[order])
    return ids


def _advance_device_ids(
    input_batch,
    st: FakeModelState,
    *,
    num_reqs: int = 2,
    q: int = 3,
    row: np.ndarray | None = None,
    computed: np.ndarray | None = None,
) -> None:
    """Set the fixture's device ``input_ids`` to what the runner holds for this step.

    Row 0 is the mirror's newest committed token and the rest are the staged
    drafts -- read from the state rather than recomputed from ``_slot_tokens``,
    because after a commit the mirror is the ground truth for row 0 and the raw
    token grid is not (``_commit`` lands values like 7 and 11, which are not in the
    grid's synthetic id space).

    In production the two move together: the post-update kernel advances
    ``num_computed`` and the next step's ids start from the newly committed token.
    A test that bumps only the counter describes a device that advanced and a
    runner that did not, and the mode-1 input gate now refuses exactly that -- so a
    test that simulates a commit landing has to advance the ids too, or it is
    testing the refusal path while its name claims it tests the commit path. The
    refusal itself is what ``test_mode_one_refuses_a_device_array_that_lags`` is
    for. One exception, in
    ``test_a_lost_commit_is_repaired_not_poisoning``: there the
    device really has advanced past the mirror, and this helper makes the two
    agree; the mismatch that test simulates is caught by the history-length check,
    which runs before the gate, so the difference never reaches the assertion.
    """
    hist = st._ple_host_history
    # Drain first: the build drains inside ``_stage_host_ple_ids``, and row 0 is
    # whatever that drain appends, so a helper that has to know the ids BEFORE the
    # build has to do the same drain. It is idempotent -- landed commits are popped
    # once -- so the build's own drain then finds nothing pending and the step
    # proceeds exactly as it would have.
    hist.drain()
    drafts = _drafts(num_reqs, q).wait_host_draft_tokens()[0]
    ids: list[int] = []
    for order, slot in enumerate([0, 3][:num_reqs]):
        # ``-1``, not ``length - 1``: the window is a rolling tail of
        # ``ngram_context_len + 1`` cells while ``length`` counts every token the
        # request has accepted. The newest cell is the last one, which is what the
        # builder reads as row 0.
        if row is not None:
            # ... unless the test is describing a device that moved and a mirror
            # that did not hear about it. Then row 0 is the token sitting at this
            # step's position in the device's own row; taking it from the mirror
            # would describe a device that agrees with the mirror the test just
            # made wrong, and the repair would trip the input gate on the
            # fixture rather than on anything real.
            ids.append(int(row[slot, int(computed[slot])]))
        else:
            ids.append(int(hist.window[slot][-1]))
        ids.extend(int(v) for v in drafts[order])
    input_batch.input_ids = torch.tensor(
        ids + [0] * (16 - len(ids)), dtype=torch.int32
    )


def _step_qsl(num_reqs: int, q: int) -> list[int]:
    """``query_start_loc`` for that step, padded the way the runner pads it."""
    num_tokens = num_reqs * q
    qsl = list(range(0, num_tokens + 1, q))
    return qsl + [num_tokens] * (4 - len(qsl))


def _batch_and_states(num_reqs: int = 2, q: int = 3, *, uva: bool = True):
    """A pure-decode batch over slots 0 and 3, with a UVA-shaped host mirror.

    The mirror is deliberately still here after P1.6: the id path no longer reads
    it, and one test below asserts that a batch with no host view at all now
    stages instead of disabling itself.
    """
    slots = [0, 3][:num_reqs]
    idx_mapping = np.array(slots, dtype=np.int64)
    scheduled = np.array([q] * num_reqs, dtype=np.int32)
    computed = np.zeros(8, dtype=np.int64)
    computed[slots] = _POSITION  # context is available for every row
    tokens = np.zeros((8, 64), dtype=np.int64)
    for slot in slots:
        tokens[slot, :] = _slot_tokens(slot)
    # Production exposes the same cells twice: ``.gpu`` is the view the device
    # writes and ``._uva_buf.np`` the host alias of it. The resync reads ``.gpu``,
    # which is also the only view a non-UVA StagedWriteTensor has, so the stub
    # carries both in both configurations.
    gpu = torch.from_numpy(tokens)
    all_token_ids = (
        SimpleNamespace(_uva_buf=SimpleNamespace(np=tokens), gpu=gpu)
        if uva
        else SimpleNamespace(_uva_buf=None, gpu=gpu)
    )
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        # The two padding fields the veto path needs: ``prepare_inputs`` builds the
        # step's PLE inputs through the real refresh, which writes the whole buffer
        # and hands forward a ``[:num_reqs_after_padding + 1]`` view. These batches
        # are not padded, so padded == real.
        num_reqs_after_padding=num_reqs,
        num_tokens=num_reqs * q,
        idx_mapping_np=idx_mapping,
        num_scheduled_tokens=scheduled,
        # The runner's own device arrays, padded to capacity. The differential
        # dump reads these (not anything forward() stashed) because torch.compile
        # runs that body at trace time only.
        #
        # Filled to agree with the host mirror, because a healthy step is what
        # this fixture claims to describe. They used to be all zeros, which was
        # harmless while only the differential read them: its gate refused and the
        # tests asserted on staging, never on the verdict. Mode 1 now asks the same
        # question before it serves, so zeros would turn every mode-1 test into a
        # test of the refusal path -- and a fixture whose device arrays disagree
        # with the batch it names tests the wrong thing. The gate's refusal on a
        # mismatch is asserted directly, by corrupting these, below.
        input_ids=torch.tensor(
            _step_device_ids(slots, tokens, num_reqs, q) + [0] * (16 - num_reqs * q),
            dtype=torch.int32,
        ),
        query_start_loc=torch.tensor(_step_qsl(num_reqs, q), dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens_np=computed,
        all_token_ids=all_token_ids,
        # The runner's own slot -> request map, which is the only authority on who
        # is in a slot. The resync's cross-check reads it before comparing a
        # mirror against a row.
        index_to_req_id={slot: _req_id(slot) for slot in slots},
    )
    return input_batch, req_states, tokens


def _drafts(num_reqs: int = 2, q: int = 3) -> FakeDrafts:
    slots = np.array([0, 3][:num_reqs], dtype=np.int64)
    values = np.stack(
        [
            np.arange(1001 + 1000 * s + 1, 1001 + 1000 * s + q, dtype=np.int64)
            for s in slots
        ]
    )
    return FakeDrafts(values, slots)


@pytest.fixture
def hostids_env(monkeypatch):
    """Force a mode and an empty registry; return setters for both."""
    state: dict[str, Any] = {"mode": 1, "layers": []}
    monkeypatch.setattr(ple_hostids, "hostids_mode", lambda: state["mode"])
    monkeypatch.setattr(
        ple_layer_module, "host_stageable_layers", lambda: list(state["layers"])
    )
    return state


def test_mode_zero_does_nothing(hostids_env) -> None:
    hostids_env["mode"] = 0
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_layers is None
    assert st._ple_host_refusals == {}
    assert st._ple_host_steps == 0


def test_model_without_ple_does_nothing(hostids_env) -> None:
    st = FakeModelState()
    st.uses_ngram_embedding = False
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_layers is None
    assert st._ple_host_steps == 0


def test_no_registered_layers_is_reported(hostids_env) -> None:
    """The knob is set but no embedding asked for it -- not a silent no-op."""
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert "no-registered-layers" in st._ple_host_refusals


def test_no_draft_source_is_reported(hostids_env) -> None:
    hostids_env["layers"] = [StagingEmbedding()]
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert "no-draft-source" in st._ple_host_refusals
    assert st._ple_host_steps == 0


def test_host_stage_source_is_wired_once(hostids_env) -> None:
    st = FakeModelState()
    handler = FakeDrafts(np.zeros((1, 1), dtype=np.int64), np.array([0]))
    st.set_host_draft_source(handler)
    assert st._host_drafts is handler


def test_id_path_survives_a_platform_with_no_host_view(hostids_env) -> None:
    """P1.6: the ids no longer depend on a host view of ``all_token_ids``.

    Before the token mirror existed, a ``req_states.all_token_ids`` without a
    ``_uva_buf`` meant the host could never know a step's history, so staging
    disabled itself permanently. That is exactly the buffer whose newest cells are
    written by a GPU kernel unordered before the host read, so depending on it was
    both fragile and unnecessary. The prefetcher still disables itself on such a
    platform -- that path reads the device array by design.
    """
    hostids_env["layers"] = [StagingEmbedding()]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states(uva=False)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert st._ple_host_steps == 1
    assert st.staged_rows() == [ROWS]


def test_pure_decode_batch_reaches_the_layer(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()
    # A decoy with the *batch-aligned* shape the same-named array has on
    # InputBatch: reading that one instead of req_states' slot-aligned array is a
    # plausible mistake and must not be invisible.
    input_batch.num_computed_tokens_np = np.array([5, 0], dtype=np.int64)

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert st.staged_rows() == [ROWS]
    assert emb.lookup_calls == 1
    assert st._ple_host_steps == 1
    # Not just "something was staged": the ids must be the ones this batch's
    # slots, positions and drafts imply. Row 0 of each request is the newest
    # committed token, from the host token mirror at the slot's computed position;
    # the rest are the staged drafts.
    expected_rows = []
    staged_drafts = _drafts().wait_host_draft_tokens()[0]
    for order, slot in enumerate((0, 3)):
        row = [int(tokens[slot, 5])]
        row += [int(v) for v in staged_drafts[order]]
        expected_rows.append(row)
    expected_input_ids = np.array(
        [v for row in expected_rows for v in row], dtype=np.int64
    )
    expected = _expected_ids(
        expected_input_ids,
        np.array([0, 3, 6], dtype=np.int64),
        # The n-gram context the builder reads out of the same mirror: the CTX
        # tokens just before each row's position.
        np.array(
            [
                [tokens[0, 5 - CTX + i] for i in range(CTX)],
                [tokens[3, 5 - CTX + i] for i in range(CTX)],
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(emb._pinned_ngram_ids[:ROWS].numpy(), expected)


def test_mode_one_refuses_a_device_array_that_lags(hostids_env) -> None:
    """The gate boot #14 said the serving path needed.

    Mode 1 used to stage whatever the mirror handed it. Boot #14 (differential
    mode, same builder, same mirror) caught the mirror returning
    ``host=4161103993993705554`` where a token id belongs -- 8 refused comparisons
    across 4 workers in 45 minutes. Mode 2 refused those steps and the device served
    them; mode 1 would have gathered rows out of a nonexistent id and handed the
    model a plausible answer built from garbage.

    So: when the runner's arrays and the mirror do not describe the same tokens,
    nothing stages. The layer's flag stays down, the op falls through to the device
    path, and the refusal is counted. Staging nothing is the whole safety argument;
    asserting it is the point of the test.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()
    # Row 0 of each request holds the token BEFORE the newest committed one: the
    # shape of a step whose sides disagree, whether the mirror ran ahead or the
    # runner's array is stale.
    input_batch.input_ids[0] = int(tokens[0, _POSITION - 1])
    input_batch.input_ids[3] = int(tokens[3, _POSITION - 1])

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {"stage-inputs-disagree": 1}
    assert st._ple_host_steps == 0
    assert st.staged_rows() == [-1], "a refused step armed the D2H skip"
    # -1 is the layer's "nothing staged" sentinel: ``take_host_stage`` will not
    # answer yes, so the op falls through to the device path for this step.
    assert emb.lookup_calls == 0
    assert emb._host_stage_refusals == 1


def test_mode_one_refuses_a_draft_column_the_device_does_not_have(hostids_env) -> None:
    """A draft position is as load-bearing as row 0.

    The staged rows are [newest committed, draft, draft]; a wrong draft column
    gathers embedding rows for a token the device never fed to its own embedding
    lookup. The gate covers the whole id vector, not just the boundary the builder
    derived its positions from -- which is the same reach the differential's gate
    has, and the reason both now share one implementation.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    input_batch.input_ids[1] = 987654  # a draft token the device does not have

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {"stage-inputs-disagree": 1}
    assert emb.lookup_calls == 0
    assert emb._host_stage_refusals == 1


def test_mode_one_refuses_a_question_it_cannot_ask(hostids_env) -> None:
    """``None`` is answered like ``False``. The rule the docstring singles out.

    ``_inputs_agree`` answers three ways: True (the two sides describe the same
    tokens), False (they do not), and None (the question could not be asked --
    inputs missing, device tensors incomplete, or reading them raised). For a
    COMPARISON None and False both mean "no verdict", which is why the difference
    went unnoticed while this was a differential-only gate. For STAGING they are not
    interchangeable: False is a step the data condemns, and None is a step the
    harness cannot see -- a lost buffer, a tensor that will not copy back. That is
    exactly the step you must not vouch for, and refusing costs one blocking D2H.

    This test is the only thing that pins it. The red control
    ``unaskable-means-stage`` -- rewriting ``is True`` as ``is not False`` in
    ``stage_inputs_agree`` -- measured STILL-GREEN on 2026-10-09: every other test
    here reaches the gate holding a real verdict, so none of them notices what the
    gate does when there is none.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    # The host half is never read on these three routes -- each returns before the
    # comparison -- so its contents do not matter. Lengths match the fake batch.
    host = (np.zeros(4, dtype=np.int64), np.zeros(2, dtype=np.int64), None)

    # The stager had no runner arrays to hand over.
    assert emb.stage_inputs_agree(host, None) is False
    # A triple arrived but its tensors did not.
    assert emb.stage_inputs_agree(host, (None, None, None)) is False

    class Unreadable:
        """A device tensor that refuses to be copied back."""

        def detach(self):
            raise RuntimeError("device tensor is gone")

    assert emb.stage_inputs_agree(host, (Unreadable(), Unreadable(), None)) is False

    # Three unaskable questions, three refusals, all on the staging counter: a step
    # the serving path declined is reported as declined, not as an unjudged
    # comparison. See test_a_staging_refusal_is_worded_as_a_refusal_to_stage.
    assert emb._host_stage_refusals == 3
    assert emb._host_input_skips == 0


def test_a_staging_refusal_is_worded_as_a_refusal_to_stage(
    hostids_env, caplog
) -> None:
    """Two different claims, so two different counters and two different verbs.

    ``SKIPPED a comparison`` means the device served the step and the harness
    declined to judge it. ``REFUSED to stage`` means the serving path declined the
    step and the device served it. The first is a hole in coverage; the second is a
    statement about served output. Blurring them is how a diagnostic starts lying,
    and this file has a counter for each because of it.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    input_batch.input_ids[0] = 999

    st._stage_host_ple_ids(input_batch, req_states)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "REFUSED to stage" in text, text
    assert "SKIPPED a comparison" not in text, text
    assert "input_ids differ at 0 of 6" in text, text
    assert emb._host_input_skips == 0, "a staging refusal counted as a comparison skip"
    assert emb._host_stage_refusals == 1


def test_the_comparison_gate_still_counts_comparisons(hostids_env) -> None:
    """The shared implementation must not merge the two counters.

    ``_inputs_agree`` now serves both paths, so a refactor that routed every
    refusal to one counter would keep every other assertion in this file green while
    quietly deleting the distinction the boot reads. A mode-2 refusal still lands on
    ``_host_input_skips`` and leaves the staging counter alone.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    assert emb._host_input_skips == 1, "the input-agreement gate did not fire"
    assert emb._host_stage_refusals == 0, "a comparison skip counted as a refusal"


def test_the_stage_gate_stays_out_of_graph_capture(hostids_env, monkeypatch) -> None:
    """During a capture there is no step to attribute, and a D2H would kill the boot.

    The gate reads three device arrays synchronously. Inside a stream capture that
    is ``hipErrorStreamCaptureUnsupported``: the capture aborts and the engine dies
    at startup -- the same class of failure that ended boot #26 at config time, one
    layer down. And it would protect nothing: replay never reads the host arrays,
    and the rows a capture leaves behind are zeroed on purpose
    (``test_ple_capture_stage``). So the gate asks the capture question first.
    """
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    input_batch.input_ids[0] = 999  # would refuse if the gate were consulted

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert emb.lookup_calls == 1
    assert emb._host_stage_refusals == 0


def test_mode_two_stages_for_comparison_only(hostids_env) -> None:
    hostids_env["mode"] = 2
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert emb._host_check_rows == ROWS
    assert emb._host_stage_rows == -1, "differential mode armed the D2H skip"
    assert emb.lookup_calls == 0, "differential mode gathered"
    assert st._ple_host_steps == 1


def test_mixed_batch_refuses_and_the_layer_is_untouched(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    # One request is a chunked prefill chunk: not a decode step at all.
    input_batch.num_scheduled_tokens = np.array([3, 12], dtype=np.int32)

    st._stage_host_ple_ids(input_batch, req_states)

    assert "not-pure-decode" in st._ple_host_refusals
    assert emb._host_stage_rows == -1
    assert emb.lookup_calls == 0
    assert st._ple_host_steps == 0


def test_missing_draft_row_refuses(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    # The mirror describes a different batch than the one being prepared.
    st.set_host_draft_source(
        FakeDrafts(
            np.arange(2 * 2, dtype=np.int64).reshape(2, 2),
            np.array([0, 1], dtype=np.int64),
        )
    )
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert "row-missing" in st._ple_host_refusals
    assert emb._host_stage_rows == -1


def test_unexpected_failure_disables_and_survives(hostids_env) -> None:
    """A knob must never be able to kill a step, let alone a boot."""

    class Exploding:
        def wait_host_draft_tokens(self):
            raise RuntimeError("mirror is broken")

    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(Exploding())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)  # must not raise

    assert st._ple_host_layers == []
    assert emb._host_stage_rows == -1


def test_layer_failure_disables_midflight(hostids_env) -> None:
    class ExplodingLayer(StagingEmbedding):
        def stage_host_step(self, *a, **kw):  # noqa: ANN002, ANN003
            raise RuntimeError("gather blew up")

    good = StagingEmbedding()
    hostids_env["layers"] = [good, ExplodingLayer()]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_layers == []
    assert st._ple_host_steps == 0, "a partial stage was counted as a success"


def test_refusals_are_counted_and_logged_once(hostids_env, caplog) -> None:
    st = FakeModelState()
    with caplog.at_level("INFO"):
        st._note_host_refusal("not-pure-decode")
        st._note_host_refusal("not-pure-decode")
        st._note_host_refusal("no-drafts")
    assert st._ple_host_refusals["not-pure-decode"] == 2
    assert st._ple_host_refusals["no-drafts"] == 1
    logged = [r.getMessage() for r in caplog.records if "HOSTIDS" in r.getMessage()]
    assert len(logged) == 2, "a repeated refusal reason logged repeatedly"


def test_a_refusal_under_a_captured_graph_denies_the_device_path(
    hostids_env, caplog
) -> None:
    """Under FULL, "device path used for that step" is a lie -- so say the truth.

    This count is the shipping gate for ``GRAPHMODE=full`` + ``HOSTIDS=1``: a step
    the host refused still replays, and the graph hands back the PREVIOUS step's
    embedding rows. The generation tripwire cannot see a sustained refusal (nothing
    advances the counter, so nothing looks stale), which is why the counter exists.
    A FULL boot that leaves this at zero never reached the cliff; a non-zero number
    means FULL needs the runner to pick a non-FULL descriptor for unstaged steps.
    """
    emb = StagingEmbedding()
    emb._staged_branch_captured = True
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_gen_layers = [emb]

    with caplog.at_level("WARNING"):
        st.begin_step("FULL")
        st._note_host_refusal("history-pending")
        st.begin_step("FULL")
        st._note_host_refusal("history-pending")

    assert st._ple_host_stale_steps == 2
    msgs = [
        r.getMessage() for r in caplog.records if "captured graph" in r.getMessage()
    ]
    assert len(msgs) == 1, "first, then every 32nd -- not one per step"
    assert "NO device path" in msgs[0]
    assert "previous step's embedding rows" in msgs[0]

    # Two different refusals inside ONE step are one stale step. The question the
    # counter answers is "how many steps served stale PLE", and counting per refusal
    # made it mean "how many things were unhappy": a step that tripped three checks
    # reported three.
    st.begin_step("FULL")
    st._note_host_refusal("history-pending")
    st._note_host_refusal("not-pure-decode")
    assert st._ple_host_stale_steps == 3, "a step counted itself twice"


def test_a_piecewise_step_after_a_capture_is_not_a_stale_replay(
    hostids_env, caplog
) -> None:
    """Boot #28's 51 ``not-pure-decode`` refusals were read as 51 stale steps.

    They were almost certainly not. The box runs ``FULL_AND_PIECEWISE`` (the v1
    default), where a mixed batch dispatches to PIECEWISE and the device PLE path
    still runs -- a refusal there costs an optimisation and changes no output. The
    instrument that was supposed to tell those apart was
    ``_staged_graph_in_play``, which returns True for every step after the first
    capture, PIECEWISE included, and so could only ever answer "yes, stale".

    This is the per-step version of the question, and it is the one the runner can
    answer exactly: ``batch_desc.cg_mode`` is decided ~170 lines before staging.
    """
    emb = StagingEmbedding()
    emb._staged_branch_captured = True
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_gen_layers = [emb]

    # A PIECEWISE step in a boot where a FULL graph exists. The old flag says
    # "stale"; the descriptor says otherwise, and the descriptor wins.
    st._ple_step_cg_mode = CUDAGraphMode.PIECEWISE.name
    with caplog.at_level("WARNING"):
        st._note_host_refusal("not-pure-decode")
    assert st._ple_host_stale_steps == 0, [
        "a PIECEWISE refusal was counted as a stale FULL replay -- boot #28's error"
    ]
    assert st._ple_host_refusals["not-pure-decode"] == 1, "the refusal vanished instead"

    # The same refusal, one step later, on a step that really does replay.
    st._ple_step_cg_mode = CUDAGraphMode.FULL.name
    st._note_host_refusal("not-pure-decode")
    assert st._ple_host_stale_steps == 1, "the real cliff stopped being counted"
    stale = [
        r.getMessage()
        for r in caplog.records
        if "cg_mode=FULL" in r.getMessage() and "NO device path" in r.getMessage()
    ]
    assert len(stale) == 1, "the stale claim must name the mode it is true in"


def test_disable_reports_the_coverage_it_is_leaving(hostids_env, caplog) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1

    with caplog.at_level("WARNING"):
        st.disable_host_staging("test reason")

    assert st._ple_host_layers == []
    # Engagement and coverage have to be recoverable from the log alone: a
    # disable that does not say how much it is switching off is not auditable.
    assert any("after 1 step" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# P1.5: the generation tripwire
#
# A frozen CUDA graph is silent: it replays its captured work and prints nothing,
# so the only way to know it is carrying this step's PLE rows is to have the graph
# itself carry a number. These tests cover the whole chain -- the counter written
# with the rows, the copy inside the graph, the non-blocking read, the verdict --
# because the failure it exists to catch (2026-09-20: wrong embeddings for five
# and a half hours, found by eye) is invisible to every other check here.
# --------------------------------------------------------------------------


def _stage(emb, *, gather: bool = True, num_reqs: int = 2) -> bool:
    inp, qsl, ctx = _step_inputs(num_reqs=num_reqs)
    dev = None
    if not gather:
        # Differential staging hands the runner's own arrays over, and the gate reads
        # them the next time the op runs. Padding to the buffer capacity is the
        # runner's shape, not a decoration: the real array is padded too, and a test
        # fixture that handed an unpadded one would be asserting a shape production
        # never produces.
        pad = max(0, CAPACITY - qsl.shape[0])
        dev = (
            torch.from_numpy(inp.copy()),
            torch.from_numpy(
                np.concatenate([qsl, np.full(pad, int(qsl[-1]), dtype=np.int64)])
            ),
            None,
        )
    return emb.stage_host_step(inp, qsl, ctx, gather=gather, device_inputs=dev)


def _agree_inputs(emb, *, rows: int = ROWS) -> None:
    """Say what both sides of a hand-driven comparison read.

    ``compare_host_ids`` refuses anything whose inputs it cannot match up (P1.10b),
    so a test that hands it an ids buffer directly has to state the tokens the compute
    consumed as well as the ones the host mirrored. This is the agreeing case; the
    refusing cases build the two arrays by hand so the disagreement is visible in the
    test rather than produced by a helper's arithmetic.
    """
    host_ids = np.arange(rows, dtype=np.int64) + 100
    host_qsl = np.array([0, rows // 2, rows], dtype=np.int64)
    host_ctx = np.zeros((rows, CTX), dtype=np.int64)
    emb._host_check_inputs = (host_ids, host_qsl, host_ctx)
    pad = max(0, CAPACITY - host_qsl.shape[0])
    emb._host_check_dev_inputs = (
        torch.from_numpy(host_ids.copy()),
        torch.from_numpy(
            np.concatenate([host_qsl, np.full(pad, rows, dtype=np.int64)])
        ),
        # P1.11: a window the device agrees with, padded to buffer capacity the way
        # ``Qwen4ExpModelState.ngram_context`` is. Handing None here instead would
        # exercise the unchecked leg, not the compared one, and every green test in
        # this file would silently be testing two legs out of three.
        torch.from_numpy(
            np.concatenate([host_ctx, np.zeros((CAPACITY - rows, CTX), np.int64)])
        ),
    )


def _interval(monkeypatch, n: int) -> None:
    monkeypatch.setattr(ple_hostids, "hostids_check_interval", lambda: n)


def _replay(emb, gen: int) -> None:
    """Stand in for the graph: the captured copy put ``gen`` into the scratch.

    Call this *before* the check that issues the read, not after. The read is a
    copy that runs in stream order, so what it samples is the scratch's value at
    the moment the issue happens -- setting the scratch afterwards would model a
    copy that reads the future, and would have made the frozen-handoff tests pass
    for the wrong reason.
    """
    assert emb._gen_device is not None
    emb._mirror_op_gen = False  # this test drives the scratch itself from here on
    emb._gen_device.fill_(gen)


def test_gen_verdict_covers_every_direction() -> None:
    """The pure half of the tripwire, testable with no torch and no device."""
    assert ple_hostids.gen_verdict(5, 5) == "ok"
    assert ple_hostids.gen_verdict(4, 5) == "frozen"
    assert ple_hostids.gen_verdict(0, 5) == "frozen"
    assert ple_hostids.gen_verdict(6, 5) == "ahead"
    # "never staged" must not borrow "ok"'s authority.
    assert ple_hostids.gen_verdict(0, 0) == "no-staging"


def test_check_interval_defaults_clamps_and_ignores_junk(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", raising=False)
    assert ple_hostids.hostids_check_interval() == 64
    monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", "1")
    assert ple_hostids.hostids_check_interval() == 1
    # 0 would sample every step and, worse, make `countdown > 0` never true in a
    # way that reads like a bug; a negative interval is not a thing.
    monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", "0")
    assert ple_hostids.hostids_check_interval() == 1
    monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS_CHECK", "yes")
    assert ple_hostids.hostids_check_interval() == 64


def test_staging_advances_the_generation_with_the_rows() -> None:
    emb = StagingEmbedding()
    assert _stage(emb) is True
    assert emb._host_stage_gen == 1
    assert int(emb._pinned_gen[0]) == 1
    _stage(emb)
    assert emb._host_stage_gen == 2 and int(emb._pinned_gen[0]) == 2


def test_a_refused_stage_does_not_advance_the_generation() -> None:
    """The same-act rule, observed from the refusing side.

    ``stage_host_step`` returns before writing anything when the rows will not
    fit, so the counter must not move either. A counter that advanced past a
    refusal would tell the next check "the graph owes you generation N" for rows
    that were never staged -- inventing a stale handoff out of a healthy step.
    """
    emb = StagingEmbedding()
    assert _stage(emb) is True
    # 3 requests x 3 tokens = 9 rows > CAPACITY: refused, nothing written.
    assert _stage(emb, num_reqs=3) is False
    assert emb._host_stage_gen == 1, "a refused step advanced the generation"
    assert int(emb._pinned_gen[0]) == 1


def test_differential_mode_never_advances_the_generation() -> None:
    """Mode 2 keeps the device in charge, so the tripwire stays inert by design.

    Not a mode check inside the op: the counter simply never leaves zero, which
    makes "frozen" unreachable for a path that never claimed to hand anything off.
    """
    emb = StagingEmbedding()
    assert _stage(emb, gather=False) is True
    assert emb._host_stage_gen == 0
    assert int(emb._pinned_gen[0]) == 0


def test_the_check_issues_a_read_only_every_nth_step(monkeypatch) -> None:
    """Sampling is throttled, and the sampled step's answer arrives on the next."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 3)
    emb._gen_check_countdown = 3
    _stage(emb)
    verdicts = []
    for _ in range(9):
        _replay(emb, emb._host_stage_gen)  # the graph keeps up
        verdicts.append(emb.check_host_gen())
    assert verdicts[:6] == ["", "", "issued", "ok", "", ""], verdicts
    # Two samples in nine steps of interval 3 (the step that reads a verdict back
    # also occupies a step, which is why this is not every third call exactly).
    assert verdicts.count("issued") == 2
    assert emb._gen_event.records == 2
    assert emb._gen_checks_ok == 2
    assert emb._hostids_disabled is False


def test_nothing_staged_yet_is_not_reported_as_ok() -> None:
    """A vacuous pass is reported as what it is.

    "ok" for a boot that has staged nothing would make the first interval of every
    run look verified, which is the same class of lie as an instrument that cannot
    go red.
    """
    emb = StagingEmbedding()
    assert emb.check_host_gen() == "no-staging"
    assert emb._gen_checks_ok == 0


def test_a_fresh_handoff_passes_and_is_counted(monkeypatch) -> None:
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    _stage(emb)
    _replay(emb, 1)  # the graph delivered generation 1
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "ok"
    assert emb._gen_checks_ok == 1
    assert emb._hostids_disabled is False
    assert emb._gen_trips == 0


def test_a_frozen_handoff_disables_while_the_device_path_is_reachable(
    monkeypatch,
) -> None:
    """No captured graph => the device path really does run, so back off."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    _stage(emb)
    _replay(emb, 0)  # the scratch is stuck at capture's value
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "frozen"
    assert emb._hostids_disabled is True
    assert emb._gen_trips == 1


def test_a_frozen_handoff_is_fatal_once_a_graph_captured_the_staged_branch(
    monkeypatch,
) -> None:
    """The case where "fall back" is not a thing a replay can do.

    Inside a captured graph the op body does not run, so the device path is
    unreachable and every further step embeds the wrong step's tokens. Continuing
    is the 2026-09-20 failure; raising is the only correct remaining action.
    """
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    emb._staged_branch_captured = True
    _stage(emb)
    _replay(emb, 0)  # capture's value: the graph never advanced it
    assert emb.check_host_gen() == "issued"
    with pytest.raises(RuntimeError, match="frozen embeddings"):
        emb.check_host_gen()


def test_a_device_value_ahead_of_the_host_is_not_treated_as_ok(monkeypatch) -> None:
    """One scratch and one producer cannot get ahead; if it does, stop anyway."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    _stage(emb)
    _replay(emb, 9)
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "ahead"
    assert emb._hostids_disabled is True


def test_a_read_that_never_lands_fails_instead_of_hanging(monkeypatch) -> None:
    """If the scratch is never written the check must still terminate loudly."""
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    emb._gen_event = FakeEvent(ready=False)
    _stage(emb)
    verdicts = [emb.check_host_gen() for _ in range(6)]
    assert verdicts[0] == "issued"
    assert "in-flight" in verdicts
    assert emb._hostids_disabled is True, verdicts
    assert emb._gen_trips == 1


def test_the_check_never_waits_on_the_device(monkeypatch) -> None:
    """The instrument must not cost the thing it protects.

    ``FakeEvent.synchronize`` raises, so reaching it anywhere in this file fails
    the test that happened to be running; this test asserts the non-blocking path
    was actually exercised rather than skipped.
    """
    emb = StagingEmbedding()
    _interval(monkeypatch, 1)
    emb._gen_event = FakeEvent(ready=False)
    _stage(emb)
    assert emb.check_host_gen() == "issued"
    assert emb.check_host_gen() == "in-flight"
    assert emb._gen_event.records == 1
    assert emb._gen_event.queries == 1
    assert emb._hostids_disabled is False


def test_the_op_copies_the_generation_into_the_scratch(monkeypatch) -> None:
    """The graph node that makes the tripwire a measurement instead of a tautology.

    RED CONTROL: delete the ``scratch.copy_`` in the staged branch and this fails,
    because the scratch would still hold capture's value while the host counter
    marched on.
    """
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    emb._pinned_gen.fill_(7)
    assert emb._gen_device is not None
    assert int(emb._gen_device[0]) == 0
    with _op_env(emb, monkeypatch):
        _run_op()
    assert int(emb._gen_device[0]) == 7


def test_the_op_records_that_the_staged_branch_was_captured(monkeypatch) -> None:
    """What makes a tripwire fire fatal rather than a quiet disable."""
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    with _op_env(emb, monkeypatch, capturing=True):
        _run_op()
    assert emb._staged_branch_captured is True

    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS
    with _op_env(emb, monkeypatch, capturing=False):
        _run_op()
    assert emb._staged_branch_captured is False


def test_the_device_path_never_touches_the_generation_scratch(monkeypatch) -> None:
    """The copy is a node of the staged branch only.

    If the device path also copied, a frozen staged graph would be "fixed" by the
    eager path writing the counter, and the tripwire would report a healthy handoff
    for a graph that delivered nothing.
    """
    emb = StagingEmbedding()
    emb._host_stage_rows = ROWS - 1  # disagreement -> device path
    emb._pinned_gen.fill_(7)
    with _op_env(emb, monkeypatch):
        _run_op()
    assert emb.lookup_calls == 1
    assert emb._gen_device is not None
    assert int(emb._gen_device[0]) == 0, "the device path advanced the tripwire"


def test_scratch_allocation_refuses_to_happen_inside_a_capture(monkeypatch) -> None:
    emb = StagingEmbedding()
    emb._gen_device = None
    emb._host_stage_rows = ROWS
    with (
        _op_env(emb, monkeypatch, capturing=True),
        pytest.raises(RuntimeError, match="never allocated"),
    ):
        _run_op()


def test_scratch_is_created_lazily_outside_a_capture(monkeypatch) -> None:
    """A module that never ran load_weights still works, and reports honestly."""
    emb = StagingEmbedding()
    emb._gen_device = None
    emb._host_stage_rows = ROWS
    emb._pinned_gen.fill_(4)
    with _op_env(emb, monkeypatch, capturing=False):
        _run_op()
    assert emb._gen_device is not None
    assert int(emb._gen_device[0]) == 4


# --------------------------------------------------------------------------
# Wiring: the model state runs the check even on steps it refused
# --------------------------------------------------------------------------


class WatchedLayer:
    """A registered embedding that records tripwire samples."""

    ngram_size = NGRAM_SIZE

    def __init__(self) -> None:
        self.samples = 0
        self._staged_branch_captured = False
        self._host_stage_rows = -1

    def check_host_gen(self) -> str:
        self.samples += 1
        return "ok"

    def take_host_stage(self):
        return None

    def stage_host_step(self, *a: Any, **kw: Any) -> bool:  # noqa: ANN002, ANN003
        return False


def test_the_check_runs_on_steps_the_host_refused(hostids_env) -> None:
    """A refusal is the interesting case, so the check sits above the refusals."""
    hostids_env["layers"] = [WatchedLayer()]
    st = FakeModelState()
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    assert "no-draft-source" in st._ple_host_refusals
    assert st._ple_host_steps == 0
    layer = hostids_env["layers"][0]
    assert layer.samples == 1, "the tripwire was skipped on a refused step"


def test_the_check_runs_once_per_registered_layer(hostids_env) -> None:
    hostids_env["layers"] = [WatchedLayer(), WatchedLayer()]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)
    # Resolution is what arms the watchers; nothing sets this by hand.
    assert st._ple_host_gen_layers == hostids_env["layers"]
    assert all(layer.samples == 1 for layer in hostids_env["layers"])


def test_a_doubt_after_capture_stops_the_boot_instead_of_serving(
    hostids_env, caplog, monkeypatch
) -> None:
    """The stop-boot mechanism, which is no longer the default -- see below.

    Boot #28 reached this exact state by a different route -- the slice-clamp false
    positive disabled the knob at 03:49:06 -- and then served 8,141 steps that
    replayed one frozen set of PLE rows. It answered a smoke test whose PLE was
    hard-zeroed perfectly, which is what makes the state worth dying on rather than
    serving. So the mechanism raises BEFORE emptying the layer list: a boot that
    stops should stop with its state intact for the post-mortem.

    It ships as opt-in now (``PLE_DOUBT_UNDER_FULL`` defaults to ``"eager"``, which
    routes these steps through the dispatch veto instead), because the state turned out
    to be reachable through real mirror corruption and not only checker false alarms,
    and three boots in a row died of the setting on 2026-10-05. So this test opts in: it
    pins that choosing stop-boot works, not that it is chosen.
    ``test_the_shipped_doubt_policy_is_eager`` pins the other half.
    """
    monkeypatch.setattr(FakeModelState, "PLE_DOUBT_UNDER_FULL", "stop-boot")
    layer = WatchedLayer()
    hostids_env["layers"] = [layer]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)

    layer._staged_branch_captured = True
    with pytest.raises(model_state_module.PLEHostDoubtUnderFullGraph) as exc:
        st.disable_host_staging("test reason")

    msg = str(exc.value)
    assert "test reason" in msg, "the reason has to survive into the crash"
    assert "no device path" in msg.lower()
    assert "8141" in msg, "the boot that proved this needs to be named out loud"
    assert "PLE_DOUBT_UNDER_FULL" in msg, "the escape hatch must be named"
    assert st._ple_host_layers, "raising must not empty the layer list on the way out"
    assert st._ple_host_gen_layers == [layer]


def test_a_repeated_disagreement_under_capture_escalates_past_the_repair_handler(
    hostids_env, monkeypatch
) -> None:
    """The dedicated exception type exists for exactly one reason.

    ``_ple_resync_mirror`` wraps the whole repair in ``except Exception`` -- rightly,
    because a repair that throws would turn a coverage bug into a crash on a step
    already handed to the device. But the disagreement budget is decided INSIDE that
    handler, and once a FULL graph is in play the budget's decision is "stop the
    boot". A stop-the-boot decision a broad handler can swallow is not a
    stop-the-boot decision, so it has a type of its own and a re-raise in front of
    the handler.

    Deleting that re-raise is control 6 of ``scripts/red-controls.sh``, and this is
    the test it turns red: without it the exception lands in ``_ple_resync_refused``
    as a string, the boot keeps serving, and the only symptom is stale PLE.

    Opted into ``stop-boot``: that is no longer the shipped default, and this test is
    about whether the raise survives the broad handler, not about who asks for it.
    """
    monkeypatch.setattr(FakeModelState, "PLE_DOUBT_UNDER_FULL", "stop-boot")
    layer = WatchedLayer()
    layer._staged_branch_captured = True
    st = FakeModelState()
    # Populated by hand: this test enters through the repair, not through a staged
    # step, and ``disable_host_staging`` returns early when there are no layers.
    st._ple_host_layers = [layer]
    st._ple_host_gen_layers = [layer]
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    for attempt in range(2):
        # A different step each time, which is what the loop describes: in production
        # these are two consecutive decode steps, and the per-call sync bound keys on
        # ``_ple_host_calls`` for exactly that reason. Without the bump the second
        # iteration is refused by the bound and the escalation this test exists for
        # never happens.
        st._ple_host_calls = attempt + 1
        # A commit the mirror never saw, and a token in the cell the cross-check
        # reads that the mirror has never held. Two of them, because the budget is
        # per-slot and the first one is "the repair"; the second is the pattern.
        req_states.num_computed_tokens_np[0] += 1
        req_states.all_token_ids.gpu[0, int(hist.length[0]) - 1] = (
            4161103993993705554 % (2**31)
        ) + attempt
        ctx = (
            pytest.raises(model_state_module.PLEHostDoubtUnderFullGraph)
            if attempt == 1
            else contextlib.nullcontext()
        )
        with ctx:
            st._ple_resync_mirror(req_states, input_batch, f"test force {attempt}")
        if attempt == 0:
            assert st._ple_resync_disagree == 1, "the first force was not a disagreement"
            assert st._ple_host_layers, "the first one disabled the knob"

    assert st._ple_resync_disagree == 2
    assert st._ple_host_layers, "the escalation emptied the layer list on the way out"


def test_degrade_policy_keeps_the_tripwire_armed_after_capture(
    hostids_env, caplog, monkeypatch
) -> None:
    """The old behaviour, verbatim, under the policy that asks for it.

    Disabling the host path must not disarm the instrument that proves it broke,
    and the disable message must not claim a device path that a replay cannot
    reach. Both of those are still true when the operator chooses ``"degrade"``;
    this test is what makes that choice a supported one rather than a bit-rotted
    fallback nobody has run since the default changed.
    """
    monkeypatch.setattr(FakeModelState, "PLE_DOUBT_UNDER_FULL", "degrade")
    # Patched on the fake, not the real class: ``FakeModelState`` copies the policy
    # constant at class-creation time, so patching production would leave this test
    # running the default and asserting a behaviour it never asked for.
    layer = WatchedLayer()
    hostids_env["layers"] = [layer]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)

    layer._staged_branch_captured = True
    with caplog.at_level("WARNING"):
        st.disable_host_staging("test reason")

    assert st._ple_host_layers == []
    assert st._ple_host_gen_layers == [layer], "disarming hid the frozen graph"
    assert any("does NOT handle" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


def test_disabling_reports_a_real_fallback_when_nothing_was_captured(
    hostids_env, caplog
) -> None:
    layer = WatchedLayer()
    hostids_env["layers"] = [layer]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()
    st._stage_host_ple_ids(input_batch, req_states)

    with caplog.at_level("WARNING"):
        st.disable_host_staging("test reason")

    assert st._ple_host_layers == []
    assert st._ple_host_gen_layers == [layer]
    assert any("handles every step" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The producer side: the model runner's epilogue handing over accepted tokens
# ---------------------------------------------------------------------------


class FakeCopyEvent:
    """``AsyncOutput.copy_event``, as far as the mirror is concerned."""

    def __init__(self, done: bool = False) -> None:
        self.done = done
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.done


def _commit(counts=(1, 1), done=False):
    """A commit for slots 0 and 3: one accepted token each by default."""
    rows = np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.int32)
    return (
        np.array([0, 3], dtype=np.int32),
        rows,
        np.array(counts, dtype=np.int32),
        FakeCopyEvent(done),
    )


def _lost_counts():
    """A commit the history must refuse, without reading an unlanded value.

    Every test that needed "the epilogue lost a commit" used to write
    ``np.array([9, 1])`` here -- a count out of range for a 4-column row. Task
    #56 made that a LEGITIMATE commit at stage time, because before the copy
    lands those numbers are whatever the allocator recycled, not this step's
    counts (that check now runs in ``drain``). A counts vector one row shorter
    than the batch is still refused at stage: shape is the only thing a
    pre-landing check is allowed to look at, which is the rule #56 exists to
    enforce.
    """
    return np.array([1], dtype=np.int32)  # 1 count for the 2 slots of _commit()


def test_epilogue_commit_is_free_when_the_knob_is_off(hostids_env) -> None:
    """The epilogue runs on every step of every boot, knob or no knob.

    So with the knob off it must queue nothing, count nothing and -- because a
    stray log line here would be read as engagement -- say nothing.
    """
    hostids_env["mode"] = 0
    st = FakeModelState()
    assert st.stage_host_tokens(*_commit()) is False
    assert st._ple_host_history.pending == 0
    assert st._ple_host_refusals == {}


def test_landed_commit_advances_the_step(hostids_env) -> None:
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()

    assert st.stage_host_tokens(*_commit(done=True)) is True
    # The commit carries one accepted token per request; the device's post-update
    # kernel advances num_computed by the same count, and so does the scheduler's
    # host copy that the batch carries.
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert st._ple_host_steps == 1
    assert st._ple_host_history.pending == 0
    # The token the commit carried is now the newest cell of each slot's window,
    # which is what the builder reads as row 0. (That the ids then follow is
    # test_pure_decode_batch_reaches_the_layer's job, with hashed values.)
    assert st.staged_rows() == [ROWS]
    assert emb.lookup_calls == 1
    hist = st._ple_host_history
    assert (hist.window[0][-1], hist.window[3][-1]) == (7, 11)
    assert (hist.length[0], hist.length[3]) == (_POSITION + 2, _POSITION + 2)


def test_unlanded_commit_refuses_the_step_rather_than_waiting(hostids_env) -> None:
    """The design's central bargain: no per-step host sync, so refuse instead.

    A commit whose D2H has not landed means the newest token is not host-visible
    yet. Blocking for it is the sync this whole path exists to remove; skipping it
    would build ids one token short. Refusing the step costs the device path one
    step and keeps the promise.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()

    slots, rows, counts, event = _commit(done=False)
    assert st.stage_host_tokens(slots, rows, counts, event) is True
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals.get("history-pending") == 1
    assert st._ple_host_steps == 0
    assert emb.lookup_calls == 0

    # The copy lands; the same step now goes through, with the counter advanced
    # by the commit it was waiting on.
    event.done = True
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1
    assert st._ple_host_history.pending == 0


def test_a_lost_commit_is_repaired_not_poisoning(hostids_env) -> None:
    """A commit that fails validation costs a step and a sync, not the request.

    The device advanced its counter whether or not the host mirror got the tokens,
    so the mirror is one token short afterwards. Until 2026-10-04 that was
    terminal: nothing could rebuild a slot -- ``seed`` ran at add and
    ``invalidate`` at remove and never again -- so the exact-length check refused
    that request for the rest of its life. Boot #26 measured the result: 411
    ``history-length`` refusals per worker and six consecutive windows of 32/32,
    with the drops that caused them all clustered behind them at batch transitions.

    What the repair does NOT give up: it reads nothing until the device has been
    synced (the row's generated cells are written by a kernel that is unordered
    before a host read), it cross-checks the row against the cells the old window
    still claims to hold, and it is budgeted per request and per boot. A mirror it
    cannot vouch for still refuses the step rather than building ids from a
    history with a hole in it.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, tokens = _batch_and_states()

    slots, rows, counts, event = _commit(done=True)
    # A commit claiming more accepted tokens than the row holds.
    assert (
        st.stage_host_tokens(slots, rows, _lost_counts(), event)
        is False
    )
    assert "stage-tokens-rejected" in st._ple_host_refusals
    # Boot #26 could not say why a commit was lost, only how many. The detail now
    # names the check that lost it.
    assert "num_sampled" in st._ple_host_history.last_drop

    # The device advanced anyway, so the mirror now lags by exactly one token.
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals.get("history-length") is None, "not repaired"
    assert st._ple_host_steps == 1
    assert st._ple_resyncs == 1, "the repair did not run"
    assert st._ple_syncs == 1, "the repair read the row without ordering the device"
    hist = st._ple_host_history
    for slot in (0, 3):
        assert hist.length[slot] == req_states.num_computed_tokens_np[slot] + 1
        assert int(hist.window[slot, -1]) == int(tokens[slot, hist.length[slot] - 1])

    # The budget is per request and finite -- when it is finite. It is not by
    # default any more (see PLE_RESYNC_PER_SLOT), so this test asks for the old cap
    # explicitly: what is under test is that a finite cap latches the slot, which is
    # exactly the behaviour that cost boot vllm.log.20261006-223026 96 % stale steps.
    st.PLE_RESYNC_PER_SLOT = 2
    for step in range(st.PLE_RESYNC_PER_SLOT - 1):
        # A fresh draft block each step, or the staleness check refuses before the
        # build and the loop measures the wrong thing.
        drafts.bump_generation(10 + step)
        req_states.num_computed_tokens_np[[0, 3]] += 1
        _advance_device_ids(
            input_batch,
            st,
            row=req_states.all_token_ids.gpu.numpy(),
            computed=req_states.num_computed_tokens_np,
        )
        st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resyncs == st.PLE_RESYNC_PER_SLOT
    drafts.bump_generation(99)
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_refusals.get("history-length") == 1
    assert st._ple_resyncs == st.PLE_RESYNC_PER_SLOT, "budget exceeded"
    assert "per-slot cap" in st._ple_resync_refused


def test_repairs_are_uncapped_by_default(hostids_env) -> None:
    """A slot that needs a repair every step keeps getting one, and keeps staging.

    This is the change of 2026-10-07, and it is a measurement, not a preference.
    ``PLE_RESYNC_PER_SLOT`` was 2 on the belief that a repair costs a device sync and
    a sync costs milliseconds. Boot ``vllm.log.20261006-223026`` measured 256 repairs:
    p50 0.08 ms, max 0.23 ms, against a 42.5 ms step -- and measured the demand at
    2 497 ``history-length`` refusals per worker against 9 918 steps, because under
    MTP an unaccepted draft leaves the mirror one behind routinely. A per-slot budget
    of 2 therefore latched every slot of a request within its first three steps and
    the request then replayed stale PLE rows for its remaining ~200: acceptance
    length 1.98 there, 3.31 with the rows zeroed (boot #37), 4.00 with them correct
    (boot #36). The cap was not converting a coverage bug into a latency bug. It was
    the coverage bug, and it was never the latency bug either.

    The total went the same way one boot later, and the way it went is the lesson.
    2000 was sized from that 25 % demand figure -- and the figure was censored: it
    came from a boot whose demand was suppressed by the per-slot cap that was being
    measured. Boot ``vllm.log.20261007-070900``, with nothing capped, needed 2 000
    paid repairs plus 3 441 steps refused for want of one, 5 441 of 8 667 decode
    steps: 63 %, not 25 %. The allowance was exhausted in twenty minutes of light
    probe traffic, and the request that ran afterwards took 768 stale steps over 766
    chunks -- every step of it, at 58.69 tok/s and acceptance 2.61. A budget that
    runs out does not stop anything; it converts a coverage bug into a quality bug
    and moves the evidence from one refusal string to another.

    What still bounds a step is the per-call sync bound, which this loop also
    exercises from the other side: one sync per step, thirty steps, thirty syncs.
    That is the bound that actually carries the risk -- 0.12 ms x one per step, so
    the worst case is 0.19 % of a 42.5 ms step whatever the demand turns out to be.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    assert st.PLE_RESYNC_PER_SLOT == 0, "the shipped default is no longer uncapped"
    assert st.PLE_RESYNC_TOTAL == 0, (
        "a total backstop is back. It has now cost a boot twice: 64 latched every "
        "slot of a request inside three steps, and the 2000 that replaced it was "
        "spent in twenty minutes because the 25 % demand it was sized on had been "
        "censored by the 64. Both degraded to stale rows, which is the silent "
        "failure a backstop is supposed to prevent. The per-call bound is the one "
        "that prices the risk; the demand rate is alarmed, not capped -- see "
        "PLE_RESYNC_ALARM_RATE."
    )
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, tokens = _batch_and_states()

    slots, rows, counts, event = _commit(done=True)
    assert (
        st.stage_host_tokens(slots, rows, _lost_counts(), event)
        is False
    )
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resyncs == 1

    for step in range(29):
        drafts.bump_generation(10 + step)
        req_states.num_computed_tokens_np[[0, 3]] += 1
        _advance_device_ids(
            input_batch,
            st,
            row=req_states.all_token_ids.gpu.numpy(),
            computed=req_states.num_computed_tokens_np,
        )
        st._stage_host_ple_ids(input_batch, req_states)

    # Thirty steps, thirty repairs, thirty syncs, and not one refused step.
    assert st._ple_resyncs == 30, st._ple_resync_refused
    assert st._ple_syncs == 30, "one sync per repair is the bound, not fewer"
    assert st._ple_host_steps == 30, "a step that was not staged"
    assert st._ple_host_refusals.get("history-length") is None, "a step went stale"
    assert "per-slot cap" not in st._ple_resync_refused, "the latch is back"
    assert "step cap" not in st._ple_resync_refused, "the per-call bound latched"


def test_the_per_step_bound_allows_exactly_one_sync_and_then_releases(
    hostids_env,
) -> None:
    """One repair sync per step, and the next step is not charged for it.

    The bound is what makes an uncapped repair safe: 0.08 ms when the host is ahead
    of the device, 13.2 ms when it is not (``synchronize()`` with one matmul in
    flight, same probe). It is counted on ``_ple_host_calls`` rather than
    ``_ple_host_steps`` because the latter only advances when a step *stages* --
    keying the bound on it would treat a run of refused steps as one step, which is
    the same class of mistake as the per-slot latch: a counter that does not move
    when the thing it counts happens.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    slots, rows, counts, event = _commit(done=True)
    st.stage_host_tokens(slots, rows, _lost_counts(), event)
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)

    st._ple_host_calls = 7
    assert st._ple_resync_mirror(req_states, input_batch, "test") is True
    assert st._ple_syncs == 1
    # Same call, second attempt: refused, and no second sync was paid for. This is
    # checked before the slots are even read, which is the point -- the bound is on
    # the sync, not on whether this step happens to still look repairable.
    assert st._ple_resync_mirror(req_states, input_batch, "test") is False
    assert "step cap" in st._ple_resync_refused
    assert st._ple_syncs == 1, "the bound let a second sync through"
    # The next step is a different call, and a fresh lag in it is repairable again:
    # the bound releases, it does not latch. A bound that latched would be the
    # per-slot bug wearing a new name.
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    st._ple_host_calls = 8
    assert st._ple_resync_mirror(req_states, input_batch, "test") is True
    assert st._ple_syncs == 2


def test_repairs_stop_at_the_total_backstop_even_with_no_slot_cap(hostids_env) -> None:
    """The backstop is a backstop: it bites when the loop has genuinely run away.

    Uncapping per slot must not mean "no bound at all" -- the total is what makes
    hitting a systematic problem a logged event rather than a silent degradation,
    which is how boot vllm.log.20261006-223026 ended (64 of 64 spent, then nothing
    repairable for the rest of the boot and no line saying so).
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.PLE_RESYNC_TOTAL = 2
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    slots, rows, counts, event = _commit(done=True)
    st.stage_host_tokens(slots, rows, _lost_counts(), event)
    for step in range(4):
        drafts.bump_generation(10 + step)
        req_states.num_computed_tokens_np[[0, 3]] += 1
        _advance_device_ids(
            input_batch,
            st,
            row=req_states.all_token_ids.gpu.numpy(),
            computed=req_states.num_computed_tokens_np,
        )
        st._ple_host_calls = step + 1
        st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resyncs == 2, "the total backstop did not hold"
    assert "total cap" in st._ple_resync_refused


def test_resync_cap_env_parsing(monkeypatch) -> None:
    """A knob that gates device syncs fails to the shipped value, not to zero.

    ``0`` means uncapped here, so a typo clamped or parsed to 0 would silently
    remove a cap somebody meant to keep -- and the whole point of the env override
    is that it is reachable from a shell, where typos happen.
    """
    cap = model_state_module._ple_resync_cap
    monkeypatch.delenv("VLLM_GFX906_PLE_RESYNC_PER_SLOT", raising=False)
    assert cap("VLLM_GFX906_PLE_RESYNC_PER_SLOT", 0) == 0
    monkeypatch.setenv("VLLM_GFX906_PLE_RESYNC_PER_SLOT", "12")
    assert cap("VLLM_GFX906_PLE_RESYNC_PER_SLOT", 0) == 12
    monkeypatch.setenv("VLLM_GFX906_PLE_RESYNC_PER_SLOT", "0")
    assert cap("VLLM_GFX906_PLE_RESYNC_PER_SLOT", 7) == 0, "explicit 0 must survive"
    monkeypatch.setenv("VLLM_GFX906_PLE_RESYNC_PER_SLOT", "-1")
    assert cap("VLLM_GFX906_PLE_RESYNC_PER_SLOT", 7) == 7, "negative fell through"
    monkeypatch.setenv("VLLM_GFX906_PLE_RESYNC_PER_SLOT", "twelve")
    assert cap("VLLM_GFX906_PLE_RESYNC_PER_SLOT", 7) == 7, "a typo fell through"


def test_resync_refuses_to_serve_when_the_row_disagrees_with_the_mirror(
    hostids_env,
) -> None:
    """A repair that takes its length from the row it read cannot notice a bad read.

    So it cross-checks the row against the cells the old window still claims --
    ``num_sampled`` commits on one side, the device's own token buffer on the
    other. Disagreement over the overlap means one of those two is lying about the
    request's history, and boot #14 already caught the host mirror handing back
    ``4161103993993705554`` where a token id belongs. The step is refused; the
    worker does not serve from a read it cannot vouch for. What it does with a
    *pattern* of these is ``test_a_disagreement_refuses_the_step_not_the_boot``.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    old_len = int(hist.length[0])
    req_states.all_token_ids.gpu[0, old_len - 1] = 4161103993993705554 % (2**31)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resync_disagree == 1
    assert st._ple_host_steps == 0, "ids built from a row that contradicts the mirror"
    assert st._ple_host_refusals.get("history-length") == 1


def test_a_mirror_ahead_of_the_device_adopts_instead_of_disagreeing(
    hostids_env,
) -> None:
    """The slice-clamp defect that killed boots #27 and #28, pinned to its one shape.

    ``row`` is ``want`` cells long. When the mirror claims MORE cells than the
    device has committed -- ``old_len > want``, which is what a rolled-back
    speculative commit looks like -- the old code asked for
    ``row[old_len - overlap : old_len]``, and numpy answers a slice that starts
    past the end of an array with the EMPTY array. Comparing that against
    ``overlap`` mirrored cells is False by construction, so *every* resync of this
    shape was recorded as a disagreement with the device's own token row. Boot #28
    reached ``PLE_DISAGREE_TOTAL`` that way and turned the knob off at 03:49:06 --
    which under a captured graph is precisely the state that serves frozen PLE
    rows, 8,141 of them. Boot #27's kill at 02:30:55 was read as two requests
    sharing a slot; this is the other candidate for the same line.

    There is nothing to compare in this shape, because the two sides describe
    different positions: the mirror's last ``overlap`` cells are device positions
    ``old_len - overlap .. old_len - 1`` and the row's are ``0 .. want - 1``. So the
    repair adopts the row and counts the event on its own ledger -- a boot that does
    this constantly is asking about commit ordering, which is not the question a
    disagreement answers.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    # The device is one commit BEHIND the mirror. Nothing about the token row has to
    # change for the shape to exist, which is exactly why the old code fell into it
    # so often: the mismatch is in the counters, not in the data.
    req_states.num_computed_tokens_np[[0, 3]] -= 1

    # Called directly, not through ``_stage_host_ple_ids``. In mode 1 the input gate
    # refuses a counters-only mismatch as ``stage-inputs-disagree`` before the
    # history check runs, so an end-to-end fixture here would assert on the gate and
    # never reach the cross-check. The clamp lives in the repair, and boot #28 is
    # the evidence that a real boot reaches it.
    before = int(hist.length[0])
    repaired = st._ple_resync_mirror(req_states, input_batch, "test mirror-ahead")

    assert repaired is True, "the caller was told not to retry after a clean adopt"
    assert int(hist.length[0]) == before - 1, "the ahead mirror was not adopted"
    assert st._ple_resync_mirror_ahead == 2, "both slots are ahead, so both count"
    assert st._ple_resync_disagree == 0, [
        "a mirror ahead of the device is not a device that disagrees. This is the "
        "assertion whose absence was boot #28's cause of death."
    ]
    assert st._ple_resyncs == 1, "one sync per repair, not one per slot"


def test_the_cross_check_only_compares_the_request_that_owns_the_slot(
    hostids_env,
) -> None:
    """Boot #27: two requests in one slot is not a disagreement, it is two requests.

    At 02:30:55 -- three seconds before its first real step -- boot #27 compared a
    mirror holding ``[4, 5, 10]`` against a row holding ``[4, 5, 19959]`` on slot 2
    and turned the knob off on all four workers. The mirror had been seeded by a
    warmup dummy that never went through ``add_request``; the row held the previous
    occupant's tokens, which ``invalidate`` does not clear because it is not the
    mirror's to clear. Neither side was wrong and the boot was dead.

    So the row is compared against the window only when the runner says the slot
    still belongs to the request the mirror describes. Otherwise the post-sync read
    of the device's own token buffer is taken on trust, which is what it is for.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    req_states.num_computed_tokens_np[[0, 3]] += 1
    old_len = int(hist.length[0])
    # A token the mirror has never seen, in the cell the overlap would compare.
    req_states.all_token_ids.gpu[0, old_len - 1] = 4161103993993705554 % (2**31)
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    # Slot 0 now holds somebody else; slot 3 still holds who the mirror says. The
    # per-slot distinction is the whole point, so the test keeps one of each.
    req_states.index_to_req_id = {0: "the-previous-occupant", 3: _req_id(3)}

    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_resync_disagree == 0, "a stranger's tokens were read as a defect"
    assert st._ple_syncs == 1
    assert st._ple_resync_owner_skip == 1
    assert st._ple_host_steps == 1, "the slot was rebuilt but the step was lost anyway"
    assert int(hist.window[0, -1]) == int(
        req_states.all_token_ids.gpu[0, hist.length[0] - 1]
    ), "the mirror did not take the row it was told to trust"
    assert st._ple_host_slot_req[0] == "the-previous-occupant"
    # The slot whose owner matched was still checked, and passed.
    assert st._ple_host_slot_req[3] == _req_id(3)


def test_a_disagreement_with_the_owner_refuses_the_step_not_the_boot(
    hostids_env,
) -> None:
    """One bad read is a refused step; a pattern of them is a dead knob.

    Under a captured graph the step has no device fallback either way, so the old
    immediate disable traded every other request in the boot for coverage it was
    not going to get on that step anyway. Disabling stays correct once the
    disagreement repeats, because then the thing that is broken is the host's
    ability to read its own memory.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    hist = st._ple_host_history
    req_states.num_computed_tokens_np[[0, 3]] += 1
    old_len = int(hist.length[0])
    req_states.all_token_ids.gpu[0, old_len - 1] = 4161103993993705554 % (2**31)
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resync_disagree == 1
    assert st._ple_host_steps == 0, "ids built from a row that contradicts the mirror"
    assert st._ple_host_layers == [emb], "one disagreement disabled the whole boot"
    assert emb._hostids_disabled is False

    # The same slot again. The mirror was deliberately not re-seeded after a
    # disagreement, so it still lags and the next step asks the same question --
    # which is the point: two on one request means the two sides will not agree.
    # A *different* bad cell, which is what "the two sides are not going to agree"
    # actually looks like. Re-reporting the SAME cell is a stuck slot, and that is
    # measured in steps instead -- see
    # ``test_the_same_comparison_repeated_does_not_take_the_knob_off``.
    req_states.all_token_ids.gpu[0, old_len - 1] = 1234567890
    drafts.bump_generation(7)
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_resync_disagree == 2
    assert st._ple_host_layers == [], "a repeating disagreement kept the knob on"
    assert st._ple_resync_disagree_slots[0] == FakeModelState.PLE_DISAGREE_PER_SLOT
    assert st._ple_resync_disagree_repeat == 0, "a different cell is not a repeat"


def test_staged_steps_stops_when_staging_is_disabled(hostids_env) -> None:
    """A counter that survives the thing it counts cannot be the shipping signal.

    Boot #27 logged ``staged_steps=5120`` for a worker that had disabled staging at
    02:30:55, whose mirror had taken 1,140 commits and whose layer had been handed
    one row -- the capture dummy. ``_stage_host_ple_ids`` tested only whether the
    layer list had been *built*, and an empty list built just as well as a full
    one, so the counter advanced through a staging loop that ran zero times. Every
    number on that line now comes from at or below the layer that does the work.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1
    assert st._ple_host_layer_steps() == 1

    st.disable_host_staging("test: what the counter does afterwards")
    before = st._ple_host_layer_steps()

    for step in range(3):
        drafts.bump_generation(10 + step)
        st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_steps == 1, "the counter kept rising after the disable"
    assert st._ple_host_layer_steps() == before
    assert st._ple_host_refusals.get("staging-disabled") == 3
    coverage = st._ple_host_coverage()
    assert "staged_steps=1" in coverage
    assert f"layers_staged={before}" in coverage
    assert "disabled=yes" in coverage


def test_a_layer_that_declines_the_ids_is_not_counted_as_staged(
    hostids_env,
) -> None:
    """``stage_host_step`` answers whether it staged. The answer is the fact.

    The layer returns False when its own hostids are off or the step is wider than
    its pinned buffer, which means these rows are not in the buffer this step's
    replay reads. The return value used to be dropped and the step counted as
    staged anyway -- the same one-line class of slip as the empty layer list, and
    invisible for exactly the same reason: nothing downstream consumed the answer.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1
    assert st._ple_host_layer_steps() == 1

    emb.disable_hostids("test: the layer stopped taking steps")
    drafts.bump_generation(11)
    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_steps == 1, "a step that staged nothing was counted"
    assert st._ple_host_layer_steps() == 1
    assert st._ple_host_refusals.get("layer-declined-staging") == 1
    coverage = st._ple_host_coverage()
    assert "staged_steps=1" in coverage
    # A layer that stopped taking steps is a boot that is not staging, whatever
    # the worker's own knob says.
    assert "disabled=yes" in coverage


def test_resync_never_runs_inside_a_capture(hostids_env, monkeypatch) -> None:
    """A device sync inside a capture aborts it, and that kills the boot."""
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    monkeypatch.setattr(
        model_state_module.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_syncs == 0
    assert st._ple_host_steps == 0
    assert "capturing" in st._ple_resync_refused
    assert st._ple_host_refusals.get("history-length") == 1


def test_refusal_detail_says_which_side_of_the_count_is_wrong(
    hostids_env, caplog
) -> None:
    """Boot #26 raised 411 of these and could not answer the question it posed.

    The exception carried ``positions [...] != committed-1 [...]`` and only
    ``exc.reason`` was logged, so the boot could establish that the mirror and the
    scheduler disagreed on every step and could not say which one was wrong or by
    how much. ``grep -c committed-1`` on that log is zero.
    """
    import logging

    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(input_batch, st)
    # No repair possible: the budget is gone, so the refusal reaches the log. The
    # cap has to be asked for -- the shipped default is uncapped (see
    # PLE_RESYNC_PER_SLOT), so restating the class value here would restate 0 and
    # let the repair run instead of refusing.
    st.PLE_RESYNC_PER_SLOT = 2
    st._ple_resync_slots.update({0: st.PLE_RESYNC_PER_SLOT, 3: st.PLE_RESYNC_PER_SLOT})
    with caplog.at_level(logging.INFO):
        st._stage_host_ple_ids(input_batch, req_states)
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "committed-1" in line, line
    # And the line that counts refusals now has the denominator next to it.
    assert "staged_steps=" in line and "dropped=" in line


def test_replaying_a_stale_draft_block_refuses(hostids_env) -> None:
    """A draft block is consumed once: the same generation is last step's tokens.

    This is the one property of the mirror that no shape check can infer. Two
    consecutive decode steps over an unchanged batch produce an IDENTICAL
    slot->row map with different values, so a map that lines up is not evidence
    of freshness -- and a block that is one step old produces exactly the
    signature the 2026-10-01 14:10 boot showed: the rows whose tokens come from
    history and draft column 0 agree with the device, the row that needs column
    1 does not.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    src = _drafts()  # generation fixed at 1, as an epilogue that staged nothing
    st.set_host_draft_source(src)
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1, "first step should stage"

    emb._host_stage_rows = -1  # the op consumed this step's rows
    st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_host_refusals.get("draft-gen-stale") == 1
    assert emb._host_stage_rows == -1, "a stale block was staged anyway"
    assert st._ple_host_steps == 1

    # The control: the very same batch proceeds the moment a new block exists, so
    # the refusal above is about the generation and not about "second step ever".
    src.bump_generation(src.wait_host_draft_tokens()[2] + 1)
    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 2, st._ple_host_refusals


def test_mismatch_dump_names_its_inputs(caplog) -> None:  # noqa: ANN001
    """A mismatch that cannot name its tokens is not a diagnosis.

    The arithmetic is proven against the device separately
    (tools/gfx906/cpu-diff-ngram.py: 8 shapes, including the failing one, all match),
    so what a production mismatch can mean is "the inputs differed". That is only
    falsifiable from a log line if the line carries both sides of the inputs and
    the draft block the host built from.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[ROWS - 1, HEADS - 1] = -1  # the LAST head: a [:4] dump would not show it
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ids_arr = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    qsl_arr = np.array([0, 3, 6], dtype=np.int64)
    ctx_arr = np.arange(ROWS * CTX, dtype=np.int64).reshape(ROWS, CTX)
    emb._host_check_inputs = (ids_arr, qsl_arr, ctx_arr)
    emb.note_draft_provenance(
        np.array([[11, 12, 999], [14, 15, 998]], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
        7,
    )

    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(ids_arr),
            torch.from_numpy(qsl_arr),
            torch.from_numpy(ctx_arr),
        ),
    )

    assert emb._host_input_checks == 1, "the dump ran on a comparison nobody attributed"
    assert emb._host_id_mismatches == 1
    assert emb._hostids_disabled is True
    text = "\n".join(r.getMessage() for r in caplog.records)
    for marker in (
        "first mismatch at 6 rows",
        # The dump now has to admit which witness it consulted: a line saying the
        # inputs agreed is what separates "the formula is wrong" from "the inputs
        # differed and nobody could tell".
        "input gate: agreed",
        "inputs host=",
        "device=",
        "qsl host=",
        "ctx host=",
        "draft gen=7",
        "rows differing=[5]",
        # The heads that actually differ, not the first four: the 14:54 boot
        # truncated every row to [:4] and its second bad row therefore looked
        # identical on both sides in the line meant to explain it.
        f"row {ROWS - 1} differs in 1 head(s) [{HEADS - 1}]",
    ):
        assert marker in text, f"missing {marker!r} in:\n{text}"
    # Consumed, so a later step cannot be compared against this step's inputs.
    assert emb._host_check_dev_inputs is None


def test_a_dump_that_cannot_name_the_device_side_says_so(
    caplog,
) -> None:
    """Silence about a missing half is worse than a line admitting it.

    The 14:54 dump dropped the input half without comment, which is why it read
    as a complete diagnosis for a whole round of analysis. A dump that is missing
    a side has to say which one.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[1, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (
        np.arange(ROWS, dtype=np.int64),
        np.arange(ROWS + 1, dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )

    emb.compare_host_ids(torch.from_numpy(dev_ids))  # no device inputs anywhere

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "inputs missing on the device side" in text, text
    assert "inputs host=" not in text, text
    # P1.10b changed what "cannot name the device side" costs: it used to disable the
    # knob, which meant a harness with no visibility took the feature away. Now it
    # refuses, so the mirror is neither blamed nor cleared.
    assert emb._hostids_disabled is False
    assert emb._host_id_mismatches == 0
    assert emb._host_input_skips == 1


def test_the_dump_prints_the_heads_that_differ_not_the_first_four(
    caplog,
) -> None:
    """``[:4]`` is only informative when the disagreement starts at head 0.

    The 14:54 boot reported two bad rows and printed four columns of each, so
    the row whose disagreement lived past column 3 looked identical on both
    sides -- the log line said "row 3 differs" and then showed row 3 matching.
    Eight heads here, disagreement at 5 and 7, so a first-four dump cannot pass.
    """
    emb = StagingEmbedding()
    heads = 8
    host = np.zeros((3, heads), dtype=np.int64)
    dev = host.copy()
    dev[2, 5] = 7
    dev[2, 7] = 9

    line = emb._id_dump(3, host, dev, None)

    assert "row 2 differs in 2 head(s) [5, 7]" in line, line
    assert "host row2=[0, 0]" in line, line
    assert "dev  row2=[7, 9]" in line, line
    assert "host row2=[0, 0, 0, 0]" not in line, line


def test_staging_captures_the_device_side_of_the_inputs(  # noqa: ANN201
    hostids_env, caplog
):
    """Staging is the only place both sides of the comparison are in scope.

    The 14:54 boot printed the draft half of its mismatch and silently dropped the
    input half. The compare read ``_host_check_dev_inputs`` and the only writer was
    ``forward()``, whose Python body dynamo runs at trace time and never again on a
    compiled path -- so on the boot that mattered the attribute was whatever tracing
    left behind. Staging is plain Python in the worker, so it hands over the runner's
    own arrays, and the gate reads those.

    What changed in P1.10b is what the compare does with them: it checks them. The
    ids under comparison below are hand-made, so the host half is restated to a step
    that agrees with them -- production stages both halves of the same step, and the
    agreeing path is what a mismatch dump needs. The capture itself is the subject.
    """
    hostids_env["mode"] = 2
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_host_steps == 1, st._ple_host_refusals
    captured = emb._host_check_dev_inputs
    assert captured is not None, "staging did not capture the device inputs"
    assert captured[0] is input_batch.input_ids, (
        "staging captured a copy where it should capture the runner's buffer"
    )
    assert emb._host_check_inputs is not None, "staging kept no host side"

    # Both halves now describe the step whose ids the test hand-builds, so the gate
    # agrees and the dump is the thing under test.
    _agree_inputs(emb)
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[0, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb.compare_host_ids(torch.from_numpy(dev_ids))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "inputs host=" in text, f"the input half is missing again:\n{text}"
    assert "device=" in text, text
    assert "SKIPPED a comparison" not in text, text
    assert emb._host_check_dev_inputs is None, "captured inputs were not consumed"


# ---------------------------------------------------------------------------
# P1.10b: a comparison is attributed by what the two sides read, not by counting
# entries into the op.
#
# P1.9 decided attribution from inside ``forward()``, by stamping an op-entry
# counter and copying the compute's inputs aside as captured graph nodes. Both
# halves were wrong. The counter counts op entries, and under piecewise graphs the
# main pass is graph-replayed and never enters the op: a boot's own dump printed
# ``op entry=42`` against ``stage seq=2``, so "gap == 1" described the harness's
# bookkeeping and not the step. And the snapshot sits inside the compiled region,
# which the AOT cache owns -- the key directory for this shape did not change for
# eight days across dozens of source edits, so whatever was traced is what ran,
# knob or no knob. The one boot that did execute those copies (a drafter-off boot,
# therefore a brand-new key) died of ``hipErrorIllegalAddress`` in CUDA-graph
# memory profiling.
#
# What replaces it asks the question directly, in eager code, at the split point:
# are the arrays the compute read the arrays the host mirrored? If not, there is
# nothing to attribute, so the comparison is refused and counted -- refused, not
# disabled, because a harness that cannot see the inputs has no business taking
# away the feature it is harnessing.
# ---------------------------------------------------------------------------


def test_the_differential_harness_keeps_nothing_in_the_compiled_region() -> None:
    """P1.9's probes stay deleted; this is what keeps them deleted.

    Asserted on the class's surface rather than on source text, because a formatter
    owns the text and dynamo owns ``forward()``. The methods below existed to observe
    the compute from inside the region the AOT cache decides how to run, and the boot
    that first executed one of them is the boot that crashed.
    """
    for gone in (
        "_snapshot_compute_inputs",
        "_read_input_snapshot",
        "_alloc_snap_scratch",
        "_begin_invocation",
    ):
        assert not hasattr(Qwen4ExpNGramEmbedding, gone), gone


def test_a_disagreement_in_the_inputs_refuses_the_comparison() -> None:
    """Two halves that describe different tokens are not a comparison at all.

    Boot #6's device ids invert -- by CRT, exactly -- to a token that appears nowhere
    in the staged step, because under MTP four passes per step write their own rows
    into one static ids buffer. Against the row-count gate that arrived as
    ``first mismatch at 6 rows`` and disabled a mirror which every offline shape
    test since has proven correct.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4  # the token the other MTP pass had put in the runner's array
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(dev_ids), emb._host_check_dev_inputs)

    assert emb._host_input_skips == 1, "the input-agreement gate did not fire"
    assert emb._host_id_mismatches == 0, "a refused comparison was counted as a verdict"
    assert emb._hostids_disabled is False, (
        "a harness limitation disabled the feature it was harnessing"
    )


def test_the_refusal_names_the_position_and_both_values(caplog) -> None:
    """ "Inputs differed" is only evidence if a reader can check it.

    The CRT work on boot #6 turned a dump into an exact token pair because the line
    carried the values; a refusal that said only "inputs differ" would need the same
    inversion all over again, this time to prove the gate was right.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "SKIPPED a comparison" in text, text
    assert "input_ids differ at 1 of 6: device=4 host=101" in text, text
    # The other half of the honesty rule: the line says what the gate checked. It
    # used to confess that the window was never compared; P1.11 compares it when the
    # worker hands it over, and counts the steps where it does not.
    assert "ngram context is never compared" not in text, text
    assert "handed over no device window" in text, text


def test_the_refusal_prints_both_id_arrays_so_rows_can_be_attributed(caplog) -> None:
    """P1.13: a refusal has to name rows, not only a position.

    Boot #11's single sign of the per-pass patchwork was
    ``input_ids differ at 0 of 8 ... (2 of 8 positions)``. That proves the ids buffer
    the op read is not the one the worker staged, but it cannot say *which row* is
    foreign, and "which row was authored by which pass" is the only question the
    differential has left. Rows are contiguous groups of ``heads_per_ngram`` in the
    flat array, so printing both arrays turns the refusal into the map -- and it is
    free at the only size this path runs (a compared step is pure decode, so the
    array is ``requests * (num_spec + 1)`` long).
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    other = ran_ids.copy()
    other[1] = 4
    other[4] = 7
    emb._host_check_dev_inputs = (
        torch.from_numpy(other),
        torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "(2 of 6 positions)" in text, text
    assert "device_ids=[100, 4, 102, 103, 7, 105]" in text, text
    assert "host_ids=[100, 101, 102, 103, 104, 105]" in text, text


def test_the_id_array_render_is_bounded_and_says_so() -> None:
    """A diagnostic on this path must not be able to grow with the batch.

    The cap is only honest if the line admits it fired: a truncation that looks like
    the whole array is worse than no array, because the reader counts rows in what
    they can see.
    """
    short = ple_layer_module._bounded_ids(np.arange(6))
    assert short == "[0, 1, 2, 3, 4, 5]", short
    long_ = ple_layer_module._bounded_ids(np.arange(20), cap=16)
    assert long_.endswith(", ... +4 more]"), long_
    assert ", 16]" not in long_, long_


def test_a_padded_query_boundary_beyond_the_host_rows_is_not_a_disagreement() -> None:
    """The gate has to be able to go green, or every refusal test is vacuous.

    The runner's ``query_start_loc`` is padded to the batch capacity and the host's
    is the rows it built, so the two arrays are routinely different lengths and
    agreement means equal over the rows. A gate that refused that shape would refuse
    every captured step -- and would pass every red control written for it.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (
        ran_ids,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    emb._host_check_dev_inputs = (
        torch.from_numpy(ran_ids),
        torch.tensor([0, 3, 6, 6, 6], dtype=torch.long),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    assert emb._host_input_checks == 1, (
        f"refused a padded but agreeing step: skips={emb._host_input_skips}"
    )
    assert emb._host_input_skips == 0
    assert emb._hostids_disabled is False


def test_a_device_array_shorter_than_the_staged_rows_is_not_a_green_light(
    caplog,
) -> None:
    """A prefix compare of a short array compares nothing and reports agreement.

    The runner's array is the padded one, so being shorter than the staged rows
    means the two sides are not describing the same work. Slicing to the shorter
    length -- which is what a careless equality check does -- would call that a
    match and hand the mismatch a green light it never earned.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (
        np.arange(ROWS, dtype=np.int64) + 100,
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )
    emb._host_check_dev_inputs = (
        torch.from_numpy(np.arange(3, dtype=np.int64) + 100),
        torch.from_numpy(np.array([0, 3], dtype=np.int64)),
        None,
    )

    emb.compare_host_ids(torch.from_numpy(host_ids.copy()), emb._host_check_dev_inputs)

    assert emb._host_input_skips == 1
    assert emb._host_input_checks == 0
    assert emb._hostids_disabled is False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "shorter than the staged rows (3 < 6)" in text, text


def test_the_gate_is_not_a_blanket_refusal() -> None:
    """A comparison the harness *can* attribute still has to reach a verdict.

    Every refusal test above is also satisfied by a compare that never compares,
    which is how a diagnostic ends up believing its own silence. The mirror being
    correct is proven elsewhere; what this proves is that a real mismatch is still
    a real finding.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    _agree_inputs(emb)
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS

    emb.compare_host_ids(torch.from_numpy(dev_ids), emb._host_check_dev_inputs)

    assert emb._host_input_checks == 1, "the agreeing step was never compared"
    assert emb._host_input_skips == 0
    assert emb._host_id_mismatches == 1
    assert emb._hostids_disabled is True


def test_the_device_query_boundaries_are_printed_whole(caplog) -> None:
    """A padded qsl is a suspect, so it cannot be printed cut to the host's size.

    The runner pads ``query_start_loc`` to the batch capacity; a padded boundary makes
    the compute split the token stream into requests the host never saw. Slicing the
    array to the host's length, which is what the 15:37 dump did, hides the only
    evidence of that. The gate is satisfied here (equal over the rows, per the test
    above) so the mismatch dump is the thing under test.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[0, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (
        np.arange(ROWS, dtype=np.int64),
        np.array([0, 3, 6], dtype=np.int64),
        np.zeros((ROWS, CTX), dtype=np.int64),
    )

    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(np.arange(ROWS, dtype=np.int64)),
            # Capacity 5, two real requests: the padded boundaries are the
            # thing being looked for.
            torch.tensor([0, 3, 6, 6, 6], dtype=torch.long),
            torch.zeros((ROWS, CTX), dtype=torch.long),
        ),
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "qsl host=[0, 3, 6] device=[0, 3, 6, 6, 6]" in text, text
    assert emb._host_input_checks == 1


# ---------------------------------------------------------------------------
# P1.11: the window leg
#
# Boot #8's CRT inversion put the single wrong term exactly in the cell the
# per-request ngram window feeds: the host mirrored 1156 where the device read 4.
# Every other input agreed, and the gate could not say anything about the window
# because it had convinced itself the worker had no device-side counterpart for
# it. It has one -- Qwen4ExpModelState owns the buffer, fills it, and hands a view
# of it to the forward -- so these tests pin the third leg: window agreement is
# required for a comparison to run, a window refusal is refused rather than
# blamed, and a window nobody handed over is announced instead of assumed.
# ---------------------------------------------------------------------------


def _window_case(
    emb: StagingEmbedding,
    *,
    host_ctx: np.ndarray,
    dev_ctx: Any,
    qsl: Any = None,
    ids_differ: bool = False,
) -> None:
    """Line up ids and qsl so only the window can disagree.

    The gate checks ids and query boundaries first and returns on the first
    disagreement, so a window test that also had a token difference would be
    testing the token check. Everything here is equal except what is passed in.
    """
    ids = np.arange(ROWS, dtype=np.int64) + 100
    bounds = np.array([0, 3, 6], dtype=np.int64) if qsl is None else qsl
    ngram = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    if ids_differ:
        ngram = ngram.copy()
        ngram[2, 0] = -1
    emb._host_check_ids = torch.from_numpy(ngram)
    emb._host_check_rows = ROWS
    emb._host_check_inputs = (ids, bounds, host_ctx)
    emb._host_check_dev_inputs = (
        torch.from_numpy(ids.copy()),
        torch.from_numpy(np.asarray(bounds, dtype=np.int64)),
        dev_ctx,
    )


def test_a_window_the_device_does_not_have_refuses_the_comparison() -> None:
    """One cell of the window differs: the host mirrored a different prompt.

    The ids the host computed are then a function of tokens the device never read,
    so calling the difference a mirror bug would be blaming the wrong suspect --
    which is precisely the mistake the boot #6 analysis almost made.
    """
    emb = StagingEmbedding()
    host_ctx = np.zeros((ROWS, CTX), dtype=np.int64)
    dev_ctx = np.zeros((CAPACITY, CTX), dtype=np.int64)
    dev_ctx[1, 0] = 4  # the boot #8 shape: a stale cell where the host has a token
    _window_case(emb, host_ctx=host_ctx, dev_ctx=torch.from_numpy(dev_ctx))

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_skips == 1, "the window leg did not refuse"
    assert emb._host_id_mismatches == 0, "a refused comparison became a verdict"
    assert emb._host_id_matches == 0
    assert emb._hostids_disabled is False


def test_the_window_refusal_names_the_request_cell_and_both_values(
    caplog,
) -> None:
    """A refusal a reader cannot check is a refusal that has to be re-proven.

    The window is per request, so the position has to be (request, cell) and not a
    flat index: a flat index would be read as a token row, and the two domains are
    already the most confused pair in this feature.
    """
    emb = StagingEmbedding()
    host_ctx = np.zeros((ROWS, CTX), dtype=np.int64)
    host_ctx[1, 0] = 1156
    dev_ctx = host_ctx.copy()
    dev_ctx[1, 0] = 4
    _window_case(emb, host_ctx=host_ctx, dev_ctx=torch.from_numpy(dev_ctx))

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "ngram_context differs at request 1 cell 0: host=1156 device=4" in text, text


def test_a_window_nobody_handed_over_still_compares_ids_and_says_so(
    caplog,
) -> None:
    """The third leg being dark is not the same as the third leg passing.

    An integration that does not hand over the buffer (a stub, or an older caller)
    must still get the ids and query verdict -- refusing everything would turn a
    missing diagnostic into a disabled feature. But the step is counted, and the
    first one is logged, so nobody reads a green gate as a full one.
    """
    emb = StagingEmbedding()
    _window_case(emb, host_ctx=np.zeros((ROWS, CTX), dtype=np.int64), dev_ctx=None)

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_checks == 1, "the ids verdict was withdrawn with the window"
    assert emb._host_id_mismatches == 0
    assert emb._host_ctx_unchecked == 1
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "the ngram context is NOT compared" in text, text


def test_a_window_with_fewer_requests_than_the_host_refuses(
    caplog,
) -> None:
    """A window for three requests cannot describe a step of six.

    Slicing it to the host's length would compare a prefix of something else and
    call the result agreement -- the same rule the ids buffer applies.
    """
    emb = StagingEmbedding()
    _window_case(
        emb,
        host_ctx=np.zeros((ROWS, CTX), dtype=np.int64),
        dev_ctx=torch.zeros((3, CTX), dtype=torch.long),
    )

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_skips == 1
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "device ngram_context describes 3 requests, the host step has 6" in text


def test_a_window_of_a_different_width_refuses() -> None:
    """A three-cell window is a different ngram_size, not a longer window."""
    emb = StagingEmbedding()
    _window_case(
        emb,
        host_ctx=np.zeros((ROWS, CTX), dtype=np.int64),
        dev_ctx=torch.zeros((CAPACITY, CTX + 1), dtype=torch.long),
    )

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_skips == 1


def test_all_three_legs_agree_and_no_leg_calls_itself_unchecked() -> None:
    """The green path has to be reachable with all three legs present.

    Without this, every refusal test above could be passing because the gate
    refuses everything -- the failure mode that makes a red control worthless.
    """
    emb = StagingEmbedding()
    _agree_inputs(emb)
    # The helper states what both sides read; this is the staged ids buffer the
    # compare runs against, and the row count that says a step is pending. Without
    # them compare_host_ids returns before the gate, which is how this test first
    # failed: a green path that never reached the gate.
    emb._host_check_ids = torch.from_numpy(
        np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    )
    emb._host_check_rows = ROWS

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    assert emb._host_input_checks == 1
    assert emb._host_input_skips == 0
    assert emb._host_ctx_unchecked == 0, "the window leg reported itself unchecked"
    assert emb._host_id_mismatches == 0
    assert emb._host_id_matches == 1, "agreement on inputs did not become a match"


def test_the_dump_prints_the_window_the_device_read(caplog) -> None:  # noqa: ANN
    """ "A dump now shows the window the mismatching rows extended into.

    Boot #8's line printed ``ctx host=[248068, 198] device=n/a``, so the analysis
    had to invert the ids by hand to learn what the device had used for the
    left-context term. Note what this line can and cannot show: a dump only ever
    runs on a comparison the gate attributed, and after P1.11 the gate attributes
    one only when the windows already matched -- so the two halves print equal by
    construction. The point is not that they differ, it is that the reader can see
    the cells the device read instead of taking the host's word for them, and that
    a missing side still says which one is missing. The window is printed whole
    (every request, both cells) because it is per request and tiny; slicing it by
    token row would be a category error.
    """
    emb = StagingEmbedding()
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    host_ctx = np.zeros((2, CTX), dtype=np.int64)
    host_ctx[0] = [248068, 198]
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ran_ids = np.arange(ROWS, dtype=np.int64) + 100
    emb._host_check_inputs = (ran_ids, np.array([0, 3, 6], dtype=np.int64), host_ctx)

    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(ran_ids),
            torch.from_numpy(np.array([0, 3, 6], dtype=np.int64)),
            torch.from_numpy(host_ctx.copy()),
        ),
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "ctx host=[[248068, 198], [0, 0]]" in text, text
    assert "device=[[248068, 198], [0, 0]]" in text, text
    assert "no device window handed over" not in text, text
    assert "input gate: agreed" in text, text


def test_an_unattributed_window_mismatch_never_reaches_the_dump(caplog) -> None:  # noqa: ANN
    """The refuse-not-disable rule, seen from the dump's side.

    A window disagreement means the host mirrored a different prompt, so printing
    the ids diff next to it would put a formula-looking diagnosis beside a sentence
    that says the inputs were different. The comparison is refused and the ids are
    never printed at all.
    """
    emb = StagingEmbedding()
    host_ctx = np.zeros((2, CTX), dtype=np.int64)
    host_ctx[0] = [248068, 198]
    dev_ctx = host_ctx.copy()
    dev_ctx[0] = [248068, 4]
    _window_case(
        emb,
        host_ctx=host_ctx,
        dev_ctx=torch.from_numpy(dev_ctx),
        ids_differ=True,
    )

    emb.compare_host_ids(
        torch.from_numpy(np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)),
        emb._host_check_dev_inputs,
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "first mismatch" not in text, text
    assert "ctx host=" not in text, text
    assert emb._host_id_mismatches == 0


def test_the_worker_hands_the_device_window_over_at_staging_time(monkeypatch) -> None:  # noqa: ANN
    """P1.11 lives or dies in ``model_state``, not in the gate.

    Every other test in this file drives ``compare_host_ids`` with a device tuple it
    built itself, so a production caller that still passed ``None`` for the window
    would leave this file entirely green while the leg stayed dark on every boot --
    the phantom-knob failure mode, this time inside our own harness. So this calls
    the real staging function, with a stand-in ``self``, and asserts the object the
    layer is handed is the state's own buffer, by identity: a copy would be the
    previous step's window, and the comparison would be quietly wrong.
    """
    import types

    from vllm.models.qwen4_exp.amd import model_state as ms
    from vllm.models.qwen4_exp.amd import ple_hostids as ph

    class Recorder:
        def __init__(self) -> None:
            self.staged: list[Any] = []

        def check_host_gen(self) -> None:
            return None

        def note_draft_provenance(self, values, row_idx, gen) -> None:  # noqa: ANN
            return None

        def stage_host_step(  # noqa: ANN
            self, ids, qsl, ctx, *, gather, device_inputs, consumed_qsl=None
        ) -> None:
            # ``consumed_qsl`` is P1.14's addition to the production signature. A
            # stub narrower than production's does not catch a wiring change, it
            # just stops the staging loop with a TypeError -- which the producer
            # swallows by design -- so it has to be restated here when it moves.
            self.staged.append((gather, device_inputs))

    class History:
        pending: list[Any] = []
        window = np.zeros((4, 8), dtype=np.int64)
        length = 0

        def drain(self) -> None:
            return None

    class Drafts:
        def wait_host_draft_tokens(self):  # noqa: ANN
            return np.zeros((2, 3), dtype=np.int64), np.array([0, 1], dtype=np.int64), 1

    layer = Recorder()
    window_buf = torch.zeros((4, CTX), dtype=torch.long)
    self_ = types.SimpleNamespace(
        uses_ngram_embedding=True,
        ngram_context_len=CTX,
        ngram_eos_token_id=EOS,
        _ple_host_layers=[layer],
        _ple_host_gen_layers=[layer],
        _host_drafts=Drafts(),
        _ple_host_history=History(),
        _last_draft_gen=0,
        _ple_host_steps=0,
        # The state's own device buffer. Identity is the assertion below, so this
        # has to be one object, not something that merely equals it.
        ngram_context=window_buf,
        _note_host_refusal=lambda reason: None,
        disable_host_staging=lambda why: None,
        # Option A's policy read, for the stub that borrows the real methods. Mode
        # 2 never runs under a FULL graph (the differential arm keeps the blocking
        # D2H, which is un-capturable), so the step answers not-FULL and the policy
        # cannot fire; it only has to be present for the attribute lookup.
        PLE_HOST_STAGE_UNDER_FULL=(
            model_state_module.Qwen4ExpModelState.PLE_HOST_STAGE_UNDER_FULL
        ),
        _ple_step_cg_mode=None,
        _ple_step_full=lambda: False,
    )
    # A SimpleNamespace does not run the descriptor protocol, so a plain function put
    # in it arrives without ``self``; bind it to the namespace after construction.
    # Borrowed rather than stubbed: it walks ``_ple_host_layers`` and calls
    # ``forget_host_stage`` only where a layer has one, so against this recorder it is
    # a no-op, and if the producer ever needs more of ``self`` than this carries, the
    # AttributeError is the honest answer.
    self_._forget_host_stage_flags = lambda: (
        model_state_module.Qwen4ExpModelState._forget_host_stage_flags(self_)
    )
    monkeypatch.setattr(ph, "hostids_mode", lambda: 2)
    monkeypatch.setattr(
        ms,
        "build_decode_step_inputs",
        lambda *a, **k: (
            np.arange(6, dtype=np.int64),
            np.array([0, 3, 6], dtype=np.int64),
            np.zeros((2, CTX), dtype=np.int64),
        ),
    )

    ms.Qwen4ExpModelState._stage_host_ple_ids(
        self_,
        types.SimpleNamespace(
            num_reqs=2,
            input_ids=torch.zeros(CAPACITY, dtype=torch.long),
            query_start_loc=torch.zeros(3, dtype=torch.long),
            idx_mapping_np=np.array([0, 1], dtype=np.int64),
            num_scheduled_tokens=[3, 3],
        ),
        types.SimpleNamespace(num_computed_tokens_np=np.array([4, 4], dtype=np.int64)),
    )

    assert len(layer.staged) == 1, "the staging loop never ran"
    gather, device_inputs = layer.staged[0]
    assert gather is False, "mode 2 must not gather"
    assert device_inputs is not None, "no device inputs were handed over"
    assert device_inputs[2] is window_buf, (
        "the gate was not handed the state's own ngram_context buffer"
    )


# ---------------------------------------------------------------------------
# P1.12 -- the dump reads the compute's own packed workspace
# ---------------------------------------------------------------------------
#
# Why this leg exists: the q=3 boot (2026-10-02 11:53) is the first step where
# ids, query_start_loc AND the ngram window were all proven identical on both
# sides, and a faithful CPU replay of _compute_ngram_ids on those exact inputs
# reproduced the HOST ids exactly -- not the device's. So the arrays the caller
# can see do not explain the array the device produced, and what is left is
# padded_buffer: the compute's own scratch, written from input_ids inside the
# compiled piece and by nobody else. Its address space is the only remaining
# witness.


def _mismatch_row2_with_workspace(
    emb: StagingEmbedding, packed_cell: int, *, four: bool = True
) -> None:
    """Drive one attributed mismatch at row 2 with a workspace under our control.

    Row 2 of a ``[0, 3, 6]`` batch is request 0, column 2, so -- by the same
    ``context = cat([ngram_context, packed])`` and ``adjusted_columns`` arithmetic
    the compute uses -- the ngram=2 group reads its left context from
    ``packed[0, 1]`` and the ngram=3 group from ``packed[0, 0]``.
    """
    host_ids = np.arange(ROWS * HEADS, dtype=np.int64).reshape(ROWS, HEADS)
    dev_ids = host_ids.copy()
    dev_ids[2, 0] = -1
    emb._host_check_ids = torch.from_numpy(host_ids)
    emb._host_check_rows = ROWS
    ids_arr = np.array([10, 11, 12, 14, 15, 16], dtype=np.int64)
    qsl_arr = np.array([0, 3, 6], dtype=np.int64)
    ctx_arr = np.arange(ROWS * CTX, dtype=np.int64).reshape(ROWS, CTX)
    packed = torch.full((4, 64), EOS, dtype=torch.long)
    packed[0, 0], packed[0, 1], packed[0, 2] = 10, packed_cell, 12
    packed[1, 0], packed[1, 1], packed[1, 2] = 14, 15, 16
    emb._host_check_inputs = (
        (ids_arr, qsl_arr, ctx_arr, packed) if four else (ids_arr, qsl_arr, ctx_arr)
    )
    emb.compare_host_ids(
        torch.from_numpy(dev_ids),
        (
            torch.from_numpy(ids_arr),
            torch.from_numpy(qsl_arr),
            torch.from_numpy(ctx_arr),
        ),
    )


def test_the_dump_names_the_workspace_cell_that_fed_the_differing_row(
    caplog,
) -> None:
    """The dump must say which cell of the compute's scratch fed the bad row.

    "The inputs agreed and the ids still differ" is only actionable if the line
    also reports what the hash read out of its own workspace at that position:
    the token means the arithmetic is the suspect, anything else means the
    compute consumed an ids tensor that is not the one the op was handed.
    """
    emb = StagingEmbedding()
    with caplog.at_level("WARNING"):
        _mismatch_row2_with_workspace(emb, 11)
    assert emb._hostids_disabled is True, "the comparison never reached its dump"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "packed device=" in text, text
    assert "row2ngram2=packed[0,1]=11" in text, text
    assert "row2ngram3=packed[0,0]=10" in text, text


def test_the_packed_leg_reports_the_workspace_and_not_the_hope(caplog) -> None:
    """Red control for the leg above: mutate the cell, the line must change.

    A leg that prints a value derived from the caller's arrays would pass the
    test above and tell us nothing at 3 a.m. This one only earns its keep by
    tracking what the scratch actually holds.
    """
    emb = StagingEmbedding()
    with caplog.at_level("WARNING"):
        _mismatch_row2_with_workspace(emb, 3)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "row2ngram2=packed[0,1]=3" in text, text
    assert "row2ngram2=packed[0,1]=11" not in text, text


def test_the_packed_leg_names_its_own_absence(caplog) -> None:
    """A leg that could not run must not read as a leg that agreed."""
    emb = StagingEmbedding()
    with caplog.at_level("WARNING"):
        _mismatch_row2_with_workspace(emb, 11, four=False)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "packed=n/a(stager handed no workspace)" in text, text


def test_staging_hands_the_dump_the_layer_s_own_workspace(hostids_env) -> None:
    """Production wiring, not a hand-built tuple.

    Every other test here assembles ``_host_check_inputs`` itself, so none of
    them can notice the stager forgetting to pass the workspace -- and a dump
    whose new leg is always ``n/a`` is the phantom-metric failure in its purest
    form: a reassuring absence that looks like a working instrument.
    """
    hostids_env["mode"] = 2
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)

    assert emb._host_check_inputs is not None, "staging kept no host side"
    assert (
        len(emb._host_check_inputs) > 3
        and emb._host_check_inputs[3] is emb.padded_buffer
    ), "the dump was not handed the workspace the compute writes into"


# ---------------------------------------------------------------------------
# The 2026-10-05 crashes: boots #29 and #30 both died inside the resync, on a
# cross-check that compared two things sitting at different absolute positions and
# then charged the guaranteed mismatch to the host's ability to read its own memory.
# The log line was ``positions [97209, 15525] != committed-1 [97206, 15525]`` -- the
# device exactly ``num_speculative_tokens`` ahead of the mirror -- and the same
# string reappeared on all four strikes, one slot, four consecutive steps. Four was
# ``PLE_DISAGREE_TOTAL``, so the disagreement budget became a timer,
# ``disable_host_staging`` ran, ``PLE_DOUBT_UNDER_FULL`` was ``"stop-boot"``, and a
# ``PLEHostDoubtUnderFullGraph`` raised through a TP worker.
# ---------------------------------------------------------------------------


def test_a_window_three_cells_behind_the_device_is_a_lag_not_a_disagreement(
    hostids_env,
) -> None:
    """The crash shape, pinned: real tokens, wrong offset, zero strikes.

    The mirror's length claims the window's last cells sit at ``old_len - overlap ..
    old_len - 1``. A dropped commit leaves that claim false by exactly the number of
    tokens the device has moved on, and the window then matches the device's own row
    once the comparison is moved. That is the coverage bug ``mirror(dropped=)``
    already counts, and the repair (adopt the row) is the right answer to it. It is
    not a host that cannot vouch for its reads.

    Called directly rather than through ``_stage_host_ple_ids`` for the same reason
    the mirror-ahead test is: in mode 1 the input gate refuses a counters-only
    mismatch before the history check runs, so an end-to-end fixture here would
    assert on the gate. The comparison lives in the repair, and boots #29 and #30 are
    the evidence that a real boot reaches it.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_layers = [emb]
    st.set_host_draft_source(_drafts())
    input_batch, req_states, tokens = _batch_and_states()

    hist = st._ple_host_history
    # Distinct, non-repeating cells, so the only shift that can explain a window is
    # the one this test puts there.
    for cell in range(12):
        tokens[0, cell] = 101 + cell
    req_states.all_token_ids.gpu.copy_(torch.from_numpy(tokens))

    # The mirror holds six committed cells; the device says the last committed
    # position is 8, i.e. three further on than the mirror's length implies.
    hist.seed(0, np.arange(101, 107, dtype=np.int64))
    # ... and its newest cells are the row's cells 0..2, three behind the cells
    # 3..5 its length claims. Real tokens of this request, at the wrong offset.
    hist.window[0, hist.width - 3 :] = [101, 102, 103]
    req_states.num_computed_tokens_np[0] = 8

    repaired = st._ple_resync_mirror(req_states, input_batch, "test lag-by-three")

    assert repaired is True, "a lag is repairable; the caller was told not to retry"
    assert int(hist.length[0]) == 9, "the row was not adopted (want = positions + 1)"
    assert st._ple_resync_mirror_offset == 1, [
        "the shift that explains the window was not counted on its own ledger"
    ]
    assert st._ple_resync_disagree == 0, [
        "a window that lags by num_speculative_tokens was charged as a disagreement "
        "with the device's own token row. That is the line that stopped boots #29 "
        "and #30, and the two crashes cost the operator two restarts."
    ]
    assert st._ple_resync_disagree_slots.get(0) is None
    assert emb._hostids_disabled is False
    assert st._ple_host_layers == [emb], "a lag took the knob off the whole boot"


def test_the_offset_test_refuses_to_explain_a_value_that_is_not_a_token(
    hostids_env,
) -> None:
    """Boot #14's corruption must still stop the boot.

    ``host=4161103993993705554`` where a token id belongs matches nothing at any
    offset, so the A3 stop-boot still fires on the thing it was written for. F1
    narrows the disagreement budget; it does not disarm it.
    """
    st = FakeModelState()
    row = np.arange(101, 107, dtype=np.int64)
    mirrored = np.array([4161103993993705554 % (2**31), 102, 103], dtype=np.int64)

    assert st._ple_offset_explains(mirrored, row, 6) is None


def test_the_offset_test_does_not_read_past_the_committed_cells(
    hostids_env,
) -> None:
    """A shift is only an explanation if the cells it points at are committed.

    The same window is explainable when the row is long enough to contain it and
    unexplainable when it is not: an explanation that could reach into the
    uncommitted tail of the row would excuse anything.
    """
    st = FakeModelState()
    cells = np.arange(101, 115, dtype=np.int64)
    mirrored = np.array([106, 107, 108], dtype=np.int64)

    # Cells 5..7 of the row: two cells past the end of a six-cell row, so the only
    # legal shifts point backwards, and none of them matches.
    assert st._ple_offset_explains(mirrored, cells[:6], 6) is None
    # Hand the same window a row that actually holds those cells and it explains.
    assert st._ple_offset_explains(mirrored, cells[:8], 6) == 2


def test_the_same_comparison_repeated_does_not_take_the_knob_off(
    hostids_env,
) -> None:
    """A stuck slot is one event, not a strike per step.

    Both crashes reached ``PLE_DISAGREE_TOTAL`` on one slot's identical comparison
    re-reported four steps running: about a tenth of a second of decode. Repeats
    still refuse the step -- under a captured graph that costs the same coverage a
    disable would -- but they do not accumulate into a boot-level decision.

    Direct calls, because this is the accounting in ``_note_resync_disagreement``;
    driving it end to end would exercise the same comparison the test above pins.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_layers = [emb]
    mirrored = np.array([7, 11, 13], dtype=np.int64)
    row_cells = np.array([7, 11, 19], dtype=np.int64)

    for _ in range(8):
        st._note_resync_disagreement(0, mirrored, row_cells, 3)

    assert st._ple_resync_disagree == 1, "the same comparison was counted as new"
    assert st._ple_resync_disagree_repeat == 7
    assert st._ple_resync_disagree_stuck[0] == 8
    assert st._ple_resync_disagree_slots[0] == 1
    assert emb._hostids_disabled is False
    assert st._ple_host_layers == [emb], [
        "eight steps of one stuck slot disabled the boot: this is the accounting "
        "that killed #29 and #30, which reached the budget at four"
    ]


def test_one_stuck_slot_is_bounded_in_steps_not_strikes(hostids_env) -> None:
    """Persistence still counts, on a step scale where it means something.

    A slot that refuses the same comparison for ``PLE_DISAGREE_STUCK_STEPS`` steps is
    a mirror that will never agree, and under a captured graph every one of those
    steps replays stale PLE rows. That is a boot-level fact and has to be able to
    stop the boot -- otherwise the availability fix has quietly bought availability
    of wrong answers, which is the trade boot #28 already made and E1 exists to undo.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_layers = [emb]
    mirrored = np.array([7, 11, 13], dtype=np.int64)
    row_cells = np.array([7, 11, 19], dtype=np.int64)

    for _ in range(FakeModelState.PLE_DISAGREE_STUCK_STEPS - 1):
        st._note_resync_disagreement(0, mirrored, row_cells, 3)
    assert st._ple_host_layers == [emb], "the budget fired early"

    st._note_resync_disagreement(0, mirrored, row_cells, 3)
    assert st._ple_host_layers == [], "a permanently stuck mirror kept the knob on"
    assert st._ple_resync_disagree_stuck[0] == FakeModelState.PLE_DISAGREE_STUCK_STEPS
    assert st._ple_resync_disagree == 1, "the stuck budget is not a strike budget"


def test_the_decline_line_names_the_graph_mode_it_is_refusing(
    hostids_env, caplog
) -> None:
    """No more "The step goes to the device path" on a step that has none.

    All four declines logged in boot #29 were on ``cg_mode=FULL`` steps, where the
    op body does not run at replay and there is nothing to fall back to. The sentence
    described a captured graph's frozen PLE as handled.
    """
    import logging

    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    mirrored = np.array([7, 11, 13], dtype=np.int64)
    row_cells = np.array([7, 11, 19], dtype=np.int64)

    full = FakeModelState()
    full._ple_host_layers = [emb]
    # The enum here and the string below on purpose: ``prepare_inputs`` stores
    # ``cudagraph_mode.name``, but a caller that stored the enum instead must not
    # silently get the "there is a device path" sentence on a captured graph.
    full._ple_step_cg_mode = CUDAGraphMode.FULL
    with caplog.at_level(logging.WARNING):
        full._note_resync_disagreement(0, mirrored, row_cells, 3)
    assert "NO device path" in caplog.text
    assert "The step goes to the device path." not in caplog.text
    assert "cg_mode=FULL" in caplog.text

    piecewise = FakeModelState()
    piecewise._ple_host_layers = [emb]
    piecewise._ple_step_cg_mode = "PIECEWISE"
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        piecewise._note_resync_disagreement(0, mirrored, row_cells, 3)
    assert "The step goes to the device path." in caplog.text
    assert "NO device path" not in caplog.text


def test_the_shipped_doubt_policy_is_zero() -> None:
    """``stop-boot``, ``degrade`` and ``eager`` are all opt-in now, and each reason is measured.

    I shipped A3 with ``"stop-boot"`` as the default on the reasoning that a boot which
    stops is recoverable and a wrong answer is not. The reasoning had a hole in it:
    under a captured FULL graph the refused steps replay stale rows **under both
    settings**, because there was no fallback either way. So the constant never chose
    between correct and wrong; it chose between correct-but-down and up-but-stale.

    That is a reasonable trade while disagreements are checker false alarms. It stopped
    being one when they turned out to be real corruption in the mirror's committed
    content -- boot #32, ``mirrored [1, 21474836484, 866]``, where 21474836484 is
    ``(5 << 32) + 4`` -- and the setting took the box down on three consecutive
    attempts. A correctness policy that guarantees an outage is not a correctness
    policy; it is an outage.

    ``eager`` replaced it on 2026-10-08 morning, on the theory that a post-disable step
    should compute its own rows on the device instead of replaying someone else's. That
    theory was right and unaffordable: the veto moves the WHOLE forward out of the
    graph, and on this launch-bound box that is ~3.1x per step (measured the same day,
    boot ``vllm.log.20261008-083018``), not the +15 % E1 Phase 0 had quoted for the
    gather alone. At tokens/s = acceptance / step_cost, eager serves fewer tokens than
    the stale replay it replaced at any acceptance the drafter can reach -- break-even
    is 6.2, the ceiling is 4.0.

    ``zero`` is the policy that survives: never downgrade, and make the graph replay
    rows this worker zeroed once at the trip, so the absence of a PLE contribution is
    the model's documented degradation path instead of another request's rows. Same
    shape as the option-A boot that has served all day; the only change is when the
    rows get zeroed.

    Pinned as a test because a default is exactly the kind of thing that gets
    "restored" by someone reading the docstring that used to justify it.
    """
    assert (
        model_state_module.Qwen4ExpModelState.PLE_DOUBT_UNDER_FULL == "zero"
    ), (
        "the default regressed: a post-disable FULL step would replay the previous "
        "step's pinned rows (degrade) or leave the graph (eager, 3.1x)"
    )
    # Precedence, in the same assertion group: the decline sits ABOVE this check, so an
    # option-A boot never reaches the zero-at-trip path and never goes fully eager.
    assert model_state_module.Qwen4ExpModelState.PLE_HOST_STAGE_UNDER_FULL == "off"
    # And the other settings still exist and still work; they are exercised by
    # test_a_doubt_after_capture_stops_the_boot_instead_of_serving (stop-boot) and
    # test_a_degrade_disable_still_replays_after_a_disable (the old default).
    assert model_state_module.Qwen4ExpModelState.PLE_DOUBT_UNDER_FULL != "stop-boot"


def test_a_declined_resync_on_a_full_step_is_counted_as_a_stale_step(
    hostids_env,
) -> None:
    """The decline route used to be missing from the staleness ledger entirely.

    Boot #32 declined three FULL steps and ``stale`` never appeared in its coverage
    line, because only the refusal path incremented the counter. Every one of those
    declines replayed the previous step's PLE rows -- the exact thing the counter
    exists to decide the shippability of -- and the ledger said nothing.

    Counted once per step, like the refusal route: three slots declining on one step
    is one stale step.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    mirrored = np.array([7, 11, 13], dtype=np.int64)
    row_cells = np.array([7, 11, 19], dtype=np.int64)

    full = FakeModelState()
    full._ple_host_layers = [emb]
    full.begin_step(CUDAGraphMode.FULL)
    full._note_resync_disagreement(0, mirrored, row_cells, 3)
    assert full._ple_host_stale_steps == 1, "a declined FULL step served stale and said nothing"

    # Two more slots decline on the same step: still one stale step.
    full._note_resync_disagreement(1, mirrored, row_cells, 3)
    full._note_resync_disagreement(2, mirrored, row_cells, 3)
    assert full._ple_host_stale_steps == 1, "one step counted three times"

    # The next step is PIECEWISE, where the device path really does exist.
    full.begin_step(CUDAGraphMode.PIECEWISE)
    full._note_resync_disagreement(3, mirrored, np.array([1, 2, 3], dtype=np.int64), 3)
    assert full._ple_host_stale_steps == 1, "a piecewise step is not a stale replay"

    # And the counter is on the coverage line, or it is not an instrument.
    assert "stale_steps=1" in full._ple_host_coverage()


# ---------------------------------------------------------------------------
# Option A -- Qwen4ExpModelState.PLE_HOST_STAGE_UNDER_FULL (2026-10-05)
#
# Boot #36 measured the alternative: 20 staged steps out of 756 (2.6 %), and a
# single-stream request with no co-tenant refused ~190 of its ~197 steps, because
# a refusal skips staging while staging is the only thing that advances the mirror.
# The policy below declines rather than attempts, on FULL-graph steps only. What
# these tests hold: declining must not look like refusing (different counters, and
# the refusal path logs), must not reach the tripwire the policy invalidates, must
# not touch non-FULL steps, and must be reversible by one class attribute.
# ---------------------------------------------------------------------------


def _armed_state(hostids_env):
    """Arm registered, draft source wired, one pure-decode batch ready to stage."""
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()
    return st, emb, input_batch, req_states


def test_host_staging_off_under_full_is_the_shipped_default() -> None:
    """The default is the finding, so a silent revert to "stage" must be visible.

    Asserted on the class attribute rather than on behaviour because the failure
    mode this guards is a merge that restores the old line: the behaviour tests
    below monkeypatch the attribute and would keep passing either way.
    """
    assert (
        model_state_module.Qwen4ExpModelState.PLE_HOST_STAGE_UNDER_FULL == "off"
    ), "option A regressed: FULL steps would attempt staging again"


def test_policy_off_declines_a_full_step_rather_than_refusing_it(hostids_env) -> None:
    """The distinction the whole policy rests on, as two numbers.

    A refusal means "we tried and lost" and comes with a log line and a strike
    budget. A decline is a decision, so it must land in its own counter, must not
    consume a refusal reason, and must still count the step as stale -- the served
    PLE rows are not this step's rows either way, and understating that is the one
    error this policy is not allowed to make.
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)

    st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)

    assert st._ple_host_policy_off == 1
    assert st._ple_host_refusals == {}, f"decline logged as refusal: {st._ple_host_refusals}"
    assert st._ple_host_steps == 0, "nothing was staged"
    assert emb.lookup_calls == 0, "the layer was never handed rows"
    assert st._ple_host_stale_steps == 1, "a declined step served stale and said nothing"


def test_policy_off_never_reaches_the_check_host_gen_tripwire(hostids_env) -> None:
    """The tripwire's premise is "we tried to keep the handoff fresh".

    Under this policy every FULL step is stale by design, so a tripwire that
    raises on a stale handoff would end the boot on step one. The decline has to
    sit above it, not below it -- which is the one thing a behavioural test can
    actually prove about the ordering inside ``_stage_host_ple_ids``.
    """
    class Doubtful(StagingEmbedding):
        def check_host_gen(self):
            raise model_state_module.PLEHostDoubtUnderFullGraph(
                "the captured graph holds rows this step did not stage"
            )

    emb = Doubtful()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _ = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)
    assert st._ple_host_policy_off == 1

    # Red control: with staging back on, the same layer on the same step raises.
    monkey = FakeModelState()
    monkey.PLE_HOST_STAGE_UNDER_FULL = "stage"
    monkey.set_host_draft_source(_drafts())
    with pytest.raises(model_state_module.PLEHostDoubtUnderFullGraph):
        monkey._stage_host_ple_ids(
            input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
        )


def test_policy_off_leaves_a_non_full_step_entirely_alone(hostids_env) -> None:
    """Prefill, mixed, and piecewise steps keep the behaviour they had before.

    A piecewise step has a real device path, so nothing about the argument for
    option A applies to it; and a GRAPHMODE=piecewise boot must stay a working
    reference for what the arm does when it works.
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)

    st._stage_host_ple_ids(
        input_batch, req_states, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )

    assert st._ple_host_policy_off == 0
    assert st._ple_host_refusals == {}, f"refused: {st._ple_host_refusals}"
    assert st._ple_host_steps == 1
    assert emb.lookup_calls == 1


def test_policy_off_stands_the_mirror_commit_down_too(hostids_env) -> None:
    """Committing while declined is bookkeeping for a reader that does not exist.

    ``HostTokenHistory.stage_step`` is what produced boot #34's
    ``stage-tokens-rejected -- num_sampled [0, 4, 8] outside 0..4`` lines. With the
    consumer gone those refusals are noise on a path that also costs a copy-event
    wait, so the epilogue stands down with the builder.
    """
    st, _emb, _input_batch, _req_states = _armed_state(hostids_env)
    st._ple_step_cg_mode = CUDAGraphMode.FULL.name

    assert st.stage_host_tokens(*_commit(done=True)) is False
    assert st._ple_host_history.pending == 0, "a commit was queued anyway"
    assert st._ple_host_refusals == {}

    # And on a piecewise step it still commits, or this is a different policy.
    st._ple_step_cg_mode = CUDAGraphMode.PIECEWISE.name
    assert st.stage_host_tokens(*_commit(done=True)) is True


def test_policy_on_is_the_only_difference_between_the_two_behaviours(
    hostids_env,
) -> None:
    """One class attribute is the whole change.

    Same layer, same batch, same step: "stage" stages, "off" declines. Without
    this pair the policy tests would still pass if some unrelated edit had broken
    staging outright, which is the failure the 2.6 % measurement nearly hid.
    """
    st_on, emb_on, ib, rs = _armed_state(hostids_env)
    st_on.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st_on._stage_host_ple_ids(ib, rs, cudagraph_mode=CUDAGraphMode.FULL)
    assert st_on._ple_host_steps == 1, "the red control no longer stages"

    st_off, emb_off, ib2, rs2 = _armed_state(hostids_env)
    st_off._stage_host_ple_ids(ib2, rs2, cudagraph_mode=CUDAGraphMode.FULL)
    assert st_off._ple_host_steps == 0
    assert emb_on.lookup_calls == 1 and emb_off.lookup_calls == 0


def test_an_unrecognised_policy_value_means_stage(hostids_env) -> None:
    """A typo in the environment must restore the old behaviour, never disable more.

    ``VLLM_GFX906_PLE_HOST_STAGE_UNDER_FULL=FALSE`` or ``diabled`` from a shell
    history is far more likely than the correct string, and the dangerous reading
    of either is "off".
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "FALSE"

    st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)

    assert st._ple_host_steps == 1, "a typo silently disabled the arm"
    assert st._ple_host_policy_off == 0
    assert emb.lookup_calls == 1


def test_a_capture_time_step_is_staged_even_with_the_policy_off(
    hostids_env, monkeypatch
) -> None:
    """The policy must not reach into a capture, or the boot dies at startup.

    This is the inverse of what a reasonable reading of "decline under FULL"
    suggests, and it is the one place where getting option A wrong is not a bad
    number but a server that never comes up. The staged branch is the only branch of
    the PLE op a stream capture can hold -- the device path reads three device
    arrays synchronously, which is ``hipErrorStreamCaptureUnsupported``. So capture
    steps stage, and ``ple_layer`` zeroes the rows it staged ("host memset on pinned
    memory: not a graph node"), which is what makes the replayed contribution exactly
    zero rather than whatever the capture dummy happened to hold.
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)

    assert (
        emb.lookup_calls == 1
    ), "capture declined: the graph would capture the device path and abort with "
    "hipErrorStreamCaptureUnsupported"
    assert st._ple_host_policy_off == 0, "a capture step is not a served step"
    assert st._ple_host_refusals == {}


def test_the_decline_says_once_that_declined_is_not_absent(
    hostids_env, caplog
) -> None:
    """The boot #35 mistake, pointed at from the other side.

    A log with no PLE HOSTIDS lines is ambiguous: it means either "the knob was
    never set" or "the arm is here and declining". Boot #35 was read as the second
    when it was the first. One line per boot, naming which it is, costs nothing and
    removes the reading that cost a measurement.
    """
    st, _emb, input_batch, req_states = _armed_state(hostids_env)

    with caplog.at_level(logging.INFO):
        for _ in range(10):
            st.begin_step(CUDAGraphMode.FULL)
            st._stage_host_ple_ids(
                input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
            )

    lines = [r for r in caplog.records if "declined under FULL" in r.getMessage()]
    assert len(lines) == 1, f"logged {len(lines)} times, expected once per boot"
    assert "NOT absent" in lines[0].getMessage()
    assert st._ple_host_policy_off == 10


def test_policy_off_appears_on_the_coverage_line(hostids_env) -> None:
    """An instrument that is not on the periodic line is not an instrument."""
    st, _emb, input_batch, req_states = _armed_state(hostids_env)
    st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)

    line = st._ple_host_coverage()
    assert "policy_off=1" in line, line
    assert "stale_steps=1" in line, line


def _resync_scenario(hostids_env):
    """The lost-commit state that forces a mirror repair, from the test above.

    Shared by the S-instrument tests: the repair is the only code in this module
    that runs a device sync, so it is the only path the timer can be tested on.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()
    slots, rows, _counts, event = _commit(done=True)
    # A commit claiming more accepted tokens than the row holds: rejected, and the
    # device advances anyway, leaving the mirror one cell short. That shortfall is
    # what makes the next step repair, and the repair is what syncs.
    assert (
        st.stage_host_tokens(slots, rows, _lost_counts(), event)
        is False
    )
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )
    return st, input_batch, req_states


def test_the_resync_instrument_reports_the_sync_it_actually_paid(hostids_env) -> None:
    """``S_sync_ms`` must be the duration of the real sync, not a plausible number.

    The whole correctness-vs-staleness decision is priced against S: break-even
    against a 42.5 ms decode step is S <= ~23 ms at the ~9 % steady refusal rate
    and S <= ~5.7 ms at the ~37 % latched rate. S has never been measured, because
    nothing outside the engine process can reach it (the engine holds 99 % of
    VRAM, a second HIP context will not fit, and on Vega two resident contexts
    time-slice the GPU). So the timer lives inside the repair path -- and a timer
    that silently reported ``n=0`` for a whole boot would be worse than no timer,
    because it would read as "the path never ran" rather than "the instrument is
    dead". Hence: make the sync sleep a known 3 ms and require the reported
    number to account for it.
    """
    st, input_batch, req_states = _resync_scenario(hostids_env)
    # Fields the production __init__ sets; this fake predates the instrument.
    st._ple_resync_sync_ms = []
    st._ple_resync_read_ms = []
    st._ple_resync_timing = True

    slept_ms = 3.0
    real_sync = st._ple_resync_sync

    def slow_sync():
        real_sync()
        time.sleep(slept_ms / 1000.0)

    st._ple_resync_sync = slow_sync

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resyncs == 1, "the repair did not run, so there is nothing to read"

    line = st._ple_host_coverage()
    m = re.search(r"S_sync_ms\(n=(\d+) p50=([\d.]+) p90=([\d.]+) max=([\d.]+)\)", line)
    assert m, f"no S_sync_ms on the coverage line: {line}"
    n, p50 = int(m.group(1)), float(m.group(2))
    assert n == 1, f"one repair must yield one sync sample, got n={n}: {line}"
    # The sleep is a floor, not a target: an instrument that timed the wrong
    # interval -- the whole repair, or the numpy compare -- would report a larger
    # number and still pass, so the ceiling is what makes this a real check that
    # it timed THE SYNC. Scheduler granularity and the timer stub's own bookkeeping
    # are what make the true value exceed the sleep; 3x is far below any of those
    # adding up, and a real device sync on this box costs 0 ms when the device is
    # idle, which is the number this instrument exists to find out.
    assert slept_ms * 0.7 <= p50 <= slept_ms * 3.0, (
        f"S_sync_ms reported p50={p50} ms for a sync that slept {slept_ms} ms -- "
        f"it is not timing the sync"
    )
    rm = re.search(r"S_read_ms\(n=(\d+) p50=([\d.]+)", line)
    assert rm, f"no S_read_ms on the coverage line: {line}"
    # Two slots (0 and 3) are short by the same lost commit, so the repair reads
    # two rows and each read is sampled on its own: an early return on a
    # disagreement must still leave the sample for the slot that paid for it.
    assert int(rm.group(1)) == 2, f"one sample per repaired slot, got: {line}"

    mix = re.search(r"refusal_mix=(\{[^}]*\})", line)
    assert mix, f"no refusal_mix on the coverage line: {line}"
    assert "stage-tokens-rejected" in mix.group(1), (
        f"the lost commit that caused the repair must appear in the reason mix: "
        f"{mix.group(1)}"
    )


def test_the_instrument_reports_nothing_when_it_is_disabled(hostids_env) -> None:
    """With the knob off the coverage line must say so, not invent a number.

    ``VLLM_GFX906_PLE_RESYNC_TIMING=0`` is the way to stop sampling without
    editing the file a serving boot runs from. The red control is the number
    itself: if the flag were ignored and the timer always ran, this reports the
    same populated line the knob is supposed to suppress and fails.
    """
    st, input_batch, req_states = _resync_scenario(hostids_env)
    st._ple_resync_sync_ms = []
    st._ple_resync_read_ms = []
    st._ple_resync_timing = False

    real_sync = st._ple_resync_sync

    def slow_sync():
        real_sync()
        time.sleep(0.003)

    st._ple_resync_sync = slow_sync

    st._stage_host_ple_ids(input_batch, req_states)
    assert st._ple_resyncs == 1, "the repair must still run with the timer off"
    line = st._ple_host_coverage()
    assert "S_sync_ms(n=0)" in line, line
    assert "S_read_ms(n=0)" in line, line


# ---------------------------------------------------------------------------
# the dispatch veto
# ---------------------------------------------------------------------------


def _desc(cg_mode: CUDAGraphMode) -> BatchExecutionDescriptor:
    """A descriptor of the shape the runner would hand this batch.

    The real frozen dataclass rather than a namespace, because the veto produces a
    new descriptor with :func:`dataclasses.replace` and a namespace would let a
    mutation of the original pass as a downgrade.
    """
    return BatchExecutionDescriptor(
        cg_mode=cg_mode, num_tokens=ROWS, num_reqs=2, max_query_len=3
    )


def test_a_refused_full_step_with_the_veto_armed_is_not_a_stale_step(
    hostids_env,
) -> None:
    """The ledger has to know which steps the runner intends to rescue.

    A FULL refusal used to have exactly one meaning: this step replayed the previous
    step's n-gram rows. The veto changes that -- the step is dispatched eager and the
    device path computes its rows -- so the same refusal now belongs in a different
    count, and a test that only checked "refusal under FULL increments stale" would
    keep passing while the ledger lied about the fix.

    Both halves are asserted: the step must leave ``stale_steps`` alone, and it must
    still appear in the per-reason counts, because "why did this step leave the fast
    path" is a question people ask of this log with the veto on exactly as often as
    with it off.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_gen_layers = [emb]
    st.arm_ple_dispatch_veto(True)

    st.begin_step("FULL")
    st._note_host_refusal("history-length")

    assert st._ple_host_stale_steps == 0, (
        "a step the runner said it would run eager was still counted as a stale "
        "replay -- the coverage line would then report the defect the veto closed"
    )
    assert st._ple_veto_steps == 1, "the veto caught nothing"
    assert st._ple_host_refusals["history-length"] == 1, "the reason vanished"

    # And unarming puts the old meaning back, exactly. This is the A/B the knob
    # exists for: ``VLLM_GFX906_PLE_HOST_VETO=off`` must reproduce the boot whose
    # acceptance length was 1.98, ledger and all.
    st2 = FakeModelState()
    st2._ple_host_gen_layers = [emb]
    st2.begin_step("FULL")
    st2._note_host_refusal("history-length")
    assert st2._ple_host_stale_steps == 1
    assert st2._ple_veto_steps == 0


def test_the_veto_log_says_eager_and_never_claims_a_stale_replay(
    hostids_env, caplog
) -> None:
    """The sentence printed on a refusal has to describe the boot it is in.

    The un-vetoed wording is the shipping-gate sentinel for FULL + HOSTIDS: "there is
    NO device path ... the graph replays the previous step's embedding rows". With the
    veto armed that sentence is false, and a log that asserts a false thing about the
    one defect a mechanism exists to prevent is worse than no log, because it is the
    thing a reader trusts without re-deriving it.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_gen_layers = [emb]
    st.arm_ple_dispatch_veto(True)

    with caplog.at_level("WARNING"):
        st.begin_step("FULL")
        st._note_host_refusal("history-length")

    msgs = [r.getMessage() for r in caplog.records if "cg_mode=FULL" in r.getMessage()]
    assert len(msgs) == 1, msgs
    assert "EAGER" in msgs[0], msgs[0]
    assert "NO device path" not in msgs[0], msgs[0]
    assert "ran on stale PLE" not in msgs[0], msgs[0]


def test_a_refused_step_is_downgraded_and_a_staged_one_is_left_alone(
    hostids_env,
) -> None:
    """The veto's whole output: the same descriptor, or the same one with NONE.

    Two properties that are easy to get wrong in opposite directions. A staged step
    must come back as the identical object -- returning a copy of a FULL descriptor
    would be invisible here and would silently defeat graph replay bookkeeping
    elsewhere. And a refused step must come back NONE *without mutating the input*,
    which is why the test builds the descriptor from the real frozen dataclass:
    ``batch_desc`` is the runner's object and other code holds it.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    # The only arm on which a FULL step stages at all. With the shipped default
    # ("off") the policy declines the step before the producer runs, which is a
    # declined case, not a staged one -- and it is asserted as one below.
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.arm_ple_dispatch_veto(True)
    st._ple_host_layers = [emb]
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    full = _desc(CUDAGraphMode.FULL)
    out = st.ple_veto_descriptor(full, input_batch, req_states)
    assert out is full, "a staged step did not get its graph back"
    assert out.cg_mode == CUDAGraphMode.FULL
    assert st._ple_veto_applied == 0
    assert st._ple_host_steps == 1
    # The work happened EARLY, once, and is sitting in the cache that
    # ``prepare_inputs`` reads. Without this the veto would be free of double cost
    # only by accident: the step is prepared here and again there, which means the
    # mirror repair runs twice and the second one is measured against a count the
    # first already fixed.
    assert st.ngram_context_calls == 1, "the early hook did not prepare the step"
    cached = st._ple_prepared
    assert cached is not None and cached[0] is input_batch, (
        "the cache is not keyed on this batch, so prepare_inputs will redo the fill"
    )
    cached_qsl = cached[1]["query_start_loc"]
    assert cached_qsl.data_ptr() == st.ple_query_start_loc.data_ptr(), (
        "the cached fragment does not carry the buffer the model reads, so the "
        "differential and the forward would describe different slices"
    )

    # Now a step that cannot stage: the layer stops taking steps, which is a refusal
    # the producer cannot argue with.
    emb.disable_hostids("test: the layer stopped taking steps")
    drafts_ref = st._host_drafts
    drafts_ref.bump_generation(11)
    refused = _desc(CUDAGraphMode.FULL)
    out = st.ple_veto_descriptor(refused, input_batch, req_states)
    assert out.cg_mode == CUDAGraphMode.NONE, "a refused step kept its FULL graph"
    assert refused.cg_mode == CUDAGraphMode.FULL, (
        "the runner's descriptor was mutated instead of replaced"
    )
    assert st._ple_veto_applied == 1
    assert st._ple_veto_steps == 1, "declined and applied must be countable apart"


def test_an_arm_that_declined_never_vetoes(hostids_env) -> None:
    """The three-state is the safety property, not a detail of the return type.

    ``None`` means "this arm is not trying to stage": the knob is off, the policy
    declined the step, or staging has been disabled for the boot. Each of those reads
    as a refusal to anything that tests ``is not True``, and each of them would then
    downgrade EVERY step of the boot to eager -- a cliff roughly the height of the one
    the veto is fixing, reached by the code that was supposed to make steps cheaper.
    So ``None`` returns the descriptor untouched, and this test is the reason.
    """
    # Knob off entirely.
    hostids_env["mode"] = 0
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.arm_ple_dispatch_veto(True)
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    desc = _desc(CUDAGraphMode.FULL)
    assert st.ple_veto_descriptor(desc, input_batch, req_states) is desc
    assert st._ple_veto_applied == 0, "an arm that was never on downgraded a step"
    assert st._ple_step_rows is None

    # The policy declined, which is the case that ships: ``PLE_HOST_STAGE_UNDER_FULL``
    # defaults to "off", so on a FULL boot today every step is declined before the
    # producer runs. A veto that treated that as a refusal would run the whole boot
    # eager -- and it would look like a correctness win in the acceptance numbers,
    # which is the most convincing way to be wrong.
    hostids_env["mode"] = 1
    st2 = FakeModelState()
    st2.arm_ple_dispatch_veto(True)
    st2._ple_host_layers = [emb]
    st2.set_host_draft_source(_drafts())
    assert st2.PLE_HOST_STAGE_UNDER_FULL == "off", "the shipped default moved"
    desc = _desc(CUDAGraphMode.FULL)
    assert st2.ple_veto_descriptor(desc, input_batch, req_states) is desc
    assert st2._ple_step_rows is None
    assert st2._ple_veto_applied == 0

    # Staging disabled mid-boot: also not a step worth vetoing, because nothing will
    # ever stage again and eager-on-refusal would be eager-forever.
    st.disable_host_staging("test: the arm is off for the rest of the boot")
    desc = _desc(CUDAGraphMode.FULL)
    assert st.ple_veto_descriptor(desc, input_batch, req_states) is desc
    assert st._ple_veto_applied == 0


def test_only_full_descriptors_are_ever_downgraded(hostids_env) -> None:
    """PIECEWISE and NONE already run the op body, so there is nothing to rescue.

    This is also why the residual the veto costs anything on is small. Boot
    ``vllm.log.20261007-070900`` refused 3 585 steps across four workers, but 117 of
    them were ``not-pure-decode`` (PIECEWISE) and 2 were ``no-drafts`` (NONE), where
    the device path ran anyway: the genuinely stale residual was 0.2 % of steps on one
    worker and 3.2 % on the worst. A veto that also downgraded those would be paying
    to fix a step that was never broken.
    """
    hostids_env["mode"] = 0  # makes every step "not attempted", so nothing to stage
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.arm_ple_dispatch_veto(True)
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
        desc = _desc(mode)
        assert st.ple_veto_descriptor(desc, input_batch, req_states) is desc
    assert st._ple_veto_applied == 0

    # Decoupled from the policy. Mode 0 makes every step "not attempted", which the
    # policy declines before it ever looks at the descriptor -- so the loop above
    # cannot tell "only FULL is ever downgraded" from "nothing is ever downgraded".
    # Make the step a genuine pure decode at FULL with nothing staged, and the policy
    # commits to a refusal; the only thing left between that and a downgrade is the
    # descriptor's own mode. A captured PIECEWISE graph has its own stale PLE rows
    # that no veto helps, and downgrading one would contradict the mode the attention
    # metadata was already built for.
    hostids_env["mode"] = 2
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st._ple_host_layers = [emb]
    st._ple_step_cg_mode = CUDAGraphMode.FULL
    emb.disable_hostids("test: the layer stopped taking steps")
    st._host_drafts.bump_generation(11)

    # Positive first, so the negatives below mean something: this configuration has
    # to actually be a refusal, or "nothing was downgraded" would be true for the
    # wrong reason.
    out = st.ple_veto_descriptor(_desc(CUDAGraphMode.FULL), input_batch, req_states)
    assert out.cg_mode == CUDAGraphMode.NONE, "the setup stopped being a refusal case"
    assert st._ple_veto_steps == 1

    for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
        desc = _desc(mode)
        assert st.ple_veto_descriptor(desc, input_batch, req_states) is desc
    assert st._ple_veto_steps == 1, (
        "the downgrade keys on the step's mode rather than the descriptor's: a "
        "non-FULL graph was asked to run eagerly"
    )
    assert st._ple_host_stale_steps == 0


def test_a_stage_that_was_replayed_cannot_leave_its_flag_for_an_eager_step(
    hostids_env,
) -> None:
    """The one hazard the veto introduces, and the invalidation that closes it.

    ``take_host_stage`` hands back staged rows when the row count matches, and under a
    captured graph nothing ever calls it: the op body does not run at replay. So a
    step that stages leaves ``_host_stage_rows`` set, indefinitely, and it matches the
    row count of any later step with the same batch shape -- which is every decode
    step of the request.

    While every later step also replays, that is harmless. The veto is the first
    mechanism that sends a step eager *because* it must not replay, and an eager op
    body that finds the previous step's flag skips its D2H and reads rows from a
    different step -- the exact failure ``take_host_stage``'s docstring warns about,
    arriving through a route that did not exist when it was written.

    The clearing runs at the top of every staging attempt, above the early returns,
    because the transition that matters is staging working and then being disabled:
    the steps after that run eager, and a clear an early return can skip is not an
    invalidation. That second case is what the second half of this test drives.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.arm_ple_dispatch_veto(True)
    st._ple_host_layers = [emb]
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    st._stage_host_ple_ids(input_batch, req_states)
    assert emb._host_stage_rows == ROWS, "the stage did not arm the skip"

    # A replay: the op body never runs, so the flag survives the step. The NEXT step
    # attempts staging and must invalidate the orphan before it can be consumed.
    st._host_drafts.bump_generation(11)
    emb.disable_hostids("test: the layer stopped taking steps")
    st._stage_host_ple_ids(input_batch, req_states)
    assert emb._host_stage_rows == -1, (
        "a refused step left the previous step's staged flag for an eager op body, "
        "which will read rows from a step it does not belong to"
    )

    # And the same invalidation happens on a step that returns before it ever looks at
    # the draft source, because that is the transition that disables staging mid-boot.
    emb2 = StagingEmbedding()
    hostids_env["layers"] = [emb2]
    st2 = FakeModelState()
    st2._ple_host_layers = [emb2]
    st2.set_host_draft_source(_drafts())
    st2._stage_host_ple_ids(input_batch, req_states)
    assert emb2._host_stage_rows == ROWS
    st2.disable_host_staging("test: the arm is off from here")
    st2._stage_host_ple_ids(input_batch, req_states)
    assert emb2._host_stage_rows == -1, (
        "a step that returns at the disabled check kept a stale flag: every step "
        "after a disable runs eager"
    )


def test_the_veto_counters_are_reported_as_a_self_check(hostids_env) -> None:
    """``veto=<declined>/<applied>`` must be comparable, or the pair proves nothing.

    An armed runner whose downgrade never lands would report hundreds of steps that
    were "handled" and served stale rows anyway. Nothing else in the line would move:
    stale stays zero because the refusal charged the veto ledger, so the coverage line
    is the only place the two can be put side by side and disagreed with.
    """
    st = FakeModelState()
    st.arm_ple_dispatch_veto(True)
    st.begin_step("FULL")
    st._note_host_refusal("history-length")
    st.note_ple_veto_applied()
    line = st._ple_host_coverage()
    assert "veto=1/1" in line, line
    assert "stale_steps=0" in line, line


def test_the_demand_alarm_announces_a_rate_and_says_nothing_below_it(
    caplog,
) -> None:
    """What replaced the total cap, and why it is not a cap in disguise.

    The 2 000-repair backstop was sized on a 25 % demand figure that had been censored
    by the 64-repair cap being measured at the same time; the real figure, from a boot
    with nothing capped, was 63 %, and spending the allowance converted a coverage bug
    into a quality bug at 58.69 tok/s and acceptance 2.61. So the bound that survives
    is per-call (one sync per step, 0.12 ms, 0.19 % of a step at any demand) and what
    the demand RATE gets is an announcement.

    Three things are worth pinning: it stays quiet below the rate, it stays quiet
    between checkpoints (a per-step warning on a path that runs 25 times a second is a
    way of not reading the log), and it does not change any behaviour -- repairs
    continue, because the rate is a pointer at the count check, not a budget.
    """
    st = FakeModelState()
    st.PLE_RESYNC_ALARM_EVERY = 8
    st.PLE_RESYNC_ALARM_RATE = 0.25

    with caplog.at_level("WARNING"):
        # Below the checkpoint: silent.
        st._ple_resyncs, st._ple_host_calls = 4, 8
        st._ple_resync_demand_alarm()
        # At the checkpoint but under the rate (8/100 = 8 %): silent. This call
        # closes the window, so the markers move up to here -- counters are
        # monotone in a real worker and the test has to stay monotone or it is
        # testing a boot that cannot happen.
        st._ple_resyncs, st._ple_host_calls = 8, 100
        st._ple_resync_demand_alarm()
        assert caplog.records == [], "the alarm fired on a normal boot"

        # Next window over the rate (8 repairs in 12 calls = 67 %, the measured
        # shape): one line, naming the number and what to do about it.
        st._ple_resyncs, st._ple_host_calls = 16, 112
        st._ple_resync_demand_alarm()
    hits = [r.getMessage() for r in caplog.records if "mirror repairs" in r.getMessage()]
    assert len(hits) == 1, caplog.records
    assert "67 %" in hits[0], hits[0]
    assert "count check" in hits[0], hits[0]

    # Off the checkpoint: one more repair is not a full window, so nothing is
    # printed even at the same rate. This is the "not per step" half of the
    # contract.
    caplog.clear()
    with caplog.at_level("WARNING"):
        st._ple_resyncs, st._ple_host_calls = 17, 113
        st._ple_resync_demand_alarm()
    assert [r for r in caplog.records if "mirror repairs" in r.getMessage()] == []


def test_the_demand_alarm_wakes_for_a_burst_inside_a_quiet_boot(
    caplog,
) -> None:
    """The window IS the unit, and this is the case the shipped instrument slept through.

    On boot ``vllm.log.20261007-204346`` a decode run needed 712 mirror repairs across
    1 024 staging calls -- 69 % of steps -- while the same boot's life-to-date ratio
    sat at 1 503/11 264 = 13 %. The alarm compares a count against a life-to-date
    denominator, so it read 13 %, decided the boot was normal, and printed nothing about
    a window in which two thirds of all steps paid a sync. A long quiet boot dilutes a
    burst inside it forever, which is the failure mode of every ratio instrument that
    averages over the wrong interval: the quieter the history, the more certainly it
    stays silent.

    So the test drives exactly that shape -- a quiet life, then a burst -- and requires
    the line. It also requires the cumulative figure to be NAMED in the line, because a
    reader who does not know the two numbers disagree will reconcile them wrongly.
    """
    st = FakeModelState()
    st.PLE_RESYNC_ALARM_EVERY = 8
    st.PLE_RESYNC_ALARM_RATE = 0.25

    with caplog.at_level("WARNING"):
        # A quiet life: 40 repairs in 4 000 staging calls (1 %). Window closes,
        # silent, and the markers land here -- which is what makes the next window
        # the burst rather than the whole boot.
        st._ple_resyncs, st._ple_host_calls = 40, 4000
        st._ple_resync_demand_alarm()
        assert caplog.records == [], "the alarm fired on the quiet part of the boot"

        # The burst, in the same proportions the boot measured: 712 repairs over
        # 1 024 calls.
        st._ple_resyncs, st._ple_host_calls = 752, 5024
        st._ple_resync_demand_alarm()
    hits = [r.getMessage() for r in caplog.records if "mirror repairs" in r.getMessage()]
    assert len(hits) == 1, "a 69 % window stayed silent, as the shipped one did"
    assert "69 %" in hits[0], hits[0]
    # The window is the unit, but the line has to admit what the boot looks like
    # overall, or the two figures get read as each other.
    assert "15 %" in hits[0], hits[0]


def test_a_dropped_mirror_commit_is_not_a_step_fate_event(hostids_env, caplog) -> None:
    """An epilogue drop belongs to the mirror, not to the ledger of step fates.

    ``stage_host_tokens`` runs after the step has been dispatched and run, and the
    dispatch veto has already returned its descriptor, so nothing it reports can be a
    step the veto caught -- or a step that replayed. Before the flag, its failures were
    noted while ``_ple_step_cg_mode`` still read ``FULL`` from this step's own producer
    call, and ``_note_stale_step``'s FULL test booked them to the veto ledger. That is
    how boot ``vllm.log.20261007-204346`` reached ``veto=12/1`` with the veto having
    fired once, at 20:53:39, and never again: eight epilogue drops in a window in which
    the producer refused nothing. A reader of that field would conclude the safety net
    was busy while the thing it was built for never happened.

    The event is still worth a count and a line -- a dropped commit makes the mirror
    short by that step, and the next step pays a repair -- but the sentence has to
    describe the mirror, and neither ledger may move.
    """
    hostids_env["mode"] = 2
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.arm_ple_dispatch_veto(True)
    st._ple_host_layers = [emb]
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    # A step that staged: FULL, veto asked, decision True, so the step mode this
    # worker reports really is FULL and a refusal noted afterwards would previously
    # have satisfied the FULL test in ``_note_stale_step``.
    desc = st.ple_veto_descriptor(
        _desc(CUDAGraphMode.FULL), input_batch, req_states
    )
    assert desc.cg_mode == CUDAGraphMode.FULL, "the step did not stage"
    assert st._ple_step_full()
    stale, vetoed = st._ple_host_stale_steps, st._ple_veto_steps

    # Now the epilogue for that step, with a sampler row the history cannot accept.
    slots, rows, _counts, event = _commit()
    with caplog.at_level("WARNING"):
        assert (
            st.stage_host_tokens(
                slots, rows, _lost_counts(), event
            )
            is False
        )

    assert st._ple_host_refusals.get("stage-tokens-rejected") == 1, "not counted"
    assert st._ple_host_stale_steps == stale, "an epilogue drop charged the stale ledger"
    assert st._ple_veto_steps == vetoed, "an epilogue drop charged the veto ledger"

    lines = [
        r.getMessage()
        for r in caplog.records
        if "mirror commit dropped" in r.getMessage()
    ]
    assert len(lines) == 1, caplog.records
    assert "neither ledger is charged" in lines[0], lines[0]
    # Both of these are claims about THIS step's PLE, and neither is true of a drop.
    assert "EAGER" not in lines[0], lines[0]
    assert "stale PLE" not in lines[0], lines[0]


def test_the_early_hook_hands_its_work_to_prepare_inputs(hostids_env) -> None:
    """The cache is a handoff, not a side effect: it is consumed exactly once.

    ``prepare_inputs`` must be able to tell three cases apart -- prepared for this
    batch, prepared for a different one, and nothing prepared -- and must clear the
    cache in all three. The dangerous one is the miss that leaves the cache set: the
    next step would find a fragment two steps old, skip its staging, and report a
    staged step that never happened. The identity test is what separates "same step"
    from "same shape", and the reference the cache holds is strong so ``is`` cannot
    alias onto a recycled address.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.arm_ple_dispatch_veto(True)
    st._ple_host_layers = [emb]
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    assert st.ple_veto_descriptor(_desc(CUDAGraphMode.FULL), input_batch, req_states)
    fragment = st._take_prepared_ple_inputs(input_batch)
    assert fragment is not None, "the handoff was lost between the hook and the model"
    assert fragment["ngram_context"] == "ngram-context"
    assert st._take_prepared_ple_inputs(input_batch) is None, (
        "the fragment survived its own step: the next prepare_inputs would reuse it"
    )

    # A second step, and a different batch object: nothing may be handed over, and
    # the attempt must not leave anything behind for the step after that.
    st._host_drafts.bump_generation(12)
    st.ple_veto_descriptor(_desc(CUDAGraphMode.FULL), input_batch, req_states)
    other, _req, _tok = _batch_and_states()
    assert st._take_prepared_ple_inputs(other) is None
    assert st._ple_prepared is None
    assert st._take_prepared_ple_inputs(input_batch) is None


def test_a_repair_is_what_checks_the_demand_rate(hostids_env, caplog) -> None:
    """The alarm has to be wired to the thing it measures.

    Asserting only that the method computes the right verdict would leave it as
    correct dead code: the number this boot is missing is exactly the one nobody
    counted, because the log line that would have shown it printed once per kind and
    the budget that would have stopped it printed nothing. So a real repair, driven
    the way a lost commit drives one, has to be what emits the line.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.PLE_RESYNC_ALARM_EVERY = 1
    st.PLE_RESYNC_ALARM_RATE = 0.0
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    # A commit that claims more accepted tokens than the row holds: the mirror is
    # behind by one, which is the repair's ordinary trigger.
    slots, rows, counts, event = _commit(done=True)
    assert (
        st.stage_host_tokens(slots, rows, _lost_counts(), event)
        is False
    )
    req_states.num_computed_tokens_np[[0, 3]] += 1
    _advance_device_ids(
        input_batch,
        st,
        row=req_states.all_token_ids.gpu.numpy(),
        computed=req_states.num_computed_tokens_np,
    )

    with caplog.at_level("WARNING"):
        st._stage_host_ple_ids(input_batch, req_states)

    assert st._ple_resyncs == 1, "the setup stopped being a repair case"
    hits = [r.getMessage() for r in caplog.records if "mirror repairs" in r.getMessage()]
    assert len(hits) == 1, (
        "the repair ran and nothing looked at the demand rate -- the alarm is dead "
        "code, which is what a control that deletes only the call site finds"
    )


def test_the_first_refusal_line_describes_the_veto_it_is_in(hostids_env, caplog) -> None:
    """The per-reason sentence, not just the cadence one.

    Boot ``vllm.log.20261007-204346`` printed, on an armed boot:

        first refusal (history-length) on a step dispatching cg_mode=FULL ... no
        device path exists for this step, so the captured graph replays the previous
        PLE rows. ... Coverage: staged_steps=1 stale_steps=0 veto=2/1 ...

    which contradicts itself sixty characters later: the same line asserts a replay
    and reports a ledger with nothing in it. The cadence WARNING below it was made
    arm-aware and this line -- the one that prints FIRST, and the only refusal line
    some boots ever emit -- was not. A test that read only the WARNING would have
    caught the first case and missed this one, so it reads both.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_gen_layers = [emb]
    st.arm_ple_dispatch_veto(True)

    with caplog.at_level("INFO"):
        st.begin_step("FULL")
        st._note_host_refusal("history-length", "positions [14] != committed-1 [6]")

    first = [
        r.getMessage()
        for r in caplog.records
        if "first refusal" in r.getMessage()
    ]
    assert len(first) == 1, first
    assert "EAGER" in first[0], first[0]
    assert "replays the previous PLE rows" not in first[0], first[0]

    # A DECLINE is not a refusal: the producer returns None, the veto leaves the
    # descriptor at FULL, and the step really does replay. Saying "EAGER" here would
    # be the same lie in the opposite direction.
    caplog.clear()
    with caplog.at_level("INFO"):
        st.begin_step("FULL")
        st._note_host_refusal("no-draft-source", refused=False)
    first = [r.getMessage() for r in caplog.records if "first refusal" in r.getMessage()]
    assert len(first) == 1, first
    assert "replays the previous PLE rows" in first[0], first[0]
    assert "EAGER" not in first[0], first[0]

    # The cadence line too, and for a different reason than the first one: it picks
    # its count from whichever ledger the step landed in, so reading the veto ledger
    # on a decline evaluates ``0 == 1 or 0 % 32 == 0`` and prints NOTHING about a
    # step that replayed. Silence on a stale step is the failure the whole ledger
    # exists to prevent, so "a line exists, and it says stale" is the assertion.
    cad = [
        r.getMessage()
        for r in caplog.records
        if "PLE HOSTIDS: refusal (" in r.getMessage()
    ]
    assert len(cad) == 1, cad
    assert "NO device path" in cad[0] and "stale PLE" in cad[0], cad[0]
    assert "EAGER" not in cad[0], cad[0]
    assert "stale=1 " in cad[0], cad[0]


def test_a_declined_step_is_stale_even_with_the_veto_armed(hostids_env) -> None:
    """The ledger rule the veto must not be allowed to soften.

    ``PLE_HOST_STAGE_UNDER_FULL=off`` declines EVERY FULL step before the producer
    runs. The veto is armed independently of that knob, so with the armed branch
    written as ``if self._ple_veto_armed:`` a boot that replays zeroed rows on every
    single FULL step reports ``stale_steps=0 veto=N/0`` -- clean-looking, on the arm
    whose whole measurement is how many steps are stale. That is the direction a
    ledger must never be wrong in, so only real refusals (the producer's ``False``,
    the one thing the veto acts on) may be charged to it, and a step whose fate is
    not known is charged as stale.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st._ple_host_layers = [emb]
    st._ple_host_gen_layers = [emb]
    st.arm_ple_dispatch_veto(True)
    st.set_host_draft_source(_drafts())
    input_batch, req_states, _tokens = _batch_and_states()

    # The real route, not the helper: policy off, veto armed, a FULL step.
    st.PLE_HOST_STAGE_UNDER_FULL = "off"
    st.begin_step("FULL")
    assert st.ple_veto_descriptor(_desc(CUDAGraphMode.FULL), input_batch, req_states) \
        .cg_mode == CUDAGraphMode.FULL, "a declined step must keep its graph"
    assert st._ple_host_policy_off == 1
    assert st._ple_veto_steps == 0, "a decline is not a veto"
    assert st._ple_host_stale_steps == 1, (
        "the step replayed and the ledger says otherwise -- this is the reading that "
        "makes an off-arm boot look clean"
    )

    # And the other half: a real refusal, armed, belongs to the veto ledger instead.
    st2 = FakeModelState()
    st2._ple_host_gen_layers = [emb]
    st2.arm_ple_dispatch_veto(True)
    st2.begin_step("FULL")
    st2._note_host_refusal("history-length")
    assert st2._ple_veto_steps == 1
    assert st2._ple_host_stale_steps == 0


# ---------------------------------------------------------------------------
# Choice #3 (2026-10-08): after a mid-boot disable on a boot whose graph already
# holds the staged branch, a FULL step runs EAGER instead of replaying.
#
# Boot ``vllm.log.20261007-204346`` is why this exists. It tripped its disagreement
# budget at 23:01:56 on ``mirrored [248068, 198, 35740] vs row [248068, 198, 760]``,
# disabled itself, and then served 14,574 more steps that replayed one frozen set of
# pinned rows -- with the veto never firing (the arm reported "not trying", and the
# veto abstains on that by design) and with no heartbeat either (``staged_steps``
# stopped advancing, and the heartbeat is keyed to it). Three counters said nothing
# was happening while the engine answered every request with last step's PLE.
# ---------------------------------------------------------------------------


def _captured_then_disabled(hostids_env, *, policy: str = "eager"):
    """A boot that staged, captured the staged branch, and then tripped.

    ``_ple_host_captured`` is the sticky boot-global flag, which is what
    :meth:`_full_graph_captured_this_boot` reads first; setting it directly is the
    existing idiom in this file for "a FULL graph holds the staged branch" and
    avoids dragging in the generation tripwire, which is not what these tests are
    about.
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.PLE_DOUBT_UNDER_FULL = policy
    st.arm_ple_dispatch_veto(True)
    drafts = st._host_drafts
    st._stage_host_ple_ids(
        input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
    )
    assert st._ple_host_steps == 1, "the setup did not stage a step first"
    st._ple_host_captured = True
    st.disable_host_staging("test: the mirror disagreed")
    assert st._ple_host_layers == []
    return st, emb, drafts, input_batch, req_states


def test_a_post_disable_full_step_is_reported_refused_not_as_not_trying(
    hostids_env,
) -> None:
    """The tri-state answer is the whole mechanism, so it is tested on its own.

    ``ple_prepare_step`` returns ``False`` for "attempted and lost" and ``None`` for
    "not trying", and :meth:`ple_veto_descriptor` treats them differently on purpose:
    only ``False`` downgrades. Before 2026-10-08 a disabled arm always answered
    ``None``, which is the right answer for a boot that captured the *zeroed* dummy
    branch and the worst possible one for a boot that captured the staged branch.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(hostids_env)

    drafts.bump_generation(11)
    decision = st.ple_prepare_step(
        input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
    )

    assert decision is False, (
        "a post-disable FULL step answered None, so the veto abstains and the step "
        "replays the previous step's pinned rows -- the 204346 failure"
    )
    assert st._ple_host_refusals.get("staging-disabled-eager") == 1, (
        st._ple_host_refusals
    )
    assert "staging-disabled" not in st._ple_host_refusals, (
        "the eager route reused the old reason, so the two fates are unreadable apart"
    )


def test_the_veto_downgrades_every_step_after_a_captured_disable(
    hostids_env,
) -> None:
    """The output that matters: a descriptor with NONE, every step, forever.

    Asserted over three steps rather than one because the latch is the thing under
    test, and a latch that works on the first step and resets is exactly as dangerous
    as one that never sets.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(hostids_env)

    for step in range(3):
        drafts.bump_generation(20 + step)
        desc = _desc(CUDAGraphMode.FULL)
        out = st.ple_veto_descriptor(desc, input_batch, req_states)
        assert out.cg_mode == CUDAGraphMode.NONE, (
            f"step {step}: a post-disable FULL step kept its graph"
        )
        assert desc.cg_mode == CUDAGraphMode.FULL, "the runner's descriptor mutated"

    assert st._ple_veto_applied == 3, st._ple_veto_applied
    assert st._ple_veto_steps == 3, "the ledger and the downgrades must both count"
    assert st._ple_host_stale_steps == 0, (
        "an eager step is not a stale step: the op body computes this step's rows"
    )


def test_the_eager_disable_charges_the_veto_ledger_and_says_so(hostids_env) -> None:
    """Which counter moves is how a reader tells the two policies apart.

    Under ``eager`` the refusals belong to the veto ledger and ``stale_steps`` must
    stay flat; under ``degrade`` they belong to the stale ledger. A boot where both
    move is a boot whose veto is not running, and the coverage line is the only place
    that is checkable from a log.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(hostids_env)

    drafts.bump_generation(31)
    st.ple_prepare_step(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)

    assert st._ple_veto_steps == 1, st._ple_veto_steps
    assert st._ple_host_stale_steps == 0
    coverage = st._ple_host_coverage()
    assert "disabled=yes" in coverage
    assert "post_disable=eager" in coverage, coverage


def test_a_degrade_disable_still_replays_after_a_disable(
    hostids_env, caplog
) -> None:
    """The old default, kept honest and kept selectable.

    ``degrade`` is what a boot whose graph captured the zeroed dummy branch wants,
    and what an A/B against the new default needs. It must behave exactly as it did
    before: answer ``None``, charge the stale ledger, leave the veto alone, and print
    the sentence that says the rows keep replaying.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(
        hostids_env, policy="degrade"
    )

    assert st._ple_disabled_eager is False, "degrade set the eager latch"
    drafts.bump_generation(41)
    with caplog.at_level("WARNING"):
        decision = st.ple_prepare_step(
            input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
        )

    assert decision is None
    assert st._ple_host_refusals.get("staging-disabled") == 1
    assert st._ple_host_stale_steps == 1, "a replayed step is stale and must be named"
    assert st._ple_veto_steps == 0, "the veto does not act on a decline"
    assert "keep replaying" in caplog.text

    # The other route has to be distinguishable in the same log field.
    assert "post_disable=replay" in st._ple_host_coverage()


def test_the_policy_off_decline_outranks_the_eager_disable(hostids_env) -> None:
    """An option-A boot must not go fully eager because of this latch.

    The decline sits ABOVE the disabled check in :meth:`_stage_host_ple_ids`, and the
    ordering is load-bearing rather than incidental: on a ``PLE_HOST_STAGE_UNDER_FULL
    = off`` boot the FULL graph captured the *zeroed* dummy branch (capture-time steps
    still stage, which is the only way a capture can succeed at all), so those steps
    were never serving this step's rows and sending every one of them eager would cost
    the boot 15 % to buy nothing.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "off"
    assert st._ple_disabled_eager is True, "the latch is what is being outranked"

    drafts.bump_generation(51)
    decision = st.ple_prepare_step(
        input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
    )

    assert decision is None, "an option-A step was reported refused: the whole boot " \
        "would run eager to buy nothing"
    desc = _desc(CUDAGraphMode.FULL)
    assert st.ple_veto_descriptor(desc, input_batch, req_states) is desc, (
        "the veto downgraded a declined step"
    )
    assert st._ple_veto_applied == 0


def test_the_heartbeat_survives_staging_being_disabled(
    hostids_env, caplog
) -> None:
    """Absence of the heartbeat is not "no traffic" -- it can be "the arm is dead".

    204346 ran 14,574 steps after its disable with no heartbeat line at all, because
    the cadence keys on ``_ple_host_steps`` and that counter stops with the arm. This
    keys a second heartbeat to staging *calls*, which keep advancing while the engine
    serves, and names the route the steps are taking so the line is actionable.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(hostids_env)

    # Land exactly on the 1024-call boundary the live heartbeat uses.
    st._ple_host_calls = 1023
    drafts.bump_generation(61)
    with caplog.at_level("WARNING"):
        st._stage_host_ple_ids(
            input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
        )
    drafts.bump_generation(62)
    st._stage_host_ple_ids(
        input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
    )

    beats = [
        r.getMessage()
        for r in caplog.records
        if "STAGING DISABLED" in r.getMessage()
    ]
    assert len(beats) == 1, f"expected one heartbeat on the boundary, got {beats}"
    assert "EAGER on the device path" in beats[0], beats[0]
    assert "staged_steps=1" in beats[0], "the line must still carry the coverage"

    # The replaying route says what it is doing too -- a silent heartbeat would be
    # the same blind spot with a new key.
    st2, _e2, d2, ib2, rs2 = _captured_then_disabled(hostids_env, policy="degrade")
    st2._ple_host_calls = 1023
    d2.bump_generation(61)
    with caplog.at_level("WARNING"):
        st2._stage_host_ple_ids(ib2, rs2, cudagraph_mode=CUDAGraphMode.FULL)
    assert any(
        "STAGING DISABLED" in r.getMessage()
        and "replaying the last staged rows" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records if "STAGING DISABLED" in r.getMessage()]


def test_the_disable_log_prices_the_route_it_chose(hostids_env, caplog) -> None:
    """A policy that takes 15 % of every step has to say so as it takes it.

    The message is the only place a reader learns that this boot is now on the device
    path for the rest of its life, and the two counters they should expect from here
    (``stale_steps`` flat, ``veto=`` rising) are stated in the same sentence. Without
    that sentence the change is a silent performance regression attributed to nothing.
    """
    st, _emb, _ib, _rs = _armed_state(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.PLE_DOUBT_UNDER_FULL = "eager"
    st._stage_host_ple_ids(_ib, _rs, cudagraph_mode=CUDAGraphMode.FULL)
    st._ple_host_captured = True

    with caplog.at_level("WARNING"):
        st.disable_host_staging("test: disagreement budget spent")

    line = [
        r.getMessage()
        for r in caplog.records
        if "host staging disabled" in r.getMessage()
    ]
    assert len(line) == 1, line
    assert "EAGER" in line[0] and "stale_steps must stop rising" in line[0], line[0]
    assert "keep replaying" not in line[0], (
        "the eager route printed the degrade sentence -- the two are opposite claims"
    )


# ---------------------------------------------------------------------------
# The "zero" policy (2026-10-08): what a tripped arm serves after it trips.
#
# ``eager`` was measured the same day it shipped and lost. E1 Phase 0 priced the
# device gather at +15 % and this policy spent the boot with that number in a
# docstring; the real cost is the whole step run without a graph, because the veto
# downgrades ``FULL`` to ``NONE`` and on this launch-bound box that is ~3.1x per step
# (probe-single-stream.py, boot vllm.log.20261008-083018: 1.89s vs 0.61s/step). At
# tokens/s = acceptance / step_cost, eager serves ~0.8 tokens/s against stale's 1.98
# and zeroed's 2.58 -- it loses to the thing it replaced at any acceptance the drafter
# can reach (break-even 6.2, ceiling 4.0). ``zero`` never downgrades: the graph still
# replays, and what it replays is rows this worker zeroed once at the trip.
# ---------------------------------------------------------------------------


def test_a_trip_under_zero_zeroes_the_rows_it_leaves_behind(hostids_env, caplog) -> None:
    """The one effect the zero policy has on the buffers, asserted on the buffers.

    Filling the pinned rows after staging and before the trip stands in for "the graph
    holds real rows": whatever was there when the arm died is what every later step
    would otherwise replay forever.
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.PLE_DOUBT_UNDER_FULL = "zero"
    st.arm_ple_dispatch_veto(True)
    st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)
    assert st._ple_host_steps == 1, "the setup did not stage a step first"
    st._ple_host_captured = True

    emb._pinned_output += 1.0
    gen_before = int(emb._pinned_gen[0])

    with caplog.at_level("WARNING"):
        st.disable_host_staging("test: slot/owner misalignment")

    assert torch.all(emb._pinned_output == 0), (
        "the trip left rows behind: every later FULL step replays them and the boot "
        "claims a PLE contribution it never computed"
    )
    assert int(emb._pinned_gen[0]) == gen_before, (
        "generation was zeroed too, which would make the tripwire report a mismatch "
        "on a boot that is behaving exactly as designed"
    )
    assert st._ple_disabled_zero is True
    assert st._ple_disabled_eager is False, "zero must not also set the eager latch"
    assert st._ple_host_layers == []
    assert "ZEROS" in caplog.text, caplog.text


def test_a_zero_disable_step_is_stale_and_keeps_its_graph(hostids_env) -> None:
    """The trade the policy makes, as four numbers.

    The step keeps its graph (that is the point: no 3.1x), is counted stale (the rows
    are not this step's rows), refuses under its own reason, and never enters the veto
    ledger. If ``veto`` ever moves on a ``post_disable=zero`` boot, the zeroing is not
    wired and the boot is running eager by accident.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(
        hostids_env, policy="zero"
    )

    drafts.bump_generation(11)
    desc = _desc(CUDAGraphMode.FULL)
    out = st.ple_veto_descriptor(desc, input_batch, req_states)

    assert out.cg_mode == CUDAGraphMode.FULL, (
        "a zeroed step went eager: that is the 3.1x this policy exists to avoid"
    )
    assert st._ple_veto_steps == 0 and st._ple_veto_applied == 0, (
        "the veto acted on a route that is not supposed to use it"
    )
    assert st._ple_host_stale_steps == 1, (
        "zero rows are still not this step's rows; calling them stale is the honest "
        "reading and the option-A boot already pays that cost"
    )
    assert st._ple_host_refusals.get("staging-disabled-zero") == 1, st._ple_host_refusals
    assert "staging-disabled-eager" not in st._ple_host_refusals
    assert "staging-disabled" not in st._ple_host_refusals, (
        "the zero route reused the degrade reason, so the two are unreadable apart"
    )
    coverage = st._ple_host_coverage()
    assert "disabled=yes" in coverage and "post_disable=zero" in coverage, coverage


def test_the_zero_route_heartbeat_says_zeros_not_replay(hostids_env, caplog) -> None:
    """A reader must be able to tell the three dead-arm routes apart from one log line.

    The heartbeat is keyed to ``_ple_host_calls`` because ``_ple_host_steps`` stops
    advancing when the arm dies. The wording is the part under test: "replaying the
    last staged rows" is a claim about stale data, "ZEROS" is a claim about no data.
    """
    st, _emb, drafts, input_batch, req_states = _captured_then_disabled(
        hostids_env, policy="zero"
    )

    st._ple_host_calls = 1023
    with caplog.at_level("WARNING"):
        st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)

    msgs = [
        r.getMessage() for r in caplog.records if "STAGING DISABLED" in r.getMessage()
    ]
    assert len(msgs) == 1, msgs
    assert "ZEROED" in msgs[0] and "post_disable=zero" in msgs[0], msgs[0]
    assert "replaying" not in msgs[0] and "eager" not in msgs[0]


def test_a_declined_boot_heartbeats_even_though_it_never_had_an_arm(
    hostids_env, caplog
) -> None:
    """The gap that made boot ``210206`` look silent: policy-off prints no heartbeat.

    The live heartbeat is keyed to staged steps and the decline route stages none, so
    an option-A boot said nothing about its own serving state after startup. A decline
    is still a route, and a route a reader cannot see is a route they will misread as
    "no traffic".
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "off"
    st.PLE_DOUBT_UNDER_FULL = model_state_module.Qwen4ExpModelState.PLE_DOUBT_UNDER_FULL
    emb._pinned_output += 1.0
    st._ple_host_calls = 1023

    with caplog.at_level("WARNING"):
        st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)
        st._stage_host_ple_ids(input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL)

    msgs = [
        r.getMessage()
        for r in caplog.records
        if "DECLINED BY POLICY" in r.getMessage()
    ]
    assert len(msgs) == 1, msgs
    assert "post_disable=" in msgs[0], msgs[0]

    # The precedence control, in the test that exercises the same route: a decline is
    # not a disable, so nothing is zeroed, no latch is set, and no refusal is charged.
    assert st._ple_disabled_zero is False
    assert st._ple_host_refusals == {}, st._ple_host_refusals
    assert torch.any(emb._pinned_output != 0), (
        "the policy-off branch zeroed rows it never staged"
    )
    # Two, because the test makes two declined calls (one on the heartbeat boundary, one
    # off it to show the cadence does not repeat). Each declined step is its own stale
    # step; the counter is per step, not per printed line.
    assert st._ple_host_stale_steps == 2, "a declined step is still served stale rows"


# ---------------------------------------------------------------------------
# Task #58: the coverage heartbeat must be reachable by a boot that never reaches
# 1024. This is the audit trail that task #53 lacked: the PLE-at-decode A/B had to
# be reported as a lean, not a measurement, because the "staged" leg's log showed
# ``staged_steps=1`` sixteen seconds before its probe and nothing afterwards -- not
# because the arm was idle, but because the heartbeat's own cadence exceeded the
# boot's entire traffic. A cadence a boot cannot reach reports nothing, and nothing
# is not the same claim as zero.


def test_the_ramp_carries_the_cadence_below_its_own_interval() -> None:
    """The predicate, on its own numbers, including the ones that must stay quiet.

    Steady-state behaviour is asserted unchanged (1024, 2048, 3072 print; 2000 does
    not) so this cannot quietly turn into a per-step log, which is the failure mode a
    "print more" change always has.
    """
    due = model_state_module._ple_heartbeat_due
    for n in (1, 16, 128, 512, 1024, 2048, 3072):
        assert due(n), f"{n} should print"
    for n in (0, 2, 15, 17, 127, 129, 511, 513, 1023, 2000, 1025):
        assert not due(n), f"{n} should stay quiet"


def test_the_first_staged_step_prints_its_own_denominator(hostids_env, caplog) -> None:
    """One staged step has to be visible in the log without serving for a minute.

    The property the A/B needed: the earliest line in a boot's log that proves the
    arm stages, and the coverage counters attached to it. Before the ramp the first
    coverage line arrived at 1024 staged steps, so a boot could stage every step of
    a probe and print nothing that said so.
    """
    st, _emb, input_batch, req_states = _armed_state(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.PLE_DOUBT_UNDER_FULL = (
        model_state_module.Qwen4ExpModelState.PLE_DOUBT_UNDER_FULL
    )

    with caplog.at_level("INFO"):
        st._stage_host_ple_ids(
            input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
        )

    assert st._ple_host_steps == 1, "the setup did not stage a step"
    lines = [
        r.getMessage()
        for r in caplog.records
        if "decode step(s) so far" in r.getMessage()
    ]
    assert len(lines) == 1, f"expected the first-step coverage line, got {lines}"
    assert "staged_steps=1" in lines[0], lines[0]


def test_a_declined_boot_reports_within_its_first_step_not_its_thousandth(
    hostids_env, caplog
) -> None:
    """The same gap on the route that never stages: option A, from step one.

    Boot ``vllm.log.20261008-210206`` printed its startup snapshots at 21:08:55 and
    then nothing for the rest of the evening, because ``policy_off`` steps advance
    ``_ple_host_calls`` and the decline heartbeat only spoke at multiples of 1024 --
    which a lightly-loaded boot reaches in minutes and a short boot never. "Declined,
    arm healthy" was indistinguishable from "engine stopped".
    """
    st, emb, input_batch, req_states = _armed_state(hostids_env)
    st.PLE_HOST_STAGE_UNDER_FULL = "off"
    st.PLE_DOUBT_UNDER_FULL = (
        model_state_module.Qwen4ExpModelState.PLE_DOUBT_UNDER_FULL
    )
    emb._pinned_output += 1.0
    st._ple_host_calls = 0

    with caplog.at_level("WARNING"):
        st._stage_host_ple_ids(
            input_batch, req_states, cudagraph_mode=CUDAGraphMode.FULL
        )

    msgs = [
        r.getMessage()
        for r in caplog.records
        if "DECLINED BY POLICY" in r.getMessage()
    ]
    assert len(msgs) == 1, f"one call must already report, got {msgs}"
    assert "staged_steps=0" in msgs[0], msgs[0]
    assert st._ple_disabled_zero is False, "a decline is not a disable"


def test_a_drain_time_drop_is_not_a_step_fate_event_either(hostids_env, caplog) -> None:
    """Task #56's moved check has to stay as loud as the check it replaced.

    The count-range check used to run in the epilogue (``stage_host_tokens``). It
    now runs inside ``HostTokenHistory.drain``, which the PRODUCER calls at the top
    of the next step -- a different caller, with a different step in flight, and
    nothing in the epilogue path left that detects the anomaly at all.

    Two things are easy to get wrong in that move, and this pins both:

    * Tempting to charge it to a ledger, because the producer's refusal happens
      before dispatch. It is still a MIRROR event: the step whose commit it
      rejected has already been dispatched and will run with correct rows, so
      neither the stale ledger nor the veto ledger may move.
    * Tempting to let it go uncounted and unlogged, since ``drain`` only returns
      an applied-count. That would make the hole SILENT -- the per-reason cadence
      line is the only thing that makes a mirror hole visible in a serving log,
      and it would go quiet exactly when the check started running somewhere
      nobody logs from. Boot #26's 67 drops per worker were invisible for exactly
      this reason.
    """
    emb = StagingEmbedding()
    hostids_env["layers"] = [emb]
    st = FakeModelState()
    st.PLE_HOST_STAGE_UNDER_FULL = "stage"
    st.arm_ple_dispatch_veto(True)
    st._ple_host_layers = [emb]
    drafts = _drafts()
    st.set_host_draft_source(drafts)
    input_batch, req_states, _tokens = _batch_and_states()

    # A step that staged, and an epilogue handing over a commit whose accepted
    # count is out of range for its row -- which is NOTICE now, not a violation.
    st.ple_veto_descriptor(_desc(CUDAGraphMode.FULL), input_batch, req_states)
    stale, vetoed = st._ple_host_stale_steps, st._ple_veto_steps
    slots, rows, _counts, event = _commit(done=True)
    assert (
        st.stage_host_tokens(slots, rows, np.array([9, 1], np.int32), event) is True
    ), "out-of-range counts are not a stage-time violation any more"
    assert st._ple_host_history.dropped == 0, "detected too early"

    # The NEXT step's producer drains the queue. That is where the row is rejected.
    # The draft block has to be fresh or the staleness check refuses first and the
    # loop measures the wrong thing (same reason as the resync tests above).
    drafts.bump_generation(2)
    with caplog.at_level("WARNING"):
        st.ple_veto_descriptor(_desc(CUDAGraphMode.FULL), input_batch, req_states)

    assert st._ple_host_history.dropped == 1, "the moved check detected nothing"
    assert "read after the copy landed" in st._ple_host_history.last_drop, (
        st._ple_host_history.last_drop
    )
    assert st._ple_host_refusals.get("mirror-drop-at-drain") == 1, "uncounted"
    assert st._ple_host_stale_steps == stale, "a mirror hole charged the stale ledger"
    assert st._ple_veto_steps == vetoed, "a mirror hole charged the veto ledger"

    lines = [
        r.getMessage()
        for r in caplog.records
        if "mirror commit dropped" in r.getMessage()
    ]
    assert len(lines) == 1, caplog.records
    assert "neither ledger is charged" in lines[0], lines[0]
    assert "mirror-drop-at-drain" in lines[0], lines[0]


def test_the_final_coverage_line_reports_a_boot_that_never_staged(
    hostids_env, caplog
) -> None:
    """A short boot's LAST counters have to reach the log, or they never existed.

    The serving heartbeat is a cadence, and task #58's lesson is that a cadence is
    part of its claim: below 1024 staged steps a boot prints no denominator at all,
    so "staged every step" and "boot too short to report" are the same silence. The
    measured case was a ~300-step boot whose last coverage line was the ramp at 128
    -- 172 steps unaccounted, and the final ``refusal_mix`` / ``dropped`` existed
    only inside a process that was about to exit.

    Two properties are pinned, both about the line being USABLE as a measurement:

    * it prints at ``staged_steps=0``. An option-A boot (policy ``off``) never
      stages, so the serving heartbeat never fires there; this line is the only
      way such a boot shows its own refusal mix, and the same line is what tells a
      reader that a quiet boot was idle rather than dead.
    * it prints ONCE. ``Worker.shutdown()`` is reachable twice in one process --
      the executor's RPC path (``WorkerWrapper.shutdown``) and ``worker_main``'s
      ``finally`` block -- and a doubled FINAL line in a boot log is a second data
      point that no measurement backs.
    """
    st = FakeModelState()
    with caplog.at_level("INFO"):
        st.log_staging_coverage_at_shutdown("worker-shutdown")
        st.log_staging_coverage_at_shutdown("worker-shutdown")

    final = [
        r.getMessage() for r in caplog.records if "FINAL coverage" in r.getMessage()
    ]
    assert len(final) == 1, final
    assert "worker-shutdown" in final[0], final[0]
    assert "staged_steps=0" in final[0], final[0]
    # The fields a reader needs to close a boot out. This is the same builder the
    # heartbeat uses, so a field that disappears here disappears from the log
    # entirely -- scripts/ple-window-state.py routes on post_disable in particular.
    for field in (
        "stale_steps=",
        "veto=",
        "policy_off=",
        "refused=",
        "post_disable=",
        "mirror(",
        "resync=",
        "disagree=",
        "refusal_mix=",
    ):
        assert field in final[0], f"{field} missing from {final[0]}"


def test_the_final_coverage_line_cannot_break_worker_teardown(
    hostids_env, caplog
) -> None:
    """A log line must never be the reason a worker skips releasing its GPU.

    The caller is ``GPUModelRunner.shutdown``, reached from the ``finally`` block of
    ``worker_main``, and everything after it in that chain frees GPU resources and
    tears down the distributed process group. The coverage builder reads live
    objects -- layer registries, draft sources, copy events -- and a worker reaches
    teardown precisely when things have gone wrong, so "should not raise" is not a
    hypothetical here: it is the failure mode that would turn a clean restart into a
    hung one. The counter is dropped and reported instead.
    """
    st = FakeModelState()

    def boom():
        raise RuntimeError("coverage builder touched a torn-down registry")

    st._ple_host_coverage = boom
    with caplog.at_level("INFO"):
        st.log_staging_coverage_at_shutdown()  # must not raise

    warns = [
        r.getMessage()
        for r in caplog.records
        if "could not print the final coverage" in r.getMessage()
    ]
    assert len(warns) == 1, warns
    assert "RuntimeError" in warns[0], warns[0]
