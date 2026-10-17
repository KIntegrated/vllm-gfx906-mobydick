# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring tests for the PLE host-ids work (docs/d1-host-ids-design.md).

These cover the *plumbing* half of the design: the pinned mirror of each step's
draft tokens (``DraftTokensHandler`` with ``host_draft_staging=True``) and the
config-time refusal of the one scheduling mode that makes that mirror stale.
The numerical half -- that the host computes the same ngram ids the device kernel
would have -- is ``test_ple_hostids.py``.

Three properties are worth more here than a happy-path assertion, because each
one is a way the knob could be silently wrong or silently destructive:

* **The scheduler channel does not change.** Upstream reports ``[[-1] * k]`` to
  the scheduler unless the batch has structured-output requests. If staging
  changed that, a perf knob would be altering scheduling.
* **Row mapping travels with the values.** Consuming staged rows by batch
  position under a different mapping is a plausible ngram id for the wrong
  request.
* **Freshness is provable.** Two decode steps over an unchanged batch have the
  same row mapping and different values, so ``gen`` -- not the mapping -- is
  what tells a consumer whether it is holding this step's drafts.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import regex as re
import torch

from vllm.config.vllm import ENV_PLE_HOSTIDS, ple_hostids_async_conflict
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the staging path allocates a copy stream and a blocking event",
)


def _batch(num_reqs: int, structured: bool = False, mapping: list[int] | None = None):
    """The three ``InputBatch`` fields ``set_draft_tokens`` touches."""
    return SimpleNamespace(
        req_ids=[f"req{i}" for i in range(num_reqs)],
        has_structured_output_reqs=structured,
        idx_mapping_np=np.array(
            mapping if mapping is not None else range(num_reqs), dtype=np.int32
        ),
    )


# --------------------------------------------------------------------------
# The config-time refusal: async scheduling takes away the wait this depends on
# --------------------------------------------------------------------------


def test_async_conflict_refuses_knob_under_async():
    env = {ENV_PLE_HOSTIDS: "1"}
    msg = ple_hostids_async_conflict(True, environ=env)
    assert msg is not None
    # The message has to name both the knob and the way out, or the operator
    # reading a ValueError at boot cannot act on it.
    assert ENV_PLE_HOSTIDS in msg
    assert "--no-async-scheduling" in msg


def test_async_conflict_silent_when_sync_or_unset():
    # Red control for the test above: the refusal must track *both* inputs, so
    # a guard that always fires would pass the first test and fail this one.
    assert ple_hostids_async_conflict(False, environ={ENV_PLE_HOSTIDS: "1"}) is None
    assert ple_hostids_async_conflict(True, environ={}) is None
    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: "0"}) is None
    assert ple_hostids_async_conflict(False, environ={}) is None


def test_async_conflict_covers_every_nonzero_value():
    """Differential mode ("2") reads the same mirror, so async breaks it too.

    It would not produce wrong output -- the device path still answers -- but the
    comparison would run against a one-step-old row set and report mismatches
    that are the scheduler's doing, not the arithmetic's. A guard that matches
    only "1" leaves that mode unguarded.
    """
    from vllm.models.qwen4_exp.amd import ple_hostids

    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: "2"}) is not None
    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: "7"}) is not None
    assert ple_hostids_async_conflict(True, environ={ENV_PLE_HOSTIDS: ""}) is None
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(ENV_PLE_HOSTIDS, "2")
        assert ple_hostids.hostids_mode() == 2
        assert ple_hostids.hostids_enabled() is False, (
            "differential mode must not be treated as the authoritative path"
        )
    finally:
        monkey.undo()


def test_env_name_agrees_between_guard_and_worker():
    """The guard, the worker and the model must all read one environment name.

    They cannot share a function call (the worker deliberately avoids importing
    model code), so the name is the thing that can drift -- and a drifted name is
    a knob that silently never engages, which is indistinguishable from one that
    worked. This reads the sources instead of importing them so the test costs
    nothing and cannot itself perturb the import graph.
    """
    import inspect
    import pathlib

    from vllm.models.qwen4_exp.amd import ple_hostids
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    literal = f'"{ENV_PLE_HOSTIDS}"'
    runner_src = pathlib.Path(inspect.getfile(model_runner_module)).read_text()
    assert literal in runner_src, "worker reads a different name than the guard"
    # hostids_enabled() must answer to the same name, not a private copy.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.delenv(ENV_PLE_HOSTIDS, raising=False)
        assert ple_hostids.hostids_enabled() is False
        monkey.setenv(ENV_PLE_HOSTIDS, "1")
        assert ple_hostids.hostids_enabled() is True
    finally:
        monkey.undo()


def test_accepted_values_agree_between_mode_and_worker():
    """``hostids_mode()`` and the worker must accept the same set of values.

    The worker cannot import model code (it would drag the model into every
    worker of every model), so it re-implements the read of the same variable. A
    value one side accepts and the other ignores is the worst drift there is:
    differential mode would boot with no draft mirror, refuse every step, and
    report a clean run.
    """
    import inspect
    import pathlib

    from vllm.models.qwen4_exp.amd import ple_hostids
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    src = pathlib.Path(inspect.getfile(model_runner_module)).read_text()
    # The accepted set is *extracted*, not string-matched. A literal like
    # 'in ("1", "2")' is text a formatter is free to reflow across lines, and a
    # test that goes red because a hook re-wrapped a tuple teaches people to
    # ignore red tests. What has to stay true is the set, so the set is what is
    # read out of the source.
    match = re.search(
        r'os\.environ\.get\(\s*"VLLM_GFX906_PLE_HOSTIDS",\s*"(?P<default>[^"]*)"\)'
        r"\s+in\s*\((?P<values>[^)]*)\)",
        src,
    )
    assert match, (
        "the worker no longer reads VLLM_GFX906_PLE_HOSTIDS against a literal "
        "tuple of accepted values; update this test and the worker together"
    )
    accepted = set(re.findall(r'"([^"]*)"', match.group("values")))
    assert accepted == {"1", "2"}, accepted
    assert match.group("default") not in accepted, (
        "the default value must not be one of the accepted ones"
    )

    monkey = pytest.MonkeyPatch()
    try:
        # The worker's rule is `raw in accepted`, ours is `hostids_mode() != 0`;
        # they must agree on every value, including the ones neither side likes.
        for raw in ("0", "", "1", "2", "3", "yes", "00", " 1"):
            monkey.setenv(ENV_PLE_HOSTIDS, raw)
            assert (raw in accepted) is (ple_hostids.hostids_mode() != 0), raw
    finally:
        monkey.undo()


# --------------------------------------------------------------------------
# the per-step graph mode reaches the code that makes staleness claims
# --------------------------------------------------------------------------


def test_the_runner_threads_this_step_graph_mode_into_model_state() -> None:
    """A2 is worth nothing unless the argument actually arrives.

    ``model_state`` now refuses to call a step stale unless it was told what graph
    mode the step was dispatched in, and it falls back -- silently, and only when
    nothing was threaded -- to the boot-global answer. A fallback that fires because
    somebody dropped a keyword during a rebase looks exactly like a working system:
    the numbers stay plausible and the refusals get over-reported. So the wiring is
    asserted from both ends.

    The call site is read out of the source rather than executed, in the style of
    ``test_accepted_values_agree_between_mode_and_worker``: running it would mean
    building a whole worker. What is *extracted* rather than string-matched is the
    call, because a formatter is free to re-flow the argument list.
    """
    import inspect
    import pathlib

    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.qwen4_exp.amd import model_state as model_state_module
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    # Imported for their effect on the class graph, not their names: the subclass
    # walk below only sees a state whose module has been imported.
    from vllm.v1.worker.gpu.model_states import (  # noqa: F401
        default,
        encoder_decoder,
        encoder_only,
        interface,
        mamba_hybrid,
    )

    src = pathlib.Path(inspect.getfile(model_runner_module)).read_text()
    call = re.search(r"self\.model_state\.prepare_inputs\((.*?)\)", src, re.S)
    assert call, (
        "the runner no longer calls model_state.prepare_inputs in one place; if that "
        "moved, the per-step graph mode has to be threaded there instead"
    )
    assert "cudagraph_mode=" in call.group(1), [
        "the shared call site stopped passing the per-step graph mode, so every "
        "staleness claim is falling back to the boot-global answer"
    ]
    assert "batch_desc" in call.group(1), call.group(1)

    # And every state the runner can be handed must tolerate the keyword, because
    # the runner calls ALL of them the same way -- one implementation that dropped
    # it is a TypeError for whoever runs that model, not for us.
    states = [interface.ModelState]
    seen: set[type] = set()
    while states:
        cls = states.pop()
        if cls in seen:
            continue
        seen.add(cls)
        states.extend(cls.__subclasses__())
        params = inspect.signature(cls.prepare_inputs).parameters
        assert "cudagraph_mode" in params, [
            f"{cls.__name__}.prepare_inputs cannot take the per-step graph mode the "
            f"runner passes to every model state"
        ]

    # ``Qwen4ExpModelState`` is the one that acts on it, and the module that owns
    # the enum is the one the runner's descriptor uses.
    assert CUDAGraphMode.FULL.name == "FULL", "_ple_step_full compares this string"
    assert "cudagraph_mode" in inspect.signature(
        model_state_module.Qwen4ExpModelState.prepare_inputs
    ).parameters


# --------------------------------------------------------------------------
# Staging off: the default must be the known-good path, unchanged
# --------------------------------------------------------------------------


@requires_cuda
def test_staging_off_by_default():
    handler = DraftTokensHandler(None)
    assert handler.host_draft_staging is False
    assert handler.draft_tokens_host is None
    assert handler.host_draft_gen == 0

    handler.set_draft_tokens(_batch(2), torch.zeros(2, 3, dtype=torch.int32))
    values, row_idx, gen = handler.wait_host_draft_tokens()
    assert values is None and row_idx is None
    assert gen == 0, "off must mean nothing was staged, not something unstaged"
    assert handler.draft_tokens_host is None


@requires_cuda
def test_scheduler_channel_is_untouched_by_staging():
    """``get_draft_tokens()`` must not learn anything new from the knob.

    Upstream only hands the scheduler real drafts when the batch contains
    structured-output requests; otherwise it reports -1 for every slot. If
    enabling a host-side mirror changed that report, the knob would be changing
    scheduling behaviour, which is a different (and much larger) change.
    """
    drafts = torch.arange(6, dtype=torch.int32).reshape(2, 3)

    reports = {}
    for staging in (False, True):
        handler = DraftTokensHandler(None, host_draft_staging=staging)
        handler.set_draft_tokens(_batch(2, structured=False), drafts)
        out = handler.get_draft_tokens()
        reports[staging] = out.draft_token_ids
        assert out.req_ids == ["req0", "req1"]

    assert reports[False] == [[-1, -1, -1], [-1, -1, -1]]
    assert reports[True] == reports[False]


@requires_cuda
def test_structured_output_channel_still_works_with_staging_on():
    """The two channels are independent: both populated, both correct."""
    drafts = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(2, structured=True), drafts.to("cuda"))
    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert staged[:2, :3].tolist() == drafts.numpy().tolist()
    assert row_idx.tolist() == [0, 1]
    assert handler.get_draft_tokens().draft_token_ids == drafts.cpu().tolist()
    assert gen == 1


# --------------------------------------------------------------------------
# Staging on: values, mapping, buffer reuse, and the freshness counter
# --------------------------------------------------------------------------


@requires_cuda
def test_cpu_staging_values_mapping_and_gen():
    handler = DraftTokensHandler(None, host_draft_staging=True)
    drafts = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32)
    handler.set_draft_tokens(_batch(2, mapping=[5, 2]), drafts)

    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert gen == 1
    assert staged[:2, :3].tolist() == [[7, 8, 9], [10, 11, 12]]
    assert row_idx.tolist() == [5, 2], "row order must follow idx_mapping, not position"

    # Waiting twice must be idempotent and must not invent a new generation:
    # a consumer that sees the same gen twice has to take the device path.
    again = handler.wait_host_draft_tokens()
    assert again[2] == gen
    assert again[0][:2, :3].tolist() == staged[:2, :3].tolist()


@requires_cuda
def test_gen_only_advances_when_something_was_staged():
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(1), torch.tensor([[1, 2]], dtype=torch.int32))
    _, _, gen1 = handler.wait_host_draft_tokens()
    assert gen1 == 1
    # A step where the drafter did not run calls nothing here, so the gen sits
    # still and the next consumer refuses. This is the assertion that makes the
    # consumer's staleness check possible at all.
    assert handler.wait_host_draft_tokens()[2] == gen1


@requires_cuda
def test_buffer_is_reused_then_grown():
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(2), torch.zeros(2, 3, dtype=torch.int32))
    buf = handler.draft_tokens_host
    assert buf is not None and buf.is_pinned()

    # Smaller request: same buffer, no new pinned allocation.
    handler.set_draft_tokens(_batch(1), torch.ones(1, 2, dtype=torch.int32))
    assert handler.draft_tokens_host is buf
    staged, _, gen = handler.wait_host_draft_tokens()
    assert gen == 2
    assert staged[0, :2].tolist() == [1, 1]

    # Larger request: grown, and the previously staged rows are gone from it.
    big = torch.full((4, 5), 9, dtype=torch.int32)
    handler.set_draft_tokens(_batch(4), big)
    assert handler.draft_tokens_host.shape >= (4, 5)
    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert gen == 3
    assert staged[:4, :5].tolist() == big.numpy().tolist()
    assert row_idx.tolist() == [0, 1, 2, 3]


@requires_cuda
def test_staging_dtype_follows_source_without_casting():
    """A cast would be a device kernel per step to achieve nothing."""
    handler = DraftTokensHandler(None, host_draft_staging=True)
    handler.set_draft_tokens(_batch(1), torch.tensor([[1, 2, 3]], dtype=torch.int32))
    assert handler.draft_tokens_host.dtype == torch.int32
    handler.set_draft_tokens(_batch(1), torch.tensor([[1, 2, 3]], dtype=torch.int64))
    # Different dtype means a new buffer, not a cast into the old one.
    assert handler.draft_tokens_host.dtype == torch.int64
    staged, _, _ = handler.wait_host_draft_tokens()
    assert staged[:1, :3].tolist() == [[1, 2, 3]]


@requires_cuda
def test_gpu_staging_lands_before_wait_returns():
    """The real path: D2H on the copy stream, visible after the event wait."""
    handler = DraftTokensHandler(None, host_draft_staging=True)
    drafts = torch.arange(12, dtype=torch.int32).reshape(3, 4).cuda()
    handler.set_draft_tokens(_batch(3, mapping=[9, 4, 77]), drafts)
    staged, row_idx, gen = handler.wait_host_draft_tokens()
    assert staged[:3, :4].tolist() == drafts.cpu().tolist()
    assert row_idx.tolist() == [9, 4, 77]
    assert gen == 1

    # Second step, same shape: the buffer must carry the new values, proving the
    # copy is re-issued rather than the first snapshot being reused.
    handler.set_draft_tokens(_batch(3, mapping=[9, 4, 77]), drafts * 100)
    staged, _, gen = handler.wait_host_draft_tokens()
    assert gen == 2
    assert staged[:3, :4].tolist() == (drafts * 100).cpu().tolist()


def test_the_runner_asks_the_veto_before_it_builds_the_metadata() -> None:
    """The ordering the dispatch veto depends on, asserted on the runner's source.

    ``model_state.prepare_inputs`` runs at the END of ``execute_model`` -- after
    ``model_state.prepare_attn`` -- so the staging decision is not available where it
    used to be made, and ``prepare_attn`` is the thing that fixes the step's graph
    mode: it picks request and token counts by mode, and it ends in
    ``recoverssm.record_step``, so calling it twice records a step that never ran.
    The veto therefore has to be asked between the input copy and that call.

    Read as text rather than imported: ``vllm.v1.worker.gpu.model_runner`` pulls in the
    whole worker stack and there is no engine here, and ``find_spec`` resolves the path
    without executing the module. This is the only cheap way to pin a property that
    lives in ORDER, and the property is the whole mechanism -- the same code asked one
    line later is a step that already replayed.
    """
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec("vllm.v1.worker.gpu.model_runner")
    assert spec is not None and spec.origin, "the runner module moved"
    src = pathlib.Path(spec.origin).read_text()

    ask = src.find("ple_veto_descriptor(")
    assert ask > 0, "the runner never asks the veto: refused steps replay stale rows"
    attn = src.find("self.prepare_attn(input_batch)", ask)
    state_attn = src.find("self.model_state.prepare_attn(", ask)
    assert 0 < attn, src[ask : ask + 400]
    assert 0 < state_attn
    # Both metadata builders come after the question. ``prepare_attn`` on the runner
    # is the block-table one; the model_state call is the mode-dependent one, and it
    # is the second of the two, so ordering against both is what says "before the
    # step is committed to a graph".
    assert ask < attn < state_attn, (
        "the veto is asked after the attention metadata exists, which is the ordering "
        "it was written to avoid: the metadata already describes a replay"
    )

    # A downgrade chosen on one rank desyncs the mode the ranks agreed on, so the
    # arm is conditional on there being one rank.
    assert "self.dp_size == 1" in src, (
        "the DP guard is gone: under data parallelism this is a cross-rank protocol "
        "decision and a local downgrade is a desync, not an optimisation"
    )
    assert "arm_ple_dispatch_veto(True)" in src
    assert "if self._ple_veto:" in src, "the per-step call is not gated on the arm"


def test_worker_teardown_prints_the_final_coverage_line_before_it_can_be_lost() -> None:
    """The FINAL PLE coverage line has one chance to fire and two ways to miss it.

    ``GPUModelRunner.shutdown`` is the hook -- it is reached from the ``finally``
    block of ``worker_main`` (multiproc_executor.py), which is how SIGINT and
    SIGTERM both arrive at a TP worker, i.e. the operator's ordinary
    ``kill -INT -$(pgid)`` restart. Two things in that same method would swallow
    the line, and the ordering is the entire content of the test:

    * the accelerator ``synchronize()`` that opens the method hangs on a wedged
      GPU -- the state a boot that is about to be killed is often in. A coverage
      line emitted after it is a line that prints exactly when it is least needed.
    * the ``del self.model_state`` further down drops the object that HOLDS every
      counter. Emitting after it is not a late line, it is no line.

    Read off the PARSED syntax tree rather than by string search, and not for
    style: the first version of this test string-matched ``del self.model_state``
    in the head of the method and went red on the COMMENT above the call, which
    mentions that statement by name. An ordering test that fires because someone
    wrote a sentence about the ordering is a test people learn to ignore. Exercising
    the real method instead would need a GPU, a model and a distributed group, and
    what is under test is where a call sits.
    """
    import ast
    import inspect
    import pathlib

    from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    src = pathlib.Path(inspect.getfile(model_runner_module)).read_text()
    fn = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef) and node.name == "shutdown"
    ]
    assert len(fn) == 1, (
        f"found {len(fn)} shutdown() methods in model_runner.py; this test has to"
        " know which one owns the model state"
    )
    stmts = fn[0].body

    def call_names(node):
        out = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                out.add(
                    f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                )
        return out

    def first(pred):
        hits = [i for i, s in enumerate(stmts) if pred(s)]
        return hits[0] if hits else None

    emit = first(lambda s: "log_staging_coverage_at_shutdown" in call_names(s))
    assert emit is not None, "shutdown() never calls the FINAL coverage emitter"

    # Both statements the ordering claim is about have to still be there, or the
    # comparisons below are against nothing.
    sync = first(lambda s: "synchronize" in call_names(s))
    assert sync is not None, (
        "shutdown() no longer synchronizes the accelerator: the blocking-call half"
        " of this ordering claim is stale, re-read the method"
    )
    dropped = first(lambda s: any(isinstance(n, ast.Delete) for n in ast.walk(s)))
    assert dropped is not None, (
        "shutdown() no longer deletes anything: the dropped-state half of this"
        " ordering claim is stale"
    )

    assert emit < sync, (
        "the FINAL coverage line is emitted after the accelerator sync, which hangs"
        " on a wedged GPU -- the state a killed boot is usually in"
    )
    assert emit < dropped, "the model state is dropped before its counters are printed"
    # The producer side has to answer, or the call above is a no-op that passes
    # every check in this file.
    assert hasattr(
        Qwen4ExpModelState, "log_staging_coverage_at_shutdown"
    ), "the runner calls a method the model state does not have"


def test_final_coverage_line_is_written_straight_to_a_file(tmp_path, monkeypatch):
    """The FINAL line must not depend on vLLM's log forwarding surviving teardown.

    Two boots on 2026-10-11 ended with the veto ARMED (all four ranks printed the
    arming line) and NO FINAL coverage line in the log -- not the line, not its
    failure form. The logger call travels worker -> forwarder -> file and the
    teardown order does not promise the forwarder is still alive when
    Worker.shutdown() runs, so the same content also goes to $LOG_DIR via write(2).
    """
    from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

    class _State:
        """Only what the hook touches. The helper must be the REAL one: a
        SimpleNamespace without it makes the hook swallow an AttributeError and
        the test passes for the wrong reason."""

        _ple_host_coverage = lambda self: "staged_steps=7 dropped=1 bad=2/mirror-tail"
        _ple_coverage_write_file = Qwen4ExpModelState._ple_coverage_write_file

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_GFX906_PLE_FINAL_FILE", raising=False)
    obj = _State()

    Qwen4ExpModelState.log_staging_coverage_at_shutdown(obj, "test-teardown")

    written = (tmp_path / "ple-hostids-final.log").read_text()
    assert "staged_steps=7" in written
    assert "reason=test-teardown" in written

    # Idempotent through the same flag the logger path uses: a doubled FINAL line
    # is a false second data point in a boot log, and now in the sink too.
    Qwen4ExpModelState.log_staging_coverage_at_shutdown(obj, "second-call")
    assert len((tmp_path / "ple-hostids-final.log").read_text().splitlines()) == 1


def test_final_coverage_file_sink_is_opt_out_and_cannot_raise(tmp_path, monkeypatch):
    from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

    class _State:
        _ple_host_coverage = lambda self: "staged_steps=0"
        _ple_coverage_write_file = Qwen4ExpModelState._ple_coverage_write_file

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_GFX906_PLE_FINAL_FILE", "0")
    obj = _State()

    Qwen4ExpModelState.log_staging_coverage_at_shutdown(obj, "test")
    assert not (tmp_path / "ple-hostids-final.log").exists()

    # A LOG_DIR that cannot be written must not break teardown: the hook swallows
    # it and still reports, rather than being the reason a worker keeps its GPU.
    # Fresh object: the idempotence flag from the call above would otherwise make
    # this assertion pass without ever attempting the write.
    monkeypatch.setenv("VLLM_GFX906_PLE_FINAL_FILE", "1")
    monkeypatch.setenv("LOG_DIR", "/proc/definitely-not-writable")
    Qwen4ExpModelState.log_staging_coverage_at_shutdown(_State(), "unwritable")
