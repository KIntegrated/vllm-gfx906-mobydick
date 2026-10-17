# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for E1 Phase 1: capturing a FULL graph with host-staged PLE rows.

Phase 0 measured the ceiling of full-graph decode with the gather skipped
(``test_ple_capture_probe.py``). Phase 1 is the shipping question: can the op be
captured while it still does the real host->device handoff?

It could not, and the reason is arithmetic two files away. The FULL ("mixed")
capture branch picks its request count with ``num_reqs = min(num_tokens,
max_num_reqs)`` (``cudagraph_utils.py:280-281``). With ``max_num_seqs=3`` and
capture sizes ``[4, 8, 12]`` that is ``(4 tok, 3 reqs)``, ``(8 tok, 3 reqs)``,
``(12 tok, 3 reqs)`` -- and only 12 is a uniform ``1 + num_speculative_tokens``
per request, which is what the staging producer insists on. So at 4 and 8 nothing
stages, the op fell through to ``pinned_ids.copy_(ngram_ids, non_blocking=False)`
``, and ``hipErrorStreamCaptureUnsupported`` killed the boot.

The clause under test takes the same graph node the staged branch takes
(fixed-extent non-blocking H2D out of ``_pinned_output``), because capture needs
the *extent* and never the dummies' *values*.

Two claims here are load-bearing and both are checked with a real capture, not a
monkeypatched flag:

* ``test_real_capture_delivers_rows_the_host_wrote_after_capture`` -- the graph
  hands back rows written AFTER capture. That is the whole handoff.
* ``test_real_capture_without_host_staging_still_dies`` -- with staging off, the
  same capture still fails. Without this, the passing test could be passing
  because the harness cannot fail.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module  # noqa: E402
from vllm.models.qwen4_exp.amd.ple_layer import (  # noqa: E402
    Qwen4ExpPLELayer,
    qwen4_exp_amd_ple_ngram_embedding,
)

ROWS = 6
# The staging buffers the real constructor allocates are max_total_tokens wide, so
# a capture shape larger than the buffer is a test bug, not a behaviour. Sized for
# the biggest capture shape used here (the 12-token dummy is this box's largest).
BUF_ROWS = 12
EMBED_DIM = 8
NGRAM_HEADS = 2
SENTINEL = -12345


class RecordingLookup:
    """Stand-in for the mmap'd sharded table: records that the gather ran."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        out = torch.zeros(
            (ids.shape[0], EMBED_DIM // NGRAM_HEADS, NGRAM_HEADS),
            dtype=torch.float32,
        )
        return out.add_(1.0)


class RecordingLogger:
    """The assertions are about log lines, so capture them directly.

    caplog would be the obvious tool, but vLLM configures its own handlers and
    propagation is not guaranteed; a stubbed logger is the honest version of
    "did this code say the thing".
    """

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def _log(self, level):
        def emit(msg, *args):
            self.lines.append((level, str(msg) % args if args else str(msg)))

        return emit

    def info(self, msg, *args):
        self._log("info")(msg, *args)

    def warning(self, msg, *args):
        self._log("warning")(msg, *args)

    def __getattr__(self, name):  # noqa: ANN001, ANN201
        return self._log(name)

    def text(self) -> str:
        return "\n".join(line for _, line in self.lines)


class FakePLEEmbedding:
    """The attributes the op actually touches, staging included."""

    def __init__(self, device: str = "cpu") -> None:
        self.lookup = RecordingLookup()
        self.device = device
        self.runtime_dtype = torch.float16
        self.layer_name = "ple"
        self._pinned_ngram_ids = torch.full(
            (BUF_ROWS, NGRAM_HEADS), SENTINEL, dtype=torch.long
        ).pin_memory()
        self._pinned_output = torch.zeros(
            (BUF_ROWS, EMBED_DIM), dtype=torch.float16
        ).pin_memory()
        # Staging state, as the real constructor leaves it.
        self._host_stage_rows = -1
        self._host_check_rows = -1
        self._hostids_disabled = False
        self._staged_branch_captured = False
        self._pinned_gen = torch.zeros(1, dtype=torch.long).pin_memory()
        self._gen_device = (
            torch.zeros(1, dtype=torch.long, device=device) if device != "cpu" else None
        )

    @property
    def ngram_embedding(self):  # noqa: ANN201
        return self.lookup

    def _gen_scratch(self, device):  # noqa: ANN001, ANN201
        # The real one allocates lazily; a capture must not allocate, so tests
        # that capture pre-seed ``_gen_device`` and this returns it.
        if self._gen_device is None:
            self._gen_device = torch.zeros(1, dtype=torch.long, device=device)
        return self._gen_device

    def compare_host_ids(self, *args, **kwargs) -> None:
        return None


@contextlib.contextmanager
def _op_env(
    monkeypatch,
    *,
    mode: str | None,
    capturing: bool,
    device: str = "cpu",
    stage_rows: int | None = None,
    disabled: bool = False,
):
    """Run the op with ``VLLM_GFX906_PLE_HOSTIDS=mode`` and capture state faked."""
    monkeypatch.delenv("VLLM_GFX906_PLE_HOSTIDS", raising=False)
    if mode is not None:
        monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS", mode)
    ple_layer_module._CAPTURE_STAGE.clear()

    embedding = FakePLEEmbedding(device)
    embedding._hostids_disabled = disabled
    if stage_rows is not None:
        embedding._host_stage_rows = stage_rows
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    layer.ple_embedding = embedding  # type: ignore[attr-defined]
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    recorder = RecordingLogger()
    monkeypatch.setattr(ple_layer_module, "logger", recorder)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    # No exit-clear: the entry-clear above already isolates tests, and clearing on
    # exit would empty the very dict the assertions below the `with` read.
    yield embedding, recorder


def _run_op(n: int = ROWS, device: str = "cpu") -> torch.Tensor:
    ids = torch.zeros((n, NGRAM_HEADS), dtype=torch.long, device=device)
    output = torch.full(
        (n, EMBED_DIM), float("nan"), dtype=torch.float16, device=device
    )
    qwen4_exp_amd_ple_ngram_embedding(ids, output, "ple")
    return output


# ------------------------------------------------------- the new clause


def test_capture_with_nothing_staged_takes_the_h2d_branch(monkeypatch) -> None:
    """The padded capture dummy the producer refused must still be capturable."""
    with _op_env(monkeypatch, mode="1", capturing=True) as (emb, _):
        out = _run_op()

    assert emb.lookup.calls == 0, "gather ran during capture: the boot would die"
    # The blocking D2H is gone too, so the planted sentinel survives.
    assert torch.all(emb._pinned_ngram_ids == SENTINEL), "the blocking D2H ran"
    assert torch.all(out == 0), "the captured node must be the staging buffer"
    assert emb._staged_branch_captured is True, (
        "a tripwire that cannot tell capture happened cannot make a fire fatal"
    )
    assert ple_layer_module._CAPTURE_STAGE[ROWS] == {"dummy": 1}


def test_capture_branch_zeroes_the_staging_rows(monkeypatch) -> None:
    """An unsanctioned replay must show up as zeros, not as the last step's rows.

    Capture-time content is what the 2026-09-20 incident served for hours. Zeroing
    converts that into "PLE contributes nothing", which the degeneration smoke
    test catches structurally. Plausible text is the worst output a lookup table
    can produce.
    """
    with _op_env(monkeypatch, mode="1", capturing=True) as (emb, _):
        emb._pinned_output.fill_(3.0)
        _run_op()

    assert torch.all(emb._pinned_output[:ROWS] == 0), "stale rows would replay"


def test_mode_two_does_not_take_the_staged_branch(monkeypatch) -> None:
    """GRAPHMODE=full and HOSTIDS=2 are mutually exclusive, by construction.

    Differential mode keeps the device path in charge and retains the blocking D2H
    on purpose. A graph that replays an H2D of host-staged rows has no device path
    left to disagree with, so mode 2 must fall through even though it is "staging
    enabled". This is why the correctness arm runs on a piecewise boot.
    """
    with _op_env(monkeypatch, mode="2", capturing=True) as (emb, _):
        _run_op()

    assert emb.lookup.calls == 1, "mode 2 must leave the device path in charge"
    assert emb._staged_branch_captured is False
    assert ple_layer_module._CAPTURE_STAGE[ROWS] == {"fell-through": 1}


def test_mode_zero_says_why_the_capture_is_about_to_die(monkeypatch) -> None:
    """HOSTIDS=1 is a prerequisite for GRAPHMODE=full, and the log must say so.

    Without this line a dead boot shows a HIP error and nothing else, which is how
    the first three full-mode attempts got misread as 'FULL is unstable'.
    """
    with _op_env(monkeypatch, mode=None, capturing=True) as (emb, log):
        _run_op()

    assert emb.lookup.calls == 1
    assert ple_layer_module._CAPTURE_STAGE[ROWS] == {"fell-through": 1}
    assert "STAGE@CAPTURE" in log.text()
    assert "VLLM_GFX906_PLE_HOSTIDS=1" in log.text()


def test_row_count_mismatch_is_labelled_apart(monkeypatch) -> None:
    """A staged extent that disagrees with the shape is a padding bug, not an
    empty buffer -- and it is labelled separately, then captured anyway.

    Outside capture a mismatch must not be honored (the device path is reachable
    and the disagreement is unresolved), and ``take_host_stage`` keeps that rule.
    During capture the fixed-extent H2D is still the right node: the graph needs a
    6-row copy either way, and the disagreement is about rows nobody replays from
    capture. Dying there would trade a loud label for a dead boot, so both labels
    are recorded and the summary shows the mismatch.
    """
    with _op_env(monkeypatch, mode="1", capturing=True, stage_rows=ROWS + 1) as (
        emb,
        _,
    ):
        _run_op()

    counts = ple_layer_module._CAPTURE_STAGE[ROWS]
    assert counts == {"staged-mismatch": 1, "dummy": 1}, counts
    assert emb.lookup.calls == 0, "capture died for a reason the boot can survive"
    assert emb._staged_branch_captured is True


def test_exact_match_is_labelled_staged(monkeypatch) -> None:
    with _op_env(monkeypatch, mode="1", capturing=True, stage_rows=ROWS) as (emb, _):
        _run_op()

    assert emb.lookup.calls == 0
    assert emb._staged_branch_captured is True
    assert ple_layer_module._CAPTURE_STAGE[ROWS] == {"staged": 1}


# --------------------------------------------------------- red controls


def test_disabled_layer_falls_through_even_in_mode_one(monkeypatch) -> None:
    """The pre-Phase-1 behaviour, reached by a route that still exists.

    ``disable_host_staging`` after a tripwire leaves the layer unable to stage, so
    the clause must not fire: staging is off, and capturing an H2D of a buffer
    nobody will write would be the frozen-embedding failure wearing a green log
    line. Flipping the ``_hostids_disabled`` read to a bare ``hasattr`` is the
    mutation this catches.
    """
    with _op_env(monkeypatch, mode="1", capturing=True, disabled=True) as (emb, _):
        _run_op()

    assert emb.lookup.calls == 1, "disabled layer took the staged branch"
    assert emb._staged_branch_captured is False
    assert ple_layer_module._CAPTURE_STAGE[ROWS] == {"fell-through": 1}


def test_outside_capture_mode_one_still_uses_the_device_path(monkeypatch) -> None:
    """The clause is capture-scoped. A live step with nothing staged is a step the
    producer refused, and the device path is the correct answer for it."""
    with _op_env(monkeypatch, mode="1", capturing=False) as (emb, _):
        _run_op()

    assert emb.lookup.calls == 1
    assert ple_layer_module._CAPTURE_STAGE == {}, "attestation is capture-only"


# ------------------------------------------------------------ attestation


def test_summary_flushes_once_on_the_first_live_step(monkeypatch) -> None:
    """One line per capture shape, after capture, not one per layer per iteration.

    The op runs once per PLE layer per capture iteration, so logging inside the
    branch would print a line per layer per shape per round and bury the one
    number the line exists to report.
    """
    with _op_env(monkeypatch, mode="1", capturing=True) as (emb, log):
        _run_op(n=4)
        _run_op(n=4)
        _run_op(n=8)
        assert set(ple_layer_module._CAPTURE_STAGE) == {4, 8}
        # Still capturing: nothing logged yet.
        assert "capture shape" not in log.text()

        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        _run_op(n=4)  # first live step

    text = log.text()
    assert text.count("PLE STAGE@CAPTURE: capture shape") == 2
    assert "n=4" in text and "n=8" in text
    assert "dummy=2" in text, "the count of layer calls at that shape is the point"
    assert ple_layer_module._CAPTURE_STAGE == {}, "the flush must clear the round"

    # A second live step says nothing more.
    with _op_env(monkeypatch, mode="1", capturing=False) as (_, log2):
        _run_op(n=4)
    assert "capture shape" not in log2.text()


# ------------------------------------------------- real CUDA graph capture

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA/HIP device"
)


@contextlib.contextmanager
def _gpu_op_env(monkeypatch, *, mode: str):
    """The same wiring with real tensors, real pinned memory and real capture."""
    monkeypatch.setenv("VLLM_GFX906_PLE_HOSTIDS", mode)
    ple_layer_module._CAPTURE_STAGE.clear()

    embedding = FakePLEEmbedding(device="cuda")
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    layer.ple_embedding = embedding  # type: ignore[attr-defined]
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    monkeypatch.setattr(ple_layer_module, "logger", RecordingLogger())
    try:
        yield embedding
    finally:
        ple_layer_module._CAPTURE_STAGE.clear()


def _capture(embedding, *, n: int = 4) -> tuple:
    """Warm up on a side stream, then capture the op. Returns (graph, ids, out)."""
    ids = torch.zeros((n, NGRAM_HEADS), dtype=torch.long, device="cuda")
    out = torch.zeros((n, EMBED_DIM), dtype=torch.float16, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Warm-up runs the device path (not capturing), which is also what makes
        # the allocator hand the capture a pool it can use.
        qwen4_exp_amd_ple_ngram_embedding(ids, out, "ple")
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qwen4_exp_amd_ple_ngram_embedding(ids, out, "ple")
    return graph, ids, out


@needs_cuda
def test_real_capture_delivers_rows_the_host_wrote_after_capture(monkeypatch) -> None:
    """The handoff, end to end, on this box's own runtime.

    Capture leaves zeros in the staging buffer; the host then writes rows and the
    replay must deliver THOSE. If capture had frozen the values instead of the
    address, this is the test that says so.
    """
    with _gpu_op_env(monkeypatch, mode="1") as emb:
        graph, _ids, out = _capture(emb)
        assert emb._staged_branch_captured is True
        assert ple_layer_module._CAPTURE_STAGE[4] == {"dummy": 1}

        emb._pinned_output[:4].fill_(7.0)
        emb._pinned_gen[0] = 99
        graph.replay()
        torch.cuda.synchronize()

        assert torch.all(out[:4] == 7.0), "replay did not deliver the staged rows"
        assert int(emb._gen_device[0]) == 99, (
            "the in-graph generation node did not follow: the P1.5 tripwire would "
            "be reporting what capture left behind"
        )


@needs_cuda
def test_real_capture_without_host_staging_still_dies(monkeypatch) -> None:
    """Red control, real: with staging off the same capture must fail.

    Placed last on purpose -- an illegal operation during capture invalidates the
    capture, and if this runtime reacted worse than that it should damage this
    test and nothing before it.
    """
    with _gpu_op_env(monkeypatch, mode="0") as emb, pytest.raises(Exception) as excinfo:
        _capture(emb)
    message = str(excinfo.value).lower()
    assert "capture" in message or "capturing" in message, (
        f"failed for an unrelated reason: {excinfo.value}"
    )
