# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the E1 Phase 0 capture probe (``PLE_CAPTURE_PROBE``).

The probe exists to answer one question cheaply: what is the ceiling of
full-graph decode (``docs/e1-full-graph-plan.md`` section 3)? It makes the PLE
custom op capturable -- during a capture it omits the device->host id copy and
the host shard gather, and records only the host->device copy out of
``_pinned_output``.

Because the probe deliberately *removes* work, the dangerous failure mode is
leakage: the probe firing outside a capture, or the capture check being
decorative while the knob alone drives the behaviour. The tests below are built
to catch both.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module  # noqa: E402
from vllm.models.qwen4_exp.amd.ple_layer import (  # noqa: E402
    Qwen4ExpPLELayer,
    qwen4_exp_amd_ple_ngram_embedding,
)

ROWS = 6
EMBED_DIM = 8
NGRAM_HEADS = 2
SENTINEL = -12345
PLE_LAYER_SRC = str(
    # Read the source of the module this test actually imported, not of a
    # hardcoded checkout: an absolute path here silently graded a *different*
    # tree than the one under test, and putting it on sys.path poisoned every
    # later test file in the same pytest process.
    Path(ple_layer_module.__file__).resolve()
)


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
        # Distinctive and non-zero: a leaked probe path produces zeros instead.
        out.add_(1.0)
        return out


class FakePLEEmbedding:
    def __init__(self) -> None:
        self.lookup = RecordingLookup()
        self._pinned_ngram_ids = torch.full(
            (ROWS, NGRAM_HEADS), SENTINEL, dtype=torch.long
        )
        self._pinned_output = torch.zeros((ROWS, EMBED_DIM), dtype=torch.float16)

    # The op calls ``ple_embedding.ngram_embedding(ids)``.
    @property
    def ngram_embedding(self):  # noqa: ANN201
        return self.lookup


@contextlib.contextmanager
def _op_env(monkeypatch, *, probe: str | None, capturing: bool):
    """Run with the probe env set as ``probe``, capture state as ``capturing``."""
    monkeypatch.delenv("VLLM_GFX906_PLE_CAPTURE_PROBE", raising=False)
    if probe is not None:
        monkeypatch.setenv("VLLM_GFX906_PLE_CAPTURE_PROBE", probe)
    ple_layer_module._reset_capture_probe_cache()

    embedding = FakePLEEmbedding()
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    layer.ple_embedding = embedding  # type: ignore[attr-defined]
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    try:
        yield embedding
    finally:
        ple_layer_module._reset_capture_probe_cache()


def _run_op(n: int = ROWS) -> torch.Tensor:
    ids = torch.zeros((n, NGRAM_HEADS), dtype=torch.long)
    output = torch.full((n, EMBED_DIM), float("nan"), dtype=torch.float16)
    qwen4_exp_amd_ple_ngram_embedding(ids, output, "ple")
    return output


# ------------------------------------------------------------- normal path


def test_probe_unset_gathers_even_during_capture(monkeypatch) -> None:
    """Default-off: with no env var the op gathers regardless of capture state.

    This is the shipping behaviour, and it is what makes ``GRAPHMODE=full`` die
    at ``hipErrorStreamCaptureUnsupported`` (mtp-cudagraph-profile-crash.md
    section 13.1) instead of silently capturing a broken PLE. Flipping this
    default would trade a loud boot failure for a silent wrong answer.
    """
    with _op_env(monkeypatch, probe=None, capturing=True) as emb:
        out = _run_op()
    assert emb.lookup.calls == 1
    assert torch.all(out == 1)


def test_probe_on_outside_capture_still_gathers(monkeypatch) -> None:
    """Capture-scoped, not step-scoped: piecewise regions must be untouched."""
    with _op_env(monkeypatch, probe="1", capturing=False) as emb:
        out = _run_op()
    assert emb.lookup.calls == 1
    assert torch.all(out == 1)


# ------------------------------------------------------------- probe path


def test_probe_during_capture_skips_gather_and_d2h(monkeypatch) -> None:
    """Nothing host-side runs, so only the H2D gets frozen into the graph."""
    with _op_env(monkeypatch, probe="1", capturing=True) as emb:
        out = _run_op()

    assert emb.lookup.calls == 0, "gather ran during capture: PLE would be frozen"
    # The D2H is gone too, so the pinned id buffer keeps the value we planted.
    assert torch.all(emb._pinned_ngram_ids == SENTINEL), "the D2H ran"
    # The replayed content is the zeroed staging buffer, not garbage.
    assert torch.all(out == 0), "probe output must be the zeroed staging buffer"


def test_pinned_output_is_zeroed_at_construction() -> None:
    """A replayed PLE reads ``_pinned_output``; ``empty`` would freeze garbage.

    The 2026-09-20 incident (section 13.2) served capture-time content for
    hours; before that, uninitialised pinned ids caused out-of-range table
    lookups and two engine deaths. Zeroing is what makes the probe's "PLE
    contributes nothing" claim mean *nothing* rather than *whatever was in the
    buffer*. Asserted against source because building a real Qwen4ExpPLELayer
    to read one constructor line would mean mapping 20M-vocab shards.
    """
    with open(PLE_LAYER_SRC) as handle:
        src = handle.read()
    assert "_pinned_output = torch.zeros(" in src
    assert "_pinned_output = torch.empty(" not in src


# ------------------------------------------------------------- red controls


def test_deleting_the_capture_check_would_leak_into_eager(monkeypatch) -> None:
    """Mutant simulation of a guard that is just ``_capture_probe_enabled()``.

    With the capture check removed, an eager/piecewise call sees the same
    ``True`` a captured call sees, so the gather is skipped on the live path and
    PLE contributes zeros to real traffic -- wrong output, no crash, no log
    line beyond the boot warning. That is the price of the one-line
    "simplification"; this test shows the behaviour it buys.
    """
    with _op_env(monkeypatch, probe="1", capturing=False) as emb:
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        out = _run_op()
    assert emb.lookup.calls == 0
    assert torch.all(out == 0), "mutant: live steps would serve zero PLE"


def test_probe_default_is_off_behaviourally() -> None:
    """Unset env must mean False, not "unset means on". Reads the real reader."""
    os.environ.pop("VLLM_GFX906_PLE_CAPTURE_PROBE", None)
    ple_layer_module._reset_capture_probe_cache()
    try:
        assert ple_layer_module._capture_probe_enabled() is False
    finally:
        ple_layer_module._reset_capture_probe_cache()
