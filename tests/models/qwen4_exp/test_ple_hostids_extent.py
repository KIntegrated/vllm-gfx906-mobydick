# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P1.14: the differential must report the extent the ngram fill consumed.

What the instrument is for. Boot #12 -- a 4-token single-request decode step --
dumped a packed workspace row 0 of ``[17383, 653, EOS, EOS, EOS, EOS]``: the fill
wrote two cells for a four-token step. Padding in the ids buffer is zero and
never EOS (``input_buffers.input_ids[:num_tokens].zero_()``), so cells 2-3 were
not written by anything, and a brute force over plausible staged boundaries
reproduces that row only when the first query length is 2. Meanwhile the gate
printed ``qsl device=[0,4]`` and concluded the inputs agreed.

The two statements are not contradictory because **the gate and the compute do not
read the same object.** The gate compares ``input_batch.query_start_loc`` sliced
to the real ``num_reqs``; forward is handed
``ple_query_start_loc[: num_reqs_padded + 1]`` (``model_state.prepare_inputs``), a
different buffer at the capture-padded width -- for that step ``[0,4,4,4]``. Two
buffers, two widths, one verdict. Nothing in the harness had ever read the buffer
the fill math consumed, and that is the file this suite covers.

Why the workspace is the witness and not another opinion. ``_compute_ngram_ids``
opens with ``packed.fill_(eos_token_id)`` and ``positions`` is
``positions_buffer[:num_tokens]`` -- a registered ``arange`` buffer that is never
written per step -- so a scatter over ``ids_len`` tokens writes cells
``0 .. qsl[1] - 1`` of row 0 and nothing past them. ``extent_row0`` is therefore
a measurement of the first boundary the fill listened to, taken off the device's
own scratch, and the line's whole job is to put it next to both boundary vectors
so a boot says which of the three numbers is the odd one out.

The evidence structure, and the two red controls:

* ``test_the_extent_line_reads_the_device_qsl_and_not_the_host_one`` is the one
  that matters. Three different qsl vectors are in scope (host ``[0,4,4,4]``,
  gate ``[0,4]``, device ``[0,2,2,2]``) and the line must show the *device* one.
  An instrument that printed the host value compared with itself would pass every
  other test in this file.
* ``test_the_op_adds_no_device_read_when_the_knob_is_off`` spies
  ``Tensor.cpu``/``Tensor.item``/``Tensor.numpy`` across a whole device-path
  lookup, because the house rule after P1.9 is that a check must not be priced in
  the same currency as the bug it catches.
* ``test_packed_dump_prints_every_row_of_the_workspace`` pins the geometry
  regression: the old print sized itself with the compare-time qsl, so every dump
  ever taken showed exactly one row of a three-row workspace.

Everything here is CPU tensors and stand-in ``self`` objects bound to the real
methods -- no GPU, no engine, no timing.
"""

import contextlib
import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

ple = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_layer",
    reason="qwen4_exp AMD model code is not importable in this build",
)
hostids = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_hostids",
    reason="host-side PLE id module is not importable in this build",
)
model_state_module = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.model_state",
    reason="qwen4_exp AMD model state is not importable in this build",
)

Embedding = ple.Qwen4ExpNGramEmbedding

NGRAM_SIZE = 3
HEADS_PER_NGRAM = 1
HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM  # 2
EMBED_DIM = 8
HEAD_DIM = EMBED_DIM // HEADS  # 4
CTX = NGRAM_SIZE - 1
CAPACITY = 8
EOS = 0
SENTINEL = -12345
ROWS = 4  # boot #12's step: one request, four rows (1 bonus + 3 drafts)


class ExtentEmbedding:
    """Stand-in ``self`` bound to the real diagnostic methods.

    Same contract as ``StagingEmbedding`` in ``test_ple_host_stage.py``: the
    behaviour under test is production code reached through method references, and
    only the state -- buffers and counters -- is local.
    """

    _consumed_report = Embedding._consumed_report
    _packed_dump = Embedding._packed_dump
    _id_dump = Embedding._id_dump
    _inputs_agree = Embedding._inputs_agree
    _ctx_leg = Embedding._ctx_leg
    _note_input_refusal = Embedding._note_input_refusal
    compare_host_ids = Embedding.compare_host_ids
    stage_host_step = Embedding.stage_host_step
    note_draft_provenance = Embedding.note_draft_provenance
    disable_hostids = Embedding.disable_hostids
    _host_hash_buffers = Embedding._host_hash_buffers

    def __init__(self, rows: int = ROWS, capacity: int = CAPACITY) -> None:
        self.layer_name = "ple"
        self.ngram_size = NGRAM_SIZE
        self.heads_per_ngram = HEADS_PER_NGRAM
        self.eos_token_id = EOS
        self.runtime_dtype = torch.float16
        self.layer_multipliers = torch.tensor([-12345, 67891, -3333], dtype=torch.long)
        self.ngram_heads_vocab_sizes = torch.tensor(
            [1000003, 2000003], dtype=torch.long
        )
        self.ngram_heads_offsets = torch.tensor([0, 1000003], dtype=torch.long)
        self._host_hash_cache = None
        self._pinned_ngram_ids = torch.full(
            (capacity, HEADS), SENTINEL, dtype=torch.long
        )
        self._pinned_output = torch.zeros((capacity, EMBED_DIM), dtype=torch.float16)
        self._host_stage_rows = -1
        self._host_check_rows = -1
        self._host_check_ids: torch.Tensor | None = None
        self._host_check_inputs: tuple[Any, ...] | None = None
        self._host_check_dev_inputs: Any = None
        self._host_check_draft: tuple[Any, Any, int] | None = None
        # P1.14: the qsl view forward is handed. None means "the stager handed no
        # consumed buffer", which the line must report as an absence, not as the
        # gate's value.
        self._host_consumed_qsl: Any = None
        self.padded_buffer = torch.full((4, 64), EOS, dtype=torch.long)
        self._hostids_disabled = False
        self._host_stage_steps = 0
        self._host_id_mismatches = 0
        self._host_id_matches = 0
        self._host_input_checks = 0
        self._host_input_skips = 0
        self._host_ctx_unchecked = 0
        self.lookup_calls = 0

    @property
    def ngram_embedding(self):  # noqa: ANN201
        def _gather(ids: torch.Tensor) -> torch.Tensor:
            self.lookup_calls += 1
            out = torch.zeros((ids.shape[0], HEAD_DIM, HEADS), dtype=torch.float32)
            out[...] = ids.abs().sum(dim=-1).reshape(-1, 1, 1)
            return out

        return _gather


def _arm(
    emb: ExtentEmbedding,
    *,
    ids_len: int = ROWS,
    host_qsl: tuple[int, ...] = (0, 4, 4, 4),
    gate_qsl: tuple[int, ...] = (0, 4),
    consumed_qsl: tuple[int, ...] | None = (0, 2, 2, 2),
    filled: tuple[int, ...] = (17383, 653),
    dev_tokens: np.ndarray | None = None,
) -> np.ndarray:
    """Stage one compared step, with all three qsl vectors distinct on purpose.

    ``host_qsl`` is what the host mirror built, ``gate_qsl`` is the runner's array
    the input gate compares against, ``consumed_qsl`` is the buffer forward reads.
    Boot #12 had host and gate agreeing at 4 while the fill behaved as if the
    first boundary were 2, so the fixture reproduces exactly that shape: any
    implementation that reports the wrong vector shows up as the wrong numbers.

    ``filled`` is written into row 0 of the workspace, leaving the rest at the
    fill value, so ``extent_row0 == len(filled)`` by construction.
    """
    tokens: np.ndarray = np.arange(ids_len, dtype=np.int64) + 100
    ids: np.ndarray = np.arange(ids_len * HEADS, dtype=np.int64).reshape(ids_len, HEADS)
    packed = torch.full((3, 64), EOS, dtype=torch.long)
    for i, value in enumerate(filled):
        packed[0, i] = value
    ctx: np.ndarray = np.full((ids_len, CTX), 7, dtype=np.int64)

    emb._host_check_ids = torch.from_numpy(ids.copy())
    emb._host_check_rows = ids_len
    emb._host_check_inputs = (
        tokens,
        np.array(host_qsl, dtype=np.int64),
        ctx,
        packed,
    )
    emb._host_consumed_qsl = (
        None if consumed_qsl is None else torch.tensor(consumed_qsl, dtype=torch.int32)
    )
    emb._host_check_dev_inputs = (
        torch.from_numpy(
            tokens.copy() if dev_tokens is None else np.asarray(dev_tokens)
        ),
        torch.tensor(gate_qsl, dtype=torch.int32),
        torch.from_numpy(np.full((CAPACITY, CTX), 7, dtype=np.int64)),
    )
    return ids


def _diff_lines(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("PLE DIFF")
    ]


# ---------------------------------------------------------------------------
# the line itself
# ---------------------------------------------------------------------------


def test_the_extent_line_reports_every_field_once(monkeypatch, caplog) -> None:
    """One line, per compared step, with the four fields the boot needs.

    ``ids_len`` is what the compute handed the op, ``qsl`` is the buffer the fill
    read *with its width*, ``packed.shape`` is the workspace's real geometry, and
    ``extent_row0`` is the extent the fill demonstrably wrote -- here two cells of
    a four-token step, which is boot #12's signature.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    ids = _arm(emb)

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.from_numpy(ids))

    lines = _diff_lines(caplog)
    assert len(lines) == 1, f"expected exactly one line per compared step: {lines}"
    line = lines[0]
    assert "PLE DIFF: ids_len=4" in line, line
    assert "qsl(width=4)=[0,2,2,2]" in line, line
    assert "packed.shape=(3,64)" in line, line
    assert "extent_row0=2" in line, line
    assert "gate_qsl(width=2)=[0,4]" in line, line
    # The comparison itself still runs, and this fixture agrees on ids, so nothing
    # is disabled by a line that merely describes the step.
    assert emb._hostids_disabled is False
    assert emb._host_id_matches == 1


def test_the_extent_line_reads_the_device_qsl_and_not_the_host_one(
    monkeypatch, caplog
) -> None:
    """RED CONTROL. The line must print the device buffer, not the host's.

    Three qsl vectors are in scope and all differ: the host mirror's
    ``[0,4,4,4]``, the gate's ``[0,4]``, and the device buffer's ``[0,2,2,2]``.
    Only the last one is what the fill math consumed, so only the last one makes
    ``extent_row0=2`` make sense. An instrument that printed the host value
    alongside itself -- which is what the gate did for twelve boots -- would pass
    every other test in this file and tell us nothing at 3 a.m.

    How this control was verified to fail for the right reason: with the
    ``consumed`` read in ``_consumed_report`` pointed at ``self._host_check_inputs
    [1]`` (the host's array) instead of ``self._host_consumed_qsl``, this test
    fails on the ``[0,2,2,2]`` assert with the line showing
    ``qsl(width=4)=[0,4,4,4]``. Pointing it at the gate's ``dev_inputs[1]`` fails
    it on the same assert showing ``qsl(width=2)=[0,4]``. Both mutations were run,
    not imagined.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    ids = _arm(emb)

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.from_numpy(ids))

    line = _diff_lines(caplog)[0]
    assert "qsl(width=4)=[0,2,2,2]" in line, line
    assert "[0,4,4,4]" not in line, f"the line reported the HOST qsl: {line}"
    # The gate field is ``gate_qsl(width=2)=[0,4]``; the leading space keeps this
    # assertion from matching its tail.
    assert " qsl(width=2)=" not in line, f"the line reported the GATE qsl: {line}"


def test_a_missing_consumed_buffer_is_named_and_not_inherited(
    monkeypatch, caplog
) -> None:
    """No silent fallback to the array that happens to be nearby.

    If the stager never handed the consumed view over, the field says so. Falling
    back to the gate's or the host's buffer here would recreate the exact confusion
    this instrument exists to end, and a bare ``n/a`` that reads as a passing leg is
    the failure mode the packed leg was rewritten for.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    ids = _arm(emb, consumed_qsl=None)

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.from_numpy(ids))

    line = _diff_lines(caplog)[0]
    assert "qsl(n/a(no buffer was handed over))" in line, line
    assert "[0,4,4,4]" not in line, line


def test_the_line_prints_before_the_input_gate_refuses(monkeypatch, caplog) -> None:
    """The gate refusing is the step worth reading, so the line comes first.

    A comparison the harness cannot attribute is precisely the step where the
    extent is informative; printing after the gate would have hidden every MTP
    refusal behind a line that only ever appears on steps that already make sense.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    other: np.ndarray = np.arange(ROWS, dtype=np.int64) + 100
    other[1] = 4  # the token a different MTP pass had put in the runner's array
    ids = _arm(emb, dev_tokens=other)

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.from_numpy(ids))

    assert emb._host_input_skips == 1, "the gate did not refuse the disagreement"
    assert emb._hostids_disabled is False, "a refusal must not disable the mirror"
    lines = _diff_lines(caplog)
    assert len(lines) == 1, lines
    assert "extent_row0=2" in lines[0], lines[0]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "SKIPPED a comparison" in text, text
    # Order is asserted, not assumed: the line must precede the refusal.
    assert text.index("PLE DIFF") < text.index("SKIPPED a comparison")


def test_the_line_prints_when_the_row_counts_disagree(monkeypatch, caplog) -> None:
    """The other silent return, now loud.

    ``compare_host_ids`` used to bail without a word when the compute produced a
    different number of rows than the host staged -- the one shape that *is* "the
    extent came from another step". The line now prints with both numbers, and the
    comparison still refuses: describing the step is not the same as trusting it.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    _arm(emb, ids_len=ROWS)
    short = torch.full((2, HEADS), 5, dtype=torch.long)

    with caplog.at_level("INFO"):
        emb.compare_host_ids(short)

    line = _diff_lines(caplog)[0]
    assert "ids_len=2" in line, line
    assert "stage_rows=4" in line, line
    assert emb._host_input_skips == 0, "the comparison ran anyway"
    assert emb._hostids_disabled is False


def test_no_line_and_no_comparison_when_nothing_was_staged(monkeypatch, caplog) -> None:
    """Mode 2 without a staged step costs no line.

    The rate claim is "one line per *compared* step", not "one line per op entry":
    under MTP the op is entered several times per step and only the staged one has
    anything to say.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    emb._host_check_rows = -1

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.zeros((ROWS, HEADS), dtype=torch.long))

    assert _diff_lines(caplog) == []


def test_mode_one_prints_nothing(monkeypatch, caplog) -> None:
    """The report belongs to the differential, not to the shipping path.

    Mode 1 never stages a comparison, so it never reaches the line; the explicit
    mode check is the belt to that brace, and it is what makes the knob-off cost
    statement in ``_consumed_report`` testable rather than aspirational.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 1)
    emb = ExtentEmbedding()
    ids = _arm(emb)

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.from_numpy(ids))

    assert _diff_lines(caplog) == []


def test_a_disabled_instrument_stops_reporting(monkeypatch, caplog) -> None:
    """After the first mismatch the mode is off, and so is its log line."""
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    emb.disable_hostids("test")
    ids = _arm(emb)

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.from_numpy(ids))

    assert _diff_lines(caplog) == []


# ---------------------------------------------------------------------------
# the workspace's geometry
# ---------------------------------------------------------------------------


def test_packed_dump_prints_every_row_of_the_workspace(monkeypatch) -> None:
    """RED: the old print sized itself with the compare-time qsl.

    ``n_reqs = min(pk.shape[0], qsl.shape[0] - 1)`` meant a one-request host step
    against a three-request workspace printed exactly one row -- and the rows the
    host never staged are the ones a second pass in the same step writes, which is
    the case the leg exists to catch. Every dump ever taken was blind in exactly
    that direction.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    packed = torch.tensor(
        [[10, 11, EOS, EOS], [21, EOS, EOS, EOS], [31, EOS, EOS, EOS]],
        dtype=torch.long,
    )
    inputs: tuple[Any, ...] = (
        np.arange(ROWS, dtype=np.int64) + 100,
        np.array([0, 4], dtype=np.int64),  # one request, as the host staged it
        np.zeros((1, CTX), dtype=np.int64),
        packed,
    )

    text = emb._packed_dump(inputs, [0])

    assert "packed.shape=(3,4)" in text, text
    assert "[10,11,0,0]" in text, text
    assert "[21,0,0,0]" in text, f"row 1 of the workspace was not printed: {text}"
    assert "[31,0,0,0]" in text, f"row 2 of the workspace was not printed: {text}"
    # And the old behaviour, positively gone: one row, sized from the qsl.
    assert "packed device=[[10,11,0,0]]" not in text, text


def test_packed_dump_caps_rows_and_says_that_it_did() -> None:
    """The cap exists so the line cannot grow with the batch; it must be visible.

    Reuse of ``_bounded_ids``' ``+N more`` marker is deliberate -- one convention
    for "this print stopped early", not one per dump leg.
    """
    emb = ExtentEmbedding()
    inputs: tuple[Any, ...] = (
        np.arange(ROWS, dtype=np.int64) + 100,
        np.array([0, 4], dtype=np.int64),
        np.zeros((1, CTX), dtype=np.int64),
        torch.arange((ple._PACKED_DUMP_ROWS + 4) * 4, dtype=torch.int64)
        .reshape(ple._PACKED_DUMP_ROWS + 4, 4)
        .contiguous(),
    )

    text = emb._packed_dump(inputs, [0])

    assert "+4 more]" in text, text
    assert text.count("[") == text.count("]")


def test_packed_dump_of_a_wide_workspace_does_not_print_the_whole_row() -> None:
    """Cells are capped too: the real workspace is ``max_total_tokens`` wide."""
    emb = ExtentEmbedding()
    wide = torch.full((1, 4096), 3, dtype=torch.long)
    inputs: tuple[Any, ...] = (
        np.arange(ROWS, dtype=np.int64) + 100,
        np.array([0, 4], dtype=np.int64),
        np.zeros((1, CTX), dtype=np.int64),
        wide,
    )

    text = emb._packed_dump(inputs, [0])

    assert f"... +{4096 - ple._PACKED_DUMP_CELLS} more]" in text, text


# ---------------------------------------------------------------------------
# the extent measurement
# ---------------------------------------------------------------------------


def test_fill_extent_counts_to_the_first_fill_cell() -> None:
    """The scratch's extent is the fill's own statement of the boundary."""
    assert ple._fill_extent(np.array([10, 11, EOS, EOS, EOS]), EOS) == 2
    assert ple._fill_extent(np.array([EOS, 1, 2]), EOS) == 0
    assert ple._fill_extent(np.array([1, 2, 3]), EOS) == 3
    assert ple._fill_extent(np.array([], dtype=np.int64), EOS) == 0


def test_fill_extent_is_a_lower_bound_when_a_token_is_the_fill_value() -> None:
    """The one thing the measurement cannot prove, stated as a test.

    A real token equal to ``eos_token_id`` truncates the count, so the field is
    "the fill wrote at least this many cells", never "exactly this many". A
    diagnostic that overstates what the scratch proves is worse than one that
    says "at most".
    """
    assert ple._fill_extent(np.array([10, EOS, 12, EOS]), EOS) == 1


def test_a_report_that_raises_says_so_instead_of_killing_the_step(
    monkeypatch, caplog
) -> None:
    """The one failure ``_consumed_report`` cannot guard from the inside.

    Its own try/except covers the reads; it cannot cover its own absence, which is
    what a mismatched ``self`` produces. The call site therefore renders a named
    absence and lets the comparison continue -- P1.9's snapshot copies died exactly
    this way, by a diagnostic raising inside the step it was watching.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    ids = _arm(emb)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    emb._consumed_report = _boom  # type: ignore[method-assign]

    with caplog.at_level("INFO"):
        emb.compare_host_ids(torch.from_numpy(ids))

    line = _diff_lines(caplog)[0]
    assert line == "PLE DIFF: n/a(the report raised RuntimeError)", line
    assert emb._host_id_matches == 1, "the comparison itself was lost with the line"


def test_the_report_survives_a_workspace_it_cannot_read() -> None:
    """A diagnostic must not kill a step, and must not read as a passing leg."""
    emb = ExtentEmbedding()
    staged: tuple[Any, ...] = (
        np.arange(ROWS, dtype=np.int64) + 100,
        np.array([0, 4], dtype=np.int64),
        np.zeros((1, CTX), dtype=np.int64),
    )  # no fourth member: the stager handed no workspace
    emb._host_check_inputs = staged
    emb.padded_buffer = None

    line = emb._consumed_report(ROWS, ROWS, None)

    assert "packed.shape=n/a(no workspace)" in line, line
    assert "extent_row0=n/a(no workspace)" in line, line


# ---------------------------------------------------------------------------
# wiring: the consumed qsl has to come from prepare_inputs, not from a fixture
# ---------------------------------------------------------------------------


def test_staging_hands_the_layer_the_buffer_forward_reads() -> None:
    """``stage_host_step`` must keep the reference, not a copy or a value.

    The report's whole claim is that it prints the cells the fill read, so the
    instrument has to hold the same object the model was handed.
    """
    monkey = torch.tensor([0, 2, 2, 2], dtype=torch.int32)
    emb = ExtentEmbedding()
    emb._host_consumed_qsl = None

    ok = emb.stage_host_step(
        np.arange(4, dtype=np.int64) + 100,
        np.array([0, 4], dtype=np.int64),
        np.zeros((1, CTX), dtype=np.int64),
        gather=False,
        consumed_qsl=monkey,
    )

    assert ok is True
    assert emb._host_consumed_qsl is monkey


def test_prepare_inputs_forwards_the_same_view_it_hands_the_model() -> None:
    """The producer leg, asserted on the wiring that exists.

    One local -- ``ple_query_start_loc[: num_reqs_padded + 1]`` -- goes into
    ``model_inputs`` AND into the staging call as ``consumed_qsl``. Identity is the
    property, so it is asserted structurally: same name, no second slice in between.

    The wiring moved out of ``prepare_inputs`` into :meth:`_ple_step_inputs` when the
    dispatch veto needed the staging decision before ``prepare_attn`` (that file is
    the shared runner's, and the decision has to be askable on its own). The property
    did not move with the prose: the same view still goes to both consumers. What is
    NEW and worth pinning here is that there is exactly ONE staging call site per
    step -- if ``prepare_inputs`` re-derived what the early hook already built, a
    step could pay for the fill and the mirror repair twice, and the second repair
    would be measured against a count the first one already fixed.
    """
    src = inspect.getsource(model_state_module.Qwen4ExpModelState._ple_step_inputs)
    prep = inspect.getsource(model_state_module.Qwen4ExpModelState.prepare_inputs)
    refresh = inspect.getsource(
        model_state_module.Qwen4ExpModelState._refresh_ple_query_start_loc
    )

    # Same name into both consumers, no re-slice between them. The kwargs form became
    # a dict literal when the block moved here; what is being asserted is the identity
    # of the value, not the punctuation that puts it in the dict.
    assert '"query_start_loc": query_start_loc' in src, src
    assert "consumed_qsl=query_start_loc" in src, src
    assert prep.count("_stage_host_ple_ids(") == 0, (
        "prepare_inputs stages on its own as well as through _ple_step_inputs: the "
        "dispatch veto's whole saving is that the step is prepared once, early"
    )
    assert src.count("_stage_host_ple_ids(") == 1, src
    # The slice used to sit in prepare_inputs; it now lives in the refresh method so
    # the whole-buffer write and the view handed forward cannot disagree. The property
    # is unchanged -- exactly one slice, handed to both consumers -- it is just asked
    # of two functions now.
    assert "self.ple_query_start_loc[" not in src and (
        "self.ple_query_start_loc[" not in prep
    ), (
        "the staged qsl is sliced again outside the refresh method: the differential "
        "can then report one view while the model reads another"
    )
    assert refresh.count("return dst[") == 1, (
        "the refresh method must hand forward exactly one view of the buffer it just "
        "wrote, or the captured graph and the report can describe different slices"
    )


# ---------------------------------------------------------------------------
# the no-sync contract
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _op_env(emb: ExtentEmbedding, monkeypatch, *, capturing: bool = False):
    """Run the real op impl against ``emb`` (same fixture as the staging suite)."""
    layer = ple.Qwen4ExpPLELayer.__new__(ple.Qwen4ExpPLELayer)
    layer.ple_embedding = emb  # type: ignore[attr-defined]
    monkeypatch.setattr(
        ple,
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


@contextlib.contextmanager
def _watch_reads():
    """Count every host-visible tensor read the op body could have added.

    Deliberately NOT counted: ``pinned_ids.copy_(ngram_ids, non_blocking=False)``.
    That blocking D2H is the device path -- it runs identically with the knob off,
    and it is the wait the whole host-ids project exists to remove. What is counted
    here is the surface the *instrument* could add on top: ``.cpu()``, ``.numpy()``
    and ``.item()``, the last being the one that turns a value into a sync.
    """
    counts = {"cpu": 0, "numpy": 0, "item": 0}
    original = {name: getattr(torch.Tensor, name) for name in counts}

    def _spy(name, fn):  # noqa: ANN001, ANN202
        def wrapper(self, *args, **kwargs):  # noqa: ANN002, ANN003
            counts[name] += 1
            return fn(self, *args, **kwargs)

        return wrapper

    for name, fn in original.items():
        setattr(torch.Tensor, name, _spy(name, fn))
    try:
        yield counts
    finally:
        for name, fn in original.items():
            setattr(torch.Tensor, name, fn)


def _run_op(emb: ExtentEmbedding, n: int = ROWS) -> torch.Tensor:
    ids = torch.arange(n * HEADS, dtype=torch.long).reshape(n, HEADS)
    output = torch.full((n, EMBED_DIM), float("nan"), dtype=torch.float16)
    ple.qwen4_exp_amd_ple_ngram_embedding(ids, output, "ple")
    return output


def test_the_op_adds_no_device_read_when_the_knob_is_off(monkeypatch, caplog) -> None:
    """With the knob off the op must not touch the host-visible read surface.

    The instrument's cost rule is absolute because P1.9's was not: a check priced
    in the currency of the bug it catches changes the bug. Everything P1.14 reads
    lives behind ``_host_check_rows``, which only mode 2 ever sets.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 0)
    emb = ExtentEmbedding()

    with _op_env(emb, monkeypatch), _watch_reads() as counts, caplog.at_level("INFO"):
        out = _run_op(emb)

    assert emb.lookup_calls == 1, "the device path did not run"
    assert counts == {"cpu": 0, "numpy": 0, "item": 0}, counts
    assert _diff_lines(caplog) == []
    assert not torch.all(torch.isnan(out)), "the device path produced no output"


def test_the_differential_reads_the_buffers_it_reports(monkeypatch, caplog) -> None:
    """The same spy, armed: the reads it claims to make, it makes.

    A no-sync test on its own can be satisfied by an instrument that never runs.
    This is the pair that makes the off-case assertion mean "the cost is gated",
    not "the cost is absent because the code is dead". Note ``item`` stays zero
    even armed: the report reads whole arrays into numpy and never asks a tensor
    for a scalar.
    """
    monkeypatch.setattr(hostids, "hostids_mode", lambda: 2)
    emb = ExtentEmbedding()
    _arm(emb)

    with _op_env(emb, monkeypatch), _watch_reads() as counts, caplog.at_level("INFO"):
        _run_op(emb)

    assert counts["cpu"] >= 1, counts
    assert counts["item"] == 0, counts
    assert len(_diff_lines(caplog)) == 1, _diff_lines(caplog)
