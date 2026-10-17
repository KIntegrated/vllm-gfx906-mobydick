# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Joo Chung
"""The static PLE boundary array must describe the step that reads it (P1 fix).

Boot #13 caught the defect this suite pins. A one-request, four-token step had its
ngram fill scatter positions 2 and 3 into **row 2** of the workspace instead of row 0,
and the scratch identified the boundary array it read: the graph was captured at three
requests, so it indexes four entries of ``ple_query_start_loc`` while
``prepare_inputs`` refreshed only the two the current step needed. Entries 2 and 3 held
the previous step's values -- ``[0, 2, 2, 2]``, the padded form of a two-token step,
whose tail is ``num_tokens`` by the runner's own convention -- and
``searchsorted`` on that array moves tokens 2 and 3 to row 2, columns 0 and 1. Cell for
cell, it is the captured workspace.

Two rules follow, and each has tests here. The *producer*: refresh the whole static
buffer every step, so no width the graph can read holds a stale boundary. The
*arithmetic*: a stale tail is not a cosmetic padding artifact -- it changes which row a
token's ngram context is computed from, which changes the ids, which changes the
embeddings the model is handed.

The scatter tests call the production ``_compute_ngram_ids`` rather than restating its
searchsorted/clamp/scatter. A first attempt at this analysis reproduced the boot with a
3-entry ``[0, 2, 2]`` and was wrong in exactly that way: it omitted
``request_indices.clamp_(max=num_reqs - 1)``, and the clamp is what makes the answer a
4-entry ``[0, 2, 2, 2]``. Duplicating the arithmetic in a test is how a test ends up
proving a mechanism the code does not have.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.amd import ple_layer as ple
from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

EOS = 248044  # the model's EOS, which is what the workspace is filled with
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 4
CTX = NGRAM_SIZE - 1
ROWS = 3  # the capture width this box replays at: 3 requests, 4 qsl entries
COLS = 64
BOOT13_TOKENS = [47983, 1892, 279, 10146]

# The workspace boot #13 printed after the bad fill, truncated to the cells that
# matter. Row 2 holding tokens 2 and 3 is the whole signature.
BOOT13_PACKED = {
    0: [47983, 1892, EOS, EOS],
    1: [EOS, EOS, EOS, EOS],
    2: [279, 10146, EOS, EOS],
}


def _state(width: int = ROWS + 1) -> Qwen4ExpModelState:
    """A state with only the buffer the producer touches.

    ``__new__`` rather than ``__init__``: the constructor wants a whole engine
    context, and the method under test needs exactly one attribute. The same trick the
    hostids suite uses for the layer.
    """
    state = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    state.ple_query_start_loc = torch.zeros(width, dtype=torch.int32)
    return state


def _batch(qsl: list[int], num_tokens: int, padded: int) -> SimpleNamespace:
    return SimpleNamespace(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        num_tokens=num_tokens,
        num_reqs_after_padding=padded,
    )


def _hash_args() -> dict:
    heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
    return dict(
        multipliers=torch.tensor([3, 5, 7], dtype=torch.long)[:NGRAM_SIZE],
        vocab_sizes=torch.full((heads,), 997, dtype=torch.long),
        head_offsets=torch.zeros(heads, dtype=torch.long),
        eos_token_id=EOS,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
    )


def _fill(qsl: list[int], n_tokens: int) -> torch.Tensor:
    """Run the production fill and return the workspace it wrote, in place.

    ``n_tokens`` is a parameter, not read off ``qsl[-1]``, because on the bad step
    ``qsl[-1]`` is precisely the stale value under test: the step had four tokens and
    a tail that said two.
    """
    packed = torch.full((ROWS, COLS), EOS, dtype=torch.long)
    ple.Qwen4ExpNGramEmbedding._compute_ngram_ids(
        torch.tensor(BOOT13_TOKENS[:n_tokens], dtype=torch.long),
        torch.tensor(qsl, dtype=torch.int64),
        torch.full((ROWS, CTX), EOS, dtype=torch.long),
        packed=packed,
        positions=torch.arange(n_tokens, dtype=torch.int64),
        **_hash_args(),
    )
    return packed


# ---------------------------------------------------------------------------
# the producer: the whole static buffer, every step
# ---------------------------------------------------------------------------


def test_a_shrinking_padded_width_leaves_no_boundary_from_the_previous_step() -> None:
    """The boot #13 sequence, in order, with the real producer.

    A three-request two-token step, then a one-request four-token step. Before the fix
    the second step refreshed two entries and left the other two holding ``2``; the
    graph, captured at three requests, read all four.
    """
    state = _state()

    state._refresh_ple_query_start_loc(_batch([0, 2, 2, 2], num_tokens=2, padded=3))
    view = state._refresh_ple_query_start_loc(_batch([0, 4], num_tokens=4, padded=1))

    assert state.ple_query_start_loc.tolist() == [0, 4, 4, 4], (
        "the padded tail still holds the previous step's boundary: a graph captured at "
        "a wider request count will read it, and the ngram fill will scatter tokens "
        "into rows that belong to no request in this step"
    )
    assert view.tolist() == [0, 4]
    assert view.data_ptr() == state.ple_query_start_loc.data_ptr(), (
        "the view handed to forward must be a view of the refreshed buffer, not a copy"
    )


def test_a_full_source_is_copied_whole_in_one_op() -> None:
    """The normal path: the runner hands over its full tail-padded buffer."""
    state = _state()
    src = [0, 2, 4, 6]

    view = state._refresh_ple_query_start_loc(_batch(src, num_tokens=6, padded=3))

    assert state.ple_query_start_loc.tolist() == src
    assert view.tolist() == src


def test_a_short_source_gets_the_tails_own_num_tokens() -> None:
    """Adaptive verification hands back a ``[num_reqs + 1]`` view; the rest is filled.

    Filling with ``num_tokens`` is the runner's own convention
    (``input_batch.py:159``), so the array stays non-decreasing and every padded
    request maps to the last token's position rather than to a stale row.
    """
    state = _state()
    state.ple_query_start_loc.fill_(9)  # hostile prior contents

    state._refresh_ple_query_start_loc(_batch([0, 4], num_tokens=4, padded=1))

    assert state.ple_query_start_loc.tolist() == [0, 4, 4, 4]


def test_the_full_source_path_does_not_fill() -> None:
    """The short-source ``fill_`` is a branch, not a second per-step kernel."""
    fills = {"n": 0}
    real = torch.Tensor.fill_

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        fills["n"] += 1
        return real(self, *args, **kwargs)

    torch.Tensor.fill_ = spy  # type: ignore[assignment]
    try:
        _state()._refresh_ple_query_start_loc(_batch([0, 2, 4, 6], 6, 3))
    finally:
        torch.Tensor.fill_ = real  # type: ignore[assignment]

    assert fills["n"] == 0, (
        "the full-buffer path issued a fill_: this runs once per decode step and the "
        "fix is supposed to cost the same single copy the prefix refresh cost"
    )


def test_the_handed_view_tracks_num_reqs_after_padding() -> None:
    state = _state()

    narrow = state._refresh_ple_query_start_loc(_batch([0, 4, 8, 12], 12, 1))
    wide = state._refresh_ple_query_start_loc(_batch([0, 4, 8, 12], 12, 3))

    assert narrow.numel() == 2 and wide.numel() == 4
    assert wide.tolist() == [0, 4, 8, 12]


# ---------------------------------------------------------------------------
# the arithmetic: what a stale tail actually does
# ---------------------------------------------------------------------------


def test_the_stale_tail_reproduces_the_boot_13_workspace() -> None:
    """Production arithmetic, given the value the fix now makes unreachable.

    This is the witness from ``logs/vllm.log.20261002-211450``: 4 tokens, one request,
    and the fill working from ``[0, 2, 2, 2]``. It is here so the claim stays checkable
    and so a future reader can see that a padded tail is not cosmetic.
    """
    packed = _fill([0, 2, 2, 2], n_tokens=4)

    for row, cells in BOOT13_PACKED.items():
        assert packed[row, :4].tolist() == cells, f"row {row}"


def test_the_refreshed_array_puts_every_token_in_its_own_row_at_capture_width() -> None:
    """The same step after the fix, read at the width the graph reads.

    Producer and consumer in one test on purpose: the buffer is refreshed by the code
    that ships, and then read through the production fill at the capture width, which
    is wider than the view forward was handed.
    """
    state = _state()
    state._refresh_ple_query_start_loc(_batch([0, 2, 2, 2], num_tokens=2, padded=3))
    state._refresh_ple_query_start_loc(_batch([0, 4], num_tokens=4, padded=1))

    packed = _fill(state.ple_query_start_loc.tolist(), n_tokens=4)

    assert packed[0, :4].tolist() == BOOT13_TOKENS, (
        "the fill did not put all four tokens of the single request in row 0"
    )
    assert packed[1, :4].tolist() == [EOS] * 4
    assert packed[2, :4].tolist() == [EOS] * 4


def test_extent_matches_the_first_query_length_after_the_refresh() -> None:
    """The invariant the P1.14 report measures, asserted off the scratch.

    ``extent_row0 == qsl[1]`` held on the good boot #13 step and failed on the bad one
    (2 vs 4), which is how the mismatch was attributed. It is the cheapest thing to
    watch on the next differential boot, so it earns a test.
    """
    state = _state()
    state._refresh_ple_query_start_loc(_batch([0, 2, 2, 2], num_tokens=2, padded=3))
    state._refresh_ple_query_start_loc(_batch([0, 4], num_tokens=4, padded=1))
    packed = _fill(state.ple_query_start_loc.tolist(), n_tokens=4)

    row0 = packed[0].tolist()
    extent = row0.index(EOS) if EOS in row0 else len(row0)
    qsl = state.ple_query_start_loc.tolist()

    assert extent == qsl[1], f"extent={extent} qsl={qsl}"


def test_only_a_four_entry_array_starting_zero_two_two_reproduces_the_witness() -> None:
    """How narrow the diagnosis is, enumerated rather than asserted.

    Searches every non-decreasing four-entry boundary array the fill could have held
    with values in the range the two steps involved, and keeps the ones that reproduce
    boot #13's workspace through the production fill. Every winner starts ``[0, 2, 2]``;
    entry 3 is unconstrained by this observation because
    ``request_indices.clamp_(max=num_reqs - 1)`` folds it away, and the runner would
    have written ``2`` there.

    A narrower array is not a candidate at all: the fill slices ``ngram_context`` to
    ``num_reqs`` and concatenates it with the workspace, so a three-entry boundary array
    against a three-row workspace raises rather than mis-scattering. That is worth
    recording, because the first analysis of this boot blamed a three-entry
    ``[0, 2, 2]`` -- a value the compute could not have consumed at all.

    Kept in the suite because it is the evidence the diagnosis rests on, and because it
    says a tail filled with anything other than *this step's* ``num_tokens`` would still
    be wrong: ``[0, 4, 4, 4]`` puts all four tokens in row 0, which is the answer the
    host path gives.
    """

    def matches(qsl: tuple[int, ...]) -> bool:
        packed = _fill(list(qsl), 4)
        return [packed[r, :4].tolist() for r in range(ROWS)] == [
            BOOT13_PACKED[r] for r in range(ROWS)
        ]

    winners = [
        q
        for q in itertools.product(range(9), repeat=4)
        if q[0] == 0 and list(q) == sorted(q) and matches(q)
    ]

    assert winners, "nothing reproduces the witness -- the fixture is wrong"
    assert all(q[:3] == (0, 2, 2) for q in winners), winners
    assert (0, 2, 2, 2) in winners, "the runner's own padded shape must reproduce it"
    assert not matches((0, 4, 4, 4)), "the fixed shape must NOT reproduce the defect"

    with pytest.raises(RuntimeError):
        _fill([0, 2, 2], 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
