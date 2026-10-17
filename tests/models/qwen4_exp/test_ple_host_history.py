# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for :class:`HostTokenHistory`, the host mirror of committed tokens.

This class exists because ``req_states.all_token_ids`` is host-*visible* but not
host-*valid*: its prompt cells are staged by the host at add time, while its
generated cells are written by ``_post_update_kernel`` on the GPU, unordered
before a host read in ``prepare_inputs``. The first differential boot caught that
on 2026-10-01. So the tests here are biased toward the two properties that make
the mirror trustworthy rather than merely addressable:

* it advances by exactly the tokens the device appended (``num_sampled``, the
  device's own trim rule), and
* it never claims a token whose copy has not landed -- ``drain`` is non-blocking
  and stops at the first un-landed commit, so ordering survives.
"""

import numpy as np
import pytest

from vllm.models.qwen4_exp.amd.ple_hostids import HostTokenHistory

W = 3  # width = ngram_context_len + 1 = 2 + 1
SLOTS = 4


class FakeEvent:
    """The two methods ``HostTokenHistory`` calls on a copy event.

    ``blocks`` models a copy that will not land, which is how the flush path gets
    tested: a wait that returns without completing must still not apply the row.
    """

    def __init__(self, done: bool = False, blocks: bool = False) -> None:
        self.done = done
        self.blocks = blocks
        self.queries = 0
        self.syncs = 0

    def query(self) -> bool:
        self.queries += 1
        return self.done

    def synchronize(self) -> None:
        self.syncs += 1
        if not self.blocks:
            self.done = True

    def complete(self) -> None:
        self.done = True


def _stage(history, slots, rows, counts, done=None, pad=0):
    """Stage one step. ``pad`` adds trailing sampler rows that belong to no
    request in the batch, which is what a padded graph descriptor produces."""
    width = max(len(r) for r in rows)
    matrix = np.zeros((len(rows) + pad, width), dtype=np.int32)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row
    return history.stage_step(
        np.array(slots, np.int32),
        matrix,
        np.array(list(counts) + [0] * pad, np.int32),
        done,
    )


def test_construction_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        HostTokenHistory(0, W)
    with pytest.raises(ValueError):
        HostTokenHistory(SLOTS, 0)


def test_seed_keeps_the_chronological_tail():
    history = HostTokenHistory(SLOTS, W)
    history.seed(1, [10, 11, 12, 13, 14])
    assert history.length[1] == 5
    assert history.window[1].tolist() == [12, 13, 14]
    # Newest at the right: that cell is the token the next step feeds first.
    assert history.window[1, -1] == 14


def test_seed_of_a_short_request_zeroes_the_unused_left():
    history = HostTokenHistory(SLOTS, W)
    history.seed(2, [42])
    assert history.length[2] == 1
    assert history.window[2].tolist() == [0, 0, 42]
    history.seed(3, [])
    assert history.length[3] == 0


def test_seed_of_a_resumed_request_includes_generated_tokens():
    """Preemption/resume: the scheduler sends the whole list again.

    Re-seeding is what makes slot reuse safe -- a window left over from a
    previous request cannot survive an add.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1, 2, 3])
    _stage(history, [0], [[9]], [1])
    history.drain()
    assert history.window[0].tolist() == [2, 3, 9]
    history.seed(0, [1, 2, 3, 9, 77])  # resumed with the generated token
    assert history.window[0].tolist() == [3, 9, 77]
    assert history.length[0] == 5


def test_commit_appends_in_order_for_every_batch_size():
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1, 2])
    history.commit(0, [3])
    assert history.window[0].tolist() == [1, 2, 3] and history.length[0] == 3
    history.commit(0, [4, 5])  # accepted bonus + one draft
    assert history.window[0].tolist() == [3, 4, 5] and history.length[0] == 5
    history.commit(0, [6, 7, 8, 9])  # more than the window holds
    assert history.window[0].tolist() == [7, 8, 9] and history.length[0] == 9
    history.commit(0, [])  # a step that committed nothing is a no-op
    assert history.window[0].tolist() == [7, 8, 9] and history.length[0] == 9


def test_stage_never_reads_the_sampled_array():
    """Staging must be safe while the D2H is still in flight.

    Copying at stage time is the bug this class exists to avoid, so the staged
    values are held by reference and only read once the event reports done.
    """

    class Exploding:
        ndim = 2
        shape = (1, 4)

        def __array__(self, dtype=None):
            raise AssertionError("HostTokenHistory read the sampled array")

    history = HostTokenHistory(SLOTS, W)
    assert _stage_history_untouched(history, Exploding()) is True


def _stage_history_untouched(history, sampled) -> bool:
    return history.stage_step(
        np.array([0], np.int32),
        sampled,
        np.array([1], np.int32),
        FakeEvent(False),
    )


def test_drain_waits_for_the_copy_and_keeps_commits_in_order():
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1])
    first, second = FakeEvent(False), FakeEvent(True)
    assert _stage(history, [0], [[2, 3]], [2], first)
    assert _stage(history, [0], [[4]], [1], second)

    # The second commit's data is host-visible, but applying it before the first
    # would put token 4 before tokens 2 and 3.
    assert history.drain() == 0
    assert history.pending == 2
    assert history.window[0].tolist() == [0, 0, 1]

    first.complete()
    assert history.drain() == 2  # both, in order
    assert history.pending == 0
    assert history.window[0].tolist() == [2, 3, 4]
    assert history.length[0] == 4


def test_drain_only_polls_never_blocks():
    history = HostTokenHistory(SLOTS, W)
    event = FakeEvent(False)
    _stage(history, [0], [[5]], [1], event)
    history.drain()
    assert event.queries == 1  # query(), never synchronize()


def test_num_sampled_is_the_device_trim_rule():
    """Rejected drafts sit in the sampled row and must not be appended.

    ``_post_update_kernel`` takes the first ``num_sampled[i]`` entries of row i;
    the mirror uses the same counts, so agreement with the device is structural.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1, 2])
    matrix = np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.int32)
    assert history.stage_step(
        np.array([0, 1], np.int32),
        matrix,
        np.array([1, 3], np.int32),
        None,
    )
    assert history.drain() == 2
    assert history.window[0].tolist() == [1, 2, 7]  # one accepted, drafts rejected
    assert history.window[1].tolist() == [11, 12, 13]
    assert history.length[0] == 3
    assert history.length[1] == 3


def test_invalidate_drops_an_in_flight_commit_for_that_slot():
    """A request removed while its copy is in flight must not poison its slot.

    The event is the point of the test: the copy lands *after* the slot has been
    handed to a new request, so the drop has to happen at ``invalidate`` time. A
    queue that still carried the dead request's row would append its token to the
    new request, and the builder's exact-length check would then refuse that slot
    for the rest of its life.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(2, [1, 2, 3])
    event = FakeEvent(False)
    _stage(history, [2], [[99]], [1], event)
    history.invalidate(2)
    # The slot is handed to a new request while the old copy is still in flight...
    history.seed(2, [7, 8])
    assert history.drain() == 0
    # ...and only afterwards does that copy land.
    event.done = True
    assert history.drain() == 0, "the dead request's row was applied"
    assert history.window[2].tolist() == [0, 7, 8]
    assert history.length[2] == 2


def test_queue_is_bounded_and_flushes_rather_than_loses_a_commit():
    """A full queue waits for the head commit; it does not silently lose the new one.

    This is the behaviour boot #26 paid for. The bound used to drop the new
    commit, and because nothing here can recover a lost commit, the affected
    slot's length stayed short and the builder refused that request on every
    remaining step -- six consecutive windows of 32/32 refusals, with the drops
    that caused them all behind them. Waiting once is the cheaper failure, so the
    queue now flushes and the commit lands.
    """
    history = HostTokenHistory(SLOTS, W)
    events = [FakeEvent(False) for _ in range(HostTokenHistory.MAX_PENDING)]
    for event in events:
        assert _stage(history, [0], [[1]], [1], event)
    assert history.pending == HostTokenHistory.MAX_PENDING

    # The head's copy has not landed, but waiting for it is allowed to succeed.
    assert _stage(history, [0], [[2]], [1], FakeEvent(False)) is True
    assert events[0].syncs == 1, "the flush did not wait for the head commit"
    assert events[1].syncs == 0, "the flush waited for more than the head"
    assert history.flushes == 1
    assert history.dropped == 0
    # Flushing applied the head, so the queue made room instead of losing work.
    assert history.length[0] == 1


def test_a_flush_that_cannot_finish_still_drops_and_says_so():
    """Waiting is a repair, not a licence to read an un-landed copy."""
    history = HostTokenHistory(SLOTS, W)
    stuck = FakeEvent(False, blocks=True)
    for _ in range(HostTokenHistory.MAX_PENDING):
        assert _stage(history, [0], [[1]], [1], stuck)
    assert _stage(history, [0], [[2]], [1], FakeEvent(False)) is False
    assert stuck.syncs == 1
    assert stuck.queries >= 2, "the flush applied a commit without re-asking"
    assert history.dropped == 1
    assert history.length[0] == 0, "an in-flight commit was applied"
    assert "pending full" in history.last_drop


def test_padded_sampler_rows_are_trimmed_to_the_batch():
    """The sampler arrays are copied out whole; the batch is the real width.

    ``AsyncOutput`` copies ``sampler_output.sampled_token_ids`` with no slice, so
    a step replayed from a padded descriptor hands over more rows than
    ``idx_mapping_np[:num_reqs]``. Those extra rows belong to no request in this
    batch, and the first ``len(slots)`` are the batch's, in the same order. The
    old code called that a shape violation and dropped the commit -- which is the
    one failure mode that used to be permanent.
    """
    history = HostTokenHistory(SLOTS, W)
    assert _stage(history, [3], [[11, 12]], [2], FakeEvent(True), pad=2) is True
    assert history.sliced == 1
    assert history.dropped == 0
    assert history.drain() == 1
    assert history.length[3] == 2
    assert history.window[3].tolist() == [0, 11, 12]


def test_fewer_sampler_rows_than_slots_is_still_a_drop():
    """Trimming answers extra rows. Missing rows have no owner to guess."""
    history = HostTokenHistory(SLOTS, W)
    assert _stage(history, [0, 1], [[1]], [1]) is False
    assert history.dropped == 1
    assert "rows" in history.last_drop


def test_drop_detail_names_the_check_that_fired():
    """A count says a commit was lost; the detail says which check lost it."""
    history = HostTokenHistory(SLOTS, W)
    assert _stage(history, [9], [[1]], [1]) is False
    assert "slot" in history.last_drop
    # Out-of-range counts are a DRAIN-time check now (task #56). At stage time the
    # array still holds whatever the allocator recycled, so judging it there
    # rejected healthy commits -- the ``num_sampled [0, 4, 8] outside 0..4``
    # drops. The detail says where the check ran.
    assert _stage(history, [0], [[1, 2, 3]], [5]) is True
    assert history.drain() == 0
    assert "num_sampled" in history.last_drop
    assert "landed" in history.last_drop
    assert history.dropped == 2
    assert _stage(history, [0], [[1]], [1]) is True
    assert history.last_drop == "", "a success left a stale explanation"


def test_stage_rejects_inconsistent_shapes():
    history = HostTokenHistory(SLOTS, W)
    cases = [
        # slots, rows, counts, why
        ([9], [[1]], [1], "slot outside the table"),
        ([-1], [[1]], [1], "padding slot"),
        ([0, 1], [[1]], [1], "counts disagree with slots"),
    ]
    for slots, rows, counts, _why in cases:
        before = history.dropped
        assert _stage(history, slots, rows, counts) is False, _why
        assert history.dropped == before + 1, _why

    # A sampled array that is not [requests, tokens] cannot be trimmed at all.
    assert (
        history.stage_step(
            np.array([0], np.int32),
            np.zeros((0, 2), np.int32),
            np.array([1], np.int32),
            None,
        )
        is False
    )
    assert history.dropped == len(cases) + 1
    assert history.staged == 0

    # The two out-of-range-count cases used to be in the list above. They are
    # STAGED now -- nothing may judge those values before the copy lands -- and
    # rejected when drain reads them. The mirror image, counts that only LOOK
    # wrong before landing, is test_num_sampled_is_read_after_the_copy_lands.
    for bad, why in ((5, "more accepted than the row holds"), (-1, "negative count")):
        assert _stage(history, [0], [[1, 2, 3]], [bad]) is True, why
    assert history.staged == 2
    assert history.drain() == 0, "an out-of-range row must commit nothing"
    assert history.dropped == len(cases) + 3


def test_num_sampled_is_read_after_the_copy_lands():
    """THE task #56 regression: the counts are the second output of the same D2H.

    ``AsyncOutput`` (``async_utils.py``) copies ``sampled_token_ids`` AND
    ``num_sampled_tokens`` to fresh host arrays on the copy stream, then records
    one event. Both are garbage until that event fires. ``stage_step`` used to
    ``num_sampled.copy()`` anyway, on the docstring's mistaken belief that the
    counts came from a persistent buffer like ``idx_mapping_np`` does.

    The field signature this reproduces, verbatim from two boots:

      * ``vllm.log.20261008-083018`` -- 14,859 drops (9.9 % of commits),
        ``num_sampled [0, 4, 8] outside 0..4``
      * ``vllm.log.20261009-161850`` -- the same four lines at startup, one per
        worker

    ``[0, 4, 8]`` is not a count vector for a 1+3-draft batch (the cap is 4). It
    is what the allocator had left in that host block -- offsets into the
    previous step's flattened copy -- read microseconds after the D2H was
    enqueued. Against the pre-#56 code this fails twice over: the stage is
    rejected on recycled junk, and even an accepted commit drains the PREVIOUS
    step's counts.
    """
    history = HostTokenHistory(SLOTS, W)
    for slot, token in ((0, 1), (1, 2), (2, 3)):
        history.seed(slot, [token])

    # 3 requests, 1 + 3 draft slots: exactly the shape that produced
    # "outside 0..4" in the field.
    counts = np.array([0, 4, 8], np.int32)
    sampled = np.zeros((3, 4), np.int32)
    sampled[0, 0], sampled[1, 0], sampled[1, 1], sampled[2, 0] = 7, 8, 9, 10
    event = FakeEvent(False)  # D2H enqueued, not landed

    assert (
        history.stage_step(np.array([0, 1, 2], np.int32), sampled, counts, event)
        is True
    ), "staging must not judge counts it cannot legitimately read yet"
    assert history.dropped == 0, "the pre-landing guard rejected a healthy commit"
    assert history.drain() == 0, "nothing may be applied before the event lands"

    counts[:] = [1, 2, 1]  # the copy lands with this step's real counts
    event.complete()

    assert history.drain() == 3
    assert history.window[0].tolist() == [0, 1, 7]
    assert history.window[1].tolist() == [2, 8, 9]
    assert history.window[2].tolist() == [0, 3, 10]
    assert history.dropped == 0, "landed counts of 1..4 are not an anomaly"


def test_invalidate_voids_an_unlanded_commit_without_writing_to_the_source():
    """A slot freed while its copy is in flight must not come back to life.

    The tempting one-line version of task #56 -- hold ``num_sampled`` by
    reference and let ``invalidate`` keep zeroing it -- is wrong twice: the
    zero goes into ``AsyncOutput``'s own array, which output processing reads
    too, and the in-flight D2H overwrites the zero when it lands, resurrecting
    the exact commit invalidation exists to cancel. ``slot`` reuse is what makes
    that a wrong-answer bug rather than a stale one: the tokens land in the
    window of whoever gets the slot next.

    Red against that tempting fix, green against ``_PendingCommit.void``.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1])
    counts = np.array([0], np.int32)
    sampled = np.array([[5, 0, 0]], np.int32)
    event = FakeEvent(False)
    assert history.stage_step(np.array([0], np.int32), sampled, counts, event)

    history.invalidate(0)  # request removed; the slot is free to be reused
    counts[:] = [1]  # the D2H lands -- this is what the copy actually does
    event.complete()

    assert history.drain() == 0, "an invalidated commit was applied"
    assert history.length[0] == 0
    assert history.window[0].tolist() == [0, 0, 0]
    assert counts.tolist() == [1], (
        "invalidate wrote through to the AsyncOutput array output processing "
        "also reads"
    )


def test_padded_rows_with_out_of_range_counts_still_stage():
    """THE task #57 gap: what the row-slice guarantees, pinned directly.

    ``_stage(..., pad=N)`` fills padded rows with count 0, which is in range, so
    task #55's ``trim-removed`` red control stayed GREEN -- the suite could not
    see the removal of the guard the slice exists to make unreachable. A real
    padded graph descriptor does not carry zeros in its unused rows; it carries
    whatever the sampler left there, which is the out-of-range junk the count
    check rejects.

    The assertion is about the QUEUE, not about the outcome, and that is the
    finding worth writing down. Removing the slice does not change a single value
    this history commits, because ``drain`` iterates ``entry.slots.size`` and the
    pad rows are therefore never read: after task #56 moved the range check into
    ``drain``, the slice stopped being the thing that saves the commit and became
    defense in depth for the row-i-belongs-to-slot-i invariant. A behavioural test
    cannot pin it any more -- so this pins the shape the queue is handed, which is
    the claim the slice actually makes, and the red control
    ``trim-removed-T57-closure`` goes RED on it.
    """
    history = HostTokenHistory(SLOTS, W)
    history.seed(0, [1])
    counts = np.array([1, 9, 7], np.int32)  # rows 1 and 2 belong to no request
    sampled = np.array([[6, 0, 0], [0, 0, 0], [0, 0, 0]], np.int32)

    assert (
        history.stage_step(
            np.array([0], np.int32), sampled, counts, FakeEvent(True)
        )
        is True
    ), "a padded row's garbage must not veto the real batch's commit"
    assert history.sliced == 1
    # The trim itself: one row per slot, and the counts that go with those rows.
    # Without the slice these are 3 and [1, 9, 7], and the next assertion is the
    # only thing standing between a padded descriptor and a future consumer that
    # reads all of them.
    assert history._pending[0].sampled.shape[0] == 1, "the queue kept the pad rows"
    assert history._pending[0].counts().tolist() == [1], "the queue kept pad counts"
    assert history.dropped == 0
    assert history.drain() == 1
    assert history.window[0].tolist() == [0, 1, 6]
    assert history.length[0] == 2


# --------------------------------------------------------------------------------
# Task #69: the drain-time anomaly instrument.
#
# The instrument exists because five explanations for the host mirror's bad newest
# cell were eliminated statically and the sixth cannot be seen in source: the values
# logged at drain are recycled allocator bytes, and `async_copy_to_np` allocates a
# FRESH host block per step. Telling "the block handed over is not the block the copy
# wrote" from "the device wrote that block" needs a live boot, so the instrument has
# to be inert, must never raise, and has to be tested for exactly those two
# properties before it is ever allowed near a serving engine.
# --------------------------------------------------------------------------------

# Boot #14's reported value where a token id belongs. Reinterpreted as two float32s
# this is (0.0037265, 0.000364706) -- plausible floats, which is the whole clue.
GARBAGE = 4161103993993705554


class _Entry:
    """Just enough of ``_PendingCommit`` for the instrument's own unit tests."""

    def __init__(self, sampled) -> None:
        self.sampled = sampled


def _stage_i64(history, slots, rows, counts, done=None):
    """``_stage`` with an int64 matrix.

    The production ``sampled`` arrives via ``async_copy_to_np(...).numpy()`` from the
    sampler's int64 output, and the recycled-block values are 64-bit, so a test that
    wants to reproduce one cannot use the int32 matrix ``_stage`` builds.
    """
    width = max(len(r) for r in rows)
    matrix = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row
    return history.stage_step(
        np.array(slots, np.int32),
        matrix,
        np.array(counts, np.int32),
        done,
    )


def test_instrument_is_inert_when_every_value_is_a_token_id():
    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    assert _stage(history, [0], [[7, 9, 11]], [3], FakeEvent(done=True))
    assert history.drain() == 1
    assert history.bad_value_events == 0
    assert history.last_bad_value == ""


def test_out_of_range_value_is_observed_but_still_committed():
    # OBSERVATION ONLY is the load-bearing property, so it is asserted on the window
    # rather than on a flag: if the instrument changed what gets committed, the
    # garbage would be missing from the mirror and the instrument would itself be a
    # correctness bug on the serving path.
    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    assert _stage_i64(history, [0], [[GARBAGE, 5]], [2], FakeEvent(done=True))
    assert history.drain() == 1
    assert history.bad_value_events == 1
    assert history.window[0, W - 1] == 5  # newest in the tail cell, as always
    assert history.window[0, W - 2] == GARBAGE  # committed, not skipped
    assert history.dropped == 0


def test_report_carries_the_fields_that_discriminate_the_remaining_theories():
    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    _stage_i64(history, [0], [[9, GARBAGE]], [2], FakeEvent(done=True))
    history.drain()
    detail = history.last_bad_value
    # Each of these separates two explanations; see the method's docstring.
    for field in (
        "slot=0",
        "row=0",
        "bound=1000(vocab_size)",
        "bad=[",
        "bad_as_float32=[",
        "row[:8]=[",
        "addr=0x",
        "window_tail=",
    ):
        assert field in detail, f"{field!r} missing from {detail!r}"
    # And the address has to be a REAL one. Asserting only that the field exists
    # would be satisfied by a stub that always printed 0x0, which is exactly what a
    # recycled-block investigation cannot use: the address is the field that
    # separates "wrong block" from "device wrote it", so a constant is not a
    # degraded report, it is no report.
    assert "addr=0x0 " not in detail, "numpy source must report its real address"
    assert int(detail.split("addr=0x")[1].split()[0], 16) > 0


def test_float_pair_is_the_actual_reinterpretation_not_a_plausible_one():
    # Computed independently here. A printed bit pattern that is subtly wrong is
    # worse than no printed bit pattern: it is the evidence someone quotes at each
    # other, and this project has already been bitten by a number written into a
    # document without being computed.
    expected = float(np.asarray([GARBAGE], dtype=np.int64).view(np.float32)[0])
    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    _stage_i64(history, [0], [[GARBAGE]], [1], FakeEvent(done=True))
    history.drain()
    assert repr(expected) in history.last_bad_value
    assert 1e-4 < expected < 1e-2  # not denormal, not huge: a plausible logit


def test_a_missing_vocab_bound_falls_back_wide_and_says_so():
    # The fallback catches the recycled-float class (bit patterns re-read as int64
    # are astronomically large) and CANNOT catch a value that is merely another
    # request's id. Both halves are asserted, because the limitation is the part an
    # operator would otherwise assume away.
    history = HostTokenHistory(SLOTS, W)
    assert history.vocab_bound == (1 << 31) - 1
    assert history.vocab_bound_source == "fallback-2p31"
    _stage_i64(history, [0], [[GARBAGE]], [1], FakeEvent(done=True))
    history.drain()
    assert history.bad_value_events == 1
    assert "fallback-2p31" in history.last_bad_value

    other = HostTokenHistory(SLOTS, W)
    _stage_i64(other, [0], [[231807]], [1], FakeEvent(done=True))
    other.drain()
    assert other.bad_value_events == 0, "wide bound must not pretend to catch this"


def test_a_negative_value_is_out_of_range_by_definition():
    # ``PLACEHOLDER_TOKEN_ID = -1`` is the sampler's padding sentinel, and zero -1
    # values appeared in the 148 logged events -- so if this ever fires on a real
    # boot, it is an overrun, not padding.
    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    _stage_i64(history, [0], [[-1]], [1], FakeEvent(done=True))
    history.drain()
    assert history.bad_value_events == 1


def test_report_is_cadenced_not_per_event():
    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    for _ in range(8):
        _stage_i64(history, [0], [[GARBAGE]], [1], FakeEvent(done=True))
        history.drain()
    assert history.bad_value_events == 8
    # The counter keeps counting; only the ~200-char detail is cadenced (1..4, then
    # every 64th). A boot where this fires every step is the failure mode, and 19
    # identical lines a second is not evidence of it.
    assert history.last_bad_value.startswith("#4 ")


def test_instrument_reports_a_source_with_no_ctypes_at_all():
    # ``entry.sampled`` is a numpy view today, but an instrument on the serving path
    # may not assume that forever: a torch tensor has no ``.ctypes``. The address is
    # reported as 0 rather than raising.
    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    history._observe_committed(
        np.array([GARBAGE], np.int64), 0, 0, _Entry([[GARBAGE, 1, 2, 3]])
    )
    assert "addr=0x0 " in history.last_bad_value
    assert history.bad_value_events == 1


def test_instrument_cannot_cost_a_boot():
    # The instrument must never raise, whatever the row is. This is not defensive
    # programming for its own sake: a module with no logger and no exceptions is
    # this file's existing contract, and breaking it turns a diagnostic into an
    # outage on a box that serves traffic.
    class Exploding:
        def __array__(self, dtype=None):  # noqa: ANN001
            raise RuntimeError("boom")

    history = HostTokenHistory(SLOTS, W, vocab_bound=1000)
    history._observe_committed(Exploding(), 0, 0, _Entry([[1, 2]]))
    assert history.bad_value_events == 0
    assert "observation failed" in history.last_bad_value
