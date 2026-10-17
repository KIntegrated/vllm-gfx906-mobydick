# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side ngram id computation for the PLE decode path.

Why this exists
---------------
The ngram table lives in host RAM (``MmapShardedNGramEmbedding``), so the
gather has to happen on the host. But the *ids* it gathers with are produced on
the GPU by ``Qwen4ExpNGramEmbedding._compute_ngram_ids``, which forces a
blocking device->host copy every step:

    pinned_ids.copy_(ngram_ids, non_blocking=False)   # ple_layer.py

``non_blocking=False`` is not a tuning choice. The host reads that buffer
immediately afterwards, so the copy has to have landed -- which means the host
waits for everything queued on the compute stream ahead of it. On this box that
wait (``drain`` in ``ple_prof.py``) is the term that the profiler measures, and
D2' proved the gather itself is no longer the cost: after the sorted-gather
rewrite the residual lookup is page-fault-bound, and ``drain`` is host *idle*
while the GPU catches up.

Removing the round trip means computing the ids where the gather already is.
The host can do that, but not as cheaply as it looks: this runner keeps **no host
array of the step's input ids** (there is no ``input_ids_np`` / ``token_ids_cpu``
here -- ``input_batch.input_ids`` is device-only, and decode rows are placed on
the device by ``combine_sampled_and_draft_tokens``). The host has to *rebuild*
them per request, from three pieces it does have:

* the tokens accepted last step, as a host list-of-lists in ``ModelRunnerOutput``
  (``sampled_token_ids``), which is the row's first token;
* this step's draft tokens, mirrored to pinned host memory by
  ``DraftTokensHandler`` when ``VLLM_GFX906_PLE_HOSTIDS`` is set -- without that
  mirror the host knows 1 of the 4 rows a speculative decode step feeds this
  layer, because ``spec_decode/utils.py`` deliberately never copies drafts to
  the host (``docs/d1-host-ids-design.md`` has the trace);
* the per-request token history in :class:`HostTokenHistory`, a rolling window
  fed from the host copy of ``sampled_token_ids`` that ``AsyncOutput`` already
  makes. It is *not* ``req_states.all_token_ids._uva_buf.np``: that is genuinely
  pinned host memory the accelerator view aliases, but its prompt region is
  written host-side at add time while its generated region is written by a GPU
  kernel (``input_batch.py:577``) unordered before a host read in
  ``prepare_inputs``, so mid-decode the newest cells -- the only ones this path
  needs -- are still zeros. The first differential boot caught exactly that on
  2026-10-01; see ``HostTokenHistory`` for the mechanism and the fix.

Bit-identity is the whole contract
----------------------------------
This module is a *reimplementation* of ``_compute_ngram_ids`` in numpy, which
makes it the one kind of optimization that can be silently wrong: an id that is
off by one is a different row of a 20-million-row table, so the model receives a
plausible embedding of the wrong ngram and nothing raises. Two things hold it in
place.

1. It reproduces the device function's semantics exactly, including the parts
   that are easy to guess wrong:

   * **int64 wraparound.** ``shifted * multiplier`` overflows int64 for ordinary
     token ids. Torch wraps two's complement; numpy wraps too but warns, so the
     arithmetic runs under ``np.errstate(over="ignore")``. ``bitwise_xor`` on
     signed int64 and ``torch.bitwise_xor`` agree because both act on the
     two's-complement bits.
   * **floor remainder.** ``torch.remainder`` signs its result like Python's
     ``%`` (non-negative for a positive divisor, even for a negative dividend).
     ``np.remainder`` is the same floor definition; ``np.mod`` would agree too
     but ``%`` on a Python int would not, so nothing here uses Python ints.
   * **eos segment boundaries.** ``_shift_precompute``/``_shift_apply`` make a
     token blind to anything at or before the nearest preceding eos: a shift is
     only honoured when ``position_in_segment >= shift``. Reproduced below with
     a running maximum of eos positions, which -- like ``torch.cummax`` followed
     by the one-element right shift in the device code -- looks strictly
     backwards, so padding cells after a request's last token cannot influence
     it either way.
   * **the column identity.** The device code indexes the concatenated
     ``[context | packed]`` row at ``columns + ngram_size - 1``, which is the
     token's own row index *only because* ``ngram_context`` is exactly
     ``ngram_size - 1`` wide. That is asserted, not assumed: a context of any
     other width means the device path is reading a different window than this
     one, and the correct response is to refuse, not to compute.

2. ``tests/models/qwen4_exp/test_ple_hostids.py`` compares this function against
   ``_compute_ngram_ids`` itself (on CPU tensors, so it runs without a GPU) and
   requires exact equality on every case -- real config-derived hash buffers
   *and* adversarial ones. A differential test against a duplicate of the math
   would prove nothing; this one calls the device function.

Refusal, not failure
--------------------
Nothing here raises into the engine. ``hostids_enabled()`` gates on the
environment, and ``compute_ngram_ids_host`` raises ``ValueError`` on any layout
it has not been shown to match; the caller catches that and takes the device
path. An optimization that cannot prove its preconditions steps aside -- the
same contract ``ple_prefetch.py`` uses (``disable(reason)``), and for the same
reason: a wrong PLE id is not a crash, it is a wrong answer.
"""

from __future__ import annotations

import os

import numpy as np

PADDING_SENTINEL = -1
"""Mirrors ``MmapShardedNGramEmbedding.PADDING_SENTINEL``.

Kept here as documentation of what the *id producer* may emit. Out-of-range ids
(including this sentinel) are not handled by this module: they are handled by
``MmapShardedNGramEmbedding.forward``, which folds them onto row 0. The host
path deliberately does not pre-fold, so that the fold -- and its warning, and
its ``VLLM_GFX906_PLE_STRICT`` raise -- happens in exactly one place for both
producers.
"""


def hostids_mode() -> int:
    """``VLLM_GFX906_PLE_HOSTIDS``: 0 off (default), 1 host ids, 2 differential.

    2 is the mode that has to run before 1 is ever trusted: it computes the host
    ids, keeps the device path fully in charge, and compares the two answers
    every step. It retains the blocking D2H, so a differential boot measures no
    performance at all -- it is a correctness boot whose subject is the position
    arithmetic that nothing host-side can check on its own.

    Deliberately read with ``os.environ.get`` rather than registered in
    ``vllm/envs.py``: like the other gfx906 experiment knobs it is local, and
    the cost of that is a benign "Unknown vLLM environment variable" warning at
    boot. Engagement is proven by the activation line the caller logs, never by
    the absence of that warning.
    """
    raw = os.environ.get("VLLM_GFX906_PLE_HOSTIDS", "0")
    if raw == "1":
        return 1
    if raw == "2":
        return 2
    return 0


def hostids_enabled() -> bool:
    """Host ids are authoritative for the steps the builder accepts."""
    return hostids_mode() == 1


def hostids_check_interval() -> int:
    """Every how many steps the generation tripwire samples (P1.5).

    The sample costs one 8-byte device->host copy issued non-blocking and one
    ``Event.query()`` on a later step, so the interval is not about cost -- it is
    about how long a frozen graph can run before it is caught. The default of 64
    means "a few hundred milliseconds of serving" at decode rates here, which is
    the difference between the 2026-09-20 incident (found after 5 h 30 m) and a
    boot-visible failure. Set to 1 to sample every step when debugging.
    """
    raw = os.environ.get("VLLM_GFX906_PLE_HOSTIDS_CHECK", "").strip()
    if not raw:
        return 64
    try:
        return max(1, int(raw))
    except ValueError:
        return 64


def gen_verdict(read_gen: int, staged_gen: int) -> str:
    """Compare what the graph delivered against what the host staged.

    ``read_gen`` is the generation the device reported for the last step that ran
    the staged branch; ``staged_gen`` is the generation the host had staged as of
    just before that step was prepared. Pure arithmetic with no torch in it, so
    the whole truth table is testable without a device:

    * ``"no-staging"`` -- the host has never staged, so there is nothing for the
      graph to have delivered. A vacuous pass is not a pass, so this is reported
      separately rather than folded into ``"ok"``.
    * ``"ok"`` -- the device is replaying exactly what the host last staged.
    * ``"frozen"`` -- the device is behind. This is the 2026-09-20 signature: the
      graph is running, or not running, but it is not carrying the host's work.
    * ``"ahead"`` -- the device is ahead of the host, which cannot happen with one
      producer and one scratch. It means the scratch is being written by something
      else (a second embedding sharing a buffer, a re-capture, a stale test), and
      an instrument being wrong about the direction of time is worth stopping for.
    """
    if staged_gen <= 0:
        return "no-staging"
    if read_gen == staged_gen:
        return "ok"
    return "frozen" if read_gen < staged_gen else "ahead"


def _as_int64(values: np.ndarray | None, name: str) -> np.ndarray:
    """Return ``values`` as a contiguous int64 array, without a copy if it can be.

    A copy is required (and allowed) if the caller handed over a different
    dtype: this is a per-step host function, so the cost is bounded by a few
    thousand int conversions, and taking ``np.asarray`` seriously (rather than
    asserting the dtype) is what keeps the pinned-buffer view case zero-copy.
    """
    if values is None:
        raise ValueError(f"{name} is required")
    out = np.asarray(values, dtype=np.int64)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def compute_ngram_ids_host(
    input_ids: np.ndarray,
    query_start_loc: np.ndarray,
    ngram_context: np.ndarray,
    *,
    multipliers: np.ndarray,
    vocab_sizes: np.ndarray,
    head_offsets: np.ndarray,
    eos_token_id: int,
    ngram_size: int,
    heads_per_ngram: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ngram embedding ids on the host.

    Numerically identical to ``Qwen4ExpNGramEmbedding._compute_ngram_ids``; see
    the module docstring for which parts of that identity are load-bearing.

    Args:
        input_ids: this step's token ids in token order, length ``T``. Values
            may be negative (the drafter's ``PADDING_SENTINEL``); range checking
            and the fold onto row 0 belong to the embedding module.
        query_start_loc: ``[num_reqs + 1]`` cumulative token offsets, exactly
            the tensor the device path is handed.
        ngram_context: ``[num_reqs, ngram_size - 1]`` -- the tokens immediately
            preceding each request's first token this step, left-padded with
            ``eos_token_id``.
        multipliers / vocab_sizes / head_offsets: the layer's hash buffers.
            Host (numpy or CPU-torch) mirrors of ``layer_multipliers``,
            ``ngram_heads_vocab_sizes`` and ``ngram_heads_offsets``.
        eos_token_id, ngram_size, heads_per_ngram: taken from the layer.
        out: optional pre-allocated ``[T, (ngram_size - 1) * heads_per_ngram]``
            int64 destination, which in production is
            ``ple_embedding._pinned_ngram_ids[:T].numpy()`` -- a view of pinned
            memory, so the ids land in the buffer the gather already reads and
            no copy happens at all.

    Raises:
        ValueError: any layout this function has not been shown to reproduce.
            The caller must fall back to the device path, not proceed.
    """
    input_ids = _as_int64(input_ids, "input_ids")
    query_start_loc = _as_int64(query_start_loc, "query_start_loc")
    ngram_context = _as_int64(ngram_context, "ngram_context")
    multipliers = _as_int64(multipliers, "multipliers").reshape(-1)
    vocab_sizes = _as_int64(vocab_sizes, "vocab_sizes").reshape(-1)
    head_offsets = _as_int64(head_offsets, "head_offsets").reshape(-1)

    ngram_size = int(ngram_size)
    heads_per_ngram = int(heads_per_ngram)
    eos_token_id = int(eos_token_id)
    num_heads = (ngram_size - 1) * heads_per_ngram

    # --- preconditions: refuse anything the device path has not been shown to
    # agree with. Every one of these is a shape the engine can legitimately
    # present (drafting, padding, a config we do not run), and the correct
    # response to all of them is the device path.
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be >= 2, got {ngram_size}")
    if heads_per_ngram <= 0:
        raise ValueError(f"heads_per_ngram must be > 0, got {heads_per_ngram}")
    if multipliers.size < ngram_size:
        raise ValueError(f"need {ngram_size} multipliers, got {multipliers.size}")
    if vocab_sizes.size < num_heads or head_offsets.size < num_heads:
        raise ValueError(
            f"need {num_heads} vocab_sizes/head_offsets, got "
            f"{vocab_sizes.size}/{head_offsets.size}"
        )
    if ngram_context.ndim != 2:
        raise ValueError(f"ngram_context must be 2D, got {ngram_context.ndim}D")
    if ngram_context.shape[1] != ngram_size - 1:
        # See "the column identity" in the module docstring: the device code
        # indexes at columns + ngram_size - 1, which only names the token
        # itself when the context is exactly ngram_size - 1 wide.
        raise ValueError(
            f"ngram_context width {ngram_context.shape[1]} != ngram_size - 1 "
            f"= {ngram_size - 1}; the device path would read a different window"
        )
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got {input_ids.shape}")

    num_reqs = int(query_start_loc.size) - 1
    if num_reqs < 0:
        raise ValueError("query_start_loc must hold at least one boundary")
    if ngram_context.shape[0] < num_reqs:
        raise ValueError(
            f"ngram_context has {ngram_context.shape[0]} rows for {num_reqs} requests"
        )
    num_tokens = int(input_ids.size)
    if num_reqs > 0 and int(query_start_loc[0]) != 0:
        raise ValueError(
            f"query_start_loc must start at 0, got {int(query_start_loc[0])}"
        )
    if num_reqs > 0 and int(query_start_loc[-1]) != num_tokens:
        raise ValueError(
            f"query_start_loc ends at {int(query_start_loc[-1])} but there are "
            f"{num_tokens} tokens"
        )
    if np.any(np.diff(query_start_loc) < 0):
        raise ValueError("query_start_loc must be non-decreasing")

    if out is None:
        out = np.empty((num_tokens, num_heads), dtype=np.int64)
    else:
        if out.dtype != np.int64:
            raise ValueError(f"out must be int64, got {out.dtype}")
        if out.shape != (num_tokens, num_heads):
            raise ValueError(
                f"out has shape {out.shape}, need {(num_tokens, num_heads)}"
            )
    if num_tokens == 0 or num_reqs == 0:
        return out

    ctx_len = ngram_size - 1
    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]

    for req in range(num_reqs):
        start, end = int(starts[req]), int(ends[req])
        qlen = end - start
        if qlen == 0:
            # The device path produces a row for every token it is given, and a
            # zero-length request contributes none, so there is nothing to do.
            continue
        # The device path's `context` row: [ngram_context | packed], where the
        # request's tokens occupy packed columns 0..qlen-1 and every later cell
        # is eos. Cells after this request's last token cannot affect any of its
        # ids (both the segment scan and the shifted gathers look backwards
        # only), so the row is cut at the last token this request owns.
        row = np.concatenate((ngram_context[req], input_ids[start:end]))
        # Row index of each of this request's tokens. `columns + ngram_size - 1`
        # in the device code; identical to the token's index because the width
        # precondition above holds.
        idx = np.arange(ctx_len, ctx_len + qlen, dtype=np.int64)

        # position_in_segment, for those indices only.
        #   previous_eos[i] = max{j < i : row[j] == eos}, or -1
        # which is torch.cummax over (j if eos else -1), shifted right by one
        # with a -1 pad -- i.e. `_shift_precompute`.
        eos_pos = np.where(row == eos_token_id, np.arange(row.size, dtype=np.int64), -1)
        previous_eos = np.concatenate((np.array([-1], dtype=np.int64), eos_pos[:-1]))
        previous_eos = np.maximum.accumulate(previous_eos)[idx]
        position_in_segment = idx - previous_eos - 1

        shifted = [row[idx]]
        for shift in range(1, ngram_size):
            source = idx - shift
            # `_shift_apply`: gather from clamp_min(source, 0) but keep the value
            # only when the source exists and the segment is at least `shift`
            # tokens deep; otherwise eos. The clamp is load-bearing for the
            # gather (a negative index would wrap the row) and harmless for the
            # result, because those lanes are replaced.
            values = np.where(
                (source >= 0) & (position_in_segment >= shift),
                row[np.clip(source, 0, None)],
                eos_token_id,
            )
            shifted.append(values)

        # One int64 multiply per (token, shift) is allowed to overflow. See the
        # module docstring: torch wraps, numpy wraps, only the warning differs.
        with np.errstate(over="ignore"):
            for ngram in range(2, ngram_size + 1):
                head_start = (ngram - 2) * heads_per_ngram
                head_end = head_start + heads_per_ngram
                mixed = shifted[0] * multipliers[0]
                for index in range(1, ngram):
                    mixed = np.bitwise_xor(
                        mixed,
                        shifted[index] * multipliers[index],
                    )
                out[start:end, head_start:head_end] = (
                    np.remainder(
                        mixed[:, None],
                        vocab_sizes[None, head_start:head_end],
                    )
                    + head_offsets[None, head_start:head_end]
                )
    return out


class HostIdsRefused(ValueError):
    """The host cannot produce this step's ids, so the device path must run.

    Subclasses :class:`ValueError` because that is the contract
    ``compute_ngram_ids_host`` already documents: a refusal is not a failure to
    propagate, it is the signal to fall back. ``reason`` is a short stable token
    so a caller can *count* which precondition fires in production instead of
    guessing from a log line -- the difference between "the host path covers
    94 % of decode steps" and a knob that quietly never engages.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class _PendingCommit:
    """One step's accepted tokens, waiting for its D2H to land.

    Holds references rather than copies, for **both** arrays the copy event
    covers. ``sampled`` and ``num_sampled`` are the two outputs of one
    ``AsyncOutput`` (``async_utils.py``: ``async_copy_to_np(sampler_output
    .sampled_token_ids)`` and ``async_copy_to_np(num_sampled_tokens)``), so each
    is a fresh host allocation per step whose CONTENTS are meaningless until
    ``done`` reports the copy landed. Copying either at stage time is precisely
    the bug this class exists to avoid.

    The docstring used to claim the opposite about ``num_sampled`` -- that it
    "comes from a persistent buffer the runner overwrites on the next step" --
    and ``stage_step`` therefore copied it. That copy is the root cause of the
    ``num_sampled [0, 4, 8] outside 0..4`` drops: ``[0, 4, 8]`` is not a count
    vector, it is what the allocator had left in that host block (offsets from
    the previous step's flattened copy), read microseconds after the D2H was
    enqueued. Boot ``vllm.log.20261008-083018`` recorded 14,859 such drops
    (9.9 % of commits) and boot ``2026-10-09 16:18`` recorded the same four at
    startup, one per worker.

    ``slots`` IS copied at stage time, and must stay that way: it is
    ``input_batch.idx_mapping_np[:num_reqs]``, a view into a persistent pinned
    buffer the runner rewrites during the NEXT step's input prep, which can happen
    before this event lands.

    ``void`` carries the rows :meth:`HostTokenHistory.invalidate` cancelled. It
    can no longer zero the source array the way the old code did, because the
    source is ``AsyncOutput``'s array and its copy may still be in flight: a zero
    written now is overwritten when the copy lands, resurrecting exactly the
    commit invalidation exists to prevent. The cancellation is remembered here and
    applied by :meth:`counts`, after the copy has landed.
    """

    __slots__ = ("done", "num_sampled", "sampled", "slots", "void")

    def __init__(self, slots, sampled, num_sampled, done) -> None:  # noqa: ANN001
        self.slots = slots
        self.sampled = sampled
        self.num_sampled = num_sampled
        self.done = done
        self.void = None  # np.ndarray | None, bool mask over rows

    def counts(self):
        """Read this step's accepted counts. CALL ONLY once ``done`` says landed.

        Returns a private copy, so the caller can clamp or zero rows without
        writing through to the array output processing is reading too.
        """
        ns = np.asarray(self.num_sampled).copy()
        if self.void is not None:
            ns[self.void] = 0
        return ns


class HostTokenHistory:
    """The last ``width`` committed tokens of each request, on the host.

    ``build_decode_step_inputs`` needs ``ngram_context_len + 1`` consecutive
    tokens per request, ending at the newest committed one. ``width`` is exactly
    that many, so ``window[slot]`` is the chronological tail of the request:
    ``window[slot, -1]`` is the token the next step feeds as its first row and
    ``window[slot, :-1]`` is its ngram context.

    Why the sampled tokens are the right source, and not ``all_token_ids``.
    The device appends generated tokens in ``_post_update_kernel``
    (``input_batch.py:577``), which takes the first ``num_sampled[i]`` entries of
    row i of ``sampler_output.sampled_token_ids`` and writes them at
    ``total_len + i``. ``AsyncOutput`` (``async_utils.py``) copies *that same
    tensor* and *those same counts* to host memory every step for output
    processing, and under sync scheduling ``get_output()`` waits for it before
    the scheduler produces the next step. So this class mirrors a copy the
    worker already pays for: no extra D2H, no extra stream, and no wait at all --
    :meth:`drain` only ever calls ``Event.query()``.

    The trim rule is the device's trim rule on purpose: ``num_sampled`` is how
    many of a row's sampled tokens were accepted, so a rejected draft is never
    appended here either, and the window agrees with the device's history by
    construction instead of by luck. ``length[slot]`` counts the same accepted
    tokens, which is what lets the builder require exact agreement with
    ``num_computed_tokens`` instead of trusting it.

    Not thread safe and not meant to be: it is staged by the model runner's
    epilogue and drained by the same worker thread at the top of the next step.
    """

    MAX_PENDING = 8
    """Bound on queued commits.

    Reaching it flushes the queue (one blocking wait for the head commit) and
    only drops if the flush did not free a slot. Eight steps of un-drained
    output copies means the copy stream is a full step behind, which is a bug
    worth seeing in the counters rather than hiding behind a bigger bound.

    It used to drop the *new* commit outright, on the theory that a drop costs
    coverage rather than correctness. Boot #26 showed the theory's blind spot:
    a dropped commit is never recovered, so the exact-length check refuses that
    request on *every* step for the rest of its life. Six consecutive windows of
    32/32 ``history-length`` refusals, with the drops that caused them all
    clustered at batch transitions. Blocking once is much the cheaper failure.
    """

    def __init__(
        self, num_slots: int, width: int, vocab_bound: int | None = None
    ) -> None:
        if num_slots <= 0 or width <= 0:
            raise ValueError(
                f"num_slots and width must be positive, got {num_slots}, {width}"
            )
        self.num_slots = int(num_slots)
        self.width = int(width)
        # Task #69: the ceiling a committed value must sit under to be a token id at
        # all. Passed in by the model state from ``hf_text_config.vocab_size``; the
        # fallback is NOT a vocab size and is labelled as such in the report, because
        # "out of range" is only interesting if the reader knows which range. A bound
        # that is too wide still catches the recycled-memory class (float bit patterns
        # reinterpret as ~1e18 integers) and misses values that merely look like
        # another request's ids -- which is exactly the split seen in the 148 logged
        # disagreement events, so the wide bound is a real observation, not a shrug.
        self.vocab_bound = int(vocab_bound) if vocab_bound else (1 << 31) - 1
        self.vocab_bound_source = "vocab_size" if vocab_bound else "fallback-2p31"
        self.bad_value_events = 0
        self.last_bad_value = ""
        self.window = np.zeros((self.num_slots, self.width), dtype=np.int64)
        self.length = np.zeros(self.num_slots, dtype=np.int64)
        self._pending: list[_PendingCommit] = []
        # Counted, not raised: a knob that can kill a boot is worse than one
        # that reports its own coverage.
        self.staged = 0
        self.dropped = 0
        # A commit's accepted tokens did not reach the window, wherever the hole
        # was detected: ``stage_step`` counts the shape/slot/queue failures it can
        # see without reading the copy, and ``drain`` counts a row whose counts
        # were out of range once the copy had actually landed. One counter because
        # the reader's question is "is the mirror missing tokens", not "which
        # function noticed"; ``last_drop`` says which check fired and when.
        # The two ways a commit goes missing, which boot #26 could not tell
        # apart because neither was ever surfaced: sampler rows trimmed to the
        # real batch, and queue overflows flushed instead of dropped. Both are
        # counted rather than logged because this runs on every step.
        self.sliced = 0
        self.flushes = 0
        # Detail of the most recent drop, for the caller to log once. A bare
        # drop count says a commit was lost; the detail says which check lost it
        # and what the row counts actually were, which is the difference between
        # a diagnosis and a suspicion.
        self.last_drop = ""

    def _check_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.num_slots:
            raise ValueError(f"slot {slot} outside 0..{self.num_slots - 1}")

    def seed(self, slot: int, tokens: object) -> None:
        """(Re)start a slot from the full token list the scheduler sends at add.

        ``RequestState.add_request`` gets ``all_token_ids`` = prompt plus, for a
        resumed request, the tokens already generated. Seeding from it is exact,
        and it is also what makes a re-preempted request usable again: the
        window is rebuilt from scratch, so a stale one can never survive a slot
        being reused.
        """
        self._check_slot(slot)
        values = np.asarray(tokens, dtype=np.int64).ravel()
        self.length[slot] = int(values.size)
        row = self.window[slot]
        row[:] = 0
        if values.size:
            take = min(values.size, self.width)
            row[self.width - take :] = values[values.size - take :]

    def commit(self, slot: int, values: object) -> None:
        """Append accepted tokens to a slot's window (empty is a no-op)."""
        self._check_slot(slot)
        row = np.asarray(values, dtype=np.int64).ravel()
        n = int(row.size)
        if n == 0:
            return
        self.length[slot] += n
        w = self.window[slot]
        if n >= self.width:
            w[:] = row[n - self.width :]
        else:
            # Chronological left-to-right, newest at the right: drop the n
            # oldest cells and shift the rest left. ``.copy()`` because the
            # slices overlap and numpy does not promise correct behaviour for
            # overlapping assignment.
            w[: self.width - n] = w[n:].copy()
            w[self.width - n :] = row

    def invalidate(self, slot: int) -> None:
        """Forget a slot, and drop any queued commit that would resurrect it.

        A request removed while its step's copy is still in flight would
        otherwise append its tokens to whoever reuses the slot next. The
        exact-length check would catch that too, but it would catch it as a
        permanent refusal for the new request -- so the queue is cleaned here.
        """
        self._check_slot(slot)
        self.length[slot] = 0
        self.window[slot] = 0
        for entry in self._pending:
            hit = entry.slots == slot
            if hit.any():
                # Remembered, not written: the source array's D2H may still be in
                # flight, and a zero written into it now is overwritten when the
                # copy lands. See ``_PendingCommit.void``.
                entry.void = hit if entry.void is None else entry.void | hit

    def stage_step(
        self,
        slots: np.ndarray,
        sampled: np.ndarray,
        num_sampled: np.ndarray,
        done: object,
    ) -> bool:
        """Queue one step's accepted tokens. Reads nothing; blocks only to avoid
        losing a commit it cannot recover (see ``MAX_PENDING``).

        ``done`` is the ``AsyncOutput`` copy event (anything with ``query()``,
        including ``None`` for a caller whose data is already host-side, which is
        how the tests stage synchronously).

        Returns False -- and counts a drop, recording why in ``last_drop`` -- on
        any inconsistency, because a malformed commit would reorder a slot's
        history.

        **Row counts are not required to match; they are required to be
        explainable.** ``slots`` is ``input_batch.idx_mapping_np[:num_reqs]``,
        the real batch, while ``sampled`` and ``num_sampled`` are the sampler's
        own arrays, copied out whole by ``AsyncOutput`` (``async_utils.py`` calls
        ``async_copy_to_np`` with no slice). A step replayed from a padded
        descriptor therefore carries MORE sampler rows than slots, and the rows
        past ``len(slots)`` belong to no request in this batch. Trimming to the
        first ``len(slots)`` rows is exact: the sampler emits one row per request
        in the same batch order ``idx_mapping_np`` is built in. Fewer rows than
        slots stays a drop -- there is no way to know which request an absent
        commit was meant for.
        """
        slots = np.asarray(slots)
        num_sampled = np.asarray(num_sampled)
        self.last_drop = ""
        if sampled.ndim != 2 or slots.ndim != 1 or num_sampled.ndim != 1:
            return self._drop(
                f"ndim sampled={sampled.ndim} slots={slots.ndim} "
                f"num_sampled={num_sampled.ndim}"
            )
        n = slots.shape[0]
        if sampled.shape[0] < n or num_sampled.shape[0] < n:
            return self._drop(
                f"rows sampled={sampled.shape[0]} "
                f"num_sampled={num_sampled.shape[0]} < slots={n}"
            )
        if sampled.shape[0] > n or num_sampled.shape[0] > n:
            # A view, so this reads no token values: the array's event has not
            # necessarily landed and staging must stay safe while it is in flight.
            #
            # Since task #56 this is DEFENSE IN DEPTH, not the load-bearing guard it
            # used to be. The count range check ran HERE, over every row of
            # ``num_sampled`` including the pad rows, so junk in an unused row vetoed
            # the whole batch's commit and trimming was the only thing that saved it.
            # The check now runs in ``drain``, whose loop is bounded by
            # ``entry.slots.size`` -- pad rows are not consulted at all any more. What
            # the trim guarantees instead is the invariant every consumer below
            # assumes without restating it: row i belongs to slot i, and there are
            # exactly ``len(slots)`` rows. A future drain that iterated
            # ``sampled.shape[0]`` would otherwise start reading whatever the sampler
            # left in the graph's unused rows -- which is junk, not zeros.
            sampled = sampled[:n]
            num_sampled = num_sampled[:n]
            self.sliced += 1
        if slots.size and (int(slots.min()) < 0 or int(slots.max()) >= self.num_slots):
            return self._drop(f"slot {slots.tolist()} outside 0..{self.num_slots - 1}")
        # NO check on the VALUES of ``num_sampled`` here, deliberately, and this
        # is the whole of task #56. The counts are the second output of the D2H
        # this commit is waiting on, so a range check here reads whatever the
        # allocator recycled into that block: it rejected healthy commits (the
        # ``[0, 4, 8] outside 0..4`` drops) and would have passed a corrupt one
        # whose garbage happened to look small. Every check that stays here is
        # shape-only or slot-only, which is exactly the set that is legal to make
        # before a copy has landed.
        if len(self._pending) >= self.MAX_PENDING:
            # Blocking, and the only blocking call in this class, for the reason
            # in the MAX_PENDING docstring: the alternative is a commit that is
            # never recovered. Counted, so a boot that does this every step is
            # visible as a slow boot rather than as a coverage hole.
            self.drain(flush=True)
            if len(self._pending) >= self.MAX_PENDING:
                return self._drop(f"pending full ({len(self._pending)})")
        self._pending.append(
            # ``num_sampled`` by reference: see ``_PendingCommit``.
            _PendingCommit(slots.copy(), sampled, num_sampled, done)
        )
        self.staged += 1
        return True

    def _drop(self, detail: str) -> bool:
        """Count a refused commit, record why, and return False for the caller."""
        self.dropped += 1
        self.last_drop = detail
        return False

    def _observe_committed(
        self, row: object, slot: int, i: int, entry: "_PendingCommit"
    ) -> None:
        """Task #69. Look at a row about to be committed and record anything that
        could not be a token id. OBSERVATION ONLY: it changes no control flow, drops
        nothing, and never raises.

        Why this exists. Across 148 logged host-vs-device disagreement events the
        defect is one cell -- the newest -- read before the worker wrote it, and the
        host values in that cell were either fused small integers or, reinterpreted
        as two float32s, plausible floats ((0.0037, 0.00036), (985.15, -619.197)).
        Five theories were eliminated statically (split-event lag, drain ignoring
        invalidation, int64 fusion of two int32s, the ``-1`` sampler sentinel, and a
        read-index error). What is left cannot be seen in source: ``async_copy_to_np``
        is ``x.to("cpu", non_blocking=True).numpy()``, a FRESH host allocation per
        step that the system allocator recycles, so the only way to tell "the block
        handed over is not the block the copy wrote" from "the device wrote that
        block" is to look at a live boot. This method is that look.

        Each field is here because it separates two explanations:

        * ``row[:8]`` versus the offending cell -- sane ids in the prefix and garbage
          in the tail means the copy DID write this block and stopped short; garbage
          across the whole row means nothing wrote it this step.
        * ``addr`` -- the host address of the buffer the commit reads. Recycled
          blocks show up as the same address carrying different content across
          events, which is the allocator's signature; a monotonically new address per
          step with the same content is not.
        * ``bad_as_float32`` -- the reinterpretation, computed here so a log line is
          evidence on its own instead of a number someone has to come back to.
        * ``window_tail`` -- the cell the next step will actually feed, which is
          ``window[slot, width-1]`` (both writers keep the newest token there for any
          length), not the length-indexed cell it is tempting to read.

        Cadenced rather than per-event: a boot where this fires every step is the
        failure mode, and a log that says so 19 times a second is not evidence.
        """
        try:
            arr = np.asarray(row)
            if arr.size == 0:
                return
            bad = arr[(arr < 0) | (arr >= self.vocab_bound)]
            if bad.size == 0:
                return
            self.bad_value_events += 1
            if not (
                self.bad_value_events <= 4 or self.bad_value_events % 64 == 0
            ):
                return
            full = np.asarray(entry.sampled)
            prefix = [int(v) for v in np.asarray(full[i])[:8]]
            # int64 buffer re-read as two float32s: the check for the recycled-float
            # class, in the same line as the integer that raised the alarm.
            pair = np.asarray([int(bad[0])], dtype=np.int64).view(np.float32)
            self.last_bad_value = (
                f"#{self.bad_value_events} slot={slot} row={i} "
                f"n={arr.size} bound={self.vocab_bound}"
                f"({self.vocab_bound_source}) "
                f"bad={[int(v) for v in bad[:4]]} "
                f"bad_as_float32={[float(v) for v in pair]} "
                f"row[:8]={prefix} "
                f"addr=0x{int(getattr(getattr(entry.sampled, 'ctypes', None), 'data', 0)):x} "
                f"shape={tuple(full.shape)} dtype={full.dtype} "
                f"window_tail={int(self.window[slot, self.width - 1])}"
            )
        except Exception as exc:  # noqa: BLE001 - an instrument must not cost a boot
            self.last_bad_value = f"observation failed: {type(exc).__name__}: {exc}"

    def drain(self, flush: bool = False) -> int:
        """Apply every queued commit whose copy has landed. Never blocks unless
        ``flush`` says to wait for the head.

        Stops at the first commit that has not landed: the window is append-only
        per slot, so applying step N+1 before step N would put a request's
        tokens in the wrong order.

        ``flush`` is the repair path :meth:`stage_step` uses when the queue is
        full: it waits for the head commit instead of returning with the queue
        still full. Only the head -- one wait is enough to make room, and waiting
        for the whole queue would turn a rare repair into a stall that grows with
        how far behind the copy stream is. Counted in ``flushes``.
        """
        applied = 0
        flushed = False
        while self._pending:
            entry = self._pending[0]
            if entry.done is not None and not entry.done.query():
                if not flush or flushed:
                    break
                entry.done.synchronize()  # type: ignore[union-attr]
                self.flushes += 1
                flushed = True
                # Re-ask rather than assume. Applying a commit whose copy has not
                # landed is the one thing this queue exists to prevent, and a
                # wait that returns without completing must not sneak past it.
                if not entry.done.query():
                    break

            self._pending.pop(0)
            # The counts are read HERE and nowhere earlier: same D2H as
            # ``sampled``, same rule about when its contents exist.
            counts = entry.counts()
            hi = int(np.asarray(entry.sampled).shape[1])
            for i in range(int(entry.slots.size)):
                n = int(counts[i])
                if n < 0 or n > hi:
                    # Post-landing this is a device-side anomaly rather than the
                    # recycled-memory artifact it used to be -- the device's own
                    # ``_post_update_kernel`` trusts these same counts. Commit
                    # nothing for that row: a wrong-length append is the history
                    # corruption the range check exists to prevent. Count the hole
                    # so the next step's exact-length check asks the device to
                    # repair it, exactly as a stage-time drop did.
                    self.dropped += 1
                    self.last_drop = (
                        f"num_sampled {counts.tolist()} outside 0..{hi} "
                        "(read after the copy landed)"
                    )
                    continue
                if n:
                    row = entry.sampled[i, :n]
                    # Observation only (task #69): reads the row, records what it saw,
                    # commits anyway. Committing is what this line always did.
                    self._observe_committed(row, int(entry.slots[i]), i, entry)
                    self.commit(int(entry.slots[i]), row)
                    applied += 1
        return applied

    @property
    def pending(self) -> int:
        """Commits queued but not yet applied."""
        return len(self._pending)


def build_decode_step_inputs(
    idx_mapping_np: np.ndarray,
    num_scheduled_tokens_np: np.ndarray,
    num_computed_tokens_np: np.ndarray,
    history_window_np: np.ndarray,
    history_len_np: np.ndarray,
    draft_values_np: np.ndarray | None,
    draft_row_idx: np.ndarray | None,
    *,
    ngram_context_len: int,
    eos_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild one pure-decode step's PLE inputs on the host.

    Returns ``(input_ids, query_start_loc, ngram_context)`` -- exactly the three
    arrays ``Qwen4ExpNGramEmbedding._compute_ngram_ids`` is handed on the device,
    so the result goes straight into :func:`compute_ngram_ids_host`.

    Why each piece lives where it lives (the derivation, not a guess):

    * **The first row's token is at sequence position ``num_computed_tokens``.**
      ``_post_update_kernel`` advances ``total_len`` by ``num_sampled`` and
      ``num_computed_tokens`` by ``query_len - num_rejected``, and those two
      deltas are the same number (the accepted rows). So a request carries
      exactly one more stored token than it has computed tokens, and the stored
      token the next step feeds first is the one at index
      ``num_computed_tokens`` -- the newest committed token, held here in
      ``history_window_np[slot, -1]``.
    * **The context window is the ``ngram_context_len`` tokens strictly before
      that index**, EOS-filled on the left. That is ``_prepare_ngram_context``
      reproduced: ``context_end = num_computed_tokens``, offsets
      ``arange(-context_len, 0)``, ``where(idx >= 0, token, eos)``. Both come
      from :class:`HostTokenHistory` rather than from
      ``req_states.all_token_ids``: that buffer's generated region is written by
      a GPU kernel that is not ordered before this host read, so reading it
      mid-decode yields zeros for exactly the newest cells. See the module
      docstring; the first differential boot proved it.
    * **Rows 1..q-1 are this step's drafts** and are deliberately *not* in
      ``all_token_ids`` (they are speculative, only the accepted prefix is
      committed). They come from the pinned mirror staged at the previous step's
      epilogue, addressed by the row index that mirror published -- never by
      batch position.

    What the exact-length check proves. ``num_computed_tokens_np`` is updated by
    the scheduler here and by ``_post_update_kernel`` on the device, and
    ``states.py`` calls the host one an *optimistic upper bound*; until P1.6
    nothing here could notice a disagreement, which is why differential mode
    (mode 2) was mandatory. ``history_len_np[slot]`` counts accepted tokens by a
    third path -- the same ``num_sampled`` the post-update kernel consumes, summed
    host-side -- so requiring ``history_len == num_computed + 1`` makes two
    independently-maintained counters prove each other's arithmetic before a
    single id is built. Disagreement of either kind, lag or lead, is a refusal.

    Differential mode is still the gate on this function: agreeing counters is
    not the same as agreeing ids, and only a comparison against the device's own
    ``_compute_ngram_ids`` output can show the reproduction is exact.

    Raises:
        HostIdsRefused: any batch shape this function has not been shown to
            reproduce. The caller must take the device path.
    """
    if draft_values_np is None or draft_row_idx is None:
        raise HostIdsRefused("no-drafts", "staging is off or nothing was staged")
    ngram_context_len = int(ngram_context_len)
    if ngram_context_len < 1:
        raise HostIdsRefused("context-width", str(ngram_context_len))

    num_reqs = int(idx_mapping_np.size)
    if num_reqs == 0:
        raise HostIdsRefused("empty-batch")
    if int(num_scheduled_tokens_np.size) < num_reqs:
        raise HostIdsRefused(
            "layout",
            f"num_scheduled_tokens has {num_scheduled_tokens_np.size} "
            f"entries for {num_reqs} requests",
        )
    if history_window_np.ndim != 2:
        raise HostIdsRefused("layout", "history_window_np must be 2D")
    num_slots, width = (
        int(history_window_np.shape[0]),
        int(history_window_np.shape[1]),
    )
    if width < ngram_context_len + 1:
        # The window has to hold the context *and* the token at the current
        # position. A narrower window means the producer and this reader
        # disagree about what ``ngram_context_len`` counts, which is the same
        # class of mistake as the column identity in compute_ngram_ids_host:
        # refuse, do not compute from a shifted window.
        raise HostIdsRefused(
            "window-width",
            f"history window is {width} wide, need ngram_context_len + 1 "
            f"= {ngram_context_len + 1}",
        )
    if int(history_len_np.size) < num_slots:
        raise HostIdsRefused(
            "layout",
            f"history_len has {history_len_np.size} entries for {num_slots} slots",
        )

    slots = np.asarray(idx_mapping_np, dtype=np.int64)
    if (slots < 0).any():
        raise HostIdsRefused("negative-slot", slots.tolist())
    queries = np.asarray(num_scheduled_tokens_np, dtype=np.int64)[:num_reqs]
    num_draft = int(draft_values_np.shape[1])
    if not bool((queries == num_draft + 1).all()):
        # A request with a different row count is a prefill, a chunked prefill, a
        # short draft, or a padded slot. Any of them means this step is not the
        # shape the host path was written for, and the batch is not separable:
        # the ids array is one flat buffer, so a partial host answer would be a
        # partial D2H -- which is the whole reason narrowed D1 was killed.
        raise HostIdsRefused(
            "not-pure-decode",
            f"queries {queries.tolist()} != draft+1 = {num_draft + 1}",
        )

    # Staged rows are addressed by the req_state slot they belonged to. The
    # previous step's batch may be larger, smaller, or shifted relative to this
    # one, and a slot appears at most once per batch, so a scatter into a
    # max_num_reqs-sized table is exact and costs one fill per step.
    row_of_slot = np.full(num_slots, -1, dtype=np.int64)
    staged_slots = np.asarray(draft_row_idx, dtype=np.int64)
    if staged_slots.size and int(staged_slots.max()) >= row_of_slot.size:
        raise HostIdsRefused("layout", "staged row index out of range")
    row_of_slot[staged_slots] = np.arange(staged_slots.size, dtype=np.int64)

    positions = np.asarray(num_computed_tokens_np, dtype=np.int64)[slots]
    if (positions < 0).any():
        raise HostIdsRefused("position-out-of-range", f"positions {positions.tolist()}")
    lengths = np.asarray(history_len_np, dtype=np.int64)[slots]
    if not bool((positions == lengths - 1).all()):
        # The window holds tokens 0..length-1 of the request, so the position
        # this step reads must be length-1. A lag means an un-drained commit or a
        # host counter that raced ahead of the device; a lead means the opposite.
        # Either way every id would be built from a shifted context, which is
        # unobservable downstream and therefore worth refusing a step over.
        raise HostIdsRefused(
            "history-length",
            f"positions {positions.tolist()} != committed-1 {(lengths - 1).tolist()}",
        )

    total_tokens = int(queries.sum())
    input_ids = np.empty(total_tokens, dtype=np.int64)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(queries, out=query_start_loc[1:])
    ngram_context = np.full(
        (num_reqs, ngram_context_len), int(eos_token_id), dtype=np.int64
    )

    for i in range(num_reqs):
        slot = int(slots[i])
        pos = int(positions[i])
        row = int(row_of_slot[slot])
        if row < 0:
            # This request was not in the step that staged the drafts, so its
            # rows 1..q-1 are unknown here. Refusing the whole batch is the only
            # safe answer: the device would use this step's real drafts.
            raise HostIdsRefused("row-missing", f"slot {slot} had no staged draft")

        # ``min(ngram_context_len, pos)`` cells of the window are real tokens;
        # the rest of the context is the device's eos fill (its
        # ``where(idx >= 0, token, eos)`` over ``context_end + offsets``).
        real = min(ngram_context_len, pos)
        if real:
            window_row = history_window_np[slot]
            ngram_context[i, ngram_context_len - real :] = window_row[
                width - 1 - real : width - 1
            ]

        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        input_ids[start] = history_window_np[slot, width - 1]
        input_ids[start + 1 : end] = draft_values_np[row, : end - start - 1]

    return input_ids, query_start_loc, ngram_context


__all__ = [
    "PADDING_SENTINEL",
    "HostIdsRefused",
    "HostTokenHistory",
    "build_decode_step_inputs",
    "compute_ngram_ids_host",
    "gen_verdict",
    "hostids_check_interval",
    "hostids_enabled",
    "hostids_mode",
]
