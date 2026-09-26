# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in PLE read-ahead (``VLLM_GFX906_PLE_PREFETCH``).

The read-ahead tells the kernel which pages the *next* prefill chunk will fault.
It cannot change a computed value -- it only issues ``posix_fadvise`` on a file
-- so the properties worth asserting are not about numerics. They are about
whether the hint points at the bytes the gather will actually read, and about
whether the thing can fail in a way that hurts the engine:

* the lookahead ids for chunk *k+1* are **identical** to the ids the real path
  computes when chunk *k+1* arrives. If these ever diverge the prefetcher keeps
  running and keeps warming pages that will never be read -- wasted NVMe traffic
  is invisible in the output, so only a test can see it;
* the advised byte ranges are the real file offsets of the real rows, derived
  from ``/proc/self/maps`` (the arithmetic that turns a ``data_ptr`` into a
  file offset, and the check that the shard is still a view rather than a copy);
* every failure mode disables the prefetcher and never raises into the model
  path -- a knob must be able to refuse itself, but not to refuse a boot.
"""

from __future__ import annotations

import contextlib
import mmap
from uuid import uuid4

import numpy as np
import torch

from vllm.models.qwen4_exp.amd import ple_prefetch
from vllm.models.qwen4_exp.amd.ple_layer import (
    MmapShardedNGramEmbedding,
    Qwen4ExpNGramEmbedding,
)
from vllm.models.qwen4_exp.amd.ple_prefetch import (
    PleRowPrefetcher,
    plan_lookahead_chunks,
    prefetch_enabled,
)

# A tiny fake table: 4 shards x 64 rows, so the whole id space fits in 256 rows.
NUM_SHARDS = 4
CAP = 64
DIM = 8
NGRAM_SIZE = 3
HEADS = 2
CTX_LEN = NGRAM_SIZE - 1
EOS = 999

_MULTIPLIERS = torch.tensor([1, 2654435761, 40503], dtype=torch.long)
_SIZES = torch.tensor([16, 16, 16, 16], dtype=torch.long)
_OFFSETS = torch.tensor([0, 16, 32, 48], dtype=torch.long)
_REAL_LIBC = ple_prefetch._libc


def _ids(input_ids, query_start_loc, context, packed_rows):
    """Call the shared id function the way the model path does."""
    n = input_ids.shape[0]
    return Qwen4ExpNGramEmbedding._compute_ngram_ids(
        input_ids,
        query_start_loc,
        context,
        multipliers=_MULTIPLIERS,
        vocab_sizes=_SIZES,
        head_offsets=_OFFSETS,
        packed=torch.full((1, packed_rows), EOS, dtype=torch.long),
        positions=torch.arange(n, dtype=torch.long),
        eos_token_id=EOS,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS,
    )


def _embedding(shard_tensors=None, rows=CAP):
    emb = MmapShardedNGramEmbedding(NUM_SHARDS, rows, DIM)
    for i in range(NUM_SHARDS):
        tensor = _shard_tensors()[i] if shard_tensors is None else shard_tensors[i]
        emb.set_shard(i, tensor)
    return emb


def _shard_tensors(rows=CAP):
    return [
        torch.arange(rows * DIM, dtype=torch.float16).reshape(rows, DIM)
        for _ in range(NUM_SHARDS)
    ]


def _prefetcher(emb, id_fn=None):
    return PleRowPrefetcher(
        emb,
        (_MULTIPLIERS, _SIZES, _OFFSETS),
        id_fn or Qwen4ExpNGramEmbedding._compute_ngram_ids,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS,
        eos_token_id=EOS,
        max_tokens=1024,
    )


@contextlib.contextmanager
def _file_backed(tmp_path, rows=256, dim=DIM):
    """An embedding whose shards really are views into a file.

    The hint path only engages on file-backed mappings, so any test that wants
    to reach the worker thread needs a real mmap, not a heap tensor. Each shard
    is mapped at its own page-aligned offset, like real shards in a safetensors
    file.
    """
    row_bytes = rows * dim * 2
    path = tmp_path / f"table-{uuid4().hex}.bin"
    path.write_bytes(np.zeros(row_bytes * NUM_SHARDS, dtype=np.float16).tobytes())
    tensors, handles = [], []
    try:
        for shard in range(NUM_SHARDS):
            # The fd is only needed to create the mapping; Linux keeps the
            # inode alive under the mmap, so it can close straight away.
            with open(path, "rb") as fh:
                mm = mmap.mmap(
                    fh.fileno(),
                    row_bytes,
                    offset=shard * row_bytes,
                    access=mmap.ACCESS_READ,
                )
            handles.append(mm)
            tensors.append(
                torch.frombuffer(mm, dtype=torch.float16, count=rows * dim).reshape(
                    rows, dim
                )
            )
        yield _embedding(tensors, rows=rows)
    finally:
        tensors.clear()  # drop the buffer exports before unmapping
        for mm in handles:
            mm.close()


class _SpyLibc:
    """Records posix_fadvise calls instead of making them."""

    def __init__(self, raises=False):
        self.calls: list[tuple[int, int, int, int]] = []
        self.raises = raises

    def posix_fadvise(self, fd, offset, length, advice):
        if self.raises:
            raise OSError("simulated EIO on the hint path")
        self.calls.append((fd.value, offset.value, length.value, advice.value))
        return 0


class _InputBatch:
    """The four numpy fields the planning hook reads off the real InputBatch."""

    def __init__(self, idx, scheduled, computed, prefill_len):
        self.num_reqs = len(idx)
        self.idx_mapping_np = np.asarray(idx, dtype=np.intp)
        self.num_scheduled_tokens = np.asarray(scheduled, dtype=np.int32)
        self.num_computed_tokens_np = np.asarray(computed, dtype=np.int32)
        self.prefill_len_np = np.asarray(prefill_len, dtype=np.int32)

    def plan(self, host, context_len=CTX_LEN, eos=EOS):
        """What the model-state hook would hand to the prefetcher."""
        n = self.num_reqs
        return list(
            plan_lookahead_chunks(
                host_tokens=host,
                idx_mapping=self.idx_mapping_np[:n],
                scheduled=self.num_scheduled_tokens[:n],
                computed=self.num_computed_tokens_np[:n],
                prefill_len=self.prefill_len_np[:n],
                context_len=context_len,
                eos_token_id=eos,
            )
        )


# ---------------------------------------------------------------------------
# the knob
def test_read_ahead_is_off_by_default(monkeypatch):
    """Default-off is the standing rule for every gfx906 perf knob."""
    monkeypatch.delenv("VLLM_GFX906_PLE_PREFETCH", raising=False)
    assert prefetch_enabled() is False
    monkeypatch.setenv("VLLM_GFX906_PLE_PREFETCH", "1")
    assert prefetch_enabled() is True
    monkeypatch.setenv("VLLM_GFX906_PLE_PREFETCH", "0")
    assert prefetch_enabled() is False


# ---------------------------------------------------------------------------
# the property that makes the whole feature mean anything
def test_lookahead_ids_equal_the_ids_the_real_path_will_compute(tmp_path):
    """Chunk k+1 warmed == chunk k+1 gathered, row for row.

    This is the only test that can catch a prefetcher that works and is useless:
    if the context or the slice is off by one, the ids hash to different rows,
    the hints land on pages nobody reads, and every visible symptom stays green.
    A deliberately wrong context is asserted to differ, so the test cannot pass
    by being insensitive.
    """
    seen: list[torch.Tensor] = []
    real = Qwen4ExpNGramEmbedding._compute_ngram_ids

    def record(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out.clone())
        return out

    # File-backed shards: resolution is a precondition for computing anything,
    # so with heap tensors this would stop before the interesting part.
    with _file_backed(tmp_path) as emb:
        pf = _prefetcher(emb, record)
        host = np.arange(2 * 32, dtype=np.int32).reshape(2, 32)
        # One request: 8 tokens claimed by the step in flight, prompt is 24
        # tokens long, so the next chunk is [8, 24) with context tokens 6, 7.
        ((req, start, end, ctx),) = _InputBatch([0], [8], [0], [24]).plan(host)
        assert (req, start, end) == (0, 8, 24)
        np.testing.assert_array_equal(ctx, host[0, 6:8])

        pf.prefetch_token_chunk(host, req, start, end, ctx)
        hint_rows = pf.drain()[1]  # drain() waits for the worker thread
        assert len(seen) == 1, "lookahead did not compute ids for the next chunk"
        assert hint_rows > 0, "ids computed but nothing advised"

    tokens = torch.from_numpy(host[0, start:end].astype(np.int64))
    qsl = torch.tensor([0, end - start], dtype=torch.long)
    expected = _ids(tokens, qsl, torch.from_numpy(ctx).reshape(1, -1), end - start)
    assert torch.equal(seen[0], expected), "lookahead ids drifted from the model path"

    # Red control: the same call with the wrong context must differ, otherwise
    # this test would pass no matter what the planner did.
    wrong = _ids(tokens, qsl, torch.zeros(1, CTX_LEN, dtype=torch.long), end - start)
    assert not torch.equal(seen[0], wrong), "test is not sensitive to the context"


def test_prompt_front_is_padded_with_eos_like_the_model_path():
    """A chunk that starts inside the padding gets its context filled with eos."""
    host = np.arange(32, dtype=np.int32).reshape(1, 32)
    # start=1 with context_len=2: only token 0 exists before the chunk, so the
    # slot in front of it must be eos -- exactly how _prepare_ngram_context
    # fills the front of a prompt.
    ((req, start, end, ctx),) = _InputBatch([0], [1], [0], [10]).plan(host)
    assert (req, start, end) == (0, 1, 10)
    np.testing.assert_array_equal(ctx, [EOS, 0])
    # A chunk that starts deep in the prompt takes both context tokens.
    ((_req, _start, _end, ctx),) = _InputBatch([0], [4], [0], [10]).plan(host)
    np.testing.assert_array_equal(ctx, [2, 3])


def test_next_chunk_starts_at_computed_plus_scheduled():
    """The chunk in flight is not the next one.

    Off by one here and the read-ahead warms the pages that are already being
    read -- which stay in the cache anyway -- and misses the ones that matter.
    """
    host = np.arange(64, dtype=np.int32).reshape(1, 64)
    ((_req, start, end, _ctx),) = _InputBatch([0], [8], [16], [40]).plan(host)
    assert (start, end) == (24, 40)


def test_decode_and_finished_requests_are_skipped():
    """Nothing to warm once the prompt is exhausted -- decode must be a no-op."""
    host = np.arange(64, dtype=np.int32).reshape(2, 32)
    # computed + scheduled == prefill_len: the prompt is done.
    assert _InputBatch([0], [4], [24], [28]).plan(host) == []
    assert _InputBatch([0, 1], [4, 1], [24, 31], [28, 32]).plan(host) == []


def test_placeholder_request_indices_are_ignored():
    """Dummy batches used by graph capture hold placeholder indices."""
    host = np.arange(32, dtype=np.int32).reshape(1, 32)
    assert _InputBatch([7], [4], [0], [16]).plan(host) == []
    assert _InputBatch([-1], [4], [0], [16]).plan(host) == []


def test_lookahead_never_runs_past_the_host_row_width():
    """prefill_len is clamped to the all_token_ids width, not trusted."""
    host = np.arange(32, dtype=np.int32).reshape(1, 32)
    ((_req, _start, end, _ctx),) = _InputBatch([0], [4], [0], [9999]).plan(host)
    assert end == 32


# ---------------------------------------------------------------------------
# the maps arithmetic
def test_advised_bytes_are_the_real_file_offsets_of_the_real_rows(
    tmp_path, monkeypatch
):
    """``data_ptr`` -> file offset, asserted on a genuinely file-backed mapping.

    The offsets come from ``/proc/self/maps``, so this also fails loudly if the
    loader ever returns a heap copy instead of a view: no file VMA, no hints.

    Each shard is mapped at a *different* page-aligned offset of one file, the
    way real shards sit at different offsets in a safetensors file. Mapping them
    all at offset 0 would make a wrong shard index indistinguishable.
    """
    rows = 2048
    row_bytes = DIM * 2
    shard_bytes = rows * row_bytes  # 32 KiB = 8 pages per shard
    total = shard_bytes * NUM_SHARDS
    path = tmp_path / "table.bin"
    path.write_bytes(np.zeros(total, dtype=np.float16).tobytes())

    spy = _SpyLibc()
    monkeypatch.setattr(ple_prefetch, "_libc", lambda: spy)

    tensors, handles = [], []
    try:
        for shard in range(NUM_SHARDS):
            # The fd only needs to live long enough to create the mapping.
            with open(path, "rb") as fh:
                mm = mmap.mmap(
                    fh.fileno(),
                    shard_bytes,
                    offset=shard * shard_bytes,
                    access=mmap.ACCESS_READ,
                )
            handles.append(mm)
            tensors.append(
                torch.frombuffer(mm, dtype=torch.float16, count=rows * DIM).reshape(
                    rows, DIM
                )
            )
        emb = _embedding(tensors, rows=rows)
        pf = _prefetcher(emb)
        host = np.zeros((1, 8), dtype=np.int32)
        ctx = np.full(CTX_LEN, EOS, dtype=np.int64)
        pf.prefetch_token_chunk(host, 0, 0, 8, ctx)
        pf.drain()

        assert spy.calls, "no hints issued on a file-backed shard"
        for _fd, off, length, advice in spy.calls:
            assert advice == ple_prefetch.POSIX_FADV_WILLNEED
            assert off % 4096 == 0, "WILLNEED must start on a page boundary"
            assert length % 4096 == 0, "WILLNEED must be a whole number of pages"

        # Independently recompute which pages the ids occupy and require that
        # every one of them was covered.
        ids = _ids(
            torch.zeros(8, dtype=torch.long),
            torch.tensor([0, 8], dtype=torch.long),
            torch.full((1, CTX_LEN), EOS, dtype=torch.long),
            8,
        ).reshape(-1)
        want: set[int] = set()
        for value in ids.tolist():
            assert 0 <= value < NUM_SHARDS * rows
            shard, local = divmod(value, rows)
            want.add((shard * shard_bytes + local * row_bytes) // 4096)
        have: set[int] = set()
        for _fd, off, length, _advice in spy.calls:
            have.update(range(off // 4096, (off + length) // 4096))
        assert want <= have, f"rows never advised: {sorted(want - have)}"
    finally:
        tensors.clear()  # drop the buffer exports before unmapping
        for mm in handles:
            mm.close()


def test_unresolvable_shards_disable_the_prefetcher_instead_of_guessing():
    """Plain heap tensors mean the table is no longer mmap-backed.

    Advising the wrong bytes is worse than advising none, so the prefetcher
    turns itself off and says why.
    """
    emb = _embedding()  # heap tensors
    pf = _prefetcher(emb)
    host = np.zeros((1, 8), dtype=np.int32)
    pf.prefetch_token_chunk(host, 0, 0, 8, np.full(CTX_LEN, EOS, dtype=np.int64))
    assert pf.disabled_reason is not None
    assert "file-backed" in pf.disabled_reason


def test_hint_failure_disables_and_never_reaches_the_model_path(tmp_path, monkeypatch):
    """An EIO on the hint path costs the optimization, not the request."""
    spy = _SpyLibc(raises=True)
    monkeypatch.setattr(ple_prefetch, "_libc", lambda: spy)
    # File-backed shards: the hint path only engages on a real mapping, so a
    # heap tensor would disable the prefetcher before the worker ever ran.
    with _file_backed(tmp_path) as emb:
        pf = _prefetcher(emb)
        host = np.zeros((1, 8), dtype=np.int32)
        pf.prefetch_token_chunk(host, 0, 0, 8, np.full(CTX_LEN, EOS, dtype=np.int64))
        pf.drain()  # the exception happens on the worker thread
        assert pf.disabled_reason is not None
        assert "hint worker failed" in pf.disabled_reason
        # A second call is a no-op rather than another crash.
        pf.prefetch_token_chunk(host, 0, 0, 8, np.full(CTX_LEN, EOS, dtype=np.int64))
        pf.drain()
        assert "hint worker failed" in pf.disabled_reason


def test_out_of_range_ids_advertise_nothing():
    """Rows outside the table must not become offsets into an unrelated file."""
    emb = _embedding()
    pf = _prefetcher(emb)
    # Bypass the id function so we can feed the row space directly.
    pf._ranges = [None] * NUM_SHARDS
    bad = torch.tensor([[CAP * NUM_SHARDS + 10_000]], dtype=torch.int64)
    pf._advise(bad)
    assert pf.stats() == (0, 0)


def test_rows_are_merged_into_runs_instead_of_one_syscall_per_row():
    """A chunk's worth of rows must not cost one syscall per row."""
    emb = _embedding()
    pf = _prefetcher(emb)
    calls: list[tuple[int, int, int, int]] = []

    class Spy:
        def posix_fadvise(self, fd, off, length, advice):
            calls.append((fd.value, off.value, length.value, advice.value))
            return 0

    ple_prefetch._libc = lambda: Spy()
    try:
        # One synthetic shard whose rows are exactly one page each, so the run
        # math is visible without depending on the fake table's dimensions.
        pf._ranges = [(7, 0, 4096)] + [None] * (NUM_SHARDS - 1)
        rows = torch.arange(10, dtype=torch.int64).reshape(-1, 1)
        pf._advise(rows)
        assert len(calls) == 1, f"expected one merged run, got {len(calls)}"
        assert calls[0][1] == 0 and calls[0][2] == 4096 * 10

        # A gap wider than a row must split the run -- merging past it would
        # read pages nobody asked for.
        calls.clear()
        rows = torch.cat([torch.arange(10), torch.arange(20, 25)]).reshape(-1, 1)
        pf._advise(rows)
        assert [(off, length) for _f, off, length, _a in calls] == [
            (0, 4096 * 10),
            (4096 * 20, 4096 * 5),
        ]
    finally:
        ple_prefetch._libc = _REAL_LIBC


class _SpyLogger:
    """Capture what the prefetcher logs, without depending on vllm's logger."""

    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warns: list[str] = []

    @staticmethod
    def _fmt(msg: str, args: tuple) -> str:
        return msg % args if args else msg

    def info(self, msg, *args) -> None:
        self.infos.append(self._fmt(msg, args))

    def warning(self, msg, *args) -> None:
        self.warns.append(self._fmt(msg, args))

    def info_once(self, msg, *args) -> None:  # pragma: no cover - not used
        self.info(msg, *args)


def test_the_success_line_appears_once_per_process(tmp_path, monkeypatch):
    """One line per process, not one line per distinct chunk size.

    The first version used `logger.info_once`, which dedupes on
    (message, args): because the token count is an argument, every new chunk
    size printed another line. Boot D emitted seven in three minutes and was
    on track for one per distinct chunk size for the life of the process. The
    line is the only in-record signal that the read-ahead is live, so it has
    to stay a signal and not become traffic.

    The guard is exercised directly with three different token counts, because
    driving it through `prefetch_token_chunk` would need a host row wide enough
    for every size -- an earlier version of this test did that and quietly made
    two of its three calls no-ops, which is how it passed a red control it
    should have failed.
    """
    log = _SpyLogger()
    monkeypatch.setattr(ple_prefetch, "logger", log)
    with _file_backed(tmp_path) as emb:
        pf = _prefetcher(emb)
        for n in (8, 64, 4096):
            pf._log_first_hint(n)
    assert sum("read-ahead active" in m for m in log.infos) == 1, (
        "the success line must not repeat once per distinct chunk size"
    )


def test_the_success_line_reports_the_resolved_shard_counts(tmp_path, monkeypatch):
    """The line that proves the read-ahead is live must not print 0/0.

    Resolution happens inside the first `prefetch_token_chunk`, so the counts
    are only meaningful once a hint has actually been planned; printing them
    before that would claim a working read-ahead over an empty shard table.
    """
    log = _SpyLogger()
    spy = _SpyLibc()
    monkeypatch.setattr(ple_prefetch, "logger", log)
    monkeypatch.setattr(ple_prefetch, "_libc", lambda: spy)
    with _file_backed(tmp_path) as emb:
        pf = _prefetcher(emb)
        host = np.zeros((1, 8), dtype=np.int32)
        ctx = np.full(CTX_LEN, EOS, dtype=np.int64)
        pf.prefetch_token_chunk(host, 0, 0, 8, ctx)
        pf.drain()
        assert spy.calls, "prefetch must have run for the line to mean anything"
        active = [m for m in log.infos if "read-ahead active" in m]
        assert len(active) == 1, f"expected one line, got {len(active)}"
        assert "%d" not in active[0], "the line must be formatted, not a raw template"
        assert f"{NUM_SHARDS}/{NUM_SHARDS}" in active[0], active[0]


def test_a_working_read_ahead_reports_its_real_shard_counts(tmp_path, monkeypatch):
    """The one success log line must state counts that resolution produced.

    Until boot C the *only* sign that this feature worked at all was a thread
    named "ple-prefetch" -- and Python 3.12 does not put threading thread names
    in /proc/<pid>/task/*/comm, so that signal did not exist (docs
    prefill-io-wall.md sec 3e: the boot was nearly written off as inert). The
    once-per-process log line is the on-the-record signal now, so the shard
    counts it prints have to be the ones _resolve_shards actually found.
    """
    spy = _SpyLibc()
    monkeypatch.setattr(ple_prefetch, "_libc", lambda: spy)
    with _file_backed(tmp_path) as emb:
        pf = _prefetcher(emb)
        host = np.zeros((1, 8), dtype=np.int32)
        ctx = np.full(CTX_LEN, EOS, dtype=np.int64)
        assert pf._shards_total == 0, "counts must not be set before resolution"
        pf.prefetch_token_chunk(host, 0, 0, 8, ctx)
        pf.drain()
        assert spy.calls, "no hints issued on a file-backed shard"
        assert pf._shards_total == NUM_SHARDS
        assert pf._shards_usable == NUM_SHARDS, "every shard here is file-backed"
        hints, rows = pf.stats()
        assert hints == len(spy.calls) and rows > 0
