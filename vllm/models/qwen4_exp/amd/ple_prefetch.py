# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-ahead for the PLE ngram table, off the critical path.

Why this exists
---------------
The ngram table is ~99 GiB of bf16 rows gathered at random out of an mmap'd
safetensors mapping, which is bigger than this node's page cache, so nearly
every gather is a cold random fault. The gather sits at PLE layer 2 of 48, i.e.
at the very bottom of the transformer stack, so almost nothing else is running
on the GPU while the host sits in ``index_select`` waiting on NVMe: a cold
prefill measured **0 % GPU busy for ~62 % of wall clock**
(``docs/prefill-io-wall.md``).

``MADV_RANDOM`` (``ple_layer.py``, ``VLLM_GFX906_PLE_MADV_RANDOM``) already
removes most of the *volume* by killing 128 KiB read-ahead windows on 320-byte
rows. What it cannot remove is the *latency*: 16-65k independent 4 KiB reads per
chunk, serialized behind the one thread that is also the critical path.

This module hides that latency by issuing ``POSIX_FADV_WILLNEED`` for the rows
the **next** chunk will touch, while the GPU is busy with the current one.

Why fadvise, and not a prefetch gather into a scratch buffer
-----------------------------------------------------------
Both were measured (``scripts/probe-ple-prefetch.py``, cold cache, rotated arm
order, real shard, 1024 tokens x 16 ids):

    arm                     best ms   MiB read    vs serial
    serial (today)             77.2       65.5        1.00x
    prefetch gather thread     76.5       65.5        1.01x   <- no win
    fadvise-ahead worker       66.0       66.1        1.17x

A gather-ahead thread does the same I/O twice and fights the main thread for
the GIL, and it was *slower* without MADV_RANDOM (0.89x). ``posix_fadvise`` is
an asynchronous hint: the issuing thread never blocks on the device, so it
never carries the GIL across an NVMe wait. That is the whole reason this module
issues hints instead of doing work.

Safety
------
``posix_fadvise`` is a hint about a *file*. It cannot change a computed value:
there is no cache of results here, no memo, no key to get wrong. The only ways
this module can hurt are wasted NVMe bandwidth (if the lookahead mispredicts)
and page-cache pressure. Both are bounded by the lookahead length, and both are
why this is opt-in.

Every failure -- an unresolvable mapping, an unexpected buffer layout, an
error inside the worker thread -- disables the prefetcher for the life of the
process after logging once, and never propagates to the caller. An engine
optimization must be able to refuse *itself*, but it must never refuse a boot.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import regex as re
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

POSIX_FADV_WILLNEED = 3
_PAGE = 4096

_MAPS_LINE = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+)\s+\S+\s+([0-9a-f]+)\s+\S+\s+\S+\s+(.*)$"
)


def prefetch_enabled() -> bool:
    """`VLLM_GFX906_PLE_PREFETCH=1` turns the read-ahead on. Off by default."""
    return os.environ.get("VLLM_GFX906_PLE_PREFETCH", "0") == "1"


def plan_lookahead_chunks(
    *,
    host_tokens: np.ndarray,
    idx_mapping: np.ndarray,
    scheduled: np.ndarray,
    computed: np.ndarray,
    prefill_len: np.ndarray,
    context_len: int,
    eos_token_id: int,
):
    """Yield ``(req_index, start, end, context)`` for each request's next chunk.

    Pure numpy in, numpy out -- deliberately kept apart from the model state so
    the arithmetic that decides *which tokens to warm* is unit-testable without
    a model, a device, or a scheduler.

    ``computed`` is the optimistic CPU mirror of already-prefilled tokens and
    ``scheduled`` the tokens claimed by the step in flight, so the next chunk
    starts at their sum and runs to the end of the prompt (``prefill_len``).
    Requests that are decoding, or whose prompt is already finished, yield
    nothing.

    ``context`` is built the way ``_prepare_ngram_context`` builds it: the
    ``context_len`` tokens immediately before ``start``, left-padded with eos
    when the chunk starts inside the padding.
    """
    max_req, max_pos = host_tokens.shape
    for row in range(len(scheduled)):
        req = int(idx_mapping[row])
        if req < 0 or req >= max_req:
            continue
        start = int(computed[row]) + int(scheduled[row])
        end = min(int(prefill_len[row]), max_pos)
        if start < 0 or start >= end:
            continue
        context = np.full(context_len, eos_token_id, dtype=np.int64)
        have = min(context_len, start)
        if have > 0:
            context[context_len - have :] = host_tokens[req, start - have : start]
        yield req, start, end, context


class PleRowPrefetcher:
    """Issue WILLNEED for the rows a future chunk of tokens will gather.

    Args:
        ngram_embedding: the ``MmapShardedNGramEmbedding`` whose shards will be
            read. Only used for ``_shards`` / ``shard_row_capacity`` /
            ``embedding_dim``; nothing is written to it.
        hash_buffers: the three ``Qwen4ExpNGramEmbedding`` hash buffers, in
            (multipliers, vocab_sizes, head_offsets) order. CPU copies are taken
            once; the lookahead computes ids on the host.
        id_fn: ``Qwen4ExpNGramEmbedding._compute_ngram_ids``, called with the
            CPU mirrors. Sharing that function with the device path is what
            keeps the lookahead honest.
        ngram_size / heads_per_ngram / eos_token_id: taken from the layer, not
            inferred. The id hash depends on the eos token (it marks segment
            boundaries in the shift math), so guessing it would compute ids that
            look plausible and match nothing -- wasted NVMe traffic is exactly
            the failure a read-ahead cannot show in its output.
        max_tokens: cap on lookahead tokens per call, so a 128k prompt cannot
            queue an unbounded hint storm.
    """

    def __init__(
        self,
        ngram_embedding,
        hash_buffers: Sequence[torch.Tensor],
        id_fn,
        *,
        ngram_size: int,
        heads_per_ngram: int,
        eos_token_id: int,
        max_tokens: int = 4096,
    ) -> None:
        self._emb = ngram_embedding
        self._id_fn = id_fn
        self._ngram_size = int(ngram_size)
        self._heads_per_ngram = int(heads_per_ngram)
        self._eos_token_id = int(eos_token_id)
        self._max_tokens = int(max_tokens)
        self._multipliers, self._vocab_sizes, self._head_offsets = (
            b.detach().to("cpu") for b in hash_buffers
        )
        # Reused host workspaces for the id computation (grown on demand).
        self._packed = torch.empty((1, self._max_tokens), dtype=torch.long)
        self._positions = torch.arange(self._max_tokens, dtype=torch.long)

        self._ranges: list[tuple[int, int, int] | None] | None = None
        self._fds: list[int] = []
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False
        self._disabled_reason: str | None = None
        self._hints = 0
        self._hint_rows = 0
        # Kept as ints so the first-hint log line below costs nothing per step
        # (`info_once` evaluates its arguments eagerly, and it is on the hot
        # path: it runs once per scheduled chunk per rank).
        self._shards_usable = 0
        self._shards_total = 0

    # ------------------------------------------------------------------ setup
    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = reason
            logger.warning(
                "PLE read-ahead disabled (%s); the gather stays on the critical "
                "path. This is a performance-only fallback, outputs are "
                "unaffected.",
                reason,
            )
            self.close()

    def _resolve_shards(self) -> bool:
        """Map each shard's rows to (fd, byte offset of row 0, row bytes).

        fadvise is keyed on a file and a file offset, but a loaded shard is only
        a pointer. The mapping from one to the other is ``/proc/self/maps``: the
        VMA that contains ``data_ptr`` carries the page-aligned file offset, so
        ``file_off = vma_off + (data_ptr - vma_start)`` is row 0's position in
        the file. This is also the check that the shard is *still* a view into
        the checkpoint file: a loader that starts returning copies yields no
        file-backed VMA, and the prefetcher turns itself off instead of advising
        the wrong bytes.
        """
        shards = self._emb._shards
        row_bytes0 = self._emb.embedding_dim
        try:
            with open("/proc/self/maps") as fh:
                maps = [m for line in fh if (m := _MAPS_LINE.match(line)) is not None]
        except OSError as exc:
            self.disable(f"cannot read /proc/self/maps ({exc})")
            return False

        ranges: list[tuple[int, int, int] | None] = []
        for idx, shard in enumerate(shards):
            if shard is None:
                # Not this rank's shard (the table is split across TP ranks);
                # the rank that owns it warms it. No rank-level gating here.
                ranges.append(None)
                continue
            if not shard.is_contiguous():
                ranges.append(None)
                continue
            addr = shard.data_ptr()
            hit = next(
                (
                    m
                    for m in maps
                    if int(m.group(1), 16) <= addr < int(m.group(2), 16)
                    and m.group(4).strip().startswith("/")
                ),
                None,
            )
            if hit is None:
                ranges.append(None)
                continue
            path = hit.group(4).strip()
            vma_start = int(hit.group(1), 16)
            byte_off = int(hit.group(3), 16) + (addr - vma_start)
            row_bytes = shard.stride(0) * shard.element_size()
            if row_bytes < row_bytes0 or byte_off < 0:
                ranges.append(None)
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                ranges.append(None)
                continue
            self._fds.append(fd)
            ranges.append((fd, byte_off, row_bytes))
        self._ranges = ranges
        usable = sum(r is not None for r in ranges)
        self._shards_usable, self._shards_total = usable, len(shards)
        if usable == 0:
            self.disable("no PLE shard resolves to a file-backed mapping")
            return False
        if usable < len(shards):
            logger.info(
                "PLE read-ahead: %d/%d shards resolve to file-backed mappings; "
                "the rest are skipped.",
                usable,
                len(shards),
            )
        return True

    # ------------------------------------------------------------- hint issue
    def _advise(self, ids: torch.Tensor) -> None:
        """Translate row ids into page runs and advise them. Runs on the worker."""
        assert self._ranges is not None
        cap = self._emb.shard_row_capacity
        flat = ids.reshape(-1).numpy()
        if flat.size == 0:
            return
        total_rows = self._emb.num_shards * cap
        if flat.min() < 0 or flat.max() >= total_rows:
            # Out-of-range rows would advise the wrong file offsets. The real
            # gather folds them onto row 0 and warns; here there is nothing to
            # warm, so just drop them.
            flat = flat[(flat >= 0) & (flat < total_rows)]
            if flat.size == 0:
                return
        shard_idx = flat // cap
        local = flat - shard_idx * cap
        order = np.argsort(shard_idx, kind="stable")
        shard_idx, local = shard_idx[order], local[order]
        bounds = np.searchsorted(shard_idx, np.arange(self._emb.num_shards + 1))
        n_runs = 0
        for shard in np.nonzero(np.diff(bounds))[0]:
            rng = self._ranges[int(shard)]
            if rng is None:
                continue
            fd, byte_off, row_bytes = rng
            rows = np.unique(local[bounds[shard] : bounds[shard + 1]])
            first = (byte_off + rows.astype(np.int64) * row_bytes) // _PAGE
            span = max(1, (row_bytes - 1) // _PAGE + 1)
            # Merge consecutive pages into one syscall per run.
            breaks = np.nonzero(np.diff(first) > span)[0]
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks + 1, first.size]
            for s, e in zip(starts, ends):
                begin = int(first[s])
                stop = int(first[e - 1]) + span
                libc = _libc()
                libc.posix_fadvise(
                    ctypes.c_int(fd),
                    ctypes.c_longlong(begin * _PAGE),
                    ctypes.c_longlong((stop - begin) * _PAGE),
                    ctypes.c_int(POSIX_FADV_WILLNEED),
                )
                n_runs += 1
        self._hints += n_runs
        self._hint_rows += int(flat.size)

    # ------------------------------------------------------------------ public
    def prefetch_token_chunk(
        self,
        token_ids: np.ndarray,
        req_index: int,
        start: int,
        end: int,
        context: np.ndarray,
    ) -> None:
        """Warm the rows for ``token_ids[req_index, start:end]``.

        ``context`` is the ``ngram_size - 1`` token ids immediately before
        ``start``; the real path derives it the same way
        (``Qwen4ExpModelState._prepare_ngram_context`` reads it from
        ``all_token_ids``), so the ids computed here match the ids the gather
        will compute later.
        """
        if self._disabled_reason is not None or self._closed:
            return
        if self._ranges is None and not self._resolve_shards():
            return
        end = min(int(end), start + self._max_tokens)
        n = end - int(start)
        if n <= 0:
            return
        try:
            if self._packed.shape[1] < n:
                self._packed = torch.empty((1, n), dtype=torch.long)
                self._positions = torch.arange(n, dtype=torch.long)
            ids_in = torch.from_numpy(
                np.ascontiguousarray(
                    token_ids[req_index, int(start) : end], dtype=np.int64
                )
            )
            ctx = torch.from_numpy(np.ascontiguousarray(context, dtype=np.int64))
            ctx = ctx.reshape(1, -1)
            query_start = torch.tensor([0, n], dtype=torch.long)
            ngram_ids = self._id_fn(
                ids_in,
                query_start,
                ctx,
                multipliers=self._multipliers,
                vocab_sizes=self._vocab_sizes,
                head_offsets=self._head_offsets,
                packed=self._packed[:, :n],
                positions=self._positions[:n],
                eos_token_id=self._eos_token_id,
                ngram_size=self._ngram_size,
                heads_per_ngram=self._heads_per_ngram,
            )
        except Exception as exc:  # noqa: BLE001 - never reach the caller
            self.disable(
                f"host-side id computation failed ({type(exc).__name__}: {exc})"
            )
            return

        # Hand the ids to a single worker thread. One thread because the hint
        # stream is a queue, not a fan-out, and because two threads would only
        # compete for the same NVMe queue.
        try:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="ple-prefetch"
                )
            self._pool.submit(self._advise_guarded, ngram_ids)
            # A read-ahead that works is otherwise invisible: it has no effect
            # on output, and until now no line in the log proved it was live.
            # (Verifying it meant a syscall tracepoint; see
            # docs/prefill-io-wall.md sec 3e.) Once per process.
            logger.info_once(
                "PLE read-ahead active: hinting %d tokens' embedding rows from "
                "%d/%d file-backed shards, one chunk ahead of the gather.",
                n,
                self._shards_usable,
                self._shards_total,
            )
        except Exception as exc:  # noqa: BLE001
            self.disable(f"prefetch worker unavailable ({exc})")

    def _advise_guarded(self, ids: torch.Tensor) -> None:
        """Worker entry point.

        An exception raised inside a submitted future is captured by the Future
        and silently discarded unless someone calls ``result()`` -- which would
        turn a real bug into a prefetch that quietly does nothing. Catch it here
        and say so instead.
        """
        try:
            self._advise(ids)
        except Exception as exc:  # noqa: BLE001
            self.disable(f"hint worker failed ({type(exc).__name__}: {exc})")

    def stats(self) -> tuple[int, int]:
        """(syscalls issued, rows advised) -- for the log line at shutdown."""
        return self._hints, self._hint_rows

    def drain(self) -> tuple[int, int]:
        """Wait for queued hints and return :meth:`stats`.

        Only for shutdown and tests: hints are fire-and-forget in the engine, so
        blocking on them is exactly what the design avoids. Without this, a test
        that inspects the issued hints would race the worker thread.
        """
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        return self.stats()

    def close(self) -> None:
        self._closed = True
        pool, self._pool = self._pool, None
        if pool is not None:
            # Do not wait: in-flight hints are hints, not work we depend on.
            pool.shutdown(wait=False, cancel_futures=True)
        fds, self._fds = self._fds, []
        for fd in fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._ranges = None


_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
        _LIBC.posix_fadvise.restype = ctypes.c_int
    return _LIBC
