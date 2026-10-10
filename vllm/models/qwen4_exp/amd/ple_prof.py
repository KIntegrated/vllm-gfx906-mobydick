# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side measurement brackets for the PLE n-gram lookup.

Why this exists
---------------
`qwen4_exp_amd_ple_ngram_embedding` (in `ple_layer.py`) is one custom op that
performs four host-visible steps in a row:

1. **drain** -- `pinned_ids.copy_(ngram_ids, non_blocking=False)`. A blocking
   device-to-host copy, so it also waits for everything queued ahead of it on
   the compute stream. This is the step that turns GPU backlog into host stall,
   and it is the reason this lookup cannot be moved off the critical path simply
   by making the *runner* asynchronous.
2. **gather** -- the fixed 128-shard `index_select` loop over the mmap'd table.
   Host work, page-fault driven, 5 KB of table read per row.
3. **memcpy** -- `pinned_out.copy_(result)`, a host-to-host copy of the gathered
   rows (~21 MB for a 4096-token chunk).
4. **H2D issue** -- `output.copy_(pinned_out, non_blocking=True)`. Only the
   *issue* cost is visible to the host; completion is stream-ordered.

Steps 2-4 are exactly what a look-ahead prefetch (plan step D2) would overlap,
and step 1 is what host-side id computation (step D1) would remove. Whether
either is worth building was last quantified when NVMe read-ahead dominated
everything else (2.346 ms/token prefill), a number overtaken twice since. This
module exists so that decision uses this configuration's numbers instead.

Design rules
------------
* **Off by default.** `VLLM_GFX906_PLE_PROF=1` opts in. With it unset, the only
  extra work on the hot path is one cached boolean test per lookup.
* **No device-side objects.** No CUDA events, no device `synchronize()` call.
  Adding a sync to measure a sync would change the thing being measured, and
  event recording during graph capture is a hazard we do not need. Everything
  here is `perf_counter` and `getrusage`, which is what a blocking copy's cost
  is actually made of.
* **Windowed aggregates, not per-call logs.** 48 layers x thousands of steps is
  a log nobody can read. One line per `VLLM_GFX906_PLE_PROF_EVERY` lookups
  (default 256) or per `VLLM_GFX906_PLE_PROF_SECS` seconds (default 30),
  whichever comes first; the window then resets.
* **Never the reason a boot fails.** Every entry point swallows its own errors
  and degrades to "no data", because this rides along with production serving.

Reading the output
------------------
`wall=` is this process's PLE host-visible time over the window's wall clock. It
dilutes when the server is idle, so read it from a benchmark run, not a quiet
box. `majflt` is the number that says whether the table is coming from page cache
or off disk. Each TP rank profiles its own process, so expect one line per rank.
"""

import os
import resource
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_PLE_PROF = "VLLM_GFX906_PLE_PROF"
ENV_PLE_PROF_EVERY = "VLLM_GFX906_PLE_PROF_EVERY"
ENV_PLE_PROF_SECS = "VLLM_GFX906_PLE_PROF_SECS"

_DEFAULT_EVERY = 256
_DEFAULT_SECS = 30.0

_ENABLED: bool | None = None


def prof_enabled() -> bool:
    """Is PLE profiling requested? Cached -- this runs on the hot path."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get(ENV_PLE_PROF, "0") == "1"
        if _ENABLED:
            logger.info(
                "PLE profiler active (%s=1): aggregating drain / gather / memcpy / "
                "H2D-issue time per lookup. Costs a few perf_counter calls and one "
                "getrusage pair per lookup; model output is unaffected.",
                ENV_PLE_PROF,
            )
    return _ENABLED


def reset_enabled_cache() -> None:
    """Forget the cached env read. Test-only."""
    global _ENABLED
    _ENABLED = None


def _env_number(name: str, default, cast):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        logger.warning(
            "PLE profiler: %s=%r is not a valid %s; using %r instead.",
            name,
            raw,
            getattr(cast, "__name__", "number"),
            default,
        )
        return default


def page_faults() -> tuple[int, int]:
    """(minor, major) page faults of this process, or (0, 0) if unavailable."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(usage.ru_minflt), int(usage.ru_majflt)
    except Exception:  # pragma: no cover - defensive, see the module docstring
        return 0, 0


class PleLookupWindow:
    """Aggregates PLE lookups over a window, logs one line, then resets.

    The window closes on a lookup count *or* an age, whichever comes first. The
    age trigger matters at low batch: at 3 concurrent requests a count-only
    window can sit unreported for minutes and be mistaken for "the lookup is not
    running", which is a false negative on exactly the question being asked.
    """

    def __init__(
        self, every: int | None = None, interval_s: float | None = None
    ) -> None:
        self.every = (
            every
            if every is not None
            else _env_number(ENV_PLE_PROF_EVERY, _DEFAULT_EVERY, int)
        )
        self.interval_s = (
            interval_s
            if interval_s is not None
            else _env_number(ENV_PLE_PROF_SECS, _DEFAULT_SECS, float)
        )
        self.reset()

    def reset(self) -> None:
        """Open a fresh window."""
        self.calls = 0
        self.rows = 0
        self.bytes_staged = 0
        self.shard_lookups = 0
        self.drain_s = 0.0
        self.gather_s = 0.0
        self.memcpy_s = 0.0
        self.h2d_s = 0.0
        self.max_drain_s = 0.0
        self.max_gather_s = 0.0
        self.max_memcpy_s = 0.0
        self.max_h2d_s = 0.0
        self.minflt = 0
        self.majflt = 0
        self._t0 = time.perf_counter()

    def record(
        self,
        *,
        drain_s: float = 0.0,
        gather_s: float = 0.0,
        memcpy_s: float = 0.0,
        h2d_s: float = 0.0,
        rows: int = 0,
        bytes_staged: int = 0,
        shards_touched: int = 0,
        minflt_delta: int = 0,
        majflt_delta: int = 0,
    ) -> None:
        """Add one lookup, then close the window if it is due.

        Keyword-only and duck-typed on purpose: the caller in `ple_layer.py`
        measures what it can and passes what it has.
        """
        try:
            self.calls += 1
            self.rows += rows
            self.bytes_staged += bytes_staged
            self.shard_lookups += shards_touched
            self.drain_s += drain_s
            self.gather_s += gather_s
            self.memcpy_s += memcpy_s
            self.h2d_s += h2d_s
            self.max_drain_s = max(self.max_drain_s, drain_s)
            self.max_gather_s = max(self.max_gather_s, gather_s)
            self.max_memcpy_s = max(self.max_memcpy_s, memcpy_s)
            self.max_h2d_s = max(self.max_h2d_s, h2d_s)
            self.minflt += minflt_delta
            self.majflt += majflt_delta
            self.maybe_dump()
        except Exception as exc:  # pragma: no cover
            logger.debug("PLE profiler: record() ignored an error: %r", exc)

    def summary(self, elapsed_s: float) -> str:
        """One readable line for the window that is about to close."""
        n = max(self.calls, 1)
        host_s = self.drain_s + self.gather_s + self.memcpy_s + self.h2d_s
        wall_pct = 100.0 * host_s / elapsed_s if elapsed_s > 0 else 0.0
        per_row_ms = 1000.0 * host_s / self.rows if self.rows else 0.0
        return (
            "PLE prof: {calls} lookups, {rows} rows, {mib:.1f} MiB staged | "
            "drain {d_avg:.3f}/{d_max:.3f} ms  gather {g_avg:.3f}/{g_max:.3f} ms  "
            "memcpy {m_avg:.3f}/{m_max:.3f} ms  h2d-issue {h_avg:.3f}/{h_max:.3f} ms "
            "(avg/max) | host-visible {host:.1f} ms = {pct:.1f}% of {wall:.2f}s | "
            "{per_row:.4f} ms/row, {shards:.1f} shards/lookup, "
            "faults +{minflt} minor / +{majflt} major"
        ).format(
            calls=self.calls,
            rows=self.rows,
            mib=self.bytes_staged / (1024.0 * 1024.0),
            d_avg=1000.0 * self.drain_s / n,
            d_max=1000.0 * self.max_drain_s,
            g_avg=1000.0 * self.gather_s / n,
            g_max=1000.0 * self.max_gather_s,
            m_avg=1000.0 * self.memcpy_s / n,
            m_max=1000.0 * self.max_memcpy_s,
            h_avg=1000.0 * self.h2d_s / n,
            h_max=1000.0 * self.max_h2d_s,
            host=1000.0 * host_s,
            pct=wall_pct,
            wall=elapsed_s,
            per_row=per_row_ms,
            shards=self.shard_lookups / n,
            minflt=self.minflt,
            majflt=self.majflt,
        )

    def maybe_dump(self, force: bool = False) -> bool:
        """Log and reset if this window is due. Returns whether it did."""
        if self.calls == 0:
            return False
        elapsed = time.perf_counter() - self._t0
        if not force and self.calls < self.every and elapsed < self.interval_s:
            return False
        try:
            logger.info("%s", self.summary(elapsed))
        except Exception as exc:  # pragma: no cover
            logger.debug("PLE profiler: summary() ignored an error: %r", exc)
        finally:
            self.reset()
        return True


_window: PleLookupWindow | None = None


def window() -> PleLookupWindow:
    """The process-wide window, built on first use (env is set by then)."""
    global _window
    if _window is None:
        _window = PleLookupWindow()
    return _window


def reset_window_for_tests() -> None:
    """Drop the process-wide window. Test-only."""
    global _window
    _window = None
