# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Per-region ms/step budget instrumentation (LAB-6 / #50).

The fork's A/B discipline leads with **ms/step** because acceptance and t/s are
chaotic on this stack, but every spec-decode argument so far has had to infer the
step budget from totals (tokens/step x t/s) instead of measuring it in parts. This
module measures the parts.

Design constraints, both from #50's gate:

* **Graph-safe.** Only ``time.perf_counter()`` on the calling thread: no CUDA
  calls, no synchronisation, no allocations. Nothing here can perturb capture or
  replay, and a region that happens to be inside a captured graph still measures
  the *host* dispatch around it (which is what a replay costs on this stack --
  see the launch-overhead findings in DEVLOG-tp2-dense.md S7).
* **Opt-in and cheap.** ``VLLM_GFX906_STEP_TIMER=1`` enables it; disabled, each
  region is one ``if`` and a shared no-op context manager, so the overhead is a
  branch (~40 ns). Enabled, it is two perf_counter calls plus a float add per
  region (~1 us), i.e. <1 % of a ~40 ms MTP step with five regions.

Reporting: every ``VLLM_GFX906_STEP_TIMER_EVERY`` steps (default 100) a one-line
table is emitted at INFO with the mean ms per region, plus the unattributed
remainder against the measured step wall. The remainder is the point: if the
regions do not sum to the step, the missing time is somewhere nobody has hooked
yet.

Where the hooks live (2026-10-26, and this map is the point of the module):

* ``prep_inputs`` and ``attn_metadata`` are hooked in **both** runners -- the V1
  ``v1/worker/gpu_model_runner.py`` and the V2 ``v1/worker/gpu/model_runner.py``,
  which is the default since 0.30. Instrumenting only V1 produced a report where
  every region was zero and ``unaccounted`` was the whole step; the gate (region
  sum versus step wall) is what caught it.
* ``gen_draft`` is V1-only. In V2 drafting happens inside the speculator
  (``v1/worker/gpu/spec_decode/speculator.py:74`` ``propose``), which subclasses
  override, so a hook needs a wrapper per implementation rather than one call site.
* ``ctx_kv`` is not hooked: our FA backend has
  ``forward_includes_kv_cache_update = False`` and the write runs inside the
  compiled/captured model, where a Python hook would only ever fire in eager
  mode -- measuring it that way would understate production.

Typical use from a bench or a serving run::

    VLLM_GFX906_STEP_TIMER=1 VLLM_GFX906_STEP_TIMER_EVERY=50 vllm serve ...
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.logger import init_logger

logger = init_logger(__name__)

# Reporting order; anything a future hook adds is appended after these rather
# than silently dropped from the line.
_REGIONS = (
    "prep_inputs",
    "attn_metadata",
    "ctx_kv",
    "gen_draft",
)


def _flag(name: str, default: str) -> str:
    return os.environ.get(name, default)


class _StepTimer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sums: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.steps = 0
        self.last_report_step = 0
        self.enabled = _flag("VLLM_GFX906_STEP_TIMER", "0") == "1"
        try:
            self.every = max(1, int(_flag("VLLM_GFX906_STEP_TIMER_EVERY", "100")))
        except ValueError:
            self.every = 100

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sums[name] = self.sums.get(name, 0.0) + (time.perf_counter() - start)
            self.calls[name] = self.calls.get(name, 0) + 1

    @contextmanager
    def step(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with self.region("step_wall"):
            yield
        self.steps += 1
        if self.steps >= self.last_report_step + self.every:
            self.report()
            self.last_report_step = self.steps

    def report(self) -> None:
        if not self.enabled or self.steps == 0:
            return
        window = self.steps - self.last_report_step
        if window <= 0:
            return
        wall = self.sums.get("step_wall", 0.0) / window * 1e3
        accounted = 0.0
        parts = []
        names = [n for n in _REGIONS if n in self.sums]
        names += sorted(n for n in self.sums if n not in _REGIONS and n != "step_wall")
        for name in names:
            per_step = self.sums[name] / window * 1e3
            accounted += per_step
            parts.append(f"{name}={per_step:.2f}")
        parts.append(f"unaccounted={wall - accounted:.2f}")
        logger.info(
            "gfx906 steptimer: steps=%d wall=%.2f ms/step  %s",
            self.steps,
            wall,
            " ".join(parts),
        )
        # The window restarts so a long run keeps reporting recent behaviour.
        self.sums.clear()
        self.calls.clear()


TIMER = _StepTimer()


def enabled() -> bool:
    return TIMER.enabled


def region(name: str):
    """Time a named region. No-op unless ``VLLM_GFX906_STEP_TIMER=1``."""
    return TIMER.region(name)


def step():
    """Bracket one engine step; drives the periodic report."""
    return TIMER.step()
