# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the PLE lookup profiler (`VLLM_GFX906_PLE_PROF`).

The profiler exists to answer one question with numbers from the current
configuration: how much of a step's wall clock is the PLE lookup's host-visible
time. Two properties matter more than the numbers themselves, and they are what
this file actually tests:

* **it is off unless asked for**, and being off means the instrumented function
  executes the same statements it did before the profiler existed;
* **it cannot be the reason a step fails** -- every entry point swallows its own
  errors, so a bug in measurement degrades to "no data", never to a dead engine.

The strongest test here is the red control at the bottom: it drives a real
`MmapShardedNGramEmbedding.forward()` and asserts the instrumentation actually
observed the gather. A measurement that never fires reads identical to a
measurement that reports zero, so the probe has to prove it ran.
"""

import pytest
import torch

ple_prof = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_prof",
    reason="qwen4_exp AMD model code is not importable in this build",
)


@pytest.fixture(autouse=True)
def _pristine_env(monkeypatch):
    for name in (
        ple_prof.ENV_PLE_PROF,
        ple_prof.ENV_PLE_PROF_EVERY,
        ple_prof.ENV_PLE_PROF_SECS,
    ):
        monkeypatch.delenv(name, raising=False)
    ple_prof.reset_enabled_cache()
    ple_prof.reset_window_for_tests()
    yield
    ple_prof.reset_enabled_cache()
    ple_prof.reset_window_for_tests()


def test_off_by_default():
    assert ple_prof.prof_enabled() is False


def test_on_when_env_says_so(monkeypatch):
    monkeypatch.setenv(ple_prof.ENV_PLE_PROF, "1")
    assert ple_prof.prof_enabled() is True


def test_any_other_value_is_off(monkeypatch):
    # "1" only. An env knob that turns on for "yes"/"true"/"0" is a knob that
    # will be set wrong at 3 a.m. during an incident.
    monkeypatch.setenv(ple_prof.ENV_PLE_PROF, "0")
    assert ple_prof.prof_enabled() is False
    monkeypatch.setenv(ple_prof.ENV_PLE_PROF, "true")
    ple_prof.reset_enabled_cache()
    assert ple_prof.prof_enabled() is False


def test_garbage_window_config_falls_back(monkeypatch, caplog):
    monkeypatch.setenv(ple_prof.ENV_PLE_PROF_EVERY, "not-a-number")
    window = ple_prof.PleLookupWindow()
    assert window.every == ple_prof._DEFAULT_EVERY
    assert "not a valid" in caplog.text


def test_window_closes_on_count(caplog):
    # record() flushes the window itself when it is due -- a caller must not
    # have to remember to poll, because a hot path that has to remember is a
    # hot path that eventually forgets.
    window = ple_prof.PleLookupWindow(every=2, interval_s=10_000)
    window.record(rows=1)
    assert window.calls == 1  # 1 of 2, not due
    assert window.maybe_dump() is False
    window.record(rows=2)
    assert window.calls == 0  # flushed and reset
    assert "PLE prof: 2 lookups, 3 rows" in caplog.text


def test_window_closes_on_age_even_with_one_lookup(caplog):
    # At 3 concurrent requests a count-only window can starve, and a starved
    # window is indistinguishable from "the lookup never ran" -- the exact false
    # negative this profiler would be blamed for.
    window = ple_prof.PleLookupWindow(every=10_000, interval_s=0.0)
    window.record(rows=1)
    assert window.calls == 0
    assert "PLE prof: 1 lookups" in caplog.text


def test_empty_window_never_logs():
    window = ple_prof.PleLookupWindow(every=1, interval_s=0.0)
    assert window.maybe_dump() is False


def test_aggregates_averages_maxes_and_percentages():
    window = ple_prof.PleLookupWindow(every=1_000, interval_s=1_000)
    window.record(drain_s=0.001, gather_s=0.002, rows=10, bytes_staged=2 << 20)
    window.record(drain_s=0.003, gather_s=0.004, rows=30, bytes_staged=2 << 20)
    assert window.calls == 2
    assert window.rows == 40
    assert window.max_drain_s == pytest.approx(0.003)
    line = window.summary(elapsed_s=1.0)
    # host-visible = 0.001+0.002+0.003+0.004 = 10 ms of a 1 s window.
    assert "10.0 ms" in line
    assert "1.0%" in line
    assert "40 rows" in line
    assert "4.0 MiB" in line
    # per-row = 10 ms / 40 rows = 0.25 ms
    assert "0.2500 ms/row" in line


def test_summary_survives_a_zero_length_window():
    window = ple_prof.PleLookupWindow()
    window.record(rows=0)
    assert "% of" in window.summary(elapsed_s=0.0)


def test_record_never_raises(monkeypatch):
    """Measurement must not be able to kill a step.

    `max()` on a string is the cheapest way to make the aggregation raise on
    purpose; the point is that `record()` is where the try/except lives, so the
    caller in `ple_layer.py` cannot be taken down by a bad measurement.
    """
    window = ple_prof.PleLookupWindow(every=1, interval_s=0.0)
    window.record(drain_s="not a number")  # must not raise
    assert window.calls == 1


def test_page_faults_returns_a_pair():
    minor, major = ple_prof.page_faults()
    assert minor >= 0 and major >= 0


def test_window_is_created_lazily_and_is_a_singleton():
    first = ple_prof.window()
    assert ple_prof.window() is first
    ple_prof.reset_window_for_tests()
    assert ple_prof.window() is not first


class TestGatherIsActuallyObserved:
    """Red control: prove the bracket sees a real gather.

    A profiler that is never reached produces the same log as one that reports
    zeros, so the instrumentation has to be shown firing on the code path that
    actually serves requests. `MmapShardedNGramEmbedding.forward` is the gather
    -- CPU-only, so it runs here without a GPU.
    """

    @staticmethod
    def _table():
        ple = pytest.importorskip(
            "vllm.models.qwen4_exp.amd.ple_layer",
            reason="qwen4_exp AMD model code is not importable in this build",
        )
        table = ple.MmapShardedNGramEmbedding(
            num_shards=3, shard_row_capacity=8, embedding_dim=4
        )
        for shard in range(3):
            table.set_shard(
                shard,
                torch.full((8, 4), float(shard), dtype=torch.float16),
            )
        return table

    def test_shard_count_recorded_when_profiling_on(self, monkeypatch):
        monkeypatch.setenv(ple_prof.ENV_PLE_PROF, "1")
        ple_prof.reset_enabled_cache()
        table = self._table()
        table.forward(torch.tensor([0, 5, 9, 23], dtype=torch.long))
        # ids 0,5 -> shard 0; 9 -> shard 1; 23 -> shard 2.
        assert table._last_shards_touched == 3

    def test_no_state_written_when_profiling_off(self, monkeypatch):
        monkeypatch.delenv(ple_prof.ENV_PLE_PROF, raising=False)
        ple_prof.reset_enabled_cache()
        table = self._table()
        out = table.forward(torch.tensor([0, 9], dtype=torch.long))
        assert not hasattr(table, "_last_shards_touched")
        # And the gather itself is unaffected by the knob.
        assert out.shape == (2, 4)
