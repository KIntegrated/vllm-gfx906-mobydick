# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Behaviour of the per-region step timer (LAB-6 / #50).

The timer's contract is narrow but load-bearing for the fork's measurement
discipline, so each property that an A/B depends on is asserted here: it is inert
when disabled, it attributes time to the right region when enabled, it survives an
exception inside a region, and it reports the unattributed remainder rather than
presenting a partial budget as complete.
"""

import time

import pytest

from vllm import gfx906_step_timer as st


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("VLLM_GFX906_STEP_TIMER", raising=False)
    monkeypatch.delenv("VLLM_GFX906_STEP_TIMER_EVERY", raising=False)
    st.TIMER.reset()
    yield
    monkeypatch.undo()
    st.TIMER.reset()


def test_disabled_is_inert():
    assert not st.enabled()
    with st.region("prep_inputs"):
        pass
    with st.step():
        pass
    assert st.TIMER.sums == {}
    assert st.TIMER.steps == 0


def test_enabled_attributes_time(monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_STEP_TIMER", "1")
    st.TIMER.reset()
    assert st.enabled()

    with st.region("prep_inputs"):
        time.sleep(0.01)
    with st.region("gen_draft"):
        pass

    assert st.TIMER.calls["prep_inputs"] == 1
    assert st.TIMER.sums["prep_inputs"] >= 0.01
    assert st.TIMER.calls["gen_draft"] == 1


def test_region_records_on_exception(monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_STEP_TIMER", "1")
    st.TIMER.reset()

    with pytest.raises(RuntimeError):
        with st.region("attn_metadata"):
            raise RuntimeError("boom")

    assert st.TIMER.calls["attn_metadata"] == 1


def test_step_counts_and_reports_unaccounted(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_GFX906_STEP_TIMER", "1")
    monkeypatch.setenv("VLLM_GFX906_STEP_TIMER_EVERY", "2")
    st.TIMER.reset()

    with caplog.at_level("INFO"):
        for _ in range(2):
            with st.step():
                with st.region("prep_inputs"):
                    time.sleep(0.002)

    assert st.TIMER.steps == 2
    reports = [r for r in caplog.records if "gfx906 steptimer" in r.getMessage()]
    assert reports, "expected a steptimer report line"
    message = reports[-1].getMessage()
    assert "prep_inputs=" in message
    # The remainder must be reported, not silently dropped: a budget that does
    # not add up has to say so, otherwise a missing hook looks like idle time.
    assert "unaccounted=" in message


def test_report_restarts_the_window(monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_STEP_TIMER", "1")
    st.TIMER.reset()
    with st.step():
        with st.region("prep_inputs"):
            pass
    st.TIMER.report()
    assert st.TIMER.sums == {}
    assert st.TIMER.calls == {}
    assert st.TIMER.steps == 1


def test_disabled_overhead_is_a_branch():
    """Ten thousand disabled regions must cost microseconds, not milliseconds."""
    start = time.perf_counter()
    for _ in range(10_000):
        with st.region("prep_inputs"):
            pass
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"disabled region cost {elapsed / 10_000 * 1e6:.2f} us each"
