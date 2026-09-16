# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The gfx906 fallback guard: a lost custom FA must never be silent."""

from types import SimpleNamespace

import pytest

from vllm.platforms import rocm
from vllm.v1.attention.backends.registry import AttentionBackendEnum

CUSTOM = AttentionBackendEnum.CUSTOM


class _LoggerStub:
    def __init__(self):
        self.warnings = []

    def warning_once(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


@pytest.fixture
def gfx906(monkeypatch):
    monkeypatch.setattr(rocm, "on_gfx906", lambda: True)
    stub = _LoggerStub()
    monkeypatch.setattr(rocm, "logger", stub)
    return stub


def test_warns_when_custom_is_rejected(gfx906, monkeypatch):
    monkeypatch.delenv("VLLM_GFX906_FA_STRICT", raising=False)
    rocm._guard_gfx906_fa_fallback(
        SimpleNamespace(attn_type="decoder"),
        {CUSTOM: ["attention sinks not supported"]},
        AttentionBackendEnum.TRITON_ATTN,
    )
    assert len(gfx906.warnings) == 1
    assert "TRITON_ATTN" in gfx906.warnings[0]
    assert "attention sinks not supported" in gfx906.warnings[0]


def test_strict_mode_fails_closed(gfx906, monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_FA_STRICT", "1")
    with pytest.raises(RuntimeError, match="custom gfx906 FA is unavailable"):
        rocm._guard_gfx906_fa_fallback(
            SimpleNamespace(attn_type="decoder"),
            {CUSTOM: ["non-causal attention not supported"]},
            AttentionBackendEnum.ROCM_ATTN,
        )


def test_silent_when_custom_was_not_a_candidate(gfx906, monkeypatch):
    monkeypatch.delenv("VLLM_GFX906_FA_STRICT", raising=False)
    rocm._guard_gfx906_fa_fallback(
        SimpleNamespace(attn_type="encoder"),
        {AttentionBackendEnum.TURBOQUANT: ["kv_cache_dtype not supported"]},
        AttentionBackendEnum.ROCM_ATTN,
    )
    assert gfx906.warnings == []


def test_silent_off_gfx906(monkeypatch):
    monkeypatch.setattr(rocm, "on_gfx906", lambda: False)
    stub = _LoggerStub()
    monkeypatch.setattr(rocm, "logger", stub)
    monkeypatch.setenv("VLLM_GFX906_FA_STRICT", "1")
    rocm._guard_gfx906_fa_fallback(
        SimpleNamespace(attn_type="decoder"),
        {CUSTOM: ["attention sinks not supported"]},
        AttentionBackendEnum.TRITON_ATTN,
    )
    assert stub.warnings == []
