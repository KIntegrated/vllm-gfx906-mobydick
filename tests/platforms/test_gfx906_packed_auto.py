# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The gfx906 packed-KV `auto` decision must key on *usable* capacity.

The hybrid aligner re-derives the scheduler block size from the (smaller) packed
attention page. Raw pool capacity is ``block_size / page_bytes``, but a request
holds ``ceil(max_model_len / block_size)`` blocks, so what a pool can actually
serve is ``pool / page * max_model_len / blocks_per_req`` — the quantity the
served "GPU KV cache size" reports, and the one that decides the default.

Geometry below is measured: ``docs/gfx906/_hybrid_kv_ratio.py`` for the pages and
blocks (it reproduces the boot logs), and a block-level in-process probe
(`num_blocks` + per-group pages) for Lightning's pool.
"""

import pytest

from vllm.platforms.rocm import RocmPlatform

gain = RocmPlatform._gfx906_packed_capacity_gain

# Measured geometries (block size, unified page bytes per block).
QWEN35B = ((1056, 2_162_688), (1376, 2_157_568))
LIGHTNING = ((2096, 2_146_304), (2672, 2_137_600))


def test_packed_wins_when_the_smaller_cell_also_needs_fewer_blocks_per_request():
    """Qwen3.5-35B-A3B at 16k: 16 blocks/request -> 12, i.e. the real win.

    Served A/B: 142,987 -> 174,762 KV tokens (+22.2 %).
    """
    assert gain(*QWEN35B, 16_384) == pytest.approx(0.336, abs=0.003)
    assert gain(*QWEN35B, 16_384) >= RocmPlatform._GFX906_PACKED_MIN_GAIN


def test_packed_loses_its_win_when_both_layouts_round_to_the_same_block_count():
    """Nemotron-3.5-Lightning-30B-A3B at 8k: ceil(8192/x) is 4 either way.

    The pool gains 28 % of raw tokens (423,392 -> 542,416 at the same 2.02 GiB)
    but a request still holds 4 blocks, so the usable gain is +0.4 % — the
    recorded served datum. Must stay on the alias layout.
    """
    assert gain(*LIGHTNING, 8_192) == pytest.approx(0.004, abs=0.001)
    assert gain(*LIGHTNING, 8_192) < RocmPlatform._GFX906_PACKED_MIN_GAIN


def test_same_model_can_flip_verdict_with_max_model_len():
    """At 16k the Lightning block rounding finally pays: 8 blocks -> 7."""
    assert gain(*LIGHTNING, 16_384) == pytest.approx(0.1475, abs=0.001)
    assert gain(*LIGHTNING, 16_384) >= RocmPlatform._GFX906_PACKED_MIN_GAIN


def test_packed_loses_when_the_page_grows_into_the_same_block_count():
    """A bigger page at an unchanged block count is a straight loss."""
    assert gain((16, 131_072), (16, 147_456), 8_192) < 0


def test_no_unified_page_predicts_nothing_and_stays_off():
    assert gain((16, None), (16, None), 8_192) == 0.0
    assert gain((16, 131_072), (16, None), 8_192) == 0.0


def test_explicit_packed_on_warns_when_gfx906_is_not_detected(
    monkeypatch, caplog
):
    """Fail-open guard: `--gfx906-fa-packed on` must never be dropped silently.

    Measured 2026-10-19: on an amdsmi-degraded boot the flag served the alias KV
    layout with no message from the packed layer, because the platform's publish
    step is gated on `on_gfx906()` and the backend then resolves "off" by default.
    """
    import logging
    from unittest.mock import MagicMock

    from vllm.platforms import rocm as rocm_mod

    monkeypatch.setattr(rocm_mod, "on_gfx906", lambda: False)
    vllm_config = MagicMock()
    vllm_config.cache_config.gfx906_fa_packed = "on"
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.worker_cls = "ray"
    vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs.return_value = (
        False
    )
    vllm_config.model_config = None

    with caplog.at_level(logging.WARNING, logger="vllm"):
        RocmPlatform.check_and_update_config(vllm_config)

    assert any(
        "gfx906_fa_packed=on was requested but gfx906 was not detected" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]
