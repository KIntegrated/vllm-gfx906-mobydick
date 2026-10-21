# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Behaviour of the gfx906 FA tuning-record cache (P32-4/#63)."""

import json

import pytest

from vllm.gfx906_fa.tuning import (
    FAConfig,
    TuningTable,
    cache_path,
    digest,
    gqa_ratio,
    identity,
    static_config,
    workload_key,
)

IDENT = identity("gfx906", "float16", 12, 2, 256, 1376)
OTHER = identity("gfx906", "float16", 16, 2, 256, 1376)


def test_static_rule_matches_the_kernel_defaults():
    """The fallback must reproduce csrc/gfx906_fa/gfx906_fa.cpp, not near it."""
    decode = static_config(seq_q=1, gqa=8)
    assert (decode.nc2, decode.kvsplit, decode.clip) == (8, 16, True)

    verify = static_config(seq_q=4, gqa=8)
    assert (verify.nc2, verify.kvsplit) == (8, 32)

    # GQA ratio that cannot fill the packing factor falls back to unpacked.
    assert static_config(seq_q=2, gqa=4).nc2 == 1


def test_static_rule_rejects_non_integral_gqa():
    with pytest.raises(ValueError, match="not a multiple"):
        gqa_ratio(12, 5)


def test_digest_ignores_key_order_and_separates_geometry():
    reordered = dict(reversed(list(IDENT.items())))
    assert digest(IDENT) == digest(reordered)
    assert digest(IDENT) != digest(OTHER)
    assert cache_path(IDENT) != cache_path(OTHER)


def test_records_roundtrip_and_fall_back_when_untuned(tmp_path):
    path = tmp_path / "table.json"
    key = workload_key(seq_q=1, seq_k=65536, batch=4)
    fallback = static_config(seq_q=1, gqa=6)

    table = TuningTable(ident=IDENT, records={})
    assert table.select(key, fallback) == (fallback, False)

    tuned = FAConfig(nc2=4, kvsplit=8, clip=False)
    table.records[key] = tuned
    table.save(path)

    reloaded = TuningTable.load(path, IDENT)
    assert reloaded.select(key, fallback) == (tuned, True)
    assert reloaded.untuned([key, "sq8-sk1024-b1"]) == ["sq8-sk1024-b1"]


def test_a_foreign_or_corrupt_file_yields_no_records(tmp_path):
    path = tmp_path / "table.json"
    key = workload_key(1, 1024, 1)
    TuningTable(ident=IDENT, records={key: FAConfig(8, 16, True)}).save(path)

    # Same file, different head geometry: records must not be reused.
    assert TuningTable.load(path, OTHER).records == {}

    path.write_text("{not json")
    assert TuningTable.load(path, IDENT).records == {}


def test_save_is_atomic_under_a_lock(tmp_path):
    path = tmp_path / "table.json"
    key = workload_key(1, 1024, 1)
    TuningTable(ident=IDENT, records={key: FAConfig(8, 16, True)}).save(path)

    assert json.loads(path.read_text())["identity"] == IDENT
    assert not path.with_suffix(".json.tmp").exists()
    assert path.with_suffix(".lock").exists()


def test_workload_key_separates_the_launch_dims():
    assert workload_key(1, 65536, 1) != workload_key(1, 65536, 4)
    assert workload_key(2, 1024, 1) != workload_key(4, 1024, 1)


def test_kv_split_mirror_agrees_with_the_cpp_probe(monkeypatch):
    """The Python mirror must equal the rule the kernels actually apply.

    `vllm._gfx906_fa_C.kv_split_default` is the C++ decision for an unset
    `GFX906_FA_KVSPLIT` (a no-GPU-work probe). Comparing against it is what
    keeps `static_config` from drifting; it already caught one: the
    paged-direct decode rule is batch-aware (8/8/5/2/2 for B=1/2/3/4/8), not the
    gather path's flat 16.
    """
    fa = pytest.importorskip(
        "vllm._gfx906_fa_C", reason="custom FA extension not built"
    )
    monkeypatch.delenv("GFX906_FA_KVSPLIT", raising=False)
    monkeypatch.delenv("GFX906_FA_KVSPLIT_SHAPEAWARE", raising=False)

    for direct in (False, True):
        for seq_q in (1, 2, 3, 4, 8, 1568):
            for batch in (1, 2, 3, 4, 8):
                mine = static_config(seq_q, 8, batch=batch, direct=direct)
                assert mine.kvsplit == fa.kv_split_default(seq_q, batch, direct), (
                    f"seq_q={seq_q} batch={batch} direct={direct}"
                )
