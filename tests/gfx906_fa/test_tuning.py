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

    # Verify/prefill keeps the unpacked path (nc2 is a decode-tile optimisation)
    # but takes the wider KV split.
    verify = static_config(seq_q=4, gqa=8)
    assert (verify.nc2, verify.kvsplit) == (1, 32)

    # GQA 4 cannot fill 8 but is even -> packed-by-2; GQA 3 is odd -> unpacked.
    assert static_config(seq_q=2, gqa=4).nc2 == 2
    assert static_config(seq_q=2, gqa=3).nc2 == 1
    # Head packing is a decode-tile optimisation only: prefill keeps nc2=1.
    assert static_config(seq_q=8, gqa=8).nc2 == 1


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


def test_nc2_mirror_agrees_with_the_cpp_probe():
    """The nc2 half of the mirror must equal the rule the launcher applies.

    The C++ rule has two parts the Python mirror originally missed: the
    ``8 -> 2 -> 1`` GQA downgrade and the decode-only clamp (``seq_q > 2``
    keeps nc2=1). Both live in ``gfx906_fa_effective_nc2`` so this probe and the
    launcher cannot disagree.
    """
    fa = pytest.importorskip(
        "vllm._gfx906_fa_C", reason="custom FA extension not built"
    )
    for heads_q, heads_kv in ((16, 2), (24, 4), (12, 2), (6, 2), (9, 3), (32, 1)):
        gqa = heads_q // heads_kv
        for seq_q in (1, 2, 3, 4, 8, 1568):
            for requested in (1, 2, 8):
                mine = static_config(seq_q, gqa).nc2
                theirs = fa.nc2_effective(requested, heads_q, heads_kv, seq_q)
                if requested != 8:
                    # An explicit non-default request is not the mirror's job;
                    # only assert the default (8) path matches.
                    continue
                assert mine == theirs, (
                    f"heads_q={heads_q} heads_kv={heads_kv} seq_q={seq_q}: "
                    f"mirror {mine} != kernel {theirs}"
                )
    # Odd GQA cannot be packed at all: 8 -> 1 (heads_q=6, heads_kv=2 -> gqa 3).
    assert fa.nc2_effective(8, 6, 2, 1) == 1
    # And an explicit, unservable request is reported, not silently clamped.
    assert fa.nc2_effective(8, 3, 2, 1) == -1


def test_warmup_records_once_and_reuses_on_the_second_boot(tmp_path, monkeypatch):
    """A cold boot records the shape set; a warm boot must not rewrite it."""
    from vllm.gfx906_fa import tuning_warmup

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.delenv("VLLM_GFX906_FA_TUNING_RECORDS", raising=False)
    tuning_warmup.reset_for_tests()

    kwargs = dict(
        device="gfx906", dtype="float16", num_heads=24, num_kv_heads=2, head_size=256
    )
    tuning_warmup.warmup_launch_config(**kwargs)
    ident = identity("gfx906", "float16", 24, 2, 256, 0)
    path = cache_path(ident)
    assert path.exists()

    cold = TuningTable.load(path, ident)
    expected = len(tuning_warmup.SHAPES) * len(tuning_warmup.SEQ_K_BUCKETS)
    assert len(cold.records) == expected
    # gqa=12 -> packs by 2 at the decode tile, unpacked in prefill.
    key_decode = workload_key(1, 65536, 1)
    assert cold.records[key_decode] == static_config(1, 12)
    assert cold.records[key_decode].nc2 == 2

    mtime = path.stat().st_mtime_ns
    tuning_warmup.reset_for_tests()
    tuning_warmup.warmup_launch_config(**kwargs)  # warm boot
    assert path.stat().st_mtime_ns == mtime, "a warm boot must not rewrite records"
    assert len(TuningTable.load(path, ident).records) == expected


def test_warmup_can_be_disabled(tmp_path, monkeypatch):
    from vllm.gfx906_fa import tuning_warmup

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("VLLM_GFX906_FA_TUNING_RECORDS", "0")
    tuning_warmup.reset_for_tests()
    tuning_warmup.warmup_launch_config(
        device="gfx906", dtype="float16", num_heads=24, num_kv_heads=2, head_size=256
    )
    assert not list(tmp_path.glob("gfx906_fa/*.json"))


def test_resolve_prefers_a_recorded_measurement(tmp_path, monkeypatch):
    """A record drives the choice; without one, the static rule is returned."""
    from vllm.gfx906_fa import tuning

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    tuning.clear_cache_for_tests()
    kwargs = dict(
        device="gfx906", dtype="float16", num_heads=24, num_kv_heads=4,
        head_size=256, block_size=0, seq_q=16, seq_k=65536, batch=2,
    )
    fallback = static_config(16, 6, batch=2)
    cfg, tuned = tuning.resolve(**{**kwargs, "fallback": fallback})
    assert (cfg, tuned) == (fallback, False), "no record yet"

    ident = identity("gfx906", "float16", 24, 4, 256, 0)
    measured = FAConfig(nc2=1, kvsplit=16, clip=True)
    table = TuningTable(ident=ident, records={})
    table.records[workload_key(16, 65536, 2)] = measured
    table.save(cache_path(ident))

    tuning.clear_cache_for_tests()  # a fresh process picks the record up
    cfg, tuned = tuning.resolve(**{**kwargs, "fallback": fallback})
    assert (cfg, tuned) == (measured, True)
    # A shape with no record still falls back to the rule.
    cfg, tuned = tuning.resolve(**{**kwargs, "seq_q": 32, "fallback": fallback})
    assert (cfg, tuned) == (fallback, False)


def test_identity_dtype_spellings_agree():
    """The three dtype spellings must key one identity, not three."""
    keys = {
        digest(identity("gfx906", spelling, 24, 4, 256, 0))
        for spelling in ("auto", "fp16", "float16", "torch.float16", "HALF")
    }
    assert len(keys) == 1, "dtype spellings must canonicalise to one identity"
    assert digest(identity("gfx906", "bfloat16", 24, 4, 256, 0)) not in keys


def test_workload_key_buckets_the_padded_context_length():
    """Serving's padded Sk must land on the tuner's round-number records."""
    from vllm.gfx906_fa.tuning import seq_k_bucket

    assert seq_k_bucket(16384) == 16384
    assert seq_k_bucket(16385) == 16384
    assert seq_k_bucket(32767) == 16384
    assert seq_k_bucket(32768) == 32768
    assert seq_k_bucket(1) == 1
    assert seq_k_bucket(0) == 0
    # A record measured at 16384 is visible to every request in that class,
    # and not to a request in the next one up.
    assert workload_key(2, 16384, 1) == workload_key(2, 20000, 1)
    assert workload_key(2, 16384, 1) != workload_key(2, 65536, 1)
