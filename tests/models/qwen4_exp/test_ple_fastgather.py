# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sorted-slice PLE gather (`VLLM_GFX906_PLE_FASTGATHER`).

T3 (2026-09-29, docs/t3-ple-prof-results.md) measured the ngram gather as a
*fixed* cost per lookup rather than a per-token one: 0.0579 ms/row at decode
sizes against 0.0073 ms/row at prefill, i.e. 3.46 ms to read 72 rows (11 KB).
The reason is structural -- `_gather_masked` runs `num_shards` (128) iterations
and about five torch CPU dispatches per iteration, while a decode batch touches
55.7 of those shards. `_gather_sorted` keeps the iteration count and makes an
untouched shard cost one Python integer compare.

What this file is actually built to prove, in order of how much it would cost to
learn it in production:

* **row-for-row bitwise identity between the two implementations.** A gather is
  a per-row copy, so a permutation of the reads must not change a single bit.
  Every case is checked against a dense reference table as well as against the
  other path, because two implementations can agree on a wrong answer;
* **which implementation actually ran.** A knob that silently falls back is
  worse than no knob: the A/B would report the old cost as the new one. The
  dispatch counters below are the red control -- the masked path performs
  `num_shards` tensor comparisons and no sort, the sorted path performs no
  comparisons and exactly one sort. If the switch ever stops working, those two
  tests fail rather than the timing measurement quietly not moving;
* **the CUDA-graph invariant survives**: the loop length is `num_shards` in both
  implementations and is not a function of which shards a batch touches;
* the guard that keeps a row from being returned uninitialized still holds on
  the new path, for out-of-range ids, the drafter's `-1` sentinel, and a shard
  that was never loaded.

Nothing here is timing-based: one decode run on this box is not evidence, and a
loaded box makes any threshold flaky. The performance claim is measured by T8
with `VLLM_GFX906_PLE_PROF`, which brackets the same gather in production.
"""

import inspect
import mmap

import numpy as np
import pytest
import regex as re
import torch

ple = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_layer",
    reason="qwen4_exp AMD model code is not importable in this build",
)

NUM_SHARDS = 3
CAPACITY = 8  # rows per shard
DIM = 5
TABLE_ROWS = NUM_SHARDS * CAPACITY


def _shard(
    base: int, dtype=torch.float16, n: int = CAPACITY, dim: int = DIM
) -> torch.Tensor:
    """Row r is filled with base + r * 0.5, so a wrong row is obvious."""
    return (
        torch.arange(n, dtype=torch.float32)
        .unsqueeze(-1)
        .expand(n, dim)
        .mul(0.5)
        .add(base)
        .to(dtype)
    )


def _build(num_shards: int, capacity: int, dim: int, dtype=torch.float16):
    e = ple.MmapShardedNGramEmbedding(num_shards, capacity, dim)
    for i in range(num_shards):
        e.set_shard(i, _shard(i * 100, dtype=dtype, n=capacity, dim=dim))
    return e


@pytest.fixture
def emb():
    return _build(NUM_SHARDS, CAPACITY, DIM)


@pytest.fixture(params=[False, True], ids=["masked", "sorted"])
def gather_path(request, monkeypatch):
    """Run the test under each implementation of the gather loop."""
    if request.param:
        monkeypatch.setenv("VLLM_GFX906_PLE_FASTGATHER", "1")
    else:
        monkeypatch.delenv("VLLM_GFX906_PLE_FASTGATHER", raising=False)
    ple._reset_fastgather_cache()
    assert ple._fastgather_enabled() is request.param
    yield request.param
    ple._reset_fastgather_cache()


@pytest.fixture
def knob(monkeypatch):
    """Set the knob for one test and leave the module cache clean after."""

    def _set(on: bool) -> None:
        if on:
            monkeypatch.setenv("VLLM_GFX906_PLE_FASTGATHER", "1")
        else:
            monkeypatch.delenv("VLLM_GFX906_PLE_FASTGATHER", raising=False)
        ple._reset_fastgather_cache()

    _set(False)
    yield _set
    ple._reset_fastgather_cache()


def _reference(shards, ids, dim):
    """What a single dense embedding table would have returned."""
    table = torch.cat(shards, dim=0)
    flat = ids.reshape(-1).long()
    return table.index_select(0, flat.clamp(min=0)).reshape(*ids.shape, dim)


def _count_tensor_eq(monkeypatch) -> dict:
    """Count `tensor == value` comparisons, the masked loop's per-shard cost."""
    calls = {"n": 0}
    original = torch.Tensor.__eq__

    def wrapper(self, other):
        calls["n"] += 1
        return original(self, other)

    monkeypatch.setattr(torch.Tensor, "__eq__", wrapper)
    return calls


def _count_attr(monkeypatch, obj, name) -> dict:
    calls = {"n": 0}
    original = getattr(obj, name)

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, name, wrapper)
    return calls


# --------------------------------------------------------------------------
# the knob, and proof of which implementation ran (the red control)
# --------------------------------------------------------------------------


def test_knob_defaults_off_and_runs_the_masked_loop(emb, monkeypatch):
    monkeypatch.delenv("VLLM_GFX906_PLE_FASTGATHER", raising=False)
    ple._reset_fastgather_cache()
    assert ple._fastgather_enabled() is False
    eq = _count_tensor_eq(monkeypatch)
    sorts = _count_attr(monkeypatch, torch, "sort")
    ids = torch.tensor([1, 9, 17])
    out = emb(ids)
    # One comparison per shard, whether or not the batch touches it: this is the
    # 128 x ~5 dispatches T3 measured. And no sort anywhere.
    assert eq["n"] == NUM_SHARDS
    assert sorts["n"] == 0
    assert torch.equal(out, _reference(emb._shards, ids, DIM))


def test_knob_on_runs_the_sorted_loop(emb, monkeypatch):
    monkeypatch.setenv("VLLM_GFX906_PLE_FASTGATHER", "1")
    ple._reset_fastgather_cache()
    assert ple._fastgather_enabled() is True
    eq = _count_tensor_eq(monkeypatch)
    sorts = _count_attr(monkeypatch, torch, "sort")
    bins = _count_attr(monkeypatch, torch, "bincount")
    selects = _count_attr(monkeypatch, torch.Tensor, "index_select")
    ids = torch.tensor([1, 9, 17])
    out = emb(ids)
    # Zero per-shard comparisons is the whole point of the rewrite.
    assert eq["n"] == 0
    assert sorts["n"] == 1, "sorted path must run exactly one sort per lookup"
    assert bins["n"] == 1, "sorted path must run exactly one bincount per lookup"
    # One read per shard the batch actually touches, and nothing for the rest.
    assert selects["n"] == len({int(i) // CAPACITY for i in ids})
    assert torch.equal(out, _reference(emb._shards, ids, DIM))


def test_knob_value_is_cached_and_reset(monkeypatch, knob):
    monkeypatch.setenv("VLLM_GFX906_PLE_FASTGATHER", "1")
    ple._reset_fastgather_cache()
    assert ple._fastgather_enabled() is True
    # Flipping the environment without resetting the cache must not change the
    # path: the cached read is what keeps this off the per-lookup critical path.
    monkeypatch.delenv("VLLM_GFX906_PLE_FASTGATHER")
    assert ple._fastgather_enabled() is True
    knob(False)
    assert ple._fastgather_enabled() is False


# --------------------------------------------------------------------------
# bitwise identity
# --------------------------------------------------------------------------


def test_both_paths_are_bitwise_identical(emb, monkeypatch):
    ids = torch.tensor([7, 0, 23, 23, 8, 1, 15, 22, 16, 2])
    monkeypatch.setenv("VLLM_GFX906_PLE_FASTGATHER", "1")
    ple._reset_fastgather_cache()
    fast = emb(ids)
    monkeypatch.delenv("VLLM_GFX906_PLE_FASTGATHER")
    ple._reset_fastgather_cache()
    slow = emb(ids)
    assert fast.dtype == slow.dtype
    assert fast.shape == slow.shape
    assert torch.equal(fast, slow), "the gather is a copy: no bit may move"


CASES = {
    "empty": [],
    "single": [3],
    "one_shard_only": [0, 1, 2, 3, 4, 5, 6, 7],
    "shard_boundaries": [7, 8, 15, 16, 23],
    "duplicates": [5, 5, 5, 12, 12, 20],
    "descending": list(range(TABLE_ROWS - 1, -1, -1)),
    "every_row": list(range(TABLE_ROWS)),
    "one_row_each_shard": [0, CAPACITY, 2 * CAPACITY],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_parity_vs_dense_reference(emb, gather_path, case):
    ids = torch.tensor(CASES[case])
    out = emb(ids)
    assert out.shape == (ids.numel(), DIM)
    assert out.dtype is emb.params_dtype
    assert torch.equal(out, _reference(emb._shards, ids, DIM))


def test_parity_on_nd_ids(emb, gather_path):
    ids = torch.tensor([[1, 9, 17], [22, 0, 7]])
    out = emb(ids)
    assert out.shape == (2, 3, DIM)
    assert torch.equal(out, _reference(emb._shards, ids, DIM))


@pytest.mark.parametrize("seed", range(20))
def test_parity_random_id_sets(emb, gather_path, seed):
    g = torch.Generator().manual_seed(seed)
    n = int(torch.randint(0, 400, (1,), generator=g).item())
    # Skewed toward one shard, which is the decode shape: most shards empty.
    shard = torch.randint(0, NUM_SHARDS, (n,), generator=g)
    if n:
        shard = torch.zeros(n, dtype=torch.long) if seed % 2 else shard
    local = torch.randint(0, CAPACITY, (n,), generator=g)
    ids = shard * CAPACITY + local
    out = emb(ids)
    assert torch.equal(out, _reference(emb._shards, ids, DIM))


def test_parity_decode_shaped_lookup_over_128_shards(gather_path):
    """The shape T3 measured: 72 rows touching ~56 of 128 shards.

    128 shards is `split_ngram_parts` in the served config, so the fixed loop
    length here is the real one and the empty-shard handling is exercised at
    the scale the knob exists to fix.
    """
    shards, capacity, dim = 128, 64, 8
    e = _build(shards, capacity, dim)
    g = torch.Generator().manual_seed(20260929)
    ids = torch.randint(0, shards * capacity, (72,), generator=g)
    out = e(ids)
    assert len({int(i) // capacity for i in ids}) < shards, "premise: shards empty"
    assert torch.equal(out, _reference(e._shards, ids, dim))


def test_parity_prefill_shaped_lookup_touches_every_shard(gather_path):
    """A full 4096-token chunk touches 128.0 of 128 shards (T3), so the
    non-empty branch has to be correct at that scale too."""
    shards, capacity, dim = 128, 64, 8
    e = _build(shards, capacity, dim)
    ids = torch.randint(0, shards * capacity, (4096 * 16,), generator=None)
    out = e(ids)
    assert len({int(i) // capacity for i in ids}) == shards, "premise: all touched"
    assert torch.equal(out, _reference(e._shards, ids, dim))


# --------------------------------------------------------------------------
# the guards still hold on the new path
# --------------------------------------------------------------------------


def test_out_of_range_ids_fold_to_row_0_on_both_paths(emb, gather_path):
    ids = torch.tensor([5, -1, TABLE_ROWS, 10_000_000, 6])
    out = emb(ids)
    row0 = emb._shards[0][0]
    assert torch.equal(out[1], row0)  # drafter's -1 sentinel
    assert torch.equal(out[2], row0)  # one past the last row
    assert torch.equal(out[3], row0)  # far outside
    assert torch.equal(out[0], emb._shards[0][5])
    assert torch.equal(out[4], emb._shards[0][6])


def test_unloaded_shard_raises_the_same_error(emb, gather_path):
    emb._shards[1] = None
    with pytest.raises(RuntimeError, match=r"PLE ngram shard 1 was never loaded"):
        emb(torch.tensor([2, 9, 10]))


def test_no_shards_loaded_at_all_raises(emb, gather_path):
    fresh = ple.MmapShardedNGramEmbedding(NUM_SHARDS, CAPACITY, DIM)
    with pytest.raises((ValueError, RuntimeError)):
        fresh(torch.tensor([1, 2, 3]))


def test_sorted_gather_refuses_unfolded_ids(emb, monkeypatch):
    """The `pos != n` guard is the last thing between a bug in the range test
    and a silently uninitialized row: it must fire, not return garbage.

    On the v0.30 line the shard index is no longer an argument a caller can get
    wrong -- `_gather_sorted` derives it from `_shard_of` -- so the same fault is
    injected one level down. An id whose shard index runs past num_shards is
    exactly what an unfolded id looks like to the gather.
    """
    ids = torch.tensor([1, 2, 3])
    out = torch.empty((ids.numel(), DIM), dtype=emb.params_dtype)
    monkeypatch.setattr(
        emb,
        "_shard_of",
        lambda _ids: (
            torch.tensor([0, NUM_SHARDS + 5, 1], dtype=torch.long),
            torch.tensor([0, 0, 0], dtype=torch.long),
        ),
    )
    with pytest.raises(RuntimeError, match="range guard|did not run"):
        emb._gather_sorted(ids, out)


def test_sorted_gather_reads_the_live_shard_not_a_copy(emb, knob):
    """A gather that cached rows would pass every equality test above and still
    serve stale embeddings once `load_weights` finished."""
    knob(True)
    ids = torch.tensor([1, 9, 17])
    before = emb(ids).clone()
    emb._shards[1][1] = 4242.0
    after = emb(ids)
    assert not torch.equal(before, after)
    assert torch.equal(after[1], torch.full((DIM,), 4242.0, dtype=emb.params_dtype))


def test_sorted_path_on_a_real_mmap_backed_shard(tmp_path, knob):
    """The table in production is the page cache, not a heap tensor, and the
    sorted gather must read the mapping it was handed."""
    knob(True)
    rows, dim = 32, 4
    arr = np.ascontiguousarray(
        np.tile(np.arange(rows, dtype=np.float32)[:, None], (1, dim))
    )
    path = tmp_path / "shard.bin"
    path.write_bytes(arr.tobytes())
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), arr.nbytes, access=mmap.ACCESS_READ)
    e = None
    tensor = None
    try:
        tensor = torch.from_numpy(
            np.frombuffer(mm, dtype=np.float32).reshape(rows, dim)
        )
        e = ple.MmapShardedNGramEmbedding(1, rows, dim)
        e.set_shard(0, tensor)
        assert e._shards[0] is tensor
        ids = torch.tensor([3, 17, 31, 3, 0])
        assert torch.equal(e(ids), tensor.index_select(0, ids))
    finally:
        # The mmap cannot be closed while torch still holds an export of it.
        if e is not None:
            e._shards = [None]
        tensor = None
        mm.close()


# --------------------------------------------------------------------------
# the CUDA-graph invariant, asserted structurally
# --------------------------------------------------------------------------


def test_both_gathers_iterate_the_fixed_shard_range():
    """Loop length is the graph-safety property, and it is a property of the
    source, not of any observable output: both implementations must walk
    `range(self.num_shards)` and nothing data-dependent.

    A behaviour test cannot see this -- correct results look identical whether
    the loop ran 56 times or 128 -- so the shape of the loop is what is checked.
    If someone "optimizes" the loop by iterating only the touched shards, this
    fails here instead of during CUDA graph capture on a production boot.
    """
    for name in ("_gather_masked", "_gather_sorted"):
        src = inspect.getsource(getattr(ple.MmapShardedNGramEmbedding, name))
        assert re.search(r"for\s+shard\s+in\s+range\(self\.num_shards\)", src), name
        assert not re.search(r"for\s+\w+\s+in\s+(?!range\(self\.num_shards\))", src), (
            f"{name} iterates something data-dependent"
        )
