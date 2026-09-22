# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the mmap-backed PLE ngram embedding.

The change under test replaces a materialized `nn.Embedding` table with
`MmapShardedNGramEmbedding`: CPU tensors that are *the mmap'd safetensors
shards themselves*, held by reference (`load_weights` does
`embedding.set_shard(i, loaded_weight.to("cpu"))` with no `.copy_()` and no
device move), so a ~100 GB table is paid once per node through the page cache
instead of once per worker process.

What is asserted, and what deliberately is not:

* row-for-row equality against a plain reference gather, across shard
  boundaries, for 1-D and N-D id tensors, with the shard-index math at the
  exact capacity boundaries;
* **no copy**: the stored shard is the same object with the same `data_ptr()`
  as the tensor handed in -- that is the entire memory claim;
* it works on a genuinely `mmap`-backed tensor, not just a heap one;
* the guards that protect a silent wrong-lookup: non-CPU shard, non-CPU ids,
  dtype drift between shards, a shard index beyond `split_ngram_parts`, an
  `embedding_dim` mismatch, a shard that was never loaded, and an id outside
  the table's row range (negative, or past the last row) -- which before the
  range check matched no shard mask and left its row holding uninitialized
  memory from `new_empty()`;
* `load_weights` bookkeeping: which names it reports as loaded, and that
  `hashstats_*` / `token_lookup` leaves are skipped.

The fixed-length `for shard in range(self.num_shards)` loop is the CUDA-graph
safety property (a data-dependent loop length would run a different number of
Python steps in capture than in replay). That property is structural -- see the
comment at the loop -- so it is asserted here as the observable consequence:
batches that touch different shards give correct results and leave no state
behind between calls.
"""

import mmap

import numpy as np
import pytest
import torch

ple = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_layer",
    reason="qwen4_exp AMD model code is not importable in this build",
)

NUM_SHARDS = 3
CAPACITY = 8  # rows per shard
DIM = 5


def _shard(base: int, dtype=torch.float16, n: int = CAPACITY) -> torch.Tensor:
    """Rows of a recognizable pattern: row r is filled with base + r."""
    return (
        torch.arange(n, dtype=torch.float32)
        .unsqueeze(-1)
        .expand(n, DIM)
        .mul(0.5)
        .add(base)
        .to(dtype)
    )


@pytest.fixture
def emb():
    e = ple.MmapShardedNGramEmbedding(NUM_SHARDS, CAPACITY, DIM)
    for i in range(NUM_SHARDS):
        e.set_shard(i, _shard(i * 100))
    return e


def _reference(shards, ids, dim):
    """What a single dense embedding table would have returned."""
    table = torch.cat(shards, dim=0)
    flat = ids.reshape(-1).long()
    return table.index_select(0, flat).reshape(*ids.shape, dim)


# --------------------------------------------------------------------------
# correctness of the lookup
# --------------------------------------------------------------------------


def test_shards_start_empty():
    e = ple.MmapShardedNGramEmbedding(NUM_SHARDS, CAPACITY, DIM)
    assert e._shards == [None] * NUM_SHARDS
    assert e.params_dtype is None
    assert e.num_shards == NUM_SHARDS and e.shard_row_capacity == CAPACITY
    assert e.embedding_dim == DIM


def test_forward_matches_reference_gather(emb):
    ids = torch.tensor([0, 1, 7, 8, 15, 16, 23])
    out = emb(ids)
    assert torch.equal(out, _reference(emb._shards, ids, DIM))
    assert out.shape == (ids.numel(), DIM)


def test_forward_shard_boundaries(emb):
    """Row 7 is the last row of shard 0, row 8 the first of shard 1."""
    ids = torch.tensor([7, 8, 15, 16])
    out = emb(ids)
    assert out[0, 0].item() == pytest.approx(3.5, abs=1e-3)  # shard 0 row 7
    assert out[1, 0].item() == pytest.approx(100.0, abs=1e-3)  # shard 1 row 0
    assert out[2, 0].item() == pytest.approx(103.5, abs=1e-3)  # shard 1 row 7
    assert out[3, 0].item() == pytest.approx(200.0, abs=1e-3)  # shard 2 row 0


def test_forward_preserves_batch_shape(emb):
    ids = torch.tensor([[1, 9], [17, 22]])
    out = emb(ids)
    assert out.shape == (2, 2, DIM)
    assert torch.equal(out, _reference(emb._shards, ids, DIM))


def test_forward_keeping_dtype_and_device(emb):
    ids = torch.tensor([2, 10, 20])
    out = emb(ids)
    assert out.dtype == emb.params_dtype is torch.float16
    assert out.device.type == "cpu"


def test_repeated_calls_over_different_shards_are_stateless(emb):
    """The graph-safety consequence: whichever shards a batch touches must not
    change what the next batch sees."""
    a = emb(torch.tensor([0, 1]))
    b = emb(torch.tensor([0, 1]))
    emb(torch.arange(NUM_SHARDS * CAPACITY))  # touches every shard in between
    c = emb(torch.tensor([0, 1]))
    assert torch.equal(a, b) and torch.equal(a, c)
    assert [t.shape for t in emb._shards] == [(CAPACITY, DIM)] * NUM_SHARDS


def test_scalar_and_empty_id_tensors(emb):
    assert emb(torch.tensor(4)).shape == (DIM,)
    assert emb(torch.tensor([], dtype=torch.long)).shape == (0, DIM)


def test_works_on_a_real_mmap_backed_tensor(tmp_path):
    """The whole point of the class: the table is the page cache, not a copy."""
    path = tmp_path / "shard0.safetensors-like"
    rows, dim = 32, 4
    arr = np.ascontiguousarray(
        np.tile(np.arange(rows, dtype=np.float32)[:, None], (1, dim))
    )
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), arr.nbytes, access=mmap.ACCESS_READ)
    e = None
    tensor = None
    try:
        tensor = torch.from_numpy(
            np.frombuffer(mm, dtype=np.float32).reshape(rows, dim)
        )

        e = ple.MmapShardedNGramEmbedding(1, rows, dim)
        e.set_shard(0, tensor)
        assert e._shards[0] is tensor, "forward must read the mmap, not a copy"
        ids = torch.tensor([3, 17, 31])
        assert torch.equal(e(ids), tensor.index_select(0, ids))
    finally:
        # The mmap cannot be closed while torch still holds an export of it.
        if e is not None:
            e._shards = [None]
        tensor = None
        mm.close()


# --------------------------------------------------------------------------
# the guards: each of these would otherwise be a silent wrong lookup
# --------------------------------------------------------------------------


def test_set_shard_rejects_non_cpu_tensor(emb):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA/ROCm device to test the device guard")
    with pytest.raises(ValueError, match="must be loaded on CPU"):
        emb.set_shard(0, _shard(0).cuda())


def test_forward_rejects_non_cpu_ids(emb):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA/ROCm device to test the device guard")
    with pytest.raises(ValueError, match="requires CPU ids"):
        emb(_shard(0).cuda().long())


def test_set_shard_rejects_dtype_drift(emb):
    with pytest.raises(ValueError, match="does not match previously loaded"):
        emb.set_shard(1, _shard(50, dtype=torch.bfloat16))


def test_first_shard_fixes_the_dtype(emb):
    assert emb.params_dtype is torch.float16


def test_missing_shard_raises_rather_than_returning_garbage():
    e = ple.MmapShardedNGramEmbedding(NUM_SHARDS, CAPACITY, DIM)
    e.set_shard(0, _shard(0))
    e.set_shard(2, _shard(200))
    # ids in shard 1's range, and shard 1 was never loaded
    with pytest.raises(RuntimeError, match="shard 1 was never loaded"):
        e(torch.tensor([9]))
    # ids that avoid the hole still work
    assert e(torch.tensor([1, 17])).shape == (2, DIM)


def test_out_of_range_and_negative_ids_raise_rather_than_returning_garbage(emb):
    """An id past the last row matches no shard mask, so before the range check
    its row kept whatever `new_empty()` found there: a wrong embedding with no
    error. Same for a negative id, which floors to a negative shard index."""
    table_rows = NUM_SHARDS * CAPACITY
    with pytest.raises(ValueError, match="out of range"):
        emb(torch.tensor([table_rows]))
    with pytest.raises(ValueError, match=r"\[0, 24\]"):
        emb(torch.tensor([0, table_rows]))
    with pytest.raises(ValueError, match="out of range"):
        emb(torch.tensor([-1]))
    # the last valid row still works, and so does the whole valid range
    assert emb(torch.tensor([table_rows - 1])).shape == (1, DIM)


# --------------------------------------------------------------------------
# load_weights: the no-copy bookkeeping
# --------------------------------------------------------------------------


class _Stub:
    """Duck-typed `Qwen4ExpNGramEmbedding`.

    Constructing the real module means running the prime search over
    `ngram_vocab_size_base`, which is not what is under test here; `load_weights`
    only ever touches the attributes below, so it is called unbound against this.
    """

    def __init__(self, split_ngram_parts=NUM_SHARDS, dim=DIM):
        self.split_ngram_parts = split_ngram_parts
        self.layer_multipliers = torch.zeros(4, dtype=torch.int64)
        self.ngram_heads_offsets = torch.zeros(2, dtype=torch.long)
        self.ngram_heads_vocab_sizes = torch.zeros(2, dtype=torch.long)
        self.ngram_embedding = ple.MmapShardedNGramEmbedding(
            split_ngram_parts, CAPACITY, dim
        )


def _load(stub, weights):
    return ple.Qwen4ExpNGramEmbedding.load_weights(stub, weights)


def test_load_weights_reports_shard_names_and_stores_by_reference():
    stub = _Stub()
    w0, w1 = _shard(0), _shard(100)
    loaded = _load(
        stub,
        [
            ("ngram_embedding.shard_0.weight", w0),
            ("ngram_embedding.shard_1.weight", w1),
        ],
    )
    assert loaded == {"ngram_embedding.shard_0", "ngram_embedding.shard_1"}
    assert stub.ngram_embedding._shards[0] is w0, "no .copy_(): by reference"
    assert stub.ngram_embedding._shards[1] is w1
    assert stub.ngram_embedding._shards[0].data_ptr() == w0.data_ptr()
    assert stub.ngram_embedding._shards[0].device.type == "cpu"


def test_load_weights_skips_hash_and_lookup_leaves():
    stub = _Stub()
    loaded = _load(
        stub,
        [
            ("ngram_embedding.hashstats_0", _shard(0)),
            ("token_lookup", _shard(0)),
            ("ngram_embedding.shard_0.weight", _shard(0)),
        ],
    )
    assert loaded == {"ngram_embedding.shard_0"}
    assert stub.ngram_embedding._shards[1] is None


def test_load_weights_fills_persistent_buffers():
    stub = _Stub()
    vals = torch.tensor([11, 22], dtype=torch.long)
    loaded = _load(stub, [("ngram_heads_offsets", vals)])
    assert "ngram_heads_offsets" in loaded
    assert torch.equal(stub.ngram_heads_offsets, vals)


def test_load_weights_rejects_buffer_shape_mismatch():
    stub = _Stub()
    with pytest.raises(ValueError, match="Shape mismatch"):
        _load(stub, [("ngram_heads_offsets", torch.zeros(7, dtype=torch.long))])


def test_load_weights_rejects_shard_index_beyond_split():
    stub = _Stub()
    with pytest.raises(ValueError, match="exceeds split_ngram_parts"):
        _load(stub, [(f"ngram_embedding.shard_{NUM_SHARDS}.weight", _shard(0))])


def test_load_weights_rejects_embedding_dim_mismatch():
    stub = _Stub()
    wrong = torch.zeros(CAPACITY, DIM + 1, dtype=torch.float16)
    with pytest.raises(ValueError, match="Shape mismatch for PLE embedding shard"):
        _load(stub, [("ngram_embedding.shard_0.weight", wrong)])
