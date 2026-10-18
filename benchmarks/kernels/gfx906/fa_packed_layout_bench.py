# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""A/B the packed KV layout (GFX906_FA_PACKED, fork PR #43) against the Q8
alias layout (GFX906_FA_LEGACY=0, the serving default).

The flag changes only the KV *layout*, so this drives `Gfx906FAImpl.forward`
directly on equal-context caches: one process builds both impls (the mode is
read at construction), fills each via the impl's own `do_kv_cache_update`, and
then interleaves the two layouts rep by rep so clock/DVFS drift cancels.

Run:  HIP_VISIBLE_DEVICES=0 .venv/bin/python \
          benchmarks/kernels/gfx906/fa_packed_layout_bench.py
Env:  BENCH_REPS (default 40), BENCH_LENS (default 2048,16384),
      BENCH_BS (default 1,4), BENCH_DIMS (default 128,256)
"""

import math
import os
import statistics
from types import SimpleNamespace

import torch

from vllm.gfx906_fa.gfx906_fa_backend import (
    Gfx906FAImpl,
    Gfx906FAMetadata,
    _packed_cell_elems,
)

BLOCK = 16
HKV = 2
HQ = 16
REPS = int(os.environ.get("BENCH_REPS", "40"))
LENS = [int(x) for x in os.environ.get("BENCH_LENS", "2048,16384").split(",")]
BS = [int(x) for x in os.environ.get("BENCH_BS", "1,4").split(",")]
DIMS = [int(x) for x in os.environ.get("BENCH_DIMS", "128,256").split(",")]
dev = "cuda"


def build(packed: bool, D: int, B: int, L: int):
    # Same seed in both arms: the two layouts must be filled with identical
    # K/V, otherwise the output comparison is meaningless.
    torch.manual_seed(1234)
    os.environ["GFX906_FA_PACKED"] = "1" if packed else "0"
    os.environ.pop("GFX906_FA_LEGACY", None)  # serving default: Q8 alias path
    impl = Gfx906FAImpl(
        num_heads=HQ,
        head_size=D,
        scale=1.0 / math.sqrt(D),
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
    )
    assert impl._packed == packed and not impl._legacy
    per_seq_blocks = (L + BLOCK - 1) // BLOCK
    nb = per_seq_blocks * B
    if packed:
        cache = torch.zeros(
            (nb, HKV, BLOCK, _packed_cell_elems(D)), dtype=torch.float16, device=dev
        )
    else:
        cache = torch.zeros((nb, HKV, BLOCK, 2 * D), dtype=torch.float16, device=dev)

    # Fill through the impl's own write path: one contiguous slot range per
    # sequence, so the block table is just a per-sequence offset window.
    key = torch.randn(B * L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    val = torch.randn(B * L, HKV, D, device=dev, dtype=torch.float16) * 0.5
    slots = torch.cat(
        [
            torch.arange(i * per_seq_blocks * BLOCK, i * per_seq_blocks * BLOCK + L)
            for i in range(B)
        ]
    ).to(device=dev, dtype=torch.int64)
    # `layer` only supplies the (unused, unit) quant scales on the alias path.
    layer = SimpleNamespace(_k_scale=torch.tensor(1.0, device=dev),
                            _v_scale=torch.tensor(1.0, device=dev))
    impl.do_kv_cache_update(layer, key, val, cache, slots)

    bt = torch.stack(
        [
            torch.arange(i * per_seq_blocks, (i + 1) * per_seq_blocks)
            for i in range(B)
        ]
    ).to(device=dev, dtype=torch.int32)
    meta = Gfx906FAMetadata(
        num_actual_tokens=B,
        max_query_len=1,
        max_seq_len=L,
        query_start_loc=torch.arange(B + 1, device=dev, dtype=torch.int32),
        seq_lens=torch.full((B,), L, device=dev, dtype=torch.int32),
        block_table=bt,
        slot_mapping=torch.empty(0, dtype=torch.int64, device=dev),
    )
    q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16) * 0.5
    out = torch.zeros(B, HQ, D, device=dev, dtype=torch.float16)
    return impl, cache, meta, q, out


def timed(impl, cache, meta, q, out, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        impl.forward(None, q, q, q, cache, meta, output=out)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts), min(ts)


def main():
    print(f"reps={REPS} HKV={HKV} HQ={HQ} block={BLOCK} (impl.forward, decode, 1 row/seq)")
    print(f"{'D':>4} {'B':>2} {'ctx':>6} | {'legacy us':>10} {'packed us':>10} "
          f"{'delta':>7} | {'legacy min':>10} {'packed min':>10} | "
          f"{'max|d|':>9} {'max|d|/max':>10}")
    for D in DIMS:
        for B in BS:
            for L in LENS:
                lo, lc, lm, lq, lout = build(False, D, B, L)
                po, pc, pm, pq, pout = build(True, D, B, L)
                for _ in range(5):  # warm both paths (buffers, plans)
                    lo.forward(None, lq, lq, lq, lc, lm, output=lout)
                    po.forward(None, pq, pq, pq, pc, pm, output=pout)
                lo.forward(None, lq, lq, lq, lc, lm, output=lout)
                po.forward(None, pq, pq, pq, pc, pm, output=pout)
                same = bool(torch.equal(lout, pout))
                d = (lout.float() - pout.float()).abs().max().item()
                rel = d / pout.float().abs().max().item()
                # interleave rep by rep so DVFS drift cancels between layouts
                lt, pt = [], []
                for _ in range(REPS):
                    lt.append(timed(lo, lc, lm, lq, lout, 1)[1])
                    pt.append(timed(po, pc, pm, pq, pout, 1)[1])
                lmed, lmin = statistics.median(lt), min(lt)
                pmed, pmin = statistics.median(pt), min(pt)
                print(f"{D:>4} {B:>2} {L:>6} | {lmed:>10.1f} {pmed:>10.1f} "
                      f"{100.0 * (pmed - lmed) / lmed:>+6.1f}% | {lmin:>10.1f} "
                      f"{pmin:>10.1f} | {d:>9.2e} {rel:>10.2e}"
                      + ("" if not same else "  (identical)"))
                del lo, lc, lm, lq, lout, po, pc, pm, pq, pout
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
