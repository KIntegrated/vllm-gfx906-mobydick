"""CPU differential: real device _compute_ngram_ids vs the host numpy replica.

Both sides get byte-identical inputs. If they disagree, the host replica's
arithmetic is wrong and no boot is needed to find it. If they agree on every
shape here, the ids difference seen in the 2026-10-01 14:10 boot is in the
INPUTS, not the formula -- and the next boot must dump tokens.
"""

import numpy as np
import torch

from vllm.models.qwen4_exp.amd.ple_hostids import compute_ngram_ids_host
from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpNGramEmbedding

NGRAM = 3
HPN = 8
CTX = NGRAM - 1
HEADS = (NGRAM - 1) * HPN
EOS = 151645
BASE = 20_000_000

rng = np.random.default_rng(20261001)
multipliers = rng.integers(1, 2**62, size=NGRAM + 1, dtype=np.int64)
vocab_sizes = np.full(HEADS, BASE, dtype=np.int64)
head_offsets = (np.arange(HEADS, dtype=np.int64) * (BASE + 3)).astype(np.int64)


def device(ids, qsl, ctx, packed_width):
    n = qsl.numel() - 1
    return Qwen4ExpNGramEmbedding._compute_ngram_ids(
        ids,
        qsl,
        ctx,
        multipliers=torch.from_numpy(multipliers),
        vocab_sizes=torch.from_numpy(vocab_sizes),
        head_offsets=torch.from_numpy(head_offsets),
        packed=torch.full((n, packed_width), EOS, dtype=torch.long),
        positions=torch.arange(ids.numel(), dtype=torch.long),
        eos_token_id=EOS,
        ngram_size=NGRAM,
        heads_per_ngram=HPN,
    )


def run(qlens, label, pack=None, seed_ctx=None):
    num_reqs = len(qlens)
    total = sum(qlens)
    qsl = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(np.array(qlens, dtype=np.int64), out=qsl[1:])
    ids = rng.integers(1, 100_000, size=total, dtype=np.int64)
    ctx = np.full((num_reqs, CTX), EOS, dtype=np.int64)
    for i in range(num_reqs):
        ctx[i] = rng.integers(1, 100_000, size=CTX, dtype=np.int64)
    if seed_ctx:
        ctx[0, 0] = EOS
    packed_width = pack if pack else max(qlens)

    host = compute_ngram_ids_host(
        ids,
        qsl,
        ctx,
        multipliers=multipliers,
        vocab_sizes=vocab_sizes,
        head_offsets=head_offsets,
        eos_token_id=EOS,
        ngram_size=NGRAM,
        heads_per_ngram=HPN,
    )
    dev = device(
        torch.from_numpy(ids),
        torch.from_numpy(qsl),
        torch.from_numpy(ctx),
        packed_width,
    ).numpy()
    same = bool(np.array_equal(host, dev))
    if same:
        print(f"  MATCH   {label}: {host.shape}")
        return True
    bad = np.argwhere(host != dev)
    print(f"  DIFFER  {label}: {len(bad)} entries, first {bad[:6].tolist()}")
    r, c = bad[0]
    print(f"          row {r} head {c}: host={host[r, c]} device={dev[r, c]}")
    rows = sorted({int(x) for x, _ in bad})
    print(f"          differing rows: {rows} of {host.shape[0]}")
    for r in rows[:2]:
        print(f"          host row {r} = {host[r, :4].tolist()} ...")
        print(f"          dev  row {r} = {dev[r, :4].tolist()} ...")
    return False


ok = True
print("CPU differential: real _compute_ngram_ids vs compute_ngram_ids_host")
ok = run([4], "1 req x 4 rows (MTP num_spec=3), packed width == qlen") and ok
ok = run([4, 4, 4], "3 reqs x 4 rows, packed width == qlen") and ok
ok = (
    run([4, 4, 4], "3 reqs x 4 rows, packed width 4096 (real workspace)", pack=4096)
    and ok
)
ok = run([4], "1 req x 4 rows, packed width 4096", pack=4096) and ok
ok = run([1], "1 req x 1 row (no drafts)") and ok
ok = run([4, 4, 4], "3 reqs, one context cell is eos", seed_ctx=True) and ok
ok = run([4, 2, 4], "ragged queries (short draft on req 1)") and ok
ok = run([4, 4, 4], "3 reqs, packed width 8", pack=8) and ok
print("ALL MATCH" if ok else "MISMATCH SEEN -> the host formula is the bug")
