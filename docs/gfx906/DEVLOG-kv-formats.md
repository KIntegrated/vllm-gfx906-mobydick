# DEVLOG — upstream KV-cache formats on gfx906 (P32-2/#61, P32-5/#64)

**VERDICT: a gfx906 path *does* exist and it is upstream's, not ours.** The
`TRITON_ATTN` backend — the sole implementation of the per-token-head formats —
runs unmodified on this box, and the formats buy large capacity for small
quality loss:

| arm (dense 27B, util 0.95, maxlen 32768) | PPL | vs its own fp16 arm | KV pool |
|---|---|---|---|
| our custom FA, packed fp16 cell | **14.3750** | — | 132,796 tokens |
| `TRITON_ATTN`, fp16 KV (`auto`) | **14.3930** | — | 105,130 tokens |
| `TRITON_ATTN` + `int8_per_token_head` | **14.4008** | +0.05 % | **179,053 tokens** |
| `TRITON_ATTN` + `int4_per_token_head` | **14.5714** | +1.24 % | **296,839 tokens** |

All four arms: 0 top-20 misses on the probe's fixed 12-prompt set.

## What this means

- **The format work in LAB-5 does not need to be written.** `int8_per_token_head`
  is +70 % capacity over the *same backend's* fp16 KV and +35 % over our custom
  FA's packed cell, for +0.05 % PPL. `int4_per_token_head` reaches 296,839
  tokens (+182 % vs triton fp16, +124 % vs our FA) for +1.24 % PPL.
- **The cost is the backend, not the format.** Upstream's per-token-head dequant
  lives inside `triton_attn`; our custom FA reads its own q8_0 cell. So adopting
  int8 KV today means running `TRITON_ATTN`, i.e. giving up whatever decode
  throughput our FA has over it — and that is the one number this
  measurement does not contain.
- The two PPL baselines differ (14.3750 custom FA vs 14.3930 triton) because
  they are different softmax accumulation orders with fp16 KV. Comparisons must
  be *within* a backend; the vs-own-arm column above is the meaningful one.

## Remaining lever (decides both issues)

Decode throughput, custom FA vs `TRITON_ATTN`, at B=1 and B=4 on the same model
and context. Then:

- if `TRITON_ATTN` decode is within a few percent, **adopt it with
  `int8_per_token_head`** and delete LAB-5's format plan (P32-2 closes as
  "adopt upstream");
- if it is materially slower, the fork's own path is to port int8 (and then
  int4 with Hadamard-rotated keys, P32-5) *into* the packed cell — for int8 the
  dequant is a per-32-group scale multiply in the tile load, which is the cheap
  end of that work.

`ppl_probe.py` gained `BENCH_KV_DTYPE` and `BENCH_ATTN_BACKEND` to make these
arms runnable; that is how the table above was produced.
