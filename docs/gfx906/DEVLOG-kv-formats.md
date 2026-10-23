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

## Decode throughput: the missing arm, and it does not favour our FA

Method: `_bench_gfx906.py` with a decode-shaped workload (256 in / 256 out, B=1,
27B dense, `BENCH_PRELOAD=25` before the timed window), arms interleaved with the
first arm repeated last, every sample mclk-gated.

| arm | t/s (5 samples) | mean | mclk |
|---|---|---|---|
| our custom FA, packed fp16 cell | 29.079 / 29.101 / 29.122 / 29.110 / 29.093 | **29.10** | 1000 MHz, all |
| `TRITON_ATTN` + `int8_per_token_head` | 29.473 / 29.496 / 29.491 / 29.471 / 29.465 | **29.48** | 1000 MHz, all |
| our custom FA, repeated (order control) | 29.067 / 29.087 / 29.050 / 29.068 / 29.040 | **29.06** | 1000 MHz, all |

**VERDICT: there is no decode-throughput uplift to claim.** Upstream's
`TRITON_ATTN` with int8 KV is **+1.3 %** over our custom FA (29.48 vs 29.10), and
the order control reproduces our arm to 0.1 % (29.10 vs 29.06), so the delta is
outside the method's noise. Both formats were also quality-checked earlier in
this log (int8: +0.05 % PPL, +70 % KV capacity at 32k).

Two things this does *not* say. It is one workload — B=1 decode on a dense model
with a 1k effective context. Our FA's documented advantages live elsewhere
(packed-KV capacity at long context, MTP verify shapes, B>1, the 35B MoE
geometry), and none of those were re-measured here. So the honest reading is
"upstream's path is competitive enough that retiring ours needs a decision, not
an assumption" — not "our FA is pointless".

Method notes, since they cost two discarded runs: a 16k-prefill workload makes
this metric measure prefill (1.97 t/s, 65 s per sample), and a prefill-heavy
window holds mclk at 800 MHz, so the decode-shaped workload is what makes the
1000 MHz gate pass at all. `attention_backend` (kwarg) conflicts with the fork's
`attention_config.backend`; `VLLM_ATTENTION_BACKEND` is the form that works.

## The arms where CUSTOM was supposed to win — it does not

Three workloads, `_bench_gfx906.py`, 256 in / 256 out, prefix caching off,
`BENCH_PRELOAD=20`, 3 samples per arm, every sample at mclk 1000 MHz. Arms:
CUSTOM (packed FP16 cell) vs upstream `TRITON_ATTN` + `int8_per_token_head`.
B=4 and B=1-dense carried an order control (CUSTOM repeated last).

| workload | CUSTOM | upstream triton+int8 | delta |
|---|---|---|---|
| dense 27B, B=1 | 29.10 (control 29.06) | **29.48** | +1.3 % upstream |
| dense 27B, **B=4** | 61.77 (control 61.46) | **62.28** | +0.8-1.3 % upstream |
| **35B MoE**, B=1 | 89.60 | **91.90** | **+2.6 % upstream** |

The 35B MoE order control aborted (rc=134, the documented abort-after-work
pattern) and is missing; the other two controls reproduce CUSTOM to 0.1-0.5 %.

**VERDICT: on every workload measured so far, the fork's custom FA has no
throughput advantage, and upstream's Triton path with int8 KV is 0.8-2.6 %
faster while also carrying +70 % KV capacity at 32k (+0.05 % PPL).** The
advantages the README attributes to CUSTOM -- GQA head-packing, KV split,
fused gather-and-quantize -- are measured *against the 0.28-era fallback*, not
against this backend; against `TRITON_ATTN` they do not show up at 256-token
context, at B=4, or on the MoE geometry.

**Not yet measured, and therefore not yet a retirement decision:**
- **long context** (16k+), where the KV-split/packed trade and int8 density
  could separate the two; this needs the prefill-vs-decode separation the
  harness does not currently do (a 16k prefill window measures prefill and pins
  mclk at 800 MHz, so it must be measured as TTFT + decode rate, not t/s);
- **MTP/DFlash verify shapes**, where the fork's spec-decode path depends on the
  custom FA's verify tile and int8 KV may change acceptance;
- **Lightning (hybrid) and Muse-Glimmer**, which have their own recipes and
  model-specific caveats.

Until those three are known, the defensible action is to stop *adding* to the
custom FA and treat it as the rollback path, not to delete it: the cost of
keeping it is maintenance, the cost of removing it early is losing the one
configuration that may still win at long context.

## Long context: CUSTOM wins, decisively

Measured with the new `_bench_longctx.py` (TTFT from a 1-token request on the
same prompt, decode rate from the TG-token request minus its TTFT, mclk sampled
and reported for the *decode window* — a 16k prefill legitimately holds 800 MHz,
so gating the whole window would have thrown away valid decode numbers).

Dense 27B, 16384-token prompt, 256 output tokens, B=1, prefix caching off, 3
samples, first arm repeated last:

| arm | decode t/s | TTFT | prefill rate |
|---|---|---|---|
| **CUSTOM** (packed FP16 cell) | **18.70** (control **18.72**) | **14.52 s** (control 14.50) | ~1128 tok/s |
| upstream `TRITON_ATTN` + int8 KV | 16.25 | 19.91 s | ~823 tok/s |

**CUSTOM is +15.1 % on decode and 37 % faster on prefill at 16k**, and the order
control reproduces it to 0.1 %. Combined with the short-context arms (upstream
+0.8-2.6 % at 256-token context), the picture is now coherent rather than
contradictory:

* **short context** — attention is a small share of the step and upstream's
  Triton path is marginally ahead (0.8-2.6 %);
* **long context** — attention dominates, and the fork's kernels (KV split, GQA
  head-packing, packed cell) are 8.2 % faster per decode step and 37 % faster at
  prefill.

**So the custom FA is worth maintaining**, and the answer to "should it be the
default" is context-dependent rather than global. The earlier "no uplift" verdict
in this log was true and incomplete: it measured 256-token contexts only, which
is where the fork's long-context machinery cannot pay. Retiring it on that basis
would have been a mistake — worth remembering the next time a three-workload
sweep looks decisive.

Still unmeasured: MTP/DFlash verify shapes, and the Lightning (hybrid) /
Muse-Glimmer recipes.

## MTP k=3 + 16k prompt: a tie, and it indicts the serving recipe

Serving both arms with the production-shaped flags (27B, `max_model_len 32768`,
`--max-num-seqs 4`, `--max-num-batched-tokens 4096`, MTP k=3, capture ladder
`[4,8,12,16]`, prefix caching off), 16k prompt, 256 output tokens, mclk verified
at 1000 MHz before each measurement:

| arm | agg t/s (3 reps) | TTFT |
|---|---|---|
| CUSTOM | 5.20 / 5.20 / 5.19 | 41.60 / 41.60 / 41.67 s |
| upstream `TRITON_ATTN` + int8 KV | 5.20 / 5.20 / 5.19 | 41.57 / 41.54 / 41.63 s |

**Identical to the noise floor.** The MTP/verify-shape arm is therefore a tie: it
does not separate the backends, so it neither supports nor weakens keeping the
custom FA. What it does separate is *this recipe from a good one*:

- The same 16k prompt measured **14.52 s TTFT (1128 tok/s prefill)** in-process
  with default batching, against **41.6 s (395 tok/s)** here. The only recipe
  difference that matters is `--max-num-batched-tokens 4096`: a 16k prompt is
  then prefilled in four 4096-token chunks, and 4096 matches no captured graph
  size (`[4,8,12,16]`), so those chunks run eager. The flag was chosen for
  **+12 % prefill at 1024-token contexts** (AGENTS.md); at 16k it costs ~2.9x
  TTFT.
- Because prefill dominates the aggregate t/s, both arms are prefill-bound and
  the attention backend cannot show — which is exactly why a "tie" here must not
  be read as "the backends are equal at long context". The clean long-context
  comparison (previous section) used a recipe without the 4096 chunking and
  found CUSTOM +15 %.

**Two follow-ups this opens**, both independent of the backend question:
1. Re-measure the serving recipe's `--max-num-batched-tokens` at long context
   (4096 vs 8192/16384) — the AGENTS note's +12 % was measured at 1024, and the
   number here suggests it inverts badly at 16k.
2. Repeat the MTP arm once that is fixed, so the arm measures the backend rather
   than the chunking.
