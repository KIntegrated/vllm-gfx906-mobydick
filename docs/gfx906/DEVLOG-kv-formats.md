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

## Production-shaped batched-tokens sweep at 16k with MTP: small effect, and my hypothesis was wrong

Same 16k prompt, MTP k=3, ladder `[4,8,12,16]`, mclk verified at 1000 MHz, three
reps per arm:

| `--max-num-batched-tokens` | agg t/s | TTFT |
|---|---|---|
| 4096 (current recipe) | 5.20 | 41.60 s |
| 8192 | 5.27 | 41.15 s |
| 16384 | **5.39** | **39.2 s** |

So the production-shaped chunk size is **16384**, and it is worth **+5.8 % TTFT /
+3.6 % agg** over 4096 — real, but nowhere near the 2.9x I attributed to it. The
earlier claim in this log that "`--max-num-batched-tokens 4096` costs ~2.9x TTFT
at 16k" is **falsified**: 4096 vs 16384 is 41.60 vs 39.2 s, not 41.6 vs 14.5.

The remaining gap to the in-process measurement (14.52 s TTFT, no spec decode,
default batching) must therefore come from **MTP/spec decode at long context**,
not from chunking. That is the next control: identical serving flags with
`--speculative-config` removed. Until it runs, the honest statement is "enabling
MTP k=3 costs ~2.7x TTFT at a 16k prompt on this stack, cause not yet isolated" —
which is itself the more important finding, since MTP k=3 is our default recipe.

## TTFT: it is not MTP, not the chunk size, and not the serving flags

Three-way isolation, 16k prompt, decode window mclk-gated at 1000 MHz:

| path / config | TTFT | decode t/s |
|---|---|---|
| **serving**, maxlen 32k, max_seqs 4, bt 16384, MTP k=3 | 39.2 s | (agg 5.39) |
| serving, same flags, **no MTP** | 39.3 s | (agg 4.51) |
| **in-process**, same flags (`_bench_longctx`, maxlen 32k, max_seqs 4, ladder, bt 16384) | **19.61 s** | 9.88 |
| in-process, plain flags (maxlen 17k, max_seqs 1) | 19.87 s | 14.04 |

Conclusions, in the order they were earned:

1. **MTP does not affect prefill** (39.2 vs 39.3 s) — as expected; it lifts aggregate
   throughput by +19 % at 16k, which *is* its job. It appears to cost nothing at
   prefill and to help decode in proportion to content, as it should.
2. **The serving flags are not the cause of the TTFT gap**: identical flags
   in-process give 19.61 s against the serving path's 39.2 s. (They do cost
   **decode**: 14.04 -> 9.88 t/s, i.e. -30 %, from padding to
   `max_num_seqs=4`/`maxlen 32k` with a single request — worth a separate look.)
3. **The remaining ~2x lives in the serving path or in the client's TTFT
   definition.** Prime suspects, in order of cheapness to test: the client's
   first-SSE-chunk timer (queueing, chat-template application, and one decode
   step are all inside it) versus a `max_tokens=1` `generate()`; then the
   chat-template/tokenization of a 16k raw-text prompt; then engine-side
   per-request setup.
4. **Inter-run drift is real and large**: the *identical* plain in-process config
   measured 14.52 s TTFT earlier in the day and 19.87 s now (+37 %), with decode
   18.70 -> 14.04 t/s (-25 %). So the original "2.7x serving penalty" was
   three effects added together — drift, flags, and path — and only the path
   term survives at ~2x. Long-context numbers on this box must be compared
   same-session, ideally interleaved, or they mislead by tens of percent.

## **Engine-log throughput is not a measurement instrument** (permanent, 2026-10-23)

While isolating the serving-path TTFT (P32-10/#70) the engine contradicted the
client outright, and its own numbers were not self-consistent:

* 16k prompt, serving path: engine logged **1121 tokens/s prompt throughput**
  (⇒ 14.6 s) while the client measured **39.2 s** to first token, twice.
* The *same configuration* re-run logged **2.7 / 21.1 / 287.9 / 574.4 tokens/s**
  in successive lines — never 1121 — with client TTFT unchanged.
* Client TTFT scaled linearly and sanely with prompt length, which is what a
  prefill measurement should look like:
  **256 tok → 1.14 s, 4k → 10.47 s, 8k → 19.03 s, 16k → 39.2 s** (≈ 415 tok/s,
  plus ~1 s fixed). Chat vs raw `/v1/completions` differed by 0.01 s and
  `max_tokens` 1 vs 256 by 0.04 s, so TTFT is prefill-bound and the endpoint is
  not the variable.

Rule, now also in `/local/git/AGENTS.md` so every future session sees it: **do
not time anything with the engine's throughput log lines.** They exclude
inter-step gaps, they are not comparable across runs, and they can be off by
2.7x against wall time. Use an external clock around the request (streaming
first chunk for TTFT), CUDA events for GPU work, or `/metrics` histograms.

Consequence for this file's earlier entries: any figure quoted from an engine log
here is engine-internal only. The client-side numbers in the tables above are the
ones to use.

## RETRACTION: the "serving path is ~2.7x slower at prefill" finding was a clock artifact

Chased the divergence properly and it dissolved. Same build, same flags, same
11,201-token prompt, **same process**, consecutive samples:

| sample | TTFT | mclk during prefill | decode |
|---|---|---|---|
| 0 | **39.84 s** | **350 MHz** (idle state) | 0.00 t/s |
| 1 | **4.36 s** | 800 MHz | 19.63 t/s |

**A 9x TTFT swing from memory-clock state alone.** Everything I had attributed to
the serving path, MTP, chunk size, `max_num_seqs`, `maxlen`, the capture ladder,
the endpoint or multiprocessing was noise on top of that.

How the error was made, so it is not repeated:

1. My preload ran *before* the request, but a 11k-token prefill takes long enough
   (20-40 s at a low clock) that the governor drops back mid-request;
2. the mclk gate therefore has to cover the **prefill window**, and on the
   serving runs I checked the clock *before* the request, which proves nothing;
3. the in-process and serving arms also used different prompts (9.2k tokens
   synthesised vs 11.2k from the file), adding a genuine ~20 % on top.

Also retracted from earlier in this file: the client-observed prefill rate of
"~415 tok/s" — at 800 MHz the same harness does **2,570 tok/s** (11,201 tokens in
4.36 s). Every prefill throughput number in this log that was not accompanied by
a prefill-window clock reading is unusable, including the
`--max-num-batched-tokens` sweep (+5.8 % claim) — that sweep's arms ran at
verified 1000 MHz *before* each request but not during the prefill, so it needs a
re-run before the 16384 recommendation is believed.

The rule now lives in `/local/git/AGENTS.md` alongside the engine-log warning.
What survives from this whole detour: the **client-side** decode measurements
(gated during the decode window, mclk 1000) and the backend comparisons that used
them — short context (upstream +0.8-2.6 %), 16k decode (CUSTOM +15 %), and the
decode-window numbers in the int8/int4 capacity table.

## The clock is not stuck — the 39.8 s was a one-time cold start

Six consecutive identical 11,201-token prefills in one process (same flags):

| sample | TTFT | prefill mclk | decode mclk |
|---|---|---|---|
| 0 (first request after engine load) | **39.83 s** | **350 MHz** | – |
| 1 | **4.34 s** | 800 MHz | 1000 MHz |
| 2-5 | 4.35 s | 800 MHz | 1000 MHz |

So the governor is not stuck at 350 MHz; it ramps after the first request and
then **holds 800 MHz through sustained prefill** (1000 during decode). The 39.8 s
was the *cold-start* case only, and it is what every serving arm in this log
measured: each serving run sent two short warm-up prompts (64 tokens), which are
too brief to ramp mclk, so request #1 of the actual measurement ran cold. The
in-process runs did a 20 s GEMM preload and therefore did not.

That fully closes the retraction above: the "serving path 2.7x slower" finding was
cold start, reproduced in-process (39.83 s) as soon as the preload was removed
from the sampled window.

**Two open questions once the perf level can be pinned** (`gfx906_perf.sh on`,
needs root):
1. does pinning remove the cold-start 39.8 s → 4.3 s transition entirely (i.e.
   make request #1 as fast as request #2)? That is worth ~35 s of first-token
   latency per engine start and per idle-to-active transition;
2. does prefill run faster at a pinned 1000 MHz than at the self-selected 800
   (a 25 % clock step; decode already reaches 1000 on its own)?

## FINAL: the 39.9 s is a one-time first-call cost, not the clock (second retraction)

Pinned the perf level (`gfx906_perf.sh on`, mclk verified **1000 MHz during the
slow sample**, sclk 1800) and re-ran:

| sample | TTFT | prefill mclk |
|---|---|---|
| 0 (first after load) | **39.90 s** | 1000 MHz |
| 1-4 | 4.34-4.35 s | 1000 MHz |

Pinning changed nothing. Then, with a **representative full-length warm-up**
instead of a 64-token one:

```
warm-up (one 11k-token prefill): 39.86 s
sample 0: 4.35 s  at mclk 350 MHz   <- idle clock, full speed anyway
sample 1: 4.35 s  at mclk 800 MHz
sample 2: 4.35 s  at mclk 800 MHz
sample 3: 4.36 s  at mclk 800 MHz
```

So:

* **prefill is insensitive to mclk at this shape** — 4.35 s at 350, 800 and 1000
  MHz alike;
* the 39.9 s is a **one-time first-call cost for that shape** (lazy JIT/compile
  or first-touch allocation), which a full-length warm-up absorbs entirely;
* my previous entry in this file ("9x TTFT swing from clock state alone") was a
  **correlation, not a cause**: a cold process has both a low clock *and* the
  first-call cost, and I changed the clock without controlling for the repeat.

Corrections to the record, in order of the mistakes made:

1. "serving path is 2.7x slower at prefill" — wrong (compared first-request
   serving against warmed in-process, and different prompt sizes).
2. "the difference is mclk state, 9x" — wrong (clock was a bystander).
3. **correct**: first call for an 11k-token prefill costs ~35 s once per process;
   every subsequent call is 4.35 s regardless of clock.

Actionable outcomes, both real:

* **measurement**: never compare a first request with later ones; warm up at the
  length being measured (`_bench_longctx.py` now takes `BENCH_PROMPT_FILE` so
  both paths run the byte-identical prompt);
* **serving**: our current warm-up (two 64-token requests) leaves ~35 s on the
  user's first long-context request. A single full-length warm-up at startup
  would move that cost to boot, where it belongs. Worth doing in the recipes.

`gfx906_perf.sh` stays as a tool (it works, sudoers-gated, and is the way to
control the clock when a workload *is* bandwidth-bound) but it is not a fix for
this.

### P32-10(c) — the −30 % decode cost was cudagraph ladder padding, not `max_num_seqs`

**VERDICT: `max_num_seqs` 1→4 costs nothing (28.03 → 28.01 t/s, −0.1 %). The
−30 % pair above (14.04 → 9.88) is `max_model_len` 32k (−6.3 %) plus an explicit
`[4,8,12,16]` ladder whose smallest captured graph is B=4 (−24.4 %), because a
single request then decodes padded to four sequences. 6.3 + 24.4 ≈ 30 %.**

Dense 27B, PP 16,384 / TG 256, util 0.93, no spec decode, 2 interleaved reps
(A B C D E then E D C B A), each cell a fresh process, sample 0 discarded
(first-call prefill), every measured sample at mclk 1000 (per-sample record, not
a sysfs pre-read):

| cell | max_model_len | max_num_seqs | resolved capture sizes | decode t/s (r1, r2) | median |
|---|---|---|---|---|---|
| A | 17,408 | 1 | [1, 2] | 28.06, 28.00 | **28.03** |
| B | 32,768 | 1 | [1, 2] | 26.25, 26.26 | **26.26** |
| C | 17,408 | 4 | [1, 2] | 28.03, 27.98 | **28.01** |
| D | 32,768 | 4 | [1, 2] | 26.18, 26.25 | **26.22** |
| E | 32,768 | 4 | [4, 8, 12, 16] | 19.79, 19.84 | **19.82** |

TTFT was flat at 3.77–3.78 s in every cell; the one-time first-call prefill cost
**36.0 s in all ten runs** (further support for P32-10: it is a per-process cost,
not a function of these flags).

Readings:

- **max_num_seqs is free** (A vs C). The earlier "padding to max_num_seqs"
  attribution is retracted.
- **max_model_len 17k→32k costs 6.3 %** at fixed seqs (A vs B), independent of
  the ladder: a block/layout effect at the longer bound, still open.
- **The ladder is the mechanism** (D vs E, −24.4 %): [4,8,12,16] makes the
  smallest replayable decode batch four sequences, so a single request pays for
  three idle slots — larger than the effect it was meant to test.
- **Production recipe is unaffected:** entries round up to multiples of k+1, so
  under MTP k=3 the documented [4,8,12,16] means 1–4 *sequences* and pads
  nothing. The 24 % is specific to a non-MTP harness whose ladder starts at 4,
  which is exactly what the `_bench_longctx` arms do (speculative_config=None).

Tooling note: the mclk gate written for this sweep read `card0`, which does not
exist here (`card1`/`card2`, and the numbering moves between boots), so the gate
was inert until changed to scan `card*/device/pp_dpm_mclk` and take the fastest of
the two. The harness already records `mclk_prefill_mhz` / `mclk_decode_mhz` per
sample, and that is what the table is gated on.

### P32-10(b)(c) — batched tokens do not move sustained prefill; `max_model_len` costs ~linearly

Interleaved sweep, dense 27B, PP 16,384 / TG 256, util 0.93, no spec decode,
sustained samples only (sample 0 = the per-process first-call cost, reported
separately), per-sample mclk from the harness.

**(b) `--max-num-batched-tokens` is not a prefill lever here.** Sustained TTFT at
a 16,384-token prompt is **3.78 s at 4096 and 3.78 s at 8192** (third arm
pending). The earlier "+5.8 % from 4096 -> 16384" was **first-call variance, not
a prefill-window effect**: the first-call column moves 35.41-36.42 s across all
runs regardless of batched size, and the sustained column does not move at all.

**(c) `max_model_len` costs decode roughly linearly**, at fixed real sequence
length (16,384 prompt + 256 decode):

| `max_model_len` | decode t/s | vs 17k |
|---|---|---|
| 17,408 | 28.03 / 28.11 / 28.02 | baseline |
| 24,576 | 27.09 | -3.4 % |
| 32,768 | 26.26 / 26.22 | -6.3 % |
| 40,960 | 25.32 | -9.7 % |

Succcessive deltas are -0.94, -0.83, -0.94 t/s for +7k/+8k/+8k tokens, i.e.
~-0.118 t/s per 1024 of `max_model_len`, steady. A step would have implicated a
geometry boundary (the packed-`auto` gate was excluded: packed was forced `on` in
every cell); a clean ramp instead implicates a launch parameter derived from the
*bound* rather than the actual length. Prime suspect: the paged launcher's
`kv_split` / tile-clip choice, which is computed from `max_seq_kv`. Confirmation
is one experiment: force a fixed `kv_split` at 40,960 and see whether the arm
returns to 28 t/s. Until then this is a hypothesis, not a mechanism.

Practical reading: the fork's serving default (`--max-model-len 256k` on the
hybrid recipes) is not free — but note the ramp was measured in the *in-process*
harness with packed KV forced on, and the earlier serving measurements at 32k
did not show a comparable penalty, so it must be re-checked on the serving path
before it constrains the documented recipes.

### P32-10(a) — RETRACTED: the serving path has no first-call premium

**VERDICT: the warm-up remedy does not apply to serving. Two identical
12,754-token requests over HTTP took 51.06 s and 50.95 s — no first-call effect
at all — so the 39.9 s -> 4.35 s pattern is an in-process phenomenon.**

Server: dense 27B AWQ, `--max-num-seqs 1 --max-model-len 17408
--gpu-memory-utilization 0.93 --no-enable-prefix-caching`, no spec decode, single
tenant. Client: `_warmup_serve.py`, wall clock around the request, the reported
`prompt_tokens` confirming both calls carried the same 12,754-token prompt.

The finding this leaves behind is larger than the one it removes: serving prefill
runs at **~250 tok/s** for that prompt while the **same length takes 3.78 s
(4,330 tok/s) in-process** (P32-10(c) table). The serving path is 10-17x slower
than the engine on identical work, and it is the number a user feels. Earlier
notes already show the same regime (client TTFT scaling at ~415 tok/s against a
4k/8k/16k sweep), so this is not a one-off configuration.

Consequences:

- The `_warmup_serve.py` recipe is demoted to a **diagnostic** that distinguishes
  the two regimes in one run (second call much faster = first-call regime;
  equal = uniformly slow serving path).
- The open question for #70 is now sharply defined: **why is HTTP-served prefill
  10-17x slower than in-process prefill for the same length, with one request,
  one tenant, no prefix cache?** Candidates: chunked-prefill scheduling with
  `max_num_seqs=1`, per-chunk graph replay/dispatch overhead, or the engine
  waiting on something between chunks.
- The tool itself now reports the token count it actually reached (its first
  version under-built the prompt at 12,754 of 16,384 tokens; the guard caught it
  and exited 1 rather than reporting a fake result).
