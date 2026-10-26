# INT8-PACKED-1 — compressed-tensors `pack-quantized` W8A16 on gfx906

## 2026-09-16 — the first INT8 W8A16 checkpoint on this fork: a two-line wiring gap, then it loads and
## generates coherent output

**VERDICT:** VIABLE (numerics gate still pending) · **GATE:** `lued/Qwen3.8-27B-INT8-W8A16-DFlash2`
served at TP=2, `--dtype float16`, `--kv-cache-dtype float16` (see the trap below), 8k ctx, no spec
decode — loads with **zero skipped tensors**, allocates a 252,196-token KV pool, and answers templated
chat prompts correctly.

## HYPOTHESIS

Our fork cannot serve compressed-tensors `pack-quantized` INT8/W8A16 checkpoints (the blocker recorded
in `DEVLOG-dflash2.md` as the DFlash2 INT8 route's gate). Falsify: load one and generate.

## What was done

1. **Checkpoint contract** (`lued/Qwen3.8-27B-INT8-W8A16-DFlash2`, the VL variant
   `Qwen3_5ForConditionalGeneration`): `format: pack-quantized`, two config groups — `group_0`
   (`Linear`) and `group_embed` (`re:.*embed_tokens$`) — both **int8, symmetric, strategy `group`,
   group_size 128**; compressed-tensors schema `version 0.18.0`; `lm_head`, `in_proj_a/b`, all `mtp`
   and the whole visual tower are in `ignore`. 1986 tensors total: 401 `weight_packed` + 401
   `weight_scale` + 401 `weight_shape` + 521 `.weight` + 166 `.bias`.
2. **Scheme resolution probe** (`tools/ct_scheme_probe.py`, CPU-only): our tree already routes both
   groups to `CompressedTensorsWNA16(strategy=group, num_bits=8, group_size=128, symmetric=True)`,
   and a dedicated embedding scheme exists — `compressed_tensors_embedding.py`
   (`CompressedTensorsEmbeddingWNA16Int`), whose Triton `_dequant_gather_kernel` gathers rows by token
   id, unpacks int32-packed INT weights and dequantises in one pass, with both `GROUP_SIZE == 0`
   (channel) and group-scale paths.
3. **Root cause of the load failure was model wiring, not quantisation support:** `Qwen3_5Model`
   built `embed_tokens` as `VocabParallelEmbedding(vocab, hidden)` — no `quant_config`, no `prefix` —
   so the layer was never handed to the quant config, never registered the packed parameters, and the
   loader died with `ValueError: There is no module or parameter named 'embed_tokens.weight_packed'`.
   `VocabParallelEmbedding` resolves its method as `quant_config.get_quant_method(self, prefix=prefix)`,
   so the checkpoint's *name* patterns (`re:.*embed_tokens$`, `ignore: lm_head`) need the prefix too.
   **Fix: two lines** (AFMoe/AXK1/bailing_moe already do this):
   `quant_config=self.quant_config, prefix=maybe_prefix(prefix, "embed_tokens")`.
4. Serve the checkpoint (TP=2, 8k ctx, no spec decode) and sanity-generate through the chat template.

## Evidence FOR (what the gate measured)

- **Load:** `Application startup complete` after 520 s (cold Triton/inductor compile), **no**
  `not initialized` / missing / unexpected-key lines — i.e. every checkpoint tensor, packed ones
  included, was consumed.
- `GPU KV cache size: 252,196 tokens`; 29.1 GB resident on GPU0 (29.6 GB checkpoint at TP=2).
- Coherent templated answers (greedy, temperature 0):
  - *"What is the capital of France? Answer with one word."* -> `The user asks for the capital of
    France and requests a one-word answer. The capital of France is Paris.\n</think>\n\nParis`
  - *"Write a Python function that reverses a string."* -> opens a proper `\`\`\`python` block.
- The resolved scheme in the server log is `CompressedTensorsWNA16` (linear + embedding).
- The compressed-tensors *library* here is **0.17.0** while the checkpoint's schema says 0.18.0 —
  no incompatibility surfaced at load, so no library bump is forced.

## Evidence AGAINST / still open

- **Numerics are not gated yet.** Two coherent answers are an output-sanity gate, not a correctness
  gate. The gate to run: first-token top-k logprobs of the INT8 model against **bf16
  `Qwen/Qwen3.8-27B`** (both on `/data`) via `benchmarks/kernels/gfx906/ift_chat_gate.py` — templated,
  because this is an instruction-tuned checkpoint and prompt-logprob PPL is not a gate for IFT models.
- **Performance is unmeasured.** The only numbers seen (3.3 prompt / 1.9 generation t/s) come from the
  first request and include compile + warmup; a proper serving A/B is needed before any claim.
- **`qwen3_5_mtp.py` has the same wiring gap** (`embed_tokens` without `quant_config`): latent for this
  checkpoint (all `mtp` weights are `ignore`d), required for one that quantises MTP.
- Upstream `main` has the same gap in `qwen3_5.py` (checked 2026-09-16), so this fix looks upstreamable
  — but that also means the club-3090 INT8 recipe cannot work as written on stock vLLM; worth asking
  the other box to confirm empirically (it is in `HANDOVER-dflash2.md`'s scope).
- 8-bit `int_quantized` (the ecosystem's usual 8-bit form) and non-group strategies are untested here;
  this log covers the `pack-quantized`/group-128 path only.

## Why it failed (and why it now doesn't)

The packing, the group-128 scales and the embedding dequant-gather were all already implemented; the
model simply never asked for a quant method on its embedding. That is why the error was a *key-name*
error rather than a kernel or dtype error — the layer had no packed parameters to load into.

## Interactions / superseded-by

- Unblocks the **DFlash2 INT8 arm** (`DEVLOG-dflash2.md`, ROADMAP DFL2-1): the matched pair
  (INT8 W8A16 target + `lued/Qwen3.8-27B-DFlash2-W8`) can now be served on this fork.
- Trap worth keeping: `--kv-cache-dtype` does **not** accept `fp16` (choices: `auto`, `bfloat16`,
  `float16`, `fp8*`). Passing `fp16` makes `vllm serve` die in argparse before any logger exists,
  which is what silently killed the earlier chained DFlash2 INT8 session (empty server log, `rows=0`).
- This boot (2026-09-15 13:20) wedged once during the first INT8 load (half-wedge, `reset(4)`
  recovered, `degradation.md` #96); the authorized retry loaded cleanly, i.e. that failure was the
  boot's load lottery rather than this change.

## 2026-09-16 (promotion) — cherry-picked to `main` and re-gated there

**VERDICT:** on `main` · **GATE:** the same load+sanity gate, re-run on main's tree.

The INT8/W8A16 work (`309862a0b8` + the dev-log correction `20f356dbc3`) was cherry-picked from
`gfx906/int8-packed` onto `main` (clean, no conflicts; plus a follow-up fixing the lint fallout in
`tools/ct_scheme_probe.py`, which had been created by header injection into a scratch script).
Re-gated on main: `Application startup complete` after 300 s, `GPU KV cache size: 240,493 tokens`,
no skipped/unexpected-key lines, and the same two coherent templated answers as on the branch
("...The capital of France is Paris."; a fenced python block). The DFlash2 records stay on
`gfx906/dflash2`; only the quantisation work is promoted.

## 2026-10-26 — INT8-PACKED-1 closed

**VERDICT: the checkpoint is served and gated; the MTP wiring gap is fixed but
latent because this checkpoint ships *no* MTP weights; and the DFlash2-INT8
pairing arm is therefore not runnable as the issue imagined.**
**GATE:** templated chat gate (per-prompt first-token logprobs) + a decode number.

### The MTP wiring gap, fixed

`qwen3_5_mtp.py` built its `embed_tokens` as
`VocabParallelEmbedding(vocab_size, hidden_size)` — no `quant_config`, no
`prefix` — while the main model passes both (`qwen3_5.py:251`). On a
compressed-tensors pack-quantized checkpoint the embedding carries
`weight_packed` / `weight_scale` / `weight_shape`, so without the wiring those
parameters never register in the MTP module and every MTP embedding weight is
skipped silently. Fixed by mirroring the main model.

**Safe for the AWQ line**, verified two ways rather than assumed: the main path
already passes `quant_config` for an embedding that is *plain*
(`model.language_model.embed_tokens.weight`) and *not* in the ignore list, and
compressed-tensors resolves an unquantized method for it; a full AWQ + MTP k=3
run after the patch loaded and served (pp512/tg64, 17.06 t/s, no errors).

### Why it is latent here

`lued/Qwen3.8-27B-INT8-W8A16-DFlash2` has **1986 weight keys and zero containing
`mtp` or `draft`**. So MTP drafting cannot be enabled with this checkpoint at
all, and the DFlash2-INT8 pairing arm needs a checkpoint that actually ships a
drafter head. The fix is for the next one that does.

### Gate and numbers

- **Templated chat gate** (`ift_chat_gate.py`): both prompts answer coherently
  with confident first tokens — `760=-0.72, 1421=-0.97, 1596=-2.22` for the
  capital question, `1596=-0.05` for the code request. (The checkpoint is a
  thinking-style model: within 24 tokens it plans rather than answering, which is
  why the gate reports logprobs, not the text.)
- **Decode: 13.43 t/s** at pp2048/tg256, single request, `--gpu-memory-utilization
  0.95`, mclk 1000. Against the AWQ-INT4 baseline on the same path (~25.6 t/s)
  that is the expected ~2x from doubling the weight bytes — W8A16 is a *quality*
  trade, not a speed one.
- The gate needed two knobs to run at all: `BENCH_GPU_UTIL` (default 0.90 OOMs on
  a ~27 GB W8A16 27B — "No available memory for the cache blocks") and
  `BENCH_EAGER`. Both added.

### Wart found and fixed on the way

Eager mode on TP=1 emitted the packed-KV layout warning **16 times** (once per
attention layer) because the "already warned" flag was per impl instance; the
non-eager path emits none. The flag is process-level now and the message says
what it actually means (impls built before the layout is resolved: TP workers
*and* eager mode). Re-measured: 1 warning, was 16.

## 2026-10-26 (addendum) — the drafter exists; DFlash2 x INT8 is 2.83x, and the bf16 gate passes

**VERDICT: two of this issue's "remaining" items were wrong for the same reason —
I looked for MTP weights when the DFlash2 family pairs a *separate* drafter model.
With that drafter (local, 2.1 GB) the INT8 pairing arm runs and delivers **2.83x**,
and the bf16 reference gate passes at <=0.4 nats.**

### Checkpoint census (local, `mtp`/`draft` keys counted from each index)

| checkpoint | keys | mtp | note |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16) | 1199 | **15** | the issue's reference; 52 GB, needs TP=2 |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 2396 | **15** | the AWQ line (regression arm) |
| `lued/Qwen3.8-27B-INT8-W8A16-DFlash2` | 1986 | **0** | the INT8 target |
| `lued/Qwen3.8-27B-DFlash2-W8` | — | — | **`DFlash2DraftModel`**, bf16, 2.1 GB — the drafter |

So the INT8 target has no MTP head *by design*: DFlash2 does not use one. Both
facts recorded earlier ("MTP drafting cannot be enabled with this checkpoint",
"the pairing arm needs a checkpoint that ships a drafter head") were wrong in
their conclusion — the drafter is a separate model and it is here.

### The bf16 numerical gate (the issue's own wording: "vs bf16 Qwen/Qwen3.8-27B")

`ift_chat_gate.py`, templated, 24 tokens; bf16 on **TP=2** (52 GB of weights),
INT8 W8A16 on TP=1 util 0.95:

| prompt | bf16 top-5 | INT8 W8A16 top-5 |
|---|---|---|
| capital of France | 1421=-0.66, 760=-0.94, 1596=-2.59, 90700=-3.98, 81266=-7.07 | 760=-0.72, 1421=-0.97, 1596=-2.22, 90700=-3.95, 81266=-6.75 |
| add two numbers | 1596=-0.06, 1421=-3.08, 760=-4.62, 22916=-6.03, 90700=-7.98 | 1596=-0.05, 1421=-3.45, 760=-4.55, 22916=-6.11, 90700=-8.37 |

Same token sets, same ordering apart from a near-tie at the top of the first
prompt (bf16 -0.66/-0.94 vs INT8 -0.72/-0.97), largest logprob delta **0.4 nats**,
and near-identical continuations. The pack-quantized embedding plus W8A16 GEMMs
are numerically sound against the bf16 reference.

### The pairing arm (the issue's original purpose)

Target INT8 W8A16 + `lued/Qwen3.8-27B-DFlash2-W8`, `{"method":"dflash",
"num_speculative_tokens":7}`, pp1024/tg128, TP=2, util 0.92, mclk 1000:

| arm | t/s |
|---|---|
| no spec decode (control, 2 samples) | **8.97 / 8.97** |
| DFlash2 k=7 | **25.41** |

**2.83x.** Two constraints worth recording: **TP=1 OOMs** (28 GB target + 2.1 GB
drafter on a 32 GB card), so the pairing is a two-card configuration; and the
control is sequential rather than interleaved (single sample on the spec arm), so
the usual order caveat applies — 2.83x is far outside this box's 2-4 % order
variance, but an interleaved re-run is what would make it a record number.
Acceptance was not measured (no per-request spec counters in the in-process
harness); the ratio is end-to-end, so it includes acceptance.
