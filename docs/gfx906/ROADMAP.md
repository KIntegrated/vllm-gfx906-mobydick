# gfx906 roadmap — open work, priority-ordered

The single active queue. Completed work → [`CHANGELOG.md`](CHANGELOG.md);
parked/reopenable work → [`REFRIGERATOR.md`](REFRIGERATOR.md); closed
negatives → [`DEAD-ENDS.md`](DEAD-ENDS.md). Item IDs (C*, G*, L*, N*, U*,
HK*) are stable across reorganizations — cite them, not filenames.

Reference workload: Qwen3.5-35B-A3B-AWQ on one MI50 — 40 MoE layers,
E=256, topk=8, hidden=2048, W4A16 group-128 experts; B=1 decode step
≈ 15 ms at 66.5 t/s. Priority = expected gain × confidence ÷ effort+risk;
tiers are do-order, sections within a tier are ordered the same way.

## High priority — user-requested (2026-09-12)

### DFL2-1 — DFlash2 n-gram chains: drafter-free verify blocks while a request copies its context (**HIGH PRIORITY**, Kevin 2026-09-12)

> **Work branch: `gfx906/dflash2`** (cut from `main` 2026-09-15). The DFlash2 bring-up records
> already on `main` — the `triton_matmul` 3-D / `[K, N]` fix, `DEVLOG-dflash2.md` and the DFL2-*
> items — are shared; the DFlash2 feature work (DFL2-3 lookup-drafting → DFL2-1 chains, DFL2-7
> GEMV coverage) proceeds on this branch until it passes its gates.
>
> **The pairing test is blocked on quantisation, not on DFlash2** (2026-09-16): the matched
> INT8 target (`lued/Qwen3.8-27B-INT8-W8A16-DFlash2` + `…-DFlash2-W8`) is a compressed-tensors
> `pack-quantized` checkpoint our fork cannot load → **INT8-PACKED-1**. The cards' own matched
> bf16 pair on a stock vLLM remains the cheap decisive test.

**Kevin 2026-09-12.** Port `patches/dflash2-ngram-chains.patch` from
`../qwen38-27b-rtx3090` (`VLLM_DFLASH2_CHAIN=1`): while a request keeps
reproducing its context, whole verify blocks are proposed from the request's
own token history and **the drafter's forward plus its graph replay are
skipped** until the first rejected token. Upstream evidence: **+7 % on the copy
cell** (256.9 → 276 tok/s at `DFLASH_TOKENS=7`), flat on prose, greedy-only by
default (`VLLM_DFLASH2_CHAIN_GREEDY_ONLY=1`), requires `LOOKUP=1`.** Its precondition is
now met: the patch forces the V2 runner, and V2 is validated as the default on this line
(DFL2-2).

**ENQUEUED 2026-09-15 (tonight, Kevin) — with subtasks.** Ready pieces, verified:

- patch: `../qwen38-27b-rtx3090/patches/dflash2-ngram-chains.patch` (15,963 B) — touches
  **one file** (`v1/worker/gpu/spec_decode/dflash2/speculator.py`);
- in-tree DFlash2 support: `DFlash2DraftModel` is registered (`registry.py:615` →
  `qwen3_dflash2::DFlash2Qwen3ForCausalLM`), `DFlash2Speculator` exists,
  `_is_dflash2_draft()` forces V2 (validated, DFL2-2);
- drafter `incoai/Qwen3.8-27B-DFlash2` downloaded (3.6 GB, RC=0): **5 layers, hidden 5120,
  32 heads / 8 KV, head_dim 128, `is_causal: false`, sliding_window 2048** — i.e. a small
  **bidirectional** drafter, and its speculator's chain/lookup machinery is **Triton**
  (`@triton.jit` in `dflash/speculator.py` and `dflash2/speculator.py`);
- serving usage (from the model card):
  `--speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'`.

**Subtask 0 — DFlash2 *without* the patch vs MTP k=3 (do this first; it is interesting on
its own).** Record whether plain DFlash2 already beats MTP k=3 on gfx906: drafter is 5
layers against the MTP head, and it drafts a block in parallel, so the case for it is
independent of the n-gram chains. Serve the dense 27B with the config above vs MTP k=3,
same boot, **interleaved arms**, agentic corpus (64k/120k), ms/step lead + per-rep
acceptance + t/s; also with and without CAT-1 (both cut drafter work, so they interact).
Reference: MTP k=3 agentic on this line is 33.30/24.54 (V1) and 33.62/23.75 (V2).

**Subtask 1 — kernel attribution (what actually costs time).** Profile one DFlash2 decode
step with **rocprofv3** (the only per-kernel GPU method on this box — torch-profiler GPU
domains are absent on this build; skill `gfx906-rocprofv3-kernel-trace`) split by phase:
draft forward (5 layers, 32/8 GQA, D=128, bidirectional + sliding-2048), verify (target
shapes, our FA), and the chain/lookup/sampling Triton kernels. Output: top kernels by
total time, with the exact serving config.

**Subtask 2 — microbenches of the top kernels.** Standalone, at the real shapes, with the
mclk gate (`docs/gfx906/dvfs-mi50.md`, skill `kernel-microbenchmark`) — and remembering the
standalone-≠-production rule: bench the *dispatched* kernel, not a lookalike.

**Subtask 3 — HIP feasibility for those kernels (Kevin's question).** For each top kernel:
(a) does the fork already have it? Our custom FA serves **bidirectional** attention at
D=128 (VIT-1 proved that path) and the dense GEMV / spec-GEMM-M4 family covers small-M
GEMMs — so the first question is *which attention backend the drafter selects* (`Using …
backend` log line; if it lands on TRITON_ATTN while our CUSTOM FA would serve it, that is a
VIT-1-style wiring win, and the sliding-window layers need the `window`/`q_abs_offset`
path); (b) for kernels with no in-tree equivalent (the Triton chain/lookup helpers), assess
a HIP port: they look like small index/dedup kernels — cheap to port, but likely not
time-dominant, so subtask 1 decides whether it is worth it. *Rule: no HIP-port decision
without a profile — the fork's history is full of standalone numbers that did not transfer.*

**Session design: subtask 0 and the patch A/B are ONE session — and CAT-1 is why it is
*not* a factorial.** Verified in code (2026-09-15): our CAT-1 shortlist is loaded inside
`qwen3_5_mtp.py` from `mtp_draft_vocab_ids.pt`, i.e. it applies to the **MTP** drafter only;
DFlash/DFlash2 use their own *trained* draft vocabulary (`draft_vocab_size`, default
`vocab_size`) and separate block machinery (`dflash_config`: `block_size 8`,
`selector_rank 256`, `mask_token_id` 248070). So a "DFlash2 + CAT-1" cell does not exist,
and one shared boot answers both questions against a common reference:

| arm | config | answers |
|---|---|---|
| A | MTP k=3, plain | reference (no extra machinery) |
| B | MTP k=3 + CAT-1 | **production baseline** for Q1 |
| C | DFlash2, no patch | **Q1** (DFlash2 vs MTP) |
| D | DFlash2 + chain patch | **Q2** (patch on/off) |
| E | A repeated last | order control (arms run sequentially; acceptance is chaotic) |

The patch is **7 Python hunks in one file** (320+/1−), so C and D can be swapped with **no
rebuild** — but note it **does not apply cleanly** to our tree (`git apply` fails at
`dflash2/speculator.py:2`: the RTX3090 fork's file has diverged), so arm D needs a manual
port. **Sequence: run A, B, C, E first** (that is subtask 0 = Q1, tonight's question) and
add D once the port is done.

**CAT-1 does NOT come from the DFlash line (checked 2026-09-15).** Our CAT-1 is upstream's
`qwen3_5-mtp-draft-vocab.patch` — an **MTP** patch — so there is no upstream CAT-1-for-DFlash
to port. The related upstream DFlash2 work is a *different mechanism* with the same goal
(draft without paying a full drafter forward): **`dflash2-lookup-drafting.patch`** (178 KB —
drafts from the request's own context via `dflash2/lookup.py`, deciding the next block length
from emitted/rejected counts). Verified that its `vocab_size`/`VocabParallelEmbedding`
references are the model's own embedding table, *not* a corpus shortlist.

**Wedge-efficient measurement plan (Kevin, 2026-09-15).** Every arm is one server load and
therefore one draw in the boot's wedge lottery, so do not pay for a live baseline unless the
answer needs it. DFlash2 **forces V2**, so the production bar is already on record:
**MTP k=3 + CAT-1 on V2 = 35.44 / 24.61 t/s** (64k/120k, 3-rep, 0.29 line) with plain MTP k=3
at 33.62 / 23.75. Boot-to-boot spread for these arms is ~1–3 % (parity restamps: MoE 58.36 vs
58.43, dense 24.90 vs 24.82).

**RESULT (arm C complete, 2026-09-15)** — DFlash2 without the patch is **2.48/2.52 t/s at 64k**
(acceptance 0.045/0.063) and **1.37/1.37 t/s at 120k** (acceptance **0.0/0.0**), at 422/728 ms per
step: ~14x behind MTP k=3, with drafts that stop being accepted entirely by 120k. The live control
(arm B, MTP k=3 + CAT-1, same boot) reproduced the recorded band within **+0.8 %/+0.4 %**
(35.97/35.44 at 64k, 24.58/24.82 at 120k), so the historical anchor is validated and the negative
result is sound. `DFL2-8` (the drafter's attention backend: upstream ROCM_ATTN + a Triton fallback,
our CUSTOM rejected) is the gate for the whole family — no downstream patch closes a 14x gap.
Details: [`DEVLOG-dflash2.md`](DEVLOG-dflash2.md).

1. **Run arm C alone first** (DFlash2, no patch; 1 load). If it lands outside the MTP+CAT-1
   band by more than ~3 %, that is the answer for Q1 — citation-grade with the boot caveat
   stated, and no extra wedge draws. *(Agreed with Kevin 2026-09-15: **C → D → B** if the
   session survives. D before B because the patch A/B is what same-boot pairing protects, while
   the method question has a historical band to lean on. The live A+B re-measure is queued for
   the next boot if tonight's session runs clean.)*
2. **Only if it lands inside the band**, escalate to the interleaved 3-arm design
   **C → B → C** (B = MTP k=3 + CAT-1, C repeated last as the order control) so the
   comparison is same-boot and same-order-position. Plain MTP (arm A) is *not* run live: it is
   not the config we would serve, and its numbers are on record (V2 33.62 / 23.75).
3. ~~Arm D (DFlash2 + chain patch) joins once the manual port lands~~ — **D is blocked on
   DFL2-3, discovered while preparing the port (2026-09-15):** the chain patch does
   `from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import …` and itself warns
   `VLLM_DFLASH2_CHAIN=1 needs VLLM_DFLASH2_LOOKUP=1; disabling`. Our tree has **no
   `dflash2/lookup.py`** (the dir holds only `__init__.py` + `speculator.py`), and core vLLM's
   ngram proposer (`v1/spec_decode/ngram_proposer*.py`) is a different mechanism. So the port
   order is **DFL2-3 (lookup-drafting) → DFL2-1 (chains)**, and D cannot run tonight.
   Second finding: both patches are written against a **~480-line** `dflash2/speculator.py`
   while ours is **217 lines** (upstream refactored the speculators for 0.28/0.29), and the
   lookup patch also touches the model (`qwen3_dflash2.py`: `_dense_kv_rows`,
   `VocabParallelEmbedding`, masked-query slots). Both are **adaptations, not `git apply`** —
   realistically multi-session work, so DFL2-1's ETA now sits behind DFL2-3.

**DFlash2 patch family as later follow-ups** (all from `../qwen38-27b-rtx3090/patches/`):
`dflash2-lookup-drafting.patch` (context-drafting, 178 KB — potentially the biggest win, and
it is the mechanism Kevin remembered as "CAT-1 for DFlash"), `dflash2-prewarm.patch` (37 KB —
compile/capture prewarm, a boot-time win we care about given the Triton/inductor stalls),
`dflash2-z-adaptive-emitted.patch` (1.7 KB — adaptive block length from emitted tokens), and
the *idea* of applying our own CAT-1 shortlist to the **DFlash2 drafter's** head (its
`draft_vocab_size` is `None` → the full 248 k vocab, so the same ~2.5 ms/step head-restriction
saving that MTP got should apply; the shortlist loader would have to move out of
`qwen3_5_mtp.py`, and the exactness argument carries over unchanged because rejection sampling
uses the distribution the draft was sampled from).

*Metric nuance for a method-vs-method comparison:* unlike a same-config build A/B, the two
methods use **different draft depths** (DFlash2's recommended `num_speculative_tokens=7`
vs our MTP k=3), so "lead with ms/step" is not the whole story — report **t/s at each
method's recommended config** (the decision metric), with ms/step *and* acceptance as the
supporting detail that separates a cost effect from a depth effect.

### DFL2-2 — V2 runner up to speed on gfx906 (**V1 removal lands in 0.32.0**) (**HIGH PRIORITY**, Kevin 2026-09-12)

**STATUS 2026-09-15 — the bring-up is DONE for every model whose gate exists; one
pin left.** V2 is validated and default for the dense 27B, MoE 35B, Nemotron 3.5
Lightning, Ornith, **and Gemma-4** (gated today through its chat template — see
GEMMA4-1); evidence and numbers in [`V2-bringup.md`](V2-bringup.md) (PPL bit-identity,
in-process bench parity, agentic ms/step parity, MoE +0.9 %). **Muse-Glimmer is the only
remaining V1 pin** (MUSE-1; its templated gate is in flight, and its PPL probe is
inapplicable by construction — see the prompt-format note in `README.md`). Upstream
removes the V1 runner in **0.32.0** (Kevin 2026-09-15), so that pin is the deadline item.

**Why now.** vLLM 0.29.0 makes the V2 model runner the default; DFlash2 and DSpark drafts
**force V2 today** (`config/vllm.py:642`). Every gfx906 optimization and serving gate on
record — custom FA backend metadata, GDN/mamba ops (incl. the SYV-10 bounds
port), mamba state-pool sizing, the trimmed capture ladder, default-ON FIX-H2
and M3 — has only ever been validated on **V1**. V2 carries a `mamba_hybrid`
model state, so it *claims* our Qwen3.5/3.8 GDN hybrids, but it has never been
measured here.

**First datapoint (in flight, 2026-09-12):** `VLLM_USE_V2_MODEL_RUNNER=1` +
MTP k=2 + the real agentic payload, one weight load
(`/local/tmp/mtp1/v2_probe_driver.sh`), compared against the V1 baseline
measured minutes earlier on the same boot.

**Work items if it loads:** (a) confirm the gfx906 FA backend is actually
selected under V2 (log the attention backend name); (b) re-run the standard
gates — single-card dense-27B and MoE-35B `docs/gfx906/_bench_gfx906.py`,
then a TP=2 serving A/B at parity vs V1; (c) re-check the trimmed-capture
assumption (`cudagraph_capture_sizes`) against V2's cudagraph utils;
(d) re-validate the default-ON fixes under V2's metadata construction (V2
passes *full-length* host/device `query_start_loc` slices in `mamba_hybrid`,
unlike V1's `[:num_reqs_padded+1]` — see the review-trains T1/M3 notes);
(e) decide the fate of the branch's V2 SYV-12 wiring
(`vllm/v1/worker/gpu/model_runner.py`, archive-bound per T6) — V2 revival of
SYV-12 needs those hunks.

**Bring-up plan:** the concrete audit (what already rides shared code, the eight
gaps, the test matrix and the session order) lives in
[`V2-bringup.md`](V2-bringup.md). V2 stays pinned off until that plan's parity
steps are signed off; the first question is the fresh-boot init retry (the Y16
wedge is unresolved, not arch evidence).

**0.29.0 merge (2026-09-13, `gfx906/v0.29.0` → merge `3c445dba56`).** Upstream
now defaults V2 for **all** models (#53183) and its own ROCm V1 list covers only
DeepSeek archs, so the fork must pin V1 explicitly: every recipe carries
`VLLM_USE_V2_MODEL_RUNNER=0` (the env override wins inside
`VllmConfig.use_v2_model_runner`). Bring-up target is now 0.29.0's V2, whose
`_get_v2_model_runner_unsupported_features()` + `HAS_TRITON` gate replaces the
fork's removed `_is_default_v2_model_runner_model()` helper. Also relevant from
the release: ROCr/CLR update (#53712, graph-replay segfault fix, ~20 % TPOT
class) and the TheRock 7.14 preview (#49925).

**⚠ Re-investigate before closing (2026-09-13).** The branch currently records
"V2 forced on Qwen3.8-GDN ⇒ init wedge ⇒ unsupported-by-design". A single
`hipErrorLaunchFailure` at init is **not** architecture evidence on this box:
wedges #67–#71 hit the pristine snapshot and non-GDN work with the identical
signature (load lottery; one authorized retry normally loads clean), so the V2
conclusion needs a **fresh-boot retry** before GDN is written off — with 0.29.0
making V2 the default this is a cadence risk, not a detail. Related: the fork's
**A3 fused-draft opt-in was stripped** (2026-09-13, `VLLM_GFX906_FUSED_DRAFT`;
brief `/local/tmp/b4/a3-strip-decision.md`, archive `archive/a3-fused-draft`).
A3 is a V2-only accelerator (the fused loop lives in
`v1/worker/gpu/spec_decode/autoregressive/speculator.py`, upstream), so if V2
becomes our path the strip has to be revisited: re-add the ~7-line opt-in and
re-run the no-op contract audit at the serving k (A3's own k=4 gate was NEUTRAL,
k=7 was never measured).

**V2 revival detail — variable draft width (2026-09-13).** The branch's V2
footprint is now **zero**: the SYV-12 ext-column wiring and the two hunks it
required in V2 files — the truncating draft assign in
`vllm/v1/worker/gpu/model_runner.py`
(`draft_tokens[idx_mapping, :draft_tokens.shape[1]] = draft_tokens`, which
leaves the tail columns holding *stale drafts from the previous step*) and the
relaxed per-position assert in `vllm/v1/spec_decode/metrics.py` — are reverted
and preserved on `archive/syv12`. If a future V2 feature reintroduces a
variable draft width (renewed ext column, or upstream's per-request
adaptive-verification budgets), **do not** re-apply the truncating assign:
size the buffer to the max, zero/pad the tail (or pass an explicit per-request
count downstream), assert the width, and add a test. Upstream's whole-row
assign fails loudly on a width mismatch — a feature to keep, not to paper over.

**Gate:** same-boot V2-vs-V1 serving A/B at parity or better on both served
models; otherwise V2 readiness becomes a merge-cadence blocker at 0.29.0.

**Effort:** medium (mostly measurement + fixing whatever V2-specific gaps
appear); **risk:** low-medium (probe first, one load, no new code).

## High priority — user-requested (2026-09-10)

### DFL2-3 — port `dflash2-lookup-drafting` (draft from the request's own context)

**Status: open — and it is the *prerequisite* for DFL2-1** (verified 2026-09-15: the chain
patch imports `dflash2.lookup` and refuses to enable itself without `VLLM_DFLASH2_LOOKUP=1`,
while our tree has no `lookup.py`). Touches the model too (`qwen3_dflash2.py`), and is written
against a speculator that upstream has since refactored (~480 lines there vs 217 here), so it is
an adaptation port. This is the mechanism Kevin remembered as "CAT-1 for DFlash": instead of a drafter forward, blocks are proposed from the
**request's own context** (`dflash2/lookup.py` picks the block length from emitted/rejected
counts) — the same goal as CAT-1 (cheap drafting on predictable text) by a different mechanism.
Largish patch (178 KB, `../qwen38-27b-rtx3090/patches/`); verified that its
`vocab_size`/`VocabParallelEmbedding` references are the model's own embedding table, *not* a
corpus shortlist. Port + the standard gates (interleaved same-boot A/B, agentic corpus, ms/step
plus acceptance, weighted to the copy-heavy end where lookup drafting should pay).

### DFL2-4 — port `dflash2-prewarm` (boot-time compile/capture prewarm)

**Status: open, queued as a follow-up (Kevin 2026-09-15).** 37 KB patch adding a prewarm set —
relevant because boot-time Triton/inductor compile and graph-capture stalls are a standing cost
here (VIT-1's own win included −55 s of cold-boot Triton JIT). Gate: boot time with fresh caches
plus a serving smoke, not a decode A/B.

### DFL2-5 — port `dflash2-z-adaptive-emitted` (adaptive block length)

**Status: open, queued as a follow-up (Kevin 2026-09-15).** Small patch (1.7 KB): the next
block's length follows the emitted/rejected counts the sampler already reports, i.e. dynamic
draft depth. Cheap to port; gate on the agentic corpus at 64k/120k with per-rep acceptance
(depth changes move tokens/step, so t/s is the metric).

### DFL2-6 — apply the CAT-1 shortlist to the **DFlash2 drafter** head

**Status: open, our own idea (2026-09-15) — not an upstream port.** The DFlash2 drafter's head is
the full vocabulary (`draft_vocab_size: null` → `vocab_size`, 248 320) — exactly the shape the
MTP CAT-1 shortlist cut by ~2.5 ms/step — so the same saving should apply. Work: lift the
shortlist loader out of its MTP-specific home (`qwen3_5_mtp.py` reads `mtp_draft_vocab_ids.pt`)
into something a DFlash2 draft model can consume too, then gate it the way CAT-1 was gated
(controlled A/B, per-rep acceptance — while remembering that CAT-1's *acceptance* effect was
retracted, so the honest expectation is the ms/step saving). Exactness carries over unchanged:
rejection sampling uses the distribution the draft was sampled from, and the target keeps its own
full head.


### DFL2-7 — cover the DFlash2 draft GEMM shapes in the gfx906 GEMV family

**Status: open — found while fixing the DFlash2 bring-up crash (2026-09-15).** The drafter's
projections are `m ∈ {256, 6144, 17408}`, `k = 5120`, `n = 2 × block_size(8) = 14` block tokens.
Our gfx906 GEMM dispatch has fast paths only for `n == 1` (LLMM1 / long-k GEMV) and `n == 2..4`
(spec-GEMV-M4), so this regime falls through to the **generic fp16 `triton_matmul`**
(`waves_per_eu=1`, no max-ilp tuning) — and, with a weight in upstream's `[K, N]` orientation, it
pays a per-call `t().contiguous()` transpose on top. Several projections × 5 layers **per draft
step**, on the critical path of every DFlash2 step. Work: extend the GEMV/skinny coverage to
`n ≤ ~16` for these `(m, k)` pairs and/or add a stride-aware variant that takes `[K, N]`
directly; then measure in serving (interleaved arms, agentic corpus). **Profile first**
(`DFL2-1` subtask 1, rocprofv3) so the target comes from measured kernel time, not shapes — see
[`DEVLOG-dflash2.md`](DEVLOG-dflash2.md).

### VIT-1 — serve the Qwen3.5-family ViT shapes from the custom FA (user request 2026-09-12: kill the Triton ViT path)

**Status 2026-09-15: DONE — default ON.** Items 1-3 landed and gated; see
[`DEVLOG-vit1.md`](DEVLOG-vit1.md). Image-prompt TTFT (fresh image per rep, same
boot, identical prompts, prefix cache OFF): **5.81 → 5.14 s @1024×1024 (−11.5 %,
3/3 reps)**, 1.71 → 1.67 s @512; 256-token probe 6.06 → 5.42 s. Fresh-boot cost
with an empty `TRITON_CACHE_DIR`: upstream **330 s** vs ours **275 s** → the ViT's
Triton JIT is **−55 s**, but ~140 s of that 195 s penalty is *other* Triton (the
LLM's GDN decode kernel is Triton), so **item 4 (dropping the triton-AMD
flash-attn package) does not follow from this win** and stays open behind the
per-model dependency audit. Two defects fixed on the way: the adapter asserted a
`[B,S,H,D]`/`B+1` layout production never passes (the VL towers pass one **packed**
`[seq_len, 1, hidden]` stream with `cu_seqlens[-1] == seq_len`, so multi-image
requests would have failed), and the decode-era `kv_split` default (32 for any
`Sq >= 4`) cost 4-9× on prefill-shaped calls until the adapter pinned `kv_split=1`
(needed an optional per-call argument on the dense binding). Follow-up queued as
**VIT-2** (head_dim-96 instantiation, ~−5 % TTFT @1024×1024, measured); an fp16-K
kernel variant (would remove the ~2e-2 Q8 error) stays on the same residue shelf.

**Kevin 2026-09-12.** Extend `gfx906_fa` (the CUSTOM attention backend) to
also serve the vision-tower shapes of the Qwen3.5 VL family so the ViT
stops going through the Triton-AMD flash-attention path. Wins: (a) **no
per-boot Triton JIT compile / graph-capture stall for the ViT** (the
`__triton_launcher.c` first-boot compile + capture adds ~tens of seconds
to every fresh boot and every fresh container), (b) **ViT prefill
performance** — the ViT runs on the critical path of every image-bearing
prompt (text+image prefill pays it), and the Triton FA is not MI50-tuned,
(c) one attention code path — the triton-AMD flash-attn package (editable
install, `flash_attn_2_cuda` build) becomes removable from the serving
deps.

Shapes (Qwen3.5/3.8 ViT, SigLIP-style, from the shipped config:
`hidden_size=1152`, `num_position_embeddings=2304` (48×48 patches),
`patch_size=16`, `spatial_merge_size=2`, `out_hidden_size=5120`,
`deepstack_visual_indexes=[]`): **full attention (no window mask in this
cfg), ~16 heads × head_dim 72, sequences ≈ 2304/merge-tile positions,
prefill-only (no decode), fp16 KV** — i.e. small head_dim, short seqs,
batch-over-image-tiles: a very different kernel regime from the LLM path
(head_dim 256, Hkv 4, 122k contexts, Q8 K).

**Recon (2026-09-13, 0.29 tree) — the kernel work is essentially zero; the effort
is an adapter + wiring:**

- **Shapes confirmed from the shipped config** (`vision_config`: hidden 1152,
  `num_heads 16`, depth 27, patch 16, merge 2, 2304 position embeddings) →
  **head_dim = 1152/16 = 72**, bidirectional (no window mask), prefill-only,
  fp16 KV, ragged batches (`cu_seqlens` / `max_seqlen`).
- **No mask or tiling work needed.** The vendored kernel masks *either* via a
  materialised mask *or* the inline-causal `q_abs_offset`; with **neither** it
  computes **full bidirectional attention** (`mask=None`, `q_abs_offset=None`,
  `window=0`, `k_VKQ_max` = per-seq length). The ViT is exactly that case.
- **A dense, non-paged entry already exists**: `gfx906_fa_forward(q_fp32
  [B,Hq,Sq,D], k_q8 [B,Hkv,Skv,D*34/32], v_fp16 [B,Hkv,Skv,D], scale, kv_max?,
  mask?, q_abs_offset?, window, kv_start?)` — no block table, plain contiguous
  K/V. `MMEncoderAttention` dispatches per backend (`forward_cuda` →
  `_forward_fa` → `vit_flash_attn_wrapper` on ROCm), so this is a new
  `_forward_gfx906_fa` arm plus a config default.
- **D=72 must be padded** (launcher requires `head_size % 32 == 0`) → pad Q/K/V
  72 → **96** (or 128 if the tile table lacks 96). Zero-padding is **exact**: the
  padded dims contribute 0 to the QK dot, they quantise to zero Q8 blocks and
  contribute 0 to P·V, and because the padding is in the *head* dim the softmax
  denominator is untouched; padded query rows are sliced off.
- **Contract detail**: the kernel wants Q in fp32 and K pre-quantised Q8, so the
  adapter reshapes ragged → `[B, H, Sq_pad, D_pad]`, zero-pads, casts Q,
  quantises K (`quantize_q8_0`), and passes V fp16.

**Work items (revised):** (1) the `_forward_gfx906_fa` adapter + `CUSTOM`
accepted by `MMEncoderAttention` and defaulted on gfx906 (env kill switch,
FLASH_ATTN fallback for unsupported shapes); (2) ncols1/kv_split tuning at D=96
using the existing ladders; (3) screens: a ViT-shaped call vs a torch-SDPA
reference (bidirectional, ≤5e-2 rel like the FA suite), the FA suite staying
green, then an image-prompt TTFT A/B **and** a boot-time measurement (killing the
Triton-AMD ViT JIT/capture stall is half the win); (4) confirm the Triton-AMD
flash-attn editable install can then be dropped from serving deps.
**Effort: low-medium** (adapter + wiring + tests; the kernel needs nothing);
**risk: low** (fallback stays).

**Dep-shedding caveat (Kevin, 2026-09-13): do NOT drop the Triton-AMD
flash-attn dependency for the ViT win alone.** It must be verified that no other
model we serve needs it — the fork serves more than the Qwen3.5 family (Gemma-4,
Muse-Glimmer, Ornith, Nemotron, and anything whose encoder path falls back to
`FLASH_ATTN`/`TRITON_ATTN` on ROCm). Audit `get_vit_attn_backend`'s ROCm
fallbacks and the mm-encoder backend list per model *before* removing the
editable install; the boot-time stall win can be banked for the Qwen3.5 family
without touching the dependency.

### TP-1 — TP-scaling probe: prefill + decode vs TP, the TP=4 question (queued after the 120k×B4 campaign)

**User request 2026-09-10.** How well do prefill and decode scale with TP;
would TP=4 pay off? Expectation to test: memory-bw-bound decode should still
improve with TP (memory access spread over the aggregate HBM of the GPUs).
**2× MI50 only → TP=4 is not available; TP=2 is the ceiling** (TP=4 answered
counterfactually). Full analysis + probe design:
[`ttft-prefill-stall.md` §13.11](ttft-prefill-stall.md).

- **Prefill** = compute-bound → scales ~linearly (measured ~2× at TP=2,
  Muse 240→500 t/s @32k). TP=4 → ~4×.
- **Decode** = memory-bw-bound; the bandwidth term (weights ~20 GB loaded LM
  [21 GB on-disk VL ckpt] + KV 64 KB/token, weight-dominated to ~234k ctx)
  **does** halve at TP=2 (expectation holds), but a large TP-invariant term
  (CPU fixed cost + 64-layer per-layer all-reduce, latency-bound at M=1)
  blunts the net to ~parity at short ctx (measured 39.74 → 39.7 t/s). KV term
  grows with ctx → scaling should improve at long ctx, but stays
  weight+fixed-bound for this model to ~234k.
- **Probe** (two wedge-light loads, GPU0 only), **1 sample/point** (Kevin's
  2026-09-10 cut, applied — the handoff is low-risk): **(a)** TP=1 greedy, no
  spec, util 0.93, B=1, at **pp ∈ {32768, 65536} × tg=256** → prefill
  TTFT/t·s (clean TP ratio vs 442/364) + decode t/s (long-ctx decode scaling,
  never measured at TP=1). **(b)** TP=1 MTP k=3, B=1, same grid → the
  **compute-regime test**: MTP verify runs M=1+k (k=3→4/req, more
  compute-bound than greedy M=1), so TP=2 MTP decode should beat TP=1 MTP by
  *more* than the greedy parity — compare vs the existing TP=2 MTP k=3 B=1
  anchors (09-09: 64k×3, 120k×2).
- **120k point DROPPED — does not fit TP=1.** 21 GB on-disk weights (LM-only
  loads ~20 GB) leave only ~6 GB KV at util 0.93 ≈ **~90k tokens** (correcting
  the earlier "~160k / fits 120k" — that assumed ~15 GB weights). 64k fits
  with headroom and still answers the scaling question. B=4 MTP is also not
  runnable at TP=1 (4×120k=480k ≫ 90k).
- **Driver ready** (`/local/tmp/b4/run_postcampaign.sh`, safe startup tears
  down the orphaned campaign server itself). **Queued after the 120k×B4
  campaign** (TP=1 = the canary load pattern, least wedge-prone).

### DFL2-8 — the DFlash2 drafter's attention backend (blocking unknown; correctness + speed)

**Status: open — highest priority in the DFlash2 family (2026-09-15).** With the bring-up crash
fixed, arm C measured **2.48 t/s / acceptance 0.045 / 421.9 ms/step** at 64k against the MTP k=3
band (33.62 plain, 35.44 with CAT-1) — a 13x gap that no downstream patch can close. The log says
why: the **target** gets `CUSTOM` (our FA), but the **drafter** is rejected by our backend and runs
upstream `ROCM_ATTN`, whose paged kernel then falls back to Triton (`Cannot use ROCm custom paged
attention kernel, falling back to Triton implementation`, drafter only). ~422 ms/step over 5
sliding layers plus 0.045 acceptance suggests the fallback is attending over far more than the
2048-token window — i.e. a *wrong* draft context, not just a slow one.

Steps: (1) **log the rejection reasons** — `ROcmPlatform` prints only the names of invalid
backends (`Found incompatible backend(s) [CUSTOM, TURBOQUANT]`), while the reasons are already
computed (`v1/attention/backend.py`: head_size / dtype / kv_cache_dtype / block_size / mm_prefix /
MLA / sinks / **sparse** / per-head-quant / compute-capability / attn_type); one line of logging
turns this into a five-minute diagnosis; (2) with the reason known, decide whether our FA can
serve the drafter's config (bidirectional + sliding-2048 + D=128 — the kernel supports each of
those individually, VIT-1 proved the bidirectional path and `supports_sliding_window()` is
already True) or whether the drafter genuinely needs the sparse path; (3) gate any wiring with the
same interleaved serving A/B, and check acceptance (a correct window should move 0.045 by a lot).
This is the gate for the whole DFlash2 family — `DFL2-1`/`DFL2-3` stay behind it.

### FA-COVER-1 — enumerate every config where CUSTOM is rejected or not selected

**Status: open (2026-09-15).** The same mechanism silently puts models on Triton-based attention
instead of the MI50-tuned FA. Two instances found so far: **Gemma-4 → TRITON_ATTN** (noticed during
GEMMA4-1) and the **DFlash2 drafter → ROCM_ATTN + Triton fallback** (DFL2-8). Work: log the
per-backend rejection reasons (same one-liner as DFL2-8 step 1), then sweep the models we serve and
record, per model, which backend was chosen *and why*; for every rejection, decide whether it is a
genuine kernel limitation or a declaration/dispatch gap that our FA already covers (its capability
surface: head sizes {64,128,256}, sliding window, bidirectional, DECODER, MM-encoder/ViT) and close
the reachable ones. Any model that silently runs Triton attention is a candidate for a VIT-1-style
wiring win, and this is the cheapest way to find them.

### FD-1 — CLOSED: the MTP fused-draft path was measured (NEUTRAL, stack-confounded) and its only reader is gone

**STATUS 2026-09-13 — EXECUTED: VERDICT NEUTRAL, stack-confounded.** FIX arm
2377.6 s vs non-FD *serving* 2447.8/2464.9 s at 4×122880 (offline arm vs
serving control — the comparison is confounded, `DEVLOG-spec-decode.md`
2026-09-12). The flag's only reader in-tree was A3's opt-in, **stripped
2026-09-13** (`f8a9400789`), so **do not re-queue this arm as-is** — it would
silently duplicate the non-FD arm. Revival: restore from
`archive/a3-fused-draft` (`A3-REVIVAL.md`) and re-gate **same-stack** (same
build/harness, flag on/off). Keep-or-strip analysis:
`/local/tmp/b4/fd1-keep-strip-decision.md`.

**Context (Kevin, 2026-09-10).** Most branch perf work is ON by default in the
B=4 bench (FA fused/persistent/fused-quant/CG-decode, direct-paged auto → on
for B=4 decode, kv_split shape-aware clamp, GDN opts, max-ilp, J2G-1 RCCL
Tree+LL). The one MTP perf improvement that is **OFF** is
`VLLM_GFX906_FUSED_DRAFT` (default `0`; the code marks it as still needing a
serving A/B before default-on). It fuses the draft-decode metadata path and is
likely to help *more* at B=4 (bigger verify M → more compute to fuse).
**Measurement:** **mtp3b4 + `VLLM_GFX906_FUSED_DRAFT=1`** same config as
mtp3b4 (port 8141, k=3, capture [4,8,12,16], util 0.93, NOCACHE=1) at 122880
B=4 ×1, A/B vs the plain mtp3b4 arm. Single variable = the env flag. SYV-12 is
**not** in scope (gated off + k=2-only, the mtp3b4 arm is k=3 — see TP-1 note
/ item 2). **LAST + OPTIONAL arm** (highest run-risk: activates a default-OFF
unvalidated serving path — skip if any wedge has occurred first).

**Wedge budget (tonight, 1 sample/arm per Kevin's 2026-09-10 cut):** campaign
greedy4 ×2 (in progress, undisturbed) → mtp3b4 ×1 (1 load) → TP-1 greedy (1
load) → TP-1 MTP (1 load) → FD-1 ×1 (1 load, optional) = 4 more loads. House
rules: 1 retry per genuine wedge, 2 consecutive → stop + reboot. **The cut is
applied only because the handoff is low-risk** (the post-campaign driver tears
down the orphaned server itself; if the handoff state looks risky, run mtp3b4
×2 instead — do not force the cut).

### MBT-1/2/3 — cut the multi-batch prefill O(live-context) tax (analysis 2026-09-11)

**Analysis:** [`prefill-multibatch-tax.md`](prefill-multibatch-tax.md).
The B=4/120k clean wall (75.4 min/sample, s0) is dominated by a **per-step
cost ∝ the SUM of live request contexts (A)** — the validated step model
`c(n)=2.63 s + 34.1 µs·n` (`ttft-prefill-stall.md` §10) gives ~5320 s for the
480k batch vs ~568 s lone (the tax is the whole B=4-vs-4×B=1 gap). The slope
owner is **UNRESOLVED — host pass vs GPU-side pool-wide op** (CORRECTED
2026-09-11, arbiter review: "CPU-side, §11 T1 strengthened / T2 weakened" was
wrong — T1 is refuted at `ttft-prefill-stall.md` §12.3/§13.3, there is no §11,
and D1c's cross-request taxation excludes per-request own-KV work: the owner
walks ALL live contexts per prefill step; the corrected CPU census shows no
ramping host thread, leaning GPU-side). Three experiments (queued **after**
the campaign arms, E1/E2 cheap flag A/Bs, E3 the owner-pinning that unlocks
the real fix):

- **MBT-1 (E1) — prefill-chunk A/B at B=4/120k. OWNER DISCRIMINATOR —
  RESULT (2026-09-11, boot Y8): CHUNK-INVARIANT.**
  `--max-num-batched-tokens`
  ∈ {1024, 2048, 4096}, same 4×122880 simultaneous, prefix OFF. bt=2048
  clean run = 4868.7 s (81.1 min) vs bt=1024 75.3 min, staggered ttfts ≈
  identical (511/1501/2943/4824 s), prefill agg 101.0 vs 108.6 t/s — the
  per-step-repeated overhead model is REFUTED (predicted ~48 min); the tax
  is **per-PREFILL-TOKEN × live-context work** (each prefill token streams
  the live KV pool from HBM — ~10–20× the attention-FLOP floor; H2 in
  `ttft-prefill-stall.md` §13.14). **THEN ROOT-CAUSED AND FIXED SAME-DAY**
  (kv_max pad-tile expansion, §13.16 + §13.16.1): the clamp was built and
  **VALIDATED on boot Y9 — 120k×B4 wall 75.3 → 44.8 min (−41%), prefill
  agg 108.6 → 182.9 t/s (+68%), outputs fingerprint-identical, decode/spec
  unaffected.** The bt4096 arm never ran (chronic wedges #57/#58 — burst;
  its OOM question is moot — the fix removes the incentive). Remaining
  slope (~1.2 ks) = own-context per-token streaming (M=1-shaped KV reads)
  — FA-line follow-up, ~3× smaller than the fix. O1 stays dead
  (chunk-invariant).
- **MBT-2 (E2) — concurrency A/B at B=4/120k.** `--max-num-seqs` ∈ {4, 2}.
  GATE: batch wall. 2×B=2 halves the quadratic term (480² → 2·240²) → expect
  ~1.5–1.6×; confirms the tax ∝ (concurrent batch tokens)².
- **MBT-3 (E3) — pin the 34.1 µs/tok owner IF host-side.** cProfile +
  py-spy during a B=4/120k prefill (engine core + workers), target the
  per-step pass. GATE: a named op + its µs/tok — or an EMPTY profile, which
  (with E1 chunk-invariant) moves the owner to a GPU-side pool-walk and makes
  the kineto per-kernel breakdown (parked on the torch build) the arbiter. If
  it IS a Python walk / un-vectorized CPU op, fusing/vectorizing/state-size-
  reducing it cuts the slope 10–100× and takes the B=4/120k wall toward the
  ~15–25 min floor. Run AFTER MBT-1: E1's outcome picks the interpretation of
  an empty E3. **E1 outcome recorded (chunk-invariant) ⇒ the live candidates
  are GPU-side per-token live-KV streaming (H2, ttft §13.14) — the H2 kernel
  hunt (where does the pool-wide read happen: varlen metadata, paged gather,
  or the custom Q8 FA long-query path) is now the primary owner search; E3
  only rules the engine-core host spot in/out.**

**Priority (boot Y9, post-FIX-H2 validation):** (0) **FIX-H2 DONE AND
VALIDATED** (§13.16.1: 120k×B4 wall 75.3 → 44.8 min, prefill agg +68%,
outputs fingerprint-identical; commit this build) — (1) re-run the
campaign 120k×B4 cells + mtp3b4 on the fixed build (their prefill walls
dropped ~40%; the 89-min/rep era is over), (2) TP-1/FD-1 (now cheaper too),
(3) FA-line follow-up: the residual own-context per-token streaming
(~1.2 ks @120k×B4) — larger effective Q-batch per KV read in the prefill
kernel, (4) MBT-2 seqs2 re-check post-fix (optional — the A-tax may now be
small enough that seqs4 vs seqs2 no longer matters), (5) E3 only if a
residual host term is still suspected (it is not, per the census).
(6) **Re-anchor the published B=1 prefill sweep**: one point (Qwen3.8-27B, 64k,
prefix-cache off) on the current tree. FIX-H2/M3 only touch multi-batch
prefill, so the numbers should stand — this is the one-point verify that makes
that claim measured rather than argued (the sweep predates them).

## High priority — user-requested (2026-09-01)

### MTP-1 — Qwen3.8-27B dense MTP long-context decode: crossover, remaining wins, dynamic depth (HIGH PRIORITY)

**User request 2026-09-01.** Reports of low long-context performance with
MTP on **TP=2** for the dense 27B model. Three subtasks, in order; do not
start a later one before the earlier has delivered its measurement.

Target: `Qwen3.8-27B-AWQ-INT4` (dense), TP=2 on the official amdgpu DKMS
driver (platform fixed, S4), `--dtype float16`, util 0.93, trimmed capture
`[1,2,3,4]`. This is dense-model work — separate from the MoE Tier-1 C*
items; do not import Qwen3.5 MoE numbers.

**Known starting evidence (S9, boot E, 2026-08-24) — a curve, not a pinned
bracket.** MTP k=2 vs greedy, TP=2, live-context decode tax:

| live ctx | MTP t/s | greedy t/s | MTP/greedy |
|---|---|---|---|
| ~2k  | 59.2 | 40.8 | **1.45×** |
| ~8k  | 44.9 | 38.1 | **1.18×** |
| ~32k | 25.2 | 30.5 | **0.83×** |
| ~64k | 16.6 | 24.1 | **0.69×** |

Crossover already observed between 8k and 32k (README: "MTP < greedy beyond
~20k ctx"). Mechanism from S8/S9: FA gather/attention is O(Sk); MTP's ~2.5
tok/step no longer beats greedy's 1× FA/draft overhead past ~20k live ctx
(step ≈ 40 ms + ~1.7 µs/token). **But** that curve is n=2–3 per point, single
boot, and conflates prefill length with live context (prefix-cache warm hits
inflate the short-ctx cells). The crossover bracket is NOT yet pinned.

**STATUS (2026-09-02): MTP-1a DONE — crossover pinned 32k–64k pp on clean
boot Q.** Full record: [DEVLOG-mtp1](DEVLOG-mtp1.md). Pinned curve (n=3, cold
prefill, separate prefill/live-context):

| pp | MTP t/s | greedy t/s | ratio |
|---:|---:|---:|---:|
| 2048 | 55.3 | 39.9 | **1.39×** |
| 16384 | 39.9 | 31.9 | **1.25×** |
| 32768 | 26.6 | 25.9 | **1.03×** (last win) |
| 65536 | 16.0 | 18.9 | **0.85×** ← crossover |
| 98304 | 11.2 | 14.8 | **0.76×** |
| 122880 | 9.2 | 12.7 | **0.72×** |

Acceptance = 2.0 stable through 120k (no collapse) → the loss is O(Sk) step
cost, not draft rejection. Optz microbench: lm_head-per-draft lead DEAD
(memory-bound, +322 µs/step = 0.4%); attention K-multiplier ~1.0 at 120k (KV
bytes shared). **Budget puzzle:** 78 ms/step @120k greedy vs ~12 ms BW floor
= 6× unexplained → rocprofv3 kernel breakdown is the real MTP-1b gate (pending,
blocked by zombie KFD handle from old-vLLM wedge — needs reboot). Old-vLLM
(0.23.1) A/B abandoned: that code path wedges GPUs loading this model on both
userlands (see degradation.md 2026-09-02 entries).

**STATUS (2026-09-02, boot S): MTP-1b gate MET — kernel breakdown + K=1 arm.**
The budget puzzle is resolved: CUDA-event phase hooks (mode-NONE, validated to
0.4% vs wall) show the greedy @120k step = **full_attn 57.3 ms (73%)** / mlp
12.6 ms (16%) / lin_attn 8.5 ms (11%). Full attention is O(Sk) and dominates;
GDN linear-attn stays flat — confirming the crossover mechanism. **K=1 arm
(boot S, n=3):** k=1 spec decode BEATS BOTH greedy and k=2 at every long-context
point — the "crossover" was K=2-specific:

| pp | k=2 (boot Q) | **k=1 (boot S)** | greedy (boot Q) | k1/greedy |
|---:|---:|---:|---:|---:|
| 65536 | 15.95 | **31.61** | 18.86 | **1.68×** |
| 98304 | 11.19 | **25.29** | 14.80 | **1.71×** |
| 122880 | 9.18 | **22.07** | 12.74 | **1.73×** |

k=1 acceptance = 1.0/draft-token (same as k=2's per-draft acceptance; k=2
accepts ~2/step, k=1 ~1/step — the extra k=2 draft token is pure O(Sk) overhead
at long ctx). **Root-cause candidate found in our own FA kernel:** both
`gfx906_fa_forward` and `gfx906_fa_forward_paged_direct` clamp
`kv_split = 1` when `seq_q > 2` (an OOM guard meant for prefill, where Sq is
thousands). Spec-decode verify presents the target model k+1 query tokens as
ONE sequence (`seq_q = k+1`, padded via `_pick_ncols1`), so **k=2 verify
(seq_q=3→pad 4) loses ALL its KV-split parallelism on exactly the O(Sk)
attention that is 73% of the step**, while greedy (seq_q=1) and k=1 (seq_q=2)
keep kv_split. This is **MTP-1b-0** below — the top MTP-1b item (user
re-prioritized 2026-09-02: "fixing the KV split for k>1 is a high priority fix,
higher than the syv ideas"). Recon docs from the two external repos (source
forks cited in each): [RECON-syv-qwen38-27b-rtx3090](RECON-syv-qwen38-27b-rtx3090.md),
[RECON-joe2gaan-localaiservers](RECON-joe2gaan-localaiservers.md).

- **MTP-1a — pin the crossover bracket.** ~~Sweep prefill / live-context~~
  **DONE 2026-09-02 (boot Q):** bracket pinned at **32k–64k pp** (1.03× at
  32k, 0.85× at 64k); n=3 reps, cold prefill, separate arms sequential. See
  [DEVLOG-mtp1](DEVLOG-mtp1.md). Bracket is ~2× wider than the S9 ~20k
  estimate — reported per stop rule before proceeding to MTP-1b.
- **MTP-1b — remaining Qwen3.8 MTP gfx906 optimization opportunities.**
  Profile the MTP draft+verify path at and beyond the crossover: the FA
  gather/attention O(Sk) cost of the draft layer, the verification step, KV
  writes for draft tokens, and any MTP-specific kernel that falls back to a
  slow path on gfx906. Identify concrete wins (kernel / dispatch / config).
  Gate: serving A/B + PPL/coherence (model is non-deterministic at temp=0 —
  token-identity gates unusable; use t/s + PPL/coherence).

  **MTP-1b-0 — fix the FA kv_split clamp for spec-decode verify (k>1). TOP
  PRIORITY (user, 2026-09-02). CLOSED (2026-09-03): merged to main a6ff64a71b +
  review fixes 7eb8b5d08e; PPL/coherence gate GATE-PASS on clean boot T
  (PPL drift 0.0; token divergence == kv_split FP noise floor, see
  DEVLOG-mtp1.md 2026-09-03 entry).** Both
  `gfx906_fa_forward` (gather path) and `gfx906_fa_forward_paged_direct` forced
  `kv_split = 1` when `seq_q > 2`, which fired for k=2 verify (seq_q=3→pad 4)
  but NOT for greedy (seq_q=1) or k=1 verify (seq_q=2). The clamp was a prefill
  OOM guard (partial buffer `[B, Sq, Hq, kv_split, D]` fp32 scales with Sq —
  multi-GB at Sq~thousands), but at verify Sq=k+1 is tiny (a few MB even at
  kv_split=16). **Fix:** replaced the hard clamp in both paths with a byte
  budget (`GFX906_FA_KVSPLIT_MAX_BYTES`, default 512 MiB): keep kv_split>1
  whenever the partial buffer fits — covers Sq≤~32 at B=1/Hq=24/D=256/split=16
  (~24 MB) and still forces y=1 for real prefill.

  **Correctness:** kv_split=1 vs =8 outputs identical for Sq∈{1,2,3,4} on both
  paths (max|d| ≤ 4.6e-4) and match a torch causal reference within 1.2e-4;
  prefill shapes Sq∈{256,1024} also verified. In-repo regression test:
  `tests/kernels/attention/test_gfx906_fa.py::test_forward_sq_multi_kv_split_vs_fp32_ref`
  (7 cases: verify Sq∈{2,3,4}, serving split=16, prefill under budget, and the
  clamped-to-y=1 path via a pinned low `GFX906_FA_KVSPLIT_MAX_BYTES`).

  **Serving A/B @120k-class points (n=3, cold prefill, S9 corpus):**

  | pp | k=2 baseline (clamp) | k=2 fixed | gain | vs k=1 same boot |
  |---:|---:|---:|---:|---:|
  | 65536 | 15.95 | **37.95** | 2.38× | beats k=1 (31.6) |
  | 98304 | 11.19 | **29.88** | 2.67× | beats k=1 (25.3) |
  | 122880 | 9.18 | **25.70** | 2.80× | beats k=1 (22.1) |

  The fix makes k=2 beat even k=1 at every long-context point: it recovers the
  full benefit of 2 draft tokens (acc_mean=2.0) on top of keeping KV-split
  parallelism. This SUPERSEDES the K=1-as-best-config conclusion — with the
  clamp fixed, **k=2 is the best static config at ≥64k** (and the earlier
  32k–64k "crossover" was an artifact of the clamp, not a fundamental MTP
  limit). Data: `/local/tmp/mtp1/data_mtp_k2fix_bootS.jsonl`.

  **Remaining before merge:** (a) PPL/coherence gate on k=2-fixed vs baseline
  (model non-deterministic at temp=0 — token-identity gates unusable); (b)
  prefill-OOM regression check (large-Sq batch, budget guard exercised at the
  boundary); (c) merge per the pre-merge protocol (self-review + claude CLI
  review of branch vs main). **Status: VALIDATED — pending PPL gate + merge.**

  **Ideas from external repos (source forks cited in the linked recon docs):**
  - **SYV-1 — split-KV for multi-query verify.** From
    [syv-ai/qwen38-27b-rtx3090](RECON-syv-qwen38-27b-rtx3090.md) (RTX 3090,
    same model family). Their FA2 only splits KV for single-query requests;
    verify (k+1 queries) ran on 24/82 SMs. **SUPERSEDED by MTP-1b-0** — our
    kernel already has split-KV; the clamp fix is the actual work. Kept as the
    external validation that this is the right lever.
  - **SYV-2 — lookahead/context drafting.** Draft from the request's own token
    history (point-mass, lossless). Their numbers: +55% verbatim, +2–3% prose.
    Our workload has heavy verbatim reproduction; acceptance already 1.0/
    draft-token at 120k so upside is content-shape. Pure scheduler+sampler
    glue, no new params. **Status: open (MTP-1c candidate — pairs with the
    dynamic-depth policy).**
  - **SYV-3 — quantize MTP draft + small draft vocab.** Their drafter was bf16
    + full 248k lm_head per draft. ~~Closed negative (marginal K1→K3 = +322 µs
    = 0.4%/step).~~ **RE-OPENED 2026-09-03 — the close used only the MARGINAL
    cost of extra draft rows; it missed the ABSOLUTE B=1 lm_head read.**
    Roofline (see `/local/tmp/mtp1/drafter_memory_bound.md`): drafter lm_head is
    a GEMV with arithmetic intensity = B (1–3) ≪ ridge (~26 gfx906, ~38 sm86) →
    memory-bound on BOTH GPUs. **Step 1 CONFIRMED 2026-09-03** (standalone bench
    `syv3_gemv.py`, exact production shape + dispatch under TP=2 — in-process
    profiler route hit the chronic weight-load wedge family, logged):
    `torch.mm`/`F.linear` at **183 GB/s = 24.8% of the ~740 GB/s copy ceiling**
    (6.9 ms vs ~1.9 ms achievable) → the skinny-GEMV dispatch is leaving
    **~3.6× on the table**. **Lever 1 PROTOTYPED + MEASURED 2026-09-03**
    (`syv3_gemv_triton.py`, plain Triton, K=1 — the production call pattern per
    `step3p5.py`'s sequential draft steps): best config BN=128/BH=512/warps=4 →
    **727 GB/s = 98% of ceiling, 3.97× vs torch.mm** (6937→1748 µs/call), max|err|
    0.002 (fp32-accum). K≥2 `tl.dot` variants are slower than torch.mm (gfx906
    64 KB smem staging) but irrelevant — B=1 serving = K=1 calls only. **SHELVED
    2026-09-04 (C3 gate: no gain).** Full integration + TP=1 A/B completed and
    the root-cause investigation killed the premise. In-context CUDA-event
    timing at the exact production shape (K=1 N=248320 H=5120 fp16, both arms):
    **stock GEMV path = 3.09 ms median (~822 GB/s effective) vs our Triton
    kernel = 3.93 ms (~647 GB/s)**. **ROOT CAUSE CONFIRMED 2026-09-04 (two
    experiments):** (1) the standalone "torch.mm = 13 ms" number was ATen mm at
    FULL clock — DVFS sampling showed mclk held 1000 MHz throughout sustained
    AND per-iter-sync runs (idle is 350 MHz, so power-saving is real but did
    NOT explain any SYV-3 number); ATen mm is simply pathological at this shape
    (~194 GB/s ≈ 19% of the 1 TB/s HBM2 peak). (2) a production torch-profiler
    trace (eager, 8k prompt, 256 decode tokens; note: this fork's torch-profiler
    records CPU ops only — zero GPU kernel events, same broken HSA layer as
    rocprofv3) shows the default path NEVER runs ATen mm for the drafter
    lm_head: n=1 → `_rocm_C::LLMM1` (`utils.py:579`), n=2–4 →
    `_rocm_C::dense_gemv_m4_gfx906` (`utils.py:340`) — the fork's custom gfx906
    GEMV family, already at ~822 GB/s ≈ 80% of peak (matches DEAD-ENDS.md:
    "LLMM1 already at HBM floor 3114 µs; the lm_head GEMV lever does not
    exist"). The 3.09 ms vs 12.9 ms gap was two DIFFERENT kernels, not two
    clock states → default path is memory-bound, no custom kernel can win.
    (3) **MISSING RUNG CLOSED post external review 2026-09-04**: the dispatched
    kernels were then measured DIRECTLY standalone at hot clock by calling the
    production dispatcher functions themselves (`/local/tmp/mtp1/syv3_gemv_standalone.py`,
    deciles + concurrent mclk ≥900 MHz hard gate): n=1 LLMM1 = **3.098 ms
    (821 GB/s ≈ 82% peak)** vs in-context 3.09 ms and the DEAD-ENDS audit 3114 µs
    — three independent anchors agree within 1%; n=4 `dense_gemv_m4_gfx906` =
    4.997 ms (509 GB/s, weight-read bound as designed for M=2–4); ATen mm
    reference 12.83 ms (19.8%). The shelve now stands on direct measurement,
    not elimination. Review also produced the **PROPOSED dispatcher-faithful
    standalone bench protocol** (`dvfs-mi50.md` — benchmark the dispatcher,
    deciles, hard mclk gate, ≥2-anchor cross-validation).
    TP=1 A/B (zero wedges): OFF 47.82/29.64 t/s @8k/32k vs ON 46.55/30.76;
    acceptance identical 0.50/step → delta within noise, no gain. **Status:
    SHELVED (confirmed)** — code preserved on branch
    `shelved/syv3-skinny-gemv` (e960be998c) incl. the head_dtype property-gate
    fix + env-gated diagnostic timers; re-arm only if a future profile shows the
    stock drafter lm_head regressing below ~700 GB/s effective. **The byte-count
    lever survives as CAT-1** (smaller draft vocab halves bytes read — that is
    where the remaining headroom actually is).
  - **SYV-4 — sort-free small-k top-k/top-p sampler.** Their gain +4%. Our ROCm
    path (`forward_native`, aiter absent) sorts all ~248k logits/row (~0.35 ms @B=1).
    **IMPLEMENTED** on `gfx906/syv4-sort-free-sampler` (`e402e85192`, 2026-09-03):
    one `torch.topk(k)` replaces the full-vocab sort when all rows' k≤64 and B<8;
    top-p keep rule = count(cumsum < p)+1 over each row's own top-k candidates
    (no argmax/bool — ROCm has no bool-argmax kernel). CPU equivalence vs
    `apply_top_k_top_p_pytorch`: 14 cases, 0 failures. Opt-out
    `VLLM_GFX906_SORT_FREE_SMALL_K=0`. **GPU bench (2026-09-03, canary 39.2 t/s
    PASS):** 12/12 correctness cases exact vs reference at V=248k; perf — B=1 k=20:
    0.91× (noise), B=1 k=64 p=.95: 1.17×, **B=4 mixed k≤64 p: 6.55×** (2.212→0.338
    ms/call). Win scales with batch; ~neutral at B=1 greedy. **End-to-end A/B
    (2026-09-03, TP=1 dense 27B, tg256, n=3 reps):** sort-free vs reference —
    temp0 (k disabled, control): +0.5%/+0.4% (noise); sample k=32 p=.95:
    B=1 +0.6%, **B=4 +3.4%**; sample k=8 p=.9: B=1 +1.4%, B=4 +1.6%. The
    microbench win translates to serving on sampling workloads (control arm
    confirms the delta is the sampler, not run noise). MERGED to main.
  - **SYV-5 — fp16 GDN recurrent state** (`--mamba-ssm-cache-dtype float16`).
    **Status: CLOSED as dead end (2026-09-08, A6 profile,
    `PROFILE-gdn-bucket-breakdown.md`)** — the rec kernel runs 3.7× the
    pure-BW state-traffic floor (28.8 µs/layer vs 7.7 µs floor: latency-
    bound, not state-BW-bound), so fp16 state saves ≤ 0.4–1.6 % of the
    step even in the impossible fully-traffic-bound case (realistic ≪ 1 %).
    The flag is live for this model family but sub-1 % levers don't clear
    the PPL-gate bar. Flag verified in-tree: `get_mamba_state_dtype_from_config`
    (`models/qwen3_5.py:378/590`).
  - **SYV-6 — int8 activations (W4A8 Marlin) + negative-scale bug fix.**
    Batch-mode only (we run B=1); park until multi-request resumes. The bug fix
    is model-portable if we ever hit it. **Status: parked.**
  - **SYV-7 — hybrid-model prefix caching.** Biggest real-workload win for
    chat-on-docs; our sweep uses cold prefill so it doesn't change MTP-1 numbers.
    **Status: DONE (2026-09-04) — nothing to port or enable:** the flag is ON by
    default in our fork (`enable_prefix_caching=True`, `cache.py`) and the model
    config auto-promotes mamba cache mode to `align` when PC is on
    (`models/config.py:602-610`). Verified working WITH MTP (Marconi pattern:
    1st req full prefill, state cached at completion, 3rd+ hit) — see
    DEVLOG-spec-decode.md 2026-08-19. Known minor subtlety: under MTP only ~800 of
    1600 aligned tokens hit vs baseline's 1568/1631 (num_reprefillable_tokens
    finalization × single cached state) — not chased.
    **Follow-up task (SYV-7b, Kevin 2026-09-04): measure whether a SMALLER
    `--mamba-block-size` is a win for agentic payloads that cache well.**
    Rationale: prefill on gfx906 is slow, so when the shared prefix length falls
    short of a block boundary (e.g. 1.9k prefilled vs 2k block size), the whole
    tail re-prefills — seconds added to interactive/agentic turns. Test matrix:
    mamba-block-size {default(=block_size), 256, 512} × agentic workload (shared
    system+tools prefix, growing conversation) → measure per-turn TTFT + hit rate
    (`prefix_cache_stats` / num_scheduled prefilled tokens). Needs a real agentic
    corpus (pair with the CAT-1 corpus capture when it lands). Low effort:
    config-only A/B, no code. **Status: open (config-only, needs agentic corpus).**
  - **SYV-8 — DFlash2 block drafter.** Different drafter arch (whole-block
    non-autoregressive). Big effort, needs V2 runner (conflicts with our
    FULLGRAPH path). **Status: parked — revisit only if SYV-1/MTP-1b-0 + SYV-2
    don't deliver.**
  - **SYV-9 — int8-QK prefill attention.** ~~Prefill-only; not our bottleneck.~~
    **Status: HIGH PRIORITY (promoted 2026-09-03, Kevin: "prefill is also very
    relevant to us").** Rationale: gfx906 compute is slow → prefill is
    COMPUTE-bound (unlike decode, which is memory-bound GEMV), so prefill t/s
    directly limits long-doc first-token latency and any batched-prefill regime.
    Their int8-QK halves the QK^T MAC cost in prefill attention specifically.
    Scope: (a) profile OUR prefill breakdown on gfx906 (FA kernel vs GEMM vs
    GDN conv1d — we already have ~300 t/s TP=2 32k records to normalize
    against), (b) port/adapt their int8-QK path if the FA share justifies it,
    (c) broader prefill levers in scope too: Triton FA tile sizes for gfx906,
    chunked-prefill sizing (`MAX_NUM_BATCHED_TOKENS`), and prefix caching
    (SYV-7) which eliminates redundant prefill entirely. **Next step: prefill
    phase profile before any port.**
  - **SYV-10 — GDN spec-decode bounds checks (upstream PR #50021; VERIFY).**
    Their vendored `vllm-pr50021-gdn-spec-bounds.patch` fixes an
    illegal-memory-access in the DeltaNet/GDN speculative-decode kernels hit
    with several *concurrent* MTP requests. We run B=1 (low exposure) but the
    agent-corpus arms and any future multi-request work hit the same kernels.
    **Status: PORTED (2026-09-07, boot Z).** Step 0 result: the fork carried
    the PRE-PR code in all four kernels — the unmasked
    `i_t = num_accepted_tokens - 1` state-index load was byte-identical in
    `mamba/ops/causal_conv1d.py` (spec branch), `mamba/ops/mamba_ssm.py`
    (zero-clamp only, no row bound), and BOTH
    `third_party/flash_linear_attention/ops/{fused_recurrent,
    fused_sigmoid_gating}.py` (the latter is the kernel the Qwen GDN
    spec path actually calls, `fused_sigmoid_gating_delta_rule_update`) —
    also correcting the SYV-13 note that `third_party/flash_linear_attention`
    is absent: it exists in this fork (the missing piece SYV-13 refers to is
    `chunk_o.py`). Patch applied verbatim (`git apply --directory=vllm`,
    clean); behavior-identical on valid inputs (mask=true load; the
    zero-fill store fires only on the invalid-state early-out). No runnable
    GPU test on ROCm (test_gdn_fused_mtp.py is CUDA-gated); the `launch_pdl`
    guard matches existing in-file usage (constexpr-compiles-out on ROCm).
    Compile-check lands at the next MTP server launch (boot-Z+1 chat-frac
    arms); field-verify there. Still the B=4 prerequisite.
  - **SYV-11 — KV-cache compression for capacity (KVarN tier / stock int4·int8
    tier; ANALYZE).** Deep-review find (2026-09-07). Their 24 GB 3090 is
    capacity-bound; ours is not *yet* (131k ctx, 64 KB/token fp16 KV = 7.5 GB
    @120k — matches our 7.4 GiB bench records) — so this is a **context
    extension** item, not a speed item (their 3090 data: KVarN decode ~20 %
    *slower* than fp8 at 100k — the dequant eats the bandwidth saving). Two
    tiers, same model family (they run Qwen3.8-27B, so the quality data
    transfers directly):
    - **KVarN** (Huawei CSL, Apache-2.0; ported to vLLM 0.28 in their repo
      `kvarn/`): Hadamard rotation + iterative variance normalization,
      4-bit K / 2-bit V per 128-token tile → **12 KB/token (vs 64 fp16)**:
      131k → ~500k-token pool. Their measured: 262k ctx fits, needle-in-haystack
      correct 4k–240k, **PPL +0.16 %**, MTP works. 1/8 the KV bytes.
    - **Stock `int4_per_token_head` / int8 KV** (their
      `int4-kv-per-token-head.patch` + `spec-decode-int8-kv.patch` fix boot
      blockers on the stock Triton backend): 1/2–1/4 the KV bytes, stock
      machinery, ~20 % decode cost on their box (Triton backend + per-step
      unpack).
    **Status: OPEN — LOW-MED priority; only worth it if Kevin wants >131k
    single-request context or multi-request KV capacity. Gate: PPL probe band
    + needle + serving A/B; the gfx906 port of the KVarN Triton kernels needs
    its own validation (dense path only; hybrid page-alignment hunks exist in
    their port).**
  - **SYV-12 — context-lookup verify extension for MTP ("long block from the prompt"); ANALYZE — the new idea from the deep review. CLOSED negative 2026-09-08 (v1 as built: production-payload A/B net loss −9.3 %/−5.3 %, verified lossless, env-gated default-OFF; the copy-saturated case is UNVERIFIED — the fill's yield was never measured; attribution correction + resurrection gate at the end of the entry; reopen note in `REFRIGERATOR.md`).** Their DFlash2
    lookup drafting (`dflash2-lookup-drafting.patch`) generalizes to our MTP
    stack: keep the drafter at k (MTP k=5), but let the **target verify a
    longer block** (k+8) when a prompt-lookup fires — positions past the
    drafter's k are filled from the most recent earlier occurrence of the
    just-generated suffix in the request's own token history (one Triton
    program per request; point-mass proposal keeps the rejection sampler
    lossless; scheduled only while two consecutive saturated steps indicate a
    genuine copy, not a prose near-tie). Their measured (25k ctx, greedy,
    same model family): **reproduce-a-document 159 → 381 tok/s (+47 %)**,
    rewrite/quote +10 %, prose +2–3 %, quality unchanged (GSM8K 96.5 %
    flat, 7/9 prompts token-identical). This is exactly the agentic shape our
    corpus arms target (code edits, RAG-quote turns), and the agent corpus
    (built 2026-09-07) can measure the copy-fraction before we build anything.
    Pairs with SYV-2 (lookahead drafting) + CAT-7 (ngram) — this item is the
    *verify-block extension* mechanism those two were missing.
    **Copy-fraction measured (CPU-only, 2026-09-07, `/local/tmp/mtp1/syv12_copy_fraction.py`;
    MIN_MATCH=5, EXT=8, last-occurrence n-gram scan over the 8 bodies/corpus):**
    full-body hit_frac (steps with ≥1 fillable position): agent 10.1 % @64k /
    15.9 % @120k; mixed 12.0 % / 16.8 %; tail-32k @120k 18.3 % (both
    corpora). Post-scheduling (run of ≥2 consecutive hits, the mechanism's
    trigger): 7.7–13.7 %. Mean fills per hit step 4.0–4.9 of the 8 budget
    (≈2.5–2.9 effective past a k=2 drafter); mean_fills amortized over all
    steps 0.41–0.79. The corpus scan measures payload self-repetition, not
    generation-time copying — the generation-time re-measure was the real
    gate. **GENERATION-TIME MEASURED (boot W', 2026-09-07, `/local/tmp/mtp1/syv12_gen_gate.py`
    over the 39-convo Qwen3.8 replay, `chat_replay_qwen38.jsonl`; same
    last-occurrence rule, MIN_MATCH=5, EXT=8, 205,091 generated positions
    = 99.98 % of the 205,173 completion tokens):** hit_frac **25.3 %**,
    run_frac (≥2 consecutive, the scheduler trigger) **21.1 %**, mean_fills
    1.39/step, run_mean_fills 5.52 of the 8 budget. **VERDICT: GO — clears
    the ≥15 % gate by ~10 points** (the corpus proxy's 10–18 % was an
    underestimate: generations re-quote context more than the corpus
    self-repeats — the "quote/reproduce turns run hotter" side of the bound).
    Caveats: 36/62 turns truncated at the replay's max_tokens=4096 (rate is
    per-step, so truncation shortens the sample but does not bias the rate);
    sample is temp-0 Qwen3.8 on the same seeds the v2 corpus is built from.
    **Instrument bug caught in the process (recorded for the log):** the
    first gate run read `rec["reasoning_content"]`, but this stack's qwen3
    parser emits the thinking text in the raw message's `reasoning` field —
    36/62 records had empty record fields, the sample silently shrank to 16k
    positions of short turns, and the verdict read PARK (5.1 %). Fixed gate
    reads the raw message (`syv12gen_w1_fixed.log`); `chat_replay_client.py`
    patched to record `reasoning` + re-send it as `reasoning_content` in the
    next-turn context. SYV-12 moves to the build queue after the boot-W' v2
    arms land. **DESIGN CONSTRAINT (review, 2026-09-07) — GO crossed, this
    pins the build spec: the fills are NOT free verify rows on our FA
    kernel.** At the k=2/3 operating point the base verify is Sq_pad=4
    (fast occ-2 tile, 8.64 ns/token); EXT=8 with generation-time
    run_mean_fills 5.52 (corpus: 4.0–4.9) pushes
    hit-steps to Sq 6–10 → pad 8 or 16 → slow occ-1 tiles (19 ns/token) —
    the whole verify step crosses the measured tile cliff. So (1) the
    economics must count marginal verify cost including the tile
    step-function, and the first design constraint should be a fill-cap that
    keeps the block on the Sq_pad=4 tile (k=2 + ≤2 fills; k=3 can only add
    ≤1 — re-derive the yield numbers under that cap before committing to a
    build), and (2) capture shapes interact: Sq varies per hit-step, so
    either a fixed max-fill shape (always paying the slow tile) or multiple
    verify graph shapes. **Status: GO (2026-09-07, boot W') — generation-time
    gate passed (25.3 % ≥ 15 %); build scheduled after the boot-W' v2 arms
    land; build spec = the design constraint above (fill-cap on the Sq_pad=4
    fast tile, yield re-derived under the cap, capture-shape decision
    included).**
    **FILL-CAP CORRECTION (2026-09-07, at implementation): the parenthetical
    above was off by one.** The fast tile holds 4 rows (padded Sq ≤ 4); the
    verify block is 1 anchor + k drafts + f fills, so the constraint is
    f ≤ 3 − k: **k=2 → f ≤ 1; k=3 → f ≤ 0** (k=3 + 1 fill = 5 rows →
    pad 8 → slow tile; k=2 + 2 fills = 5 rows → likewise). The v1 build
    takes the only positive cell: k=2 + EXT=1.
    **IMPLEMENTED v1 (2026-09-07, boot W''; env `GFX906_SYV12`, default OFF
    until the same-boot serving A/B passes):** the extension slot is
    *always* scheduled (uniform shape — every MTP-k=2 decode step verifies
    1+k+ext = 4 rows), so the fill consumes the row the k=2 verify already
    pays for when padded 3→4: per-step FA cost is unchanged, a fired+accepted
    fill is +1 token/step, a fired-but-rejected or unfired slot (the -1
    placeholder) costs nothing. Fill = last-occurrence continuation of the
    trailing 5-gram in the request's own history (one stream-ordered Triton
    program per request, no CPU-GPU sync; `GFX906_SYV12_WINDOW` default
    8192), gated on the previous step having been fully accepted
    (num_sampled == k+1). The single-step gate (vs their two-consecutive-
    saturated-steps trigger) is deliberately looser: with the always-extended
    shape, extra firings are free, so the gate only needs to avoid fills
    with near-zero acceptance. Fills go through the standard one-hot draft
    path of the standard rejection method → lossless (asserts off for
    `synthetic`). Files: `vllm/v1/spec_decode/utils.py` (`syv12_ext`, single
    source of truth), `vllm/v1/worker/gpu/spec_decode/syv12.py` (fill
    kernel), scheduler pad sites, `RequestState.draft_tokens` width,
    `decode_query_len`, `RejectionSampler.num_speculative_steps`. k=3
    stays unextended (f ≤ 0) — its 4 rows leave no room on the fast tile.
    **BOOT-Y CORRECTION (2026-09-08): the worker-side files above are the
    V2 model runner's — NOT the live path.** Live ROCm workers use the V1
    runner (`vllm/v1/worker/gpu_model_runner.py`; V2 is behind
    `VLLM_USE_V2_MODEL_RUNNER`, default off) — the probe's ON arm was a
    no-op (its fill kernel never executed; the kernel also had a Triton
    `break` bug + a 4-token occurrence-window off-by-4, both fixed and the
    kernel is now GPU unit-verified 9/9, `/local/tmp/syv12/kernel_test.py`).
    Inert at default OFF; flag-ON was inconsistent (scheduler pads, V1
    worker does not) → staged A/B was withheld.
    **BOOT-Y2 UPDATE (2026-09-08): V1 port COMPLETE.** Worker side now
    on the live V1 runner (`gpu_model_runner.py`): the fill runs
    GPU-side — a per-request 2.1 MB history buffer (`_syv12_hist`,
    stable rows, mapping rebuilt per launch) + new
    `_syv12_append_history_kernel` (compacts the step's non-(-1)
    rejection output into the row) + the fill kernel at the common tail
    of `propose_draft_token_ids` — the CPU history is unusable in async
    mode (placeholder -1s). Also: both async placeholder sites,
    `draft_token_ids_cpu` width, zero-fallback width, mamba
    `num_speculative_blocks` = k+ext, drafter `CudagraphDispatcher` unit;
    kernels moved to neutral `vllm/v1/spec_decode/syv12.py` (V1+V2
    share); fill kernel's draft store fixed to batch-index (was
    row-index → OOB on batch-sized drafts). Rejection/
    `_calc_spec_decode_metadata`/`_prepare_input_ids` needed no changes
    (data-driven on per-request draft count). Unit-verified 16/16 on
    GPU (`/local/tmp/syv12/kernel_test.py`). **Probe run on Y2:** OFF
    PASS (25.335 t/s decode); the ON arm exposed + fixed two more
    software bugs — scheduler spec-stats sizing (k vs k+ext) and the
    **GDN state-slot width**: the GDN attention backend's `num_spec`
    (`vllm/v1/attention/backends/gdn_attn.py`) and the GDN layer's
    conv-state width (`vllm/model_executor/layers/mamba/gdn/base.py`)
    must also add `syv12_ext` (lockstep with MambaSpec) — without it
    the ext verify token is under-processed (`max_query_len` = k+1 in
    `causal_conv1d_update`) and the GDN state corrupts (all-zero
    output, self-consistent zero fixed point). 16k debug probe
    post-fix: correct loop, both kernels fired. The 120k ON re-run
    wedged at weight load (#42; #41 at 07:49) → wear-based stop (2
    spontaneous chronic-family resets on boot Y2, the #40 criterion).
    **BOOT-Y3 FINAL (2026-09-08): probe PASS + same-boot A/B = NET LOSS,
    CLOSED.** 120k probe: identity 512/512 vs the boot-Y2 OFF reference,
    both kernels fired, decode 24.561 t/s (s9 ceiling, −3 % vs OFF's
    25.335). Serving A/B (`mtp` k=2, TP=2, mixed-v2 64k+120k ×3, OFF
    first as canary 28.97/22.84 — the 120k within +0.8 % of the W''
    record): **ON 26.27/21.62 → −9.3 % @64k, −5.3 % @120k (medians;
    5/6 reps negative).** Port verified lossless (no crashes) and
    stays env-gated default-OFF. Dev log: SYV-12 entries, boot Y2/Y3.
    **ATTRIBUTION CORRECTION (2026-09-08, verdict review — the
    "mechanism-predicted / ~5 % accepted vs ~100 % on s9" text above
    is RETRACTED):** per-position acceptance shows the fill row at
    **0.000 in every s9 probe window** (drafted 3/step, accepted
    never; mean capped 3.00) and 0.006 step-weighted on mixed-v2 —
    the yield path never produced a meaningful acceptance, so the
    measured A/B loss is the pure always-paid 4th-row cost (GDN +33 %
    state traffic; FA unchanged — both pad to Sq_pad=4) and the
    mechanism's true yield was never measured. A static
    reconstruction proves the correct s9 fill value is 511/511 = 100 %
    on the real history, and an end-to-end wiring audit (gate, history
    append, fill kernel, input scatter, sampler alignment; the
    "sampler caps at k" candidate refuted by the boot-Y2 spec-stats
    IndexError + "Drafted: 3/step") found no static flaw — a
    **runtime defect** was declared open at that point. **ROOT CAUSE
    FOUND + FIXED (2026-09-08, boot Y3, pre-probe; static derivation +
    unit-verified — the "runtime" defect was a static CONTRACT bug the
    wiring audit's link checks and the 16/16 unit test had both baked
    in):** the v1 fill kernel built its lookup suffix from the trailing
    history only, which ends at the step's ANCHOR, so its continuation
    targets the d0 slot — but the value is stored in the FILL slot
    (after d0..d_{k-1}), off by k=2 positions. s9 (period-9 loop): the
    old fill ≈ the d0 value, never equal to the FILL-slot argmax →
    0.000 at every position; mixed-v2: accepted only when a token
    repeats across the anchor→d1 span → the measured ~0.6 % incidental
    hits. The recorded "511/511 static reconstruction" measured the
    kernel's own contract (d0-continuation vs the next token), not
    FILL correctness — corrected here. Fix: suffix = last
    (MIN_MATCH−k) history tokens + the k base drafts (which end at
    d_{k-1}, the token before the FILL position); unit-verified 16/16
    under the corrected contract. The close stands for **v1-as-built**;
    the copy-saturated-payload case is **UNVERIFIED, not closed as
    structural** (s9 is where MTP already saturates — the revival
    regime is verbatim-span-heavy generation where the MTP head
    misses). **Resurrection gate (updated):** the instrumented s9
    probe is now a VALIDATION run (per-step debug dump landed
    in-tree; expect s9 pos3 ≈ 1.000 and mean acceptance ≈ 4.0) —
    blocked on a reboot by wedges #44/#45 (burst, 2026-09-08 21:22)
    → then s9 + copy-heavy corpus same-boot A/B before any
    copy-saturated claim. Reopen note in `REFRIGERATOR.md` (SYV-12
    entry). **VALIDATION PROBE PASS (2026-09-09, boot Y4 — gate step 1
    DONE):** after wedge #46 (probe attempt 1) and a diagnosed software
    failure on the retry (Triton compile assert: the fixed kernel's
    match if/else mixed the int32 history load with the int64 live
    draft-tensor load — the unit test had compiled only the all-int32
    signature; fixed `fb971c54a0`, test now uses the live int64 draft
    dtype, 16/16), the probe passed on all three links: (A)
    kernel-write-vs-reference 96/96, (B) input path 65/65,
    gate/append inconsistencies 0/0 (on the rank-separated dump — both
    TP ranks were interleaving into one file, `2eacd8e8ed`); **pos3
    (FILL) = 0.987 steady state** (engine window 1.000/1.000/0.987;
    the single miss is the prefill->decode boundary step, where the
    gate is off by design), mean acceptance ~3.99 (vs ~3.0 under v1),
    identity perfect (512-token exact s9 loop), eager decode 31.65
    t/s (vs 24.56 for v1-as-built, +28.9%). The copy-saturated case
    is now MECHANISM-VERIFIED on the saturation payload; next gate
    step: the production (compiled) same-boot A/B (mtp2 OFF vs ON,
    mixed-v2 64k+120k — directly comparable to boot Y3: OFF
    28.97/22.84, broken-ON 26.27/21.62 — plus the s9 120k point).
    **PRODUCTION A/B DONE (2026-09-09, boot Y4 — gate step 2):**
    same-boot, boot-Y3 protocol (mtp k=2 TP=2, mixed-v2 64k+120k ×3,
    OFF first): **ON 27.99/23.80 vs OFF 29.55/22.37 → −5.3 % @64k,
    +6.4 % @120k (medians)**; @120k clean separation (min ON 23.28 >
    max OFF 23.06), @64k overlapping. Realized fill acceptance
    ≈25 % of steps @120k (above the review's 15–20 % bar) vs ≈0–5 %
    @64k (fill effectively never fires → ON = OFF + the 4th-row GDN
    cost). **GATE VARIABLE ADJUDICATED (2026-09-09, research agent,
    CPU-only windowed-copy-density analysis on the served corpus —
    handover `/local/tmp/handover-syv12-context-vs-payload.md`,
    repro `/local/tmp/mtp1/syv12_content_vs_context.{py,log}`): the
    flip is a PAYLOAD-POSITION effect, NOT context scaling** — the
    fill's 8,192-token lookup window is structurally blind to context
    length; the served 64k point front-slices the bodies (windowed
    5-gram repeat density 12.6 %) while generation at 120k sits in
    the document tails (21.5 %; same bodies first-64k-sliced: 13.6 %).
    Amended parked verdict: payload-conditional,
    position-of-generation-tail-gated; NOT context-length-gated. Both arms self-canary vs Y3 (OFF within ±2 %); text probes
    clean both arms (lossless in serving). **VERDICT: the fixed fill
    is a PAYLOAD-CONDITIONAL win, NOT a uniform default.** Boot Y4
    ended here on the #40/#42 wear criterion (3 spontaneous
    chronic-family wedges #46/#47/#48, GPU1, all self-recovered; 3/8
    TP=2 loads). **PARKED by Kevin 2026-09-09 until further review**
    (no crossover A/B, no compiled s9 120k for now); the B=4 campaign
    runs first (Kevin-directed, same boot) — **RE-ORDERED 2026-09-09
evening (handover #2, `ttft-prefill-stall.md` §11): the TTFT stall
    investigation now runs BEFORE the campaign's remaining cells —
    120k×B4 ×2 + the mtp3b4 grid are DEFERRED (65536-point cells done
    and valid). The per-step O(f) model is characterized (2.63 s +
    34.1 µs/tok per 1024-token step); the owner hunt leads with T6
    (torch intra-op thread cap, `Reducing Torch threads from 8 to 1`)
    + an in-process 2×2 matrix (OMP × pp, 2 loads); existing B=4
deode data is mined first; future cells at pp=4096.**
    **UPDATED 2026-09-09 late evening (matrix load A, `ttft-prefill-stall.md`
    §12): OMP 1→8 = ratio 1.00 (T6 out). The per-step owner is the
    previous step's GPU tail, observed through a 100% spin-wait on
    `num_accepted_tokens_event.synchronize()` (gpu_model_runner.py:2213)
    — T1 (CPU O(n)) refuted. **VERDICT (fresh boot, §13): the
    production chunk-size A/B (pp=1024 vs 256, 8k prefill: 2.74 vs
    0.79 s/step; total 21.9 vs 25.4 s) shows ~92 % of the per-step
    cost is chunk-linear GPU work + a 0.14 s fixed per-step overhead,
    at an effective ~10.3–10.4 TFLOPS ≈ 78 % of MI50 fp16 peak in
    both regimes — the "stall" is raw W4A16 prefill throughput
    (GEMM-dominated) + the 34.1 µs/tok long-context FA slope (~62 % of
    a 120k step): a throughput question, not a bug.** Levers: GEMM
    mass/efficiency at prefill M (T-1 int8 direction, re-bench at
    M≥256), the long-context FA slope, and only secondarily chunk
    size (fixed-overhead saving ≈ 0.42 s per 4k). The per-kernel
    split is parked: torch 2.13.0+gfx906 (2026-08-02) ships no kineto
    GPU backend (zero GPU-domain events on every profiler route,
    §13.1). **SLOPE-OWNERSHIP UPDATE (2026-09-09, §13.4): the 34.1 µs/tok
    slope is CUMULATIVE (pool-wide), NOT per-request** — 4×64k: the
    own-context model 964 s vs the cumulative model 1822 s vs the
    observed 1770 s (both reps). The §13.3 verdict holds for the GEMM
    bulk but is PREMATURE for the slope term: a pool-wide per-step op
    (dense rebuild/gather candidate class, prefill-only, decode-immune)
    would be a REAL fixable inefficiency at ~62–63 % of long-context
    prefill time. OPEN discriminators (§13.5): D1 holder+probe
    (minutes — batch-local ~5.4 s vs persistent ~9.7 s probe ttft) →
    per-kernel breakdown (parked on the kineto gap). **Campaign note:
    the 120k×B4 cells run AFTER the slope question is resolved (or with
    clocks logged) — the slope is the dominant term at that scale.**
    If SYV-12 is revisited
    (per the adjudication): the gate is PAYLOAD/RUNTIME density, not
    context tokens — enable per request when the recent fill-hit rate
    (or a host-side trailing-window copy-density estimate) is
    sustained above ~10 %; the runtime fill path already computes the
    match statistics, so the gate is nearly free. If the crossover A/B
    runs: FRONT-SLICES of the 122880-point bodies (same text, shallow
    vs deep — only front-slices isolate depth) at 80k+100k, with the
    served-64k point as the content control (2×2 content × depth,
    2 loads) + compiled-mode s9 120k (2 loads) + the B=4-era question
    (the 4th row adds the same cost at B=4; B=4 cells run at pp=4096 per
    the §11 re-order above).
  - **SYV-13 — mamba/GDN chunked-prefill align fixes (CLOSED 2026-09-08 as
    N/A — verify-only, no code change).** Diffed their
    `mamba-chunked-prefill-align.patch` (qwen38-27b-rtx3090) against our tree,
    both parts: (1) **`src_col` fix** — the live V1 path
    (`preprocess_mamba`, `vllm/v1/worker/mamba_utils.py`) already implements
    all three of the patch's guard conditions on the CPU side: fresh request
    → `prev_state_idx = (0-1)//block_size = -1` → no copy; unchanged column
    → `prev_state_idx != curr_state_idx` false → no copy; the GPU `src_col`
    tensor is defaulted to -1 and only written when a copy is actually
    needed — functionally identical to the patch's
    `src_col = where((num_computed==0)|(state_idx<0)|(state_idx==new_state_idx), -1, state_idx)`.
    The standalone GPU kernel the patch modifies
    (`preprocess_mamba_align_fused_kernel`) is reached only via
    `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` — the V2 runner, not
    live on ROCm. (2) **`chunk_o.py` NaN guard** — absent in our tree, but
    our file is the faithful pre-patch upstream state (not a fork
    divergence): the load already carries `boundary_check` (default
    other=0.0), and no NaN has ever been observed on this stack's chunked
    GDN prefill (identity-verified 120k probes + months of replay/sweep/chat
    serving). The RTX3090 fix targets a codegen artifact on their stack; if
    NaNs ever appear in a partial-chunk GDN prefill here, the one-line
    `tl.where(m_t, b_g, 0.0)` mask is the cheap reviver.
  - **Ideas from [1CatAI/1Cat-vLLM](RECON-1cat-vllm.md) (V100/SM70, "Make Volta
    Fast Again" — same generation class as gfx906; runs Qwen3.6-27B-AWQ TP2, the
    near-identical model to ours). Full recon + estimates in the linked doc.**
  - **CAT-1 — draft-vocabulary shortlisting (DONE — PASS, merge candidate).** Their biggest MTP
    win: shrink the *drafter's* lm_head vocab from full 248k to a static 131K or
    dynamic 98K+2×512 shortlist → **+21.9% e2e** (80.1→97.7 tok/s) on their TP2
    Qwen3.6-27B-AWQ, lossless by construction (target dist stays full-vocab for
    accept/recover; one-time prefill `topk=2048` bootstrap builds the shortlist).
    This is SYV-3's surviving lever: the kernel-level route (SYV-3) was shelved
    2026-09-04 because the stock GEMV path already runs at ~822 GB/s effective,
    so **the byte count itself is the remaining headroom** — a 131K draft vocab
    cuts the drafter lm_head read ~1.9× (in-context measured: full 248k =
    3.09 ms/call at K=1), and int8 on top of that halves it again. **Status:
    DONE — ported + validated (branch `cat1-draft-vocab`, devlog
    DEVLOG-draft-vocab.md): k=2 pilot with syv's 40,960-id list = +3.86/4.38%
    zero acceptance loss; k=4 stacking A/B (2026-09-05) = +5.6/+3.9/+3.8% at
    64k/96k/120k, perfect 4/4 acceptance at every draft position → gain holds at
    depth and stacks on the k=4 win (total ≈ +20–24% over plain k=2 decode).
    Env-gated by construction (sliced work-dir; plain dir = full-vocab path
    unchanged). Remaining before merge to main: Kevin's real corpus → our own
    ~131K list → final A/B acceptance gate on real traffic (s9 filler saturates
    both arms at perfect acceptance — validates mechanism+speed, not coverage).**
    **UPDATE 2026-09-13 — MERGED (T2, `190bee0582`); our own corpus landed.**
    Final artifact: `cat1_ids_v3.json`, **35,251 ids** (every id observed in our
    own pi/hermes traffic + the tokenizer's 33-id control family), head 361 MB
    vs 2.54 GB full. Raw-continuation coverage 97.7–98.0 % → **100 %** once the
    added-token markup (`<tool_call>`, `<tool_response>`, `<think>`, …) is
    forced — parsed logs can never contain it (`CAT1-corpus-build.md`).
    Controlled A/B on identical prompts (11 agentic 8k prompts × 2 reps/arm,
    TP=2 k=2): **−2.52 ms/step [−2.91, −1.94] ⇒ +4.8 % mean / +5.9 % median
    t/s** (Note 2026-09-14: the same arm-labelled client was in use here, so the *acceptance* column of that A/B is void — the arms ran different prompts. The ms/step result is unaffected (at fixed k the per-step cost does not depend on acceptance), and "no acceptance penalty" is now independently supported by the clean V2 re-measure: acceptance 1.98 vs 1.98 @64k, 2.07 vs 2.10 @120k.) The original "acceptance no detectable penalty (MWU z = −0.83)" claim came from the flawed client and is superseded by that re-measure. Agentic-coding
    headline with MTP k=3: **33.25 @64k / 24.95 @120k t/s** (greedy
    19.80/13.17 = 1.68×/1.89×).
    *Follow-ups:* (a) **T2-5 hygiene** — move the lazy `draft_vocab_ids` device
    migration out of `forward` (it relies on warmup preceding capture);
    (b) **the shipped list is not reproducible** from `corpus15` with the
    current builder (a recount gives 34,392 ids — the list predates the builder
    hardening), so a rebuild is a *new* artifact that must re-run the
    controlled A/B, and `cat1_manifest.json` carries the ids↔head pairing +
    snapshot but **no provenance block** until a fresh build writes one;
    (c) the 131K-list option is still unvalidated — our N = observed ids, and
    only a corpus that demands more ids should justify re-gating it.
  - **CAT-2 — FA prefill D256 Split-D + GQA multi-head packing (GO/ANALYZE — feeds
    SYV-9).** Their Volta D=256 prefill kernel = **1.66–2.2× over generic FA2** on
    the same D=256 shape. Techniques: Split-D (D=256→4×D64, paired warps share QK,
    more PV parallelism), **N32 online-softmax as a *quality* requirement** (their
    N64 variant's 1.27e-4 L2 error amplified across layers and changed sampled
    tokens; N32 → ~4.6e-6), GQA multi-head packing (pack 6 GQA query heads into
    wider Tensor-Core work — matches our Hq/Hkv=6 ratio), K-stage ping-pong,
    prefix/causal-tail separation. Our SYV-9 profile: FA kernel = **45% of prefill
    at 120k** → this is the direct target. **Status: OPEN — analyze our Triton
    `gfx906_fa_forward` against head-packing + D-split axes before any port; HIGH
    effort (real FA-kernel rebuild).**
  - **CAT-3 — FP8 E5M2 KV via one-pass expansion (ANALYZE).** Their biggest *prefill*
    FP8-KV speedup: **4.5–4.9×** by a single vectorized `fp8_e5m2_paged_kv_to_fp16`
    gather/expansion into a shared FP16 page-784 workspace (old path re-converted E5M2
    inside *every* query CTA → 96 KiB smem, 1 CTA/SM, ~4% tensor activity). Workspace
    ~512 MiB/rank @256K, reused serially by all full-attn layers. We run **fp16 KV
    today**, so bigger change — but halves KV bytes/bandwidth and the expand-once
    pattern serves SYV-7 prefix caching + long-context decode. **Status: OPEN — needs
    its own FP8-vs-FP16-KV model-level quality gate; HIGH effort.**
  - **CAT-4 — 128-bit wide aligned KV loads in decode XQA (ANALYZE).** PR #268: one
    aligned 128-bit load replaces narrow `half8` fragments, reusing page ID → L1
    global-load requests **−41.5%**, kernel **−23%** (B16/17.8K). Our long-context
    decode is memory-bound on the FA gather; gfx906 equivalent = `v_load_dwordx4`.
    **Status: OPEN — LOW-MOD effort, good first probe for our decode FA path.**
  - **CAT-5 — prefix/causal-tail separation for chunked prefill (ANALYZE).** Their
    superlinear cold-prefill root cause: fixed 1024-token chunks each attend over an
    increasingly long KV prefix → O(L²) work; last 32K of a 64K request = **75%** of
    the prefix-attention sum. Fix: schedule the fully-visible prefix separately from
    the exact causal tail, merge online-softmax state. Directly relevant to our SYV-9
    (FA dominant at 120k prefill). **Status: OPEN — MOD-HIGH effort, pairs with CAT-2.**
  - **CAT-6 — CTA-local K-parallel small-M GEMM (ANALYZE).** Their M=5 verify AWQ GEMM
    = 6–12% occupancy (68 CTAs/72 SMs); intra-CTA K-split (`1x4x1`→`1x4x2`, FP32
    partials reduced in smem, no extra global workspace). Their target-forward AWQ
    GEMM = 44% of verifier forward (same as ours), but our decode GEMMs are near the BW
    ceiling (SYV-3) so value is uncertain. **Status: OPEN — only if a decode-GEMM profile
    shows occupancy loss at our shapes; MOD effort.**
  - **CAT-7 — DFlash2 block drafter + LABD/ngram lookup (POSTPONE).** Their ~260 tok/s
    headline uses the NVFP4 whole-block non-autoregressive DFlash2 drafter (+ optional
    lookup-augmented / prompt-ngram drafting). Same idea as parked **SYV-8** (NVFP4
    checkpoint doesn't transfer to our AWQ; needs V2 runner conflicting with FULLGRAPH).
    The *ngram/lookup* sub-idea maps to **SYV-2** lookahead-drafting. **Status: POSTPONE —
    revisit only if MTP stops delivering.**
  - **CAT-8 — persistent partition-grid cap (NO-GO).** Their Flash-V100 fixed decode
    grid-capping was bitwise-exact but *slower* (register growth + persistent control
    ate the saving; regressed at 65K/262K). Recorded as a dead-end reference so we don't
    re-tread it. **Status: NO-GO.**
  - **CAT-9 — FP8 prefill tile-selection pitfall (ANALYZE — caution for SYV-9/CAT-3).**
    Their FP8 prefill regressed with context until they fixed page-size→BM32-phase
    selection and removed per-CTA E5M2 expansion. Adopt as a *design constraint* on any
    int8/FP8 prefill port: right tile/phase at every page size, never convert inside each
    query CTA. **Status: OPEN — design constraint, not a standalone task.**

  - **J2G-1 — persistent all-reduce** (from
    [joe2gaan/localaiservers](RECON-joe2gaan-localaiservers.md), TP=8 host).
    Attacks TP comm cost — the per-step work our phase profile could NOT
    attribute (outside any hookable module; hooked modules = ~38% of MTP step).
    Their prebuilt `.so` is topology-specific (TP=8); the env-only RCCL knobs
    (`NCCL_ALGO/PROTO/CHANNELS`) are the transferable first probe.

    **ENV A/B COMPLETE 2026-09-03 — WIN, default ON.** Sequential TP=2 greedy
    arms (corpus s9, n=5/point, medians; design + raw data in
    `/local/tmp/j2g1/`):

    | arm | env | @120k ctx | Δ vs A0 | @64k ctx | Δ vs A0 |
    |-----|-----|----------:|--------:|---------:|--------:|
    | A0 | none | 12.762 t/s | — | 18.916 t/s | — |
    | **A1** | `NCCL_ALGO=Tree` + `NCCL_PROTO=LL` | **13.115 t/s** | **+2.77%** | **19.723 t/s** | **+4.27%** |
    | A2 | A1 + `MIN/MAX_NCHANNELS=4` | 13.105 t/s | +2.69% | 19.759 t/s | +4.46% |

    - Pass bar (≥2% @120k, no new wedges): **A1 PASSES** (+2.77%, all reps
      within ±0.3% — not noise). A2 ≈ A1: channel pinning adds nothing; the win
      is Tree+LL alone. No wedges in any arm; clean teardowns.
    - MTP-workload check: canary with Tree+LL = 39.1 t/s (baseline class
      ~39–47) → no regression on the real speculative path.
    - **Applied default-on** in `run_server.sh` (`NCCL_ALGO=Tree`,
      `NCCL_PROTO=LL`) with knob-source attribution to joe2gaan's profile
      (standard env vars, no code port).
    - **Remaining: full persistent-AR port** (their prebuilt `.so` is TP=8; a
      gfx906 TP=2 build is the escalation — see J2G-3) and J2G-2 AR pre-fold.

  - **J2G-2 — AR residual pre-fold** (`VLLM_GFX906_AR_PREFOLD_ENABLE`, from
    `communication_op.py`). Algebraic identity: `allreduce(partial + residual/TP)
    == allreduce(partial) + residual` — fold the layer's residual into the AR
    input so the *next* layer's reduction carries it, saving one fused add per
    TP step (only where shape[-1]==5120, fp16/bf16, contiguous). Strict opt-in:
    dtype/rounding semantics must be validated before trusting numbers. This is
    a **decode** win (per-step comm + FLOPs), complementary to J2G-1's env knobs
    and the persistent-AR port. **Status: new candidate — read their gate logic,
    scope a strict A/B on our TP=2 dense 27B.**
  - **J2G-3 — hand-tuned RCCL_TREES + custom librccl overlay.** Beyond the env
    knobs in J2G-1, they ship a prebuilt `librccl.so.1` and an explicit
    `RCCL_TREES='(0(1(3)(4))(2(5(6(7))))|...)'` (4 ring/tree permutations for 8
    ranks). For our TP=2 there's only one edge, so the *trees* don't transfer —
    but if J2G-1's env A/B shows a win, the next step is testing whether a
    gfx906-specific RCCL build beats stock. **Status: new candidate — only worth
    it if J2G-1 lands positive.**
  - **J2G-4 — row-parallel mutable AR + boundary cut** (`VLLM_GFX906_ROWPAR_...`,
    `ROWPAR_BOUNDARY_MLP_SHAPES=2176x5120`). Splits the row-parallel GEMM/AR
    boundary at a specific MLP shape so the reduction overlaps the next compute.
    Topology/shape-specific (their Qwen3.6 27B MLP); would need re-deriving for
    our model's shapes. **Status: new candidate — low priority, high effort.**
  - **J2G-5 — tuned Triton MoE block configs** (`vllm_tuned_moe_configs/
    E=256,N=128,device_name=AMD_GFX906.json`). Per-batch-size BLOCK_M/N/K +
    warps/stages/waves_per_eu/matrix_instr_nonkdim tuned for gfx906 MoE. Only
    applies to MoE models (we run dense 27B now); note the shipped config is
    Qwen3.6-shaped (E=256, N=128) — a Nemotron-H (g64) port would need its own
    autotune sweep. **Status: new candidate — relevant when we serve a gfx906 MoE.**

  - **J2G-6 — post-AR consumer fusion (allreduce + residual + RMS epilogue).
    ANALYZE.** Deep-review find (2026-09-07, `gfx906-key-learnings-20260606.md`
    + source inventory). Their dense 27B TP8 profile: the post-AR
    residual/RMSNorm consumer costs 0.0325 ms of the 0.0893 ms
    `1x5120` MLP-down boundary; a fused `allreduce → add+RMS` kernel beat the
    decomposed chain by 0.023–0.217 ms/call (lower bound ~1.47 ms/token over
    64 boundaries) — but **no serving win at TP8** because NCCL itself
    dominates (54 %). Our TP2 geometry inverts that ratio: the single P2P
    edge is far cheaper than their 3-channel TP8 primitive, so the consumer
    pass is a bigger fraction of our boundary. J2G-2 (pre-fold) is the
    algebraic variant; this is the fused-kernel variant. **Status: OPEN —
    LOW-MED; needs a per-boundary profile of OUR TP2 AR+consumer first
    (we have no TP2 AR-cost measurement — our phase profile couldn't
    attribute the unexplained 1.55 ms/step, G1 territory).**
  - **J2G-7 — custom interleaved SwiGLU MLP GEMV (weight-interleave repack +
    fused activation epilogue). ANALYZE.** Their "native interleaved SwiGLU"
    is 13 % of TP8 decode kernel time (2.8 s of 21.7 s profiled) and part of
    their high-water stack: gate/up rows interleaved in the weight layout so
    one GEMV pass + fused SiLU-mul epilogue replaces GEMV + activation
    kernels; their corrected lower bound = ~3.2–3.5 µs/layer saved, and the
    serving form needs a packed/interleaved weight repack. For us: the Qwen
    MLP is AWQ INT4 — the same interleave applies to the packed layout
    (gate/up rows interleaved in the INT4 pack), with the fused activation in
    our existing dequant-GEMV epilogue. Our decode GEMMs are near the BW
    ceiling (SYV-3), so the win is launch/epilogue elimination, not bytes:
    ~3.5 µs × 48 dense layers ≈ 0.17 ms/step ≈ ~1.2 % of a 14 ms step. ISA
    note from their work: the gfx906 assembler rejects `v_dot2c_f32_f16`; a
    valid `v_dot2_f32_f16`-style half-dot replacement exists (their
    `gfx906_llmm1_dot2` / wvSplitK patches) — but their own dot2 replacement
    *regressed* the dominant shape, so treat it as an available instruction,
    not an automatic win. **Status: OPEN — MED-HIGH effort (weight repack +
    GEMV variant + serving gate); value ~1 % class — do only if a decode-step
    profile confirms the MLP activation pass is still a separate kernel on
    our path.**
  - **J2G-negative evidence (deep review 2026-09-07, recorded so we don't
    re-walk it):** (1) sequence parallelism — token-shard SP is a 35–59 %
    boundary win in microbench but **rejected in serving**: vLLM overrides
    SP cudagraph capture sizes to [8,16], killing the c1 num_tokens=1 graph
    path; reduce-scatter for c1 decode is slower than allreduce on
    gfx906. (2) vLLM's CUDA custom-allreduce substrate is not a gfx906 path
    (peer IPC init faults / post-prefill hangs; matches our
    `--disable-custom-all-reduce`). (3) grouped/coalesced allreduces are
    infeasible on the Qwen dense graph: 128/128 adjacent AR boundaries are
    blocked by true hidden-state dependencies (64 MLP + 64 attention
    producers). (4) Marlin tile autotuning is a wash end-to-end under
    sustained power/clock throttling (+2–20 % standalone, +0.4 % e2e on
    their 250 W-capped 3090) — the DVFS analog of our standalone-≠production
    trap. (5) their persistent-AR sidecar history: per-call AR start/stop is
    ~3.6 ms/call (resident worker required); sub-1 % descriptor/primitive
    tweaks never promote at serving scale — the gate is launch count / p99
    tail, not median latency.

- **MTP-1c — dynamic MTP logic.** Investigate runtime-adaptive spec decode:
  (a) disable MTP when context length exceeds the crossover or draft
  acceptance is low, or (b) change MTP depth dynamically (k=2 → k=1/k=0) by
  context length / content shape. First determine what this fork's
  spec-decode engine exposes for per-request or per-step control; if absent,
  scope the patch. Gate: correctness + a serving A/B on a **mixed-context**
  workload showing net positive vs the best static config (a dynamic policy
  must beat the static optimum it is trying to replace).

  **NOTE (2026-09-02): MTP-1b-0 changes this item's premise.** With the kv_split
  clamp fixed, k=2 may win at long context again, so the dynamic policy becomes
  "pick k by context/content" rather than "disable MTP past a crossover." The
  fork's existing dynamic-SD mechanism (`vllm/v1/spec_decode/dynamic/`) keys K
  on `len(num_scheduled_tokens)` — per-step batch token count, NOT context
  length (scheduler.py:1256) — so a context-length-keyed policy is new machinery.

**Stop rules:** (1) MTP-1c is design work only until MTP-1a pins the bracket
and MTP-1b shows a real remaining win — no dynamic-MTP machinery on an
unmeasured crossover. (2) Long-context runs sit in the GPU-degradation-risk
zone (`degradation.md`) — run the canary before each sweep and stop after any
reset burst. (3) TP=2 requires the official amdgpu DKMS driver; if it is not
the active driver, do not force a fallback and record it.

## High priority — user-requested (2026-09-04): startup-time items

Both items below were requested 2026-09-04 ("add to roadmap: improve graph
and inductor creation speed; improve shard loading time"). Each item's
**Step 0 is a search for existing work** (upstream + our repo) before any
implementation — the seeds listed are starting pointers, not conclusions.

### S1 — improve graph + inductor creation speed (startup compile/capture) — **COMPLETE (2026-09-04)**

**Result:** the startup bottleneck was NOT compile/capture. The 27B checkpoint is
multimodal (`Qwen3_5ForConditionalGeneration`, 333 `vision.*` tensors) and every
startup ran a max-feature-size **dummy image through the ViT** in `profile_run()` —
measured at ~213 s of the 234 s warm engine-init (stack-dump proof in
`DEVLOG-s1-startup.md`). Fix = stock flag `--language-model-only`, now the arm
default in `/local/tmp/mtp1/run_server.sh` (`FULL_MM=1` opts out). **Measured:
warm engine init 233.85 s → 14.54 s (16×)**; dev boot is now weights-load-bound
(~90–100 s end-to-end — S2 territory, deprioritized). KV pin (`KV_MEM_BYTES`)
opt-in for dev boots; `-O1` deprioritized (cache audit: AOT cache survives reboots,
cold compile is one-time per config key). No code port → no README attribution
needed (flag-only adoption; documented in the dev log).

**Scope:** cut the torch.compile (dynamo trace + inductor codegen/autotune)
and CUDA-graph capture portion of engine startup on this host — cold cache
first (new model / config change / venv rebuild), warm-start second.

**Known baseline (boot U, 2026-09-04, TP=1 dense 27B A/B logs):**
weight load 29–31 s/arm; graph capture **~1 s** (already trimmed to
`cudagraph_capture_sizes [1,2,3,4]` — little left there); the compile phase
is the open unknown for a COLD cache (warm runs hit
`~/.cache/vllm/torch_compile_cache`, which persists on this host, so warm
startup is not yet measured either). Multi-arm A/Bs pay startup per arm
(~1 h of today's run was ~4 launches), so this compounds in dev workflow.

**Step 0 — existing work (search before implementing):**
- Upstream parent issue **vllm-project/vllm#19824 "Improve startup time UX"**
  (breakdown: P2P check, weight load, dynamo trace, inductor compile +
  autotune caching, cudagraph capture, PTX JIT; proposals: lazy cudagraph
  capture, faster-startup regimes, Inductor parallel codegen).
- Upstream RFC **#20283 / PR #26847 — `-O` optimization levels** (`-O1` fast
  startup vs `-O2` full optimization): if our fork predates it, adopting a
  dev-fast `-O1`-class mode is the cheapest win.
- Upstream RFC **#27080 — Inductor partition**: 2–5× COLD compile cost on
  torch 2.9; check whether our fork has `use_inductor_graph_partition` on —
  if so, disabling it for dev iterations is a direct lever (warm start is
  actually slightly faster with it).
- PR **#10460** (reduce inductor compile time: single symbolic-shape graph),
  **#10482** (limit inductor threads / lazy quant import).
- Ours: `docs/gfx906/DEVLOG-boot-failure.md` (compile-cache-hit line on a
  clean boot); verify the cache dir survives reboots and venv rebuilds, and
  whether it is shared across TP=1/TP=2 arms (cache key includes parallelism).

**Task:** (a) measure the cold-start phase breakdown on this host (clear
`~/.cache/vllm/torch_compile_cache`, time import/config/compile/capture);
(b) rank levers by effort÷gain (cache sharing across arms, `-O1`-class dev
mode or equivalent flags, inductor thread/parallel settings, capture-size
policy); (c) implement the top lever(s) behind env flags.

**Gate:** measured cold-start reduction with **no decode t/s regression**
(compile-mode changes must pass a same-boot t/s A/B per house recipe).
Record in ROADMAP + relevant devlog; upstream-derived levers need
attribution (README + inline comment).

### S2 — improve shard loading time (weight load) — **DEPRIORITIZED (Kevin, 2026-09-04)**

> **Status:** "Shard loading time improvements are not high priority though so do not spend too much time on it." fastsafetensors — the main existing lever — is ruled out: it reserves more VRAM than the usual loaders ("This is not a good solution if this is the case"). Everything below is retained for reference; do not execute without a new trigger (e.g. load >3 min cold, or NFS-backed serving).

**Scope:** cut the "Loading weights took" portion of startup for the dense
27B AWQ checkpoint. It is currently the single largest measured startup cost.

**Known baseline:** 29–31 s/arm on boot U (TP=1, NFS cache
`/data/cache/huggingface`, ~5 shards ≈ 14.8 GB INT4 → effective read+copy
≈ 0.5 GB/s). **Prior in-repo work — build on it, don't redo it:**
`docs/gfx906/running.md` §0: `fastsafetensors` measured **41 s vs 117 s
(2.6×)** but NOT adopted for the dense NFS model (GDS unsupported here;
fork's one-line GDS-fallback fix = U1 below; +2.8 GiB live VRAM at init →
forced util 0.95).

**Step 0 — existing work (search before implementing):**
- Our U1 item (upstream queue): fastsafetensors GDS-fallback catch, local
  commit `128e948baf` — check whether it is in the fork's HEAD.
- Upstream **PR #40183** — fastsafetensors `ParallelLoader` + pipelining
  (`VLLM_FASTSAFETENSORS_QUEUE_SIZE`): ~10% on released fastsafetensors,
  ~4× once fastsafetensors' unified-memory copier (#60) lands. Check our
  fork's loader path against it (pre- or post-PR).
- Upstream **PR #29410** (make fastsafetensors the default load format) and
  the fastsafetensors paper (arXiv:2505.23072, 4.8–7.5× — but on local NVMe;
  our storage is NFS).
- **NFS-specific:** measure the actual read ceiling of
  `192.168.33.240:/volume2/ai` (single-stream vs parallel-shard reads,
  page-cache warm vs cold) before choosing a loader — if ~0.5 GB/s is the
  NFS ceiling, no deserializer change helps and the lever becomes storage
  (local mirror / tmpfs staging), which is an infra decision for Kevin.

**Task:** (a) measure where the 29 s goes (NFS read vs deserialization vs
H2D copy); (b) rank levers: parallel shard reads in the current default
loader, porting #40183-class pipelining if absent from our fork, or storage
change; (c) implement + A/B.

**Gate:** load time ≤ baseline with no VRAM regression beyond a documented
amount and identical loaded weights (greedy fingerprint match vs baseline).
Record in ROADMAP + devlog; attribution per house rule for any ported code.

## Tier 0 — cheap, decisive, low-risk

### GDN-1 — SYV-10 bounds-port test coverage (T4-2 test debt)

**Status: open, low effort, GPU tests.** The upstream PR #50021 port is
merged (T4, `fd6895e789`) and is a real safety fix — the accepted-token-derived
state index (`i_t = num_accepted − 1`) could read before/past the request's row
and fault the SM, because the `state_idx <= 0` guard alone accepted a garbage
positive int. It was **inspected but never gated**: add one test per zero-fill /
early-out path (`causal_conv1d`, `mamba_ssm`, `fused_recurrent`,
`fused_sigmoid_gating`), plus one that feeds an out-of-range accepted count and
asserts the output is unchanged rather than a fault. Cheap, and it retires the
only "shipped without a runnable check" item in the T4 train.

### DE-1 — dead-end register-spill / compiler-structural audit (HIGH PRIORITY)

**User request 2026-08-31.** Every HIP-kernel row in `DEAD-ENDS.md` is
re-examined for compiler-caused underperformance: VGPR/AGPR pressure,
register spills, occupancy loss, bad vectorization — i.e. failures that may
be *fixable by restructuring* rather than dead by hypothesis. Method per
kernel: (1) locate the source (in-tree flag, branch, or `/tmp` prototype),
(2) compile it standalone for gfx906 with register-usage reporting and
record VGPRs/AGPRs/spills, (3) check whether the measured shortfall is
consistent with spill/occupancy loss (gfx906: 64-lane wavefronts, 40
waves/CU max = 2560 threads; register file per CU pinned via runtime API —
see audit), (4) verdict — **fixable → open a branch** with the restructure +
serving gate; **not fixable** (HBM floor, structural design flaw,
graph-regime transfer failure, or scope cap) → mark the row `FULLY DEAD` in
`DEAD-ENDS.md` with the compiler evidence. Non-HIP rows (Triton codegen,
Python/serving, analytic, memory-fit) are classified and closed without
branches.

**STATUS: DONE 2026-08-31.** Verdict: **zero dead-ends failed from register
spills or measurable register pressure** — `vgpr_spill_count = 0` on every
in-tree HIP-kernel dead-end (VGPRs 12–93; highest, gemm1 re-tile family at 79,
is LDS-limited by design and failed on wall-clock transfer, not per-wave
throughput). The two primary suspects confirmed non-pressure: FA V2 = 16 VGPR
vs shipped V1's 12 (both spill-free → the 7× serving loss is grid-shape/
scheduler, not compiler); gemm1 V1 single-wave full-K was structural by
construction (one wavefront/block, K-looped; 64 × 128 KB streams can't stay in
flight). 13 rows annotated `FULLY DEAD` in `DEAD-ENDS.md`; **no branch opened**
(no fixable-by-restructure case found). Open rows T1/T5 untouched. Full record:
[`DEAD-ENDS-AUDIT.md`](DEAD-ENDS-AUDIT.md); rerunnable harness
`/local/tmp/spill_audit.py`, raw tables `/local/tmp/de1_audit_results.txt`.

### G1 — decode-graph per-node replay-cost probe

**STATUS: DONE 2026-08-31 — NODE COUNT IS NOT THE OWNER (hypothesis killed).**
Probe `benchmarks/kernels/gfx906/g1_node_replay_probe.py` (decode-shaped
graph, N ∈ {0,16,32,64} dummy no-op kernels/layer, wall-clock A/B replay):
**~1.2 µs/node TP=1, ~1.1 µs/node TP=2** — linear across the whole range, an
order of magnitude below the ~10 µs needed for node count to own the
1.55 ms/step. 16–32 extra nodes/decode step ≈ 0.02–0.04 ms/step (~2 % of the
unexplained cost). The remainder lives in TP=2 sync placement / other
LEGACY=0-common per-step work (eager TP=2 can't isolate it — documented).
Consequences: the refrigerated Q8-fusion lever stays refrigerated (halving
~16–32 nodes saves ≤ ~0.03 ms/step, cannot close a 6 % gap); future
adds-nodes-per-step proposals now carry a citable budget of **~1 µs/node**.
Full record: `DEVLOG-fa-legacy0-b1-decode.md` (G1 addendum).

The 2026-08-29 same-boot LEGACY adjudication
(`DEVLOG-fa-legacy0-b1-decode.md`, boot O) left a bounded-but-unexplained
**~1.55 ms/step** serving cost common to both LEGACY=0 arms, with the
extra captured-graph nodes (~16–32 per decode step: one Q8 side-buffer
write + slot cast per full-attn layer) as the leading unmeasured
hypothesis. Not further decomposed there — the obvious tools are blocked
on this stack (chrome-trace GPU timestamps are not wall-aligned; eager
TP=2 collapses ~3× from launch overhead). This is NOT LEGACY=0-specific:
any change that adds per-layer kernels to the captured decode graph
(MoE routing fusion, spec-decode extensions, future KV-side writes) pays
the same invisible per-node replay cost, so the measurement gates a
family of decisions — including how much C1's fusion is worth.

**Probe (cheap, no model change):** capture the standard decode graph,
then A/B replay wall time with N dummy no-op kernel launches appended
per layer (N ∈ {0, 16, 32, 64}), TP=1 and TP=2, same boot. Falsifiable
both ways:

- per-node ≤ ~10 µs → 16–32 nodes explain ≤ ~0.3 ms of the 1.55 ms →
  node count is NOT the owner; suspect TP=2 sync placement / other
  LEGACY=0-common per-step work; the Q8-fusion lever stays refrigerated.
- per-node ≈ 50–100 µs → nodes explain the remainder → the refrigerated
  lever (fuse the Q8 write into `triton_reshape_and_cache_flash`,
  halving the nodes) reopens with a real bound — and every future
  adds-nodes-per-step proposal must budget it.

Gate is wall-clock A/B only (harness or serving). Prior-probability
note: ~50–100 µs/node would be unusually high for graph replay, so G1
is more likely to kill the hypothesis than confirm it — either outcome
is cheap and decisive.

### HK-1 — drop the legacy `~/env-rocm-7.14-gfx906.sh` sourcing

**Status: in-repo recipes DONE (2026-08-31, branch
`gfx906/hk1-drop-env-sourcing`); `/local/git/AGENTS.md` pending — its edit
is a protected-file write awaiting user approval.** The machine has a
single ROCm toolchain now (/opt/rocm is the default), so sourcing it is
unnecessary — **confirmed 2026-08-29 (boot N)**: both prime dense models'
TP=2 serving boots, the 74/74 in-process suite, and the FA micro-bench runs
all worked without it; re-verified 2026-08-31 with a torch HIP matmul under
`env -u ROCM_PATH -u LD_LIBRARY_PATH`. Removed from the ACTIVE recipes:
`running.md` (§0 + build section), `docs/gfx906/README.md` (bench recipe),
`.agents/skills/gfx906-mem-attribution/SKILL.md` (the in-repo skill's
recipe). Dev logs and `degradation_details.md` keep their lines (historical
record); the session `canary.sh` sources it — drop there too when next
touched. No code change needed: without `ROCM_PATH` the build resolves the
toolchain via torch's cpp_extension (wheel-download path via
`setup.py is_rocm_system()`), both landing on /opt/rocm here.

### N1 — quiet the expected AutoAWQMoEMarlin fallback

**Status: SHIPPED, merged to `main` (2026-08-31, ff of
`gfx906/n1-awq-fallback-quiet`; self-review + Claude CLI review — no
blockers, dead-import cleanup + once-log cache fixture applied).**
On gfx906 the fallback to the custom WNA16 path is intentional:
`get_quant_method` now emits one `info_once` line per process instead of a
per-layer warning when `current_platform.is_rocm() and on_gfx906()`; all
other platforms keep the (deduped) warning. Returned quant method identical
in both branches. Behavior gate:
`tests/quantization/test_auto_awq_gfx906_fallback.py` (2 tests; verified to
FAIL if the production change is reverted). See
`vllm/model_executor/layers/quantization/auto_awq.py`.

## Tier 1 — decode fast path

### C1 — fuse the routing pipeline (~1 ms/step)

**Status: stage 1 SHIPPED and merged to `main` (verified 2026-09-15:
`feat/moe-c1-routing-fusion` is an ancestor of `main`); stage 2 DEAD-END. Item closed as an active fusion item — the remaining
topk component is conditional (see below), gated on G1 + a design that
does not replace the production topk kernel in place.**

The M=1 decode routing chain is 3 kernels/layer (topk 11.8 + align
3.8 + count_and_sort 3.8 µs/node isolated; ≈ 0.8 ms/step at 40 layers,
M-independent — latency-bound; structural probe in
`c1_routing_structural_probe.py`). Both stages were gated by serving
A/B, not isolated kernel numbers, as the item required:

- **Stage 1 — fused align+count (120 → 80 nodes): SHIPPED.** One
  128-thread CTA replaces the align 2-block + count_and_sort pair,
  bit-equal to the generic chain
  (`moe_align_block_size_m1_gfx906`, `VLLM_GFX906_ALIGN_M1` default ON).
  Serving A/B (Qwen3.5-35B, pp2048/tg256, 4 samples/arm, same boot,
  back-to-back control): **+1.18 % (207 µs/step) / +1.73 %**, within 8 %
  of the isolated prediction (224 µs/step). Node removal transfers to
  serving.
- **Stage 2 — fused topk+align+count (120 → 40 nodes): DEAD-END.**
  One-CTA kernel (S2's bit-exact topk phase + stage-1's align phase),
  28 % faster per node in isolated graphs (10.0 vs 13.8 µs/layer), yet
  **−1.10 % in serving** (A-B-A: 57.42 → 56.79 → 57.46 t/s). Third
  confirmation of the S2 flip (2026-08-29); the stage comparison
  pinpoints the mechanism: REMOVING redundant nodes transfers; REPLACING
  the proven production topk kernel does not (S2: −1.03 %, stage 2:
  −1.10 %). Landed behind `VLLM_GFX906_ROUTING_FUSE_M1` (default OFF)
  with router→expert meta plumbing; **removed from the tree 2026-09-01**
  (maintainability sweep — code preserved on branch
  `gfx906/preserve-dead-kernels`, including the now-orphaned
  `_fused_align_meta` plumbing).

The standalone M=1 top-k specialization (S2) and stage 2 both lost in
the CUDA-graph regime, so the remaining topk cost (~470 µs/step
isolated) is open only under a design that does not swap the production
topk kernel in place (e.g. fold top-k into the router GEMV epilogue, as
originally scoped) — and G1 should price the per-node tax first. See
`DEVLOG-moe-m1-sprint.md` for the negative top-k result and
`DEVLOG-moe-c1-routing-fusion.md` (both stages, evidence + plumbing).

### C2 — decode-sized routed GEMM (decide the built wins; finish the axes)

**Status: M=1 default-on decision SHIPPED (2026-08-31); V1 N-split axis
CLOSED + harness PASS flow re-run (2026-08-31, `gfx906/c2-finish-axes`);
BM≥2 grouped path still open.** The combined TP=2 M=1 A/B + numerics gate
resolved the
default-on question: `VLLM_GFX906_MOE_M1` (gemm2 v2 tile) and
`VLLM_GFX906_MOE_NPT=2` (gemm1 `<1,2>` re-tile) are now **default-on
for the M=1 decode path** — +5.0 % decode at TP=2 M=1 (81.58 →
85.65 t/s), +2.9 % at TP=1 M=1, output fingerprint identical across
all 8 arms. Non-qualifying gemm2 shapes fall back silently; env opt-out
retained (`MOE_M1=0`, `MOE_NPT=4`). The tested batch arm was neutral
because it takes the unretiled BM≥2 grouped path; that path is still
unmeasured, not rejected. Remaining:

- measure and, if useful, re-tile the BM≥2 grouped path for concurrent
  decode (the only open axis — needs a multi-hour serving A/B session);
- ~~build the V1 N-split/direct-store variant (128/256/512 blocks)~~
  **CLOSED 2026-08-31**: all five V1 variants correct; every new N-split
  point is SLOWER than the existing best V1 point (v1b, 64 blocks @ 59.0 µs),
  and none comes within 2.1× of current (best N-split v1d, 256 blocks:
  74.8 vs 28.7). Adding blocks to shorten the per-block stream buys nothing;
  the v1a-vs-v1b pair isolates wavefront config as a ~2× effect, but the
  N-split variants confound stream length with wavefront count (mechanism
  not cleanly isolated — see devlog). Axis is measured-and-rejected at every
  block count, no serving gate reachable at that margin (`DEVLOG-moe-c2v.md`
  "V1 N-split axis"; DEAD-ENDS row annotated);
- ~~rerun the corrected standalone harness PASS flow~~ **DONE 2026-08-31**:
  `HARNESS PASS` ×4 (boot P, clean host), v1a/v1b bands match the 08-19
  records within 2.5 % — old S5 microbenchmark numbers re-validated.

See `DEVLOG-moe-c2v.md` (incl. "Combined default-on decision" and
"V1 N-split axis") and `DEVLOG-moe-gemm1-retiling.md`.

- **C8 — expert-weight residency measurement (feeds C2's target
  selection).** Measure L2/TCC hit and miss behavior for the roughly
  12 MB of active W4 weights per layer. The result determines whether
  C2's target should be based on the HBM floor or on latency/occupancy
  rather than on the current kernel's apparent bandwidth.
  **DONE 2026-08-31** (measurement; `DEVLOG-moe-residency.md`): combined
  active W4 set = 12.47 MB > 8 MB L2/TCC ⇒ not fully resident, but the
  production gemm1 `<1,4>` M=1 kernel achieves only ~195 GB/s ≈ **24% of the
  HBM floor** / <64% of its working set's achievable read BW. Binding
  constraint at M=1 is **latency/occupancy (MLP), not the HBM floor** ⇒ C2's
  target = close that read-BW gap (~2×+ headroom before bandwidth binds).

### C3 — fold the two MoE zeroings (234 µs/step)

**NO-GO 2026-09-01** (measured; `DEVLOG-moe-c3-zeroing-fold.md`). Phase A
folded `w1_out.zero_()` into the single-CTA M=1 align kernel (stream-ordered,
not P2-0b's racy in-GEMM clear) — bit-correct (fingerprint identical; 78/78 +
67/67 green) and confirmed firing (~37 MoE layers/step). But the FULL_DECODE_ONLY
serving A/B (N=1 M=1, pp2048/tg256, TP=1) is a **wash within noise**: on median
85.42 vs off median 85.62 t/s (the +0.7% "mean" was one noisy off rep). Root
cause: this model's `w1_out` is only 8 KB (moe_intermediate=512), so removing
~40 tiny memset nodes saves ~tens of µs/step (~0.3% of a ~15 ms step) — below
the A/B's noise floor, and the fold adds an 8 KB store to align that offsets it.
`output.zero_()` (the other half) is alias-blocked and not pursued for the same
reason. Fails the user gate ("no gain ⇒ do not merge"); **not merged**. Would
only clear the bar on a larger-moe_intermediate MoE model, none in scope.

`w1_out.zero_()` and `output.zero_()` cost about 234 µs per step and are
required by the current atomic K-split/aliased-workspace design. If C2
does not replace that design, fold the zeroing operations into the
neighboring routing/activation kernels without changing the
`gemm1 -> activation -> output.zero_() -> gemm2` ordering. The gate is
bit-correctness plus serving A/B; the common-workspace alias must be
preserved.

### N3 — GDN state-bookkeeping copies (~180 µs/step)

**CLOSED 2026-08-31** (measured, no code change; `DEVLOG-gdn-n3-state-copies.md`).
Attribution probe (`benchmarks/kernels/gfx906/n3_state_copy_probe.py`,
in-process, 32 profiled decode tokens, Qwen3.5-35B-A3B-AWQ) run in both
regimes: eager and production `FULL_DECODE_ONLY`. Copy-class op
invocations per step drop **~214 → ~57 (−73 %)** under graph serving —
`clone`/`contiguous` −99 %, `copy_` −68 % (residual + `_to_copy` are the
state/metadata bookkeeping outside the captured region). The eager
~180 µs/step was per-op CPU **launch overhead** on 192-B `[3,1,32]`
copies, not GPU work; CUDA-graph capture absorbs it (same mechanism as
the FA decode copy pile). Residual ~57 tiny copies/step is bounded well
under 60 µs/step and realistically sub-µs against a ~1.5 ms step — not
worth an upstream patch or custom kernel. Disposition: closed, "upstream
code, small", now backed by a measurement. Graph-serving gate (required
because launch-latency-bound) = the graph arm above.

### C5 — fuse the shared-expert chain (150–250 µs)

The shared expert is already dense fp16. Its w13, activation, and w2
operations are individually near their measured GEMV/LLMM1 optima; the
remaining opportunity is a single chain kernel that removes two launches
per layer. Expected benefit is roughly 150–250 µs after the serving
critical-path discount. The existing shared down-projection GEMV is
shipped and is not a separate open item. See `DEVLOG-moe-m1-sprint.md`.

### N2 — B>1 FA direct store

For real single-token batched decode, store directly into `[B,Hq,D]` and
remove the remaining per-layer BSHD reshape copy. B=1 is already
copy-free; this is a separate decode-specialized kernel and launcher
change.

## Tier 2 — bigger / conditional bets

### INT8-PACKED-1 — compressed-tensors `pack-quantized` int8/W8A16 support (blocks the DFlash2 INT8 arm)

**Status: open.** `lued/Qwen3.8-27B-INT8-W8A16-DFlash2` (W8A16, 29.6 GB — fits 2x MI50 at
TP=2) stores every quantised linear layer in compressed-tensors' *packed* form
(`weight_packed` + `weight_scale` + `weight_shape`, 401 tensors including `embed_tokens`
and the GDN `in_proj_qkv`/`in_proj_z`), and loading fails with `ValueError: There is no
module or parameter named 'embed_tokens.weight_packed' in Qwen3_5Model`. Our tree ships
unpacked W8A16/W8A8 schemes (`compressed_tensors_w8a16_channel_dequant`, `w8a16_fp8`,
`wNa16`, ...) but the packed int8 path exists only for nvfp4/mxfp4
(`kernels/linear/nvfp4/humming.py`). Scope: the unpack kernel plus the scheme/loader
plumbing. Gate: the INT8 model serves and passes PPL; then DFL2-1's INT8 pairing arm runs.

### VIT-2 — head_dim-96 instantiation for the ViT (cut the 72 → 128 padding waste)

**Status: open, queued follow-up to VIT-1 (2026-09-15).** The ViT's real head dim
is 72 and the launcher dispatches only {64,128,256}, so it is padded to 128 and the
kernel does 128/72 = 1.78× the arithmetic the model needs. Measured basis
(`bench_vit_dscale.py`, launch-regime, H=16 S=2304, mclk 800 MHz, DEVLOG-vit1.md):
cost tracks the **padded** dim — head_size 72/80/96/112/128 (all padding to 128)
cost **7.68–8.00 ms**, the D=64 instantiation **3.205 ms** — so the 56 zero dims are
pure cost. A 96-wide instance removes exactly 25 % of the head-dim arithmetic:
**~5.5–6.0 ms vs 7.78 ms (−22…−29 % on the ViT attention)**, worth ≈ **−0.25 s TTFT
at 1024×1024 (−5 %)** and −0.7 % at 512×512 (the ViT is ~19 % of the custom path's
TTFT there; VIT-1's own win is already banked). **Accuracy is unaffected**: q8_0
blocks are 32-wide, so D=96's three blocks are the first three of today's D=128
(dims 64–95 = 8 real + 24 zeros either way) — only the redundant all-zero fourth
block disappears; measured rel err is padding-independent (0.0156 / 0.0172 / 0.0199
/ 0.0182 at head dim 64 / 72 / 96 / 128). Note 72 itself is unusable (not a
multiple of 32), which is why 96 is the target.
*Work*: add a `(DKQ=96, DV=96)` tile-config entry (nthreads, occupancy, nbatch_fa,
nbatch_K) and instantiate only the `ncols1` the ViT selects (64 for Sq > 32, per the
ladder in `gfx906_fa_launcher.cu`); read the VKQ/LDS paths for DV assumptions first;
then `_pad_head_dim` learns 96. *Risks*: config quality dominates — the ±25 %
per-dim spread between the 64 and 128 entries means an untuned 96 entry can come out
**slower** than the padded 128 path; build time/TU size grows; the launcher's
`head_dim` switch is shared (a mis-keyed entry can shadow D=128 users) and any other
caller with head_size 80–96 moves onto the new kernel. *Gate*: standalone at the ViT
shapes **and** the FA suite (D=128 unchanged) **and** a one-image-prompt TTFT A/B
(−5 % @1024) — keep it behind an opt-in (`GFX906_FA_VIT_PAD=96`) until that passes,
so the reviewed default stays the validated 128. Residue on the same shelf: a
**fp16-K** variant (removes the ~2e-2 Q8 error and the quantise pass; accuracy win,
not a speed win at these S).

### KVLAYOUT-1 — verify the opt-in LEGACY=0 Q8 side-buffer under 0.29's fused layout

**Status: VERIFIED + adopted for spec-decode serving (2026-09-16); default not yet flipped.** 0.29 standardised the
KV-cache layout (#51718): one tensor with a fused content axis `[B, H, N, 2*D]`
per layer. The default path is ported and validated (FA suite + PPL + smoke),
and the opt-in `GFX906_FA_LEGACY=0` Q8 side-buffer *should* still be correct —
its K view is the `split(D, -1)` half, whose last dim is stride-1 and whose
`bytes_per_row <= row_bytes` guard (136 B into a 512 B K segment for D=256) still
holds, so the uint8 byte-alias writes stay inside K's own segment and never
touch V. That reasoning is static; the path has not been run on 0.29. Verify with
one serving A/B before enabling LEGACY=0 for anything.
**Resolved 2026-09-16.** `GFX906_FA_LEGACY=0` gives **PPL 10.5472** — bit-identical to the
current build's LEGACY=1 value (0 top-20 misses) — so the static reasoning above is confirmed
numerically. Interleaved serving A/B (L1 -> L0 -> L1, MTP k=3, 64k/120k, ms/step): **L0
71.4/76.8 @64k and 103.7/103.8 @120k vs L1 87.8/87.5 and 128.1/128.2** = **-15.5 % / -19.1 %**
with acceptance unchanged (1.7634/1.7634/2.0476); L1's own repeat ran 6-7 % faster and L0 still
beat it, so the win exceeds the per-process drift. Regime split: B=1 greedy pays ~6 % for this
path (2026-08-29), spec-decode serving saves 15-19 %. Hence **adopt LEGACY=0 for serving, keep
LEGACY=1 as the rollback**, and flip the default (code + the four LEGACY=1-assuming test guards,
then the FA suite under LEGACY=0). See `DEVLOG-fa-legacy0-b1-decode.md`.

**Unit-suite check (2026-09-15): the suite is not the instrument for this configuration.** Run with
`GFX906_FA_LEGACY=0 GFX906_FA_LEGACY_ALLOW_UNVERIFIED=1` it gives **4 failed / 93 passed**, and all
four are the fork's own preconditions/diagnostics rather than numerics: two bare `assert False`
guards in tests written for the LEGACY=1 path, `test_a3_draft_step_reuse_reads_live_seq_lens`
("test must exercise the LEGACY=1 fp16 gather path"), and `test_gather_multi_retire_warns`, which
asserts on the warning set and trips over the fail-closed warning this override emits. The gates are
therefore **numerics — the PPL probe, expected 10.5516 (the LEGACY=1 value)** — and the serving A/B.
**ENQUEUED 2026-09-15 (tonight, Kevin):** run that A/B — LEGACY=1 vs LEGACY=0, same boot,
interleaved, agentic corpus, ms/step lead. The outcome either enables the Q8 side-buffer or
retires the opt-in with evidence; note it currently *fails closed* (refuses without
`GFX906_FA_LEGACY_ALLOW_UNVERIFIED=1`), so only the LEGACY=0 arm needs that override.

### KVLAYOUT-2 — migrate the three skipped fork capture/lifecycle tests to the fused layout

**Status: open, small.** 0.29's fused KV-cache content axis (#51718) made the
fork's capture/lifecycle tests hand-build the pre-0.29 `[N, 2, B, Hkv, D]`
fill/reference conventions. Eight of them were migrated (fused cache helper +
`_kv_split`/`_write_v_fused`); three remain **skipped** with that reason:
`test_q_pad_buffer_survives_capture_then_prefill_grow`,
`test_gather_buffers_lifecycle_postfix`,
`test_forward_mixed_batch_pad_tile_clamp_and_host_cu`. Work: re-derive each
test's fill + reference construction against the fused layout (the K/V views are
now strided halves of one tensor, so the staging/`view()` idioms also need
`reshape`). The engine paths they cover are already validated end-to-end on 0.29
(PPL 10.5516 == 0.28, serving smoke, parity restamp), so this is coverage debt,
not an unknown.

**Status: CLOSED (2026-09-14, `gfx906/v2-bringup`).** The three tests were already
written against the fused-layout helpers; the skip decorators were stale. All
three pass unchanged, and the suite is **91 passed, 0 skipped**
(`tests/kernels/attention/test_gfx906_fa.py`). The M3 test was additionally
extended to assert that a full-length (V2-style) host `cu_seqlens` slice with a
garbage tail is bit-identical, which is the guard the V2-bringup plan asked for.

### V2-CAT1-1 — the V2 CAT-1 acceptance boost was a harness bug (RETRACTED; the real effect is ms/step)
**Status: CLOSED (2026-09-14, `gfx906/v2-bringup`) — the acceptance effect was a
measurement artifact.** The A/B client put the arm *name* in the prompt header
(`RESEARCH-BRIEFING-{arm}-…`); because the header's token count differs per arm
(26 vs 29) the body slice `pp - len(header)` shifted too, so each arm ran a
*different prompt* — the exact trap `AGENTS.md` forbids. The three-arm reading
(matched 2.44/2.49 vs control 1.72/1.71 vs none 2.05/1.93) is therefore void, as
is 42.60 t/s. Re-measured with the fixed client (arm out of the prompt; a
`prompt_sha1` digest is now logged so prompt identity can be asserted), same boot,
V2, 3 reps: plain 34.41/24.00 vs CAT-1 35.44/24.61 t/s, ms/step 86.3→83.7 @64k
and 128.5→124.4 @120k ⇒ **+3.0 % / +2.5 %**, **acceptance unchanged** (1.98 vs
1.98 @64k, 2.07 vs 2.10 @120k). The mechanism is the cheaper per-step head read
(35,251 rows vs 248,320), exactly as the V1 controlled A/B found (−2.52 ms/step,
no acceptance penalty). Exactness still holds by audit (`gumbel_sample` caches the
masked draft logits; `rejection_sampler_utils.py` computes the ratio from that same
cache; the target keeps its own head) and the draft path at temp 0 is a plain
argmax. Quote CAT-1's gain as **+3.0 % / +2.5 % on V2** (and ~+5 % on V1), never
as an acceptance effect.

(Original entry, kept for the record.) On V2 the CAT-1 shortlist arm's acceptance is ~20 %
higher than V2's plain MTP k=3 arm (2.44/2.49 vs 2.05/1.93 @64k, 2.37 vs 2.13/2.00
@120k; the server's own `Mean acceptance length` agrees: 3.3–3.5 vs 3.0), while
under V1 the same list showed **no** acceptance effect (controlled A/B, z = −0.83).
The t/s gain on V2 (+26.7 % @64k over V2's own plain arm) therefore mixes two
effects, and the *headline* number should carry that caveat until this is settled.

Hypothesis to test: V2's draft sampling is seeded/stochastic, so restricting the
draft head to a corpus-matched prior shapes the draft *distribution* (raising
agreement with the target), whereas V1's greedy drafts are unchanged. Exactness
should be unaffected (rejection sampling corrects any draft distribution; the
target keeps its own full `lm_head`).

Work: (1) run the **mismatched control list** of the same size
(`/local/tmp/mtp1/cat1_32768ctrl`) under the identical V2 config — if it shows the
same boost, the list's content is irrelevant and something structural is
happening (investigate the V2 speculator's draft path, `logits_cache`/
`gumbel_sample`); (2) a correctness cross-check with the shortlist active under
V2 — in-process PPL with the MTP spec config, or a greedy token-identity A/B
against the no-shortlist arm (remember: token identity alone is *not* a gate on
this stack, so lead with the numbers, not the diff); (3) only then decide whether
the V2 CAT-1 config becomes the recommended one (it is currently the fastest
configuration measured on this box).

### GEMMA4-1 — CLOSED: Gemma-4 is fine, was never gated, and now is

**Status: CLOSED (2026-09-15).** Three readings settled by evidence, kept as the worked
example of the trap:

1. *"The PPL probe can't gate it"* — the original entry, and correct.
2. *"It's a 0.29 regression"* (my raw-text generation probe on 0.29) — **wrong**: the
   **0.28 image** on the same checkpoint+probe reproduces the garbage byte-identically.
3. *"It's broken on both lines"* — **also wrong**: the checkpoint is
   **instruction-tuned** and does not continue raw text. Through its chat template it
   answers `'Paris//'` at first-token logprob **0.00** and writes a correct Python
   snippet at ≈0.00 (`/local/tmp/b4/gemma_diag.log`).

**V2 gate: passed** — templated in-process comparison, V1 vs V2: identical text and
logprobs agreeing to ≤0.05 (`/local/tmp/b4/gemma_v1v2.log`). So Gemma-4's V1 pin is
lifted on this evidence; it was never a model or kernel defect, and the earlier
"degenerate PPL" (84261 V1 / 108909 V2) was a prompt-format artifact.

**Why it survived so long:** the model's table row *did* say "chat template required
(thinking model); PPL/prompt_logprobs unreliable on this model", but it was an aside in a
notes column with nothing enforcing it, and the recorded 67.79 t/s is a **speed** number —
so the model looked validated while never having been gated. Both harnesses now enforce
the prompt form (`ppl_probe.py` warns and takes `BENCH_CHAT_TEMPLATE=1`;
`_bench_gfx906.py` records `prompt_form`/`has_chat_template` and warns that tokens/s is
not a correctness gate), and `README.md` carries a prominent prompt-format block with the
per-model split.

**Follow-ups:** a templated reference PPL for Gemma-4 and Muse-Glimmer (the run that
exercises `BENCH_CHAT_TEMPLATE=1` end-to-end is in flight), and the same templated gate
for Muse-Glimmer (MUSE-1).

### MUSE-1 —### MUSE-1 — Muse-Glimmer: V2 parity looks good, but the PPL probe is not its gate

**Status: open (2026-09-15) — checkpoint obtained, gate run, verdict: V1/V2 parity holds
at the token that matters; the RBLOCK workaround is obsolete.** The 24 GB AWQ
checkpoint came down to `/data/cache` (see the pull note below). Findings from the
first session (0.29 line, stock triton 3.8.0, in-process, single GPU):

- **V1/V2 parity on generation is exact**: greedy completions are **byte-identical**
  across the two runners on both probe prompts (the raw-text quicksort continuation
  and the capital-of-France one), i.e. the same class of evidence as the triton
  greedy-identity control. The quicksort continuation is sensible; the capital
  prompt loops ("… is Paris. The capital of France is Paris. …"), which is a
  raw-text-on-instruct-model artifact, identical on both runners.
- **The in-process PPL probe cannot gate this model**: it renders the prompts
  through the chat template and loads it as a **VLM** (`MuseGlimmerForConditionalGeneration`
  with a `vision_config`; the log shows the encoder cache being profiled with image
  items), and it reports **362 of 363 top-20 misses** on *every* arm (PPL 36.12 V1 /
  36.19 V2 / 36.19 V2-with-rblock-default) while the same model generates sanely
  outside the probe. So those numbers are a prompt/template artifact, not a model
  defect — unlike Gemma-4, where raw-text *generation* is garbage.
- **The `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` question is unresolved**: that arm
  (V2) was killed by a GPU wedge (`hipErrorLaunchFailure`, wedge #93) 2 s before its
  failure was logged, and the arm that ran after it (rblock default) succeeded — no
  controlled comparison yet. The workaround existed because the rblock *variant*
  compile crashed in the triton v3.6.0 fork.
- Being a VLM, its vision tower also exercises **VIT-1** on this line — worth
  checking which ViT backend it selects (the new loud fallback warning makes a
  fall-through visible).

**Gate result (clean, post-reset arms; the earlier NaNs were post-wedge contamination —
see below).** Three arms with `RBLOCK` **unset**: V1, V2, V1 (repeat). The first token is
identical and confident in every arm (`328` at logprob **0.00** — Glimmer's own
`to=self` recipient/reasoning marker, which its system prompt defines via "Valid
recipients: self, user"), and V1 reproduces itself exactly at ranks 2–5. V2's ranks 2+
differ by ≤0.7 logprob at −18…−25 (numerically irrelevant near-zero probabilities).
**So: V1/V2 parity holds at the top token, with tail differences inside the
per-process variation the dense model also shows.** Two consequences:

- **`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` is no longer needed** — the variant-compile crash
  (`AttributeError: 'NoneType' object has no attribute '__code__'`) was a **triton v3.6.0
  fork** defect; every clean arm above ran with it unset on stock Triton 3.8.0. Drop it from
  Muse-Glimmer recipes.
- The in-process gate **cannot judge Glimmer's answer quality**: its output is its own
  recipient/reasoning format (`' to=self…'` then the task restated), so the "should have
  said Paris" heuristic does not apply and the *text* is not stable across processes (V1
  said "We", a second V1 run said "Answer" after identical logprobs). Its **flip gate is a
  serving A/B** (chat endpoint, tokens + ms/step, interleaved arms) — which is also the
  deadline item, since upstream removes V1 in 0.32.0.

**A caveat recorded for the record:** the *first* gate session's arms (V2 with
`RBLOCK=0` and V1) both returned **`nan` logprobs**; they ran immediately after wedge #94
and the clean re-run on the reset GPU produced rc=0 with 0 NaNs — i.e. NaN output is a
**post-wedge symptom**, not a model or kernel defect. Do not read a post-reset session's
numbers as evidence.

**Next**: gate it the way its siblings were gated where the probe applies — a
**serving A/B** (V1 vs V2, same boot, identical prompts, MTP k=3 since MUSE-1's own
point is that Muse-Glimmer should use MTP rather than ngram) with the rblock arms
interleaved (A,B,A) to survive the wedge lottery, and check the ViT backend line.


**Status: open; the AWQ checkpoint is downloading (2026-09-15).** Only the GGUF was
local. The AWQ-INT4 checkpoint (24 GB) is now in `/data/cache/huggingface/hub` — see the findings above.

### TRITON-1### TRITON-1 — move to stock Triton with native gfx906 support (perf secondary)

**Status: DONE — ADOPTED (2026-09-15).** Stock upstream **Triton 3.8.0** is the
default; the ai-infos fork (v3.6.0 + a 7-line gfx906 patch) is retained only as a
rollback (`pip install -e /local/git/triton-gfx906`). Upstream carries gfx906
natively since `aa53dba7455` "[AMD] Add GCN5.1 / gfx906 target (#9628)" —
`ISAFamily::GCN5_1`, wave64, v_dot, DPP, no MFMA, deliberately not CDNA/RDNA — so
the fork had nothing left to carry.

Gates passed on stock 3.8.0 (full detail + the retractions in
[`RECON-triton-1.md`](RECON-triton-1.md)): FA suite **97/97**; dense 27B PPL
**10.5472** vs 10.5516 (−0.04 %); MoE 35B (whose layer-0 experts run Triton
`fused_moe`) **57.97** vs 58.36 t/s; Nemotron **26.9937** vs 27.0066 (band
26.96–27.02); Ornith **16.6664** vs 16.7824; all with 0 top-20 misses; the
`GFX906_FA_VIT=0` flash-attn/Triton-AMD ViT fallback compiles and runs; serving
ms/step parity (85.4/127.9 vs 85.8/128.2 @64k/120k).

Artifact: build from the **upstream v3.8.0 tag** (no patches) with the recipe in
`README.md`; the **PyPI 3.8.0 wheel segfaults on import here**, and AMD's
ROCm-index wheel (`3.7.1+git0263a6a6.rocm7.14.0`, upstream's own `rock.txt` pin)
downloads but is untested — so adoption carries a small build step rather than a
stock download.

Follow-ups this created:
- **one clean in-tree extension rebuild** with 3.8.0 installed (all gates so far
  ran against the existing vLLM build) — the release build recipe must still work;
- **a triton-adopting image build** if the docker images should move off the fork
  (the published `0.29.0-e730ef4066` image still ships `v3.6.0+gfx906`), which
  needs a new tag shape rather than a retag of the published one;
- **re-test the two fork-specific workarounds** the new Triton may obsolete:
  Muse-Glimmer's `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` (the rblock *variant*
  compile crashed in the fork — part of the MUSE-1 investigation) and Nemotron's
  mamba2 `ssd_chunk_scan` restructure (a triton 3.6.x `CanonicalizePointers`
  assertion; the shipped kernel fix is harmless either way and now upstream-report
  is moot).
- Measurement hygiene this item produced: acceptance/t/s cannot carry a
  build comparison (see §"interleaved" in the recon and the AGENTS.md rule).

**Original recon (2026-09-15) — the answer was better than expected:
stock Triton supports gfx906 since v3.8.0, so this is an *upgrade*, not a port.**
See [`RECON-triton-1.md`](RECON-triton-1.md). Findings: (**i**) the fork carries
**nothing but a 7-line ISA classification** (its history is upstream source drops
with one gfx906 commit after each), so nothing else has to be carried forward;
(**ii**) upstream landed gfx906 natively in
`aa53dba7455` "[AMD] Add GCN5.1 / gfx906 target (#9628)" — `ISAFamily::GCN5_1`,
wave64, DPP broadcast, `supportsVDot`, no MFMA, *not* classified RDNA/CDNA — which
is contained in **v3.8.0** and not in v3.7.1; (**iii**) parity with the fork is
complete except `supportsDirectToLdsLoadBitWidth`, where v3.8.0 has no `GCN5_1`
case (gfx906 → false) while the fork allowed 32-bit direct-to-LDS — a one-line
upstreamable follow-up to #9628, gated on a measurement showing that path is taken
and pays.
**Work now**: build stock v3.8.0 (recipe in the recon: gcc, *no*
`TRITON_BUILD_WITH_CLANG_LLD`, plus
`TRITON_APPEND_CMAKE_ARGS=-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON` for the Ninja RPATH
error the 3.8.0 prebuilt LLVM triggers), install the wheel over the editable fork
(rollback: `pip install -e /local/git/triton-gfx906`), then screen: FA suite (97)
→ in-process PPL (dense 27B, expect **10.5516 / 359 tokens**) → serving A/B (MTP
k=3, agentic) → ViT-fallback smoke (`GFX906_FA_VIT=0`). **Version caution**: vLLM
0.29 nominates `triton==3.7.1+git0263a6a6`, so 3.8.0 is newer than the pin — watch
API drift on `triton_kernels`, `vllm.triton_utils` and `triton_prefill_attention`.
The triton-touching code we depend on is the **GDN decode kernel**, the mamba ops,
`triton_mla` and the flash-attn Triton-AMD fallback. A 3.7.1 port was written
before the upstream commit was found (`RECON-triton-1.md` §4b) and is **superseded**
— kept as a reference implementation of the design upstream chose.

### MUSE-1 — Muse-Glimmer spec decode: MTP instead of ngram

**Status: open, medium.** Kevin 2026-09-13: Muse-Glimmer should use **MTP**, not
ngram (ngram is deprecated for now across our models). The Muse rows in the
model table were measured with ngram n=5 on a **repetitive filler** corpus where
acceptance saturates (100 %, acceptance-length 6.0), so they are a ceiling and
say nothing about real prompts. Work: (1) confirm the checkpoint exposes an MTP
draft head (the local cache currently has only the GGUF build — the AWQ-INT4
checkpoint referenced by the table is not in `/local/cache`), (2) same-boot A/B
on a real prompt set: greedy vs MTP k=3 (capture ladder multiples of 4), with
acceptance and ms/step reported, (3) fold the result into the model table and
drop the ngram recipe for this model. Reuses the VIT-1/v2 machinery: nothing new
in the kernels.

### SMLA-1 — re-port the fork's fp16 sparse-MLA to 0.29.0's ROCm path (only if needed)

**Status: parked, inert.** The fork carries an fp16 variant of the ROCm AITER
sparse-MLA indexer (`VLLM_ROCM_MLA_SPARSE_FP16`, default **off**; gfx906 fp16
logits path, optional MLA metadata, prev-extent tracking) spread over
`v1/attention/backends/mla/rocm_aiter_mla_sparse.py`,
`model_executor/models/deepseek_v2.py` and the ops file. The 0.29.0 merge kept
upstream's restructured **ops file** (its local `rocm_fp8_mqa_logits` /
`rocm_fp8_paged_mqa_logits` implementations are what upstream's indexer calls;
the fork's variant had deleted them, so mixing the two was incoherent) while
keeping the fork's backend/hook files. Consequence: the fp16 path is
**half-ported** — inert at its default, and not expected to work if enabled.
Re-port only if we ever serve a DeepSeek/GLM sparse-attention model on gfx906
(needs AITER + gfx942/950 for the upstream path; our models are Qwen3.5/3.8,
Muse-Glimmer, Ornith, Gemma-4 and Nemotron — none use sparse MLA). The fork's
variant is preserved in git history (`main`) and in this branch's merge parents.

### FA-STRUCT — remaining FA decode headroom (structural only)

**Status: open, high effort.** Every config-level FA decoder lever is measured
dead (`DEVLOG-fa-verify-sq8.md`), so what is left is structural:
(1) **KV re-read elimination across q-tiles** — ~6–9 % of step, weeks-scale
rewrite (ref llama.cpp/llaminar FA), and it only pays when grid_x>1 (moot at the
Sq=5 verify shape); (2) **online-softmax rescale batching** (defer the max
update across N KV tiles) — ~2.5–4 %, 1–2 weeks; (3) **M6: a Q4-KV format** to
unlock native `v_dot8_i32_i4` (2× dot4 MAC rate at half the operand bytes, no
unpack ALU) — the only instruction-level upside left, **PPL-gated**; the M5
work showed the current Q8 dot is already full-rate and the B=1 path is
gather-HBM-bound, so this is a numerics bet, not a free win
(`DEVLOG-fa-kernel-batches.md`). Housekeeping: (i) `GFX906_FA_DIRECT_PAGED` still lacks the FIX-H2 pad-tile
clamp (documented, not fixed — the path is default-off for the mixed
shapes); add it if direct-paged is ever enabled at long context; (ii) the
FIX-H2 test's length-hardening is unfixed (a length mismatch raises
IndexError, not a clear assertion) — fold into the next touch of that test;
(iii) the `GFX906_FA_KVSPLIT_MAX_BYTES` default (512 MiB) carries a queued
decision (packet F3): at B=1 it admits y=32 (403 MB transient is under the
cap) while B=4 forces y=1 — leave it alone unless a same-boot serving A/B
shows a y=32 loss at B=1; (iv) drop the `GFX906_FA_GATHER_EXACT` kill switch
at the next
gather-lifecycle change
(byte-for-byte pre-fix policy = the OOM-repro value), re-gated on a serving A/B
(drop notes are in the code and in `plan-gfx906-fa-fix.md` §6).

### C7 — persistent/cooperative MoE block

**Status: open, high effort.** A cooperative kernel for routing, gemm1,
activation, and gemm2 could remove much of the launch floor and avoid
the current CAS/zeroing structure. First verify HIP cooperative-launch
support and resident-grid capacity on Vega 20. Follows C1–C3; requires
serving, correctness, and memory gates.

### C4 — quantize layer-0 routed experts (414 µs/step)

**Status: GO 2026-09-01 (measured) and merged to `main` (verified 2026-09-15:
`gfx906/c4-layer0-quant` is an ancestor of `main`).**
The checkpoint leaves layer 0's routed experts in fp16, so the unquantized
Triton path costs ~740 µs per call at M=1 (C4 scoping probe) vs 182 µs for
the gfx906 W4A16 kernel — ~558 µs/step ≈ 4.8% of the ~11.7 ms step.

Implementation: `c4_layer0_moe.py` installs a method on the skipped
RoutedExperts layer that loads fp16 as usual, then in
`process_weights_after_loading` quantizes to int4 (asymmetric AWQ, group size
from the checkpoint config; codepoints chosen against the stored fp16 scale)
and delegates to the shared gfx906 WNA16 repack + kernel — no new kernel
code. The fp16 storage is released via `register_parameter(name, None)`
(~1.5 GiB back for graph capture). Gated by
`VLLM_GFX906_QUANT_LAYER0_MOE=1` (default off until soak).

Gates (all passed):
- Unit: 8/8 (`tests/kernels/moe/test_c4_layer0_quant.py`) — bit-exact packing
  vs an independent reference, round-trip error bounds, cross-check against the
  production gfx906 repack.
- Quality: PPL off 15.9531 → on 15.9929 (Δ +0.04, noise; gate < 0.5); greedy
  serving fingerprint bit-identical across arms (`d2e5262183c6b92f`); coherent
  text all samples.
- Serving A/B (M=1 decode, pp2048/tg256, 3 reps/arm, same boot): off
  84.95 ± 0.08 → on **87.51 ± 0.33 t/s = +3.0%**, above the ~1.8% A/B noise
  floor (C3's wash was 0.3%). Firing confirmed in the ON-arm log.

Pre-merge review (self + Claude CLI, validated): synced the runner-visible
quant-method state from the delegate (`moe_kernel`/`moe_quant_config`/
`experts_cls`) and made `supports_eplb` follow the active path — without the
sync the MoE runner would have treated layer 0 as a non-MK method.

Open: default-on decision after soak (keep opt-in until at least one full-day
serving window); T1 (int8 family, PROBE GO) may supersede part of this work
if it lands on the unquantized mass instead.

## Model onboarding queue (own cadence)

General rule: gfx906 dispatch is selected by weight format and shape,
not by model family — verify every shape before adding a model-specific
gate. Any compatible AWQ W4A16 MoE checkpoint benefits from the existing
expert kernel without new kernel work, but remains an onboarding task,
never an automatic support claim.

Procedure for a new model:

1. Confirm the model loads and identify its attention/linear-attention
   kernels.
2. Run a shape spy and build a per-step kernel table; do not infer
   transfer from Qwen3.5 numbers.
3. Microbenchmark each candidate GEMV/GEMM shape before changing
   dispatch.
4. Use a greedy-hash gate for bit-equal changes, or a PPL/coherence gate
   when accumulation order changes.
5. Run graph and eager serving A/Bs, with the default chosen from the
   A/B.

The portable design notes and hardware constraints are in
`latency-hiding.md`, `lds-layout.md`, `dequant-instructions.md`, and
`README.md`. The active MoE kernel is useful to AWQ W4A16 checkpoints
with compatible group size, layout, and dimensions; BF16, FP8, FP4,
MLA, and DSA paths need separate validation.

### Ling-3.0-tiny (`BailingMoeV3ForCausalLM`)

**Status: open; load and baseline first.** The on-disk checkpoint is an
approximately 7.5B BF16 model that should fit one MI50, but it is not yet
measured on gfx906. It has 24 layers, hidden size 1536, E=128/topk=8 MoE
layers, sigmoid plus `noaux_tc` routing with expert bias, KDA linear
attention, and MLA-style full attention. These properties do not match
the Qwen3.5 GDN, standard GQA, or W4A16 paths.

- **L1 — get it running.** Load the BF16 checkpoint and verify the
  `BailingMoeV3` model, KDA linear attention, and MLA decode path on
  gfx906. Check for CDNA-only intrinsics or aiter assumptions before
  porting anything. Record a greedy probe and PPL baseline.
- **L2 — establish a profile.** Collect shape and kernel profiles for
  the full decode step, including layer-0 dense work, KDA, MLA, routing,
  and shared expert. Produce a measured budget before selecting an
  optimization.
- **L3 — BF16 MoE expert GEMM.** If the model runs and profiling
  justifies it, benchmark a new W16A16 grouped skinny-GEMM family for
  E=128, hidden=1536, and expert intermediate size 512. The candidate
  dimensions are K=1536 and K=512. The existing lane-column,
  wave-per-K-slice, single-wave-epilogue design is a starting point, not
  proof that the Qwen3.5 W4A16 kernel transfers.
- **L4 — routing.** The sigmoid/`noaux_tc`/bias configuration is handled
  by the generic routing path. Consider an M=1 specialization only if
  the measured profile shows a large routing gap; the Qwen3.5 E=256
  top-k result is not a sufficient reason.
- **L5 — attention.** Treat KDA recurrent decode and MLA paged decode as
  separate workstreams. The Qwen3.5 GDN and custom FA implementations
  provide methodology only; they do not establish correctness or
  performance for these kernels.

**Stop rule:** if L1 finds a hard gfx906 blocker in MLA or KDA, park
Ling (→ REFRIGERATOR) and do not build an expert kernel for an
unservable model.

### Nemotron-3.5-Lightning-30B-A3B mixed INT4/INT8 (`NemotronHForCausalLM`)

**Status: NH-1 + NH-3 + NH-4 + NH-5 SHIPPED, merged to `main` (2026-08-30 ff of
`gfx906/nh2-int8-gemv`, code review `nemotron-nh-code-rev.md` — no blocking
findings); NH-2 NO-GO as Triton (measured; opt-in in-kernel int8 code on
`main`, env default off); NH-2′ (CUDA int8 GEMV family) MERGED 2026-08-31 as
opt-in after a NO-GO serving A/B gate (M-mismatch — kernel's M≤4 support
misses the m=6 spec-decode steps).** Serves at **70.4 tok/s** (graph, pp2048/tg256, 4
samples, GPU0; boot-dependent — boot O window 2026-08-30 PM: 106.8 →
**114.6 t/s** after NH-5, A–B–A) after five fixes that landed on `main`:
fp32-router LLMM1 dtype
guard, the ssd_chunk_scan pointer-yield restructure (triton-gfx906
CanonicalizePointers workaround), a new
`CompressedTensorsW8A16ChannelDequant` scheme replacing Conch
(3.79 ms → ~62 µs per dense GEMV), the gfx906 W4A16 kernel with the
group gate widened to any positive multiple of 32 (g64 here) +
`RELU2_NO_MUL` support (+88.8% vs Triton WNA16), and the fp32
router-gate GEMV on hipBLAS sgemv (24 × 128 µs triton fp32 matmul →
8 µs `torch.mv`; 59.4 → 70.4 tok/s).
See `DEVLOG-nemotron-h.md` for gates (PPL 26.96–27.02 band, A/B tables).

Per-step decode budget at 70.4 tok/s (14.2 ms; pre-NH-3 profile at 59.4,
16.8 ms, clean 32-step profile, `/tmp/nemotron_prof3.log`): LLMM1 dense
GEMVs 3.57 ms · ~~fp32 router gates 3.08 ms~~ 0.2 ms after NH-3 (24 ×
8 µs sgemv) · MoE experts 1.77 ms · mamba elementwise/mul ~3 ms ·
shared experts 1.0 ms · topk chain ~1.2 ms · SSU+conv 0.5 ms.

- **NH-2 — int8-channel GEMV kernel (dense INT8 layers): NO-GO as
  Triton (2026-08-30, measured).** Triton int8 GEMV/GEMM at all six
  Nemotron shape families (`bench_w8a16_gfx906.py`, devlog): M=1 total
  1.10× (wins 1.29–1.60× on the K=2688/large-N shapes, loses 0.69–0.72×
  on K=4096/small-N — mid-N is the hand-tuned CUDA's band); M=4
  0.55–0.80×; M=4096 0.19–0.47×. The serving mode (ngram spec M=6/step
  + M=4096 prefill) is exactly the losing zone; an M=1-only hybrid
  needs 3× VRAM. Code + probe + tests land on `main` behind
  `VLLM_GFX906_W8A16_INT8=1` (default off). Real win exists for N ≥ 10K
  M=1 lm_head-class shapes (1.60× measured).
- **NH-2′ — int8 CUDA GEMV family: MERGED as opt-in (2026-08-31; serving
  A/B gate NO-GO).** Byte-load + in-register per-channel dequant kernels
  (`dense_gemv_i8_gfx906` M=1, `dense_gemv_i8_m4_gfx906` M≤4), env-gated
  behind `VLLM_GFX906_W8A16_INT8_CUDA=1` on top of the NH-2 int8 path (both
  default off; dequant path bit-identical when off — verified by review).
  Kernel-level GO: M=4 in_proj [10304,2688] 239 → 72 µs (3.3×), 10/10 unit
  tests vs fp64 + Triton cross-check. Serving A/B gate (TP=2+EP, ngram spec
  n=5, warm median-of-3): armA 119.2 t/s → armB **46.1 t/s (−61%)**, PPL
  24.9260 vs 24.8826 (noise). Root cause = M-mismatch: the real serving M
  distribution is m=1 72% / m=6 28% / m=4 ~1% (eager-mode MLOG, devlog) —
  the micro-bench's headline M=4 operating point essentially never occurs,
  and the 28% m=6 steps fall back to the slow Triton int8 GEMM. Revival path:
  extend the kernel family to M≤6 (or a dedicated M=6 variant) so all
  spec-decode steps hit the CUDA path; then re-gate. See devlog "NH-2′
  serving A/B gate" section.
- **NH-3 — fp32 router-gate GEMV: SHIPPED, merged to `main`.** hipBLAS
  sgemv (`torch.mv`) replaces the 128 µs fp32 triton matmul at M=1
  (8 µs measured at the [128, 2688] gate shape); 59.4 → 70.4 tok/s
  (+18.4%), PPL 26.9757 (noise band). Only fires for fp32 operands,
  which previously crashed LLMM1 on gfx906 — no existing model's route
  changes. M=2..32 fp32 batches still take the triton path (~118 µs);
  a batched fp32 GEMV remains open if spec-decode/batched decode of an
  fp32-router model ever matters.
- **NH-4 — mamba2 grouped gated-norm fused path: SHIPPED (2026-08-30,
  env default OFF).** `Mixer2RMSNormGated.forward_cuda` routes the
  n_groups>1 case through the existing fused Triton `rms_norm_gated`
  kernel when `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM=1` and
  `per_rank_hidden_size % group_size == 0` (≡ `n_groups % tp_size == 0`,
  excludes the redundant all-gather case). Isolated: ~68 → ~55 µs/layer
  (1.2–1.6×, ~0.29 ms/step over 23 mamba layers). Serving A–B–A, TP=2+EP:
  109.8 → **110.05** → 109.37 t/s (+0.4 %, within inter-arm noise — the
  decode step is MoE-GEMV-bound at this batch) and PPL 24.9034 vs
  24.8944 (Δ +0.04 %). Correctness: 11/11 unit (incl. production TP=2
  geometry + TP-driven partial-group refusal), TP=2 regression driver
  6/6 bit-equal, ruff clean. Ship the opt-in; flip the default when a
  non-GEMV-bound config (spec-decode mid-N, small batch) shows the win.
- **NH-5 — topk chain (~1.2 ms/step): SHIPPED (2026-08-30, node removal
  only, per C1's fold-don't-replace rule).** (a) single-group degenerate
  fast path in the torch-compiled `grouped_topk` (n_group=1/topk_group=1
  — Nemotron) removes 2 of 3 `aten::topk` + the group-mask no-op
  machinery (`VLLM_GFX906_TOPK_SINGLE_GROUP`, default ON); (b) C1 fused
  align+count extended to (128, 6) (templated `moe_align_m1_gfx906`).
  3 kernels/layer removed. Serving A–B–A (boot O): 106.8 → **114.6** →
  107.8 t/s = **+7.3–7.8 %** (0.63 ms/step vs 1.09 ms isolated
  prediction; the in-graph topk nodes are cheaper than eager).
  Correctness: fast path bit-equal to the generic chain (19/19 unit
  incl. ties + compiled toggle), (128,6) align bit-equal (51/51), PPL
  27.05 vs 27.00 (Δ = historical inter-arm band). Note: the fully-fused
  `ops.grouped_topk` kernel stays dead on this fork (its gate needs
  `current_platform.is_cuda()`, False here) — enabling it would be a
  topk replacement (C1-negative); the surviving top-6/128 + gate-GEMV
  epilogue fold remain open. See `DEVLOG-nemotron-h.md` (NH-5).
- **NH-6 — MTP head (parked).** The BF16 MTP layer is present in the
  checkpoint; nemotron_h_mtp drafting with mamba-state rewind is
  unvalidated on this fork and MTP was already too heavy for these GPUs
  on Qwen3.8. Revisit only with ngram numbers first
  (`--speculative-config '{"method":"ngram",...}'`).
- **TP=2 untested** for this model (mamba state pool + shared-expert
  overlap under TP not validated); single-card 32 GB fits maxlen 8k
  comfortably, 131k needs the second card.

Serve recipe (validated 2026-08-29): `--dtype float16` (bf16 config
would route shared experts off Exllama and experts off the gfx906
kernel — both are fp16-acts-only), `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`,
util 0.90–0.95, graph mode with cudagraph capture sizes matched to
max_num_seqs; attention runs ROCM_ATTN (the CUSTOM gfx906 FA backend is
rejected for hybrid DECODER attention — investigate separately if the
6 GQA layers ever show in the profile; they do not at B=1).

## Upstream contribution queue (owner time)

Implemented or source-confirmed in the fork but not upstream
vLLM/ROCm merges. Keep until an owner performs the required
duplicate-work check, rebase, tests, and submission.

- **U1 — fastsafetensors GDS fallback.** Catch the non-`RuntimeError`
  GDS failure so unsupported systems fall back instead of killing the
  engine; preserve the successful GDS fast path. Local commit
  `128e948baf`.
- **U2 — hipify in-source build guard.** Avoid `copytree` onto itself
  during Py3.12 in-source builds. Local commit `225448d93f`.
- **U3 — GemmaRMSNorm fused dispatch.** Preserve Gemma's `(1+w)` algebra
  in the input dtype so the fused RMS norm path remains available. Local
  commits `19c1d41cf5` and `70ec1d0e79`.
- **U4 — compressed-tensors asymmetric W4A16 qzeros repack.** Correct
  the Triton backend's K-first qzeros layout handling. See
  `DEVLOG-ornith-wna16.md`.
- **U5 — asymmetric-W4A16 review hardening.** Share the `g_idx` gate,
  make qzeros repacking fail closed, and centralize the stored-zero-point
  backend capability set. Local commit `d160fb2ad0`; see
  `DEVLOG-ornith-wna16.md`.
- **ROCR-1 — `IPCRecvHandle` EOF spin.** Treat `recvmsg()==0` as peer
  EOF rather than retrying forever in ROCR-Runtime.
- **ROCR-2 — EventPool permanent allocation latch.** Retry event
  creation after a transient `hsaKmtCreateEvent` failure rather than
  permanently forcing userspace polling. Separate `/local/git/TheRock`
  changes; see `cpu-stuck-threads.md`.

## Open questions

- What exact call site accounts for the remaining roughly 158 µs/step of
  MoE-adjacent `[1,2048]` copies?
- What is llama.cpp's component-level kernel budget on the same MI50?
- Does `topkGating`'s cost come from structure or a hidden memory round
  trip? (feeds C1)

(The former "MI50 L2 size / expert residency" question is C8, folded
into C2 above.)
