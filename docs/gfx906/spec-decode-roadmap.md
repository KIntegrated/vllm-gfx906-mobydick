# gfx906 Speculative Decoding Roadmap

> Status: **Phase 0 pending.** n-gram probe complete 2026-08-18
> (0.68× — see below); the two prerequisite FA fixes are in-tree on this
> branch. Baseline refs: dense 27B-AWQ **27.46 t/s** (server, 4 seqs,
> graph), MoE 35B-A3B-AWQ **66.56 t/s** (same session).

## 1. Where we are

n-gram spec decode (k=3, min2/max5) on dense Qwen3.5-27B-AWQ, 3
multi-turn agentic-coding prompts (tool-call JSON + code context),
512 tok, greedy:

| arm | t/s | notes |
|---|---|---|
| baseline | **27.46** | server path (LLM-harness record 25.25) |
| ngram k=3, pre-fix | 22.99 | artifact: engine demoted to all-PIECEWISE |
| ngram k=3, post-fix | **18.73** | 0.68×; drafts on ~40% of steps, ~36% accepted |

**Fixed in-tree** (commit `4e40e3e`): `Gfx906FABackend` now declares
`UNIFORM_BATCH` CG support when spec tokens are configured, and
`forward_paged` has capture-safe uniform fast paths (the per-seq
`int(cu_seqlens_q[...])` D2H syncs are illegal under capture). Spec
steps run on FULL graphs. **Without these, every spec method on this
backend runs all decode steps piecewise (~100 ms/step).**

**Model facts that shape the plan** (Qwen3.5-27B text_config):

- 64 layers = **48 GDN linear_attention + 16 full_attention**
  (`full_attention_interval=4`). GDN dominates the spec-step overhead.
- **No MTP head** (no `num_nextn_predict_layers`) — `method="mtp"`,
  EAGLE, draft models are unavailable for this checkpoint. The
  no-draft-model family (ngram / ngram_gpu / suffix) is all we can
  serve on the 27B today; the 35B MoE is the same family, check
  before assuming.

**The three remaining cost blocks** (torch.profiler A/B + controlled
probes; full numbers in DEVLOG-moe-opt.md "n-gram spec decode probe"):

| block | cost | affected steps | shared by |
|---|---|---|---|
| B1: GDN spec path uses **chunk (prefill) kernels** — `ChunkGatedDeltaRule*` 1536 calls, ~800 µs/layer, vs ~20 µs/layer for the packed-decode M=1 fast path | ~20–30 ms/step (48 layers) | every draft step | **all** spec methods |
| B2: **no-draft 1-token steps** don't qualify for the uniform-4q FULL graph → PIECEWISE | ~53–64 ms vs 36.5 ms | the ~60% of ngram/suffix steps without a draft | ngram, suffix (MTP/EAGLE always draft → immune) |
| B3: spec bookkeeping (~6× `copy_`, ~300 ms `index_*`) + CPU ngram proposer D2H serialization | ~8–13 ms/step | every draft step | all methods (B3b is ngram-CPU only) |

Ceiling math at current draft stats (r=0.40, a=1.1): fixing B1 →
~55 ms draft steps; fixing B2 → 36.5 ms no-draft steps; net
**~32.8 t/s ≈ 1.19×**. Modest but real — and the same two levers
scale the MoE 35B (15 ms baseline step → spec is even more
overhead-sensitive there; B1 is mandatory before any method works).

## 2. Phase 0 — cheap data, no kernel work (½ day)

Config-only A/Bs on the existing 3-prompt bench
(`benchmarks/kernels/gfx906/spec_ngram_dense.py`):

1. `method="ngram_gpu"` — GPU vectorized proposer, removes the CPU
   D2H serialization (B3b). Pure config change.
2. `method="suffix"` — in-tree `SuffixDecodingProposer` (suffix tree
   over prompt + cached responses); likely better drafts on code.
   Knobs: `suffix_decoding_max_tree_depth` (24), `..._max_spec_factor`.
3. ngram `prompt_lookup_min=1` — raises the draft rate r (fewer B2
   steps); watch acceptance drop.
4. k sweep {1,2,3,4} on the winner of 1–3.

**Gate:** any combo ≥ 27.5 t/s → ship the config, stop. (Expected:
none pass — B1/B2 are structural — but it's an afternoon, not a
project.)

## 3. Phase 1 — GDN small-M spec kernel (B1, the shared lever)

The GDN state recurrence `h_t = A_t·h_{t-1} + B_t·x_t` is inherently
sequential in t, so "M>1 decode" is **not** a parallelism problem:
the local fused GDN decode kernel (M=1, ~20 µs/layer) extended with an
inner token loop is the right shape, and it should stay within a
factor of M of the M=1 cost.

- **Kernel**: extend the gfx906 fused GDN decode kernel to
  `1 < M ≤ 8` (uniform per-seq token counts; spec steps are uniform
  by construction). Keep the local micro-opts' invariants: non-spec
  keeps the zero-fill skip; the spec path keeps the fill (upstream
  invariant — pad rows feed causal-masked compute).
- **Dispatch** (Python, `qwen_gdn_linear_attn.py`): `on_gfx906()` and
  uniform `1 < num_tokens ≤ 8` → small-M kernel; else upstream chunk
  path. Env kill-switch `VLLM_GFX906_GDN_SPEC_SMALL_M` (default on),
  matching project convention.
- **Numerics gate**: the M=4 result must be **bit-equal to four
  sequential M=1 calls** (same per-token op order — this is the
  reference math the packed-decode kernel already implements). Unit
  test that directly; end-to-end via greedy probe + PPL A/B.
- **Expected**: ~20–30 ms/step back on every draft step, all models,
  all spec methods. Largest single lever; do this before spending
  anything on B2.

## 4. Phase 2 — q1 FULL graphs for no-draft steps (B2)

1-token spec steps today dispatch to piecewise because
`_is_uniform_decode` requires `max_num_scheduled_tokens == 1 + k`.

- **Preferred**: capture **both** uniform graph families when spec is
  configured — q1-uniform [1..max_num_seqs] and q(1+k)-uniform
  [1+k .. max_num_seqs(1+k)] — and treat a uniform 1-token spec step
  as a q1-uniform decode (extend `_is_uniform_decode` + the capture
  size derivation; the dispatcher's `BatchDescriptor` keying already
  separates uniform vs not). Core vLLM change, inert when no spec
  config is set.
- **Alternative** (heavier, park): schedule PAD tokens to pad
  no-draft steps to 1+k (scheduler/`num_scheduled_tokens` accounting
  surgery, line ~2327 of `gpu_model_runner.py`).
- **Budget**: +~0.3–0.5 GiB graph memory for the q1 set; verify
  against the 4-seq dense budget (KV 7.4 GiB at 0.95) and the MoE
  budget before capture.
- **Expected**: no-draft 53–64 ms → ~37 ms; on ngram/suffix this
  touches the *majority* of steps.

## 5. Phase 3 — tune + adoption gate

- Re-run the agentic bench per arm: method × k × min_n (ngram) /
  tree_depth (suffix); add one **low-repetition control prompt**
  (e.g. novel-identifier listing) to measure the floor, not just the
  ceiling.
- **Adoption gate**: ≥1.15× vs 27.46 on the agentic set, PPL-neutral
  (spec-vs-nospec is S3-class fp drift by construction — gate on PPL
  + coherence, not token identity), zero regression on the
  no-spec baselines (27.46 / 66.56).
- On pass: recommended `--speculative-config` in
  `docs/gfx906/running.md`. On fail: record the measured ceiling and
  park — the FA fixes stay (any future drafter needs them).

## 6. Other spec-decode approaches — applicability on this machine

The cost blocks are **method-agnostic rails**; the rails fix once and
every drafter rides them:

| method | available on 27B? | expected | notes |
|---|---|---|---|
| ngram (CPU) | yes | 0.68× today → ~1.19× post P1+P2 | prompt-repetition-bound; r≈0.4 on agentic code |
| ngram_gpu | yes (test) | removes B3b | same drafts as ngram, async H2D + GPU unfold/argmax |
| suffix | yes (test) | ≥ ngram drafts expected | suffix tree, in-tree proposer; same rails |
| MTP | **no head in checkpoint** | 1.1–1.3× *if a head existed* | draft = one extra layer fwd (~1–2 ms) — the user's original intuition (cheap draft on a weight-bound GPU) applies maximally; r=1.0 so B2 is irrelevant, only B1 matters. Re-evaluate if an MTP-capable Qwen3.5 checkpoint lands |
| EAGLE | no head on disk | n/a | would need a trained head; same rails |
| draft model | none on disk | n/a | a 1B draft reads ~2 GB/step ≈ 2 ms — only worth it at high acceptance; same rails |

MoE 35B: baseline step ~15 ms means B1 (48→30 GDN layers) is a
*larger* fraction of the spec step there; Phase 1 is a hard
prerequisite for any spec method on the MoE, and the absolute payoff
is bigger (+~10–15 t/s at 1.2×).

## 7. Artifacts

- FA fixes: `vllm/gfx906_fa/gfx906_fa_backend.py` (UNIFORM_BATCH when
  spec configured), `vllm/gfx906_fa/gfx906_fa_paged.py`
  (capture-safe uniform scatter/gather). FA suite 15/15.
- Bench/probe scripts (moved from /tmp): `benchmarks/kernels/gfx906/
  spec_ngram_dense.py` (serving A/B + acceptance),
  `spec_step_probe.py` (step-cost probes), `spec_prof_probe.py`
  (torch.profiler A/B for B1/B3 attribution).
- Numbers: DEVLOG-moe-opt.md "n-gram spec decode probe".
