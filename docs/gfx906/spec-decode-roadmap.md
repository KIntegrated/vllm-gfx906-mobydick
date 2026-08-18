# gfx906 Speculative Decoding Roadmap

> Status: **Phase 0 done, Phase 1 re-scoped after kernel-path
> forensics (2026-08-18).** n-gram probe complete (0.68× on agentic
> prompts; **1.12× on highly repetitive prompts** — see §1); the two
> prerequisite FA fixes are in-tree on this branch. Baseline refs:
> dense 27B-AWQ **27.46 t/s** (server, 4 seqs, graph), MoE
> 35B-A3B-AWQ **66.56 t/s** (same session). Two adversarial review
> passes were merged into this doc on 2026-08-18
> (`spec-decode-roadmap-plan-rev_claude.md`); the afternoon's
> kernel-path spy then **overturned the B1 attribution** — the
> revised cost model and levers are in §1.

## 1. Where we are

n-gram spec decode (k=3, min2/max5) on dense Qwen3.5-27B-AWQ, 3
multi-turn agentic-coding prompts (tool-call JSON + code context),
512 tok, greedy:

| arm | t/s | notes |
|---|---|---|
| baseline | **27.46** | server path (LLM-harness record 25.25) |
| ngram k=3, pre-fix | 22.99 | artifact: engine demoted to all-PIECEWISE |
| ngram k=3, post-fix | **18.73** | 0.68×; drafts on ~40% of steps, ~36% accepted |
| ngram min1 k3 (Phase 0) | 19.40 | 0.71×; r=0.84, a=0.71 — higher r, lower a |
| ngram_gpu k3 min2 (Phase 0) | 18.29 | **0.428 acc/step vs CPU 1.08 — rejected on draft quality** |
| repetitive-prompt probe, nospec | 29.0 | LLM harness, 128 tok, 39-tok prompt |
| repetitive-prompt probe, ngram | **32.5** | **spec already wins here (1.12×)** |

**The verdict is prompt-dependent.** n-gram drafts only exist where
the prompt (or cached context) repeats. On the most repetitive
workload spec decode **already beats the baseline** with zero kernel
work; on agentic coding prompts it loses. The Phase 1/2 levers below
target the agentic (low-repetition) case, which is where the
production gap is.

**Fixed in-tree** (commit `4e40e3e`): `Gfx906FABackend` now declares
`UNIFORM_BATCH` CG support when spec tokens are configured, and
`forward_paged` has capture-safe uniform fast paths (the per-seq
`int(cu_seqlens_q[...])` D2H syncs are illegal under capture). Spec
steps run on FULL graphs. **Without these, every spec method on this
backend runs all decode steps piecewise (~100 ms/step).**

**Model facts that shape the plan** (Qwen3.5-27B text_config):

- 64 layers = **48 GDN linear_attention + 16 full_attention**
  (`full_attention_interval=4`).
- **No MTP head** (no `num_nextn_predict_layers`) — `method="mtp"`,
  EAGLE, draft models are unavailable for this checkpoint.

### The cost model (revised 2026-08-18 afternoon — supersedes the
B1/B2/B3 table that preceded it)

**B1 as previously characterized does not apply to single-request
serving.** The GDN kernel-path spy
(`benchmarks/kernels/gfx906/gdn_step_spy.py`, single prompt,
`enforce_eager` so every step is visible — **CUDA-graph replay skips
Python, so routing spies only see eager or capture-time steps**)
shows, for a single-sequence spec run of 128 tokens:

```
TOTALS: chunk: 48 (the 39-token prefill, one per GDN layer)
        packed: 336 (7 no-draft steps × 48)
        fused_seq: 1968 (36 draft steps × 48)
draft step GDN: fused_sigmoid_gating_delta_rule_update
                q=(1,4,16,128) ssm_state_indices=(1,4) num_accepted_tokens set
no-draft step:  fused_recurrent_packed_decode (M=1 fast path)
```

The spec-path GDN kernel — the sequential FLA kernel **with 2D
per-token state-slot indices and `num_accepted_tokens`**, i.e. exactly
the Phase-1 kernel this roadmap originally planned to build —
**already exists and already runs** on every draft step, at
~32 µs/layer (~1.2 ms/step total). The chunk kernel appears only in
prefill and, in **multi-request batches**, for the *no-draft*
sequences: when a batch mixes spec and non-spec sequences, the
metadata builder reclassifies the non-spec decodes as 1-token
"prefills" (`qwen_gdn_linear_attn.py`:
`if num_decodes > 0 and num_spec_decodes > 0: num_prefills +=
num_decodes`), and 1-token prefills cost ~415 µs/layer vs ~20 µs
packed — a **multi-request serving pathology** (~20 ms/step of waste
per no-draft sequence in a mixed batch), not a single-request one.
See §6 (work item W3).

**The actual draft-step excess is in the AWQ GEMM family — not the
GDN, and not the fp16 family.** Per-step kernel breakdown (structure
from the in-process single-prompt profiler pair
`/tmp/spec_prof3_{nospec,spec}.log`; absolute costs from kernel
microbenches — the profiler totals include warmup/capture runs and
inflate ~3×):

| family | nospec M=1 | draft step M=4 | M=4 excess |
|---|---|---|---|
| AWQ `gptq_gemm` (q_gemm kernel), ~240 GEMMs/step | ~18 ms (kernel 7.3 + launch/split ~11) | ~45 ms in-engine (kernel 12.5 per `bench_awq_m_scaling`; +~20 ms unexplained launch/K-split inflation, 263 vs 234 calls/step) | **~26 ms** |
| fp16 projections (`triton_matmul`, ~90 GEMMs/step) | ~19 ms | **~19 ms (M-invariant: 174 µs fa_q at M=1..16 — weight-bound)** | **0** |
| GDN (packed / fused_seq) | 0.7 ms | 1.2 ms | 0.5 |
| FA q1 / q4 | 0.3 ms | 0.5 ms | 0.2 |
| B3 bookkeeping (copy_/elementwise/copyBuffer) | 0.5 ms | ~2 ms | ~1.5 |
| CPU ngram proposer (off-GPU) | — | ~5 ms | ~5 |
| **measured step wall** | **36.5 ms** | **~95 ms** | **~57 ms** |

- The AWQ M-scaling: `bench_awq_m_scaling.py` shows `q_gemm` at M=4
  costs 1.4–1.9× M=1 (75→141 µs on the 17408×5120 shape) even
  though the kernel is weight-read-bound — the M=1 per-file max-ilp
  tuning does not carry to M=4; the in-engine gap on top of the
  kernel gap (the ~20 ms) is launch/K-split inflation and is what
  Phase 1 (L2) must kill.
- The fp16 family is **not** an M=4 problem: `triton_matmul` is
  weight-bound and M-invariant, and on MI50 it already beats
  rocBLAS (`fp16_skinny_m.py`: 174 vs 340 µs on fa_q 3072×5120;
  838 vs 1751 µs on layer0_down 17408×5120). Its ~19 ms/step is a
  **general decode-latency** issue (custom weight-row-parallel
  skinny fp16 GEMM could cut it to ~4–6 ms) → work item W4, parked
  behind Phase 1 because it moves the no-spec baseline underneath
  the spec A/B.

**Model calibration.** With T_draft = 95 ms, T_nodraft = 64 ms
(piecewise, B2), min1 stats r=0.84, a=0.71:
tokens/step = 1 + r·a = 1.597; step time = 0.84·95 + 0.16·64 =
85.4 ms → **18.7 t/s predicted vs 19.40 measured** (min1 server
bench). The model is calibrated to within ~4%.

### The levers (measured, single-request, per draft step)

| # | lever | savings | scope |
|---|---|---|---|
| **L2** | AWQ `q_gemm` at M≤4: skinny-M variant (re-tile the M=1 tuning to M=4, or a GEMV-family 4-row kernel reading weights once for 4 rows) | **~10–26 ms** (kernel ~5, launch/split inflation ~20 — kill both) | gfx906-local kernel (`csrc/rocm/quantization/gptq/`) |
| L3 | CPU ngram proposer D2H serialization | ~5 ms | in-tree (`ngram_proposer.py`); the GPU proposer exists but has a **draft-quality bug** (0.428 vs 1.08 acc/step — match selection/tie-break divergence, §2 item 1) |
| L4 | q1 FULL graphs for no-draft spec steps (old B2) | 25–27 ms on no-draft steps (16% of steps at min1; 40% at min2) | core vLLM (§4) |
| L5 | B3 bookkeeping | ~2 ms (small — the 10 ms figure came from a 2-seq run) | partly overlaps L2 kernel work |
| **W4** | custom skinny fp16 GEMM (weight-row-parallel, GEMV structure) for the ~19 ms/step `triton_matmul` family — **all M, spec and no-spec** | ~15 ms per decode step (~40% baseline latency) | gfx906-local kernel; does NOT change the spec ratio — parked after Phase 1 |

GDN needs **no new kernel** for single-request spec decode.

### Revised ceiling (agentic prompts; T_n = 37 ms with L4)

| scenario | T_d | r, a (min1) | t/s | vs 1.15× gate |
|---|---|---|---|---|
| today | 95 | .84, .71 | 19.4 (measured) | 0.71× |
| +L2 (T_d ≈ 58) | 58 | .84, .71 | 26.5 | 0.97× |
| +L2+L4 | 58 | .84, .71 | 29.2 | 1.06× |
| +L2+L4, a = 1.0 | 58 | .84, 1.0 | 33.7 | **1.23×** |
| +L2+L4+L3 (T_d ≈ 53) | 53 | .84, .71 | 31.8 | 1.16× |

Same L2+L4 stack at min2 (r=0.40, a=1.08, T = 0.4·58+0.6·37 =
45.4): 31.5 t/s = **1.15×**.

W4 (skinny fp16 GEMM) lifts **both arms** ~40% (no-spec 36.5 → ~22
ms ≈ 45 t/s; spec T_d 58 → ~43, T_n 37 → ~23) without changing the
ratio.

**Readout.** The stack **L2+L4(+L3)** lands at 1.06–1.16× at
today's measured draft quality (a = 0.71) and **1.15–1.23× at
a = 1.0–1.1** (min2 already reaches 1.15× at a = 1.08). **Draft
quality `a` is the swing variable**: 0.71 → 1.0 flips 1.06× →
1.23×. Note a = 1.0 is only 0.16 above the CPU ngram min2 average
measured on this prompt set (1.08) — the gate pass is plausible
without any drafter improvement, via min2 + L2 + L4. **Stop rule
(kept from review, re-anchored):** if the measured post-Phase-1
draft-step cost exceeds 60 ms, stop after Phase 1 and record the
ceiling. (Expected ~58 — inside the rule, barely.)

## 2. Phase 0 — config-only A/Bs (DONE 2026-08-18)

Run with `benchmarks/kernels/gfx906/spec_ngram_dense.py` (repeats=3,
mean+CI gate per the review fix):

1. **`ngram_gpu` — REJECTED on draft quality.** 18.29 t/s (CI 17.9–18.6)
   vs CPU ngram 18.73; the GPU proposer accepts **0.428
   tokens/draft-step vs the CPU proposer's 1.08**, and output text
   SHAs diverge from the CPU arm. It is a reimplementation
   (unfold/argmax match selection) whose tie-breaking on repeated
   n-grams selects different (worse) continuation tokens. Removing
   the CPU serialization (L3) via this path is not worth a 2.5×
   drop in acceptance. The match-selection divergence is itself a
   small, targeted bug-fix candidate if a GPU proposer is wanted
   (compare its argmax/first-occurrence logic against
   `ngram_proposer.py` line by line).
2. **ngram `prompt_lookup_min=1`, k=3 — best spec arm: 19.40 t/s**
   (CI 18.9–19.8). r=0.84, a=0.71: higher draft rate, lower
   acceptance, net still 0.71× of baseline. Confirms the §1
   a-vs-r trade; the ceiling math above uses these stats.
3. **`suffix` — deferred.** Requires `arctic-inference==0.1.1`
   (sdist, built locally, install + ROCm verification pending);
   dynamic per-request draft length → PIECEWISE-only (neither L4 nor
   the uniform rails apply); its only remaining value is a
   draft-quality measurement, which no longer changes the plan
   (L2/L4 dominate). Revisit only if L2+L4 lands at <1.15× and a
   better drafter is needed.
4. **k sweep — skipped** (k=1/2 are dominated by the same cost
   structure at worse tokens/step; k=4 changes T_d +~8 ms and
   T_nodraft weighting — revisit if Phase 1 leaves headroom).

**Gate result:** no config-only arm beats the baseline
(27.46, band 26.81–27.96) on agentic prompts — as predicted, the
gaps are structural (L2/L4). Phase 1 proceeds on the revised
scope.

## 3. Phase 1 — AWQ `q_gemm` at M≤4 (L2, the shared lever)

The draft step's M=4 excess is in the AWQ family (~26 ms in-engine:
kernel ~5 + launch/K-split inflation ~20). Everything else
measured is small (GDN 0.5, FA 0.2, B3 1.5, proposer 5) or
M-invariant (fp16 → W4). The work is gfx906-local.

**Two candidate shapes (bench both, pick):**
- (a) **Re-tile/tune the existing q_gemm kernel for M≤4**: extend
  the M=1 per-file max-ilp tuning matrix to M=4; check the K-split
  launch counts in-engine (263 vs 234 calls/step — the M=4 path
  changes splitting; that inflation may be most of the gap).
- (b) **GEMV-family 4-row kernel**: the exllama GEMV structure
  (weight-row-parallel, x shared across the block) extended to
  output 4 rows per weight row — same weight traffic as M=1, ~4×
  the ALU, still weight-bound. Larger kernel effort, bigger ceiling.
- Dispatch gate: uniform M≤4 (spec small-M steps); M=1 keeps
  today's path untouched (regression gate: nospec 27.46 t/s +
  greedy probe token identity). Env kill-switch per project
  convention.
- **Numerics**: per-GEMM max-abs-diff vs the M=1 path on synthetic
  inputs; PPL A/B (S3-class: fp argmax flips allowed, PPL/
  coherence the bar); greedy probe.

**Exit criteria (Phase 1 = L2):**
- Draft-step wall measured (spec_step_probe + profiler): **≤ 60 ms**
  (stop rule). Expect ~58.
- nospec regression: 27.46 t/s band + greedy probe token identity
  (M=1 paths untouched).
- Unit tests for the new kernel path; A/B bench (agentic 3-prompt,
  min1 and min2) with the §1 ceiling re-derived from measured r/a.

## 4. Phase 2 — q1 FULL graphs for no-draft steps (L4)

1-token spec steps today dispatch to piecewise because
`_is_uniform_decode` requires `max_num_scheduled_tokens == 1 + k`.
(Scoping notes from the review are unchanged and stand:
`uniform_decode_query_len` is a scalar that becomes a set; the
spec-aware q1 graph is a *new* capture shape, not the no-spec q1
graph; spike first; PAD-token alternative remains the
one-file fallback.)

- **Expected**: no-draft 53–64 ms → ~37 ms. Worth 16% of steps at
  min1, 40% at min2.
- Budget: +~0.3–0.5 GiB graph memory; verify against the 4-seq dense
  budget.

## 5. Phase 3 — L3 (proposer) + tune + adoption gate

- L3: either fix the GPU ngram proposer's match selection (then
  `ngram_gpu` removes ~5 ms) or accept the CPU proposer. Decide from
  Phase 1's residual.
- Re-run the agentic bench per arm (method × k × min_n); add one
  **low-repetition control prompt** to measure the floor.
- **Adoption gate**: ≥1.15× vs 27.46 on the agentic set, PPL-neutral
  (spec-vs-nospec is S3-class fp drift by construction), no-spec
  baselines within the observed noise band (mean+CI, not point
  parity), greedy-probe token identity for the M=1 path.
- On pass: recommended `--speculative-config` in
  `docs/gfx906/running.md`. On fail: record the measured ceiling and
  park — the FA fixes stay (any future drafter needs them).

## 6. Other spec-decode approaches + work items

| method | available on 27B? | expected | notes |
|---|---|---|---|
| ngram (CPU) | yes | 0.71× today (agentic) → ~1.15–1.23× post L2+L4 (a-dependent) | prompt-repetition-bound; already 1.12× on repetitive prompts |
| ngram_gpu | yes | **rejected: draft-quality bug** (0.428 vs 1.08 acc/step) | fixable (match selection), small task |
| suffix | yes (dep pending) | draft-quality probe only | PIECEWISE-only (dynamic draft length); §2 item 3 |
| MTP | **no head in checkpoint** | r=1.0 (no L4 dependency) | re-evaluate if an MTP-capable checkpoint lands |
| EAGLE / draft model | no weights on disk | n/a | would ride the same rails |

**Work items outside the single-request rail:**

- **W1 (multi-request GDN chunk reclass).** In mixed batches the
  no-draft sequences run the chunk kernel as 1-token "prefills"
  (~415 µs/layer vs ~20 µs packed, ~20 ms/step per such sequence).
  Small dispatch fix in `qwen_gdn_linear_attn.py` (route 1-token
  non-spec rows to the packed/sequential path instead of
  reclassifying). Matters only for concurrent serving (production
  config 4–8 seqs); measure with a 2-request mixed probe.
- **W2 (MoE 35B).** 30 GDN + 10 FA layers, ~15 ms baseline step. The
  same rails apply (the 35B's MoE experts are the W4A16 path —
  different kernels, but the fp16/GDN and AWQ-dense families are
  shared). Phase 1 lands first on the 27B; port after.
- **W3 (GDN small-M kernel — CLOSED, no work).** The sequential
  kernel with per-token state slots + `num_accepted_tokens` already
  exists and is on the spec path. The original Phase-1 scope is
  satisfied by upstream FLA; nothing to build.
- **W4 (skinny fp16 GEMM — general decode opt).** `triton_matmul`
  costs ~19 ms/step at every M (weight-bound, M-invariant) and is
  ~5× off HBM roofline on the big shapes (fa_q 31.5 MB → 31 µs
  roofline vs 174 µs measured; rocBLAS is *slower*, 340 µs). A
  weight-row-parallel skinny fp16 GEMM (GEMV structure, M≤16) would
  cut it to ~4–6 ms: ~40% off every decode step, spec and no-spec
  alike. Parked behind Phase 1: it moves the no-spec baseline
  (36.5 → ~22 ms) underneath the spec A/B, so do it after the spec
  work is gated (or use it to lift the parked ceiling).

## 7. Artifacts

- FA fixes: `vllm/gfx906_fa/gfx906_fa_backend.py` (UNIFORM_BATCH
  when spec configured), `vllm/gfx906_fa/gfx906_fa_paged.py`
  (capture-safe uniform scatter/gather). FA suite 15/15.
- Bench/probe scripts: `benchmarks/kernels/gfx906/spec_ngram_dense.py`
  (serving A/B + acceptance + `--repeats` CI summary),
  `spec_step_probe.py` (step-cost probes), `spec_prof_probe.py`
  (torch.profiler A/B — in-process only), `gdn_step_spy.py` (GDN
  kernel-routing spy — use `enforce_eager`, single prompt),
  `bench_awq_m_scaling.py` (q_gemm M-scaling microbench),
  `bench_fp16_skinny_m.py` (triton_matmul vs rocBLAS at M=1..16).
- Numbers: DEVLOG-moe-opt.md "n-gram spec decode probe";
  **DEVLOG-spec-decode.md** (this branch's log: Phase 0 results, B1
  reconciliation, kernel-path spy, revised cost model).
- Review: `spec-decode-roadmap-plan-rev_claude.md` (merged two-pass
  adversarial review).
