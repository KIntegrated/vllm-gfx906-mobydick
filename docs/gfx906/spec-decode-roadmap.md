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

**The actual draft-step excess is in the GEMM families.** Per-step
kernel breakdown, in-process single-prompt profiler pair
(`/tmp/spec_prof3_{nospec,spec}.log`; spec run = 36 draft + 7
no-draft + 1 prefill):

| family | nospec M=1 (per step) | draft step M=4 | excess |
|---|---|---|---|
| AWQ `gptq_gemm` (q_gemm kernel) | ~19 ms (234 calls, 80 µs avg) | ~45 ms (263 calls, 153 µs avg) | **~26 ms** |
| fp16 projections (GDN in/out etc.) | ~5 ms (inductor fusions valid at M=1) | **~35 ms raw `triton_matmul` (77 calls × 377 µs)** | **~30 ms** |
| GDN (packed / fused_seq) | 0.7 ms | 1.2 ms | 0.5 |
| FA q1 / q4 | 0.3 ms | 0.5 ms | 0.2 |
| B3 bookkeeping (copy_/elementwise/copyBuffer) | 0.5 ms | ~2 ms | ~1.5 |
| CPU ngram proposer (off-GPU) | — | ~5 ms | ~5 |
| **measured step wall** | **36.5 ms** | **~95 ms** | **~57 ms** |

- The AWQ excess is **pure M-scaling of the same GEMMs**: a kernel
  microbench (`benchmarks/kernels/gfx906/bench_awq_m_scaling.py`)
  shows `q_gemm` at M=4 costs 1.4–1.9× M=1 (75→141 µs on the
  17408×5120 shape) even though the kernel is weight-read-bound —
  the M=1 launch config/ILP tuning (per-file max-ilp work) does not
  carry to M=4.
- The fp16 excess is an **inductor shape-guard cliff**: the M=1
  compiled fusions (norm+gemm+act pointwise around the projections)
  don't apply at M=4, falling back to raw `triton_matmul` at a
  377 µs/call config that is ~8× off the rocBLAS `Cijk` cost for the
  same shapes.

**Model calibration.** With T_draft = 95 ms, T_nodraft = 64 ms
(piecewise, B2), min1 stats r=0.84, a=0.71:
tokens/step = 1 + r·a = 1.597; step time = 0.84·95 + 0.16·64 =
85.4 ms → **18.7 t/s predicted vs 19.40 measured** (min1 server
bench). The model is calibrated to within ~4%.

### The levers (measured, single-request, per draft step)

| # | lever | savings | scope |
|---|---|---|---|
| **L1** | fp16 projections at M≤4: route off the raw inductor `triton_matmul` to a tuned skinny path (rocBLAS `Cijk`/`rocm_unquantized_gemm` first — measure; custom skinny GEMM if not) | **~25–30 ms** | gfx906-local dispatch + possibly inductor config |
| **L2** | AWQ `q_gemm` at M≤4: skinny-M variant (M=4-specialized tile/ILP, or a GEMV-family 4-row kernel reading weights once for 4 rows) | **~20–25 ms** | gfx906-local kernel (`csrc/rocm/quantization/gptq/`) |
| L3 | CPU ngram proposer D2H serialization | ~5 ms | in-tree (`ngram_proposer.py`); the GPU proposer exists but has a **draft-quality bug** (0.428 vs 1.08 acc/step — match selection/tie-break divergence, §2 item 1) |
| L4 | q1 FULL graphs for no-draft spec steps (old B2) | 25–27 ms on no-draft steps (16% of steps at min1; 40% at min2) | core vLLM (§4) |
| L5 | B3 bookkeeping | ~2 ms (small — the 10 ms figure came from a 2-seq run) | partly overlaps L1/L2 kernel work |

GDN needs **no new kernel** for single-request spec decode.

### Revised ceiling (agentic prompts; T_n = 37 ms with L4)

| scenario | T_d | r, a (min1) | t/s | vs 1.15× gate |
|---|---|---|---|---|
| today | 95 | .84, .71 | 19.4 (measured) | 0.71× |
| +L1+L2 (T_d ≈ 55) | 55 | .84, .71 | 28.2 | 1.03× |
| +L1+L2+L4 | 55 | .84, .71 | 30.8 | 1.12× |
| +L1+L2+L4+L3 (T_d ≈ 50) | 50 | .84, .71 | 31.9 | 1.16× |
| +L1+L2+L4, a = 1.0 | 55 | .84, 1.0 | 35.5 | **1.29×** |

Same stack at min2 (r=0.40, a=1.08, T = 0.4·55+0.6·37 = 44.2):
32.4 t/s = **1.18×**.

**Readout.** The stack **L1+L2+L4 (+L3)** clears the 1.15× gate at
today's min1 draft quality by a hair (1.16×) and more comfortably if
the drafter gets even slightly better (a=1.0 → 1.29×). **Stop rule
(kept from review, re-anchored):** if the measured post-Phase-1
draft-step cost exceeds 60 ms, stop after Phase 1 and record the
ceiling.

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
   (L1/L2 dominate). Revisit only if L1+L2+L4 lands at <1.15× and a
   better drafter is needed.
4. **k sweep — skipped** (k=1/2 are dominated by the same cost
   structure at worse tokens/step; k=4 changes T_d +~8 ms and
   T_nodraft weighting — revisit if Phase 1 leaves headroom).

**Gate result:** no config-only arm beats the baseline
(27.46, band 26.81–27.96) on agentic prompts — as predicted, the
gaps are structural (L1/L2/L4). Phase 1 proceeds on the revised
scope.

## 3. Phase 1 — skinny-M (M≤4) GEMM paths (L1 + L2, the shared lever)

The draft step pays ~55 ms to run the *same weights* at M=4 that
cost ~24 ms at M=1. Both families are gfx906-local.

**L1 — fp16 (unquantized) projections, ~25–30 ms:**
- Measure first: rocBLAS `Cijk` / `rocm_unquantized_gemm` at M=4 on
  the GDN projection shapes (8192×5120, 2048×5120, 128×5120) —
  the profiler shows Cijk at M=4 already costs only ~6 ms/step
  aggregate where inductor spends 35. If rocBLAS wins, the fix is a
  dispatch/routing change (route these projections to
  `rocm_unquantized_gemm` for M≤8, or make the inductor keep its
  M=1 fusions for M≤4 / autotune a proper M=4 config).
- Fallback: a custom skinny fp16 GEMM for M≤8 (the
  `dense_gemv_gfx906` kernel is M=1 by construction — a 4-row
  variant reusing its weight-read-once structure is the template).
- **Numerics**: routing change = bit-equal expected; new kernel =
  fp32-reference unit test per shape.

**L2 — AWQ `q_gemm` at M≤4, ~20–25 ms:**
- The q_gemm kernel's M=1 per-file max-ilp tuning does not carry to
  M=4 (weight-bound but 1.9× slower at M=4 — bad tile/ILP config,
  not a memory problem). Two shapes: (a) re-tile/tune the existing
  kernel for M≤4 (extend the M=1 tuning matrix to M=4, check the
  K-split launch counts — 263 vs 234 calls/step suggests the M=4
  path also changes splitting); (b) a GEMV-family skinny kernel:
  the exllama GEMV structure (weight-row-parallel, x shared across
  the block) extended to output 4 rows per weight row — same weight
  traffic as M=1, ~4× the ALU, still weight-bound.
- Dispatch gate: M≤4 (uniform small-M spec steps); M=1 keeps today's
  path untouched (regression gate: nospec 27.46 t/s + greedy probe
  token identity).
- **Numerics**: new/re-tiled kernel must pass the PPL A/B gate
  (S3-class: fp argmax flips allowed, PPL/coherence the bar) plus a
  per-GEMM max-abs-diff check vs the M=1 path on synthetic inputs.

**Shared exit criteria (Phase 1 = L1+L2):**
- Draft-step wall measured (spec_step_probe + profiler): **≤ 60 ms**
  (stop rule). Expect ~55–60.
- nospec regression: 27.46 t/s band + greedy probe token identity
  (M=1 paths untouched).
- Unit tests per new kernel path; A/B bench (agentic 3-prompt,
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
| ngram (CPU) | yes | 0.71× today (agentic) → ~1.16–1.29× post L1+L2+L4 | prompt-repetition-bound |
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
  same L1/L2 rails apply (the 35B's MoE experts are the W4A16 path —
  different kernels, but the fp16/GDN and AWQ-dense families are
  shared). Phase 1 lands first on the 27B; port after.
- **W3 (GDN small-M kernel — CLOSED, no work).** The sequential
  kernel with per-token state slots + `num_accepted_tokens` already
  exists and is on the spec path. The original Phase-1 scope is
  satisfied by upstream FLA; nothing to build.

## 7. Artifacts

- FA fixes: `vllm/gfx906_fa/gfx906_fa_backend.py` (UNIFORM_BATCH
  when spec configured), `vllm/gfx906_fa/gfx906_fa_paged.py`
  (capture-safe uniform scatter/gather). FA suite 15/15.
- Bench/probe scripts: `benchmarks/kernels/gfx906/spec_ngram_dense.py`
  (serving A/B + acceptance + `--repeats` CI summary),
  `spec_step_probe.py` (step-cost probes), `spec_prof_probe.py`
  (torch.profiler A/B), `gdn_step_spy.py` (GDN kernel-routing spy —
  use `enforce_eager`, single prompt), `bench_awq_m_scaling.py`
  (q_gemm M-scaling microbench).
- Numbers: DEVLOG-moe-opt.md "n-gram spec decode probe";
  **DEVLOG-spec-decode.md** (this branch's log: Phase 0 results, B1
  reconciliation, kernel-path spy, revised cost model).
- Review: `spec-decode-roadmap-plan-rev_claude.md` (merged two-pass
  adversarial review).
