# gfx906 Speculative Decoding Roadmap

> Status: **CLOSED (2026-08-24), build updated 2026-08-19 (q_gemm
> max-ilp split).** Final serving A/B (agentic 3-prompt, 3 repeats,
> `68243a61b2`): **mtp k=2 = 1.503×** (39.74 t/s, CI 38.58; 1.819
> tokens/step, 90.95% draft acceptance, token-identical output) — the
> recommended spec method for this model. ngram3 = **1.094×** (28.92
> t/s) after the L5 cg-small fix; its ceiling is prompt repetition
> (1.12× on repetitive prompts, ~1.1× agentic). L2 (AWQ M≤4) deferred
> (dequant-ALU-bound ceiling 2–8 ms); L3 moot (MTP supersedes the CPU
> ngram proposer); W4 (skinny fp16 M≤16) parked — it lifts both arms
> ~40% and is now the bigger prize (a 40 t/s baseline would make k=2
> MTP ~2.1×).
>
> **Build (2026-08-19, supersedes the VLLM_NO_MAX_ILP note below):**
> the 2026-08-24 open item (revisit per-file max-ilp) is resolved by
> **splitting q_gemm's 4-bit kernel on M** — M=1 compiles with
> `-amdgpu-sched-strategy=max-ilp` (new TU `q_gemm_m1_maxilp.cu`),
> M≥2 unflagged; FA/skinny keep the flag. Measured: baseline
> 26.44→27.99 t/s (+5.9%), mtp2 39.74→39.37 (within CI), ngram3
> 28.92→28.03 (marginal). Kill-switch
> `VLLM_GFX906_QGEMM_M1_MAXILP=0`. See DEVLOG-spec-decode.md
> "max-ilp split" section. Build env: **no `VLLM_NO_MAX_ILP`**
> (flag on by default; the split makes it safe). The GPU-wedge
> suspicion of the flag was NOT confirmed: 5+ clean 27B weight loads
> and a full A/B on a max-ilp build, both wedge incidents traced to
> zombie-VRAM/other causes.
>
> Build note (superseded): branch built `VLLM_NO_MAX_ILP=1` — the
> per-file max-ilp flag under the new ROCm 7.14 LLVM was suspected of
> the weight-load GPU faults (two `hipErrorLaunchFailure` wedge
> incidents, 2026-08-19 and 2026-08-24; both recovered via BACO/reboot).
>
> Prior status (2026-08-18): Phase 1 in progress — L1' DONE (committed
> `751eacb37d`, 13.4 ms/step measured in-engine); serving A/B then
> 0.945× (was 0.71×). L1''/L5/L2 were the remaining levers; see the
> levers table for their final dispositions.
>
> Prior status (2026-08-18): Phase 0 done, Phase 1 re-scoped after
> kernel-path forensics. n-gram probe complete (0.68× on agentic
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
- **MTP head present**: `mtp_num_hidden_layers=1`; `mtp.*` weights =
  `mtp.fc` ([5120, 10240] fp16) + one full decoder layer (fp16 via
  `modules_to_not_convert`), shared embed/lm_head with the target.
  `method: "mtp"` resolves to `Qwen3_5MTP` on the same checkpoint.
  (Earlier "no MTP head" note was wrong — corrected 2026-08-24.)

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

**The actual draft-step excess is in both GEMM families — not the
GDN.** Per-step kernel breakdown (call structure from the exact
eager-mode census `gemm_step_census.py`; costs from kernel
microbenches — profiler totals include warmup/capture runs and
inflate ~3×):

| family | nospec M=1 | draft step M=4 | M=4 excess |
|---|---|---|---|
| AWQ `gptq_gemm` — **same 4 shapes, same ~231–236 calls/step at M=1 and M=4** (`5120×6144`×63, `34816×5120`×63 fused gate+up, `5120×17408`×63, `16384×5120`×47) | ~21 ms | ~38 ms (pure per-call kernel time, 1.4–1.9× per `bench_awq_m_scaling`) | **~17 ms** |
| fp16 projections — **dispatcher is M-gated**: M=1 takes the GEMV op, M=4 falls to `triton_matmul` (N=96 ×~43, N=14336 ×~14, one inductor-fused [248320, 5120] layer-0 mega-GEMM) | ~3.5 ms (GEMV) | ~16.5 ms (triton; itself M-invariant, 174 µs fa_q at M=1..16) | **~13 ms** |
| GDN (packed / fused_seq) | 0.7 ms | 1.2 ms | 0.5 |
| FA q1 / q4 | 0.3 ms | 0.5 ms | 0.2 |
| B3 bookkeeping (copy_/elementwise/copyBuffer) | 0.5 ms | ~2 ms | ~1.5 |
| CPU ngram proposer (off-GPU; grows with context length) | — | ~5 ms | ~5 |
| **measured step wall** | **36.5 ms** | **~95 ms (agentic context; ~70 in the short-context probe)** | **~30–57 ms** |

- The AWQ delta is **pure per-call kernel time** — the census
disproved the "launch/K-split inflation" hypothesis (identical call
structure at M=1/M=4; the earlier 263-vs-234 was profiler
contamination). `q_gemm` at M=4 is 1.4–1.9× M=1 despite being
weight-read-bound: the M=1 per-file max-ilp tuning does not carry.
- The fp16 delta is a **dispatcher gate**, not an inductor cliff:
`rocm_unquantized_gemm_impl` routes n==1 to the GEMV ops and
0 < n ≤ 16 to `triton_matmul`. On MI50 triton already beats
rocBLAS (`bench_fp16_skinny_m.py`: 174 vs 340 µs on fa_q) and is
M-invariant — so the fix is an M≤4 GEMV-family kernel (extend
dense_gemv), not a routing change. The M≤16 extension of the same
kernel is W4 (replaces the triton 174 µs floor for all M — general
decode opt, ratio-neutral).

**Model calibration.** With T_draft = 95 ms, T_nodraft = 64 ms
(piecewise, B2), min1 stats r=0.84, a=0.71:
tokens/step = 1 + r·a = 1.597; step time = 0.84·95 + 0.16·64 =
85.4 ms → **18.7 t/s predicted vs 19.40 measured** (min1 server
bench). The model is calibrated to within ~4%.

### The levers (measured, single-request, per draft step)

| # | lever | savings | scope | status |
|---|---|---|---|---|
| **L1'** | fp16 M≤4: extend `dense_gemv_gfx906` to ≤4 rows (census: M=1 takes the GEMV op, M=4 falls to `triton_matmul`) | **13.4 ms measured in-engine** (draft step 66.6 → 53.2 ms eager) | gfx906-local kernel + dispatch branch | **DONE** `751eacb37d`; M-templated follow-up `68243a61b2` (M=1..3 → 816/730/544 GB/s; templated M=4 regressed 507→311 unattributed, so M=4 keeps the original runtime-M kernel) |
| **L1''** | extend the m4 dispatch to **n>16** fp16 | — | dispatch-only | **CLOSED as non-issue** (2026-08-24): the "inductor intercept" theory was wrong — an eager dispatch spy showed every decode fp16 GEMM (incl. the LM head) reaches the dispatcher and already routes to m4 when the shape qualifies (k%1024==0 covers K=1024 fc/gate shapes too); the residual ~7.3 ms/step inductor rows are piecewise-compiled fragments, not missed GEMM dispatches |
| **L2** | AWQ M≤4: 4-row GEMV (exllama structure) or re-tiled `q_gemm` | **2–8 ms** (re-scoped 2026-08-18: the family is dequant-ALU-bound — M=1 q_gemm runs 44 MB in 75 µs = 590 GB/s, ~2× off the HBM roofline — and the tiled M=4 kernel already shares the dequant across m-rows; GEMV structure removes only atomics/LDS/m-tiling overhead, not the 17 ms M=1→M=4 delta) | gfx906-local kernel (`csrc/libtorch_stable/quantization/gptq/`) | **DEFERRED** — low ROI vs MTP (the drafter is fp16, not AWQ) |
| L3 | CPU ngram proposer D2H serialization | ~5 ms | in-tree (`ngram_proposer.py`); the GPU proposer exists but has a **draft-quality bug** (0.428 vs 1.08 acc/step — match selection/tie-break divergence, §2 item 1) | **MOOT** — MTP supersedes ngram (1.50× vs 1.077×) |
| **L5** | no-draft spec step penalty: **measured ≈ 13 ms** (graph-mode no-draft ≈ 48 ms vs nospec 35–37, back-solved from the A/B counters; eager shows only +3–5, so ~10 ms is graph-mode — suspected cudagraph padding of spec steps to the draft capacity M=4) | **~13 ms on 63% of agentic steps** — the dominant remaining loss | core vLLM (`vllm/config/compilation.py`) | **DONE** `68243a61b2` — confirmed at kernel level (graph-mode profile: zero M=1 AWQ calls; batchdesc_probe.py: no-draft steps pad to num_tokens=4). Root cause: `adjust_cudagraph_sizes_for_spec_decode` rounds capture sizes up to multiples of `uniform_decode_query_len`, dropping sizes 1..q-1 from the PIECEWISE key set, so 1-token no-draft steps replay the size-q graph. Fix re-adds them (gfx906-gated, `VLLM_GFX906_SPEC_CG_SMALL=0` off). A/B: ngram3 0.945× → 1.077× |
| **W4** | the same `dense_gemv` extension shipped at M≤16 (replaces the GEMV floor AND the 174 µs triton floor) — **all M, spec and no-spec** | ~15 ms per decode step (~40% baseline latency) | gfx906-local kernel; does NOT change the spec ratio — parked after Phase 1 | parked |

GDN needs **no new kernel** for single-request spec decode.

### Ceiling, re-anchored on measured values (min2 mix: r = 0.37,
a = 1.09 — the config the serving A/B actually runs)

Anchor points: pre-L1' draft step 95 ms (profiler, contaminated
era) / post-L1' **56 ms** (graph profile 54.7 + launch margin);
no-draft **48 ms** (back-solved, L1' build); nospec 37 ms.

| scenario | T_d | T_nd | t/s (decode) | all-in vs baseline 26.5 |
|---|---|---|---|---|
| **measured (L1')** | 56 | 48 | 28.6 | **25.06 = 0.945×** ✓ model fits |
| +L1'' | 51 | 48 | 31.2 | ~1.03× |
| +L1''+L2 (5 ms) | 46 | 48 | 33.7 | ~1.09× |
| +L5 (T_nd → 35) | 46 | 35 | **35.9** | **~1.16×** |

(The old min1 table is superseded: min2's draft share is 37% vs
84%, so the no-draft lever L5 — not the draft kernel — dominates
the agentic mix. At min1 the draft-side levers dominate instead;
the two configs are lever-dual.)

**Final measured (2026-08-24, `68243a61b2`, post-reboot run)**: the
ngram stack (L1' + L5) landed at **1.094×** (28.92 t/s vs 26.44
baseline; pre-reboot run 28.55 = 1.077×, within noise) — between
the "measured (L1')" and "+L5" rows above, i.e. L5's predicted ~13 ms
no-draft saving materialized but the draft-step cost stayed ~56 ms
(the L1'' row turned out to be a non-issue). The 1.15× gate failed
for ngram by construction: at min2's mix the draft step is
AWQ-dominated (~29 ms of the 56), and L2's ceiling is 2–8 ms. **MTP
bypassed the whole ngram cost model** — r=1.0 (every step drafts),
no no-draft steps, and its drafter is fp16 (L1' family already
fast): **1.503×** (39.74 t/s) with no L2 dependency.

W4 (skinny fp16 GEMM, M≤16) — the "19 ms/step at every M" figure in
its work-item entry predates the M-gated-dispatcher finding: at M=1
the fp16 projections already take the GEMV ops (~3.5 ms/step, near
roofline), so W4 does **not** lift the single-request baseline. Its
real value is **multi-request decode**: 5–16 concurrent sequences
fall to `triton_matmul` (M-invariant 174 µs floor, ~5× off roofline),
and the m4 kernel covers only M≤4. That is the highest-leverage
parked item on this branch — it also directly benefits concurrent
MTP serving (drafter batch = 3×seqs, which lands in the M=5..16
triton gap at 2–5 seqs).

**Readout (updated 2026-08-18 post-L1').** At min2's measured
quality (a = 1.09, r = 0.37), the stack **L1''+L2+L5** lands at
~1.16× all-in (35.9 t/s decode); L1'' is dispatch-only, L2 is
2–8 ms, L5 needs its padding hypothesis verified first (core-vLLM
risk). **Order of attack: L1'' → verify L5 padding → L5 → L2.**
L3 (proposer) stays available if the ceiling still falls short.
**Draft quality `a` remains the swing variable** (a = 1.09 → 1.3
would add ~1.3 t/s per 0.1 of a at the min2 mix). **Stop rule
(kept from review, RE-ANCHORED 2026-08-18):** the draft-step cost
is now 56 ms measured (≤ 60 rule met) — the rule is consumed; the
gate going forward is the **all-in serving A/B ≥ 1.15×** after
L1''+L2+L5. If that fails, stop and record the ceiling.

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

## 3. Phase 1 — skinny-M (M≤4) GEMV-family kernels (L1' + L2)

**Status (2026-08-18):** L1' implemented, tested (25 passed / 2
skipped), microbench-verified (25.6 → 11.0–11.2 ms/step fp16
worst-case M=4) and **confirmed in-engine: draft step 66.6 →
53.2 ms eager; serving A/B 0.71× → 0.945×** (dev log). L1''
(m4 dispatch to n>16) and the L5 no-draft finding re-ordered the
remainder — see the §1 levers table. L2 below is re-scoped.

The GEMM census (`gemm_step_census.py`, eager single-prompt, exact
per-step counts) gives the final shape of the work:

- **AWQ (L2)**: the M=4 draft step makes the *same* GEMM calls as
  M=1 (231–236/step, four shapes: `5120×6144`, `34816×5120`
  (fused gate+up), `5120×17408`, `16384×5120`) — the delta is pure
  per-call kernel time, **~17 ms/step** over the exact shape set.
  *(Re-scoped 2026-08-18: that 17 ms is the M=1→M=4 cost, but the
  family is dequant-ALU-bound and the M=4 tiled kernel already
  shares the dequant — a GEMV-structure M≤4 kernel saves only the
  atomics/LDS/m-tiling overhead: 2–8 ms. See levers table.)*
- **fp16 (L1')**: the dispatcher is M-gated — M=1 takes the GEMV op
  path (~3.5 ms/step), M=4 falls to `triton_matmul` (~16.5 ms/step:
  N=96 ×~43, N=14336 ×~14, one inductor-fused [248320, 5120] layer-0
  mega-GEMM). Delta **~13 ms/step**. (The earlier "fp16 is
  M-invariant / L1 dead" finding was about triton_matmul's own
  M-scaling, not the engine's M=1 baseline.)

**Implementation — one kernel family, two weight formats:**
- **L1'**: extend `dense_gemv_gfx906` (W16A16, gfx906-owned,
  M=1-by-construction, `template<int RPT, int KCHUNK>`) to ≤4
  output rows — same weight-read-once structure, 4 dots per weight
  element, still weight-bound. Dispatch for 1 < M ≤ 4 in
  `rocm_unquantized_gemm_impl` (gfx906-local branch already
  exists). The same kernel extended to M≤16 is W4 (replaces the
  current GEMV floor AND the triton 174 µs floor — general decode
  opt, ratio-neutral).
- **L2**: a 4-row GEMV for packed-4bit weights (exllama GEMV
  structure: weight-row-parallel, x shared across the block, 4
  output rows per weight row — same weight traffic as M=1), or
  re-tile `q_gemm`'s M=1 tuning to M=4. Bench both at the census
  shapes, pick.
- Dispatch gate: uniform M≤4 (spec small-M steps); M=1 keeps
  today's path untouched (regression gate: nospec 27.46 t/s +
  greedy probe token identity). Env kill-switch per project
  convention.
- **Numerics**: per-GEMM max-abs-diff vs the M=1 path on synthetic
  inputs; PPL A/B (S3-class: fp argmax flips allowed, PPL/
  coherence the bar); greedy probe.

**Exit criteria:**
- Draft-step wall measured (spec_step_probe): **≤ 60 ms** (stop
  rule). Expect ~64 probe-context / ~60–70 agentic — the rule is
  tight; the ~30 ms GEMM delta is well-evidenced, the residual
  context-dependent part (FA KV growth, proposer on long context)
  is what makes 60 borderline.
- nospec regression: 27.46 t/s band + greedy probe token identity
  (M=1 paths untouched).
- Unit tests for both new kernel paths; A/B bench (agentic 3-prompt,
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
| ngram (CPU) | yes | **1.094× final** (agentic; L5 cg-small fix) | prompt-repetition-bound; 1.12× on repetitive prompts; ceiling ~1.1× agentic — see the L1''/L5 closure above |
| ngram_gpu | yes | **rejected: draft-quality bug** (0.428 vs 1.08 acc/step) | fixable (match selection), small task |
| suffix | yes (dep pending) | draft-quality probe only | PIECEWISE-only (dynamic draft length); §2 item 3 |
| **MTP** | **yes — head in checkpoint** (1 layer, fp16, shared embed/lm_head) | **1.503× final** (k=2: 39.74 t/s, 1.819 tok/step, 90.95% acceptance, token-identical) | needs `Gfx906FAMetadata` in the proposer allowlist (gfx906-gated, `68243a61b2`); drafter is fp16-only → 3.4 GB HBM read/forward (~25% of the step); fc K=10240 GEMV dispatch + m4 M-templating applied; k=3 can't break even (needs ≥2.17 tok/step) |
| EAGLE / draft model | no weights on disk | n/a | would ride the same rails |

**Work items outside the single-request rail:**

- **W1 (multi-request GDN chunk reclass).** In mixed batches the
  no-draft sequences run the chunk kernel as 1-token "prefills"
  (~415 µs/layer vs ~20 µs packed, ~20 ms/step per such sequence).
  Small dispatch fix in `qwen_gdn_linear_attn.py` (route 1-token
  non-spec rows to the packed/sequential path instead of
  reclassifying). Matters only for concurrent serving (production
  config 4–8 seqs); measure with a 2-request mixed probe.
- **W2 (MoE 35B) — DONE (2026-08-23, `DEVLOG-moe-spec-decode.md`,
  branch `gfx906/moe-spec-decode`).** The rails ported with **zero
  code changes** (all in-tree from the merged 27B phase;
  `Qwen3_5MoeMTP` registered, fc K=4096 in the GEMV KCHUNK set).
  mtp2 k=2: **graph 89.9 vs 76.2 t/s steady = 1.18×** (80.4 %
  acceptance, 1.609 tok/step; break-even 1.39) and **eager 45.5 vs
  24.5 = 1.86×** (launch-bound baseline amplifies the win). The 35B
  MTP head is a weaker proposer than the 27B's (80 % vs 91 %);
  k=3 needs ~2.4 tok/step — not viable. Recommendation: mtp2 +
  cg-small for 35B serving. 35B caveats: temp=0 baseline is
  non-reproducible (token-identity gates unusable — fp16-atomic
  MoE epilogue suspected); the 35B baseline is also
  non-determinism-sensitive, so spec A/Bs there stand on perf +
  acceptance only.
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
  `gemm_step_census.py` (per-step GEMM call census by shape —
  `enforce_eager`, single prompt), `bench_awq_m_scaling.py` (q_gemm
  M-scaling microbench), `bench_fp16_skinny_m.py` (triton_matmul vs
  rocBLAS at M=1..16).
- Numbers: DEVLOG-moe-opt.md "n-gram spec decode probe";
  **DEVLOG-spec-decode.md** (this branch's log: Phase 0 results, B1
  reconciliation, kernel-path spy, revised cost model).
- Review: `spec-decode-roadmap-plan-rev_claude.md` (merged two-pass
  adversarial review).
