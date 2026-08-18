# Dev Log — gfx906 Speculative Decoding

> Branch: `gfx906/spec-decode`. Fresh log for the spec-decode work
> (sibling of `DEVLOG-moe-opt.md`, which keeps the n-gram probe
> record). Copyright Kevin Read <me@kevin-read.com>.

## 2026-08-18 — session: post-review implementation start

### Roadmap review absorbed

`spec-decode-roadmap.md` was revised post-review (two independent
passes, `spec-decode-roadmap-plan-rev_claude.md` merging
`spec-decode-roadmap-plan-rev-glm.md`). Incorporated findings that
change execution order:

- Ceiling 1.19× demoted to **upper bound**; sensitivity table +
  **stop rule: if post-P1 measured draft-step cost > 60 ms, stop
  after Phase 1 and record** (62 ms → 1.13×, 70 ms → 1.08×, both
  below the 1.15× gate).
- B1 unit accounting flagged as unreconciled (800 µs/layer vs
  20–30 ms/step vs 38.4 ms naive product) — resolved below, first.
- Phase 1 re-scoped: new kernel (vendored FLA
  `fused_recurrent.py` is M=1 by hardcoded shape/grid), per-token
  state writes into the existing `mamba_cache_mode=align` block-slot
  scheme, numerics gate = fp32 reference within tolerance at every
  token boundary (bit-equal-to-4×M=1 is informational only — fp32
  resident state skips the per-call quantization round-trips).
- Phase 2: scalar `uniform_decode_query_len` architecture limit +
  spec-aware q1 graph is a *new* capture shape, not the no-spec q1
  graph; spike before committing; no-spec q1 dispatch regression
  test is an exit criterion.
- Phase 0: gate restated as mean + CI (flat 27.5 sits inside the
  26.81–27.96 baseline band); `spec_ngram_dense.py` needs a repeat
  knob; ngram_gpu needs a draft-equivalence check (reimplementation,
  tie-breaking may differ); suffix is **not** config-only
  (`arctic-inference==0.1.1` dependency, dynamic per-request draft
  length → PIECEWISE-only → Phases 1/2 as scoped don't apply; pin
  `num_speculative_tokens` explicitly, treat as draft-quality probe).

### B1 reconciliation (from `/tmp/spec_prof_{nospec,spec}.log`)

Exact profiler rows, spec run (128 committed tokens, agentic prompt,
thinking off, k=3):

- Decode steps: packed_decode 3312 calls / 48 GDN layers = **69
  one-token steps**; `ChunkGatedDeltaRuleFunction` 1536 / 48 =
  **32 multi-token steps**; total **101 steps**.
- Consistency: 128 = 69×1 + 32×(1+a) → **a = 0.84 accepted/draft
  step; r = 32/101 = 0.32** for this prompt set (server 3-prompt
  probe: r ≈ 0.40, a ≈ 1.08 — prompt-dependent, as expected).
- `ChunkGatedDeltaRuleFunction`: CUDA **total** 637.694 ms / 1536 =
  **415.2 µs per layer per multi-token step**. Child kernels
  (`h_blockdim64` 312.5 + `chunk_fwd_o` 141.8 + `kkt` 68.8 +
  `recompute_w_u` 82.0 = 605 ms) are *inside* that total — the
  roadmap table's "~800 µs/layer" double-counted wrapper + children.
  **Correct B1 = 415.2 µs × 48 layers = 19.9 ms per draft step**
  (devlog's 18–25 ms band was right; "38.4 ms naive" is the
  double-count).
- +`fused_sigmoid_gating_delta_rule_update` 66.6 ms / 101 steps ≈
  0.66 ms/step (32 draft + 11 one-token calls/layer — small,
  included).
- **B1 (final): ≈ 20.5 ms per draft step**, vs 0.96 ms/step for the
  packed_decode fast path (20.05 µs/layer).

B3 (spec bookkeeping), spec-vs-nospec CUDA deltas:
- `aten::copy_` 158.6 − 24.8 = 133.8 ms → 4.2 ms/draft step
  (133.8/32), assuming all on draft steps (KV×4 writes, state
  staging).
- `index_select` 87.5 + `index_put_` 93.5 = 181.0 ms → 5.7
  ms/draft step (index_put is the align-slot path;
  `precopy_mamba_align_fused_kernel` JITs in-run, inside this).
- **B3 ≈ 10 ms per draft step.**

### Revised cost model and ceiling (replaces the 1.19× case)

Per-draft-step budget, reconciled:
```
measured draft step (server, k=3)  ≈ 82 ms
  = base 4-token compute (GEMM M=4 + FA q4 + proposer sync) ≈ 47
  + B1 GDN chunk                                    20
  + B3 copy/index                                   10
  + slack ~5
```
Post-P1 (chunk → fused M=4, ≈ 48 × 4 × 20 µs ≈ 4 ms; B3 kept, no
credit taken):
**draft step ≈ 82 − 20 + 4 ≈ 66 ms.**

| scenario | T_draft | T_nodraft | r, a | t/s | vs gate |
|---|---|---|---|---|---|
| post-P1 only (no-draft still piecewise 64 ms) | 66 | 64 | .40/1.1 | 22.2 | 0.81× |
| post-P1+P2 (no-draft → 36.5 ms FULL q1) | 66 | 36.5 | .40/1.1 | 29.8 | **1.09×** |
| post-P1+P2 + B3 −5 ms (kernel owns align-slot writes) | 61 | 36.5 | .40/1.1 | 31.3 | 1.14× |
| post-P1+P2+B3 with a = 1.3 (better drafter) | 61 | 36.5 | .40/1.3 | 33.4 | **1.22×** |

**Implications (change the plan's emphasis):**

1. At today's draft quality (a ≈ 1.1), P1+P2 lands at ~1.09–1.14× —
   **at or below the gate**. The swing variable is now **draft
   quality a**, not step cost alone. Phase 0's suffix acceptance
   probe is therefore a go/no-go input, not just data.
2. The stop rule stays (post-P1 draft step > 60 ms → stop).
   Prediction (66 ms) is *just over* it — so Phase 1's exit check
   must measure, and the B3-overlap claim (kernel writes align slots
   directly ⇒ index_put/precopy disappear) is the only in-P1 lever
   to get under 60. It is testable from the Phase-1 profiler run.
3. min_n=1 (Phase 0 item 3) trades a for r; with the reconciled
   costs, higher r at lower a is *neutral-to-negative*
   (r=0.6, a=0.5 → 25 t/s). Run it to measure the a(r) trade, not
   to bank a win.

### Artifacts updated this session

- `docs/gfx906/spec-decode-roadmap.md`: B1 row fixed (415 µs/layer,
  19.9 ms/step; double-count noted), B3 folded into Phase 1 with the
  align-slot reasoning, revised ceiling table + stop rule, Phase 0
  gate = mean + CI with repeat knob, Phase 0 item ordering
  (ngram_gpu → min_n=1 → suffix, suffix acceptance = swing input).
- `benchmarks/kernels/gfx906/spec_ngram_dense.py`: `--repeats` knob,
  mean/sd/95%-CI-lower summary.

## Phase 0 results (2026-08-18, all arms run)

`spec_ngram_dense.py --repeats 3` (repeats knob added this session),
server port 8931, 3 agentic prompts, 512 tok, greedy:

| arm | mean t/s | CI95 | drafts/step | acc/step | vs baseline 27.46 |
|---|---|---|---|---|---|
| ngram_gpu k3 min2/max5 | 18.29 | 17.9–18.6 | 0.40 | **0.428** | 0.67× — **rejected** |
| ngram k3 min1/max5 | 19.40 | 18.9–19.8 | 0.84 | 0.71 | 0.71× — best spec arm |
| suffix | — | — | — | — | deferred: `arctic-inference==0.1.1` is an sdist (built in `/tmp/arctic_check`, install pending); dynamic draft length → PIECEWISE-only; only remaining value is a draft-quality measurement |

**ngram_gpu rejection detail:** same draft rate as CPU ngram (0.40) but
0.428 accepted/draft-step vs the CPU proposer's 1.08, and output text
SHAs diverge. The GPU proposer is a reimplementation (unfold/argmax)
whose tie-breaking on repeated n-grams picks different, worse
tokens. Not a config issue — a match-selection bug. Logged as L3
sub-item in the roadmap (small targeted fix if a GPU proposer is
ever wanted).

**Gate:** no arm beats the baseline band (26.81–27.96) — structural,
as predicted. Phase 1 proceeds.

## Fork in the road: the B1 attribution was wrong for single-request

Kernel-path spy (`/tmp/bench/gdn_step_probe.py`, promoted to
`benchmarks/kernels/gfx906/gdn_step_spy.py`) wraps the four GDN kernel
entry points + the conv op and prints per-step kernel routing. Three
pitfalls found:

1. **CUDA graphs hide steps from Python spies.** Replay executes the
captured kernel list directly — the model forward (and any spy)
runs only at capture time. First spy run (graph mode) saw 7 packed
steps + prefill and *zero* draft steps. **Fix: `enforce_eager=True`**
(same routing as capture, all steps visible).
2. **The conv anchor double-counts on mixed-batch steps.** When a
batch mixes spec and non-spec sequences, the spec branch calls
`causal_conv1d_update` for the spec rows *and* the non-spec path
calls it again → 96 convs on one step → step index `conv//48`
shifts. Only valid for single-sequence runs (one conv per layer per
step).
3. **Exit-134 shutdown hang** on this probe (known vLLM teardown
heap corruption): `os._exit(0)` after the report.

Clean single-prompt spec run (128 tok, eager):

```
TOTALS: chunk: 48 (39-token prefill)   packed: 336 (7 no-draft steps)
        fused_seq: 1968 (36 draft steps)
draft step:  fused_sigmoid_gating_delta_rule_update
             q=(1,4,16,128) ssm_state_indices=(1,4) num_accepted_tokens=set
no-draft step: fused_recurrent_packed_decode B=1
```

**The spec-path GDN kernel already is what old-Phase-1 planned to
build**: the sequential FLA kernel with 2D per-token state-slot
indices and `num_accepted_tokens` — per the code, the align-slot
rollback mechanism is already wired. Cost ~32 µs/layer ≈ 1.2
ms/step. **GDN is not the single-request draft-step bottleneck.**

Where the chunk kernel actually bites: the GDN metadata builder
(`qwen_gdn_linear_attn.py`, `_build_...` metadata path):
`if num_decodes > 0 and num_spec_decodes > 0:` reclassifies the
non-spec 1-token sequences as **prefill** ("the prefill kernel
correctly handles 1-token sequences with initial state") → they run
the chunk kernel at ~415 µs/layer (~20× packed). A **multi-request
mixed-batch pathology** (~20 ms/step of waste per no-draft sequence),
irrelevant to the single-request production bench. → roadmap W1.

## Revised cost model (replaces the B1/B2/B3 budget in §"Revised
cost model")

Per-step kernel breakdown, in-process single-prompt profiler pair
(`/tmp/spec_prof3_{nospec,spec}.log`; spec = 36 draft + 7 no-draft):

| family | nospec M=1 | draft M=4 | excess |
|---|---|---|---|
| AWQ gptq_gemm | 18.9 ms (234 calls, 80 µs) | 44.6 ms (263 calls, 153 µs) | **25.7** |
| fp16 projections | ~5 ms (inductor fusions) | **35 ms raw `triton_matmul` (77×377 µs)** | **~30** |
| GDN | 0.7 | 1.2 | 0.5 |
| FA | 0.3 | 0.5 | 0.2 |
| B3 (copy_/elementwise/copyBuffer) | 0.5 | ~2 | ~1.5 |
| CPU proposer | — | ~5 | ~5 |
| **wall** | **36.5** | **~95** | **~57** |

- `bench_awq_m_scaling.py` (promoted): q_gemm M=1→M=4 = 1.4–1.9×
  (75→141 µs on 17408×5120) at M where the kernel is weight-bound —
  the M=1 per-file max-ilp tuning does not carry to M=4.
- The fp16 excess is an **inductor shape-guard cliff**: the M=1
  compiled fusions don't apply at M=4 → raw triton_matmul at a
  config ~8× off rocBLAS Cijk (which already costs only ~6 ms/step
  aggregate at M=4 in the same run).
- B3 single-seq is ~2 ms, not 10 — the 10 ms came from the 2-seq
  profiler run (state save/restore for two sequences).

**Calibration:** T_d=95, T_n=64 (piecewise), min1 r=.84 a=.71 →
predicted 18.7 t/s vs **measured 19.40** (4% — model trusted).

**Prompt sensitivity (new finding):** on a maximally repetitive
prompt ("repeat the sentence 30 times", 128 tok, LLM harness) spec
already **wins**: 32.5 t/s vs 29.0 nospec (1.12×), zero kernel work.
The agentic loss is entirely the low-repetition case — where L2/L4
apply.

## Levers and revised plan (roadmap restructured accordingly)

- **L2** AWQ q_gemm M≤4 skinny path: re-tile the M=1 tuning to M=4
  or a GEMV-family 4-row kernel. In-engine M=1→M=4 delta ~26 ms
  (19→45 ms/step); the kernel-only microbench delta is ~5 ms
  (7.3→12.5 ms/step over the 27B's AWQ set) — the rest is
  launch/K-split inflation (263 vs 234 calls/step). Target: bring
  M=4 to M=1 cost.
- **L3** CPU proposer (~5 ms) / GPU proposer match-selection bug.
- **L4** q1 FULL graphs for no-draft steps (old Phase 2, unchanged
  scoping + review caveats).
- **L5** B3 (~2 ms, small).
- **W4 (new, general decode opt — not spec-specific)**: the fp16
  family (self_attn.q/k/v + linear_attn.in_proj_a/b + layer 0,
  ~90 GEMMs/step) runs `triton_matmul` at **~19 ms/step at EVERY M**
  (weight-bound; M-invariant: 174 µs at M=1..16 on fa_q). Microbench
  (`fp16_skinny_m.py`, promoted): Triton *beats* rocBLAS on MI50
  (174 vs 340 µs on fa_q; 838 vs 1751 on layer0_down) — so the
  original L1 "route to rocBLAS" is dead on arrival. The win is a
  custom weight-row-parallel skinny fp16 GEMM (GEMV structure):
  fa_q's 31.5 MB weight read is 31 µs at HBM speed vs Triton's 174.
  ~15 ms off *every* decode step (spec and no-spec alike) — lifts
  the baseline 36.5 ms → ~22 ms (~45 t/s no-spec) but does NOT
  change the spec ratio. Parked as W4; do after Phase 1 (it makes
  the A/B baselines move underneath the spec work).
- **Old L1 (fp16 M≤4 routing): DEAD** — triton_matmul is M-invariant
  and already the MI50-best of {triton, rocBLAS}; nothing to route.
- **Old Phase 1 (GDN small-M kernel): CLOSED — already exists**
  (W3). GDN needs no new kernel for single-request spec.
- Stop rule re-anchored: measured post-Phase-1 draft step > 60 ms →
  stop. Expected ~58–65 (L2 on 95; the unexplained ~20 ms AWQ M=4
  inflation is what L2 is supposed to kill).
- Ceiling (microbench-verified costs; T_d ≈ 58 post-L2, T_n = 37
  with L4): min1 (r=.84, a=.71) → **1.06×**; a=1.0 → **1.23×**;
  min2 (r=.40, a=1.08) → **1.15×**. Draft quality `a` is *the*
  swing variable (0.71→1.0 flips 1.06→1.23×). W4 lifts both arms
  ~40% without moving the ratio.

## Artifacts this session

- Promoted into `benchmarks/kernels/gfx906/`: `gdn_step_spy.py`
  (from /tmp/bench/gdn_step_probe.py), `bench_awq_m_scaling.py`.
- `spec_ngram_dense.py`: `--repeats` knob + CI summary (committed
  earlier this session).
- Roadmap restructured (Phase 1 = L2 AWQ skinny-M; old Phase 1
  closed as W3; W1 = multi-request chunk reclass; W2 = MoE port;
  W4 = skinny fp16 GEMM, general decode opt).
- `bench_fp16_skinny_m.py` promoted (triton_matmul vs rocBLAS at
  M=1..16): **triton wins on MI50, and is M-invariant** → old L1
  (route fp16 to rocBLAS) is dead; the fp16 family becomes W4
  (custom skinny fp16 GEMM, ~15 ms/step off ALL decode, ratio-
  neutral).
- `spec_prof_probe.py`: in-process env fix (multiprocess EngineCore
  empties the profiler table).

## GEMM census (eager, single prompt, exact per-step counts)

`gemm_step_census.py` (promoted) wraps `ops.gptq_qgemm`,
`triton_matmul` and the GDN path markers; enforce_eager single-prompt
run, counts per decode step by (N, K):

- **AWQ call structure is M-invariant**: nospec 236/step, spec draft
  231/step — the *same four shapes*:
  `5120×6144` ×63, `34816×5120` ×63 (fused gate+up),
  `5120×17408` ×63 (down), `16384×5120` ×47 (GDN in_proj_qkvz).
  The "263 vs 234" in-engine gap was profiler warmup/capture
  contamination — **no launch inflation exists**. The AWQ M=4 delta
  is pure per-call kernel time: **~17 ms/step** over the exact shape
  set (34816 and 16384 dominate; microbench-scaled).
- **fp16 is M-gated in the dispatcher, not M-invariant in the
  engine**: at M=1 the unquantized projections take the GEMV op path
  (`_llmm1_tiny_m`/`_gfx906_gemv_long_k`/`dense_gemv` — the
  `triton_matmul` call count is **0** at M=1); at M=4 they take
  `triton_matmul` (N=96 ×~43/step for GDN a/b, N=14336 ×~14/step
  for FA qkv, one ~[248320, 5120] inductor-fused mega-GEMM per
  step = layer 0). fp16 M=1 cost ~3.5 ms (GEMV) → M=4 ~16.5 ms
  (triton): **delta ~13 ms** — the fp16 family IS an M=4 lever
  after all (my "L1 dead" call was about triton_matmul's own
  M-invariance, not the engine's M=1 baseline).
- Layer 0 is entirely fp16 (`modules_to_not_convert`); inductor
  fuses some of its GEMMs into a single N=248320 call at M=4.

**Implementation consequence (Phase 1 final shape):** one kernel
family, two weight formats —
- **L1' (fp16)**: extend `dense_gemv_gfx906` (W16A16, ours,
  M=1-by-construction, template `<RPT, KCHUNK>`) to M≤4 output rows
  (same weight-read-once structure, 4 dots per weight element);
  dispatch for 1 < M ≤ 4 in `rocm_unquantized_gemm_impl`. Kills ~13
  ms/draft step. The same kernel at M=1..16 is W4 (general decode
  opt: replaces both the current GEMV and the triton 174 µs floor).
- **L2 (AWQ)**: 4-row GEMV for the packed-4bit weights (exllama
  GEMV structure, x shared, 4 output rows per weight row) or
  re-tile q_gemm for M=4. Kills ~17 ms/draft step.
- Combined target: T_d 95 → ~64 (probe context); the ceiling table
  is unchanged (the spec-specific delta is ~30 ms either way).

## Next

1. Phase 1: bench the two candidate shapes (dense_gemv M=4 extension
   vs status-quo triton; AWQ 4-row GEMV vs re-tiled q_gemm) at the
   census shapes; implement the winners; unit tests; draft-step
   probe; stop-rule check (≤ 60 ms).
2. Phase 2 (L4): q1 FULL graphs for no-draft steps (spike first per
   review caveats).
3. Phase 3: L3 (proposer) + adoption gate (1.15×, PPL-neutral,
   M=1 token identity).
4. W4 = the dense_gemv M≤16 extension shipping past M=4 (lifts both
   arms ~40%) — do after the spec gate.
