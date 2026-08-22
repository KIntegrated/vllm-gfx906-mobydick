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

## Phase 1 L1' implementation (dense_gemv M≤4 kernel) — built, awaiting GPU

Kernel (`csrc/rocm/dense_gemv_gfx906.cu`, new `dense_gemv_m_kernel`
+ `dense_gemv_m4_gfx906` op, bound in `torch_bindings.cpp` /
`_custom_ops.py`):
- Row-parallel weight structure identical to the M=1 GEMV; per-M-row
  x-slices in registers (x is M*K*2B ≤ 40 KB, L2-resident across
  blocks); `acc[RPT][4]` fp32, RPT ∈ {2,4} (the packed-CAS epilogue
  needs adjacent rows; per-M-row outputs are N apart so RPT=1 packing
  is impossible).
- ksplit==1: per-(m, row) direct stores. ksplit>1: per-M-row packed
  CAS (32-bit RPT=2 / 64-bit RPT=4), lane-0 gather via shfl (single
  warp) or LDS (multi-warp). Launcher requires N % RPT == 0 (kills
  the ragged-tail OOB class the M=1 kernel only avoids in practice).
- KCHUNK ∈ {512, 1024, 2048} (4096 bench-only as before); host op
  takes kchunk, RPT via VLLM_GFX906_GEMVM_RPT (default 2).
- M=1 path untouched (old kernel + op unchanged) → nospec regression
  is structurally zero; dispatch gate is 2 ≤ M ≤ 4.

Dispatch (`vllm/model_executor/layers/utils.py`, new
`_gfx906_spec_gemv_m4`): gfx906 + fp16 + 2≤M≤4 + k%8==0 + N%2==0 +
k%{512,1024}==0, kchunk = largest of 2048/1024/512 dividing K;
excludes the tuned hipBLAS special case (m==5120, 2048≤k≤2304);
kill switch `VLLM_GFX906_SPEC_GEMM=0` (default on — M=1 untouched).

Tests (`test_rocm_unquantized_gemm.py`): numeric M∈{2,3,4} at
K=5120 (ksplit=5 CAS chain), K=17408 M=4 (ksplit=17, the census
down_proj shape), mock dispatch test (M=4 → op+kchunk, M=1 → not
called, special-case shape → not called, kill switch → triton),
never-off-gfx906 guard. Ruff-clean vs baseline.

Microbench (`/tmp/bench/bench_fp16_m4.py`, not yet promoted): census
shapes (96×5120 ×43, 14336×5120 ×14, 248320×5120 ×1 LM-head,
16384×5120, 5120×6144, 34816×5120, 5120×17408) — M=1 ref vs
triton M=4 vs m4 M=4 (kchunk 512/1024 sweep). Built (incremental,
~5 min) into `_rocm_C`; **GPU held for RCCL bench + unit tests —
microbench, pytest, and the in-engine A/B run when it frees.**

Expected (weight-bound model): m4 M=4 ≈ M=1 cost per shape
(248320×5120 ≈ 2.0 ms vs triton ~11.9; 14336×5120 ≈ 0.15 vs ~0.7);
L1' step saving ≈ 13-20 ms if the weight-read speed holds at 4× the
ALU work.

## ROCm reinstall + clean rebuild (2026-08-18)

ROCm moved: the old `/opt/rocm-7.14` custom build (7.13.60850 runtime)
is now `/opt/rocm-7.14-gfx906-old`; the new install is `/opt/rocm`
(7.14.60850, Debian-packaged, dev cmake packages via dpkg — the first
clean build failed CMake configure on the missing `hip-lang` package,
resolved by the dev-tools dpkg install landing them under
`/opt/rocm/core-7.14/lib/cmake`, symlinked through
`/etc/alternatives/rocm-lib`). `~/env-rocm-7.14-gfx906.sh` now points
at `/opt/rocm` (top-level `lib` symlinks everything; gfx906 rocblas
Tensile libs confirmed present). Full clean rebuild (`rm -rf build/`,
stale .so's had `RUNPATH /opt/rocm-7.14/lib` baked in): configure OK
(HIP 7.14.60850), ccache made it ~4 min. The new .so's carry **no
RUNPATH** — runtime resolution is via the env script's
LD_LIBRARY_PATH (system ld cache has no /opt/rocm entries). The
optional `vllm-rs` Rust CLI cargo build fails (vllm-server crate
build-script error) — tolerated, precompiled binary from Aug 16
stays; Python extension build is unaffected. Import + gfx906_fa
backend registration + m4 op smoke all pass on the new stack.

## Phase 1 L1' — measured WIN (microbench + tests)

`benchmarks/kernels/gfx906/bench_fp16_m4.py` (census shapes,
kchunk 512/1024 sweep, RPT 2/4 sweep):

| per draft step (all census fp16 shapes) | µs |
|---|---|
| M=1 ref (GEMV family) | 6.9 ms |
| triton_matmul M=4 (status quo) | **25.6 ms** |
| dense_gemv_m4 M=4 (kc1024, RPT=2) | **11.2 ms** |
| dense_gemv_m4 M=4 (kc1024, RPT=4) | 11.0 ms |

L1' saving at the M=4 worst case: **~14.5 ms/step**; at the typical
draft-step M (1+accepted, a≈0.71 → M≈1.7–2) the saving is larger
(~17–18 ms, cost scales ~linearly in M between 6.9 and 11). The M=4
path runs 1.5–1.6× the M=1 cost (4× the dot work — ALU-bound, as
modeled; the 5120×6144 shape is 1.08×). kc1024 beats kc512 on every
shape; RPT=4 wins ~2% overall but RPT=2 wins at N=96 — default
stays 2 (matches the M=1 GEMV rule; env sweep knob kept).

pytest: 25 passed / 2 skipped (all 4 new tests green, incl. M=4
K=17408 ksplit=17 CAS chain). M=1 paths untouched.

## ROCm 7.14 rebuild: GPU wedge incident + max-ilp disabled (2026-08-18)

After the Debian-packaged ROCm switch and full clean rebuild
(30 min, only the optional vllm-rs Rust CLI failed), weight loading of
the dense 27B AWQ model faulted the GPU twice:
`hipErrorLaunchFailure` mid-shard-copy (sticky error), CP
"unrecoverable state due to unsuccessful queues preemption" in dmesg,
queue eviction, BACO reset (self-recovered once; wedged the driver
once, fixed by reboot). Intermittent (2 hits, then clean runs).

Only compile difference in the load path: `q_gemm.hip` (contains
gptq_shuffle / gptq_shuffle_awq_qweight) is the single file in
csrc/libtorch_stable built with the per-file LLVM **max-ilp**
scheduler flag. Under the new LLVM (ROCm 7.14.60850 toolchain) that
flag is suspected of emitting a faulting schedule. Decision: rebuild
with `VLLM_NO_MAX_ILP=1` (all 4 max-ilp files lose the flag). Since
then: **4 full clean runs** (3× weight load + generate, plus the
serialized-kernel probe). Cost carried: q_gemm +15..26%, FA decode
−2..5% vs the max-ilp build — absolute serving numbers on this build
are ~1–2 t/s below the max-ilp records; A/B ratios within one build
stay valid. Revisit per-file max-ilp under the new LLVM later
(candidates: flag only for the FA files, or a scheduler-variant).

## In-engine step-wall probe (L1' confirmed in-engine)

`/tmp/bench/step_wall_probe.py` (promote when settled): eager
single-prompt (maximally repetitive), conv-anchor step segmentation,
one CUDA event per step boundary (conv #48). `AMD_SERIALIZE_KERNEL=3`
inflates walls ~2.2× (35→82 ms M=1) — use it only to identify
faulting kernels, never for timing.

| arm (eager) | wall | no-draft step | draft step min/mean |
|---|---|---|---|
| nospec M=1 | 24.9 t/s | 35.2 / 37.3 ms | — |
| spec triton M≤4 | 36.2 t/s | 40.3 ms | 46.9 / 66.6 ms |
| spec m4 GEMV (L1') | **43.7 t/s** | 39.8 ms | 41.9 / **53.2 ms** |

L1' in-engine saving: **13.4 ms/draft step** — matches the
microbench prediction (14.5 ms M=4 worst case). This
maximally-repetitive prompt already wins 1.76× in eager (consistent
with the graph-mode 1.12× finding). Draft-step floor ≈ 42 ms
(min); the mean's excess is eager launch/CPU gaps (292 GEMMs) that
CUDA graphs remove in serving.

L2 re-scope: the AWQ family is dequant-ALU-bound, not
weight-read-bound (M=1 q_gemm runs 44 MB in 75 µs = 590 GB/s, ~2× off
the HBM roofline) and the tiled M=4 kernel already shares the
dequant across m-rows. An M≤4 AWQ GEMV removes atomics/LDS/m-tiling
overhead only — estimated **2–8 ms/step**, not the census's 17 ms
M=1→M=4 delta. Deprioritized behind the serving A/B and the in-tree
ngram GPU-proposer quality fix (~5 ms, no kernel work).

## flash-attn: no rebuild needed

`flash_attn` (editable, /local/git/flash-attention-gfx906) selects
backend at import: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` → pure
Triton `flash_attn_triton_amd` (verified: varlen func finite and
correct); otherwise `import flash_attn_2_cuda` (compiled CK kernel —
not present in the tree, and CK does not support gfx906 anyway). The
compiled .so vanished at some point (reboot/migration), which broke
any process lacking the env var; with the var (mandatory for vLLM
ROCm regardless) the missing extension is never touched. (The "triton
packed path returns inf" seen during the import test was an fp16
`.sum()` overflow in the test, not a kernel bug — re-verified with
deterministic values 2026-08-18; also checked GPU health after the
TP=2 tests: clean.)

## Serving A/B (no-max-ilp build, L1' on) + graph-mode profile

`run_spec_ab.sh` (3 agentic prompts × 3 repeats, 512 tok greedy,
server per arm, `--served-model-name qwen27` — client 404s without
it):

| arm | mean t/s | CI95 lo |
|---|---|---|
| baseline | 26.52 | 25.48 |
| ngram3 (L1') | 25.06 | 24.67 |

L1' recovered most of the Phase 0 gap (0.71× → **0.945×**) but the
agentic case still loses. Per-prompt: baseline 29.3/26.0/25.7,
ngram3 24.3/25.3/25.6 — the spec arm is flatter (the repetitive
bug-fix prompt p0, baseline's fastest, loses most).

Counters (ngram3): a = 1.09 accepted/step, ~37% of steps are
drafts (134/366 on p0), ~63% no-draft. Back-solving step walls from
elapsed (incl. ~2–2.5 s TTFT): **no-draft spec step ≈ 48 ms vs
nospec 35–37 ms** — a ~13 ms penalty on 63% of steps. The eager
probe showed only +3–5 ms there (n=2), so the extra ~10 ms is
graph-mode: suspected cudagraph padding of spec steps to the draft
capacity (M=4) so no-draft steps pay M=4 GEMM costs (AWQ 29 vs 21
ms, fp16 16.7 vs ~5). UNVERIFIED — needs a padded-size trace or the
dispatcher's captured-size log. This, not the draft step, is the
dominant agentic loss (B3-family lever, in-tree core-vLLM risk).

`spec_prof_probe.py --spec` (repetitive prompt, graph mode, m4
build): 128 tok, Self CUDA 2.736 s, ~50 steps → 54.7 ms/step,
matching the eager probe (53 ms). Per step:

| kernel | ms/step |
|---|---|
| gptq 4bit GEMM (AWQ M=4, 226 calls) | 29.1 |
| dense_gemv_m_kernel (L1', 65 calls) | 9.4 |
| inductor FxGraph (fp16 n>16 GEMMs) | 7.3 |
| Cijk/hipBLAS (M=1 fp16) | 5.2 |
| reconstruct_exllama_4bit (AWQ z/s) | 1.85 |
| GDN fused_seq (1968 calls = 4/step… 48 layers × 2?) | 1.1 |
| FA decode (q8 + gather) | 0.8 |

**L1'' found**: the fp16 n>16 GEMMs (LM head 248320×5120, layer-0
FA projections) bypass the m4 hook (dispatcher only hooks the
n≤16 branch) and run inductor at M=4: 5.6 ms/call. The m4 kernel
does that shape in ~2.0 ms (microbench). Extending the m4 dispatch
to n>16 for M∈{2,3,4} is ~5 ms/step of further L1' saving —
dispatch-only, no new kernel. (M=1 stays on inductor/hipBLAS, which
is at the roofline there.)

### 2026-08-24: L1'' investigation — dispatch gap was an artifact; L5 root-caused + fixed

**L1'' (m4 dispatch for n>16) — re-scoped, mostly a non-issue.**
Dispatch spy on the eager m4 build (monkeypatch
`rocm_unquantized_gemm_impl`, log all shapes): in EAGER every decode
fp16 GEMM reaches the dispatcher, including the LM head (n=4, m=248320,
k=5120), and the m4 kernel handles the whole M=4 decode set (400
launches/step). The graph-mode "FxGraph 7.3 ms/step" row was NOT a
missed dispatch — it is the 7 no-draft steps' inductor piecewise
fragments (see L5). Optional leftover: extend the m4 dispatch to the
K=1024 gate shapes (m=128, k=1024, ~0.7 ms → 0.25 ms each, ~2 ms
/step) — dispatch-only, low priority.

**L5 (no-draft spec step cost) — ROOT-CAUSED AND FIXED** (pure Python,
no build). Root cause: `compilation.py
adjust_cudagraph_sizes_for_spec_decode` rounds ALL capture sizes up to
multiples of `uniform_decode_query_len` (= 4 for ngram-3) — the
upstream stopgap for issue 28207 that predates separate decode/mixed
capture-size handling. Side effect: the PIECEWISE key set loses sizes
1–3, so a no-draft spec step (1 token, non-uniform → PIECEWISE) pads
up to the size-4 graph and runs every GEMM at M=4: ~8 ms AWQ + ~6 ms
fp16 ≈ the observed ~13 ms no-draft penalty on 63% of agentic steps.

Evidence: (a) batchdesc spy — no-draft steps dispatched as
`BatchDescriptor(num_tokens=4, uniform=False) PIECEWISE` despite
num_tokens=1; (b) kernel symbols — graph profile showed only
gptq `<true,4>` (M=4) and zero `<true,1>`; eager profile showed both
(1652 M=1 calls = the 7 no-draft steps); (c) the graph-only "Call
CompiledFxGraph" row (65 calls, 5.64 ms avg) = the 7 no-draft
PIECEWISE steps' ~9 inductor fragments each — FULL draft steps show no
such rows.

Fix (env-gated, gfx906-only): after the rounding, re-add sizes 1..q-1
(`VLLM_GFX906_SPEC_CG_SMALL`, default on). FULL keys already filter
sizes < q, so this only adds small PIECEWISE graphs. Verified:
no-draft steps now dispatch `num_tokens=1 PIECEWISE`; profile shows
gptq `<true,1>` (1652 calls @ 90.6 µs, was absent) and M=1 GEMV
kernels on the no-draft steps (477 launches = 7 × 68).

### 2026-08-24 (later): L5 fix serving A/B — ngram3 now 1.077×

run_spec_ab.sh (recreated post-reboot), 3 agentic prompts × 3 repeats,
no-max-ilp build, L1' on + cg-small fix on:

baseline 26.53 t/s (CI 25.49) — matches the 26.52 pre-fix baseline.
ngram3 28.58 t/s (CI 28.12) — **1.077×** (was 0.945× pre-fix).

+3.5 t/s from the cg-small fix alone (no-draft steps 48→~40 ms; the
size-1 PIECEWISE graph still pays ~9 small inductor-fragment replays
per step, so it lands between the M=4-padded 48 ms and the 35–37 ms
eager M=1 cost).

ngram ceiling assessment: the 1.15× all-in gate is NOT reachable with
ngram on agentic prompts — even the full L2 (2–8 ms on the 37% draft
steps) adds ~+0.3 t/s → ~1.09×. ngram's acceptance (1.09/step agentic)
is the ceiling; a model-based drafter with real acceptance on
non-repetitive text is the remaining lever. L2 deferred as optional
low-ROI polish; MTP investigation is next (the model HAS an MTP head:
mtp_num_hidden_layers=1, 15 mtp.* tensors, 'mtp' in
modules_to_not_convert → unquantized fp16).

### 2026-08-24 (later): MTP investigation — model HAS an MTP head; k=2 = 1.49×

(Correction: an earlier note in this log claimed the model has no MTP
head — WRONG. config.json text_config has mtp_num_hidden_layers=1 and
mtp_use_dedicated_embeddings=false; the checkpoint carries 15 mtp.*
tensors: mtp.fc + mtp.layers.0.{self_attn,mlp,*layernorm,*norm}. The
'mtp' entry in modules_to_not_convert makes the MTP layer unquantized
fp16 (verified: no M=1 AWQ calls in the profile; all decode AWQ is
M=3 target).)

Setup: `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'`
— vLLM auto-resolves the drafter to the same checkpoint (Qwen3_5MTP
registered in the model registry). MTP layer = 1 full-attention decoder
layer (q_norm/k_norm, attn_output_gate) + fp16 MLP + mtp.fc
(10240→5120, takes [h ⊕ e]). embed_tokens and lm_head are SHARED with
the target (no mtp.* embedding/lm_head tensors) — the drafter's
per-forward weight traffic is fc(0.10) + attn(0.28) + mlp(0.54) +
lm_head(2.54) ≈ 3.4 GB.

First crash: the proposer's ROCm allowed_attn_types whitelist did not
include our Gfx906FAMetadata → ValueError on the first decode step.
Fixed by adding it to llm_base_proposer.py (gfx906-gated import).
Our FA builder needs no build_for_drafting override — the base
default (build with the mutated common_attn_metadata) matches what
ROCM_ATTN uses.

**Serving A/B (3 agentic prompts × 3 repeats): baseline 26.53 → mtp2
39.47 t/s (CI 37.17) = 1.488×** (first run, pre-fc-fix). Counters:
1.82 tokens/step, 86–95% draft-token acceptance, token-identical to
baseline on both agentic prompts (one benign divergence on the
repetitive prompt). Every step drafts (no no-draft penalty: MTP
always proposes; target always runs uniform M=3).

k=2 vs k=3: the drafter forward costs ~8 ms each while the marginal
third draft token's conditional acceptance is low (k=2 already
captures 1.82/step) — k=3 would add ~8 ms for ~+0.1 tokens/step;
k=1 halves drafting but loses the second-token acceptance. k=2 is the
sweet spot (measured, not yet A/B'd against k=3 — TODO if time).

### 2026-08-24 (later): MTP drafter cost decomposition + first optimization

Profile (torch profiler, graph + eager, repetitive 128 tokens, ~47
decode steps): step ≈ 60 ms = target M=3 (~45 ms: AWQ 23.9 + m4-fp16
10.1 + GDN/FA/norms ~11) + **drafter ~15 ms**.

Shape spy (dispatcher + gptq_gemm + F.linear) + dispatch microbench
(MI50 HBM 1.02 TB/s) — drafter GEMMs per step:

| shape (N×K) | M | time | GB/s | path |
|---|---|---|---|---|
| lm_head 248320×5120 (2.54 GB) | 1 | 3.13 ms | 811 | LLMM1 (near roofline) |
| fc 5120×10240 (0.10 GB) | 1 | 544 µs | 193 | triton (k>8192, ≠17408) |
| qkv 14336×5120 | 1 | 184 µs | 796 | LLMM1 |
| out 5120×6144 | 1/3 | 81/131 µs | 777/482 | LLMM1/m4 |
| gate_up 34816×5120 | 3 | 685 µs | 521 | m4 kernel |
| down 5120×17408 | 3 | 350 µs | 510 | m4 kernel |

Two findings:

1. **fc (K=10240, M=1) fell through to triton at 193 GB/s.** The
K-split dense GEMV does it in 148.7 µs at kchunk=2048 (705 GB/s;
1024→395, 512→254). Fix (dispatch-only, gfx906, gated by
VLLM_GFX906_DENSE_GEMV): extended `_gfx906_gemv_long_k` to the
(5120, 10240) shape with kchunk=2048, and the n==1 branch conditions
to `k in (10240, 17408)`. Saves ~0.8 ms/step. Dispatcher unit tests
25 passed / 2 skipped.

2. **The m4 kernel (L1') plateaus at ~520 GB/s for ALL M (even M=1
through the m4 op: 552) while the M=1 kernel does 700–815 on the same
shapes.** Root cause: the m4 kernel allocated `xa[4]` x-slices +
`acc[RPT][4]` + `acc_flat[RPT*4]` unconditionally (runtime M) →
~130+ VGPRs → ~35% occupancy loss at KCHUNK=1024. Fix (in progress):
template the kernel on M (`dense_gemv_m_kernel<RPT, KCHUNK, M>`),
static-size the arrays, drop the acc_flat copy (in-place shfl
reduction). Expected: M=2..4 up to ~700–800 GB/s → ~2.5 ms/step on
the MTP drafter (M=3) and ~3 ms/draft-step on ngram (M=4).

Structural note (user's theory): the MTP drafter IS compute/traffic-
hungry on this memory-bandwidth-bound GPU — 6.8 GB of extra weight
reads per step (2×3.4 GB) on top of the target's ~25 GB. Its floor is
~6.8 ms/step; measured 15 ms before the two fixes above. The lm_head
GEMV (2.54 GB ×2/step, 811 GB/s) is the irreducible ~6.3 ms; the
rest is recoverable.

### 2026-08-24 (later): L3 — ngram proposer CPU cost

Roadmap L3: the CPU ngram proposer runs in the scheduler (engine
process) every step — Python string/suffix matching on the prompt.
Measured via the step probe: with spec on, the nospec→spec step-wall
delta includes proposer time. The proposer cost is small vs the
measured step cost (steps are 35–56 ms GPU-bound; proposer is single
digit ms in Python for our prompts). Not worth a GPU proposer for
this workload; `ngram_gpu` (roadmap L3 alt) was already rejected on
quality (Phase 0). Closing L3 as not-beneficial-for-this-workload.

Remaining roadmap items: L2 (AWQ M≤4 GEMV, 2–8 ms/step, kernel work)
and the optional m4 K=1024 dispatch extension (~2 ms). Both small;
L2 is the only remaining >2 ms lever.

### 2026-08-24 (later): m4 kernel templated on M — M=2 +37%, M=4 kept old

The m4 kernel (L1') was runtime-M: `xa[4]` + `acc[RPT][4]` +
`acc_flat[RPT*4]` allocated unconditionally → ~130+ VGPRs → ~35%
occupancy loss at KCHUNK=1024 (measured 520–550 GB/s for all M, vs
700–815 for the M=1 kernel on the same shapes).

Fix: `dense_gemv_m_kernel<RPT, KCHUNK, M>` — static arrays, in-place
shfl reduction (no acc_flat copy). Results (default RPT=2, KC=1024,
GB/s, 2.54/0.36/0.18 GB shapes):

| M | lm_head | gate_up | down | (old m4: 552/548/539 … 507/501/486) |
|---|---|---|---|---|
| 1 | 816 | 795 | 730 | +45% |
| 2 | 730 | 717 | 677 | +37% |
| 3 | 544 | 535 | 525 | +3% |
| 4 | 311 (REGRESSION) | 307 | 298 | — |

M=4 regressed 507→311 in the templated form (cause unattributed —
RPT=4 makes everything worse, kc=512 worse too; the x/w load ratio
M/RPT=2 is the floor but 311 is below even that prediction). Practical
resolution: keep both kernels — the launcher dispatches M=1..3 to the
templated kernel and M=4 to the restored runtime-M kernel
(`dense_gemv_m_kernel_rt`, byte-identical to the L1' version). M=4
in-engine (ngram draft steps) is therefore bit-identical to before.

The residual M=3/4 ceiling is the activation re-read ratio (x bytes per
w byte = M/RPT): M=3@RPT2 = 1.5:1 → ~540 GB/s. Beating that needs an
LDS-tiled GEMM-style kernel (each x tile read once per many w tiles) —
a full rewrite, deferred: the in-engine gain is ~1–2 ms/step.

In-engine effect: MTP drafter M=3 forward ~6.5→6.1 ms (lm_head
4862→4671 µs, gate_up 685→666, down 350→340); M=2 (unused by current
workloads) +37%. Dispatcher suite 25 passed / 2 skipped (includes the
m4 real-kernel numeric tests).

k=3 vs k=2 MTP: not A/B'd — the third drafter forward costs ~8.5 ms
while the third draft's conditional acceptance is ~15–25% (k=2 already
1.815 tokens/step); break-even needs ≥2.17 tokens/step. k=2 stands as
the sweet spot pending a model with a stronger MTP head.

### 2026-08-24 (later): final A/B + GPU wedge #2

Final 3-arm A/B (agentic prompts, 3 repeats, all fixes in):

- baseline: 26.50 t/s (CI 25.45) — matches all prior baseline bands
- ngram3: 28.55 t/s (CI 28.12) = **1.077×** — unchanged from the
  pre-m4-templating run (expected: ngram's M=4 path is the restored
  rt kernel, bit-identical; no-draft steps use LLMM1 / cg-small fix)
- mtp2: the mtp2 server of this run died at weight-load time with
  "CUDA error: unspecified launch failure" and **wedged the MI50
  (wedge #2, same signature as the 2026-08-19 incident)**: rocm-smi
  shows GPU0 temp/clock N/A with 80% zombie VRAM, fresh contexts fail
  with `amdgpu_query_gpu_info_init failed`. Reboot required.

  Probable cause: the runner killed the ngram3 server and started the
  mtp2 server after a fixed 8 s sleep; vLLM teardown had not released
  the ~25 GB (zombie 80% VRAM visible after), so the mtp2 load ran
  into memory pressure. The earlier mtp2 A/B (same build except the
  m4 M-templating) measured **39.66 t/s (CI 38.49) = 1.500×**.
  **RESOLVED post-reboot (final A/B, same build + m4 M-templating):**
  baseline 26.44, ngram3 **28.92 (1.094×)**, mtp2 **39.74 t/s
  (1.503×**, CI 38.58; 1.819 tokens/step, 90.95% draft acceptance).
  The m4 M=3 templated kernel added no measurable delta over the
  pre-templating 39.66 (the ~0.4 ms/step saving is within noise),
  and no regression (ngram3 unchanged within noise). MoE 35B soak
  on the same build: 66.30/66.31/66.27/66.27 t/s (record band
  65.9–67.0) — no cross-path regression.

  Runner rule for the future: between arms, wait until rocm-smi VRAM
  is < 5% before starting the next server (8 s sleep is not enough
  after a kill). Implemented in /tmp/bench/run_final_ab.sh (wait_vram
  loop); the post-reboot run with the fix completed all three arms
  cleanly. NOTE: the reboot also left the NFS /data mount down
  (fstab `auto` entry) — first A/B attempt after reboot failed with
  HFValidationError (model path missing), not a GPU fault.

### MTP work state (final, 2026-08-24 post-reboot)

- k=2 MTP is the recommended spec method for this model: **1.503×**
  (39.74 t/s final A/B, CI 38.58; 1.819 tokens/step, 90.95% draft
  acceptance), token-identical to baseline on the agentic prompts.
- The MTP drafter is fp16-only (modules_to_not_convert) — every
  drafter weight byte hits HBM (3.4 GB/forward); k=2 = 2 forwards/step
  (~6.8 GB) ≈ target's M=3 GEMM cost. On a fast-compute GPU the
  drafter is nearly free; here it is ~25% of the step. Remaining
  levers (deferred): M=3/M=4 LDS-tiled kernel (beats the x/w re-read
  ratio floor, ~1–2 ms/step), k=3 (needs ≥2.17 tokens/step), AWQ
  MTP (quality risk, out of scope).
- ngram3 stands at 1.094× (28.92 t/s final A/B; 1.077× pre-reboot
  run — within noise) — the cg-small fix consumed its ~1.4 ms no-draft
  penalty; L2 (AWQ M≤4) remains the only unmeasured >1 ms lever but
  its ceiling was revised down (dequant-ALU bound).


### 2026-08-24 (evening): external code reviews absorbed

Two independent critical reviews of `38ceb5d957..68243a61b2` landed:
`docs/gfx906/spec-dec-code-rev-glm.md` and
`docs/gfx906/spec-dec-code-rev-ds4.md`. Disposition:

**ds4 F1 (claimed BLOCKER: cg-small fix unreachable — MRV2) —
REFUTED.** The serving runs use MRV1, not MRV2:
- `VLLM_USE_V2_MODEL_RUNNER` unset; Qwen3.5 is not a default-V2
  architecture (`use_v2_model_runner` property → False).
- The serve logs' `gpu_model_runner.py:6821` line is an EXACT match for
  `vllm/v1/worker/gpu_model_runner.py` (MRV1) — the MRV2 file
  (`vllm/v1/worker/gpu/model_runner.py`) does not have 5424 lines.
- The per-arm PIECEWISE graph counts in the serve logs (ngram3: 8,
  mtp2: 7) equal the rounded sizes + the restored small sizes per
  method (q=4 → 5+3=8; q=3 → 5+2=7) — the block executes.
- Pre-fix kernel profile (only `<true, 4>` AWQ in graph mode) vs
  post-fix (M=1 calls present on no-draft steps) is independent
  confirmation.
The fix did, however, **break an upstream test**:
`test_resolve_cudagraph_mode_adjusts_spec_decode_sizes_only_for_v1`
asserts the rounded-only sizes [4,8,12,16] and fails on gfx906 with
the fix on. Resolved: that test now pins `VLLM_GFX906_SPEC_CG_SMALL=0`
(upstream behavior) and a new
`test_resolve_cudagraph_mode_gfx906_spec_cg_small_restores_sizes`
asserts the gfx906 behavior ([1,2,3,4,8,12,16]). Both pass.

**Fixed this session:**
- glm 1.2: `VLLM_GFX906_GEMVM_RPT=4` + N≡2 mod 4 previously tripped
  the launcher's TORCH_CHECK mid-serving; `_gfx906_spec_gemv_m4` now
  mirrors the rpt env and returns None (triton fallback) when
  `m % rpt != 0`.
- glm 1.3: ksplit==1 (plain-store epilogue) numeric test added
  (K=1024/kc=1024, M=2..3..4, 14336 rows). Suite now 28 passed /
  2 skipped.
- ds4 F8: cg-small unit tests (above).

**Accepted / documented (no code change):**
- glm 1.4 / ds4 F5: the ksplit>1 fp16-CAS epilogue is
  run-to-run order-nondeterministic (atomic order), and "token-
  identical to baseline" was measured once, not guaranteed. This is
  the S3-class fp-drift bar (PPL/coherence, not token identity); the
  MTP acceptance counter (90.95%) absorbs any draft-logit jitter
  downstream. Stated here so it is not re-litigated.
- glm 2.1: cg-small graph memory cost — ngram3 serve log:
  "Estimated CUDA graph memory: 1.60 GiB total" (8 PIECEWISE incl.
  the three small graphs + 3 FULL) at GPU_UTIL 0.95, max_num_seqs=4;
  the A/B runs prove the 4-seq GDN budget still fits (the size-1..3
  pools are small; largest graph unchanged).
- ds4 F6: [5120, 10240] fp16 is unique to the MTP fc in this
  checkpoint (tensor list checked); the long-k kernel itself is
  shape-generic, K is the tuned dimension. RPT: the default RPT=2
  beats RPT=4 on every measured m4 shape (today's mscan: RPT=4
  lm_head M=1..4 = 792/453/324/320 vs 816/730/544/506), so no RPT
  sweep is pending.
- glm 1.7: Gfx906FAMetadata contract — the 90.95% acceptance +
  token-identical MTP A/B is the contract test in action (wrong
  draft-attention fields would collapse acceptance). A field-presence
  test would only pin attribute names; deferred.
- glm 1.5 / ds4 F4: the templated M=4 is NOT compiled — the launcher
  never instantiates `dense_gemv_m_kernel<_,_,4>` (the M=4 branch
  calls `_rt`); the kernel template's M parameter is generic (the
  same source serves M=1..3). The 311 GB/s measurement was of a
  transient build that did instantiate it. Root-causing remains an
  open afternoon task (rocprof register/occupancy comparison).

**Open items handed back (decisions, not quick fixes):**
- ds4 F3 / glm §5: ALL branch numbers are from the `VLLM_NO_MAX_ILP=1`
  build (the per-file max-ilp flag is the suspect in both GPU-wedge
  incidents). To close: root-cause the launch failures, or re-run the
  gate A/Bs (mtp2 + baseline) on a max-ilp build. ~1 hr build +
  ~20 min A/B + wedge risk. **Needs user OK** (reboot risk).
- ds4 F2: the m4 hook is intentionally NOT spec-gated — M=2..4 fp16
  decode (2-4 concurrent seqs, the production config) also takes it,
  replacing triton_matmul (strictly slower on this range). Empirical
  gate for the non-spec path: the standard dense 27B 4-seq bench
  (record band 25.14-25.60 t/s) re-run on this build; also the
  uniform-prefill case (ds4 F7) since 4 uniform prompts fire the FA
  uniform fast path in prefill. The F7 pad-row concern is
  structurally void: `num_tokens == num_seqs * max_seqlen_q` implies
  every sequence has exactly max_seqlen_q tokens (sum of lengths =
  num_tokens, each ≤ max), so no padded rows exist when the fast
  path fires.

**ds4 F2/F7 empirical gates (run this session):**
- Dense 27B 4-seq bench (4× uniform 2048-tok prompts → uniform
  prefill fast path + M=4 non-spec decode), m4 ON: 23.70 t/s;
  `VLLM_GFX906_SPEC_GEMM=0` (m4 OFF): 23.69 t/s — identical to
  noise. The m4 kernel is neutral on the non-spec path (no
  regression; the record band 25.14–25.60 was set on a max-ilp /
  old-ROCm build, so 23.70 vs 25.25 is the build delta, not this
  branch's code — see the max-ilp open item above).
- Serial-vs-batch token probe (`benchmarks/kernels/gfx906/
  uniform_batch_probe.py`): prompt 0 diverges at char 59 — a benign
  fp argmax flip near a logit tie ("...using Python's standard
  library and the `croniter` library (which is the industry
  standard" vs "...using the `croniter` library, which is the
  industry standard"), both continuations coherent; prompt 1
  identical. No scatter/gather or row-mapping bug (those would show
  incoherent text). S3-class bar met. The flip location varies
  run-to-run (CAS-order nondeterminism, as documented above).

## 2026-08-18 — n-gram spec decode probe (dense 27B)

> Moved here from DEVLOG-moe-opt.md (topic consolidation, 2026-08).

Question: does n-gram speculative decoding beat the no-spec baseline on
dense Qwen3.5-27B-AWQ (compute/weight-bound decode)? Theory: the draft
is free (no draft model), so it should fit this GPU better than MTP.

**Setup.** OpenAI server, `--max-num-seqs 4 --max-model-len 2816
--gpu-memory-utilization 0.95`, greedy, 3 multi-turn agentic-coding
prompts (tool-call JSON + code context; outputs re-use context
identifiers), 512 tokens each. Arms: baseline (no spec) and ngram
k=3, `prompt_lookup_min=2, prompt_lookup_max=5`. Acceptance from
`vllm:spec_decode_num_*_total` Prometheus deltas + engine
"SpecDecoding metrics" log lines. Scripts: `/tmp/bench/
spec_ngram_dense.py`, `spec_step_probe2.py`, `spec_prof_probe.py`
(prompts + metric scrape; not yet in-tree).

**VERDICT: no speed-up — 0.68×.**

| arm | t/s (mean) | notes |
|---|---|---|
| baseline | **27.46** | 26.81 / 27.96 / 27.62 (server path; LLM-harness record 25.25) |
| ngram k=3, first attempt | 22.99 | artifact: whole engine demoted to PIECEWISE (see below) |
| ngram k=3, after FA fixes | **18.73** | 18.42 / 18.62 / 19.16 |

Spec arm stats: drafts on ~40% of steps, ~36% draft-token acceptance
(per-position ≈ 0.6 / 0.35 / 0.25), 1.08 accepted per draft step.
Outputs differ from baseline at the first near-tie token (benign fp
argmax flips — q4 vs q1 paths; texts coherent, S3-class, not
corruption).

**Blocker 1 (fixed in-tree): the engine was demoted to PIECEWISE.**
`Gfx906FABackend` declared `UNIFORM_SINGLE_TOKEN_DECODE`; vLLM only
FULL-graphs uniform decodes at support ≥ `UNIFORM_BATCH` (the level
whose docstring says it covers spec decodes = 1 + num_spec queries).
So every decode step ran piecewise (~100 ms/step). Fix (commit after
the upstream merge): `UNIFORM_BATCH` when `num_speculative_tokens > 0`
+ capture-safe uniform fast paths in `forward_paged` (the Q-scatter /
out-gather had `int(cu_seqlens_q[...])` D2H syncs, illegal under
capture; gated on the host identity
`num_tokens == num_seqs * max_seqlen_q`). Spec steps now FULL-graphed
(capture 2 s vs 44 s). FA suite 15/15.

**Blocker 2 (open): draft-step overhead.** Step-cost decomposition
(controlled no-draft vs repeat-phrase probes + torch.profiler A/B):

- no-draft step (1 tok, falls back to PIECEWISE — 1-token spec steps
  are not uniform-4q): ~53–64 ms vs 36.5 ms baseline;
- draft step (4 tok, FULL graph): ~80–110 ms vs ~45 ms ideal.

Profiler diff (spec vs no-spec, same 128-token generation) attributes
the draft-step excess to:

1. **GDN spec path uses the chunk (prefill) kernel family**, not the
   fused-recurrent decode kernel: `ChunkGatedDeltaRule*` calls 48 →
   1536 at ~800 µs/layer/step (~18–25 ms/step) vs ~0.5 ms/step for the
   packed-decode fast path. The fused-recurrent kernel is M=1-only;
   upstream routes any multi-token GDN step through chunk kernels whose
   O(chunk) preprocessing dominates at M=4.
2. ~6× `aten::copy_` + ~300 ms `index_select/index_put/index`
   spec bookkeeping (state save/restore, per-position scatters, 4× KV
   writes).

**Blocker 3 (open): no-draft steps.** With min2/max5 only ~40% of
steps get a draft; the 60% without run at piecewise cost. Even at
100% draft rate the current per-step costs give ~25.6 t/s < 27.5
baseline, so draft-rate alone can't win; the two cost blocks must
drop first.

**Ceiling math** (T1=36.5 ms baseline step): with draft steps at ~55 ms
(GDN chunk → recurrent fix) and no-draft steps at 36.5 ms (1-token
spec steps routed to a q1 FULL graph — needs dual q1+q4 uniform FULL
capture when spec is on), r=0.40, a=1.1 → ~32.8 t/s ≈ **1.19×**;
r=0.60 (min_n=1), a=0.9 → ~1.18×. Real but modest.

**Verdict.** n-gram spec decode as configured does not help dense 27B
on gfx906 today (0.68×). The FA CG-support fix stays (required
prerequisite for *any* spec decode on this backend). If revisited:
(1) small-M fused-recurrent GDN spec path (biggest lever, ~27 ms/step),
(2) q1 FULL graphs for no-draft spec steps, (3) re-bench with
min_n=1. MTP/EAGLE would hit the same GDN spec-path wall, so the
lever is shared.

---

## max-ilp split: per-M q_gemm scheduling (2026-08-19)

The 2026-08-24 max-ilp open item is resolved by compiling q_gemm's
4-bit kernel **twice** — M=1 with `-amdgpu-sched-strategy=max-ilp`,
M>=2 without — and routing on m_count at the single choke point
(`gemm_half_q_half_cuda_part`).

**Why split (measured, this session):**

Max-ilp 3-arm A/B (build with flag on q_gemm + FA + skinny):
- baseline 28.56 t/s (+8.0% vs no-max 26.44 — M=1 wins)
- ngram3 27.80 = 0.973x (vs no-max 1.094x)
- mtp2 36.67 = 1.284x (vs no-max 1.503x)
The spec steps are M=3/4-dominant, so the M=1 win is overwhelmed by
the M=3/4 loss.

AWQ M-scaling microbench (27B shapes, us, [max-ilp | no-max]):
| shape            | M=1      | M=2      | M=4      | M=8      |
|------------------|----------|----------|----------|----------|
| down 17408x5120  | 80 \| 95 | 118\|97  | 138\|105 | 160\|160 |
| gate 5120x17408  | 78 \| 103| 118\|115 | 134\|122 | 182\|183 |
| gdn 8192x5120    | 46 \| 47 | 52 \| 52 | 60 \| 61 | 100\|100 |
| fa_q 3072x5120   | 29 \| 36 | 42 \| 42 | 56 \| 53 | 63 \| 59 |

M=1: max-ilp wins 19-24% (down/gate/fa_q); M>=2: no-max wins 14-23%
(down/gate; gdn/fa_q 3072-row shapes ~neutral). The split is strictly
better than either uniform choice.

**Implementation:**
- New TU `csrc/libtorch_stable/quantization/gptq/q_gemm_m1_maxilp.cu`:
  renamed copy of the 4-bit kernel (`gemm_half_q_half_gptq_4bit_
  kernel_m1mi`, m_count=1 only) + `dot22_8_f_m1mi` + host launcher
  `qgemm_m1_maxilp_launch`. **~149 lines of duplication** (140-line
  kernel + 9-line helper + ~20 launcher grid math); the zero-
  duplication alternative (shared macro-renamed header) was
  considered and rejected for this local branch — the copy can't
  perturb the shared file, and both copies carry explicit
  **SYNCHRONIZATION WARNING** headers + SYNC-COPY markers at both
  kernel sites and both dot-helper sites (a one-sided edit silently
  changes M=1 numerics/perf with no compile/test signal).
- `q_gemm.cu`: forward declaration + branch at the top of
  `gemm_half_q_half_cuda_part`: m_count==1 && bit==4 && arch guard &&
  env not disabled -> `qgemm_m1_maxilp_launch` + return.
- CMake: the max-ilp flag moves OFF `q_gemm.hip` ONTO
  `q_gemm_m1_maxilp.hip` (verified per-object in build.ninja; FA x3 +
  skinny x2 keep their flags; q_gemm.hip unflagged).
- Kill-switch: `VLLM_GFX906_QGEMM_M1_MAXILP=0` (M=1 falls back to the
  unflagged kernel).

**Debug saga (all fixed, all gfx906/stable-ABI landmines worth
knowing):**
1. First build: `use of undeclared identifier 'qgemm_m1_maxilp_
   launch'` — the extern forward declaration was missing (added).
2. Second build compiled + linked, but the dispatch probe (torch
   profiler kernel names) showed M=1 still on the ORIGINAL kernel:
   `__gfx906__` is NOT defined in the stable-ABI host compile, so
   both the branch and the whole twin TU compiled to nothing (0
   m1mi symbols in the .so). Convert to a **runtime arch guard**.
3. Runtime guard via `at::cuda::getCurrentDeviceProperties` failed:
   stable-ABI builds (`TORCH_STABLE_ONLY`) #error on ATen/cuda
   headers. Convert to the plain runtime API
   (`cudaGetDeviceProperties`, hipified to `hipGetDeviceProperties`).
4. Branch still dead: `gcnArchName` on this machine is
   **`gfx906:sramecc+:xnack+`** (suffixed), so `== "gfx906"` failed.
   Prefix match (`strncmp(..., "gfx906", 6)`) — same convention as
   skinny_gemms.cu's `find()`.
Verification after fixes: dispatch probe shows M=1 ->
`..._4bit_kernel_m1mi<1>`, M=4 -> `..._4bit_kernel<true, 4>`;
build.ninja per-object flag placement exactly
{m1mi, skinny x2, fa x3} flagged / q_gemm unflagged.

**Verification (this build):**
- AWQ M-scaling microbench: M=1 = 80/78/47/29 (== max-ilp numbers
  exactly); M=2 = 97/115/51/42 (== no-max); M=8 = 161/179/99/58
  (== no-max). OUTLIER: down M=4 = 115/116 (stable across runs) vs
  105 on the no-max build — same unflagged kernel binary, so
  environmental (clock/thermal or I-cache residency of the twin);
  the serving A/B is the arbiter.
- PPL gate (MoE 35B probe): twin ON 15.9730 vs kill-switch OFF
  16.0450 (0.07 delta, 0 top-20 misses, both coherent) — S3-class
  pass (algorithm identical; delta is CAS-order nondeterminism).
- Dense 27B 4-seq bench: 25.35/25.36/25.23/25.36 t/s (~25.32) —
  recovers the full max-ilp dense number (25.31-25.34), +6.8% vs
  no-max 23.70, as predicted (dense decode = M=1 = twin).
- Unit tests: 12 passed / 1 failed (test_auto_awq Qwen2-1.5B —
  pre-existing: the custom FA launcher supports head_dim 128/256
  only, Qwen2-1.5B is head_dim=64; FA suite 15/15 on this build).
- 3-arm serving A/B (3 repeats x 3 prompts, VRAM-safe runner):

| arm      | no-max   | full max-ilp | split build |
|----------|----------|--------------|-------------|
| baseline | 26.44    | 28.56        | **27.99**   |
| ngram3   | 28.92 (1.094x) | 27.80 (0.973x) | **28.03 (1.002x)** |
| mtp2     | 39.74 (1.503x) | 36.67 (1.284x) | **39.37 (1.407x)** |

  mtp2 acceptance 90.64% / 1.813 tok-step (no-max: 90.95% / 1.819).
  CIs: split mtp2 [38.19, ...] overlaps no-max 39.74 (CI 38.58);
  ngram3 no-max CI lower 28.34 vs split 28.03 (marginal no-max win).

**Verdict: split build is the branch default.**
- baseline (production, no spec): +5.9% vs no-max (27.99 vs 26.44),
  2% below full max (28.56) — M=1 gets the max-ilp win.
- mtp2 (recommended spec method): 39.37 ~= no-max 39.74 (CI overlap,
  -0.9%), +7.3% vs full max. The spec workload stays at its best
  level while the plain serving path gets the M=1 gain.
- ngram3: 28.03, -3% vs no-max (28.92) — ngram is the weak spec
  method anyway (1.002x); the delta tracks the down_proj M=4
  microbench outlier (115 vs 105 us, ~1 ms/step), not a dispatch
  problem (M=4 verified on the unflagged kernel).
- full max-ilp remains the wrong default (mtp2 -7.8%).

**Open (documented, not chased):** down_proj M=4 115 vs 105 us on the
split build (same unflagged `<true,4>` kernel binary as the no-max
build; suspected module-level I-cache/constant-memory effect from the
twin's presence, or environmental; serving-level impact <=3%, CIs
overlap — revisit only if the spec arms regress further).

## Final branch review: m4 WARPS==1 shfl-gather bug (fixed)

Critical pre-merge review of the branch found a **silent wrong-numerics
bug** in `dense_gemv_m_kernel`'s CAS epilogue (WARPS==1, ksplit>1 path),
latent for Qwen3.5-27B (all K's are 1024-multiples → kchunk=1024 →
WARPS=2 smem path; the kchunk=512 path only fires for K=1536/2560/3584
models):

- Templated kernel gathered the butterfly sums with
  `s[i] = __shfl(acc[i/M][i%M], i)` inside `if (t == 0)`.
- On ROCm 7.14 clang this lowers to
  `ds_bpermute_b32 vdst, slane, vdata offset:4/8/12` — the offset
  encoding reads `addr(vdata)+offset` from the source lane's file, which
  is only correct for a specific consecutive-register layout the register
  allocator does not guarantee. Result: s[0] (self-read, offset 0)
  correct, s[1..NV-1] wrong. Test: M=2 → 74.5% mismatched elements,
  even/odd column split matching the half2 CAS packing.
- The rt kernel had the same path with the earlier uniform-expression
  form (`__shfl(acc_flat[0], i)`) — fixed during review to read lane 0's
  own registers; the templated kernel is now fixed the same way
  (butterfly is a full-wavefront broadcast, verified by the ksplit==1
  direct-store tests passing and by a standalone probe showing
  post-butterfly "own acc" = full sum).
- The divergent cross-lane `__shfl` gather is the only unsafe form;
  offset-0 forms (butterfly, uniform broadcast, e.g. the FA gather
  kernel's `__shfl(x, 0, 64)`) are unaffected and empirically verified
  (FA suite 15/15, serving A/Bs).
- Regression test: `test_rocm_unquantized_gemm_spec_gemv_m4_kc512_ksplit2`
  (m=2,3,4; x m×1024, w 1024×1024 fp16, kchunk=512 → WARPS=1 ksplit=2),
  compared to F.linear. Full file: 31 passed / 2 skipped.
- A minimal standalone repro of the pattern misleads (plain clang -x hip
  spills the accumulators to scratch and shuffles lower to flat loads —
  a different artifact); the fix was validated through the in-tree test
  against the actual CMake build.

## Parallel 2-request serving A/B (dense 27B, split build)

`benchmarks/kernels/gfx906/spec_parallel_dense.py` — staggered 2-request
(A at t=0, B at t=2s), 512 output tokens, 3 repeats, streaming TTFT.
Prompts: 797 / 621 tokens. Prefix cache does not help (both arms
re-prefill ~1418 tok/window).

| clean reps 1-2 | baseline | mtp2 |
|---|---|---|
| A decode (resident) | 22.6 t/s | **26.8 (+19%)** |
| B decode | 25.2 t/s | **32.1 (+27%)** |
| aggregate | 40.4 t/s | **42.7 (+5.7%)** |
| B TTFT (prefill under load) | 2.51 s | **3.76 s (+1.24 s)** |
| A TTFT (prefill alone) | 0.14 s | **3.15 s** |
| token identity | — | identical across arms |

Per-request decode uplift is real (+19-27%); aggregate nets only +5.7%
(M=6 verify steps + drafter forwards). Prefill cost of enabling MTP
under concurrent load: B's TTFT +50%.

**Open item (not chased):** mtp2 A-TTFT is 3.15 s with sd 0.007 across
reps — a deterministic ~3.0 s per-sequence first-token cost on top of
the 797-token prefill, present even with warm cache and no JIT
(warmup JIT of `precopy_mamba_align_fused_kernel` /
`postprocess_mamba_fused_kernel` + 3 eagle metadata kernels only
explains rep 0 per jit_monitor). Needs a dedicated per-chunk probe to
root-cause (suspect mamba-state path on the prefill→decode boundary
running non-graphed).

## Environment note: standalone HIP programs fail on physical GPU #0

2026-08-19: freshly built standalone HIP binaries (even an empty
`<<<1,64>>>` kernel) fail on launch with hipErrorIllegalState (401) on
physical GPU #0 (HIP_VISIBLE_DEVICES=0) but work on #1; torch/vLLM work
on both. Both cards are identical 0x66a0 gfx906 32 GiB (the old
"GPU #1 = RDNA2 12GB" note is stale). Not chased; use .venv python for
GPU probes on this machine.

## 2026-08-19 — Prefill/TTFT A/B: MTP prefill cost, cache behavior, one OOM bug

Question set: prefill tok/s with/without MTP drafting; MTP's impact on
prefill; root-cause of the 3.15 s mtp2 A-TTFT open item above.

**Design** (`benchmarks/kernels/gfx906/prefill_ttft_probe.py`, new):
sequential max_tokens=1 requests, TTFT + derived prefill rate. Warmup
phase on a distinct prompt (PROMPTS[2]) absorbs per-server one-time
costs; measured prompts A=795 tok, B=619 tok (both < one attention
block), L=1631 tok (two blocks; built from PROMPTS[0] + extra messages,
so L shares A's prefix).

**Block sizes** (engine log, `mamba_cache_mode=align` auto-selected for
both arms): baseline attention block = 784, mtp2 = 800 (MTP lookahead
shifts the page-size fit). Cacheable prefix = `round_down(len, block)`.

**Results** (warm, split build, dense 27B, sequential):

| req (tok)     | baseline            | mtp2                     |
|---------------|---------------------|--------------------------|
| A fresh (795) | 3.176 s (250 t/s)   | 3.141 s (253 t/s)        |
| A rep1/2      | 0.142/0.141 s (784 hit) | 3.140/3.147 s (sub-block: 0 cacheable) |
| B fresh (619) | 2.404 s (257 t/s)   | 2.484 s (249 t/s)        |
| L fresh (1631)| 3.547 s (784 hit via A's prefix + 847 prefill) | 6.410 s (true fresh) |
| L rep1        | 0.455 s (1568 hit)  | 6.415 s (2nd observation; state cached at completion) |
| L rep2        | 0.454 s (1568 hit)  | 3.257 s (800-tok hit)    |

**Findings**

1. **MTP solo-prefill cost: negligible.** 249–254 (mtp2) vs 250–258
   (baseline) tok/s, within run-to-run noise.
2. **The 3.15 s mtp2 A-TTFT (open item) is RESOLVED: block alignment,
   not a per-sequence MTP cost.** A (795) is sub-block under mtp2
   (`round_down(795, 800) = 0`) → full re-prefill every rep,
   795/253 = 3.14 s, sd 0.007. Under baseline (block 784) the same
   prompt hits 784 tokens → 0.14 s. Not a bug.
3. **MTP + align-mode prefix caching works.** Upstream #33705, #33726,
   #37898 (Marconi), #47782 verified in-tree. Repeated 1631-tok prompt
   under mtp2 shows the Marconi admission pattern exactly: 1st full
   prefill, 2nd full prefill (shared prefix detected, mamba state
   cached at completion), 3rd+ recurring hit.
   **Subtlety for upstream:** only 800 of the 1600 aligned tokens hit
   under MTP (baseline: 1568 of 1631, from rep1). Candidate:
   `num_reprefillable_tokens` finalization in
   `HybridKVCacheCoordinator.cache_blocks` interacting with Marconi's
   single cached state. Not chased.
4. **One-time effects, characterized:** first prefill after a rebuild
   pays ~10 s Triton JIT (mamba-align + eagle kernels, per jit_monitor);
   the on-disk cache shrinks that to ~1 s residual. Warmup phase makes
   all measured requests one-time-free. B (619, sub-block both arms)
   never caches in either arm — consistent.
5. **BUG (ours + upstream): serving baseline OOM at 0.95 with a warm
   inductor cache.** 2nd request (795 → 784-tok chunk) dies with
   `aten::empty` 356 MiB (free: 0) inside the inductor piecewise graph
   (`piecewise_backend.py:198 compiled_graph_wrapper`). 3/3
   reproductions on a monitored-clean GPU (5 s VRAM+KFD sampler: only
   our processes). Correlation: cold inductor cache → profiled KV
   7.54 GiB → OK; warm cache → KV 7.70 GiB → OOM. Warm-cache init peak
   is ~0.16 GiB lower (autotune/combo-benchmark workspaces cached
   away), so the KV pool is sized too large for a runtime allocation.
   mtp2 arm at 0.95 is immune (KV 6.39 GiB — MTP weights + spec pool
   leave headroom). **Workaround: `--gpu-memory-utilization 0.93`**
   (validated: full baseline arm clean). Upstream report candidate.
6. **External docker-crash report (other agent):** "default config
   crashes at engine init in `gather_paged_kv_quant_kernel`" — separate
   issue from the OOM above (my OOMs are explained without GPU
   collision; default config = `GFX906_FA_LEGACY=1` + prefix caching =
   our production config, 6+ clean local inits same day). His
   workaround (`GFX906_FA_LEGACY=0` + `--no-enable-prefix-caching`) is
   our documented experimental path. Need his timestamps + full error
   text to sort collision vs env-specific kernel bug.

**Serving recipe update:** dense 27B serving servers need
`--gpu-memory-utilization 0.93` (warm inductor cache) — see item 5.

## 2026-08-22 — TP=2 mtp2 engine-cadence overhead root-caused (the "regression" vs S5/S8 records)

**VERDICT:** OPEN (mechanism characterized; no code fix yet) · **GATE:**
TP=2 mtp2 chrome trace (`/tmp/mtp2_trace/`, `--profiler-config` +
`/start_profile`) + same-harness TP=1 in-process matrix.

Origin: the N4 re-baseline (DEVLOG-masked-fa post-commit 3) found mtp2
TP=2 steady 24.9 t/s vs the S8 record 39.9 (same recipe). Read-only
diagnosis in `fa-masked-mtp-regression-glm5.md` (folds the ds4 + qwen
reviews; all claims adjudicated) + this session's GPU work:

- **TP=1 is healthy:** record-config rerun (2816, agentic, in-process)
  49.8 t/s @ 2.72 acc = 54.6 ms/step; fox matrix 62.6/62.9 ms/step at
  maxlen 2816/32768 (maxlen exonerated; the earlier "+36 ms TP=1-common"
  was acceptance-confounded). Plain TP=1 unchanged vs record era
  (40.4 vs 39.5 ms). M=1-vs-M=3 kernel tables show the mtp2-only GPU
  work is the drafter's two fp16 forwards + the 1.45× M=3 GEMM premium —
  no pathology.
- **TP=2 GPU side also healthy:** trace shows gptq-M3 17.6 ms/step
  (TP=1's 33.6 halved ✓), rccl 8.5 (130 collective kernels/step), FA
  2.7, gather 0.35; GPU busy 96% of the kernel window; the rocBLAS
  MT-monsters are prefill chunks.
- **The pathology is engine cadence:** client/engine 144 ms/step vs
  ~45-50 ms of GPU work. Worker spends ~57 ms/step blocked in
  `hipEventSynchronize` waiting for the next step's inputs (3813 ms /
  272 calls / 71 steps, dedicated thread); visible spec CPU on the
  worker ≈ 8.5 ms/step (`propose_draft_token_ids` 6.6 incl. ~1 ms Triton
  re-specialization checks, spec metadata 1.1, rejection 0.8); the rest
  (~30-40 ms) is EngineCore-side scheduler bookkeeping (untraced
  process). Plain decode's lighter bookkeeping fits inside its 24.5 ms
  cadence — that's why only mtp2×TP=2 suffers.
- **`--async-scheduling` A/B: NO EFFECT** (24.93 vs 24.90 steady;
  confirmed engaged) — the chain is a per-step GPU→CPU→GPU
  **data dependency**, not schedulable idle.
- **Record→now delta:** still unexplained by any diffable code (engine
  code byte-identical; async default identical; capture topology
  identical; drafter eager in both). H1 stands: all S5-S8 TP=2 numbers
  came from one never-rebuilt dirty Aug-19 binary (state unrecoverable).
  The S5-S8 mtp2/mtp3 records keep their asterisk; the clean-rebuild
  A/B of `69f615b98` remains the confirmation experiment.
- **Platform:** the mtp1 TP=2 arm could not boot (3×
  `hipErrorLaunchFailure` at weight load — the S4-residual wedge; BACO
  reset needs root). mtp1 scaling point unmeasured.

Refrigerated: `--stream-interval` batched-output test for the
EngineCore-side share; propose-path CPU trimming (the 6.6 ms/step);
Triton specialization caching; EngineCore-side trace (untraced process
needs its own profiler attach); the clean-rebuild confirmation.

Cross-refs: `fa-masked-mtp-regression-glm5.md` (full evidence +
three-review adjudication), `docs/gfx906/fa-masked-mtp-regression-qwen.md`,
`fa-masked-mtp-regression-ds4.md`, DEVLOG-masked-fa post-commit 3,
DEVLOG-tp2-dense S5-S8.

### 2026-08-22 (post-reboot correction) — the "TP=2 mtp2 regression" was HOST-STATE DEGRADATION

**VERDICT: RETRACTED (not a code regression).** A host reboot at 12:39
restored mtp2 TP=2 serving to **74.9 t/s steady (40.1 ms/step,
acceptance 3.00, usage-based client)** — 3× the same-boot-prior 24.9,
on identical binary/config/harness. The build now beats the S8 record
(62.4 ms/step, which paid the N4 gather tax): healthy-host matrix
(131k, TP=2, P1): plain 40.9 · mtp1 61.9 (@2.00, 33 ms/step) ·
mtp2 74.9 · mtp3 88.1 (@4.00, 45.4) t/s — spec decode beats plain
1.83–2.15× on TP=2 again. The earlier entry's trace analysis described
the degraded host (worker `hipEventSynchronize` stalls ~57 ms/step =
the degradation signature, not engine overhead); async-scheduling /
stream-interval no-ops were no-ops for that reason. The clean-rebuild
confirmation of `69f615b98` is cancelled (unnecessary).

Two operational lessons, now protocol: (a) SSE-chunk counting
under-reports spec-decode t/s by ~acceptance (chunks ≈ steps) — use
`stream_options.include_usage`; (b) **degradation canary**: after GPU
wedge/reset bursts, run the 60-s TP=1 mtp2 probe in
`docs/gfx906/degradation_details.md` before recording any spec number —
slow canary ⇒ reboot. All wedges/degradation get timestamped rows in
`docs/gfx906/degradation.md` (49 half-wedges + 3 full-wedges +
1 degradation event logged 2026-08-21→22; the 14:06 full-wedge killed
the 4-arm mtp2 re-baseline mid-run — P0/P1 tax arms need one clean
re-run; the P1 74.9 stands from the stream-interval boots).
