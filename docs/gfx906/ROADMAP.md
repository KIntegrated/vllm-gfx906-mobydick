# gfx906 roadmap — open work, priority-ordered

The single active queue. Completed work → [`CHANGELOG.md`](CHANGELOG.md);
parked/reopenable work → [`REFRIGERATOR.md`](REFRIGERATOR.md); closed
negatives → [`DEAD-ENDS.md`](DEAD-ENDS.md). Item IDs (C*, G*, L*, N*, U*,
HK*) are stable across reorganizations — cite them, not filenames.

Reference workload: Qwen3.5-35B-A3B-AWQ on one MI50 — 40 MoE layers,
E=256, topk=8, hidden=2048, W4A16 group-128 experts; B=1 decode step
≈ 15 ms at 66.5 t/s. Priority = expected gain × confidence ÷ effort+risk;
tiers are do-order, sections within a tier are ordered the same way.

## Tier 0 — cheap, decisive, low-risk

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

**Status: stage 1 SHIPPED (unmerged, `feat/moe-c1-routing-fusion`);
stage 2 DEAD-END. Item closed as an active fusion item — the remaining
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
  with router→expert meta plumbing, kept for future kernel-design
  iterations.

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

### C7 — persistent/cooperative MoE block

**Status: open, high effort.** A cooperative kernel for routing, gemm1,
activation, and gemm2 could remove much of the launch floor and avoid
the current CAS/zeroing structure. First verify HIP cooperative-launch
support and resident-grid capacity on Vega 20. Follows C1–C3; requires
serving, correctness, and memory gates.

### C4 — quantize layer-0 routed experts (414 µs/step)

**Status: open — the 70 t/s target is ACTIVE (user decision
2026-08-29).** The checkpoint leaves layer 0's routed experts in fp16, so
the unquantized Triton path costs about 414 µs per decode step. A
load-time calibration-free W4A16 repack could remove the last
routed-expert Triton dependency and reduce the layer's weight traffic.
This is a quantization-path change, not a kernel-only change: it needs a
PPL/coherence gate and a serving A/B.

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
