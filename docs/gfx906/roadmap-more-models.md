# More model families on gfx906 — open roadmap

Completed onboarding and portability findings are recorded in
[`CHANGELOG.md`](CHANGELOG.md). This file contains only model work that is
still open or blocked. The general rule is that gfx906 dispatch is selected by
weight format and shape, not by model family: verify every shape before adding
a model-specific gate.

## Onboarding procedure

For a new model:

1. Confirm that the model loads and identify its attention/linear-attention
   kernels.
2. Run a shape spy and build a per-step kernel table; do not infer transfer
   from Qwen3.5 numbers.
3. Microbenchmark each candidate GEMV/GEMM shape before changing dispatch.
4. Use a greedy-hash gate for bit-equal changes, or a PPL/coherence gate when
   accumulation order changes.
5. Run graph and eager serving A/Bs, with the default chosen from the A/B.

The portable design notes and hardware constraints are in
`latency-hiding.md`, `lds-layout.md`, `dequant-instructions.md`, and
`README.md`. The active MoE kernel is useful to AWQ W4A16 checkpoints with
compatible group size, layout, and dimensions; BF16, FP8, FP4, MLA, and DSA
paths need separate validation.

## Ling-3.0-tiny (`BailingMoeV3ForCausalLM`)

**Status: open; load and baseline first.** The on-disk checkpoint is an
approximately 7.5B BF16 model that should fit one MI50, but it is not yet
measured on gfx906. It has 24 layers, hidden size 1536, E=128/topk=8 MoE
layers, sigmoid plus `noaux_tc` routing with expert bias, KDA linear attention,
and MLA-style full attention. These properties do not match the Qwen3.5 GDN,
standard GQA, or W4A16 paths.

### L1 — get it running

Load the BF16 checkpoint and verify the `BailingMoeV3` model, KDA linear
attention, and MLA decode path on gfx906. Check for CDNA-only intrinsics or
aiter assumptions before porting anything. Record a greedy probe and PPL
baseline.

### L2 — establish a profile

Collect shape and kernel profiles for the full decode step, including layer-0
dense work, KDA, MLA, routing, and shared expert. Produce a measured budget
before selecting an optimization.

### L3 — BF16 MoE expert GEMM

If the model runs and profiling justifies it, benchmark a new W16A16 grouped
skinny-GEMM family for E=128, hidden=1536, and expert intermediate size 512.
The candidate dimensions are K=1536 and K=512. The existing lane-column,
wave-per-K-slice, single-wave-epilogue design is a starting point, not proof
that the Qwen3.5 W4A16 kernel transfers.

### L4 — routing

The sigmoid/`noaux_tc`/bias configuration is handled by the generic routing
path. Consider an M=1 specialization only if the measured profile shows a
large routing gap; the Qwen3.5 E=256 top-k result is not a sufficient reason.

### L5 — attention

Treat KDA recurrent decode and MLA paged decode as separate workstreams. The
Qwen3.5 GDN and custom FA implementations provide methodology only; they do
not establish correctness or performance for these kernels.

**Stop rule:** if L1 finds a hard gfx906 blocker in MLA or KDA, park Ling and
do not build an expert kernel for an unservable model.

## DeepSeek-V4-Flash

**Status: blocked; no optimization work is open on the current machine.** The
model has 43 layers plus MTP, E=256/topk=6 MoE with FP4 experts, FP8 dense
weights, DSA sparse-indexer attention, compressed KV, and a hidden size of
4096. The expert memory estimate is approximately 140 GB at FP4 (and more at
FP8), beyond the 64 GB available across the two cards. TP=2 is not a reliable
sharding path on this machine and DP would replicate the model.

Revisit only if a smaller variant, a substantially smaller checkpoint, or a
working multi-card sharding path becomes available. At that point, validate
format conversion, the K=4096 W4A16 extension, DSA/MLA attention, and
sqrtsoftplus/topk-6 routing independently.

## Generic AWQ MoE queue

The next compatible AWQ W4A16 MoE checkpoint can benefit from the existing
expert kernel without a new kernel, but it remains an onboarding task rather
than an automatic support claim. Qwen3-30B-A3B-AWQ is a candidate: E=128,
topk=8, hidden=2048, and expert dimensions that may fit the M=1 tile. Its
routing (E=128), dense attention, and shared-expert dimensions still require
shape-specific measurement. Follow the procedure above and the open items in
`moe-decode-roadmap.md`.

## Muse-Glimmer-30B (`MuseGlimmerForCausalLM`) — post-onboarding follow-ups

**Status: onboarded 2026-08-28; follow-ups M0–M3, M5, M6 Parts A/B
and the M3 dot2 refutation are CLOSED (records in `CHANGELOG.md`
2026-08-27–29; evidence in `DEVLOG-muse-glimmer.md` rounds 1–11 and
`DEVLOG-fa-attention.md`).** M6 Part A was **merged 2026-08-29 as
loader hygiene** (`02d197189f`; flip question stays DEAD-END — same-
boot B=1 A/B PASS, slope −4.3…−4.8 %, bit-identical, 74/74). 
`GFX906_FA_LEGACY=1` remains the serving default; LEGACY=0 B≥2 routes
through the fused-Q8 gather (`GFX906_FA_DIRECT_PAGED_Q8=0`, round
10); direct-paged is opt-in. Knobs: `README.md` table.

### M4 — long-context split-K accuracy point (qwen #4a)

Direct-paged split-K stores unscaled fp16 partials per split; the
partial magnitude grows with keys-per-slice × |V|. Tests cover
L ≤ 512 (plus the 4353/2048 clip case). One accuracy point at
L=16k–32k, split 8 vs split 1 vs fp32 torch ref (cheap, no server)
closes the claim that the default `clamp(16/B,2,8)` is safe at
long context.

### M6 — residuals (Parts A and B closed; see CHANGELOG)

- **Part C — Q4-KV via native `v_dot8_i32_i4`: SHELVED 2026-08-28
  (user decision, `5d8d4c7f59`).** Quality unproven — Q4 K *and* Q4 Q
  (or Q8→Q4 requant) accuracy is unvalidated on this model family;
  the 7-level q4_0 codebook roughly doubles the KQ quantization
  error with no measured PPL evidence. Reopens only behind a dedicated
  accuracy gate (PPL probe bands on the 442-token set, Q4-KV vs
  Q8-KV arms) that must pass *before* any kernel work; the measured
  ISA rates motivating it are in `dequant-instructions.md`
  (`v_dot8_i32_i4` 49.6 T MAC/s, 2× dot4 at half the operand bytes).
- **LEGACY-flip adjudication: closed in practice.** Part A's gate
  fired (the B=1 gap is not load-instruction-bound) and no other
  candidate closes the B=1 −2.5…−3.7 % delta, so the flip question
  stays shut unless new evidence appears. The never-run same-boot
  B=1 adjudication (107.2 boot M vs 111.5 boot L) remains available
  if the question ever reopens.

### Housekeeping — drop the legacy `~/env-rocm-7.14-gfx906.sh` sourcing

The machine has a single ROCm toolchain now (/opt/rocm is the default),
so `source ~/env-rocm-7.14-gfx906.sh` (PATH/LD_LIBRARY_PATH to
/opt/rocm) is unnecessary — **confirmed 2026-08-29 (boot N)**: both
prime dense models' TP=2 serving boots, the 74/74 in-process suite,
and the FA micro-bench runs all worked without it. Remaining: remove
it from the ACTIVE recipes: `/local/git/AGENTS.md` (single-card bench
recipe), `docs/gfx906/running.md` (§0 + build section),
`docs/gfx906/README.md` (bench recipe). Dev logs and
degradation_details.md keep their lines (historical record); the
session `canary.sh` sources it — drop there too when next touched.

## Decode-graph node-overhead point (shared infra, from the LEGACY=0 B=1 closure)

The 2026-08-29 same-boot adjudication (`DEVLOG-fa-legacy0-b1-decode.md`,
boot O) left a bounded-but-unexplained **~1.55 ms/step** serving cost
common to both LEGACY=0 arms, with the extra captured-graph nodes
(~16–32 per decode step: one Q8 side-buffer write + slot cast per
full-attn layer) as the leading unmeasured hypothesis. Not further
decomposed there — the obvious tools are blocked on this stack
(chrome-trace GPU timestamps are not wall-aligned; eager TP=2
collapses ~3× from launch overhead). This is NOT LEGACY=0-specific:
any change that adds per-layer kernels to the captured decode graph
(MoE routing fusion, spec-decode extensions, future KV-side writes)
pays the same invisible per-node replay cost, so the measurement
gates a family of decisions, not just the refrigerated lever below.

**G1 — per-node replay-cost probe (cheap, no model change):** capture
the standard decode graph, then A/B replay wall time with N dummy
no-op kernel launches appended per layer (N ∈ {0, 16, 32, 64}), TP=1
and TP=2, same boot. Falsifiable both ways:

- per-node ≤ ~10 µs → 16–32 nodes explain ≤ ~0.3 ms of the 1.55 ms →
  node count is NOT the owner; suspect TP=2 sync placement / other
  LEGACY=0-common per-step work; the Q8-fusion lever stays
  refrigerated.
- per-node ≈ 50–100 µs → nodes explain the remainder → the
  refrigerated lever (fuse the Q8 write into
  `triton_reshape_and_cache_flash`, halving the nodes) reopens with a
  real bound — and every future adds-nodes-per-step proposal must
  budget it.

Gate is wall-clock A/B only (harness or serving). Prior-probability
note: ~50–100 µs/node would be unusually high for graph replay, so G1
is more likely to kill the hypothesis than confirm it — either
outcome is cheap and decisive.
