# More model families on gfx906 — open roadmap

Copyright Kevin Read <me@kevin-read.com>

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

**Status: onboarded + window-FA shipped (2026-08-27, `feat/muse-glimmer`,
not yet in main); LEGACY=0 (direct-paged) validated, LEGACY=1 remains the
serving default.** Onboarding and all gate numbers: `DEVLOG-muse-glimmer.md`.
Knobs: `README.md` table. The independent review
`muse_glimmer_opt2_code_rev_qwen.md` (repo root, untracked) is kept until
M3/M4 below close its open findings; its siblings
(`muse_glimmer_opt2_code_rev_claude.md`,
`docs/gfx906/muse_glimmer_opt2_code_rev_ds4.md`) are kept too — their
residuals are M1/M2.

### M1 — window clip on the gather path (B=1 decode)

The Phase C clip fires only on the direct-paged (B≥2) dispatch; B=1
decode (the gather path — the model's B=1 hot path) and all prefill
scan the full KV. The gather path materializes `[B, Hkv, L, bpr]` K/V
in HBM before the FA kernel, so this is a gather-side change (per-row
gather start), not just a kernel loop start. Gate: bit-identity unit
test + pp8192/B=1 A/B; the FA share of the B=1 step is large enough
that the kernel micro-bench's −48% should show up e2e.

### M2 — per-row (2D) prefill clip

Prefill rows need only `[q_abs+1-W, q_abs]`; today the per-sequence
`k0_base` covers the whole q-tile, so early rows in a prefill chunk
still scan (and the gather materializes) out-of-window keys. A
conservative per-q-tile start (smallest row's window start) is
implementable without per-row loops. Gate: bit-identity + pp4096
prefill/TTFT A/B.

### M3 — kernel hygiene batch (one rebuild)

From `muse_glimmer_opt2_code_rev_qwen.md`, bundle into one build/test
cycle:
- **#8**: device-side `k0_base = max(0, kv_start[sequence])` clamp in
  `fattn-q8-paged.cuh` — a negative start would walk the k-loop into
  pages before token 0 (illegal access / wedge, not a wrong number).
- **#10**: overflow-free cutoff `q_abs_row - k_pos_abs >= window` at
  all four LOCKSTEP sites (the current `k < q_abs_row - window + 1`
  overflows for absurd windows → UB + silently disabled mask). Must
  preserve the unaligned bit-identity tests.
- **#4b/c**: allocate only the meta buffer actually used (`o_meta` is
  dead when `kv_split > 1`); note the `o_part`/`o_meta_split` per-call
  cliff at `_DIRECT_PAGED_MAX_SQ=16` (~35 MiB/layer) — shared arena
  only if a real workload hits Sq>2 direct-paged.
- Test hardening: amplified-V window-boundary case (probe-B trick,
  ~400× discriminative, 3 lines).

### M4 — long-context split-K accuracy point (qwen #4a)

Direct-paged split-K stores unscaled fp16 partials per split; the
partial magnitude grows with keys-per-slice × |V|. Tests cover
L ≤ 512 (plus the 4353/2048 clip case). One accuracy point at
L=16k–32k, split 8 vs split 1 vs fp32 torch ref (cheap, no server)
closes the claim that the default `clamp(16/B,2,8)` is safe at
long context.

### M5 — default read-path decision after a bake

LEGACY=0 (Q8-aliased direct-paged read) is validated (46/46 suite,
default-config + prefix-cache smokes, clip +3.6% / KVSPLIT +1.8%
gates) but stays experimental: B=1 still runs gather, and the
default LEGACY=1 records predate it. Flip the default only after a
serving bake on the target workload. Gate: B=1 + B=4 A/B with the
degradation canary green.
