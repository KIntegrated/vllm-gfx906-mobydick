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

### G1 — decode-graph per-node replay-cost probe

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

The machine has a single ROCm toolchain now (/opt/rocm is the default),
so sourcing it is unnecessary — **confirmed 2026-08-29 (boot N)**: both
prime dense models' TP=2 serving boots, the 74/74 in-process suite, and
the FA micro-bench runs all worked without it. Remaining: remove it from
the ACTIVE recipes: `/local/git/AGENTS.md` (single-card bench recipe),
`running.md` (§0 + build section), `README.md` (bench recipe). Dev logs
and `degradation_details.md` keep their lines (historical record); the
session `canary.sh` sources it — drop there too when next touched.

### N1 — quiet the expected AutoAWQMoEMarlin fallback

On gfx906 the fallback to the custom WNA16 path is intentional. Replace
the repeated per-layer warnings with one info/debug message on gfx906
while retaining warnings on platforms where the fallback is unexpected.
Run a behavior sanity gate. See
`vllm/model_executor/layers/quantization/auto_awq.py`.

## Tier 1 — decode fast path

### C1 — fuse the routing pipeline (~1 ms/step)

**Status: open, IN FLIGHT (`feat/moe-c1-routing-fusion`).**
`topkGating`, `moe_align_block_size`, and `count_and_sort_expert_tokens`
account for roughly 1 ms per B=1 decode step. The standalone M=1 top-k
specialization was tested and rejected for the CUDA-graph regime; the
remaining candidates are:

- fuse top-k, align, and count into one capture-safe kernel while
  preserving `sorted_token_ids`, `expert_ids`, and
  `num_tokens_post_padded`; or
- fold top-k into the router GEMV epilogue, then fuse the align/count
  work.

The gate is a serving A/B, not an isolated kernel number. Microbenchmarks
at M=1/8/32/128 and TCC/occupancy counters should establish whether the
cost is structural before model-path work begins. See
`DEVLOG-moe-m1-sprint.md` for the negative top-k result. Run G1 first or
in parallel — it prices the per-node tax this fusion saves.

### C2 — decode-sized routed GEMM (decide the built wins; finish the axes)

**Status: partially open, TP=2 M=1 scope.** The gemm2 lane-column re-tile
and the `VLLM_GFX906_MOE_M1` path are local, default-off options. C2-V
showed a positive TP=2 M=1 result (about +1.47% for gemm2); the TP=1
result was neutral. The NPT=2 gemm1 trial was similarly positive only in
the TP=2 M=1 arm (about +1.23% graph, +1.32% eager). Decide whether
either flag should become default-on after a combined TP=2 A/B and
numerics gate. The tested batch arm was neutral because it takes the
unretiled BM≥2 grouped path; that path is still unmeasured, not
rejected. Remaining:

- measure and, if useful, re-tile the BM≥2 grouped path for concurrent
  decode;
- build the V1 N-split/direct-store variant (128/256/512 blocks) before
  closing the untested block-count axis; and
- rerun the corrected standalone harness PASS flow before quoting old S5
  microbenchmark numbers.

See `DEVLOG-moe-c2v.md` and `DEVLOG-moe-gemm1-retiling.md`.

- **C8 — expert-weight residency measurement (feeds C2's target
  selection).** Measure L2/TCC hit and miss behavior for the roughly
  12 MB of active W4 weights per layer. The result determines whether
  C2's target should be based on the HBM floor or on latency/occupancy
  rather than on the current kernel's apparent bandwidth.

### C3 — fold the two MoE zeroings (234 µs/step)

`w1_out.zero_()` and `output.zero_()` cost about 234 µs per step and are
required by the current atomic K-split/aliased-workspace design. If C2
does not replace that design, fold the zeroing operations into the
neighboring routing/activation kernels without changing the
`gemm1 -> activation -> output.zero_() -> gemm2` ordering. The gate is
bit-correctness plus serving A/B; the common-workspace alias must be
preserved.

### N3 — GDN state-bookkeeping copies (~180 µs/step)

Attribute and reduce the upstream `[3,1,32]` state copies (about
180 µs/step eager). The graph serving gate is required because this is
launch-latency-bound.

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
