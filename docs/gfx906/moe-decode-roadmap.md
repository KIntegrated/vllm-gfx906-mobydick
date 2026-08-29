# MoE decode on gfx906 — open roadmap

Completed experiments and shipped work have moved to
[`CHANGELOG.md`](CHANGELOG.md). This document keeps only open, deferred, or
locally landed-but-not-yet-upstream work. The main reference workload is
Qwen3.5-35B-A3B-AWQ on one MI50: 40 MoE layers, E=256, topk=8, hidden=2048,
and W4A16 group-128 experts.

## Current open candidates

### C1 — Fuse the routing pipeline

**Status: open.** `topkGating`, `moe_align_block_size`, and
`count_and_sort_expert_tokens` account for roughly 1 ms per B=1 decode step.
The standalone M=1 top-k specialization was tested and rejected for the
CUDA-graph regime; the remaining candidates are:

- fuse top-k, align, and count into one capture-safe kernel while preserving
  `sorted_token_ids`, `expert_ids`, and `num_tokens_post_padded`; or
- fold top-k into the router GEMV epilogue, then fuse the align/count work.

The gate is a serving A/B, not an isolated kernel number. Microbenchmarks at
M=1/8/32/128 and TCC/occupancy counters should establish whether the cost is
structural before model-path work begins. See `DEVLOG-moe-m1-sprint.md` for
the negative top-k result.

### C2 — Decode-sized routed GEMM and zeroing fold

**Status: partially open, TP=2 M=1 scope.** The gemm2 lane-column re-tile and
the `VLLM_GFX906_MOE_M1` path are local, default-off options. C2-V showed a
positive TP=2 M=1 result (about +1.47% for gemm2); the TP=1 result was neutral.
The NPT=2 gemm1 trial was similarly positive only in the TP=2 M=1 arm
(about +1.23% graph, +1.32% eager). Decide whether either flag should become
default-on after a combined TP=2 A/B and numerics gate. The tested batch arm
was neutral because it takes the unretiled BM>=2 grouped path; that path is
still unmeasured, not rejected.

The remaining experiments are:

- measure and, if useful, re-tile the BM>=2 grouped path for concurrent
  decode;
- build the V1 N-split/direct-store variant (128/256/512 blocks) before
  closing the untested block-count axis; and
- rerun the corrected standalone harness PASS flow before quoting old S5
  microbenchmark numbers.

See `DEVLOG-moe-c2v.md` and `DEVLOG-moe-gemm1-retiling.md`.

### C3 — Fold the two MoE zeroings

**Status: open.** `w1_out.zero_()` and `output.zero_()` cost about 234 us per
step and are required by the current atomic K-split/aliased-workspace design.
If C2 does not replace that design, fold the zeroing operations into the
neighboring routing/activation kernels without changing the
`gemm1 -> activation -> output.zero_() -> gemm2` ordering. The gate is
bit-correctness plus serving A/B; the common-workspace alias must be preserved.

### C4 — Quantize layer-0 routed experts

**Status: open, conditional.** The checkpoint leaves layer 0's routed experts
in fp16, so the unquantized Triton path costs about 414 us per decode step.
A load-time calibration-free W4A16 repack could remove the last routed-expert
Triton dependency and reduce the layer's weight traffic. This is a
quantization-path change, not a kernel-only change: it needs a PPL/coherence
gate and a serving A/B. Do it only if the 70 t/s target remains active.

### C5 — Fuse the shared-expert chain

**Status: open.** The shared expert is already dense fp16. Its w13, activation,
and w2 operations are individually near their measured GEMV/LLMM1 optima;
the remaining opportunity is a single chain kernel that removes two launches
per layer. Expected benefit is roughly 150–250 us after the serving critical-
path discount. The existing shared down-projection GEMV is shipped and is not
a separate open item. See `DEVLOG-moe-m1-sprint.md`.

### C7 — Persistent/cooperative MoE block

**Status: open, high effort.** A cooperative kernel for routing, gemm1,
activation, and gemm2 could remove much of the launch floor and avoid the
current CAS/zeroing structure. First verify HIP cooperative-launch support and
resident-grid capacity on Vega 20. This should follow C1–C3 and requires
serving, correctness, and memory gates.

### C8 — Expert-weight residency measurement

**Status: open, informational.** Measure L2/TCC hit and miss behavior for the
roughly 12 MB of active W4 weights per layer. The result determines whether
C2's target should be based on the HBM floor or on latency/occupancy rather
than on the current kernel's apparent bandwidth.

### C9 — Overlap shared and routed work

**Status: parked.** The shared-expert chain is independent of routed work, so a
multi-stream fork/join might hide part of its cost. vLLM currently captures a
single stream. Revisit with a concurrent/batched decode project where the
overlap window is large enough to measure.

## Other open gfx906 work

These items were parked in this MoE roadmap because they affect the same
serving path but are not MoE-kernel rewrites.

- **N1 — quiet the expected AutoAWQMoEMarlin fallback.** On gfx906 the
  fallback to the custom WNA16 path is intentional. Replace the repeated
  per-layer warnings with one info/debug message on gfx906 while retaining
  warnings on platforms where the fallback is unexpected. Run a behavior
  sanity gate. See `vllm/model_executor/layers/quantization/auto_awq.py`.
- **N2 — B>1 FA direct store.** For real single-token batched decode, store
  directly into `[B,Hq,D]` and remove the remaining per-layer BSHD reshape
  copy. B=1 is already copy-free; this is a separate decode-specialized
  kernel and launcher change.
- **N3 — GDN state-bookkeeping copies.** Attribute and reduce the upstream
  `[3,1,32]` state copies (about 180 us/step eager). The graph serving gate is
  required because this is launch-latency-bound.
- **P2-1(e) — persistent-CTA MoE prefill GEMM.** The earlier prefill effort
  stalled well below the practical dot2 peak. It remains out of the decode
  scope, but is not a closed technical question if prefill becomes a target.

## Model and platform follow-ups

### Unmerged upstream candidates

These changes are implemented or source-confirmed in the fork but are not
upstream vLLM/ROCm merges. Keep them here until an owner performs the required
duplicate-work check, rebases, tests, and submits them.

- **U1 — fastsafetensors GDS fallback.** Catch the non-`RuntimeError` GDS
  failure so unsupported systems fall back instead of killing the engine;
  preserve the successful GDS fast path. Local commit `128e948baf`.
- **U2 — hipify in-source build guard.** Avoid `copytree` onto itself during
  Py3.12 in-source builds. Local commit `225448d93f`.
- **U3 — GemmaRMSNorm fused dispatch.** Preserve Gemma's `(1+w)` algebra in
  the input dtype so the fused RMS norm path remains available. Local commits
  `19c1d41cf5` and `70ec1d0e79`.
- **U4 — compressed-tensors asymmetric W4A16 qzeros repack.** Correct the
  Triton backend's K-first qzeros layout handling. See
  `DEVLOG-ornith-wna16.md`.
- **U5 — asymmetric-W4A16 review hardening.** Share the `g_idx` gate, make
  qzeros repacking fail closed, and centralize the stored-zero-point backend
  capability set. Local commit `d160fb2ad0`; see
  `DEVLOG-ornith-wna16.md`.
- **ROCR-1 — `IPCRecvHandle` EOF spin.** Treat `recvmsg()==0` as peer EOF
  rather than retrying forever in ROCR-Runtime.
- **ROCR-2 — EventPool permanent allocation latch.** Retry event creation
  after a transient `hsaKmtCreateEvent` failure rather than permanently
  forcing userspace polling. These are separate `/local/git/TheRock` changes;
  see `cpu-stuck-threads.md`.

### Open model onboarding

The reusable W4A16 kernel is shape-gated, not model-name-gated. For a new
checkpoint, run a shape spy, microbenchmark every candidate shape, then use a
PPL/greedy gate and graph+eager serving A/B before enabling a dispatch.
`roadmap-more-models.md` contains the model-specific onboarding queue.

## Open questions

- What exact call site accounts for the remaining roughly 158 us/step of
  MoE-adjacent `[1,2048]` copies?
- What is llama.cpp's component-level kernel budget on the same MI50?
- What are the MI50 L2 size and active-expert weight residency behavior?
- Does `topkGating`'s cost come from structure or a hidden memory round trip?
