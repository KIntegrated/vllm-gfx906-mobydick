# gfx906 refrigerator — parked work with reopen gates

The index of shelved/parked/deferred items: not active targets, but
cheap to reopen if their gate fires. Symmetric with `DEAD-ENDS.md`
(closed negatives) and `CHANGELOG.md` (closed positives). Every entry
states **why parked** and the **reopen gate** — an item leaves this file
only by meeting its gate (or by a user decision). Dev-log-level
refrigerated levers stay in their `DEVLOG-*.md` residue sections; this
file indexes roadmap-level items only (see Cross-references).

## DeepSeek-V4-Flash

**Parked: hardware-blocked, not an active target (user decision
2026-08-29 — others can test it for us).** 43 layers plus MTP,
E=256/topk=6 MoE with FP4 experts, FP8 dense weights, DSA sparse-indexer
attention, compressed KV, hidden size 4096. Expert memory ≈ 140 GB at
FP4 (more at FP8) vs 64 GB across the two cards; TP=2 is not a reliable
sharding path on this machine and DP would replicate the model.
**Reopen gate:** a smaller variant, a substantially smaller checkpoint,
or a working multi-card sharding path. If reopened, validate format
conversion, the K=4096 W4A16 extension, DSA/MLA attention, and
sqrtsoftplus/topk-6 routing independently.

## M6 Part C — Q4-KV via native `v_dot8_i32_i4`

**Parked: user decision 2026-08-28 (`5d8d4c7f59`).** Quality unproven —
Q4 K *and* Q4 Q (or Q8→Q4 requant) accuracy is unvalidated on this
model family; the 7-level q4_0 codebook roughly doubles the KQ
quantization error with no measured PPL evidence. **Reopen gate:** a
dedicated accuracy gate (PPL probe bands on the 442-token set, Q4-KV vs
Q8-KV arms) that passes *before* any kernel work. The measured ISA rates
motivating it are in `dequant-instructions.md` (`v_dot8_i32_i4`
49.6 T MAC/s, 2× dot4 at half the operand bytes).

## C9 — overlap shared and routed MoE work

**Parked: no overlap window.** The shared-expert chain is independent of
routed work, so a multi-stream fork/join might hide part of its cost,
but vLLM currently captures a single stream. **Reopen gate:** a
concurrent/batched decode project where the overlap window is large
enough to measure.

## P2-1(e) — persistent-CTA MoE prefill GEMM

**Parked: out of decode scope.** The earlier prefill effort stalled well
below the practical dot2 peak. **Reopen gate:** prefill becomes a
performance target.

## Speculative decoding (former spec-decode-roadmap.md)

Shipped recommendation on record: MTP k=2 for Qwen3.5-27B/-35B
(`running.md`, `DEVLOG-spec-decode.md`); Qwen3.8-27B serving uses ngram
n=5 (repo serving defaults). Completed phases are in `CHANGELOG.md`;
detailed evidence in `DEVLOG-spec-decode.md`,
`DEVLOG-gdn-mixed-decode.md`, `DEVLOG-fp16-skinny.md`.

### SD-L2 — AWQ M≤4 draft-step GEMV

**Parked: estimate says small win.** The M=1-to-M=4 AWQ cost is
approximately 17 ms per agentic draft step, but the q_gemm family is
dequant-ALU-bound: M=1 already reads about 44 MB in 75 µs (roughly
590 GB/s), and the existing tiled M=4 kernel shares dequantization
across rows. An exllama-style four-row GEMV or a q_gemm re-tile is
therefore expected to save only the atomics/LDS/M-tiling overhead,
estimated at 2–8 ms. **Reopen gate:** MTP is no longer the preferred
drafter, or a serving profile shows the estimate is materially wrong.
Gates if reopened: per-shape microbenchmark, the gfx906 MoE/GPTQ tests,
a PPL or greedy gate as appropriate, and an agentic serving A/B.

### SD-suffix — suffix draft-quality probe

**Parked: dependency-blocked.** The suffix proposer needs
`arctic-inference==0.1.1` and has dynamic draft length, so it remains
PIECEWISE-only and does not use the uniform speculative-decode graph
rails. Its only useful result would be a draft-quality comparison
against MTP/ngram. **Reopen gate:** a better drafter is needed AND the
dependency can be installed and verified on ROCm.

### SD-gram — GPU n-gram proposer match-selection fix

**Parked: not an adoption candidate.** The GPU proposer produced 0.428
accepted tokens per draft step versus 1.08 for the CPU proposer and
diverged in repeated-match tie breaking. A line-by-line match-selection
fix could make it useful for deployments without an MTP head.
**Reopen gate:** a no-MTP-head deployment need, behind a draft-quality
and serving A/B gate. Keep the CPU proposer as the default until then.

### SD-future — future drafter models (EAGLE etc.)

**Parked: unplanned.** Would use the existing FA/speculative-decode
rails. **Reopen gate:** a checkpoint exists locally AND a target model
AND an acceptance gate AND a memory budget. Do not add implementation
work before all four.

## Cross-references (dev-log refrigerated levers, not restated)

- LEGACY=0 Q8-write fusion into `triton_reshape_and_cache_flash` —
  parked in `DEVLOG-fa-legacy0-b1-decode.md`; gated on ROADMAP G1
  (node-overhead measurement).
