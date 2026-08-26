# gfx906 speculative-decoding roadmap

Copyright Kevin Read <me@kevin-read.com>

Completed speculative-decoding work has moved to
[`CHANGELOG.md`](CHANGELOG.md). This file intentionally contains only work
that is still open or deferred. The current shipped recommendation is MTP k=2
for Qwen3.5-27B and Qwen3.5-35B; see `running.md` and the dev logs for the
measured configurations.

## Open and deferred work

### L2 — AWQ M<=4 draft-step GEMV

**Status: deferred.** The M=1-to-M=4 AWQ cost is approximately 17 ms per
agentic draft step, but the q_gemm family is dequant-ALU-bound: M=1 already
reads about 44 MB in 75 us (roughly 590 GB/s), and the existing tiled M=4
kernel shares dequantization across rows. An exllama-style four-row GEMV or a
q_gemm re-tile is therefore expected to save only the atomics/LDS/M-tiling
overhead, estimated at 2–8 ms.

Reopen only if MTP is no longer the preferred drafter or a serving profile
shows that this estimate is materially wrong. Required gates are a per-shape
microbenchmark, the gfx906 MoE/GPTQ tests, a PPL or greedy gate as appropriate,
and an agentic serving A/B. The original measurements and the reason for the
re-scope are in `DEVLOG-spec-decode.md`.

### Suffix draft-quality probe

**Status: deferred.** The suffix proposer needs `arctic-inference==0.1.1`
and has dynamic draft length, so it remains PIECEWISE-only and does not use the
uniform speculative-decode graph rails. Its only useful result would be a
draft-quality comparison against MTP/ngram. Revisit if a better drafter is
needed and the dependency can be installed and verified on ROCm.

### GPU n-gram proposer match-selection fix

**Status: optional, low priority.** The existing GPU proposer is not an
adoption candidate: it produced 0.428 accepted tokens per draft step versus
1.08 for the CPU proposer and diverged in repeated-match tie breaking. A
line-by-line match-selection fix could make it useful for deployments without
an MTP head, but it is not on the current Qwen3.5 path. Keep the CPU proposer
as the default until a draft-quality and serving A/B gate passes. The rejected
experiment is recorded in `DEVLOG-spec-decode.md`.

### Future drafter models

**Status: unplanned.** EAGLE or another external draft model would use the
existing FA/speculative-decode rails, but no weights are available locally and
there is no target model or acceptance gate yet. Do not add implementation
work until a checkpoint and a memory budget exist.

## Related records

The completed spec-decode phases, including the GDN attribution, no-draft
capture fix, MTP results, and concurrent-request rails, are summarized in
[`CHANGELOG.md`](CHANGELOG.md). Detailed evidence remains in
`DEVLOG-spec-decode.md`, `DEVLOG-gdn-mixed-decode.md`, and
`DEVLOG-fp16-skinny.md`.
