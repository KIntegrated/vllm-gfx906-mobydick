# gfx906 dev-log conventions

> This is the gfx906 domain guide. Read this before **writing or updating**
> any `DEVLOG-*.md` / `*-dead-ends*.md` file in `docs/gfx906/`. It is the
> standing rule that keeps the dev logs searchable, verdict-first, and under
> budget. See `editing-agent-instructions.md` before editing *this* file.

## Purpose of a dev log

A dev log records **what was tried and what the verdict was** — experiments,
hypotheses, and outcomes (good and bad) — for a future reader to look up
"did X make sense?" in one pass. It is **not** a scratchpad, not a proposal,
and not an ISA reference.

- Forward-looking material (ideas, plans, candidates, "todos") lives in a
  **roadmap**, not a dev log. Cross-link from log to roadmap, then leave it.
- Measured ISA facts, latency-hiding patterns, LDS layout → keep in the
  kernel-notes files (`latency-hiding.md`, `lds-layout.md`), not the log.
- A dev log may reference `/tmp/*.log` paths by name but the **numbers must
  be written into the log** — `/tmp` is wiped on reboot and is not the record.

## The searchability contract

Every experiment/session entry must be queryable with a single grep, so:
- **Every entry ends with `VERDICT:`** using a fixed vocabulary:
  `DEAD-END` | `SHIPPED` | `NEUTRAL` | `SUPERSEDED` | `OPEN`.
- **Every hypothesis is a `## HYPOTHESIS` section** written falsifiably
  ("If X, then Y") so its verdict maps trivially.
- **Every entry names its `GATE:`** — the one measurement that decides the
  verdict. See the gate rules below.

A `grep 'VERDICT: DEAD-END'` across the topic logs, plus the one-line
`DEAD-ENDS.md` index, is the whole answer to "what didn't work."

## The gate rules (evidence vs verdict)

On gfx906, **serving wall-clock A/B is THE gate** for µs-scale verdicts.
Per-kernel census and standalone-harness numbers are **evidence, not the
gate** — they have repeatedly failed to transfer to serving.

- Label standalone/census numbers explicitly as **launch-regime evidence**.
- Eager A/B can tie even when kernels differ (launch-bound); graph serving
  `_bench_gfx906.py` A/B is authoritative. Keep the exact config (pp/tg,
  samples, `BENCH_MAX_SEQS`, GPU util) with every number.
- **Never** judge a kernel change on perf alone without the serving A/B.

## Entry shape (per experiment/session)

Prefer the template in `docs/gfx906/_devlog-template.md`:

```markdown
# <TOPIC> — one-line claim

**VERDICT:** <fixed vocabulary> · **GATE:** <serving-config>
**DROPPED/REVERTED at commit:** <hash>  (if DEAD-END/SUPERSEDED)

## HYPOTHESIS      — falsifiable one-liner
## What was done   — terse; kernels/toggles touched, configs swept, /tmp log paths
## Evidence FOR    — numbers + the gate each was measured at (label launch-regime)
## Evidence AGAINST— numbers + the gate; the decisive transfer-failure row last
## Why it failed   — mechanism, measured or inferred
## Interactions / superseded-by · Refrigerated residue (near-hit calls, cross-linked not restated)
```

A `DEAD-END`/`SUPERSEDED` verdict must state the **revert** explicitly
(`DROPPED at commit`, "prepared for git-revert" for the vLLM no-busywork
rule). Cheap-but-not-shipped calls go in **Refrigerated residue** so they
aren't re-fancied as new work. When a dead-end changes a roadmap expectation,
also add a one-line revert note to the roadmap candidate and cross-link.

## Grouping & naming

Keep dev logs **combined by topic** (kernel family / model train), file per
major ongoing theme, with the model/branch/date in the file header:

- `DEVLOG-moe-*.md` — the MoE expert-kernel train (Qwen35, M=1 sprint, gemm1 retiling)
- `DEVLOG-fa-attention.md` — the custom Q8 FA / decode backend + fused-gather track
- `DEVLOG-fa-splitk-accuracy.md` — the FA split-K accuracy track (M4 closure;
  split-defaults-vs-fp32-ref pins)
- `DEVLOG-fa-legacy0-b1-decode.md` — the LEGACY=0 B=1 decode-gap track
  (flip adjudication; see also `roadmap-more-models.md` G1)
- `DEVLOG-dense-decode.md` — Qwen3.5-27B dense decode (GEMV, max-ilp, load tests)
- `DEVLOG-spec-decode.md` — speculative decoding (incl. the n-gram dense probe)
- `DEVLOG-gemma4-*.md` — Gemma-4 family (kernel + onboarding/prefill-logprob incident)
- `DEVLOG-qwen*.md` — Qwen-family onboarding / crash notes
- `DEAD-ENDS.md` — one-pass index: hypothesis → gate → verdict → commit → refs

A short incident/crash/anomaly goes in its model/topic log, not a fresh file —
a log that covers only one incident is a weak search target.

## Merge-train rules (stop the logs from regrowing)

1. **Hard size budget: ~20–25 KB per dev log.** When a log exceeds it, run a
   staleness/archive pass instead of appending to the tail: prune old
   superseded numbers, move settled entries to a topic index line, skim prose.
2. **Never let one file become a catch-all.** If a log mixes unrelated
   topics, split by topic; if a log starts accumulating todos/ISA notes, move
   them to the roadmap / kernel-notes where they belong.
3. **Date-ordered `## YYYY-MM-DD` headings, not phase-number nesting.**
   Phase counters (`P3-3a`, `PHASE 3`) grow organically into bloat — a dated
   session heading plus `VERDICT:` keeps history linear and re-sortable.
4. **Verdict precedes details.** Future readers scan verdicts and read
   details only for the tag they need.
5. **Roadmap vs dev-log stays strict** — roadmap says *what we might do*,
   dev log records *what we did and the outcome* — but keep the cross-links
   so each dead-end is reachable from both directions.

## Copyright

New dev logs/files use the project convention:
`Copyright Kevin Read <me@kevin-read.com>`.