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

## Next

1. Phase 0 arms (each = server restart + 3-prompt bench, ~15 min):
   ngram_gpu (k3 min2/max5) + draft-equivalence (text SHA vs CPU
   ngram texts), then ngram min_n=1, then suffix (if
   `arctic-inference` installs; pin `num_speculative_tokens=4`).
2. Re-derive ceiling with measured r/a; record go/no-go for Phase 1.
3. Phase 1 spec (kernel contract: grid/strides, align-slot output,
   numerics bar) → implement → unit test (per-boundary state vs fp32
   reference) → draft-step profiler → stop-rule check.
