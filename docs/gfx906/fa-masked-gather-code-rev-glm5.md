# Critical code review — `gfx906/fa-masked-gather` (persistent live-bounded gather+quantize)

Copyright Kevin Read <me@kevin-read.com>

Reviewer: GLM-5 (agent), 2026-08-22. Scope: the code-carrying commits of
the branch — `49d4876e72` (kernel + launcher + pybind + dispatch),
`907ec900f3` (gate probes + suite test) — plus everything that landed
during the review window (`ddd2adbdeb` dispatch guard + tests + probe
width fix, `c9cf0dec17` qwen review + devlog addendum) and the fixes from
this review session itself. **This revision folds in and adjudicates every
claim of the two sibling reviews** (`fa-masked-gather-code-rev-ds4.md`,
`docs/gfx906/fa-masked-gather-code-rev-qwen.md`): each is marked
VALIDATED / PARTLY / REJECTED below. Sources checked line-by-line:
`csrc/gfx906_fa/gfx906_fa_gather.cu` (persistent kernel 530–704, fused V1
404–499, V2 206+), `csrc/gfx906_fa/gfx906_fa.cpp` (wrapper 880–987, knobs
92–114), `csrc/gfx906_fa/kernel/fattn-q8.cuh` (tile loaders 128–200,
284–330; K-loop bound 576–641), `csrc/gfx906_fa/kernel/q8_0_quantize.cuh`,
`vllm/gfx906_fa/gfx906_fa_paged.py`, `vllm/gfx906_fa/gfx906_fa_backend.py`,
`vllm/config/compilation.py`, `vllm/engine/arg_utils.py`, the four probes,
`test_gfx906_fa.py`, `plan_masked_fa.md`, `DEVLOG-masked-fa.md`. I also
**ran** things (below), not just read them.

---

## Verdict

**SHIP — with the blocking defect already fixed on-branch (`ddd2adbdeb`)
and this session's hardening applied.** The fix itself is sound: the
diagnosis is code-verified, the persistent kernel's design (fixed
capture-time grid, live-`seq_lens`-bounded work, in-kernel q8_0) is the
right shape for the problem, the kernel math checks out, and the gate
evidence (NaN-tail → standalone bit-exact → capture/replay → PPL →
serving A/B) is real and internally consistent. The one blocking defect
(reproduced end-to-end during this review) was the unguarded `num_seqs >
16` throw on the default path; it is fixed, tested, and devlog-recorded.
Remaining items are follow-ups (mtp2 in-model A/B, D=128 probe, prefix
widen to B=32), not merge blockers.

---

## What I verified and it holds (keep)

- **The load-bearing mechanism claim is true in source.** The dense FA
  tile loaders never issue a global read at or beyond `i_sup` (= live
  `kv_max`): V loader selects a register `zero` instead of the source
  pointer (`fattn-q8.cuh:161`), the K_q8 loader `break`s per row
  (`:189`), the paged variants do the same (`:257, :318`), and the K-loop
  trip count is `k_VKQ_sup = k_VKQ_max - k_VKQ_0` (`:593`). So not
  writing gather rows ≥ `seq_len` is safe by mechanism, and the NaN-tail
  probe confirms it empirically — now also at `Sq > 1` (see F3).
- **Kernel math is correct.** Row→(seq,head,tok) mapping is a strict
  partition (per-seq counts prefix-summed uniformly per lane; one row =
  one workgroup; no cross-WG overlap → deterministic, matching the
  bit-exact gates). `sl` clamped to `Sk`; margin clamped to `Sk − sl`;
  `int` widths fine; `uint4` V stores 16B-aligned for all D % 32 == 0
  (D∈{64,128,256} all verified in layout math); K writes byte-wise
  (34-byte-block safe); `__shfl(...,0,64)` legal on wave64. `rph[s]=0`
  can never be selected (selection implies `rem < Hkv·rph[s]`, i.e.
  `rph[s] ≥ 1`) — no div-by-zero. `sl=0` sequences emit only margin
  V-zero rows and never read `block_table`.
- **Bit-equality to the old paths is structural**: same
  `quantize_block_q8_0_halfwarp` helper, per-row/per-block independence.
- **Capture-safety is by construction**: both `GFX906_FA_PERSIST_*`
  knobs are process-start `static`s; all live quantities are read from
  device tensors inside the kernel.
- **I ran the code** (this session): suite **21/21**; B=17 crash repro
  pre-fix and fallthrough post-fix through `forward_paged`;
  `fa_capture_replay_probe.py` **16/16 PASS** (re-run twice — before and
  after the full-width table fix); `fa_nantail_probe.py` **11/11 PASS**
  (incl. the four new Sq>1 cases); `_DOUBLE_CHECK` smoke OK.

---

## Findings (mine), with current status

### F1 [was BLOCKING — FIXED in `ddd2adbdeb`, validated]: `num_seqs <= 16` throw on the default path

Committed dispatch (`49d4876e72`) called
`gather_paged_kv_quant_persistent` whenever `_PERSISTENT` (default ON),
and the wrapper `TORCH_CHECK(num_seqs <= 16)` (`gfx906_fa.cpp:928`) —
**reproduced**: `B=17 forward_paged CRASH: RuntimeError: num_seqs must be
<= 16, got 17` (old fused path handles B=17 fine). Blast radius confirmed
in-tree: default capture sizes `[1,2,4]+range(8,256,8)+range(256,512,16)`
(`vllm/config/compilation.py:701`) and default `max_num_seqs=1024`
(`arg_utils.py`) → engine-start crash for any default config; house MoE
recipe (`BENCH_MAX_SEQS=32`) likewise. **Fix validated**: guard
`if _PERSISTENT and num_seqs <= _PERSIST_MAX_SEQS` + B=17 dispatch
regression test + B=16 ragged bit-equal test; I re-ran B=17 through
`forward_paged` — returns via the fused path, suite 21/21. Devlog
addendum records it. Residual (accepted): B>16 batches pay the old
Sk-frozen tax at 131k/262k; prefix widen to B=32 is the refrigerated
follow-up.

### F2 [PARTLY pre-fixed, closed this session]: capture/replay probe coverage + dead constant

Pre-`ddd2adbdeb`, the B=2..4 tables covered only 65 280 tokens/seq, so
high-`sl` "bit-equal" points compared identical skipped/aliased rows.
`ddd2adbdeb` sized `nb` so no rows are skipped — but its fill
(`bt[s_, :off-16]`) still left the **last 16 blocks/seq aliased to phys
block 0** through the zero-filled table tail (deterministic, identical
across paths — gate validity intact, but "fully materialized" was
overstated). **Fixed this session**: full-width fill + dead `REPLAY_SLS`
constant removed (the devlog-recorded sweep `{32,1536,2177,8192,…}` never
matched the committed loop `{128,1536,262112,262144}` — the dead constant
was the residue of that drift). Re-run: 16/16 PASS with every row from
its own physical blocks.

### F3 [closed this session]: NaN-tail gate covered only Sq=1 decode

The gate licensing tail-write removal never ran at `Sq > 1`, where the
persistent kernel also serves (prefill inline-causal, mtp2 spec-decode
verify — different tile geometry: ncols1=64/4, causal boundaries inside
the last tail tile). **Fixed this session**: probe now takes `sq` +
`q_abs_offset = sl − sq`; added mtp2 (sq=3), prefill chunk (sq=64,
ncols1=64), worst-case tail (sl=Sk−1, sq=2), GQA-packed mtp2 B=2 ragged.
**11/11 PASS** — tail-write removal is now gated on all query-tile
geometries the kernel serves. (The in-model mtp2 *serving* A/B remains a
separate follow-up — qwen P1-1, agreed.)

### F4 [fixed this session]: `GFX906_FA_FUSED_QUANT=0` dead under PERSIST

With `_PERSISTENT` winning the dispatch, the documented "reverts to the
two-kernel path" knob was silently ignored (two-kernel reachable only via
`PERSIST=0` or B>16). Documented at the knob (precedence
PERSIST > FUSED_QUANT > two-kernel). (= ds4 F2 / qwen P2-1.)

### F5 [fixed this session]: stale margin comment

`gfx906_fa.cpp` said "Set to 0 once the NaN-tail gate passes" — the gate
passed and the recorded decision is to **keep 128**. Comment now states
the decision; note 128 is exactly the max `nbatch_fa` for D=256
(config table: ncols=2 → 128), so the value itself is right.

### F6 [fixed this session]: `tp_decode_investigation.md` contradicted the branch outcome

The uncommitted 321-line rewrite left the RESOLUTION verdict "OPEN fix
(not built)" and the bounded-capture/multi-tier proposals reading as the
live recommendation, while this branch built and gated the superseding
kernel. Now bannered **SUPERSEDED → DEVLOG-masked-fa.md** per house
convention (analysis retained as history).

### F7 [open, refrigerated]: no small-`max_model_len` in-model A/B

All serving gates ran at 131072/262144; the persistent kernel also
replaces fused V1 for Sk ≤ 65535 (B ≤ 16) with kernel-level evidence
only (27.8 µs vs ~40 µs at 3328 — launch-regime evidence). Expected win;
one `_bench_gfx906.py` run at 8k/32k closes it.

### F8 [fixed this session]: no in-model torch reference on the persistent path

(ds4 F3, adopted) `GFX906_FA_DOUBLE_CHECK=1` now cross-checks the
persistent branch against `_gather_kv` + `quantize_q8_0` (per-seq
in-range rows — tails legitimately differ; raises on mismatch).
Smoke-run OK at B=3 ragged.

---

## Cross-review adjudication

### `fa-masked-gather-code-rev-ds4.md` (10 findings)

| Claim | Verdict | Notes |
|---|---|---|
| F1 B≤16 hard cap, no fallback, blocking; recommends dispatcher shunt | **VALIDATED (defect + severity); recommendation PARTLY** | Reproduced B=17 throw; default-config blast radius confirmed. But its suggested mechanism ("shunted to eager inside the dispatcher") is wrong — the landed fix keeps >16 batches *inside* the graph on the fused/two-kernel paths (capturable, no dispatcher change needed), which is simpler and correct. |
| F2 `FUSED_QUANT=0` dead knob | **VALIDATED** | = my F4; fixed by comment. |
| F3 no `_DOUBLE_CHECK` on persistent path | **VALIDATED** | = my F8; fixed. |
| F4 NaN-tail only at D=256; `supports_head_size` = {64,128,256} | **VALIDATED** | Backend line 246 confirms {64,128,256}; gate measured at D=256 only. Its own analysis (margin conservative → safe by construction at D=64/128) is correct; D=128 probe stays a follow-up (qwen P1-2). |
| F5 MARGIN re-read every launch vs GRID read-once asymmetry | **REJECTED** | `get_fa_persist_margin()` is a `static` read-once initializer, same as GRID (`gfx906_fa.cpp:107-114`). The *value* is passed as a kernel arg, but that arg is baked at capture like all launch args — no asymmetry, nothing to fix. |
| F6 margin 128 hardcoded, not derived from a kernel constant | **VALIDATED (as nit)** | True; the stale comment (my F5) made it worse. Kept as conservative default per devlog decision; deriving from `nbatch_fa` is optional polish. |
| F7 suite 2e-2 tolerance vs "matches two-kernel fallback" wording | **VALIDATED (as nit)** | The bit-exact-vs-eager assert carries correctness; tolerated. |
| F8 plan deviations (no `bench_gfx906_fa_gather.py` extension; rocprofv3 outstanding) | **VALIDATED** | Already candid in devlog; "RESOLVED" should not be read as "every plan step ran". |
| F9 style (`hipLaunchKernelGGL` double-paren, jam) | **VALIDATED (as nit)** | hipify-form in a `.cu`; cosmetic. |
| F10 kernel self-consistency (alignment, OOB, uniform control flow, shfl width) | **VALIDATED** | Independently re-derived; all correct (see "Verdict" section). |
| Greedy-hash divergence properly refuted; PPL corpus small | **VALIDATED** | Matches devlog; acceptable per house protocol. |

### `docs/gfx906/fa-masked-gather-code-rev-qwen.md`

| Claim | Verdict | Notes |
|---|---|---|
| P0-1 unguarded B>16 → engine-start crash; masked by `--max-num-seqs 4` | **VALIDATED** | Default capture sizes / `max_num_seqs=1024` confirmed in-tree; my B=17 repro; fix `ddd2adbdeb` present and tested. Its note that 5–16 is also *untested* was addressed by the B=16 ragged test in the same commit. |
| P1-1 mtp2 + PERSIST=ON untested in-model | **VALIDATED (open)** | Production record config (mtp2, 39.7 t/s) not re-run on this branch. My Sq=3 nantail cases close the *kernel-level* spec-decode coverage; the in-model mtp2 A/B before re-baselining S8 remains required — agreed, refrigerated. |
| P1-2 D≠256 untested under default-ON | **VALIDATED (open)** | Kernel is D-generic by construction; no D=128 verification. Follow-up before widening default-ON to other model families. |
| P2-1 `FUSED_QUANT=0` interaction undocumented | **VALIDATED** | = ds4 F2 / my F4; fixed. |
| P2-2 LEGACY=0 direct-paged path untouched; don't claim N4 covers it | **VALIDATED (scope note)** | Confirmed: `forward_paged_direct` still scales with Sk; plan §1 already scoped this out. |
| P2-3 long-live-context in-model perf unmeasured | **VALIDATED** | Already flagged in devlog caveats; traffic math sound. |
| P2-4 perf nits (row-mapping divide, static knobs, grid sweep) | **VALIDATED (accepted)** | 26.7 µs/layer at 262k confirms not divide-bound; grid sweep refrigerated. |
| P2-5 probe coverage; capture-probe width fix applied | **VALIDATED, fix PARTLY** | The `ddd2adbdeb` sizing removed the skip-guard, but the `off-16` fill still aliased the last 16 blocks/seq to phys block 0 (both sides identical → gate valid, "fully materialized" overstated). Completed this session (full-width fill, re-run 16/16). Its "reviewer's own slip" note on test block-table arithmetic is honest and matches the committed tests. |
| "Verified correct" list (margin clamp, rph≥0 selection, OOB guard, D-generality, K bit-eq, V alignment, strides, buffer lifecycle, capture semantics, test hygiene, commit-message traceability) | **VALIDATED** | Independently spot-checked each; all correct. |
| Review header hash `25729a2560` | **NOTE** | Pre-rewrite hash of the N4 docs commit (now `0b38949440`); same content, no issue. |

### Disagreements resolved

- **ds4's fallback mechanism vs the landed fix**: ds4 wanted dispatcher-
  level eager shunting for >16; the landed in-graph fused/two-kernel
  fallthrough is strictly better (no eager cliff, no dispatcher change).
  ds4's F1 severity stands; its remedy is superseded by the actual fix.
- **ds4 F5 (margin per-launch re-read)**: factually wrong — rejected.
- **"One code path at every Sk" (original commit message)**: both
  reviews + mine converge: after `ddd2adbdeb` it holds **for B ≤ 16
  only**; B>16 falls back (old Sk-frozen cost at 131k/262k). The devlog
  addendum now records this; no further action needed.

---

## Evidence-validity notes (devlog vs what I could check)

- Serving A/B numbers are consistent with standalone per-layer deltas
  ×16 FA layers (17.8/35.7 ms predicted vs 20.2/38.4 measured).
- The greedy-hash divergence refutation is sound methodology; PPL
  identical (10.5516) carries the numeric gate. PPL ran `enforce_eager`,
  so graph-replay correctness rests on the capture/replay probes + suite
  test — acceptable layering, now with full-width tables.
- Long-live-context A/B absent but flagged; mechanism kernel-verified.

## Remaining actions (post-fold, ordered) — status after the 2026-08-22 GPU session

1. ~~**mtp2 in-model A/B**~~ — **DONE** (devlog post-commit 3): tax
   removal transfers to mtp2 (P0 tax −24%, P1 residual 0.0%, +51%/+98%
   steady). **New OPEN finding from the same run**: mtp2 on the
   post-merge branch is absolutely regressed (24.9 t/s steady vs 39.9
   pre-merge record; slower than plain greedy 40.9 despite acceptance
   3.0 — ~120 ms/verify step). Pre-existing (P0 shows it), not an N4
   artifact; blocks mtp2 S8 record re-baselining, needs its own
   investigation.
2. ~~**D=128 probe/PPL**~~ — **DONE (kernel level)**: probe matrix +
   nantail + suite test all PASS (bit-equal, 16.7–18.3 µs flat). A D=128
   *model* PPL run remains only if/when a D=128 model is actually
   onboarded.
3. **Register-prefix widen to B=32** — OPEN (refrigerated): kernel
   change; needs the full gate protocol plus a >16-concurrent-decode
   serving workload to A/B, which doesn't exist in the house harnesses
   yet. Deferred deliberately.
4. ~~**Small-`max_model_len` serving A/B**~~ — **DONE, result is a win**:
   8k +1.1%, 32k **+10.5%** (P1 flat 24.75 across 8k→32k; P0 pays −8.6%
   at 32k). No regression at small Sk — the persistent kernel also
   beats fused V1 in-model, not just standalone.
5. Optional: derive `MARGIN` from the tile constant or drop to 0;
   `GFX906_FA_PERSIST_GRID` 512/2048 sweep (refrigerated).
