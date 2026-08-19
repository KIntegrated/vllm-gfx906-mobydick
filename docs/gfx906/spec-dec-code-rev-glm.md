# Critical code review — `gfx906/spec-decode` branch

Reviewed range: `38ceb5d957..68243a61b2` (12 commits, 28 files, ~3800
insertions) plus the uncommitted `spec-decode-roadmap.md` status update.
Scope: production code (`csrc/`, `vllm/`), tests, bench/probe harnesses,
and consistency between the docs' claims and the code.

Overall verdict: **solid, measurement-driven work with good kill switches
and honest negative results recorded.** The kernel logic checks out on
manual verification (reduction indexing, CAS epilogue addressing, tail
guards). The findings below are ranked; none is a release blocker for
this local branch, but items 1–4 should be addressed before anything is
proposed upstream.

---

## 1. Correctness / robustness

### 1.1 Dead tail-guard contradicts the launch invariant (low, cosmetic→confusing)
`dense_gemv_m_kernel` / `_rt`: the ksplit>1 epilogue guards
`if (row0 + RPT - 1 >= N) continue;  // ragged tail`, but the launcher
enforces `TORCH_CHECK(N % rpt == 0)` and grids at `N / RPT`, so a ragged
tail block can never exist. Harmless, but the guard implies a code path
that is unreachable; a future refactor that relaxes the divisibility
check will find this guard silently dropping *valid* partial rows (it
skips the whole CAS for the block instead of masking rows). Either
remove the guard or make it a real per-row mask.

### 1.2 `VLLM_GFX906_GEMVM_RPT=4` can turn a fall-back into a hard abort (medium, env-only)
`_gfx906_spec_gemv_m4` gates on `m % 2 == 0` (RPT=2 default). If an
operator sets `VLLM_GFX906_GEMVM_RPT=4` and a weight with `N % 4 != 0`
(e.g. N=2048? fine, but any N≡2 mod 4) hits the path, the kernel's
`TORCH_CHECK(N % rpt == 0)` throws mid-serving instead of falling back
to triton. Env is documented as a sweep knob, but the failure mode is a
crash. Cheap fix: replicate the rpt env read in `_gfx906_spec_gemv_m4`
and return `None` when `m % rpt != 0`.

### 1.3 ksplit==1 path of the M≤4 kernel has no test (medium)
`test_rocm_unquantized_gemm_spec_gemv_m4_real_kernel` covers ksplit=5
(K=5120/kc=1024) and ksplit=17 (K=17408) — both take the packed-CAS
epilogue. The `ksplit == 1` plain-store epilogue (both WARPS==1 and
WARPS>1 variants, e.g. K=1024 `fc` shape at kc=1024, or K=2048 at
kc=2048 — shapes the roadmap says the dispatch *does* select) is
untested. It's different code (direct store vs CAS, warp-0 store vs
lane-0 gather). Add one ksplit==1 case.

### 1.4 fp16 CAS accumulation: nondeterminism and error growth (accepted risk, worth stating)
The ksplit>1 path accumulates partials into fp16 via 32/64-bit CAS
atomics. Two consequences: (a) run-to-run nondeterminism in draft logits
(atomic order), (b) rounding error grows with ksplit (tests need
atol=0.5 at ksplit=17 vs 0.3 at ksplit=5). This matches the M=1 design
and the A/B showed token-identical output, so it's accepted — but the
roadmap/devlog never states the nondeterminism explicitly. It should,
because "token-identical output" was measured once, not guaranteed.

### 1.5 M=4 templated-kernel regression is unattributed and shipped around (medium, process)
The templated M=4 kernel measured 507→311 GB/s (1.6× slower) and the
response was to keep the runtime-M kernel for M=4 and *leave the slow
templated M=4 instantiation compiled but unreachable*. That's ~300 lines
of dead dispatch (`LAUNCHM_BY_KC(4)` unreachable; the `else` branch
always takes `_RT`). Options: delete the unreachable arm, or keep it but
gate with a comment referencing a filed follow-up. Right now a reader
must reverse-engineer that `LAUNCHM_BY_KC(3)`'s `else` is the M=4 path
into `_RT`. The "cause unattributed" note is honest but this smells like
a register-pressure/occupancy artifact worth one afternoon with
`rocprof` — the fix could recover ~40% on the M=4 LM head.

### 1.6 `forward_paged` uniform-batch fast path (verified OK, one nit)
The `num_tokens == num_seqs * max_seqlen_q` inference is sound (equality
with the max forces uniformity), the permute/copy indexing is correct,
and the output gather matches. Nit: `query.view(...)` assumes a
contiguous `query`; a `.reshape` would be strictly safer. Also this
branch is only reached when `max_seqlen_q > 1`, so no overlap with the
M=1 branch — good. No D2H sync — capture-safe as claimed.

### 1.7 `Gfx906FAMetadata` allowlisted for multi-step drafting on faith (medium)
`llm_base_proposer.py` appends `Gfx906FAMetadata` to the metadata
allowlist with a comment asserting the same shape contract as
`RocmAttentionMetadata` (`num_actual_tokens` / `max_query_len` /
`query_start_loc` / `seq_lens`). There is no test asserting the draft
loop actually consumes those fields correctly for this backend. If the
contract drifts, the symptom would be *wrong draft attention*, i.e.
silently lower acceptance. The MTP k=2 A/B (90.75% acceptance,
token-identical) is strong indirect evidence it works today, but a
cheap field-presence test would pin the contract.

## 2. Integration / config

### 2.1 L5 cg-small fix (verified reasonable; memory cost unstated)
The `compilation.py` change re-adds capture sizes 1..q-1 after
`adjust_cudagraph_sizes_for_spec_decode` rounds them away. Correctness
of the *mechanism* is confirmed by the A/B (ngram3 0.945×→1.077×).
Concerns:
- Each extra capture size is another graph + memory-pool reservation.
  On a 32 GB MI50 with the GDN mamba-state pool this is exactly the
  resource that OOMs (cf. `BENCH_MAX_SEQS=4`). The devlog should record
  the graph-memory delta of the added sizes (q-1 extra graphs).
- Gating is `is_rocm() and on_gfx906()` inside a `not use_v2_model_runner
  and FULL-decode and uniform_decode_query_len > 1` block — i.e. only
  spec-decode MRV1 runs, good. The env default-on (`"1"`) means all
  gfx906 spec users get it; consistent with the kill-switch convention
  elsewhere in the branch.
- The lazy `from vllm.platforms.rocm import on_gfx906` inside
  `CompilationConfig` post-init works but is a layering smell; fine for
  a local branch, would draw review upstream.

### 2.2 FA backend `UNIFORM_BATCH` declaration (OK)
`vllm_config.num_speculative_tokens > 0` is safe (property returns int,
diffusion fallback covered). The claim that the Q8 paged kernel is
q_len-generic and the LEGACY inline-quant path is FULL-capture-safe is
backed by earlier P3-3a work; the new `forward_paged` uniform path is
the piece that makes the declaration true for multi-token queries.
Consistent.

### 2.3 Dispatch placement of `_gfx906_spec_gemv_m4` (OK, one gap)
Placed at the top of the `n <= 16 and bias is None` triton branch, after
the M=1 GEMV paths, correctly excluding the tuned hipBLAS special case
(m==5120, 2048≤k≤2304). Gap: `n` in 5..16 still goes to
`triton_matmul` — fine (W4 is parked, documented). Also note
`_gfx906_spec_gemv_m4` re-reads env and `on_gfx906()` per call; matches
existing style (`_gfx906_gemv_long_k`), negligible cost.

### 2.4 `_gfx906_gemv_long_k` widening (fine, but the guard inverted shape)
The rewrite from one hardcoded shape to `K ∈ {17408, 10240}` with the
`weight.shape[0] == 5120` precondition still hardcoded is correct for
the two measured shapes, and returns None otherwise. The N=5120 check
now silently covers shapes it was never measured for (any [5120, 10240]
fp16 weight) — acceptable given the M=1 kernel is shape-generic, but
the docstring should say the *kernel* is generic and only K was tuned.

## 3. Tests

- Good: dispatch tests are thorough — routing (m4 vs LLMM1 vs hipBLAS
  special case vs kill switch) and the never-off-gfx906 guard are both
  covered with mocks; the fake-tensor registration for torch.compile is
  present.
- Missing: (a) ksplit==1 numeric test (see 1.3); (b) RPT=4 numeric test
  (only the default RPT=2 is exercised; the `pk4` CAS code is untested
  at any ksplit); (c) odd-M (M=3) is parameterized — good — but only at
  ksplit=5.
- Tolerances (atol 0.3–0.5, rtol 2e-2) are loose but honest for fp16
  CAS accumulation; keep them but consider also asserting max-abs-error
  trends in the bench rather than the unit test.

## 4. Bench/probe scripts (`benchmarks/kernels/gfx906/`)

Consistent with repo conventions (correctness via pytest, perf via
benchmarks). `spec_ngram_dense.py` (386 lines) is the heaviest; it
duplicates serving-loop scaffolding that exists in
`docs/gfx906/_bench_gfx906.py` — if it's one-shot it's fine in-tree, but
flag it as probe-grade in the header so it isn't mistaken for a
maintained harness. The `_bench_gfx906.py` `BENCH_MAX_SEQS` knob is a
clean minimal change.

## 5. Docs / claims consistency

- The roadmap correctly records the reversal history (B1 overturned,
  "no MTP head" corrected, L1'' closed as a non-issue after an eager
  dispatch spy disproved the inductor-intercept theory, L5 root-caused
  at kernel level with `batchdesc_probe.py`). This is exemplary
  adversarial self-review.
- One discrepancy: the devlog says L5's "~13 ms no-draft saving
  materialized but the draft-step cost stayed ~56 ms" while the levers
  table predicted ~43 ms with W4-class improvements — the residual is
  attributed to L1'' being a non-issue; the numbers reconcile only
  roughly (~4% model error claimed earlier). Fine for a devlog; just
  don't upstream the cost model as precise.
- The uncommitted roadmap edit (CLOSED status, MTP 1.50× recommendation)
  matches the final commit `68243a61b2`. Commit it.
- The branch is built `VLLM_NO_MAX_ILP=1` with two unattributed GPU
  faults (hipErrorLaunchFailure + BACO) suspected against the max-ilp
  flag. **This is the biggest open risk on the branch**: all perf
  numbers on this branch are from a build that disables an optimization
  credited with ~2% on dense. Before closing, either reproduce the fault
  and root-cause it, or re-run the final MTP A/B on a max-ilp build.

## 6. Copyright / headers

New files carry the Kevin Read SPDX notice or the upstream contributor
notice as appropriate; the modified test file appends the notice without
replacing the existing one — correct per AGENTS.md. No issues.

---

## Recommended actions (priority order)

1. Re-validate the final MTP 1.50× number on a max-ilp build (or
   root-cause the launch failures) — affects every recorded number.
2. Add ksplit==1 and RPT=4 numeric tests for `dense_gemv_m4_gfx906`.
3. Make `VLLM_GFX906_GEMVM_RPT=4` fall back instead of TORCH_CHECK-abort
   (1.2).
4. Add a `Gfx906FAMetadata` draft-loop contract test (1.7).
5. Record the graph-memory cost of the L5 cg-small sizes (2.1) and the
   fp16-CAS nondeterminism (1.4) in the devlog.
6. Clean up or root-cause the unreachable templated M=4 instantiation
   (1.5).

---

# Addendum (2026-08-24, second pass): merge of `spec-dec-code-rev-ds4.md`

A second independent review (`spec-dec-code-rev-ds4.md`, findings F1–F10)
landed after this one. Each claim was re-validated against the tree
before merging. Verdicts below; items that survive are folded into the
findings list that follows. Note that the working tree at this pass
already contains fixes for my items 1.2, 1.3, 2.1 and ds4 F8 (see
"Already fixed in working tree").

## Verdicts on ds4 findings

### F1 — "L5 cg-small fix unreachable (MRV2)" — **REJECTED** (but one valid residual)

ds4's blocker claims the cg-small restore is dead code because the v1
GPU model runner passes `use_v2_model_runner=True`
(`vllm/v1/worker/gpu/model_runner.py:582`). Independently checked and
refuted:

- That file is the **MRV2** runner. The serving runs for Qwen3.5 use
  **MRV1** (`vllm/v1/worker/gpu_model_runner.py`), which passes
  `use_v2_model_runner=False` (line 7332).
- Runner selection (`VllmConfig.use_v2_model_runner`, `vllm/config/vllm.py:615`):
  env override absent, and `Qwen3_5ForConditionalGeneration` is **not**
  in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (`vllm/config/vllm.py:69` —
  Deepseek V2/V4, GraniteMoe, Inkling, KimiK3, LongcatFlashNgram,
  Qwen2Moe only). The hybrid GDN model also forces V1
  (`is_hybrid` + not-default-V2 → False).
- So `resolve_cudagraph_mode_and_sizes(..., use_v2_model_runner=False)`
  executes the MRV1 block, including the cg-small restore. The serve-log
  PIECEWISE graph counts cited in the devlog (q=4 → 5+3=8 graphs for
  ngram3; q=3 → 5+2=7 for mtp2) match the restored-size arithmetic —
  the code demonstrably ran.

**Valid residual (adopted as N1):** the fix lives only in the MRV1 arm.
If this architecture (or ROCm defaults) ever moves to MRV2, the fix
silently stops applying and the ~13 ms no-draft penalty returns with no
error. Worth a warning-level log line when spec-decode + MRV2 + gfx906
is detected, or a note in the devlog flagging the coupling.

### F2 — "M≤4 kernel is not spec-decode-only; nospec regression claim too strong" — **VALID, adopted (N2)**

Missed by my first pass. `_gfx906_spec_gemv_m4` gates on gfx906 + fp16 +
contiguous + `2 <= n <= 4` — there is **no speculative-decoding context
guard**. A non-spec deployment doing batched decode with 2–4 sequences,
or small uniform prefill tails, now routes to the custom kernel with
loose fp16-CAS accuracy and no dedicated no-spec A/B. The commit claim
"nospec regression structurally zero" is true only for M=1. Acceptable
as-is for this branch (the kernel is numerically gated), but the claim
should be narrowed in the commit/roadmap text, and a no-spec M=2..4
serving A/B should be run before any upstream pitch.

### F3 — "numbers on a non-default/unstable build" — **VALID, duplicate of my §5 item 1**
Same finding, same disposition; keep as top open item.

### F4 — "RT kernel kept on an unattributed 1.6× regression" — **VALID in substance (duplicate of my 1.5), one detail corrected**
Both reviews stated the slow templated M=4 ships as dead code; the
corrected fact is that it is **never instantiated** — the launcher's
M=4 branch calls `_rt` directly, so no `dense_gemv_m_kernel<_,_,4>`
binary exists. The complexity concern (two long-lived kernel variants,
unattributed regression) stands; root-causing remains an open task.

### F5 — "fp16-CAS epilogue nondeterministic and loosely accurate" — **VALID, duplicate of my 1.4**
Same finding. ds4's addendum that the M=1 kernel comment implies
"K-split never on model path" overreads the comment, but the core point
(order-nondeterministic drafts; "token-identical" ≠ bit-exact
run-to-run) is correct and now recorded in the devlog disposition.

### F6 — "k=10240 routing broadened; RPT rules unverified for M>1" — **PARTIALLY VALID**
The k=10240 behavior change is real but verified scoped: that shape is
unique to the MTP `fc` in this checkpoint, the M=1 long-K GEMV is
shape-generic with only K tuned, and the caller only reaches it at
n==1. The RPT=2-vs-4 point is a fair perf nit but the measured sweep
(RPT=2 wins on every m4 shape) resolves it for now. No action beyond
keeping the docstring honest (my 2.4).

### F7 — "uniform fast path not spec-only; prefill could hit it" — **PARTIALLY VALID (generality, not correctness)**
The `num_tokens == num_seqs * max_seqlen_q` branch can indeed fire for
uniform non-spec batches. This is correctness-neutral: attention is
per-query-row, only `[:n_q]` output rows are gathered, and the padded
`q` rows' garbage cannot contaminate real rows. It is a perf-path
generality note, not a bug. Worth a one-line comment in
`gfx906_fa_paged.py` stating the fast path is intentionally
shape-gated, not spec-gated.

### F8 — "no test for the compilation.py L5 block" — **VALID, was my gap too**
Adopted; both this review and ds4 under-tested the config change. Fixed
in the working tree (see below).

### F9 — docs/devlog sync (zombie-VRAM hardware state, "unverified" TODOs
vs headline numbers) — **VALID, minor**; fold into the closing devlog
entry when the branch is finalized.

### F10 — copyright/attribution — **VALID, no issues** (agrees with my §6).

ds4's "cheap wins verified OK" section agrees with my verification of
the reduction, CAS tail coverage, and dispatch exclusions — no
conflicts to reconcile.

## New findings adopted from ds4

- **N1 (from F1 residual):** cg-small fix is MRV1-only; add a
  detection/warning for spec+MRV2+gfx906 or document the coupling.
- **N2 (from F2):** M=2..4 dispatch is not spec-gated; narrow the
  "nospec regression structurally zero" claim to M=1 and run a no-spec
  M=2..4 A/B before upstreaming.

## Already fixed in the working tree (uncommitted at time of this pass)

Verified in `git diff` of the working tree, responding to my items and
ds4's:

- **1.2 fixed:** `_gfx906_spec_gemv_m4` now mirrors `VLLM_GFX906_GEMVM_RPT`
  and returns None (triton fallback) when `m % rpt != 0` — no more
  mid-serving TORCH_CHECK abort.
- **1.3 fixed:** ksplit==1 plain-store epilogue numeric test added
  (`test_..._m4_ksplit1_real_kernel`, M=2..4, K=1024, 14336 rows).
- **F8 fixed:** `tests/test_config.py` pins the upstream test to
  `VLLM_GFX906_SPEC_CG_SMALL=0` and adds
  `test_resolve_cudagraph_mode_gfx906_spec_cg_small_restores_sizes`
  asserting [1,2,3,4,8,12,16].
- **2.1 answered:** devlog records the graph-memory cost (1.60 GiB total
  incl. the three small graphs, within the 4-seq GDN budget).

Still open after this pass: max-ilp re-validation (item 1), RPT=4
numeric test (item 2, partially narrowed — devlog sweep data covers it
but no unit test), Gfx906FAMetadata contract pin (item 4, devlog defers
it with a defensible argument), M=4 regression root-cause (item 6), and
the new N1/N2.

## Updated priority list

1. Max-ilp build re-validation / launch-failure root cause (mine §5.1,
  ds4 F3).
2. N2: no-spec M=2..4 A/B + narrow the nospec-regression claim.
3. RPT=4 (`pk4` CAS) numeric test.
4. Gfx906FAMetadata draft-loop contract pin.
5. N1: MRV2 coupling warning for the cg-small fix.
6. M=4 templated-kernel regression root-cause (F4/my 1.5).
7. Commit the working-tree fixes + updated roadmap/devlog.
