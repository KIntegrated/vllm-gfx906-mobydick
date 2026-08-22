# Adversarial self-review — plan_masked_fa.md (merged with external reviews)

Copyright Kevin Read <me@kevin-read.com>

Status: review (2026-08-21), updated same day to fold in two independent
external reviews (`plan_masked_fa_rev_glm5.md`, `plan_masked_fa_rev_qwen.md`)
and an R3 experiment run against the live TP=2 server. This revision
**supersedes** the original self-review below the "Original self-review
findings" heading — several of my own original findings (R2, R3) are
now shown to rest on a false premise and are corrected here rather than
silently dropped, per this repo's convention of keeping the record of
how a diagnosis evolved.

## Summary verdict (revised)

Both external reviews (GLM-5, Qwen) independently converged on the same
two fatal findings against `plan_masked_fa.md`, and I verified the
load-bearing claims myself rather than accepting them on authority:

1. **The plan analyzes the wrong kernel for the 131072/262144 scenario.**
   `launch_gather_paged_kv_q8` V1 (the plan's §0 target) cannot run at
   `Sk_pad > 65535` — confirmed directly in code
   (`gfx906_fa_gather.hip`: `if (cached_version == 1 && Sk > 65535)
   cached_version = 2;`). **Both configs that motivated this entire
   investigation (131072 and 262144) force V2, unconditionally,
   regardless of `GFX906_FA_GATHER_V`.** My own R3 ("V1 vs V2 A/B")
   was therefore attempting to compare two arms that are actually the
   same arm — I discovered this the hard way by trying to run it (see
   "R3 experiment outcome" below) before finding the line that makes it
   impossible.
2. **The tax is dominated by real O(Sk_pad) HBM traffic (V-zero writes
   in the gather kernel, plus a *fourth*, previously-unmentioned kernel
   — `quantize_q8_0_dense_kernel`, invoked on the two-kernel fallback
   path that both 131072 and 262144 actually take), not by dispatch
   overhead.** This directly overturns my own §0 diagnosis and R1
   finding. I re-derived the arithmetic independently (not just
   trusting the reviews) using the **actual model geometry**, read
   from `config.json` (both reviews guessed this, with GLM guessing
   wrong): 16 FA layers, 4 KV heads total → 2/GPU under TP=2, head_dim
   256. Under the corrected V2-forced assumption, explaining the full
   8.4 ms/step S8 gap via dispatch alone requires ~8 ns/block — an
   order of magnitude less physically plausible than the ~0.8 ns/block
   my original (V1-based) model needed. This independently corroborates
   the reviews' claim that memory traffic, not dispatch, dominates.
3. **Option 1 (conditional graph nodes) is not just architecturally
   doubtful (my original R2) — it is dead by direct header evidence.**
   Both reviews grepped the ROCm 7.14 HIP headers and found zero
   conditional-node types. This confirms and sharpens my original R2:
   the blocking question was answerable in seconds by reading a header,
   not requiring a spike.
4. **Both reviews independently identify the design I dismissed
   (Option 2) as actually the correct target, once "fixed grid" is
   read as "fixed *small* constant" rather than "fixed at `Sk_pad`."**
   I mischaracterized Option 2 in the plan and both reviews caught it
   independently — a persistent/grid-stride kernel with a small
   capture-time-constant grid and a live `seq_lens`-bounded work loop
   (the same live-bounding pattern the attention kernel already uses)
   is capture-safe by construction, needs no conditional nodes, no
   tiers, and no dispatcher fallback.

**My own R3 recommendation (ship `GFX906_FA_GATHER_V=2` as a cheap
win) is now known to be moot** — V2 is already forced at both
motivating configs; there is no V1-vs-V2 choice to make there. This is
the single biggest correction to my own work in this review: I
proposed an experiment that, had I read one more line of the launcher
before proposing it, I would have known was untestable as designed.

## R3 experiment outcome (this session, before the external reviews were read)

I attempted to run the R3 experiment (serving A/B of
`GFX906_FA_GATHER_V=1` vs `=2` at `max_model_len=262144`) against the
live TP=2 server. Two server boots hit unrelated problems (a 120k-token
prompt crashed the engine on an `aten::empty` allocation failure inside
a compiled GPTQ gemm at prefill — not a gather-kernel issue, not
investigated further; a second boot at 60k tokens was still loading
when I pivoted). Before completing a clean run, I re-read
`gfx906_fa_gather.hip`'s launcher while setting up the profiler capture
and found the `Sk > 65535` auto-switch — at that point the experiment
as designed was already known to be invalid, independent of the
external reviews (which confirmed it minutes later). No V1-vs-V2 serving
numbers were collected, because there is no V1 arm reachable at
131072/262144. This is recorded as a negative result: **the experiment
that motivated this whole side-quest cannot be run at the configs that
matter**, which is itself the useful finding — it forced the
kernel-selection question that the external reviews then answered
definitively.

A kernel-trace attempt (`rocprofv3 --attach`) was blocked by the target
process not exposing the attach thread even with `ROCP_TOOL_ATTACH=1`
set at launch (env var did not propagate through vLLM's fork-based
multiproc worker spawn). A follow-up TP=1 direct-launch trace
(`rocprofv3 ... -- python _bench_gfx906.py`) got further — the profiler
attached across the process's internal spawn boundary and the model
began loading — but crashed with `hipErrorLaunchFailure` /
`c10::AcceleratorError: unspecified launch failure` at `SetDevice`
before any kernel data was collected, in a way that looks like an
interaction between rocprofv3's instrumentation and this process's
CUDA/HIP device-init path rather than anything related to the gather
kernel itself. **No kernel trace was obtained this session.** Given
time already spent (two TP=2 server boots + one TP=1 profiled attempt,
~40 minutes of GPU-boot time total) and that the source-level evidence
below closes the load-bearing questions without a trace, further
profiler debugging was not pursued. Treat the numbers below as
launch-regime/source-level evidence per house protocol — **a real
kernel trace is still the formal gate and remains an open action for
whoever picks up implementation**, not a "nice to have."

**Independent source-level verification performed instead** (in lieu
of a trace, and beyond what either external review's authors reported
verifying themselves): I read the actual store instructions, not just
the reviews' characterizations —
- `csrc/gfx906_fa/gfx906_fa_gather.hip`, V2's V-pass: confirmed
  `tok_valid = !full_oob && (tok_global < seq_len)`; on the invalid
  branch, `val = make_uint4(0,0,0,0)` and the code **unconditionally
  executes** `reinterpret_cast<uint4*>(v_dst_tok)[c] = val` — a real
  16-byte global-memory store per invalid token-chunk, not a skipped
  write. This directly confirms F2/F3's "V-zero writes are real
  traffic, not a cheap early-exit" claim from the store instruction
  itself.
- `csrc/gfx906_fa/gfx906_fa.cpp:382-411`, `quantize_q8_0`: confirmed
  the function signature takes only `k_fp16` (the tensor) — `N` is
  computed purely from `k_fp16.dim()`/shape, with **no `seq_lens`
  parameter anywhere in the signature or body**. This kernel cannot
  possibly skip padding rows because it has no information about where
  real data ends; its cost is unconditionally `O(B×Hkv×Sk_pad)`. This
  independently confirms both reviews' F2/F3 claim about the "fourth,
  unmentioned kernel."
- `/opt/rocm/include/hip/hip_runtime_api.h`, `hipGraphNodeType` enum:
  confirmed directly (see "Validated external findings" below) — runs
  0-14, no conditional variant, on both the active and the
  `-gfx906-old` ROCm installs on this machine.
- `config.json` for the actual served model: confirmed 16 FA layers
  (not the ~10 guessed by the original plan/investigation), 4 KV heads
  total (2/GPU under TP=2), head_dim 256 — settling the model-geometry
  uncertainty both external reviews flagged and partially guessed at.

## Validated external findings (both reviews, cross-checked against code myself)

- **V1 is unreachable at `Sk_pad > 65535`** — verified directly:
  `gfx906_fa_gather.hip`, `launch_gather_paged_kv_q8`:
  `if (cached_version == 1 && Sk > 65535) cached_version = 2;`. At
  block_size=16, this is `Sk_pad > 65535` tokens — both 131072 and
  262144 trip it. My plan's §0/§4-Q2 named V1 as "the default" for
  these configs; that was wrong.
- **The fused gather+quant kernel is also capped and falls back** —
  verified: `gfx906_fa_paged.py:500`, `if _FUSED_QUANT and Sk_pad <=
  65535:` — false at both configs, so the LEGACY=1 path actually taken
  is `gather_paged_kv_fp16` (V2, byte-generic) + a **separate**
  `quantize_q8_0(K_bhsd)` call. Verified this second kernel exists and
  its grid formula: `gfx906_fa_quant.hip:177`, `dim3 grid((N +
  ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1)` where `N` is computed
  from the full gathered-buffer shape with **no `seq_lens` parameter in
  its signature at all** — confirmed by reading the launch call; this
  kernel cannot early-exit per-sequence because it has no way to know
  where each sequence's real data ends once flattened into `N` rows.
  My plan's §3 step 2 ("change `launch_gather_paged_kv_q8` … both V1
  and V2 variants") would leave this kernel completely untouched —
  exactly the gap both reviews flag as F2/F3's central point.
- **The "V tail must be zeroed" premise in the existing code comments
  is stronger than what the FA kernel actually requires** (Qwen's F4,
  independently corroborated by GLM's F4 with the same conclusion). I
  confirmed the V-write side of this myself this session (see the
  "R3 experiment outcome" section above — the unconditional
  `reinterpret_cast<uint4*>(v_dst_tok)[c] = val` store) but did not
  re-derive the FA-kernel-side half (that `fattn-q8.cuh`'s tail-tile
  loader zero-fills OOB rows in LDS without a global read, making the
  gather's zero-write genuinely unnecessary) line-by-line myself — both
  reviews independently reached that half from the same
  `oob_check`/tail-tile code path. Net: **half independently confirmed
  by me, half validated by convergence of two reviews only.** The
  plan's revision should still gate any implementation that relies on
  removing the zero-fill behind the NaN/Inf-tail-injection test both
  reviews correctly insist on.
- **Model geometry**: I independently pulled `config.json` (both
  reviews reported it as unreachable/NFS-blocked in their sessions) and
  got authoritative numbers: 64 total layers, 16 `full_attention` (FA)
  layers via `full_attention_interval: 4`, `num_key_value_heads=4`
  (→2/GPU under TP=2, confirming Qwen's figure and correcting GLM's
  "4/rank"), `head_dim=256`. Any future arithmetic in the plan or
  investigation docs should use these confirmed numbers, not guesses.
- **Option 1 dead by header evidence — independently re-confirmed by
  me, not just accepted from the two reviews.** I ran
  `grep -n "hipGraphNodeType" -A 20 /opt/rocm/include/hip/
  hip_runtime_api.h` myself: the enum runs `hipGraphNodeTypeKernel = 0`
  through `hipGraphNodeTypeBatchMemOp = 14` with no conditional variant
  anywhere in the list, on both the active `/opt/rocm` and the
  `/opt/rocm-7.14-gfx906-old` install on this machine. A broader
  `grep -rli conditional /opt/rocm*/include/hip/` returns only files
  matching unrelated substrings (`host_defines.h`, `helpers.hpp`, etc.
  — almost certainly "condition"/"conditionally" in comments, not an
  API). This matches both external reviews' citations exactly. Option 1
  is dead on three independent readings now (two AI reviews plus this
  one), not just convergence — this fact is settled and should be
  recorded as DEAD-END in the plan without further hedging.
- **Option 2's correct form** (small fixed grid, grid-stride, live
  `seq_lens`-bounded work) is a real, buildable design that my plan
  dismissed by attacking a strawman version of it. Both reviews independently
  reconstructed the same correct version and reached the same
  conclusion (this is the actual target design). I accept this
  correction to my own §2.2 — re-reading my own text, the tell is
  exactly where both reviews point: "Launch the gather kernel with a
  **fixed** grid sized to the largest practical case (**or even
  `Sk_pad`**, unchanged from today)" — the parenthetical shows I
  conflated "fixed" with "fixed-at-worst-case," which is the error.

## Where the two external reviews disagree with each other (adjudicated)

- **Magnitude estimates (F3 in both) differ in method and headline
  number** — GLM estimates ~3.5-4 ms of the 8.4 ms gap from traffic
  (leaving room for other terms); Qwen estimates ~6.7 ms of 8.4 ms from
  traffic (leaving less room). Both explicitly label these as
  order-of-magnitude, launch-regime estimates pending a real kernel
  trace, and both used guessed model geometry (GLM's guess was wrong
  per my config.json read; Qwen's was right). Given Qwen's model
  geometry is now confirmed correct, **Qwen's F3 numbers are more
  trustworthy than GLM's**, but neither should be treated as gate-level
  until the actual trace (attempted, not yet completed, this session)
  produces real per-kernel timings.
- **GLM's proposed re-anchor keeps a "tiered FULL graphs" fallback
  (§4.3) as a fallback path if the persistent-kernel rewrite stalls;
  Qwen's proposed re-anchor also keeps this as its "fallback if the
  rewrite stalls," essentially identical.** No real disagreement here,
  just independent convergence — worth noting because it strengthens
  confidence in that specific piece of the re-anchored plan (tiered
  FULL graphs via the existing `BatchDescriptor`/`CudagraphDispatcher`
  machinery, no conditional nodes needed, each tier is an ordinary
  captured graph) as a legitimate secondary option, not just a
  first-reviewer's idiosyncratic take.
- **Qwen's F9 finding** (the `Sk`-linear FA coefficient from
  `DEVLOG-fa-attention.md` is a 40-60× extrapolation and the attention
  kernel is latency-bound at short `Sk`, not bandwidth-bound, so the
  short-`Sk` coefficient "does not transfer" to long-`Sk` predictions)
  is not addressed by GLM at all. I find Qwen's specific point here
  persuasive and independently checkable in principle (compare 327 µs
  at Sk~2176 against a naive bytes/bandwidth floor) but did not
  re-derive it myself this session — flagged as validated-by-one-review-
  only, lower confidence than the doubly-corroborated findings above.

## Original self-review findings — status after this revision

- **R1 (my original finding: "0.8ns/block is plausible"): SUPERSEDED.**
  Built on the false V1-at-262144 premise. Re-derived under the correct
  V2-forced assumption with confirmed model geometry: dispatch-only
  would need ~8ns/block to explain the full gap, an order of magnitude
  less believable — this now argues *against* my original conclusion
  and *for* the external reviews' traffic-dominant model. Kept in the
  historical record below for transparency, not as current guidance.
- **R2 (Option 1 architecturally doubtful): CONFIRMED AND SHARPENED.**
  My original instinct (stream-capture API mismatch) was directionally
  right; the external reviews supply the decisive header-level evidence
  that makes it a settled "no" rather than a "probably no, needs a
  spike."
- **R3 (ship V2 as a cheap win): INVALIDATED.** V2 is not a choice at
  the configs that matter — it's already forced. This recommendation
  should be struck from the plan, not merely reprioritized.
- **R4 (LEGACY=0 paged-direct kernel's own Sk-freeze, unchecked):**
  still open, not addressed by either external review; still a fair,
  minor gap.
- **R5 (Q3 tier-threshold data is a precondition for Option 1, not a
  peer open question): MOOT** — Option 1 is dead (R2/F1), so this
  finding's object no longer exists as a live design. Its point
  (threshold/tier choices need real traffic data) still applies
  verbatim to the tiered-FULL-graph fallback both external reviews
  keep as a secondary option, so re-target it there if that path is
  pursued.
- **R6 (prefill gather cost, unaddressed): still open**, not addressed
  by either external review; still a fair, minor gap, now sharpened by
  Qwen's F10(c) note that `kv_split` is forced to 1 for `seq_q > 2`
  (prefill) — confirming prefill takes a different code shape worth
  checking for the same `Sk_pad`-grid issue independently.

## What to do next

The plan (`plan_masked_fa.md`) needs a substantial rewrite, not a
patch: §0's kernel identification, §2's option analysis, and §3's step
ordering are all built on the wrong target kernel. Recommend adopting
the converged external-review re-anchor: (1) a real kernel trace as
step 0 (in progress this session, not completed — see experiment
outcome above), (2) the persistent grid-stride fused gather+quantize
as the target design (kills dispatch, V-zero traffic, and the
`quantize_q8_0` traffic in one change, capture-safe by construction,
no tiers/conditional-nodes/dispatcher-fallback needed), (3) tiered FULL
graphs (plain `BatchDescriptor` tiering, no conditional nodes) as the
fallback if the kernel rewrite stalls, (4) Option 1 recorded as
DEAD-END with the header-grep evidence so it isn't revived, (5) the
NaN/Inf-tail-injection gate before relying on the "V tail doesn't need
zeroing" finding, and (6) my own R4/R6 gaps folded in as additional
open items.

---

## Original self-review (2026-08-21, pre-merge) — kept for the record

The section below is the original, unmodified content of this review
before the external-review merge. Where it conflicts with the merged
findings above, **the merged findings above are authoritative.**

### Summary verdict

The plan's §0 diagnostic correction (gather kernel, not attention
compute kernel, is the mechanism) holds up and is now **quantitatively
supported**, not just qualitatively argued (finding R1). The §1 path
decision (LEGACY=1) holds up unconditionally. But the plan's
**recommended fix (Option 1, HIP conditional graph nodes) rests on an
architectural assumption that is likely wrong** (finding R2) — it
significantly understates the engineering cost, to the point that
Option 1 may not be buildable through any supported API at all, not
just "gated behind a version check" as the plan states. Separately,
the plan **buried a nearly-free, zero-new-code mitigation** (switching
the existing `GFX906_FA_GATHER_V=2` default) that quantitatively could
close most of the measured gap on its own, and should have been
sequenced before any of the conditional-node design work (finding
R3). Net effect: the plan's ordering of effort is backwards — it
should gate on the cheap experiment first, not last.

*(R1/R3's content is preserved verbatim below for the record; both are
now superseded per the "Original self-review findings — status after
this revision" section above.)*

### R1 (strengthens the plan) — the §0 gather-vs-attention split is quantitatively consistent, not just plausible

The plan's §0 correction rests on "even at a few nanoseconds of
per-block overhead, [wasted gather dispatch] could plausibly dominate"
— stated without arithmetic in the plan itself. Checked: using the S8
numbers (131072→39.9 t/s, 262144→29.9 t/s ⇒ 8.38 ms/step delta) and
the gather grid sizes (`Sk_pad` = 131072 vs 262144, `gridDim.z` = `Sk`
directly for V1), the required per-block overhead to fully explain the
measured delta at 4 seqs × ~2 KV heads × ~10 FA layers is **~0.0008
µs (0.8 ns) per scheduled-but-early-exiting block**. [SUPERSEDED — see
above: this used V1's grid, which cannot run at these Sk values.]

### R2 (likely invalidates Option 1 as scoped) — HIP conditional graph nodes are not reachable from vLLM's capture path, and the plan understates this

[Content unchanged from original — see merged section above,
"CONFIRMED AND SHARPENED."]

### R3 (should have been sequenced first, not discovered as a footnote) — V2 gather may already close most of the gap for free

[Content unchanged from original — see merged section above,
"INVALIDATED."]

### R4-R6 and "What does NOT need correction"

[Unchanged from the original review; see the live findings above for
current status of each.]
