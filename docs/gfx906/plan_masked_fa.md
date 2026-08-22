# Plan — masked early-exit for GFX906_FA's gather kernel (max_model_len decode tax)

Copyright Kevin Read <me@kevin-read.com>

Status: **IMPLEMENTED + GATED (2026-08-22, branch `gfx906/fa-masked-gather`).**
Plan dated 2026-08-21, revised same day after two independent
adversarial reviews (`plan_masked_fa_rev_glm5.md`,
`plan_masked_fa_rev_qwen.md`) plus a self-review
(`plan_masked_fa_plan_rev_claude.md`) found the original §0/§2/§3
targeted the wrong kernel and an unbuildable fix; this revision is the
converged, cross-verified design. All §3 steps ran (step 1's rocprofv3
remained unusable — standalone probes + the serving A/B substituted;
step 6's bit-exact greedy is not a valid instrument on this hybrid
model — PPL + kernel-level gates carry correctness, see devlog). §4
open items resolved: NaN-tail gate PASSED (margin stays 128 as
defensive default), prefill path covered by the same dispatch + PPL,
grid 1024 validated. Serving A/B (THE gate): 262k 15.9→40.9 t/s
(+157%), 131k 22.4→40.9 t/s (+83%), P1 residual tax 0.07%. Full
record: `DEVLOG-masked-fa.md`. House protocol (`docs/gfx906/AGENTS.md`)
applied: micro-bench per shape before the model path, PPL gate, serving
A/B, separate commits, positive AND negative results recorded.

## 0. What's actually happening (revised diagnosis, triple-verified)

At `--max-model-len` 131072 or 262144 — the two configs whose ~25%
decode regression motivated this whole investigation — the LEGACY=1
(default) decode path takes the **two-kernel fallback**, not the
single fused kernel, because `Sk_pad` (= `pad32(max_model_len)`, frozen
at FULL-graph capture to `self.max_model_len`) exceeds HIP's
`gridDim.z` cap of 65535 at both values:

```python
# gfx906_fa_paged.py
if _FUSED_QUANT and Sk_pad <= 65535:      # false at 131072 AND 262144
    K_q8, V_bhsd = gfx906_fa.gather_paged_kv_quantized(...)
else:
    K_bhsd, V_bhsd = gfx906_fa.gather_paged_kv_fp16(...)   # V2, forced
    K_q8 = gfx906_fa.quantize_q8_0(K_bhsd)                 # separate kernel
```

Two kernels actually run per FA layer per decode step at these
configs:

1. **`gather_paged_kv_q8_kernel_v2`** (`gfx906_fa_gather.hip`) — grid
   `(num_seqs, num_kv_heads, ceil(Sk_pad/block_size))`. V1 (grid
   `(num_seqs, num_kv_heads, Sk_pad)` directly) is unreachable here:
   `launch_gather_paged_kv_q8` contains `if (cached_version == 1 &&
   Sk > 65535) cached_version = 2;` — an unconditional, sticky,
   process-global override. **V1 vs V2 is not a real choice at
   `max_model_len` ∈ {131072, 262144}; both are already V2, regardless
   of `GFX906_FA_GATHER_V`.** (An earlier version of this plan proposed
   an A/B between them — that experiment is not runnable at these
   configs, discovered by attempting to run it; see
   `plan_masked_fa_plan_rev_claude.md`.)
2. **`quantize_q8_0_dense_kernel`** (`gfx906_fa_quant.hip`,
   `gfx906_fa.cpp:382-411`) — a **separate, generic dense quantize**
   over the entire gathered K buffer. Its signature is
   `quantize_q8_0(torch::Tensor k_fp16)` — **no `seq_lens` parameter at
   all**; `N` (row count) is computed purely from the tensor's shape.
   It cannot skip padding rows because it has no way to know where they
   are. Grid: `ceil(N/ROWS_PER_BLOCK)` where `N = B × Hkv × Sk_pad`.
   This kernel was **not identified in the original version of this
   plan** and is the single largest missed piece of the mechanism.

Both kernels do real, unconditional global-memory work over the full
`Sk_pad` extent, not cheap early-exits:
- V2's out-of-range blocks still execute
  `reinterpret_cast<uint4*>(v_dst_tok)[c] = val` (a zeroed value, but a
  real 16-byte store) for every invalid token-chunk — confirmed by
  reading the store instruction directly, not inferred.
- `quantize_q8_0_dense_kernel` reads and quantizes **every** row of the
  gathered K buffer, valid or not, because it has no per-row validity
  signal.

**The attention compute kernel itself (`flash_attn_tile_q8`,
`fattn-q8.cuh`) is not part of this bug.** Its grid has no `Sk`
dimension (`dim3(x=(seq_q+NC1-1)/NC1, y=kv_split, z=batch·heads_q/NC2)`,
`kv_split` a fixed env constant), and its K-loop is bounded by a live,
per-sequence `KV_max` value fed from the runner's persistent, every-step-
refreshed `seq_lens` GPU buffer — a real, already-shipped, correctly-
functioning masked early-exit. This part of the original diagnosis
(§0 in the first version of this plan) was correct and survives
review.

**Model geometry** (confirmed from `config.json`, not guessed): the
served model (`cyankiwi/Qwen3.8-27B-AWQ-INT4`) has 64 total layers, 16
`full_attention` (FA) layers (`full_attention_interval: 4`), 4 KV heads
total (2/GPU under TP=2), `head_dim=256`. Use these numbers, not the
"~10 layers" guess in earlier drafts of this investigation, for any
future magnitude arithmetic.

**Quantitative sanity check** (launch-regime estimate, not the gate):
under the corrected V2-forced assumption, explaining the full 8.4
ms/step S8 gap (39.9→29.9 t/s, 131072→262144) via gather-dispatch
overhead *alone* requires ~8 ns of overhead per scheduled block — an
order of magnitude less physically believable than the ~0.8 ns/block a
(now-known-incorrect) V1-based model needed. This is consistent with
—though does not on its own prove — the reviews' conclusion that real
HBM traffic (V-zero writes + the unconditional `quantize_q8_0` pass),
not dispatch count, is the dominant term. **A real kernel trace
(attempted this session via `rocprofv3`, not completed — see the
self-review for the failure mode) is still needed to confirm the exact
V-zero/quantize/dispatch split and remains the first implementation
step below.**

## 1. Path selection — LEGACY=1 vs LEGACY=0/direct-paged

Unchanged from the original plan and re-verified by both external
reviews independently: **build on LEGACY=1.**

- All 8 tests in `tests/kernels/attention/test_gfx906_fa.py` target the
  LEGACY=1 gather-then-dense path; several assert `impl._legacy`
  directly. Zero tests exercise `GFX906_FA_LEGACY=0`/
  `forward_paged_direct`.
- `GFX906_FA_LEGACY=0` fails closed (`RuntimeError`) when combined with
  prefix caching (`gfx906_fa_backend.py` `get_cudagraph_support`, R2 in
  `moe-decode-roadmap.md` §9.1) — the Q8 K side-buffer misses COW'd
  prefix-cache blocks. Untested, undocumented in depth, and a separate,
  harder project to fix. Prefix caching is exactly what makes long
  multi-turn conversations (the workload this whole investigation is
  about) affordable, so this gap can't just be worked around for this
  plan's purposes.
- The LEGACY=1 fix below is self-contained: it changes gather/quantize
  kernel behavior and does not touch KV-cache write consistency at
  all, so it carries none of LEGACY=0's corruption-class risk.

## 2. The fix — kernel-side live bounding (persistent grid-stride gather+quantize)

### 2.1 What does NOT work (ruled out, keep out of scope)

- **Conditional HIP graph nodes** (the original plan's Option 1,
  "target design"). **Dead, confirmed independently three times**: two
  external reviews and this plan's author each grepped ROCm 7.14's HIP
  headers (`/opt/rocm/include/hip/hip_runtime_api.h`,
  `hipGraphNodeType` enum) and found no conditional-node type (0-14,
  ending at `hipGraphNodeTypeBatchMemOp`). Even if a future ROCm added
  one, vLLM's `CUDAGraphWrapper` captures via PyTorch's stream-capture
  API (`torch.cuda.graph`), which has no hook to insert nodes into an
  already-captured graph — conditional nodes require the explicit
  graph-construction API, a different, non-overlapping code path.
  Recorded as **DEAD-END**; do not revisit without a fundamentally
  different capture mechanism than vLLM uses today.
- **A/B'ing `GFX906_FA_GATHER_V=1` vs `=2` at 131072/262144.** Moot —
  both configs force V2 unconditionally (see §0). This was the
  original plan's top-priority "cheap experiment"; it cannot be run as
  designed. (V1 vs V2 *is* a real, live question at `max_model_len ≤
  65535`, where the existing launcher comment records V1 as 15% faster
  — out of scope for this plan, which is about the large-`Sk`
  regression specifically.)
- **Naively "compute the grid size live and pass it at replay."**
  HIP/CUDA graph capture bakes kernel launch geometry (grid/block dims)
  at the moment of capture; it is not recomputed from live data on
  replay under any mechanism vLLM uses. Any design must keep the
  captured grid a **fixed constant** and push all live-length-
  dependent behavior *inside* the kernel body, reading from data (a
  live tensor), not from launch parameters.
- **A per-block early-return inside the existing grid shape alone**
  (the original plan's "Lever A"/"Option 2, strawman form"). Reducing
  what an out-of-range block *does* without reducing how many blocks
  get *scheduled* does not address the dispatch component and does
  nothing for the `quantize_q8_0` traffic component at all, since that
  kernel doesn't have per-row validity information to check against in
  the first place.

### 2.2 Target design — persistent, fixed-small-grid, live-bounded, fused gather+quantize

Launch a **single fused kernel** (folding gather + in-kernel quantize
into one, eliminating the separate `quantize_q8_0` pass entirely — the
existing `gather_paged_kv_quantized` fused kernel already does this,
just not beyond the 65535 cap) with:

- **A fixed, small grid, chosen as a capture-time constant** — e.g.
  enough workgroups to fill the 60 CUs on this hardware (a number in
  the low hundreds, not `Sk_pad`-scaled). This is legal under graph
  capture *by construction*: the grid never depends on any live value,
  so the same captured graph is valid for every `max_model_len` and
  every live context length. No conditional nodes, no tiers, no
  dispatcher fallback, no batch-wide cliff.
- **A grid-stride loop inside the kernel over the (seq, head,
  token-chunk) work space**, with the total amount of real work bounded
  by a value **read from the live `seq_lens` tensor inside the
  kernel** — the exact same live-bounding pattern the attention compute
  kernel's `KV_max` already uses successfully (confirmed working,
  confirmed live-refreshed every step, per §0). This is what makes the
  design "masked early-exit" in the sense the original investigation
  asked for, applied correctly this time: work is `O(real_seq_len)`,
  dispatch is `O(fixed constant)`, and both properties hold at every
  `max_model_len` and every live context length in a single capture.
- **No output write beyond what the FA kernel will actually read.**
  Per both external reviews' F4 finding (partially, not fully,
  independently re-verified this session — see the self-review): the
  FA kernel's tail-tile loader zero-fills out-of-range K/V rows in LDS
  without a global memory read, so it likely never actually needs the
  gather buffer's padding to be pre-zeroed. **This must be confirmed
  by a dedicated correctness gate (§4) before being relied on** — fill
  the K/V tail beyond `seq_len` with NaN/Inf in a test and assert FA
  output is unchanged. Until that gate passes, keep a defensive
  zero-write for at least the one tail tile width (`nbatch_fa` rows)
  past `seq_len`, not the full `Sk_pad` extent — this alone (bounding
  the zero-write from `O(Sk_pad)` to `O(nbatch_fa)`, a small constant)
  removes the overwhelming majority of the V-write traffic even before
  the more aggressive "no zero-write at all" version is gated in.

The gather **buffer** itself is unchanged: still allocated at
`Sk_pad`-capacity (no realloc, no change to `_ensure_gather_buffers`'s
VA capture-safety design, R3/R8 in `moe-decode-roadmap.md` §9.1/9.3
stay intact). Only the **kernel doing the writing** changes — what
work it does and how many workgroups it takes to do it.

This design also **deletes the `Sk > 65535` two-kernel fallback
branch entirely**: since the grid is now a fixed small constant
regardless of `Sk_pad`'s magnitude, the `gridDim.z` cap that forced the
V1→V2 switch and the fused-kernel→two-kernel fallback never applies.
One code path serves every `max_model_len`.

### 2.3 Fallback if the kernel rewrite stalls (correctness or scheduling risk)

**Tiered FULL graphs, keyed by (batch size × Sk tier), via the existing
`BatchDescriptor`/`CudagraphDispatcher` machinery.** Each tier is an
*ordinary* captured graph (this is legal — no conditional nodes needed,
because tiering happens at the *outer* Python dispatch level, choosing
which whole graph to replay, not by branching inside one graph); the
investigation's capture-time `max_seq_len` bound is what differentiates
the captures (each tier captures with `for_cudagraph_capture`'s
`max_seq_len` set to the tier's bound instead of the true
`max_model_len`). No dispatcher-level `invalid_modes` cliff to eager —
every tier is still a FULL graph.

Residual cost if this fallback is used instead of §2.2: each tier still
pays that tier's `Sk_pad`-proportional gather+quantize cost with the
*current* (unbounded-zero-write, two-kernel-above-65535) kernel body
unless the §2.2 kernel changes are also applied within each tier — so
this fallback is not a substitute for the kernel rewrite, it's a
different way to bound the worst case if the rewrite proves infeasible.
Tier thresholds need real traffic/context-length distribution data to
choose well (not devlog anecdotes) — this is a precondition for this
fallback's cost/benefit case, not an independent nice-to-have.

## 3. Implementation steps

1. **Kernel trace (blocking, do first).** Get a real per-kernel-call
   timing breakdown (gather-V2 / `quantize_q8_0` / FA-compute) at
   `max_model_len=262144`, ~1.5k live tokens, FULL-graph decode —
   matching the S8 operating point. Attempted this session via
   `rocprofv3` (both `--attach` to a running TP=2 server and a direct
   TP=1 launch); both attempts failed on tooling/environment issues
   unrelated to the kernel logic (attach-thread not exposed despite
   `ROCP_TOOL_ATTACH=1`; direct-launch crashed with
   `hipErrorLaunchFailure` at device init under the profiler). Retry
   with: (a) confirm `ROCP_TOOL_ATTACH=1` actually reaches the worker
   process post-fork (may need to set it via a different mechanism than
   the parent's env, e.g. inside the worker init code, or use
   `VLLM_WORKER_MULTIPROC_METHOD=spawn` instead of `fork` for the trace
   run specifically, since spawn was what the TP=1 attempt auto-forced
   into anyway); (b) if `rocprofv3` continues to fail under this
   process topology, fall back to the P3-4-style eager torch-profiler
   correlation technique documented in `DEVLOG-moe-opt.md`, applied to
   the gather/quantize call sites specifically. This step's output is
   both the mechanism confirmation and the serving-A/B baseline.
2. **Correctness gate for tail-write removal**: NaN/Inf-tail-injection
   test (fill K/V past `seq_len` with NaN/Inf, assert FA output
   unchanged) across the decode and GQA-packed (`ncols2>1`) paths,
   before any implementation relies on skipping the zero-write.
3. **Build the fused persistent gather+quantize kernel** (§2.2): fixed
   small grid, grid-stride over (seq, head, token-chunk), live
   `seq_lens`-bounded work, in-kernel quantize (reuse
   `quantize_block_q8_0_halfwarp`, as the existing fused kernel already
   does), margin-only (or, once step 2 passes, zero) tail writes. This
   replaces `gather_paged_kv_q8_kernel_v2` + `quantize_q8_0_dense_kernel`
   with one kernel valid at every `Sk_pad`, deleting the 65535-cap
   fallback branch.
4. **Standalone kernel test**: extend
   `tests/kernels/attention/test_gfx906_fa.py`'s existing
   `test_gather_buffers_capture_sweep_keepalive`/
   `test_cudagraph_capture_replay_legacy_decode_path` pattern (capture
   at large `Sk_pad`, replay at small live `Sk`) to assert **end-to-end
   FA-output equality** against today's two-kernel path — not
   bit-identical gather-buffer contents (rows beyond `seq_len` may
   legitimately differ once tail-write removal lands, by design; see
   step 2's gate), across a sweep of live lengths including values near
   the old 65535 boundary (now irrelevant to correctness but worth
   confirming performance continuity across).
5. **Micro-bench** the new kernel alone (extend
   `benchmarks/kernels/gfx906/bench_gfx906_fa_gather.py`) at a matrix
   of (`max_model_len`, live length) pairs spanning both sides of the
   old 65535 boundary and up to 262144, confirming (a) it beats the
   current two-kernel path at large `Sk_pad`, and (b) it does not
   regress the already-fast small-`Sk` fused-kernel case.
6. **PPL + greedy correctness gate**
   (`benchmarks/kernels/gfx906/ppl_probe.py` pattern).
7. **Serving A/B** (`_bench_gfx906.py`-style / the S8
   `tp2_serve_bench2.py` harness, per house protocol) at
   `max_model_len` 131072 and 262144, matched real context, reproducing
   and (hopefully) closing the S8 gap. Also a concurrency/mixed-length
   A/B (per the C2-V precedent in `moe-decode-roadmap.md`) since this
   design has no batch-wide cliff to check for, but concurrency effects
   on the new kernel's occupancy should still be measured.
8. **If the rewrite stalls** (correctness issues that don't resolve
   quickly, or the persistent-kernel occupancy/coalescing design proves
   harder than expected on gfx906): fall back to §2.3's tiered-graph
   approach, pricing in its residual cost honestly rather than assuming
   it away.

## 4. Open items

- **Kernel trace still outstanding** (step 1) — the load-bearing
  quantitative gate for this whole plan. Everything above step 1 is
  launch-regime/source-level evidence, not a measurement.
- **LEGACY=0's paged-direct launcher** (`gfx906_fa_launch_paged_impl`)
  — unchecked whether it has an analogous `Sk`-shaped grid freeze. Not
  blocking for this plan (LEGACY=1 is the chosen path), but relevant if
  a future project ever revisits closing LEGACY=0's COW-prefix-caching
  gap.
- **Prefill-path gather cost** — `forward_paged` is also the prefill
  entry point; whether prefill pays an analogous capture-frozen cost
  under `FULL_AND_PIECEWISE`/mixed-batch FULL-graph configurations
  (`kv_split` is forced to 1 for `seq_q > 2`, confirmed, so at least the
  attention-compute side is unaffected — but the gather/quantize side
  for prefill chunks hasn't been checked) is unverified either
  direction.
- **The `Sk`-linear FA-kernel coefficient** from `DEVLOG-fa-attention.md`
  (327 µs @ Sk~2176 / 72 µs @ Sk~500) was flagged by one of the two
  external reviews as likely a 40-60× extrapolation outside its
  calibration range, with the attention kernel plausibly latency-bound
  at short `Sk` and bandwidth-bound at long `Sk` — meaning the
  short-`Sk` coefficient may not predict long-`Sk` behavior accurately.
  This doesn't change this plan's target (the attention kernel isn't
  in scope), but any future arithmetic that leans on that coefficient
  should re-derive or re-measure it rather than reuse the two-point fit.

## Cross-references

- `plan_masked_fa_plan_rev_claude.md` — this plan's self-review,
  including the R3 experiment's failed premise (V1/V2 A/B unrunnable
  at these configs) discovered by attempting it, and the merged/
  cross-verified external review findings.
- `plan_masked_fa_rev_glm5.md`, `plan_masked_fa_rev_qwen.md` — the two
  independent external reviews this revision is built on.
- `tp_decode_investigation.md` — origin; RESOLUTION section's
  "attention compute kernel is frozen" claim is corrected by this
  plan's §0 (gather+quantize, not attention compute, is the mechanism).
- `DEVLOG-fa-attention.md` — gather V1/V2/fused-quant development
  history (Route B) and the `Sk`-linear FA-kernel measurement (see §4
  caveat on its extrapolation range).
- `moe-decode-roadmap.md` §9.1/9.3 — R3/R8, the gather-buffer
  retired-list cap and shared-buffer fixes this plan's buffer-lifecycle
  assumptions build on unchanged.
- `tests/kernels/attention/test_gfx906_fa.py` — existing LEGACY=1
  capture/replay test pattern to extend (§3 step 4).
- `benchmarks/kernels/gfx906/bench_gfx906_fa_gather.py` — existing
  gather micro-bench harness to extend (§3 step 5).
