# Adversarial review — plan_masked_fa.md (masked early-exit for the GFX906_FA gather)

Copyright Kevin Read <me@kevin-read.com>

Date: 2026-08-21. Reviewer: Qwen (coding agent), independent pass over
`plan_masked_fa.md` against the current tree. Method: full re-read of
every file the plan cites (gather/quantize/FA kernels, launchers,
`gfx906_fa_paged.py`, backend, model-runner capture/dispatch paths),
header inspection of the ROCm 7.14 toolchain on this machine, model
config check, and cross-checks against `tp_decode_investigation.md`,
`DEVLOG-tp2-dense.md` (S5/S8), and `DEVLOG-fa-attention.md`. No GPU runs
were performed — findings F2/F3 rest on code reading plus
order-of-magnitude arithmetic from the S8 measured delta (8.4 ms/step);
finding F6 (a kernel trace) is required to confirm them before design
work starts.

## 1. Verdict

**The plan's §0 "correction" is directionally right but incomplete in
three load-bearing ways, and its recommended design (Option 1) is
unbuildable on this stack — Q1 is answerable today, and the answer is
NO.** Specifically:

1. **The plan analyzes the wrong kernel for the tax scenario.** At
   `max_model_len` 131072/262144 (both `Sk_pad > 65535`), the LEGACY=1
   default path does **not** run `launch_gather_paged_kv_q8` V1 (the
   plan's §0 target). It runs the **two-kernel fallback**: V2 gather
   (forced by the `gridDim.z` cap auto-switch) + a dense
   `quantize_q8_0` kernel over the **entire** `Sk_pad`, which the plan
   never names (F2).
2. **The tax is mostly real O(Sk_pad) HBM traffic, not dispatch.** V2's
   out-of-range paged blocks each *write* 8 KB of V zeros, and
   `quantize_q8_0` does full read-quantize-write on every out-of-range
   K row. Order-of-magnitude for the TP=2 27B: ~13 ms/step of tail work
   at the 262k config vs ~6.7 ms at 131k — matching the measured
   8.4 ms delta (F3). "Lever A" (cheaper early-exit) therefore has ~zero
   headroom, and any tiered design that keeps the kernel body unchanged
   leaves O(tier) tail traffic (~1 ms/step residual at a 32k tier).
3. **The "V tail must be zeroed" requirement the plan inherits does not
   exist.** The FA kernel's `oob_check` tail tile zero-fills OOB K/V
   rows in LDS without reading global memory — it never reads K or V
   rows at or beyond the live `kv_max` (F4). This removes the last
   apparent obstacle to live-bounded gather grids and makes the OOB
   zeroing pure waste.
4. **The plan's option space omits the design that is graph-safe by
   construction**: a persistent grid-stride fused gather+quantize with a
   *fixed small grid* and live-bounded work read from the `seq_lens`
   tensor in-kernel (F5). The plan's Option 2 paragraph is where this
   design was available and was missed: it redefines Option 2 as "keep
   the Sk_pad grid" and then correctly rejects that strawman, concluding
   one "can't" change the grid under a single capture. One *can* — to a
   small constant frozen at capture. Frozen ≠ frozen-at-worst-case.

Recommended re-anchor: validate the corrected mechanism with a single
kernel trace first (F6), build the persistent fused gather as the target
(F5, with F4's tail-write removal gated by a NaN-injection test), keep
tiered FULL graphs (keyed by batch-size × Sk-tier) as the
no-kernel-rewrite fallback, and demote the investigation's Option 3 to a
short-context mitigation. Details in §4.

---

## 2. What the plan gets right (verified against the tree)

- **FA grid is Sk-independent** — `gfx906_fa_launcher.hip:217-220`:
  `dim3 grid((seq_q+NC1-1)/NC1, kv_split, batch*((heads_q+NC2-1)/NC2))`.
  `kv_split` is the `GFX906_FA_KVSPLIT` env (default 16, `gfx906_fa.cpp:
  84-89`), and is force-1 for `seq_q > 2` (prefill), so the tax discussion
  is decode-graph-only. ✓
- **The FA K-loop is live-bounded** — `fattn-q8.cuh:961-996`:
  `k_VKQ_max = KV_max[sequence*gridDim.x + blockIdx.x]`, loop bounded by
  it. `KV_max` is `seq_lens.to(torch.int32).contiguous()`
  (`gfx906_fa_paged.py:587`) — a **no-op** cast (the persistent
  `self.seq_lens` is already int32, `gpu_model_runner.py:825-827`), so
  the captured graph references the persistent buffer directly, refreshed
  every step at `gpu_model_runner.py:2252-2255`. The plan's
  "already-shipped masked early-exit, fed live data" claim for the
  *attention compute* kernel is correct. ✓
- **The gather grid is frozen at `Sk_pad = pad32(max_model_len)` for
  every FULL capture** — `gpu_model_runner.py:2387-2391`
  (`for_cudagraph_capture: max_seq_len = self.max_model_len`) →
  `gfx906_fa_paged.py:316` → launchers. ✓
- **§1 path selection (build on LEGACY=1) is sound.** Verified: all 8
  tests in `tests/kernels/attention/test_gfx906_fa.py` target the
  LEGACY=1 gather path (several assert `impl._legacy`); zero exercise
  `forward_paged_direct`/LEGACY=0. The direct path requires
  `key_cache_q8 is not None` (LEGACY=0), which is fail-closed with prefix
  caching. ✓
- **Buffer lifecycle correctly left unchanged** — keeping the
  `Sk_pad`-capacity buffer (VA capture-safety, `_ensure_gather_buffers`,
  R3/R8) is the right constraint. ✓
- **The gate ladder (bit-equal capture/replay tests → micro-bench matrix
  → PPL/greedy → serving A/B incl. concurrency) is right** and matches
  house protocol. ✓
- **The dismissal of Option 3 for long-conversation serving is
  quantitatively defensible** — but the plan states the reason only as
  "regresses to PIECEWISE/eager for the rest of its life" without the
  measurement that makes it so: the TP=2 eager cliff (S8: 19.5 vs
  39.9 t/s at matched 1.5k context; `DEVLOG-tp2-dense.md` line 68:
  "Eager TP=2 ≈ 7 t/s is an artifact… graphs are mandatory"). Cite it. ✓

## 3. Findings (ranked)

### F1 (fatal) — Option 1 is unbuildable on this stack; Q1 is answered NO today

Q1 ("does PyTorch's ROCm/HIP graph capture path expose conditional
nodes?") is posed as the plan's blocking open question. It can be
settled without a spike:

- **HIP 7.14 has no conditional-node API.** The graph API is present in
  `/opt/rocm/core-7.14/include/hip/hip_ext.h` (`hipGraphAddNode`,
  `hipGraphAddKernelNode`, `hipGraphAddChildGraphNode`,
  `hipGraphAddMemAlloc/Memcpy/Memset/Host/Event*` nodes, …), but the
  complete node-type list contains **no conditional type**: there is no
  `hipGraphConditionalHandle`, no `hipGraphSetConditional`, no
  `NodeTypeConditional` anywhere under `/opt/rocm/core-7.14/include`
  (grep for `conditional` in the hip headers: zero hits). CUDA 12.4's
  conditional graph nodes were not ported to this ROCm.
- **Even where the API exists, the capture flow can't reach it.**
  vLLM's `CUDAGraphWrapper` captures via `torch.cuda.CUDAGraph()` +
  `torch.cuda.graph(...)` (`vllm/compilation/cuda_graph.py:283,313`) —
  stream capture that yields an instantiated exec graph, with **no
  exposed raw `hipGraph_t`** to insert nodes into. Conditional nodes are
  an explicit-graph-construction feature; no amount of "drop to raw HIP
  for this one kernel call" is reachable from inside an already-captured
  FULL model graph.

**Consequence:** the plan's §2.3 "target design", §3 step 2, and most of
the §2.2 Option 1 machinery assume an API that does not exist here. The
plan's own contingency ("Option 1 collapses to Option 3") is therefore
the only outcome its option space can produce — which is why the missing
option in F5 matters. Rewrite Q1 as **resolved: NO**, with this
evidence.

### F2 (fatal) — the plan targets the wrong kernel for the 131k/262k scenario

`forward_paged`'s LEGACY=1 branch (`gfx906_fa_paged.py`, "Legacy-path"
block) is:

```python
if _FUSED_QUANT and Sk_pad <= 65535:
    K_q8, V_bhsd = gfx906_fa.gather_paged_kv_quantized(...)
else:
    K_bhsd, V_bhsd = gfx906_fa.gather_paged_kv_fp16(...)
    K_q8 = gfx906_fa.quantize_q8_0(K_bhsd)
```

`Sk_pad` is 131072 or 262144 in **both** tax scenarios (> 65535), so the
fused kernel is never used there; the kernels actually running per FA
layer per step are:

1. **`gather_paged_kv_q8_kernel_v2`** — `gather_paged_kv_fp16` calls
   `launch_gather_paged_kv_q8` with `bytes_per_row = 2D`
   (`gfx906_fa.cpp:708-718`), and the launcher's
   `if (cached_version == 1 && Sk > 65535) cached_version = 2;`
   (`gfx906_fa_gather.hip`) **forces V2** (grid `(B, Hkv, ceil(Sk/16))`;
   V1 would need `gridDim.z = 262144`, beyond the 65535 cap).
2. **`quantize_q8_0_dense_kernel`** — grid
   `ceil(B·Hkv·Sk/4)` blocks (`gfx906_fa_quant.hip:175-179`), full
   read-quantize-write over **all** `Sk` rows including out-of-range
   ones (the K tail is unmasked garbage by design).

Against the plan:

- §0 names `launch_gather_paged_kv_q8` (V1, "default") as *the* gather
  kernel. V1 is unreachable at `Sk > 65535`, and that launcher's
  V1/V2 split is only live for the LEGACY=0 fast path or LEGACY=1 at
  `max_model_len ≤ 65535` — not the tax scenario.
- **Q2 is moot**: "is V2 alone worth A/B'ing before any
  conditional-node work?" — at 131k/262k V2 is *already the kernel in
  flight* (auto-forced); there is nothing to switch. (The V1-vs-V2
  question only exists for `max_model_len ≤ 65535`, where the launcher
  comment records V1 15% faster at D=256, Sk=3328.)
- **Q4 is answerable now**: `gather_paged_kv_fp16` does have the
  frozen-grid issue — via the same launcher (as V2). And the plan
  overlooks the **fourth** Sk-dependent launch entirely:
  `quantize_q8_0`.
- §3 step 2 ("change `launch_gather_paged_kv_q8` (both V1 and V2
  variants)") as written would leave the quantize kernel — roughly half
  of the tax, per F3 — untouched in exactly the regime where the tax was
  measured.

### F3 (fatal) — the tax is mostly real O(Sk_pad) HBM traffic, not dispatch

The plan's §0 frames the cost as "many blocks, mostly early-exiting" — a
launch/dispatch count problem. In the actual 131k/262k path, the
out-of-range work is not early-exit:

- **V2 OOB blocks write V zeros.** `full_oob` blocks skip the K pass but
  the V pass *stores* `16 × D × 2` bytes of zeros per block (D=256 →
  **8 KB/block**) — `gfx906_fa_gather.hip` V2 V pass: `tok_valid =
  !full_oob && tok_global < seq_len; ... val = 0 ... store`.
- **`quantize_q8_0` does full work on every OOB K row** (read 2D bytes,
  quantize, write (D/32)×34 bytes) — there is no early exit; the K tail
  is unmasked by construction.

Order-of-magnitude for the TP=2 27B (Qwen3.8-27B-AWQ-INT4: **16 FA
layers** of 64, 24 q heads, **4 kv heads → 2/GPU under TP=2**,
**head_dim 256**; B=1 single request; live ~1.5k; ~798 GB/s HBM):

| config | OOB V-zero writes | OOB quantize work | tail total/step |
|---|---|---|---|
| 262144 | (16384−96) blk/head × 8 KB × 2 heads × 16 ≈ 4.3 GB | 2×(262144−1536) rows × 784 B × 16 ≈ 6.5 GB | **≈ 13 ms** |
| 131072 | (8192−96) × 8 KB × 2 × 16 ≈ 2.1 GB | 2×(131072−1536) × 784 B × 16 ≈ 3.3 GB | **≈ 6.7 ms** |
| Δ | | | **≈ 6.7 ms** vs 8.4 ms measured (S8) |

(The residual ~1.7 ms is plausibly in-range attention growth, dispatch,
and my bandwidth estimate; the match is within the crudeness of the
model.) Consequences:

- **Lever A has ~zero headroom** — OOB blocks aren't exiting cheaply,
  they're doing HBM work.
- **Any design that keeps tail processing over the full captured width
  (Option 1 tiering with the kernel body unchanged) leaves O(tier)
  traffic**: at a 32k tier ≈ (32768−1536)×2 heads×512 B V-zeros
  [fused kernel, in-range quantize only] + dispatch ≈ **~0.7–1.1
  ms/step** — 10× the ~0.1 ms a true live-bounded design gets (F5).
- The `> 65535` two-kernel fallback is a **hidden tax multiplier**: per
  OOB token it costs ~1296 B (V-zero + garbage quantize, D=256) vs 512 B
  for the fused kernel. Part of the 262k penalty is paying for a code
  path that exists only to dodge the `gridDim.z` cap.

Note these are launch-regime order-of-magnitude estimates from code
reading + the S8 delta, not measurements — F6's trace is the gate that
splits V-zero vs quantize vs dispatch precisely.

### F4 (enabling) — the "V tail must be zeroed" requirement does not exist

The gather header (`gfx906_fa_gather.hip`: "V rows beyond
`seq_lens[seq_idx]` are zeroed inline (a kernel requirement: the V
'tail' must not contribute to softmax)") and the plan inherit a
contract the FA kernel does not impose. In `fattn-q8.cuh`, the tail
tile that crosses `kv_max` (`oob_check = true`, lines 972-977 /
981-990) self-masks:

- **V loader** (`flash_attn_tile_q8_q8_load_tile`, line ~161):
  `!oob_check || i < i_sup ? KV + i*stride_KV + j : zero` — OOB rows
  are **zero-filled into LDS without any global read**.
- **K loader** (`flash_attn_tile_q8_q8_load_tile_q8`, lines 189/257):
  OOB rows are zero-padded in LDS; the KQ→exp stage then explicitly
  masks OOB scores (`!oob_check || i_KQ < k_VKQ_sup`, lines
  657/689/736/769).

Non-tail tiles satisfy `k_VKQ_0 + nbatch_fa ≤ kv_max`, so their loads
are fully in-range. **The FA kernel never reads a K or V row at or
beyond the live `kv_max` from the dense buffer.** Therefore:

1. The true gather contract is: rows `[0, seq_len)` valid per sequence;
   **beyond that is never read**. No `+ nbatch_fa` headroom is needed
   for a live-bounded grid.
2. The OOB V-zero writes (F3's ~5 ms/step at 262k) are pure waste and
   removable.
3. Live-bounded grid designs (F5, tiering, the investigation's bound)
   are strictly simpler and safer than the plan assumes — *once gated*.

**Gate required before relying on this** (removing a defensive
invariant): a targeted test that fills the K/V tail beyond `seq_len`
with NaN/Inf and asserts the FA output is unchanged (bit-equal or at
least finite and correct) across the decode and GQA-packed (`ncols2>1`)
paths. Cheap, and it should run before any design in §4 depends on it.
(Scope check: this backend always passes `kv_max` —
`gfx906_fa_paged.py:587` — so the `KV_max == nullptr` mask-only path,
which would read the full `Sk`, is dead here.)

### F5 (major) — the plan's option space omits the design that is graph-safe by construction

**Persistent grid-stride fused gather+quantize:** launch the gather with
a **fixed** grid (a small constant chosen at capture — e.g. 128–512
blocks; a frozen constant, which is exactly what graphs allow); each
block grid-strides over `(seq, head, token-chunk)` work items; the work
bound (per-sequence `seq_len`, hence total work) is **read from the live
`seq_lens` tensor inside the kernel** — the same pattern the plan itself
credits to the FA kernel's `KV_max`; K is quantized in-kernel (reusing
`quantize_block_q8_0_halfwarp`, as the existing fused kernel does),
which deletes the second kernel *and* the `> 65535` two-kernel fallback
branch; no tail writes (F4).

- Grid dims are capture-time constants → **legal under graph capture by
  construction**; no conditional nodes, no tiers, no dispatcher change,
  no Python fallback, no cliff, no step function.
- Kills all three F3 components in one change: dispatch count
  (O(fixed)), OOB V-zero traffic (O(0)), OOB quantize work (O(0)).
  Residual gather cost at 1.5k live context ≈ 3072 rows × 784 B ≈ 2.4 MB
  ≈ tens of µs/step — vs ~13 ms today at 262k.
- The buffer stays at `Sk_pad` capacity (VA/capture-safety unchanged);
  rows beyond the live work are simply never written — and never read
  (F4).
- Bit-equality is inherited per row: quantization is row-local and
  partition-independent (the existing fused kernel is already bit-equal
  to the two-kernel sequence "by construction").
- Precedent for the risk class: the vendor-ported in-repo rewrite of
  `csrc/rocm/moe_q_gemm_gfx906.cu` (Phase 1/2, 12-test suite, shipped).
  The plan's steps 3–6 (bit-equal capture/replay test, micro-bench
  matrix, PPL/greedy, serving A/B) apply unchanged.
- Real risks to price in: (a) work partitioning must keep HBM
  coalescing (process contiguous token runs per `(seq, head)`; with
  B ≤ 4 in the S5 config, static partitioning is trivial); (b) it must
  not regress the already-fast fused small-`Sk` case — the bench matrix
  (plan step 4) covers this; (c) the gather also serves prefill (ragged
  chunked `Sk`) — either scope the new kernel to decode first or verify
  the prefill path in the same test.

**Where the plan lost this design:** §2.2 Option 2 is where a
fixed-grid + grid-stride + live-bounded-work gather was on the table,
and the plan redefined it: "Launch the gather kernel with a **fixed**
grid sized to the largest practical case (or even `Sk_pad`, unchanged
from today)" — i.e. it equated "fixed" with "fixed-at-the-frozen-worst
case", then correctly rejected *that* ("does not solve the
dispatch-count problem"). The actual Option 2 sets the grid to a small
constant *at capture time*; that is not a live-computed value, so the
"can't [change grid size], under a single capture" objection does not
apply. The plan's Option 2 paragraph is a strawman of the only
no-conditional-node design.

### F6 (major) — the plan's first step must be mechanism validation, not a build

The §0 "correction" (attention fine, gather at fault) is a
code-reading claim; per house protocol, per-kernel census is
launch-regime evidence and the serving A/B is the gate — and the
*corrected mechanism itself* is the hypothesis that determines which
kernels to touch and what residual any design leaves. A single kernel
trace on the **current build** (rocprofv3 `--kernel-trace`, or the P3-4
eager torch-profiler correlation technique) at the 262k config with
~1.5k live context predicts, under F2/F3:

- V2-gather + `quantize_q8_0` together ≈ 10–13 ms/step, scaling with
  `max_model_len` (not with live context); the split between V-zero
  writes and quantize work is the number the design depends on;
- the FA compute kernel ≈ O(live) (tens of µs).

If that trace does not show the split, every design choice in the plan
is in question. The trace is also the baseline for the final serving
A/B. The plan's step 4 (standalone gather micro-bench) cannot substitute:
an Option-1-style tiered design *passes* a dispatch micro-bench (fewer
blocks) while leaving the tail traffic — only the in-model trace plus
serving A/B reveals the residual (F3).

### F7 (major) — §2.1's legality paragraph is wrong and self-contradictory

§2.1 (Lever B) asserts: "grid dimensions computed from a live
GPU/host scalar at replay time are allowed to vary, so long as the value
is read from memory (…) rather than baked in as a
compile-time/graph-capture-time constant." That is false — launch
geometry (grid/block dims and kernel arguments) is frozen at capture on
HIP/CUDA graphs, full stop. The plan states the correct rule itself in
§2.2 ("HIP graph capture bakes the kernel-launch parameters (…) that
were used at the moment of capture — it does not 'compute grid dims
live' on replay"). As written, §2.1 would lead an implementer to build
"compute `Sk_live` live and pass it as the grid size" and be surprised
when it silently re-bakes at capture. Delete or rewrite the paragraph;
it also implicitly argues for a design (live grid) that F5 shows is
unnecessary.

### F8 (moderate) — "no correctness risk" underestimates the boundary hazard

The plan's correctness note for the bounded/tiered designs: "there is
nothing to guard against beyond 'don't dispatch FULL when it doesn't
apply' — a request over the bound simply never enters a FULL graph, so
there is no truncated-context/garbage-output failure mode." For a
tiered/bounded design **with the current kernel body** (which zeroes V
only up to the grid end), a request with live length `L ∈ (tier −
nbatch_fa, tier]` passes the `L ≤ tier` dispatch check, and its FA tail
tile (pre-F4, when zeroing is still believed necessary — and in any
implementation that keeps it) touches V rows `[L, L + nbatch_fa)` that
this step's gather never zeroed → stale buffer contents (fp16 Inf/NaN
possible) → corrupt output; at `L = tier` with a tier-sized buffer, the
read is out of bounds. So:

- **Before F4 is established and gated**, the dispatch invariant must be
  `L + nbatch_fa ≤ tier` with the buffer ≥ `tier + nbatch_fa`.
- **After F4 is gated**, the invariant is exactly `L ≤ captured Sk`
  (FA never reads ≥ `kv_max`).
- Either way, **boundary tests are mandatory and absent from the plan**:
  `L = tier−1 / tier / tier+1` (the last falling back to eager/PIECEWISE
  if a bound exists), across the decode and GQA-packed paths. The
  investigation's Option 3 has the identical hole.

### F9 (moderate) — an extrapolated coefficient is carried as fact

§0 leans on the `Sk`-linear 327 µs@2176 / 72 µs@500 coefficient
(`DEVLOG-fa-attention.md`) to argue the attention cost is
"legitimately O(real_seq_len)". That coefficient (0.152 µs/token/layer)
is a two-point fit the investigation itself flagged as a 40–60×
extrapolation outside its calibration range; applied at 131k it predicts
19.6 ms/**layer** — more than the entire 25 ms step. The attention
kernel is latency-bound at short `Sk` (327 µs at 2176 tokens is ~100×
off the ~3 µs bandwidth peak for that data) and bandwidth-bound at long
`Sk`; the short-`Sk` coefficient does not transfer. Directionally the
plan's claim stands (attention cost is live-proportional, not
capture-artifact), but the plan should not carry the number into design
expectations — measure it (F6 + serving A/B).

### F10 (minor) — secondary observations the plan misses

- **(a) Two-kernel path K-buffer churn.** `forward_paged` passes only
  `v_out` to `gather_paged_kv_fp16`; its `use_or_alloc(k_out=None, …)`
  does a fresh `torch.empty` fp16 `[B, Hkv, Sk, D]` **per FA layer per
  step** in eager mode (128 MB at 131k, 256 MB at 262k, per layer).
  Under FULL capture the allocations are baked into the graph pool
  (reused per-layer, so VRAM cost ≈ peak concurrency, not 16×), but the
  eager long-context path churns device memory per layer per step —
  likely a contributor to the slow eager floor (S8's 19.5 t/s). The R8
  grow-buffer pattern is a cheap one-line-class fix worth bundling with
  whatever lands.
- **(b) The V1→V2 auto-switch is process-global.**
  `static int cached_version` in `launch_gather_paged_kv_q8` is flipped
  to 2 by the *first* `Sk > 65535` call and stays there for the process
  lifetime — a mixed workload (one long request, many short) permanently
  loses V1. Harmless today (V2 is the only option at large Sk), a
  footgun otherwise; the F5 design eliminates the switch.
- **(c) `kv_split` = 16 is decode-only** (force-1 for `seq_q > 2`,
  `gfx906_fa.cpp:264-272`), with a separate split-combine kernel —
  kv_split-sized, not Sk-sized, so it is not part of the tax. Verified;
  the plan's claim is right.
- **(d) Endgame note (one line, not the target).** The paged FA twin
  (`kernel/fattn-q8-paged.cuh`) exists but consumes the Q8 side-buffer —
  i.e. it inherits LEGACY=0's COW corruption class. The true endgame is
  a direct paged FA that quantizes K **in-kernel from the fp16 paged
  cache** (no gather, no side-buffer, no dense buffer at all); it is a
  larger kernel project and the natural follow-up if the F5 design
  lands, not a candidate for this plan.

## 4. What the plan should become (proposed re-anchor)

1. **Step 0 (before any design work, ~half a day): the F6 kernel trace**
   on the current build (262k config, ~1.5k live context, graph mode).
   Confirms/refutes F2+F3 and produces the V-zero / quantize / dispatch
   split — the number that decides how much residual the fallback
   designs leave. Also the A/B baseline.
2. **Target design: the F5 persistent grid-stride fused gather+quantize**
   (fixed small grid, live-bounded work via the `seq_lens` tensor,
   in-kernel quantize, no tail writes), with the F4 tail-write removal
   gated first by the NaN/Inf-tail-injection test. Buffer stays at
   `Sk_pad` capacity; no lifecycle change. One graph, no dispatcher
   change, no cliff.
3. **Fallback if the rewrite stalls (correctness or perf): tiered FULL
   graphs keyed by (batch size × Sk tier)** via the existing
   `BatchDescriptor`/`CudagraphDispatcher` machinery — pure Python, no
   conditional nodes (each tier is an ordinary captured graph; the
   investigation's capture-time `max_seq_len` bound is what differentiates
   the captures). Price in the F8 boundary invariant (pre-F4:
   `L + nbatch_fa ≤ tier`; post-F4: `L ≤ tier`) and the F3 O(tier)
   residual (~0.7–1.1 ms/step at a 32k tier with the current kernel
   body; less if tail writes are dropped).
4. **Option 3 (single bound + eager/PIECEWISE fallback) demoted to a
   short-context mitigation**, with its long-context dismissal citing the
   measured TP=2 eager cliff (S8: 19.5 vs 39.9 t/s; "graphs are
   mandatory", `DEVLOG-tp2-dense.md`) rather than only the
   "regresses for the rest of its life" phrasing.
5. **Keep the plan's gate ladder** (bit-equal capture/replay tests
   extended across tier/bound crossings, micro-bench matrix, PPL/greedy,
   serving A/B at matched context below/above the boundary, and the
   concurrency/mixed-length A/B per the C2-V precedent), and **add**:
   the F4 NaN-tail test, the F8 boundary tests, and the F10(a)
   K-buffer grow fix as a bundled cleanup.

## 5. Disposition of the plan's open questions

| Q | Plan's state | This review |
|---|---|---|
| Q1 (conditional nodes) | blocking, unresolved | **NO** — no conditional-node API in ROCm 7.14 HIP headers (evidence in F1); unreachable from `torch.cuda.graph` capture on any platform. Option 1 unbuildable. |
| Q2 (V2 at 262k A/B) | open | **Moot** at 131k/262k — V2 is already the in-flight kernel (forced by the `gridDim.z` cap, F2). Only relevant for `max_model_len ≤ 65535`, where V1 is the documented 15%-faster default. |
| Q3 (tier thresholds) | open, needs traffic data | Moot under the F5 target (no tiers). Only needed if the §4.3 tiered fallback is built; then yes, real-traffic distribution data. |
| Q4 (fp16/quant launchers) | "skimmed, confirm later" | **Resolved** — `gather_paged_kv_fp16` routes through the same launcher (V2 forced at `Sk > 65535`); and the plan missed the fourth Sk-dependent launch, `quantize_q8_0` (F2). |

## 6. Cross-references

- `plan_masked_fa.md` — the reviewed plan.
- `tp_decode_investigation.md` — diagnosis history; its RESOLUTION
  mechanism (capture bakes `pad32(max_model_len)`) stands; its
  "dense attention" phrasing is what this review sharpens into
  gather+quantize specifics (F2/F3), and its Option 3 carries the F8
  boundary hole.
- `DEVLOG-tp2-dense.md` S5/S8 — serving config
  (`-tp 2 --gpu-memory-utilization 0.93 --max-num-seqs 4
  --max-model-len {131072,262144} --compilation-config
  '{"cudagraph_capture_sizes":[1,2,3,4]}'`), the 8.4 ms/step tax, the
  eager-vs-graph A/B, and the eager TP=2 cliff.
- `DEVLOG-fa-attention.md` — the (extrapolated, F9) `Sk`-linear FA
  numbers and the gather V1/V2/fused-quant history.
- `moe-decode-roadmap.md` §7 (N4), §9.1/9.3 (R2/R3/R8) — roadmap item,
  buffer-lifecycle constraints this review keeps intact.
- `docs/gfx906/AGENTS.md` — gate rules this review applies (serving
  A/B is the gate; census is launch-regime evidence).
