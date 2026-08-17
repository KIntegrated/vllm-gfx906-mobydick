# MoE decode on gfx906 — future optimization roadmap
Copyright Kevin Read <me@kevin-read.com>

Status: roadmap (2026-08-16) — **not a committed plan.** Phase 3 is
declared done at **64.08 t/s** (gap 1.10× vs llama.cpp 70.3 t/s); this
document catalogs the remaining MoE-side headroom so the next phase can
start from measurements instead of memory. Every item carries an
evidence state: **measured** (number from a trace/probe this repo has),
**derived** (arithmetic from measured numbers), or **hypothesis**
(untouched). House protocol applies to anything that gets built:
micro-bench per shape before the model path, PPL/greedy gates, serving
A/B, separate commits, positive AND negative results in the DEVLOG.

Model facts (Qwen3.5-35B-A3B-AWQ, this deployment): 40 layers, all MoE
(E=256, topk=8, hidden 2048, expert w13 N=1024×K=2048 fused gate+up,
w2 N=2048×K=512, AWQ w4a16 128-group), shared expert N=512 (dense
**fp16** — the checkpoint's `modules_to_not_convert` excludes
`shared_expert` — through the LLGemm1/LLMM1/GEMV surface), 30 GDN +
10 FA layers, single-request serving, **no spec decode** (the GDN
empty-core-out gate in v15 relies on that). The checkpoint also
excludes `model.layers.0.` — layer 0's routed experts ship fp16 and
run the unquantized Triton path (C4, O1 resolved).

## 1. Where the MoE time goes (B=1 decode step)

Measured from the post-FA-track rocprofv3 trace (4500 ms / 256-step
window, `/tmp/bench/trace_fa/.../719850_results.db`) and the P3-4 eager
torch-profiler attribution (DEVLOG "P3-4", `/tmp/bench/fillprof_result.txt`):

| component | calls/step | µs/step | notes |
|---|---|---|---|
| `moe_gemm_q4_kernel_gfx906<1,4>` (routed gemm1+gemm2) | 77.7 | **1922** (24.73 µs/call) | BM=1, NPT=4; grid.z 64/16-way K-split + fp16 CAS atomics. (The v14 budget table's 2662 µs is window-amortized and includes the 2.7 ms prefill `<8,2>` calls; 1922 is the decode-step figure.) |
| `topkGating<8,256,...>` | 40.0 | **568** (14.19 µs/call) | softmax+top8 over 256 experts, 1 token |
| `moe_align_block_size_kernel` | 39.0 | **299** (7.66 µs/call) | sorts 8 (token,expert) slots into per-expert blocks |
| `count_and_sort_expert_tokens_kernel` | 39.0 | **178** (4.57 µs/call) | ~at the launch floor |
| `w1_out.zero_()` [8,1024] | 38.7 | **121** | required by the atomic K-splits |
| `output.zero_()` [1,2048] | 38.7 | **113** | required; **aliases** w1_out's memory (see §2) |
| MoE-adjacent `copy_` [1,2048] | ~40 | **~158** (half of the 316 µs/80.3 group) | exact call site not yet pinned (P3-4 left the other half = GEMV pad, now fixed) |
| `fused_moe_kernel.kd` (Triton) | 2.0 | **414** (206.8 µs/call) | **layer 0's routed experts** — checkpoint `modules_to_not_convert` leaves `model.layers.0.` fp16 → unquantized TritonExperts oracle (O1 resolved 2026-08-16; C4) |
| `moe_sum_vec_kernel` | 1.0 | 6 | — |
| **total** | ~245 | **≈ 3.80 ms** | |

Weight-bandwidth floor (derived): per layer the 8 routed experts' W4
weights are 8 × (1.0 MB w13 + 0.5 MB w2) = 12 MB; × 40 layers = 480 MB
per step → **602 µs/step at 798 GB/s** (the P3-4 fill cuts did not move
this; the GEMM kernel is ~3.2× it at 1922). Routing floor: ~40 layers ×
3 kernels × ~4.6 µs launch floor = **~550 µs** if kernels stay separate,
**~200 µs** if fused to one per layer. Zeroings floor: **0** under a
direct-store design. Realistic whole-MoE floor ≈ **0.9–1.0 ms/step**.

**Headroom ≈ 2.7–2.9 ms/step — larger than the entire remaining e2e gap
to llama.cpp (~1.4 ms).** That is the case for a MoE-decode phase; it is
also why nothing in §3 is scheduled yet: the plan's own P2-4 gate was
"gap > 2 ms" and the gap is 1.4 ms, so this roadmap only activates if
the 70 t/s target stays the goal.

## 2. Structural facts (constraints for any design)

1. **Atomic K-splits force two zeroings.** The gemm kernel tiles K via
   `grid.z` (K/32) and accumulates with packed fp16 CAS; both gemms need
   pre-zeroed outputs (121 + 113 µs/step). P3-4 proved the zeroings
   cannot be reordered or merged: `modular_kernel._allocate_buffers`
   aliases `workspace13` (gemm1 out) and the fused `output` in one
   `common_workspace`, so the gemm2 zero must follow gemm1+activation
   over the same memory. Any buffer-lifecycle change must respect the
   aliasing (prefill PPL 6.69 → 1.09e7 when it was violated).
2. **B=1 gives em=8, all distinct experts.** BM=1 is forced (each
   (token,expert) slot is its own group in `sorted_token_ids`); no
   slot-pairing benefit exists at B=1 (that only arrives with
   multi-batch, a separate project per the Phase 3 scope note).
3. **gfx906 has no int8 fast path.** No MFMA, no int8 matrix cores, no
   DP4A (Vega 20 / GCN5). The fp16 `v_dot2_f32_f16` 16B-dot is the
   cheapest dot available; an int8-activation (Q8_1) dot must expand to
   int32 multiply-accumulate — see C6 for the likely-negative verdict.
4. **Graph replay removes CPU launch cost, not GPU dispatch cost.**
   Under FULL_DECODE_ONLY each kernel still takes ~3–5 µs of GPU-side
   dispatch at these sizes (P3-4 measured the fill pile at median
   4.64 µs). "Fewer kernels" is the only launch-side lever; at 99.5%
   GPU busy, only the critical-path share of each removed kernel is
   actually recovered (P3-4 net was +0.7 t/s against ~0.4 ms removed —
   discount all estimates in §3 by a similar factor).
5. **No spec decode** in this deployment: rejected-token padding is not
   a correctness concern (it is for upstream; see the GDN zeros gate).

## 3. Candidates

Gains are per-decode-step, from §1. "Gate" is the measurement that must
pass before code changes (house protocol).

### C1 — Routing pipeline: small-M specialization + fusion (measured 1045 µs)

`topkGating` is 14.19 µs/call for one token over 256 experts — ~3× the
launch floor with trivial work; `moe_align_block_size` is 7.66 µs to
sort **8 items**; `count_and_sort` is already at floor. Options, in
increasing order of surgery:

- (a) **Small-M fast path in each kernel** (M ≤ 32): skip the
  histogram-over-256-buckets / prefix-sum structure; topk via a single
  wavefront, align via straight insertion of 8 (expert, slot) pairs.
  Target: all three ≤ ~5 µs → −350 to −450 µs/step.
- (b) **Fuse topk+align+count into one kernel per layer** (they already
  form a 3-stage pipeline on 8 elements). Target: 40 × ~6 µs = 240 µs →
  −800 µs/step. Needs one output-layout design (the gemm kernel consumes
  `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded`).

Risk: low–medium (standalone kernels, unit-testable in isolation; the
gemm contract is unchanged for (a)). Gate: micro-bench the three kernels
at M=1/8/32/128 with PMC counters (TCC + wavefront occupancy) to confirm
the 14 µs/call is structure, not DRAM latency.

### C2 — Routed-gemm re-tile at decode sizes (measured 1922 µs vs 602 µs floor)

The BM=1/NPT=4 tiling launches 4096 blocks (gemm1: 8 slots × 8 n-tiles ×
64 K-splits, 32 threads each) per layer; each (slot, n-column) cell is
CAS'd by 64 blocks. Variants to micro-bench (isolated bench, per
house rule, before touching the model path):

- V1: **full-K direct store** (grid.z=1): 64 blocks (gemm1), each a
  1-wavefront 2048-K loop. Kills atomics + both zeroings. Risk:
  under-occupancy (64 wavefronts on 60 CUs).
- V2: **in-block K-parallel GEMV tiling** (the classic shape):
  256-thread blocks, 8 wavefronts each take a 256-K slice, LDS
  cross-warp reduce, **one direct store per cell**. Grid ≈ 64 blocks
  (gemm1). Same kill as V1 with 8× the threads per block. This is the
  leading design; it restructures the k-loop (currently fixed 32-wide,
  `BLOCK_KN == THREADS_X` static assert) and the dequant prefetch.
- V3: **2–4-way K-split into fp32 scratch + tiny add**: keeps CAS-free
  stores, adds scratch traffic (2 MB/layer, ~2.5 µs) + one add kernel
  (40 × 4.6 µs). Only worth it if V1/V2 lose on occupancy.
- V4: keep the split, **halve the CAS fan-in** (NPT=8, z=32) — cheap
  ablation to quantify the atomic cost itself.

Target: gemm → 700–900 µs/step (1.2–1.5× floor; the floor is not
reachable at 64–512 blocks/CU) **plus** the 234 µs zeroing kill →
**−1.2 to −1.5 ms/step**. Risk: medium (kernel rewrite of a vendor-ported
but in-repo file, `csrc/rocm/moe_q_gemm_gfx906.cu`; the 12-test suite in
`tests/kernels/moe/test_gfx906_moe_gemm.py` + PPL gate cover it).

### C3 — Fold the zeroings into neighbor kernels (measured 234 µs)

If C2 lands, this is subsumed. If C2 stalls: the `w1_out.zero_()` can be
issued from the tail of the align/topk kernel (disjoint buffers, same
stream, capture-safe) and the `output.zero_()` from the activation
kernel (disjoint from `act_out`; verified in P3-4 that the activation
never writes the common buffer). Saves one launch each ≈ −190 µs/step
(after the critical-path discount of §2.4, realistically −100–150).
Risk: low (no numerics change) but it is cross-kernel surgery in the
align + activation paths.

### C4 — Layer-0 fp16 routed experts (measured 414 µs) — identity RESOLVED

`fused_moe_kernel.kd` (2×/step, 206.8 µs/call) is **layer 0's routed
MoE**: the AWQ checkpoint's `modules_to_not_convert` lists
`model.layers.0.`, so that layer's 256 experts ship fp16 and the
quant-method oracle routes them to the unquantized Triton path
(`unquantized.py: Using TritonExperts MoE backend` in the load log;
`int_wna16.py: Using Gfx906WNA16Experts` covers layers 1–39).
Attribution done 2026-08-16 with the P3-4 method (kernel External id →
`vllm::moe_forward_shared` cpu_op → enclosing layer frame; 114/114
ops in `Qwen3NextSparseMoeBlock_0`, eager trace
`/tmp/bench/fillprof/`). Options: **(a) leave** — 414 µs is ~2.4% of
the 17.5 ms step; **(b) re-quantize layer 0 to AWQ at load** —
calibration-free per-group W4A16 of the fp16 weights for the excluded
modules inside the AWQ loader; removes the only Triton dependency in
the decode path and shrinks layer 0's weight bytes 4×, but changes
layer 0 numerics → PPL gate mandatory, and it is a quant-method
change (upstream-class, nontrivial). An fp16-dense replacement
(gather + per-expert aiter GEMM, like the shared expert) was
estimated at no better than ~200–300 µs for 8 active experts × 2
legs — not a clear win. Verdict: **(b) only if the 70 t/s target is
live**; otherwise (a).

### C5 — Shared-expert chain fusion (derived ~300–400 µs)

The shared expert (dense **fp16**, see header) is three sequential
small GEMMs/acts (w13 [1024,2048] → SiLU·mul → w2 [2048,512]) at B=1;
P3-2(b) put each leg at its GEMM-kernel optimum ("3.6–14× floor rows,
launch/latency bound, no GEMM kernel closes them") — but that
adjudicated the legs separately. Fusing the **chain** (w13+act+w2 in
one kernel with an in-block or grid-sync barrier) removes 2 launches ×
40 layers. At B=1 the intermediate is [1,1024] fp16 = 2 KB —
trivially shareable inside a block or via one barrier. Effort: high
(new fp16 kernel + fused SiLU·mul — simpler than a W4A16 version
since there is no dequant); risk: medium; expected −150 to −250 µs
after the §2.4 discount.

### C6 — Q8_1 activation quant (llama.cpp's decode mechanism) — likely NO on gfx906

llama.cpp quantizes activations to Q8_1 and dots int8×int4 (mmq).
Adopting it here means: (a) a per-layer activation-quant kernel (80 ×
~4 µs ≈ 320 µs **added**), and (b) replacing `v_dot2_f32_f16` (2 fp16
products/instruction, free fp32 accumulate) with int8 products expanded
through int32 MACs — on a part with **no DP4A and no int8 matrix cores**
(§2.3), that is ~2–3× the instruction count per 8 elements. The
activation-byte savings are irrelevant at M=1 (4 KB rows). llama.cpp's
MoE advantage on this box is structural (kernel shape, launch count),
not format: its Q4_K weights are ~12.5% **heavier** in bytes than our
packed AWQ W4 (1.125 MB vs 1.0 MB per expert per leg, 4.5 vs 4.0
bits/elem) yet it still wins overall — the weight format is not its
fast path. **Verdict to record without building: expected net-negative
on gfx906.** Reopen only if C2 lands and the gemm is still > 2× the
weight floor.

### C7 — MoE-block persistent/cooperative kernel (the endgame, ~1 ms launch floor)

One cooperative kernel per layer doing topk → align → gemm1 → act →
gemm2 with grid-wide barriers removes the whole 7-kernel pipeline's
launch floor (~300–400 µs critical-path) and lets the K-splits live in
LDS/L2 without CAS. Feasibility is unproven on gfx906 (HIP cooperative
launch support on Vega 20 must be verified; resident-grid capacity for
the ~4096-block work must be checked). Highest effort, highest risk;
only after C1+C2, and only if the gap target still stands.

### C8 — L2-residency probe for expert weights (informational)

12 MB/layer of W4 weights vs MI50's L2 (a few MB — verify with
`rocagent`/PMC before citing): if weights stream cold from HBM every
step (no cross-step reuse; each layer routes to different experts), the
602 µs floor is HBM-bound and C2's target should be set from HBM
bandwidth measurements, not from the current kernel's L2 behavior. The
P3-0 Q1 TCC_HIT/TCC_MISS mechanism already exists for dense kernels;
repoint it at `moe_gemm_q4`.

### C9 — Multi-stream overlap of shared vs routed (engine-level)

Within a layer the shared-expert chain is independent of the routed
chain; a two-stream (fork/join) capture could overlap ~60 µs of shared
work behind ~49 µs of routed per layer. vLLM captures single-stream;
this is engine surgery, not MoE surgery. Parked — revisit only with a
multi-batch project where the overlap window is larger.

## 4. Recommended sequencing (if the phase is ever started)

1. **Phase 0 — characterization (days, no model-path changes):** C8
   (TCC on the gemm) + C1 gate micro-bench (routing kernels at
   M=1/8/32/128) + C2 ablation bench (V1–V4 at gemm1/gemm2 shapes).
   (C4's identity question was resolved 2026-08-16 — layer 0's fp16
   routed experts; see §6.) Output: a DEVLOG table deciding which of
   C1/C2 is real.
2. **Phase 4a — C1 + C2** (the ~2–2.5 ms pair): routing small-M path or
   fusion, then the gemm re-tile with the zeroing kill riding along.
   Gates per item: isolated micro-bench, 12/12 MoE tests, PPL probe
   (accept the MoE-atomic run noise band, ~0.003 abs), MB greedy heads,
   serving A/B (2 samples each, sequential, §2.4-discounted expectation).
3. **Phase 4b — C5** (shared-expert chain fusion) if 4a nets < +1 ms.
4. **C7** only if the 70 t/s target is still the goal after 4a+4b.
5. **C6 stays rejected on paper** until C2 evidence says otherwise.

Stop conditions: Phase 0 shows the routing/gemm time is DRAM-latency
bound (not structure) → close as "at the floor" with the TCC evidence;
any item netting < +0.3 ms/step measured in serving A/B → reject and
record (P3-4 precedent: 350 µs removed → +0.7 t/s, ~2× discount).

## 5. Open questions

- **O1 (resolved 2026-08-16):** the 2×/step `fused_moe_kernel.kd`
  Triton calls (206.8 µs each) are layer 0's routed experts — the
  checkpoint's `modules_to_not_convert` excludes `model.layers.0.`, so
  they ship fp16 and take the unquantized Triton oracle. See C4.
- **O2:** exact call site of the ~40/step MoE-adjacent `copy_` [1,2048]
  (158 µs/step).
- **O3:** llama.cpp's per-component kernel budget on the same box
  (we have its e2e 14.2 ms/step but never a kernel-level trace) —
  needed to compare "where does it spend 1.4 ms less than us".
- **O4:** MI50 L2 size + expert-weight residency (C8).
- **O5:** does `topkGating`'s 14 µs include an HBM round trip
  (256-expert logits row = 1 KB, likely L2) or pure structure? (C1 gate)

## 6. Phase-2 (MoE) open-items cross-reference

Phase 2 (MoE prefill tuning; see `DEVLOG-moe-opt.md` "PHASE 2") closed
with these items open; where each
lands in this roadmap:

- **P2-4 fused topk+align** (≈1 ms/step routing, high correctness
  risk) → **C1** (routing small-M path or fusion).
- **P2-5 "shared-expert Triton elimination"** (~0.55 ms/step as
  profiled in P2) → premise corrected 2026-08-16 with the O1
  resolution: the P2 profile's "shared expert (Triton fused_moe)" line
  was **layer 0's routed experts** (fp16 unquantized), not the shared
  experts. The shared experts were **already dense fp16** all along
  (`modules_to_not_convert` excludes `shared_expert`) — there was no
  Triton to eliminate. P2-5's residual substance splits into **C4**
  (the layer-0 Triton, 414 µs) and **C5** (shared-expert chain fusion,
  now fp16-fp16 rather than W4A16 — simpler).
- **P2-1(e) persistent-CTA B-in-LDS prefill gemm** (the only path to
  the ~2× prefill gemm goal; a/b/c landed for +26% and stalled at
  ~5.9 TFLOPS ≈ 30% of the practical dot2 peak) → **out of scope for
  this decode roadmap** (prefill). Parked here so it is not lost; the
  DEVLOG "P2-1" section holds the measurements.
- **P2-3 decode-MoE small-M latency** (skipped in P2 on the premise
  "MoE is a small fraction of the step") → premise superseded by the
  Phase 3 budget (§1: MoE ≈ 3.8 ms of a 17.5 ms step); this document
  is the rescope of P2-3.
- **P2-6** (dense GEMMs, decode paged attention, elementwise/norm
  fusion — non-MoE) → became Phase 3's scope (P3-1/2/4); no MoE
  residue.

## 7. Non-MoE items (dense/general; parked from the dense takeover, 2026-08-17)

- **N1 — silence the expected AutoAWQMoEMarlin fallback warning
  (cleanup, expected-on-gfx906).** `auto_awq.py`
  `get_quant_method` (RoutedExperts branch): `check_moe_marlin_supports_layer`
  fails for this checkpoint, so every MoE layer logs
  `Layer '...mlp.experts' is not supported by AutoAWQMoEMarlin. Falling
  back to Moe WNA16 kernels.` — one `logger.warning_once` per layer
  prefix, ~39 lines per engine start. On gfx906 this fallback is
  **expected and is the intended fast path** (Moe WNA16 → the custom
  gfx906 W4A16 kernel); the warning mis-signals a problem. Proposed
  fix: platform gate — on gfx906 (where Marlin W4 MoE is unavailable)
  emit a single info/debug line; keep the per-layer warning on other
  platforms where the fallback may be a genuine surprise. Log-only
  change, but still run the PPL + serving sanity gate (behavior must
  not change). Evidence state: **measured** (site pinned;
  `vllm/model_executor/layers/quantization/auto_awq.py`).
- **N2 — FA B>1 decode direct kernel store (derived ~192 KB/layer at
  B=8).** After the 2026-08-17 native-BSHD output work, B=1 decode has
  zero output-path copies; B>1 decode still pays one reshape copy per
  layer (the [B,Hq,D] row block extracted from BSHD is 192 KB at B=8).
  The real fix is a decode-specialized kernel store: when the real
  seq_q == 1, allocate [B,Hq,D] (no Sq dim), store only the j==0 column
  into (b,h) rows, and make kv_split partials [B,Hq,kv_split,D]. The
  kernel needs a `decode_single` flag — the launcher sees Sq_pad=2,
  not the real seq_q, so it cannot infer this today. Only matters for
  batched decode (the production bench is single-request); B=1 is
  already copy-free. Evidence state: **derived** (per-call probe +
  stride analysis in the DEVLOG "FA decode per-layer copy pile").
- **N3 — GDN [3,1,32] state-bookkeeping copies (measured 32/step,
  ~180 µs/step eager).** The timeline probe
  (`/tmp/bench/dense_ewp_timeline.py`) attributes the 32 [3,1,32]
  copies/step (~180 µs, launch-latency-bound) to upstream vLLM
  mamba/GDN state management around `_causal_conv1d_update` +
  `fused_recurrent_gated_delta_rule` — not FA, not model code. A fix
  means reducing the count in upstream state handling or folding it
  into the GDN core custom op; deferred because it is upstream code,
  small, and the production graph path amortizes the launch cost.
  Evidence state: **measured** (attribution); fix = **hypothesis**.

## 8. Upstream candidates (platform-generic fixes to offer vllm-project/vllm)

Three landed changes in this branch are not gfx906-specific. If they are
ever offered upstream (rebased onto `vllm-project/vllm` main, per vLLM's
contribution policy: human submitter who can defend the change, tests,
AI-assistance disclosure), these are the items:

- **U1 — fastsafetensors GDS fallback** —
  `vllm/model_executor/model_loader/weight_utils.py` (`128e948baf`).
  The GDS→non-GDS fallback only catches `RuntimeError`, but
  fastsafetensors raises a bare `Exception` when GDS reads fail
  (cuFileRead errno 22 on unsupported systems) — engine death instead
  of fallback. Broadening the catch (keeping the `"gds" in str(e)` +
  not-yet-yielded guards) gives 2.6× faster loads where GDS works and
  a working boot where it doesn't. One-line diff; evidence state:
  **measured** (41 s vs 117 s load, DEVLOG "Local-venv bench
  environment").
- **U2 — hipify in-source build guard** — `cmake/hipify.py` (part of
  `225448d93f`). `shutil.copytree(project_dir, output_dir)` raises
  `SameFileError` when both are the same directory (in-source rebuilds
  on Py3.12); guard with an `abspath` compare. Two-line diff; evidence
  state: **measured** (repro'd the crash, DEVLOG "Build-system note:
  hipify.py in-source guard").
- **U3 — GemmaRMSNorm fused-kernel dispatch** —
  `vllm/model_executor/layers/layernorm.py` (`19c1d41cf5`,
  `70ec1d0e79`). `forward_cuda` delegates to `forward_native`, whose
  fp32 `(1 + w)` breaks the fused `vllm_c` rms-norm kernels'
  `weight.dtype == x.dtype` requirement → decomposed elementwise
  fallback (~131 extra launches/step on Qwen3.5-family hybrids in
  eager mode). Gemma's `(1 + w)` factorization is a plain scaled RMS
  norm with `w' = 1 + w` in the input dtype; dispatching with that
  (plus a per-instance `(1 + w)` cache keyed on
  data_ptr/dtype/device/_version) restores the fused path. PPL-gated
  on two models. Evidence state: **measured** (unit maxdiff 0.002,
  PPL in band both models). Caveat for upstream reception: the
  benefit is eager-path only — inductor fuses the decomposition in
  compiled mode.

## 9. Code-review items (parked from the pre-merge review, 2026-08-17)

The pre-merge code review of the branch (`gfx906_qwen_impr_code_rev_qwen.md`
in `docs/gfx906/` plus the combined review at the repo root) resolved its
four severity findings in `01499157a8` (symmetric W4A16 repack crash +
wrong zp fill, GDN platform gate, oracle GPTQ exclusion, `top_k` guard).
The remainder is parked here. None of it affects the default
configuration (LEGACY=1, gfx906 build); it is edge-path correctness,
cross-platform hygiene, and cleanup. IDs are R*, severities as
P2/P3/P4 per the review. House rule applies: P3/P4 items get bundled
with the next substantive touch of the affected file, not their own
commit.

### 9.1 Edge-path correctness (fix before enabling the affected mode)

- **R1 — direct-paged path builds an fp16 Q buffer; the C API requires
  fp32 (P2).** `forward_paged`'s direct branch checks
  `q_pad_buf.dtype == query.dtype`, but `q_pad_buf` is always allocated
  fp32 → the fallback `torch.zeros(..., dtype=query.dtype)` is fp16 →
  `forward_paged_direct`'s `TORCH_CHECK` fails. Reachable only under
  `GFX906_FA_LEGACY=0` + B≥2/Sq≤16 (the experimental Q8 side-buffer
  mode); dormant in the default configuration, but the documented
  `GFX906_FA_DIRECT_PAGED` A/B knobs would hit it. One-line fix: use
  the fp32 buffer / drop the dtype condition. Evidence: **verified**
  (code walk; probe-able via the knobs in `running.md`).
- **R2 — LEGACY=0 + prefix caching logs an error but continues (P3).**
  `get_cudagraph_support` logs the corruption warning (the Q8
  side-buffer lags the fp16 cache during warmup/COW/graph-replay
  writes) but keeps the backend. Should fail closed (raise, behind an
  explicit override if a diagnostic use case exists) — the README
  already says "do not use"; the code should agree.
- **R3 — `_ensure_gather_buffers` retired-buffer growth is unbounded in
  LEGACY=0 (P3).** Exact-shape-match realloc plus a retired list that
  keeps the old tensors alive: every `Sk_pad` growth after the first
  capture (decode grows it in 32-token steps over long contexts)
  permanently retains the previous K+V pair (tens of MiB each at
  serving shapes). Inert in the default LEGACY=1 (buffers are `None`).
  Fix: grow-only capacity buffer with an exact-size view.
- **R4 — MoE kernel caller contract is partially unenforced (P3).**
  `moe_gptq_gemm_gfx906` checks dims/dtypes/`N % 4` (and `top_k > 0`
  since `01499157a8`) but not `K % 8`, `groups | K`, or `N % 8` (the
  qzeros row width); the oracle-side shape gates are deferred (the
  repack's layout detection rejects unrecognized shapes loudly — the
  observed failure mode for exotic shapes is a load-time `ValueError`,
  not silent miscomputation). Add the cheap `TORCH_CHECK` group and the
  oracle divisibility check when the kernel/oracle is next touched.

### 9.2 Cross-platform / build hygiene (fix before the fork is built
or run on non-gfx906 hardware)

- **R5 — plugin entry point logs a full traceback on every non-gfx906
  vLLM startup (P2).** The pyproject registers `gfx906_fa = vllm.gfx906_fa
  .gfx906_fa_backend:register` unconditionally; importing that module
  hard-imports the gfx906-only `_gfx906_fa_C` extension. The plugin
  loader catches it (graceful: backend simply absent) but via
  `logger.exception` → a traceback in every CUDA/non-gfx906 startup
  log. The explicit-registration path in `platforms/rocm.py` already
  has the `try/except ImportError`; the plugin path doesn't. Fix: make
  `register()` import-tolerant.
- **R6 — setup.py requests a CMake target that may not exist (P2).**
  `_targets_gfx906()` returns True when `PYTORCH_ROCM_ARCH` is *empty*
  (auto-detect), but the `_gfx906_fa_C` CMake target is defined only
  when `VLLM_GPU_ARCHES` matches gfx906. On a non-gfx906 ROCm machine
  with arch auto-detection the build fails on a nonexistent target.
  Fix: append the extension only when `"gfx906" in rocm_arch` (the
  CMake gate is the source of truth).

### 9.3 Cleanup / debt (bundle with the next touch of the file)

- **R7 — ncols1 tile ladder copied three times (P3).** The
  `Sq>32→64 … Sq≤2→2` ladder lives in `forward_paged` (Python), the C
  launcher dispatch, and the C kv_max expansion (annotated "keep in
  sync!"). Any ladder change (e.g. a new NC2 cap) must hit all three;
  a divergence mis-pads Q or mis-expands kv_max. Consolidate or pin
  the mapping with a unit test.
- **R8 — `gather_paged_kv_quantized` allocates fresh outputs per call
  (P3).** Unlike its two sibling gather paths it has no `use_or_alloc`
  grow-buffer parameter; the fused-quant gather is the one actually
  used on the default LEGACY decode path. Benign under graph capture
  (private pool) and cheap in eager (caching allocator); inconsistent
  with the VRAM-spike rationale documented for the siblings.
- **R9 — MoE `apply()` aliasing order dependency undocumented (P3).**
  `workspace13`/`fused_out` alias one `common_workspace` (§2.1); the
  sequence gemm1 → activation → `output.zero_()` → gemm2 is load-
  bearing (an earlier zero would wipe `w1_out`). Correct today; one
  reorder away from silent corruption. Add the warning comment (or
  split the buffer for this experts class).
- **R10 — stale docs/comments (P4).** The `kchunk 512|2048|4096`
  docstrings in `csrc/rocm/ops.h` + `vllm/_custom_ops.py` omit 1024
  (a supported and *used* value — K=17408 down_proj); the
  `forward_paged_direct` pybind help says output `[B, Hq, Sq, D]`, it
  is native BSHD `[B, Sq, Hq, D]`; the launcher header's "TODO: vLLM
  block table" is implemented; the MoE kernel header's `grid = (…, 
  N/1024, …)` is only true for NPT=4 (NPT=2 → N/512); `/tmp/bench/…`
  references in code comments (volatile); the dead `mask_buf=None`
  parameter on `forward_paged` (Level-3a leftover).
- **R11 — lint debt in vendored/bench files (P4).** ~63 ruff errors in
  `vllm/gfx906_fa/*.py`, `benchmarks/kernels/gfx906/*`, and the docs
  bench scripts (B023/E501/F821 in the timeit-closure bench patterns;
  the F821s are outer-scope names, verified harmless); pre-existing
  E501s + ruff-format drift in `utils.py`. Pre-commit will flag these
  if the files are restaged — a `per-file-ignores` entry or a cleanup
  pass, bundled, not standalone.
- **R12 — debug env knobs (P4).** 8 debug-only `GFX906_FA_*` switches
  (`_DOUBLE_CHECK`, `_DUMP`, `_FUSED`, `_FWD_DEBUG`, `_NO_BUF_REUSE`,
  `_QPAD_EMPTY`, `_TORCH_GATHER`, `_ZERO_KTAIL`) — candidates for a
  single `GFX906_FA_DEBUG=1` master switch; all are read once at
  import and documented in the README knob table.

## Appendix A — `moe_gemm_q4_kernel_gfx906` facts (csrc/rocm/moe_q_gemm_gfx906.cu)

- Template `<BM, NPT>` with instantiations `<1,4>`, `<4,4>`, `<8,2>`;
  BM chosen by `em = M×topk`: ≤32 → 1, ≤512 → 4, else 8 (decode B=1:
  em=8 → BM=1).
- `BLOCK_KN_SIZE = THREADS_X = 32` (static assert); block covers
  `NPT×32` output columns for `BM` slots of **one expert** (expert from
  `expert_ids[blockIdx.x]`; slots from `sorted_token_ids`).
- Grid: x = token-blocks, y = N/(32·NPT), z = K/32. K is **not** looped
  per block beyond its 32-wide slice — grid.z tiles K and the epilogue
  CAS-accumulates (`atomic_add_pk4_f16` / `pk2`, packed 64/32-bit CAS
  loops) into the pre-zeroed output. **Neither gemm can direct-store in
  the current tiling.**
- gemm1 (`output_topk=0`): out row = slot id, [em, N]; gemm2
  (`output_topk=topk`): out row = token id, [M, K], with the topk
  router weight fused into the epilogue (no separate reduce kernel;
  `moe_sum` is unused on this path).
- Weights: packed int32 `[E, K/8, N]` (exllama shuffle) + fp16 scales
  `[E, groups, N]` + packed zeros `[E, groups, N/8]`; per-column dequant
  constants (`z1z16/y1y16`) refreshed at group boundaries (128-group
  AWQ); `zero_offset=0` for AWQ (GPTQ-v1 needs +1).
- Activations stage through LDS (`block_a[BM][32+8]`, 16B-padded for
  `ds_read_b128`); double-buffered weight prefetch was tried and
  rejected (256 VGPRs + spills at BM=16, 5× slower — DEVLOG P2-1).

## Appendix B — bench / probe recipes

- Serving A/B (local venv, sequential): the standard recipe in
  `README.md` §Bench recipes (`_b.py`, pp=2048/tg=256,
  util 0.95, fastsafetensors, FULL_DECODE_ONLY); check `uptime` first.
- MoE kernel tests: `tests/kernels/moe/test_gfx906_moe_gemm.py`
  (12 tests, covers both source layouts).
- PPL probe: `/tmp/bench/ppl_probe.py` (prefill logprobs; the AWQ
  gemm2 atomics make it non-deterministic at the ~0.003-abs level across
  runs of identical code — compare against that band).
- Isolated micro-benches live in `benchmarks/kernels/gfx906/`
  (gemv, fa, gather, llmm1); a MoE-decode variant bench (em ∈ {8, 32},
  tiling variants) does not exist yet — Phase 0 item.
- Kernel trace: `rocprofv3 --kernel-trace` → SQLite (`agg_db.py`);
  remember the ~10–15% per-dispatch inflation under the tracer and the
  untrustworthy grid-axis columns (use timestamps/durations only).
- Eager torch-profiler attribution (fill/copy method):
  the eager torch-profiler correlation method documented in
  `DEVLOG-moe-opt.md` "P3-4" (probe script since removed) — the same technique
  answers O1/O2/O5.

## Appendix C — file map

- `csrc/rocm/moe_q_gemm_gfx906.cu` — the routed gemm (ours, Phase 1/2).
- `vllm/model_executor/layers/fused_moe/experts/gfx906_w4a16_moe.py` —
  the expert class (apply: zero_ → gemm1 → activation → gemm2;
  workspace lifecycle via `_resize_cache`).
- `vllm/model_executor/layers/fused_moe/modular_kernel.py` —
  `_allocate_buffers` (**common_workspace aliasing**, §2.1).
- `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` +
  `csrc/moe/*` (v1) — routing kernels (C1 targets).
- `vllm/model_executor/layers/utils.py` — `_llmm1_tiny_m` (dense-side
  dispatch incl. the shared expert; P3-2(b)/P3-4 GEMV m==1).
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — GDN
  (context only; its zeros gate is the §2.5 spec-decode precedent).

## Appendix D — pitfalls (from P3-4 and earlier)

- **Alias the buffers and the PPL detonates** (workspace13 == output;
  §2.1). Any "reorder the zero" idea must be re-derived against
  `_allocate_buffers` first.
- **Atomic K-splits** rule out direct stores in the current tiling
  (C2 must re-tile, not patch the epilogue).
- **No int8 fast path** on gfx906 (§2.3) — Q8_1-activation proposals
  should be rejected at review without new hardware evidence (C6).
- **BM=1 at B=1** — any tiling that assumes ≥2 same-expert slots per
  block is dead at B=1 (multi-batch only).
- **Critical-path discount** (§2.4): at 99.5% GPU busy, kernel-time
  savings convert to e2e at roughly half the arithmetic sum (P3-4:
  ~0.4 ms removed → +0.7 t/s of the possible ~1.6).
- **rocprofv3** grid-axis columns untrustworthy; kernel-trace inflates
  per-dispatch times 10–15%; use timestamps/durations.
- **PPL run-to-run noise** (~0.003 abs) from the fp16 atomic-add order
  — do not chase it as a regression signal.
