# Dev Log — gfx906 Speculative Decoding (dense 27B)

> Branch: `gfx906/spec-decode`. The 27B Phase-1 spec log (the MoE
> port is DEVLOG-moe-spec-decode.md).
> Copyright Kevin Read <me@kevin-read.com>.

Top-line state: **mtp2 (k=2) is the recommended spec method for
Qwen3.5-27B — 1.503× (39.74 t/s, 1.819 tok/step, 90.95% draft
acceptance, token-identical on agentic prompts)**; ngram3 stands at
1.094× (ceiling — agentic acceptance is the limit); the branch
default build is the per-M q_gemm max-ilp **split** (M=1 max-ilp,
M≥2 unflagged). The TP=2 mtp2 "regression" of 2026-08-22 was
**retracted** — host-state degradation, not code.

## 2026-08-18 — n-gram spec decode probe (dense 27B)

> Moved here from DEVLOG-moe-opt.md (topic consolidation).

Question: does n-gram spec (free draft) beat no-spec on the
weight-bound dense 27B? Setup: server, max_num_seqs 4, maxlen 2816,
util 0.95, greedy, 3 agentic-coding prompts × 512 tok; arms baseline
vs ngram k=3 min2/max5; acceptance from Prometheus deltas.

**VERDICT: no speed-up as configured — 0.68× (DEAD-END for that
config).** baseline 27.46 (26.81/27.96/27.62); ngram first attempt
22.99 (artifact — engine demoted to PIECEWISE, fixed below); ngram
after FA fixes **18.73**. Drafts on ~40% of steps, 1.08 accepted per
draft step. Outputs diverge at first near-tie token (benign fp argmax
flips, q4 vs q1 paths — S3-class).

**Blocker 1 (fixed in-tree, prerequisite for ANY spec on this
backend):** `Gfx906FABackend` declared `UNIFORM_SINGLE_TOKEN_DECODE`,
so vLLM never FULL-graphed spec decodes (needs support ≥
`UNIFORM_BATCH`). Fix: `UNIFORM_BATCH` when
`num_speculative_tokens > 0` + capture-safe uniform fast paths in
`forward_paged` (the Q-scatter/out-gather had `int(cu_seqlens_q[...])`
D2H syncs, illegal under capture; gated on the host identity
`num_tokens == num_seqs * max_seqlen_q`). Spec steps now FULL-graphed
(capture 2 s vs 44 s).

**Blocker 2 (as then understood): draft-step overhead** — profiler
diff attributed it to (a) GDN spec path using the chunk (prefill)
kernel family: 48 → 1536 calls, "~800 µs/layer" (double-counted — see
B1 below), (b) ~6× copy_ + index_select/index_put bookkeeping.
**Refuted for single-request the same day** (see "Fork in the
road").

**Blocker 3: no-draft steps** — at 40% draft rate the 60% no-draft
steps ran at piecewise cost. Ceiling math at the time: ~1.19× with
draft steps at ~55 ms and q1 FULL graphs for no-draft steps.

## 2026-08-18 — cost model reconciled, Phase 0 arms, revised plan

**B1 reconciliation:** `ChunkGatedDeltaRuleFunction` 637.7 ms / 1536
= **415.2 µs/layer/multi-token step** — the child kernels (h_blockdim64
312.5 + chunk_fwd_o 141.8 + kkt 68.8 + recompute_w_u 82.0) are *inside*
that total; the roadmap's "~800 µs/layer" double-counted wrapper +
children. **B1 = 415.2 × 48 = 19.9 ms/draft step** (+0.66 ms
fused_sigmoid_gating). B3 bookkeeping ≈ 10 ms from the 2-seq run
(later revised to ~2 single-seq). Measured draft step ≈ 82 ms =
base-4-tok ~47 + B1 20 + B3 10 + slack.

**Fork in the road: the B1 attribution was wrong for single-request.**
Kernel-path spy (`benchmarks/kernels/gfx906/gdn_step_spy.py`, wraps
the four GDN kernel entry points + conv): clean single-prompt spec run
showed draft steps already on `fused_sigmoid_gating_delta_rule_update`
(sequential FLA, 2D per-token state-slot indices,
`num_accepted_tokens` — the align-slot rollback is already wired) at
~32 µs/layer ≈ **1.2 ms/step. GDN is NOT the single-request
draft-step bottleneck.** Where the chunk kernel actually bites: the
metadata builder reclassifies non-spec 1-token seqs as *prefill* when
a batch mixes spec and non-spec rows (~20 ms/step waste per no-draft
seq) — a **multi-request mixed-batch pathology** → roadmap W1. Spy
pitfalls: (1) CUDA graphs hide steps from Python spies (replay runs
the captured list; the spy only saw capture) → `enforce_eager=True`
for spy runs; (2) the conv anchor double-counts on mixed batches
(96 convs/step) — single-seq runs only; (3) exit-134 teardown hang →
`os._exit(0)` after the report.

**Phase 0 results (all arms, `spec_ngram_dense.py --repeats 3`,
baseline band 26.81–27.96):**

| arm | mean t/s | acc/draft-step | verdict |
|---|---|---|---|
| ngram_gpu k3 min2/max5 | 18.29 | 0.428 | **REJECTED** — GPU proposer is a reimplementation whose n-gram tie-breaking picks different, worse tokens (same draft rate as CPU, output SHAs diverge) — match-selection bug, not config (roadmap L3 sub-item) |
| ngram k3 min1/max5 | **19.40** | 0.71 | best spec arm, 0.71× |
| suffix | — | — | deferred: `arctic-inference==0.1.1` sdist; dynamic draft length → PIECEWISE-only |

**Revised per-step breakdown (in-proc profiler pair, spec = 36 draft
+ 7 no-draft):** AWQ gptq_gemm 18.9 → 44.6 ms (+25.7) · fp16
projections ~5 → ~35 ms raw triton_matmul (inductor shape-guard
cliff: M=1 fusions don't apply at M=4; rocBLAS Cijk would cost ~6 ms
aggregate) (+~30) · GDN +0.5 · FA +0.2 · B3 +~1.5 · CPU proposer ~5 ·
wall 36.5 → ~95 ms. **Calibration: predicted 18.7 vs measured 19.40
t/s (4%) — the cost model is trusted.**

**Prompt sensitivity (new finding):** on a maximally repetitive
prompt spec already WINS 1.12× with zero kernel work; the agentic
loss is entirely the low-repetition case.

**Levers (roadmap restructured):** L2 AWQ M≤4 (in-engine +26 ms;
kernel-only ~5 — the rest launch/K-split… later refuted by the census)
· L3 proposer (~5 ms) · L4 q1 FULL graphs for no-draft steps · L5 B3
(~2 ms) · **W4 (new, general decode opt): the fp16 family (~90
GEMMs/step) runs triton_matmul ~19 ms/step at EVERY M (weight-bound,
M-invariant 174 µs on fa_q) — and triton BEATS rocBLAS on MI50 (174
vs 340 µs), so old-L1 "route to rocBLAS" is DEAD; the win is a custom
skinny fp16 GEMM, ~15 ms off every decode step, ratio-neutral** (became
its own branch — DEVLOG-fp16-skinny.md) · old Phase 1 (GDN small-M
kernel) **CLOSED — already exists** (the fork-in-the-road finding).

**GEMM census (`gemm_step_census.py`, eager single prompt):** AWQ
call structure is **M-invariant** (236/step nospec vs 231 spec, the
same four shapes) — the "263 vs 234" in-engine gap was profiler
warmup/capture contamination; **no launch inflation exists**, the M=4
AWQ delta is pure per-call time ~17 ms. fp16 is **M-gated in the
dispatcher**: at M=1 the projections take the GEMV op path
(`triton_matmul` count 0), at M=4 triton_matmul — M=1 ~3.5 ms (GEMV)
→ M=4 ~16.5 ms: **delta ~13 ms**. Layer 0 is entirely fp16.

**Phase 1 final shape:** L1' = extend `dense_gemv_gfx906` (W16A16
GEMV, `<RPT,KCHUNK>`) to M≤4 rows (same weight-read-once structure) —
kills ~13 ms/draft step; L2 = 4-row AWQ GEMV or q_gemm re-tile —
kills ~17 ms.

## 2026-08-18 — L1' built + measured WIN (SHIPPED)

Kernel `dense_gemv_m_kernel` (+`dense_gemv_m4_gfx906` op):
row-parallel weight structure identical to the M=1 GEMV; per-M-row
x-slices in registers (x ≤ 40 KB, L2-resident across blocks);
`acc[RPT][4]` fp32, RPT ∈ {2,4} (the packed-CAS epilogue needs
adjacent rows — M=1-style RPT=1 packing impossible: per-M-row outputs
are N apart); ksplit>1 → per-M-row packed CAS (32/64-bit), lane-0
gather shfl (1 warp) or LDS (multi); launcher requires N % RPT == 0
(kills the ragged-tail OOB class). KCHUNK ∈ {512,1024,2048}, RPT via
`VLLM_GFX906_GEMVM_RPT` (default 2). **M=1 path untouched** (dispatch
gate 2 ≤ M ≤ 4, gfx906 + fp16 + k%8==0 + N%2==0 + K divisible;
excludes the tuned hipBLAS 5120×[2048..2304] special case; kill
switch `VLLM_GFX906_SPEC_GEMM=0`) → nospec regression structurally
zero.

Micro-bench (census fp16 shapes, per draft step): M=1 ref (GEMV)
6.9 ms · triton_matmul M=4 **25.6 ms** · m4 (kc1024, RPT=2) **11.2**
· RPT=4 11.0 — **saving ~14.5 ms/step** at the M=4 worst case (larger
at the typical draft M≈1.7–2). M=4 runs 1.5–1.6× M=1 cost (4× dot
work — ALU-bound, as modeled). kc1024 > kc512 every shape; RPT=4
wins ~2% overall but RPT=2 wins at N=96 → default stays 2.

**In-engine (step-wall probe, repetitive prompt, eager):** nospec
24.9 t/s (35.2/37.3 ms) · spec triton 36.2 (draft 46.9/66.6 ms) ·
**spec m4 43.7 t/s (draft 41.9/53.2 ms)** — **13.4 ms/draft step,
matches the microbench**. `AMD_SERIALIZE_KERNEL=3` inflates walls
~2.2× — identify-only, never for timing.

**L2 re-scoped down:** the AWQ family is **dequant-ALU-bound, not
weight-read-bound** (M=1 q_gemm runs 44 MB in 75 µs = 590 GB/s, ~2×
off roofline; the tiled M=4 kernel already shares dequant across
m-rows) → an M≤4 AWQ GEMV removes only atomics/LDS/tiling overhead:
**2–8 ms/step, not the census's 17**. Deferred as low-ROI polish.

## 2026-08-18 — ROCm reinstall, GPU wedge #1, max-ilp disabled

ROCm moved to a Debian-packaged install (`/opt/rocm`, 7.14.60850);
clean rebuild (stale .so's had the old RUNPATH baked in). New .so's
carry **no RUNPATH** — runtime resolution is the env script's
LD_LIBRARY_PATH. The optional vllm-rs Rust CLI build fails
(pre-existing; tolerated).

Weight loading of the 27B AWQ then faulted the GPU twice
(`hipErrorLaunchFailure` mid-shard-copy, sticky; BACO self-recovered
once, wedged the driver once → reboot). Intermittent (2 hits, then
clean). Only compile difference in the load path: `q_gemm.hip` was
the single file built with per-file LLVM **max-ilp** — suspected
faulting schedule under the new LLVM. **Decision: `VLLM_NO_MAX_ILP=1`
rebuild** (all 4 max-ilp files unflagged); 4 clean full runs since.
Cost: q_gemm +15–26%, FA decode −2–5% → absolute numbers on this
build ~1–2 t/s below max-ilp records; A/B ratios within one build
stay valid. (Resolved properly by the per-M split below.)

Env note: `flash_attn` needs no rebuild — with
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (mandatory for vLLM ROCm)
the pure-Triton backend is selected and the missing compiled
`flash_attn_2_cuda` .so is never touched.

## 2026-08-18 — serving A/B + L5 root-caused and fixed (SHIPPED)

Serving A/B (3 agentic prompts × 3 repeats, per-arm server,
`--served-model-name` required or the client 404s): baseline 26.52 ·
ngram3 (L1') 25.06 = **0.945×** — L1' recovered most of the Phase 0
gap (0.71× → 0.945×) but agentic still loses. Counters: a = 1.09,
37% of steps are drafts. Back-solving step walls: **no-draft spec
step ≈ 48 ms vs nospec 35–37 — a ~13 ms penalty on 63% of steps;
the dominant agentic loss is the no-draft step, not the draft step.**

**L5 root cause (pure Python, no build):**
`compilation.adjust_cudagraph_sizes_for_spec_decode` rounds ALL
capture sizes up to multiples of `uniform_decode_query_len` (= 4 for
ngram-3 — the upstream stopgap for issue 28207). The PIECEWISE key
set loses sizes 1–3, so a no-draft spec step (1 token, non-uniform →
PIECEWISE) pads to the size-4 graph and runs every GEMM at M=4
(~8 ms AWQ + ~6 ms fp16 ≈ the observed 13 ms). Evidence: (a)
batchdesc spy — no-draft steps dispatched `num_tokens=4 uniform=False
PIECEWISE` despite num_tokens=1; (b) graph-mode profile showed only
gptq `<true,4>` kernels, eager showed both; (c) the graph-only
"Call CompiledFxGraph" rows = the no-draft PIECEWISE steps' ~9
inductor fragments each.

**Fix (`VLLM_GFX906_SPEC_CG_SMALL`, default on, gfx906-only):** after
the rounding, re-add sizes 1..q-1. FULL keys already filter sizes < q
→ only small PIECEWISE graphs added. Verified: no-draft steps
dispatch `num_tokens=1 PIECEWISE`; profile shows gptq `<true,1>`
(1652 calls @ 90.6 µs, was absent) + M=1 GEMV kernels.

**L1'' (2026-08-24) — the dispatch gap was an artifact:** the
graph-mode "FxGraph 7.3 ms/step" row (L1' build) was NOT a missed m4
dispatch — an eager dispatch spy shows every decode fp16 GEMM
reaches the dispatcher (incl. the LM head n=4, m=248320, k=5120;
m4 handles the whole M=4 set, 400 launches/step) — the FxGraph row
is the no-draft steps' inductor piecewise fragments (see L5). Leftover
(optional, low priority): extend the m4 dispatch to the K=1024 gate
shapes (m=128, k=1024: ~0.7 → 0.25 ms each, ~2 ms/step),
dispatch-only.

**L5 serving A/B (2026-08-24, post-reboot):** baseline 26.53 ·
ngram3 **28.58 = 1.077×** (was 0.945×) — +3.5 t/s from the fix alone
(no-draft 48 → ~40 ms; the size-1 graph still pays ~9 small
fragment replays). **ngram ceiling:** the 1.15× all-in gate is NOT
reachable with ngram on agentic prompts (even full L2 → ~1.09×);
agentic acceptance (1.09/step) is the ceiling — a model-based drafter
is the remaining lever.

## 2026-08-19 — max-ilp split: per-M q_gemm scheduling (SHIPPED, branch default)

The max-ilp question resolved by compiling the 4-bit kernel **twice**
— M=1 with `-amdgpu-sched-strategy=max-ilp`, M≥2 without — routed on
m_count at `gemm_half_q_half_cuda_part`. Measured (AWQ M-scaling, 27B
shapes): M=1 max-ilp wins 19–24% (down 80 vs 95, gate 78 vs 103,
fa_q 29 vs 36 µs); M≥2 no-max wins 14–23%; gdn/fa_q 3072-row shapes
~neutral. Spec steps are M=3/4-dominant → the split is strictly
better than either uniform choice.

Implementation: new TU `q_gemm_m1_maxilp.cu` (~149 lines of
duplication: 140-line kernel + helper + launcher; the zero-dup
macro-header alternative was rejected — the copy can't perturb the
shared file). Both copies carry **SYNCHRONIZATION WARNING** headers +
SYNC-COPY markers at both kernel and dot-helper sites (a one-sided
edit silently changes M=1 numerics with no compile/test signal).
CMake: the flag moves off `q_gemm.hip` onto the twin. Kill switch
`VLLM_GFX906_QGEMM_M1_MAXILP=0`.

**Debug saga (all gfx906/stable-ABI landmines worth knowing):**
(1) missing extern decl → link error; (2) `__gfx906__` is **NOT
defined in the stable-ABI host compile** → the whole twin TU
compiled to nothing (0 m1mi symbols) — convert to a **runtime arch
guard**; (3) the guard via ATen device properties #errors in
stable-ABI builds → plain `cudaGetDeviceProperties` (hipified);
(4) `gcnArchName` here is **`gfx906:sramecc+:xnack+`** (suffixed) →
`== "gfx906"` fails — prefix match (same convention as
skinny_gemms.cu). Verified: M=1 → `..._m1mi<1>`, M=4 →
`..._kernel<true,4>`; build.ninja flags exactly {m1mi, skinny ×2, fa
×3}.

**Verification (3-arm, 3×3, VRAM-safe runner):**

| arm | no-max | full max-ilp | **split (default)** |
|---|---|---|---|
| baseline | 26.44 | 28.56 | **27.99** (+5.9% vs no-max) |
| ngram3 | 28.92 (1.094×) | 27.80 (0.973×) | **28.03 (1.002×)** |
| mtp2 | 39.74 (1.503×) | 36.67 (1.284×) | **39.37 (1.407×)** |

Full max-ilp is the wrong default (mtp2 −7.8%); the split keeps the
spec workload at its best level while plain serving gets the M=1
gain. PPL gate: twin ON 15.9730 vs OFF 16.0450 (0.07, 0 top-20
misses — S3-class). Dense 27B 4-seq 25.32 (recovers the full max-ilp
dense number). Unit 12 passed / 1 pre-existing fail (test_auto_awq
Qwen2-1.5B, head_dim=64 unsupported by the custom FA launcher).
**Open (not chased):** down_proj M=4 115 vs 105 µs on the split build
(same unflagged binary — suspected I-cache/constant-memory effect of
the twin's presence or environmental; serving impact ≤3%, CIs
overlap).

## 2026-08-19 — Prefill/TTFT A/B: MTP prefill cost, cache behavior, one OOM bug

`benchmarks/kernels/gfx906/prefill_ttft_probe.py`: sequential
max_tokens=1, warmup on a distinct prompt absorbs one-time costs.
Block sizes (mamba_cache_mode=align): baseline 784, **mtp2 800**.

1. **MTP solo-prefill cost: negligible** (249–254 vs 250–258 tok/s).
2. **The 3.15 s mtp2 A-TTFT open item (2-request A/B) RESOLVED: block
   alignment, not an MTP cost.** A (795 tok) is sub-block under mtp2
   (`round_down(795, 800) = 0` cacheable) → full re-prefill every
   rep, 795/253 = 3.14 s, sd 0.007. Under baseline (block 784) the
   same prompt hits 784 → 0.14 s. Not a bug.
3. **MTP + align-mode prefix caching works** (upstream Marconi
   pattern verified: 1st full prefill, 2nd full (state cached at
   completion), 3rd+ hit). Subtlety: only 800 of 1600 aligned tokens
   hit under MTP (baseline 1568 of 1631) — `num_reprefillable_tokens`
   finalization × Marconi's single cached state; not chased.
4. **One-time effects:** first prefill after a rebuild pays ~10 s
   Triton JIT (mamba-align + eagle kernels) → ~1 s with warm on-disk
   cache; the probe's warmup phase makes measured requests
   one-time-free.
5. **BUG (ours + upstream): serving baseline OOM at util 0.95 with a
   WARM inductor cache.** 2nd request (784-tok chunk) dies in
   `aten::empty` 356 MiB (free: 0) inside the inductor piecewise
   graph; 3/3 on a monitored-clean GPU. Warm-cache init peak is
   ~0.16 GiB lower (autotune workspaces cached away) → KV pool sized
   too large for the runtime allocation. mtp2 immune (KV 6.39 GiB —
   MTP weights + spec pool leave headroom). **Workaround:
   `--gpu-memory-utilization 0.93` (validated; upstream-report
   candidate).**
6. External docker-crash report (other agent,
   `gather_paged_kv_quant_kernel` at init): separate issue; his
   workaround (`GFX906_FA_LEGACY=0` + no prefix caching) is our
   documented experimental path; 6+ clean local inits of the
   production config same day — need his timestamps to sort
   collision vs env bug.

## 2026-08-22 — TP=2 mtp2 engine-cadence overhead (OPEN → RETRACTED)

**VERDICT: RETRACTED — the "regression" was host-state degradation,
not a code regression.** (Section name kept — DEVLOG-masked-fa.md
references it.)

Origin: N4 re-baseline found mtp2 TP=2 at 24.9 t/s vs the S8 record
39.9. Read-only diagnosis (`fa-masked-mtp-regression-glm5.md` folds
three reviews; all claims adjudicated) + this session's GPU work:
TP=1 healthy (record-config rerun 49.8 t/s @ 2.72 acc; fox matrix
62.6/62.9 ms at maxlen 2816/32768 — maxlen exonerated); TP=2 GPU side
healthy too (gptq-M3 17.6 ms = TP=1's 33.6 halved, rccl 8.5, GPU busy
96% of the kernel window). **Apparent pathology: engine cadence 144
ms/step vs ~45–50 ms GPU work — the worker blocked in
`hipEventSynchronize` ~57 ms/step** (3813 ms / 272 calls / 71
steps); spec CPU on the worker ~8.5 ms/step; the rest untraced
EngineCore bookkeeping. `--async-scheduling` A/B: NO EFFECT (24.93
vs 24.90, confirmed engaged) — a per-step GPU→CPU→GPU data
dependency, not schedulable idle. H1: all S5–S8 TP=2 numbers came
from one never-rebuilt dirty Aug-19 binary.

**Post-reboot correction (the same day):** a 12:39 host reboot
restored mtp2 TP=2 to **74.9 t/s steady (40.1 ms/step, acc 3.00) — 3×
the same-boot 24.9, identical binary/config/harness**; the build
beats the S8 record. Healthy-host matrix (131k, TP=2, P1): plain
40.9 · mtp1 61.9 (@2.00) · mtp2 74.9 · mtp3 88.1 (@4.00) — spec beats
plain 1.83–2.15× on TP=2 again. The trace analysis above described
the **degraded host** (the 57 ms/step hipEventSynchronize stall IS
the degradation signature); the no-effect A/Bs were no-ops for that
reason; the clean-rebuild confirmation of `69f615b98` cancelled.
Protocol now: (a) SSE-chunk counting under-reports spec t/s by
~acceptance — use `stream_options.include_usage`; (b) **degradation
canary** (`docs/gfx906/degradation_details.md` 60-s probe) before
recording any spec number — slow canary ⇒ reboot; (c) every wedge /
degradation gets a timestamped row in `degradation.md`. Re-run CLEAN
post-reboot (canary-passed): 131k P0 49.44 / 262k P0 37.69 / 131k P1
74.74 / 262k P1 73.51 — P0 tax −23.8%, P1 residual 0.0%, mtp2 = 1.83×
plain, identical output hashes across arms.

## 2026-08-24 — MTP: model HAS an MTP head; k=2 = 1.503× (SHIPPED)

(Correction: an earlier note claimed the model has no MTP head —
WRONG. config text has `mtp_num_hidden_layers=1`; the checkpoint
carries 15 mtp.* tensors (mtp.fc + mtp.layers.0.{self_attn,mlp,norms});
'mtp' in `modules_to_not_convert` → the drafter is unquantized fp16.)

Setup: `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
(auto-resolves the drafter to the same checkpoint). Drafter = 1
full-attn layer + fp16 MLP + mtp.fc (10240→5120 on [h ⊕ e]);
embed/lm_head SHARED with the target → per-forward weight traffic
fc 0.10 + attn 0.28 + mlp 0.54 + lm_head 2.54 ≈ **3.4 GB**. First
crash: the proposer's ROCm allowed_attn_types whitelist lacked
Gfx906FAMetadata — added (gfx906-gated import); no
build_for_drafting override needed.

**Serving A/B (3 agentic × 3): baseline 26.53 → mtp2 39.47 = 1.488×**
(1.82 tok/step, 86–95% draft-token acceptance, token-identical on
both agentic prompts; one benign divergence on the repetitive
prompt). Every step drafts (MTP always proposes; target always
uniform M=3) — **no no-draft penalty, unlike ngram.** k=2 is the
sweet spot: each drafter forward ≈ 8 ms, the marginal third token's
conditional acceptance is low; k=3 break-even needs ≥2.17 tok/step.

**Drafter cost decomposition** (step ≈ 60 ms = target M=3 ~45 +
drafter ~15): lm_head 2.54 GB @ 811 GB/s = 3.13 ms (LLMM1, near
roofline — the irreducible ~6.3 ms of the 2 forwards) · **fc
(5120×10240, M=1) fell through to triton at 193 GB/s → extended
`_gfx906_gemv_long_k` (kchunk=2048: 148.7 µs @ 705 GB/s; 1024→395,
512→254), dispatch-only, saves ~0.8 ms/step** · qkv/out via LLMM1 at
777–796 · gate_up/down (M=3) on the m4 kernel at ~510–521 GB/s.
Structural note (user's theory, confirmed): the drafter is
compute/traffic-hungry on this BW-bound GPU — 6.8 GB extra weight
reads/step on top of the target's ~25 GB; floor ~6.8 ms, measured 15
pre-fix.

**m4 kernel templated on M:** the runtime-M m4 plateaus ~520 GB/s for
ALL M (even M=1 through it) vs 700–815 for the M=1 kernel — `xa[4]`
+ `acc[RPT][4]` + `acc_flat` allocated unconditionally → ~130+ VGPRs
→ ~35% occupancy loss at KC=1024. Fix
`dense_gemv_m_kernel<RPT,KCHUNK,M>` (static arrays, in-place shfl
reduction): M=1 +45%, M=2 +37%, M=3 +3%, **M=4 REGRESSED 507→311
(cause unattributed — RPT=4 and kc=512 worse too)** → **keep both
kernels**: launcher dispatches M=1..3 to the templated kernel, M=4 to
the restored runtime-M `dense_gemv_m_kernel_rt` (byte-identical to
L1') — ngram's M=4 path is bit-identical to before. Residual M=3/4
ceiling = x/w re-read ratio (M/RPT=1.5:1 → ~540 GB/s); beating it
needs an LDS-tiled GEMM (~1–2 ms/step in-engine — deferred).

**L3 (ngram proposer CPU cost) closed: not beneficial for this
workload** — the proposer is single-digit ms Python vs 35–56 ms
GPU-bound steps; the GPU proposer was already rejected on quality
(Phase 0).

**Final A/B (3-arm, all fixes in, VRAM-safe runner):** baseline
26.44 (CI 25.45) · ngram3 **28.92 = 1.094×** (CI 28.34) · mtp2
**39.74 = 1.503×** (CI 38.58; 1.819 tok/step, 90.95% acceptance).
The m4 M=3 templating added no measurable delta over the pre-templating
39.66 (within noise) and no regression. MoE 35B soak same build:
66.30/66.31/66.27/66.27 (record band) — no cross-path regression.

**Wedge #2 + runner rule:** the mtp2 arm of the pre-reboot A/B died
at weight load ("unspecified launch failure") and wedged the MI50
(rocm-smi N/A, 80% zombie VRAM) — the runner's fixed 8 s sleep after
killing the ngram3 server wasn't enough for the ~25 GB to drain.
**Rule: wait for rocm-smi VRAM < 5% between arms** (wait_vram loop in
the runner). Post-reboot NOTE: the NFS /data fstab `auto` mount came
up down — the first post-reboot A/B failed with HFValidationError
(model path missing), not a GPU fault.

**MTP work state (final):** k=2 MTP recommended (1.503×, token-identical
agentic). The drafter is fp16-only — every byte hits HBM; on a
fast-compute GPU it would be nearly free, here ~25% of the step.
Remaining levers (deferred): M=3/4 LDS-tiled kernel (~1–2 ms/step),
k=3 (needs ≥2.17 tok/step), AWQ MTP (quality risk, out of scope).

## 2026-08-24 — external code reviews absorbed

Two independent critical reviews of the branch
(`spec-dec-code-rev-{glm,ds4}.md`):

**ds4 F1 (claimed BLOCKER: cg-small fix unreachable under MRV2) —
REFUTED.** The serving runs use MRV1: `VLLM_USE_V2_MODEL_RUNNER`
unset, Qwen3.5 is not a default-V2 arch; the serve log's
`gpu_model_runner.py:6821` line is an EXACT match for MRV1 (the MRV2
file has 5424 lines); the per-arm PIECEWISE graph counts (ngram3: 8,
mtp2: 7) = rounded sizes + restored small sizes per method (q=4 →
5+3=8; q=3 → 5+2=7); pre/post kernel profiles independently confirm.
The fix DID break an upstream test
(`test_resolve_cudagraph_mode_adjusts_spec_decode_sizes_only_for_v1`
asserts rounded-only [4,8,12,16]) → that test now pins
`VLLM_GFX906_SPEC_CG_SMALL=0` + a new gfx906 test asserts
[1,2,3,4,8,12,16]. Both pass.

Fixed: glm 1.2 (`VLLM_GFX906_GEMVM_RPT=4` + N≡2 mod 4 tripped the
launcher's TORCH_CHECK mid-serving — the m4 hook now mirrors the rpt
env and falls back to triton when `m % rpt != 0`) · glm 1.3
(ksplit==1 plain-store numeric test) · ds4 F8 (the cg-small tests).
Accepted/documented: the ksplit>1 fp16-CAS epilogue is run-to-run
order-nondeterministic → "token-identical" is measured once, not
guaranteed — **the S3-class bar (PPL/coherence), stated so it is not
re-litigated** (the 90.95% acceptance absorbs draft-logit jitter) ·
cg-small graph memory 1.60 GiB total at 4-seq (the A/Bs prove the GDN
budget still fits) · ds4 F6 (the [5120,10240] fp16 shape is unique to
the MTP fc here; RPT=2 beats RPT=4 on every measured m4 shape — no
RPT sweep pending) · glm 1.7 (acceptance + token-identity IS the
Gfx906FAMetadata contract test in action) · glm 1.5/ds4 F4 (the
templated M=4 is NOT compiled — the launcher never instantiates it;
the 311 GB/s measurement was a transient build that did).

**ds4 F2/F7 empirical gates (run this session):** dense 27B 4-seq
(4× uniform 2048-tok prompts → uniform prefill fast path + M=4
non-spec decode): m4 ON 23.70 vs OFF 23.69 t/s — **the m4 hook is
neutral on the non-spec path** (no regression; 23.70 vs the 25.25
record band is the no-max-ilp build delta, not this branch's code).
Serial-vs-batch token probe (`uniform_batch_probe.py`): prompt 0
diverges at char 59 — benign fp argmax flip near a logit tie (both
coherent), flip location varies run-to-run (CAS-order). F7 pad-row
concern structurally void (`num_tokens == num_seqs*max_seqlen_q`
implies no padded rows).

## 2026-08-24 — m4 WARPS==1 shfl-gather bug (fixed) — TRAP

Critical pre-merge review found a **silent wrong-numerics bug** in
`dense_gemv_m_kernel`'s CAS epilogue (WARPS==1, ksplit>1): the
templed kernel gathered butterfly sums with
`s[i] = __shfl(acc[i/M][i%M], i)` inside `if (t == 0)` — a divergent
cross-lane gather. On ROCm 7.14 clang this lowers to
`ds_bpermute_b32 vdst, slane, vdata offset:4/8/12` — the offset
encoding reads `addr(vdata)+offset` from the source lane's file,
correct only for a consecutive-register layout the allocator does not
guarantee → s[0] (offset 0) right, s[1..] wrong: **M=2 → 74.5%
mismatched elements**, even/odd column split matching the half2 CAS
packing. Latent for Qwen3.5-27B (all K's 1024-multiples → kchunk=1024
→ WARPS=2 smem path; the kchunk=512 path only fires for K=1536/2560/
3584 models). The rt kernel had the same path (uniform-expression
form) — fixed to read lane 0's own registers; the templated kernel
the same way. **The divergent cross-lane `__shfl` gather is the only
unsafe form; offset-0 forms (butterfly, uniform broadcast, e.g. the
FA gather kernel's `__shfl(x, 0, 64)`) are safe and empirically
verified.** Regression test
`test_..._spec_gemv_m4_kc512_ksplit2` (m=2,3,4; kchunk=512 →
WARPS=1 ksplit=2 vs F.linear); suite 31 passed / 2 skipped. A minimal
standalone repro misleads (plain clang spills the accumulators →
flat loads, a different artifact) — validate through the in-tree test
against the actual CMake build.

## 2026-08-24 — parallel 2-request serving A/B (dense 27B, split build)

`benchmarks/kernels/gfx906/spec_parallel_dense.py`: staggered 2-request
(B at t=2 s), 512 out-tok, 3 repeats, streaming TTFT; prefix cache
does not help (both arms re-prefill ~1418 tok/window).

| clean reps 1–2 | baseline | mtp2 |
|---|---|---|
| A decode (resident) | 22.6 | **26.8 (+19%)** |
| B decode | 25.2 | **32.1 (+27%)** |
| aggregate | 40.4 | **42.7 (+5.7%)** |
| B TTFT (prefill under load) | 2.51 s | **3.76 s (+1.24 s)** |
| A TTFT (prefill alone) | 0.14 s | **3.15 s → resolved: block alignment (2026-08-19 entry)** |

Per-request uplift real (+19–27%); aggregate nets +5.7% (M=6 verify
steps + drafter forwards). Prefill cost of MTP under concurrent load:
B-TTFT +50%.

## Environment notes

- **Standalone HIP programs fail on physical GPU #0** (2026-08-19):
  even an empty `<<<1,64>>>` kernel → hipErrorIllegalState (401) on
  GPU #0, works on #1; torch/vLLM work on both. Not chased — use
  .venv python for GPU probes.
- vLLM in-proc probes: the spawn/guard/max_model_len/late-binding
  checklist is in DEVLOG-moe-opt.md "PROBE PITFALLS".
