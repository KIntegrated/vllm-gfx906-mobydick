# INT8 / Q8 on gfx906 — what transfers from the gfx908 int8-vllm fork

Investigation (read-only), 2026-08-30. Inputs: `/local/git/int8-vllm`
(gfx908/MI100, `curvedinf/int8-vllm` @ `0dc62c98e`), `/home/kread/git/llama.cpp`
@ `b7386eeae` (gfx906 Q8_0 repack), the two Qwen checkpoints we serve, and
this hub. **No code was changed and nothing was run for this document.**

**VERDICT:** OPEN (analysis complete; all perf numbers here are analytic or
imported from the records cited, not measured by this document).
**GATE:** the two probes in §6, then serving A/B per `AGENTS.md`.

## 0. Bottom line

Ranked by (expected gain × confidence) ÷ effort. "fp16 mass" = weights that
**our checkpoints ship unquantized** (§2.2) — the class where int8 is a strict
byte reduction over what we read today, with no W4 comparison problem.

| # | item | what transfers | expected gfx906 effect | conf | gate |
|---|---|---|---|---|---|
| **T1** | int8 for the **fp16 weight mass** (lm_head, FA q/k/v/o, GDN in_proj_qkv/z/out, shared expert, layer 0) — load-time quant, int8 GEMV | yes, fully (byte-bound, no MFMA/aiter needed); their load-time quant pattern (`_quantize_embedding_int8_`) is the template | **dense 27B +8–11 % decode, MoE 35B +10–19 % decode ceiling**; frees 1.3–3.8 GB for KV | **high** (analytic: 33 % of decode weight bytes, both shapes measured at 98–101 % of HBM floor) | probe P2 + serving A/B + KLD/PPL |
| **T0** | **P82 lossy spec-decode acceptance** (rejection-sampler OR-clause) | yes, arch-free (pure sampler) | their measured **+19…+26 % decode, GSM8K neutral**; ours rides on MTP k=2 (78.7–90.9 % acceptance today) | high (imported) | GSM8K + greedy-coherence + serving A/B; **explicit opt-in only** |
| **T2** | **int8 KV storage** (`int8_per_token_head` doctrine, but in our Q8_0 layout) | partial: format+doctrine yes, their Triton/aiter kernel no | **1.88× KV capacity** (256k → ~480k-token pool at fixed bytes); decode +1–3 % (halves FA gather traffic, kills the read-side quantize); V-side accuracy is the only new loss | med | FA-side accuracy gate + serving A/B (the LEGACY=0 lesson, §3.2) |
| **T3** | **A8W8 (int8 act × int8 weight) GEMM for prefill only** | **as Triton source** — our Triton port lowers int8 `tl.dot` to `v_dot4_i32_i8` (§2.3) | ceiling 1.96× the fp16 MAC rate; their measured analogue **1.62× on the 17408-wide MLP, 0.81–0.95× at N=5120, 0.61× small-N** → only dense fat-N prefill qualifies | low-med | probe P1 (GO/NO-GO) |
| **T4** | accuracy + measurement methodology (KLD probe, record-once replay budget, paired boots, metric definitions) | yes, pure tooling | our dense PPL band is ±2 % — too coarse to gate a 0.1 % weight perturbation; KLD is ~100× more sensitive | high | adopt on first use |

The single most important framing: **the int8 fork's GEMM layer does not
transfer at all (aiter + CK, both MFMA-based), but the int8 fork's *byte*
doctrine transfers better than it did on MI100** — because on our Qwen
checkpoints a third of the decode-step weight traffic is still fp16.

## 1. Hard constraints (do not re-litigate)

1. **No aiter, no CK on gfx906.** Every GEMM, the act quant, unified
   attention and the CAR in the gfx908 stack are aiter/Composable-Kernel
   objects; aiter does not support gfx906. All of it is out.
2. **No MFMA/WMMA** (`README.md` line 4). Integer compute on this chip is
   `v_dot4_i32_i8` only: full rate, 4.44× fp32 FMA, **1.96× packed fp16**
   (`dequant-instructions.md`, SCEV-proof probe). `v_dot4c_i32_i8`,
   `v_dot8_i32_i8`, `v_dot8c_i32_i8` are assembler-rejected; the i4
   `v_dot8_i32_i4` exists but needs int4 operands on both sides.
3. **Weights are already int4** (AWQ) for the MLP/expert/GDN-in_proj bulk.
   W8 for those tensors is **2× the bytes** → strictly worse on byte-bound
   decode. Any A8W8 proposal must live where the workload is
   compute-bound, not where it is byte-bound (§2.1).
4. Our custom Q8 FA backend declares `supported_kv_cache_dtypes = ["auto",
   "float16", "half"]` (`vllm/gfx906_fa/gfx906_fa_backend.py:171-175`).
   `--kv-cache-dtype int8_per_token_head` today silently leaves the CUSTOM
   backend → Triton attention (the 18.89-vs-25.60 t/s class of gap). "Just
   turn on int8 KV" is not available.
5. The int8 fork's numbers are **their** metric definitions
   (`docs/recipes/README.md` §NUMBER DEFINITIONS: steady-state TG =
   concurrency ÷ mean TPOT vs `vllm bench serve` wall-clock ≈ 0.55× of it),
   on a box with a 24 % per-rank sclk skew. Never mix them with ours.

## 2. Three facts that decide most of the surface list

### 2.1 int8 GEMM pays only where compute-bound — and the fork measured the boundary

`vllm/model_executor/kernels/linear/mixed_precision/triton_w8a16.py:583-587`
(gfx908 fork) is the whole doctrine in one comment:

```
_W8A8_DISPATCH_MIN_M = 256      # decode (M<256) stays on W8A16 (bandwidth-bound)
_W8A8_DISPATCH_MIN_N = 8192     # microbench 2026-08-22: 17408-wide MLP up/gate 1.62x,
                                # 5120-wide 0.81-0.95x, small-N 0.61x
```

MI100's int8 advantage over its own fp16 path is the same ~2× ours is
(MI100: INT8 374 TOPS vs FP16 184.6 TFLOPS; MI50: dot4 25.9 T MAC/s vs
packed fp16 13.2 T MAC/s). So those ratios are the best available prior for
A8W8 on gfx906: **the win is real on fat-GEMM prefill and negative
everywhere byte-bound.** Independent corroboration from the llama.cpp gfx906
Q8 repack PR (commit `b7386eeae`, recorded in `dequant-instructions.md`):
layout/repack deltas were **neutral (−1.1…+6.4 %) on single-token mat-vec**
and **+21…+51 % on the LDS-staged multi-token GEMM**.

Our own dense prefill sits at ~41 % of the fp16 MAC ceiling (442 t/s @32k,
TP=2, `README.md` long-context sweep vs 13.2 T MAC/s/GPU), i.e. there is
headroom for a higher-ceiling dot — but only on the fat shapes:
`gate/up = [34816, 5120]` ✓, `qkv = [14336, 5120]` ✓, `down/o_proj =
[5120, 17408]/[5120,5120]` ✗ (their 0.81–0.95× band), MoE experts
`N=512/1024` ✗✗.

### 2.2 Our fp16 weight mass (checkpoint scan, 2026-08-30)

Safetensors-header scan of the two models we actually serve. This is the
number the int8 fork's whole `INT8_AUDIT_RESULTS.md` exists to find, and on
our checkpoints it is **big**.

**Qwen3.5-27B-AWQ** (`/data/models/qwen/Qwen3.5-27B-AWQ`, 21.85 GB:
11.47 GB packed int4, 10.02 GB BF16, 0.36 GB fp16 scales):

| fp16/bf16 tensor group | size | read per decode token |
|---|---|---|
| `lm_head.weight` [248320, 5120] | 2.543 GB | **yes** |
| `embed_tokens.weight` [248320, 5120] | 2.543 GB | no (row gather) |
| FA `q_proj` [12288, 5120] ×16 layers | 2.013 GB | **yes** |
| FA `k_proj`/`v_proj` [1024, 5120] ×16 | 0.336 GB | **yes** |
| layer 0 (`modules_to_not_convert`, incl. 3×178 MB MLP) | 0.780 GB | **yes** |
| `mtp.*` sidecar | 0.703 GB | only in MTP arms |
| GDN `in_proj_a`/`in_proj_b` [48, 5120] ×48 | 0.047 GB | yes (negligible) |
| **fp16 read per decode step** | **≈ 5.72 GB** | **≈ 7.17 ms @ 798 GB/s** |

That reconciles with the measured `LLGemm1 … ~7,100 µs/step` bucket in
`DEVLOG-dense-decode.md` to within 1 %. (It does **not** reconcile with that
devlog's *labelled* floor table, which attributes 3.79 + 1.25 ms/step to
`in_proj_b×48`/`in_proj_a×48` — those tensors are 47 MB total, i.e. ~60 µs.
The labelled rows are stale; the total is right. §7 correction C-1.)

**Qwen3.5-35B-A3B-AWQ** (`/data/models/QuantTrio/…`, 25.45 GB: 15.83 GB
packed int4, 9.05 GB BF16): per-step fp16 read = `lm_head` 1.017 GB +
GDN `in_proj_qkv` [8192,2048]×30 = 1.007 + `in_proj_z`×30 = 0.503 +
`out_proj`×30 = 0.503 + FA `q_proj`×10 = 0.336 + `o_proj`×10 = 0.168 +
shared experts ×40 = 0.246 + layer-0 active experts ≈ 0.05 →
**≈ 3.8 GB/step ≈ 4.8 ms @ 800 GB/s**. This matches the per-shape floor
table in `DEVLOG-moe-m1-sprint.md` ("S3 LLMM1 2.09 ms" + the `dense_gemv`
floors: lm_head 1274.6 µs, GDN in_proj 63.1 µs×30, FA qkv 47.3×10,
o_proj 21.0×40, shared gate_up 5.3×40, shared down 2.6×40 = **4.80 ms**)
line for line.

Consequences, and they are the reason this document exists:

* **32 % of the MoE decode step and ~20 % of the dense decode step is
  fp16 weight streaming that is already at 98–101 % of the HBM floor**
  (`DEVLOG-moe-m1-sprint.md` "no lever" column; `DEVLOG-dense-decode.md`
  "LLMM1 is at the HBM floor for every dense shape"). Nothing in our
  roadmap can win there — except moving fewer bytes.
* int8 halves those bytes: **dense −3.6 ms of a 35.6 ms decode-only step
  (≈ +11 %), MoE −2.4 ms of a 14.8 ms step (≈ +19 %, i.e. the 70 t/s
  roadmap target in one move if it lands at floor).** Budget 50–70 % of the
  ideal delta for small-shape launch/occupancy costs.
* It also frees **1.27 GB (MoE) / 3.8 GB (dense, embed+lm_head)** resident —
  directly relevant to the `--gpu-memory-utilization 0.93` and 256k-prefill
  pressure documented in `/local/git/AGENTS.md` and `oom-256k-prefill.md`.
* **Their doctrine matches ours on what NOT to int8**: they keep `A_log`,
  `dt_bias`, depthwise conv1d, norms, KV scales, GDN recurrent state and
  softmax/P·V float, and they explicitly refuse to quantize `in_proj_a/b`
  because those feed `exp(A)` gates (`INT8_AUDIT_RESULTS.md` #5). On our
  checkpoints those tensors are 47 MB — nothing to win there anyway.

### 2.3 Triton on gfx906 lowers int8 `tl.dot` to `v_dot4_i32_i8`

Source-verified in our Triton (`/local/git/triton-gfx906`, 3.6.0, the tree
`requirements`/venv resolves to):

* `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp:48-53` — for
  `i8 × i8 → i32` the chosen intrinsic is `llvm.amdgcn.sdot4`,
  `vectorSize = 4` (operand pack bitcast to `i32`).
* `.../TargetUtils.cpp:44-56` — `supportsVDot()` includes `VEGA20`.
* `.../TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp:1665-1669` — the
  I8×I8→I32 v_dot form is legal when `k % 4 == 0`.
* `.../DotOpToLLVM.cpp:53-55` — a blocked-layout `DotOp` goes to
  `convertAMDFMADot` (the path above); `f16×f16→f32` takes `fdot2` there.

So the gfx908 fork's **portable** int8 GEMM (`triton_w8a8_gemm_kernel`,
`triton_w8a16.py:248-341` — `tl.dot(a_q, b_q, out_dtype=tl.int32)` with
`BLOCK_K == group_size == 128` and one A-scale/B-scale descale per tile,
exact by construction) can be lifted as *source* with no aiter dependency.
**This is ISA-level unverified on our toolchain** — probe P1 settles it
(compile → check the emitted ISA via the `.amdgcn` asm dict / the
objcopy→objdump chain in the `gfx906-isa-disassembly` skill, then exactness
vs an int32 reference, then rate). Open risk: the blocked-layout FMA path
may not stage operands well (our Triton W4A16 `has_zp` branch is 267 ms/tok
pathological, `README.md`), which is exactly why P1 is a gate and not a
formality.

## 3. Surface by surface

Format: what gfx908 did (file/commit) → why it won there → gfx906 transfer →
verdict → gate.

### 3.1 T1 — int8 the unquantized projections (incl. lm_head + embedding)

* **They did:** `VLLM_GFX908_INT8_EMBEDDING` (default ON, `d9e11f233`) —
  load-time per-row int8 + fp16 scale via `replace_parameter`, dequant fused
  into the gather; `VLLM_GFX908_INT8_LM_HEAD` (production ON, `64627cb78`) —
  untied lm_head → per-channel int8 + W8A8. Their audit's "ship int8
  storage" item halves a 30 GB checkpoint toward 18 GB.
  **Trap they hit first** (`ab2139abf`): the initial version cast the whole
  table to fp16 *before* `F.embedding`, materializing 2.5 GB per call. The
  fix gathers int8 rows and dequantizes only the selected rows.
* **Us:** the same load-time conversion applies to the §2.2 mass, not just
  the vocab pair. No checkpoint work, no third-party quant pipeline, and we
  already do load-time repacks (W4 MoE repack ~65 s). Quantization choice:
  per-output-row symmetric (their embedding/audit choice; simplest, keeps a
  GEMV's inner loop scale-free) vs per-128-group along K (better accuracy,
  one extra scale load per 128 elements — matches the granularity our AWQ
  scales already use).
* **Kernel work:** one int8 weight-stationary GEMV in the
  `csrc/rocm/dense_gemv_gfx906.cu` family (M=1, K≤8192) reading int8 +
  per-row scale, fp16 `x`, `__ockl_fdot2` accumulate. Byte-bound ⇒ the
  dequant ALU is free (the `dequant-instructions.md` warning about
  i8→half2 expansion costing 0.17× applies to *dot-throughput*-bound loops,
  not to a stream at 800 GB/s). Alternative: quantize `x` per token (K
  elements — trivial at M=1) and use `v_dot4` with exact int32 accumulation;
  P2 should measure both shapes cheaply before the kernel is written.
  `LLMM1`-served shapes (`out_proj`, shared expert) can be routed to the new
  GEMV rather than porting an int8 LLMM1.
* **Verdict: OPEN, high confidence.** **GATE:** probe P2 (int8 GEMV ≥ 1.7×
  the fp16 GEMV at `lm_head` and `q_proj` shapes) → serving A/B
  (`_bench_gfx906.py`, MoE + dense, both arms same boot) → accuracy: PPL
  bands (6.6817–6.6942 / 6.6993–6.7197) **plus** a KLD gate (§3.8) because
  the dense PPL band is ±2 % and cannot see a 0.1 % weight change.

### 3.2 T2 — int8 KV cache (their `int8_per_token_head`, our Q8_0)

* **They did:** `--kv-cache-dtype int8_per_token_head` (fp32 inline scales)
  on target *and* draft, through aiter unified attention / the Triton unified
  kernel; per-token-head dynamic quant at cache-write; replay error "0.85 %
  per token-head"; the int8-KV + int8-mamba-state **combination** corrupts
  generation (bisected; fp16 state blew states to 63k → fp32 state is
  mandatory).
* **Us:** our FA already computes on **Q8_0 K** (per-32-block, inline half
  scale — *finer* granularity than per-token-head, so equal-or-better
  numerics per byte) and reads V in fp16. Per `README.md`, LEGACY=1 (the
  default) quantizes K on every read; LEGACY=0 stores a Q8 view aliased into
  the fp16 K half. The thing **nobody has tried** is making Q8 the *primary
  pool*:
  * capacity: per (layer, token) fp16 K+V = 2·(D·2 B·Hkv) = 4096 B vs Q8_0
    2·((D/32)·34 B·Hkv) = 2176 B → **1.88× pool** (K-int8-only: 1.31×);
  * cost: the append-time Q8 write we measured at **+94.6 µs/step** in
    LEGACY=0 — but there it was an *extra* write on top of the fp16 write.
    In an int8-primary pool the fp16 write disappears, so that specific tax
    is structurally gone (the G1 per-node tax question stays open and is
    the same measurement);
  * read: gather traffic halves and the in-kernel quantize disappears —
    i.e. LEGACY=0's stated benefit without LEGACY=0's measured cost.
    Note the honest counter-evidence: LEGACY=0's kernel-level wins
    (Q8 gather 22–45 % faster per step) did **not** transfer (−6.3 % B=1).
  * **V is the new loss and the new kernel work.** Storing V in int8 does
    *not* let P·V use `v_dot4` (P is fp16 probabilities; the P·V accumulate
    is 1024× in-place `v_pk_fma_f16` — `dequant-instructions.md` 2026-08-29
    objdump). V must be dequantized back to fp16 in the kernel, adding ALU
    to an issue-bound loop at Sq>1. So: **stage A = int8 K only (zero new
    accuracy loss vs the production read path — K is already Q8 there),
    stage B = int8 V with its own gate.**
* Plumbing cost is real: `int8_per_token_head` is already plumbed for the
  Triton backend in our tree (`vllm/v1/kv_cache_interface.py`,
  `triton_attn.py`) but the attention *pool* is a single
  `[blocks,2,bs,Hkv,D]` tensor — a mixed/staged int8 layout for the CUSTOM
  backend is new allocation + `do_kv_cache_update` + COW work
  (`GFX906_FA_LEGACY=0` already proved the COW-safe alias pattern).
* **Verdict: OPEN, medium.** GATE: (i) PPL/coherence gate on V-int8
  (K-int8 needs none beyond a bit-equality check vs the current Q8 read);
  (ii) serving A/B at B=1/B=4 and pp2048/pp16384; (iii) the win is mostly
  *capacity*, so the acceptance criterion is tokens-in-pool and
  long-context TTFT, not t/s.

### 3.3 T3 — A8W8 prefill GEMM

* **They did:** aiter CK W8A8 (`gemm_a8w8_CK`, per-token act scale ×
  per-channel weight scale) as the production path at **every** M, with a
  load-time requant of the GPTQ gs128 scales to per-channel
  (`triton_w8a16.py:674-697`, `aiter_w8a16.py:380-500`), plus a Triton
  blockscale fallback and their own Triton `tl.dot` int8 kernel for
  large-M/fat-N (`d9e11f2..16e2b51`, `199db58`, `89e808d`).
* **Us:** only the portable Triton kernel is on the table (§2.3), only for
  **prefill / spec-verify shapes with M ≥ 256 and N ≥ 8192** (their
  measured gate, §2.1) — i.e. dense `gate/up` [34816,5120] and `qkv`
  [14336,5120]. Two extra costs they also paid and we would too:
  a per-call activation quant (§3.4) and a **2× weight copy** — int8 weights
  for prefill on top of the int4 weights we need for decode is 1.9× the
  weight footprint, which only fits at TP=2 and only if we free the §2.2
  fp16 mass first (T1). Also note the accuracy asymmetry: int8-ing an int4
  checkpoint's *activations* is fine, but requantizing group scales is not
  free — their own per-channel requant is lossy and they kept the gs128
  scales around "for the blockscale fallback and the fused context-KV
  dequant".
* **Verdict: OPEN, low-medium — hold until P1 says GO.** GATE: P1 must show
  `v_dot4_i32_i8` in the emitted ISA, exact int32 accumulation, and
  **≥ 1.3× over our current prefill GEMM path at [4096×34816×5120]** after
  charging the activation-quant cost. Otherwise close it as a DEAD-END with
  P1 as the evidence.

### 3.4 Activation quantization numerics (round-to-nearest) — cheap, and it is a numerics lesson not a perf lever

Their `pertoken_quant` replacement (`act_quant_rn.py`) is a two-sweep Triton
per-token int8 quant that **rounds to nearest instead of truncating toward
zero**: replay convicted the truncation leg at 10–15 % mean rel-L2 per GEMM
output, rounding recovers ~half of it, and it was *also faster* than the
4-pass aiter eager chain (TPOT 14.4 → 12.5 ms came with it). Portable
gotcha they hit: **`libdevice.rint` is not available on HIP** — they use
`floor(x + 0.5)`. Our in-tree `_per_token_quant_int8`
(`vllm/model_executor/layers/quantization/utils/int8_utils.py:47-61`) uses
`tl.extra.hip.libdevice.round`; P1 should confirm which one actually lowers
to `v_rndmath_f32`/`cvt` on our build rather than assuming.
Perf relevance for us: only as a component of T3 (its cost at M=4096 is the
"quant overhead payback" their N-gate is about) — at M=1 it is noise.
**Verdict: OPEN (adopt as method); GATE: P1 act-quant timing at the T3
shapes.**

### 3.5 int8 GDN state / conv cache / attention internals — matches our doctrine, no action

Their verdicts: GDN recurrent state **must stay fp32** (int8 corrupts with
int8 KV; fp16 round-trip broke delta-rule cancellation, states → 63k); conv
cache fp16; softmax and P·V stay float. We already keep all of these float.
Nothing to do; recorded so a future "let's int8 the mamba state" proposal
meets their bisection record instead of re-running it.

### 3.6 T0 — P82 lossy spec-decode acceptance (no int8, biggest non-kernel win)

`VLLM_GFX908_MTP_ACCEPT_THRESHOLD` (`vllm/v1/sample/rejection_sampler.py:99`)
adds an OR-clause: also accept a draft token whose *target* probability is
above a threshold. Their measurement (4×MI100, full bench + GSM8K 300 Q,
temp 0.6, seeded, default budget): **+19 % (27B dense decode), +24 %
(35B MoE decode), GSM8K 91.7 → 92.0 % (neutral)**; ~75–80 % of tokens
diverge from strict sampling; lossy and explicitly opt-in; they also
record an n=5 long-context buffer overflow that they sidestepped by
shipping n=3.
For us this is a sampler-only change that rides MTP k=2 (78.7–90.9 %
acceptance, `README.md`) and ngram n=5. Two caveats before believing the
+19 % here: (a) our spec numbers already assume exact verification, and
`DEVLOG-spec-decode.md` gates are acceptance-based — an OR-clause inflates
"accepted tokens" while silently changing output distribution, so the
accuracy gate must be task-level (GSM8K-style) **and** a
stop-probability/coherence check (see §3.8); (b) our own record warns that
spec-decode numbers are the first casualty of host-state degradation
(`/local/git/AGENTS.md` MI50 degradation note) — run the canary first.
**Verdict: OPEN, high value / low cost. GATE:** canary probe → seeded
task-accuracy A/B + first-token-stop probability + serving A/B; ship
default-OFF behind an env flag exactly as they do.

### 3.7 Fused AR+RMSNorm+quant epilogue — their result is a *negative*, and one warning

They built the full chain (aiter `fused_ar_rms_int8_per_token_quant`, a vLLM
fusion pass, an eager seam) and shipped it **OFF**: neutral on TPOT
(prefill-only by design), and — the transferable warning — enabling the
fused norm+quant **dropped spec-decode acceptance 71.2 → 46.5 %** while
being numerically "close" (`docs/recipes/README.md` §Current production
status). Second instance of the same lesson in this repo: quant-adjacent
numerics changes show up in acceptance rates long before they show up in
PPL. For our TP=2 work the corollary is cheap: keep vLLM CUSTOM all-reduce
(they measured 63.49 vs aiter CAR 58.34 vs PYNCCL 53.04 tok/s at TP4/C8 —
the same ordering conclusion we reached independently for TP=2), and do not
add per-layer fused epilogues without first pricing the graph-node cost
(roadmap **G1** — this is exactly the measurement G1 was built for).
**Verdict: no action; cross-linked.**

### 3.8 Their accuracy + measurement methodology is the most portable artifact

* `scripts/kld_probe_v2.py`: greedy continuations + per-position top-K
  logprob dumps along the *reference's* greedy path (so supports always
  intersect — v1's temp=1.0 sampling produced ∞ KLD), fixed 52-prompt
  corpus (code/math/strict-format/chat/long-form), `capture`/`compare`
  modes, `KLD_OUT_DIR/<tag>.npz`. Reported gate format: median KLD, greedy
  agreement n/m, **first-token stop probability vs the BF16 reference** —
  that last one is how they caught a 10× first-token-stop inflation that
  produced empty responses and that PPL never showed.
* `quant_audit_recorder.py` + `scripts/quant_replay_gemms.py`: record-once
  (pinned ring, non-blocking copies, ARM-file arming so warmup records
  nothing) then replay offline to attribute error per quantization leg.
  This is how they *convicted* act-quant and the fp16 GDN state and
  *exonerated* weights/AR/KV.
* Measurement doctrine worth copying verbatim: paired same-regime boots
  (their box is bimodal from DVFS skew, sub-15 % single-boot deltas are
  unmeasurable), one named metric per claim, and "audited exceptions are
  the optimum at the current measurement floor" as a stopping rule.
  Our equivalents of these are the `_bench_gfx906.py` graph A/B (the gate),
  PPL (±0.3 % MoE / ±2 % dense — too coarse here) and `greedy_probe.py`.
  **KLD is the missing gate in our kit** and T1 needs it.
* **Verdict: ADOPT (tooling, no kernel risk).**

### 3.9 Autotuned per-shape kernel tables

Their aiter tuning program produced a 299-row gfx908 tuned a8w8 CSV + a
45-variant `module_gemm_a8w8.so`: **1.37× geo-mean (max 2.64×) kernel-level
decode GEMM speedup, +3.5 % median wall-clock paired A/B** — i.e. a large
kernel-level win that moved end-to-end by single digits (the same transfer
ratio we report constantly). The method (per-(N,K,M-bucket) measured table,
shipped next to the kernel) is applicable to our own `q_gemm`/GEMV/skinny
dispatch — several of our dispatch gates are analytic (K≤8192, M≤7/≤32 MB)
rather than table-driven. **Verdict: OPEN (method), low priority; gate = a
per-shape table for the §3.1 int8 GEMV, which we need anyway.**

## 4. Do-not-transfer list (their own negative verdicts)

| tempting idea | their verdict (evidence) |
|---|---|
| quantize the draft model's `fc` GEMM | REJECT — "not a bottleneck", TPOT +3.4 % paired despite 0.49 % GEMM err |
| int8 ctx-KV projection in the drafter | REJECT — +10.4 % TPOT from acceptance drop, even with RN act-quant |
| int8 GDN/mamba state | corrupts generation (with int8 KV); fp16 state blew up to 63k; fp32 mandatory |
| fused norm+quant for spec decode | acceptance 71.2 → 46.5 % → shipped OFF |
| int8 all-reduce transport | research-only; fp16 payload stays (AR is ~10 % of decode) |
| A8W8 at decode M / narrow N | their dispatch gate excludes it (§2.1): 0.61–0.95× |
| quantizing `in_proj_a/b` (exp(A) gates) | doctrine-excluded on sensitivity; on our checkpoints it's 47 MB anyway |
| "enable int8 KV" without a CUSTOM-backend path | lands on Triton attention → multi-× slowdown for us |
| quoting their tok/s against ours | different metric definitions + a skewed box (§1.5) |

## 5. Suggested sequence and roadmap entries

1. **P1 + P2 probes** (§6) — GPU-free queue, no model load, minutes.
2. **T1** int8 GEMV for the §2.2 mass, staged by expected value:
   `lm_head` first (single shape, 2.5 GB→1.27 GB dense / 1.02→0.51 GB MoE),
   then FA q/k/v (2.35 GB dense), then GDN in_proj_qkv/z + out_proj
   (MoE 2.0 GB), then shared expert / layer 0. Kill switch
   `VLLM_GFX906_INT8_F16_MASS=0`; per-stage serving A/B + KLD.
   → propose **ROADMAP `I1`**.
3. **T0** P82-style acceptance threshold (independent track, opt-in).
   → **ROADMAP `I2`**.
4. **T2** int8 KV: stage A (K only, capacity), then stage B (V).
   → **ROADMAP `I3`** (cross-ref G1: it must price the per-node tax first).
5. **T3** A8W8 prefill — only if P1 clears its bar; and only after T1 freed
   the footprint. → **ROADMAP `I4`** (Tier 2).

## 6. Probes queued for the GPU-free run

Both are standalone (torch + triton only, no vLLM import), take
`HIP_VISIBLE_DEVICES` from the caller, and print an explicit `GO`/`NO-GO`.

* **P1 `benchmarks/kernels/gfx906/int8_triton_dot_probe.py`** — decides T3
  and §3.4. (a) compiles an int8 `tl.dot` kernel, asserts exact equality vs
  an int32 reference, and dumps whether `v_dot4_i32_i8` appears in the
  emitted ISA (asm dict `kernel.asm['amdgcn']`, with the
  `objcopy --only-section`/`llvm-objdump` fallback from the
  `gfx906-isa-disassembly` skill); (b) measures int8-vs-fp16 `tl.dot`
  GMAC/s (expect ~1.96× if dot4 is real: records 25 877 vs 13 210);
  (c) runs the ported blockscale A8W8 GEMM against a same-quality fp16 Triton
  GEMM at our production shapes × M ∈ {1, 64, 512, 2048, 4096}, charging the
  activation-quant pass; (d) same for the two in-tree act-quant roundings.
  **GO iff** dot4 present ∧ exact ∧ ≥ 1.3× at [4096, 34816, 5120].
* **P2 `benchmarks/kernels/gfx906/int8_gemv_probe.py`** — decides T1.
  Triton M=1 int8-weight GEMV with per-row and per-128-group scales, at the
  real shapes ([248320,5120], [12288,5120]×16, [8192,2048]×30,
  [2048,4096], [512,2048]), against the fp16 baseline and the recorded HBM
  floor; prints achieved GB/s and the projected serving delta.
  **GO iff** ≥ 1.7× at lm_head and ≥ 1.6× at the ×30 GDN shapes.

## 7. Corrections and open questions

* **C-1 (our record):** `DEVLOG-dense-decode.md` §"Dense model facts"
  attributes 3.79 + 1.25 ms/step to `in_proj_b×48` / `in_proj_a×48` fp16
  GEMVs; the checkpoint ships those as `[48, 5120]` bf16 × 48 = 47 MB
  (~60 µs). The bucket *total* (LLGemm1 ≈ 7.1 ms) is confirmed by the
  §2.2 scan, so only the attribution is wrong. Worth a one-line fix in that
  log next time it is touched (do not re-derive the budget from those rows).
* **C-2 (our record):** `README.md`'s KV sizing line ("Qwen3.5-27B 20 KB;
  Qwen3.5-35B 10 KB per token") disagrees with the config arithmetic
  (27B: 16 FA layers × 4 KB = 64 KB/token TP=1; 35B: 10 layers × Hkv 2 ×
  4 KB/2 = 20 KB/token). Recompute before sizing anything for T2.
* **Q-1:** does Triton's blocked-layout int8 dot stage operands well enough
  to matter (P1a/b)? If it emits dot4 but at < 60 % of the recorded rate,
  T3 needs the llama.cpp `q8_repack` structure (BM=64, BK=4 sub-blocks,
  `sW_lo/sW_hi` double buffer, k-major `sWdh` scales, `sXd[BN][BK+1]`
  padding, DPP reduce with `s_nop` padding — `repack-common.cuh`) ported
  instead, which is a hand-kernel project.
* **Q-2:** does an int8 lm_head GEMV stay at the HBM floor once it carries
  a scale load and a cvt, at 256 threads / K-chunk ≤ 2048 (P2)?
* **Q-3:** int8 K/V with **Q8_0 inline scales** (finer than
  `int8_per_token_head`) — do we keep it, or adopt the upstream
  per-token-head format so upstream ops (paging tests, `triton_attn`,
  `kv_cache_interface`) keep working? Keeping Q8_0 is better numerics per
  byte and zero extra stores, but it is a bespoke pool layout.
* **Q-4:** their `VLLM_GFX908_CK_FREE_GS128` pattern (free the superseded
  weight copy after requant) is required for any T1/T3 combination — do we
  quantize in place (`replace_parameter`) or keep an fp16 master?

## 8. Cross-links

`dequant-instructions.md` (dot rates, P·V contraction), `lds-layout.md`
(power-of-two strides, the q8 repack padding result), `latency-hiding.md`,
`README.md` (model records, knobs), `DEVLOG-dense-decode.md` /
`DEVLOG-moe-m1-sprint.md` (the fp16 GEMV floors), `DEVLOG-fa-attention.md` +
`DEVLOG-fa-legacy0-b1-decode.md` + `plan_fa_part_A.md` (Q8 read path,
LEGACY=0 negative), `oom-256k-prefill.md` (fp16 lm_head scratch),
`ROADMAP.md` (G1, C4, C7 = the 70 t/s target), `DEAD-ENDS.md`,
`REFRIGERATOR.md` (M6 Part C Q4-KV gate — the T2 neighbour).
gfx908 side: `docs/recipes/README.md` (canonical stack + numbers),
`INT8_AUDIT_RESULTS.md` (coverage inventory + experiment ledger),
`docs/mi100_decode_opt/p82_lossy_acceptance.md`, `scripts/kld_probe_v2.py`.
llama.cpp side: `ggml/src/ggml-cuda/q8_repack/README.md` @ `b7386eeae`.
