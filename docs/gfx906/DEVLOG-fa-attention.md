# Dev log — gfx906 custom Q8 FA & decode backend

Copyright Kevin Read <me@kevin-read.com>

> Split from DEVLOG-moe-opt.md (2026-08, topic consolidation) — the
> custom Q8 FlashAttention backend saga, the B=1 decode parallelism
> track, and the fused-gather/fill-pile work. MoE kernel and dense 27B
> trails live in DEVLOG-moe-opt.md / DEVLOG-dense-decode.md.

**VERDICT (top-level):** the custom Q8 FA backend went from dead code
to the gfx906 default; the decode-stack attention+copy work is what
took MoE to 67.39 t/s and dense to 25.60 t/s. `CUSTOM` is both the
win and several hard-won traps (stride bugs, capture lifecycle, the
V1/V2 gather serving degradation). Individual verdicts inline.

### P3-3 — paged attention: the CUSTOM (Q8 FA) backend saga (2026-08-15)

**Starting point**: vLLM attention 1.94 ms/step (10 FA layers × 194 µs) vs
llama.cpp 0.19 ms — P3-3 was planned as "partition the Triton kernel".
While starting that, discovered the repo **already vendors a custom Q8
FlashAttention backend** (`vllm/gfx906_fa/` + `csrc/gfx906_fa/`, llama.cpp's
`flash_attn_tile_q8`, head_size 256 supported) integrated in `67d2a813a2`
as the gfx906 default — **but it was dead code at runtime**: the
`vllm.general_plugins` entry point is absent from the tree's stale
`vllm.egg-info` AND the image's dist-info, so `CUSTOM.is_overridden()` was
always False and everything fell to Triton. Nobody noticed because the
fallback is silent.

Fix: `vllm/platforms/rocm.py` `_get_backend_priorities` now registers the
plugin explicitly on gfx906 when not already registered (idempotent;
entry-point installs still win). Verified priorities become
`[CUSTOM, ROCM_ATTN, TRITON_ATTN, TURBOQUANT]`.

**Then the real bug hunt.** With the backend live, `GFX906_FA_LEGACY=0`
(the fast path: Q8 side-buffer + fused HIP gather) produced garbage from
the first token ('!!!!!...'), while LEGACY=1 and FUSED=0 worked. Isolation
ladder (each step verified before moving on):

1. `reshape_and_cache_q8` vs `quantize_q8_0` on same data: **byte-identical**
   (D=256 fine).
2. Synthetic gather test vs torch `_gather_kv_q8`: byte-identical,
   V tail zeroed (Sk_pad, varied seq_lens).
3. Synthetic end-to-end (gather → `fa.forward` vs fp32 SDPA, decode AND
   prefill shapes, tails, B=1/B=2): all correct (rel err ~2.5e-3 = Q8 noise).
   NOTE: my first two "references" were wrong (einsum axis bugs — GQA
   broadcast + softmax over the wrong axis); the *pairwise* A/B/C identity
   checks are what kept the investigation honest.
4. In-model double-gather compare (`GFX906_FA_DOUBLE_CHECK=1`): **K identical,
   V corrupted with NaN** — synthetic had passed because my test caches were
   contiguous.

**Root cause**: `value_cache` in the backend is `kv_cache.unbind(1)` of
`[num_blocks, 2, block_size, Hkv, D]` — non-contiguous, block stride 2×.
`gather_paged_kv_q8` (and `forward_paged_direct`) **computed strides from
shapes, ignoring the real tensor strides** → the kernel read K-cache bytes
as V → NaN/garbage V. Only block 0 happened to look sane. Same bug class in
`reshape_and_cache_q8` (latent — its side buffer is contiguous today).
Fixed all three sites to use `tensor.stride(i)` (+ element size for fp16 V,
the launcher wants bytes) with contiguity TORCH_CHECKs on the last dim.

After the fix, live double-check reports `K=True V=True`, and the probe
generates the exact correct greedy output in both LEGACY modes.

**Benchmarks** (pp=2048/tg=256, single request):

| config | eager t/s | notes |
|--------|-----------|-------|
| Triton paged + P3-1 | **19.49** | best eager |
| CUSTOM LEGACY=1 (fp16 gather + per-step Q8 quant) | 18.49 | 2.1 ms/step gather/quant tax; FA kernel itself 194→72 µs/layer (2.7×) |
| CUSTOM LEGACY=0 FUSED (fixed) | 19.33 | fused gather removes quant tax |
| CUSTOM LEGACY=0 DIRECT forced | 19.21 | block-table indirection tax at B=1 |
| serving (cudagraph) + CUSTOM | **crashes** | `value_cache blocks mismatch` during piecewise capture — CGSupport.NEVER does not protect the torch.compile path; deeper integration issue, deferred |
| serving (cudagraph) + Triton (P3-1) | **44.09** | current best decode |

Also fixed along the way: `_bench_gfx906.py` counted tokens by re-encoding
the output *text* — garbage output re-encodes to fewer tokens (the
mysterious "32 tokens" was 256 real tokens of garbage, 19.05 t/s). Now
counts `token_ids`.

**Net P3-3 outcome so far**: attention itself can be 2.7× faster
(72 µs/layer), but at B=1 eager the win is eaten by the gather/conversion
tax and eager is CPU-launch-bound anyway; serving mode — where decode time
actually matters — cannot use CUSTOM until cudagraph capture is fixed
(capture calls attention with a different/aliasing kv_cache view; the
side-buffer alloc also assumes the first cache shape). Learnings:
- **Stride bugs hide from synthetic tests that build contiguous caches** —
  always mirror the real allocation path (`unbind` views) in tests.
- Silent registration fallbacks make dead backends invisible; assert the
  backend you expect in logs.
- The bench's text-re-encode token counting is wrong on degenerate output.
- llama.cpp-style Q8 K quant changes logits ~1e-3 — greedy outputs diverge
  from fp16 runs after ~10-25 tokens (both fluent); same trade llama.cpp
  makes.
- A/B pairwise identity checks (two implementations on same inputs) are
  more reliable than building a mathematical reference from scratch.

Next candidates: (a) cudagraph-safe CUSTOM (fix capture view handling or
pad the side buffer at capture sizes), (b) prefill uses CUSTOM (it wins
there per the vendored docs; our serving prefill could improve), (c) back
to the original P3-3 Triton partitioning for the serving path.

### Day-1 pre-step pair — gather micro-bench + P3-2(a) probe (2026-08-15)

Plan v6's two Day-1 gates, run in the same session. Scripts:
`benchmarks/kernels/gfx906/bench_gfx906_fa_gather.py` and
`bench_llmm1_rows_per_block.py` (7.14 image, repo source-mounted,
HIP_VISIBLE_DEVICES=0).

**Gather micro-bench (P3-3a go/no-go) — GO.**
`gather_paged_kv_q8` at B=1, Hkv=2, D=256, bs=16, pre-allocated out
buffers (serving steady state); correctness vs torch reference
byte-identical incl. V-tail zeroing:

| Sk | µs/layer | GB/s | ×floor(798 GB/s) |
|----|----------|------|------------------|
| 2048 | 18.6 | 345 | 2.3× |
| 2816 | **21.7** | 407 | 2.0× |
| 3328 | 25.3 | 412 | 1.9× |

Q-fp32 side costs per FA layer (Sq=1): q.float 3.9 + q_pad.zero_ 2.7 +
q copy 7.0 + out unpack 8.3 = **21.9 µs/layer** (launch-bound small
copies). Combined tax **43.6 µs/layer** vs 122 µs/layer FA kernel win
(194−72) → net **~0.78 ms/step** over 10 layers — inside the plan's
M1 expectation (0.7–1.2 ms → 46–48 t/s). **P3-3a RESUMES** per plan §4,
as the time-fenced parallel line; the Triton PIECEWISE baseline bench
(deconfound vs FULL_DECODE_ONLY) is required before reporting M1
numbers. Note for M2: the Q-side 21.9 µs is as large as the gather
itself — zero+copy+unpack are the natural next trim.

**P3-2(a) aiter probe — STOPPED (structurally a no-op on gfx906).**
Code audit of `rocm_unquantized_gemm_impl` (vllm/model_executor/layers/
utils.py):
- `VLLM_ROCM_USE_AITER` defaults **False** (envs.py).
- Every aiter gemm path (aiter triton `gemm_a16w16`, `wvSplitKrc`) is
  inside `if not on_gfx906():` — unreachable on MI50.
- `ops.wvSplitK` explicitly excluded on gfx906 ("matrix cores not
  supported", GCN5/Vega20).
- aiter triton-gemm whitelist is GPT-OSS shapes (m==5120/k==2880, …);
  none match this model.
→ aiter rejection reason recorded: **arch exclusion, not shape/dtype
selection** — the P3-1-era "collapse into P3-2(b)" path confirmed.

**LLGemm1 `rows_per_block` sweep** (the one real knob; dispatch
hardcodes 4):

| shape (M×K) | ×/step | rpb2 | rpb4 (cur) | rpb8 | rpb16 | floor |
|-------------|--------|------|------|------|-------|-------|
| 12288×2048 in_proj | 30 | 57.8 | 64.7 | 60.1 | 61.0 | 63.1 |
| 248320×2048 LM head | 1 | 1134.6 | 1209.8 | 1244.2 | 1178.6 | 1274.6 |
| 9216×2048 qkv | 10 | 43.9 | 60.9 | 49.7 | 51.0 | 47.3 |
| 2048×4096 o_proj | 40 | 21.5 | 23.0 | 25.3 | 28.9 | 21.0 |
| 1024×2048 shared gate_up | 40 | **47.7** | 7.6 | 8.8 | 9.9 | 5.3 |
| 2048×512 shared down | 40 | 5.5 | 7.1 | 6.8 | 7.8 | 2.6 |
| 256×2048 router | 40 | 4.7 | 5.0 | 5.1 | 5.9 | 1.3 |
| 64×2048 GDN small | 30 | 4.6 | 4.6 | 4.6 | 5.7 | 0.3 |

Weighted per-step: rpb4 5604 µs (current), rpb8 5523 µs (+1.4% best),
rpb2 6626 µs, rpb16 5793 µs. No shape moves ≥20% vs current → probe
stop condition met; config retune alone is not worth a change.
Anomaly: rpb=2 is 6× slower than rpb=4 on shared gate_up (M=1024)
while winning 6–11% on the big shapes — block-count tail/occupancy
effect (512 blocks vs 256 on 60 CUs); rpb=2 net negative.

**Scoping for P3-2(b)** (custom M=1 W16A16):
- Big rows (in_proj/LM head/qkv/o_proj) are **BW-bound at rpb=4**:
  0.95–1.29× floor. Realistic capture there ≈ 0.2–0.5 ms/step (qkv
  1.29× is the main one).
- Small rows (gate_up/down/router/GDN-small, 150 calls/step) are
  **launch/latency-bound**: 3.6–14× floor. LLGemm1 at M=64 is 16 blocks
  × 256 threads — it does not fill 60 CUs; a K-split (MoE-kernel-style)
  M=1 kernel attacks this. Ceiling ≈ 0.5–0.6 ms/step.
- Total realistic P3-2(b) capture ≈ **0.7–1.1 ms/step**, consistent
  with (slightly under) the plan's 1–1.5 ms estimate.

### P3-2(b) — custom W16A16 dense GEMV for M=1 (2026-08-15)

Kernel: `csrc/rocm/dense_gemv_gfx906.cu`, op `_rocm_C.dense_gemv_gfx906(weight[N,K] fp16,
x[1,K] fp16, kchunk) -> out[1,N]`. Row-parallel (LLGemm1-style, `__ockl_fdot2`, fp32 acc),
templates `RPT ∈ {1,2,4}` (rows/thread) × `KCHUNK ∈ {512,2048,4096}` (K per block,
threads = KCHUNK/8). K-split (KSPLIT>1) accumulates via packed fp16 CAS into a
pre-zeroed out; KSPLIT==1 stores directly. RPT selectable at runtime via
`VLLM_GFX906_GEMV_RPT` (bench sweeps); default is the measured rule below.

**v1 correctness bugs (caught in static review, before first GPU run was trusted):**
64-thread K-split CAS overcount (4 lanes CAS same 8 bytes → guarded by `t==0`);
256-thread OOB LDS read in the sibling-row epilogue (lane 3 read `red_smem[4..6]`);
host accepted N%4!=0 with K-split (now: RPT=1 requires kchunk>=K).

**v1 micro-bench — K-split hypothesis was WRONG** (negative result): kc=512 splits are
2.4–4.2× *slower* than LLMM1 on every K=2048 shape; o_proj (K=4096) via 2-chunk split
3.5× slower. fp16-CAS + `zero_` + tiny-block latency dominates at M=1 — unlike the MoE
kernel, where dequant makes blocks compute-heavy and atomics amortize over token rows.
Small rows (router/GDN-small/shared-down) are **launch/latency-bound, not CU-occupancy
bound**: 64-block single-pass does not beat LLMM1's 16-block single-pass. The "3.6–14×
floor" rows cannot be closed with a better GEMM kernel; they belong to kernel-count
reduction (the §2 inter-kernel-gap row).

**v2 — RPT=2 + kc=4096 single pass: real win.** Best-per-shape vs LLMM1 rpb=4:

| shape (n/step) | LLMM1 µs | best cfg µs | delta |
|---|---|---|---|
| qkv 9216×2048 (×10) | 63.2 | kc2048/r2 48.5 | **−23%** |
| router 256×2048 (×40) | 5.4 | kc2048/r2 4.5 | **−17%** |
| in_proj 12288×2048 (×30) | 64.1 | kc2048/r2 59.9 | −6% |
| LM head 248320×2048 (×1) | 1216.5 | kc2048/r2 1137.9 (0.9× floor) | −6% |
| o_proj 2048×4096 (×40) | 22.0 | LLMM1 (kc4096/r2 +4%) | keep |
| shared gate_up 1024 (×40) | 7.6 | LLMM1 (r2 +533%!) | keep |
| shared down / GDN small | 7.1 / 4.5 | LLMM1 / tie | keep |

**Weighted step total: 5203 µs vs 5604 µs (Day-1 rpb=4) = −401 µs/step (−7.2%).**
Pattern: K=2048 single-pass RPT=2 wins for N==256 or N≥2048; RPT=2 is pathologically
bad at N=1024 (512 blocks) — the shape rule is N==256 ∨ N≥2048, not a threshold.
Plan predicted 0.7–1.1 ms/step capture; measured 0.40 ms/step — the small-row
latency-bound reframing above is why the top of the estimate was unreachable.

**Integration:** single choke point `_llmm1_tiny_m` (both n==1 call sites in
`rocm_unquantized_gemm_impl`); routes to the op when K==2048 ∧ fp16 ∧
(N==256 ∨ N≥2048) — everything else stays LLMM1. Kill switch
`VLLM_GFX906_DENSE_GEMV=0`. C++ default RPT = measured rule.

**Build gotcha:** a missing `\` on an interior macro line silently truncated
`LAUNCH_BY_RPT` at `do { if (rpt == 4)` and made the rest parse as code at the
definition site (errors pointed at the #define body). `clang -E | grep` on the file
alone revealed it in seconds — use that for macro surgery, not full rebuilds.

Serving A/B (FULL_DECODE_ONLY, Triton ROCM_ATTN, source-mounted): running.

### Serving-mode backend findings (P3-2b A/B detour, 2026-08-15)

The first source-mounted serving A/B (BENCH_CG_MODE=FULL_DECODE_ONLY,
"VLLM_ATTENTION_BACKEND=ROCM_ATTN") produced **22.44 t/s (GEMV off) /
22.58 (GEMV on)** — half the 44.09 reference. Root cause, two independent
facts:

1. **`VLLM_ATTENTION_BACKEND` no longer exists in this vLLM** (0.27.2rc1.dev
   tree): backend choice is the new priority-based selector
   (`platforms/rocm.py get_valid_backends`); the knob is now
   `attention_config={"backend": ...}` (AttentionConfig.backend). The env var
   was silently ignored.
2. **The fork's gfx906 FA plugin now wins selection by default** ("Overriding
   with CUSTOM out of potential backends: ['CUSTOM', 'ROCM_ATTN',
   'TRITON_ATTN']") — the P3-3 "dead code at runtime" state is gone (plugin
   entry now active).

Then the surprising one: the third run of the sequence requested
**cudagraph_mode=PIECEWISE** (not FULL_DECODE_ONLY) and hit **52.07 t/s**
(CUSTOM FA + GEMV on) — beating the 44.09 Triton-FULL reference. Side-by-side
with identical everything else:

| requested CG mode | backend | GEMV | t/s |
|---|---|---|---|
| FULL_DECODE_ONLY → downgraded to PIECEWISE (warning: CGSupport.NEVER) | CUSTOM | off | 22.44 |
| FULL_DECODE_ONLY → downgraded to PIECEWISE | CUSTOM | on | 22.58 |
| PIECEWISE (requested, compiled piecewise) | CUSTOM | on | **52.07** |

Mechanism (hypothesis, pending diagnosis): FULL_DECODE_ONLY requests a
non-piecewise-compiled model graph; the CGSupport.NEVER downgrade then gives
PIECEWISE *runtime* mode over a non-piecewise-compiled graph → decode
degrades toward eager. Plain PIECEWISE compiles the model with attention
split out → proper piecewise graphs + the 72 µs/layer CUSTOM FA kernel wins
→ 52 t/s. Either way: **requested-PIECEWISE + CUSTOM is the new best
serving config**, and the downgrade path is a real engine bug (P3-3a scope:
fix or guard it; with a working FULL decode capture, CUSTOM+FULL may beat
52). GEMV's 0.40 ms/step GPU win is largely hidden in the downgraded config
(+0.7% only — CPU-launch-bound); its clean A/B is pending under the
52 t/s config (GEMV=0 leg not yet run).

Consequences:
- The "44.09 t/s current best decode" record is **stale**: the current
  default (source-mounted fork, FULL_DECODE_ONLY request) serves at 22.44;
  the current best achievable is 52.07 (requested PIECEWISE). llama.cpp gap
  narrows from ~1.6× to ~1.35×.
- P3-3a sub-plan rescoped: "make CUSTOM serving-viable" is largely met by
  the plain-PIECEWISE path; remaining: correctness under prefix-cache COW /
  multi-batch (probe running), the FULL-downgrade bug, M1 side-buffer
  lifecycle items.
- `_bench_gfx906.py` gained `BENCH_ATTN_BACKEND` (maps to attention_config).

### P3-3a: CUSTOM serving correctness probe — PASSED (2026-08-15)

`/bench/probe_custom_fa.py` (scratch): 2048-token filler prompt, 128 greedy
tokens, same seed/params:
- A = ROCM_ATTN (Triton) + FULL_DECODE_ONLY — reference
- B = CUSTOM (Q8 FA) + requested PIECEWISE — the 52.07 t/s path

**RESULT: IDENTICAL** (first_diff_index=None, 128/128). The degenerate
prompt produces a tight repetition loop — a strong fingerprint; the
V-cache stride-bug class (garbage V → early divergence) would break it
immediately. KV growth across 128 decode steps (Sk 2048→2176) is covered.
Residual correctness gaps for the record: prefix-cache COW and multi-batch
decode are not exercised by this probe (P3-3a items (ii) continue).

Also added `GFX906_FA_CG` env knob to `Gfx906FAMetadataBuilder`
(default never; decode → UNIFORM_SINGLE_TOKEN_DECODE; always → ALWAYS) to
test whether the LEGACY decode path is actually FULL-capture-safe.
Hypothesis basis: the first FULL capture runs with
profile_seq_lens=max_model_len (gpu_model_runner), so Sk-sized buffers
inside forward_paged are allocated at capacity at capture time; the
metadata (seq_lens/block_table) is runner-staged into fixed buffers and
re-read live at replay.

### P3-3a M2 experiment: FULL decode capture works on LEGACY=1 (2026-08-15)

`GFX906_FA_CG=decode` (UNIFORM_SINGLE_TOKEN_DECODE) + requested
FULL_DECODE_ONLY + CUSTOM FA + GEMV on: **53.09 t/s** (18.8 ms/step) —
capture succeeds ("Capturing CUDA graphs (decode, FULL)"), no downgrade
warning, no crash. Beats the 52.07 PIECEWISE number (+1.9%; the step is
GPU-kernel-bound, so the CPU-launch tax on eager-between-pieces attention
was only ~1 t/s, not the 30–40-launch concern of RC3).

Hypothesis confirmed: the LEGACY=1 decode path is FULL-capture-safe as
is. The first FULL capture runs at profile_seq_lens=max_model_len →
Sk-sized buffers inside forward_paged are allocated at capacity at capture
time; seq_lens/block_table/query_start_loc are runner-staged into
pointer-stable buffers re-read live at replay; the decode fast path
(max_seqlen_q==1) takes no host loop / dtype-conversion branch that would
dangle. No W5 buffer surgery needed for LEGACY=1.

**PENDING: correctness probe for the FULL path** (the passing probe covered
PIECEWISE only; 53.09 is not a "new best" until its greedy tokens match the
Triton reference). Then the W8 default flip.

### Serving A/B matrix (pp=2048/tg=256, single req; GEMV = P3-2b)

| # | attention | requested CG | GEMV | FA_CG | t/s | step | notes |
|---|-----------|--------------|------|-------|-----|------|-------|
| 1 | CUSTOM (default) | FULL_DECODE_ONLY | off | never | 22.436 | 44.57 ms | downgrade bug (parent v9) |
| 2 | CUSTOM | FULL_DECODE_ONLY | on | never | 22.584 | 44.28 ms | downgrade bug; GEMV +0.7% |
| 3 | CUSTOM | PIECEWISE | on | never | **52.074** | 19.20 ms | probe-verified correct |
| 4 | CUSTOM | PIECEWISE | off | never | 50.877 | 19.66 ms | clean GEMV A/B: **+0.45 ms/step (+2.3%)** ≈ micro-bench 0.40 ms |
| 5 | CUSTOM | FULL_DECODE_ONLY | on | decode | **53.094** | 18.83 ms | M2 experiment; correctness probe pending |
| 6 | ROCM_ATTN (Triton) | FULL_DECODE_ONLY | off | — | 43.986 | 22.73 ms | reproduces 44.09 archive (−0.2%) |
| 7 | ROCM_ATTN | FULL_DECODE_ONLY | on | — | 44.808 | 22.32 ms | GEMV +1.9% |
| 8 | ROCM_ATTN | PIECEWISE | on | — | 43.955 | 22.75 ms | M0-2 mode-matched FA reference |
| 9 | CUSTOM | FULL_DECODE_ONLY | on | (default) | 52.90 | 18.90 ms | W8 default flip; 5-sample mean, σ≈0.06 |
| 10 | CUSTOM | FULL_DECODE_ONLY | on | (default) | 49.56 | 20.18 ms | Route B stage 1, V2 fused fp16 gather — REGRESSION |
| 11 | CUSTOM | FULL_DECODE_ONLY | on | (default) | 56.92 | 17.57 ms | Route B, V1 fused gather (`GATHER_V=1`), single |
| 12 | CUSTOM | FULL_DECODE_ONLY | on | (default) | **57.09** | 17.52 ms | Route B, V1 default; 5-sample mean, σ≈0.09 — **new best = default config** |

Conclusions:
- 44.09 archive confirmed reproducible (43.99); the 22.44 default-request
  config is the downgrade bug, not model/engine drift. Triton PIECEWISE
  (43.96) ≈ Triton FULL (43.99) — piecewise mode itself costs nothing for
  Triton, so 22.44 cannot be a "piecewise penalty" of any kind.
- GEMV (P3-2b) end-to-end verified in both modes: +0.45 ms/step serving vs
  0.40 ms/step micro-bench (prediction held); +1.9% under Triton-FULL too.
- M2 (FULL capture, LEGACY=1) works: 53.09 t/s; +1.0 t/s over PIECEWISE
  (step is GPU-kernel-bound, CPU launch tax was smaller than RC3 feared).
- Attention win over Triton, mode-matched: FULL 53.09 vs 44.81 = +8.3 t/s
  (+18.5%); PIECEWISE 50.88 (GEMV off) vs 43.99 (GEMV off) = +6.9 t/s
  (+15.7%).
- Route B (fused fp16 gather, V1 default): 57.09 t/s — +4.2 t/s (+7.9%) over
  the 52.90 torch-gather default; probe-verified correct (128/128 greedy,
  bit-exact vs Triton FULL). Default-request config is now 22.44 → 57.09
  (2.54×).

### P3-3a M2 closed (2026-08-15)

- **FULL-path correctness probe: PASSED.** probe2 (Triton-FULL ref vs
  CUSTOM-FULL, 128 greedy tokens @ pp=2048): IDENTICAL (128/128). Case A
  ids reproduced bit-exact across probe runs (stable reference).
- **W8 default flip applied**: `Gfx906FAMetadataBuilder.get_cudagraph_support`
  now returns UNIFORM_SINGLE_TOKEN_DECODE by default (GFX906_FA_CG=never|always
  override retained for experiments). Requested FULL_DECODE_ONLY no longer
  downgrades for this backend; the downgrade path is dormant here and
  remains an upstream-class bug for other NEVER-support backends.
- **Final 5-sample bench** (new default, FULL_DECODE_ONLY + GEMV on):
  52.93 / 52.92 / 52.94 / 52.93 / 52.83 / 52.87 t/s → **mean ≈ 52.90, σ ≈ 0.06**.
  No downgrade warnings; "Capturing CUDA graphs (decode, FULL)" confirmed.
- **T3 capture/replay test added and passing**
  (`test_cudagraph_capture_replay_legacy_decode_path`): warmup@small-Sk →
  capture@capacity; multi-size capture B=1→B=2 with B=1 replay after
  (dangling-buffer class); live seq_lens growth 100→200 with K/V refill
  matching eager at the new length. Existing 2 FA tests stay green.
  (Debug detour: a `.tolist()` inside a debug print during capture raises
  "Cannot copy between CPU and CUDA tensors" — the error is the print, not
  the path; and `arange(n).view(2,-1)` silently loses half the block-table
  columns.)

**P3-3a headline: 22.44 → 52.90 t/s on the default-request config (2.36×).**
Remaining P3-3a items (optional, see sub-plan v4): M0-3 attention-slice
profile, LEGACY=0 fused-gather track (W1/W2/T1/T2) if an A/B ever shows it
beats the LEGACY=1 PyTorch gather.

### Re-baselined decode budget at 52.90 t/s (rocprofv3 --kernel-trace, 2026-08-15)

Profiled the final default config (FULL_DECODE_ONLY + GEMV) under
`rocprofv3 --kernel-trace` (46.2 t/s under tracer; per-dispatch overhead
inflates absolute values ~10-15%, shares are the reliable signal).
Steady-state window: last 4 s of the decode cluster, 185 steps, GPU 99%
busy. Top rows (µs/step, profiler scale): dense_gemv 4366 (91.5 calls —
GEMV covers FA qkv/in_proj/router + GDN in_proj_qkvz/router + LM head),
FA kernel 3272 (11.3 calls ≈ 327 µs/layer @ Sk~2176, i.e. ~4.3× the
Sk~500 72 µs — Sk-linear, as expected), LLMM1/LLGemm1 2505 (the non-GEMV
rows: o_proj K=4096, gate_up N=1024, shared expert, GDN small), MoE wna16
2390 + routing/fused_moe ~1900, GDN rec/conv ~590, and the LEGACY FA
gather+side pile (torch gather + contiguous/mask/permute copies + Fill +
quantize_q8_0 + q_pad zero/copy) ≈ 4-5 ms.

**M0-3 resolved (the sub-plan's outstanding question)**: the LEGACY=1
serving attention slice is NOT "FA 327 µs + gather ~40 µs" — the PyTorch
fancy-index `_gather_kv` costs **128-190 µs/layer** in isolation
(micro-bench, bench_gfx906_fa_gather.py LEGACY section: 189.5 @ Sk=2048,
128.3 @ Sk=2816) vs the fused gather at 19-25 µs/layer. **The v3 demotion
of the M1 fused-gather track was premature** — it is now the biggest
remaining lever: ~0.9-1.4 ms/step (10 FA layers).

**Route B chosen** (stage 1): fused fp16-K gather op
(`gather_paged_kv_fp16`, reuses the byte-generic v2 gather kernel with
bytes_per_row=2D — K stays fp16 in the cache, quantize_q8_0 still runs on
the gathered K, no Q8 side buffer, no RC1/RC2 lifecycle). Expected +4-6
t/s. Stage 2 (if quantize remains visible): fused fp16→q8
quantize-during-gather. Stage 3 (the LEGACY=0 Q8 side-buffer track, W1/W2)
only if more is needed.

### Route B stage 1: fused fp16 gather — built, correct, the V2 serving trap (2026-08-16)

**Implementation.** `gather_paged_kv_fp16` (C++ binding over the byte-generic
gather kernel, `bytes_per_row = 2D`): K stays fp16 in the cache,
`quantize_q8_0` still runs on the gathered K, no Q8 side buffer, no RC1/RC2
lifecycle. Python: LEGACY branch of `forward_paged` now calls the fused op at
`Sk_pad`; `GFX906_FA_TORCH_GATHER=1` reverts to the torch `_gather_kv` for A/B.
Test: `test_fused_fp16_gather_matches_torch_gather`.

**Bug found + fixed:** stride-domain mixup — K is `const uint8_t*` (byte
strides = element stride × 2) but V is `__half*` (element strides, no ×2).
L=32 "worked" by luck (doubled offsets still in-bounds); L=512 faulted.

**Correctness:** probe3 (128/128 greedy tokens) — Triton-FULL vs CUSTOM-FULL
with the fused gather: **bit-exact**, same degenerate-repetition fingerprint as
the earlier probes. The fused path is correct end-to-end.

**Serving regression (V2):** 49.56 t/s vs 52.83 (torch) — the fused op was
SLOWER in serving. Investigation:

- Isolated, the fused fp16 gather is **27-42 µs in every state**: contiguous /
  unbind-view / identity / random bt, pools up to 6.6 GB, with and without a
  512 MB L2 evictor, per-call synced or pipelined. The kernel is not slow.
- Serving profile (rocprofv3 --kernel-trace): the gather kernel runs
  **~285 µs/call, uniform** (p10-p90 = 282-287), vs 41 µs isolated.
- Graph replay IS visible to the kernel trace: the final decode burst contains
  ~2560 gather calls = 256 steps × 10 FA layers — so the 285 µs is real
  replay time, not a windowing artifact.
- **rocprofv3 grid-axis columns are untrustworthy in this build**: for known
  kernels the reported Grid_Size_X/Y/Z does not match the source-computed
  grid under any consistent axis mapping. Use timestamps/durations, not grids.
- Probe artifact (avoid): one L2 test put the evictor `zero_()` INSIDE the
  timed window — 256 MB of zeroes ≈ 320 µs masqueraded as "L2-miss gather
  cost". With the evictor outside the window: 41 µs.

**The decisive A/B (serving, FULL_DECODE_ONLY, same config):**

| LEGACY FA gather | t/s |
|---|---|
| V2 fused (416 WG × 128 thr, `__syncthreads`) | 49.56 |
| torch `_gather_kv` (fancy index) | 52.83 |
| **V1 fused** (`GFX906_FA_GATHER_V=1`, per-token, grid (B,Hkv,Sk), 64 thr, no barriers) | **56.92** |

V1 — the "old" per-token kernel with 16× more WGs and no shared memory — wins
the serving context; V2 degrades 7× (isolated 41 → serving 285 µs) only in
serving. Mechanism not isolated (wave-scheduling / barrier + low-WG-count
interaction with the graph context is the leading candidate), but the
empirical result is unambiguous. **Launcher default flipped to V1**
(`GFX906_FA_GATHER_V=2` selects the old V2). All 4 FA tests pass on the new
default. 5-sample confirmation bench: see below.

**5-sample confirmation (new default, no env):** 57.13 / 57.14 / 57.18 /
57.00 / 57.00 → **mean 57.09 t/s, σ≈0.09**. New best; the default-request
config is now 22.44 → 57.09 (2.54×). llama.cpp gap narrows from 1.43× to
≈1.23×.

**Decision: Route B stage 1 LANDED** (V1 default; V2 behind
`GFX906_FA_GATHER_V=2`; torch path behind `GFX906_FA_TORCH_GATHER=1`).
Remaining FA-side levers, re-ranked by the 52.90 profile: FA kernel itself
(327 µs/layer, Sk-linear — the big one, P3-3 track 2), quantize_q8_0
(~312 µs/step — stage 2 quantize-during-gather candidate), q_pad zero/copy
pile. The V2-serving-degradation mechanism stays an open note (barrier +
low-WG-count kernel in FULL-graph context) — if future kernels for gfx906
serve low-WG shapes, prefer many-small-WG layouts or A/B both.

### Phase-3 code-review fixes (`phase3_code_rev_combined.md`, 2026-08-16)

Addressed the combined adversarial review (qwen + ds4). All items below
landed unless noted.

**C1 (CRITICAL) — GEMV dispatch arch-gating.** `dense_gemv_gfx906` was
routed on every ROCm arch under the skinny condition. Added the
`on_gfx906()` gate in `_llmm1_tiny_m` (both call sites). Mock dispatch
tests (m ∈ {256, 2048, 1024} + an off-gfx906 never-routes guard) added to
`tests/model_executor/layers/test_rocm_unquantized_gemm.py`.

**F1/F6 (HIGH) — capture-safe q_pad + gather buffer lifecycle.**
`_ensure_forward_buffers` / `_ensure_gather_buffers` no longer
free-then-realloc + `empty_cache()` on grow. A buffer that was current
during a capture (tracked by `_q_pad_captured` / `_gather_captured`
latches) is retired into a keep-alive list instead of freed, because the
graph bakes in its VA. The capture-state poll runs only until the first
capture latches the flag → zero steady-state cost (the first version
polled `is_current_stream_capturing()` every step; the latch removes it
from the hot path). New test
`test_q_pad_buffer_survives_capture_then_prefill_grow` drives the real
`Gfx906FAImpl` in the hazardous order — small decode → capture → large
prefill (grow) → decode replay — and asserts retired-buffer liveness +
replay numerics.

**F2/M1 — GEMV numeric tests.** Real kernel vs `F.linear` at K=2048:
kchunk=2048 (model path, RPT=2) and kchunk=512 (K-split atomic CAS
epilogue, bench-only path), m ∈ {256, 2048}, atol=0.15 / rtol=2e-2.
All pass. (M1 resolved as numeric tests rather than a bench-only
carve-out.)

**F4 — RPT env hardening.** `VLLM_GFX906_GEMV_RPT=0` is a hard error;
non-{1,2,4} values warn and fall back to the default rule.

**F5/M2 — V1 gather robustness.** V1 now has the `block_tab_idx >=
max_blocks_per_seq` bounds guard (V2 parity); the launcher switches
V1→V2 when `Sk > 65535` (HIP `gridDim.z` limit).

**F7 — LEGACY=0 RC2 guards.** `get_cudagraph_support` logs a loud ERROR
when LEGACY=0 with prefix caching enabled and a WARNING that LEGACY=0 is
inconsistent with FULL capture and prefix caching. F7b: the three debug
env hooks in `gfx906_fa_paged.py` are documented as eager-only
(host-device syncs, illegal during capture).

**F9 — dead code / stale docs.** Dead `gathered_sk` assignments removed;
`ops.h` kchunk doc corrected; the vendored Russian comments/docstrings in
`gfx906_fa_backend.py`, `gfx906_fa_paged.py`, `__init__.py` translated to
English (the files had never been reviewed); Kevin Read SPDX notice
added alongside the vendor notice in the three `vllm/gfx906_fa/` files;
stale "MVP" header docstrings rewritten.

**F10 — repo hygiene.** `.gitignore` gained `.rocprofv3/` and
`gpucore.*.gpu` (the root-owned 207 MB dumps remain on disk — need sudo
to delete). Root duplicates consolidated: `_bench_gfx906.py` /
`_pp_bench.py` are canonical in `docs/gfx906/` (this phase's
established bench-script home); `run_bench_gfx906.sh` checked in there
(path-fixed); probes `_p31_ab.py`, `probe_custom_fa.py`,
`probe2_custom_fa_full.py` checked in there too (W4 "tests/ or tools/" —
used the project's established scripts area). `bench_ab2.py` /
`test_backend_vs_legacy.py` no longer exist (one-off; their results are
recorded above). The two stale tables fixed: parent plan §1 (44.09 /
1.59× → 57.09 / 1.23×) and sub-plan §0 (52.90 "new best" → 57.09 row).

**F3 (evidence) — perplexity point + multi-batch probe.**
- PPL (fixed 12-prompt natural-text set, 442 prompt tokens,
  prompt-logprob PPL with k=20 top-k, actual-token lookup):
  CUSTOM **6.6811** vs Triton **6.6775** → **+0.05%** (acceptance
  ≤ 2%) — the Q8-K attention quantization is PPL-negligible.
- Multi-batch greedy (2 requests, req2 = req1 prefix + continuation so
  APC COW's the shared blocks; B=2 decode graph; 128 tokens):
  - req1: **128/128 bit-identical** CUSTOM vs Triton, and identical
    across repeated runs.
  - req2: 127/128 vs Triton — first diff at the LAST token, inside a
    degenerate repetition loop (near-tie).
  - req2 vs no-share control (fresh engine, P2 alone): exactly one
    near-tie position (token 120), sequences re-sync afterwards.
  - **Pure Triton shows the same class of non-determinism**: two runs of
    the Triton reference differ on req2 at 2 loop-region positions
    (req1 identical) → engine-level property (likely MoE routing
    tie-breaks), not introduced by the CUSTOM backend.
  - Production B=1 path (probe2, two independent launches): logs
    **byte-identical** → bit-deterministic.
  Interpretation: no corruption in the multi-batch / prefix-sharing
  path; greedy near-tie resolution can differ between runs/backends at
  tied positions (model/engine property); the PPL point is the correct
  aggregate metric.

**H3/M3 — V2 7× in-graph regression, root-cause pass.** Reduced harness
(gather-only graph, no model): V1 eager 33.7 / graph 40.5 µs; **V2 eager
36.5 / graph 38.6 µs (ratio 1.06)**. A gather-only graph does NOT
reproduce the 7× — the anomaly is not a kernel-local graph effect; it
needs the full decode graph context. Mechanism re-characterized: V2's
416 wavefronts fill ~43% of the MI50's 960 wavefront slots, so under a
graph the other branches (MoE / GDN / elementwise) co-reside and
interleave, inflating the observed duration; V1's 6656 wavefronts
saturate the machine, so nothing co-resides. (Full proof would need a
serving kernel trace showing the overlap window — optional follow-up.
Supersedes the earlier "barrier + low-WG-count in graph context"
leading candidate.)

**Bench — no regression from the fixes.** Default-config 3-sample
bench after the fixes: 56.75 / 56.81 / 56.73 (before the capture-poll
latch: 56.77); the same bench on HEAD (`01526dfc69`, the 57.09 commit)
run today: 56.79 / 56.83 / 56.58 → mean 56.73. The 57.09 → ~56.7 drift
is machine-state drift, not code (HEAD ≈ current tree within 0.05%);
57.09 remains the 5-sample record.

**Test status.** `test_gfx906_fa.py`: 5/5 (existing 4 + new lifecycle
test; the fp16-gather test gained a B=2 disjoint-blocks case).
`test_rocm_unquantized_gemm.py`: 8 new GEMV tests pass; the 8
pre-existing mock-based failures (CPU-tensor mocks vs this ROCm/Triton
build) fail identically on HEAD — not ours.

---

## FA kernel track (P3-3a) — B=1 decode parallelism — LANDED

`flash_attn_tile_q8` was the largest remaining non-MoE decode cost
(3.27 ms/step = 10 × 327 µs, Sk-linear). Root cause at B=1: the
launcher hardcoded NC2=1 (no GQA head-packing) and gridDim.y=1 (no KV
split) → 16 blocks = 64 of 960 wavefront slots (6.7%). The vendored
kernel already supported both; the launcher never used them.

**Implementation.** `GFX906_FA_NC2` / `GFX906_FA_KVSPLIT` env knobs in
`gfx906_fa_launcher.cu` (dispatch ladders per ncols1; grid
`(ceil(Sq/NC1), kv_split, B·ceil(Hq/NC2))`), new
`fa_split_combine_kernel` (flash-decoding merge of the per-split m/l
partials, one warp per row) wired through `gfx906_fa::forward` (y>1
allocates `o_part`/`o_meta_split` + combine; y≤1 no-op/memcpy).

**Bugs found & fixed (3).**
1. **Vendor null-mask deref**: `(ncols2 > 1 || mask)` dereferenced
   `mask` unconditionally when NC2>1 → GPU fault at 0x0 with mask=null.
   → `mask != nullptr` in 4 sites (both fattn-q8 .cuh files).
2. **NC2=8 × prefill fault**: ncols=64 config OOB-faults at large Sq
   (first serving run of g8s16 died in prefill). GQA-packing is only
   validated at the decode tile → launcher guard: `nc2>1 && seq_q>2`
   falls back to NC2=1.
3. **Vendor OOB-tail bug (NC2>1 + KV split)**: the strided KV loop
   (step `gridDim.y·nbatch_fa`) never enabled `oob_check=true` for the
   tail tile, unlike the NC2==1 branch → when kv_max is not a multiple
   of nbatch_fa (128 for ncols=16) padding tokens enter the softmax
   (rel err 0.24–0.60 in tests). Fixed in both fattn-q8 .cuh files:
   per-tile `k_VKQ_0 + nbatch_fa > k_VKQ_max` → oob_check=true variant.

**Micro-bench** (`bench_gfx906_fa_decode.py`, B=1, Hq16/Hkv2/D256,
Sq=2, correctness vs fp32 ref at every Sk, maxerr ≤ 0.0048):
legacy 111 ns/token @ 14% HBM; @Sk=2176: NC2=1/y=1 245 µs →
NC2=1/y=8 82.9 → **NC2=8/y=16 58.3 µs (4.2×)**. y=16 is the knee
(y=32/64 regress: combine + empty-split overhead).

**Serving A/B** (docker 0.85, FULL_DECODE_ONLY, default backend;
note: an earlier all-44.7 matrix was self-inflicted — a stale
`BENCH_ATTN_BACKEND=ROCM_ATTN` forced the Triton backend, diagnosed via
kernel trace showing `kernel_paged_attention_2d` 6.46 ms/step and no
`flash_attn_tile_q8`):
| config | t/s |
|---|---|
| NC2=1, y=1 (legacy default) | 57.08 / 57.16 |
| NC2=1, y=8 | 62.13 / 62.15 |
| **NC2=8, y=16** | **62.81 / 62.92** |

**Correctness.** 12/12 `test_gfx906_fa.py` (7 new subprocess tests:
split ± empty trailing splits, GQA pack ± split, short Sk=123 vs
nbatch_fa=128, kv_max 481/512 — the OOB-tail cases fail without fix
3). PPL (12-prompt, 442 tokens, deterministic): legacy 6.6999 vs new
6.6895 = **−0.15%** (Triton 6.6775) — inside noise, far under the 2%
bar. Greedy 4×128-token A/B is **not a valid gate** for this probe set:
legacy×2, new×2, and Triton×2 all diverge across launches (8–115
diffs/req, first-diff positions prompt-specific) — engine-level
non-determinism (MoE routing near-ties), consistent with the earlier
multi-batch finding; the PPL point is the accepted aggregate metric.

**Default flipped** to NC2=8/KVSPLIT=16 (kill switch:
`GFX906_FA_NC2=1 GFX906_FA_KVSPLIT=1`).

**Regression (new default).** Local venv, util 0.95 + fastsafetensors
(see bench-env note below): **62.677 / 62.668 / 62.671** (σ≈0.005) —
matches the docker 0.85 g8s16 pair (62.81/62.92) within machine drift.
Default-request decode: 57.09 → **~62.7 t/s (+9.8%)**; vs the original
44.09 Triton-FULL record 1.42×; llama.cpp ~70 t/s gap 1.23× → **1.12×**.

## Local-venv bench environment (replaces docker for serving benches)

- `source ~/env-rocm-7.14-gfx906.sh` — sets `LD_LIBRARY_PATH` to
  `/opt/rocm-7.14/lib` (the gfx906 ROCm build; the system `/opt/rocm`
  libs are the wrong vintage: libhipsparse symbol mismatch, then RCCL
  missing `ncclCommResume` until the 7.14 point release was updated).
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` **required**: the venv's
  `flash_attn` is the `/local/git/flash-attention-gfx906` fork without a
  built C ext; the env selects the Triton-AMD path. Needed at import
  time for this model's ViT attention wrapper (`fa_utils.py` →
  `flash_attn_varlen_func`).
- `.venv` vllm is an editable install of this repo; the compiled exts
  live in the tree, so docker `pip install -e .` rebuilds are picked up
  without reinstall.
- **fastsafetensors**: GDS unsupported here (cuFileRead errno 22).
  vLLM's GDS→nogds fallback only caught `RuntimeError`; fastsafetensors
  raises a bare `Exception` → engine death. One-line fix in
  `weight_utils.py` (catch `Exception`, keeping the `"gds" in str(e)`
  + not-yielded guards). Load: 41 s vs 117 s default (2.6×).
  Cost: +2.8 GiB live at init (25.21 vs 22.41 GiB) → needs
  `gpu_util 0.95` (KV 1.37 GiB ≈ the 2.11 GiB docker-0.85 headroom;
  B=1 decode numbers unaffected — KV capacity is not the bottleneck).
- Bench recipe: `HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
  BENCH_EAGER=0 BENCH_PP=2048 BENCH_TG=256 BENCH_MAXLEN=3328
  BENCH_GPU_UTIL=0.95 BENCH_CG_MODE=FULL_DECODE_ONLY
  BENCH_LOAD_FORMAT=fastsafetensors BENCH_SAMPLES=N .venv/bin/python
  /tmp/bench/_b.py /local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`
  (`_b.py` gained `BENCH_LOAD_FORMAT`; model path is the real /local
  path now, no docker remap).

## Post-FA-track trace + stage 2: fused gather-and-quantize — LANDED (2026-08-16)

### Fresh rocprofv3 serving trace (NC2=8/KVSPLIT=16 default)

`rocprofv3 --kernel-trace` of the post-FA-track default (55.48 t/s under
tracer; shares only, absolute times inflated 10-15%): 17.49 ms/step kernel
budget, GPU busy 99.5%. Top rows vs the pre-FA-track budget:

| Kernel | µs/step | notes |
|---|---|---|
| dense_gemv<2,2048> | 3936 | 80.7 calls; LM head (N=248320) is 1 call ≈ 1138 µs at 0.9× HBM floor — nothing left |
| LLGemm1<Half,4> | 2021 | 189.3 calls — o_proj/gate_up/shared-down/GDN-small shapes; micro-bench already adjudicated these AGAINST GEMV (o_proj kc4096/r2 +4% vs LLMM1). Dispatch is at its measured optimum |
| moe_gemm_q4 | 2662 | P2-4 territory |
| FillFunctor<Half> + copyBuffer | 1178 | **uncharacterized pile** (~118 fills + 115 D2D copies/step); FA path contributes only ~10 small q_pad zeros + tiny staging; rest is GDN/MoE/runner — candidate P3-4 pass |
| topkGating + align + count_sort | 1044 | P2-4 routing pipeline |
| flash_attn_tile_q8<256,256,2,8> | 475 | 10 calls — FA tile decode (was ~3.0 ms/step pre-FA-track) |
| fa_split_combine | 146 | +146 combine; **FA stack total ≈ 621 µs/step vs 3272 pre-FA-track** |
| quantize_q8_0_dense | 284 | ← stage 2 target |
| gather_paged_kv_q8 (V1) | 174 | ← stage 2 target |

Dense GEMM verdict: dispatch already at the micro-bench optimum (P3-2(b)
evidence: o_proj 2048×4096 GEMV kc4096/r2 is +4% SLOWER than LLMM1; N=1024
gate_up same). No lever there.

### Stage 2: quantize-during-gather (GFX906_FA_FUSED_QUANT, default on)

Replaces the LEGACY decode two-kernel sequence
(`gather_paged_kv_fp16` + `quantize_q8_0`, 174+284 = 458 µs/step under
tracer) with one fused kernel per FA layer: V fp16 copy (V1 semantics,
tail zeroed) + K read from the fp16 paged cache and quantized to q8_0
in-kernel. At B=1 both original kernels are latency/launch-bound
(78-18 GB/s effective vs 798 GB/s HBM), so fusion saves a launch per
layer plus the K fp16 round trip.

Implementation:
- `csrc/gfx906_fa/kernel/q8_0_quantize.cuh` — `quantize_block_q8_0_halfwarp`
  extracted from gfx906_fa_quant.cu (both TUs include it; bit-exact shared
  helper, per-32-block amax via shfl_xor width 32).
- `gather_paged_kv_quant_kernel` in gfx906_fa_gather.cu: grid (B, Hkv,
  Sk), 64 threads/token; halfwave 0 → q8 blocks 0,2,4,6; halfwave 1 →
  1,3,5,7; tail tokens zero V / leave K (same as V1); Sk ≤ 65535
  (gridDim.z cap, same as V1; Python falls back beyond it).
- C++ binding `gather_paged_kv_quantized` (per-call allocs, same
  capture-safety properties as the fp16 gather).
- Python: LEGACY branch of `forward_paged`; `GFX906_FA_FUSED_QUANT=0`
  kill switch reverts to the two-kernel path.

Correctness: `test_fused_gather_quantized_bit_equal_to_gather_then_quantize`
(3 shapes: B=2 [100,300], B=1 [3328], B=1 [33]) asserts the fused K_q8 is
**bit-equal** to quantize_q8_0(gather_paged_kv_fp16) and V bit-equal, on the
production unbind(1) non-contiguous cache layout. 15/15 file pass (incl.
the cudagraph capture test, which now exercises the fused kernel). Because
the FA inputs are bit-identical to the previous default, PPL is unchanged by
construction (6.6895) — no separate PPL gate needed.

Numbers (B=1, Hkv=2, D=256, isolated): Sk=2176: 41.7 → 25.6 µs/call;
Sk=3328: 64.3 → 36.9 µs/call (−27.4 µs/call × 10 layers ≈ −274 µs/step).

Serving A/B (local venv, util 0.95, fastsafetensors, FULL_DECODE_ONLY,
pp=2048/tg=256): OFF (two-kernel) 62.594 / 62.695; **DEFAULT (fused)
63.534 / 63.581 → new record 63.56 t/s** (+1.47% over the 62.67 record).

### Build-system note: hipify.py in-source guard

`cmake/hipify.py` did `shutil.copytree(csrc, csrc)` for in-source builds —
`copytree(dir, dir)` raises SameFileError on this image's Python 3.12, so
any rebuild after a `.cu` edit crashed at the hipify step. Added a 3-line
`abspath(project) != abspath(output)` guard (the copy is a no-op in that
case). Also: never run a bare `cmake <repo>` diagnostic in the repo root —
it pollutes the in-source build state (CMakeCache.txt/build.ninja/CMakeFiles
at the root) and derails the pip editable flow; the canonical build dir is
the pip-generated one, and `.deps/*-subbuild` caches are path-bound
(delete stale subbuilds, keep -src, if FetchContent complains).


---

## 2026-08-19 — FA gather-buffer use-after-free (init Memory Fault) — found & fixed

Symptom: Qwen3-0.6B init (MRV2, default `GFX906_FA_LEGACY=1`) faults 100%
during post-capture warmup with `gather_paged_kv_quant_kernel` in the HW
record; the record's grid `[16384,8,2048]` and name were garbage (proved by
LEGACY-independent constancy, a no-FA control still naming the kernel, and a
launch-API spy seeing no such dispatch).

Root cause: `_ensure_gather_buffers` allocated one exact-shape K+V gather
buffer pair per batch size; FULL-graph capture bakes 35 pairs' VAs (B sweep
1..256), but the keep-alive list held only 4 generations. The descending
capture sweep freed the first-captured (B=256) pair; warmup replayed
`graph_256` -> writes through stale VAs into freed segments.

Fix (`vllm/gfx906_fa/gfx906_fa_backend.py`): smaller-B requests slice the
current buffer `[:B]` (same base VA, one generation for all sizes); real
growth (Sk/Hkv/D) retires into an unbounded dict so captured VAs are never
freed. Latch `_gather_captured` on the slice path too.

**Review hardening (takeover):** the retire dict was keyed by
`(shape, device)` — two generations of identical shape (reachable via
post-capture eager decode + prefill Sk ping-pong) would collide and the
latter entry would free the captured one (UAF recurs). Re-keyed by
`data_ptr`, which is unique among live tensors and a retained tensor is
never freed, so an entry can never be overwritten. New regression test
`test_gather_buffers_capture_sweep_keepalive` drives the real
`Gfx906FAImpl` through warmup -> capture (B=2, then B=1 slice, same base
VA) -> post-capture Sk/B churn that recreates same-shape generations, and
asserts every retired generation (incl. the captured one) stays referenced
and both graphs replay numerically correct.

Verified: repro 4/4 clean on the hardened build (was 10/10 fault);
FA kernel suite 18/18 (17 + new lifecycle test); temp instrumentation
reverted (gather debug prints, C++ dbg block, GFX906_FA_DISABLE gate)
and the .so rebuilt clean. Also added a no-view fast path when
`b.shape[0] == num_seqs` (the exact-size decode case was getting a fresh
TensorImpl per FA layer per step).

Serving re-validation (post-fix build): dense 27B 4-seq 25.33 t/s
(25.26-25.36, record band 25.25-25.34 — no regression); MoE 35B
65.98 / 65.81 t/s over two 4-sample runs vs 65.71 for the OLD backend in
an in-session A/B (stashed fix) — the ~0.5% offset vs the historical
66.3-66.5 band is day-to-day environment variance (old code reproduces
it too), not the fix. Note: MoE 35B production (max_num_seqs=32 ->
7+ captured sizes > old bound of 4) had been exposed to *silent*
corruption under the old bound. Full investigation trail:
/tmp/fa-analysis.md (§11).
