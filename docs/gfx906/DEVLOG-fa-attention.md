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

**GATE (all serving A/Bs in this log, unless a row states otherwise):**
`_bench_gfx906.py`, pp=2048/tg=256, single request, `FULL_DECODE_ONLY`
(the physical-venv bench-env recipe is in `running.md`, not re-stated
here). GEMV = the P3-2(b) `dense_gemv` op. Supporting kernels
(gather, fused-quant) also state their isolated micro-bench config.

## P3-3 — paged attention: the CUSTOM (Q8 FA) backend saga (2026-08-15)

**Starting point:** attention 1.94 ms/step (10 FA layers × 194 µs) vs
llama.cpp 0.19 ms — planned as "partition the Triton kernel". Found the
repo already vendors a custom Q8 FlashAttention backend
(`vllm/gfx906_fa/` + `csrc/gfx906_fa/`, llama.cpp's `flash_attn_tile_q8`,
head_size 256) integrated in `67d2a813a2` as the gfx906 default — **but
dead code at runtime**: the `vllm.general_plugins` entry point is absent
from the stale `vllm.egg-info` AND the image's dist-info, so
`CUSTOM.is_overridden()` was always False → everything fell to Triton.
Silent fallback, never noticed.

**Fix:** `vllm/platforms/rocm.py` `_get_backend_priorities` registers the
plugin explicitly on gfx906 when not already (idempotent; entry-point
installs still win). Priorities become `[CUSTOM, ROCM_ATTN, TRITON_ATTN,
TURBOQUANT]`.

**The real bug hunt** — with the backend live, `GFX906_FA_LEGACY=0` (Q8
side-buffer + fused gather) produced garbage ('!!!!!...'); LEGACY=1 and
FUSED=0 worked. Isolation ladder (each step verified):
1. `reshape_and_cache_q8` vs `quantize_q8_0`: **byte-identical**.
2. Synthetic gather vs torch `_gather_kv_q8`: byte-identical (V tail zeroed).
3. Synthetic end-to-end vs fp32 SDPA (decode+prefill, tails, B=1/2): correct
   (rel ~2.5e-3 = Q8 noise). My first two "references" were wrong (einsum
   axis bugs); pairwise A/B/C identity checks kept it honest.
4. In-model double-gather (`GFX906_FA_DOUBLE_CHECK=1`): **K identical, V
   NaN** — synthetic passed because my test caches were contiguous.

**Root cause (reusable trap):** `value_cache` = `kv_cache.unbind(1)` of
`[num_blocks, 2, block, Hkv, D]` — non-contiguous, block stride 2×.
`gather_paged_kv_q8` (and `forward_paged_direct`) **computed strides from
shapes, ignoring real tensor strides** → read K bytes as V. Only block 0
looked sane. Same latent class in `reshape_and_cache_q8`. Fixed all three
sites to use `tensor.stride(i)` (× element size for fp16 V) + contiguity
TORCH_CHECKs on the last dim. After: `K=True V=True`, exact correct greedy.

**Benchmarks (eager t/s, single req):**

| config | eager t/s | notes |
|---|---|---|
| Triton paged + P3-1 | **19.49** | best eager |
| CUSTOM LEGACY=1 (fp16 gather + per-step Q8 quant) | 18.49 | gather/quant tax; FA kernel 194→72 µs/layer (2.7×) |
| CUSTOM LEGACY=0 FUSED (fixed) | 19.33 | fused gather removes quant tax |
| CUSTOM LEGACY=0 DIRECT forced | 19.21 | block-table indirection tax at B=1 |
| serving (cudagraph) + CUSTOM | **crashes** | `value_cache blocks mismatch` during piecewise capture — deferred |
| serving (cudagraph) + Triton (P3-1) | **44.09** | current best decode at the time |

Also fixed: `_bench_gfx906.py` counted tokens by re-encoding output *text* —
garbage re-encodes to fewer tokens (the mystery "32 tokens" was 256 real
tokens of garbage, 19.05 t/s). Now counts `token_ids`.

**P3-3 near-term outcome:** attention 2.7× faster (72 µs/layer), but at
B=1 eager the win is eaten by gather/conversion tax and eager is
CPU-launch-bound anyway; serving can't use CUSTOM until cudagraph capture
is fixed (capture calls attention with a different/aliasing kv_cache view;
side-buffer alloc assumes the first cache shape). Learnings:
- **Stride bugs hide from synthetic tests that build contiguous caches** —
  always mirror the real allocation path (`unbind` views) in tests.
- Silent registration fallbacks make dead backends invisible; assert the
  backend you expect.
- Q8 K quant changes logits ~1e-3; greedy diverges from fp16 after
  ~10-25 tokens (both fluent) — same trade llama.cpp makes.
- Pairwise A/B identity checks beat building a math reference from scratch.

Next: (a) cudagraph-safe CUSTOM, (b) prefill uses CUSTOM, (c) back to
Triton partitioning.

## Day 1 — gather micro-bench + P3-2(a) probe (2026-08-15)

**Gather go/no-go — GO.** `gather_paged_kv_q8` B=1, Hkv=2, D=256, bs=16,
pre-allocated (serving steady state); byte-identical vs torch incl. V-tail
zeroing:

| Sk | µs/layer | GB/s | ×floor(798) |
|----|----------|------|-------------|
| 2048 | 18.6 | 345 | 2.3× |
| 2816 | **21.7** | 407 | 2.0× |
| 3328 | 25.3 | 412 | 1.9× |

Q-fp32 side costs/layer (Sq=1): q.float 3.9 + q_pad.zero_ 2.7 + q copy 7.0 +
out unpack 8.3 = **21.9 µs/layer** (launch-bound small copies). Combined tax
**43.6 µs/layer** vs the 122 µs/layer FA win (194−72) → net **~0.78 ms/step**
over 10 layers. **P3-3a resumes**; Triton-PIECEWISE baseline bench required
before M1 numbers. M2 note: the Q-side 21.9 µs is as big as the gather
itself.

**P3-2(a) aiter probe — STOPPED (structurally a no-op on gfx906).** Code
audit of `rocm_unquantized_gemm_impl`: `VLLM_ROCM_USE_AITER` defaults False;
every aiter gemm path is inside `if not on_gfx906()` (`gemm_a16w16`,
`wvSplitKrc`); `ops.wvSplitK` explicitly excluded ("matrix cores not
supported"); aiter triton-gemm whitelist is GPT-OSS shapes, none match. →
**arch exclusion, not shape/dtype selection.**

**LLGemm1 rows_per_block sweep** (the one real knob; dispatch hardcodes 4):

| shape (M×K) | ×/step | rpb2 | rpb4 (cur) | rpb8 | rpb16 | floor |
|---|---|---|---|---|---|---|
| 12288×2048 in_proj | 30 | 57.8 | 64.7 | 60.1 | 61.0 | 63.1 |
| 248320×2048 LM head | 1 | 1134.6 | 1209.8 | 1244.2 | 1178.6 | 1274.6 |
| 9216×2048 qkv | 10 | 43.9 | 60.9 | 49.7 | 51.0 | 47.3 |
| 2048×4096 o_proj | 40 | 21.5 | 23.0 | 25.3 | 28.9 | 21.0 |
| 1024×2048 shared gate_up | 40 | **47.7** | 7.6 | 8.8 | 9.9 | 5.3 |
| 2048×512 shared down | 40 | 5.5 | 7.1 | 6.8 | 7.8 | 2.6 |
| 256×2048 router | 40 | 4.7 | 5.0 | 5.1 | 5.9 | 1.3 |
| 64×2048 GDN small | 30 | 4.6 | 4.6 | 4.6 | 5.7 | 0.3 |

Weighted/step: rpb4 5604 µs, rpb8 5523 (+1.4% best), rpb16 5793, rpb2 6626
(rpb2 is 6× slower on gate_up while winning 6-11% on big shapes — block-tail
effect; net negative). No shape moves ≥20% → config retune not worth it.

**P3-2(b) scoping:** big rows (in_proj/LM/qkv/o_proj) are **BW-bound at
rpb=4** (0.95–1.29× floor, capture ~0.2–0.5 ms/step); small rows
(gate_up/down/router/GDN-small, 150 calls/step) are **launch/latency-bound**
(3.6–14× floor; a K-split M=1 kernel targets this, ceiling ~0.5–0.6
ms/step). Total realistic ~0.7–1.1 ms/step.

## P3-2(b) — custom W16A16 dense GEMV for M=1 (2026-08-15)

Kernel `dense_gemv_gfx906.cu`, op `_rocm_C.dense_gemv_gfx906(weight[N,K]
fp16, x[1,K], kchunk)->out[1,N]`. Row-parallel (LLGemm1-style,
`__ockl_fdot2`, fp32 acc), templates `RPT×KCHUNK`; K-split >1 accumulates
packed fp16 CAS (pre-zeroed out), ==1 direct store.

**v1 correctness bugs (caught in static review):** 64-thread K-split CAS
overcount (4 lanes CAS same 8B → guard `t==0`); 256-thread OOB LDS read in
sibling-row epilogue (lane 3 read `red_smem[4..6]`); host accepted N%4!=0
with K-split.

**v1 micro-bench — K-split hypothesis WRONG:** kc=512 splits 2.4–4.2×
*slower* than LLMM1 on every K=2048 shape; o_proj 3.5× slower. fp16-CAS +
`zero_` + tiny-block latency dominates at M=1 (unlike the MoE kernel, where
dequant makes blocks compute-heavy). Small rows are launch/latency-bound,
not CU-occupancy-bound — a better GEMM can't close them; belongs to
kernel-count reduction.

**v2 — RPT=2 + kc=4096 single pass, real win.** Best-per-shape vs LLMM1
rpb4:

| shape (n/step) | LLMM1 µs | best cfg µs | delta |
|---|---|---|---|
| qkv 9216×2048 (×10) | 63.2 | kc2048/r2 48.5 | **−23%** |
| router 256×2048 (×40) | 5.4 | kc2048/r2 4.5 | **−17%** |
| in_proj 12288×2048 (×30) | 64.1 | kc2048/r2 59.9 | −6% |
| LM head 248320×2048 (×1) | 1216.5 | kc2048/r2 1137.9 (0.9× floor) | −6% |
| o_proj 2048×4096 (×40) | 22.0 | LLMM1 (kc4096/r2 +4%) | keep |
| shared gate_up 1024 (×40) | 7.6 | LLMM1 (r2 +533%!) | keep |
| shared down / GDN small | 7.1 / 4.5 | LLMM1 / tie | keep |

**Weighted step: 5203 vs 5604 µs (−401 µs/step, −7.2%).** Shape rule:
K=2048 single-pass RPT=2 wins N==256 ∨ N≥2048 (RPT=2 pathological at
N=1024). Plan predicted 0.7–1.1 ms/step; measured 0.40 — small-row
latency-bound reframing explains the miss.

**Integration:** single choke point `_llmm1_tiny_m`, routes to the op when
K==2048 ∧ fp16 ∧ (N==256 ∨ N≥2048); else LLMM1. Kill switch
`VLLM_GFX906_DENSE_GEMV=0`.

**Build gotcha:** a missing `\` on an interior macro line silently truncated
`LAUNCH_BY_RPT`; `clang -E | grep` on the file alone revealed it — use that
for macro surgery, not full rebuilds.

## Serving-mode backend findings (P3-2b A/B detour, 2026-08-15)

First source-mounted A/B gave **22.44 (GEMV off) / 22.58 (on)** — half the
44.09 reference. Two independent facts:
1. **`VLLM_ATTENTION_BACKEND` no longer exists** (0.27.2rc1.dev): backend is
   the priority-based selector (`platforms/rocm.py get_valid_backends`); the
   knob is now `attention_config={"backend": ...}`. The env var was silently
   ignored.
2. **The gfx906 FA plugin now wins by default** (the P3-3 dead-code state is
   gone).

Then the surprise: the third run requested **PIECEWISE** (not
FULL_DECODE_ONLY) and hit **52.07 t/s** (CUSTOM + GEMV) — beating the 44.09
Triton-FULL ref:

| requested CG mode | backend | GEMV | t/s |
|---|---|---|---|
| FULL_DECODE_ONLY → downgraded to PIECEWISE (CGSupport.NEVER warn) | CUSTOM | off | 22.44 |
| FULL_DECODE_ONLY → downgraded to PIECEWISE | CUSTOM | on | 22.58 |
| PIECEWISE (requested, compiled piecewise) | CUSTOM | on | **52.07** |

**Mechanism (hypothesis):** FULL_DECODE_ONLY requests a non-piecewise graph;
the CGSupport.NEVER downgrade then runs PIECEWISE *runtime* over it → decode
degrades toward eager. Plain PIECEWISE compiles attention out → proper
piecewise graphs + the 72 µs/layer CUSTOM FA wins → 52. The downgrade path
is a real engine bug (P3-3a scope). **requested-PIECEWISE + CUSTOM = new
best serving config**; GEMV's 0.40 ms win is hidden in the downgraded config
(+0.7% only). `_bench_gfx906.py` gained `BENCH_ATTN_BACKEND`.

## P3-3a: CUSTOM serving correctness probe — PASSED (2026-08-15)

`/bench/probe_custom_fa.py`: 2048-tok filler, 128 greedy. A=ROCM_ATTN+
FULL_DECODE_ONLY (ref) vs B=CUSTOM+PIECEWISE (the 52.07 path).
**RESULT: IDENTICAL** (128/128). Degenerate-repetition fingerprint breaks
any V-stride bug instantly; KV growth Sk 2048→2176 covered. Residual gaps
(not exercised): prefix-cache COW, multi-batch.

Also added `GFX906_FA_CG` knob (default-never; decode→
UNIFORM_SINGLE_TOKEN_DECODE; always→ALWAYS) to test if LEGACY decode is
FULL-capture-safe. Hypothesis: first FULL capture runs at
profile_seq_lens=max_model_len → Sk-sized buffers allocated at capacity at
capture; metadata side-staged into fixed buffers, re-read live at replay.

## P3-3a M2 experiment: FULL decode capture works on LEGACY=1 (2026-08-15)

`GFX906_FA_CG=decode` + FULL_DECODE_ONLY + CUSTOM + GEMV: **53.09 t/s** (18.8
ms/step) — capture succeeds, no downgrade, no crash. +1.9% over PIECEWISE
(the step is GPU-bound; CPU launch tax on eager-between-pieces attention was
only ~1 t/s, not the RC3-feared 30-40 launches). Hypothesis confirmed:
LEGACY=1 decode is FULL-capture-safe as-is; the decode fast path
(max_seqlen_q==1) takes no dangling host loop. **PENDING: FULL-path
correctness probe** before 53.09 is a "new best"; then the W8 default flip.

## Serving A/B matrix (GATE; GEMV = P3-2b)

| # | attention | requested CG | GEMV | FA_CG | t/s | step | notes |
|---|---|---|------|-------|-----|------|-------|
| 1 | CUSTOM (default) | FULL_DECODE_ONLY | off | never | 22.436 | 44.57 ms | downgrade bug |
| 2 | CUSTOM | FULL_DECODE_ONLY | on | never | 22.584 | 44.28 ms | downgrade bug; GEMV +0.7% |
| 3 | CUSTOM | PIECEWISE | on | never | **52.074** | 19.20 ms | probe-verified correct |
| 4 | CUSTOM | PIECEWISE | off | never | 50.877 | 19.66 ms | clean GEMV A/B: **+0.45 ms/step (+2.3%)** |
| 5 | CUSTOM | FULL_DECODE_ONLY | on | decode | **53.094** | 18.83 ms | M2; correctness probe pending |
| 6 | ROCM_ATTN (Triton) | FULL_DECODE_ONLY | off | — | 43.986 | 22.73 ms | reproduces 44.09 (−0.2%) |
| 7 | ROCM_ATTN | FULL_DECODE_ONLY | on | — | 44.808 | 22.32 ms | GEMV +1.9% |
| 8 | ROCM_ATTN | PIECEWISE | on | — | 43.955 | 22.75 ms | mode-matched FA ref |
| 9 | CUSTOM | FULL_DECODE_ONLY | on | (default) | 52.90 | 18.90 ms | W8 flip; 5-sample, σ≈0.06 |
| 10 | CUSTOM | FULL_DECODE_ONLY | on | (default) | 49.56 | 20.18 ms | V2 fused gather — REGRESSION |
| 11 | CUSTOM | FULL_DECODE_ONLY | on | (default) | 56.92 | 17.57 ms | V1 fused (`GATHER_V=1`), single |
| 12 | CUSTOM | FULL_DECODE_ONLY | on | (default) | **57.09** | 17.52 ms | V1 default; 5-sample, σ≈0.09 — new best |

**Takeaways (the non-obvious ones the table doesn't already say):**
- The 22.44 rows are the **downgrade bug**, not engine drift: Triton
  PIECEWISE (43.96) ≈ Triton FULL (43.99), so "piecewise penalty" is refuted.
- **GEMV prediction held**: ~0.40 ms/step micro-bench → +0.45 serving (+2.3%).
- **Route B (fused fp16 gather, V1 default) is the config**: 57.09 t/s,
  +2.54× over the 22.44 default-request start; llama.cpp gap 1.23×.

## P3-3a M2 closed (2026-08-15)

- **FULL-path correctness: PASSED** (probe2 Triton-FULL vs CUSTOM-FULL, 128/128
  identical).
- **W8 default flip:** `get_cudagraph_support` returns
  UNIFORM_SINGLE_TOKEN_DECODE by default; FULL_DECODE_ONLY no longer downgrades
  here (the bug stays for other NEVER-support backends).
- **Final 5-sample:** 52.93/52.92/52.94/52.93/52.83/52.87 → **52.90, σ≈0.06**.
- **T3 test added** (`test_cudagraph_capture_replay_legacy_decode_path`):
  warmup@small-Sk → capture@capacity; multi-size B=1→2 with B=1 replay
  (dangling-buffer class); live seq_lens 100→200 with K/V refill.
  (Debug detour: a `.tolist()` in a debug print raises "Cannot copy between
  CPU and CUDA" — the error is the print, not the path; `arange(n).view(2,-1)`
  silently loses half the block-table columns.)

**P3-3a headline: 22.44 → 52.90 t/s default-request (2.36×).** Remaining
(optional): M0-3 attention-slice profile, LEGACY=0 fused-gather track.

## Re-baselined decode budget at 52.90 t/s (rocprofv3 --kernel-trace, 2026-08-15)

Profile of the default config (46.2 t/s under tracer; per-dispatch overhead
inflates absolutes ~10-15%; shares reliable). Top (µs/step): dense_gemv 4366
(91.5 — qkv/in_proj/router/LM head), FA kernel 3272 (11.3 ≈ 327 µs/layer @
Sk~2176 ≈ 4.3× the Sk~500 72 µs — Sk-linear), LLMM1 2505, MoE wna16 2390 +
routing ~1900, GDN rec/conv ~590, and the **LEGACY FA gather+side pile**
(torch gather + copies + Fill + quantize + q_pad zero) ≈ 4-5 ms.

**M0-3 resolved:** the LEGACY=1 attention slice is NOT "FA 327 + gather ~40
µs" — the torch fancy-index `_gather_kv` costs **128-190 µs/layer** (189.5 @
Sk=2048, 128.3 @ Sk=2816) vs the fused gather's 19-25. **The v3 demotion of
the M1 fused-gather track was premature** — it's the biggest remaining lever,
~0.9-1.4 ms/step.

**Route B chosen (stage 1):** fused fp16-K gather (`gather_paged_kv_fp16`,
byte-generic v2 kernel, bytes_per_row=2D; K stays fp16, quantize still runs
on gathered K). Expected +4-6 t/s. Stage 2 (if quantize visible): fused
fp16→q8 quantize-during-gather.

## Route B stage 1: fused fp16 gather — built, correct, the V2 serving trap (2026-08-16)

`gather_paged_kv_fp16` (C++ over the byte-generic gather, bytes_per_row=2D):
K stays fp16 in cache, `quantize_q8_0` runs on gathered K, no Q8 side buffer.
Python LEGACY branch of `forward_paged` calls it at `Sk_pad`;
`GFX906_FA_TORCH_GATHER=1` reverts to torch. Test added.

**Bug: stride-domain mixup** — K is `const uint8_t*` (byte strides ×2) but V
is `__half*` (element strides). L=32 "worked" by luck; L=512 faulted.

**Correctness:** probe3 128/128 **bit-exact** (Triton-FULL vs CUSTOM-FULL).

**Serving regression (V2):** 49.56 vs 52.83 torch — SLOWER in serving.
Isolated the fused gather is 27-42 µs in every state (contiguous / unbind /
identity / random bt / 6.6 GB pools / synced / pipelined); serving profile
shows **~285 µs/call uniform** (p10-p90 282-287) vs 41 isolated. Graph replay
IS visible (256 steps × 10 layers ≈ 2560 calls). rocprofv3 grid-axis columns
are untrustworthy in this build — use timestamps/durations, not grids.
(Probe-artifact to avoid: one L2 test put the evictor `zero_()` inside the
timed window — 256 MB of zeroes ≈ 320 µs masqueraded as gather cost.)

**Decisive A/B (serving, FULL_DECODE_ONLY):**

| LEGACY FA gather | t/s |
|---|---|
| V2 fused (416 WG × 128 thr, `__syncthreads`) | 49.56 |
| torch `_gather_kv` (fancy index) | 52.83 |
| **V1 fused** (`GFX906_FA_GATHER_V=1`, per-token, grid(B,Hkv,Sk), 64 thr, no barriers) | **56.92** |

**V1 — the per-token kernel with 16× more WGs and no smem — wins the serving
context**; V2 degrades 7× (isolated 41 → serving 285 µs) only in serving.
Mechanism: V2's low-WG + barrier co-resides with other decode branches (see
H3/M3 below). **Launcher default flipped to V1** (`GFX906_FA_GATHER_V=2`
selects V2). **5-sample confirm:** 57.13/57.14/57.18/57.00/57.00 → **57.09,
σ≈0.09**; llama.cpp gap 1.23×.

**Decision: Route B stage 1 LANDED (V1 default).** Remaining FA-side levers:
FA kernel itself (327 µs/layer, Sk-linear), quantize_q8_0 (~312 µs/step —
stage 2 candidate), q_pad zero/copy pile.

## Phase-3 code-review fixes (`phase3_code_rev_combined.md`, 2026-08-16)

- **C1 (CRITICAL) — GEMV dispatch arch-gating:** `dense_gemv_gfx906` was
  routed on every ROCm arch. Added `on_gfx906()` in `_llmm1_tiny_m` (both
  call sites) + dispatch tests.
- **F1/F6 (HIGH) — capture-safe q_pad/gather buffer lifecycle:** no longer
  free-then-realloc+`empty_cache()` on grow; a buffer current during capture
  (`_q_pad_captured`/`_gather_captured` latches) is retired to a keep-alive
  list (graph bakes its VA). Latch makes the poll zero in steady state. New
  test drives small-decode → capture → large-prefill-grow → decode replay.
- **F2/M1 — GEMV numeric tests:** real kernel vs `F.linear`, kchunk=2048
  (RPT=2) + kchunk=512 (K-split), m∈{256,2048}. Pass.
- **F4 — RPT env hardening:** `VLLM_GFX906_GEMV_RPT=0` errors; non-{1,2,4}
  warns+falls back.
- **F5/M2 — V1 gather robustness:** `block_tab_idx >= max_blocks_per_seq`
  guard (V2 parity); V1→V2 auto when `Sk > 65535` (gridDim.z).
- **F7 — LEGACY=0 RC2 guards:** loud ERROR on LEGACY=0 + prefix caching;
  WARNING it's inconsistent with FULL capture. F7b: debug env hooks are
  eager-only (syncs illegal under capture).
- **F9 — dead code / stale docs:** removed dead `gathered_sk`; kept `ops.h`
  kchunk doc; translated vendored Russian comments in the 3 `gfx906_fa/`
  files; added Kevin Read SPDX alongside vendor; rewrote stale MVP headers.
- **F10 — repo hygiene:** `.gitignore` + `.rocprofv3/` `gpucore.*.gpu`;
  bench scripts canonical in `docs/gfx906/`; stale tables re-pointed.
- **F3 (evidence):** PPL CUSTOM 6.6811 vs Triton 6.6775 = +0.05% (Q8-K
  PPL-negligible). Multi-batch greedy: req1 128/128 identical + across runs;
  req2 127/128 (near-tie at loop end); **pure Triton shows the same
  run-to-run non-determinism** → engine-level (MoE routing near-ties), not
  the CUSTOM backend. B=1 path logs byte-identical (bit-deterministic).
- **H3/M3 — V2 7× in-graph regression root-caused:** gather-only graph does
  NOT reproduce it (V1 eager 33.7/graph 40.5; V2 eager 36.5/graph 38.6,
  ratio 1.06). Needs the full decode graph. Re-characterized: V2's 416 WGs
  fill ~43% of 960 wavefront slots → co-reside with MoE/GDN/elementwise and
  interleave, inflating observed duration; V1's 6656 WGs saturate, nothing
  co-resides. (Supersedes the "barrier + low-WG" candidate.)

**Bench — no regression:** 56.75/56.81/56.73 post-fix; HEAD `01526dfc69`
today 56.79/56.83/56.58 → mean 56.73. The 57.09→~56.7 drift is machine-state,
not code.

**Test status:** `test_gfx906_fa.py` 5/5; `test_rocm_unquantized_gemm.py` 8
new GEMV pass, 8 pre-existing mock failures not ours.

---

## FA kernel track (P3-3a) — B=1 decode parallelism — LANDED

`flash_attn_tile_q8` was the largest remaining non-MoE decode cost (3.27
ms/step, Sk-linear). Root cause at B=1: the launcher hardcoded NC2=1 (no GQA
head-packing) and gridDim.y=1 (no KV split) → 16 blocks = 64/960 wavefront
slots (6.7%). The vendored kernel already supported both.

**Implementation:** `GFX906_FA_NC2`/`GFX906_FA_KVSPLIT` knobs (dispatch
ladders; grid `(ceil(Sq/NC1), kv_split, B·ceil(Hq/NC2))`); new
`fa_split_combine_kernel` (flash-decoding merge of per-split m/l partials,
one warp/row; y≤1 no-op/memcpy).

**Bugs found & fixed (3):**
1. **Vendor null-mask deref:** `(ncols2 > 1 || mask)` derefs `mask`
   unconditionally when NC2>1 → GPU fault at 0x0. → `mask != nullptr` in 4
   sites.
2. **NC2=8 × prefill fault:** ncols=64 OOB-faults at large Sq (GQA-packing
   validated only at decode tile). Guard: `nc2>1 && seq_q>2` → NC2=1.
3. **Vendor OOB-tail (NC2>1 + KV split):** the strided KV loop never enabled
   `oob_check` for the tail tile → kv_max not a multiple of nbatch_fa lets
   padding into the softmax (rel 0.24-0.60). Fixed: per-tile
   `k_VKQ_0 + nbatch_fa > k_VKQ_max` → oob_check.

**Micro-bench** (`bench_gfx906_fa_decode.py`, B=1, Hq16/Hkv2/D256, Sq=2,
correctness vs fp32 ref, maxerr ≤ 0.0048): @Sk=2176, NC2=1/y=1 245 µs →
NC2=1/y=8 82.9 → **NC2=8/y=16 58.3 µs (4.2×)**. y=16 knee (32/64 regress).

**Serving A/B** (docker 0.85, FULL_DECODE_ONLY; a stale
`BENCH_ATTN_BACKEND=ROCM_ATTN` showing 44.7 was self-inflicted — reused
Triton):

| config | t/s |
|---|---|
| NC2=1, y=1 (legacy default) | 57.08 / 57.16 |
| NC2=1, y=8 | 62.13 / 62.15 |
| **NC2=8, y=16** | **62.81 / 62.92** |

**Correctness:** 12/12 `test_gfx906_fa.py` (7 new: split ± empty trailing,
GQA pack ± split, short Sk, kv_max edge — OOB cases fail without fix 3).
PPL legacy 6.6999 vs new 6.6895 = −0.15% (Triton 6.6775) — inside noise.
Greedy 4×128 A/B is **not a valid gate** here: legacy×2/new×2/Triton×2 all
diverge across launches (engine-level non-determinism); **PPL is the metric.**

**Default flipped:** NC2=8/KVSPLIT=16 (kill: `GFX906_FA_NC2=1
GFX906_FA_KVSPLIT=1`). **Regression:** local venv **62.677/62.668/62.671**
(σ≈0.005). Default-request decode 57.09 → **~62.7 (+9.8%)**; llama.cpp gap
1.12×.

*(Bench env for the local numbers is in `running.md` — not re-stated.)*

## Post-FA-track trace + stage 2: fused gather-and-quantize — LANDED (2026-08-16)

`rocprofv3 --kernel-trace` of the post-FA-track default (55.48 t/s under
tracer; shares reliable), 17.49 ms/step kernel budget, GPU 99.5% busy. Top
rows:

| Kernel | µs/step | notes |
|---|---|---|
| dense_gemv<2,2048> | 3936 | LM head 1 call ≈ 1138 µs at 0.9× floor — nothing left |
| LLGemm1<Half,4> | 2021 | micro-bench already adjudicated AGAINST GEMV — at optimum |
| moe_gemm_q4 | 2662 | P2-4 territory |
| FillFunctor + copyBuffer | 1178 | uncharacterized pile; FA ~10 small q_pad zeros; candidate P3-4 |
| topkGating + align + count_sort | 1044 | P2-4 routing |
| flash_attn_tile_q8<256,256,2,8> | 475 | 10 calls; FA stack ≈ 621 vs 3272 pre-FA-track |
| fa_split_combine | 146 | in the 621 |
| quantize_q8_0_dense | 284 | ← stage 2 target |
| gather_paged_kv_q8 (V1) | 174 | ← stage 2 target |

**Stage 2: quantize-during-gather (`GFX906_FA_FUSED_QUANT`, default on)** —
replaces the LEGACY two-kernel (`gather_paged_kv_fp16` + `quantize_q8_0`,
174+284 = 458 µs/step) with one fused kernel/layer: V fp16 copy (V1
semantics, tail zeroed) + K read from the fp16 paged cache quantized to q8_0
in-kernel. At B=1 both originals are latency/launch-bound (78-18 GB/s), so
fusion saves a launch + the K fp16 round trip.

- Kernel: `q8_0_quantize.cuh` `quantize_block_q8_0_halfwarp` (bit-exact
  shared helper) + `gather_paged_kv_quant_kernel` (grid(B,Hkv,Sk), 64 thr,
  halfwave→q8 blocks; Sk≤65535). C++ `gather_paged_kv_quantized`.
- Python LEGACY branch; `GFX906_FA_FUSED_QUANT=0` reverts.
- Correctness: `test_fused_gather_quantized_bit_equal_...` (3 shapes) asserts
  fused K_q8 **bit-equal** to quantize(gather) on the production unbind(1)
  layout. 15/15 pass. PPL unchanged by construction (6.6895).
- Numbers (B=1, Hkv=2, isolated): Sk=2176 41.7→25.6 µs/call; Sk=3328
  64.3→36.9 (−27.4 × 10 ≈ −274 µs/step).
- **Serving A/B:** OFF 62.594/62.695; **DEFAULT (fused) 63.534/63.581 → 63.56
  t/s new record** (+1.47% over 62.67).

---

## 2026-08-19 — FA gather-buffer use-after-free (init Memory Fault) — found & fixed

Symptom: Qwen3-0.6B init (MRV2, default LEGACY=1) faults 100% during
post-capture warmup; `gather_paged_kv_quant_kernel` in the HW record with
garbage grid `[16384,8,2048]` and name (proved by LEGACY-independent
constancy, a no-FA control still naming it, a launch-API spy seeing no such
dispatch).

**Root cause:** `_ensure_gather_buffers` allocated one exact-shape K+V pair
per batch size; FULL-graph capture bakes 35 pairs' VAs (B sweep 1..256), but
the keep-alive list held only 4 generations. The descending capture sweep
freed the first-captured (B=256) pair; warmup replayed `graph_256` → wrote
through stale VAs into freed segments.

**Fix (`gfx906_fa_backend.py`):** smaller-B requests slice the current buffer
`[:B]` (same base VA, one generation for all sizes); real growth retires into
an unbounded dict so captured VAs are never freed. Latch `_gather_captured`
on the slice path too.

**Review hardening (takeover):** the retire dict was keyed by `(shape,
device)` — two generations of identical shape would collide (latter entry
frees the captured one; UAF recurs). Re-keyed by **`data_ptr`** (unique among
live tensors; a retained tensor is never freed, so an entry can't be
overwritten). New regression test drives warmup→capture (B=2 then B=1 slice,
same VA)→Sk/B churn recreating same-shape generations; asserts every retired
generation stays referenced + both graphs replay correct.

**Verified:** repro 4/4 clean (was 10/10 fault); FA suite 18/18 (+new);
instrumentation reverted. Also added a no-view fast path when
`b.shape[0] == num_seqs` (the exact-size decode case was making a fresh
TensorImpl per FA layer per step).

**Serving re-validation:** dense 27B 4-seq 25.33 t/s (record band, no
regression); MoE 65.98/65.81 (in-session A/B ~0.5% vs 66.3-66.5 = day variance,
not the fix). **MoE production (max_num_seqs=32 → 7+ captured sizes > old
bound 4) had been exposed to silent corruption under the old bound.** Full
trail: `/tmp/fa-analysis.md` (§11).