# Phase 3 code review — deepseek (ds4) — critical pass

Copyright Kevin Read <me@kevin-read.com>

Scope: every commit in `864915074...d6daa7..HEAD` (20 commits) — P3-1 tiny-m
padded LLMM1, P3-2(b) custom W16A16 GEMV, P3-3/P3-3a gfx906_FA serving work
(gather stride fixes, fused fp16/FP16 gather, CGSupport flip, backend
registration fallback), plus the bench/plan/devlog/docs additions. Reviewed
tree state at `HEAD` (`01526dfc69`, Route B stage 1 landed; default serving
22.44 → 57.09 t/s).

Verdict: the work is disciplined, measured, and honest about unknowns — this
is above the normal bar. The headline numbers are credible. My findings are
about **scope leakage** (a gfx906-named kernel routed into all ROCm archs),
**correctness-evidence gaps** against the plan's own acceptance criteria, and
one **unexplained serving regression that was accepted without root cause**.
None invalidates the 57.09 t/s claim.

---

## 1. Strengths (worth retaining)

- **Measured, gated process.** Every lever went micro-bench → A/B → commit
  (`_llmm1_tiny_m`, dense GEMV per-shape table, gather micro-bench as the
  P3-3a go/no-go gate, M2 `GFX906_FA_CG=decode` experiment before the W8
  default flip). Negative results are recorded (K-split false, aiter arch
  exclusion, V2 serving regression).
- **Kill switches / env A/B everywhere** (`VLLM_GFX906_DENSE_GEMV=0`,
  `GFX906_FA_GATHER_V`, `GFX906_FA_TORCH_GATHER`, `GFX906_FA_CG`,
  `GFX906_FA_FUSED`), so regressions are one knob away from baseline.
- **Real stride-domain fix** (`gfx906_fa.cpp`): deriving strides from the
  tensor's real strides instead of shapes closed a genuine K-reads-V-poison
  bug class, and the test builds the cache via `kv_cache.unbind(1)` exactly
  like serving so it can't hide again. Good defensive `TORCH_CHECK`s on the
  last-dim-contiguous invariant.
- **T3 capture/replay test** (`test_cudagraph_capture_replay_legacy_decode_path`)
  is a real capture-safety test (multi-size capture + live seq_lens replay),
  not a smoke test. That class of bug is normally invisible.
- **Documented, reproducible benchmark discipline** (`_bench_gfx906.py`,
  5-sample σ, captured cudagraph_mode, env state next to every number).
- Naming, copyright headers, and the plan/DEVLOG cross-references are
  consistent.

---

## 2. Critical findings

### C1 — The gfx906-named GEMV is routed into ALL ROCm archs with no arch gate
`vllm/model_executor/layers/utils.py` (P3-2b), `_llmm1_tiny_m` + both
dispatch sites.

The new `dense_gemv_gfx906` op is selected purely by
`VLLM_GFX906_DENSE_GEMV≠0 ∧ fp16 ∧ contiguous ∧ K==2048 ∧ (m==256 ∨ m≥2048)`
— there is **no `on_gfx906()` check**. It sits in `_llmm1_tiny_m`, which is
now called from **both** `n==1` branches:

- the `use_skinny` block (`on_gfx9() or on_gfx1x() or on_gfx906()`) **and**
- the **non-skinny fallback block at `utils.py:~426`** which has **no arch
  condition at all** (just `m%4==0 or m<4 ∧ n==1 ∧ k≤8192 ∧ bias is None`).

So on CDNA2/3 (gfx9x) and RDNA (gfx11x) ROCm targets that are **not gfx906**,
any M=1 fp16 GEMV with N==256 or N≥2048 at K=2048 that doesn't get caught by
aiter/wvSplitK now silently routes through a kernel that:
- was only ever measured on gfx906 (Vega20, no MFMA) and is untuned (likely
  slower) on MFMA-class targets where `LLMM1`/`wvSplitK` use matrix cores; and
- changes the fp reduction order to fp32-acc + IEEE round-to-nearest
  (`__float2half_rn`), where the previous path used the target's native path.

The lane `if (m % 4 == 0) return ops.LLMM1(...)` still short-circuits many
cases, but for m==256 and m≥2048 (exactly the shapes the rule matches) the
gfx906 GEMV wins on non-gfx906 archs too. The C++ header even documents the
intent ("Works on any ROCm target; selected at runtime on gfx906") — but the
**selection is not actually restricted to gfx906**. That mismatch between the
comment and the runtime gate is the bug.

This is both a **cross-arch perf regression** and a **numerics-change** risk on
archs the author isn't testing. The op should be gated, minimally, on
`on_gfx906()` (matching how `wvSplitK`, `use_skinny_reduce_counting`, and the
gfx906 MoE work are arch-gated). `dense_gemv_gfx906` not being MFMA-accelerated
on CDNA archs is exactly why it must not spread.

**Fix:** `on_gfx906()` in the `_llmm1_tiny_m` condition (or in the non-skinny
block where the rule is currently arch-blind).

---

## 3. High findings

### H1 — The dense GEMV op has no checked-in numerical correctness test
The op (`csrc/rocm/dense_gemv_gfx906.cu`, `_custom_ops.dense_gemv_gfx906`) has
a fake-registration and the mock LLMM1 test — but **zero ROCm numeric assert**
in the tree. The two new tests in `test_rocm_unquantized_gemm.py`:

- `test_rocm_unquantized_gemm_tiny_m_real_kernel` uses m=1 → **excluded by the
  rule** (`m==256 ∨ m≥2048` does not match m=1), so it exercises the LLMM1-pad
  path, **not `dense_gemv_gfx906`**. The custom GEMV op is never numerically
  validated in CI.
- The mock tests validate the pad/slice, not the kernel.

The v1 kernel shipped with documented correctness bugs ("caught in static
review, before first GPU run was trusted": 64-thread CAS overcount, 256-thread
OOB LDS read, host N%4!=0+K-split). That this required *manual static review*
instead of a unit test is a process gap. For a custom fp32-acc fp16-rounding
kernel, add
`dense_gemv_gfx906(w, x, 2048) ≈ F.linear(x, w)` (atol~1e-2) and a K-split
variant numeric test on ROCm.

### H2 — Correctness evidence is weaker than the plan's own acceptance bar
The plan (`plan-gfx906fa-serving.md` §4) names the FA acceptance criteria as
"perplexity on a fixed prompt set within 2% of the fp16 path (not fluency —
not measurable)". That **was not run**. The FA-side evidence is 128/128 greedy
tokens on ONE prompt, bit-exact vs Triton-FULL — a good smoke test but far
narrower than the stated goal, and Q8-K's ~1e-3 logit divergence can be
invisible to greedy-on-one-prompt while still drifting rank/CLS.

More importantly there is **no greedy/coherence divergence record for the
`dense_gemv` GEMV integration** at all. P3-1 documented its gate-lossit
divergence ("one prompt diverges ~token 11 on the sigmoid gate — accepted");
P3-2(b)'s DEVLOG section (which *also* changes the gate/router arithmetic —
it swaps the exact gate projections) records only throughput A/B, no greedy or
logit-corr check. Given a fused MoE Top-8 router and a dense gate sample softmax
sit right on these M=1 projections, a change to their reduction order deserves
the same divergence record P3-1 gave. Add the prompt-diff (even 1–2 prompts, a
fixed seed) next to the P3-2b integration notes.

### H3 — Unresolved V2 serving regression was accepted without root cause
`DEVLOG-moe-opt.md` "Route B stage 1": V2's paged-block gather is 41 µs/call
isolated but **285 µs/call (7×)** only inside the FULL decode graph; V1 rescued
it and became default. The mechanism is explicitly not isolated ("wave-
scheduling / barrier + low-WG-count interaction **is the leading candidate**").

Picking V1 pragmatically is defensible — but an unexplained 7× degradation of a
kernel *only in the cudagraph replay path* is a potential graph-safety
signature (persistent shared-memory buffers, `__syncthreads` + low WG count,
or a barrier/lane-width assumption interacting with replay). The same class of
trap could bite V1 at other configs (larger Sk, higher B, or a future
low-WG kernel). Recommend a short root-cause pass (rocprof or a reduced
multi-Sk capture harness) before treating this as settled — the current plan
records it as an open note, which is honest but unsatisfying for a 7× effect.

---

## 4. Medium findings

### M1 — The fp16-CAS K-split path in `dense_gemv_gfx906.cu` is unreachable from the default integration and untested
Default selection is K==2048, `kchunk` hardcoded to 2048 → `KSPLIT = K/2048 = 1`
always for the model rule. Every RPT=2/RPT=4 KSPLIT>1 atomic-CAS epilogue
(`atomic_add_pk2_f16`, `pk4`, the multi-wave LDS reduction) is dead from the
model path, reachable only via `VLLM_GFX906_GEMV_RPT` + a direct `kchunk`
call. That is a substantial body of complex, lock-free, correctness-risky code
(v1's bug class) that is neither exercised in CI nor by the default path. Either
add a K-split numeric test (H1 covers it) or document/restrict it to bench-only.

### M2 — V1 gather kernel has a latent gridDim.z limit
`gfx906_fa_gather.cu` V1 launches `dim3 grid(num_seqs, num_kv_heads, Sk)`.
HIP caps gridDim.z at 65535. With Sk_pad = ceil(max_model_len/32)·32, any
`max_model_len ≳ 65K` overflows grid.z (the comments elsewhere already reference
Sk=61K/60K contexts). Safe for the bench model (32K) but a one-line guard or
splitting on grid.z would make it robust to the larger contexts the project
explicitly manages VRAM for elsewhere. Cutover: pick `< 65535` else keep `V2`.

### M3 — Capture-safety relies on an unstated invariant (first capture at capacity)
`Gfx906FAMetadataBuilder._ensure_forward_buffers` still does exact-shape
realloc + `torch.cuda.empty_cache()`, and it is invoked before
`forward_paged`. `build_for_cudagraph_capture` is just `build(0, ...)` (still
says "not supported (MVP)" — stale comment). The FULL-capture-safety argument
rests entirely on *the first FULL capture runs at
profile_seq_lens=max_model_len, so buffers are already at capacity*. The T3
test mutates `sl` in place (fixed Sk=512) so it never exercises a growing-Sk
realloc *during* replay; it validates live re-read, not realloc-under-capture.
If any capture (e.g. a prefill-skewed graph or a future max_model_len bump)
presents a smaller Sk first, the exact-match `empty_cache()` fires inside
capture. The plan documents W5 (capacity `narrow()` handling) as not needed —
fine — but the current exact-match buffer logic should at least get the
"≥ capacity" reuse semantics so the safety doesn't depend on capture order.

### M4 — `rocm.py` backend-registration fallback imports in a hot/cold path
`vllm/platforms/rocm.py` now does a deferred `from vllm.gfx906_fa... import
register; register()` inside `get_valid_backends` when CUSTOM isn't overridden.
Registration already happens at module import (`register()` at the module
bottom), so this is a belt-and-suspenders path that also runs whenever the
plugin was already imported. It's cheap (guarded by `not is_overridden()`), but
calling a side-effecting `register()` twice could double-register if the guard
ever races on order; the else-branch ordering (`if on_gfx906() and not
overridden → register`, then `if on_gfx906() and overridden → append CUSTOM`)
is correct today, just fragile. No action beyond a comment.

---

## 5. Low / nits

- **Stale/inconsistent comments:** `build_for_cudagraph_capture` still says
  "CUDA Graph capture пока не поддерживаем (MVP)" after M2 made decode
  capture-safe. `_gather_kv` docstring still says "Будет заменена custom HIP
  kernel'ом в v2" after that kernel exists. Both mislead future readers.
- **`dense_gemv_gfx906` kernel distinguishes the two dispatch sites poorly:** the
  `use_skinny` block and the fallback block both funnel through the same
  `_llmm1_tiny_m`; the arch-blindness (C1) is the real issue, but a reviewer
  has to trace two callers to confirm the shape rule is consistent. Add one
  comment that this helper is now the single M=1 fp16 GEMM choke point.
- **No unit test for `get_cudagraph_support`** default (UNIFORM_SINGLE_TOKEN_DECODE
  vs env overrides) — trivial to add and would pin the W8 flip.
- **Bench `get_tokenizer`/prompt build** is hardcoded and prompt length relies on
  token re-encode; fine for reproducibility but it's a 5-sample mean on one
  prompt — adequate for a point-in-time record, not a stability claim.
- Overall line-coverage for the new C++ is mostly via serving A/B + manual
  probes; the reusable-strides C++ changes (`gather_paged_kv_q8`,
  `forward_paged_direct`) have a good test (`test_fused_gather_matches_...
  _unbind_cache`), but the legacy fp16 gather test (`test_fused_fp16_gather_...`)
  covers only K=500, one seq, B=1 — no multi-seq/multi-block COW-style layout.

---

## 6. Verdict vs the plan's own acceptance criteria

| plan criterion | met? |
|---|---|
| gather micro-bench gate (< ~80 µs/layer) | ✅ PASSED (21.7 µs, re-scored) |
| M1 stop (< +0.3 ms/step vs Triton-PIECEWISE) | ✅ exceeded (52 range) |
| M2 = FULL-capture-safe, 128/128 probes, T3 | ✅ PASSED |
| prefill CUSTOM ≥ Triton −5% | ⚠️ **not measured** (still listed out of scope) |
| perplexity-within-2% correctness bar (§4/§5) | ❌ replaced by 128/128 single-prompt greedy |
| GEMV P3-2b divergence record | ❌ absent (only throughput A/B) |
| 5-sample serving with σ | ✅ PASSED (57.09, σ≈0.09) |

The functional milestones are met and well documented. The two missing pieces
are numerical-evidence hardening (H1/H2) and the arch-gating bug (C1) that
stands between "gfx906 custom kernel" and "shared ROCm model path".

## 7. Recommended action order

1. **C1** — gate `dense_gemv_gfx906` selection on `on_gfx906()` (one line,
   protects all other ROCm targets).
2. **H1** — add a ROCm numeric assert vs `F.linear` for `dense_gemv_gfx906`
   (single-pass + a K-split variant).
3. **H2** — record a greedy/logit divergence for the P3-2b integration (mirror
   the P3-1 gate divergence note), and at least one perplexity point for the
   FA Q8 path if claimable.
4. **H3/M3** — timebox a root-cause pass on the V2-in-graph 7× regression and
   tighten the capture-time buffer reuse to "≥ capacity" so FULL-safety doesn't
   silently depend on capture order.
5. **M1/M2/M4, nits** — K-split numeric test or bench-only carve-out, a
   gridDim.z guard in V1, comment cleanup.

None of these blocks the 57.09 t/s claim; all of them protect the repo from
accidentally carrying a gfx906-specific assumption onto other ROCm targets or
into unverified numeric territory.