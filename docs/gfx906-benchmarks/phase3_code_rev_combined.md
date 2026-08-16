# Phase 3 code review — combined verdict

Copyright Kevin Read <me@kevin-read.com>

Scope: reconciles `phase3_code_rev_qwen.md` and `phase3_code_rev_ds4.md`,
two independent critical passes over the same diff (`864915074a..HEAD`,
HEAD = `01526dfc69`, "Route B stage 1" — default serving 22.44 → 57.09 t/s).
Every claim in both source reviews was re-checked directly against the
current tree (source, tests, docs) rather than taken on faith. Findings are
numbered by their origin review (`qwen F#`, `ds4 C#/H#/M#`); where both
reviews independently found the same thing, that is called out.

**Verdict on the reviews themselves: both are accurate.** Every claim
checked below is either CONFIRMED as stated or CONFIRMED with a minor
precision correction (a slightly imprecise word, not a wrong fact). Nothing
was rejected. The two reviews are highly complementary — qwen goes deeper on
the buffer-lifecycle hazard and env/hygiene surface, ds4 goes deeper on the
arch-gating bug and the unexplained V2 regression — and neither contradicts
the other anywhere they overlap (both flag F2/H1, F3/H2, and the V2
regression independently and reach the same conclusions).

**Verdict on the underlying code: unchanged from both source reviews.** The
57.09 t/s number is credible and not in question. The gaps are real,
concentrated in exactly the four places qwen already named: a structural
use-after-free in the shipped default path, correctness evidence thinner
than the plan's own bar, a critical cross-arch dispatch bug, and repo/record
hygiene. Priority order below merges both reviews' recommendations.

---

## 1. Confirmed findings, by severity

### CRITICAL

**ds4 C1 — `dense_gemv_gfx906` has no `on_gfx906()` gate; it silently routes
onto every ROCm arch.** CONFIRMED exactly as described.
`vllm/model_executor/layers/utils.py`:
- `use_skinny` (line 402-408) is true whenever
  `on_gfx9() or on_gfx1x() or on_gfx906()` — i.e. CDNA2/3 and RDNA targets
  qualify, not just gfx906 (Vega20/GCN5).
- Inside `use_skinny`, the `elif` branch at line 421-423 calls
  `_llmm1_tiny_m(weight, x_view)` with **no arch check at all**.
- The non-skinny fallback block at line 425-434 (reached when
  `VLLM_ROCM_USE_SKINNY_GEMM` is off, or `use_skinny`'s preconditions
  fail) calls the same `_llmm1_tiny_m` at line 433, **also with no arch
  condition** — confirmed "arch-blind" exactly as ds4 states.
- `_llmm1_tiny_m` itself (`utils.py:221-244`) gates only on tensor dtype,
  contiguity, and shape (`weight.shape[1] == 2048 and (m == 256 or
  m >= 2048)`) — never on architecture. Any ROCm target hitting that shape
  rule gets routed onto a kernel "only ever measured on gfx906 ... untuned
  (likely slower) on MFMA-class targets," per ds4's own framing, which
  matches the kernel's doc comment at `utils.py:225-229`.

This is a one-line fix (`and on_gfx906()` in the condition, or wrapping both
call sites) and should land before anything else in this list — it is the
only finding here that actively mis-routes model math on hardware other
than the one this phase targets.

### HIGH

**qwen F1 — the captured decode graph's `q_pad` buffer is freed by the
first real prefill; the real buffer lifecycle is structurally untested.**
CONFIRMED, including every step of the causal chain:

1. `Gfx906FAImpl.forward` (`gfx906_fa_backend.py:470`) returns early at
   line 486-488 (`if attn_metadata is None: return output.fill_(0)`) — so
   during `profile_run` (no forced attention, mode NONE), `q_pad` is never
   touched.
2. `_ensure_forward_buffers` (line 336-375) is only reached once real
   attention metadata exists. The engine call order, confirmed in
   `vllm/v1/worker/gpu_worker.py`, is `profile_run()` (line 479/506) →
   `profile_cudagraph_memory()` (line 519) → `capture_model()` (line 719) —
   matching the claimed sequencing: profiling passes first, then capture.
3. The grow branch (line 361-375) is a **free-then-realloc, not
   grow-in-place**: `self._q_pad_buf = None; del cur;
   torch.cuda.empty_cache(); self._q_pad_buf = torch.empty(new_shape, ...)`.
   A brand-new tensor object (new VA) is allocated; nothing narrows or
   reuses the old storage. There is no
   `assert not torch.cuda.is_current_stream_capturing()` or any other guard
   in this function (confirmed absent by direct grep).
4. Consequence: if 4 decode graphs (B=8/4/2/1) are captured referencing one
   small `q_pad` buffer, and a subsequent **eager** prefill with a larger
   `max_seqlen_q` triggers this grow branch, the VA the captured graphs
   reference is freed back to the allocator while those graphs still exist.
   Replay then writes into freed memory. This is empirically tolerated (the
   driver evidently keeps/re-maps the VA) but is not a structural
   invariant — it depends on incidental allocator behavior.

**Test-bypass claim CONFIRMED by direct read of
`tests/kernels/attention/test_gfx906_fa.py:84-176`
(`test_cudagraph_capture_replay_legacy_decode_path`).** Line 117 manually
constructs `q_pad = torch.zeros(2, HQ, 2, D, dtype=torch.float32,
device=dev)` and passes it straight into `forward_paged(..., q_pad_buf=
q_pad)` (line 123) — the exact comment qwen quoted appears verbatim at
lines 115-116: "Shared q_pad buffer across both graphs, as the backend's
lazy-grown class buffer would be at capture capacity." This test never
calls `_ensure_forward_buffers` and cannot exercise the grow/free path at
all. What it *does* validate (Sk_pad growth mid-replay, B=1↔B=2 capture
order, live `seq_lens` re-read) is real and valuable — it is simply a
different test than the one this code path needs.

This is the single highest-priority code fix in either review: it is a
latent use-after-free on the path every default request now takes, made
safe today only by the specific profiling/capture order vLLM happens to
use, not by anything the backend enforces.

**qwen F2 / ds4 H1 — `dense_gemv_gfx906` has no pytest numeric coverage.**
CONFIRMED by direct read of `tests/model_executor/layers/
test_rocm_unquantized_gemm.py`:
- `test_rocm_unquantized_gemm_tiny_m_llmm1_padded` (line 180, parametrized
  m∈{1,2,3}) and `test_rocm_unquantized_gemm_tiny_m_real_kernel` (line 209,
  m=1) both use `weight.shape == (m, 2048)` with m never equal to 256 and
  never ≥ 2048 — the exact dispatch rule in `_llmm1_tiny_m`
  (`m == 256 or m >= 2048`) is never satisfied. Both tests exercise the
  padded-LLMM1 fallback, not `dense_gemv_gfx906`.
- The only in-tree numeric gate for the kernel is the `assert md < atol`
  in `benchmarks/kernels/gfx906/bench_dense_gemv_gfx906.py` — a benchmark
  script (loose atol 0.25), not run by CI, and explicitly out of scope per
  this repo's own AGENTS.md ("prove correctness in existing pytest
  suites," "no one-off kernel benchmarks in tests/"). qwen's framing
  ("shipped with three correctness bugs that static review caught before
  first trust — precisely the class a small unit test catches cheaply") is
  accurate to the devlog's own account of the v1 kernel's bug history.

**qwen F3 / ds4 H2 — default-path correctness evidence is narrower than
the plan's own acceptance bar.** CONFIRMED on every sub-claim:
- The stated acceptance criterion — "perplexity on a fixed prompt set
  within 2% of the fp16 path" — exists at
  `plan-gfx906fa-serving.md:427` and nowhere does any perplexity number
  appear in the tree (checked `DEVLOG-moe-opt.md` and both plan files).
  Not run, as claimed.
- `DEVLOG-moe-opt.md:884` records verbatim: "diverge from fp16 runs after
  ~10-25 tokens (both fluent)" — confirming greedy divergence on
  non-degenerate prompts is a known, recorded phenomenon, which is exactly
  why a 128/128 bit-exact match on one degenerate (tight-repetition-loop)
  prompt is weak evidence for the actual Q8-quantization numerics change.
- `DEVLOG-moe-opt.md:1082` records verbatim: "prefix-cache COW and
  multi-batch decode are not exercised by this probe (P3-3a items (ii)
  continue)" — confirming the engine-level multi-batch gap is
  self-acknowledged and, per current HEAD, still open.
- Both positive claims qwen adds also check out: `gpu_model_runner.py:2241`
  is exactly `self.seq_lens[num_reqs:].fill_(0)` (line number unchanged
  from the review), and block-table padding via `NULL_BLOCK_ID` is
  confirmed at `gpu_model_runner.py:2398`. The V1 gather kernel's
  `tok_pos >= seq_len` early return (`gfx906_fa_gather.cu` ~line 108-114)
  is confirmed to precede the block-table read (~line 118) — so the
  padded-row-is-inert argument for LEGACY=1's COW safety is sound.

### MEDIUM

**qwen F4 — `VLLM_GFX906_GEMV_RPT=0` crashes the host; invalid values
misroute silently.** CONFIRMED exactly.
`csrc/rocm/dense_gemv_gfx906.cu:245-256`: `rpt = atoi(e)`, then
`if (rpt < 0) { ...default rule... }` — the `< 0` check does not catch
`rpt == 0`. `TORCH_CHECK(N % rpt == 0, ...)` with `rpt=0` is `N % 0`, UB on
host (SIGFPE class). For `rpt ∈ {3,5,6,7,...}` that happens to divide `N`,
the check passes but `LAUNCH_BY_RPT` (line 274-282) only special-cases
`rpt == 4` and `rpt == 2`; every other value — including whatever
was actually requested — falls through to the final `else LAUNCH(1,
KCVAL)`. The kernel still computes a correct answer at RPT=1 semantics, but
silently ignores the env value's stated intent.

**qwen F5 — the new default V1 gather kernel lacks the block-table bounds
check V2 has.** CONFIRMED exactly. `csrc/gfx906_fa/gfx906_fa_gather.cu`:
V1 (line 118) reads `block_table[seq_idx * max_blocks_per_seq +
block_tab_idx]` unguarded (only the `tok_pos >= seq_len` early-return above
it protects against out-of-range `tok_pos`). V2 (line 233-235) has
`s_phys_block = (block_tab_idx < max_blocks_per_seq) ?
block_table[seq_idx * max_blocks_per_seq + block_tab_idx] : -1`. The
current usage contract makes this safe, but the safety is enforced three
layers away, matching qwen's assessment.

**qwen F6 — `torch.cuda.empty_cache()` on the forward path.** CONFIRMED.
Both `_ensure_forward_buffers` (`gfx906_fa_backend.py:374`, the F1
mechanism) and `_ensure_gather_buffers` (`gfx906_fa_backend.py:414`, the
LEGACY=0 class-buffer path) call `torch.cuda.empty_cache()` inside a
free/realloc branch with **no `is_current_stream_capturing()` guard**
anywhere in the file (confirmed absent by grep). `empty_cache()` is illegal
during capture and a full allocator sync in eager; it is also the specific
mechanism that makes F1 possible.

**qwen F7 — live env footguns on the default path.** CONFIRMED on all
three sub-claims:
- RC1 ("`value_cache blocks mismatch`" crash, LEGACY=0) is documented at
  `DEVLOG-moe-opt.md:864` and `plan-gfx906fa-serving.md:94`; RC2 (COW
  bypass) at `plan-gfx906fa-serving.md:114`. The plan's own RC2-invariant
  note (`plan-gfx906fa-serving.md:127-128`) calls for "a loud log/assert
  when a connector is enabled with LEGACY=0" — grepping the actual
  `vllm/gfx906_fa/*.py` source finds **no such log or assert**; the
  guard was specified in planning but never implemented.
- The three debug hooks are confirmed capture-unsafe and confirmed
  undocumented as such: `_DOUBLE_CHECK`
  (`gfx906_fa_paged.py:383-390`) calls `.item()` (line 388) and
  `.tolist()`/`print` (line 390); `_DUMP_DIR`
  (line 495-497) calls `torch.save`; `_DBG` (multiple sites, e.g. line
  331-332, 508) calls `torch.cuda.synchronize()`. The module docstring
  (`gfx906_fa_paged.py:8-17`) documents the fast/legacy K-quant paths but
  says nothing about which env knobs are eager-only.
- Env read-time inconsistency confirmed by direct grep:
  `_DBG`/`_FUSED`/`_ZERO_KTAIL`/`_NO_BUF_REUSE`/`_DOUBLE_CHECK`/
  `_TORCH_GATHER`/`_DUMP_DIR` are all module-level reads in
  `gfx906_fa_paged.py` (lines 30-74); `GFX906_FA_LEGACY` is read
  per-instance in `Gfx906FAImpl.__init__` (`gfx906_fa_backend.py:303`);
  `GFX906_FA_CG` is read inside a method body
  (`gfx906_fa_backend.py:90`), i.e. effectively per-call. No central
  "environment variables" table exists anywhere in the module. The
  characterization is fair.

### LOW

**qwen F8 / ds4 H3 — the V2 7× in-graph degradation is an unexplained,
accepted regression.** CONFIRMED. `DEVLOG-moe-opt.md:1232-1254` records:
V2 isolated at 41 µs/call vs **285 µs/call (p10-p90 282-287) only inside
the FULL decode graph** — a real, uniform, 7× in-context degradation. The
"leading candidate" mechanism (line 1253-1254: "wave-scheduling /
barrier + low-WG-count interaction with the graph context") is explicitly
hedged, not confirmed. The cited probe artifact (line 1239-1241: a 256 MB
`zero_()` inside the timed window inflating an unrelated measurement to
"masquerade" as 320 µs of gather cost) is also confirmed verbatim in the
devlog. Both W6 follow-up items (D=64/128 early-out verification;
`block_size==16` assert) were searched for and not found done anywhere in
plan or devlog.

**ds4 M1 — the K-split path is unreachable from the default integration.**
CONFIRMED. The model dispatch (`_llmm1_tiny_m`, `utils.py:240`) always
calls `ops.dense_gemv_gfx906(weight, x_view, 2048)` — `kchunk` is
hardcoded to 2048, and the dispatch rule requires `weight.shape[1] ==
2048` (i.e. `K == 2048`), so `ksplit = K/kchunk` is always exactly 1 on
the model path. The RPT=2/4 K-split atomic-CAS epilogue is real,
complex, lock-free code that is genuinely dead from serving — reachable
only via `VLLM_GFX906_GEMV_RPT` + a direct kernel call with a smaller
`kchunk`.

**ds4 M2 — V1 gather kernel has a latent `gridDim.z` limit.** CONFIRMED.
`gfx906_fa_gather.cu:409`: `dim3 grid(num_seqs, num_kv_heads, Sk);` — `Sk`
(padded sequence length) is used directly as `gridDim.z` with no cap
check anywhere nearby. HIP's `gridDim.z` limit of 65535 would be hit at
`max_model_len` in the ~65-70K token range (`Sk_pad` rounds up to a
multiple of 32). Safe today at the bench model's 32K context; a real
limit for larger contexts this project manages VRAM for elsewhere.

**ds4 M4 — `rocm.py` has a belt-and-suspenders registration path.**
CONFIRMED, with one precision correction: `vllm/gfx906_fa/
gfx906_fa_backend.py:573` does call `register()` unconditionally at
module import time, and `vllm/platforms/rocm.py:503-517` (inside
`_get_backend_priorities`, called from `get_valid_backends`) has a second,
guarded (`not is_overridden()`) `register()` call. Structurally exactly as
ds4 describes. **Correction**: ds4 calls this a "hot/cold path" — direct
grep of all callers shows `get_valid_backends` has exactly two call
sites in the whole tree (`cuda.py:427`, `rocm.py:700`), both inside
backend-selection setup, not a per-request or per-step path. "Hot path"
overstates the exposure; it is a startup/config-time path invoked at most
a handful of times per process, so the double-registration risk is real
but lower-frequency than "hot" implies. No action needed beyond ds4's own
"no action beyond a comment" recommendation.

**qwen F9 / ds4 nits — stale comments and small code smells.** CONFIRMED,
sampled directly:
- `csrc/rocm/ops.h:56` documents "kchunk 512|2048"; the actual kernel
  (`dense_gemv_gfx906.cu:236-238`) accepts and correctly handles 4096 too
  (its own in-file comment at line 218 correctly lists all three) — the
  drift is specifically in the binding header, confirmed.
- `gfx906_fa_backend.py:110` still reads "CUDA Graph capture пока не
  поддерживаем (MVP)" — stale post-M2, confirmed present verbatim.
- `gfx906_fa_paged.py:117` still reads "Будет заменена custom HIP
  kernel'ом в v2" after that kernel exists — confirmed present verbatim.
- `gathered_sk` in `gfx906_fa_paged.py` is assigned at lines 382, 402,
  414, 423 and never read anywhere in the file — confirmed dead.
- `test_fused_fp16_gather_matches_torch_gather`
  (`test_gfx906_fa.py:179-204`) uses `L=500`, one seq, B=1 (`bt =
  torch.arange(n_blocks, ...).view(1, -1)`) — confirmed no multi-seq/B>1
  coverage for this specific test, as ds4's nits section states.

**qwen F10 — records and repo hygiene drift.** CONFIRMED, all sub-claims:
- `git status --short` shows exactly the untracked set claimed:
  `.rocprofv3/` (measured 462 MB, close to the review's "≈479 MB" —
  difference is measurement-time noise, not a wrong claim),
  `gpucore.1.gpu` / `gpucore.10.gpu` (confirmed root-owned,
  `-rw-------`, 207 MB each, genuinely unreadable to this user —
  `file` reports "no read permission"), `_pp_bench.py`,
  `run_bench_gfx906.sh`. None of these patterns appear in `.gitignore`
  (grepped, zero matches).
- All four named probe scripts (`_p31_ab.py`, `probe_custom_fa.py`,
  `bench_ab2.py`, `test_backend_vs_legacy.py`) are absent from the tree —
  confirmed by both `find` and `git ls-files`, zero hits for any of them.
  The sub-plan's own W4 item ("check probes into tests/ or tools/") is
  confirmed unmet.
- Doc staleness confirmed precisely, with one location correction: the
  stale "44.09 t/s / 1.59× gap" table qwen describes is in
  `docs/gfx906-benchmarks/plan-decode-phase3.md` (the v11 file, parent of
  the sub-plan), not the sub-plan itself. Its status header (line 5-20)
  correctly states 57.09 t/s / ~1.23× gap, but the §1 "Where we are" table
  further down (line 220-230) still shows the old 44.09 t/s / **1.59×**
  numbers, unedited. Separately, in `plan-gfx906fa-serving.md` (the
  sub-plan), the §0 table at **line 66** still labels **52.90** t/s "new
  best = default config," while the same file's own changelog at line 202
  records 57.09 as the new best — confirming the second staleness claim
  in a different location than the first. Both are real, both are minor
  reconciliation misses, not fabricated by either review.

### INFO

**qwen F11 — default-path resource characteristics.** Not independently
re-derived (this is an informational note about VRAM scaling, not a
falsifiable code claim); the underlying formula and O(context) HBM-traffic
observation about LEGACY=1 re-gathering/re-quantizing full KV history per
step are consistent with the LEGACY=1 code path as read during F1/F3
verification (no separate K representation is cached — confirmed while
checking the COW-safety claim in F3). No correction needed.

---

## 2. What to fix, in order (merged from both reviews)

1. **ds4 C1** — gate `dense_gemv_gfx906` dispatch on `on_gfx906()`. One
   line, protects every non-gfx906 ROCm target from an untuned,
   numerically-different kernel path today. Highest severity because it
   is the only finding that affects hardware beyond this phase's target.
2. **qwen F1** — make `q_pad` (and the LEGACY=0 class gather buffers)
   capture-order independent: capacity pre-size or never-shrink +
   `narrow()` views, drop `empty_cache()` from the forward path (F6), and
   add a test that drives the real `Gfx906FAImpl` (not hand-fed buffers)
   through capture → real prefill → decode replay. This is the gap
   between "the bench flow is safe" and "the default path is
   structurally safe."
3. **qwen F2 / ds4 H1 + qwen F4 + qwen F5** — the remaining one-liners:
   a ROCm numeric pytest for `dense_gemv_gfx906` vs `F.linear`; clamp
   `VLLM_GFX906_GEMV_RPT` to {1,2,4} and reject 0 explicitly; add the V1
   gather kernel's block-table bounds guard to match V2.
4. **qwen F3 / ds4 H2** — one perplexity point (fixed prompt set, CUSTOM
   vs Triton) and one two-request multi-batch greedy probe (ideally with
   a partial-block prefix hit); check probes into `tests/` or `tools/`
   per W4 while at it (folds in qwen F10's probe-hygiene gap).
5. **qwen F6/F7 + ds4 H3/M3** — `assert not
   torch.cuda.is_current_stream_capturing()` in both realloc paths; a
   loud log/assert for LEGACY=0 + (cudagraphs or prefix caching), per the
   plan's own already-written RC2 note; one comment marking the three
   debug env hooks eager-only; a short root-cause pass on the V2 7×
   in-graph regression (rocprof or a reduced multi-Sk capture harness)
   before treating the "leading candidate" theory as settled.
6. **ds4 M1/M2 + qwen F9/F10 + ds4 nits** — K-split numeric test or
   explicit bench-only carve-out; a `gridDim.z` guard (or V1→V2 cutover)
   for very large `max_model_len`; comment/doc cleanup (stale MVP/Russian
   comments, `ops.h` kchunk doc, dead `gathered_sk`); `.gitignore` the
   profiler/gpucore output at repo root; fix the two stale-number tables
   in `plan-decode-phase3.md` §1 and `plan-gfx906fa-serving.md` §0.

None of this changes the headline number. The 57.09 t/s figure is real and
well-measured by both reviews' independent assessment. The work above is
the difference between "the bench flow is verified" and "the default
serving path is structurally safe, cross-arch-clean, and durably tested" —
item 1 protects other hardware, item 2 protects this hardware's own
default path, and the rest closes the gap to the plan's own stated bar.
