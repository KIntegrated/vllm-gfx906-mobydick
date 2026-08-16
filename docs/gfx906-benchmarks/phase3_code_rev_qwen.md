# Phase 3 code review — qwen — critical pass

Copyright Kevin Read <me@kevin-read.com>

Scope: commits `864915074a..HEAD` (23 commits; `01526dfc69` at review time) —
P3-0 diagnostics, P3-1 tiny-m padded LLMM1, P3-2(b) custom W16A16 GEMV,
P3-3/P3-3a gfx906 FA serving work (stride fixes, fused fp16 gather + V1
default, CGSupport flip, registration fallback), tests, bench harness,
plan/devlog updates. Static review of tree state; no test runs performed
(per reviewer instruction — results were already verified). Companion to
`phase3_code_rev_ds4.md`; where we overlap I say so rather than repeat.

Verdict: the process quality is the standout — reconciled budgets, gated
go/no-gos, negative results recorded, mode-matched A/B matrix with a
reproduced archive baseline, 5-sample σ, and honest "open note" items. The
57.09 t/s claim is credible. My findings concentrate in four places:

1. a **structural use-after-free hazard in the shipped default path** that
   the T3 test structurally cannot see (F1);
2. **correctness/test evidence thinner than the plan's own acceptance bar**
   (F2, F3);
3. a handful of **one-line hardening gaps** in the new C++ (F4, F5);
4. **records/hygiene drift** (F10) that will cost future sessions time.

None of F1–F5 invalidates the 57.09 t/s number; F1 is the only one I would
treat as blocking before this default path is trusted in anything other than
the bench's single-request flow.

---

## 1. Findings

### F1 (HIGH) — the captured decode graph's `q_pad` buffer is freed by the
first real prefill; the real buffer lifecycle is untested

`vllm/gfx906_fa/gfx906_fa_backend.py:336–375` (`_ensure_forward_buffers`),
M2 default path (LEGACY=1, FULL decode capture).

Tracing the actual init/serving sequence (not the T3 test):

1. `profile_run` (`_dummy_run(max_num_tokens, is_profile=True)`) runs with
   `force_attention=False` and mode NONE, so `_build_attention_metadata` is
   never called and `Gfx906FAImpl.forward` returns at
   `if attn_metadata is None: return output.fill_(0)` — **q_pad is not
   sized during profile_run**.
2. `profile_cudagraph_memory` warmups run uniform-decode dummies
   (`max_query_len=1` → `Sq_pad=2`) with `force_attention=True` → q_pad
   first allocated as `(8, H, 2, D)` fp32 (largest capture B).
3. `capture_model` captures B=8/4/2/1 (descending); all four decode graphs
   reference that one `(8, H, 2, D)` buffer (grow-only logic + descending
   order — this part is fine).
4. **First real request**: prefill 2048 tokens → `max_query_len=2048` →
   `Sq_pad=2048` → the grow branch fires: `self._q_pad_buf = None; del cur;
   torch.cuda.empty_cache(); torch.empty((8, H, 2048, D))` (line ~372–375).
   The `(8, H, 2, D)` buffer captured into all four decode graphs is freed
   back to the driver.
5. Every subsequent decode replay writes `q_padded.zero_()` + the query
   into that freed VA (32 KiB for the B=1 graph, which is the only one
   replayed by the single-request bench).

Empirically this is tolerated on this ROCm/driver (128/128 probe bit-exact,
5-sample σ≈0.09 — the driver evidently keeps or harmlessly re-maps the VA),
but it is a **latent use-after-free, not an invariant**:

- The sub-plan's RC4-additional explicitly identified this buffer class
  ("class-level buffers dangle across capture sizes") and W5 specified the
  fix — "Ensure all captured graphs share one buffer object that is never
  shrunk … Same treatment for q_pad_buf." M2 shipped **without** that
  treatment for `q_pad`, and the devlog's "No W5 buffer surgery needed for
  LEGACY=1" is only true for the bench flow, by accident of vLLM-internal
  ordering (uniform-decode captures at Sq_pad=2; prefill chunks ≤
  max_num_batched_tokens).
- `test_cudagraph_capture_replay_legacy_decode_path` **cannot** catch this:
  it hands `forward_paged` a manually pre-allocated, manually shared
  `q_pad` (`torch.zeros(2, HQ, 2, D)`, comment: "as the backend's
  lazy-grown class buffer would be at capture capacity") — i.e. it bypasses
  `_ensure_forward_buffers` entirely. What it validates (Sk_pad transition,
  multi-size capture+replay, live `seq_lens` re-read) is real and valuable;
  what it does not validate is the one allocation the backend actually
  manages.
- The single-request bench cannot catch it either: only the B=1 graph
  replays, writing only 32 KiB, and the freed 256 KiB block's reuse pattern
  happens to be harmless here.

This is not hypothetical: any config change that (a) makes the profile
dummies non-uniform or decode-only metadata, (b) changes capture order, or
(c) puts a live tensor into that VA, converts "works" into "silent decode
corruption." Fix options, cheapest first:

1. Pre-size q_pad at capacity before capture (`Sq_cap` from
   `max_num_batched_tokens`, `B_cap` from `max_cudagraph_capture_size`), or
   reuse "≥ capacity + `narrow()` views" (W5 as written), or at minimum
   move the grow to "grow only, never free a buffer a captured graph
   references" and drop `empty_cache()` from the forward path (see F6).
2. Add a test that drives the **real** `Gfx906FAImpl` (not `forward_paged`
   with hand-fed buffers) through: multi-size FULL capture → prefill with
   `max_query_len > 2` → decode replay → compare against eager. That test
   would have failed on the current code path the moment the freed VA was
   reused (or at least would pin the invariant so the ordering dependency is
   explicit in the suite rather than in a devlog paragraph).

### F2 (HIGH) — `dense_gemv_gfx906` has no pytest numeric coverage

(Agrees with and extends ds4 H1.) The two new tests in
`tests/model_executor/layers/test_rocm_unquantized_gemm.py` never execute the
op:

- `test_rocm_unquantized_gemm_tiny_m_llmm1_padded` monkeypatches
  `ops.LLMM1` and uses m=1/2/3 — none satisfy the routing rule
  (`m == 256 or m >= 2048`), so `_llmm1_tiny_m` takes the pad-LLMM1 branch;
  the GEMV op is never called.
- `test_rocm_unquantized_gemm_tiny_m_real_kernel` uses m=1, K=2048 — again
  excluded by the rule; it tests the padded LLMM1 path.

The only in-tree correctness gate for the op is the `assert md < atol` block
inside `benchmarks/kernels/gfx906/bench_dense_gemv_gfx906.py` — a benchmark
script, not run by CI, with a loose atol (0.25). Per this repo's own
AGENTS.md ("prove correctness in existing pytest suites"), that is not a
substitute. A v1 of this kernel shipped with three correctness bugs that
static review caught before first trust — which is precisely the class a
small unit test catches cheaply. Add one pytest (ROCm-gated), e.g.
parametrize `(N, K, kchunk, rpt-env)` over `{(256, 2048, 2048, auto),
(2048, 2048, 512, 4), (2048, 2048, 2048, 4), (2048, 4096, 2048, 4)}` vs
`F.linear` in fp32. That covers: the routed shapes, the K-split path
(ksplit=4), and the RPT=4 64-bit-CAS epilogue. (Note the bench does sweep
`kc=512/rpt=4`, so the 64-bit CAS has at least been *run* — but a bench
assert is the wrong durable gate, and nothing pins it in CI.)

### F3 (HIGH) — default-path correctness evidence is narrower than the
plan's own acceptance bar

The sub-plan (`plan-gfx906fa-serving.md` §4) names the acceptance as
"perplexity on a fixed prompt set within 2% of the fp16 path (not
'fluency' — not measurable)". It was not run — there is no perplexity
number for the Q8-FA path anywhere in the devlog (checked). What exists:

- 128/128 greedy tokens **bit-exact vs Triton-FULL on one degenerate
  prompt** (tight repetition loop). That is an excellent *fingerprint* for
  the stride-bug corruption class (garbage V diverges immediately), but it
  is weak evidence for the actual numerics change: Q8 K quantization moves
  logits ~1e-3, and the devlog itself records that greedy runs "diverge
  from fp16 runs after ~10-25 tokens (both fluent)" on non-degenerate
  prompts. Greedy-argmax on a one-token-margin prompt can be bit-exact
  while rank/CLS drift — the plan anticipated exactly this and set the
  perplexity gate for it.
- **Engine-level multi-batch decode was never probed.** The devlog (line
  ~1082) acknowledges "prefix-cache COW and multi-batch decode are not
  exercised by this probe (P3-3a items (ii) continue)" — and as of HEAD
  they still are not. T3 covers B=1→B=2 at the kernel level only. The
  B≥2 replay graphs (block-table condense, multi-request KV allocation,
  padded-row staging) are now part of the *default* serving path, yet no
  two-request probe exists.
- (GEMV-side divergence record — ds4 H2 — also absent; I concur.)

Two positives worth pinning in the record while it's fresh:

- **LEGACY=1 inherits prefix-cache COW safety by construction.** It holds no
  separate K representation: K lives only in the fp16 cache, which
  `copy_kv_cache_blocks_inplace` mirrors correctly, and quantization
  re-reads the fp16 cache every step. That is the strongest argument for
  shipping LEGACY=1 as the default (RC2 simply cannot fire) — the devlog
  implies it but never states it as the design rationale.
- The padded-row invariants it depends on are real and verified in the
  runner: `seq_lens[num_reqs:].fill_(0)` (gpu_model_runner.py:2241) and
  block-table rows filled with `NULL_BLOCK_ID`; the V1 gather's
  `tok_pos >= seq_len` early-return happens *before* any block-table read,
  so padded rows are inert. Good.

Recommended: one perplexity point (fixed prompt set, CUSTOM vs Triton,
same config) + one two-request multi-batch greedy probe (ideally with a
partial-block prefix hit, which doubles as the long-deferred T4). Both are
cheap relative to what's now shipping as the default.

### F4 (MEDIUM) — `VLLM_GFX906_GEMV_RPT=0` crashes the host (div-by-zero);
invalid values misroute silently

`csrc/rocm/dense_gemv_gfx906.cu:246–256`. `rpt = atoi(env)`; the `rpt < 0`
fallback-to-default does not catch `0`: with `VLLM_GFX906_GEMV_RPT=0`,
`TORCH_CHECK(N % rpt == 0)` evaluates `N % 0` — undefined behavior / SIGFPE
on the host. With `rpt ∈ {3,5,6,7,...}` and `N % rpt == 0`, the check
passes but `LAUNCH_BY_RPT` falls through to `LAUNCH(1, …)` — the kernel
still computes the right answer (grid.x = N, one row per block), so the
contract is silently different from what the env value claims. This is a
bench-sweep knob, but it's exported as a public-looking env with no
validation. Clamp: anything not in {1,2,4} → default rule (and reject 0
explicitly).

### F5 (MEDIUM) — the new default V1 gather kernel lacks the block-table
bounds check that V2 has

`csrc/gfx906_fa/gfx906_fa_gather.cu:120` vs `:233`. V2:
`s_phys_block = (block_tab_idx < max_blocks_per_seq) ? block_table[…] : -1`.
V1 (now the default): unguarded
`block_table[seq_idx * max_blocks_per_seq + block_tab_idx]` with
`block_tab_idx = tok_pos / block_size` up to `Sk/bs − 1`.

Under the current usage contract this is safe — a block-table read only
happens for `tok_pos < seq_len`, and `seq_len ≤ max_model_len` ≤ the
block table's covered token count — but the safety lives in an invariant
three layers away (kernel ← forward_paged ← runner metadata), and a lying
`seq_lens`, or a future caller passing `Sk` beyond block-table coverage
(e.g. the `max_model_len` not-multiple-of-32 corner where
`Sk_pad > mml`), becomes an out-of-bounds read of up to a row of ints. The
guard is one comparison per workgroup (lane 0 only) and costs nothing; add
it to V1 to match V2.

### F6 (MEDIUM) — `torch.cuda.empty_cache()` on the forward path

`gfx906_fa_backend.py:374` (q_pad grow) and `:414` (class gather buffers,
LEGACY=0). `empty_cache()` is illegal inside CUDA graph capture and is an
expensive allocator sync in eager. In the current default flow the q_pad
grow happens during eager prefill (legal but a full sync on the first
prefill) — and it is the mechanism of F1. The LEGACY=0 class-buffer path
would hit it on every 32-token `Sk` boundary in PIECEWISE serving (the W3
hysteresis item, still deferred). Minimum: `assert
not torch.cuda.is_current_stream_capturing()` in both realloc paths (the
sub-plan W5 already specifies this); proper fix: capacity-based never-shrink
buffers per F1.

### F7 (MEDIUM) — live env footguns on the default path

- **`GFX906_FA_LEGACY=0` still crashes in serving and corrupts with prefix
  caching** (RC1: `_ensure_q8_sidebuffer` binds once to the first/profile
  cache shape — `value_cache blocks mismatch` on the real-cache capture
  warmup; RC2: COW copies bypass the Q8 side buffer). Both are documented
  and demoted — fine — but the env is still live, unguarded, and there is
  no loud warning when it's set in a serving config. The sub-plan's own
  RC2-invariant note says "add a loud log/assert when a connector is
  enabled with LEGACY=0" — neither guard shipped. At minimum, log once at
  init when LEGACY=0 + (cudagraphs or prefix caching) is detected.
- **Capture-unsafe debug hooks are undocumented as eager-only**:
  `GFX906_FA_DOUBLE_CHECK=1` (`.item()` + `print` in forward),
  `GFX906_FA_DUMP` (`torch.save`), `GFX906_FA_FWD_DEBUG=1` (`.tolist()`,
  `synchronize()`). Any of these under cudagraph capture aborts the capture
  (the devlog itself records the `.tolist()` detour). One comment line
  "eager diagnostics only — will abort graph capture" on the block at the
  top of `gfx906_fa_paged.py` would do.
- **Env read-time is inconsistent**: `GFX906_FA_TORCH_GATHER`, `_FUSED`,
  `_ZERO_KTAIL`, `_NO_BUF_REUSE`, `_DOUBLE_CHECK`, `_DUMP_DIR` are read at
  module import; `GFX906_FA_GATHER_V` at first kernel launch;
  `GFX906_FA_CG` per call; `GFX906_FA_LEGACY` per impl instance. None of
  this is documented. For a backend with this many knobs, a one-paragraph
  "environment variables" table (value, effect, read time, capture-safe?)
  in the module docstring would save the next person the rediscovery.

### F8 (LOW) — the V2 7× in-graph degradation is an unresolved anomaly, and
two related W6 items are unrecorded

(Concurs with ds4 H3; adding methodological context.) The same investigation
session found a probe artifact where a 256 MB `zero_()` inside the timed
window "masqueraded as L2-miss gather cost" (320 µs). That is the kind of
error that inflates a kernel's apparent in-context cost; the 285 µs figure
is from a kernel trace (more trustworthy) but deserves one re-check with a
minimal capture harness before the "barrier + low-WG-count in graph
context" theory is treated as settled. Separately:

- W6's "Verify the early-out covers D=64/128 variants" — no record of it
  being done (devlog or plan).
- W6/W8's "Pin the `block_size == 16` assumption explicitly … add an assert
  so a future block-32 model fails loudly" — not done. A non-16 block size
  today *silently* routes to the gather path (`key_cache_q8.shape[1] == 16`
  check in `forward_paged`), which works but defeats the feature with no
  log line.

### F9 (LOW) — stale comments / small code smells in the new C++

- `csrc/rocm/ops.h:56` documents `kchunk 512|2048`; the implementation
  accepts 4096 too.
- The "measured rule" exists in two places: the Python gate
  (`weight.shape[1] == 2048 and (m == 256 or m >= 2048)` in
  `_llmm1_tiny_m`) and the C++ default-RPT heuristic (`kchunk == 2048 &&
  (N == 256 || N >= 2048) → RPT=2`). They agree today and only the Python
  side is on the model path, but they will drift; one comment pointing at
  the other would help. (The model-specific constants in generic
  `vllm/model_executor/layers/utils.py` are fine as a conservative
  no-op-for-other-models rule — the arch-gating gap is ds4's C1 and I
  concur it's the top fix.)
- `gather_paged_kv_fp16`'s `use_or_alloc(buf, dim3)` parameter shadows the
  CUDA `dim3` type (cosmetic, but confusing in a .cpp that launches
  kernels).
- `gathered_sk` in `forward_paged` is assigned in all four branches and
  never read — dead.
- `build_for_cudagraph_capture` still says "CUDA Graph capture пока не
  поддерживаем (MVP)" after M2 (ds4 also flagged).
- `VLLM_GFX906_DENSE_GEMV` is read via `os.environ.get` on every call in a
  hot path (~91 calls/step); vLLM convention is the cached `envs` module.
  Negligible cost, but it's the pattern the codebase already solves.

### F10 (LOW) — records and repo hygiene drift

- **Untracked junk at repo root**: `.rocprofv3/` (≈479 MB of counter
  data), `gpucore.1.gpu` / `gpucore.10.gpu` (GPU fault dumps, unreadable),
  `_pp_bench.py`, `run_bench_gfx906.sh`. Add to `.gitignore` or move out;
  the fault dumps in particular are a tripwire for anyone running
  `git status`.
- **Key probes are not in the tree.** The devlog/plan cite
  `/tmp/bench/_p31_ab.py`, `/bench/probe_custom_fa.py`, `bench_ab2.py`,
  `test_backend_vs_legacy.py` — none checked in, despite the sub-plan's own
  W4: "check probes into `tests/` or `tools/` (not `/tmp`)". The three
  bit-exact 128-token probes that are the entire engine-level correctness
  evidence for the shipped default cannot be re-run from the tree. This is
  the most costly item in this list: when the next person reproduces (or
  disputes) 57.09 t/s, they have to rebuild the probes from prose.
- **Doc staleness**: plan v11 §1 "Where we are" table still shows 44.09
  t/s / 1.59× gap (superseded by 57.09 / 1.23× in the same file's status
  header); sub-plan §0 table still labels 52.90 "new best = default config"
  while its own v5 changelog records 57.09; "5-sample mean" labels next to
  six listed numbers (52.93/52.92/52.94/52.93/52.83/52.87); the P3-1 note
  "that file has 8 pre-existing failures on the base commit in this env"
  has no supporting detail (which tests, why) — worth one line so the claim
  is auditable.

### F11 (INFO) — default-path resource characteristics worth one paragraph
in the record

- Per-call `gather_paged_kv_fp16` K/V + `quantize_q8_0` K_q8 allocations in
  the LEGACY path are captured into the graph pool, so pool size ∝
  `max_capture_size × Sk_cap × Hkv × (2·2D + (D/32)·34) × #FA_layers`
  (≈0.6 GB at B=8, max_model_len=2816, D=256, 10 layers on this model).
  Fine on 32 GB; the sub-plan §6 VRAM formula (written for the LEGACY=0
  class buffers) now applies to the default path too.
- LEGACY=1 re-gathers and re-quantizes the *entire* KV history per step per
  FA layer — O(context) HBM traffic (≈2.9 MB K read + quantize write per
  layer at Sk=2816). Already tracked as stage-2 (fused quantize-during-
  gather); restating here because at long context this, not the FA kernel,
  becomes the dominant attention cost.

---

## 2. Verdict vs the plan's own acceptance criteria (incremental to ds4 §6)

| item | state |
|---|---|
| M2 exit: T3 passes / mode FULL / 5-sample σ / probes | ✅ as recorded — **but T3 validates `forward_paged` internals, not the backend's own buffer lifecycle (F1)** |
| M1 demotion (LEGACY=0 optional) | ✅ consistent — **but the live env still crashes/corrupts with no guard (F7)** |
| Sub-plan W5 ("same treatment for q_pad_buf") | ❌ not done; safety is an emergent property of vLLM's init order (F1) |
| Sub-plan W6 (D=64/128 early-out check; block_size=16 assert) | ❌ unrecorded / not done (F8) |
| Sub-plan W4 (probes in tests/ or tools/) | ❌ probes live in /tmp (F10) |
| FA perplexity ≤2% gate | ❌ not run (F3) |
| Multi-batch decode probe (P3-3a item ii) | ❌ not done at engine level (F3) |
| GEMV numeric pytest | ❌ bench-script asserts only (F2) |
| arch gating of GEMV dispatch | ❌ (ds4 C1) |

## 3. Recommended action order

1. **F1** — make the q_pad (and class gather) buffers capture-order
   independent: capacity pre-size or never-shrink + `narrow()` views, drop
   `empty_cache()` from forward, add the "real impl through capture →
   prefill → decode replay" test. This is the difference between "the bench
   flow happens to be safe" and "the path is safe."
2. **ds4 C1 + F2 + F4 + F5** — the four one-liners: `on_gfx906()` gate in
   `_llmm1_tiny_m`; GEMV numeric pytest; RPT clamp; V1 block-table guard.
3. **F3** — one perplexity point + one two-request (ideally partial-block
   prefix) multi-batch probe; check them into the tree per W4 while you're
   at it (F10).
4. **F6/F7/F8** — assert-not-capturing in realloc paths; LEGACY=0
   loud-warning + env table doc; one re-check of the V2 in-graph number and
   the two W6 records.
5. **F9/F10** — comment/doc cleanup, `.gitignore` the profiler output.

Net: the phase delivered what it claimed, and the measurement discipline is
the kind of thing that makes the next phase cheaper. The gaps are all in the
space between "the bench flow is verified" and "the default path is
structurally safe and durably tested" — F1 is the one to close first.
