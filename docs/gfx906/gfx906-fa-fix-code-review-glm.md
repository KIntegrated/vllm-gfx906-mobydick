# Adversarial review — `/local/tmp/gfx906-fa-fix.md` (gather-buffer lifecycle fix)

Copyright Kevin Read <me@kevin-read.com>

Reviewed 2026-08-23 against `gfx906/main` @ `4a9e24b5ca` + the uncommitted
OOMHUNT instrumentation. Static review: every code-level claim in the plan was
checked against the actual sources (`vllm/gfx906_fa/gfx906_fa_backend.py`,
`gfx906_fa_paged.py`, `csrc/gfx906_fa/gfx906_fa.cpp`, `gfx906_fa_gather.cu`
/`.hip`, `kernel/fattn-q8_hip.cuh`, `gfx906_fa_launcher.cu`,
`vllm/v1/worker/gpu_model_runner.py`, `vllm/config/compilation.py`) and
against the companion docs (`oom-256k-prefill.md`, `degradation_details.md`,
`DEVLOG-masked-fa.md`, `DEVLOG-tp2-dense.md`, `DEVLOG-qwen38.md`). No code was
changed, nothing was run.

**Overall verdict: the kernel-level mechanism is sound and I verified every
claim the plan makes about the persistent gather + FA kernels — including the
two claims the sibling reviews dispute (see §6). "No C++ change required" is
correct *for the default persistent path*. However the plan as specified is
NOT merge-ready: the shared buffer-check change in `gfx906_fa_paged.py`
silently breaks buffer reuse on every non-persistent consumer path (§1, must
fix), the doubling hysteresis has no clamp and can overshoot `max_model_len`
by up to 2× (§2, must fix), and the plan's framing as *the* 256k-OOM root
cause is an unproven hypothesis whose headline number is not reproducible
from the artifacts it cites — three of the four referenced validation scripts
do not exist on disk (§4). Fix §1–§2, demote the root-cause claim to a
falsifiable hypothesis with the control arms in §5, then implement.**

Findings ordered by severity. §6 adjudicates the conflicting claims in the
sibling reviews (`gfx906-fa-fix-code-review-claude.md`, `-ds4.md`).

---

## 1. F1 (BLOCKER — design bug): the shared `kbuf`/`vbuf` check feeds ALL paths, but only the persistent call is changed to `Sk_buf`

Plan §2.2(c) changes the *shared* buffer-selection block in
`gfx906_fa_paged.py` (L446–466) from exact-tuple to capacity checks
(`shape[2] >= Sk_pad`), then changes `Sk` only at the persistent call site
(L518–528). But that shared block is the *only* place `kbuf`/`vbuf` are
selected, and three other consumers receive them further down:

- `gather_paged_kv_q8` (the `_FUSED`/LEGACY=0 branch, L468+),
- `gather_paged_kv_quantized` (the `_FUSED_QUANT and Sk_pad <= 65535`
  fallback, taken when `num_seqs > _PERSIST_MAX_SEQS` = 16, L540),
- `gather_paged_kv_fp16` (`v_out=vbuf` in the long-context fallback, L556+).

Every one of those goes through a C++ `use_or_alloc` lambda that checks
**`t.size(2) == Sk` exactly** (`gfx906_fa.cpp` L614–634 for q8, L829–850 for
quantized, L933–950 for persistent — all identical). Under the plan, whenever
the buffer is wider than the live `Sk_pad` (i.e., almost always, since the
buffer sits at capture width), those paths pass `Sk = Sk_pad` with a
`k_out` of width ≠ `Sk_pad` → the C++ check fails → **`use_or_alloc` silently
allocates a fresh `[num_seqs, Hkv, Sk_pad, …]` tensor per layer per step**,
and the class buffer is bypassed without any error. The plan even acknowledges
the fused path must "keep exact-match" — but the shared check change defeats
that; the plan is internally inconsistent here.

Concrete blast radius (all are *silent* — no exception, just lost reuse and
allocator churn):

- **Any eager/piecewise forward with `num_seqs > 16`** (e.g. the standing MoE
  35B bench config, `BENCH_MAX_SEQS=32`): falls to `gather_paged_kv_quantized`
  → per-layer per-step `torch::empty` of the full K+V size. Today the class
  buffer tracks `Sk_pad` exactly (that's the churn the plan fixes) and the
  C++ reuse *works* on this path.
- **LEGACY=0 (fused q8)** whenever width > `Sk_pad`.
- **The fp16 fallback** (`Sk_pad > 65535` with `num_seqs > 16`, or
  `_FUSED_QUANT=0`): `v_out=vbuf` mismatch → per-call V allocs.

Required fix (pick one, first is cleanest):

1. Keep the capacity check **only for the persistent branch**: select
   `kbuf_pers`/`vbuf_pers` with `>=` and pass `Sk_buf = width` to the
   persistent call; for all other branches keep today's exact-tuple check
   (so they get `kbuf=None` when the buffer is wide, exactly as today, and
   the C++ alloc is at least *consistent* with Python's view).
2. Or pass `Sk_buf` in every branch and change each C++ `use_or_alloc` to
   accept `t.size(2) >= Sk` — a C++ change the plan explicitly wants to
   avoid, and it would also widen the fused q8 path's full-width V-tail
   memset (correctly called out as a ~0.8–3.3 GB/layer/step cost in plan
   §2.2(d) — keep that path narrow).

Also note the same interaction for `vbuf` when `kbuf` is None (plan computes
`Sk_buf` from `kbuf` only): if the two checks can ever disagree, the
persistent call gets mixed reuse — make `Sk_buf` fall back consistently.

## 2. F2 (must fix): doubling hysteresis is unclamped — width can overshoot `max_model_len` by up to 2×

Plan §2.2(a): new width = `max(Sk_pad, 2 * old_width)` with the claim it is
"bounded to ≤ 8 growth steps from 1024→262144". That bound holds only for
power-of-two-aligned histories. `old_width` can be an arbitrary `Sk_pad`
value (any `ceil32(seq_len)`), e.g. 200,064 after a 200k prefill in a
262,144 model; the next growth to `Sk_pad = 262144` then allocates
`max(262144, 400128)` = **400,128 wide — 52% overshoot, permanent** (width
only ever grows). At B=2/Hkv=2/TP=2 that is ~1.25 GB vs the needed 822 MB —
paid out of exactly the headroom this fix is supposed to protect.

`_ensure_gather_buffers` does not know `max_model_len` (its docstring says
so), so the clamp needs either a signature change or a class-level cap set
from somewhere that does know it. Simpler alternative that avoids the whole
class of problems: drop doubling and mirror the **in-tree precedent the plan
never cites — `_ensure_forward_buffers` for `q_pad_buf` is already
grow-only** (`shape[2] < Sq_pad` → realloc at `max(Sq_pad, cur)`), with
capture-aware retirement (`_q_pad_retired`) — i.e. the plan is porting
proven q_pad semantics to the gather buffers. The q_pad pattern has no
hysteresis and no overshoot; its only cost is more (cheap, freed-immediately)
reallocs in no-capture modes, where policy (b) makes each replacement free.

Also account for the **B dimension** the same way: plan §2.3's table says
`_gather_retired` is "expected: empty" under FULL_AND_PIECEWISE, which holds
only while serving `num_seqs` never exceeds the largest captured size. A
serving batch larger than the largest capture size (trimmed capture lists
are the norm per the TP=2 recipe) triggers one B-grow realloc that retires
the full-width captured generation (~1.6–3.3 GB retained forever, plus a
same-size new gen). Either allocate the first generation at
`B = max_num_seqs` (needs the same plumbing as the width clamp) or document
the precondition "capture sizes must cover max_num_seqs" next to the table.

## 3. F3 (verified correct — recorded because it is the load-bearing claim): width-shaped execution is live-bounded end to end

I verified every link in the plan's §2.1 chain in the actual build sources
(note: the `.hip` files are untracked hipify artifacts; the `.cu` files are
what builds — both were checked and agree):

- Capture metadata: `gpu_model_runner.py:2389–2392` — `for_cudagraph_capture`
  forces `max_seq_len = self.max_model_len`, so FULL capture does call
  `_ensure_gather_buffers` at width 262,144, and capture descs are ordered
  descending (`descs[0]` is "largest"), so the first capture allocates the
  B=cap full-width generation and later sizes slice `[:num_seqs]` (same base
  VA) — one generation, `_gather_retired` stays empty. ✓
- Persistent gather: fixed 1-D grid `dim3(grid,1,1)` (`gfx906_fa_gather.cu`
  launcher, `grid` default 1024), grid-stride over a flat row space whose
  `total` is computed from the **device** `seq_lens` (`rph[s] =
  min(seq_len[s], Sk) + margin`, clamped), `rph[16]` with a `num_seqs <= 16`
  TORCH_CHECK. Work is live-bounded; **no `gridDim.z` dependence, no 65535
  cap** on this kernel. ✓
- FA kernel: `fattn-q8_hip.cuh:963` `k_VKQ_max = KV_max ? KV_max[…] : ne11`
  — with `kv_max` always passed by `forward_paged` (`kv_max_tensor =
  seq_lens`), the k-loop (L982) is bounded by the per-sequence live length,
  never by `Sk`; the launch grid (`gfx906_fa_launcher.cu` L219–225) is
  `(ceil(seq_q/NC1), kv_split, B×tiles)` — **no Sk dependence at all**, so
  there is no oversized-grid cost either. `ne11 = seq_kv` (width) is only the
  fallback when `KV_max == nullptr`, which never happens on this path. ✓
- Tail-tile OOB: the tile crossing `k_VKQ_max` runs with `oob_check=true`;
  K loads skip rows ≥ `i_sup` (L191) and V loads substitute zero **without a
  global read** (L163) — the wide-buffer tail garbage (uninitialized K q8
  bytes, uninitialized V past `seq_len+margin`) is never read. Same situation
  as today's mixed-length batches. ✓
- Addressing/strides: launcher computes `nb11/nb12` from the passed
  `seq_kv` = the tensor's actual width, `nb13` is int64; kernel K/V base
  offsets use int64. int32 `nb12 = 272 × 262144 = 71.3M` fits comfortably
  (overflow would need ~7.9M tokens). ✓
- C++ `use_or_alloc` passes trivially for the persistent path as claimed:
  Python passes `Sk_buf = kbuf.shape[2]` and the backend hands `forward_paged`
  the `[:num_seqs]` slice, so `t.size(0) == num_seqs` and `t.size(2) == Sk`
  both hold. ✓
- `gfx906_fa.forward` needs no change: `seq_kv = v_fp16.size(2)` feeds only
  shape checks (K/V both width → consistent) and `ne11` (unused). Inline
  causal via `q_abs_offset` is width-independent. ✓
- "Decode already runs this way": **true** — FULL decode graphs bake the
  capture-time launch (gather at `Sk=262144`, FA at `ne11=262144`) and replay
  it every step with live `seq_lens`/`KV_max` device tensors bounding the
  work. Corroborated in-tree: `DEVLOG-masked-fa.md` N4 ("gather was frozen at
  131k-wide for ~1.5k-live decode") and `DEVLOG-tp2-dense.md` S8 ("capture
  bakes pad32(max_model_len)"). ✓
- Piecewise-never-captures-attention: `vllm/config/compilation.py:766` lists
  `vllm::unified_attention_with_output` in the attention-ops/splitting set. ✓

So the plan's central engineering claim — grow-only width reuse with
`Sk = buffer width` on the persistent path, zero C++ changes — is correct.
The defects are in the *surrounding* policy, not the kernel math.

## 4. F4 (evidence — major): the root-cause framing is an unproven hypothesis, and its supporting artifacts are missing

- **The plan's §1 presents the gather-generation accumulation as established
  root cause.** The project's own same-day, evidence-backed diagnosis
  (`oom-256k-prefill.md`, VERDICT line) identifies the *failing* allocation
  as the gptq `temp_dq` dequant scratch (weight-shaped, 178 MB MLP / 2.37 GiB
  lm_head) and lists "FA buffer growth" as one *unprofiled headroom consumer*
  among several, plus "an unidentified ~1–2 GiB long-context transient". The
  two stories are compatible (FA accumulation could be the drain that leaves
  no room for the scratch), but the plan does not cite or reconcile with the
  OOM doc, and asserts more than the evidence supports. The mechanism is real
  in code (the exact-match realloc at `gfx906_fa_backend.py` ~L539–545 plus
  the unbounded `_gather_retired` is exactly as §1 describes) — whether it is
  *the binding constraint at 250k* is untested.
- **The headline number is not reproducible.** `/local/tmp/oom_hunt/
  verify_theory.py` does not exist (nor do `launch_9b_instrumented.sh`,
  `/local/tmp/launch_tp2_27b.sh`, or `/tmp/needle_256k.py` — §3's validation
  plan references four artifacts, zero are on disk; the OOM doc itself points
  at `/tmp` paths, which is tmpfs). My independent arithmetic for run 4's
  shape (B=1, chunk 1024 → `Sk_pad` = 1024·k, k=1..256; gen size
  1568·Sk bytes at B=1/Hkv=2) gives Σ ≈ 52.8 GiB of serving-time generations
  + 3.29 GiB capture gen ≈ **56 GiB, not 89.4**. Not damning — B=2 or a
  finer step schedule closes the gap — but the assumptions (B, step schedule,
  whether the capture gen is counted) must be published with the script, in
  a persistent location (`/local/tmp`, not `/tmp`).
- **The simulation makes a sharp prediction the plan never tests: headroom
  exhaustion at a roughly *fixed token count* (~30k raw / ~52k refined),
  independent of prompt length.** If true, a 100k-token prefill should die
  the same way at the same point — `needle_100k.py` exists per the OOM doc
  and was "never run". That is the cheapest decisive experiment available and
  it is absent from §3.
- **Unreconciled counter-evidence: "131k is the validated context ceiling on
  this model" (`oom-256k-prefill.md` L20).** A completed 131k-token prefill
  on post-Aug-19 code (`5d960a503c` introduced unbounded retire) would have
  retained ~14 GiB by the plan's own mechanism and should have OOM'd near
  the same ~30–52k point. Either that validation predates the UAF fix, ran
  eager/PIECEWISE (latch never set → generations freed, no accumulation), or
  the mechanism's magnitude is wrong. The OOM doc gives no run config or
  date for the 131k claim (the 131k records in `DEVLOG-tp2-dense.md` are the
  *dense* Qwen3.5-27B with ~1.5k-token prompts, not 131k prefills). The plan
  must reconcile this before claiming the fix closes 256k.

None of this argues against implementing the fix — an unbounded keep-alive
dict fed by per-chunk reallocs is a real bug worth fixing on its own — but
the plan should be re-titled/reframed as "fix a proven unbounded-retention
bug; *hypothesis*: this closes the 256k OOM", with the validation plan
below.

## 5. F5 (validation plan gaps)

Keep §3's sequence, add:

1. **TRITON_ATTN control arm**: one 250k prefill with
   `--attention-backend TRITON_ATTN` (config-level escape hatch the plan
   itself lists in §4). If it OOMs identically, the FA theory is dead and
   the fix is "worthwhile bugfix, not OOM closure". This is the single most
   informative run and it is missing.
2. **Fixed-token-count probe**: needle at 60k/100k (predicted to OOM at the
   same ~30–52k point if the mechanism is binding). `needle_100k.py` already
   exists per the OOM doc.
3. **Restore/recreate the missing artifacts** (`verify_theory.py`,
   `launch_9b_instrumented.sh`, `launch_tp2_27b.sh`, `needle_256k.py`) under
   `/local/tmp` (persistent) — or drop the references. A validation plan
   gated on scripts that don't exist cannot be executed.
4. **A unit test for F1's blast radius**: eager forward with
   `num_seqs > 16` against a wider-than-`Sk_pad` buffer, asserting no
   per-call C++ allocation (e.g. via `torch.cuda.memory_allocated` deltas or
   `data_ptr` identity of the returned `K_q8`/`V_bhsd` across layers).
   Without it, test item 4 (width≫live) only covers the persistent branch.
5. 9B probe expectations (§3.2) should also assert `alloc` count stays zero
   for a *multi-request mixed batch* (num_seqs > 16 step included), not just
   the single-request 100k prefill — that's where F1 would show up first.

## 6. Adjudication of the sibling reviews (both contain verifiable errors)

- **vs `gfx906-fa-fix-code-review-ds4.md`** — its BLOCKER ("collides with a
  hard `gridDim.z ≤ 65535` limit", "not implementable without a C++ change")
  is **wrong for the default path**: the 65535 cap lives in
  `launch_gather_paged_kv_quant` (non-persistent, grid `(B, Hkv, Sk)`,
  guarded in Python by `Sk_pad <= 65535`) and in the q8 V1 launcher (which
  self-switches to V2 above 65535); the **persistent** kernel the plan
  actually modifies has a fixed 1-D grid and no Sk-shaped launch dims at all.
  Its second pillar ("decode passes the live `max_seq_len`, not the capture
  width") confuses serving-time *metadata* (live, true) with the
  *capture-time baked launch config* (width 262144, replayed every decode
  step — see F3). The plan's "decode already runs this way" argument is
  correct. DS4's cap observation *is* relevant to F1's fallback paths,
  though.
- **vs `gfx906-fa-fix-code-review-claude.md`** — its premise-contradiction
  point is directionally right (see F4) but overstated as a refutation: the
  OOM doc's own mechanism paragraph lists "FA buffer growth" as a headroom
  consumer, so the plan is a quantification of a listed suspect, not a
  contradiction of the verified diagnosis. Its "part (b) does not fix what
  it claims" point is partially answered below (§7): (b) is a safety net for
  no-capture modes, which the plan does say, though §2.3's table under-sells
  the cases where retired is non-empty (F2's B-grow).

## 7. F6 (medium — policy (b) detail): the "per-generation" flag is not per-generation, and replacement inherits the latch

The plan's §2.2(b) says "Track captured-ness **per generation**" but the
snippet shows the same single bool with `old or capturing` semantics. Two
consequences:

1. On replacement, the *new* generation inherits `captured=True` from a
   captured predecessor even if the new gen is allocated eagerly. It will
   then be retired (retained forever) on its own later replacement —
   over-retention, i.e. precisely the memory the fix is trying to reclaim,
   in any mode that mixes capture and post-capture growth. Fix: on
   replacement set `_gather_buf_captured = capturing` (the new gen's own
   state), and only carry `or` semantics in the no-realloc refresh paths.
2. If the intent really is per-generation, store the flag with the
   generation (e.g. in `_gather_retired`'s tuple, or reset at alloc) — the
   current single-bool formulation will bite the rewritten keepalive test's
   item 3 (eager-mode grow → freed) whenever a captured gen precedes it.

The *policy* itself (free never-captured generations; PIECEWISE graphs never
contain attention — verified, F3) is sound, and the data_ptr-keyed
`_gather_retired` stays collision-free because only live tensors are ever
inserted.

## 8. Minor findings

- **F7 — test item 3's "tensor refcount dropped" is not reliably assertable**
  in Python (CPython refcounts include interpreter-held temporaries). Assert
  `data_ptr() not in _gather_retired` + that a fresh allocation reuses the
  old VA (allocator-level evidence of the free), or drop the clause.
- **F8 — missing tests**: (i) width≫live at `num_seqs > 1` with mixed
  `seq_lens` (margin clamp `extra > Sk - sl` behaves at width ≫ live);
  (ii) `GFX906_FA_GATHER_EXACT=1` A/B actually restores byte-identical
  behavior — the kill switch must gate **all three** sites (backend
  grow-only check, paged.py capacity checks, `Sk_buf` choice); the plan
  mentions the switch without enumerating the sites, and a partially-gated
  switch produces a meaningless A/B.
- **F9 — PIECEWISE/NONE-mode standing cost is new, and §2.3's table hides
  it**: today in no-capture modes generations are freed on replacement
  (latch never set), so steady state is one live gen at ~live width. After
  the fix, a long request leaves a full-width (up to `max_model_len`) buffer
  resident permanently — ~822 MB at B=2/Hkv=2/TP=2, ~3.3 GiB at B=8. Almost
  certainly the right trade (it's what FULL modes already pay), but the
  memory table should carry a PIECEWISE row instead of implying the fix is
  free everywhere.
- **F10 — line citations drift**: `_ensure_gather_buffers` is ~L465–596 (not
  L498), `_gather_retired` ~L327 (not L320), exact-match block ~L539–545
  (not L541–547). Also cite the `.cu` build sources, not the `.hip` hipify
  artifacts (DEVLOG-masked-fa build note); `fattn-q8_hip.cuh:963` is exact.
- **F11 — pre-existing oddity inherited unchanged** (no action needed, but
  worth one line in the plan): the full-width generation is allocated
  *during* the first capture, i.e. from the graph-capture private memory
  pool (in fact from the *profiling* pool first — the profiling capture
  phase runs before the real capture with the same capture metadata). The
  class var holding it across pool teardown works today (live reference
  keeps the block alive) and the plan preserves that behavior; just don't
  "clean this up" in the same change.

## 9. What the plan gets right (so it isn't re-litigated)

- Root-cause mechanism in code: exact-match realloc per `Sk_pad` growth +
  unbounded `_gather_retired` since `5d960a503c` — verified at
  `gfx906_fa_backend.py` L539–545/L327; decode/W4-soak flatness explained by
  FULL-graph replay skipping Python. ✓
- The pinned keepalive test encodes the old contract and must be rewritten;
  the shape-key-collision hazard it guards disappears entirely under
  grow-only (no same-shape regeneration). ✓
- §2.2(d) "what NOT to change" list is correct, including keeping
  `use_or_alloc` exact in C++ (given F1's fix, Python and C++ stay in
  agreement) and keeping the fused-q8 path narrow (its V-tail zeroing is
  width-proportional). ✓
- Validation is properly gated on the boot failure and ordered
  unit → 9B instrumented → 27B repro → accuracy → perf A/B. ✓

## Verdict summary

| Aspect | Verdict |
|---|---|
| Kernel-level mechanism (§2.1, persistent path) | **Verified correct, no C++ change** |
| Plan §2.2(c) as written (shared capacity check) | **Broken — see F1, gate per path** |
| Doubling hysteresis | **Broken — see F2, clamp or drop** |
| Policy (b) retire rules | Sound direction, latch-inheritance fix needed (§7) |
| Root-cause framing (§1) | **Hypothesis, not established — see F4** |
| Validation plan (§3) | Good bones, missing control arms + artifacts (F5) |
| DS4 review's "not implementable" | **Wrong for the default path** (§6) |

VERDICT: revise plan (F1, F2, §7, F5) before implementation; the code change
itself, once revised, is low-risk and worth landing regardless of whether it
closes the 256k OOM.
