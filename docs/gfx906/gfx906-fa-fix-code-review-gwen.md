# Code review (adversarial) — `gfx906-fa-fix.md` (gather-buffer lifecycle fix)

2026-08-23, review by Gwen. Target: `/local/tmp/gfx906-fa-fix.md` (design, not yet
implemented) against `gfx906/main` @ `4a9e24b5ca` working tree
(incl. the uncommitted OOMHUNT instrumentation). No code was changed by this
review; every line reference below is to the working tree. Companion:
`oom-256k-prefill.md` (the OOM it fixes), `DEVLOG-masked-fa.md` (N4).

## VERDICT

**The core fix is sound, and its safety case verifies to kernel level** — the
FA tile loop is `kv_max`-bounded (width never leaks into FA work), the
persistent gather is live-bounded, the margin (128) covers the max FA tail tile
(128), capture allocates at full width in descending B order, and the default
serving path (LEGACY=1, `_PERSISTENT=1`) is exactly the path the fix touches.
**Two substantive gaps must be closed before implementation:**

- **F1**: the proposed (b) retire-policy code does not reset the per-generation
  captured-flag on allocation — it is a sticky latch, so an uncaptured
  generation can be retired (kept alive) when replaced. One-line fix.
- **F2**: the plan only rewrites the *persistent* kernel call. The other three
  gather call sites (`gather_paged_kv_q8` L485, `gather_paged_kv_quantized`
  L529, `gather_paged_kv_fp16` L535 in `gfx906_fa_paged.py`) keep passing
  logical `Sk_pad` with the (now wide) `k_out`/`v_out` — the C++ `use_or_alloc`
  then **silently falls back to per-layer `torch::empty`** (no crash, no log).
  Buffer reuse is silently lost on every `num_seqs > 16` forward (e.g. the
  benchmarked 35B-A3B N=32 config) and on LEGACY=0. The plan must state the
  behavior and gate it (see §2 F2).

Plus test gaps (F3/F4), a docstring that must be rewritten (F5 — it carries the
exact wrong premise that caused the OOM), one upstream-coupling note (F6), and
minor items (§4).

## 1. Claims verified against the code (evidence)

| # | Plan claim | Verdict | Evidence |
|---|---|---|---|
| V1 | Exact-`Sk_pad` match forces a realloc on every Sk growth | **CONFIRMED** | `gfx906_fa_backend.py` L541-547: `b.shape[2] != Sk_pad` in the realloc condition. Shrink also reallocs. |
| V2 | FULL capture allocates at full width (`max_model_len`) | **CONFIRMED** | `build_for_cudagraph_capture` (L164) → runner-staged `profile_seq_lens=max_model_len` (comment L105-108, P3-3a M2); `_ensure_gather_buffers` allocates `torch.empty((..., Sk_pad, ...))` with `Sk_pad = ceil(max_seqlen_k/32)*32`. |
| V3 | One generation survives the whole capture sweep | **CONFIRMED** (with the F6 caveat) | Descending order: `vllm/v1/cudagraph_dispatcher.py` L326-352 — "Batch descriptors are sorted largest-first", PIECEWISE then FULL. Leading-dim slice reuse (backend L548-559) keeps the base VA across sizes. PIECEWISE capture runs attention *eagerly* (splitting op) → `is_current_stream_capturing()` False there; FULL capture bakes the one full-width gen. |
| V4 | The path actually serving 27B is the N4 persistent one | **CONFIRMED** | `GFX906_FA_LEGACY` defaults `"1"` (backend `__init__`, L~368) → `key_cache_q8=None` passed to `forward_paged`; `GFX906_FA_PERSIST` defaults `"1"`, `_PERSIST_MAX_SEQS=16` (paged L113/L120); 27B run-4 has B≤8 → `gather_paged_kv_quant_persistent` (paged L518-527). LEGACY=0 additionally fails closed with prefix caching (backend L117-131) — another reason it is not the default. |
| V5 | Passing `Sk = width` is safe: FA never iterates width tiles | **CONFIRMED** | `kernel/fattn-q8_hip.cuh` L963: `k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11` — `ne11` (= `seq_kv` = width) is only the no-`KV_max` fallback. Paged always passes `kv_max=seq_lens` (paged L641, L668). Tile loops (L966-994) bound on `k_VKQ_max` only. `seq_kv` elsewhere: consistency `TORCH_CHECK` (cpp L295, both tensors share the width) and the direct-paged launcher (L538) — neither in this path. |
| V6 | Wide-buffer tails are never read by FA (margin covers the tail tile) | **CONFIRMED** | Persistent gather: `gfx906_fa_gather.hip` L557-574 — `rph[s] = min(seq_len[s], Sk) + min(margin, Sk−seq_len)`, fixed grid grid-striding the flat live row space (work live-bounded, width-independent). Margin default **128** = "max D=256 nbatch_fa is 128" (`gfx906_fa.cpp` L104-115). FA tail read ≤ `seq_len + nbatch_fa − 1 ≤ seq_len + 127` < `seq_len + 128` = margin boundary. The stale region `[seq_len+128, width)` is never read. Decode already runs at width 262144 every step (soaked) — the status quo is the strongest proof. |
| V7 | C++ `use_or_alloc` exact match passes trivially when Python passes `Sk = buffer width` | **CONFIRMED** | `gfx906_fa.cpp` L933-950: checks `t.size(2) == Sk`, `t.size(0) == num_seqs` — Python passes the `[:num_seqs]` slice (exact B) and `Sk_buf = kbuf.shape[2]` (exact width). int64 strides throughout (L951+). |
| V8 | (d) fused-q8 path must stay exact (width-shaped tail zeroing) | **CONFIRMED** | The non-persistent kernels process every row up to `Sk`, zeroing V tail inline (`gfx906_fa_gather.hip` L21, L446; the "V tail zeroed up to Sk" is width-bound *work*) — plan correctly refuses wide reuse there. |
| V9 | `_q_pad_buf` slice+`contiguous()` is fine for kernel-*read* staging | **CONFIRMED, and already grow-only in-tree** | `gfx906_fa_backend.py` L430-470: q_pad already uses `shape[0] < num_seqs or shape[2] < Sq_pad → grow-to-max` — the exact pattern the plan proposes for gather buffers; it works. (q_pad is chunk-sized — `Sq_pad ≤ max_query_len ≤ chunk` — so it is not an OOM actor.) |

## 2. Findings

### F1 (must-fix, one line) — (b)'s flag is sticky, not per-generation

The plan says "track captured-ness **per generation**" but the proposed code
never resets the flag on allocation:

```python
cls._gather_buf_captured = (cls._gather_buf_captured
                            or torch.cuda.is_current_stream_capturing())
```

After a captured generation is retired and a fresh one allocated, the flag
remains `True`, so the fresh (never-captured) generation is retired instead of
freed when it is eventually replaced — reintroducing the keep-alive the fix
exists to remove, bounded to one full-width generation (~0.8 GiB at B=2/TP=2,
~3.3 GiB at B=8) in the rare Hkv/D/B-change-after-capture scenario. In the
normal FULL-capture flow the scenario doesn't arise (no post-capture
replacement), which is why it's easy to miss — but the code as written does not
implement the stated contract. **Fix: set `cls._gather_buf_captured = False`
immediately after each allocation** (both the first-alloc and realloc branches).

### F2 (must-fix, scope gap) — the other three kernel call sites silently lose reuse

The plan rewrites only the persistent call (paged L524-527). After the fix, the
shared buffer checks (L451-466, now `≥`) pass the **wide** `kbuf`/`vbuf` into:

- L485 `gather_paged_kv_q8(..., Sk_pad, k_out=kbuf, v_out=vbuf)` (LEGACY=0+FUSED),
- L529 `gather_paged_kv_quantized(..., Sk_pad, k_out=kbuf, v_out=vbuf)`
  (**num_seqs > 16**, Sk_pad ≤ 65535),
- L535 `gather_paged_kv_fp16(..., Sk_pad, v_out=vbuf)` (num_seqs > 16, Sk_pad > 65535).

C++ `use_or_alloc` (all sites, e.g. cpp L614-634, L719-737, L829-849) checks
`t.size(2) == Sk` and on mismatch **silently returns `torch::empty(...)`** — no
`TORCH_CHECK`, no log. So every `num_seqs > 16` forward (and every LEGACY=0
forward) allocates fresh per-FA-layer K/V outputs instead of reusing the class
buffer — the exact "each attention layer allocates 24-200+ MiB per step → peak
VRAM spike → OOM" failure mode the reuse mechanism was built for (paged L449-451
comment), now *silent* (today the exact-width buffer reuses fine on these
paths; the fix is what breaks it).

Reachability is concrete: the gfx906 platform makes this backend the **default
for dense decoders** (`vllm/platforms/rocm.py` L498-519), and the benchmarked
35B-A3B config runs `max_num_seqs=32` → decode at B=32 hits `gather_paged_kv_
quantized`. **Required:** (i) state explicitly what each non-persistent call
does with a wide buffer — recommend passing `k_out=None, v_out=None` there
when `kbuf.shape[2] != Sk_pad` (deliberate non-reuse, one log line at first
occurrence), since extending `Sk_buf` is *not* an option for these kernels
(their tail zeroing is width-bound work — V8); (ii) add a B=17…32 prefill/decode
smoke test for correctness; (iii) add a 35B N=32 decode t/s A/B to the perf
gate (expect flat; a drop means allocator churn is costing more than
predicted).

### F3 (test gap) — the width≫live test must poison the stale tail

Plan §2.4's new unit test probes the *output* for NaN. The adversarial form is
to **fill the stale region `[seq_len + margin, width)` of the wide buffer with
NaN/Inf before the forward** (it is exactly the data that would be read if the
tail mask were broken — e.g. by a future change dropping `kv_max`), then assert
bitwise equality of in-range rows vs the exact-width reference AND no NaN/Inf
in the output. With V6's margin arithmetic the test should pass, and it pins
the invariant that currently rests on a comment.

### F4 (test gap) — §2.4 test 2 is not constructible in the real flow

"Genuine width grow post-capture → previous **captured** gen lands in
`_gather_retired`" cannot happen: FULL capture always runs at full width (V2),
so post-capture `Sk_pad > width` is unreachable in vLLM. Restructure: drive the
retire branch directly (capture against a forced narrow width — e.g. a test-only
`max_seq_len` override on the dummy metadata — then grow past it), or unit-test
the flag/retire logic with the flag set manually. As written, the test would
either fail to trigger the branch or silently pass without exercising it.

### F5 (doc) — the class docstring carries the wrong premise that caused the OOM

`gfx906_fa_backend.py` L305-322 (the `_gather_retired` docstring): "Replacements
after the first capture only happen on Sk/Hkv/D growth … so this set stays
small." The Sk-growth clause is *the bug* (chunked prefill grows Sk every
chunk; 89.4 GiB retired by 250k in the simulation). The plan's (a) docstring
note is not enough — this class-level comment must be rewritten to the post-fix
bound: replacement happens only on Hkv/D/B change (Sk is a capacity now), so
the set holds at most the captured generation(s) — normally exactly one, and it
is the VA the FULL graphs replay.

### F6 (robustness) — the "one generation across the sweep" property couples to vLLM capture order

Verified today (V3) that `get_capture_descs` sorts largest-first (PIECEWISE
then FULL). If upstream ever reorders to ascending B, each B-growth during
FULL capture would retire a full-width generation: B=1→2→4 keeps ~2.3 GiB
alive at TP=2/Hkv=2 (1+2)×2×195 MiB — more than the 1.94 GiB run-4 headroom,
i.e. the fix would silently OOM at capture instead of prefill. Cheap guard:
warn (or assert in tests) when `len(_gather_retired) > 0` at end of capture;
the OOMHUNT instrumentation already logs `retired=` on every alloc — keep it
through validation so the first regression is loud.

### F7 (minor) — validation config inconsistency

§3.3 says "capture [1,2,4,8]" but the final 256k config is `--max-num-seqs 2`
(house recipe trims capture to max_num_seqs → [1,2]). Reconcile: with
max_num_seqs=2 the capture-time generation is the 822 MB line; with 8 it is the
3.3 GiB line — the headroom verdict in §2.3 changes accordingly. State which
one the needle/A/B runs use.

### F8 (minor) — "256 hipMalloc/hipFrees" framing misses the sharper OOM mechanism

The per-chunk reallocs go through the caching allocator, and the transient
*double* peak (old gen still alive while the new gen allocates — plus all
retired gens) is the more direct OOM driver than the dict itself. The fix
removes both; the plan should say so, because it predicts the 256k prefill
completes even if some residual retire growth were observed.

## 3. Validation-plan gaps (on top of F2-iii/F3/F4)

1. **MTP draft path**: the draft layer's attention shares the same class-level
   buffers (same Hkv/D). The 27B repro covers it end-to-end, but add the
   draft-prefill case to the width≫live unit (uniform multi-token q batch,
   paged L597-604 path) — it is the one forward shape the unit plan doesn't
   include.
2. **Live confirmation of the diagnosis itself is still outstanding**: the
   89.4 GiB figure is a simulation; the OOMHUNT 9B probe (§3.2) is the first
   live data point. Until it passes, "fixes the 256k OOM" is a prediction —
   the plan should say the 9B probe is a *diagnosis gate* (expect: one alloc at
   capture, zero allocs during the 100k prefill, `retired=` constant) ahead of
   the 27B repro.
3. **Host state**: the plan is "blocked on the weight-load failure"; that fault
   is now known to be an intermittent GPU die/fabric flap with multi-minute
   good windows (DEVLOG-boot-failure.md §7). Multi-minute 256k attempts will
   get interrupted mid-run; expect re-runs and log any wedge per protocol.
4. **House gate**: this needs its own branch + dev log (`DEVLOG-fa-gather-
   lifecycle.md` or similar) with the serving-wall A/B (graph + eager) as the
   gate, per house rules. The OOMHUNT instrumentation stays in-tree only for
   validation and must be reverted before merge.

## 4. Minor / nits

- Couple the k/v buffer decision in paged (currently independent checks): use
  the pair only if **both** pass, so a mixed state (k reused, v fresh-alloc
  per layer at `Sk_buf`) can't arise. Pre-existing risk, not introduced by the
  fix, but the rewrite is the moment to fix it.
- Plan §2.2a "bounded to ≤ 8 growth steps from 1024→262144" — from a cold
  first alloc (e.g. 128) it is 11 doublings; harmless, but the "no-capture
  mode" path (PIECEWISE-only serving) is where those happen, so the number
  should be right.
- `GFX906_FA_GATHER_EXACT=1` kill switch: good; also document that
  `GFX906_FA_CG=never` removes the captured-gen entirely (retire set stays
  empty) — that is the cheapest bisect knob if anything regresses.
- The plan's line references (backend L320/L498-596/L541-547/L569-574; paged
  L334/L451-466/L524-528/L641/L668) all check out against the working tree.

## Bottom line

Implement as designed, with F1 (flag reset) and F2 (non-persistent call sites
+ B>16 smoke + 35B N=32 A/B) closed in the same change, F3/F4 test fixes, F5
docstring rewrite, and F6's end-of-capture warning. The memory and safety
arguments hold at every level checked (Python reuse logic → C++ `use_or_alloc`
→ launcher → FA kernel tile loop → margin arithmetic); the residual risk is
scope, not mechanism.

Copyright Kevin Read <me@kevin-read.com>
