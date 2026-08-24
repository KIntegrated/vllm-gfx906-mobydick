# Adversarial review — `/local/tmp/gfx906-fa-fix.md` (gather-buffer lifecycle fix)

Copyright Kevin Read <me@kevin-read.com>

Reviewed 2026-08-23 against `gfx906/main` @ `4a9e24b5ca` +
uncommitted `vllm/gfx906_fa/gfx906_fa_backend.py` (OOMHUNT_LOG
instrumentation only).

**This revision folds in and adjudicates three sibling reviews**
(`gfx906-fa-fix-code-review-ds4.md`, `-glm.md`, `-gwen.md`) against the
actual source, resolving the direct contradictions between them. The
consolidated, adjudicated findings were used to rewrite the plan at
`docs/gfx906/plan-gfx906-fa-fix.md`; this file is kept as the review
record. Original (pre-fold) findings are preserved below where they
still hold.

## Adjudication summary

| Claim | Source | Verdict |
|---|---|---|
| Root-cause framing overstated vs `oom-256k-prefill.md` | claude, glm | **Confirmed** — kept, softened per glm's nuance (not a flat contradiction) |
| Headline "89.4 GiB" sim number unverifiable (`verify_theory.py` missing) | claude, glm | **Confirmed** — file does not exist on disk |
| Part (b) captured-flag is sticky, not per-generation | claude, glm, gwen (F1) | **Confirmed by all three independently** — real bug |
| `_PERSIST_MAX_SEQS=16` scope hole (fused/quantized paths unfixed) | claude | **Confirmed, and sharper than I originally stated** — see ds4/glm/gwen below |
| Shared `kbuf`/`vbuf` capacity check silently breaks non-persistent call sites | glm (F1), gwen (F2) | **Confirmed, distinct bug from the `_PERSIST_MAX_SEQS` gap** — I missed this in the original pass |
| Doubling hysteresis unclamped, can overshoot `max_model_len` permanently | glm (F2) | **Confirmed by arithmetic** — real bug |
| `gridDim.z <= 65535` blocks the persistent-path fix / "no C++ change" is false | ds4 | **Rejected** — verified wrong; see below |
| "Decode already runs at width 262144" is false (conflates live metadata with baked launch args) | ds4 | **Rejected** — verified wrong; see below |
| Joint `(num_seqs, Sk_pad)` growth unspecified | claude | Superseded by glm's more precise F2 formulation — folded in |
| `splitting_ops=[]` is a present-day reachable config, not just hypothetical | claude | **Confirmed, unchallenged by others** — kept |
| Capture order (largest-first) is load-bearing and coupled to upstream behavior | gwen (F6) | **Confirmed** — new finding, folded in |
| Test §2.4 item 2 unconstructible in the real flow (FULL always captures at full width) | gwen (F4) | **Confirmed** — new finding, folded in |

## Adjudicating the ds4-vs-gwen contradiction (decisive, checked in source)

ds4's BLOCKER claims passing `Sk = buffer width` up to 262144 hits a
hard `gridDim.z <= 65535` cap and that decode does not actually run at
width 262144 today. Both claims were checked directly:

**gridDim.z claim — wrong for the path the plan modifies.** The
65535 cap is real but lives in the *non-persistent* kernels only:
`gather_paged_kv_quant_kernel` (V1, `dim3 grid(num_seqs, num_kv_heads,
Sk)`, comment at `gfx906_fa_gather.hip:401` "Sk must fit in gridDim.z"),
its V2 fallback, and the matching `TORCH_CHECK(Sk <= 65535, ...)` in
`gfx906_fa.cpp:821` (`gather_paged_kv_quantized`). The **persistent**
kernel the plan's §2.2 actually rewrites
(`gather_paged_kv_quant_persistent`, `gfx906_fa_gather.hip:660-698`)
launches a **fixed 1-D grid** — `dim3 grid_d(grid, 1, 1)` with `grid`
defaulting to 1024 — and grid-strides over a flat live-row space
computed from the device-side `seq_lens` tensor (`rph[s] = min(seq_len,
Sk) + margin`). `Sk` here is a data value bounding the row-count math,
never a launch dimension. No gridDim.z dependence exists on this path.
ds4's citations (L93/L401/L421 "Sk must fit in gridDim.z") are real
lines but belong to the sibling V1/V2 kernels, not the one being
changed — ds4 conflated the two kernel families.

**"Decode executes at live Sk, not 262144" — wrong.** ds4 is correct
that `attn_metadata.max_seq_len` (the *Python metadata object*, used
by the *eager*, non-captured code path) reflects the live per-step
length. But that is not what determines the kernel launch during a
**replayed CUDA graph**. `max_seqlen_k=attn_metadata.max_seq_len`
(`gfx906_fa_backend.py:688` etc.) is a plain Python `int`; it becomes
`Sk_pad`, a literal scalar kernel-launch argument. CUDA graph capture
bakes Python-side scalar launch arguments into the graph as fixed
constants — replay does not re-run the Python code that computed them.
`build_for_cudagraph_capture` (`gpu_model_runner.py:2387-2390`) sets
`max_seq_len = self.max_model_len` specifically `for_cudagraph_capture`,
so the FULL graph's baked `Sk` argument for every persistent-gather and
FA launch is `max_model_len`-derived, replayed unchanged on every
decode step. Only `seq_lens`/`kv_max` — genuine device tensors read
*inside* the kernel — carry the live length at replay time. This is
exactly gwen's V2/V3/"decode already runs this way" chain, and it
checks out. **ds4's BLOCKER and its "factually wrong" charge against
the plan are themselves incorrect**; ds4's redesign recommendation
(Z-tile the persistent kernel) is unnecessary. ds4's grid.z observation
does remain useful as input to the `_PERSIST_MAX_SEQS`/non-persistent
scope-gap finding below, where it's actually relevant.

## Distinct scope-gap findings, both confirmed and complementary (not redundant)

Two different reviews independently found two different problems in
the same neighborhood; both are real and address different call sites:

1. **`_PERSIST_MAX_SEQS = 16` gate** (my original finding): when
   `num_seqs > 16`, execution falls through to
   `gather_paged_kv_quantized`/`gather_paged_kv_fp16` entirely — those
   paths are explicitly out of scope per the plan's own §2.2(d), so
   batches above 16 get no benefit from the fix at all, growth bug and
   all.
2. **Shared buffer-check regression** (glm F1 / gwen F2, confirmed by
   reading `gfx906_fa_paged.py:451-466` and `gfx906_fa.cpp`'s three
   `use_or_alloc` lambdas, which are byte-identical `t.size(2) == Sk`
   checks): the plan's §2.2(c) snippet changes the **shared** Python
   `kbuf`/`vbuf` selection block from exact-match to `>=`, but only
   changes the `Sk` value passed to the *persistent* call site. The
   `gather_paged_kv_q8` (LEGACY=0), `gather_paged_kv_quantized`
   (`num_seqs > 16`, `Sk_pad <= 65535`), and `gather_paged_kv_fp16`
   (`num_seqs > 16`, `Sk_pad > 65535`) call sites still receive
   `Sk_pad` (not the wide buffer width) alongside a now-wider
   `k_out`/`v_out`. C++'s exact check silently fails and falls back to
   `torch::empty` per call — **buffer reuse breaks silently on paths
   that work fine today**, which is a regression the plan introduces,
   not merely an unfixed gap. This is worse than finding 1: finding 1
   is "no improvement" for `num_seqs > 16`; this is "actively worse
   than current behavior" for those same call sites once the shared
   check changes.

Both must be fixed in the same change: gate the capacity-check rewrite
to the persistent branch's buffer selection only, leaving the other
three call sites on today's exact-match selection (so they get
`kbuf=None`/fresh-alloc exactly as before — no regression, no
silent divergence).

## Doubling hysteresis: confirmed unclamped overshoot (glm F2)

Verified by arithmetic: `new_width = max(Sk_pad, 2 * old_width)` with
no `max_model_len` clamp. Example: `old_width = 200,064` (a real
`ceil32` value reachable via chunked growth before the fix's grow-only
policy fully takes over, or via any request just under 200k), next
growth to `Sk_pad = 262,144` allocates `max(262144, 400128) = 400,128`
— a **52.6% permanent overshoot**, paid out of the same headroom the
fix is meant to protect (~1.25 GiB vs. the needed 822 MiB at
B=2/Hkv=2/TP=2). `_ensure_gather_buffers` has no visibility into
`max_model_len` today (its own docstring says so), so the clamp needs
either a threaded-through cap or — the simpler fix, and the one
adopted in the rewritten plan — dropping doubling hysteresis
entirely and mirroring the in-tree `_q_pad_buf` precedent
(`gfx906_fa_backend.py:449-469`), which is already grow-to-exact-need
with no hysteresis and works.

## Other confirmed findings folded in

- **Part (b)'s captured flag is sticky, not per-generation** — all
  three sibling reviews independently found this by reading the same
  six lines of proposed code; confirmed identical bug across all three
  independent analyses. The proposed `_gather_buf_captured =
  (old or capturing)` is never reset to `False` on allocation, so once
  any capture has occurred, every later replacement is retired forever
  — functionally unchanged from today's bug. Fix: reset to `capturing`
  (the new generation's own state) at the point of allocation, not
  `or`ed forward from the retired generation.
- **Capture-order coupling** (gwen F6): the "one generation survives
  the whole sweep" property depends on `get_capture_descs` sorting
  batch sizes largest-first (verified at
  `vllm/v1/cudagraph_dispatcher.py:326-344`). If a future vLLM upgrade
  changes that order, ascending-B capture would retire a full-width
  generation at every B-growth step during capture itself — the fix
  would then OOM at capture time instead of prefill time. Cheap to
  guard (assert/warn if `_gather_retired` is non-empty at end of
  capture); folded into the rewritten plan's test/validation section.
- **Test item 2 unconstructible** (gwen F4): "genuine width grow
  post-capture" cannot happen in the real flow, since FULL capture
  always runs at `max_model_len` (confirmed above) — so `Sk_pad >
  captured width` is unreachable post-capture. The test needs to drive
  the retire branch directly (forced narrow capture width, or manual
  flag manipulation) rather than relying on organic post-capture
  growth.
- **`splitting_ops=[]` is a present-day config**, not hypothetical
  (my original finding, unchallenged by the other three reviews):
  `vllm/config/compilation.py:1135-1200` allows `splitting_ops=[]`
  explicitly (warns, does not block, for PIECEWISE/FULL_AND_PIECEWISE).
  This defeats part (b)'s "PIECEWISE never contains attention" safety
  argument at the config level, independent of the sticky-flag bug.
- **Root-cause framing**: kept as a finding but softened per glm's
  more careful reading — `oom-256k-prefill.md` does list "FA buffer
  growth" as one of several unprofiled headroom consumers, so the plan
  is better described as *quantifying a listed suspect* than
  *contradicting the verified diagnosis* outright. It still asserts
  more certainty than the evidence supports (the missing simulation
  script, the un-reconciled 131k-ceiling claim from the OOM doc, and
  the absence of the single most informative control experiment —
  a `TRITON_ATTN` backend A/B on the same 250k request, which glm
  correctly flags as missing and cheap).

## Not adopted / rejected

- ds4's Z-tile redesign recommendation — unnecessary; the persistent
  kernel has no gridDim.z dependency (see adjudication above).
- ds4's claim that `Sk_pad <= 65535` already forces an allocating
  fallback "before the plan's change" as evidence against "no C++
  change required" — true only for the non-persistent paths, which the
  plan never claimed required no C++ change; the persistent path (what
  "no C++ change" actually refers to) has no such limit.

## Disposition

The plan has been rewritten at `docs/gfx906/plan-gfx906-fa-fix.md`
incorporating every confirmed finding above. Do not treat the original
`/local/tmp/gfx906-fa-fix.md` as current; it is superseded by the
in-tree plan.
