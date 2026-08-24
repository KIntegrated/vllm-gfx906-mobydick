# STAS — Custom-FA gather-buffer OOM fix: adversarial code review

Branch: `feat/fa-oom-fix` (created from `gfx906/main` @ 4a9e24b5ca) · 2026-08-23.
Adversarial review of the design plan in `/local/tmp/gfx906-fa-fix.md` against the
actual code (`vllm/gfx906_fa/gfx906_fa_backend.py`, `gfx906_fa_paged.py`,
`csrc/gfx906_fa/gfx906_fa_gather.hip`, metadata builders). Review is static —
no code change, no run.

**VERDICT:** the root-cause diagnosis and the grow-only-reuse direction are
correct and well-quantified, but the plan's core mechanism ("pass the buffer
width as `Sk`") collides with a **hard C++ `gridDim.z ≤ 65535` limit**, and the
plan's central justification for it ("decode already executes at width 262144")
is **factually wrong** (decode executes at the *live* `max_seq_len`). The primary
patch as written is not implementable without a C++ change the plan explicitly
claims to avoid. A redesign decoupling width-reuse from Sk-passing (tile/grid-stride
the gather for `Sk > 65535`) is required.

**GATE:** `pytest tests/kernels/attention/test_gfx906_fa.py -k gather` with the
new width≫live unit (bitwise vs exact-width reference + NaN tail probe) — must
pass *before* any memory/perf claim is trusted.

## HYPOTHESIS — falsifiable one-liner

Plan §2: reuse any gather buffer whose width ≥ needed `Sk_pad` and pass the
buffer's own width as `Sk` to the persistent gather kernel, with **no C++ change**
(grow-only + free-eager-generations). If this is correct, the shipped persistent
kernel must be able to execute at `Sk = 262144` and decode must already operate
at that width today.

Falsified: (a) the persistent kernel's `blockIdx.z` is used as `tok_pos` up to
`Sk` with `gridDim.z` capped at 65535 (kernel header L401, launch L93/L421), and
(b) decode passes `max_seq_len = seqlens_k_local.max()` — the **live** longest
context, not the FULL-graph capture width. Both verified in code.

## What was done

- Created `feat/fa-oom-fix` from `gfx906/main`.
- Read plan `/local/tmp/gfx906-fa-fix.md` in full (§1–§5).
- Verified against code:
  - `vllm/gfx906_fa/gfx906_fa_backend.py` L490–596 `_ensure_gather_buffers`
    (exact `shape[2] == Sk_pad` guard L541, `_gather_retired` ClassVar dict L320,
    single rolling `_gather_captured` latch).
  - `vllm/gfx906_fa/gfx906_fa_paged.py` L451–528 (exact `shape ==` checks for
    kbuf/vbuf), L524 persistent call with `Sk_pad`, **L537 `elif _FUSED_QUANT and
    Sk_pad <= 65535`** fallback.
  - `csrc/gfx906_fa/gfx906_fa_gather.hip` L93/L421 `tok_pos = blockIdx.z`,
    L401 header "Sk must fit in gridDim.z (<= 65535)", launch `dim3 grid_d(grid,1,1)`
    L688, grid-stride variants at L504 (LEGACY decode path for Sk>65535).
  - Metadata: `Gfx906FAMetadataBuilder.build` → `max_seq_len =
    common_attn_metadata.max_seq_len`; backend calls `forward_paged(..., max_seqlen_k=
    attn_metadata.max_seq_len)`. `CommonAttentionMetadata.max_seq_len =
    int(seqlens_k_local.max())` (vllm/v1/attention/backends/utils.py L475) — the
    max **live** sequence length per step.

## Evidence FOR (the plan's correct parts)

- Root cause sound: unbounded `_gather_retired` keep-alive (Aug-19 `5d960a503c`),
  exact-match realloc per chunk-width growth, sim 89.4 GiB retired by 250k tokens;
  matches observed "free: 0" at ~2.5–4.5 min into prefill. Consistent with decode
  staying flat (decode replays FULL graphs; `_ensure_gather_buffers` not re-run).
- Grow-only (≥ capacity) reuse is the right direction; doubling hysteresis and the
  ≤8 growth-step bound are sound.
- Freeing eager/PIECEWISE-only generations (§2.2b) is the correct real fix (only
  graph-baked VAs must be kept alive).
- Validation plan structure, kill switches (`GFX906_FA_GATHER_EXACT`,
  `--attention-backend TRITON_ATTN`, PIECEWISE/NONE escapes) are well-formed.

## Evidence AGAINST (decisive, from code)

1. **gridDim.z limit.** `gather_paged_kv_quant_persistent` codes `blockIdx.z`
   (== `tok_pos` up to `Sk`) and the paged path's persistent call passes
   `Sk_pad`; the kernel header states `Sk <= 65535` hard limit. Passing
   `Sk = 262144 = buffer width` (the plan's invariant) exceeds gridDim.z → launch
   failure / undefined. No Z grid-stride in this kernel.
2. **Plan's "$ status-quo decode at width 262144" is false.** `attn_metadata.max_seq_len
   = seqlens_k_local.max()` is the **live** per-step max context length (e.g. 2816 in
   the canary; the growing chunk length in prefill). Graphs are *captured* against a
   wide buffer but *replayed* with live `max_seq_len` — the shipped kernel executes at
   live Sk, never 262144. §2.1's appeal to decode is a capture-width/execute-width
   conflation.
3. **The `Sk_pad <= 65535` guard already forces the allocating fallback at long
   context.** paged L537: `elif _FUSED_QUANT and Sk_pad <= 65535` else → fp16
   `gather_paged_kv_fp16` + `quantize_q8_0`, which allocates a fresh `[B,Hkv,Sk,D]`
   buffer per step — the very per-step allocation churn the plan claims to remove.
   The plan's §2.1 "no C++ changes required" is contradicted by the fact that
   `Sk_pad > 65535` already diverts to an allocating path before the plan's change.

## Why it fails

The design conflates **buffer capacity** (storage) with **kernel `Sk`** (grid
extent + shape). Grow-only storage reuse is valid; routing the raw width into a
Z-extent-limited kernel is not. And the "decode already does this" premise is the
load-bearing justification, so removing it collapses the "no C++ change" claim.

## Secondary correctness risks (must be landed with (a), not after)

- **Per-generation capture tracking.** Today `_gather_captured` is ONE rolling
  ClassVar latch, never cleared. The plan's §2.2(b) "free" branch would apply the
  *last-seen* capture state to every subsequent replacement, freeing a VA an
  *earlier* graph still replays → reintroduces the 2026-08-19 UAF. Captured-ness
  must be bound to the specific `(ptr, gen)` retired, not a rolling latch.
- **PIECEWISE capture-detection is asserted, not proven.** The claim "attention is
  never inside a PIECEWISE graph" (splitting-op `unified_attention_with_output`)
  matches the config but must be a hard assertion during the capture sweep, else a
  stale latch frees a baked VA.
- **Wide-buffer slice reuse `[:num_seqs]`** changes meaning from "exact-shape
  reuse" to "capacity reuse" — base-VA sharing of a *wide* parent is only safe if
  graph-baked generations are correctly kept (§2.2b). (a) and (b) are coupled.

## Recommended redesign (intent preserved, contract respected)

1. **Keep grow-only capacity reuse** of the buffer (no realloc on width grow).
2. **Pass the kernel `Sk = min(buffer_width, 65535)`.** For `Sk_pad > 65535`,
   add a **Z-tiled / grid-stride gather launch** (the grid-stride variant already
   exists for `Sk > 65535` at gfx906_fa_gather.hip L504 "LEGACY decode path").
   Long-context prefill then gets wide-buffer reuse *and* stays within grid.z.
3. FA cut already happens at `kv_max = seq_len` (live) — a 65535-capped gather
   width + live mask is correct and tail-safe regardless.
4. Land per-gen captured tracking (§2.2b) *before* (a)/(c), with a capture-sweep
   assertion that sweep-gen VAs live until all referencing graphs die.

## Refrigerated residue (keep for later, cross-linked)

- Fused q8 path (`gather_paged_kv_q8`, `key_cache_q8 is not None` + `_FUSED`) zeroes
  the whole V tail up to Sk — at width 262144 that is ~800 MB–3.3 GB memset per
  layer/step. Plan's carve-out is right; gate out or tail-bound before it can serve
  256k. Out of scope for this fix.
- `Qk8`/`use_or_alloc` exact-match C++ (gfx906_fa.cpp L933): keep exact once Python
  passes `Sk = min(width,65535)` — it guards stride/addressing.
- GPU-authorized hands-on experiments this session (GTT refuted: full 19.57 GiB load
  peak 20 MiB / 12,260 MiB; 3× in-process passes; 2× full canary 38.2/38.6 t/s) are
  separate from this code review and are recorded in `DEVLOG-boot-failure.md` §8.
