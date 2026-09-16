# FA non-causal attention (for DFlash2 drafters)

## 2026-09-16 — design: what the drafter needs, what our FA lacks, and the cheapest way to know it matters

**VERDICT:** OPEN (design complete; no code yet) · **GATE:** DFlash2 per-position acceptance with
the recommended pair — currently 0.0 because the drafter cannot use our FA at all.

## Why this exists

`DEVLOG-dflash2.md` (2026-09-16) root-caused DFlash2's acceptance collapse to the drafter's
attention path rather than to the model, the checkpoint or the quantisation pairing: with the
*matched* pair (AWQ-INT4 target + `syvai/Qwen3.8-27B-DFlash2-W4A16`, k=7) acceptance is still 0.0,
and the reasons line added for DFL2-8 says why:

```
Found incompatible backend(s) [CUSTOM, TURBOQUANT] with AttentionType.DECODER.
  Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN'].
  Reasons: {CUSTOM: [non-causal attention not supported], ...}
```

The predicate is `vllm/v1/attention/backend.py`: `if use_non_causal and not cls.supports_non_causal()`
— and our `Gfx906FABackend` simply does not define `supports_non_causal`, so it is rejected for any
layer built with non-causal attention. The DFlash2 drafter is such a layer: all 5 of its layers are
`sliding_attention` with `sliding_window: 2048` and `is_causal: false`, and `DFlashQwen3Attention`
(`qwen3_dflash.py`) is built with `causal=False` and drives a vLLM `Attention` layer
(`self.attn = Attention(...)`, then `self.attn(q, k, v)` — the context K/V are pre-inserted into the
paged cache by `precompute_and_store_context_kv`). The fallback it gets instead (ROCM_ATTN with the
Triton paged kernel) cannot be CUDA-graph captured — `RuntimeError: Cannot copy between CPU and CUDA
tensors during CUDA graph capture` through `rocm_attn.py` -> `chunked_prefill_*` — and drafts nothing.

## The semantics to implement

Match the reference helper `_maybe_symmetrize_window` (`vllm/v1/attention/backends/flash_attn.py`):
a causal window `(w, 0)` becomes **symmetric `(w, w)` when attention is non-causal**, so
bidirectional queries attend in both directions; already-symmetric windows and full attention are
left alone. For the drafter that means: **no causal clip, and a symmetric ±(window) clip** — i.e.
keys are visible when `|k_pos - q_pos| <= window`, with `window = sliding_window` (2048).

## What our kernel does today

`csrc/gfx906_fa/kernel/fattn-q8.cuh` (and the three sibling kernels) mask with the absolute
position, gated on `q_abs_offset`:

```cpp
if (q_abs_offset) {
    const int q_abs_row = q_abs_base + j;
    if (k_pos_abs > q_abs_row || (window > 0 && q_abs_row - k_pos_abs >= window)) {
        KQ_acc[jc0] = -INFINITY;    // future keys, and keys older than the window
    }
}
```

That is a causal clip plus a **left-only** window: there is no right-side clip and no way to allow
future keys. Non-causal needs exactly two changes at each site: drop `k_pos_abs > q_abs_row`, and add
the symmetric right clip `k_pos_abs - q_abs_row > window`.

## Edit inventory (measured, not estimated)

- **Kernel mask sites, 8** — `k_pos_abs > q_abs_row` appears twice per file (the two inner-loop
  variants) in each of `fattn-q8.cuh`, `fattn-q8-paged.cuh`, `fattn-q8_hip.cuh`,
  `fattn-q8-paged_hip.cuh`.
  - Caveat: the two `*_hip.cuh` sites have the causal test **without** the window clause, so those
    variants do not implement sliding windows at all. Before wiring non-causal in, confirm which
    variant actually serves this shape (the launcher picks between the generic and hip paths), or the
    flag would be honoured on a path that is never taken.
- **Kernel signatures**: 4 files, plus the launcher (`gfx906_fa_launcher.cu`, ~8 sites: two launch
  wrappers and their `flash_attn_tile_q8*` calls) and the op (`gfx906_fa.cpp`, ~6 sites incl. the
  `window > 0 requires q_abs_offset` style `TORCH_CHECK`s).
- **Python**: one call site (`gfx906_fa_backend.py`, `window=self.sliding_window`) plus the impl
  constructor (read `vllm_config.attention_config.use_non_causal`) and the backend class
  (`supports_non_causal()` -> True, only once the path above is real).
- The plumbing is a flat argument list, not a struct, which is why this is ~25-30 edits rather than
  a handful.

## Cheaper decisive experiment first

Before spending that change: **does correct non-causal attention actually restore acceptance?** If it
does not, the kernel work buys nothing and the DFlash2 family stays parked.

Plan: an env-gated (`VLLM_DFLASH2_EAGER_ATTN=1` or similar) path in `DFlashQwen3Attention` that
computes the drafter's block attention in torch with a **symmetric-window mask** instead of calling
`self.attn`. The context K/V are available densely at the moment they are projected:
`precompute_and_store_context_kv` builds `all_k, all_v` (`_project_context_kv`, `qwen3_dflash.py`) —
so the experiment can stash those and run
`scaled_dot_product_attention(q, cat([ctx_k, blk_k]), cat([ctx_v, blk_v]), attn_mask=window_mask)`.
Slow (eager, per step) — but it answers the scientific question, and its acceptance number is the
justification (or the refutation) for the kernel work.

## Gates

1. Experiment: per-position acceptance with the eager symmetric-window path, 2 reps, ~12k prompt,
   compared against 0.0 today and syv-ai's 3.1-3.6 tokens/step reference.
2. If positive: implement the kernel flag + the `supports_non_causal()` flip, then re-run 1 with the
   real backend, plus the FA suite (must stay green; the flag defaults off so behaviour is unchanged
   until the flip).
3. Only then do DFL2-3 (lookup drafting) and DFL2-1 (chains) make sense.
