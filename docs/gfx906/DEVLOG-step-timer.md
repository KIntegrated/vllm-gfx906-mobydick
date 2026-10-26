# LAB-6 — per-region step timer (#50)

**VERDICT:** adopted. The timer is graph-safe by construction (CPU
`perf_counter` only), costs ~0.3 µs/step against a ~95 ms step, and its first
useful reading is negative: **host-side preparation is ~1 % of an MTP k=3
step**, so the step is GPU-bound and every argument that attributed step time to
"host prep on this stack" was wrong.
**GATE:** regions + unaccounted = measured step wall (exact by construction);
overhead < 1 % (`tests/gfx906_fa/test_step_timer.py` bounds the disabled path).
**Refs:** #50, `vllm/gfx906_step_timer.py`, `AGENTS.md` "Bench the metric, not the
proxy".

## What it measures, and what it cannot

`VLLM_GFX906_STEP_TIMER=1` (+ `..._TIMER_EVERY`, default 100) brackets each
engine step and a handful of named regions, then logs one line per window:

```
gfx906 steptimer: steps=120 wall=95.09 ms/step  prep_inputs=0.14 attn_metadata=0.69 unaccounted=94.26
```

- **Graph-safe:** `time.perf_counter()` on the calling thread, no CUDA calls, no
  syncs, no allocations. Regions inside a captured graph measure the *host*
  dispatch around it, which is what a replay costs on this stack.
- **`unaccounted` is a feature, not a leftover.** A budget that does not add up
  must say so; otherwise a missing hook reads as idle time.

Hook map (this is the part that cost a debugging round):

| region | where |
|---|---|
| `prep_inputs` | V1 `_prepare_inputs` call site **and** V2 `execute_model`'s request-state block |
| `attn_metadata` | V1 `_build_attention_metadata` call site **and** V2 `model_state.prepare_attn` |
| `gen_draft` | V1 `propose_draft_token_ids` call site only — V2 drafting lives in `spec_decode/speculator.py:74 propose`, which subclasses override |
| `ctx_kv` | **not hooked**: our FA backend has `forward_includes_kv_cache_update = False` and the write runs inside the compiled/captured model, so a Python hook would only fire in eager mode and would understate production |

**The first run measured a 90.24 ms/step wall with every region at zero** — the
hooks had been written against `v1/worker/gpu_model_runner.py` while this build
defaults to the **V2** runner. Without the "regions + unaccounted = wall" check
that would have looked like a successful instrumentation run with a mysterious
remainder.

## The measurement (dense 27B, MTP k=3, ladder `[4,8,12,16]`, maxlen 8192, B=1)

| region | ms/step | share |
|---|---|---|
| `prep_inputs` | 0.14-0.23 | 0.2 % |
| `attn_metadata` | 0.68-0.75 | 0.7 % |
| unaccounted (forward + sample + draft) | 93.6-97.3 | **99 %** |
| **step wall** | **94.5-98.3** | |

Consequences for the fork's measurement discipline:

1. **Stop attributing step time to host prep.** ~0.9 ms of ~95 ms is host-side;
   the remaining 99 % is GPU work plus the sampling/draft chain. An A/B that
   moves ms/step by 5 % is moving GPU work, not Python.
2. **The step budget is now derivable rather than inferred.** Previously ms/step
   was only obtainable as `tokens/step × t/s`; that arithmetic mixes the two and
   cannot separate a draft-chain change from a forward change.
3. **`gen_draft` and `forward` are the two regions still missing on V2**, and
   they are exactly the 99 %. Hooking them means wrapping the V2 forward call and
   a per-subclass wrapper for `Speculator.propose` — deliberately not done here:
   the cheap version (V1 hooks) is what exists, and a half-hooked V2 that silently
   reports zero would be worse than the honest `unaccounted` bucket.

## Overhead

Disabled: one branch per region (`< 50 µs` for 10,000 regions, asserted in the
test). Enabled: two `perf_counter` calls plus a float add (~0.3 µs per region,
four regions per step) against a ~95 ms step — 1.4e-5 of the step, i.e. four
orders below the 1 % budget.
