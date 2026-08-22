# Code review — branch `gfx906/fa-masked-gather` (N4 max_model_len tax fix)

Copyright Kevin Read <me@kevin-read.com>

Reviewed: 2026-08-22, against `49d4876e72..25729a2560` (the three original
commits), re-reading every changed line of
`csrc/gfx906_fa/gfx906_fa_gather.cu`, `csrc/gfx906_fa/gfx906_fa.cpp`,
`vllm/gfx906_fa/gfx906_fa_paged.py`, the four committed probes, the suite
test additions, and the `ppl_probe.py` param. Findings below; the P0 was
fixed on-branch during the review as `ddd2adbdeb`.

## VERDICT

**SHIP-WORTHY AFTER P0 FIX (applied).** The kernel design is sound and the
gated evidence is real; the code is cleaner than the plan suggested
(margin clamping, D-generality, and the block-table OOB guard are all
present in-kernel). One P0 (unguarded batch bound → engine-start crash for
any `max_num_seqs > 16`, invisible because every gated run used
`--max-num-seqs 4`), two coverage gaps that matter for the default-ON
flip (MTP, D≠256), and a set of P2s. All findings below have evidence and
status.

## Findings

### P0-1 — unguarded `num_seqs > 16` → engine-start crash (FOUND, FIXED in `ddd2adbdeb`)

- **What**: `forward_paged`'s legacy branch dispatched
  `gather_paged_kv_quant_persistent` whenever `_PERSISTENT` (default ON) —
  no `num_seqs` check. The C++ side `TORCH_CHECK(num_seqs <= 16, ...)`
  (kernel prefix-sums per-seq counts in a fixed 16-entry register array),
  i.e. it **throws**.
- **Blast radius**: vLLM's default FULL-cudagraph capture sizes are
  `[1,2,4] + range(8,256,8) + range(256, cap≤512, 16)`
  (`vllm/config/compilation.py`), so any `max_num_seqs > 16` — the vLLM
  default is 1024; the house MoE 35B config is 32 (AGENTS.md) —
  captures B=24..512 batches and/or the i==0 dummy-run profiling passes
  B>16 sequences → `RuntimeError: num_seqs must be <= 16` at engine
  start. Every gated run on this branch used `--max-num-seqs 4` with a
  trimmed capture list, which masked it completely.
- **Fix applied** (`ddd2adbdeb`): dispatch is now
  `if _PERSISTENT and num_seqs <= _PERSIST_MAX_SEQS` (16); B>16 falls
  through to the fused (Sk≤65535) or two-kernel (Sk>65535) paths — old
  behavior, still Sk-bounded, never throws. Regression guard:
  `test_persistent_dispatch_fallback_large_batch` (B=17, PERSIST on == off
  bit-equal) + `test_persistent_gather_bit_equal_to_fused_at_batch_bound`
  (B=16 ragged, persistent bit-equal to the fused kernel). Suite 21/21.
- **Residual**: B=17..∞ configs still pay the Sk tax (old behavior).
  Follow-up: widen the register prefix to 32 (house MoE) or 64.

### P1-1 — MTP + PERSIST=ON untested

The S8 record config (39.7 t/s, `--num-speculative-tokens 2`) is mtp2;
the serving A/B that gated this change was plain greedy. MTP does not
change B or the per-seq semantics (gather stays live-seq_lens-bounded;
the FA kernel handles the multi-row verify), so the mechanism argues for
safety, but the production record config has not been re-run on this
branch. **Status: follow-up** — re-run an mtp2 A/B (or at least a smoke +
t/s check vs the 39.7 record) before re-baselining S8 numbers on this
branch.

### P1-2 — D ≠ 256 untested under default-ON

The kernel is D-generic by construction (V copies `D/16` uint4s; the K
loop guards `b < blocks_per_row`; 16 B alignment holds for D%32==0 with
2 B strides), and both house models are D=256 (MoE 35B FA: head_dim 256,
Hkv 2; dense: 256). But default-ON means any future gfx906 model with
D=128 (e.g. a Qwen3-class dense) hits this kernel with zero direct
verification. **Status: follow-up** — before widening default-ON past the
current model family, run `fa_persist_probe.py` (and ideally PPL) against
a D=128 model, or add a D=128 case to the suite test.

### P2-1 — `GFX906_FA_FUSED_QUANT=0` no longer reaches the two-kernel path

With PERSIST=1 the debug switch is bypassed (persistent wins the
dispatch). The two-kernel path is still reachable via
`GFX906_FA_PERSIST=0` (and, post-P0-fix, automatically for B>16), so the
debugging capability survives — but the interaction is undocumented.
**Status: doc note only.**

### P2-2 — LEGACY=0 (direct-paged) path is untouched

`forward_paged_direct` (the pre-quantized-`key_cache_q8` path) does not
use the persistent kernel; its gather still scales with Sk. Dormant for
the current models (they don't carry the q8 side buffer; roadmap N2/N3),
so no regression — just do not claim the N4 fix covers that path.
**Status: scope note.**

### P2-3 — long-live-context (128k) in-model performance unmeasured

Gated at live ~1.1k. The traffic math (≈465 MB/layer at live 128k →
~0.5 ms/layer bandwidth-bound at full HBM rate vs ~1 ms for P0's 262k
rows) says P1 still wins, and the kernel-level probe bounds the
mechanism, but no in-model number exists above ~1.5k live. **Status:
follow-up if the serving population is long-context-heavy** (already
noted in the devlog).

### P2-4 — performance nits (accepted)

- Per-row row→(seq,head,tok) mapping is a 32-bit divide + a B-iteration
  scan, uniform per workgroup (one wavefront on MI50) — amortized and
  latency-bound at decode shapes; measured 26.7 µs/layer at 262k
  confirms no divide-bound regime.
- Env knobs are `static` read-once (no per-launch `getenv`) — good.
- Grid 1024 is validated at every probed shape; a 512/2048 sweep is
  deferred (devlog refrigerated list).
All fine as shipped; recorded so a future perf pass knows the shape of
the headroom (it's in the memory pattern, not the control flow).

### P2-5 — probe coverage shape (accepted, with one fix applied)

- `fa_persist_probe.py` is B=1-only (bit-exact matrix); B=2 ragged is
  covered by the NaN-tail probe, B=1..4 by the capture probe, and now
  B=16 by the new suite test. Adequate for the dispatch invariants
  (B≤4 serving, B≤16 kernel bound, B>16 fallback).
- **Fix applied** (`ddd2adbdeb`): `fa_capture_replay_probe.py` sized
  `nb = SK_PAD//BLOCK + 16*max_b` — only ~4144 blocks/seq — so the
  sl=262144 replays hit the kernel's `block_tab_idx < max_blocks_per_seq`
  guard past ~66k rows: rows were *skipped identically in every path*
  (bit-equal, but never materialized), making the largest sweep point
  shallower than the docstring claimed. Now `nb = SK_PAD//BLOCK*max_b +
  16*max_b` (full materialization); re-run: B=1..4 all PASS.
- Reviewer's own slip, for the record: the two new suite tests first
  failed on my block-table arithmetic (`n_blocks + 4/+8` not divisible by
  B), not on the kernel — fixed pre-commit; the B=16 kernel result above
  is the first-run kernel result.

## Verified correct (positive evidence from the re-read)

- **Margin clamping in-kernel**: `extra = min(margin_zeros, Sk - sl)` —
  no out-of-region V write when `sl + margin > Sk` (the sl==Sk case the
  probe exercises).
- **No div-by-zero in row mapping**: a zero-width seq (`rph[s]=0`) can
  never be selected — the scan breaks only on `rem < Hkv*rph[s]`, which
  implies `rph[s] ≥ 1`.
- **Block-table OOB guarded**: `phys_block = -1 → continue` when
  `block_tab_idx ≥ max_blocks_per_seq` (production never triggers:
  `sl ≤ max_model_len < max_blocks*block_size`; the probe did, see P2-5).
- **D-generality** as in P1-2: no hardcoded 256 anywhere in the kernel.
- **K bit-equality**: same `quantize_block_q8_0_halfwarp` primitive as
  the fused/dense paths; per-32-block independence makes WG row
  partitioning irrelevant (plus probe + PPL evidence).
- **V alignment**: `uint4` copies at `v_out + ((seq*Hkv+head)*Sk + tok)*D`
  and cache-stride sums — 16 B aligned for all D%32==0 with 2 B strides.
- **Stride-flexible caches**: wrapper takes `stride(0..2)` and only
  requires a contiguous last dim — the suite tests mirror the backend's
  non-contiguous `unbind(1)` allocation, so the real allocation path is
  covered.
- **Buffer lifecycle unchanged**: ClassVar capacity + leading-dim prefix
  slice (`b[:num_seqs]` is contiguous) → `use_or_alloc` reuses; the
  i==0 sizing pass stays tied to `max_model_len` (constraint honored).
- **Capture/replay semantics**: grid frozen, `seq_lens` re-read in-kernel
  per replay — the exact FULL-graph pattern; bit-exact at B=1..4,
  sl 32→262144 (now fully materialized).
- **Test hygiene**: `_PERSISTENT` monkeypatch restored in `finally`;
  `ppl_probe.py` param keeps its old default.
- **Commit messages**: every number in the three commit bodies traces to
  a probe/devlog entry (spot-checked the 26.7 µs/84×, the 22.4/15.9/40.9
  t/s A/B, and the PPL 10.5516).

## Recommended actions (ordered)

1. ~~P0-1 guard~~ — done (`ddd2adbdeb`), suite 21/21, capture probe
   re-verified.
2. mtp2 smoke/A/B before re-baselining S8 records (P1-1).
3. D=128 probe/PPL before widening default-ON to other model families
   (P1-2).
4. Follow-up commit: widen the register prefix to B=32 (house MoE) and
   re-run the A/B arms at `--max-num-seqs 32` (P0-1 residual, P2-4).
5. Optional cleanup: `GFX906_FA_PERSIST_MARGIN=0` (NaN gate passed; 128
   is a 64 KB/head defensive default) — devlog refrigerated list.
