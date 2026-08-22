# Critical review — `gfx906/fa-masked-gather` (persistent live-bounded gather+quantize)

Copyright Kevin Read <me@kevin-read.com>

Review of the two code-carrying commits on `gfx906/fa-masked-gather`:

- `49d4876e72` feat(gfx906): persistent live-bounded gather+quantize — the N4 fix
- `907ec900f3` test(gfx906): N4 gate probes + capture/replay suite test

Reviewed against the surrounding code (`gfx906_fa_gather.cu`, `gfx906_fa.cpp`,
`gfx906_fa_paged.py`, `gfx906_fa_backend.py`), the plan
(`docs/gfx906/plan_masked_fa.md`), and the gate record
(`docs/gfx906/DEVLOG-masked-fa.md`).

---

## Bottom line

This is a well-diagnosed, correctly-engineered fix that removes a real cost. I
**largely accept it**, but there is **one hard correctness/robustness regression I
consider blocking**: the new kernel hard-errors on `num_seqs > 16`, a limit that
did not exist in the paths it replaces and is never auto-escaped. Everything else
is minor-to-moderate. The change should not go in with `GFX906_FA_PERSIST` **ON
by default** until the >16 batch boundary is either supported or auto-falls back.

---

## What the change does well (validated)

1. **Diagnosis is sound and triple-verified.** The mechanism (two-kernel fallback
   at `Sk > 65535`: V2 gather + a `quantize_q8_0` pass that reads/writes the full
   `Sk_pad` because it has *no `seq_lens` parameter at all*) is correct. The
   cost of the selected fix — bounding work by the live `seq_lens` tensor read
   inside the kernel rather than by the frozen `Sk_pad` launch dim — is the same
   live-bounding pattern the FA compute kernel already uses via `kv_max`, which is
   confirmed working and live-refreshed. Good prior-art reuse.

2. **The design is legitimately capture-safe.** A fixed 1024-workgroup grid baked
   at capture, with all length-dependent work moved inside the kernel body, is the
   correct approach under vLLM's stream-capture mechanics. The "grid must be a
   capture-time constant" premise is proved correct by the `gridDim.z` cap that
   forced the original V1→V2→two-kernel ladder. No conditional nodes (absent in
   ROCm 7.14) needed. One code path at every `Sk_`.

3. **The bit-equal-K claim is well-founded.** `quantize_block_q8_0_halfwarp`
   quantizes each independent 32-value block, so partitioning rows across
   workgroups cannot change the result — bit-equality with the dense/fused paths
   holds by construction, not by luck.

4. **The NaN/Inf tail gate is the right correctness instrument** and was run
   *before* relying on tail-write removal. It directly establishes that the FA
   kernel never reads rows at/beyond `kv_max`, which is the load-bearing
   precondition for the "don't write tail rows" optimization. Well-constructed
   (aligned/misaligned tail tiles, 1-row-before-`Sk`, ragged B=2, GQA-packed
   `Hkv=2/4`).

5. **Gates are appropriately heavy and ordered:** NaN-tail → standalone
   bit-exact+timing → capture/replay (B=1..4, incl. `sk-32`/`sk`) → suite test
   → PPL A/B → serving A/B. The serving A/B (THE gate) is a genuine controlled
   experiment (3 reps, PERSIST=0 vs =1, both 131k and 262k) with strong,
   reproducible results (+83% / +157%, P1 residual tax 0.07% = noise).

6. **Rollback path retained.** `GFX906_FA_PERSIST=0` returns to the prior
   behavior, so the feature is reversible without a revert. Good.

7. **Capture-safety machinery correctly reused.** The gather buffers keep a single
   base VA across the capture sweep (`_ensure_gather_buffers` leading-dim slice,
   retired-generation retention); the new kernel sources/outputs ride that existing
   infrastructure, so the earlier use-after-free class is re-covered.

---

## Findings

### [BLOCKING] F1 — Hard `num_seqs <= 16` cap with no auto-fallback

- `gather_paged_kv_quant_persistent` declares a stack `int rph[16]` and the
  launcher returns `hipErrorInvalidValue` for `num_seqs > 16`; the pybind layer
  adds a `TORCH_CHECK(num_seqs <= 16, ...)`.
- **Neither the V2 two-kernel fallback nor the existing fused
  `gather_paged_kv_quant_kernel` had any batch cap** — their grids scale with
  `num_seqs` (e.g. `dim3 grid(num_seqs, num_kv_heads, Sk)`). So this is a
  **new functional regression for decode batches > 16**, silently converting what
  previously worked into a runtime `RuntimeError`.
- This matters concretely here: the shared MoE 35B bench/serving workload
  (`/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`) runs at `BENCH_MAX_SEQS=32`
  (AGENTS.md). With `GFX906_FA_PERSIST` **ON by default**, a 32-way decode
  batch on that model now raises. The gate evidence is all `B <= 4`
  (capture probe B=1..4, nantail B=1/2, serving A/B with capture sizes
  [1,2,3,4]), so the 5–16 range is also *untested*, not just >16 unsupported.
- **Recommendation (minimum):** keep the old two-kernel path reachable and
  dispatch to it automatically when `num_seqs > 16` (the batch count is a live
  value that *can* drive Python-level dispatch; only the inside-the-graph grid
  must stay frozen). Since FULL-graph capture bakes grid per capture, a >16
  capture is simply shunted to eager/fallback inside the existing dispatcher, or
  the persistent kernel is only selected for the capture-size set actually used.
  Do not ship default-ON without this, and document the cap loudly either way.

### [MODERATE] F2 — `GFX906_FA_FUSED_QUANT=0` is now a dead knob under the default

- Dispatch order is `if _PERSISTENT: ... elif _FUSED_QUANT and Sk <= 65535:
  ... else: two-kernel`. With `_PERSISTENT` defaulting ON, setting
  `GFX906_FA_FUSED_QUANT=0` (documented as "reverts to the two-kernel path")
  is silently ignored. The two-kernel fallback you might think `FUSED_QUANT=0`
  selects is only reachable with `PERSIST=0` too. This is a configuration
  semantics inconsistency that should be reconciled (make FUSED_QUANT defeat the
  persistent branch, or drop FUSED_QUANT's now-misleading doc).

### [MODERATE] F3 — In-model torch reference double-check does not cover the persistent path

- `_DOUBLE_CHECK` (compares against the torch `_gather_kv_q8` reference) only
  exists in the direct-paged / `key_cache_q8` branch; the legacy `else` branch —
  where the persistent kernel actually runs — has no in-model reference check.
  Correctness rests entirely on the standalone probes + PPL + serving A/B. For a
  hand-tuned kernel this is a weak in-model safety net; an env-gated
  torch-reference cross-check on the persistent branch would materially harden it.

### [MODERATE] F4 — Gate geometry coverage narrower than the supported surface

- The NaN-tail gate (the load-bearing precondition) was exercised only at `D=256`.
  `supports_head_size` advertises 64/128/256, and the kernel handles all three
  (V `uint4` copy `v_n_u4 ∈ {8,16,32}`; K `blocks_per_row ∈ {2,4,8}`). The
  "FA never reads ≥ kv_max" and the `margin`-only-zeroing behavior are only
  *measured* at D=256. At D=64/128 the margin default (128) is conservative and
  thus safe by construction, so this is not a latent bug — but the gate's
  conclusion is over-generalized in the docs relative to what was measured. State
  the D=256 scoping, or add a D=128 nantail case.

### [MINOR] F5 — Per-launch vs capture-frozen knob asymmetry

- `GFX906_FA_PERSIST_GRID` is read once at process start (a static; genuinely
  capture-frozen), but `GFX906_FA_PERSIST_MARGIN` is re-read on **every**
  launch and passed as a live kernel arg. If the intent is "capture-time
  constants," margin is inconsistent (though harmless: changing it only affects
  how many V-zero rows are written, and the FA kernel cuts at `kv_max` anyway).
  Worth a comment so a future reader doesn't treat margin as baked.

### [MINOR] F6 — Margin "128 = max FA tail-tile width" is a hardcoded empirical constant

- `get_fa_persist_margin()` default 128 is hardcoded and justified by "max D=256
  `nbatch_fa`." It is not derived from any kernel constant at build/source level,
  so if the FA tail-tile width ever changes, this defensive headroom silently
  becomes stale. Given the NaN gate passed, margin is belt-and-braces; either
  derive it from the tile constant or drop the "max tail-tile width" justification
  and document it as a purely conservative safety margin.

### [MINOR] F7 — Loose two-kernel equality tolerance in the suite test

- `test_persistent_gather_capture_replay_large_sk` asserts bit-exactness vs eager
  persistent (`e_persist == 0.0`) — good, that's the strong check — but only a
  2e-2 relative error vs the two-kernel fallback. The function/message says it
  "matches the two-kernel fallback," which overstates what a 2e-2 tolerance
  proves. Acceptable (different code path, exact equality not expected), but the
  name/docstring promise more than the assertion delivers. The bit-exact-vs-eager
  assertion is the one carrying correctness.

### [MINOR] F8 — Process/plan deviation, acknowledged

- Plan §3.5 asked to extend `bench_gfx906_fa_gather.py` with a micro-bench
  matrix; it was not modified (last touched `1691d1dd29`, unbranched). The
  standalone probes (`fa_persist_probe.py`) substitute, and the plan explicitly
  allows the substitution, so this is cosmetic — but the plan's step 1 kernel-trace
  (`rocprofv3`) is **still outstanding** and the devlog is candid that the serving
  A/B substitutes for it. Fine, prefer honesty here; just keep N4 "RESOLVED" from
  reading as "everything in the plan ran," because the block trace did not.

### [NIT] F9 — Code style

- `hipLaunchKernelGGL(( gather_...))` has an odd double-open-paren and the
  argument list is poorly indented/jammed; cosmetic. The `#pragma unroll 4`
  over a runtime-trip `B` loop is fine but the magic 4 appears un-coupled from
  anything. No functional impact.

### [NIT] F10 — Kernel self-consistency notes checked and found correct

- V copy alignment is safe: both source and dest addresses are multiples of `D`
  `__half`s (≥ 128 B, 16-B aligned) → `uint4` misalignment is impossible for
  D ∈ {64,128,256}.
- The row→(seq,head,tok) flatten is OOB-safe: valid rows have `tok < sl`, so
  `block_tab_idx < ceil(sl/block_size) <= max_blocks_per_seq`; `sl=0` pad
  sequences produce only margin/V-zero rows and never read `block_table`. The
  `__shfl(phys_block, 0, 64)` broadcast is correct for the 64-thread block.
- Control flow is uniform within each 64-thread wavefront (all lanes share `row`),
  so the margin branch and `continue` introduce no intra-wavefront divergence.

---

## Validity risk assessment of the evidence

- The greedy-hash divergence between P0/P1 (3/12 prompts) was properly chased down
  with controls (P0-vs-P0 also diverges; `batch=1` nondeterministic) and
  attributed to GDN decode nondeterminism of this hybrid model, not the gather. The
  reasoning is sound, and the PPL gate (10.5516 both arms) plus the
  bit-exact-vs-eager assertions carry decode correctness. This is not a flaw in the
  change. Minor residual: 12 prompts is a small PPL corpus, and decode correctness
  is not bit-for-bit proven — acceptable for the gfx906 house protocol.

---

## Recommended action list (before merge to `gfx906/main`)

| # | Severity | Action |
|---|----------|--------|
| 1 | **BLOCKING** | Auto-fallback to the two-kernel path (or refuse-with-clear-error + doc) for `num_seqs > 16`; do not ship `PERSIST` default-ON without it. Validate B ∈ {8, 16} at least. |
| 2 | Moderate | Reconcile `GFX906_FA_FUSED_QUANT=0` with `PERSIST` (or fix its doc). |
| 3 | Moderate | Add an env-gated torch-reference double-check to the persistent backend path. |
| 4 | Moderate | Document the D=256 scoping of the NaN-tail gate (or add a D=128 case). |
| 5–10 | Minor/nit | Margin-derived-from-constant, marg/asym comment, test-docstring accuracy, plan trace-status honesty, style fixes — all non-blocking polish. |

---

## Verdict

**Conditional SHIP.** The mechanism, design, and gate record are strong and honest,
and the serving A/B establishes a large, reproducible win. Blocking on F1: the
>16-batch hard cap is a real functional regression on the shared MoE workload and is
completely uncovered by the (B≤4) evidence. With the fallback (or a loudly
documented + validated cap) in place, and F2–F4 addressed at reasonable effort,
this is a solid, well-documented contribution.
