# MTP2 regression diagnosis — post-merge build (Qwen3.8-27B, gfx906) — OPEN, narrowed

Copyright Kevin Read <me@kevin-read.com>

Status: OPEN (read-only diagnosis). · 2026-08-22
Build: `v0.28.0rc2.dev318+gfed58511` (branch `gfx906/fa-masked-gather`)
Blocks: S8 mtp2 re-baseline gate (`DEVLOG-masked-fa.md` post-commit 3).

## VERDICT (first pass)

Not an N4/persistent-gather artifact (gather = 0.44 ms/step; P0 shows it
too). Not lost cudagraph coverage (3-token verify hits the captured
size-3 FULL graph). Not GDN (~4.5 ms/step). Not FA (~2.7 ms/step). The
step is GPU-kernel-bound; dominant kernels are the **4-bit GEMM/GEMV
family**. The regression decomposes:

| basis (tax-free, acceptance 3.0) | pre-merge | post-merge | Δ |
|---|---|---|---|
| TP=1 mtp2 (in-process, 32k, P1) | ≈39.4 (spec-decode branch record) | **36.75 t/s** (82 ms/step) | −7% (≈ noise; 1.45× plain-TP1 vs 1.56× pre-merge) |
| TP=2 mtp2 (serving, 131k, P1) | 39.9 t/s (S8 — was paying N4 tax) | **24.90 t/s** (120.5 ms/step) | **−38% headline; ≈ −35% vs TP=1 parity** |

Pre-merge mtp2 had TP=1/TP=2 **parity** (S7: 39.7–39.9). Post-merge TP=1
is where it should be (3 tokens/step at weight-bound GEMM ≈ 1.45×
plain-TP1 25.3 t/s — consistent with M=3 GEMM ~1.2× M=1), while **TP=2
lost ≈12 t/s that TP=1 did not**. Plain greedy is fine at TP=2 (40.86
t/s P1). The pathology is **TP=2 × MTP-specific**.

## Established facts

- **F1 — Reproduces in-process at TP=1: 36.75 t/s** (248 tokens, 32k,
  util 0.95, graphs). First short window (27 steps) read 23.4 t/s —
  one-time Triton JIT spikes of the spec-path kernels
  (`precopy_mamba_align_fused_kernel`, `postprocess_mamba_fused_kernel`,
  `eagle_prepare_{inputs,next_token}_padded_kernel`,
  `eagle_step_slot_mapping_metadata_kernel`) during warmup; steady state
  is 36.75.
- **F2 — Step cost:** TP=2 mtp2 = 120.5 ms/step vs plain 24.5 ms. Linear
  3-token expectation ≈ 73.5 + draft ≈ 78–84 ms → **≈40–45 ms/step of
  excess at TP=2 only** (TP=1 = 82 ms ≈ expected).
- **F3 — GPU-kernel-bound.** torch profiler (TP=1, 83 steps): total
  self-CUDA ≈ 116 ms/step ≈ wall/step; no CPU gap. Caveat: under FULL-
  graph replay, op-level entries (`_C::gptq_gemm` 18.3, `aten::mm`,
  `aten::copy_`, `ChunkGatedDeltaRuleFunction` 1.1) may carry
  misattributed kernel time — read per-line as a RANKING, not an exact
  sum; kernel-leaf names are trustworthy.
- **F4 — Dominant kernels (TP=1, ms/step, kernel leaves):**
  | kernel | ms | note |
  |---|---|---|
  | `gemm_half_q_half_gptq_4bit_kernel<true,3>` | 33.6 | M=3 4-bit GEMM (target weights) |
  | `Cijk MT128x64x32` + `MT64x32x16` (rocBLAS) | 16.8 | fp16 GEMMs — draft MTP layer / combo-kernel fallback? |
  | `LLGemm1_kernel` + `LLMM1` (compressed-tensors LL) | 15.0 | 4-bit LL path |
  | `dense_gemv_m_kernel<2,1024,3>` + `dense_gemv_m4_gfx906` | 11.7 | M=3/M=4 gfx906 gemv |
  | `flash_attn_tile_q8<256,256,{4,64},1>` | 1.8 | 16 FA layers, sq=3 |
  | GDN: `fused_sigmoid_gating_delta_rule_update` 1.2 + chunk residual ~2.5 + conv 0.33 | ~4.5 | spec kernel IS the one running |
  | `gather_paged_kv_quant_persistent_kernel` | 0.44 | **16 calls/step = 1 per FA layer (not 3×); N4 path fine** |
  GEMM/GEMV family ≈ 77–95 ms of ~116 ms.
- **F5 — Graph coverage intact.** mtp2 startup: `Capturing CUDA graphs
  (decode, FULL): 1/1 (largest=3)`; BatchDescriptor(3, num_reqs=1,
  uniform=True) is a captured size; both FA and GDN builders declare
  `UNIFORM_BATCH` CG support (unchanged since v0.11).
- **F6 — Merge diff archaeology (v0.27.2rc0→v0.28.0rc1):** every
  GDN/MTP/spec decode change is gated INACTIVE for this model — fused-
  CUDA GDN decode needs BF16+SM80+ (we're fp16 gfx906); MTP MoE-TP
  plumbing is no-op for dense; gdn_attn metadata refactor is prefill-
  only. The MTP draft model (1-layer Qwen3_5MultiTokenPredictor,
  embedding+lm_head tied to target, fp16) runs `propose()` = 2 draft
  forwards (M=3 then M=1) + lm_head/argmax.

## Open sub-questions

- **S1 — TP=2-specific excess (≈40–45 ms/step).** Suspects, in order:
  1. **Drafter runs EAGER at TP=2**: the capture log shows ONE graph
     section (the target model). If the two draft passes (~10–20 kernel
     launches each + TP=2 collectives) run eager, S5's launch-overhead
     collapse (eager TP=2 → 7 t/s full model) scaled to 1/64 of a model
     could be ~30–60 ms/step. Check drafter capture + a TP=2 profile.
  2. **Spec-path CPU work × 2 workers**: `llm_base_proposer.py`
     builds CPU tensors per pass (`query_start_loc_cpu`,
     `num_accepted_tokens.cpu()`-class syncs, `spec_token_indices`
     bookkeeping); in serving each worker does this per pass with
     process-boundary serialization.
  3. **RCCL pattern change in verify/draft** (per-GEMM allreduce ×3
     passes; lm_head is vocab-parallel + local argmax, so that's not it).
- **S2 — mtp2 P0−P1 delta anomaly:** at 131k, mtp2 P0−P1 = 181.8−120.5
  = **61.3 ms/step** vs plain P0−P1 = **20.2 ms/step** — ≈3× at the
  same max_model_len, though P1's gather runs 16 calls/step (not 3×).
  P0/P1 differ only in the gather kernel, so either the two-kernel is
  ~3× slower in the SPEC context, or the mtp2 P0 "steady" number is
  contaminated. Affects the pre-merge tax-adjusted baseline (39.9 was
  paying this tax) but NOT the P1 regression. Re-verify with per-arm
  acceptance + a P0 profile.
- **S3 — M=3 GEMM efficiency (TP-common, mild):** 33.6 ms for the M=3
  gptq 4-bit kernel; whether pre-merge used a different dispatch for
  M∈{2,3} (or inductor `combo_kernels` — new in this build's compile
  config — picks bad tiles at M=3). Compare against a plain M=1 profile
  (rerun — first plain-profile attempt died rc=134, known flake family).

## Next probes (ordered)

1. **TP=2 mtp2 chrome trace** (serving `start_profile` endpoint, B=1,
   200 out) → attribute the ≈40–45 ms: draft-pass kernel launches /
   RCCL / CPU gaps. (Decisive for S1.)
2. **Rerun plain TP=1 profile** (rc=134 flake) → M=1 GEMM table; direct
   M=1 vs M=3 per-kernel delta (S3).
3. **mtp1 run** (2 tokens/step): per-token GEMM scaling point; if mtp1
   ≈ 3×-linear-expected, S3 is minor and S1 dominates.
4. **Drafter graph status**: grep serving log for drafter graph capture;
   if absent, test forcing drafter capture (or eager→graph) as the fix.

## Notes

- Plain greedy unaffected: 40.86 t/s (TP=2 P1) — S8 plain re-baseline
  proceeds independently.
- `fa-masked-gather-code-rev-*.md` (parallel reviewers) + this file are
  uncommitted working-tree files; commit only at a natural boundary.
- Profiler caveats (F3): op-level entries under graph replay are
  approximate; kernel-leaf entries + total self-CUDA are the solid
  numbers.

VERDICT: OPEN — root cause narrowed to TP=2 × MTP-specific step
overhead (~40–45 ms/step beyond 3-token expectation), GEMM/GEMV family
dominant in the kernel mix; GDN/FA/N4/graph-coverage exonerated. S1
(drafter-eager / spec-path CPU ×2 workers) is the leading hypothesis;
TP=2 trace is the decisive next probe.
