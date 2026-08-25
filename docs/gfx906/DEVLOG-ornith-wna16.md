# Ornith WNA16 MoE (asymmetric compressed-tensors W4A16) — the CT
# asymmetric expert checkpoint class now loads on gfx906 and runs the
# custom kernel: 65.03 t/s harness A/B (decode-only 81.1), 18.6× over
# the Triton fallback arm.

Copyright Kevin Read <me@kevin-read.com>

Model: `cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4` (Qwen3.5-MoE VLM, 40 layers,
128 experts × 8 top-k, moe_inter 512, 24.33 GiB checkpoint — the same
decode-shape class as the flagship Qwen3.5-35B-A3B-AWQ). Quant:
compressed-tensors **asymmetric** W4A16, group-32, int8 zero points,
pack-quantized, routed experts only. Branch `gfx906/moe-ct-asym-zp`
(base `gfx906/v0.28.0rc2` @ 67ae6c3f96), MI50 GPU0, 2026-08-25.

**VERDICT:** SHIPPED (branch finalized 2026-08-25, unmerged) ·
**GATE:** serving wall-clock A/B, `_bench_gfx906.py`, graph
(`BENCH_EAGER=0`), pp2048/tg256, maxlen 2816, `BENCH_MAX_SEQS=8`,
gpu_util 0.95, 4 samples/arm

## 2026-08-25 — onboarding + CT-asym W4A16 MoE support

## HYPOTHESIS

If the on-disk zps of a compressed-tensors asymmetric pack-quantized
W4A16 checkpoint are the kernel's native K-first `[E, G, N/8]` int32
layout (as the `is_transposed` MoE loader presents them), then opening
the oracle gate + a pass-through repack — no kernel change — lets
Ornith load on gfx906 and decode at flagship-class speed; if not, the
zp path in the repack or the kernel needs real work.

## What was done

Pre-change state on stock v0.28.0rc2: **Ornith cannot load at all** —
`AssertionError: Only symmetric quantization is supported for MoE`
(`compressed_tensors_moe_wna16.py:127`). CT-asymmetric MoE was
unreachable on this platform: the Marlin branch is CUDA-only, and the
non-Marlin branch asserted symmetric for every other backend.

Changes (all Python; no kernel work — CT-asym dequant `(q − zp)·scale`
is the kernel's existing AWQ-with-zp convention, `zero_offset=0`):

1. `oracle/int_wna16.py` — `_gfx906_asym_ct_reason()`: accept
   QuantizationArgs asymmetric 4-bit, static scales, GROUP strategy,
   group size ∈ {32, 128}, no g_idx actorder, for GFX906_HIP; AutoGPTQ
   still rejected ("GPTQ-style zero-point encoding is not supported").
2. `oracle/int_wna16.py` — `_repack_w4a16_gptq_kfirst_layout`: zps
   validated against `[E, G, N/8]` (wrong shape fails closed) and
   passed through bit-exact (`.to(int32).contiguous()`).
3. `Gfx906WNA16Experts._supports_quant_scheme`: accept
   `kInt4StaticAsym` / `kInt4Static32Asym`.
4. `CompressedTensorsWNA16MoEMethod.__init__`: the symmetric assert is
   scoped to zp-consuming backends (TRITON, GFX906_HIP); any other
   backend still fails closed for asymmetric.
5. `TritonWNA16Experts._supports_quant_scheme`: accept the asym keys
   (the Triton WNA16 kernel consumes `w1_zp`/`w2_zp` in both gemm
   passes) — required for the A/B baseline arm.
6. `oracle/int_wna16.py` TRITON branch of
   `convert_to_wna16_moe_kernel_format` — `_repack_qzeros_kfirst_for_triton`:
   the passthrough read a **transposed** layout (garbage text before
   the fix); the kernel indexes column n from word `n // 2` (axis 1),
   nibble `(n % 2) * 4`, group `g` (axis 2) — i.e. `[E, N/2, G]`,
   2 zps per int32. Repack converts the checkpoint's 8-zp-per-word
   K-first packing; result stored physically `[E, G, N/2]` and returned
   as a transposed view (coalesced along the kernel's n-walk).
7. Probes: `BENCH_MODEL` + `BENCH_MOE_BACKEND` env support in
   `benchmarks/kernels/gfx906/ppl_probe.py` and `greedy_probe.py`.

One load-time GPU wedge on the first graph-mode launch of the boot
(22:19 UTC, GPU0, BACO reset, self-recovered) — recorded in
`degradation.md`/`degradation_details.md`; retried once per house
recipe, clean.

## Evidence FOR

- Unit: oracle suite 34/34; GPU kernel suite 51/51 (incl. 8 new
  `gptq_kfirst` E2E asym cases + repack pass-through/shape-failure
  tests).
- Smoke (eager, max_len 4096, gpu_util 0.95): coherent text on both
  arms. gfx906: "Paris. Paris is located in the north-central part of
  the country, on the Seine River…" triton pre-repack: garbage
  (`::::::::ALES::::::::对付::::::zi`). KV 123,699 tokens @4096.
- PPL probe (12 prompts, 359 tokens, 0 top-20 misses, eager):
  gfx906 **16.6716** / **16.6876** (run-to-run 0.016); triton
  **16.4539**. Inter-arm 0.218 ≈ fp16-noise-level divergence between
  the two kernel implementations (cf. the gemma-4 ΔLP note), not a
  correctness failure; both arms coherent.
- **GATE — serving A/B (graph, config above):** gfx906 arm
  **65.079 / 65.063 / 64.999 / 64.995 t/s** (mean 65.03, band 0.08 %);
  triton arm **3.498 / 3.501 / 3.500 / 3.500** (mean 3.50). Ratio
  **18.6×**.
- TTFTI probe (same graph config, 2048-prefill + 256-decode split):
  gfx906 TTFT **0.77 s**, decode **12.3 ms/tok = 81.1 t/s**; triton
  TTFT **4.38 s**, decode **267.6 ms/tok = 3.7 t/s**. Decode-only A/B
  **21.9×**; prefill 5.7×.
- Class parity: 65.03 harness vs the flagship's 67.39 record — the
  asym-zp path performs on par with the symmetric path on identical
  decode shapes.

## Evidence AGAINST

None against the fix itself. The Triton arm's slowness (Entry 2) is a
property of the upstream Triton kernel's `has_zp` branch, not of this
change — and that path was unreachable before this work.

## Why it worked

The loader already presents the checkpoint zps in the kernel's native
layout, so the gate + pass-through is sufficient; the only real
layout work was for the Triton baseline arm (Entry 2).

## Interactions / superseded-by

- The Triton int4-zp path is now a *reachable* (was: assert-failed)
  fallback for CT-asym checkpoints on ROCm; its gfx906 slowness is
  tracked below and cross-linked to `moe-decode-roadmap.md` §8 (U4).
- Precedent: gemma-4 no-zp CT W4A16 onboarding (`DEVLOG-gemma4-*.md`,
  1.79× over Triton) — same oracle/repack machinery, symmetric branch.

VERDICT: SHIPPED

## 2026-08-25 — Triton W4A16 int4 `has_zp` branch is pathologically slow on gfx906

## HYPOTHESIS

The Triton arm of the A/B is a fair no-zp-equivalent baseline — i.e.
within ~2× of the no-zp class (gemma-4 Triton baseline 37.8 t/s).

## What was done

Measured the A/B baseline arm; when it read 3.50 t/s (harness) / 3.7
t/s (decode-only), decomposed with a TTFT/decode probe and a 9B dense
control (`cyankiwi/Qwen3.5-9B-AWQ-INT8-INT4`, same GDN-Triton-fallback
stack, no MoE, same boot/config).

## Evidence AGAINST

- 9B dense control (no MoE): decode **13.7 ms/tok = 73.2 t/s**,
  TTFT 0.61 s → the stack, GDN Triton fallback, and host state are
  healthy on this boot (the canary concern was refuted by the
  healthy dense control + the 81.1 t/s gfx906 arm).
- The excess over the control concentrates in the 40 MoE blocks:
  ~6 ms/block in the Triton zp path vs ~0.2 ms/block class.
- **Layout-independent:** the repack was measured with both physical
  zp layouts (contiguous `[E, N/2, G]` and transposed-view
  `[E, G, N/2]`) — identical 267–270 ms/tok. Not a storage-layout
  artifact of this work.
- Values are correct (PPL 16.4539, coherent text) — pure speed
  pathology in the `has_zp` branch of `fused_moe_kernel_gptq_awq` on
  gfx906. Suspect: the per-element data-dependent shift
  `(b_zp >> (offs_bn % 2) * 4) & 0xF` over `[BLOCK_K, BLOCK_N]` blocks
  on triton-hip/CDNA1 — **unmeasured mechanism**.
- Prefill affected too (TTFT 4.38 s vs 0.77 s) — consistent with the
  same branch at large M.
- Not a regression: the path was unreachable before this work (the CT
  scheme asserted symmetric), so this is new visibility.

## Why it failed

Upstream Triton kernel characteristic on gfx906; out of scope for the
backend fix (the whole point of the work is the fast custom kernel,
which the oracle selects by default).

Refrigerated residue: a standalone microbench of
`fused_moe_kernel_gptq_awq` with `has_zp` True/False at M=64
(128 experts, w13 N=1024/K=2048) would pin the mechanism in minutes —
worth doing only if the Triton zp path ever becomes a supported
fallback on gfx906.

VERDICT: OPEN
