# Nemotron-H onboarding (Nemotron-3.5-Lightning-30B-A3B mixed INT4/INT8) — serves at 59.4 tok/s on one MI50 after three gfx906 fixes

> Branch `gfx906/nemotron-h-onboard` off `main` (97a6fbe11a) · model
> `primitive-ai/Nemotron-3.5-Lightning-30B-A3B-mixed-INT4-INT8` (NFS HF
> cache snapshot `1a1521fb`) · 2026-08-29 · roadmap item `NH-1`–`NH-3`.

Model: 52 layers = 23 mamba2 + 23 MoE (E=128, topk 6, 1 shared expert,
non-gated relu² experts, sigmoid+bias router) + 6 GQA attention (2 KV
heads). compressed-tensors mixed: routed+shared experts INT4 **group-64**
symmetric; mamba in/out_proj, qkvo, lm_head INT8 **channel-wise**; MTP
head BF16 (unused). 19.2 GiB on disk, 3B active.

---

## 2026-08-29 — bring-up: three blockers, all fixed on this branch

**VERDICT:** SHIPPED (decode path) · **GATE:** greedy two-prompt smoke +
PPL probe A/B (365 tokens, 0 top-20 misses) + `_bench_gfx906.py` graph
serving below.

### Blocker 1 — fp32 router gate crashed `ops.LLMM1`

`GateLinear(force_fp32_compute=True)` stores fp32 weights on ROCm (no
specialized tiers apply) → tier-6 `F.linear` fp32 → the fork's
`rocm_unquantized_gemm` tiny-M interception fed fp32 into LLMM1, whose
TORCH_CHECK only accepts fp16/bf16. First MoE layer crashed the engine.

Fix: `_llmm1_tiny_m` (vllm/model_executor/layers/utils.py) returns None
for non-fp16/bf16 operands; callers fall back to `triton_matmul` /
`torch.nn.functional.linear`. Generic correctness fix — an unquantized
fp32 GEMM must not enter the fp16 skinny-kernel family.

### Blocker 2 — mamba2 SSD chunk-scan does not compile on triton-gfx906

`_chunk_scan_fwd_kernel` with `HAS_INITSTATES=True` aborts the triton
compile worker:

```
TritonAMDGPUTransforms/CanonicalizePointers.cpp:1430:
Assertion `(fatPtrs.at({thenFatPtrBase,...}) == fatPtrs.at({elseFatPtrBase,...}))
&& "expected then fat ptr canNarrow and else fat ptr canNarrow to be equal"' failed
```

Mechanism: the runtime `scf.if` selecting the previous-state source
yielded **two different base pointers** (`initstates_ptr` vs
`states_ptr`) plus per-branch strides; the fat-pointer narrowing pass
cannot merge that. `HAS_INITSTATES=False` (constexpr-folded branch)
compiled fine — the shape of the clue.

- `AMDGCN_USE_BUFFER_OPS=0` works around it globally (repro
  `/tmp/repro_ssd_chunk_scan.py`, both variants OK) but disables buffer
  ops for *every* triton kernel — not acceptable as a serving default.
- **Shipped fix:** restructure `ssd_chunk_scan.py` so each branch loads
  its own tile and the `scf.if` yields the *tile*, never a pointer
  (both the `BLOCK_SIZE_DSTATE <= 128` fast path and the k-loop path).
  Zero double loads; addresses are branch-local.
- Gate: `tests/kernels/mamba/test_mamba_ssm_ssd.py` — **94/94 passed**
  against the `ssd_minimal_discrete` reference (initial-states variants
  included). Engine multi-chunk prefill (init-states path) verified by
  the smoke tests.

This is a triton-gfx906 (3.6.x base) compiler limitation, not a vLLM
bug; the kernel-side restructure avoids depending on a triton bump.
Worth reporting upstream against their fork regardless.

### Blocker 3 — INT8-channel dense layers landed on Conch at 3.8 ms/GEMV

`choose_mp_linear_kernel` on ROCm has no kernel for channel-wise 8-bit
except `ConchLinearKernel` (installed in the venv). It "worked" — and
cost **3.79 ms per M=1 GEMV** (46 dense projections per decode step =
75% of GPU time; 4.95 tok/s total). Profile:
`/tmp/nemotron_prof.log` (eager, 24-step window).

Fix: new scheme `CompressedTensorsW8A16ChannelDequant`
(compressed_tensors/schemes/compressed_tensors_w8a16_channel_dequant.py)
— on gfx906 only, symmetric pack-quantized int8-channel dense layers
dequantize to fp16 at load (bias-128 convention, [N, K/4] i32 →
[N, K] fp16, chunked over N; packed tensors freed) and run the
optimized unquantized GEMV family via `dispatch_unquantized_gemm`.
Selection wired in `compressed_tensors.py` `_get_scheme_from_parts`.

Cost: +1.8 GiB VRAM (18.05 → 19.87 GiB weights+non-torch at util 0.90).
Gate below.

### Evidence (gates)

| config | result |
|---|---|
| greedy smoke (short + 300-tok multi-chunk) | coherent, `Paris.` ✓ (all arms) |
| PPL fp16 + gfx906 MoE + conch dense | 27.0216 |
| PPL fp16 + triton MoE + conch dense | 27.0026 |
| PPL bf16 + triton MoE + conch dense | 27.0026 |
| **PPL fp16 + gfx906 MoE + dequant dense (shipped)** | **26.9555** |

All within ±0.25% — noise band; the dequant path is exact int8→fp16.

---

## 2026-08-29 — MoE group-64 on the custom gfx906 W4A16 kernel

**VERDICT:** SHIPPED · **GATE:** serving A/B, graph mode, pp=2048/tg=256,
4 samples, same boot, `_bench_gfx906.py`, GPU0, util 0.95.

## HYPOTHESIS

If the oracle's 32/128 group gate is widened to any positive multiple of
32 (the kernel derives `groupsize = K/groups` arithmetically and the CT
repack passes `[E, G, N]` scales through unchanged), the Nemotron g64
experts run on the custom kernel bit-compatibly with the Triton WNA16
fallback, and the serving gain matches the historical Qwen g128 gap.

## What was done

- `oracle/int_wna16.py` `_gfx906_no_zp_reason`: (32, 128) → positive
  multiple of 32. Asymmetric gate left at 32/128 (no asym-g64 checkpoint
  to validate).
- `Gfx906WNA16Experts._supports_activation`: + `RELU2_NO_MUL` (kernel is
  activation-agnostic; `apply` already routes non-gated activations
  through `apply_moe_activation`). Without this the oracle silently fell
  to Triton even with the gate open.
- Tests: `test_gfx906_moe_gemm.py` + Nemotron shapes
  (1856/2688 × 2688/1856, gs 64, identity inter-GEMM for the non-gated
  layout) → 63 passed; `test_moe_wna16.py` gfx906 oracle gates → 24
  passed (g64 moved to the accept list, g48 is the new rejection case).

## Evidence — FOR

- PPL: 26.9555 (gfx906) vs 27.0026 (triton) — Δ0.17%, accumulation-order
  noise, same magnitude as the historical g128 A/B.
- **Serving A/B (the gate):** gfx906 **59.44 tok/s** (59.42–59.46) vs
  triton **31.48 tok/s** → **+88.8%**. Per-kernel census (launch-regime
  evidence): 38.4 µs/gemm × 46 gemms/step = 1.77 ms/step on the custom
  kernel.

## Evidence — AGAINST

None at the gate. Standalone numbers not measured (shape covered by the
unit tests instead).

## Interactions

- Reinforces: dispatch by weight format + shape, not model family — the
  only model-specific fact that mattered was relu²-no-mul.
- The `_supports_activation` gap is the second time a silent oracle
  fallthrough (not a crash) hid the custom kernel; cf. Gemma-4 no-zp
  (180f030ee3). Lesson: when a CT model "runs slow", grep the backend
  selection log line first.

## Refrigerated residue

- Asymmetric CT g64 widening: one-line gate change once a checkpoint
  exists to validate against.
- `moe_backend` override made it easy to A/B — keep using
  `BENCH_MOE_BACKEND=triton` as the control arm for future expert-kernel
  work on this model.

---

## 2026-08-29 — W8A16-channel dequant vs Conch (dense INT8 path)

**VERDICT:** SHIPPED · **GATE:** same serving A/B harness as above.

## HYPOTHESIS

If Conch's 3.79 ms M=1 GEMV is structural (generic triton, untuned for
MI50 skinny shapes), dequantizing the 71 int8-channel tensors to fp16 at
load and running the fork's GEMV family must recover decode throughput
at modest VRAM cost.

## What was done

New `CompressedTensorsW8A16ChannelDequant` scheme (see blocker 3).
Control arm: same tree with the scheme disabled (conch selected).

## Evidence — FOR

- **Serving (the gate):** 4.95 tok/s (conch) → **59.44 tok/s**
  (dequant) — **12.0×**, same MoE backend both arms.
- PPL: 26.9555 vs 27.0216 (conch arm) — dequant marginally *better*
  (exact dequant vs in-kernel rounding), both in the noise band.
- VRAM: weights+non-torch 18.05 → 19.87 GiB (+1.8); KV pool still
  ~8.2 GiB at util 0.90 / maxlen 8k.

## Evidence — AGAINST

- Decode reads 2× the weight bytes vs a native int8 GEMV: LLMM1 family
  is now 3.57 ms/step (57 × 62 µs, ≈890 GB/s — near the HBM floor for
  fp16, so the *only* way further down is an int8 GEMV, roadmap NH-2).
- Prefill pays a full fp16 GEMM on dequantized weights — measured TTFT
  impact not isolated; total pp2048+tg256 wall 4.31 s/sample.

## Interactions

- Removes the last Conch dependency from this model's serving path.
- Same pattern applies to any future CT int8-channel dense checkpoint on
  gfx906 (selection is platform+format-gated, not model-gated).

## Refrigerated residue

- `AMDGCN_USE_BUFFER_OPS=0` (global triton knob) also unblocks the
  chunk-scan compile; rejected as a default, kept as a diagnostic.
- Exllama channel-wise acceptance (groups=1 via `group_size=K`) was
  drafted and dropped: M>32 reconstructs per call (dequant + hipBLAS per
  invocation) would tax prefill; the fp16-dequant path dominates it.

---

## Search keys

`HYPOTHESIS:` `VERDICT:` Nemotron, nemotron_h, group-64, g64, relu2,
RELU2_NO_MUL, W8A16 channel, int8 channel, dequant, Conch, ConchLinearKernel,
CanonicalizePointers, HAS_INITSTATES, chunk_scan, ssd_chunk_scan, fp32 router,
LLMM1, GateLinear, force_fp32_compute, AMDGCN_USE_BUFFER_OPS.
