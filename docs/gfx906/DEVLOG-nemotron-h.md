# Nemotron-H onboarding (Nemotron-3.5-Lightning-30B-A3B mixed INT4/INT8) — serves at 70.4 tok/s on one MI50 after five gfx906 fixes

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

## 2026-08-29 — fp32 router-gate GEMV on hipBLAS sgemv (NH-3)

**VERDICT:** SHIPPED · **GATE:** serving A/B, graph mode, pp=2048/tg256,
4 samples, same boot, plus the PPL probe.

## HYPOTHESIS

If the 128 µs fp32 triton matmul per router gate is a skinny-shape
artifact (1.3 MB weight read ⇒ ~10 µs at HBM speed), routing the M=1
fp32 GEMV to hipBLAS sgemv (`torch.mv`) must cut ~3 ms/step and show
≥ +10% in serving.

## What was done

`rocm_unquantized_gemm_impl` (vllm/model_executor/layers/utils.py):
fp32 single-token GEMVs take `torch.mv(weight, x_view[0])` before the
fp16 GEMV family (which rejects fp32). Micro-bench at the gate shape
[128, 2688]: F.linear fp32 141.8 µs · triton 117.6 µs · mv **8.0 µs**.

## Evidence — FOR

- **Serving (the gate):** 59.44 → **70.40 tok/s** (+18.4%), samples
  70.34–70.44 (`/tmp/bench_arm4.log`).
- PPL 26.9757 vs 26.9555 — Δ0.07%, fp32 accumulation-order change,
  same band as every other kernel swap on this model.

## Evidence — AGAINST

- M=2..32 fp32 (batched/spec-decode steps) still take the ~118 µs
  triton path — batched fp32 GEMV left open (roadmap NH-3 residue).

## Interactions

- The path only fires for fp32 operands — previously those *crashed*
  at LLMM1 on gfx906 (blocker 1), so no previously-working model's
  route changes. Blast radius: fp32-router models only.

---

## 2026-08-30 — TP=2 crash reports investigated: EP flag required + amdsmi wrapper fix

**VERDICT:** SHIPPED (wrapper fix + verification) · **GATE:** TP=2 +
`--enable-expert-parallel` serving boot + greedy A/B vs TP=1 (7 prompts,
token-ids/text diff) + multi-chunk prefill.

## HYPOTHESIS

If a user launches this model at TP=2, (1) the default no-EP config dies
in `CompressedTensorsWNA16MoEMethod.create_weights` because the g64 scale
groups do not divide the per-rank intermediate (1856/2 = 928, 928 % 64 = 32),
and (2) on boots whose post-torch amdsmi is broken (0 handles, `shut_down`
returns NOT_INIT), the first unprotected `get_device_name` (SSD config
lookup in the cudagraph-profile dummy run) kills the rank-1 worker inside
`with_amdsmi_context`'s `finally`, leaving rank 0 hung in an NCCL spin.

## What was done

- Reproduced (1) exactly: TP=2 default → `ValueError: ... intermediate
  size per tensor-parallel partition (928) to be divisible by group_size
  (64)` in `create_weights` (pre-existing upstream check, untouched by the
  onboard branch — the message itself prescribes the fix).
- Reproduced (2): rank-1 worker died in
  `selective_state_update → get_ssm_configs → get_device_name` with
  `AMDSMI_STATUS_NOT_INIT` from the wrapper's `finally: amdsmi_shut_down()`
  (`/local/tmp/nemotron_tp2_ep_server.log`); rank 0 then spun at 100 % CPU
  + GPU (NCCL) until SIGTERM. Fresh-process probe: post-torch
  `amdsmi_init()` returns success with **0 handles** and `shut_down()`
  raises NOT_INIT — deterministic this boot (single- and dual-GPU).
- Fixed `with_amdsmi_context` (vllm/platforms/rocm.py): cleanup failure in
  `finally` no longer masks a successful query (warning only); the
  0-handles GCN-arch fallback (`AMD_GFX906`) then carries the lookup.
  This is the permanent fix the 2026-08-22 C2-V entry in
  `degradation_details.md` called for (same crash signature, 35B-MoE
  rank-1 death in `get_device_name` during `profile_run`; that boot's
  workaround was a per-run sitecustomize shim). Test:
  `tests/test_rocm_amdsmi_context.py` (2/2).
- Verified TP=2 + `--enable-expert-parallel`: boots (64 local experts/rank,
  `GFX906_HIP` MoE backend still selected under EP), multi-chunk prefill
  (5041 tok = 2 chunks) + decode coherent.

## Evidence — FOR

- Greedy A/B TP1 vs TP2+EP, same tree (7 prompts, temp 0):
  5/7 token-identical; 2/7 (`story`, `mito`) diverge mid-output and stay
  coherent (quantized-kernel + all-reduce float diffs — expected). The
  whitespace-loop `transformer` prompt loops **identically** on both —
  model behavior, not EP corruption. Probe: `/local/tmp/nemotron_tp_ab.py`
  (JSONs `nemotron_tp_ab_tp{1,2}.json`).
- Steady-state decode TP2+EP B=1: ~100 tok/s (128-tok runs 99.4/101.4/
  101.0; 1.26–1.29 s) — above the 70.4 TP=1 healthy-host record, as
  expected (EP halves per-GPU expert traffic). **Caveat:** this boot's
  amdsmi is broken (fallback marker in every log) — the number stands but
  host state is suspect; re-confirm on a clean boot.

## Evidence — AGAINST

- None at the gate. Note the TP=1 arm on this same boot ran abnormally
  slow (first request ~7 tok/s, warming to ~35) while TP=2 ran at
  record level — a TP-dependent perf anomaly on a suspect boot, not
  adjudicated (canary + clean boot pending; see degradation.md rows).

## Interactions

- **Operational rule:** this model at TP>1 needs
  `--enable-expert-parallel` (g64 + pure-TP shard is unsatisfiable:
  1856/TP must be a multiple of 64, which only holds at TP=1 — or TP=29,
  which is not a deployment; 128 % 2 == 0, so EP=2 is exact).
  Recorded in the workspace AGENTS.md local-serving notes.
- The wrapper fix is generic (any `get_device_name`/`get_device_uuid`/
  memory query on a broken-amdsmi boot); blast radius: none on healthy
  boots (shut_down succeeds → no warning path).

## Refrigerated residue

- This boot's perf anomaly (TP1 decode ~7–35 tok/s vs TP2 ~100, boot O,
  amdsmi broken since ~08-30 morning) is an open question for the
  degradation TP-dependence line — needs the canary on a clean boot.
- Tuned SSD-decode configs for `AMD_GFX906` do not exist (lookup falls to
  the default launch config on every boot here); tuning is a perf item,
  orthogonal to this incident.

## 2026-08-30 (later) — review follow-up on the wrapper fix

**VERDICT:** SHIPPED · **GATE:** `tests/test_rocm_amdsmi_context.py` (2/2,
with cleanup + diagnostic-contract asserts) + module import + plugin probe
on this box.

Self-review of `e4a4dd0edc` found four diagnostic-integrity gaps, all in
the exact scenario the fix exists for; fixed same day:

- The "after a successful query" warning also fired on the failed-query
  path (finally runs both ways). Now branched in `_shut_down_amdsmi()`:
  warning on success, debug on failure — the propagating query exception
  dominates there.
- `warning_once` defaulted to `scope="local"` → dropped on non-first
  local ranks, i.e. the very rank that died in the TP=2 incident. Now
  `scope="process"`.
- Same `finally: amdsmi_shut_down()` pattern in
  `platforms/__init__.py::rocm_platform_plugin` fed the NOT_INIT error
  into the outer except and misattributed the debug log ("not available
  because: AmdSmiException" instead of "no GPU found"); return value was
  unaffected (assignment precedes the finally). Fail-soft swallow+debug
  applied symmetrically; no dedicated test — the only observable delta
  is log text, which is brittle to pin.
- Tests now pin the cleanup half of the contract (`init/shutdown
  assert_called_once`) and the warning contract (fires once at process
  scope on success; absent on the failure path); failure test switched
  to `pytest.raises`.

Not actioned (deliberate): decorator order on `get_device_name`
(`with_amdsmi_context` outside `lru_cache` → per-call init/shutdown even
on cache hits) and broken-boot `_GCN_ARCH` initializing CUDA at import —
both pre-existing design, not regressions; noted for the roadmap if the
broken-amdsmi boots persist. No commit trailer on `e4a4dd0edc` (policy
miss, local fork, left as-is).

---

## 2026-08-30 (later) — NH-2: int8 in-kernel W8A16 GEMV/GEMM (CPU side done)

**VERDICT:** OPEN (implementation + CPU gates on
`gfx906/nh2-int8-gemv`; kernel probe + serving A/B + PPL pending on the
GPU) · **GATE:** probe at Nemotron's exact dense shapes
(`benchmarks/kernels/gfx906/bench_w8a16_gfx906.py`: M=1 vs the production
`rocm_unquantized_gemm_impl` dispatch, M=4 + M=4096 arms, correctness vs
fp32 reference with 0x80 zero-codes pinned) → serving A/B
(`VLLM_GFX906_W8A16_INT8=0` arm, same boot, 70.4-tok/s recipe) + PPL
26.96–27.02 band + 7-prompt greedy A/B.

## HYPOTHESIS

Nemotron's per-step dense weight traffic is 1.32 GiB of int8 across the
71 CT W8A16 channel tensors (23× mamba in_proj [10304, 2688] + 23×
out_proj [2688, 4096] + 6× GQA q/k/v/o + lm_head [131072, 2688]). The
scheme dequants to fp16 at load (2× the traffic, +1.3 GiB VRAM) and
serves M=1 via LLMM1 rpb=4 — 3.57 ms/step in the profile (K=2688/4096
miss the custom `dense_gemv_gfx906` whitelist). If in-register dequant
(bias-128, per-channel scale) is cheap, serving the packed bytes directly
halves the traffic; the P2 probe (`DEVLOG-int8-transfer.md`) measured
exactly this pattern at 1.93–2.03× on HBM-bound shapes, never worse than
0.96×.

## What was done

- `compressed_tensors_w8a16_channel_dequant.py`: the int8 in-kernel path
  is now the default on gfx906 + fp16 (`VLLM_GFX906_W8A16_INT8`, default
  on; `=0` gives the old dequant path, bit-identical). `process_weights`
  keeps the packed int32 storage and registers `weight_i8` — a uint8
  [N, K] **view** of it (no copy, no extra VRAM) — freeing only the load
  metadata. New Triton kernels: `k_w8a16_gemv` (M=1; per-row scale after
  the f32 reduction; SPLIT-K via atomic on a zeroed output) and
  `k_w8a16_gemm` (M>1; the tile dequant is bit-identical to the old
  load-time dequant — f32 (w−128)·s → f16 — so M>1 numerics are the
  dequant path's numerics; `tl.dot` with f32 accumulation).
- `_gemv_geometry` = P2's `_pick` with a split guard: P2's `_pick`
  *asserts* at Nemotron's K=2688 (split=8 → per=336 → no aligned BK);
  the guard rejects a split whose slice cannot take BK ≥ 32.
- The probe measures the "current" arm through
  `rocm_unquantized_gemm_impl` — the actual production dispatch (LLMM1
  rpb=4 at these K's), not a re-implementation.
- Unit tests extended (6/6 pass on CPU): the `weight_i8` view aliases
  the packed storage (data_ptr identity), the byte convention
  `(word >> 8b) & 0xFF`, numerics equal the dequant reference, env-off
  fallback, bf16 → dequant path, fail-closed on scale shape / K.

## Evidence — FOR (CPU side only)

- 6/6 unit tests; `_gemv_geometry` verified on all six shapes (K=2688 →
  BK=64 SPLIT=2; lm_head → 8192 programs SPLIT=1; k/v_proj → SPLIT=4
  BK=32 — no degenerate tiles); ruff clean.
- Prior: P2 int8 row-GEMV at 1716.9 µs on the 248320×5120 lm_head shape
  = 1.81× the 3114–3193 µs recorded production fp16 floor, at 93 % of
  the int8 floor of its own.

## Evidence — AGAINST

- Nothing measured yet (probe pending). The known risks the probe
  settles: (a) Triton launch overhead on the 12×/step [256, 2688]
  k/v_proj (P2: ~15 µs floor on both sides — expected ~neutral);
  (b) `k_w8a16_gemm` at M=4096 prefill — the in-register dequant adds
  3–4 ALU ops per element on top of the FMA-bound dot (no MFMA on
  gfx906; hipBLAS fp16 is FMA-bound too), so prefill is expected ≈
  neutral, not 2×; (c) fp16 atomic_add partials in the SPLIT>1 GEMV
  (only the q_proj family) — P2 passed ≤3e-4 rel-err with the same
  scheme.

## Interactions

- Blast radius: only CT W8A16 channel-symmetric fp16 layers on gfx906 —
  Nemotron-H is the only served model of this shape (its experts are
  INT4 WNA16, a different scheme). No other model's route changes; the
  env kill switch reproduces the old path bit-identically for the A/B.
- lm_head is one of the int8 tensors: 352 MB/step — the single largest
  decode byte item on this model.

## Refrigerated residue (when the GPU is free)

1. `W8A16_PROBE_QUICK=1` first, then the full probe →
   `/local/tmp/w8a16_probe.log`.
2. If any M=1 shape lands < 0.95× of cur: fix the geometry before the
   serving gate (the probe prints per-shape geometry).
3. Serving A/B + PPL + 7-prompt greedy A/B (item 3 of the gate above);
   the dequant arm runs with `VLLM_GFX906_W8A16_INT8=0` on the same
   boot.
4. If the M=4096 prefill arm regresses > 10 %: either accept (log it)
   or revert the whole scheme via the kill switch — there is no
   int8-GEMV-only hybrid without re-materializing the fp16 weight.

---

## Search keys

`HYPOTHESIS:` `VERDICT:` Nemotron, nemotron_h, group-64, g64, relu2,
RELU2_NO_MUL, W8A16 channel, int8 channel, dequant, Conch, ConchLinearKernel,
CanonicalizePointers, HAS_INITSTATES, chunk_scan, ssd_chunk_scan, fp32 router,
LLMM1, GateLinear, force_fp32_compute, AMDGCN_USE_BUFFER_OPS, TP=2,
tensor-parallel, enable-expert-parallel, EP, expert parallel, 928,
with_amdsmi_context, AMDSMI_STATUS_NOT_INIT, amdsmi, get_device_name,
_shut_down_amdsmi, rocm_platform_plugin, scope=process,
profile_run, rank-1 worker, NCCL hang, NH-2, weight_i8, bias-128, 0x80,
k_w8a16_gemv, k_w8a16_gemm, SPLIT, VLLM_GFX906_W8A16_INT8,
bench_w8a16_gfx906.
