# Plan — make the gfx906 CUSTOM (Q8 FA) backend the serving decode path
Copyright Kevin Read <me@kevin-read.com>


Status: v8 (2026-08-16) — **Stage 2 (quantize-during-gather) LANDED.**
The LEGACY decode two-kernel sequence (gather_paged_kv_fp16 +
quantize_q8_0, 458 µs/step under tracer; both latency/launch-bound at
B=1) is now one fused kernel: V fp16 copy + K gathered and quantized to
q8_0 in-kernel via the same `quantize_block_q8_0_halfwarp` helper as
the standalone quantizer → output **bit-equal** to the two-kernel path
(3-shape bit-exact unit test; PPL unchanged by construction, 6.6895).
Micro: Sk=3328 64.3 → 36.9 µs/call (−27.4 µs × 10 layers ≈ −274
µs/step). Serving A/B (local venv): OFF 62.594/62.695 vs
**DEFAULT 63.534/63.581 → record 63.56 t/s** (+1.47% over 62.67).
Default flipped: `GFX906_FA_FUSED_QUANT` default on, `=0` kill switch.
New rocprofv3 trace also firmed up the per-step budget: FA stack now
≈ 621 µs/step (was 3272); dense-GEMV/LLMM1 dispatch confirmed at its
micro-bench optimum (no lever); a ~1.18 ms/step fill+D2D-copy pile is
uncharacterized (FA contributes only ~10 small q_pad zeros) →
candidate P3-4 pass. Build note: `cmake/hipify.py` gained a
same-directory copytree guard (in-source rebuilds crashed on Python
3.12 SameFileError). Detail: DEVLOG §"Post-FA-track trace + stage 2".

Status: v7 (2026-08-16) — **FA kernel track LANDED** (the §"M0-3 /
remaining P3-3a" item: the FA kernel itself, 327 µs/layer at B=1).
Root cause: the launcher hardcoded NC2=1 + gridDim.y=1 → 16 blocks =
6.7% of the MI50's wavefront slots at B=1 (latency-bound, 111
ns/token, 14% HBM). Enabled GQA head-packing (`GFX906_FA_NC2`) + KV
split (`GFX906_FA_KVSPLIT`) + new `fa_split_combine_kernel`; fixed
three bugs along the way (vendor null-mask deref at NC2>1; NC2=8×
prefill ncols=64 OOB → packing restricted to decode `seq_q≤2`; vendor
OOB-tail: the NC2>1 strided KV loop never enabled `oob_check` for the
tail tile → wrong softmax for kv_max % nbatch_fa ≠ 0). Evidence: 12/12
`test_gfx906_fa.py` (7 new split/GQA subprocess cases incl. empty
trailing splits); PPL 6.6999→6.6895 (−0.15%, bar ≤2%; the greedy
4×128 A/B is not a valid gate — legacy×2, new×2 and Triton×2 all
diverge cross-run: engine-level MoE near-tie non-determinism). Serving:
legacy 57.08 / NC2=1,y=8 62.13 / **NC2=8,y=16 62.81+62.92 (docker
0.85)**; default flipped to NC2=8/KVSPLIT=16 (kill switch: both =1);
new-default 3-sample on the local venv bench (util 0.95 +
fastsafetensors): **62.677/62.668/62.671**. Default-request decode:
57.09 → ~62.7 t/s (+9.8%); llama.cpp gap 1.23× → 1.12×. Also this
session: serving benches moved from docker to the local `.venv`
(editable vllm + ROCm 7.14 gfx906 env script +
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` + fastsafetensors with a
one-line vLLM GDS-fallback catch fix) — DEVLOG "Local-venv bench
environment". M0–M3 below unchanged; M3 outcome stands with the new
number. Bench-history detail: DEVLOG §"FA kernel track (P3-3a)".

v6 (2026-08-16) — **code-review fixes LANDED** (combined review
`phase3_code_rev_combined.md`): C1 arch-gate on the GEMV dispatch
(CRITICAL), F1/F6 capture-safe q_pad + gather buffer lifecycle (retired
captured buffers stay alive; real-impl lifecycle test), F2/M1 GEMV
numeric tests incl. K-split, F4 RPT hardening, F5/M2 V1 bounds guard +
gridDim.z cutover, F7 LEGACY=0 RC2 loud guards, F9 doc/comment cleanup
(vendored Russian → English), F10 repo hygiene (gitignore, probes
checked in, stale tables fixed), F3 evidence (PPL 6.6811 vs 6.6775 =
+0.05%; multi-batch probe: no corruption; the cross-run near-tie
non-determinism is an engine property — pure Triton shows the same),
H3/M3 V2 mechanism re-characterized (wavefront under-fill →
co-resident interleaving; reduced-harness negative result).
**57.09 t/s unchanged** (3-sample post-fix bench ≈ HEAD bench ≈ 56.7,
machine drift).

v5 (2026-08-16) — **M2 LANDED + Route B stage 1 LANDED**:
LEGACY decode path FULL-capture-safe, CGSupport default flipped to
UNIFORM_SINGLE_TOKEN_DECODE, fused fp16 gather (V1) default —
**57.09 t/s mean (σ≈0.09, n=5) on the default-request config** (was
22.44 via the downgrade bug, 52.90 after M2), 128/128 greedy probes
identical on both PIECEWISE and FULL paths, T3 capture/replay test
passing. Sub-plan of Phase 3 **P3-3a** (parent:
`plan-decode-phase3.md` v10). Evidence, bench history and the bug-hunt
narrative live in `DEVLOG-moe-opt.md` (§"P3-3", §"Serving-mode backend
findings", §"P3-3a: CUSTOM serving correctness probe"). Review findings
that drove v2: `gfx906fa-serving-plan-rev-claude.md` (merged claude + ds4
+ qwen).

**v2→v3 changes** (all from the 2026-08-15 serving findings):
- **The premise "dead code at runtime" is stale**: the plugin entry point
  is now active — CUSTOM wins backend selection by default (and
  `VLLM_ATTENTION_BACKEND` was dropped upstream; the knob is
  `attention_config["backend"]`).
- **New anchor: 52.07 t/s** (LEGACY=1 default + requested PIECEWISE +
  GEMV on) — already beats the 44.09 Triton-FULL record. The v2 M1 stop
  condition ("< +0.3 ms/step vs Triton-PIECEWISE → P3-3b") is moot unless
  Triton-PIECEWISE itself exceeds 52 (queued in run_ab2).
- **M0 re-scored**: item 1 (gather micro-bench) passed (21.7 µs/layer —
  but that measured the LEGACY=0 *fused* gather; serving default is
  LEGACY=1 with the PyTorch `_gather_kv` + inline `quantize_q8_0` — the
  52.07 end-to-end number is the real gate and already passes). Items 2–3
  (Triton-PIECEWISE baseline, attention slice) still queued/pending.
- **Correctness probe PASSED** (128/128 greedy tokens identical vs
  Triton-FULL at pp=2048) — LEGACY=1 serving decode is correct under KV
  growth; COW/multi-batch items stand (they are LEGACY=0-adjacent and only
  bite if the Q8 side-buffer path is enabled).
- **RC1/RC2/W1/W2/T1/T2 demoted**: they are LEGACY=0-only. The LEGACY=1
  path has no Q8 side buffer, so none of that lifecycle work is needed for
  the default serving path. M1-as-specified (enable LEGACY=0 fused
  gather) is now an *optional optimization track*, not the critical path.
- **M2 is now the critical path** — experiment in flight: the LEGACY=1
  decode path may already be FULL-capture-safe (first FULL capture uses
  `profile_seq_lens=max_model_len` → Sk-sized buffers allocated at
  capacity; metadata is runner-staged into pointer-stable buffers and
  re-read live at replay). `GFX906_FA_CG=decode` knob added to test
  without flipping the default. If it passes: M2 = W8 support flip +
  T3 capture test, no W5 buffer surgery needed for LEGACY=1.
- **New engine bug (parent v9)**: requested FULL_DECODE_ONLY +
  CGSupport.NEVER downgrades to PIECEWISE *after* the model is compiled
  non-piecewise → decode degrades toward eager (22.44 t/s). With M2's
  support flip (≠ NEVER) the downgrade stops firing for this backend;
  the bug remains for other NEVER backends (documented, upstream class).

## 0. Goal and target

The vendored Q8 FlashAttention kernel is measured **72 µs/layer vs Triton
paged attention's 194 µs (2.7×)** but cannot run in serving mode today.
Goal: CUSTOM as the serving decode attention path with prefix caching ON
and cudagraphs as strong as the Triton baseline's.

| config | decode | attention slice/step |
|--------|--------|----------------------|
| serving, Triton, FULL_DECODE_ONLY (baseline) | 43.99 t/s (archive 44.09 reproduced) | 10 × ~194 µs ≈ 1.94 ms ¹ |
| serving, Triton, PIECEWISE | 43.96 t/s | — |
| serving, CUSTOM + PIECEWISE + GEMV | 52.07 t/s | 10 × (FA + LEGACY gather) ≈ TBD (M0-3) |
| serving, CUSTOM + PIECEWISE, GEMV off | 50.88 t/s | — |
| serving, CUSTOM + FULL_DECODE_ONLY + GEMV, torch gather (n=6) | 52.90 t/s mean, σ≈0.06 | 10 × (FA + gather + quantize) ≈ 3.6 ms (rocprofv3, post-52.90) |
| serving, CUSTOM + FULL_DECODE_ONLY + GEMV, **V1 fused gather (n=5)** | 57.09 t/s mean, σ≈0.09 | torch gather was 128–190 µs/layer isolated (M0-3); V2 in-graph trap (285 µs/call) documented — DEVLOG "Route B stage 1" |
| serving, CUSTOM + FULL_DECODE_ONLY + GEMV + V1 gather + FA NC2=8/KVSPLIT=16 + **fused gather-quantize (current default, v8)** | **~63.6 t/s** (local venv 0.95: 63.534/63.581; OFF A/B: 62.594/62.695) | gather+quant 64.3 → 36.9 µs/layer @Sk=3328 (1.7×) — DEVLOG "Post-FA-track trace + stage 2" |
| serving, CUSTOM + FULL_DECODE_ONLY + GEMV + V1 gather + **FA NC2=8/KVSPLIT=16 (v7)** | **~62.7 t/s** (docker 0.85: 62.81/62.92; local venv 0.95: 62.677/62.668/62.671) | FA 245 → 58.3 µs/layer @Sk=2176 (4.2×; HBM floor ~8 µs) — DEVLOG "FA kernel track" |
| serving, CUSTOM + requested FULL_DECODE_ONLY, CGSupport=NEVER (`GFX906_FA_CG=never`) | 22.44 t/s — **downgrade bug** (dormant; upstream class) | — |
| eager, CUSTOM LEGACY=0 (works today) | 19.33 t/s | — |

¹ "10 × 194 µs" comes from a P3-0 profile at seq~500. The 44.09 t/s bench
runs at average Sk≈2176. M0 must produce the real attention slice from a
rocprofv3 pass over the 44.09 t/s run itself before these projections are
trusted.

**Why the §5 math is provisional:** the only measured full-path data point
at B=1 is eager parity (CUSTOM 19.33 vs Triton 19.49 t/s). Since
`_DIRECT_PAGED_MIN_BATCH=2` and the serving bench is single-request (B=1),
every serving decode step routes through the gather path — direct-paged
never engages. The eager parity implies the gather+dtype tax is absorbing
essentially the entire 122 µs kernel win at B=1, suggesting gather alone is
~100+ µs/layer, not the 20–50 µs assumed in §5. The M0 gather micro-bench
exists to measure this directly.

**M1 target (revised):** "CUSTOM-PIECEWISE ≥ Triton-PIECEWISE + 0.3 ms/step"
(mode-matched). The 44.09 t/s baseline is context only, not the comparison
point, because CUSTOM runs PIECEWISE while that baseline is FULL_DECODE_ONLY.

Stop condition: if M1 nets **< +0.3 ms/step vs the Triton-PIECEWISE
baseline** after W3, abandon and fall back to P3-3b (Triton KV
partitioning).

## 1. What blocks it (root causes, all verified in code/traceback)

**RC1 — crash in serving, LEGACY=0** (`RuntimeError: value_cache blocks
mismatch`, `gfx906_fa_paged.py:373` → TORCH_CHECK in
`csrc/gfx906_fa/gfx906_fa.cpp:431`):
`profile_cudagraph_memory` (not `profile_run` — see note) →
`_init_minimal_kv_cache_for_profiling` →
`initialize_kv_cache(is_profiling=True)` allocates a **minimal** KV cache;
the dummy forwards run `do_kv_cache_update`, whose `_ensure_q8_sidebuffer`
binds `_k_cache_q8` to that shape; the real cache is allocated later with a
different `num_blocks`; the first real-capture warmup calls
`do_kv_cache_update` with `value_cache` (real shape) + `key_cache_q8`
(minimal shape) → TORCH_CHECK fires. LEGACY=1 never allocates the side
buffer, which is why the default never crashed.

*Note on trigger:* `_init_minimal_kv_cache_for_profiling` is called from
`profile_cudagraph_memory` (`gpu_model_runner.py:6765`), which runs
**after** `profile_run`. During `profile_run` itself, `kv_cache_config`
does not exist yet → `_get_slot_mappings` returns None → no
`do_kv_cache_update` at all. T1 must simulate the
`profile_cudagraph_memory` → real-cache path, not `profile_run`.

**RC2 — correctness hole with prefix caching (LEGACY=0)**: COW block copies
(`gpu_model_runner.execute_model` → `copy_kv_cache_blocks_inplace`,
`vllm/v1/worker/utils.py:563`) mirror only the fp16 caches; the Q8 side
buffer keeps stale rows → garbage on prefix hits that fork a partially
shared block. Full-block prefix hits reuse blocks **in place** (no copy) —
that is why the `[cached]` probe passed. Related:
`_zero_block_ids` zeroes fresh fp16 blocks only — **harmless** for Q8
(rows beyond `kv_max` are masked by the kernel; every live row is rewritten
by `reshape_and_cache_q8` before it can be read) — document, no change.

**Invariant**: correctness of the LEGACY=0 path rests on *every live K row
being written via `do_kv_cache_update` before it is read*. Exceptions to
guard: (a) KV connectors / disaggregated prefill write blocks directly into
the fp16 cache bypassing `do_kv_cache_update` — add a loud log/assert when
a connector is enabled with LEGACY=0; (b) COW copies (RC2/W2).

**RC3 — cudagraph mode downgrade**: `Gfx906FAMetadataBuilder
._cudagraph_support = AttentionCGSupport.NEVER` →
`resolve_cudagraph_mode_and_sizes` downgrades the engine to PIECEWISE,
while the Triton baseline serves with FULL_DECODE_ONLY. Even a working
CUSTOM would be measured in a weaker mode (attention eager between graph
pieces: ~30–40 extra launches/step, graph-boundary overhead). **This makes
M1's serving number incomparable to the 44.09 t/s baseline** — see revised
M1 target in §0 and M1 exit in §2.

**RC4 — capture debt (only needed for M2/FULL mode)**: forward-time
reallocs with `torch.cuda.empty_cache()` (class-level gather buffers
re-keyed by `Sk_pad` every 32-token boundary; `q_pad_buf` grow), host
branching on `max_seqlen` metadata (`_should_use_direct_paged`,
`ncols1`), dtype conversion `q.float()` per call (cost, not blocker).
All legal eagerly (PIECEWISE), illegal/fragile inside FULL capture.

**RC4 additional — two capture landmines not in the original RC4 list:**

- **Warmup→capture Sk_pad mismatch**: pre-capture warmup builds metadata
  with `for_cudagraph_capture=False` → `max_seq_len=1` → `Sk_pad=32`.
  The capture itself uses `for_cudagraph_capture=True` →
  `max_seq_len=max_model_len` → `Sk_pad≈2816`. `_ensure_gather_buffers`
  requires an **exact** shape match; the mismatch triggers
  `torch.cuda.empty_cache()` **inside the CUDA graph capture** — illegal.
  W5 must fix the reuse semantics (see §2/M2), not just pre-allocate.

- **Class-level gather buffers dangle across capture sizes**: in ascending
  capture order (B=1, 2, 4, 8), each capture writes into the class-level
  buffer that exists at that moment. Capturing B=2 reallocates and frees
  the B=1 buffer; the B=1 graph then writes K/V into freed memory on every
  replay. W5 must ensure all captured graphs share one buffer object that is
  never shrunk.

## 2. Milestones

### M0 — pre-work (go/no-go gate) — **item 1 PASSED; items 2–3 in flight/pending**

Required before M1/M2 coding per parent plan v6 (re-scored v3):

1. **Gather micro-bench**: measure `gather_paged_kv_q8` + `q.float()` cost
   per layer at serving shapes (B=1, Sk≈2176–2816, Hkv=8, D=256) in
   isolation. This is the decisive go/no-go gate. If gather > ~80 µs/layer
   the realistic M1 serving gain collapses to ≤0.3 ms and P3-3a stays
   suspended.
2. **Triton-PIECEWISE baseline**: run the standard serving bench
   (`BENCH_EAGER=0`, pp=2048/tg=256) with the Triton backend forced into
   PIECEWISE mode. Record t/s and the resolved `cudagraph_mode`. This is
   the mode-matched comparison point for all M1 numbers.
3. **Baseline attention slice**: rocprofv3 pass over the 44.09 t/s run to
   record the real Triton attention time at Sk≈2176, replacing the
   seq~500 estimate from §0.

**M0 exit / go condition**: gather ≤ ~80 µs/layer → proceed to M1.
If gather > ~80 µs/layer → P3-3a suspended; switch to P3-2.

**v3 re-score**: item 1 measured 21.7 µs/layer (2.0× HBM floor) — gate
passed, though the measured kernel was the LEGACY=0 fused path; the
serving default (LEGACY=1) PyTorch gather is validated end-to-end by the
52.07 t/s number (which already beats the 44.09 baseline by 3.5 ms/step —
more than the raw FA kernel delta, i.e. the LEGACY=1 gather path is
cheaper than the §0/§5 pessimism assumed). **Proceed.**

**v5 (2026-08-16) — M0-3 resolved, Route B stage 1 landed.** Item 3
(LEGACY attention slice) measured at the 52.90 baseline: the torch
`_gather_kv` costs **128-190 µs/layer** isolated (the v3 demotion of the
fused-gather track was premature). Route B stage 1 — fused fp16-K gather
op, no Q8 side buffer — implemented, stride-domain bug fixed
(K byte-strides vs V element strides), correctness probe PASSED (128/128
bit-exact vs Triton FULL). Serving A/B exposed a trap: the V2
(paged-block, 416 WG + `__syncthreads`) kernel is 41 µs isolated but
285 µs/call in the FULL-graph serving context (49.56 t/s — regression);
the V1 (per-token, grid (B,Hkv,Sk), 64 thr, no barriers) kernel wins:
**57.09 t/s 5-sample mean** (new best = default config; launcher default
flipped to V1, `GFX906_FA_GATHER_V=2` / `GFX906_FA_TORCH_GATHER=1` keep
the alternatives). V2's serving degradation mechanism is not isolated —
see DEVLOG "Route B stage 1".

**v6 (2026-08-16) — combined code-review fixes landed** (details in
DEVLOG §"Phase-3 code-review fixes"). Structurally: the default path is
now capture-order independent (q_pad + LEGACY=0 gather buffers: retired
captured buffers stay alive, `empty_cache()` gone from the forward path,
capture-state poll latches after first capture) with a real-impl
capture→prefill→replay test; the GEMV dispatch is gfx906-gated (C1);
V1 gather gained the bounds guard + the Sk>65535 V1→V2 cutover; LEGACY=0
+ prefix-caching/FULL-capture now log loudly (F7/RC2). Evidence bar met:
PPL point +0.05% vs the fp16 Triton reference (acceptance ≤2%) and the
two-request multi-batch greedy probe (prefix-overlap, B=2 graph) shows no
corruption — the only cross-run differences are greedy near-ties, which
the pure-Triton reference exhibits equally (engine property, likely MoE
routing tie-breaks). H3/M3: the V2 7× in-graph regression does NOT
reproduce in a gather-only graph (ratio 1.06) — re-characterized as
wavefront under-fill (416/960 ≈ 43%) letting other graph branches
co-reside and interleave; V1's 6656 wavefronts saturate the machine.
Numbers unchanged: 57.09 t/s 5-sample record; post-fix 3-sample bench
≈ HEAD 3-sample bench (≈56.7, machine drift).

**v8 (2026-08-16) — stage 2 (quantize-during-gather) landed.** One fused
kernel per FA decode layer replaces gather_paged_kv_fp16 + quantize_q8_0
(bit-equal outputs; `GFX906_FA_FUSED_QUANT` default on). Fresh
rocprofv3 trace: FA stack ≈ 621 µs/step (was 3272); dense dispatch at
micro-bench optimum; ~1.18 ms/step fill/copy pile uncharacterized
(next: P3-4 characterization). Record: 63.56 t/s; llama.cpp gap 1.11×.

### M1 — LEGACY=0 serving path (v3: demoted to optional optimization track)

v3: the default serving path is LEGACY=1 (inline quant, no Q8 side
buffer), which already serves at 52.07 t/s with a passed correctness
probe. M1-as-specified (enable the LEGACY=0 fused gather + Q8 side
buffer) remains valuable only if a later A/B shows the fused gather
beats the LEGACY=1 PyTorch gather by a meaningful margin at serving
shapes; W1/W2/W4-T1/T2 are then required (RC1/RC2 apply to LEGACY=0
only). Attention runs eagerly between piecewise graphs → dynamic shapes
and allocations stay legal. Work items (unchanged):

- **W1 side-buffer lifecycle**: in `do_kv_cache_update`, if `_k_cache_q8`
  is None or `(key_cache.data_ptr(), key_cache.shape, key_cache.device)`
  does not match the stored binding → free, realloc at the real cache
  shape, `zero_()`. (Checking `shape[0]` alone misses coincidentally-equal
  block counts on engine re-init; bind by full identity.)
  Update the W2 registry entry in the same operation (data_ptr changes on
  realloc). Trigger: `profile_cudagraph_memory` → real-capture warmup path,
  not `profile_run` (see RC1 note). Keep the C++ TORCH_CHECK as the
  invariant guard.

- **W2 COW mirror**: the copy utility (`copy_kv_cache_blocks_inplace`,
  `vllm/v1/worker/utils.py:563`) runs in `_update_states`
  (`gpu_model_runner.py:1267–1271`) before the forward. The mirror must be
  in the same call window (a post-forward hook would miss prefix hits that
  affect the current step).

  Registry design (required specifics before writing the hook):
  - **Key**: the fp16 backing storage pointer of the `kv_cache` tensor
    for each FA layer (e.g. `kv_cache.untyped_storage().data_ptr()`), not
    the Q8 side-buffer pointer (which has a separate storage).
  - **Scope**: only gfx906 FA layers. GDN/mamba state blocks also appear
    in `kv_caches` — applying `(src, dst)` block indices to the wrong
    storage is a silent corruption path. Guard by registry emptiness for
    non-registered storages.
  - **KV-sharing**: if two FA layers share one fp16 backing storage
    (`kv_sharing_target_layer_name`), the data_ptr key collides. Key by
    `(data_ptr, layer_name)` or register a per-layer list.
  - **On W1 realloc**: the data_ptr changes → update the registry entry
    in the same `_ensure_q8_sidebuffer` call (see W1).
  - **Copy semantics**: a `view(num_blocks, -1)` uint8 replay of the same
    `(src_block_id, dst_block_id)` pairs is valid — `_k_cache_q8` is
    block-major contiguous with the same `num_blocks` as the fp16 cache
    (verified). Also replay V (fp16) if a V-Q8 side buffer is added later.

  Stopgap if the runner hook is invasive: `GFX906_FA_NO_PREFIX_CACHE` to
  disable prefix caching **for A/B measurement only** — if used, record it
  explicitly next to the bench number. M1 numbers measured with this flag on
  are incomparable to the 44.09 baseline (which ran with prefix caching).

- **W3 gather-buffer hysteresis** (perf, gated on flame-trace evidence):
  grow the class-level gather buffers in capacity steps (round `Sk` up to,
  say, 512 tokens; serve per-step calls from `narrow()` views) so
  realloc+`empty_cache()` stops firing every 32-token boundary. **Do not
  bundle this into the M1 milestone** — gate on measuring whether the
  realloc is visible in a flame trace first. Include as a separate A/B row
  in the §4 bench matrix rather than as a prerequisite.

- **W4 tests + probes** (§3): T1 lifecycle, T2 COW, partial-block probe
  (see M1 exit below); check probes into `tests/` or `tools/` (not `/tmp`);
  DOUBLE_CHECK live gather.

**M1 exit** (all required before declaring M1 done):
1. Serving bench (`BENCH_EAGER=0`, pp=2048/tg=256) runs LEGACY=0
   end-to-end without crash.
2. **Partial-block COW probe correct** (hard gate): two consecutive requests
   where the shared prefix length is **not** a multiple of 16 (e.g.
   520-token first request; second request reusing that prefix → last shared
   block is partially filled → COW fires → Q8 rows must be correct).
   This is the only path that exercises RC2/W2; the standard bench and the
   existing `[cached]` probe both use block-aligned prefixes and never fire
   COW.
3. Resolved `cudagraph_mode` asserted PIECEWISE (assert, not just record).
4. FA-layer backend selection asserted in logs (not silently falling back
   to Triton).
5. Record t/s + mode vs **Triton-PIECEWISE** (M0 reference), with 44.09
   as context. Gate: CUSTOM-PIECEWISE ≥ Triton-PIECEWISE + 0.3 ms/step.
6. Report state: `NO_PREFIX_CACHE` flag on/off, bench entrypoint, sample
   count.

### M2 — FULL_DECODE_ONLY (CGSupport.UNIFORM_SINGLE_TOKEN_DECODE) — **DONE (v4)**

**Outcome**: the LEGACY=1 decode path is FULL-capture-safe as-is; no W5
buffer surgery was needed (first FULL capture at
profile_seq_lens=max_model_len → capacity-sized buffers; runner-staged
metadata re-read live at replay; Sq=1 fast path takes no host loop). W8
flipped the default; T3 test passes (including the warmup→capacity
transition, multi-size capture+replay, and live-seq_lens growth cases);
probes identical on both paths. Work items (resolved as noted):

v3: promoted ahead of M1 (see v2→v3 changes). The LEGACY=1 decode path
has no Q8 side buffer, so M2 on the default path needs no W1/W2; the
open question is capture-safety of the LEGACY gather allocations, which
the `GFX906_FA_CG=decode` experiment answers empirically (hypothesis:
safe, because the first FULL capture runs at
profile_seq_lens=max_model_len → capacity-sized buffers, and metadata is
runner-staged). Work items:

- **W5 static decode shapes** (correctness requirement, not just perf):
  Replace the exact-shape reuse check in `_ensure_gather_buffers` with a
  ≥-capacity check + `narrow()` view. Pre-allocate at max capture sizes:
  `B_cap = max(max_cudagraph_capture_size, max_num_seqs)`,
  `Sk_cap = ceil(max_model_len/32)·32`. Remove all `torch.cuda.empty_cache()`
  from the forward path. Add `assert not torch.cuda.is_current_stream_capturing()`
  in any remaining realloc path. Ensure all captured graphs (B=1, 2, 4, 8)
  share **one** buffer object that is never shrunk — the B=2 capture must
  not free the B=1 buffer (see RC4 additional). Same treatment for
  `q_pad_buf`.

- **W6 audit host-branching in the decode path**: `_should_use_direct_paged`
  (B, Sq are static per capture; the branch is on values known at capture
  time — bake the result), `ncols1` (decode Sq=1 → ncols1=2, baked), any
  `max_seqlen_k` use. For the gather grid: it is **frozen at capture**
  (kernel launch parameters cannot be per-call in FULL mode). Rely on the
  existing per-workgroup `full_oob` early-out in `gather_paged_kv_q8_kernel_v2`
  (`(block_start_tok >= seq_len) || (phys_block < 0)`) which zeroes V tails
  and skips K for out-of-length blocks — this is the capture-safe substitute
  for a kv_max-driven grid. Verify the early-out covers D=64/128 variants.
  Pin the `block_size == 16` assumption explicitly (direct-paged is
  hard-coded to block_size=16; add an assert so a future block-32 model
  fails loudly rather than silently falling back to gather-only).

- **W7 metadata**: `build_for_cudagraph_capture` already delegates to
  `build()`; confirm `seq_lens`/`block_table`/`query_start_loc` are the
  runner's persistent buffers (pointer-stable across replays). Verified:
  `seq_lens` is a persistent GPU buffer zero-filled for padded rows
  (`gpu_model_runner.py:2241`); `query_start_loc.gpu` and block-table device
  tensors are persistent; padded rows use `NULL_BLOCK_ID` / `-1` / `kv_max=0`
  → inert. Also verify `kv_max` expansion (`seq_lens [B] → [B, grid_x]`
  via `unsqueeze/expand/contiguous` in gfx906_fa.cpp:217) is pointer-stable
  under capture: source is the persistent `seq_lens` buffer, so replay
  re-reads current values — safe, but requires `seq_lens` to be int32
  contiguous (else an extra captured copy).

- **W8 flip support**: `_cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`
  (not `ALWAYS` — `ALWAYS` claims mixed prefill+decode full-cudagraph
  support, which is false: `forward_paged`'s non-decode branch contains a
  host Python loop with `int(cu[s+1]-cu[s])` — a D2H sync inside capture.
  `UNIFORM_SINGLE_TOKEN_DECODE` passes the decode-FULL gate, rejects
  spec-decode and mixed FULL, and accurately describes what is capture-safe.)
  Gate behind `GFX906_FA_CG_MODE=always` env (rename from `GFX906_FA_CG=always`
  to match the `GFX906_FA_FUSED` / `GFX906_FA_DIRECT_PAGED` valued-key
  convention). Assert that the Sq=1 fast path
  (`max_seqlen_q == 1 and num_tokens == num_seqs`) is taken during every
  capture — if not, the Python loop is on the captured stream and must be
  rewritten before capture.

**M2 exit** (all met, v4):
- T3 passes (warmup→capture transition, multi-size capture+replay,
  live-seq_lens growth) ✓
- Serving t/s ≥ M1 number: 52.90 (FULL) ≥ 52.07 (PIECEWISE) ✓
- Mode-matched Triton delta: 52.90 vs 43.96 (+20%) — the graph reduction
  is real even though the marginal FULL-vs-PIECEWISE win is only +1.0 t/s
  (step is GPU-kernel-bound) ✓
- Mode = FULL_DECODE_ONLY ✓; `BENCH_SAMPLES ≥ 5` + σ recorded
  (52.93/52.92/52.94/52.93/52.83/52.87, σ≈0.06) ✓
- (W8 naming note: the knob landed as `GFX906_FA_CG`, not the proposed
  `GFX906_FA_CG_MODE` — shorter, same semantics, default `decode`.)

### M3 — decision

Keep the best of {Triton FULL_DECODE_ONLY, CUSTOM M1, CUSTOM M2}. If all
CUSTOM variants < +0.3 ms/step over Triton-PIECEWISE → reopen P3-3b
(design sketch preserved in the devlog; do not re-derive).

## 3. Tests (extend `tests/kernels/attention/test_gfx906_fa.py`)

- **T1 lifecycle**: allocate side buffer at N1 blocks, simulate the
  `profile_cudagraph_memory` → real-cache path (not `profile_run`), then
  `do_kv_cache_update` with real-shape cache → `forward_paged` passes and
  output matches reference. Use N1 and N2 that differ by more than one
  alignment class (not just ±1 block) to ensure the mismatch check fires.

- **T2 COW**: fill blocks, run the Q8 mirror copy with (src, dst) lists,
  re-gather → Q8 rows match freshly re-quantized fp16 rows. Also assert
  that Q8 rows for COW-copied blocks are correct post-copy (the "every live
  row is rewritten before read" invariant does not hold for COW-copied rows
  until W2 mirrors them).

- **T3 capture/replay** (extended from v1): `torch.cuda.CUDAGraph` capture
  — must cover (a) the warmup→capture `max_seq_len` transition
  (Sk_pad=32 warmup → Sk_pad=max_model_len capture) to exercise the RC4
  Sk_pad mismatch landmine; (b) multi-size capture+replay: capture B=1,
  then B=2 (which would free B=1's buffer under the old exact-match design),
  then replay B=1 → output must match eager. A single-size capture cannot
  catch the dangling-buffer bug.

- **T4 partial-block COW probe** (replaces the v1 "model-level" T4 which
  required scheduler machinery): a checked-in probe script (in `tests/` or
  `tools/`, not `/tmp`) that runs two consecutive completions via a live
  engine where the shared prefix is **not** a multiple of 16 tokens
  (e.g. 520-token prompt → second request reuses prefix → COW fires on the
  last partially-filled block). Assert output correctness. This is also the
  hard gate in the M1 exit.

## 4. Bench matrix (standard docker recipe; pp=2048/tg=256)

All numbers must record: resolved `cudagraph_mode`, `GFX906_FA_NO_PREFIX_CACHE`
state, bench entrypoint, `BENCH_SAMPLES` count.

- **M0 gather micro-bench**: `gather_paged_kv_q8` + `q.float()` cost at B=1,
  Sk≈2176–2816 in isolation. Go/no-go gate for P3-3a.
- **M0 Triton-PIECEWISE baseline**: Triton backend forced PIECEWISE,
  pp=2048/tg=256. This is the mode-matched M1 comparison point.
- **M0 attention slice**: rocprofv3 over 44.09 t/s run → real Triton FA
  time at Sk≈2176 (replaces the seq~500 estimate).
- **eager**: Triton 19.49 / CUSTOM LEGACY=0 ~19.33 (regression check only).
- **M1 serving**: CUSTOM PIECEWISE — compare vs Triton-PIECEWISE (M0), not
  vs 44.09.
- **W3 A/B** (if flamed): gather buffer hysteresis on vs off, same serving
  recipe. Separate row, not part of M1 milestone.
- **M2 serving**: CUSTOM FULL_DECODE_ONLY — compare vs M1 + kernel launch
  count delta.
- **prefill**: CUSTOM vs Triton pp-only run. Acceptance bar: CUSTOM prefill
  ≥ Triton prefill − 5%. Record a CUSTOM-vs-Triton prefill baseline before
  M1 (no such number exists in the devlog yet).
- **B≥2 concurrent-decode** (M2 only, if feasible): 4–8 concurrent requests,
  decode-dominated, to exercise direct-paged. Otherwise mark B≥2/direct-paged
  measurement explicitly out of scope for the M3 decision.
- **sanity greedy A/B vs Triton**: Q8 K quant → expect ~1e-3 logit
  divergence. Acceptance: perplexity on a fixed prompt set within 2% of the
  fp16 path (not "fluency" — not measurable).

## 5. Expected outcome / stop conditions

**Projection (provisional until M0 gather micro-bench):**
attention slice 1.94 ms → 10 × 72 µs (FA) + 10 × (gather TBD µs) ≈ TBD ms.
The 46–47 t/s headline from v1 rests on the 20–50 µs gather assumption,
which is unverified and contradicted by the eager parity data. Update
after M0.

**Stop conditions:**
- M0 gather > ~80 µs/layer → P3-3a suspended; proceed with P3-2.
- M1 < +0.3 ms/step vs Triton-PIECEWISE after W3 → P3-3b.
- M2 correctness unresolved within scope bounds (see W5–W8 per-item):
  keep best M1/M3 result, document.
- Prefill regresses beyond −5% vs Triton: if CUSTOM prefill is below the
  bar, real options are (a) accept the regression; (b) optimise CUSTOM's
  prefill path; (c) add an in-forward dispatch fallback inside
  `Gfx906FAImpl.forward` based on a `max_query_len` threshold (dispatching
  to the Triton impl for prefill — a hack but implementable); (d) abandon
  M1. **There is no per-phase backend selection in vLLM** (one backend per
  attention group serves both prefill and decode of every layer in that
  group); "revert backend for prefill only" is not a mechanism that exists.

## 6. Risks

- **Gather tax at B=1 may be fatal**: the M0 micro-bench is the primary
  gate. The eager parity strongly suggests it; the micro-bench confirms or
  refutes it directly.
- **Piecewise boundary overhead** may eat the kernel win — measure at M0
  (Triton-PIECEWISE baseline) before assuming any kernel win will survive
  to the M1 number.
- **Capacity gather buffers in M2**: the buffer-never-shrunk invariant is
  a correctness requirement for multi-size capture, not just a VRAM
  concern. VRAM: `B_cap × Hkv × Sk_cap × (bytes_per_row + D×2)` — at
  B=32, ctx 2816 ≈ 137 MB; at max_model_len=32K ≈ 0.4 GB. Fine on MI50
  32 GB, but state the formula rather than a single datapoint.
- **Class-level buffers + multiple engines in one process** (probes):
  key the registry by `(device, layer_name)` or equivalent; the reset hook
  must refuse while live CUDA graphs exist (resetting under live graphs
  dandles them), or be test-only.
- **The COW copy hook must not affect non-gfx906 paths** — guard by
  registry emptiness (correct design; the scoping requirement in W2 makes
  this automatic).
- **Numerics**: Q8 K changes logits ~1e-3; greedy divergence from the
  fp16 path is expected and accepted (documented trade), but T1–T4
  byte/parity checks must stay strict.
- **Block-size assumption**: direct-paged is hard-coded to block_size=16
  (gfx906_fa.cpp:580). The bench model is 16; add an assert in W8 so a
  future block-32 model fails loudly rather than silently falling back to
  gather-only.

## 7. Out of scope

- Direct-paged at B=1 (needs B≥2; becomes relevant at M2 capture sizes).
- B≥2 concurrent-decode measurement (add to §4 or explicitly exclude
  from M3 decision — do not leave it implicit).
- fp16-Q kernel variant, sink/alibi/cascade support, TP.
- Changing the Triton backend (P3-3b owns that fallback).
- KV connectors / disaggregated prefill with LEGACY=0 (guard with
  log/assert per RC2 invariant note; do not develop compatibility).
