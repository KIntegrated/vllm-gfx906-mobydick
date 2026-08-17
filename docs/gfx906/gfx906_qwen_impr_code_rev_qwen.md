# Code review — `gfx906/moe-opt` (Qwen3.5 improvements) before merge into `gfx906/main`

Copyright Kevin Read <me@kevin-read.com>

Date: 2026-08-17. Reviewer: Pi (coding agent), independent pass over the
final branch state (`613caa72c6...HEAD`, 67 commits, 50 files, +12 870/−19).
Method: full read of every changed production file (C++ kernels, pybind
layer, Python backend/runner/oracle, build system), mechanical audits
(lint, headers, env-knob census, TODO scan), and re-run of the test
suites on the gfx906 box. Prior in-branch reviews (phase 2 code-rev
`08387aad9d`, phase 3 fixes `5a0c45988f`) are assumed; this review is for
the *merged* state.

## 1. Verdict

**Ready to merge after the four P2 fixes below** (each is small; none
touches the hot path). No P1 findings: nothing in the default
configuration (`GFX906_FA_LEGACY=1`, production env) is broken, and the
default-configuration behavior changes are measured, gated, and
documented. The kernel code quality is high: the most bug-prone classes
in this branch (non-contiguous KV-cache strides, CUDA-graph buffer
lifetimes, NC2 packing invariants, K-split atomic accumulation) all have
both the fix *and* the regression test.

Re-verified during this review, on the final per-file-max-ilp build:

| gate | result |
|---|---|
| `pytest tests/kernels/attention/test_gfx906_fa.py` (15) + `moe/test_gfx906_moe_gemm.py` (12) + `test_rocm_unquantized_gemm.py` (21) | **46 passed, 2 skipped** (the 2 skips are by-design wvSplitK-on-gfx906) |
| PPL dense 27B / MoE 35B | 6.7122 / 6.6832 — both inside the historical bands |
| serving records (this build) | dense **25.60 t/s**, MoE **67.39 t/s** (1.04× gap to llama.cpp 70.3) |
| new-file copyright audit | all new files carry SPDX + Kevin Read line; vendored files keep the upstream (Nick) copyright alongside |

## 2. What the branch does (behavior-change inventory)

1. **CUSTOM Q8 FlashAttention backend** (`csrc/gfx906_fa/`,
   `vllm/gfx906_fa/`): vendored llama.cpp-gfx906 kernel port wrapped as a
   vLLM v1 attention backend; **now the default attention backend on
   gfx906** (`vllm/platforms/rocm.py` priority) for fp16-KV decoder
   attention, head sizes 64/128/256, block sizes multiple of 16, no
   sliding window / sinks / softcap / cascade (all fail-closed with
   `NotImplementedError` at init).
2. **W4A16 MoE fused GEMM** (`moe_q_gemm_gfx906.cu` +
   `experts/gfx906_w4a16_moe.py` + oracle `GFX906_HIP` backend): the
   first-priority WNA16 MoE backend on gfx906 for AWQ-style group-scale
   checkpoints; weight repack at load.
3. **Dense M=1 GEMV** (`dense_gemv_gfx906.cu` + `layers/utils.py`
   routing): K=2048 rows (N∈{1,256,≥2048}) and the K=17408 down_proj,
   gfx906-gated, `VLLM_GFX906_DENSE_GEMV` kill switch. Also routes
   m<4 (LLMM1 padding) on **all** ROCm — a generalization of upstream
   behavior, safe (same upstream kernel) but wider blast radius than the
   gfx906 shapes.
4. **GDN `core_attn_out` zero-fill skip** (default on, spec-decode-safe
   guard, `GFX906_GDN_EMPTY_CORE_OUT` kill switch).
5. **GemmaRMSNorm `(1+w)` cache** + fused-kernel dispatch on ROCm.
6. **Per-file `max-ilp` LLVM scheduler** in CMake for q_gemm / gfx906_fa /
   skinny_gemms (`VLLM_NO_MAX_ILP=1` off); explicitly *not* applied to
   the MoE routed kernel (measured −2.2% MoE serving).
7. Build: `_gfx906_fa_C` extension (gfx906 arch-gated), pyproject plugin
   entry point, `setup.py` `py_limited_api` param, hipify in-source guard,
   `weight_utils.py` fastsafetensors GDS fallback widening.

## 3. Findings

Severity: **P2 = fix before merge** (small, no perf impact), **P3 = fix
soon / follow-up issue**, **P4 = nit**.

### P2-1 — MoE oracle selects the kernel without shape gates; exotic
AWQ-MoE shapes crash at runtime instead of falling back

`oracle/int_wna16.py`: the `GFX906_HIP` priority is granted to *any*
gfx906 ROCm process with the op present, and
`_backend_incompatibility_reason` only checks `may_have_zp`. The kernel
side only checks `size_n % 4 == 0` (C++ `TORCH_CHECK`), and the C++
entry point does **not** check `size_k % 8`, `groups | size_k`, or
`size_n % 8` (qzeros row width). A hypothetical AWQ W4 MoE model on MI50
with, say, N % 4 != 0 or K % 8 != 0 would be *selected* by the oracle and
then fail (or, for the unchecked cases, read OOB) at first forward
instead of falling back to Triton as the oracle contract intends.
Realistic exposure is low (common shapes are all divisible), but the fix
is a few lines: add N/K/group divisibility to
`_backend_incompatibility_reason` (it has `moe_config` for the sizes),
and/or mirror the N%4 check into `Gfx906WNA16Experts._supports_*`.

### P2-2 — direct-paged path builds an fp16 Q buffer; the C API requires
fp32 → guaranteed `TORCH_CHECK` failure (dormant by default)

`gfx906_fa_paged.py` direct-paged branch: the buffer reuse check is
`q_pad_buf.dtype == query.dtype`, but the backend always allocates
`q_pad_buf` **fp32** while the query is fp16 → the check fails and the
fallback `torch.zeros(..., dtype=query.dtype)` allocates **fp16**, which
`forward_paged_direct` rejects (`q must be fp32`). The path is only
reachable when a Q8 side-buffer exists (`GFX906_FA_LEGACY=0`,
experimental) and `B≥2`/`Sq≤16`, so it never fires in the default
configuration — but the `GFX906_FA_DIRECT_PAGED` A/B knobs documented in
`running.md` would hit it. Fix: use the fp32 buffer (or cast into it) in
the direct branch; drop the `== query.dtype` condition.

### P2-3 — plugin entry point logs a full traceback on every
non-gfx906 vLLM startup

`pyproject.toml` registers `gfx906_fa = vllm.gfx906_fa.gfx906_fa_backend:register`
unconditionally; importing that module runs
`vllm/gfx906_fa/__init__.py` → `from vllm import _gfx906_fa_C`, a hard
import that fails on CUDA and on non-gfx906 ROCm installs. The plugin
loader catches the exception (graceful: backend simply not registered),
but via `logger.exception` — so every such startup prints a traceback.
The explicit-registration path in `rocm.py` already handles this with
`try/except ImportError`; the plugin path does not. Fix: make
`register()` (or `__init__.py`) import-tolerant and return early when
the extension is absent.

### P2-4 — `setup.py` `_targets_gfx906()` can request an extension whose
CMake target does not exist

`_targets_gfx906()` returns True when `PYTORCH_ROCM_ARCH` is *empty*
(auto-detect), but the CMake target `_gfx906_fa_C` is only defined when
`VLLM_GPU_ARCHES MATCHES "gfx906"`. On a non-gfx906 ROCm machine with
arch auto-detection, setup.py appends `CMakeExtension(name="vllm._gfx906_fa_C")`
and the build fails on a nonexistent target. Fix: add the extension only
when `"gfx906" in rocm_arch`, or have the build tolerate the missing
target (the CMake gate is the source of truth).

### P3-1 — LEGACY=0 + prefix caching: `logger.error`, not fail-closed

`gfx906_fa_backend.py` `get_cudagraph_support` logs an error (corruption
warning) but continues. Given the documented corruption mode, this
combination should raise (behind an explicit override env if a
diagnostic use case exists). The README already says "do not use"; the
code should agree.

### P3-2 — `_ensure_gather_buffers` retired-buffer growth is unbounded in
LEGACY=0

Exact-shape-match realloc + retired-list-keeps-alive means every
`Sk_pad` growth (decode grows it in 32-token steps over a long context)
after the first capture permanently retains the previous
K+V pair (~tens of MiB each at serving shapes). Inert in the default
LEGACY=1 (the buffers are `None`), but a memory leak on the documented
experimental path. A grow-only capacity buffer with an exact-size view
would fix it.

### P3-3 — `workspace13`/`fused_out` aliasing order dependency is
undocumented in `Gfx906WNA16Experts.apply`

`modular_kernel._allocate_buffers` makes `workspace13` and `fused_out`
views of one `common_workspace`; `apply()` relies on the sequence
gemm1 → activation → `output.zero_()` → gemm2 (the re-zero would wipe
`w1_out` if it ran earlier). Correct today; one reorder away from
silent corruption. Add the warning comment (or split the buffer for
this experts class).

### P3-4 — three new lint errors in production files

`ruff` (vs `gfx906/main`): `utils.py` I001 (unsorted `import os`),
`qwen_gdn_linear_attn.py` I001 (same), `int_wna16.py` SIM102 (nestable
`if` in the new GFX906 block). Pre-existing E501s in those files are
not branch-introduced. (The vendored `vllm/gfx906_fa/*.py` +
`benchmarks/kernels/gfx906/*` + `docs/gfx906/*` bench scripts carry 63
ruff errors — mostly B023/E501/F821 in timeit-closure bench patterns —
see P4-5.)

### P3-5 — the ncols1 tile ladder is copied three times

The `Sq>32→64 … Sq≤2→2` ladder appears in `forward_paged` (Python, for
`Sq_pad`), the C++ launcher dispatch, and the C++ kv_max expansion
(latter annotated "keep in sync!"). Any future change to the ladder
(e.g. a new NC2 cap) must hit all three; a divergence would mis-pad Q or
mis-expand kv_max. Consolidate (one place, or a comment cross-ref at
each) or at least a unit test pinning the mapping.

### P3-6 — `gather_paged_kv_quantized` allocates fresh outputs per call

Unlike its two siblings, the fused-quant gather (the one actually used
on the default LEGACY decode path) has no `use_or_alloc` grow-buffer
parameter; it `torch::empty`-allocates K+V per layer per step. Benign
under graph capture (private pool) and cheap in eager (caching
allocator), but inconsistent with the VRAM-spike rationale documented
for the other two. Low priority.

### P3-7 — MoE kernel contract partially unenforced in C++

Beyond P2-1's shape gates: `topk_weights` layout (flat [M*topk]),
`sorted_token_ids` id range, and `expert_ids` sizing are caller
contracts with no checks; a future caller (non-vLLM) could feed
mismatched metadata and get OOB. The vLLM-side caller is correct
(verified against `moe_align_block_size` semantics and covered by
`test_gfx906_moe_gemm.py`); noting for API hygiene.

### P4 nits

1. **Stale docstring**: `csrc/rocm/ops.h` and `vllm/_custom_ops.py`
   say `kchunk 512|2048|4096` — 1024 is a supported and *used* value
   (K=17408 down_proj).
2. **Stale pybind help**: `forward_paged_direct` docstring says output
   `[B, Hq, Sq, D]`; it is native BSHD `[B, Sq, Hq, D]`.
3. **Stale comment**: launcher header "Вед mask, sinks, KV_max (paged KV
   — TODO: vLLM block table)" — the paged block-table path is
   implemented.
4. **MoE kernel header** says `grid = (…, N/1024, …)`; only true for
   NPT=4 (NPT=2 → N/512).
5. **Lint debt in vendored/bench files** (63 ruff errors; the F821s are
   bench-script outer-scope names, verified harmless). Pre-commit will
   flag these if the files are ever restaged; consider a `per-file-ignores`
   entry or a cleanup pass pre-merge.
6. **`/tmp/bench/...` references in code comments** (e.g.
   `bench_dense_gemv_gfx906.py` docstring, `utils.py`): /tmp is volatile;
   use the in-tree `benchmarks/kernels/gfx906/` paths.
7. **Dead parameter**: `forward_paged(..., mask_buf=None, ...)` — Level-3a
   leftover; the parameter is ignored.
8. **Env-knob surface**: 21 knobs, 9 debug-only (`_DUMP`,
   `_DOUBLE_CHECK`, `_FWD_DEBUG`, …). All are read-once at import and
   documented in the README table; acceptable, but the debug ones are
   candidate for a single `GFX906_FA_DEBUG=1` master switch.
9. **Branch history**: 67 commits include heavy plan/version docs churn.
   History is internal to the fork, so no action required; a squash
   merge would keep `gfx906/main` bisectable for the kernel work.

## 4. Positive findings (what held up under scrutiny)

- **Stride handling is the branch's best work**: every cache read uses
  *real* tensor strides (the shape-derived-strides bug that read K bytes
  as V was found, fixed, and pinned by
  `test_fused_gather_matches_torch_gather_on_unbind_cache`, which
  mirrors the production `unbind(1)` allocation exactly).
- **Graph-capture buffer lifetime** is designed, not patched:
  capture-state latching + retired-list keep-alive for q_pad,
  q_pad_decode, and the class-level gather buffers, with
  `empty_cache()` explicitly removed from the forward path;
  `test_cudagraph_capture_replay_legacy_decode_path` and
  `test_q_pad_buffer_survives_capture_then_prefill_grow` cover the two
  failure modes.
- **Fail-closed dispatch**: NC2 packing rejects explicit GQA-invalid
  values (clamping would run the NC2=8 kernel and silently mispack —
  the comment says exactly this), kv_split is clamped at prefill (OOM
  math in the comment), head_dim/block_size/dtype checks at every
  pybind entry, oracle falls back instead of guessing.
- **Precision claims are evidenced, not asserted**: bit-equal
  gather-vs-quantize tests, fp32-reference kv_split/GQA-pack tests,
  fp16-CAS accumulation reordering A/B-diffed at integration, PPL bands
  tracked across every commit that touches numerics.
- **Kernel guards are correct where they matter**: wave64 full-warp
  reduction (mask 32..1), 16B-aligned vector loads verified against the
  alignment invariants, early-exit returns placed *before*
  `__syncthreads()` (no deadlock), uninitialized-LDS slots provably
  unread (K-tail guard), degenerate all-empty kv_split rows guarded in
  the combine kernel.
- **Defaults are the fast path with kill switches**, and the one
  measured-negative flag (max-ilp on the MoE kernel) was deliberately
  excluded rather than globally applied.
- **Docs match code** on every spot-checked claim (knob table, records,
  test counts, known limitations, roadmap N1–N3).

## 5. Resolution status (2026-08-17, after the combined review)

The combined review (`gfx906_qwen_impr_code_rev_claude.md`, repo root)
added four findings of its own; together with this document's list, the
following were fixed in the follow-up commit(s) right after this review
landed:

| finding | source | fix |
|---|---|---|
| repack `UnboundLocalError` on symmetric W4A16 (`zf` fall-through) + wrong symmetric zp fill (`8` vs `0x88888888` per nibble) | combined #1 | `int_wna16.py` both repack layouts; 12 new kernel-test cases (`*_sym`) |
| GDN zero-fill skip not platform-gated | combined #2 | `on_gfx906()` added to the condition |
| `GFX906_HIP` accepted GPTQ-style zp configs | combined #3, this doc P2-1 | oracle excludes `AutoGPTQConfig`/`QuantizationArgs`; new rejection + acceptance oracle tests |
| `top_k > 0` guard missing in MoE GEMM | combined #4, this doc P3-7 | `TORCH_CHECK` in `moe_q_gemm_gfx906.cu`; probe-verified |
| MoE oracle shape gates (N%4/K%8/groups) | this doc P2-1 | deferred — repack layout detection rejects unknown shapes loudly; revisit if a new layout is added |

Still open from this document: P2-2 (direct-paged fp16-Q), P2-3 (plugin
traceback), P2-4 (setup.py/CMake gate), P3-1…P3-6, P4 nits, and the
three new lint errors (I001 ×2, SIM102).

## 6. Suggested pre-merge actions (ordered)

1. P2-1 oracle shape gates (+ C++ K%8/groups checks if cheap).
2. P2-2 direct-paged fp32 buffer fix.
3. P2-3 import-tolerant plugin `register()`.
4. P2-4 setup.py/CMake gate alignment.
5. P3-4 the three lint errors (one-line fixes).
6. Re-run the three suites + PPL after 1–4 (1 and 2 touch selection
   paths; 3–4 are build/import only).
7. P3 items and P4 nits → follow-up issues (or fold into the same PR if
   trivial).
