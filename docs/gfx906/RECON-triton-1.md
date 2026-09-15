# TRITON-1 recon — what the gfx906 Triton port carries, and what a move to upstream costs

> 2026-09-15, branch `gfx906/triton-1`. Roadmap item
> [`TRITON-1`](ROADMAP.md). Fork under test: `/local/git/triton-gfx906`
> (remotes: `origin` = ai-infos/triton-gfx906, `upstream` = triton-lang/triton).

**Question.** vLLM 0.29's `requirements/build/rocm.txt` nominates
`triton==3.7.1+git0263a6a6` (commented out in our tree); we instead run an
editable build of the fork at **3.6.0+gfx906** (`triton 3.6.0+git82957a51`). Do we
still need the fork, and if we move to upstream, what must we carry with us?

## 1. The fork carries exactly one thing: 7 lines of ISA classification

Its history is upstream source drops with one gfx906 commit after each
(`7d7fe9648b` v3.5.1 drop → `90c4b262b0` +6 lines; `b9e30e974a` v3.6.0 drop →
`82957a5112` +7 lines). Every other commit is upstream. So there are **no other
local optimizations** to carry forward.

`82957a5112`, 3 files, 7 lines — all of it, with what each line buys:

| line | file | what it buys |
|---|---|---|
| `GK_GFX906 → ISAFamily::VEGA20` | `TargetUtils.cpp` (`deduceISAFamily`) | the arch mapping; without it gfx906 resolves to `Unknown` |
| `case ISAFamily::VEGA20: return 64` | `TargetInfo.cpp` (`getWarpSize`) | **critical** — the default is **32**, gfx906 is wave64 |
| `case ISAFamily::VEGA20` in `supportsVDot` | `TargetUtils.cpp` | enables `v_dot4_i32_i8` / `v_dot2_f32_f16` for integer dots (**performance**) |
| `case ISAFamily::VEGA20` with CDNA3 in `supportsDirectToLdsLoadBitWidth` | `TargetInfo.cpp` | direct-to-LDS limited to 32-bit (**perf + correctness**) |
| `case ISAFamily::VEGA20` in `isCDNA` | `TargetUtils.cpp` | family **alias** — the questionable part, see §3 |

Two further answers come out correct *by default* and need no case:
`getSharedMemorySize` → 64 KB (gfx906's LDS size) and `getMfmaVersion` → 0, i.e.
FMA-lowered dots. The 3.6 fork therefore does **not** use MFMA on gfx906.

## 2. Upstream still has no gfx906 support — and VEGA20 was *removed*

Checked `upstream/main`, `v3.7.1` and `v3.8.0`: no `GK_GFX906` mapping anywhere,
and `ISAFamily::VEGA20` no longer exists — v3.7.1's enum is
`{Unknown, CDNA1..4, RDNA1..4, GFX1250}`.

So the port is **not** "re-apply 7 lines": VEGA20 has to be *reintroduced* (enum
member plus a case at every site). Port surface, counted as `switch` sites over
`ISAFamily` in `third_party/amd`:

| base | sites | files | notes |
|---|---|---|---|
| 3.6.0 (today) | 6 (1 enum + 5 cases) | 3 | the current patch |
| **v3.7.1** | **11** | 4 | `TargetInfo.cpp` ×6, `TargetUtils.cpp` ×3, `LoadStoreOpToLLVM.cpp` ×1, `AccelerateAMDMatmul.cpp` ×1 |
| v3.8.0 | 15 | 6 | family/feature logic **restructured** into `Dialect/TritonAMDGPU/IR/TargetFeatures.cpp` (6 sites) — a feature-model port, not a case list |

**Recommendation: v3.7.1.** It is the version vLLM 0.29 nominates, so the API
surface is what upstream expects; 3.8.0 adds four sites and a restructured
feature model for no functional need.

## 3. The semantic decisions (the real work, not the lines)

- `getWarpSize` → 64, `supportsVDot` → true, `supportsDirectToLdsLoadBitWidth` →
  32-bit: carry as-is.
- `getMfmaVersion` → 0 (FMA dots) must **stay** 0. Aliasing gfx906 to CDNA here
  would emit gfx908/CDNA MFMA shapes that gfx906 does not have.
- `isCDNA()` is the hazard. The fork aliases VEGA20 into `isCDNA` wholesale; on
  v3.7.1 that function gates only three sites — `MemoryOpToLLVM.cpp:456`,
  `TargetInfo.cpp:386`, `UpdateAsyncWaitCount.cpp:371` — and each needs its own
  decision rather than a blanket alias, because upstream has since added
  **CDNA-only features** (async copies, buffer atomics, TDM/Gluon paths) that
  gfx906 lacks. A blanket alias can turn those on.
- The remaining sites take conservative defaults; they need an audit, not code.

## 4. Screens (what a new Triton has to keep working)

Triton-touching code we actually depend on:

- **the GDN decode kernel** — `qwen_gdn_linear_attn.py:526` logs
  `GDN decode kernel: triton`; every hybrid model we serve (Qwen3.5/3.8 dense and
  MoE, Nemotron) runs it on the decode path;
- `vllm/model_executor/layers/mamba/ops/` (`replayssm_config.py`,
  `ssd_combined.py`), `vllm/v1/attention/backends/mla/triton_mla.py`;
- the **flash-attn Triton-AMD** path — the ViT fallback (`GFX906_FA_VIT=0`) and
  the platform-level FA requirement — compiles through the same Triton.

Gates, in this order: build → FA suite (97) → in-process PPL probe
(dense 27B, expected **10.5516 / 359 tokens**, bit-identical to V1 and 0.28) →
serving A/B (MTP k=3, agentic corpus) → ViT-fallback smoke with
`GFX906_FA_VIT=0`. Plus: confirm nothing we use is Triton-version-pinned
(`triton_prefill_attention`, `vllm.triton_utils`, and the `triton_kernels`
build variable `TRITON_KERNELS_SRC_DIR` in the fork's build recipe).

## 4b. The port, as implemented (2026-09-15)

`/local/tmp/triton-v371` = a **worktree of upstream `v3.7.1`** (tag `f797708c06`,
LLVM pin `1f126a6dea50…`, triton `3.7.1`), branch `gfx906/triton-1` in that
repository, commit **`549ce65245`**: **28 insertions / 2 deletions across 5
files** — smaller and more precise than the fork's blanket alias:

| site | change |
|---|---|
| `TargetUtils.h` | re-add `ISAFamily::VEGA20` (upstream dropped it after 3.6.0) |
| `TargetUtils.cpp` `deduceISAFamily` | `GK_GFX906 → VEGA20` |
| `TargetUtils.cpp` `supportsVDot` | VEGA20 → true |
| `TargetInfo.cpp` `getWarpSize` | VEGA20 → 64 |
| `TargetInfo.cpp` `supportsDirectToLdsLoadBitWidth` | VEGA20 with CDNA3 → 32-bit |
| `MemoryOpToLLVM.cpp` (BarrierOp) | name VEGA20 explicitly (parity with the fork) |
| `UpdateAsyncWaitCount.cpp` | name VEGA20 explicitly — its own comment says the pass exists for GFX9 parts with direct-to-LDS and no async loads |

**Deliberately *not* done:** adding VEGA20 to `isCDNA()`. The fork aliased it, but
on v3.7.1 that function also gates CDNA-only features (async copies, buffer
atomics, TDM, direct-to-LDS scattering) that gfx906 does not have; the two sites
above get the behaviour by name instead. Left at upstream defaults, all correct
for gfx906: `getMfmaVersion` 0 (FMA dots — gfx906's MFMA shapes are not CDNA's),
`getSharedMemorySize` 64 KB, `requiresAliasInfoForAsyncOps` false,
`supportsDirectToLDSScattering` false, `supportDppBroadcast` false (a *perf*
candidate, not parity — gfx906 has DPP, so this is a follow-up A/B).

**Build:** `pip wheel . --no-build-isolation --no-deps` in the worktree, using the
vLLM venv (cmake 3.31 / ninja / pybind11 present). Triton downloads a prebuilt
LLVM for its pin (`~/.triton/llvm/llvm-<hash>-ubuntu-x64`; the 3.6.0 one is already
cached at 8.2 GB), so this is a download plus a compile — CPU-only, no GPU
impact, runnable while the GPUs do release work. **Nothing is installed into the
venv until the wheel exists and every screen is ready to run**, so the live
`3.6.0+gfx906` editable install stays untouched; rollback is
`pip install -e /local/git/triton-gfx906`.

## 5. Options

- **A — stay on 3.6.0+gfx906.** Zero work, known-good. The fork remains a
  source-drop fork (not rebase-able), and drifts further from what upstream vLLM
  expects.
- **B — port to v3.7.1 (recommended).** Re-add `ISAFamily::VEGA20`; ~11 sites;
  per-site `isCDNA` decisions; keep it as a **patch series on upstream v3.7.1**
  (branch off the tag, so it rebases) instead of another source drop; then §4.
- **C — port to v3.8.0.** As B plus the `TargetFeatures.cpp` feature model:
  more surface, same outcome, no functional gain today.

Cost note: a full Triton build is CPU-only (tens of minutes) and can run while
the GPUs are busy with release/serving work.
