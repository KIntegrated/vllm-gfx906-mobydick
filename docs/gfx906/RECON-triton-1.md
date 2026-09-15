# TRITON-1 recon — what the gfx906 Triton port carries, and what a move to upstream costs

> 2026-09-15, branch `gfx906/triton-1`. Roadmap item
> [`TRITON-1`](ROADMAP.md). Fork under test: `/local/git/triton-gfx906`
> (remotes: `origin` = ai-infos/triton-gfx906, `upstream` = triton-lang/triton).

**Question.** vLLM 0.29's `requirements/build/rocm.txt` nominates
`triton==3.7.1+git0263a6a6` (commented out in our tree); we instead run an
editable build of the fork at **3.6.0+gfx906** (`triton 3.6.0+git82957a51`). Do we
still need the fork, and if we move to upstream, what must we carry with us?

> **Answer, corrected 2026-09-15: upstream supports gfx906 natively since
> v3.8.0, so we need neither the fork nor a port — see §2.** (The first draft of
> this recon concluded "upstream still has no gfx906 support"; that was wrong: it
> was based on grepping for `ISAFamily::VEGA20`, the *fork's* name. Upstream
> landed the support under a different name, `GCN5_1`, and v3.7.1 predates it.)

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

## 2. Upstream *does* support gfx906 — since v3.8.0 (`GCN5_1`)

**`aa53dba7455` "[AMD] Add GCN5.1 / gfx906 target (#9628)"** (Luna Nova,
2026-03-06) introduced native gfx906 support in stock Triton, under a **new
family name**: `ISAFamily::GCN5_1`, mapped from `GK_GFX906`. It is contained in
**`v3.8.0`** and *not* in `v3.7.1` (which predates it). The commit's own
rationale matches both the fork's classification and this recon's conclusions,
independently:

> gfx906 has v_dot2_f32_f16 and v_dot4_i32_i8 via VOP3P but **no MFMA** … New
> `ISAFamily::GCN5_1` mapped from `GK_GFX906` with wave64, DPP broadcast, and
> `supportsVDot`; this is GCN so **not marked as RDNA/CDNA** and most gates don't
> need updated. `warpReduce` refactored from isCDNA/isRDNA negative enumeration to
> a `getIsaVersion()` check (gfx90a+ || gfx11+).

The five `GCN5_1` sites in v3.8.0: the enum (`TargetFeatures.h`),
`deduceISAFamily` (`TargetFeatures.cpp:85`), `getWarpSize` (grouped with CDNA →
**64** ✓), `supportDppBroadcast` (→ **true** — upstream enables DPP for gfx906,
which the 3.6 fork left off), and a read-lane/shuffle path in `Utility.cpp:170`.
`getMfmaVersion` stays 0 (FMA dots) and `getSharedMemorySize` stays 64 KB, both
correct for gfx906.

**Feature parity vs the fork (3.6.0+gfx906):**

| feature | fork | stock v3.8.0 `GCN5_1` |
|---|---|---|
| arch recognised | VEGA20 (patch) | **GCN5_1 (upstream)** |
| wave64 | patch | ✓ |
| `v_dot` | patch | ✓ |
| DPP broadcast | default (off) | ✓ **better** |
| MFMA | 0 (FMA) | 0 (FMA) |
| shared memory 64 KB | ✓ | ✓ |
| pre-CDNA2 warp-reduce exclusion | separate logic | ✓ via `getIsaVersion()` |
| **direct-to-LDS 32-bit** | patch (with CDNA3) | ✗ `supportsDirectToLdsLoadBitWidth` has **no GCN5_1 case** → `false` |

So exactly **one** item is missing versus the fork: the 32-bit direct-to-LDS
allowance. That is a one-line change in `TargetFeatures.cpp` (add `GCN5_1`
alongside `CDNA3`) and is a good candidate to offer upstream as a follow-up to
#9628 — but it only matters if a measurement shows the LDS-direct path is
actually taken and pays on gfx906; the screens below decide that.

**Consequence:** TRITON-1 reduces to *validating and adopting a stock release* —
no patch, no fork, no source drops. The 3.7.1 port in §4b is therefore
**superseded** (kept on the triton-repo branch as a reference; it is also an
independent implementation of the same design upstream chose).

**Version caution:** vLLM 0.29 nominates `triton==3.7.1+git0263a6a6`, so v3.8.0 is
*newer* than the nominated pin — the screens must watch for API drift
(`triton_kernels`, `vllm.triton_utils`, `triton_prefill_attention`).

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

## 4b. The 3.7.1 port, as implemented — **SUPERSEDED by §2** (kept for the record)

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

## 4c. Build recipe for stock v3.8.0 (what actually works here)

`/local/tmp/triton-v380` = worktree of upstream **`v3.8.0`** (tag `c01b6774b1`,
LLVM pin `5f07f818b51b…` → prebuilt `~/.triton/llvm/llvm-5f07f818-ubuntu-x64-1`),
built with the **same recipe as the known-good fork build** (`/usr/bin/cc` +
`/usr/bin/c++`, no `TRITON_BUILD_WITH_CLANG_LLD`), plus one variable:

```bash
env MAX_JOBS=14 TRITON_APPEND_CMAKE_ARGS="-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" \
  .venv/bin/python -m pip wheel . --no-build-isolation --no-deps -w wheel
```

- `TRITON_BUILD_WITH_CLANG_LLD=1` **must not** be set: it makes triton's CMake ask
  for bare `clang`/`clang++` on PATH, which this box does not have (ROCm's clang is
  at `/opt/rocm/llvm/bin`), and the failure is a confusing "not a full path".
- The 3.8.0 prebuilt LLVM's exported targets request an install-RPATH relink that
  the Ninja generator refuses (7 × `CMake Error at CMakeLists.txt:415`); CMake's
  own suggested `CMAKE_BUILD_WITH_INSTALL_RPATH=ON`, forwarded through
  `TRITON_APPEND_CMAKE_ARGS`, clears it.
- Wheel only: **nothing is installed into the venv** until the screens run, so the
  live `3.6.0+gfx906` editable install is untouched (rollback:
  `pip install -e /local/git/triton-gfx906`).

## 4d. Screens on stock v3.8.0 — all four gates passed (2026-09-15)

Wheel installed over the editable fork (rollback `pip install -e
/local/git/triton-gfx906`), same boot for the A/B, same vLLM build throughout.

| gate | result |
|---|---|
| **FA suite** | **97 passed** (`tests/kernels/attention/test_gfx906_fa.py`) |
| **in-process PPL** (dense 27B, V2) | **10.5472** (359 tokens, **0** top-20 misses) vs the fork's 10.5516 → **−0.04 %** |
| **greedy identity control** | a fixed 32-token temp-0 completion is **byte-identical** between the two triton builds — the PPL shift is fp-accumulation rounding below the argmax threshold, not a semantic change |
| **ViT fallback smoke** (`GFX906_FA_VIT=0`) | the upstream flash-attn/Triton-AMD ViT path compiles and runs under 3.8.0; both image requests returned sane output (1024×1024 TTFT 6.43 s, `sum_logprob` −2.229 vs the fork's −2.205 on the same `prompt_sha1`) |
| **serving parity** (MTP k=3, agentic corpus, 64k+120k, 2 reps, same boot) | ms/step **85.4 vs 85.8 @64k** (−0.5 %) and **127.9 vs 128.2 @120k** (−0.2 %) → parity |

**Caveat worth keeping:** *throughput* is not comparable across a triton change.
The two arms ran identical prompts (`prompt_sha1` matched) yet acceptance diverged
(2.19/1.74 vs 1.76/1.50 @64k) → t/s means of 34.9 vs 30.5 for identical per-step
cost. A different codegen perturbs the draft path's numerics just enough to send
spec-decode trajectories down different branches; **ms/step is the metric**, exactly
as the CAT-1 investigation concluded.

**Adoption state:** the A/B left the **fork** installed (the known-good default).
Switching to stock 3.8.0 is one command (install the wheel in
`/local/tmp/triton-v380/wheel/`); a fresh triton *version* recompiles every triton
kernel it uses on first boot (both arms booted ~450 s here with `NOCACHE=1`), which
is a one-time cost per version change, not a recurring one.

**Open, small:** the `supportsDirectToLdsLoadBitWidth` gap (§2) — worth a
measurement before carrying a patch, and a good upstream PR against #9628 either
way.

## 5. Options (superseded by §2 — kept for the record)

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
