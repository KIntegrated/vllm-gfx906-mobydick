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

**Caveat — corrected 2026-09-15 (full interleaved dataset in the follow-up
block below):** the A/B's acceptance difference **cannot be attributed to the
triton build**. Re-running *the same build* (3.8.0) on *the same corpus body*
(`prompt_sha1 795844ca5794`) in a second process gave acceptance **2.1875 → 1.8132
(−0.374)**, i.e. a within-build process-to-process swing as large as the
cross-build delta (−0.424 at that body, −0.247 at the next). The fork's single
sample (1.7634) sits inside 3.8.0's two-sample range, and ms/step shows the same
per-process spread (82.8 vs 86.0 = 3.9 % on that body).

Consequences:

- **Acceptance/t/s cannot carry a build comparison at all here.** The A/B ran the
  arms *sequentially* (3.8.0 first, fork second), so arm order is confounded with
  the build, and per-process variance is of the same magnitude as the effect being
  measured. Any future triton/patch A/B must either interleave arms (A,B,A,B) or
  repeat the first arm last, and must lead with **ms/step**.
- ms/step itself has ~2–4 % per-process spread, so "parity" here means *no
  difference beyond that spread* — a <2 % regression would not have been
  detectable at n=2 arms × 2 reps. That is a limitation of the verdict, not a
  claim of exactness.
- The mechanism of the per-process variance is **not pinned**. Candidates: the
  chunked-prefill GDN path is `@triton.autotune`d (`ssd_bmm.py`, timing-based
  config choice per process → different fp accumulation orders → different GDN
  state → different trajectory), host/kernel state, or allocation-dependent
  kernel selection. What the evidence excludes is a *build* cause.
- The gates that do survive as build comparisons are the deterministic ones: FA
  suite (97), the in-process PPL (10.5472 vs 10.5516, −0.04 %) and the greedy
  target-text identity (byte-identical at 32 tokens) — plus the ViT-fallback
  smoke.

**Adoption state:** the A/B left the **fork** installed (the known-good default).
Switching to stock 3.8.0 is one command (install the wheel in
`/local/tmp/triton-v380/wheel/`); a fresh triton *version* recompiles every triton
kernel it uses on first boot (both arms booted ~450 s here with `NOCACHE=1`), which
is a one-time cost per version change, not a recurring one.

**Follow-up, interleaved A → B → A (2026-09-15, fresh boot, canary 38.7 t/s).** Same
corpus body (`prompt_sha1 795844ca5794`, 64k), acceptance and ms/step per *process*:

| build | process samples (acceptance, ms/step) |
|---|---|
| stock 3.8.0 | (2.1875, 82.8) · (1.8132, 86.0) · (1.7634, 82.4) · (1.7128, 87.7) |
| fork 3.6.0 | (1.7634, 83.7) · (**1.7634**, 81.6) |

Reading:

- **No mean-level build difference is detectable.** The A/B's apparent delta
  (2.1875 vs 1.7634 = 0.424) is *smaller than the same build's own
  process-to-process spread* (0.475 over four processes), and the fork's value
  (1.7634) is exactly reproduced by one of 3.8.0's own samples. ms/step spreads the
  same way (3.8.0: 82.4–87.7 = 6.4 %; fork: 81.6–83.7 = 2.6 %).
- **The only surviving build-flavoured hint is a variance asymmetry**: over four
  processes 3.8.0 spread 1.71–2.19 (ms/step 82.4–87.7) while the fork gave 1.7634
  twice (ms/step 81.6–83.7). At n=4 vs n=2 that is **not established** — it would
  take 2–3 more fork runs to test, and if real it is a config-selection instability
  in the newer Triton rather than a numerical difference.
- **Numeric differences exist, but they are prompt-dependent, not build-dependent.**
  128-token greedy probes (temp 0): the *code* prompt's continuation was
  byte-identical across **every** run of both builds (`16172ed9edfc`), while the
  *prose* prompt's differed **between two runs of the same build**
  (`43531fd7aa41` vs `f93789285849`; the fork produced a third variant). So
  close-call argmax flips happen per process on sensitive prompts, on either build.
- Consistent with that: the in-process PPL probe (359-token prompt, no chunked
  prefill) was bit-reproducible per build (10.5516 on the fork across many earlier
  runs; 10.5472 on 3.8.0) — long-context measurements are where the per-process
  variance enters, which points at the **autotuned chunked-prefill GDN kernels**
  (`ssd_bmm.py`) as the leading mechanism candidate (timing-based config choice per
  process → different accumulation order → different long-context trajectory).
  Unpinned; a test would be to force a single autotune config and re-measure.

**Closed: the `supportsDirectToLdsLoadBitWidth` "gap" is inert, in both builds.**
Checked the mechanism rather than the symptom (the parity A/B showed no perf
difference, which is *why*): direct-to-LDS loads are only ever *created* for the
async-copy path, gated in `canBeConvertedToAsyncLoad` on
`{CDNA3, CDNA4, GFX1250}` in v3.8.0 — and in the fork's own 3.6.0 that list is
`{CDNA3, CDNA4}`, i.e. **VEGA20 was excluded there too**. So the fork's `VEGA20`
case in that switch was dead code, and v3.8.0's missing `GCN5_1` case is equally
unreachable; the function is only consulted from inside async-copy lowering/
coalescing, which gfx906 never enters. There is no behavioural gap and nothing to
measure; the only residue is a latent inconsistency that would fail **loudly**
(the `LoadStoreOpToLLVM` asserts) if upstream ever opens that path for gfx906.

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
