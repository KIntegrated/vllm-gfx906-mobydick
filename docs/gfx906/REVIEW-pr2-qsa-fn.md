# PR #2 review (joochung) — the first end-to-end Qwen3.8-Flash-Next result, and what to merge

> Review of `KIntegrated/vllm-gfx906-mobydick#2` (head `joochung:gfx906/qsa-fn`,
> base `main`) and its sibling `#1` (PLE mmap offload), plus the test run behind
> them. Reviewer: this session. **Verdict: merge-worthy in parts — take 6 of their
> commits with 3 required edits; drop 2 sets; ask for 2 test artifacts.**

## Headline: the FN-8 gate is met

The tester served the real checkpoint on **4× MI50 (32 GB, PCIe-only, no XGMI)**,
`TP=4`, fp16, `max-model-len=147456`, `max-num-seqs=3`, MTP k=3, piecewise
cudagraphs, and measured **46.8 t/s at B=1**. That is the end-to-end number the
whole train was missing (we cannot load it on 2 cards), and it confirms the fp16
enablement (QSA-FN-1) works on the real model rather than only in kernel probes.

Recorded findings worth keeping (generic or model-specific, not box-specific):

- fp16 QSA is the working arm; bf16 unchanged at ~1.00× — consistent with our
  probes (fp16 `v_dot2_f32_f16` vs emulated bf16).
- **Drafter CUDA graphs are worth +6 % and cost 2.3 % of the KV cache** with MTP
  k=3 on this stack. Their `enforce_eager` attempt was a no-op: that flag is read
  only by the legacy `v1/spec_decode/` proposer, never by
  `v1/worker/gpu/spec_decode/`, and is not propagated to the draft `ModelConfig`.
- **Boot segfault (MTP + graphs)**: dies inside `gc.collect()`'s traversal
  (`deduce_unreachable`), i.e. GC is the *victim* of a heap corruption whose
  culprit is unidentified (their prime suspect: a stale tvm_ffi torch-C-DLPack
  addon keyed by filename without a torch-version hash). `gc.freeze()` +
  `gc.disable()` do **not** prevent it (an explicit third-party `gc.collect()`
  still traverses); shadowing the `gc.collect` attribute for the region does.
  **Root cause OPEN; the guard is a mitigation, not a fix.**
- `tvm_ffi` installs its own `SIGSEGV` handler and *replaces* `faulthandler`'s,
  which is why their crash dumps had no Python frames.
- Their topology has **no P2P** (each GPU behind its own root port + its own AMD
  `14a1` switch DSP; `ACSCtl ReqRedir+ CmpltRedir+`): a reconstructed fullbar
  amdgpu patch was rejected — ACS redirect is the real blocker. Dead end, worth an
  `AGENTS`-level note rather than a code change.

## Read the history shape first: do not merge their branch

Their branch's merge base with ours is `f79ebf2d44` (pre-QSA), and it **re-commits
our own work with new SHAs** (they applied the tester bundle, then merged `main`),
so a branch merge would duplicate ~20 commits of our history and fight our
`gfx906/v0.29.0-final` release. Cherry-pick their unique commits instead, and ask
them to re-target the PR at **`gfx906/qsa-fn`**.

## Their unique commits, triaged

| commit | what | call |
|---|---|---|
| `d65391dc61` | PLE ngram table → pinned host RAM via mmap-shaped CPU shards (`MmapShardedNGramEmbedding`), replacing the TP-sharded on-device embedding | **TAKE, but gate it** (see edit 2) |
| `745c35e098` | PLE ngram D2H-sync fix (stale shard index) + graph-safe shard loop; V2 GC-freeze guard; `GFX906_DRAFTER_GRAPHS` | **TAKE with edit 1**; the two PLE correctness fixes are the valuable part |
| `0aae70351e` | GC freeze: shadow `gc.collect` (the part that actually stops the crash) | **TAKE, comments trimmed** |
| `4ca518b908` | GC guard exit path: park-at-exit by default instead of unfreeze+collect (`GFX906_GC_THAW` restores the fatal exit) | **TAKE** |
| `d37e9d2aae` | `style:` repo pre-commit over 19 files | **TAKE the Python hunk only** (our files are not all ruff-format clean); it is mechanical |
| `23c2115c72` | `style:` whole-file clang-format of `csrc/rocm/dense_gemv_gfx906.cu` (273+/293−) | **DROP** — keep only their +34 content lines; a 600-line whitespace rewrite of a hand-tuned fork file hides real changes and conflicts |
| 4× 1-line commits | append rows to `docs/gfx906/degradation.md` with their ops narrative (`/ai/...`, pids, `run-vllm`, `BOOT_TRIES`, cache paths) | **DROP/REWRITE** — keep the three generic findings above, without the box-specific runbook |
| `ae6492f7c2`, `ac830d0e6a`, `78279001b5` | merge of `main`; our handover re-applied; a "log PR #2" line | drop (already ours / bookkeeping) |

Also trivial and takeable: `# shellcheck disable=SC2086` pragmas in both serve
recipes, and the missing SPDX line in `_qsa_tiny_model.py`.

## Required edits before merging

1. **Invert the drafter-graph default.** As written, an unset
   `GFX906_DRAFTER_GRAPHS` gives the drafter `CUDAGraphMode.NONE`, i.e. it
   **changes default behaviour** for every spec-decode run — contradicting the
   PR's own claim that "unset env ⇒ upstream behaviour". Our house recipes serve
   MTP k=3 with graphs, so their default would cost ~6 % (the inverse of the +6 %
   they measured). Required form: default = inherit the target's mode (upstream);
   `GFX906_DRAFTER_GRAPHS=0` disables, with their measurement in the note.
2. **Gate the PLE host-table path** behind `GFX906_PLE_HOST_TABLE=1` (default
   off). It is a *structural replacement* of the TP-sharded embedding — also not
   covered by the PR's "all changes gated" claim — and our validated path must
   stay the default; a 4×32 GB / 147 k-ctx config needs it on, which the dev-log
   note should say. It also swaps `dtype=params_dtype` for a `runtime_dtype`, so
   the two paths need a shared dtype contract.
3. **Trim the code comments to house style.** Several of their additions are
   100+-line dev-log essays inside `model_runner.py` citing boot timestamps, pids
   and their log files. Keep the mechanism (a few lines) + a dev-log pointer; the
   narrative belongs in a log, and the repo style says minimise comments. Also drop
   the `# or a cheaper sanity check` aside they left in the `rocm.py` amdsmi
   fallback — that fallback itself is a **generic, takeable robustness fix**.

## Ask them for

- The **GC-guard unit harness** (they cite 18 assertions incl. nesting, exception
  path, enabled-state restore) and the PLE-path test they used. Both are referenced
  in commit messages but are not in the PR; a shadowing guard and a structural PLE
  swap are not mergeable credit-blind.
- Re-target to `gfx906/qsa-fn`, and correct the PR body's two "default-off ⇒
  byte-identical" claims (drafter graphs, PLE host table).

## Verification we can and cannot do here

Can: build (the `.cu` content lines), `tests/models/qwen4_exp/` +
`tests/kernels/mamba/` + `tests/kernels/moe/test_c4_layer0_quant.py`, the tiny rig
(which exercises PLE + mamba align + QSA in one server), the house 35B bench
(59.77 t/s baseline) and the PPL probe (15.9361) for non-regression.

Cannot: 4×MI50 TP=4, 147 456 ctx, the boot-segfault class (never reproduced on
this box) — those stay the tester's evidence.

## Next actions

1. Cherry-pick the 6 commits above onto `gfx906/qsa-fn`, applying edits 1–3 and
   dropping the `.cu` reformat + the degradation rows.
2. Build, run the suites + tiny rig + the two house gates; record in
   `CHANGELOG.md` / `DEVLOG-qwen38-flash-qsa.md` as **QSA-FN-9** (tester result +
   merged fixes).
3. Ask the tester for the two test artifacts before the merge lands.
