# gfx906 changelog

This file records roadmap items that are complete, rejected, superseded, or
otherwise closed. Active work, deferred work, and changes that are local but
still need upstream merging remain in the roadmap files. Dates are landing or
merge dates where the repository history provides one; they are not necessarily
the date an investigation began.

## 2026-09-16 (KVLAYOUT-1 — LEGACY=0 default flip)

- **`GFX906_FA_LEGACY=0` (Q8 side-buffer KV read path) is now the default.** Verified
  against 0.29's fused KV-cache layout (#51718): PPL **10.5472 / 10.5460** vs **10.5472** for
  LEGACY=1 (the same to within the probe's own ~0.001 run-to-run spread), 0 top-20 misses in
  every run, and **−15.5 % / −19.1 % ms/step** (MTP k=3, dense 27B
  AWQ, 64k/120k; acceptance unchanged at 1.7634/1.7634/2.0476; interleaved L1 → L0 → L1,
  so the order control is included — L1's own repeat ran faster and L0 still beat it).
- The fail-closed refusal and its `GFX906_FA_LEGACY_ALLOW_UNVERIFIED` override are
  removed as obsolete, and the always-on "experimental read path" warning is demoted to
  debug. `GFX906_FA_LEGACY=1` remains as the rollback: ~6 % faster for B=1 greedy decode,
  the one regime it wins (`DEVLOG-fa-legacy0-b1-decode.md`).
- Tests: the five LEGACY=1-assuming tests (gather-buffer lifecycle, q_pad capture grow,
  the A3 fused-loop contract, the gather-retire warning) now pin `GFX906_FA_LEGACY=1`
  themselves, and the fail-closed test became
  `test_legacy_default_is_side_buffer_after_kvlayout1`.

## 2026-09-15 (MUSE-1 first signals)

- **Muse-Glimmer loads and runs on both runners, and V1/V2 generation is
  byte-identical** (greedy, two raw-text prompts) — the first parity evidence for
  the model, and the same class of control as the triton greedy-identity test.
- **The in-process PPL probe cannot gate it**: Muse-Glimmer is a VLM
  (`MuseGlimmerForConditionalGeneration` + `vision_config`) and the probe renders
  its prompts through the chat template and profiles the encoder cache, reporting
  **362/363 top-20 misses on every arm** (PPL 36.12 V1 / 36.19 V2) while raw-text
  generation is sane — a prompt/template artifact, unlike Gemma-4's genuinely broken
  output. Its gate has to be a serving A/B.
- The `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` arm (V2) was killed by a GPU wedge
  (#93, `hipErrorLaunchFailure`), so whether stock Triton 3.8.0 makes that fork-era
  workaround unnecessary is still open; the arms need interleaving to survive the
  load lottery.

## 2026-09-15 (TRITON-1 rebuild + MUSE-1 gate + DFL2-1 prep)

- **The in-tree extension rebuild with stock Triton 3.8.0 passes** (task 2 of the triton
  adoption): `setup.py build_ext --inplace` rc=0, then `import vllm` (0.29.0 + triton 3.8.0)
  and the **FA suite 97/97**. The release build recipe therefore works on the adopted
  triton; no patches were needed.
- **MUSE-1 gate: V1/V2 parity at the first token, and the RBLOCK workaround is obsolete.**
  Clean arms (V1, V2, V1-repeat; `RBLOCK` unset) all start with the same confident token
  (`328` @ 0.00 — Glimmer's `to=self` recipient marker) and V1 reproduces its ranks 2–5
  exactly; V2 differs by ≤0.7 at −18…−25. Since the fork-era crash was a **triton 3.6.0**
  defect, `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` can be dropped from Muse-Glimmer recipes.
  Its answer-quality gate must be a **serving A/B** (its output is its own recipient/
  reasoning format, and text is not stable across processes), which is also the 0.32 V1
  deadline item.
- **Post-wedge NaN caveat recorded**: the first gate session's arms returned `nan` logprobs
  because they ran straight after wedge #94; the clean re-run gave rc=0 and 0 NaNs. NaN
  output is a post-wedge symptom — do not treat a post-reset session's numbers as evidence.
- **DFL2-1 prepared**: drafter `incoai/Qwen3.8-27B-DFlash2` downloaded (3.6 GB; 5 layers,
  hidden 5120, 32/8 GQA, head_dim 128, `is_causal: false`, sliding-2048), the chain patch
  touches one file, and `DFlash2DraftModel` is already registered. Subtasks added: a
  **baseline DFlash2-vs-MTP-k=3 comparison without the patch**, then **rocprofv3 kernel
  attribution**, **microbenches** at the real shapes, and a **HIP-feasibility assessment per
  top kernel** (starting with which attention backend the drafter selects — its attention is
  bidirectional, which our custom FA already serves).

## 2026-09-15 (0.29.0 release readiness)

- **Validation matrix on the 0.29.0 line (release candidate state):** FA suite **97/97**;
  in-process PPL on the dense 27B **10.5516** (bit-identical to V1 and the 0.28 line, so the
  V2 bring-up, VIT-1 and the `kv_split` override left the text path numerically untouched);
  MoE 35B 57.97 t/s (v2 restamp 58.36, 0.28 record 58.43); Nemotron PPL 26.9937 (band
  26.96–27.02); Ornith 16.6664 (fork 16.7824); Gemma-4 gated via its chat template
  (templated V1/V2 identical, logprobs ≤0.05); Muse-Glimmer gate in flight.
- **Default flips on this line:** V2 model runner (dense 27B, MoE 35B, Nemotron, Ornith,
  Gemma-4); **VIT-1** (ViT attention on the custom FA: −11.5 % image-prompt TTFT @1024²,
  −55 s fresh-boot Triton JIT); **stock Triton 3.8.0** (upstream gfx906 support, fork kept
  as rollback); MTP k=3 spec config. Muse-Glimmer remains the only V1 pin.
- **Published:** `unverbraucht/vllm-gfx906:0.29.0-rocm-7.14` + `:0.29.0-e730ef4066`
  (digest `sha256:bf3caead…`; built from `preset.0.29.0-rocm-7.14-kintegrated.sh` pinned to
  `gfx906/v0.29.0` @ `e730ef4066`; verified in-container: vllm 0.29.0, transformers 5.15.0,
  triton 3.6.0+gfx906, torch 2.13.0+gfx906, FA extension loads). Note the image still ships
  the triton fork; a triton-adopting image needs its own tag.
- **Open before publishing the next artifact:** push the branch (needs the `gh` `workflow`
  scope or the user's credentials — 18 commits ahead of the pushed `e730ef4066`), the
  in-tree extension rebuild against stock Triton 3.8.0, and a triton-adopting image build.

## 2026-09-15 (IFT gate tool + the PPL probe's real limit)

- **`BENCH_CHAT_TEMPLATE=1` does not make the PPL probe valid for IFT checkpoints** —
  verified, not assumed: templating Gemma-4's prompts makes the number *worse*
  (PPL **1278491** vs 84261 raw, 0 top-20 misses in both) because prompt-logprob PPL asks
  the model to predict the **user's** tokens, which an instruct model was never trained to
  model. Both are prompt-format artifacts.
- So the shipped gate for those models is a new repo tool,
  **`benchmarks/kernels/gfx906/ift_chat_gate.py`**: renders each prompt through the model's
  chat template and prints the greedy continuation plus the first-token top-k logprobs —
  a confident first token (≈0.00) with a sensible completion is the signal, and running it
  under two configurations gives a parity gate by comparing text and logprobs (that is how
  Gemma-4's V2 validation was done). The prompt-format note, `running.md` and the probe's
  own warning now point there instead of promising the flag fixes it.

## 2026-09-15 (GEMMA4-1 CLOSED + prompt-format guards)

- **Gemma-4's V2 gate passed**: a *templated* in-process V1-vs-V2 comparison gives
  identical text and logprobs agreeing to ≤0.05 (`'Paris//'` at 0.00), so its V1 pin is
  lifted on evidence. The model was never broken (raw text on an IFT checkpoint returns
  garbage; the same garbage reproduces on the 0.28 image).
- **Prompt-format guards shipped** so the trap cannot repeat: the PPL probe warns loudly
  when a tokenizer has a chat template while `BENCH_CHAT_TEMPLATE` is unset and renders
  prompts through the template when set; the throughput harness records
  `prompt_form`/`has_chat_template` and warns that tokens/s is not a correctness gate; and
  `docs/gfx906/README.md` now carries a "Prompt format — read before gating any model"
  block with the raw-text-valid vs template-required model split.
- Lesson recorded (with AGENTS.md): the Gemma-4 row already said "chat template
  required", but an aside in a notes column is not a guard — and a speed number is not a
  correctness gate.

## 2026-09-15 (GEMMA4-1: the model is fine, the gate was wrong)

- **Gemma-4 was never broken and there is no 0.28→0.29 regression.** Two probes misled
  this item: (a) the in-process PPL probe (degenerate 84261/108909, 0 top-20 misses —
  a prompt/format artifact), and (b) my raw-text generation probe on 0.29, which showed
  garbage and was then *reproduced byte-identically by the 0.28 image*, proving the
  lines agree rather than that the model is broken. Root cause of both: Gemma-4 is an
  **instruction-tuned** checkpoint that does not continue raw text. With its chat
  template it answers correctly and confidently — `'Paris//'` at first-token logprob
  0.00, and a correct Python function snippet at ≈0.00 (`/local/tmp/b4/gemma_diag.log`).
- So the model's "supported, 67.79 t/s" row is a *speed* claim that was never backed by
  an output gate — the same class of proxy error as the acceptance and CAT-1 lessons.
  The gate for it (and for its V2 flip) is a **templated** in-process comparison or a
  serving A/B; the templated V1-vs-V2 parity run is in flight.

## 2026-09-15 (0.29 line: missing-model investigation)

- **(CORRECTED 18:55 — see the next entry: Gemma-4 is *not* broken and it is not a 0.29
  regression; the raw-text probe below was invalid for an instruction-tuned
  checkpoint, and the 0.28 image reproduces the identical output.)**
  ~~Gemma-4 is broken on the 0.29 line — a regression, not a probe limitation.~~
  In-process greedy completions at temperature 0 (V1) are garbage
  (`' it it it it most is it it ...'`, `'<|||||||로. ...'`), which is what the
  degenerate PPL (84261.54 V1 / 108909.96 V2) was telling us. The same checkpoint
  was the fastest model on record on the 0.28 line (67.79 t/s). The 0.29 merge
  rewrote a lot of Gemma-4 code (`gemma4_mm.py` +181 lines, `gemma4.py` +64,
  new `gemma4_dspark.py`, the Gemma-4 MTP/unified paths, 43 files / +1702 lines
  including the quantization utils) — that is the suspect area. GEMMA4-1 is
  reframed accordingly; it stays V1-pinned but V1 is *also* broken for it.
- **Muse-Glimmer: the AWQ-INT4 checkpoint pull is in progress.** 24 GB / 16 files,
  throttled unauthenticated at ~2.5 MB/s (~2.7 h); it goes to
  `/data/cache/huggingface/hub` because the `/local` cache has only ~27 GB free.
  On arrival: the V2 parity run plus a test of whether stock Triton 3.8.0 makes
  Muse-Glimmer's `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` workaround unnecessary
  (that env exists because the rblock variant compile crashed in the triton fork).

## 2026-09-15 (TRITON-1 adopted)

- **Stock upstream Triton 3.8.0 is now the default** (the ai-infos v3.6.0+gfx906 fork
  is retained as rollback only). Upstream has carried gfx906 since `aa53dba7455`
  `ISAFamily::GCN5_1`; all gates pass (FA 97, dense/MoE/Nemotron/Ornith numerics,
  ViT fallback, serving ms/step parity) and the install docs
  (`README.md`, `requirements/build/rocm.txt`, `running.md`) now describe building
  from the upstream v3.8.0 tag instead of cloning the fork. Two build gotchas and
  the PyPI-wheel-import segfault are recorded; adoption therefore carries a small
  build step.
- **The 0.29.0 docker image was built and published**: `unverbraucht/vllm-gfx906:
  0.29.0-rocm-7.14` and `:0.29.0-e730ef4066` (digest `sha256:bf3caead…`, both the
  same image), from `preset.0.29.0-rocm-7.14-kintegrated.sh` pinned to
  `gfx906/v0.29.0` @ `e730ef4066`. Verified in-container: vllm 0.29.0,
  **transformers 5.15.0** (the pinset's stale `5.7.0` violated upstream's
  `>= 5.10.4` and was updated), triton 3.6.0+gfx906, torch 2.13.0+gfx906, FA
  extension loads. This image still ships the fork Triton; a triton-adopting image
  would be a separate build.

## 2026-09-15 (TRITON-1, model gates)

- **All per-model gates pass on stock Triton 3.8.0.** MoE 35B (layer-0 Triton
  `fused_moe`) **57.97 t/s** (58.03/57.97/57.96/57.90, mclk 1000) vs the recorded
  58.36; Nemotron **PPL 26.9937** vs 27.0066 (band 26.96–27.02); Ornith **PPL
  16.6664** vs 16.7824 — all 0 top-20 misses. Dense 27B and the FA suite were
  already green. Ornith's −0.7 % is the largest numeric shift (PPL is
  deterministic per build+model here, so it is a real codegen effect on that
  model's kernel mix) and is the one number worth remembering when adopting.
- **Artifact caveat:** the published PyPI `triton-3.8.0` wheel segfaults on import
  on this box (AMD backend and `gfx906` present; no missing libs, no runtime deps),
  so adoption means building the unpatched upstream tag with our documented recipe
  — no patches, reproducible, but a build we produce. AMD's ROCm-index wheel
  (`3.7.1+git0263a6a6.rocm7.14.0`, vLLM's own `rock.txt` pin) downloads and is an
  untested alternative.

## 2026-09-15 (TRITON-1)

- **Stock Triton supports gfx906 since v3.8.0 — validated at parity, so the
  patched fork is no longer needed.** Upstream's `aa53dba7455` "[AMD] Add GCN5.1 /
  gfx906 target (#9628)" introduces `ISAFamily::GCN5_1` (wave64, DPP broadcast,
  `supportsVDot`, no MFMA, deliberately not CDNA/RDNA); v3.7.1 predates it. Built
  stock v3.8.0 (recipe: gcc, *no* `TRITON_BUILD_WITH_CLANG_LLD`, plus
  `TRITON_APPEND_CMAKE_ARGS=-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`) and ran the
  screens on the 0.29.0 line: **FA suite 97 passed**; **PPL 10.5472 vs the fork's
  10.5516 (−0.04 %)** with 0 top-20 misses and a **byte-identical** 32-token greedy
  completion (so the drift is fp accumulation rounding); the
  **`GFX906_FA_VIT=0` ViT fallback** (flash-attn/Triton-AMD) compiles and runs; and
  serving parity in the same boot (MTP k=3, agentic, 64k+120k, 2 reps) gives
  ms/step **85.4/127.9 vs 85.8/128.2**. Two claims were walked back by an
  interleaved A→B→A follow-up: (a) the A/B's acceptance difference is **not** a
  build effect — at one corpus body 3.8.0 gave 2.1875 / 1.8132 / 1.7634 / 1.7128
  over four processes (spread 0.475) against the fork's 1.7634 twice, i.e. the
  cross-build delta is *inside the same build's own spread* (lead with ms/step,
  interleave arms; the variance-asymmetry hint did not survive the second
  corpus body — the fork's own two samples there differ by 0.19 and straddle 3.8.0's
  range, so per-process variance is common to both builds); 128-token greedy probes are byte-identical across all
  runs of both builds on a code prompt yet differ between two runs of the *same*
  build on a prose prompt, so close-call flips are per-process and prompt-dependent;
  the one surviving hint is that the newer Triton may be more process-variable
  (candidate mechanism: timing-based `@triton.autotune` config choice in the
  chunked-prefill GDN kernels — the PPL probe, which does not chunk, stays
  bit-reproducible per build); (b) the `supportsDirectToLdsLoadBitWidth` "gap" is
  **inert** — direct-to-LDS is only created for async copies, gated on
  `{CDNA3,CDNA4,GFX1250}` in 3.8.0 and `{CDNA3,CDNA4}` in the fork, so the fork's
  `VEGA20` case was dead code and 3.8.0's missing `GCN5_1` case is unreachable. Remaining: the adoption decision, and one
  small parity gap (`supportsDirectToLdsLoadBitWidth` has no `GCN5_1` case —
  upstreamable against #9628). Full record: `RECON-triton-1.md`.

## 2026-09-15

- **0.29.0 line updated for release: `gfx906/v2-bringup` fast-forwarded into
  `main` and `gfx906/v0.29.0` (all three at `8c147037f3`, 23 commits, 0 behind).**
  The release branch now carries the V2 bring-up, VIT-1, the CAT-1 correction, the
  A3 revival, KVLAYOUT-2's closure and the k=3 spec-decode default.
- **Decisions recorded for the release.** V2 is the default runner for the
  validated models (dense 27B, MoE 35B, Nemotron 3.5 Lightning, Ornith);
  **Gemma-4 and Muse-Glimmer stay pinned to V1** (`VLLM_USE_V2_MODEL_RUNNER=0`)
  because their parity gate has not passed. **VIT-1 stays default ON** (kill
  switches `GFX906_FA_VIT=0` / `GFX906_FA_VIT_AUTO=0`, documented in the root
  README). **V1's removal is upstream 0.32.0**, which sets the deadline for the
  two pins — tracked in ROADMAP `DFL2-2` / `GEMMA4-1` / `MUSE-1`.
- **Release docs re-anchored on the 0.29.0/V2 basis:** the root README's install
  section was empty and now states the stack (ROCm 7.14 + the official AMD DKMS
  driver for TP=2 P2P), the mandatory `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`,
  the build recipe, the MTP k=3 + capture-ladder defaults and the wedge/canary
  protocol; `docs/gfx906/README.md` carries the release basis, the V1-pin list,
  the VIT-1 knobs and the 0.29/V2 performance rows.
- **Release validation on the merged tree passed.** FA suite **97 passed** and
  the in-process PPL probe on the dense 27B is **10.5516 (359 tokens, 0 top-20
  misses)** — bit-identical to V1 and to the 0.28 line, i.e. the V2 bring-up +
  VIT-1 + the `kv_split` override left the text path numerically untouched.
- **`VLLM_GFX906_FUSED_DRAFT` (A3) is revived but inert** (default OFF, neutral at
  k=3 and k=7) and **SMLA-1 is parked/inert** (fork fp16 sparse-MLA, default off,
  DeepSeek-MLA only) — neither blocks the release.

## 2026-09-14

- **The 0.29.0 line was promoted to `main`** (fast-forward, `main` ==
  `gfx906/v0.29.0` @ `7311119d67`; `gfx906/v0.28.0` stays as the previous release
  branch). Parity gate all green before the promotion: build + extensions, FA
  suite (88 pass / 3 tracked skips), PPL probe **bit-identical** to the 0.28 line
  (10.5516), V1 serving smoke, V1 restamp (MoE 65.40/58.17, dense 24.90/16.33),
  and the agentic corpus re-measured (greedy 19.91/13.25 — parity; MTP k=3
  33.30/24.54 — parity arm-level; MTP k=3 + CAT-1 34.62/25.55). V2 was shown
  viable on gfx906 in the same window (graph serve + PPL 10.5516); its
  performance/spec-decode parity work continues off `main` (`V2-bringup.md`).
- **V2 bring-up (`gfx906/v2-bringup`) — V2 now serves spec decode at parity, and
  CAT-1 works under it.** Dense 27B on the agentic corpus (same boot, 2 reps):
  greedy 20.37/13.27 vs V1 19.91/13.25; MTP k=3 33.62/23.75 vs 33.30/24.54;
  MTP k=3 + CAT-1 42.60/26.94 vs V1 34.62/25.55 — **RETRACTED 2026-09-14**: the
  A/B client put the arm name in the prompt header, so every arm ran a different
  prompt (the AGENTS.md trap). Re-measured same-boot with the fixed client, 3 reps:
  plain 34.41/24.00 vs CAT-1 35.44/24.61 t/s (ms/step 86.3→83.7 @64k, 128.5→124.4
  @120k ⇒ **+3.0 % / +2.5 %**) with **acceptance unchanged** — the shortlist's
  effect is the cheaper per-step head read, not agreement. The `MTP draft-vocab
  shortlist ACTIVE` marker fires under V2 with graph capture on and no eager
  fallback, so item 3's silent-loss risk is refuted both by code audit (V2 drafts
  go through the draft model's `compute_logits`; the one bypass path fails closed
  at init) and live. V2-CAT1-1's earlier three-arm reading (matched 2.44/2.49 vs
  control 1.72/1.71 vs none 2.05/1.93) is retracted with it; exactness still
  holds because the rejection sampler reads the same masked draft logits it
  sampled from. V2 reserves more VRAM for the same flags (KV pool 454,536 vs
  496,693 tokens greedy; 386,513 vs 442,368 spec; capture 2.49–3.02 GiB vs
  0.71 GiB), so capture ladders matter more on V2. M3's host-`cu_seqlens` path is
  length-agnostic and now has a full-length-slice guard test; KVLAYOUT-2's three
  skipped capture tests were stale skips and pass (suite: 91 passed, 0 skipped).
- **A3 revived (V2-only, opt-in, still default OFF) and gated: NEUTRAL at the
  serving k.** The fused-draft opt-in + no-op `update_draft_decode_metadata` are
  back in `gfx906_fa_backend.py` behind `VLLM_GFX906_FUSED_DRAFT` (default 0),
  with the three archive tests (3 passed; the reuse test migrated to the 0.29
  fused KV layout) and the no-op contract re-audited against 0.29 + V2. Serving
  A/B under V2 (agentic, 2 reps, ms/step lead): off 81.2/88.7 ms @64k and
  128.9/129.4 @120k vs on 81.9/89.1 and 128.5/129.6 — neutral, like the archive's
  k=4. k=7 stays open: both `mtp7` launches wedged at load (burst) and GPU work
  stopped.
- **V2 parity extended to Nemotron 3.5 Lightning (PPL 27.0066 vs V1 26.9986) and
  Ornith 1.5-35B-A3B (16.7824 vs 16.7724)** — both flipped to V2. Gemma-4 cannot
  be gated by the in-process probe (both runners degenerate, PPL ~10^5 over a
  multimodal load) → ROADMAP GEMMA4-1, stays pinned to V1.
- **VIT-1 DONE — the Qwen3.5-family ViT now runs on the custom FA by default**
  (`DEVLOG-vit1.md`). Serving gate, fresh image per rep, prefix cache OFF, same
  boot, identical prompts: **5.81 → 5.14 s TTFT @1024x1024 (−11.5 %)**, 1.71 →
  1.67 s @512, and the fresh-boot Triton JIT for the ViT is **−55 s** (330 → 275 s
  with an empty `TRITON_CACHE_DIR`; the remaining ~140 s is the GDN Triton kernel,
  so this does *not* make the triton-AMD flash-attn package droppable). The
  swap is not bit-equivalent: same greedy answer content, different wording, top-1
  preserved, max tail |dlogprob| 0.66 (Q8-K features vs fp16). Fixed en route: the
  adapter's non-production layout assumption (packed `[seq_len,1,hidden]` is what
  the VL towers pass — multi-image requests would have failed) and the decode-era
  `kv_split` default (32), now a per-call override.
- VIT-1 step 1 landed (custom-FA arm for the Qwen3.5 ViT, opt-in, unit-validated);
  the aiter/spec-decode blocker found on the 0.29 line was fixed.

## 2026-09-13

- **Upstream v0.29.0 merged** into `gfx906/v0.29.0` (merge `3c445dba56`; parents
  `4b7e0b7eb2` + tag `98dff2a81d`). Upstream delta vs the merge base: 561
  commits / 1848 files; 104 files touched by both sides, **23 conflicts**
  resolved by hand (record: `/local/tmp/b4/v0290-resolutions.md`, condensed in
  the merge message). Headlines: **Model Runner V2 is upstream's default for all
  models** — the fork pins **`VLLM_USE_V2_MODEL_RUNNER=0`** in every recipe until
  DFL2-2 brings V2 up; upstream now ships **dflash2** itself (the fork's backport
  converged and its two superseded files went to upstream: `qwen3_dflash2.py`
  and `dflash2/__init__.py` are byte-identical post-merge); native Hunyuan VL
  moved to the Transformers backend (file deleted, our compat rename dropped);
  FA4-hd256 block sizes and the FlashInfer CuTeDSL BF16 path arrive with our
  GEMV dispatch kept intact; the WNA16 oracle keeps the fork's tested qzeros
  repack helper and learns about upstream's new `EMULATION` backend.
  **Validation (2026-09-13, boot a27a894e, all three gates green):**
  worktree build `rc=0`; **FA suite 89/89**; **PPL probe bit-identical to the
  0.28.0 line — 10.5516 both** (Qwen3.8-27B-AWQ-INT4, fp16, 359 tokens, 0
  top-20 misses); **V1 serving smoke**: coherent completion + clean teardown
  (`VLLM_USE_V2_MODEL_RUNNER=0`). Three merge defects were found and fixed
  *after* the static pass, each only by running something:
  (1) `csrc/.../gptq/q_gemm.cu` duplicate `dot22_8_f` declaration (caught by the
  build); (2) `qwen3_vl.py` `NameError: video_max_pixels_per_frame` in
  `get_dummy_mm_data` — the merge kept one of the fork's lines inside the method
  upstream rewrote; the fork's per-item video pixel cap was restored on top of
  upstream's definitions (caught by the PPL probe); (3) the 0.29 **KV-cache
  layout standardisation (#51718)** broke the FA backend's K/V split —
  `kv_cache.unbind(1)` now sees heads on dim 1 because the content axis is
  fused `K||V` (`[B, H, N, 2*D]`). Ported: both split sites now use
  `kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` (identical
  `[blocks, block_size, Hkv, D]` view our kernels already took) and the backend
  declares `supported_kv_cache_layouts()` = `(KVCacheLayout.LBHNC,)`; the
  engine logs `Using LBHNC KV cache layout` (caught by the first PPL run).
  `ops/rocm_aiter_mla_sparse.py` was taken wholesale from upstream (the fork's
  fp16 sparse-MLA is half-ported, tracked as SMLA-1).
  **Spec-decode blocker found on the 0.29 line (2026-09-14):** the MTP/EAGLE
  drafter's base proposer imports `vllm.v1.attention.backends.rocm_aiter_fa`
  after a `find_spec` existence check, but that module imports AITER's gluon PA
  kernel at module scope — on a box without the `aiter` package (the gfx906
  venv; AITER is unusable here) **every spec-decode serve died at drafter init
  with `ModuleNotFoundError: No module named 'aiter'`**, i.e. our production MTP
  k=3 config could not start. Fixed by making that import optional (skip the
  metadata type with a one-shot warning); MTP k=3 then serves normally
  (`MTP3 READY ~395 s`). The deeper, upstream-worthy fix is to guard the
  module-level aiter import in `rocm_aiter_fa.py` itself.
  **Agentic-corpus re-measure on 0.29 (boot eefacc1e, V1 pinned, 2 reps/cell,
  mclk 1000):** greedy 20.39 @64k / 13.26 @120k, MTP k=3 32.54 / 24.53, MTP k=3
  + CAT-1 **34.62 / 25.55** — the CAT-1 shortlist reproduces here (+6.4 % /
  +4.2 % over plain k=3 in-session) and its `MTP draft-vocab shortlist ACTIVE`
  marker is logged in serving, which also de-risks the V2 variant of that check.
  An arm-level re-check of bare MTP k=3 (3 reps) then **retired that reading**:
  64k {35.20, 35.14, 29.56} → mean 33.30 (0.28: 33.28), 120k {24.26, 25.14,
  24.21} → mean 24.54 (0.28: 24.74, -0.8 % inside the spread), acceptance
  2.05/2.05/1.55 and 2.06/2.15/2.06 — the 2-rep "-1-2 %" was the arm's own
  acceptance variance, and spec decode is **at parity on 0.29**. Two load
  wedges hit the session (#81 mtp3 attempt 1, #82 CAT-1 attempt 1); both retried
  clean and are recorded in `degradation.md`.

  **Parity + V2 status (boot eefacc1e, same day).** V1 restamp of the 0.29 line
  against the 0.28 numbers: MoE **65.40 warm / 58.17 cold**, dense **24.90 /
  16.33** t/s — all within ±0.8 % (mclk 1000 in every window), i.e. **0.29 V1
  parity with 0.28 is established**. And **V2 is viable on gfx906**: the graph-mode
  V2 serve returned a coherent completion with a clean teardown, and V2
  in-process PPL = **10.5516**, bit-identical to V1's and to the 0.28 baseline
  (the Y16 "unsupported-by-design" record was the box's load lottery — a BACO
  reset landed in the same second as the eager attempt that failed). Remaining
  V2 work is performance/spec-decode parity: `V2-bringup.md`.

## 2026-09-02

- **MTP-1a closed: Qwen3.8-27B dense MTP k=2 crossover pinned at 32k–64k pp**
  (clean boot Q, TP=2, n=3, cold prefill, arms sequential). MTP wins ≤32k
  (1.03×), loses ≥64k (0.85× → 0.72× @120k); acceptance 2.0 stable through
  120k so the tax is O(Sk) step cost, not draft rejection. Bracket is ~2×
  wider than S9's ~20k estimate. Optz microbench: lm_head-per-draft lead
  DEAD (memory-bound, +322 µs/step = 0.4% of the 78 ms @120k step); attention
  K-multiplier ~1.0 at S=120k (KV bytes shared). Open residue: the 78 ms vs
  ~12 ms bandwidth-floor gap needs a rocprofv3 kernel breakdown (MTP-1b gate;
  blocked by a zombie KFD VRAM handle left by the old-vLLM wedge below).
  Old-vLLM 0.23.1 A/B abandoned: that code path wedged GPUs loading this AWQ
  model at shard ~2/5 on BOTH userlands (docker ROCm 7.2, in-process ROCm
  7.14) — incompatible with the model+host, not an env issue. Record:
  `DEVLOG-mtp1.md`, degradation entries 2026-09-01/02.

## 2026-09-01

- **Post-C4 maintainability sweep: removed the two FULLY-DEAD M=1 routing
  experiments (S2 topk kernel + C1 stage 2 fused routing).** Per the
  post-C4 inventory directive (inventory all merged-but-not-default-enabled
  gfx906 work; remove dead code for maintainability with a preservation
  branch first), branch `gfx906/preserve-dead-kernels` was cut at main so
  every removed line stays in git history, then: S2 (`moe_topk_gfx906.cu`,
  the `VLLM_GFX906_TOPK_M1` dispatch, bindings, `_custom_ops` wrapper +
  fake, tests, bench) and C1 stage 2
  (`moe_routing_fused_m1_gfx906.cu`, the `VLLM_GFX906_ROUTING_FUSE_M1`
  dispatch, bindings, tests) were deleted, along with the
  `_fused_align_meta` router→expert plumbing they orphaned (C1 stage 2 was
  its only producer; verified zero non-None assignments remain) — touching
  `moe_runner.py`, `routed_experts.py`, `modular_kernel.py`,
  `fused_moe_modular_method.py`, `gfx906_w4a16_moe.py`, and
  `fused_topk_router.py`. Net −406/+9 lines, 13 files. Both were already
  recorded FULLY DEAD in `DEAD-ENDS.md` (3rd isolated→serving flip pattern;
  C1 stage-1 — the align+count kernel that shipped default-on — is
  untouched). The structural probe `c1_routing_structural_probe.py` is kept
  (hasattr-guarded, degrades gracefully) as the cited evidence record.
  Verification: incremental build rc=0; unit suites green; live M=1/M=4
  `fused_topk` through the edited router OK; removed ops confirmed absent
  from the compiled extension; full-engine e2e probe (Ornith-1.5-35B-A3B,
  C4 ON) loading + coherent decode. Remaining default-off knobs after the
  sweep: `SKINNY_M16` (cleared to flip default-on — pending decision),
  `MAMBA_FUSED_GROUP_NORM` NH-4 (+0.4 % = noise; keep, revisit when a
  non-MoE-GEMV-bound config exists), the int8 W8A16 family
  (`W8A16_INT8*`, T1 PROBE GO dependency — keep), and C4
  `QUANT_LAYER0_MOE` (GO, opt-in pending soak). The stale S5 row in
  `DEAD-ENDS.md` was corrected: that kernel is live default-on as
  `VLLM_GFX906_MOE_M1` since the C2 combined A/B (+2.72 %).

- **C4: load-time int4 quantization of the unquantized first MoE layer (GO,
  measured).** Qwen3.5-35B-A3B-AWQ leaves `model.layers.0.` in fp16
  (`modules_to_not_convert`), so its routed experts ran on the Triton
  unquantized path at ~4× per-call cost (740 vs 182 µs, C2 profile). New
  method loads fp16 as usual, then quantizes to int4 in
  `process_weights_after_loading` (asymmetric AWQ, codepoints against the
  stored fp16 scale) and delegates to the shared gfx906 WNA16 repack — no new
  kernel code; fp16 storage released (~1.5 GiB). Gates: unit 8/8; PPL Δ +0.04
  (noise); greedy serving fingerprint bit-identical across arms; serving A/B
  (M=1, pp2048/tg256, same boot) **84.95 → 87.51 t/s = +3.0%** (above the
  ~1.8% noise floor). Pre-merge review (self + Claude CLI): runner-visible
  quant-method state sync + `supports_eplb` path-following fixed; one finding
  rejected with evidence. Ships opt-in (`VLLM_GFX906_QUANT_LAYER0_MOE=1`)
  pending a soak window. See `DEVLOG-c4-layer0-quant.md`.

## 2026-08-30

- **NH-4: mamba2 grouped gated-norm fused path (SHIPPED, env default
  OFF).** `Mixer2RMSNormGated.forward_cuda` routes the n_groups>1 case
  through the existing fused Triton `rms_norm_gated` kernel behind
  `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM=1`, gated on
  `per_rank_hidden_size % group_size == 0`. Isolated ~68 → ~55 µs/layer
  (~0.29 ms/step over 23 mamba layers); serving A–B–A (TP=2+EP, fresh
  boot per arm) 109.8 → 110.05 → 109.37 t/s (+0.4 %, within inter-arm
  noise — step is MoE-GEMV-bound at this batch) with PPL 24.9034 vs
  24.8944 (Δ +0.04 %). Correctness: 11/11 unit tests (incl. production
  TP=2 geometry and TP-driven partial-group refusal), TP=2 regression
  driver 6/6 bit-equal, ruff clean. Review protocol: self-review +
  Claude CLI review of branch vs main, both findings fixed before merge.
  See `DEVLOG-nemotron-h.md` (NH-4 section) and `ROADMAP.md`.

## 2026-08-29

- **NH-1 + NH-3: Nemotron-3.5-Lightning-30B-A3B mixed INT4/INT8
  onboarding (`gfx906/nemotron-h-onboard`, unmerged).** Serves at
  70.4 tok/s (graph, pp2048/tg256, 4 samples) from 4.95 tok/s at first
  load (14.2×): fp32-router LLMM1 dtype guard; ssd_chunk_scan
  pointer-yield restructure working around the triton-gfx906
  CanonicalizePointers fat-pointer assertion (94/94 SSD reference
  tests); new `CompressedTensorsW8A16ChannelDequant` scheme replacing
  Conch for int8-channel dense layers (3.79 ms → 62 µs per M=1 GEMV,
  +1.8 GiB VRAM); gfx906 W4A16 MoE oracle gate widened to any positive
  multiple of 32 (group-64) + RELU2_NO_MUL experts (+88.8% vs Triton
  WNA16); fp32 router-gate GEMV on hipBLAS sgemv (+18.4%). PPL gate
  26.96–27.02 band across all arms. Open follow-ups NH-2 (int8 GEMV),
  NH-4/5 (mamba2/topk tails) in `ROADMAP.md`; records in
  `DEVLOG-nemotron-h.md`.
- **M2: per-q-tile prefill clip merged to `main` (`06c0614379`).** Two
  bit-identical per-q-tile scan bounds in both FA kernels — a window
  raise of `k0_base` (the tile's first row has the smallest window
  start; keys below it are masked for every row) and a causal cap of
  `k_VKQ_max` (the tile's last valid row bounds the scan tail) — plus
  the DIRECT_PAGED backend clip extended from decode-only to prefill
  chunks; knob `GFX906_FA_TILE_CLIP` (default on). Kernel A/B at the
  pp4096/full-context shape: 3.19×/2.81× (windowed, both kernels) and
  2.22×/1.96× (causal-cap-only, first-chunk full-attention geometry —
  the cap is a general chunked-prefill win, not a window feature).
  Review-gated e2e: Muse pp16384/B=2 windowed **+11.8 % wall / +14.8 %
  prefill**; Qwen3.8-27B pp2048 full-attention +0.73 % (GEMM-dominated;
  its FA component is the 1.96–2.22× above). Decode/spec paths provably
  unchanged (cap = seq_len; raise ≡ the existing floor). Residual:
  per-row granularity within a 64-row tile (~1/32 of the effect) left
  open. Records: `DEVLOG-fa-kernel-batches.md` (M2 + 2026-08-29 review-fix
  entries, `m2-code-rev-glm5.md` findings closed by `04e6ab7c60`).
- **M3: kernel hygiene batch merged (`feat/fa-m3-hygiene`).** #8
  device-side `k0_base = max(0, kv_start[seq])` clamp (a negative start
  walked the paged k-loop into token-negative space — illegal access /
  wedge, not a wrong number); #10 overflow-free window cutoff
  (`q_abs_row - k_pos_abs >= window`, provably equivalent for all int32
  window; the old form could not actually wrap — hardening/clarity);
  #4b `o_meta` `[B,Sq,Hq,2]` allocation dropped entirely (the kernel's
  only `dst_meta` write is guarded by `gridDim.y != 1` and
  `gridDim.y == kv_split`, so the buffer is dead at `kv_split==1` too
  — ~300 KB/layer at Sq=1568/Hq=24); amplified-V window-boundary
  regression pin (~400× discriminative). The branch's dot2 P·V rewrite
  premise was REFUTED by ISA and the item closed: objdump of the
  production build shows the P·V accumulate already compiles to
  `v_pk_fma_f16` (1024× in `flash_attn_tile_q8<128,128,16,2>`, 0×
  `v_pk_add_f16`), so `v_dot2_f32_f16` buys zero instruction count —
  precision-only candidate, revisit only behind a numerics gate
  (`dequant-instructions.md` corrected, old paragraph SUPERSEDED).
  Post-merge suite 70/70 (60 base + 5 M2 + 5 M3 parametrized cases);
  both review rounds (`m3-code-rev-glm5.md`, external fold) closed at
  `cf5ccbd685`/`9d98aca9ab`.
- **M4: long-context split-K accuracy point closed (qwen review #4a).**
  Production split defaults are safe — in fact MORE accurate — at
  16k–32k context: in-process probe (sk 16384/32768, D=256/Hq16/Hkv2
  + D=128/Hq32/Hkv2, seed 20260829) shows gather kv_split=16 (the B=1
  default) at 5.2e-3/6.6e-3 rel vs fp32 ref and direct-paged
  kv_split=8 (the B≥2 clamp default) at 4.0e-3/5.0e-3 — all ≤ half the
  5e-2 tolerance, and the no-split baseline is WORSE (1.9e-2/2.6e-2):
  the split partials are fp32 (the M4 "unscaled fp16 partials" framing
  was stale) and the fp16 P·V accumulator error scales with
  accumulator length, which splitting shortens. Suite 74 → 78 (two
  16k gather arms + direct-paged L=16384 split-8 pin, both
  geometries); probe kept at
  `benchmarks/kernels/gfx906/m4_splitk_accuracy_probe.py`.
  Records: `DEVLOG-fa-splitk-accuracy.md`.
- **B=1 LEGACY=1-vs-0 decode gap closed (roadmap item #1): LEGACY=0
  stays OFF.** Same-boot (boot O) serving A/B, Qwen3.8-27B TP=2 B=1
  pp2048/tg256: LEGACY=1 40.11/40.12 vs LEGACY=0 37.61/37.56 (−6.3 %)
  t/s; the M5-era direct-paged B=1 config lands within 0.2 % of the
  Q8-gather config (37.55/37.54) despite very different FA/gather
  subcomponents (kernel probe: the Q8-gather read path is 22–45 %
  FASTER per step than fp16-gather+quantize, growing with Sk; direct
  paged is +8–35 % slower). The serving gap is therefore a
  LEGACY=0-common per-step cost, not FA/gather: the append-time Q8
  side-buffer write is +60–105 us/step eager (16 full-attn layers;
  q8-alone ×16 = 105.6 us bound), and
  the ~1.55 ms/step remainder is a serving-harness/graph-node
  interaction (unmeasured). M5's "LEGACY=0 LOSES, default stays 1"
  verdict confirmed by a proper same-boot adjudication. Probes kept
  (`benchmarks/kernels/gfx906/legacy0_b1_step_probe.py`,
  `legacy0_append_cost_probe.py`); `_serve_tp2_gfx906.sh` gained
  EXTRA_SERVE_ENV passthrough. Records:
  `DEVLOG-fa-legacy0-b1-decode.md`.
- **Roadmap reorganization: three per-topic roadmaps → single
  priority-ordered `ROADMAP.md` + `REFRIGERATOR.md`.** The per-topic
  split had leaked (G1/housekeeping in more-models, non-MoE N-items and
  the upstream queue in the MoE file) and the spec-decode roadmap was
  100 % parked work. Closures folded in: the spec-decode file is deleted
  (all four items → REFRIGERATOR with reopen gates); the Muse follow-ups
  section is empty and gone (LEGACY-flip closed this date, Part C →
  REFRIGERATOR); DeepSeek-V4-Flash → REFRIGERATOR (not an active
  target); Qwen3-30B-A3B → DEAD-ENDS (SUPERSEDED — not an active goal,
  model superseded by the supported Qwen3.5/3.8 line). C4 stays active
  (70 t/s target active, user decision 2026-08-29). Item IDs (C*, G*,
  L*, N*, U*, HK*, SD-*) are stable; README/AGENTS references updated;
  the C8 L2/residency open question is folded into C2. Historical
  filename mentions inside devlogs/plans are left as records.
- **MoE C1 stage 1: M=1 fused align+count kernel landed (opt-in flag
  defaulted ON after gate).** The M=1 decode routing chain is 3 kernels
  per layer (topk + align 2-block + count_and_sort = 120 graph nodes/step
  ≈ 0.8 ms); the new 1-CTA kernel replaces the align pair (120 → 80
  nodes), bit-equal to the generic chain. Structural probe + S2
  re-validation: isolated-graph kernel numbers can flip sign in the
  production graph (S2 topk swap: −1.03% serving vs −28% per node in
  isolated graphs), but **node removal transfers** — serving A/B
  (in-process MoE 35B, pp2048/tg256, 4 samples/arm, back-to-back):
  **+1.18% (207 µs/step), +1.73% on the second session**; within 8% of
  the isolated prediction. `VLLM_GFX906_ALIGN_M1=0` opts out. Stage 2
  (fused topk+align+count, 120 → 40 nodes) is the follow-up. Records:
  `DEVLOG-moe-c1-routing-fusion.md`,
  `benchmarks/kernels/gfx906/c1_routing_structural_probe.py`.
- **MoE C1 stage 2: fused topk+align+count — DEAD-END in production
  (flag OFF, kernel + plumbing + tests landed).** The one-CTA fused
  routing kernel (`moe_routing_fused_m1_gfx906`) is bit-equal to the
  3-kernel chain (27/27 tests) and 28 % faster in isolated graphs
  (40 nodes: 10.0 µs/node vs 13.8 µs/layer for the stage-1 pair) — yet
  the A-B-A serving gate shows **−1.10 %** (57.42 → 56.79 → 57.46
  control t/s, Qwen3.5-35B, pp2048/tg256): the third S2-pattern flip,
  and the stage comparison pinpoints it: node REMOVAL transfers
  (stage 1, +1.2–1.7 %), REPLACING the proven production topk does not
  (S2: −1.0 %, stage 2: −1.1 %). Router→expert meta plumbing
  (optional `fused_align_meta` kwarg, signature-gated, dropped for
  unquantized/ignored layers) is in place and production-neutral with
  the flag off. Records: `DEVLOG-moe-c1-routing-fusion.md` (stage-2
  section), `tests/kernels/moe/test_moe_routing_fused_m1_gfx906.py`.

## 2026-08-27–28

- **Muse-Glimmer-30B-AWQ-INT4 onboarding + window FA + M1 gather clip
  merged to `main` (2026-08-28, `feat/muse-glimmer` fast-forward).**
  Sliding-window support in the custom Q8 FA (window arg, both kernel
  copies; all-CUSTOM 1.59× vs hybrid at B=1), direct-paged split-K +
  Phase C clip, LEGACY=0 Q8 side view aliased into the fp16 K half
  (zero extra KV memory, COW-safe; prefix-cache fail-closed removed),
  and the M1 gather-path window clip (absolute-position gather layout,
  +8.1% e2e at pp8192/B=1). Root-caused and fixed the boot J/K
  first-prefill OOM: the q_pad buffer was per-impl (v1 creates one
  backend impl per attention layer) — 52 × 256 MiB = 13.3 GiB;
  ClassVar share cut the transient 3.785 → 1.285 GiB and made bt4096
  TP=2 serving viable (the bt2048 workaround is droppable). Records:
  `DEVLOG-muse-glimmer.md` (rounds 1–5), `degradation*.md` boots I–L,
  working TP=2 recipe in `README.md`. Review rounds 1–3 + the
  post-boot-L review set (`fa_oom_fix_clip_code_rev_*.md`) closed;
  the two robustness gaps they flagged (raw-fp16 branch assert,
  GATHER_CLIP_MARGIN/config-table static_assert) landed in
  `52ff21f9d9`.
- **M6 Part B: LEGACY=0 B≥2 default route flipped to the fused-Q8
  gather (2026-08-28, round 10).** The M5 bake's B=4 @2k
  −27…−31 % deficit was Sq>1-specific (the in-process Sq=1 A/B on
  the identical strided-read path was a wash; mechanism — strided
  Q8-slice reads leading but unconfirmed, Sq>1 machinery at least a
  co-contributor — round-10 erratum). `GFX906_FA_DIRECT_PAGED_Q8`
  (default `0` since the flip) routes LEGACY=0 B≥2 through the
  fused-Q8 gather: B=4 @2k aggregate 35.7 → 46.3 t/s (parity with
  the 46.7 LEGACY=1 control within cross-boot uncertainty; B=1 and
  prefill unchanged; 60/60 suite). A no-op under the production
  LEGACY=1 default; direct-paged stays opt-in (=1). The M5
  LEGACY-flip gate's B=4 half is now green; the flip itself still
  needs the B=1 same-boot adjudication.
- **M5: LEGACY read-path bake executed — keep `GFX906_FA_LEGACY=1`
  (`a6780408a8`).** The TP=2 ngram bake measured LEGACY=0 slower at
  every controlled point (B=1 −2.5…−3.7 %, B=4 −27…−31 %, prefill
  wash); per the flip rule only a win flips, so the default stands.
  The bake's original "no int8 path / fp32-ALU" reading was refuted by
  the SCEV-proof dot-rate probe (`v_dot4_i32_i8` full-rate, 4.44×
  fp32 FMA — AMD's 53 TOPS INT8 figure is this instruction; rates in
  `dequant-instructions.md`); the deficit is read-path/layout, not
  the dot. LEGACY=0 remains an experimental opt-in.
- **M6 Part A: planar Q8 repack executed — DEAD-END for the flip
  question; merged to main 2026-08-29 as loader hygiene** (merge
  `02d197189f`). The rev-2 plan's hard stop-rule fired: loader global
  loads 10→6 per tile-row (1.67× < the 2× rule) despite a −2.4 %
  standalone B=1 win, so the B=1 gap is not load-instruction-bound.
  Merged for the aligned-loader win (production LEGACY=1 shares the
  loader) and Part C groundwork: merged-tree suite 74/74 (incl. 4
  byte-level layout pins), same-boot B=1 decode-step A/B (Muse
  geometry D=128/Hq=32, NC2=1/KVSPLIT=1, boot N): slope 36.0→34.4
  ns/token (−4.3/−4.8 %), @Sk=2176 83.6→79.1 us (−5.0/−5.6 %),
  bit-identical (maxerr equal at every Sk) — gate PASS. **Caveat
  (post-merge review): both A/B arms ran under contention — the same
  merged `.so` measures 42.0 us @Sk=2048 / slope 12.86 ns/token on an
  idle GPU (1.6–1.8× faster absolute), so the recorded µs/ns are
  contended-boot numbers; the −4…−5 % delta is directionally
  supported (16/16 points, round-11 −2.4 %, ISA mechanism) but the
  merge never depended on it (abort condition was slower-than-noise;
  bit-identical).** Record:
  `DEVLOG-muse-glimmer.md` round 11, `DEAD-ENDS.md` MG row, plan
  `plan_fa_part_A.md`.
- **M6 Part C (Q4-KV via `v_dot8_i32_i4`): SHELVED (`5d8d4c7f59`).**
  Quality unproven (Q4 K *and* Q requant; 7-level codebook ≈ doubles
  KQ quantization error with no PPL evidence). Reopens only behind a
  dedicated accuracy gate that must pass before any kernel work.
- **MI50 vLLM memory-attribution skill.** Personal skill
  (`~/.agents/skills/gfx906-mem-attribution/SKILL.md`) + in-repo probe
  (`docs/gfx906/_probe_mem_attribution_gfx906.py`): the 3-arm OOM
  attribution recipe, per-layer hooks, bisection, and the env traps
  (AOT workers, inductor fork/spawn HSA). Validated on the M0 hunt.

## 2026-08-14–16

- **Phase 3 gfx906 performance stack.** The custom W4A16 MoE grouped GEMM
  fixed the 3.49 t/s routed-MoE regression, and the custom Q8 FlashAttention
  backend was integrated and made serving-viable. The dense M=1 GEMV path,
  FA GQA head packing/KV split, fused KV gather and quantization, NC2/kv-split
  guards, and bit-exact fill/copy reductions were landed. The resulting
  Qwen3.5-35B decode progression reached 64.08 t/s before the later sprint
  work. See `DEVLOG-moe-opt.md`, `DEVLOG-fa-attention.md`, and
  `DEVLOG-dense-decode.md`.
- **Phase 2 prefill close-out.** The useful prefill tuning and its negative
  results were recorded; the remaining persistent-CTA prefill idea is still
  parked as an open item in `moe-decode-roadmap.md`.

## 2026-08-17

- **Initial gfx906 roadmap and review close-out.** The Qwen3.5 improvement
  branch was merged into `gfx906/main` (`e861d0b30f`). The twelve parked
  pre-merge review items were resolved: direct-paged fp32-Q handling,
  LEGACY=0 prefix-cache guarding, bounded gather-buffer retention, MoE caller
  validation, non-gfx906 plugin/build tolerance, duplicated FA helper cleanup,
  gather-buffer reuse, MoE workspace-alias documentation, stale comments,
  lint debt, and the FA debug switch. See `DEVLOG-moe-opt.md`.
- **S2 M=1 top-k experiment.** The dedicated E=256/topk-8 softmax kernel was
  bit-equal and faster in isolation, but lost in CUDA-graph serving replay.
  It was retained behind `VLLM_GFX906_TOPK_M1` with the default off; the
  standalone-kernel approach is rejected for the gapless serving regime.
  See `DEVLOG-moe-m1-sprint.md`.

## 2026-08-18

- **MoE M=1 sprint results.** The gemm2 lane-column re-tile shipped behind
  `VLLM_GFX906_MOE_M1` (default off), improving the graph result by about
  0.60 t/s; the gemm1 version did not yet have a serving-gated win. The
  shared-expert down-projection moved to the gfx906 dense GEMV path, with the
  default-on decision recorded as provisional. See `DEVLOG-moe-m1-sprint.md`.
- **Speculative decoding Phase 0 and L1'.** The n-gram, GPU-n-gram, suffix,
  and prompt-lookup experiments established that n-gram quality is
  repetition-bound and GPU n-gram has a draft-selection mismatch. Suffix was
  deferred because its dependency and dynamic-length path were not justified
  for the current target; the k sweep was likewise not pursued. The fp16
  M<=4 GEMV-family extension (L1') shipped
  and moved the dense draft step from 66.6 ms to 53.2 ms eager; the first
  serving A/B was 0.945x. See `DEVLOG-spec-decode.md`.
- **Speculation cost-model correction.** Kernel-path census overturned the
  original GDN-small-M attribution: the required sequential GDN kernel was
  already in-tree. The dominant draft cost was the AWQ and fp16 GEMM mix, not
  a missing GDN kernel. The old L1/L2 plan was consequently re-scoped. The
  capture-safe FA uniform-batch rails required by speculative decoding were
  also landed. See `DEVLOG-spec-decode.md`.
- **C6 activation-quantization disposition.** The proposed Q8_1 activation
  path was rejected on gfx906: it adds quantization launches and lacks DP4A or
  int8 matrix hardware, so its expected net cost is negative.
- **Layer-0 MoE attribution.** The residual Triton expert calls were resolved
  to layer 0's fp16 routed experts, not the shared expert. Layer-0 quantization
  remains an open conditional candidate in `moe-decode-roadmap.md`. See
  `DEVLOG-moe-opt.md`.
- **Upstream vLLM merge.** `gfx906/main` absorbed upstream `main` in
  `38ceb5d957`; the gfx906 attention, quantization, and platform behavior was
  retained and revalidated.

## 2026-08-19

- **Speculative decoding rails completed.** The small-capture-size fix (L5)
  removed the graph padding penalty from one-token no-draft steps. MTP k=2
  support was added for the Qwen3.5-27B target, and the final agentic result
  was 39.74 t/s, 1.503x over the no-spec arm, with 90.95% draft acceptance.
  The dispatch-only L1'' investigation was closed as a non-issue: the
  relevant fp16 GEMMs already reached the dispatcher. See
  `DEVLOG-spec-decode.md`.
- **Per-file max-ilp split.** The q_gemm 4-bit build was split by M: M=1
  uses max-ilp while M>=2 does not (`cfe09d8611`). This resolved the build
  concern without regressing the MTP result. See `DEVLOG-spec-decode.md`.
- **Gemm1 re-tiling close.** The V1 design and the NPT surface were measured;
  the apparent isolated gain did not transfer to TP=1 serving, so no gemm1
  dispatch change shipped. The V3/V4 follow-ups were closed by the same
  evidence. See `DEVLOG-moe-gemm1-retiling.md`.
- **Gemma-4 onboarding.** Gemma-4 26B-A4B was loaded and characterized on
  gfx906. Its raw-prompt degeneration and hybrid-attention logprob issue
  were documented, and the model was retained as a supported test target.
  See `DEVLOG-gemma4-onboarding.md`.

## 2026-08-20

- **Upstream release `v0.28.0rc1`.** The tag was merged into
  `gfx906/v0.28.0rc1` on 2026-08-20 (`fc777b87dd`).
- **Gemma-4 symmetric no-zero-point MoE support.** The existing gfx906 W4A16
  kernel was extended through Python-side compressed-tensors gates and the
  GPTQ-K-first repack path; no new kernel was required. Serving improved from
  37.81 to 67.79 t/s (1.793x), and the flagship Qwen3.5-35B result was
  unchanged. See `DEVLOG-gemma4-moe.md`.
- **Gemma-4 review follow-ups.** The bit-width/group-size/strategy gate,
  activation-ordering (`g_idx`) guard, no-fabricated-zero-point storage, and
  the numerical divergence record were all closed. See
  `DEVLOG-gemma4-moe.md`.
- **Prefill/TTFT and build investigations closed.** The MTP2 TTFT question
  was resolved, the Qwen3.8 launch failures were attributed to the NAS rather
  than a model or kernel defect, and gfx906 auto-dtype fallback to fp16 was
  landed for bf16 checkpoints. See `DEVLOG-spec-decode.md` and
  `DEVLOG-qwen38.md`.

## 2026-08-21

- **TP=2 transport diagnosis.** The TP=2 investigation closed the initial
  RCCL/P2P failure hypotheses and identified the driver/topology issue. The
  official amdgpu DKMS driver made TP=2 serving viable; the remaining
  communication-bound ceiling and capture behavior were recorded rather than
  treated as a decode-kernel regression. See `DEVLOG-tp2-dense.md`.
- **N4 capture-width diagnosis.** The long-context decode tax was traced to
  `max_model_len` being baked into the captured FA gather dimensions, and the
  persistent live-bounded gather design was selected. See
  `DEVLOG-masked-fa.md` and `tp_decode_investigation.md`.

## 2026-08-22

- **N4 persistent gather shipped.** The persistent gather+quantize path removed
  the max-context replay tax. TP=2 serving improved from 22.4 to 40.9 t/s at
  131k context and from 15.9 to 40.9 t/s at 262k; the short-context tax was
  within noise. `GFX906_FA_PERSIST` is on by default. See
  `DEVLOG-masked-fa.md` and `DEVLOG-tp2-dense.md`.
- **C2-V validation completed.** The additional TP=2 and batch-regime tests
  showed that the M=1 re-tiles are positive at TP=2 (about +1.47% for gemm2
  and +1.23%/+1.32% for gemm1) but neutral at TP=1 and in the tested batch
  arm. The TP=2-scoped follow-up and the default-on decision remain open in
  `moe-decode-roadmap.md`. See `DEVLOG-moe-c2v.md`.
- **Upstream release `v0.28.0rc2`.** The tag was merged into
  `gfx906/v0.28.0rc2` on 2026-08-22 (`19e23ffedd`). Subsequent gfx906 work
  was periodically merged into that release branch; the release merge date
  is recorded here so version provenance is unambiguous.
- **Qwen3.8-27B support.** TP=1 and TP=2 execution was brought to a
  functional, measured state, including the fp16 dtype fallback and the
  long-context FA validation. See `DEVLOG-qwen38.md` and
  `DEVLOG-tp2-dense.md`.

## 2026-08-23

- **W2: Qwen3.5-35B MTP2.** The speculative-decoding rails transferred to the
  MoE model without code changes. Graph serving measured 89.9 vs 76.2 t/s
  (1.18x) and eager serving 45.5 vs 24.5 t/s (1.86x), with 80.4% acceptance.
  MTP2 is the recommended 35B configuration; MTP3 was not viable. See
  `DEVLOG-moe-spec-decode.md`.
- **W4: skinny fp16 M=5..16 GEMM.** The weight-row-parallel GEMV extension
  shipped behind `VLLM_GFX906_SKINNY_M16` (default off). The original
  all-decode-step gain estimate was falsified by the x-L2 re-read bound, but
  concurrent decode improved by 14.5% on 35B and 6.1% on Qwen3.8-27B; a
  30-repetition soak passed. See `DEVLOG-fp16-skinny.md`.
## 2026-08-24

- **FA gather-buffer lifecycle fix.** Capacity-width reuse and a
  per-generation capture flag replaced the unbounded retired-generation
  behavior. The Qwen3.8 250k prefill completed with needle retrieval and
  flat decode A/B; the fix was merged in `21c69a8ead`. See
  `DEVLOG-fa-attention.md`, `oom-256k-prefill.md`, and
  `plan-gfx906-fa-fix.md`.
- **Release-branch sync.** The completed FA lifecycle fix and the W2/W4/C2-V
  results were merged from `gfx906/main` into `gfx906/v0.28.0rc2` on
  2026-08-24 (`7e4567053e`), keeping the release branch aligned with the
  feature line.
- **Final-build records.** The dense and MoE serving numbers were restamped
  on the max-ilp split build, including the MTP and skinny-GEMM comparisons.
  See `README.md` and `DEVLOG-spec-decode.md`.

## 2026-08-25

- **Ornith asymmetric compressed-tensors W4A16 support.** The stored-int8
  zero-point checkpoint was admitted through the gfx906 oracle and repacked
  safely; no kernel change was needed. The model reached 65.03 t/s versus
  3.50 t/s on the Triton arm. The Triton `has_zp` performance problem remains
  an open portability/kernel issue, while the supported gfx906 path is
  complete. See `DEVLOG-ornith-wna16.md`.
- **TP CPU-spin mitigation.** The HIP blocking-sync shim was added after the
  TP stuck-thread investigation. Two independent ROCR-Runtime fixes were
  documented as unmerged upstream candidates; they remain in
  `moe-decode-roadmap.md`. See `cpu-stuck-threads.md` and
  `moe-decode-roadmap.md`.

## 2026-08-26

- **W1: mixed-request GDN decode peel.** Non-spec sequences in spec-mixed
  batches no longer take the expensive one-token chunk path. The 27B
  two-request serving A/B improved from 55.60 to 59.35 t/s (+6.7%), and the
  review follow-ups were closed before merge. W1 was merged into
  `gfx906/main` as `5b15152431`. See `DEVLOG-gdn-mixed-decode.md`.
- **L3 n-gram proposer follow-up closed.** The proposer-cost battery and
  revalidation did not justify replacing the CPU proposer; GPU n-gram remains
  rejected for draft-quality reasons. See `DEVLOG-spec-decode.md`.
- **Ornith review hardening.** The shared `g_idx` gate, fail-closed qzeros
  repack checks, and stored-zero-point backend capability set were merged as
  `d160fb2ad0`. See `DEVLOG-ornith-wna16.md`.
- **Upstream release `v0.28.0`.** The final upstream tag was merged into the
  gfx906 fork on 2026-08-26 as `a4cb86c4aa`, after `v0.28.0rc2` had already
  been merged into the gfx906 release branch. The merge brought the three
  upstream rc2-to-final commits listed in that merge commit.
