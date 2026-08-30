# gfx906 changelog

This file records roadmap items that are complete, rejected, superseded, or
otherwise closed. Active work, deferred work, and changes that are local but
still need upstream merging remain in the roadmap files. Dates are landing or
merge dates where the repository history provides one; they are not necessarily
the date an investigation began.

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
  open. Records: `DEVLOG-fa-attention.md` (M2 + 2026-08-29 review-fix
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
