# More model families on gfx906 — portability findings and per-model roadmap
Copyright Kevin Read <me@kevin-read.com>

Status: findings + roadmap (2026-08-18) — **not a committed plan.**
Written after the MoE M=1 sprint (`DEVLOG-moe-m1-sprint.md`) to answer:
*do the Qwen3.5-35B-A3B-AWQ optimizations transfer to other models on
this machine?* Short answer: the MoE GEMM work transfers to any
**W4A16 (AWQ int4, group-128) MoE with compatible shapes**; the
routing and attention work is Qwen3.5-specific; the design patterns,
methodology, and platform facts are model-independent. Two concrete
candidates were evaluated: **inclusionAI/Ling-3.0-tiny** (on disk in
the HF cache) and **DeepSeek-V4-Flash** (config from the HF hub).
Evidence states follow the house convention: **measured** /
**derived** / **hypothesis**.

## 1. What gates each optimization (portability of the stack)

The stack is layered. A model only gets a piece of it if it satisfies
the layer's conditions — none of which are "model family" tests:

| layer | conditions | Qwen3.5-35B-AWQ | Ling-3.0-tiny | DeepSeek-V4-Flash |
|---|---|---|---|---|
| **W4A16 MoE expert GEMM** (incl. S5 M=1 re-tile) | layout: int4 static + group scales [E,G,N] fp16, packed zeros, fp16 activations; kernel: BM∈{1,2,4,8,16}, N%8==0, N%4==0, K%groups==0; S5 tile adds N%256==0, K%64==0, K≤2048, M=1 decode | ✅ measured | ❌ **BF16 weights** — the WNA16 oracle never routes it | ❌ native **FP8 dense / FP4 experts** (`expert_dtype: fp4`, e4m3 block-128) |
| **S2 M=1 topk kernel** | hard-coded in `moe_topk_gfx906.cu`: `K_TOP=8`, 32 lanes × VPT=8 → **E=256, topk-8, SCORING_SOFTMAX, no expert bias** | ✅ measured (default OFF) | ❌ `sigmoid` scoring, `noaux_tc` group-limited (n_group=8, topk_group=4), expert bias, **E=128** | ❌ `sqrtsoftplus`, `noaux_tc`, **topk=6 of 256** |
| **Dense GEMV dispatch** (P3-2/4, S3) | fp16 weights, M=1 (`n==1`), exact measured shapes: K=2048 & m∈{1,256,≥2048}; K=512 & m=2048 (shared down); K=17408 & m=5120 (27B down). The *kernel* is shape-generic (any N, K%8==0, kchunk∈{512,1024,2048,4096}) | ✅ measured | ❌ hidden **1536** — no condition fires; BF16 also rejected (fp16-only check) | ❌ hidden **4096**, FP8 weights |
| **GDN decode / mamba-state work** | vLLM's Qwen3-Next GDN implementation (`mamba/gdn/`) | ✅ measured | ❌ Ling's linear attention is **KDA** (`kda_lower_bound`, `kda_safe_gate`) — a different family; note the tree *does* ship `mamba/linear/bailing_linear_attn.py` (generic upstream impl, unmeasured on gfx906) | ❌ no GDN (MLA/DSA attention) |
| **FA decode (q8 paged GQA)** | standard paged GQA, head_dim 128, vLLM FA backend path | ✅ measured | ❌ FA layers are **MLA-style** (`kv_lora_rank=512`, `q_lora_rank=256`, `qk_head_dim=192` = 128 nope + 64 rope) — latent KV, not paged GQA | ❌ **DSA sparse-indexer attention** (`index_topk=512`, `index_head_dim=128`) + per-layer KV compression (`compress_ratios` 0/4/128) + sliding window 128 |
| **Platform** (DP 2-card serving, RCCL 6.3.3-only dispatch, PCIe topology facts, build recipes, AWQ load fixes, thermal/measurement rules) | machine-level | ✅ | ✅ | ✅ |

**Model-class support in this vLLM tree** (checked, not assumed):
- `BailingMoeV3ForCausalLM` → `vllm/model_executor/models/bailing_moe_v3.py`
  **exists** (plus `bailing_moe_v3_mtp.py`); KDA linear attention
  implemented at `vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py`.
- `DeepseekV4ForCausalLM` → `vllm/models/deepseek_v4/` **exists** with an
  `amd/` variant and ROCm aiter MLA attention (plus MTP, DSpark draft
  model, tokenizer/renderer/warmup). gfx906-specific kernel support for
  those paths is **unverified** (hypothesis: the aiter/MLA kernels
  target CDNA and will need the same gfx906 porting treatment our FA
  work got).

## 2. The portable assets (what to reuse, not re-derive)

1. **Kernel design patterns** (`docs/gfx906/README.md`,
   `latency-hiding.md`, the gfx906 hub): lane-based columns +
   wave-per-K-slice + **single-wave epilogue** (the S5 ×8-CAS lesson:
   with lane-based columns all waves hold the same reduced value for
   the same cells — direct stores hide it, CAS must fire exactly once);
   per-lane **distinct CAS targets** (intra-wavefront same-address CAS
   contention is pathological on ROCm 7.14 gfx906: lost updates at 2¹¹,
   aperture-violation aborts); NPT/`uint4` load discipline; the LDS
   layout standard; `__device__` globals are silently broken in
   standalone `clang++ -x hip` probes (pass buffers as kernel args).
2. **Methodology** (the sprint's working process, model-independent):
   standalone A/B harness **before** touching the model path
   (`benchmarks/kernels/gfx906/harness/moe_m1_harness.cu` takes K/N/E/topk as args — it is
   the re-validation tool for other shapes); per-kernel torch-profiler
   µs rows are unreliable (trust wall-clock); greedy-hash gate for
   bit-equal work, **PPL gate** (`benchmarks/kernels/gfx906/ppl_probe.py`) when
   accumulation order changes; serving A/B (graph + eager) as the
   final gate; thermal-noise awareness (2026-08-18 serving numbers
   flagged directional for this reason).
3. **Platform facts**: DP (two single-GPU vLLM servers) is the only
   validated multi-card path; TP=2 is closed (chipset-attached card
   poisons PCIe under RCCL); RCCL 2.21.5/ROCm 6.3.3 is the only
   dispatching RCCL; MI50 HBM ≈ 800 GB/s effective floor;
   GDN mamba state ≈ 72 MB/seq makes `max_num_seqs` the memory lever.
4. **Shape-generic kernels**: the base `moe_gemm_q4_kernel_gfx906`
   (any E/shapes within §1 conditions) and `dense_gemv_kernel`
   (any N, K%8, the four kchunks). S5's M=1 V2 tile is the newest
   addition and is currently capped at K≤2048.

## 3. Ready-to-benefit: any AWQ W4A16 MoE (no new kernels)

Any MoE checkpoint in AWQ int4 group-128 form **with stored zero
points** served on gfx906 with fp16 activations is routed to the same
expert kernel by layout, not by model class
(`gfx906_w4a16_moe.py` / `oracle/int_wna16.py`). The gate was
AWQ-only until 2026-08-20: symmetric no-zp (GPTQ-style) checkpoints are
now accepted too — the kernel already dequanted `(q-8)*scale` for missing
zero points, and only the Python-side gates + one repack layout branch
had to open (gemma-4-26B-A4B-AWQ, §6; `DEVLOG-gemma4-moe.md`).
On this disk the only AWQ-with-zp MoE is the 35B itself (Qwen3.5-27B-AWQ
is dense and already gets the K=17408 GEMV). The nearest untested family member is
**Qwen3-30B-A3B-AWQ** (E=128, topk-8, hidden 2048, inter 768):

- gemm1 N=1536×K=2048, gemm2 N=2048×K=768 — both satisfy the S5 V2
  constraints (N%256, K%64, K≤2048) → the M=1 re-tile should apply
  as-is; **per-shape micro-bench first** (harness args, 30 min).
- Dense linears are standard MHA (no GDN): qkv N≈6144/K=2048 and
  o_proj N=2048/K=2048 hit the existing K=2048 GEMV dispatch
  (m≥2048); router [128, 2048] does not (m=128) — measure, likely
  stays LLMM1.
- S2 topk does **not** apply (E=128 vs the hard-coded 256) — the
  generic topkGating handles it; a templated M=1 variant is a small
  follow-up if the generic one shows up in a profile.
- No shared-expert K=512 shape (its shared inter is 768) — S3's
  dispatch condition won't fire; measure [2048, 768] if a profile says
  so.

**Checklist to onboard any W4A16 MoE** (each step exists today):
1. `shape_spy` pass: attribute every per-layer linear (M, K, x/step).
2. Micro-bench each shape (GEMV bench + MoE harness) → dispatch
   decisions. 3. Greedy-hash (bit-equal work) **or** PPL A/B
   (order-changing work). 4. Serving A/B graph+eager, 4 samples.
5. Kill-switch env per new dispatch, default decided by the A/B.

## 4. Ling-3.0-tiny (`BailingMoeV3ForCausalLM`)

Config facts (from the on-disk snapshot, HF cache
`models--inclusionAI--Ling-3.0-tiny`): 24 layers, hidden **1536**,
layer 0 dense (inter 4608, `first_k_dense_replace=1`), 23 MoE layers:
**E=128, topk=8**, moe_inter **512**, 1 shared expert (inter 512),
`sigmoid` scoring + `noaux_tc` (8 groups, 4 selected) + expert bias,
`routed_scaling_factor=2.5`; hybrid attention: KDA linear layers +
MLA-style full attention (`kv_lora_rank=512`, `q_lora_rank=256`,
`qk_head_dim=192`, `v_head_dim=128`, `layer_group_size=4`,
`max_window_layers=20`, partial rotary 0.5, short-conv 4); **BF16,
unquantized**; vocab 157184. Estimated ≈7.5B params → ≈15 GB BF16,
fits one MI50 32 GB (derived, unmeasured).

**What applies now:** platform layer only (DP, build, measurement
protocol). Model class + KDA + MLA model code exist in-tree but are
unmeasured on gfx906.

**Work needed (phases, in dependency order):**
- **L1 — get it running** (hypothesis: days, not hours): load the
  BF16 checkpoint, confirm `BailingMoeV3` + `bailing_linear_attn` +
  MLA decode all function on gfx906 (the MLA/KDA kernels likely need
  the gfx906 porting pass our FA/GDN kernels got — check for
  CDNA-only intrinsics/aiter paths first). Greedy probe + PPL
  baseline.
- **L2 — profile + baseline table**: shape spy + kernel_prof_probe →
  the per-step µs table (the re-anchor discipline: no assumptions).
- **L3 — MoE expert GEMM, BF16**: new kernel family (W16A16 grouped
  skinny GEMM, E=128, inter 512, hidden 1536). The S5 design ports
  directly (lane columns, wave-per-K-slice, single-wave epilogue,
  per-lane distinct cells); K=1536 (gemm1) and K=512 (gemm2) both fit
  the K≤2048 re-tile. Micro-bench vs the Triton/aiter path per shape
  before dispatch wiring.
- **L4 — routing**: `sigmoid`+`noaux_tc`+bias is the generic
  topkGating's territory (it exists for DeepSeek-style models); an
  M=1 dedicated kernel is a follow-up **only if** the profiler shows
  the 713 µs/step-class gap here (it won't match 35B's numbers —
  E=128 halves the search space).
- **L5 — attention decode**: KDA recurrent decode + MLA paged decode
  on gfx906 (separate workstream; the GDN/FA experience is the
  pattern, not the code).

**Stop rule:** if L1 surfaces a hard gfx906 blocker in the MLA/KDA
kernels, park the model — the expert-GEMM work (L3) is the only
high-value piece and shouldn't be built for a model that can't run.

## 5. DeepSeek-V4-Flash

Config facts (HF hub `deepseek-ai/DeepSeek-V4-Flash-0731`, pulled
2026-08-18): 43 layers + 1 MTP, hidden **4096**; attention:
`head_dim=512`, DSA sparse indexer (`index_n_heads=64`,
`index_topk=512`, `index_head_dim=128`), `o_groups=8`,
`o_lora_rank=1024`, `q_lora_rank=1024`, per-layer KV compression
(`compress_ratios` pattern 0/0/4/128/...), sliding window 128,
YaRN-16; MoE: **E=256 + 1 shared, topk=6**, moe_inter **2048**,
`sqrtsoftplus` scoring, `noaux_tc`, `routed_scaling_factor=1.5`,
`swiglu_limit=10`, hyper-connections (`hc_mult=4`, sinkhorn),
**`expert_dtype: fp4`**; quantization: **FP8 e4m3** weights, 128×128
block, ue8m0 scales, dynamic activations; `torch_dtype: bfloat16`;
vocab 129280.

**Hard blockers (before any optimization is in scope):**
1. **Capacity (measured arithmetic):** ≈256 experts × 25.2M params per
   MoE layer ≈ 6.4B/layer × 43 ≈ **≈140 GB at FP4** (≈280 GB at FP8)
   for the experts alone. One MI50: 32 GB. Two cards: 64 GB, and
   **TP=2 is closed on this machine** — DP replicates, it doesn't
   shard. **DeepSeek-V4-Flash is not servable on this machine** at any
   realistic quantization (even AWQ-int4 experts ≈ 70–75 GB > 64 GB),
   absent a much smaller "flash" variant. *This is the deciding
   fact; the rest of this section is for the record.*
2. **Format:** gfx906 (Vega20/GCN5) has no FP8/FP4 hardware — compute
   would be dequant-to-fp16/bf16, where the W4A16 kernel family is the
   natural design (if the experts were ever AWQ-quantized, the S5
   re-tile applies with a **K=4096 extension** — the current V2 tile
   caps at K≤2048; gemm1 N=4096/K=4096, gemm2 N=4096/K=2048).
3. **Attention:** DSA indexer + compressed KV + sliding window is the
   `vllm/models/deepseek_v4/` ROCm path (exists, aiter-based) —
   gfx906 support unverified; our FA work (standard paged GQA) does
   not transfer to it.
4. **Routing:** `sqrtsoftplus` + `noaux_tc` + topk-6 — generic
   topkGating territory; S2's kernel (softmax, topk-8, E=256) does not
   apply as-is.

**Verdict: no gfx906 optimization roadmap exists for V4-Flash on this
machine** — the model doesn't fit. If a smaller V4 variant (or a
community int4 quant under 32 GB) appears, the onboarding checklist
(§3) plus items 2–4 above is the plan; the design patterns (§2.1)
transfer unchanged.

## 6. Gemma-4-26B-A4B-AWQ (onboarded 2026-08-19, `DEVLOG-gemma4-onboarding.md`)

`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (17.2 GB, local, standard HF
layout under `/local/cache/huggingface/hub/`). Supersedes the §3 note
that gemma-4 was GGUF-only. Facts that broke the "reuse the existing
kernel" premise of P1:

- **All 30 layers are MoE** (E=128, topk=8, moe_inter 704) **plus a
  per-layer shared expert** (unquantized bf16 in layers 0–26).
- Quantization is compressed-tensors W4A16 **group-32, symmetric, no
  zero points** (`weight_packed` + `weight_scale` only) → initially
  routed to **Triton WNA16** by the then-AWQ-only oracle gate (correct
  at the time; our kernel path needed the gates opened, 2026-08-20,
  below).
- Hybrid attention (25 sliding hd-256 + 5 full hd-512 layers with
  `attention_k_eq_v` + proportional RoPE) → `Gemma4Config` forces the
  **TRITON_ATTN** backend here (no FA4). Works; 20% of the decode step.
- fp16 (forced by the auto-dtype gfx906 fallback — second model to
  exercise `69f615b98a`) is numerically fine; bf16 is structurally
  blocked (our fp16-only dense path asserts).
- **Raw-prompt degeneration trap:** bare continuation prompts produce
  confident repetition loops on this thinking-mode instruct model;
  templated messages are coherent. Gates must use the chat template.

Record: **37.6 t/s** (graph, pp=2048 tg=256, 4 samples, util 0.95,
max-seqs 32); KV pool 53,434 tokens. Census: MoE Triton expert GEMMs
**46% of the step** (232.8 µs/call × 59.8/step ≈ ~38 GB/s effective),
attention 17%, shared-expert `LLGemm1` 12%.

**Lever — DONE 2026-08-20 (`DEVLOG-gemma4-moe.md`).** The no-zp path
turned out to need **zero kernel changes**: `moe_gemm_q4` already
dequants `(q-8)*scale` when zero points are missing (the repack
fabricates `0x88888888`). The entire blocker was Python-side: two
oracle gates (may-have-zp, GPTQ-style rejection), one missing
GPTQ-K-first repack layout branch, and a latent upstream bug where
`compressed_tensors_moe_wna16._setup_kernel` never forwarded
`backend=` to `make_wna16_moe_kernel` (would have crashed any gfx906
W4A16 MoE load). Serving A/B (same recipe): **37.81 → 67.79 t/s
(1.793×)**, 26.4 → 14.75 ms/step; numerics at fp16-noise level vs
Triton (|ΔLP| of sampled token, 491 matching steps: median 0.0017,
p99 0.31); flagship Qwen3.5-35B unchanged (66.27, band 65.9–67.0);
58/58 MoE tests. Note: PPL is invalid as a gate on this model
(prefill-logprob anomaly in the hybrid sliding/`k_eq_v` attention,
both arms equally — separate investigation).

### 6.1 Review todos (post-`180f030ee3` review, 2026-08-20)

Ordered by risk; **HIGH** items are fails-open paths that will bite
when the next non-gemma-4 symmetric checkpoint arrives:

- [x] **HIGH — gate doesn't check bit-width / group size / strategy —
  RESOLVED (`253942905c`).** New `_gfx906_no_zp_reason()` in the
  GFX906_HIP branch of `_backend_incompatibility_reason` rejects
  symmetric no-zp configs that are not 4-bit, static (non-dynamic)
  scaled, group-strategy, and group size 32 or 128 (checkpoint-
  validated; the kernel's per-32-K-slice group tracking would accept
  any multiple of 32 — widen with a per-shape micro-bench). Rejected
  configs fall through to the Triton backend. (`int_wna16.py`)
- [x] **HIGH — GPTQ activation ordering (`g_idx`) unguarded —
  RESOLVED (`5e3cf6d780`).** The same helper rejects `actorder in
  (group, dynamic)`: those store weights in original column order and
  need a runtime g_idx reordering the kernel and repack lack (a
  silent mis-dequant). `weight`/`static` are format-identical to no
  activation ordering and keep passing, matching the Marlin/Triton
  treatment. (`int_wna16.py`)
- [x] **MED — fabricated-zp memory waste — RESOLVED
  (`20df23b80f`).** The repack returns `None` for symmetric input, the
  op wrapper passes an empty tensor, and both gfx906 kernels inline
  the constant zero point 8 when `b_qzeros` is null — no `[E, G, N/8]`
  tensor is materialized (~16 MiB/layer at gemma-4) or streamed.
  Symmetric layers no longer carry `zero_point` parameters; the CPU
  path is unchanged. (`int_wna16.py`,
  `csrc/rocm/moe_q_gemm_gfx906.cu`, `vllm/_custom_ops.py`)
- [x] **MED — strengthen the numerics-gate record — RESOLVED
  (2026-08-20, `DEVLOG-gemma4-moe.md` "Post-review MED items").**
  Per-step re-run (probe at `benchmarks/kernels/gfx906/
  gemma4_divergence_probe.py`): first-diff positions are spread
  through the decode (median 20 of 64; 2/8 in the first two steps),
  diff-position |ΔLP| sits inside the matching-step noise band, and
  prefill flatness does not separate divergent from matching prompts —
  the divergence is a near-tie argmax flip in pure decode, not
  coupled to the garbage-prefill regime. Verdict unchanged: no
  systematic dequant error.
- [ ] **LOW — doc drift.** §7 item 3 still lists "gemm1 M=1 re-tile"
  among the 35B roadmap's open levers; it was closed 2026-08-19
  (`DEVLOG-moe-gemm1-retiling.md`). Update when next editing §7.
- [ ] **LOW — quote the 1.793× carefully.** It is a single-request
  (batch-1) figure; under parallel requests the Triton arm's MoE cost
  grows superlinearly, so the true factor likely *widens* — never
  quote 1.793× as a universal multiplier without the regime.

## 7. Priorities

1. **P1 (cheap, high confidence):** onboard the next AWQ W4A16 MoE
   checkpoint as it arrives (Qwen3-30B-A3B-AWQ-shaped): checklist §3,
   ~1 day per model, zero new kernels. (gemma-4-26B-A4B done: onboarding
   2026-08-19 + no-zp kernel gate 2026-08-20, §6 — now 67.79 t/s.)
2. **DONE 2026-08-20:** the gemma-4 no-zp W4A16 MoE expert kernel (§6)
   — 37.81 → 67.79 t/s (1.793×); Python-gate work, no kernel changes.
3. **P2 (one machine, one model at a time):** Ling-3.0-tiny L1+L2
   (make it run, get the table) — only after the 35B roadmap's big
   open levers (gemm1 M=1 re-tile, S2' router-GEMV fusion,
   `moe-decode-roadmap.md`) are spent, or if the 35B work is declared
   done.
4. **P3 (park):** DeepSeek-V4-Flash — capacity-blocked (§5.1); revisit
   only on a smaller variant or a 3rd+ card with a working TP/PP path.

**House protocol applies to all of it**: micro-bench per shape before
the model path, PPL/greedy gates, serving A/B, separate commits,
positive **and** negative results in the DEVLOG.
