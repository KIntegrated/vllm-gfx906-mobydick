# Nemotron-3.5-Bolt 74B-A9B (AutoRound W4A16) on 2x MI50 — bring-up

**VERDICT:** SHIPPED for serving/perf, QUALITY-OPEN · **GATE:** TP=2 boot +
in-context generation + decode t/s · **Model:**
`SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym` at
`/biglocal/models/SergiioB/…` (41 GB).

## HYPOTHESIS

A 41 GB AutoRound/GPTQ W4A16 NemotronH-Omni MoE (74B total / 9B active,
top-k 9, `moe_latent_size=1024`) can serve on 2x32 GB MI50 with the fork's
gfx906 WNA16 MoE kernel as the fastest backend.

## What was done

Downloaded with `hf download --local-dir /biglocal/models/<org>/<name>`
(`HF_HOME=/biglocal/cache/hf`), ~59 min (xet ~5-9 MB/s). Reference recipe:
the model card's XPU one (vLLM 0.31.0, TP=2) minus flashinfer-attention /
flashinfer-mamba / fp8-KV / bf16, all unavailable or unsupported on gfx906;
added `--mamba-cache-mode align`, fp16, maxlen 16384, image-only mm limits,
both model-card support files (`ultra_v3_reasoning_parser.py`, bf16).

Weight layout (verified by HTTP-range reading shard headers): GPTQ K-first
`qweight [K/8, N]`, `scales [G, N]`, `qzeros [G, N/8]` with stored nibbles
**7** (= 8-1, GPTQ convention), `g_idx` = per-channel group id (desc_act
false, ignore-safe), vision tower / projector / embeddings / lm_head / MoE
router gates bf16-f32 (unquantized).

## Evidence / fixes (each a commit on `gfx906/v0.31.0`)

- **Boot blocked: "auto_gptq quantization is currently not supported in
  rocm".** `vllm/platforms/rocm.py`'s `supported_quantization` had
  `"auto_gpt"` — commit `69f615b98a` (an unrelated dtype change) had
  clobbered `"auto_gptq"`. Restored.
- **Profiling run died: `expected mat1 and mat2 to have the same dtype:
  Half != BFloat16`.** `nano_nemotron_vl.py` built the RADIO tower / `mlp1`
  from `self.language_model.config.dtype` (the HF config says bfloat16) and
  cast `vit_embeds` with two hardcoded `torch.bfloat16`. Now uses the
  resolved `model_config.dtype` and `self.llm_dtype`.
- **MoE backend: TritonWNA16 -> GFX906_HIP.** The AutoGPTQ MoE fallback on
  gfx906 rewrites the layer to `MoeWNA16Config(has_zp=False,
  linear_quant_method="gptq")`, which the oracle rejected ("zero points are
  required") because `_is_symmetric_no_zp` only knew compressed-tensors
  `QuantizationArgs`. Taught it the MoeWNA16 symmetric layout (N-first
  uint8, constant zp 8) plus a symmetric-AutoGPTQ path (drop the stored
  zp 7 so the kernel inlines 8). Log now reads `Using 'GFX906_HIP' WNA16
  MoE backend`; `tests/quantization/test_moe_wna16.py` 50 passed.

## Gates

- **Boot**: TP=2, fp16, V1 and V2 both boot; maxlen 16384; KV 7.1 GiB/card
  at util 0.90; ~24.7 GiB/card resident.
- **Decode** (greedy, `ignore_eos`, 128 tok, 3 reps, gfx906 MoE):
  **48.7 t/s @216-prompt**, **39.7 t/s @856** (the XPU card reports 59.5 t/s
  @886 on bf16/fp8-KV/TRITON_ATTN).
- **In-context generation is correct**: capitals (`Berlin`, `Rome`,
  `Madrid`), counting (`6,7,8…`), times-table chains (`48`, `54`), weekday
  lists, `fibonacci` body — all right.
- **Logits are sane**: `The capital of France is` -> ` Paris` at logprob
  −0.07, top-2 −4.04 (confident, not flat).

## Evidence AGAINST — quality is OPEN

- **Open-ended / chat generation degenerates into repetition** after a few
  tokens: `The capital of France is` -> ` Paris. Paris is the capital of
  France. …`; chat thinking on -> `<think>` repeated; thinking off ->
  `\n<think></think>` repeated. Same on V1 and V2, and on the raw
  `/v1/completions` path with the exact chat-token ids (so it is **not** the
  `ultra_v3` reasoning parser).
- Arithmetic slips under the reasoning scaffold: few-shot "84 * 3 / 2" ->
  reasoning "84*3 = 102, 102/2 = 51", answer `51` (repeated). Fluent but
  wrong.
- Tokenization is not the cause: `/tokenize` round-trips
  `<|im_start|>system\n…assistant\n<think>\n` exactly (added tokens present).
- Leading candidate: the checkpoint was calibrated with **bf16
  activations**; gfx906 is fp16-only (the unquantized router gates / norms /
  vision are bf16 in the checkpoint). Not yet excluded, and not testable
  here. Second candidate: NemotronH-Omni hybrid state handling on this
  stack (V1 and V2 behave identically).

**Next**: A/B the router/unquantized-path precision, compare a greedy
decode against a short fp32 reference on a fixed prompt set, and try the
NVIDIA chat template variant with `medium_effort`/reasoning-effort set.

## 2026-10-16 (later) — the repetition is a chat-template/reasoning-effort issue, not dequant

**VERDICT:** the `<think>` loop is fixed by the template args · **GATE:** chat answers.

The model needs the template's **reasoning-effort hint**: with
`chat_template_kwargs={"enable_thinking": true, "medium_effort": true}` (the
template appends `{reasoning effort: efficient}` to the last user message) the
loop disappears and the model answers normally:

- "What is 84 * 3 / 2?" -> reasoning "84 * 3 = 252. 252 / 2 = 126." +
  content `126` (stop).
- "capital of France/Japan" -> `Paris.` / `Tokyo.`
- first five primes -> `2 3 5 7 11`.
- Without `medium_effort` the same requests emit `<think>` / `<think></think>`
  forever (V1 and V2 alike); the reasoning parser is not involved (raw
  `/v1/completions` with the exact chat ids reproduces it).

Residual quality (QUALITY-OPEN): arithmetic is unreliable (17*23 -> 413,
3 KiB -> 3096), some prompts never close the think block within a normal
`max_tokens` (empty content; `force_nonempty_content` exists for that), and a
few still repeat with a stray `</s></think>`. The template knobs are
**per-request** in 0.31.0 — there is no `--default-chat-template-kwargs` CLI
flag — so clients must send them. Leading residual cause is still the
bf16-calibrated checkpoint running on fp16 (gfx906 has no bf16).

## Official `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` — NOT runnable on gfx906

**VERDICT:** DEAD-END for this box. The card documents **ModelOpt** quant:
routed+shared experts at **IQ2-XXS (2.0625 bits, ggml-style block packing)**,
Mamba layers at **NVFP4** (4-bit float, group 16), embeddings/output at
**Q8_0**. Hardware compatibility is stated as **NVIDIA Blackwell SM120/SM121**
(RTX 5090 / Spark), and the recipe needs a vLLM *nightly*
(`nightly-77154ca6116…`) plus `vllm download-kernels` (the IQ2/NVFP4 kernels
are CUDA/Blackwell artefacts). gfx906 has no FP4 and no IQ2 packing path in
vLLM 0.31.0's `modelopt`/compressed-tensors stack, so the 2-bit experts would
not dequant at all — and even the FP4 Mamba layers are unsupported. The
41 GB AutoRound W4A16 build is the runnable choice here (this log).

## 2026-10-17 — LiLiCorr draft model (DFlash spec decode): loads, runs, 0 acceptance

**VERDICT:** OPEN — the drafter runs and spec decode is exact, but it accepts 0
draft tokens on this target. · **GATE:** `spec_decode_num_accepted_tokens_total`.

Model: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` (962 MB,
`LiLiCorrDraftModel`, qwen3 backbone + LiLiCorr head, `W4A16_NVFP4`). Official
config is `{"method":"dflash","model":…,"num_speculative_tokens":7,
"rejection_sample_method":"standard","draft_load_config":{"load_format":"auto"}}`
(the card's `attention_backend: FLASHINFER` dropped — unavailable on ROCm).

**FP4 is not runnable on gfx906 as-is.** `modelopt_fp4` + `W4A16_NVFP4` maps to
"FP4 Marlin GEMM", and `init_nvfp4_linear_kernel` *forces* the Marlin kernel for
W4A16 (`ValueError: Forced NVFP4 kernel MarlinNvFp4LinearKernel is not supported:
Marlin FP4 not available`). The `emulation` NVFP4 kernel is W4A4 (it quantizes
activations), so it does not cover this checkpoint either.

**Fix: convert to bf16 offline** (`/local/tmp/dequant_lilicorr.py`). 25 FP4
linears (`weight` U8 packed e2m1, `weight_scale` F8_E4M3 per 16, `weight_scale_2`
F32 global) dequantized to bf16 = 2.2 GB. Verified **bit-identical** to vLLM's
`dequantize_to_dtype(..., swizzle=False)` (max abs diff 0.0). The 104 BF16 / 25
F32 tensors are copied as-is; `quantization_config` dropped from `config.json`.
Result: `/biglocal/models/nvidia/…-LiLiCorr-bf16`. The GGUF variant
(`…-LiLiCorr-GGUF`, one 998 MB `dflash`-arch GGUF) is llama.cpp-oriented and was
not needed — the safetensors dequant was simpler and bit-exact.

**Also fixed en route** (`49e17fdca7`): `triton_attn.py` still read the v0.30-
removed `CommonAttentionMetadata._seq_lens_cpu` (`seq_lens_cpu_upper_bound` now).
Latent before: serving always picked the custom gfx906 FA; the drafter forces
`TRITON_ATTN` (the custom FA has no attention sinks) and hit AttributeError.

**What works.** Boots TP=2, fp16, maxlen 16384: `LiLiCorrForCausalLM` loads, the
LiLiCorr speculator captures its graphs, drafts are produced (7/step), and
spec decode is **exact** — greedy outputs are byte-identical to the non-spec
runs. Embeddings/lm_head are shared from the target by the LiLiCorr speculator;
the aux hidden-state width check (6 taps x 4096) passes.

**What does not.** `accepted_tokens_total = 0` at every draft position 0..6,
over ~2000 draft tokens, at temperature 0 **and** 1.0 → decode 18 t/s vs the
48.7 t/s non-spec baseline (pure overhead). Per-position 0 is also 0, so the
very first proposal never matches the target's choice.

Candidate causes, not yet separated:

1. **Target-specific drafter.** The card ties this drafter to the 24GB **IQ2**
   target; our target is the 41 GB W4A16 AutoRound build (same BF16 base,
   different quant). Hidden states/logits differ, and the LiLiCorr head was
   trained to correct the IQ2 target specifically.
2. **A gfx906 bug in the draft's Triton path.** The drafter forces TRITON_ATTN
   (no sinks in the custom FA) for a *non-causal + 1024-SWA + attention-sink*
   stack, and its layers use the DFlash2 grouped-conv kernel plus the LiLiCorr
   head's Triton sampler/scorer. None of those were exercised on gfx906 before.
3. The hybrid-target warning `no KV cache group could be identified as the
   draft model's` (mamba groups present, no `is_eagle_group` annotation); the
   draft does get a group (the speculator asserts on it), so this is likely
   benign, but it is unverified.

**Next**: dump the draft's top-1 per slot against the target's argmax on a fixed
prompt (distinguishes 1 from 2), and run the same drafter against a BF16 target
if one fits; if it is the Triton path, the sink/non-causal/SWA handling in
`triton_attn.py` is the place to look.

## 2026-10-17 (later) — lab fixes ported; the LiLiCorr 0-acceptance survives the fp16 rescale

Ported from `0c33/mi50-vllm-lab` (Apache-2.0; README "Ported optimizations"):
the DFlash2/LiLiCorr **fp16 residual rescale** (`cc9e994ac9`), the **non-causal
sliding window** in the custom FA + its freed-block test (`3161af4cda`), and the
**triton_attn** `getattr(..., None)` form (`a7b29d3c4c`). The test suites also
surfaced a merge miss of my own: upstream's
`test_context_kv_uses_quantized_projection_fallback` (from the LiLiCorr PR
#57934) — the fork's `_dense_kv_rows` had replaced upstream's per-projection
fallback instead of sitting behind it (`7763dd7ee1`).
Suites: `test_dflash2` + `test_dflash_causality` + `test_gfx906_fa` = 133
passed / 10 skipped.

**The LiLiCorr drafter still accepts 0.** With the rescale active (log:
`Draft in float16: residual stream scaled by 1/32
(VLLM_LILICORR_FP16_RESIDUAL_DIVISOR…)`), 3 requests gave
drafts=285, draft_tokens=1995, **accepted=0, position-0 accepted=0**, 17.9 t/s
(vs 48.7 non-spec). So the fp16 residual overflow is **not** the LiLiCorr
failure mode: either its stream does not overflow at this divisor, or the cause
is elsewhere (the draft's candidates, the LiLiCorr lattice scoring, or the
target aux-hidden capture for the Omni target). The port itself stands — it is
output-preserving and is the lab's measured DFlash2 fix (10.2 → 48.4 t/s).

Note the window fix does not touch this gate either: the LiLiCorr drafter runs
`TRITON_ATTN` (the custom FA has no attention sinks), and it fails at short
context, before the freed-block regime.

**Next diagnostic:** dump the draft's per-slot top-1 (and the target's chosen
token) on a fixed prompt, and the aux-hidden magnitudes the drafter receives
(`target_layer_ids` [1,18,35,51,68,85]) — that separates "draft produces
garbage" from "lattice/candidate scoring broken" from "wrong target features".

### BOLT-1 diagnostic — the NaN is inside the drafter's own forward

Gated instrumentation (`VLLM_LILICORR_DEBUG`; patch saved at
`/local/tmp/lilidbg.patch`), fp16 rescale active, one real greedy request:

**Healthy upstream.** The target→drafter features are fine: 6 aux hidden states
captured, concatenation absmax 533, `fc` output / `self.hidden_states[7]`
absmax 3764 with `nan=0`, and the anchor is valid and non-zero (absmax 21.4,
`anchor_valid=[True]`, `qsl=[0,8]`, anchor index 7 inside the filled range).
The drafter's weights are fine too: `hidden_norm.w` 2.219, `fc.w` 5.188,
`attention_sink_bias` 6.97 (rank 1: 6.53), no NaNs anywhere.

**The drafter output is 100 % NaN.** `last_hidden_states` is all-NaN
([1,7,4096] = 28672 elements), so the candidate logits are NaN, the top-k list
degenerates to the constant `[3,2,1,0,6,7,4,5]`, the LiLiCorr head scores are
NaN (448/448), and every slot argmaxes to candidate 0 ⇒
`draft_tokens = [5,5,5,5,5,5,5]` on every step ⇒ 0 acceptance. (The earlier
2-boot dump that showed `anchor=0` was the zero-input profile dummy run.)

**Ruled out:** target aux-hidden capture / `target_layer_ids` indexing; the
`prepare_context_anchor` indexing; uninitialised `sink` / `hidden_norm` / `fc`
weights; the shared-embedding and LM-head path (`maybe_share_target_embed`
replaces both and the checkpoint ships neither — it has only
`lilicorr.rank_embedding` / `slot_embedding`); absence of the rescale.

**Remaining, in order:** (1) fp16 overflow *inside* the drafter — the rescale
covers only `o_proj`/`up`/`down` + the layer-0 residual, while the grouped
dynamic conv multiplies un-rescaled activations; (2) `TRITON_ATTN` semantics:
the model card says the draft is *bidirectional within the block* with a
1024-token sliding window and sinks, but the Triton decoder path encodes the
window as `(1023, 0)` (left only), which cannot express the block mask — a
fully-masked row NaNs. The lab never ran a drafter on TRITON_ATTN: their DFlash2
drafter ran on the custom FA, which refuses this drafter (no sink support).

**Next discriminators (one boot each):** `VLLM_LILICORR_FP16_RESIDUAL_DIVISOR`
256/1024 (overflow), then a drafter-eager run with per-layer NaN dumps (mask /
kernel).

**VERDICT (2026-10-17): the LiLiCorr drafter works — the lab's rescale needs divisor
2048 for this architecture, not 32.**
`/biglocal/models/SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym`
(TP=2, fp16) + the bf16 LiLiCorr drafter, k=7, graphs: **39 drafts / 264 accepted
= 6.77 tokens per step**, acceptance 100 % at positions 0–3, **96.2 t/s** over
three 96-token requests (vs **48.7** non-spec and **17.9** with the broken
drafter). Greedy output unchanged.

Root cause and the fix (both measured, not guessed):

- **The drafter's grouped dynamic conv overflows fp16.** Per-layer bisect
  (eager run, `DFlash2Qwen3DecoderLayer.forward` traces at L0):

  ```
  L0 in_hidden         absmax=0.166   nan=0
  L0 attn_out          absmax=716     nan=0
  L0 attn_conv_finish  absmax=inf     ← `_grouped_conv`: 716 × |tap| (≤151)
  L0 post_attn_ln      nan=7
  ```

  The conv taps are the *learned* `base_kernel` (+ a small projected delta), so
  the gain (~150×) is **not** scaled by the residual rescale, and it is applied
  after the attention **and** again after the MLP. With the DFlash2 value (32)
  the attention conv alone hits 1.1e5; at 128 the attention conv fits (46k) but
  the MLP conv overflows (`mlp_out` 470 × ~150); the unscaled stream is ~4e7 by
  the last layer, i.e. fp16 needs a divisor >~1024. At **2048** every layer is
  finite with ~3.5× headroom (L4 `mlp_conv_finish` 18.7k) and acceptance is
  6.77/7. The lab's 32 was calibrated on the conv-less z-lab DFlash2 drafter.

- The rescale itself is still output-preserving here: every branch reads the
  stream through an RMSNorm, and the conv is linear in the branch output (the
  `base_kernel` term dominates the projected delta), so a uniform divisor only
  changes the representation, not the result — confirmed by the greedy output
  matching the non-spec arm token for token.

Implementation: `fit_residual_stream_to_fp16(..., default_divisor=2048.0)` at the
LiLiCorr call site (DFlash2 keeps 32); `VLLM_LILICORR_FP16_RESIDUAL_DIVISOR`
still overrides, `=1` disables. Values 32/128/256 were each tested and fail
(128: MLP conv inf; 256: layers 0–1 finite, L4 NaN).

### Follow-ups after the 2048 fix

**Chat works, and the earlier kwargs recipe was wrong.** With the template's
defaults (no `chat_template_kwargs`; the jinja already defaults
`enable_thinking=True`, `medium_effort=False`) a chat request to the 74B target
+ LiLiCorr drafter answers correctly (`5` for the 5-machines/5-widgets puzzle),
`finish_reason=stop`, 262 tokens, **acceptance 6.53 and 108–133 t/s**. Both
non-default modes degenerate: `medium_effort=true` appends
`{reasoning effort: efficient}` and the model never leaves `</think>` (loops to
`finish_reason=length`), and `enable_thinking=false` emits `<think></think>`
repeatedly. So the recipe recorded at bring-up ("needs
`enable_thinking/medium_effort`") is inverted for this checkpoint; drop the
kwargs.

**TP=1 is infeasible.** The W4A16 checkpoint is 41 GB on disk and a single MI50
offers 34.3 GB, so the weights do not fit before any KV cache. TP=1 would need
expert offload (issue #44's territory), not a different launch flag.

**BOLT-2 (new): acceptance collapses intermittently past ~4k context.** The same
~7.3k-token prompt gives `acc/draft = 1.0, 1.0, 5.65` across three greedy reps,
while ~2.5k is stable at `5.52, 5.52, 5.52`. `1.0` means only the bonus token's
candidate is accepted — the classic shape of a drafter reading context it should
not, and the same family as the lab's freed-block finding, but this drafter runs
on **TRITON_ATTN** (the custom FA refuses it: no sink support), which the lab
never exercised with a drafter. Prefill and decode themselves are fine at 7.3k.

### Kernel ports (LAB-1/2/3) — measured on this Bolt setup

Ported from `0c33/mi50-vllm-lab` (Apache-2.0; sources under
`tools/gfx906_skinny/`, hooks in `layers/utils.py`, `exllama.py`,
`cuda_communicator.py`). Build: `.venv/bin/python tools/gfx906_skinny/build.py`
→ `tools/gfx906_skinny/build/gfx906_skinny.so`; enable with
`VLLM_GFX906_SKINNY8_LIB` (fp16 + W4) and `VLLM_GFX906_HOSTAR_LIB` (TP=2 AR).

Microbenchmarks (`bench_skinny.py`, `check_w4.py`, `test_hostar.py`):

| shape (M=7) | current dispatch | skinny v20 | speedup |
|---|---|---|---|
| lm_head 124160×5120 | 6249 µs | **1808 µs** | 3.5× |
| gate_up 17408×5120 | 875 µs | 274 µs | 3.2× |
| fc 5120×25600 | 1312 µs | 402 µs | 3.3× |
| ctx_kv 5120×5120 | 265 µs | 97 µs | 2.7× |

M=1 is a wash or slower, so the hook starts at M=2 (the lab's starts at 1).
W4 vs the float64 reference: **3.3–4.9e-4** rel err vs Exllama's 1.2–1.4e-3
(closer to exact, as the lab reported). hostar: **0 mismatches over 3000
random calls**, 73.7 → **18.0 µs** for 80 KB, 18.8 µs and bit-identical inside a
CUDA graph.

Served A/B (Bolt 74B W4A16 + LiLiCorr k=7, TP=2, same prompt, 3×96 tokens,
greedy, two reps per arm, arms sequential):

| arm | t/s (2 reps) | ms/step | acc/draft |
|---|---|---|---|
| kernels OFF | 96.0, 103.3 | 77.0, 71.5 | 6.77 |
| kernels ON | **119.5, 130.7** | **61.8, 56.5** | 6.77 |

**+25 % t/s / −20 % ms/step at identical acceptance** (non-overlapping ranges,
so the effect is well outside this host's noise; the arms were sequential, not
interleaved, which is the remaining caveat). Hooks confirmed engaged in the log
(`gfx906 skinny8 GEMM enabled`, `gfx906 host-memory all-reduce enabled for TP=2
(<= 131072 bytes)`).

**LAB-4 (amdsmi-first `.pth`) — not adopted.** Tested in the venv: with
`import amdsmi` in a `.pth`, the `amdsmi_shut_down failed …(32)` warning
disappears, but `current_platform.get_device_name()` changes from
`AMD_GFX906` to the generic `AMD Radeon Graphics` (a different amdsmi query
order). `on_gfx906()` is `True` either way, so nothing breaks, but there is no
functional gain on this host — the lab needed it for container amdsmi boot
failures, which we do not have. Documented and dropped.

### BOLT-2 discrimination — the backend is not the cause

Four arms, same two prompts (3 greedy reps each; `acc/draft`):

| arm | drafter backend | sinks | prefill chunk | 2.5k | 7.3k |
|---|---|---|---|---|---|
| control | TRITON_ATTN | on | 4096 | 5.52, 5.52, 5.52 | 1.0, 1.0, 5.65 |
| A | TRITON_ATTN (`attention_backend`) | **off** | 4096 | 5.38, 2.18, 5.38 | **1.0, 5.65, 1.0** |
| D | **CUSTOM (gfx906 FA)** | **off** | 4096 | 5.52, 5.38, 5.65 | **1.0, 5.65, 1.0** |
| E | TRITON_ATTN | on | **8192 (single chunk)** | 5.52, 5.38, 5.38 | **5.65, 1.0, 1.0** |

**Widening the custom FA would not fix it.** The custom FA reproduces the
collapse with the *same values* (1.0 and 5.65) as Triton, and the sink path is
irrelevant (arm A), as is prefill chunking (arm E). 2.5k is stable in every arm;
the trigger is context length, and the failure is intermittent on identical
input — so it is state in the **drafter's context/KV path** (speculator side),
not the attention implementation.

Arm D also exposed a genuine fork bug: with the sink bias dropped the custom FA
becomes eligible for the drafter, and the engine died on
`Gfx906FABackend.get_supported_kernel_block_sizes() takes 0 positional
arguments but 1 was given` — v0.31 added a `kv_cache_spec` argument that the
fork's override never got, unreachable until today. Fixed in `68e00234c2` with
a merge-drift guard test that fails on the old signature.

Next for #54: instrument the drafter's attention metadata at 2.5k vs 7.3k
(`seqused_k`, window, causal, block-table span) plus the number of context slots
`precompute_and_store_context_kv` actually writes; then poison the drafter's KV
cache with NaN and see whether the draft output depends on the poisoned region.

## 2026-10-17 — DFLASH2-GATE (#51): both ported fixes validated on a real drafter

Setup: `cyankiwi/Qwen3.8-27B-AWQ-INT4` target + **`syvai/Qwen3.8-27B-DFlash2-W4A16`**
drafter (compressed-tensors pack-quantized, conv layers, `sliding_window: 2048`,
no sink bias → the drafter runs on the **custom FA**), TP=2, fp16, k=7,
`--max-model-len 16384`, `--max-num-seqs 1`, capture `[8]`. Harness: the ported
`tools/gfx906_lab_bench/{bench,longctx}.py` (#56).

**The rescale needed a new code path: quantized drafters.** The ported
`fit_residual_stream_to_fp16` assumed a dense `o_proj.weight`; on this drafter
that raised `AttributeError: 'RowParallelLinear' object has no attribute
'weight'` at load. It now rescales **`weight_scale`** when there is no dense
weight (pack-quantized dequantizes as `(q - zp) * scale`, so scaling the scale
is exactly equivalent), sliced the same way (up half of `gate_up_proj`,
`down_proj` by `up_divisor/divisor`), and it runs before the quant scheme's
`process_weights_after_loading`.

| arm | 2.0k prompt | 7.3k prompt |
|---|---|---|
| rescale **off** (dense path skipped) | **0.3 % acc**, 16.7 t/s mean | — |
| rescale **on** (1/32, `bench.py`, 3 prompts x 2 reps) | **51.4 % acc**, **73.2 t/s** mean | — |
| window fix **on** (`longctx.py`) | 53.3 % acc, 73.4 t/s | **29.4 % acc**, 45.9 t/s |
| window fix **reverted** (`window = 0` for non-causal) | 51.7 % acc, 71.9 t/s | **7.3 % acc**, 23.1 t/s |

- **Rescale:** 0.3 % → 51.4 % acceptance, 16.7 → **73.2 t/s** — and 73.2 is
  within noise of the lab's own 73.7 t/s mean on their (fp16) drafter. Their
  "every draft rejected" signature reproduced exactly before the fix.
- **Window:** dropping it costs **4x acceptance** at 7.3k (29.4 % → 7.3 %) and
  nothing at 2.0k (53.3 % vs 51.7 %) — the freed-block regime starts past
  ~2-4k, matching the lab's finding. Not the full 0 % they saw (our prompt is a
  repeated source, and the drafter's block cache still held live data), but the
  direction and magnitude match.

Verdict: **both ported fixes are validated end-to-end on hardware.** The
`_rescale_layer` unit-test helper only covers the dense path; the packed path is
covered by this gate rather than by a unit test.

### BOLT-2 root cause: the drafter's context KV is written with two holes

Post-gather logging + a per-request NaN poison (`VLLM_DFLASH_CTXDBG` /
`VLLM_DFLASH_POISON`, patch at `/local/tmp/ctxdbg2.patch`) settled it. The
failure is **on the write side of the speculator**, not in attention:

- **Hole 1 — the first 8 rows.** `_precompute_context_kv` is called as
  `(num_context, num_target_tokens)` with `num_context = min(reqs*(k+1),
  max_num_tokens) = 8`; rows 0..7 are left to the captured graph. The outside
  write therefore starts at row 8, and the log shows it:

  ```
  write n=1600 slots=1600 slot_min=151160 slot_max=152759 pos_min=8 pos_max=1607
  ```

- **Hole 2 — whole earlier prefill chunks.** On a multi-chunk prefill only the
  **last** chunk is stored, because the speculator is only invoked on the step
  that has draft tokens. For a 6.4k prompt the single write is

  ```
  write n=2320 slots=2320 slot_min=159761 slot_max=166592 pos_min=4097 pos_max=6416
  ```

  i.e. **positions 0..4096 never enter the drafter's KV cache.**

- **Poison test.** Filling every draft `kv_cache` with NaN once per request
  before its prefill drops acceptance to **0 at both 1.6k and 6.4k** (`accept
  1.00` on both agentctx prompts), so the draft *depends on slots this request
  does not write* — either the graph's rows 0..7 or positions outside the
  written range.

This explains the threshold exactly: below 4096 only the first 8 positions are
missing (negligible — those tokens rarely decide the next one, so acceptance
stays ~5.5); above 4096 up to 4096 positions are missing, so the drafter's
context is largely stale and acceptance collapses to 1.0. It also explains why
the failure looked backend-, sink- and chunk-agnostic in the earlier arms.

Fix directions: (a) precompute from row **0** in the `else` branch above (the
graph rewrites the same rows, so it is idempotent); (b) store the context KV on
**every** prefill chunk, not just the last one — that is the real fix, and it
needs the target's hidden states for the earlier chunks, which the target only
produces for the step it runs in. (b) is therefore a scheduling-side change
(run the store during chunked prefill) or an explicit prefix store when the
first decode step runs.

### LAB-7 (#55) — the blocking-sync lever is a no-op on a healthy boot

Ported the lab's `.pth` to `tools/gfx906_blocking_sync/` and measured it on the
Bolt + LiLiCorr setup with the kernels enabled (2 reps per arm, same prompt,
sequential arms):

| arm | t/s | ms/step | acc/draft |
|---|---|---|---|
| lever off (recorded arm) | 119.5, 130.7 | 61.8, 56.5 | 6.77 |
| lever on | 119.1, 130.6 | 62.0, 56.5 | 6.77 |

**No effect** (≤0.3 %, inside this host's noise), as expected on a healthy
TP=2 boot with the official DKMS driver. Kept as an opt-in lever for the
sync-cadence wedge family; not installed in the venv.

### BOLT-2 FIXED — store the drafter's context KV from row 0

`_precompute_context_kv(num_context, num_target_tokens)` → `(0, num_target_tokens)` in the
FULL-graph branch of `propose()` (upstream `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`,
regressed by upstream #57632; the file has no fork commits). The extra rows are the ones the captured
graph also writes, so the store is idempotent, and it closes the hole where rows
`[0, num_context)` were never stored on prefill steps.

Validation (Bolt 74B + LiLiCorr k=7, production config; `agentctx.py`, distinct-source prompts,
greedy):

| prompt | before | after |
|---|---|---|
| 1.6k | 2.97 accept | **4.29, 4.23** |
| 6.4k | **1.0, 1.0, 5.65** | **3.75, 5.94, 3.80** |

No collapse in six 6.4k runs (three before at 1.0/1.0/5.65, three after at 3.75/5.94/3.80). Upstream
report written up in `UPSTREAM-dflash-context-kv.md` (no duplicate upstream; #57632 suspected
introducer). The remaining acceptance spread (3.75–5.94) is this stack's normal draft chaos, not the
collapse signature (which was exactly 1.0).

## 2026-10-18 — LAB-9 measurement campaign (does the packed KV layout cost throughput?)

Answer: **no** — and at the kernel level it is *faster*. Two layers of
measurement, both with the layout as the only variable.

**1. Kernel, interleaved per rep** (`benchmarks/kernels/gfx906/fa_packed_layout_bench.py`,
20 reps, decode 1 row/seq, bk=16, HKV=2, HQ=16, equal token counts):

| D | B | ctx | legacy µs | packed µs | Δ |
|---|---|---|---|---|---|
| 128 | 1 | 2048 | 231.9 | 211.9 | −8.6 % |
| 128 | 1 | 16384 | 318.5 | 297.8 | −6.5 % |
| 128 | 4 | 2048 | 245.4 | 221.4 | −9.8 % |
| 128 | 4 | 16384 | 736.6 | 715.1 | −2.9 % |
| 256 | 1 | 2048 | 254.9 | 230.1 | −9.7 % |
| 256 | 1 | 16384 | 384.6 | 356.2 | −7.4 % |
| 256 | 4 | 2048 | 286.2 | 264.5 | −7.6 % |
| 256 | 4 | 16384 | 1051.1 | 1000.8 | −4.8 % |

Outputs are **bit-identical** in all eight shapes (`max|Δ| = 0`), i.e. the same
Q8 K values and fp16 V — the layout claim holds. The cell is 22-23 % smaller and
the decode step is bandwidth-bound, so it gets faster, not slower.

**2. Model level** (35B MoE, single MI50, 16k ctx, **prefix caching ON** — the
combination that had never been run with packed):

| | KV tokens | decode t/s | first TTFT | prefix-cache queries/hits |
|---|---|---|---|---|
| `GFX906_FA_PACKED=0` | 142,987 | 57.1 | 0.64 s | 44 / 0 |
| `GFX906_FA_PACKED=1` | **174,762 (+22.2 %)** | **57.2** | 0.64 s | 44 / 0 |

So end-to-end decode is **neutral** at B=1/16k (the 35B's weight streaming
dominates, which masks the KV-path win) while capacity rises 22 %. Prefix
caching behaves identically; the hit counter stayed at 0 in *both* arms, so the
honest statement is "no change", not "reuse verified".

**Not yet measured:** B=4 model-level, TP=2 + a drafter (the 8-row verify), and
the hybrid aligner case (448 → 576). The aligner is the only *config-visible*
consequence and therefore the only remaining blocker for a default; the 16
legacy-layout tests are the blocker for ever deleting the alias path.

### LAB-9 aligner arm — hybrid models: packed is all risk, no benefit

Hybrid (mamba+attention) `primitive-ai/Nemotron-3.5-Lightning-30B-A3B-mixed-INT4-INT8`,
`--mamba-cache-mode align`, 8k ctx, `GFX906_FA_PACKED` off vs on:

| | scheduler block size | KV tokens | `--prefix-cache-retention-interval 576` |
|---|---|---|---|
| off | **2096** | 526,336 | rejected (not a multiple of 2096) |
| on | **2672** | **528,384 (+0.4 %)** | rejected (not a multiple of 2672) |

Three conclusions:

1. **The flag changes a config-visible number on hybrid models** — the scheduler
   block size moves 2096 → 2672 — so any pinned `--block-size` /
   `--kv-cache-memory-bytes` / `--prefix-cache-retention-interval` is affected.
2. **The PR's motivating case does not reproduce on the 0.31 line.** Their 0.30
   tree moved the aligner 448 → 576 and thereby *unlocked* a pinned 576; here
   both arms reject 576 (the base aligner result is 2096, not 448), and packed
   makes the block size *larger*, not smaller.
3. **The capacity win collapses for hybrid models**: +22 % on the non-hybrid 35B,
   but **+0.4 %** here, because the pool is dominated by the mamba/state groups
   whose size the attention page barely moves.

⇒ **The conditional default's condition is now measured, not guessed:** packed by
default for models **without** hybrid KV groups (where the block size is
unchanged, e.g. 35B: 8192 in both arms) and **off** for hybrid models, where it
must stay opt-in.

**Not implemented yet — and the reason is ordering, not hesitation:** flipping
the default on makes the 16 legacy-layout tests fail, because they construct the
fused cache directly. So the 16-test parameterization is now the blocking item
for the default (not just for deleting the alias path). Doing it first keeps the
suite green; the conditional-default change itself is then small (a
`_packed_enabled()` default keyed on a hybrid-vs-not flag set where
`_align_hybrid_block_size` runs).

### LAB-9 items 3–4 — TP=2 + drafter measured; the default flip needs a better condition

**TP=2 + drafter (8-row verify)**, Qwen3.8-27B-AWQ-INT4 + DFlash2-W4A16 k=7,
`GFX906_FA_PACKED` off vs on, `tools/gfx906_lab_bench/bench.py` (3 prompt
classes × 2 reps, greedy):

| | KV tokens | mean decode | acceptance | accept len |
|---|---|---|---|---|
| off | 184,178 | **73.74 t/s** | 51.9 % | 4.63 |
| on | **205,381 (+11.5 %)** | **73.46 t/s** | 51.5 % | 4.61 |

Neutral throughput and equal acceptance (expected: the kernel output is
bit-identical), with the capacity win holding on the TP=2/drafter path. Together
with the kernel A/B and the 35B arm, no serving path we run regresses.

**Item 4 (flip the default) is deferred, and implementing items 1–2 is what
showed why.** The planned condition was "exclude hybrid models", but
`model_config.is_hybrid` is not a usable proxy for "the aligner would move the
block size":

- Qwen3.5-35B-A3B is flagged **hybrid**, yet its block size is 8192 with *and*
  without packed (only a non-attention group's size moves 1056 → 1376) and it is
  where the +22 % win was measured. Excluding it would drop the win from our own
  main bench model.
- Nemotron-3.5-Lightning-30B (mamba+attention) is where the block size moves
  2096 → 2672 and the win is +0.4 %.

The discriminator is therefore the *aligner outcome*, not the model class: the
flip needs `_align_hybrid_block_size` dry-run twice (packed/alias page) and a
comparison of the resulting `block_size`, restoring the fields it mutates
(`block_size`, `mamba_block_size`, `mamba_page_size_padded`,
`skip_page_size_padded`) in between. That is a change inside the platform's
config path and is not something to land unverified at the end of a session.

So: `gfx906_fa_packed` stays `"off"` by default (config field + CLI + auto/fallback
all landed and tested), and the flip waits for the dry-run comparison. Until
then, the capacity win is available per model with `--gfx906-fa-packed on` or
`GFX906_FA_PACKED=1`.

### LAB-9 item 4 — the default is now `auto`, gated on the aligner's block size

`CacheConfig.gfx906_fa_packed` defaults to **`"auto"`**. On a hybrid model
`auto` dry-runs `_align_hybrid_block_size` for both layouts and keeps packed only
when the derived **scheduler block size is unchanged**; otherwise it logs the
numbers and stays on the alias layout. The dry run runs on a **copy** of the
config, because restoring a snapshot is not equivalent to a fresh run (measured:
2672 restored vs 4192 on a dirty config), and its logging is suppressed so the
log never shows a phantom "packed enabled".

Verified on the two hybrid models we have:

| model | dry run (alias → packed) | `auto` decision | pool |
|---|---|---|---|
| Nemotron-3.5-Lightning-30B | 2096 → **2672** | **off** | 526,336 tok (identical to the control) |
| Qwen3.5-35B-A3B | 1056 → **2112** | **off** | 142,987 tok (identical to the control) |

**Correction to an earlier note in this log:** I had recorded the 35B's block
size as "unchanged (8192)". That was a misreading of `kv cache group sizes`
(8192 is the attention group's context capacity, not the scheduler block size);
the 35B's block size really does move with the layout (1056 → 1376 with the env
override, 1056 → 2112 in the engine's config stage). So the block-size rule
correctly excludes both of our hybrid models, and their **+22 % / +11.5 %**
wins remain opt-in via `--gfx906-fa-packed on`.

Known wart, recorded: the dry run runs in both the API-server and engine-core
config paths and the two report different "would-be" block sizes (2672 vs 4192
for Lightning; 1376 vs 2112 for the 35B) because the config is at a different
stage in each. The decision matches; computing it once (or passing the decision
through) is a follow-up.

Net effect of the flip: models the config does **not** flag as hybrid get the
packed layout by default (no aligner, so no block-size question), with the
auto-fallback covering the unsupported shapes; hybrid models keep today's
layout unless a user opts in, which is the state LAB-10/#58 assumes.

### Our gates now run the packed layout by default

Per the LAB-9 decision, the harnesses pass `--gfx906-fa-packed on` unless
overridden (`BENCH_FA_PACKED=auto|off`), so the layout we ship is the layout we
gate:

- `benchmarks/kernels/gfx906/ppl_probe.py` — in-build A/B on the 35B:
  **alias 15.9104 vs packed 15.9021** (–0.05 %), 0 top-20 misses in both. That
  moves the 35B reference band slightly (the old 16.0066 tablet was alias, and an
  older build); the packed band is ~15.90. The small delta is expected: on this
  hybrid-flagged model packing also moves the scheduler block size (1056 → 2112),
  which changes the kernel tiling, not the maths.
- `docs/gfx906/_bench_gfx906.py` (house serving bench) — same default.

Both print nothing new to parse: the layout shows up in the boot log as
`GFX906_FA_PACKED: KV cell … B` (packed) or its absence (alias).
