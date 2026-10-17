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
