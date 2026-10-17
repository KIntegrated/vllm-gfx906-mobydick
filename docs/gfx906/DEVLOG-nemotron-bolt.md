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
