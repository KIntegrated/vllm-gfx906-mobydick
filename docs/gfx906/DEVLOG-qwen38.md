# Dev Log — Qwen3.8-27B on gfx906

Copyright Kevin Read <me@kevin-read.com>

Getting `cyankiwi/Qwen3.8-27B-AWQ-INT4` to load and serve on the local
vLLM build (gfx906/main), after two unrelated machine-killing NAS
outages (see `qwen3.8_crash.md`).

## Model facts

- `Qwen3_5ForConditionalGeneration`, `qwen3_5_text`, 64 layers,
  hidden 5120, 24 q-heads / 4 kv-heads, **FA head_dim 256** (27B is
  128), GDN head_k_dim 128, `max_position_embeddings 262144`, MTP head
  present (`mtp_num_hidden_layers: 1`).
- **compressed-tensors W4A16** (`quant_method: compressed-tensors`,
  `format: pack-quantized`, I32 packed weights + I32 zero points),
  *not* the legacy auto_awq format of the 27B/MoE. Dense linears run
  on `TritonW4A16LinearKernel` (log-verified), GDN on the Triton/FLA
  kernels, q/k/v projections **quantized** (27B left them unconverted).
- Unquantized tensors (norms, embed, layer-0 GDN, mtp) are **BF16** in
  the checkpoint; config has **no `torch_dtype`**.
- Local copy: `/local/cache/huggingface/hub/models--cyankiwi--
  Qwen3.8-27B-AWQ-INT4` (moved from the non-canonical
  `/local/cache/huggingface/models--...` on 2026-08-19; all internal
  symlinks relative, `refs/main` intact; offline resolution verified).

## The bf16 crash (third incident, clean failure)

Server launch (local weights, `--enforce-eager`) crashed in the
`_dummy_run` profile forward with:

```
RuntimeError: fused_add_rms_norm, .../layernorm_kernels.hip:320
```

`.hip:320` = `STD_TORCH_CHECK(input.scalar_type() ==
residual.scalar_type())` — a **dtype mismatch** at a decoder layer's
`post_attention_layernorm` (qwen3_next.py:533).

Root-cause chain:

1. No `torch_dtype` in the config → `get_torch_dtype` falls back to
   the safetensors weight dtype → **bf16** (dominant unquantized
   dtype).
2. `_resolve_auto_dtype` trusts it: bf16 is in the ROCm
   `supported_dtypes` → model dtype = bf16.
3. **gfx906 has no native bfloat16** (CDNA1/Vega20 ISA; user-confirmed
   silicon fact), and the gfx906 kernel stack (AWQ q_gemm, gfx906 FA,
   GEMV) is fp16-only → the attention output / residual stream dtypes
   diverge inside a layer and the fused_add host check fires.

Cross-check: another user runs this exact model on gfx906 vLLM with
`--dtype float16` (plus tool/reasoning parser flags) — confirmed
fp16 is a workable dtype for the model (it is not in
`_FLOAT16_NOT_SUPPORTED_MODELS`).

## Fix (committed with this log)

- `Platform.supports_native_bf16` property (base: `True`).
- `RocmPlatform` override: `not _ON_GFX906` (the arch flag already
  existed; `_GCN_ARCH` comes from amdsmi with no CUDA init, so the
  check is safe at config-parse time).
- `_resolve_auto_dtype`: when `config_dtype == bf16` and float16 is in
  the model's supported set and the platform lacks native bf16 →
  **fall back to float16 with a warning** (explicit `--dtype` still
  overrides). The `float16 in supported_dtypes` test keeps
  fp16-forbidden models (gemma2/3, glm4, plamo2) on bf16.
- `_get_and_verify_dtype`: explicit `--dtype bfloat16` on such a
  platform now **warns** (still honored — user's choice).
- `tests/config/test_dtype_resolution.py`: 4 unit tests (native
  stays bf16; non-native auto falls back; fp16-forbidden model keeps
  bf16; fp16 unaffected). All pass.

## Validation

- Real (unmocked) resolution on the actual checkpoint:
  - `auto` → warning + **float16** (previously: silent bf16 → crash)
  - explicit `bfloat16` → warning + bfloat16 (honored)
- Server with `--dtype float16 --enforce-eager`
  (`/local/tmp_q38/fp16_eager.log`): loads clean (60 s from local
  NVMe), **KV pool 92,521 tokens**, serves coherent completions
  (Eiffel-Tower probe). ~9 min of startup is vision-encoder
  profiling (VL model) + Triton JIT warmup.
- D=256 FA path: eager forward passes; **graph-capture mode and any
  speed number for this model are still open** (eager run chosen first
  to de-risk after the NAS incidents).

## Open items

- Graph-mode run + single-request serving speed for Qwen3.8 (and the
  D=256 capture path at serving scale).
- MTP k=2 on 3.8 (checkpoint carries the head; recipe exists from the
  27B work) — speculative decoding on a bf16-native model running
  fp16.
- PPL/coherence sanity probe (existing `ppl_probe.py` recipe).
- `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (17 G, local) still sits in
  the non-canonical cache layout; move to `hub/` when it's picked up.
  Note: gemma models are the fp16-forbidden family — on gfx906 they
  hit the exact tension this fix documents (bf16 required for
  accuracy vs. no native bf16 silicon).
