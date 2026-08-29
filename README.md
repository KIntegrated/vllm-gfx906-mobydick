## Mini Install Guide for GFX906

## Fork heritage

This repository is the gfx906 vLLM port
[**ai-infos/vllm-gfx906-mobydick**](https://github.com/ai-infos/vllm-gfx906-mobydick),
based on [**nlzy/vllm-gfx906**](https://github.com/nlzy/vllm-gfx906), the
original gfx906 port of vLLM. The custom Q8 FlashAttention kernels below are
vendored from
[**cassettesgoboom/gfx906-fa-vllm**](https://github.com/cassettesgoboom/gfx906-fa-vllm).
See [`docs/gfx906/`](docs/gfx906/) for the full optimization record.

## Custom FlashAttention backend (gfx906 FA, `CUSTOM`)

This fork vendors a custom Q8 FlashAttention attention backend for gfx906
(`AttentionBackendEnum.CUSTOM`, built from
`https://github.com/cassettesgoboom/gfx906-fa-vllm`) and makes it the **default**
for attention on gfx906 (prefill and decode). No `--attention-backend` flag is needed;
`CUSTOM` is automatically selected and the extension ships inside this wheel.

### What it accelerates

The vendor kernels originally target **prefill** on **long contexts** of
**full-attention models**; on *any*-attention hybrids (e.g. Qwen3.5, few
full-attention layers) the prefill gain is small. This fork adds the decode
path (B=1 parallelism via GQA head-packing + KV split, fused
gather-and-quantize, native BSHD output), making `CUSTOM` the default for
**decode** as well: 18.9 → 25.6 t/s serving on dense Qwen3.5-27B, and the
MoE flagship at 66.1 t/s single-request (67.4 record) / 193 t/s concurrent
(N=8, 191.0 record) — final-build restamps 2026-08-24.
See [`docs/gfx906/`](docs/gfx906/) for the full change inventory, numbers,
and bench recipes.

### Model support and performance on gfx906 (single MI50/MI60)

| model | status | decode t/s |
|---|---|---|
| Qwen3.5-35B-A3B-AWQ (MoE) | flagship, fully optimized | **66.1** (restamp; 67.4 record; ~2140 t/s prefill) |
| ↳ N=8 concurrent decode | W4 (`VLLM_GFX906_SKINNY_M16=1`) | **193** (restamp; 191.0 record; +14.5 % vs 166.9) |
| ↳ with MTP k=2 speculative decoding | recommended spec config | **88.6** (restamp; 89.9 record; 1.16× vs 76.7 greedy) |
| Qwen3.5-27B-AWQ (dense) | optimized | **25.6** |
| ↳ with MTP k=2 speculative decoding | recommended spec config | **39.4** (1.41×) |
| Gemma-4-26B-A4B-it-AWQ-4bit | optimized | **67.8** |
| Qwen3.8-27B-AWQ-INT4 (dense) | fully functional (TP=1 + TP=2) | **59.2** (MTP k=2, TP=2, 2k ctx; 2026-08-24 final) |
| ↳ MTP k=2 context curve (TP=2) | live-ctx tax — MTP < greedy past ~20k ctx | 44.9/25.2/16.6 @ 8k/32k/64k (greedy 38.1/30.5/24.1) |
| ↳ N=8 concurrent decode | W4 (`VLLM_GFX906_SKINNY_M16=1`) | **104.2** (TP=1, util 0.90) |
| ↳ 256k context | FA gather fix (2026-08-24) | 250k needle PASS; 16.6 t/s MTP @ ~64k ctx |
| Qwen3.6 fp16 checkpoints (52–67 GB) | do not fit 32 GB | — |

Details, per-model caveats, and bench recipes:
[`docs/gfx906/README.md`](docs/gfx906/README.md) §Model support status.

### Long-context performance (TP=2, 2× MI50 32 GB)

Prefill sweep for the two prime dense models at their max context
(Qwen3.8-27B: 256k; Muse-Glimmer-30B: 128k). Deep prompts, B=1,
tg=128, mean of 2 samples; prefix caching OFF, `--max-num-batched-tokens
4096`, float16, trimmed cudagraph capture `[1,2,3,4]`. 2026-08-29,
boot N (canary 38.9 t/s healthy); csrc @ `cf5ccbd685` (M2 merged + M3
hygiene, bit-identical). Re-run recipe: `docs/gfx906/_serve_tp2_gfx906.sh`
(start/wait/stop; Qwen3.8 at 256k needs `KVBYTES=10737418240`) +
`docs/gfx906/_bench_serve_grid_gfx906.py` with
`'[[32768,128],[65536,128],[112640,128]]' 2`.

| model (max ctx) | 32k prefill | 64k prefill | 110k prefill | TTFT @ 32k/64k/110k |
|---|---:|---:|---:|---|
| Qwen3.8-27B-AWQ-INT4 (256k) | **443.9** | **364.8** | **289.0** | 73.8 s / 179.6 s / 389.7 s |
| Muse-Glimmer-30B-AWQ-INT4 (128k) | **500.0** | **442.1** | **379.6** | 65.5 s / 148.1 s / 296.7 s |

Prefill t/s = pp / TTFT. Live-ctx tax: prefill rate falls ~12–14 % per
doubling for Muse and ~18–21 % for Qwen3.8 (head_dim 256 makes its
attention share scale harder). Decode (byproduct, tg=128, no spec
decode): Muse 30.5 → 26.3 → 21.9 t/s; Qwen3.8 25.6 → (–) → 13.3 t/s
(the 64k sample ended at out=1 — the model hit EOS on the repetitive
filler, a content artifact; TTFT is unaffected). KV budgets: 6 GiB
(783,892-token pool) for Muse, 10 GiB (323,414-token pool) for
Qwen3.8 — the 256k max-len needs ≥ 8.09 GiB of KV.

### Benchmarks

**gfx906 fork — dense AWQ `QuantTrio/Qwen3.5-9B-AWQ` (few full attention
layers), pp = prefill throughput (tok/s), single MI60, eager, pp/tgen two-phase:**

| pp | `CUSTOM` prefill | stock `ROCM_ATTN` prefill | Δ |
| ---: | ---: | ---: | ---: |
|  256 | 590 | 575 | +2.6% |
|  512 | 757 | 764 | −0.9% |
| 1024 | 1483 | 1427 | +3.9% |
| 2048 | 1399 | 1288 | **+8.6%** |

Decode throughput in that table reflects the vendor baseline; this
fork's decode path (above) changes these models' decode numbers
substantially.

**Upstream gfx906-fa-vllm — full-attention `MiniMax-M2.7-AWQ-4bit` (8× MI50,
TP=8, BS=1, from the upstream repo's README):**

| ctx | `CUSTOM` TG (tok/s) | Δ vs stock `TRITON_ATTN` |
| ---: | ---: | ---: |
|  1K | 27.7 | — |
|  32K | 7.7  | +6% |
| 100K | 3.9 | **+32%** |
| 130K | 3.0 | **+29%** |

On a full-attention model at long context the custom kernels give roughly
**+20–40%** prefill/overall throughput and stay functional where the stock
Triton kernels stall.

### Escaping back to the default attention backend

To bypass `CUSTOM` and use vLLM's stock ROCm backend instead:

```bash
vllm serve ... --attention-backend ROCM_ATTN
# or in Python:
# LLM(..., attention_backend="ROCM_ATTN")
```

Set the env `VLLM_ATTENTION_BACKEND=ROCM_ATTN` as well for earlier-stack paths.
If you built without gfx906 (no FA extension compiled), or the backend is not
registered, vLLM automatically falls back to the stock ROCm/TRITON backends.

### 🐳 Using Pre-built Docker Image (Recommended)

If you have Docker and the AMD ROCm drivers/kernel modules installed on your host system, you can totally bypass the complex manual source-build installation by using our pre-built Docker image.

```bash
# Pull the latest image (or specify a tag instead of latest, e.g. v0.19.1rc0.x)
docker pull aiinfos/vllm-gfx906-mobydick:latest

# Run the container interactively (Make sure to pass ROCm devices into the container and have your models in host /home/ as we map /home:/home; feel free to edit the command below to a safer one, without priviledged and others)
sudo docker run -it --name vllm-gfx906-mobydick -v /home:/home --network host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add $(getent group render | cut -d: -f3) \
  --cap-add=SYS_ADMIN --volume /sys:/sys:ro --pid=host --privileged \
  --ipc=host aiinfos/vllm-gfx906-mobydick:latest
```

Once inside the container, you are all set! You can immediately start serving models (see the Quickstart example below).

---

### 🛠️ Manual Build from Source

If you prefer to build and install from source on your bare metal instead, follow the steps below:

### ROCm 6.3.4 & amdgpu drivers

```code
# Get the script that adds the AMD repo for 24.04 (noble)
wget https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
sudo apt install ./amdgpu-install_6.3.60304-1_all.deb

# Install ROCm  6.3.4 including hip, rocblas, amdgpu-dkms etc (assuming the machine has already the advised compatible kernel 6.11)
sudo amdgpu-install --usecase=rocm --rocmrelease=6.3.4    

sudo usermod -aG render,video $USER

# Verify ROCm installation
rocm-smi --showproductname --showdriverversion
rocminfo


# Add iommu=pt if you later grow beyond two GPUs
# ROCm’s NCCL-/RCCL-based frameworks can hang on multi-GPU rigs unless the IOMMU is put in pass-through mode
# see https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.3/reference/install-faq.html#multi-gpu

sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /' /etc/default/grub
sudo update-grub
sudo reboot
cat /proc/cmdline  # >>> to check: must return: "BOOT_IMAGE=... iommu=pt"

```

### vllm-gfx906-mobydick fork with its dependencies (python, torch, triton, flash-attn, etc)

```code

pyenv install 3.12.11
pyenv virtualenv 3.12.11 venv312
pyenv activate venv312

# PYTORCH 2.11.0

git clone --branch v2.11.0 --recursive https://github.com/pytorch/pytorch.git
cd pytorch

# Install Python Dependencies
pip install -r requirements.txt
pip install mkl-static mkl-include

# Hipify the Source (Convert CUDA to ROCm code)
python tools/amd_build/build_amd.py

# Build the wheel and install
export MAX_JOBS=96 # to be adjusted according to your setup to avoid OOM / freeze / crash
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx906
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH}"

pip wheel --no-build-isolation -v -w dist -e . 2>&1 | tee build.log
pip install ./dist/torch*.whl


# TORCHVISION 0.26.0

# Install dependencies
sudo apt-get update && sudo apt-get install -y libpng-dev libjpeg-dev ffmpeg

# Build and Install
git clone --branch v0.26.0 https://github.com/pytorch/vision.git
cd vision
export FORCE_CUDA=1
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx906

python setup.py install


# TORCHAUDIO 2.11.0

# Build and Install
git clone --branch v2.11.0 https://github.com/pytorch/audio.git
cd audio
export PYTORCH_ROCM_ARCH=gfx906
export USE_ROCM=1

python setup.py install


# TRITON-GFX906 V3.6.0

git clone --branch v3.6.0+gfx906 https://github.com/ai-infos/triton-gfx906.git
cd triton-gfx906 
pip install -r python/requirements.txt
TRITON_CODEGEN_BACKENDS="amd" pip wheel --no-build-isolation -w dist . 2>&1 | tee build.log
pip install ./dist/triton-*.whl  


# FLASH-ATTENTION-GFX906 (triton backend)

git clone https://github.com/ai-infos/flash-attention-gfx906.git
cd flash-attention-gfx906
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install

# VLLM-GFX906-MOBYDICK main

git clone https://github.com/ai-infos/vllm-gfx906-mobydick.git
cd vllm-gfx906-mobydick
pip install 'amdsmi>=6.3,<6.4'
pip install -r requirements/rocm.txt
pip wheel --no-build-isolation -v -w dist . 2>&1 | tee build.log
pip install ./dist/vllm-*.whl

# TRANSFORMERS (v5.7.0 or any other version <6 supporting your model)
pip install transformers==5.7.0
```

### Quickstart example (with Qwen3.5-0.8B)

```code
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3.5-0.8B \
  --dtype float16 \
  --kv-cache-dtype float16 \
  2>&1 | tee log.txt
```

NB: --dtype float16 is recommended to add for this gfx906 fork. If not set, vllm will take the dtype from config.json model which might be bfloat16, not natively supported on gfx906 (with potential fallback to float32, leading to slower inference)

CREDITS
-------

- https://github.com/nlzy/vllm-gfx906
- https://github.com/Said-Akbar/vllm-rocm
- https://github.com/vllm-project/vllm

---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
