# Running & building the gfx906 fork

Quick reference so this doesn't have to be rediscovered. Two environments:
the **local editable `.venv`** (canonical since 2026-08-16, §0) and the
legacy **docker images** (§1–4). Hardware + toolchain selection + source
mount are the three things that consistently trip people up.

**Host:** single AMD MI60 (32 GB, gfx906). All examples use
`HIP_VISIBLE_DEVICES=0`. This is a **hostless ROCm** -> every container needs
`--device /dev/kfd --device /dev/dri` plus the video/render group IDs.

---

## 0. Local venv (canonical environment)

The serving benches moved out of docker on 2026-08-16. The `.venv` holds an
editable install of this repo; the compiled extensions live in-tree.

```bash
# /opt/rocm is the default ROCm toolchain on this box — no env sourcing
# needed (HK-1, confirmed 2026-08-29: serving boots + in-process suite +
# FA micro-bench all run without it). rocminfo resolves via /usr/bin.
```

- **TP=2+ serving needs the HIP blocking-sync `.pth` shim, once per venv**
  (2026-08-25): without it, every TP worker permanently pegs 1-2 host
  cores at ~100%+ each for the life of the process (HIP's default
  `hipDeviceScheduleAuto` resolves to active-wait whenever GPU count <
  CPU thread count — true of almost any TP box — and the flag can only
  be flipped before the process's first HIP queue is created, so it
  must run before torch/vLLM import anything; see
  `degradation_details.md` 2026-08-25 for the full trace). One-time
  setup:

  ```bash
  cp docs/gfx906/gfx906-blocking-sync.pth .venv/lib/python3.12/site-packages/
  ```

  Then export `VLLM_GFX906_HIP_LIB_PATH` (the absolute path to your ROCm
  install's `libamdhip64.so`) alongside `LD_LIBRARY_PATH` in every launch
  — a bare SONAME search fails silently at this early point if
  `LD_LIBRARY_PATH` isn't already populated:

  ```bash
  export VLLM_GFX906_HIP_LIB_PATH=/opt/rocm/lib/libamdhip64.so.7
  ```
  (2026-08: ROCm 7.14 lives at `/opt/rocm`; the old custom
  `/opt/rocm-7.14` install was removed — update the path to match your
  install's `lib/`.)

  Set `VLLM_GFX906_HIP_BLOCKING_SYNC=0` to disable (e.g. if
  lowest-dispatch-latency active-wait matters more than idle CPU for a
  given deployment). This `.pth` file must be re-copied after any fresh
  `.venv` rebuild — it does not survive `uv venv`/`pip install -e .`.

- The system `/opt/rocm` libs are the wrong vintage (libhipsparse symbol
  mismatch; RCCL missing `ncclCommResume` until the 7.14 point release).
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is **required** at import time:
  the venv's `flash_attn` is the `/local/git/flash-attention-gfx906` fork
  without a built C ext; the env selects the Triton-AMD path the ViT
  attention wrapper needs.
- **fastsafetensors** loads: `BENCH_LOAD_FORMAT=fastsafetensors`, 41 s vs
  117 s (2.6×). GDS is unsupported here; the fork's one-line fallback fix
  catches the bare `Exception`. Cost: +2.8 GiB live at init → needs
  `BENCH_GPU_UTIL=0.95` (dense bench uses 0.92 + explicit KV cap instead).
- MoE serving recipe: see `README.md` §Bench recipes. Dense 27B (NFS model,
  no fastsafetensors): `BENCH_GPU_UTIL=0.92 BENCH_KV_MEM=6442450944
  BENCH_MAXSEQS=8 BENCH_BATCHED_TOKENS=4096 BENCH_TEXT_ONLY=1`.

### Building the C/HIP extensions locally

- **`pip install -e .` does NOT compile** (PEP 517 editable flow only links
  the package). Use:

  ```bash
  # No env sourcing needed: /opt/rocm is the default toolchain (HK-1,
  # confirmed 2026-08-29 — build + serving + in-process suite run without
  # it; ROCm resolves to /opt/rocm by default when ROCM_PATH is unset).
  export PATH="$PWD/.venv/bin:$PATH"          # venv cmake must win
  # VLLM_VERSION_OVERRIDE: the branch tag `gfx906-main-pre-promotion` does not
  # parse as a version (setup.py's setuptools_scm step crashes on it); set it
  # to match the existing vllm/_version.py so the build proceeds.
  export VLLM_VERSION_OVERRIDE=0.28.0rc2
  FETCHCONTENT_BASE_DIR=/tmp/vllm-deps \
  HIP_VISIBLE_DEVICES=0 .venv/bin/python setup.py build_ext --inplace
  ```

- `FETCHCONTENT_BASE_DIR` is needed because the in-tree `.deps` is
  root-owned (docker-era); setup.py honours the env var. Reboots wipe
  `/tmp/vllm-deps`, so also export `TRITON_KERNELS_SRC_DIR` at the
  already-fetched package dir (in-tree `.deps/triton_kernels-src` is
  kept) to skip the ROCm/triton git clone:

  ```bash
  export TRITON_KERNELS_SRC_DIR=$PWD/.deps/triton_kernels-src/python/triton_kernels/triton_kernels
  ```
- `ccache` is wired in automatically by setup.py; incremental rebuilds are
  minutes, a full flag-change rebuild of all HIP objects ~5 min on 16 cores.
- **Extra HIP flags**: the `CMAKE_HIP_FLAGS` env var is NOT imported into
  the CMake cache by CMake. Either pass via `CMAKE_ARGS` (no spaces in the
  value; setup.py splits it) or edit
  `build/temp.linux-x86_64-cpython-312/CMakeCache.txt` directly
  (`CMAKE_HIP_FLAGS:STRING=...`) and re-run; ninja rebuilds every object
  whose flags changed.
- **Docker-originated build trees**: a `CMakeCache.txt` created inside
  docker (different ROCm/prefix paths) does not reconfigure cleanly with
  different host paths; delete it for a fresh configure.
- Verify an extension loads:
  `.venv/bin/python -c "import torch; from vllm import _gfx906_fa_C as e; print('OK', e.forward)"`

---

## 1. Stable launch recipe

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --ipc host \
  --group-add 993 --group-add 44 \
  -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -v /data/cache/huggingface:/root/.cache/huggingface:ro \
  -v /tmp/bench:/bench \
  --entrypoint bash \
  <image> -c '...'
```

- `--group-add 993` (render) + `--group-add 44` (video) are required to
  enumerate the GPUs.
- `--ipc host` so the spawn'd engine-core shares memory.
- `HF_HUB_OFFLINE=1` + read-only `/data/cache/huggingface` mount = offline
  model loading. Switch the mount to read-write if a model isn't cached and
  you need to download it.
- Keep leftover processes dead: `VLLM::EngineCore` **hogs VRAM** after a
  crash/kill. Clean with:
  ```bash
  for pid in $(ps -eo pid,args | grep -iE "VLLM::EngineCore" | grep -v grep | awk '{print $1}'); do
    docker run --rm --pid=host --privileged docker.io/library/busybox:latest kill -9 "$pid"
  done
  ```

### Container images (this fork)

| image | code | ROCm | toolchain / arch override |
|-------|------|------|---------------------------|
| `aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0` | upstream 0.23 | 7.2.1 | **`HSA_OVERRIDE_GFX_VERSION=9.0.6` REQUIRED** |
| `mixa3607/vllm-gfx906:0.26.0-rocm-7.2.1-kintegrated` | gfx906 0.26 | 7.2.1 | **`HSA_OVERRIDE_GFX_VERSION=9.0.6` REQUIRED** |
| `mixa3607/vllm-gfx906:0.27.99rc0-rocm-7.14-kintegrated` | gfx906 main | **7.14** | **NO HSA override** (7.14 has native gfx906) |

**Arch override rule:** the 7.2.1 images are built for a different arch and
need `-e HSA_OVERRIDE_GFX_VERSION=9.0.6`; the 7.14 image must **NOT** get that
env var.

### PyTorch / HIP interaction (7.14 gotcha)

On ROCm 7.14, importing torch breaks `amdsmi` (it returns 0 handles), so the
"detect ROCm via amdsmi" path is unreliable. This fork already detects ROCm
via `torch.version.hip` and derives the device name from the GCN arch — just be
aware if you see empty device/ROCm detection on 7.14.

---

## 2. GPU memory pressure

The MI60/MI50 has 32 GB. Current validated practice (2026-08):
`--gpu-memory-utilization 0.82` for spec-decoding serving, `0.93` for
dense TP=1 serving (0.95 OOMs on the second request with a warm
inductor cache — the warm-cache profiling peak is ~0.16 GiB lower than
cold), `0.95` for the in-process bench harness with an explicit KV
cap (`--kv-cache-memory-bytes`). See README §Recommended serving
configuration. Large fp16 models (Qwen3.6-27B = 52 G,
Qwen3.6-35B-A3B = 67 G) do **not** fit single-GPU.

---

## 3. Running a benchmark against the **installed** vLLM (no source shadow)

The `docker-bake` `vllm-v2` preset pins `VLLM_COMMIT` and builds from the
**remote GitHub** (`git fetch` over SSH), so images contain their own vLLM
install. To benchmark the image's vLLM exactly as shipped, **DO NOT** mount the
repo over it (that shadows the installed package with an un-compiled tree).
Instead mount only a `/bench` dir with the script:

```bash
mkdir -p /tmp/bench && cp <fork>/docs/gfx906/_bench_gfx906.py /tmp/bench/_b.py
docker run ... -v /tmp/bench:/bench <image> -c \
  "BENCH_PP=2048 BENCH_TG=256 BENCH_GPU_UTIL=0.85 BENCH_MAXLEN=3328 python3 -u /bench/_b.py 'QuantTrio/Qwen3.5-9B-AWQ'"
```

Full zero-ambiguity runner scripts lived here historically; the current
canonical runner is the local venv recipe in §0.

### `_bench_gfx906.py` env-var surface (W1 additions, 2026-08-26)

Core knobs (documented in the harness header): `BENCH_PP`, `BENCH_TG`,
`BENCH_GPU_UTIL`, `BENCH_MAX_SEQS`, `BENCH_SAMPLES`, `BENCH_EAGER`.
The W1 work (`DEVLOG-gdn-mixed-decode.md`) added:

- `BENCH_NREQS=<n>` — send `n` prompts as one batch (default 1).
- `BENCH_SPEC_CONFIG='<json>'` — speculative-decoding config for the
  harness engine (e.g. the production ngram config
  `{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":2}`).
- `BENCH_CG_MAX=<n>` — cap the cudagraph capture sizes (use a multiple
  of `num_speculative_tokens + 1` for ngram; e.g. 12 for n=5, 2 requests).
- `BENCH_MIXED=1` — with `BENCH_NREQS>=2`: request 0 gets a 2048-token
  repetitive filler (always drafts), the rest get the 190-token diverse
  sentence pool ending mid-sentence (rarely drafts) → spec-mixed
  batches on most decode steps. The W1 gate recipe:
  `BENCH_NREQS=2 BENCH_MIXED=1 BENCH_CG_MAX=12` + the ngram
  `BENCH_SPEC_CONFIG`, graph, 0.82, `BENCH_MAX_SEQS=4`, 4 samples/arm.

## 4. Developing against / modifying the vLLM **source**

For compiling and validating changes to the fork's Python/C++ (e.g. the custom
`CUSTOM` FA backend), source-mount the repo and put it on `PYTHONPATH` so the
editable/compiled tree wins over the image's installed copy:

```bash
docker run ... \
  -e PYTHONPATH=/workspace/vllm \
  -e VLLM_WORKER_MULTIPROC_METHOD=fork \
  -w /workspace/vllm -v "$PWD:/workspace/vllm" \
  <image> -c "cd /workspace/vllm && python3 -u /bench/your_test.py"
```

- `VLLM_WORKER_MULTIPROC_METHOD=fork` (not `spawn`) is required because vLLM's
  engine-core spawns a subprocess; spawn forces `__main__`-guard requirements.
  When the CUDA/HIP runtime is touched before `__main__`, vLLM may force
  `spawn` — guard with `if __name__ == "__main__"`.
- **Register third-party backends at module level** (not inside `main()`) so
  the spawn'd engine-core re-import sees them — e.g. the CUSTOM FA backend.
- The built `_gfx906_fa_C.cpython-312-*.so` lands in-tree and is gitignored; a
  build rebuilds it under `vllm/`.

### Rebuilding the C/C++ extension in the 7.14 image

```bash
pip install cmake && apt install -y pkg-config
export PKG_CONFIG_PATH=/opt/rocm/core-7.14/lib/rocm_sysdeps/lib/pkgconfig  # libdrm for amdgpu-arch
# launch with --device /dev/kfd /dev/dri and -e PYTORCH_ROCM_ARCH=gfx906
pip install -e . --no-build-isolation --no-deps
```

Without cmake/pkg-config/PKG_CONFIG_PATH the `amdgpu-arch` probe fails and the
build produces no kernel/extension. Build the wheel for `gfx906`. Verify load:
`python3 -c "import torch; from vllm import _gfx906_fa_C as e; print('OK', e.forward)"`.

---

## 5. Platform / backend overrides for testing

- CUSTOM (gfx906 FA) is the **default** for attention on gfx906 (prefill
  AND decode); you don't pass `--attention-backend`.
- Escape back to stock: `--attention-backend ROCM_ATTN` (or
  `VLLM_ATTENTION_BACKEND=ROCM_ATTN`).
- The validated decode path is LEGACY inline-Q8 KV mode (default
  `GFX906_FA_LEGACY=1`, FULL-capture-safe). Setting `GFX906_FA_LEGACY=0`
  selects the Q8 side-buffer fast path which desyncs on warmup (garbage output) — don't use
  it as a default.