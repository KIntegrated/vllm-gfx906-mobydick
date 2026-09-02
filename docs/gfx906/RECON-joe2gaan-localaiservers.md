# joe2gaan/localaiservers qwen36-gfx906 — recon notes (2026-09-01)

Cloned to /local/git/localaiservers. Qwen3.6 27B dense + MoE deploy bundle for
gfx906 MI50, ROCm 7.2. Runs TP=8 on an 8-GPU full-BAR/P2P-on host (custom VBIOS
113-D1631700-111). Different topology than ours (TP=2) but the OPTIMIZATION
TECHNIQUES are directly relevant to long-ctx TP decode.

## Key gfx906 perf levers (from profiles/vnext/hf-dense27b-tp8.env)
1. **PERSISTENT ALL-REDUCE** — `VLLM_GFX906_PERSISTENT_AR=1` + prebuilt
   `libgfx906_persistent_tree_ll_ar_default_20260613.so`. Persistent tree LL AR,
   preinit on graph capture, multi-row + multi-work variants, watchdog. This
   attacks TP all-reduce cost directly — the #1 suspect for long-ctx decode
   where comm is a big fraction of step time. **Most relevant lead for us.**
2. **ROW-PARALLEL MUTABLE AR** — `VLLM_GFX906_ROWPAR_MUTABLE_AR=1`, boundary cut
   at MLP shape 2176x5120.
3. **INTERLEAVED SWIGLU GEMV ext** — custom `.so` for the Qwen MLP
   (`gfx906_swiglu_gemv_ext_*.so`).
4. **Custom RCCL overlay** — `/rccl-overlay/install/lib/librccl.so.1` + hand-tuned
   `RCCL_TREES`, `NCCL_ALGO=Tree NCCL_PROTO=LL NCCL_MIN/MAX_NCHANNELS=4`.
5. `FLASH_ATTENTION_TRITON_AMD_REF=TRUE` (ref FA mode), `GPU_MEMORY_UTILIZATION=0.95`,
   `MAX_NUM_BATCHED_TOKENS=4`, `MAX_NUM_SEQS=2`.

## How it's wired
All loaded via a **patch bundle** (`/opt/vllm_patch_bundle`, manifest SHA in env)
+ prebuilt `.so` libs inside their docker image
(`joe2gaan/localaiservers:qwen36-gfx906-rocm72-dense-moe-runtime-archive-...`).
So adopting = pulling their image, extracting the `.so` + patch code. The Python
hooks that load these are in their vLLM source overlays (overlays/hf/minimal-bundle/),
NOT necessarily in our tree — need to diff.

## Runnables of interest
- run_qwen36_live_tps.py (live TPS bench)
- run_v02_profile_benchmark.sh + run_vnext_profile_benchmark.sh (profile benches)
- overlays/hf/minimal-bundle/vllm/... (their vLLM source patches, incl gpu_model_runner.py)

## Relevance to MTP-1 speedup
If our 120k decode profiler shows all-reduce = large chunk of step time →
porting persistent-AR (lead 1) is the highest-value gfx906-specific win.
RCCL/NCCL tuning (lead 4) is a quick env-only A/B to try first.
