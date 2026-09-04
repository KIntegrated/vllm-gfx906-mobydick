# MI50 (gfx906) DVFS / clock facts — benchmarking ground rules

Measured 2026-09-04, boot U, GPU0, ROCm 7.14. Scripts:
`/local/tmp/mtp1/syv3_dvfs_test.py`, `/local/tmp/mtp1/syv3_dvfs_gap.py`.

## The facts

| state | mclk | effective HBM2 BW |
|---|---|---|
| idle | **350 MHz** | ~358 GB/s (≈35% of peak) |
| any sustained kernel load | **1000 MHz** | ~1 TB/s theoretical (4096-bit @ 2 Gbps) |

- The boost is fast: mclk was at 1000 MHz within the first 0.5 s sample after
  load started, for BOTH a well-utilizing kernel and a poorly-utilizing one
  (ATen `torch.mm` at ~19% of peak still boosts).
- Per-iteration `torch.cuda.synchronize()` does **not** drop mclk between
  calls (sampled every 0.5 s through a per-sync bench: flat 1000 MHz, times
  flat to within 2%). So "hot" standalone numbers are real.
- COLD benches run right after idle ARE inflated ~3× by the 350 MHz memory
  clock. If a standalone number looks off vs an in-context one, sample
  `rocm-smi --showclocks` (mclk line) during the bench before drawing
  conclusions — do not assume downclocking, and do not assume it's absent.

## The standalone-≠production trap (SYV-3, 2026-09-04)

The biggest error this cost: benchmarking ATen `torch.mm` standalone and
concluding "the stock path leaves 3.3× on the table". Wrong on two counts:

1. **ATen mm is not what vLLM runs.** On gfx906 the Linear dispatch goes to
   custom kernels first — see `vllm/model_executor/layers/utils.py`:
   - n=1, k≤8192 → `_llmm1_tiny_m` (LLMM1)
   - n=1, long-k (5120×{10240,17408}) → `_gfx906_gemv_long_k`
   - n=2–4 → `_gfx906_spec_gemv_m4` = `dense_gemv_m4_gfx906`
   - kill-switches: `VLLM_GFX906_DENSE_GEMV=0`, `VLLM_GFX906_SPEC_GEMM=0`
2. **ATen mm at N=248320 K=5120 fp16 is pathological on MI50** (~194 GB/s ≈
   19% of peak, even at full clock), while the fork's GEMV family runs ~822
   GB/s in-context (≈80% of peak). The "gap" was two different kernels, not
   two clock states.

**Rule:** before benchmarking any "stock" op, read the `utils.py` dispatch and
identify which kernel actually runs (profiler cpu-op stream or CUDA-event
timers on the live path). A standalone ATen number is evidence about ATen mm
only.

## Profiling methods — official vLLM vs our CUDA-event harness

Compared 2026-09-04 (boot U) against the upstream docs
(<https://docs.vllm.ai/en/latest/contributing/profiling/>). Three official
methods exist; status on THIS host:

1. **vLLM torch-profiler** — `--profiler-config '{"profiler":"torch",
   "torch_profiler_dir":...}'` + `/start_profile`/`/stop_profile` (or
   `vllm bench serve --profile`). Writes a real trace but with **zero GPU
   kernel events** (cpu_op only; verified 2026-09-04: 19 MB trace, 1.1M
   cpu_ops, 0 kernels — same in `--enforce-eager`) → same broken HSA tracing
   layer as rocprofv3. Still useful: the cpu-op stream names the exact
   custom-op dispatch per decode step (parse between
   `execute_context_*_generation_*` annotations) — this is how SYV-3 found
   that production runs the fork GEMV family, not ATen mm.
2. **Nsight Systems** (`nsys profile --trace-fork-before-exec=true
   --cuda-graph-trace=node --capture-range=cudaProfilerApi ...` +
   `vllm bench serve --profile`) — upstream's recommended low-overhead path
   for per-kernel timing under CUDA graphs. **NOT INSTALLED on this host**
   (no nsys binary). Untested; if the HSA tracing layer is broken for
   torch-profiler/rocprofv3 it may be broken here too, but nsys is a
   different capture stack — worth trying before assuming dead. Requires a
   user install (~100 MB, NVIDIA package on ROCm).
3. **cProfile helpers** (`vllm.utils.profiling.cprofile`) — CPU-side Python
   only; irrelevant to GPU kernel questions.

**Verdict (2026-09-04): the self-built CUDA-event forward-hook harness
STAYS.** It is the only method on this host that yields per-module *GPU*
time attribution (validated 0.4% vs wall clock, `phase_profile_results.md`).
Official methods cover a different question (which kernel family dispatches)
and are complementary, not replacements: torch-profiler cpu-op stream =
kernel identity; CUDA-event harness = kernel/module timing. Remove the
harness only if nsys is installed AND verified to produce GPU kernel events
on this ROCm build — re-test then and update this section.
