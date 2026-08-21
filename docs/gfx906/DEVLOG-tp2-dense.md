# TP=2 dense 27B serving on 2× MI50 — one-line claim

Branch: `gfx906/tp2-dense-serving` (off `gfx906/main` @ `b1f164a46c`) · started 2026-08-20
Model: `cyankiwi/Qwen3.8-27B-AWQ-INT4` (snapshot `63768c10`, local HF cache)
Platform delta vs. old records: both GPUs now gfx906 MI50 32GB, PCIe P2P
enabled bidirectionally (`hipDeviceCanAccessPeer` = 1 both ways, 2 hops,
weight 40). Old "chipset-attached card poisons PCIe" blocker no longer
reproduces. RCCL UnitTests: 62/120 run (killed early for time), 0 failures,
incl. MP 2-rank cases.

## Session log

### S0 — baseline bring-up (2026-08-20, night session)

**VERDICT:** OPEN · **GATE:** `_bench_gfx906.py`-equivalent serving A/B at pp=2048/8192, tg=128/256, ±MTP; target ≥1.5× TP=1 dense decode (~25.3 t/s record → need ≥38 t/s)

Server (first boot, defaults): `-tp 2 --gpu-memory-utilization 0.93
--max-num-seqs 4`, `HIP_VISIBLE_DEVICES=0,1`, NCCL_DEBUG=INFO.
Log: `/tmp/vllm-tp2-27b.log`.

Open issues found at first boot:

1. **RCCL `graphUsageMode=0` vs CUDA-graph capture** — warning fired 1244×
   during graph capture ("can lead to hangs"). Comm not created in graph
   usage mode while vLLM captures allreduce into graphs. Prime suspect for
   upstream TP=4 reports (graph capture OOM/hangs, decode collapse to 4-5
   t/s). Fix: RCCL comm config (graphUsageMode=1) / env knobs; verify no
   hangs + allreduce not bouncing through host.
2. **Context window**: default run targets modest KV; requirement is
   ≥120k max_seq. GDN mamba state (~72 MB/seq full, TP halves per-GPU KV
   share) + inductor buffers constrain max_num_seqs; plan
   `--max-model-len 131072`, trimmed cudagraph capture sizes
   (`--compilation-config` / capture list) or cudagraphs off for debugging.
3. **Allreduce transport unverified**: RCCL logged "Using network Socket";
   no P2P/via lines surfaced. Must confirm PCIe P2P (not host bounce) is
   used for TP allreduce before judging TP=2 perf.

### S1 — graph-capture decode collapse, isolated to RCCL-in-graph (2026-08-20)

First boot loaded clean (exllama gptq W4 path, GFX906_FA CUSTOM backend,
NC2 auto-downgrade to 2 for gqa_ratio 6 per `1a895e8a01`, GDN triton paths,
in-tree qwen_triton_warmup). Booted to 262144 max_model_len, 463k-token KV
pool, graph capture 4/4 shapes. But first real request: prefill fine,
then decode frozen — **6 tokens in ~10 min (~0.01 t/s)**; rocm-smi showed
alternating single-GPU 99% (serialized ranks). Same class as upstream
TP=4 reports (4-5 t/s, graph capture OOM).

**RCCL transport exonerated** (`/tmp/rccl-allreduce.log`, GPUs free):
P2P/direct channels both ways, 17-19 µs latency at 1-16 KB, 8.2 GB/s
algbw at 64 MB, 0 wrong values. ~1 ms/decode-step allreduce cost — not
the bottleneck.

Prime suspect: RCCL comm created with `graphUsageMode=0` while vLLM
captures allreduce into CUDA graphs (warning fired 1244x during capture,
"can lead to hangs"). Graph replay of RCCL ops outside graph usage mode.

Next: eager A/B (same server config + `--enforce-eager --max-model-len
131072`, log `/tmp/vllm-tp2-eager.log`). If eager decodes at sane t/s,
verdict on the graph path is confirmed and fix = graph-safe RCCL init
(comm config graphUsageMode=1 or NCCL env knobs) vs graph capture shapes.

## HYPOTHESIS (night session)

If TP=2 allreduce runs over PCIe P2P with graph-safe RCCL config, then
dense 27B decode at ≥120k ctx reaches ≥1.5× the TP=1 record (≥38 t/s);
if allreduce falls back to socket/host, decode will be <10 t/s and the
fix is RCCL comm/env config, not kernels.

### S2 — decode stall isolated to GPU-side RCCL P2P (2026-08-21)

**VERDICT:** OPEN (root cause narrowed to RCCL P2P transport under serving
load) · **GATE:** offline in-process repro (`VLLM_ENABLE_V1_MULTIPROCESSING=0`,
`/tmp`→`/local/tmp/tp2-debug/tp2_offline.py`), 128-tok greedy generate.

Timeline of the isolation (all logs preserved under
`/local/tmp/tp2-debug/`):

- Symptom: ~27 s per engine step (0.04 t/s decode), batch-size independent;
  init/memory-profiling/graph-capture phases equally slowed (~500 s init).
  Reproduces in eager, with/without custom AR, fork or spawn, and with
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (our FA backend exonerated).
- Workers complete each step CPU-side in <1 s; the 27 s is consumed at
  `async_copy_ready_event.synchronize()` in
  `AsyncGPUModelRunnerOutput.get_output()` (async scheduling on) — i.e.
  real GPU execution time of the step, surfacing at the first host sync.
- Exonerated: shm MessageQueue/SpinCondition transport (ENQ-RESP→DEQ-RESP
  ~1 ms), scheduler busy loop, CUDA graphs, custom AR, multiproc method,
  RCCL in isolation (`all_reduce_perf` clean: P2P/direct, 17 µs / 8.2 GB/s),
  torch.distributed AR loops (sync, pipelined, side-stream, copy-stream
  variants all 0.35–9 ms/step for 64 ARs).
- In-stall transport (NCCL_DEBUG_SUBSYS=INIT): RCCL 2.30.4 picks
  **P2P/IPC**, 2 channels, chunksize 128 KiB — same transport as the *fast*
  torch probe. So the trigger is the combination (P2P transport + the real
  model's kernel/stream pattern), not transport selection alone.
- `NCCL_PROTO=Simple`: no effect (27 s cadence unchanged).
- **`NCCL_P2P_DISABLE=1` (SHM/host fallback): stall gone.** Init 503→137 s,
  128-tok generate 18.55 s = **6.9 t/s** (correct text output). Far below
  the TP=1 record (~25.3 t/s) — SHM transport is the new bottleneck.
- Topology: both MI50s 2 switch-hops apart through the CPU root complex
  (00:03.1→0a→0b and 00:03.2→0d→0e), same IOMMU group, `iommu=pt`.
  Peer *copies* are healthy (hipMemcpyPeerAsync 9.6/14.2 GB/s both dirs).
- A raw HIP spin-kernel-on-flag repro (cross-GPU *and* single-GPU
  same-device variants) hangs beyond 30 s — harness suspect (needs a
  bounded-iteration rerun after reboot; do not trust that result yet).

## HYPOTHESIS

If RCCL's P2P/IPC flag-sync between the two CPU-root-complex-attached MI50s
is starved by concurrent compute kernels on the peer GPU, then serving
steps (128 interleaved small ARs + GEMM/FA/GDN work) deadlock each AR for
~27 s while pure-AR loops stay fast. `NCCL_P2P_DISABLE=1` already restores
function; the open question is making P2P usable (ACS bits on the GPU
upstream bridges 0a:00.0/0d:00.0 are the prime suspect — host is rebooting
to change them; see tp2-claude.md) or accepting SHM + optimizing elsewhere.

## Instrumentation added (env-gated, `VLLM_TP2_DEBUG=1`)

- `vllm/envs.py`: `VLLM_TP2_DEBUG`
- `multiproc_executor.py`: worker RPC start/duration, ENQ-RESP/DEQ-RESP
- `gpu_model_runner.py`: FORWARD/LOGITS/SAMPLE markers, copy-event
  RECORD/sync timing, use_async_scheduling log
- `vllm/v1/engine/core.py`: busy-loop iter timing, batch-queue
  future.result() timing
All pure logging; keep until the P2P question closes.

## Next after reboot (ACS changed)

1. Re-run offline repro (no env override) → still 27 s?
2. If yes: `NCCL_DEBUG_SUBSYS=INIT` (expect P2P/direct now), perf matrix.
3. If still broken: docker upstream A/B (mixa3607/vllm-gfx906:0.20.1-rocm-7.2.1,
   aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x) — TP=2 without our kernels
   discriminates our-tree vs platform.
4. GPU-health caveat: SIGKILLing stalled runs wedges the GPUs
   (hipErrorLaunchFailure, drm_open hang) — always SIGTERM first.

### S3 — cross-stack confirmation, ACS exonerated, PXB workaround (2026-08-21 pm)

**VERDICT:** OPEN (platform/driver-level P2P/IPC pathology; workaround confirmed)
**GATE:** offline repro, 128-tok greedy; docker A/B same model/flags.

- **ACS override kernel (6.8.12-acso, pcie_acs_override=downstream,multifunction):
  stall persists** (P2P/IPC selected, 26 s/step). ACS is NOT the mechanism.
  All P2P primitives healthy on this kernel too (peer copy 10/14.2 GB/s,
  cross-GPU flag poll 0 ms — note: earlier "flag poll hangs" was a harness
  bug, setter on legacy default stream serialized against the spinner).
- **Docker A/B** (`mixa3607/vllm-gfx906:0.20.1-rocm-7.2.1-aiinfos`): TP=2
  with P2P/IPC **hard-hangs the GPU during weight load** (amdgpu "GPU Hang",
  BACO reset); TP=1 in the same image loads/serves fine. Failure is
  version-independent (old RCCL hangs, RCCL 2.30.4 stalls) → driver-level.
- **Workaround (confirmed 3×): `NCCL_P2P_LEVEL=PXB`** → SHM/direct/direct
  transport: init 105–198 s, no stalls, 128-tok generate 6.6–12.5 t/s
  (run-to-run variance to investigate; transport pick may vary).
  `NCCL_P2P_DISABLE=1` = 6.9 t/s. Matrix so far:
  auto=27 s/step stall · Simple=stall · PHB=stall · PXB=OK · P2P_DISABLE=OK(slow).
- Wedge hazard: TP=2 P2P failures can escalate to amdgpu reset storms
  (BACO recovery works but colliding runs fail with hipErrorLaunchFailure).
  Sequence attempts carefully; SIGTERM before SIGKILL.

## HYPOTHESIS (S3)

If the stock Ubuntu amdgpu driver mishandles P2P/IPC doorbell traffic on
this dual-root-port topology, then the official AMD amdgpu driver build
will either fix or change the failure mode; if it is RCCL-side, driver
swap changes nothing and the fix lives in RCCL transport selection
(default away from P2P/IPC on gfx906 PCIe-only topologies — candidate
upstream patch or platform quirk in vllm ROCm platform code).

## Next

1. pp/tg ± MTP benchmark matrix on `NCCL_P2P_LEVEL=PXB` (131k ctx server).
2. Official amdgpu driver test (user-gated; needs DKMS vs acso kernel or
   stock kernel boot — ACS irrelevant now).
3. Candidate upstream fix: default NCCL_P2P_LEVEL for gfx906 PCIe-only
   topology in vllm ROCm platform init.
