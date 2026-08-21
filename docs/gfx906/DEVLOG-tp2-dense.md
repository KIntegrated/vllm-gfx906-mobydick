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

### S4 — official amdgpu driver 6.19.14: P2P/IPC stall FIXED; perf ceiling remains (2026-08-21)

**VERDICT:** SHIPPED (platform fix: official amdgpu driver) · **GATE:** offline
repro, default env (no NCCL overrides), P2P/IPC transport.

- Official AMD DKMS driver (6.19.14.31400100, on 6.8.12-acso) fixed the
  27 s/step P2P/IPC stall entirely: default env now selects P2P/IPC and
  runs clean — init 135-145 s (vs 505), both GPUs 100% concurrently
  (serialized 99/0 pattern gone), zero copy-event stalls.
- Root cause chain closed: stock Ubuntu amdgpu mishandled P2P/IPC on this
  dual-root-port CPU-rooted topology (soft 27 s stall w/ RCCL 2.30.4,
  hard GPU hang w/ ROCm 7.2.1 RCCL); official driver handles it.
- Residual flakiness: ~1/3 inits wedge GPU1 mid-load (BACO recovers,
  retry succeeds). Watch; may be warm-cache/profiling related.
- **Decode throughput with working P2P: ~7.1 t/s** (256-tok: 36.15 s) —
  same as SHM workaround (6.9), below PXB best (12.5). vs TP=1 record
  25.3 t/s. TP=2 for dense decode is comm-bound and NOT a win on this
  topology; 2× speedup target unreachable via TP.
- Where TP=2 still pays: 131k-context capacity (463k-token KV pool across
  two cards, max_model_len 262144 boots), at ~7-12 t/s decode.

## HYPOTHESIS (S4)

If the remaining ~7 t/s is per-step allreduce latency (128 ARs/step), then
merging ARs (fuse_allreduce_rms) or MTP (fewer steps per token) recovers
some t/s, but the PCIe topology caps TP=2 dense decode well below TP=1;
expected ceiling ~12-15 t/s. VERDICT pending matrix.

## Next (night-session deliverables, adjusted)

1. Benchmark matrix on the fixed platform: pp 2048/8192 × tg 128/256 ± MTP
   at default (P2P) — for the record and MTP ratio at TP=2.
2. Recommendation: dense-27B serving stays TP=1+DP; TP=2 reserved for
   ctx-capacity-constrained workloads.
3. Flake watch on GPU1 init wedges; escalate to AMD if persistent.

### S5 — final matrix: TP=2 beats TP=1 with graphs + MTP (2026-08-21 eve)

**VERDICT:** SHIPPED · **GATE:** serving matrix on TP=2 server, graph mode,
131072 ctx, official amdgpu driver, default NCCL (P2P/IPC), batch=1 greedy.
Harness: `/tmp`→`/local/tmp/tp2-debug/tp2_serve_bench2.py` (streaming,
per-phase rates); server flags: `-tp 2 --gpu-memory-utilization 0.93
--max-num-seqs 4 --max-model-len 131072
--compilation-config '{"cudagraph_capture_sizes":[1,2,3,4]}'`.

## Results (tg decode tok/s, streaming-measured)

| arm | pp2048 tg128 | pp2048 tg256 | pp8192 tg128 | pp8192 tg256 |
|---|---|---|---|---|
| baseline | ~31 | 38-39 | ~34 | 34-35 |
| mtp2 | **39.7** | 38.2 | 34.0 | 34.4 |

- Best: **39.7 t/s = 1.57× TP=1 record (25.3)**. Target ≥1.5× met.
- MTP: mean accepted length 2.49, draft acceptance ~74% at TP=2 — parity
  with TP=1 mtp2 acceptance. Gains concentrated at short prompts.
- TTFT ~3.1 s at ~1.5k-token prompts (pp_rate ~492 tok/s cold;
  higher when prefix cache warms).
- Pure-decode single-stream offline (no server): 40.7 t/s steady
  (3×256-tok, repeatably). First-gen warmup 26.6.
- e2e (incl. TTFT) rates 14-20 t/s at these shapes.

## Key facts learned this session (not obvious from the matrix)

1. **Eager-mode TP=2 numbers (~7 t/s) were an artifact** — graph capture
   is mandatory for TP=2 on this stack; the eager per-op launch overhead
   × ~96 AR/token dominated. Claude UPDATE-11 arithmetic vindicated.
2. **Debug logging costs real t/s**: instrumented run measured 29.8 t/s
   where clean code does 40.7. Never benchmark with VLLM_TP2_DEBUG on
   (reverted at `49c935332d`, kept in history for reuse).
3. **Trimmed graph capture**: `[1,2,3,4]` (max_num_seqs=4) captures in
   3 s/phase vs 2+ min, frees capture VRAM; KV pool 448-480k tokens,
   3.4-3.7× concurrency at 131k ctx.
4. Init flake (~1/3 GPU1 wedge during load, BACO recovers, retry works)
   persists on official driver — watch; not blocking.
5. Profiler left enabled in bench scripts skews numbers — the final
   matrix ran clean (no profiler, no debug env).

## Session verdict

TP=2 dense 27B serving is viable and record-setting on this machine when:
official amdgpu driver + graph capture + trimmed capture sizes. Dense
decode record moves 25.3 → 39.7 t/s (mtp2, pp2048/tg128 shape). Context
capacity doubles-plus (131k boots with 3.4x concurrency). Platform fix
(official driver 6.19.14 DKMS) is a hard prerequisite — stock Ubuntu
amdgpu both soft-stalls (RCCL 2.30.4) and hard-hangs (RCCL 7.2.1-era)
TP=2 on this dual-root-port topology.
