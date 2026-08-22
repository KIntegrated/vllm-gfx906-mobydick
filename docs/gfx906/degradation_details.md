# Degradation & wedge details — 2× MI50, ROCm 7.14 + amdgpu DKMS 6.19.14

Copyright Kevin Read <me@kevin-read.com>

Companion to `degradation.md` (the timestamped table). This file holds
the mechanism notes, the kernel evidence, and the open questions. Event
rows live in the table.

## The three failure modes (kernel signatures, from /var/log/kern.log)

1. **Half-wedge (HW, recoverable).** A compute/context queue is left
   mid-operation (usually a worker dying mid-P2P/memcpy or a harness
   teardown racing the engine). Kernel sequence:
   ```
   qcm fence wait loop timeout expired
   The cp might be in an unrecoverable state due to an unsuccessful queues preemption
   Failed to evict process queues
   GPU reset begin!. Source:  4
   BACO reset
   GPU reset succeeded, trying to resume
   [drm] VRAM is lost due to GPU reset!
   ```
   The GPU comes back, but **every context on it is dead** ("VRAM is
   lost") — any server that was up must be torn down and rebooted.
   Observed 49× between 2026-08-21 14:21 and 2026-08-22 14:06, almost
   all at **server boot-failure / teardown boundaries** (weight-load
   SIGABRTs, `hipErrorLaunchFailure` bursts, half-dead shm-broadcast
   boots). This is the kernel-level view of the "SIGKILL/abort leaves
   the driver mid-P2P-op" note in the workspace AGENTS.md.

2. **Full wedge (FW, GPU dead until power state cleared).** The reset
   itself fails:
   ```
   GPU reset begin!. Source:  4
   VM_L2_PROTECTION_FAULT_STATUS:0x0000473C ... (MP0/UTCL2 walker error)
   PSP load sys drv failed!
   PSP resume failed
   resume of IP block <psp> failed -62
   GPU reset end with ret = -62
   ```
   After this `rocm-smi` shows the GPU with temp/clock/VRAM `N/A` and a
   zombie VRAM percentage; new contexts fail. **Only a host reboot (or
   root BACO/power-cycle) recovers it.** Observed 2026-08-21 03:17,
   11:26, 2026-08-22 14:06 (all GPU0, 0000:0b:00.0).

3. **Host-state degradation (DEG) — the insidious one.** Everything
   boots, everything "works", but **sync-cadence-heavy inference
   collapses ~3× while GPU-bound work stays at full speed**. It
   survives server restarts (four fresh boots, same slowness) and is
   cleared only by a host reboot.

## The 2026-08-22 DEG event in detail (the one we have full data for)

**What it looked like** (this is how to recognize it):

- mtp2 (spec-decode) TP=2 serving: steady **24.9 t/s = 120.5 ms/step**
  (acceptance 3.00, healthy) — vs **40.1 ms/step (74.9 t/s)** on the
  identical binary/config/harness after the 12:39 reboot. 3× step-time
  inflation.
- plain greedy TP=2 serving in the SAME degraded boot: **40.86 t/s —
  completely normal.** TP=1 plain (F7 A/B, 06:09–06:33): normal.
- torch-profiler chrome trace of a degraded step: GPU kernel work
  normal and dense (~45–50 ms/step worth, gptq-M3 17.6 ms, GPU 96 %
  busy in-window); the worker spends **~57 ms/step blocked in
  `hipEventSynchronize`** waiting for the next step's inputs; CPU-side
  `execute_model` spans are normal. I.e. the *CPU↔GPU event/sync
  cadence* is inflated, not the kernels. Spec decode is the canary
  because it does ~10+ small syncs per step (draft passes, propose,
  reject, per-token output) — plain decode does ~1 big sync over a
  24 ms GPU step and absorbs the penalty.
- Engine "SpecDecoding metrics" windows agree with the client (so it
  is not a client artifact).

**Boot context:** degradation was first observed at 06:46 in a boot
that had accumulated **≥14 half-wedge resets over ~17 h** (14:21 →
02:42) plus roughly ten 20-GB weight-load cycles. No mtp2 serving ran
in that boot before the resets, so the onset threshold is unmeasured.

**What rules it out:** not the N4 gather (PERSIST=0 equally slow), not
capture coverage (identical topology), not engine code (byte-identical
binaries before/after), not the client (usage-based recount after
reboot: 74.9 t/s same wall-time), not `--async-scheduling` (no effect
on the degraded host), not `--stream-interval` (no effect — it is a
no-op on the healthy host).

## Open questions (record answers here as evidence lands)

1. **Onset:** does degradation need N accumulated resets, long uptime,
   many big weight-load cycles (host page-cache/mem pressure), or a
   specific single event? Data so far: 2 resets post-reboot did NOT
   degrade (n3 served fast at 13:02); ≥14 resets + 14 h uptime DID.
   Need a canary probe scheduled after each HW burst to bisect (see
   "detection" below).
2. **TP=2-specific?** All clear-cut DEG observations are TP=2 serving.
   In-process TP=1 mtp2 in the same window was mixed (one slow
   81.6 ms/step reading, later same-boot readings 54–71 ms/step) —
   ambiguous. TP=2 stresses the P2P/IPC paths the resets hit, so it is
   plausible, not proven.
3. **Load-time only?** Resets TRIGGER at load/teardown boundaries, but
   the DEG manifests during steady inference (spec decode cadence).
   Two distinct things: trigger (load/teardown) vs symptom (inference
   sync latency).

## Detection (cheap canary — run this before trusting any spec numbers)

~60 s, TP=1, GPU0, no server needed — after any suspected wedge burst:

```bash
cd /local/git/vllm-gfx906-mobydick && source ~/env-rocm-7.14-gfx906.sh
HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 BENCH_MODEL=/local/cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea \
.venv/bin/python - <<'EOF'
import os, time, torch
from vllm import LLM, SamplingParams
llm = LLM(model=os.environ["BENCH_MODEL"], max_model_len=2816, max_num_seqs=4,
          gpu_memory_utilization=0.95,
          speculative_config={"method": "mtp", "num_speculative_tokens": 2},
          seed=0, compilation_config={"cudagraph_capture_sizes": [1, 2, 3, 4]})
p = "The quick brown fox jumps over the lazy dog. " * 40
llm.generate([p], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
t0 = time.perf_counter()
o = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=False)
dt = time.perf_counter() - t0
n = len(o[0].outputs[0].token_ids)
# healthy host: ~55-63 ms/step at acceptance ~2.9  =>  ~40-47 t/s
print(f"CANARY: {n} tok / {dt:.1f}s = {n/dt:.1f} t/s "
      f"(healthy ~40-47; degraded <25 => REBOOT before benching)")
EOF
```

A canary run in the degraded state would have read <25 t/s; on the
healthy host it reads ~40+.

## Boot/recovery procedure

- FW (PSP −62): host reboot required (BACO reset needs root, which the
  bench user lacks). After reboot, verify `rocm-smi` shows both GPUs
  with real temps/clocks before starting anything.
- HW: GPU recovers but all contexts are dead — SIGTERM any servers,
  verify VRAM 0 %, then relaunch clean. If boots keep failing in a
  burst (13:55–14:06 pattern), a full wedge is likely imminent; stop
  retrying and reboot.
- DEG: reboot. Do NOT record spec-decode numbers from a boot whose
  canary reads slow — they are host artifacts (this cost us a day of
  misattribution on 2026-08-22: the "mtp2 TP=2 regression"
  investigation, see `fa-masked-mtp-regression-glm5.md`).
