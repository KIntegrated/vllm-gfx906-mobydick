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

## 2026-08-22 evening: TP=2 35B-MoE amdsmi crash (C2-V t2n1_off) — no kernel reset

First TP=2 run of the 35B MoE on this box (offline `LLM()`,
`HIP_VISIBLE_DEVICES=0,1`). ~20:05:50Z the rank-1 worker died in
`RocmPlatform.get_device_name` (the torch-compile-cache-dir query,
`vllm/utils/platform_utils.py:72`) during `profile_run`, while the
compiled region was executing `torch.ops.vllm.moe_forward_shared`:
the final exception was `AMDSMI_STATUS_NOT_INIT` from
`amdsmi_shut_down()` — i.e. the `with_amdsmi_context` wrapper's
`finally` masked the primary error. Rank 0 then hung on the shm
broadcast (GPU0 pinned 100 %) until SIGTERM at ~20:11Z.

- **No kernel amdgpu events** in the 19:55–20:15 window
  (`journalctl -k` / kern.log) — not a wedge in the HW/FW sense;
  pure software crash.
- **Transient VRAM observation**: 20:11–20:12Z `rocm-smi` showed
  44 % VRAM on GPU1 with **no owning KFD process** (`rocm-smi
  --showpids`). Attributed in the end to the next config's
  in-flight weight-load allocation (the driver's VRAM-release check
  had raced a 0 % reading and launched t2n1_m1on while I was
  inspecting); both GPUs read 0 % after killing everything. Flagging
  it here because it matches the zombie-VRAM symptom pattern and
  cost a false-alarm.
- **amdsmi is broken on this boot across ALL runs** (TP=1 included):
  every run logs `Failed to get total memory via amdsmi, falling
  back to torch.cuda` (rocm.py:913, protected path). Only
  `get_device_name` is unprotected (no caller try/except), which is
  why it was fatal in the rank-1 worker and harmless in TP=1.
- **Workaround** (C2-V only): `sitecustomize.py` on PYTHONPATH
  (`/tmp/c2v/shim/`) swallows `amdsmi_init/shut_down` failures and
  gives `get_device_name` the same `AMD_<arch>` fallback the code
  already has for the 0-handles case. TP=2 arm retried with the shim
  (stage0b). A permanent fix belongs upstream in
  `vllm/platforms/rocm.py` (the wrapper's `finally` must not mask).
- Canary this boot: 38.8 t/s (band ~40–47) — see
  `DEVLOG-moe-c2v.md`; the official 35B harness run is the
  tie-breaker for host health.

## 2026-08-23 05:18Z: GPU0 half-wedge — comp_1 fence timeout, driver reset

First HW reset this boot. Trigger: the 27B mtp2 canary (W2
spec-decode session) — SIGABRT rc=134 (core dumped) during/after the
measured generate. Kernel log: `Fence fallback timer expired on ring
comp_1.0.0` → `GPU reset(1) succeeded!` → `[drm] device wedged, but
recovered through reset` (05:18:28–29Z, 0000:0b:00.0 = GPU0).
rocm-smi back to 0/0 use after.

Pre-wedge symptoms (unexplained, possibly early degradation, possibly
unrelated): amdsmi broken since boot; 35B-MoE smoke A/B at 04:5x–
05:1xZ showed (a) baseline greedy output FP NON-REPRODUCIBLE across
two identical runs (`270672f1…` vs `147420f5…`) and (b) mtp2 decode
~39.5 t/s vs baseline ~81 (0.49×) with a mid-run Triton JIT spike
(`eagle_prepare_inputs_padded_kernel` compiled during the measured
generate). Both numbers are suspect; the FP non-reproducibility in a
temp=0 baseline is the stronger anomaly. Canary re-run post-reset is
the arbiter (if it reads <25 t/s or 100 % acceptance — reported as a
degradation symptom — the boot is degraded: REBOOT).

Open: does the 100 %-acceptance symptom (reported by Kevin 2026-08-23,
unconfirmed) accompany DEG? The canary should print the spec stats to
check (in-process runs need `VLLM_LOG_STATS=1` for the
SpecDecoding-metrics line).

## 2026-08-23 05:47Z: second GPU0 half-wedge (W2 mtp2 eager boot) + FP resolution

Same signature: `Fence fallback timer expired on ring comp_1.0.0` →
`GPU reset(2) succeeded` (05:47:46Z), during the w2_mtp2_e arm's boot
(`hipErrorLaunchFailure` at SetDevice, core dumped). Four clean engine
cycles (w2_base_g, w2_base_g2, w2_mtp2_g, w2_base_e) ran in the
05:18–05:47 window between resets.

**FP-mystery resolution:** post-reset baseline re-runs (w2_base_g /
w2_base_g2) showed the SAME temp=0 greedy non-reproducibility
(per-prompt FPs differ across reps; partial cross-process overlap:
8c4c58ea… / 3b96c1fe… appear in both runs). The pre-wedge FP drift
was therefore NOT a host artifact — the 35B MoE baseline is
non-deterministic in this build (hypothesis: fp16-atomic K-split
epilogue in the M=1 MoE q_gemm → last-bit logit noise → argmax flips
at near-ties). Consequence: token-identity gates are unusable for the
35B; the W2 A/B stands on perf + acceptance + output sanity. (The
27B dense baseline was deterministic in its spec A/B — consistent
with the dense M=1 path being the non-atomic dense_gemv.)

## 2026-08-23 06:08Z: GPU0 FULL WEDGE — PSP -62, host reboot required

The post-reset#2 canary hit the third comp_1 fence timeout this boot
(50 min apart: 05:18, 05:47, 06:08). This one did not recover:
`MAPPING_ERROR: 0x1` → `PSP load sys drv failed!` → `PSP resume
failed` → `resume of IP block <psp> failed -62` → `GPU reset end with
ret = -62` (06:08:32–37Z). Same signature as the 2026-08-22 14:06Z
full wedge (the one that needed the ~14:50 reboot). GPU0 is dead
until a host reboot; BACO reset needs root, which the bench user
lacks. All W2 GPU work stopped. Session data safe in /tmp (survives
reboot).
## 2026-08-23 OOM-teardown collateral (W4 A/B, not a wedge)

W4 serving A/B on the 06:33 boot. 27B (Qwen3.8) N=8 off arm OOM'd at
util 0.93 (356 MB inductor prefill buffer, free: 0 — Qwen3.8's FA KV
is 655 KB/token; 64 layers) and aborted; the *next* arm (on) died
`hipErrorLaunchFailure` rc=134 at boot (08:52). GPU0 read clean
afterwards (0 % VRAM, 0 % busy, no zombies); re-run of both arms at
util 0.90 / maxlen 1280 passed clean (off 98.2 / on 104.2 t/s, no
launch failures, ksplit=5 atomicAdd path graph-safe). Verdict:
one-off reset collateral of the aborted OOM arm, NOT a half-wedge
(nothing was wedged afterwards; no PSP failure; the next two engines
on the same GPU booted fine). Counts as a reset for the onset
bracket (see the 08-22 15:41 row pattern).

## 2026-08-23: Qwen3.8-27B 256k-prefill OOM cluster — verified diagnosis

Seven OOMs in one session, all on the first big prefill of a ~250k-
token request, never during steady serving. The failing allocation in
four of the arms is exactly **178,257,920 B at `torch.ops._C.gptq_gemm`
(`free: 0`)** — the exllama AWQ **per-call dequant scratch**
(`temp_dq = [N×32/bit, K/8]` fp16, weight-shape-sized, *not
token-scaled*): 8×17,408×640×2 for the MLP gate_up. The lm_head
(vocab 248,320, quantized) needs **2.37 GiB** of the same scratch on
every forward. Ruled out by direct test: `mamba_cache_mode: align`
(auto-enabled for Qwen3.5/3.6/3.8 with prefix caching; the scheduler
only *clips* chunks to the 784/800-token block, never bumps — the
`--no-enable-prefix-caching` arm OOMed identically), MTP, chunk size
(8192→1024), and util (0.93→0.82; the post-capture headroom
`profiled + graph_est − graph_actual` ≈ 1.9-2.5 GiB is util-
independent by construction). The 250k sequence is the common factor:
it drains the headroom via unprofiled request-time consumers (lazy Q8
side-buffer ~0.4 GiB, FA buffer growth, inductor dynamic shapes) plus
an unidentified ~1-2 GiB long-context transient, leaving no contiguous
block for the next scratch. W4 soak (same pool size, maxlen 1536,
30 reps) ran flat — not a leak.

**Full mechanism + verbatim evidence (log lines, C++ allocation site,
accounting arithmetic, arm matrix, fix directions): `oom-256k-prefill.md`.**
131k is the validated ceiling on this model (dense Qwen3.5-27B serves
256k fine).

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


## 2026-08-23 17:59–18:06Z: OOM-hunt canary burst — 3 weight-load aborts, 3 GPU0 resets

Successive canary attempts (the detection protocol probe, 27B mtp2
in-process) all SIGABRT rc=134 with `hipErrorLaunchFailure` at
safetensors shard 2/5→3, each leaving `Fence fallback timer expired
on ring comp_1.0.0` → `GPU reset(N) succeeded` in kern.log (18:00:02,
18:01:35, 18:06:15Z; boot's reset #1 was the 12:48 GPU1 event). VRAM
0 % and rocm-smi clean between attempts; attempt 3 ran with
AMD_SERIALIZE_KERNEL=3 with no change. 4 resets this boot (06:33
onward) = the documented burst pattern → stopped retrying
(full-wedge risk); host-health verdict deferred until a canary can
run on a fresh boot. Context: the 256k-OOM-hunt session
(oom-256k-prefill.md follow-up) needed the canary only as a
pre-bench health gate; the OOM mechanism itself is allocator-level
and unaffected by DEG.

## 2026-08-23 18:23Z: OOM-hunt 9B boot — GPU1 reset (5th+ this boot)

The instrumented Qwen3.5-9B TP=2 boot (gather-generation probe for the
256k OOM hunt) aborted at rank-1 SetDevice with
`hipErrorLaunchFailure`, triggering `GPU reset(2) succeeded` on GPU1
(0000:0e:00.0, 18:23:00Z). Combined with the 17:59–18:06 canary burst
(GPU0 ×3) the boot is at 5+ resets — confirmed wedge-burst state.
Host reboot requested (Kevin, 18:2xZ); all OOM-hunt GPU experiments
deferred to the fresh boot. Teardown clean: SIGTERM, VRAM 0 % both
GPUs, no zombie KFD procs.

Interim (static) conclusion of the OOM hunt — see
oom-256k-prefill.md follow-up notes: the ~1–2 GiB "unidentified
long-context transient" is predicted to be the FA gather-buffer
generations retired into the unbounded `_gather_retired` keep-alive
(5d960a503c) as chunked prefill grows Sk_pad every 32 tokens —
simulation of the 27B TP=2 shape (B=2, Hkv=2, D=256, chunk 1024)
predicts 89.4 GiB accumulated by 250k tokens; the 1.94 GiB run-4
headroom is exhausted at ~30k tokens (~2.7 min at prefill rate),
matching every observed arm. Live confirmation on the fresh boot.

## 2026-08-23 18:35Z→20:15Z: boot-failure wedges — on-die RAS latches + GTT refutation

The 18:30 warm reboot did NOT clear the wedge-burst state: GPU0 reset
18:35:55 (canary, weight load) with the day's first **on-die RAS latch**
(`ERREVENT_ATHUB_INTERRUPT` uncorrectable — the on-die host fabric hub);
GPU1 reset 18:39:40; **pcie_bif correctable latch on BOTH cards within
6 ms** (18:50:19.637/.643 — simultaneous dual-card timing; shared
host-side cause or delayed post-reset flush, ambiguous); the 19:14–19:20
full power cycle did NOT clear it either (first canary on the 19:20 boot
failed 19:23:30; failures continued 19:35–19:46).

Second session (20:00–20:30, pi) established the failure is
**intermittent, not deterministic**, and refuted the GTT-exhaustion
theory:

- Flap pattern on the 19:20 boot: 4 fails (19:23–19:46) → 5 passes
  (repro ×4 + full 27B-mtp2 canary 38.9 t/s, 20:03–20:08) → 2 fails
  (20:15:16/20:15:48, live-caught) → 5+ passes (20:16–20:23). Bad
  windows ~60-90 s; good windows 30+ min to hours. Same-day totals:
  **19 BACO resets, GPU0 ×16 / GPU1 ×3** (GPU0 primary).
- The 20:15:16 wedge was preceded 41 s earlier by a GPU0 `pcie_bif`
  correctable RAS latch (20:14:35) — the second such latch on GPU0
  (first 18:50:19). All on-die error events of the day (mmhub no-retry
  page fault 06:08:32, ATHUB uncorrectable 18:35:55, pcie_bif
  18:50:19 + 20:14:35) are on GPU0's host-fabric/PCIe-interface blocks.
  PCIe AER device counters: all zero, link x16/16 GT/s (the faults are
  on-die, not link-level).
- **GTT refutation (the decisive measurement):** `mem_info_gtt_total` =
  12,553,486,336 B (11.68 GiB; kernel: "11971M of GTT memory ready");
  `mem_info_gtt_used` sampled at 150 ms during the full 19.57 GiB /
  2396-tensor distinct-mmap weight load peaked at **20 MiB (0.16 %)**;
  idle baseline 14 MiB. A 19.57 GiB load has also PASSED many times
  today — impossible under a strict 11.7 GiB capacity mechanism. The
  "tensor #801 / ~8 GiB" failure point is a time-to-failure artifact
  (cold load reaches tensor 801 at ~40 s; runs die when they enter a bad
  window), not a capacity crossing. fwupd installed nothing today;
  nothing ran on the GPUs 13:06–17:59 (the 17:59 failure followed a 5 h
  idle).

Classification: **HW-class, GPU die/fabric flap — GPU0 primary** (NOT
GTT-pressure origin). Leading hypotheses: degrading GPU0
(ATHUB/pcie_bif/mmhub) and/or a shared host-side contributor (both GPUs
sit behind parallel two-bridge chains off adjacent root ports 03.1/03.2;
the 18:50 simultaneous dual-card latch). Discriminating experiment:
card swap between the symmetric slots. Full experiment record +
artifacts (`/local/tmp/boot_fail/`): `DEVLOG-boot-failure.md` §7.

## 2026-08-23 21:46Z — fa-gather-lifecycle arm A: isolated GPU1 wedge at worker init

First 256k needle-harness attempt (TP=2, Qwen3.8-27B, `GFX906_FA_GATHER_EXACT=1`
arm — the pre-fix-policy OOM repro) died at worker init, ~1 min in:

```
21:46:53 amdgpu 0000:0e:00.0: GPU reset begin!. Source: 4
21:46:53 amdgpu 0000:0e:00.0: BACO reset
21:46:55 VRAM is lost due to GPU reset!
21:46:56 GPU reset(1) succeeded!
21:46:56 [drm] device wedged, but recovered through reset
```

Worker log: `c10::AcceleratorError` / `hipErrorLaunchFailure` at
`SetDevice` (both ranks), `WorkerProc initialization failed`.

- 85 min of quiet between wedges (20:21:33 → 21:46:53) with passing
  canaries at 20:40/20:41 (38.2/38.6 t/s) — an **isolated wedge inside a
  good window**, the pattern established in DEVLOG-boot-failure.md §7.1.2.
- This one hit **GPU1** (0e:00.0 — 4th reset on GPU1 today vs 17 on GPU0),
  so the flap is not strictly GPU0-primary at the per-event level even
  though GPU0 dominates the count.
- Clean recovery (no PSP ret −62, no zombie VRAM, rocm-smi 0 % both cards);
  harness process exited on its own. Arm A retried after this entry.

## 2026-08-23 21:46–22:42Z — fa-gather-lifecycle session: GPU1 flap → dual-card common-cause reset

Session context: arm A (pre-fix policy, `GFX906_FA_GATHER_EXACT=1`) and arm
B (the fix) of the 256k needle harness. Wedges at worker init (SetDevice /
first kernel dispatch) in the ~6-min boot window:

- 21:46:53 GPU1 (0e) — arm A attempt 1
- 22:02:01 GPU1 — arm A attempt 2 (comp_1 fence fallback)
- 22:25:12 GPU1 — arm A attempt 4 (attempt 3, launched 22:05, ran clean
  22:07–22:16 through full boot + graph capture — then died on a harness
  assert bug, not a wedge)
- **22:42:12 BOTH cards (0b + 0e) in the same millisecond** — arm B attempt 1

The dual-card same-millisecond reset matches the 18:50:19.637/.643
dual-card pcie_bif latch: a shared host-side contributor (PCIe fabric /
root-complex) is the leading common-cause candidate; card-local
degradation alone would not reset both in one ms.

**Arm A SUCCEEDED on attempt 5 (22:29–22:38, a wedge-free window):**
the pre-fix OOM reproduced byte-exact (178,257,920 B, free: 0, gptq_gemm,
3.3 min into prefill) with OOMHUNT pinning the unbounded `_gather_retired`
dict (152 generations, 7.79 GB retired at the 60k-token OOM point).
Arm B pending a clean boot window.

Host state: 12 resets this boot (boot C, 19:20). Cadence accelerating
(15 min → 23 min → 17 min between single-card wedges, then dual-card).
If arm B cannot get a clean window, reboot is the next step (needs root).

### 22:48–22:58 addendum — stop-retrying decision

- 22:48:02 GPU0 (0b) — arm B attempt 1 (launched 22:47:09)
- 22:57:38 window_watch confirmed a 5.5-min clean probe window (10×30s,
  both cards) and auto-launched arm B; 22:58:30 GPU1 (0e) wedged at
  SetDevice 52 s later.

Cadence over the session: 21:46 → 22:02 → 22:25 → 22:42 (dual) →
22:48 → 22:58 — 12 resets this boot. Good windows have shrunk below
the ~6-min boot time, so every launch attempt now lands in (or creates)
a bad window. **Retrying stopped.** This matches the AGENTS.md
degraded-state signature (many half-wedges in one boot → only a reboot
clears it). After a reboot: run `window_watch.sh` (it auto-launches
arm B on a confirmed window) or launch arm B directly; arm A evidence
is already on record (`/local/tmp/fa_fix/arm_A.log`, `oomhunt_A.log`).

## 2026-08-24 (boot D) — first reset: 07:46:46 GPU0, 256k server launch

Boot D (05:20:44) held **zero resets** through a ~60-min heavy window
(arm A2 byte-exact OOM repro 05:28–05:37, arm B 250k PASS 05:44–06:19,
MoE decode A/B through ~06:40). First GPU use after that window: the
Qwen3.8-27B 256k TP=2 + MTP *serving* server (27B AWQ, util 0.82,
maxlen 262144, capture [1,2,3,4]) launched 07:45:44. ~60 s in, at
worker init: `hipErrorLaunchFailure` at SetDevice → comp_1 fence
fallback timeout → `GPU reset(1) succeeded` on `0000:0b:00.0` (GPU0)
at 07:46:46 — "device wedged, but recovered through reset".

**First reset since boot D** — boot-D flap onset at ~2.5 h, on GPU0
(the historical primary). Post-reset: both cards pass the 200-round
probe, VRAM 0 %, no zombie procs. Launch retried — per the boot-C
pattern, an isolated wedge inside a good window is not the degradation
signature; the burst (≥ several resets close together) is. If the retry
also wedges, treat as a burst and stop retrying (reboot remedy).

Note: the launch initially failed twice earlier the same hour for a
non-GPU reason — `python -m vllm.entrypoints.openai.api_server` does
not map the positional `model_tag` to `args.model` (only the `vllm
serve` CLI does), so the engine tried to resolve its default model id
offline. `vllm serve` is the correct entry point on this tree.
