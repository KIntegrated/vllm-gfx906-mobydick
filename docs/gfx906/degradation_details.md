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

**Resolution:** serving metrics over the following ~1 h confirmed the
degraded signature at SHORT context (serving canary 20-25 t/s vs 56+ on
boot E; the 08-23 20:41/20:43 canaries on the 19:20 boot had passed at
38.2/38.6). The 16.4 t/s long-ctx agentic decode (≈61k live) was NOT a
degradation symptom — boot E re-measure: 16.6 t/s @64k; the residual
live O(Sk) FA gather+attention cost caps long-context decode on any
healthy boot (see 13:00 section). Reboot 11:37 (boot E) cleared it.

## 2026-08-24 13:00Z: boot E first wedge — GPU0 comp_1 fence at greedy-server weight load

Boot E ran ~75 min clean before this: the rc2-image 27B TP=2 MTP serve
(Kevin's; 5 shards in 55 s, fault-free), serving canary 56.2/56.7 t/s,
the 27B MTP context curve (59.2/44.9/25.2/16.6 t/s at 2k/8k/32k/64k
live ctx), and the 35B re-stamp session (28/28 + 43/43 suites; single
65.7/66.1; MTP 88.6 vs 76.7; N=8 192.9/194.0). First reset 13:00:53:
GPU0 (0000:0b:00.0) `qcm fence wait loop timeout expired` → BACO →
`Fence fallback timer expired on ring comp_1.0.0` → `GPU reset(1)
succeeded` mid weight-load (shard 2/5) of the boot's second launch
(greedy 27B TP=2 docker); rocr `HW Exception by GPU node-1 reason :GPU
Hang` ×4 → worker abort (SIGSEGV, exit 139). Post: rocm-smi OK,
VRAM 0/0, 31 °C. Isolated-wedge-in-good-window pattern (cf.
2026-08-23 20:21:33, 21:46:53).

Retry 13:05: clean — 5/5 shards in 58 s, init + capture OK, canary
34.8/39.8 t/s, full greedy context curve completed (40.8/38.1/30.5/24.1
t/s at 2k/8k/32k/64k). Isolated wedge confirmed; boot E window remains
good. (Note: an early monitor false-alarmed a "second wedge" — its
grep scanned the whole T13: hour and re-matched the 13:00 events; no
second wedge existed.)

## 2026-08-24 14:47–14:56Z (boot E): CPU stuck-threads on HSA P2P-IPC + Worker_TP0 death

New failure signature — **CPU-side**, not a GPU reset. Observed on both
boot-E serving instances (13:25 docker, 14:37 host-venv; identical
config: rc2 image/tree, 27B TP=2 MTP, maxlen 262k):

- ~15–20 min after start, each TP worker process had **two threads at
  ~100% CPU each** (4 cores total; container CPU ~410%). Serving numbers
  were unaffected (canaries 55–59 t/s, full curves clean) — so this is
  **not** the sync-cadence degradation; it is idle-core burn.
- By ~14:55 the threads were **frozen, not spinning**: rip constant on a
  *trivial* instruction (register `mov`/post-syscall `cmp`) across
  minutes and repeated gdb attaches; utime ticking at 100%, stime flat
  (pure user mode); `voluntary_ctxt_switches` ~0–1. A frozen rip on a
  register move is not software — it is a core endlessly replaying an
  instruction (microarch stuck state).
- Locations (file offsets, identical lib in both host and image,
  md5-verified): two threads in/near `rocr::core::Runtime::IPCClientImport`
  in `libhsa-runtime64.so.1.21.0` (the HSA P2P-IPC import — the GPU0↔GPU1
  path on this dual-root-port topology), one at glibc `__poll`
  post-syscall, one in a libhsa stack-save. So the stuck context is the
  **P2P IPC channel establishment**, the same hardware path as the
  wedges — first CPU-side manifestation of that failure family.
- The vLLM Python side was clean: py-spy showed MainThread correctly
  parked in `SpinCondition.wait` (zmq poll); all other Python threads
  idle. So the "200% CPU" is **not** the RPC reader's
  `sched_yield` busy-branch and not a vLLM busy-wait.
- Taskset test: pinning a frozen thread to another core kept the frozen
  rip → the stuck state travels with the thread, not a dead core.
  `kill -9 <tid>` did remove a frozen thread (kernel delivery works).
- 14:56:10: Worker_TP0 "died unexpectedly (exit code: None)" — no OOM,
  no MCE, no segfault/fence in kern.log; engine cascade-shut at
  14:56:16. The death landed ~5 min after my gdb attach/detach cycles to
  that worker (ptrace stop/resume of stuck threads) — correlation, not
  proven causation; the stuck threads had been frozen before any attach.
- Cleared by the 15:01 reboot (boot F: 0 resets/wedges since boot) —
  but **RECURRED on boot F**: fresh docker TP=2 MTP server (15:09, same
  image/config), canary healthy (33.5/55.1 t/s), and by ~15:25 the same
  signature again — 2 threads per worker at 99.9% (workers at 200%
  instantaneous), ALL FOUR frozen at the SAME location as one boot-E
  thread: glibc `__poll` post-`syscall` (libc+0x11b5fd, constant rip
  across samples), one-to-one pairing per worker (one "python3" + one
  "VLLM::Worker" thread), symmetric across ranks.
- **Verdict: deterministic host-level defect** — reproduces across
  reboots, docker/venv, and boots (E + F) within ~15-20 min of a fresh
  TP=2 start; the P2P-IPC handshake path locks the threads in a CPU
  instruction-replay stuck state. Not a vLLM bug; not boot-state
  degradation (reboot does not fix it). Impact so far: 4 idle cores; no
  serving degradation; boot E additionally lost Worker_TP0 (14:56:10).
- Next step: `NCCL_P2P_DISABLE=1` (SHM transport) A/B to confirm the
  mechanism and test as a workaround; escalation candidate for
  ROCm-HSA / amdgpu-DKMS (gfx906, 2× MI50 dual-root-port P2P).

Tracing notes (for the next time): worker procs set `dumpable=0`
(HSA) — `/proc/<tid>/syscall` and same-user py-spy/gdb are EPERM; use a
helper container `--pid=host --cap-add SYS_PTRACE --security-opt
seccomp=unconfined --security-opt apparmor=unconfined` (AppArmor
`docker-default` denies ptrace even with the cap). Single-shot gdb
`info threads`/`info registers` only — a gdb `while`-loop in batch mode
hung with the inferior stopped (killing gdb released it).

## 2026-08-25 — boot E/F CPU-spin: root cause found, fix built, GPU1 wedge burst blocked validation

Follow-up to the boot-E/F entry above. Full analysis and source citations
live in `tp_stuck_threads_analyze_claude.md` (repo root) — this section
is the terse pointer + the same-day GPU-wedge interaction.

**Root cause (source-confirmed, §8 of the analysis doc):** the 100%-CPU
frozen-looking threads are `rocr::core::Runtime::AsyncEventsLoop` and
`InterruptSignal::WaitRelaxed`, both stuck in `Signal::WaitMultiple`'s
`HSA_WAIT_STATE_ACTIVE` busy-poll branch (`signal.cpp:315-317`, never
falls through to the real kernel sleep). That branch is forced
permanently once any watched signal has `EopEvent() == NULL`
(`signal.cpp:213-220`), which happens forever after the **first** failed
`hsaKmtCreateEvent` ioctl call in the process's life —
`InterruptSignal::EventPool::alloc()` used to latch
`allEventsAllocated = true` on that first failure and never retry
(`interrupt_signal.cpp:50-63`, old code). Independently corroborated by
unrelated reporters (gfx1151, torch, ComfyUI) hitting the identical
`AsyncEventsLoop` stack in
[ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051) — not
gfx906/ACS-workaround-specific. (That issue thread also contains what
reads as a prompt-injection payload aimed at AI agents — a fake
"agent-reviewed" `LD_PRELOAD` shim dressed up with fabricated benchmark
tables; not used, flagged for the record only.)

**Fix built:** two source patches to a local TheRock checkout
(`/local/git/TheRock`, `rocm-systems/projects/rocr-runtime`), rebuilt as
a minimal-scope `ROCR-Runtime`-only build (system clang toolchain
override, trimmed `BUILD_TOPOLOGY.toml` deps, local `LibElf` CMake shim
— avoids a full amd-llvm/LLVM rebuild):
1. `os_linux.cpp` `IPCRecvHandle`: bounded EOF check instead of
   unbounded `while(!rcv) recvmsg(...)` retry (separate bug, startup-time
   IPC-handle-import race — real, but not the CPU-spin cause; see
   `tp_stuck_threads_analyze_claude.md` §0).
2. `interrupt_signal.cpp` `EventPool::alloc()`: retry
   `hsaKmtCreateEvent` on every call instead of latching a permanent
   give-up flag after the first failure — this is the actual CPU-spin
   fix.

Deployed via `LD_LIBRARY_PATH` (not touching `/opt/rocm`) ahead of the
installed 1.21.0.

**Validation blocked by an unrelated GPU1 wedge burst, same day:** two
back-to-back server launches with the patched lib both hit the
already-documented `comp_1.0.0` fence-timeout/BACO-reset signature on
GPU1 (0000:0e:00.0) during the **drafter (MTP) model** weight-load phase
— 07:45:35 and 07:48:15 (see `degradation.md` rows). Per house recipe
(2nd wedge = burst → stop retrying), did not attempt a 3rd launch, so
the server never reached steady-state serving on the patched build and
the `EventPool::alloc()` fix's effect on the live CPU-spin symptom is
**still unconfirmed** — only confirmed so far via static analysis +
symbol/offset verification against the rebuilt binary, not a live
re-trace. This wedge pattern is not obviously related to the library
patch (GPU fence/hardware failure, not HSA-runtime control flow) and the
identical `comp_1.0.0`/GPU1 signature predates any of this session's
patches (recurs across many boots — 2026-08-23 18:23, 2026-08-24
13:00:53, etc.) — but ruling the patch in/out with a clean stock-library
run for comparison is still open.

**Next steps:** (a) fresh boot, retry the patched-lib server once
cleanly to get the live re-trace (per-thread CPU deltas + gdb/nm offset
resolution against the rebuilt `libhsa-runtime64.so`, method in
`tp_stuck_threads_analyze_claude.md` §8.1) and confirm the spin is
actually gone; (b) if it recurs, dig into the true proximate cause of
the first `hsaKmtCreateEvent` failure (kernel-side `kfd_events.c` trace
— signal-page mapping vs. event-ID-space exhaustion) rather than only
the userspace symptom.

## 2026-08-25 (same day, later) — EventPool fix validated as ineffective; real root cause found in HIP (clr), not ROCR

The (b) above happened, and the answer is more interesting than a KFD
trace: **the `EventPool::alloc()` fix does not touch the actual
mechanism.** Full trace in `tp_stuck_threads_analyze_claude.md`
("Update, same day" section after §8) — summary here.

Live re-test on a fresh boot (reboot per house recipe after the prior
GPU1 wedge burst) confirmed the patched lib loads and runs correctly
(server reached steady-state serving), but the CPU-spin symptom was
**unchanged** — same ~236% CPU, same two hot functions
(`AsyncEventsLoop`, `WaitRelaxed`). A live `strace -f -e trace=ioctl` on
the hot TIDs over a 4-second window showed **zero**
`AMDKFD_IOC_WAIT_EVENTS` calls — confirmed pure userspace spin, never
calling into the kernel wait at all. This rules out the
`hsaKmtCreateEvent`-failure theory entirely.

Traced the actual mechanism into `clr` (HIP's implementation,
`rocm-systems/projects/clr`, separate from `rocr-runtime`) instead:
`WaitRelaxed`'s wait-state hint is a straight pass-through of HIP's
per-device `ActiveWait()` flag
(`clr/rocclr/device/device.hpp:2381-2383`), set by `hipSetDeviceFlags()`
(`clr/hipamd/src/hip_device_runtime.cpp:800-843`). **`hipDeviceScheduleAuto`
(the default when nothing overrides it) resolves to permanent active-wait
whenever `device_count < hardware_concurrency()`** — true here (2 GPUs,
16 threads) and true on essentially any multi-GPU server. Neither torch
nor vLLM calls `hipSetDeviceFlags()` anywhere, so this default just
applies silently. **Not a ROCm bug — documented, intentional low-latency
behavior**, which also explains why the identical `AsyncEventsLoop`
signature is reported across completely unrelated projects/GPU families
in ROCm/TheRock#7051: they're all just default HIP clients.

`ROC_ACTIVE_WAIT_TIMEOUT` (an env var string found via `strings` on
`libamdhip64.so`) does **not** override `ActiveWait()` — tested live,
confirmed present in the worker env, no effect on the hot-thread
signature.

**Fix candidate (untested live yet):** call
`hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` per device before the
hot loops start. Built a `.pth`-file injection
(`_hip_blocking_sync_test.pth` in the venv's site-packages, gated on
`VLLM_HIP_BLOCKING_SYNC_TEST=1`) — `.pth` chosen over `sitecustomize.py`
because the venv's `sitecustomize.py` gets shadowed by the system
Python's own copy (stdlib precedes venv site-packages in `sys.path`,
only one `sitecustomize` module loads). First two live-test attempts
were blocked: one by a library-path bug in the hook itself (bare
`libamdhip64.so` isn't resolvable via `ctypes.CDLL` at `.pth`-exec time;
fixed by using the absolute path), one by an unrelated GPU1 wedge burst
(2 resets this boot — 08:21:31 isolated + retried clean, then 09:10:50
during the still-broken hook's run — see `degradation.md`) that stopped
further retries per house recipe before a clean run with the corrected
hook completed. Hook is fixed and verified working standalone
(`hipSetDevice`/`hipSetDeviceFlags` both return 0 for both devices) —
ready for next boot.

Also per Kevin: if `hipDeviceScheduleBlockingSync` does eliminate the
spin, a follow-up idea (lower priority, not yet designed) is toggling
`ActiveWait` dynamically — spin (`hipDeviceScheduleSpin`) while actively
serving requests for lowest latency, blocking-sync while idle to save
the CPU core — rather than a static per-process choice. Would need
hooking vLLM's request-scheduler idle/busy transitions.

**CONFIRMED on next reboot, same day: fix works.** Fresh boot, `.pth`
hook fired in both TP worker processes (`hipSetDevice`/`hipSetDeviceFlags`
both `ret=0` for devices 0 and 1), server reached steady state cleanly
(no GPU wedge this run). Per-thread CPU delta: hottest thread per worker
dropped from ~330 ticks/3s (~110%) to 7 ticks/5s (~1.4%); `ps` per-worker
total dropped ~236% → ~88%. Functional check: `curl` chat completion
returned correct output in 230ms round-trip, no regression. `rocm-smi`/
`journalctl -k` clean after.

**Verdict: root cause is HIP's default active-wait scheduling
(`hipDeviceScheduleAuto` → `SetActiveWait(true)` whenever GPU count <
CPU thread count, `clr/hipamd/src/hip_device_runtime.cpp:823-843`), not
a ROCR/HSA bug.** Full trace + fix details in
`tp_stuck_threads_analyze_claude.md` (the section after §8).

**Follow-up (same day): the obvious next move — move the
`hipSetDeviceFlags` call into `vllm/platforms/rocm.py`'s `set_device()`
+ call it from `gpu_worker.py`'s `init_device()`/`load_model()` — was
tried and does NOT work, and cannot be made to work as an in-process
vLLM call at any point.** Root cause of *that* failure, traced into HIP
source: `VirtualGPU::HwQueueTracker::Create()`
(`clr/rocclr/device/rocm/rocvirtual.cpp:536-566`) reads `ActiveWait()`
**once, at queue-creation time** to decide whether each signal in that
queue's pool is created with a real interrupt event or with
`HSA_AMD_SIGNAL_AMD_GPU_ONLY` (permanently active-wait for that signal's
whole life). Flipping the device flag *after* the queue already exists
is a no-op for that queue. torch/vLLM's default HIP queue gets created
very early — well before `init_device()` runs, per `rocm.py`'s own
`_get_gcn_arch()` fallback comment ("Ultimate fallback: use torch.cuda
... will initialize CUDA") — so there is no reliably-early-enough
in-process call site. Confirmed live: `set_device()` called from
`init_device()` correctly set and read back the flag (`hipGetDeviceFlags`
→ `0x4`), yet the hot threads' `strace` still showed zero KFD wait
syscalls — the setting took for *future* queues only. (This attempt also
surfaced and required fixing a real, separate bug on the way: looping
`hipSetDevice()` over every visible device to flag each one leaves the
process's "current device" pointed at the last one in the loop, which
broke NCCL — `"this nccl communicator is created to work on cuda:0, but
the input tensor is on cuda:1"` — any future in-process attempt must
restore the caller's intended device afterward.) The `rocm.py`/
`gpu_worker.py` changes from this attempt were reverted; `git diff`
against upstream is clean.

**Actual shipped fix:** `docs/gfx906/gfx906-blocking-sync.pth`, tracked in
this repo, copied into the venv's `lib/python3.12/site-packages/`.
`.pth` files execute at interpreter startup, before torch/vLLM import
anything — the only point that's reliably before any HIP queue exists.
It loads `libamdhip64.so` via `ctypes` and calls `hipSetDeviceFlags`
for every visible device, gated on `VLLM_GFX906_HIP_BLOCKING_SYNC`
(default on), resolving the library via `VLLM_GFX906_HIP_LIB_PATH` (set
this explicitly — a bare SONAME search at `.pth`-execution time silently
fails when `LD_LIBRARY_PATH` hasn't been populated yet, which is the
normal case for a from-scratch launch script; the original `try/except:
pass` swallowed this without a trace on one early test run, which is why
that run looked "successful" in the log but wasn't). Live-validated on a
fresh boot with `rocm.py`/`gpu_worker.py` fully reverted to upstream:
hottest thread/worker 6-7 ticks/5s (~1.2-1.4%), correct chat completion,
no wedge. Install step: `docs/gfx906/running.md` §0.

## 2026-08-25 22:19Z (boot G, 19:58:22): first wedge — GPU0 qcm fence at Ornith triton-arm weight load

Boot G (19:58:22). Session: Ornith-1.5-35B-A3B-AWQ-INT4 onboarding on branch
`gfx906/moe-ct-asym-zp` (compressed-tensors asymmetric W4A16 MoE support).
Five clean eager runs preceded the wedge (smoke gfx906-arm 21:47, smoke
triton-arm 21:50, PPL gfx906 22:12, PPL triton 22:15, PPL gfx906-run2
~22:17 — all loaded + ran to completion, no resets in that window).

First graph-mode serving launch of the boot (`_bench_gfx906.py`,
`BENCH_EAGER=0 BENCH_MAX_SEQS=8`, `BENCH_MOE_BACKEND=triton` arm) died at
safetensors shard 3/5 with `c10::AcceleratorError: CUDA error: unspecified
launch failure` (SIGABRT, exit 134). Kernel log (22:19:02):

```
amdgpu 0000:0b:00.0: qcm fence wait loop timeout expired
amdgpu 0000:0b:00.0: The cp might be in an unrecoverable state due to an unsuccessful queues preemption
amdgpu 0000:0b:00.0: GPU reset begin!. Source:  4
amdgpu 0000:0b:00.0: BACO reset
amdgpu 0000:0b:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:0b:00.0: VRAM is lost due to GPU reset!
```

GPU0 (0b:00.0) wedged; GPU1 clean. Post-reset probe: rocm-smi healthy,
VRAM 0%/0%, 36 °C, clocks normal; no vllm procs left. Log:
`/tmp/ornith_serve_trt.log` (wiped on reboot — the load progress lines
above are transcribed from it: shards 1-2 at 15.9/13.4 s/it, abort during
shard 3). Note the crash is during plain safetensors weight load — no
custom gfx906 kernel in the triton arm's load path; consistent with the
recurring load-time fence-timeout signature (cf. 2026-08-24 13:00:53,
2026-08-25 08:21:31).

Verdict: isolated wedge (1st of boot G) → retried once per house recipe.

## 2026-08-26 00:20Z (boot G): second isolated GPU0 half-wedge, ~2 h after the 22:19 one

Qwen3.8-27B ngram-battery run 1 (nospec_r1, launched 00:19:49)
SIGABRTed at weight-load shard 1/5 (same `hipErrorLaunchFailure` in
`copy_()` → `SetDevice` signature). ~25 s after that process died,
kernel logged `qcm fence wait loop timeout expired` → `GPU reset
begin! Source: 4` → BACO → `Fence fallback timer expired on ring
comp_1.0.0` → `GPU reset(2) succeeded` on 0000:0b:00.0, "device
wedged, but recovered through reset". The *next* battery process
(ngram_r1, launched 00:20:00 — i.e. before the reset landed)
completed weight load, an 89 s torch.compile, and cudagraph capture
with no visible effect. The reset is best read as cleanup of queue
state the aborted nospec_r1 left mid-operation (failure mode 1:
"worker dying mid-memcpy"), not a fresh die event hitting ngram_r1.

Assessment: two HW events on boot G (22:19:02, 00:20:25), ~2 h
apart, both GPU0 (0000:0b:00.0), both self-recovered via BACO;
neither killed a live workload (the 22:19 one got its house-recipe
retry; the 00:20:25 one hit no live process). Not a burst per the
house recipe (bursts on prior boots were ≤~30 min apart: 08-25
07:45/07:48, 08-23 21:46/22:02/22:25). Battery continues; a 3rd
event — especially one that kills a live run, or two close together
— stops the session and reboots (root).

## 2026-08-26 ~00:58Z (boot G): same-minute dual-card weight-load wedges — BURST, session stopped

Two independent launches, both on boot G, both at weight-load, both
`hipErrorLaunchFailure`, both within the same minute:

- **GPU0, ~00:58** — Qwen3.8-27B `nggpu_r4` re-run (the ngram_gpu
  arm of the 27B spec-decode battery; launched 00:57:16 to replace a
  battery run that silently ran without the ngram_gpu spec config —
  see below) SIGABRTed at weight-load shard 3/5. Kernel-side
  recovery per the usual BACO path; rocm-smi clean after (0% VRAM,
  32 °C).
- **GPU1, ~00:58** — the GDN mixed-batch before-probe (9B, first GPU1
  use this boot) SIGABRTed at weight-load shard 1/3. Follow-up torch
  matmul on GPU1 clean; VRAM back to ~11 MB.

House-recipe reading: two wedges in the same minute (across both
cards) = burst. **All GPU work stopped.** Boot G has now had three HW
events (22:19:02, 00:20:25, 00:58) over ~6 h — the two 00:58 ones
happened while both GPUs were loading 20 GB-class weights
concurrently, which is the highest-HBM-bandwidth moment of any run
and the moment prior boots have shown the flaps. A reboot (root) is
required before further inference; the 00:58 window matches the
"degraded state" signature (everything boots, weight-load hangs).

Side finding (harness, not HW): the *original* battery `nggpu_r4`
(00:51, the run that produced 24.9 t/s / 256 tokens / 1 prompt) ran
on a working tree whose `_bench_gfx906.py` had been reverted —
`BENCH_NREQS`/`BENCH_CG_MAX`/`BENCH_SPEC_CONFIG` are read only by the
harness commit on `gfx906/ngram-cpu-d2h` (`18c235772d`), not by
`gfx906/main`. The battery script was launched from the ngram branch
but the 27B re-runs after the branch switch hit the main-branch
harness, so `nggpu_r4` silently ran as a plain 1-request nospec
number (24.9 t/s ≈ the 25.25 t/s 27B nospec record — consistent).
`nggpu_r1`, `ngram_r1`, `ngram_r4` (all before the switch) are valid.
Lesson: pin the harness to the branch under test; env vars that are
silently ignored by the older harness read as "no spec / 1 request".
## 2026-08-26 ~06:03Z (boot H): first weight-load hang on the fresh boot

Boot H = 2026-08-26 05:48 (post the 00:58 dual-card burst). mtp2 27B
canary at ~05:53: **38.9 t/s — healthy** (prior passing canaries
38.6–38.9). Seven minutes later, the W1 after-probe (9B, GPU0,
first non-canary load of the boot) hit the recurring
`hipErrorLaunchFailure` weight-load hang at shard 2/3
(`copy_()` → `SetDevice`, c10::AcceleratorError abort). GPU0 clean
after (0% VRAM, matmul OK). This is the 5th occurrence of this
exact signature across boots (22:19, 00:20, 00:58×2, 06:03) —
independent of the degradation model (canary passed minutes earlier;
boot is 15 min old). House recipe: log + one retry. (The retry
succeeded — all subsequent 9B/27B loads on boot H clean.)

### 2026-08-26 ~06:52 — 27B W1-before serving A/B, weight-load
hang #6

Same signature at shard 5/5 (`hipErrorLaunchFailure` in
`copy_`→SetDevice). ~49 min after the 06:03 event with a ~45-min
clean window between (10 successful loads, incl. the 27B W1
after-arm) — NOT within a 30-min burst; the house recipe's burst
rule was not triggered. Read: the chronic intermittent weight-load
hang (see boot-E "13:55–14:06 burst" history for the same
intermittent shape). GPU0 probe-clean after; one retry per house
recipe succeeded (the W1 before-arm 27B number: 55.60 t/s).

### 2026-08-26 20:05 — TP=2 promotion smoke half-wedge on GPU1

The second proper TP=2 smoke attempt for the main-branch promotion used
Qwen3.8-27B, `max_num_batched_tokens=4096`, `max_num_seqs=4`,
`gpu_memory_utilization=0.82`, and trimmed capture sizes `[1,2,3,4]`.
The first TP=2 attempt was invalid because it ran Python from stdin under
`spawn`; the corrected file-based retry reached model load and generated 64
tokens successfully, but exited with a multiprocessing teardown status of 1.

The next corrected retry began loading the five checkpoint shards and failed
on worker TP1 at `SetDevice`/`copy_()` at 20:05:23Z. Kernel evidence was:

```text
qcm fence wait loop timeout expired
The cp might be in an unrecoverable state due to an unsuccessful queues preemption
Failed to evict process queues
Failed to quiesce KFD
GPU reset begin!. Source: 4
BACO reset
GPU reset succeeded, trying to resume
VRAM is lost due to GPU reset!
Fence fallback timer expired on ring comp_1.0.0
GPU reset(1) succeeded
[drm] device wedged, but recovered through reset
```

This is a **HW half-wedge**, GPU1 (`0000:0e:00.0`), not a full wedge: both
cards returned to 0% VRAM and `rocm-smi` remained responsive afterward. The
TP2 smoke therefore has no clean exit gate; further dual-card inference is
stopped pending the normal reboot/recovery procedure. Persistent copies of
all session logs and the TP2 probe script are in
`/local/tmp/gfx906-promotion-2026-08-26/`.

## 2026-08-26 22:30–23:59Z (boot I, 20:56:19): Muse-Glimmer onboarding — 2 allocator OOMs + 2 collateral weight-load launch-failures

Boot I (20:56:19) started with a clean TP=2 promotion-validation window
(20:05–20:5x, see the section above for the 20:05 GPU1 half-wedge), then
the Muse-Glimmer-30B-AWQ-INT4 onboarding session (TP=1, GPU0, local venv,
branch `feat/muse-glimmer`):

**Timeline (all GPU0):**

1. **22:30Z — util-0.95 OOM abort.** First graph-mode bench attempt
   (`gpu_memory_utilization=0.95`, no explicit KV cap): warm-cache
   profiling peak undershot the runtime inductor prefill buffer (532 MiB,
   `aten::empty` in the piecewise inductor graph) → OOM on the first
   request. Process aborted.
2. **22:41:28Z — weight-load `hipErrorLaunchFailure` → GPU reset(1).**
   The next launch (util 0.93, still no cap) wedged at weight-load shard
   1/5→2 → `Fence fallback timer expired on ring comp_1.0.0` → BACO →
   `GPU reset(1) succeeded` on 0000:0b:00.0; VRAM 0% / rocm-smi clean
   after. ~11 min after the OOM abort → **OOM-teardown collateral**
   (cf. 2026-08-23 ~08:51 precedent), not an independent wedge. Retry
   per house recipe.
3. **22:55:42Z — warm-cache OOM abort again** (util 0.93): booted clean
   (load 58 s, capture 0.71 GiB, KV pool 5.17 GiB from profiling) then
   OOM'd on the first prefill (same 532 MiB buffer, free: 0 ×2) → HSA
   `HSA_STATUS_ERROR_OUT_OF_RESOURCES` abort in `_fwd_kernel`
   (rocdevice.cpp:4207), no kernel reset in journal. Root cause:
   profiling-based KV sizing vs the 532 MiB runtime buffer (the 532 MiB
   buffer is larger than Qwen3.8-27B's 356 MiB, so 0.93 — which works for
   27B — is too tight here). Fix: explicit `kv_cache_memory_bytes` cap +
   `BENCH_KV_MEM` harness hook. SIGTERM teardown, VRAM 0%.
4. **~23:38Z — weight-load `hipErrorLaunchFailure` (eager probe, 2 GiB
   KV cap).** SIGABRT at shard 3/5; journal clean (no BACO/reset this
   time), rocm-smi 0% after. ~60 s after the previous run's OOM-teardown
   exit (23:37) → same collateral pattern.
5. **~23:59Z — weight-load `hipErrorLaunchFailure` (graph run, 0.75 GiB
   KV).** SIGABRT at shard 3/5; journal clean, rocm-smi 0% after. Again
   immediately after an OOM-teardown exit (23:57) → collateral pattern.
6. **~00:0xZ (08-27) — retry clean.** Weights loaded 100%, capture OK;
   at this KV cap the first prefill still OOMed (free 128–132 MiB —
   allocator-level; the all-CUSTOM arm's per-layer buffers are ~2.6 GiB
   larger than the hybrid's — see `DEVLOG-muse-glimmer.md`, memory
   forensics). A `BENCH_BATCHED_TOKENS=1024` re-run then loaded clean
   and completed the full gate bench.

**Assessment:** two allocator-level OOM aborts (expected — the model's
first-request memory profile exceeded the profiling-based KV sizing) and
two weight-load launch-failures, both within a minute or two of an
OOM-teardown exit. No independent wedge signature (no unexplained BACO,
no full wedge, journal otherwise clean). Both cards clean at session
end; no vllm procs left. If weight-load failures start appearing WITHOUT
a preceding OOM abort, treat as independent wedges and apply the burst
rule.

Logs: `/local/tmp/muse/` (`bench_hybrid_graph*.log`,
`bench_allcustom*.log`, `smoke_*.log`, `probe_allcustom*.log`); dev log:
`DEVLOG-muse-glimmer.md`.

## 2026-08-27 08:2x–08:3xZ (boot I): weight-load `hipErrorLaunchFailure` under a concurrent 16-way build — retry clean

**Context.** Boot I is ~11 h 45 min old; the 08:1x–08:2x window-FA
follow-up session ran two full B=4 bench launches that loaded clean.
The 3rd launch (B=4 `GFX906_FA_KVSPLIT=2` arm) SIGABRT'd at weight-load
shard 4/5 with `c10::AcceleratorError: CUDA error: unspecified launch
failure` (HIP `hipErrorLaunchFailure`).

**Unusual confounder.** A 16-way ccache/clang rebuild of the gfx906 FA
extension (`setup.py build_ext --inplace`, Phase C kernel change) was
running concurrently — the first observed overlap of a heavy CPU build
with a weight load. No OOM preceded it (unlike both boot-I events from
the onboarding session), so per the onboarding-session rule this counts
as an independent launch-failure, not OOM collateral.

**Unknowns.** `dmesg`/`journalctl -k` unreadable (no root) — cannot tell
whether a BACO/kernel reset ran; rocm-smi showed both cards clean
immediately after (VRAM 0%/0%, 38/31 °C, normal SCLK/MCLK), and a torch
matmul probe was not needed because the retry loaded and served clean.

**Outcome.** Isolated (1st since the ~00:0x session) → one retry per
house recipe at 08:41Z: weights loaded clean, full 4-sample bench
completed (20.55 t/s). No further launch-failures; the two subsequent
pp4096 long-context bench launches also loaded clean.

**Open question for the table:** does sustained high CPU load (16-way
clang) during a 20 GB weight load raise the launch-failure rate? One
data point; if weight-load failures recur under concurrent builds,
serialize builds and loads.

## 2026-08-27 14:19–14:32Z (boot I): OOM teardown → 2 consecutive weight-load/init `hipErrorLaunchFailure` — burst, session stopped

**Context.** Review-round-2 session: the LEGACY=0 Q8-side-buffer smoke
(first one with an *uncapped* KV pool on this model) OOM'd at
14:19:25Z — `aten::empty` in `gptq_gemm` inside the AOT/inductor
runtime (the 104k-token pool + ~1.5 GiB Q8 side buffer + inductor
headroom exceeds 32 GiB; allocator-level, not HW). The same smoke with
a 0.375 GiB KV cap then ran to completion (garbage output = the
expected side-buffer desync finding).

**The wedge.** ~11 min after the OOM teardown, the standard
LEGACY=1 smoke SIGABRT'd at weight-load shard 2/5
(`c10::AcceleratorError: CUDA error: unspecified launch failure`,
`hipErrorLaunchFailure`, raised in SetDevice) — `/local/tmp/muse/
smoke_final.log`. rocm-smi remained fully responsive throughout
(both cards VRAM 0%, 33/31 °C, SCLK 938 MHz) — the silent-wedge
variant: the device answers management queries but rejects launches.
The one permitted retry (14:32Z) also SIGABRT'd, at engine init before
any weight load — `/local/tmp/muse/smoke_final2.log`.

**Classification.** The OOM-teardown collateral pattern (precedents:
2026-08-26 22:41, ~23:38, ~23:59; 2026-08-23 08:51) — both failures
follow the 14:19:25 teardown with no other GPU work between. But two
consecutive failures = **burst per house recipe**: GPU work stopped;
reboot (root) required before further inference.

**What completed before the wedge.** All of review round 2's
verification: the pre-fix repro of the P1 clip bug (old .so), the
post-fix verification (new .so), 45/45 suite, the LEGACY=0 garbage
smoke (desync blocker), the window-check rejection probes, and the
pp8192 clip A/B (later identified as a null test — both arms ran the
gather path). Nothing GPU-dependent is pending.

## 2026-08-27 17:40–17:52Z (boot J): TP=2 serving OOM (pool sizing) → force-kill → 2 consecutive `hipErrorLaunchFailure` relaunches — stopped, reboot required

**Timeline.**

1. **17:30Z** — Muse-Glimmer-30B **TP=2** ngram serving launched clean
   (first TP=2 launch this boot; official-driver stack). Weights 12.68
   GiB/GPU, KV pool **9.02 GiB/GPU** (1,358,787 tokens; util 0.82, no cap),
   graphs 1.28 GiB. Single-request checks + a 4-parallel batch (short
   prompts, all <100 tokens) + tool/reasoning parser checks all clean.
2. **17:40:33Z** — first request with a real 4096-token prefill (bench
   sanity, pp4097): OOM `aten::empty` in `gptq_gemm` (AOT runtime).
   Budget math from the engine log: steady state 14.48 (weights +
   non-torch) + 1.28 (graphs) + 9.02 (KV) = 24.78 GiB of 31.98 physical
   → <7.2 GiB headroom; the runtime bt4096 inductor prefill buffer
   exceeded it (profiled peak activation was only 2.73 GiB — the
   warm/cold gap is much larger than the documented 0.16 GiB 27B case
   at this model size/batch). Allocator-level, the same class as the
   TP=1 532 MiB case (boot I 22:30) — the pool was simply oversized.
   **The engine force-killed the surviving worker** (`force killing
   remaining process EngineCore`) — a SIGKILL mid TP=2 P2P op.
3. **17:45:49Z** — relaunch (with `--kv-cache-memory-bytes 6 GiB` cap):
   `hipErrorLaunchFailure` at worker init/SetDevice, both ranks.
4. **~17:49Z** — TP=1 mtp2-27B canary: **38.4 t/s, healthy** (no P2P).
5. **17:52:06Z** — retry: `hipErrorLaunchFailure` again, both ranks,
   22 s into weight load. Journal unreadable (no root) — BACO/reset
   status unknown. rocm-smi clean after (both 0%/0%, 32–33 °C, 938 MHz),
   no stale processes.

**Classification.** Two components, both matching documented patterns:

- The OOM itself is pure pool-sizing (SW): fixed by the explicit KV cap
  (`--kv-cache-memory-bytes 6442450944` → ~900k-token pool, ~10 GiB
  headroom/GPU). The uncapped 0.82 config is NOT usable for this
  model at bt4096 — the README row must carry the cap.
- The two consecutive `hipErrorLaunchFailure`s after a mid-P2P SIGKILL
  match the AGENTS.md TP=2 teardown note exactly ("SIGKILL leaves the
  driver mid-P2P-op and the next init wedges GPU1 … BACO reset + retry
  needs root"). The healthy TP=1 canary between the two failures argues
  against host degradation (which would slow TP=1 spec decode too);
  this is a P2P-path stall. Per house recipe, 2 consecutive launch
  failures = stop; a BACO reset (root) or a reboot is required before
  further TP=2 work on this boot.

**Pending on next clean boot (reboot, then canary first).**

- Muse TP=2 ngram serving validation + benchmark grid
  (`/local/tmp/muse/bench_serve_grid.py`: pp2048/8192/16384 ×
  tg256/512, B=1 streaming TTFT/decode split + one B=4 point at
  pp2048/tg256) with the capped pool; server args preserved in
  `/local/tmp/muse/muse_tp2_ngram3.log` (header).
- README model-row numbers for the serving config (decode t/s per
  context + prefill t/s) — see the DEVLOG-muse-glimmer round-3 notes.
- Optional: greedy (no-spec) baseline B=1 for the ngram-lift number
  (needs a spec-off server).

## 2026-08-27 18:23–18:29Z (boot K): chronic weight-load hang (GPU1 reset) → clean retry → silent process-group kill (operator-aborted launcher), not HW

**Timeline.**

1. **~18:19Z** — boot K (after the 17:52 stop). rocm-smi both cards clean
   (0%/0%, 32–33 °C, 938 MHz). mtp2-27B canary **38.8 t/s** (healthy;
   prior passing canaries 38.4–38.9).
2. **18:22:31Z** — Muse TP=2 serve launched with the 6 GiB KV cap
   (`--kv-cache-memory-bytes 6442450944`; args unchanged otherwise).
3. **18:23:30Z** — weight load hung at shard 2/5→3 on both ranks
   (`hipErrorLaunchFailure` in `copy_`→SetDevice) — the chronic
   weight-load-hang signature (6th occurrence across boots).
4. **18:23:47Z** — kernel: `Fence fallback timer expired on ring
   comp_1.0.0` → `GPU reset(1) succeeded` on **0000:0e:00.0 (GPU1)** →
   "device wedged, but recovered through reset". First HW event of
   boot K. rocm-smi clean after.
5. **~18:25:30Z** — retry per house recipe: weights 5/5 in 42 s (clean),
   **KV pool 904,164 tokens — exactly the 6 GiB cap** (3.45× the 256k
   max, vs 1,358,787 uncapped), graph capture 9 s / 0.89 GiB (vs 1.28
   uncapped — smaller pool, smaller piecewise workspace), "Application
   startup complete" 18:28:07Z.
6. **~18:29Z** — the entire process group (API + EngineCore + both
   workers) died with **no log output, no kernel events, no OOM-kill**
   in kern.log, and full VRAM release (both cards 0%/0%, 34 °C).

**Classification.**

- Event 3/4 = the chronic intermittent weight-load hang, isolated on a
  fresh boot with a healthy canary and no OOM/SIGKILL precursor —
  same class as 08-26 06:03/06:52 (boot H) and 08-27 08:2x (boot I);
  all of those cleared on one retry, and so did this one.
- Event 6 is **not a GPU event**. The launcher was a single shell call
  (`nohup vllm serve … & sleep 240; grep …`); when that call was
  interrupted by the operator ("Loaded" interjection), the tool killed
  the call's process group — `nohup` only ignores SIGHUP, so the
  backgrounded server died with it. Distinguishing evidence: kern.log
  is empty for 18:28–18:31 (no fence timeout, no reset, no amdgpu
  error), no "Killed process" OOM lines, and the GPUs released all
  VRAM cleanly (a mid-op GPU kill would leave zombie VRAM / N/A
  rocm-smi). Boot J's launches used the identical pattern but *completed*
  their calls, which is why their servers survived.
- **Process-management lesson (recurring trap):** launch long-lived
  servers detached from the interactive call (`setsid nohup … &` from
  a stable terminal, or a plain `nohup … &` whose call is never
  interrupted) and check readiness in *separate* short calls.

**Validation state of the 6 GiB-cap config (as of this entry).**
Load, pool sizing (exactly the requested 904,164 tokens), and graph
capture all clean on the retry. The one remaining validation is the
**first real 4096-token prefill** — the exact site of the boot-J OOM.
Pending with the operator-relaunched server: sanity request (pp4096) →
the prefill grid (`docs/gfx906/_bench_serve_grid_gfx906.py`) → the
README serving row.

## 2026-08-27 20:40–20:42Z (boot K): in-process OOM-attribution probe wedges GPU0 mid weight-load; driver self-recovers

**What ran.** The OOM-attribution probe for the boot-J/K first-prefill
OOM (`/local/tmp/muse/probe_oom_attribution.py`, custom arm = our
GFX906_FA backend, TP=1, in-process, 0.5 GiB explicit KV cap, PP=4097
so the first prefill chunk is 4096 = the OOM site,
`VLLM_USE_AOT_COMPILE=0` after the AOT-worker crash, see
DEVLOG-muse-glimmer.md round 3 for the inductor-host-crash chase).

**Timeline.**
1. **~20:40:59Z** — model load starts (5th in-process probe launch on
   boot K: custom3/4/6 died earlier in host-level inductor crashes,
   custom5 in an import-time AttributeError — none of those touched a
   GPU kernel launch hard, but the boot has also carried the 17:40/17:52
   boot-J relaunch failures and the 18:23 GPU1 reset).
2. **~20:41:0xZ** — at 20% into weight load (shard 1/5→2 boundary):
   `terminate called after throwing an instance of 'c10::AcceleratorError'
   what(): CUDA error: unspecified launch failure` — the **7th
   occurrence** of the chronic weight-load hang across boots (same
   family: 08-26 06:03/06:52, 08-27 08:2x ×2, 17:40/17:52, 18:23).
   kern.log/journal unreadable (no root), so the reset type is
   inferred, not observed.
3. **~20:42Z** — rocm-smi: both cards back at the 10.8 MB VRAM
   baseline, 0% util. The driver completed its own reset and the GPU
   is usable again — an *isolated* wedge with self-recovery.

**Assessment.** Isolated per the house recipe → one retry of the probe.
If the retry wedges again that is a burst (2 consecutive launch
failures on this boot's tail) → stop GPU work, reboot (root). Note the
boot has now accumulated 3 GPU-side incidents (18:23 GPU1, 20:40 GPU0,
plus the boot-J carry-over); the degradation-onset question (when does
a boot enter the state where weight loads wedge?) stays open — see the
Open questions section.

## 2026-08-27 21:57–21:59Z (boot K): OOM-attribution probe (q_pad-fix verification run) wedges GPU0; driver self-recovers

**Context.** Boot K's 4th GPU-side incident (after 18:23 GPU1, 20:40
GPU0, plus the 21:0x/21:1x/21:2x in-process probe launches that ran
clean). The run was the post-fix verification of the q_pad ClassVar
change (DEVLOG-muse-glimmer round 4 root cause;
`attr_tp1_custom18_fixed.log`): in-process custom arm, TP=1, 0.5 GiB
KV cap, PP=4097, weight load started ~20 s in.

**Timeline.**
1. **21:57:28Z** — engine init: backend registered, weight load
   begins (GPU0).
2. **~21:58:3xZ** — `terminate called after throwing an instance of
   'c10::AcceleratorError' what(): CUDA error: unspecified launch
   failure` — the 8th occurrence of the chronic launch-failure
   family across boots. kern.log/journal unreadable (no root).
3. **~22:00Z** — rocm-smi: GPU0 back at the 11.2 MB VRAM baseline,
   0% util; the driver completed its own reset (self-recovery, same
   pattern as 20:40). GPU1 unaffected (a test suite ran on it
   throughout).

**4. ~22:03Z** — a 30 s torch canary on GPU0 (200 fp16 matmuls)
passed: 3.01 s, clean. GPU1 unaffected throughout (the 51/51
q_pad-fix test suite completed on it at ~22:06).
5. **~22:06Z** — the ONE allowed retry of the probe
(`attr_tp1_custom19_fixed_retry.log`) wedged GPU0 again — 2nd
consecutive launch failure, same `unspecified launch failure`
signature. And after this one the driver did NOT self-recover: GPU0
stuck at **24.9 GB zombie VRAM**, 0% util (the 20:40 and 21:58
crashes both released to the ~11 MB baseline; this one held the
reservation).

**Assessment.** **BURST per house recipe (2 consecutive launch
failures) → all GPU work stopped; the host needs a reboot (root)** —
BACO reset also needs root, and the 2nd-failure rule stops short of
even attempting it. Boot K's GPU incident count: 18:23 (GPU1), 20:40
(GPU0, self-recovered), 21:58 (GPU0, self-recovered), 22:06 (GPU0,
ZOMBIE VRAM). This boot is done for GPU work.

**Pending post-reboot** (in order): (1) canary (Qwen3.8-27B mtp2,
expect 38–47 t/s); (2) q_pad-ClassVar fix verification — re-run the
OOM-attribution custom arm (expect: one-time 256 MiB grow, survival
at the 0.5 GiB KV cap, transient ≈ model core + 0.26 + churn, i.e.
~1.5 GiB vs the 3.785 pre-fix); (3) M1 gather-clip e2e A/B
(pp8192/B=1 tg256, `GFX906_FA_GATHER_CLIP` 1 vs 0, record recipe);
(4) bt4096 TP=2 serving re-validation — the fix removes the
6.7 GiB/GPU q_pad growth that forced the bt2048 workaround, so the
boot K launch recipe can drop `--max-num-batched-tokens 2048` (and
the 6 GiB KV cap can shrink) if prefill clears; (5) write the
gfx906-mem-attribution skill (roadmap Housekeeping) with the
validated recipe (3-arm matrix + per-layer `memory_allocated()`
hooks + the env traps: `VLLM_USE_AOT_COMPILE=0`, thread compile
pool, `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`).

## 2026-08-28 ~06:14Z (boot L): post-reboot canary + q_pad-fix verification — clean

1. **~06:14Z** — fresh boot (uptime 1 min at check); both GPUs at the
   10.8 MB baseline, 0% util.
2. **Canary** (Qwen3.8-27B mtp2, GPU0, `canary_bootL.log`): **38.8
   t/s** — in the recent healthy band (38.4–38.9). Host clear.
3. **q_pad-ClassVar fix verification** (pending list item 1; custom
   attribution arm, TP=1, 0.5 GiB KV cap, PP=4097,
   `attr_tp1_custom20_bootL.log`): **SURVIVED**, peak transient
   **1.285 GiB** (vs 3.785 pre-fix on boot K), 4.89 GiB free after
   (vs 0.00). The boot J/K first-prefill OOM root cause is confirmed
   fixed (DEVLOG-muse-glimmer round 4). Remaining pending: M1
   gather-clip e2e A/B, bt4096 TP=2 serving re-validation, the
   gfx906-mem-attribution skill.
4. **~06:40Z** — **M1 gather-clip e2e A/B: PASS, +8.1%** (harness
   record recipe, pp8192/B=1/tg256, `GFX906_FA_GATHER_CLIP` 1 vs 0:
   6.042 vs 5.587 t/s; DEVLOG-muse-glimmer round 5).
5. **~06:55Z** — **bt4096 TP=2 serving re-validation: PASS** (boot K
   recipe + `--max-num-batched-tokens 4096`; first real 8192 request
   cleared — the exact boot J/K OOM site — cold prefill 452 t/s,
   warm ~99 t/s decode @8k/B=1, 8.7 GiB headroom/GPU; clean SIGTERM
   teardown to the 10.8 MB baseline; the bt2048 workaround is
   droppable — README updated).
6. **~07:05Z** — **gfx906-mem-attribution skill written**
   (`/home/kread/.agents/skills/gfx906-mem-attribution/SKILL.md`);
   the attribution probe persisted in-repo
   (`docs/gfx906/_probe_mem_attribution_gfx906.py`). All boot-K
   post-reboot pending items complete. Boot L clean throughout
   (canary 38.8 t/s; no wedges).
7. **~12:43Z** — **1st wedge of boot L: dual weight-load launch
   failures (GPU0 + GPU1 simultaneous)**. Two in-process harness runs
   (G1 B=2 / G2 B=4 `GFX906_FA_WINDOW_CLIP` A/B for the LEGACY=0
   default-flip gate, Muse-Glimmer, default LEGACY=1) both aborted at
   weight load — `terminate called after throwing an instance of
   'c10::AcceleratorError' ... CUDA error: unspecified launch
   failure` (`hipErrorLaunchFailure`), in the Exllama load path
   (checkpoint stats logged, then terminate — before any FA kernel
   ran, so the round-6 fused-clip rebuild is not implicated by the
   crash site). Both cards affected in the same ~30 s window ~6.5 h
   into boot L; rocm-smi self-recovered (0%/0% both cards, no zombie
   VRAM) within ~1 min of the crashes — matches the chronic
   weight-load-hang family (9th occurrence across boots) and the
   host-degradation signature (sync-cadence-heavy load work dies,
   driver recovers). Evidence: `/tmp/g1_clipon_b2.log`,
   `/tmp/g2_clipon_b4.log` (boot-volatile; signatures quoted here).
   **Canary skipped on operator instruction** ("card was working just
   now") — if the retry loads but t/s land low, the host is the
   suspect. Protocol state: isolated → 1 retry (G1 on GPU0); a 2nd
   launch failure = burst → stop all GPU work + reboot (root).
8. **~12:54Z** — **2nd boot-L wedge pair: concurrent G1(n=2)/G2(n=4)
   relaunch, both cards, same weight-load `unspecified launch
   failure`** (evidence: `/tmp/g1_clipon_b2_n2.log`,
   `/tmp/g2_clipon_b4_n4.log`; self-recovered to 10.8 MB both).
   Context: the 12:43 pair's single-card retry (G1, ~12:52) loaded
   clean and ran to completion at **6.06 t/s** (B=1/pp8192/tg256 vs
   the 6.042 pre-wedge record — host healthy mid-interval). Pattern
   so far on boot L: every double wedge followed a CONCURRENT
   two-card launch; every clean run was single-card. Hypothesis
   (unverified): two heavy HSA loads in the same ~30 s window race a
   host/driver resource on this dual-root-port topology. Mitigation
   from here: serialize all GPU work (one in-process or one TP=2 job
   at a time). Protocol state: 2nd observation, chain was broken by
   the 12:52 success → G1 retried once; a 3rd observation or a retry
   failure = burst → stop + reboot (root).
9. **~14:38Z** — **3rd boot-L wedge: G3 LEGACY=0 TP=2 bake retry,
   `hipErrorLaunchFailure` in weight `copy_`→SetDevice (Worker_TP1,
   both ranks; evidence: `/tmp/g3_legacy0_serve.log` 2nd write, boot-
   volatile; signatures quoted here).** First G3 attempt (~14:20)
   died with a CODE error instead — a capture-unsafe D2H sync
   (`int(cu[s+1]-cu[s])`) in the direct-paged branch's Sq>1 Q-pad
   loop, hit for the first time in production (LEGACY=0 + ngram spec
   + B≥2 decode = Sq=6 direct-paged under FULL decode capture);
   fixed in-tree (uniform-batch fast paths, mirroring the gather
   branch's capture-safe idiom; 57/57 suite incl. the nq=6/B=2
   bit-identity A/B now exercising them). The retry then wedged at
   weight load — the 14:20 code death had already torn down both
   workers, so the wedge may be collateral to that teardown (cf. the
   documented SIGKILL-mid-P2P → next-init wedge), or a 3rd
   accumulation. Either way: **3rd wedge observation of boot L =
   BURST per the protocol recorded at 12:54Z → ALL GPU WORK STOPPED;
   host needs a REBOOT (root).** Pending post-reboot: the G3 LEGACY=0
   TP=2 serving bake (recipe: `HIP_VISIBLE_DEVICES=0,1
   GFX906_FA_LEGACY=0` + README TP=2 flags; grid
   `_bench_serve_grid_gfx906.py` default ×3; control = the boot L
   LEGACY=1 records 111.5/99/46.7 @2k/8k/B=4). Pattern note: all 3
   boot-L wedges involved two-card launches; all 6 single-card
   in-process runs (incl. the post-wedge A/Bs, 13:0x–14:3x) were
   clean — the concurrent-two-card race hypothesis from 12:54Z is
   untested by design (TP=2 IS two cards), and boot L is ~8 h in
   with 3 wedges accumulated — consistent with the degradation-onset
   model (enough half-wedge resets in one boot degrade the host).
10. **~15:00Z (boot M)** — **1st boot-M wedge: G3 LEGACY=0 TP=2
    attempt 1, `hipErrorLaunchFailure` both ranks at weight load**
    (~5.5 min in; evidence: `/tmp/g3_legacy0_serve.log` boot-M
    write, boot-volatile; signatures quoted here). Context: fresh
    boot (uptime ~8 min), canary **38.9 t/s** at ~14:58Z (2 min
    prior) — host healthy at launch. Chronic weight-load-hang
    family, 10th occurrence across boots; the TP=2 two-rank weight
    load is the usual trigger (cf. boot K 18:23:30, 7th). Driver
    self-recovered to the 10.8 MB baseline within ~4 min; no zombie
    VRAM. Protocol state: isolated → 1 retry; a 2nd launch failure
    on boot M = burst → stop + reboot (root).
