# GPU wedge / host-degradation event log — 2× MI50 (gfx906), official amdgpu DKMS

Copyright Kevin Read <me@kevin-read.com>

Protocol: **every** half-wedge, full wedge, and host-degradation
observation gets a row here (timestamped UTC), with a pointer to the
details in `degradation_details.md`. Sources: `journalctl -k` /
`/var/log/kern.log` (`GPU reset begin/end`, `PSP resume failed`,
`qcm fence wait loop timeout`), server logs, bench harness logs.

Legend: **HW** = half-wedge (GPU reset Source:4, recovered, "VRAM is
lost"), **FW** = full wedge (reset → `PSP resume failed` / ret −62,
GPU dead until reboot/power-cycle), **DEG** = host-state degradation
(serving stays up but sync-cadence-heavy paths collapse; see
`degradation_details.md`), **BOOT** = host boot boundary.

| UTC timestamp | GPU | type | context / trigger | evidence, outcome |
|---|---|---|---|---|
| 2026-08-21 03:17 | 0 | FW | (pre-gfx906-session server work) | PSP resume failed, ret −62; reboot followed |
| 2026-08-21 11:26 | 0 | FW | TP=2 serving session server churn | PSP resume failed, ret −62; reboot 11:52 |
| 2026-08-21 14:21–17:35 (8×) | mostly 1 | HW | TP=2 server boot/teardown churn (S4-S6 era) | Source:4 resets, recovered each time |
| 2026-08-21 21:47–23:52 (5×) | 0/1 | HW | TP=2 serving, server restarts | recovered; plain decode unaffected |
| 2026-08-22 00:01–02:42 (9×) | 0/1 | HW | N4 plain-greedy TP=2 A/B night (fa_ab) | recovered; plain decode 40.86 t/s — unaffected |
| 2026-08-22 06:22 | 0 | HW | F7 arm weight-load SIGABRT (m32768_p0, exit 134) | recovered |
| 2026-08-22 06:46–11:39 | — | **DEG** | mtp2 TP=2 serving 3× slow (120.5 vs 40.1 ms/step post-reboot) in a boot with ≥14 prior HW resets, ~14 h uptime, ~10× 20-GB weight-load cycles | 4 separate server boots all degraded (24.90/24.93 steady); plain TP=2/TP=1 fine; trace: worker blocked in `hipEventSynchronize` ~57 ms/step; cleared ONLY by reboot |
| 2026-08-22 08:33, 10:00, 10:02 | 0 | HW | in-process TP=1 runs SIGABRT (rc=134, weight load) | recovered |
| 2026-08-22 10:26, 10:41×2, 10:43×2, 10:45 | 0/1 | HW | mtp1 TP=2 arm: 3× `hipErrorLaunchFailure` at weight load (wedge burst) | recovered per-reset; arm abandoned |
| 2026-08-22 11:03, 11:13 | 1/0 | HW | trace + async-scheduling runs, boot failures | recovered |
| — | | **BOOT** | 2026-08-22 12:39:47 reboot | DEG gone: mtp1/mtp3/mtp2 serving all healthy (33 / 45.4 / 40.1 ms/step) |
| 2026-08-22 12:53, 12:55 | 1, 0 | HW | mtpscale n3 boot attempts 1-2 failed | recovered; n3 booted 3rd try, served FAST (88 t/s) — 2 resets did NOT degrade |
| 2026-08-22 13:44 (×2), 13:46 | 0, 1 | HW | during/after si8 serve window (qcm fence timeout, failed queue evict) | si8 bench still completed at 74.8 t/s |
| 2026-08-22 13:55–14:04 (6×) | 1 | HW | si32 + mtp2-rebaseline boot failures (half-wedge cascade) | each "recovered" but next boot still failed |
| 2026-08-22 14:06 | 0 | **FW** | mtp2_rb m131072_p1 boot attempt | reset → PSP resume failed ret −62; GPU0 dead (rocm-smi N/A, 31 % zombie VRAM) at 14:09; **needs reboot** |
| — | | **BOOT** | 2026-08-22 ~14:50 reboot (GPU0 recovered, temps/clocks normal) | canary 38.6 t/s = healthy; **2 prior resets were insufficient to degrade; ≥14 were** (onset bracket) |
| 2026-08-22 15:24, 15:40, 15:41 | 1 | HW | mtp2 re-baseline boot-attempt failures | recovered; all 4 arms then benched HEALTHY (49.4/37.7/74.7/73.5 t/s) — **3 resets did not degrade** (onset data point) |
| 2026-08-22 20:05–20:12 | 1 (GPU1 44% transient, no owner) | SW | first TP=2 35B-MoE run (C2-V t2n1_off): rank-1 worker SIGSEGV-class crash in `RocmPlatform.get_device_name` (amdsmi `NOT_INIT`, masked by `with_amdsmi_context` finally); rank-0 hung on shm broadcast till SIGTERM. **No kernel reset** in window (journal clean); amdsmi broken on every run this boot (protected fallback at rocm.py:913; `get_device_name` the one unprotected caller). TP=2 arms re-ran clean with a sitecustomize shim | no GPU damage; software crash |
| 2026-08-23 05:18:29 | 0 | HW | 27B mtp2 canary (W2 session) SIGABRT rc=134; `Fence fallback timer expired on ring comp_1.0.0` → `GPU reset(1) succeeded` — first HW reset this boot. Pre-wedge: amdsmi broken since boot (08-22 20:05Z); 35B-MoE mtp2 smoke ~39.5 t/s decode (vs ~81 baseline) + baseline greedy FP non-reproducible 04:5x–05:1xZ (**later explained: the 35B MoE baseline is non-deterministic at temp=0 in this build — post-reset re-runs drift identically; not a host artifact**) | recovered; canary 38.9 t/s = healthy-for-this-boot |
| 2026-08-23 05:47:46 | 0 | HW | W2 w2_mtp2_e arm (35B-MoE mtp2 eager boot) — `hipErrorLaunchFailure` at SetDevice; same comp_1 fence timeout → `GPU reset(2) succeeded`. 4 clean engine cycles (base_g/base_g2/mtp2_g/base_e) in the 05:18–05:47 window | recovered at the time; canary re-run next |
| 2026-08-23 06:08:37 | 0 | **FW** | post-reset#2 canary: `MAPPING_ERROR 0x1` → `PSP resume failed` → `GPU reset end with ret = -62` — **GPU0 dead, needs host reboot** (3rd comp_1 fence timeout this boot in 50 min: 05:18 / 05:47 / 06:08 — burst pattern per boot/recovery procedure) | **REBOOT REQUIRED**; W2 mtp2 arms (acceptance re-run) paused until after reboot |
| — | | **BOOT** | 2026-08-23 06:33 boot (Kevin) | clean: GPU temps 33/32 °C, 0 amdgpu errors since boot, canary 38.6 t/s (healthy-for-this-model); W4 A/B on the fresh boot: 35B N=8 off/on 166.9/191.0 t/s (+14.5 %, off = C2-V record 167.4 → host clean) |
| 2026-08-23 ~08:51, ~08:52 | 0 | HW | W4 27B (Qwen3.8) N=8 arms: off arm OOM'd at util 0.93 (356 MB inductor buf, free: 0) and aborted; next (on) arm `hipErrorLaunchFailure` rc=134 at boot | closed: clean re-run at util 0.90 passed (98.2/104.2 t/s, no recurrence, 2 engines booted fine after) — OOM-teardown collateral, NOT a wedge (details file § 2026-08-23) |
| 2026-08-23 ~11:36–11:44 | 0, 1 | SW/HW? | Qwen3.8-27B TP=2 256k serve (3rd boot attempt, util 0.85): worker stuck in init on shm-broadcast wait (~11:36), 39 % VRAM pinned per GPU, GPU use 0 %, SIGTERM not honored (stuck in C extension) | SIGKILL'd (GPU idle — low P2P-mid-op risk); 2 prior TP=2 boots this session were clean, so this is a one-off init deadlock, pattern = C2-V t2n1_off shm hang; watch next boot for GPU1 wedge (known SIGKILL-TP=2 risk) |
| 2026-08-23 ~12:48 | 1? | HW? | TP=2 no-prefix-caching boot (Qwen3.8-27B 256k OOM investigation): `hipErrorLaunchFailure` at init (~12:48:46), EngineCore failed to start. Prior teardown was a clean SIGTERM (VRAM 0 % verified), two earlier boots since the 11:47 SIGKILL were clean | VRAM 0 % both GPUs, rocm-smi responsive, no zombie procs; retrying once (house recipe); if it fails again → BACO/reboot. Possible drift toward host degradation — canary probe if any further anomaly |
| 2026-08-23 17:59–18:06 (3×) | 0 | HW | OOM-hunt session canary attempts 1-3 (27B mtp2 canary, weight load): each SIGABRT rc=134 `hipErrorLaunchFailure` at shard 2/5→3, each triggering `Fence fallback timer expired on ring comp_1.0.0` → `GPU reset(2/3/4) succeeded` (18:00:02, 18:01:35, 18:06:15; reset #1 was 12:48 GPU1). VRAM clean between attempts; AMD_SERIALIZE_KERNEL=3 on attempt 3 did not change the failure point | 4 resets this boot (since 06:33) — burst pattern; STOPPED retrying per boot/recovery procedure (full-wedge risk). Canary never completed: host-health verdict deferred. Not a DEG signature per se (failures are at load, not sync-cadence), but treat all subsequent boots as suspect until a canary passes |
| 2026-08-23 18:23 | 1 | HW | OOM-hunt 9B TP=2 instrumented boot (Qwen3.5-9B, gather-gen probe): rank-1 worker `hipErrorLaunchFailure` at SetDevice → `GPU reset(2) succeeded` on 0000:0e:00.0 (GPU1, 18:23:00). VRAM 0 % after; killed cleanly (SIGTERM, no vllm procs left) | 5+ resets this boot — confirmed wedge-burst state; **host reboot requested** (Kevin offered) before any further GPU work. All OOM-hunt GPU experiments deferred to the fresh boot |
| 2026-08-23 18:35:55 | 0 | HW | OOM-hunt canary (27B, weight load): comp_1 fence timeout → reset; **`ERREVENT_ATHUB_INTERRUPT` uncorrectable hardware error latched** (on-die host fabric hub) — first on-die RAS latch of the day | recovered |
| 2026-08-23 18:39:40 | 1 | HW | OOM-hunt GPU1 canary (canary_gpu1.sh): comp_1 fence timeout → reset | recovered; GPU1 quiet after this |
| 2026-08-23 18:50:19 | 0, 1 | HW | **pcie_bif correctable RAS latch on BOTH cards within 6 ms** (18:50:19.637 GPU0 / .643 GPU1) — simultaneous dual-card timing; shared host-side cause or delayed post-reset flush (ambiguous) | counters re-latched; no wedge |
| — | | **BOOT** | 2026-08-23 19:20:55 boot (full power cycle, 6 min off per OOM-hunt doc) | first canary on the fresh boot FAILED at 19:23:30 — power cycle did NOT clear the fault |
| 2026-08-23 19:23:30, 19:35–19:46 (3×) | 0 | HW | OOM-hunt on fresh boot: canary_postpc, trace/nosdma canaries, safeopen/mater repros — all `hipErrorLaunchFailure` mid weight-load (DEVLOG-boot-failure.md §1) | each recovered via BACO reset |
| 2026-08-23 20:14:35 + 20:15:16, 20:15:48 (2×) | 0 | HW | boot-failure 2nd session (pi): pcie_bif correctable latch 41 s before a live-caught comp_1 wedge; double-wedge 32 s apart; same boot had PASSING windows (repro ×4 + full canary 38.9 t/s at 20:03–20:08; repro ×5+ after 20:16) — **intermittent die/fabric flap, GPU0 primary (16/19 resets today); GTT-exhaustion theory REFUTED (20 MiB peak of 12,260 MiB during the full 19.57 GiB load)** | recovered per reset; host usable in good windows; card-swap test recommended — DEVLOG-boot-failure.md §7 |
| 2026-08-23 20:21:33 | 0 | HW | isolated comp_1 fence timeout during the "good window" (30 s after the repro loop ended; no wedge before/after within minutes — runs 20:23:50, 20:40, 20:41 all passed) | recovered; good windows contain isolated wedges — flap, not strict good/bad phases |

Maintained by: whoever hits the next one. Update BOTH files.
