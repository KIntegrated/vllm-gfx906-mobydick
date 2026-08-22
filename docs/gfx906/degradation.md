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

Maintained by: whoever hits the next one. Update BOTH files.
