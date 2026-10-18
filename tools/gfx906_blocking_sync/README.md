# gfx906 blocking-sync `.pth` (LAB-7, #55)

Ported from [0c33/mi50-vllm-lab](https://github.com/0c33/mi50-vllm-lab)
(Apache-2.0, `docker/gfx906-blocking-sync.pth`).

Installing this file into site-packages makes every process call
`hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` on all visible devices before
torch initialises — a **diagnostic lever** for the sync-cadence wedge family
(spec decode collapsing to a fraction of its normal ms/step while plain decode
is unaffected; see `docs/gfx906/degradation.md`), not a default.

```bash
cp tools/gfx906_blocking_sync/gfx906-blocking-sync.pth \
   "$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
# opt out without uninstalling:
export VLLM_GFX906_HIP_BLOCKING_SYNC=0
```

**Measured here (2026-10-18):** no effect on a healthy TP=2 boot. Bolt 74B +
LiLiCorr k=7, kernels on, same prompt and 2 reps per arm:

| arm | t/s | ms/step | acc/draft |
|---|---|---|---|
| lever off | 119.5, 130.7 | 61.8, 56.5 | 6.77 |
| lever on | 119.1, 130.6 | 62.0, 56.5 | 6.77 |

So it is kept as a lever only: a ~30-second way to confirm "sync cadence, not
kernels" if a future TP=2 spec run wedges, without reaching for a reboot.
