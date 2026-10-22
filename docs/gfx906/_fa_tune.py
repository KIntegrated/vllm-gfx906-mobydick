#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Measure gfx906 FA launch-config candidates per shape (P32-4/#63, slice 4).

The static rules (`tuning.static_config`) encode two measured swings: the KV
split at Sq=2 vs Sq>=4, and the batch-aware clamp on the paged-direct path.
Everything else is interpolation. This tool measures the alternatives directly
through the per-call overrides (`nc2=`, `kv_split=`) and prints, per shape, the
median time of each candidate -- so a record can be justified by data instead of
by inheriting a default.

It does not write records: run it, read the table, and only then decide whether
a shape is worth pinning (the point of the exercise is to find out whether the
rules are already right).

Usage:
    HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        .venv/bin/python docs/gfx906/_fa_tune.py [--reps 30]
"""

from __future__ import annotations

import argparse
import glob
import re
import statistics
import threading
import time

import torch

from vllm import _gfx906_fa_C as fa
from vllm.gfx906_fa.tuning import static_config

# (label, B, Hq, Hkv, D, Sq) — one decode, one batched decode, one MTP verify,
# one short prefill.
SHAPES = [
    ("decode B=1 Sq=1", 1, 24, 4, 256, 1),
    ("decode B=4 Sq=2", 4, 24, 4, 256, 2),
    ("verify B=1 Sq=8", 1, 24, 4, 256, 8),
    ("verify B=2 Sq=8", 2, 24, 4, 256, 8),
]

SEQ_K = 65536
NC2_CANDIDATES = (1, 2, 8)
KVSPLIT_CANDIDATES = (8, 16, 32)


class MclkSampler:
    """Concurrent mclk sampling, per docs/gfx906/dvfs-mi50.md.

    Not ``rocm-smi --showclocks``: that call blocks while a deck is under load
    and reports DPM levels rather than the live clock (it read a flat 350 MHz
    through a 30 s GEMM preload). The sysfs active-level line is the truth.
    """

    _ACTIVE = re.compile(r"^\s*\d+:\s*(\d+)\s*Mhz\s*\*", re.M)

    def __init__(self, period: float = 0.25):
        self.period = period
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._cards = sorted(glob.glob("/sys/class/drm/card*/device/pp_dpm_mclk"))
        self._t = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> None:
        for path in self._cards:
            try:
                with open(path) as handle:
                    match = self._ACTIVE.search(handle.read())
            except OSError:
                continue
            if match:
                self.samples.append(int(match.group(1)))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._read()
            time.sleep(self.period)

    def __enter__(self) -> MclkSampler:
        self._read()
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=2)

    def report(self) -> str:
        if not self.samples:
            return "mclk: unreadable (no sysfs samples)"
        loaded = [s for s in self.samples if s >= 400] or self.samples
        share = 100.0 * sum(1 for s in loaded if s >= 900) / len(loaded)
        med = statistics.median(loaded)
        verdict = "" if med >= 900 else "  <-- BELOW 900 MHz: cold-clock data"
        return (f"mclk: median {med:.0f} MHz, {share:.0f} % of loaded samples "
                f">= 900 MHz ({len(self.samples)} samples){verdict}")


def ramp_mclk(seconds: float) -> None:
    """Sustained fp16 GEMM load so the memory clock is up while we measure.

    These kernels run in 3-6 ms bursts, which never lifts mclk off its 350 MHz
    idle state on this MI50; a kvsplit optimum measured there is measured in the
    wrong regime, because splitting trades parallelism for traffic.
    """
    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    deadline = time.time() + seconds
    while time.time() < deadline:
        a = a @ a
        a = a / a.abs().max().clamp(min=1e-3)  # keep values finite
    torch.cuda.synchronize()


def make_inputs(batch, hq, hkv, d, sq, sk):
    q = torch.randn(batch, hq, sq, d, dtype=torch.float32, device="cuda")
    packed_row = (d // 32) * 34  # q8_0 row: D quants + D/32 fp16 scales
    k_q8 = torch.zeros(batch, hkv, sk, packed_row, dtype=torch.uint8, device="cuda")
    v = torch.randn(batch, hkv, sk, d, dtype=torch.float16, device="cuda")
    return q, k_q8, v


def time_config(q, k_q8, v, scale, nc2, kv_split, reps):
    kv_max = torch.full((q.shape[0],), SEQ_K, dtype=torch.int32, device="cuda")
    # Warm up (allocator + any lazy setup), then measure.
    for _ in range(3):
        fa.forward(q, k_q8, v, scale, kv_max=kv_max, nc2=nc2, kv_split=kv_split)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fa.forward(q, k_q8, v, scale, kv_max=kv_max, nc2=nc2, kv_split=kv_split)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--preload-seconds", type=float, default=25.0,
                    help="sustained GEMM load before measuring, to raise mclk")
    args = ap.parse_args()

    print(f"seq_k={SEQ_K} reps={args.reps} (median ms per forward)")
    if args.preload_seconds > 0:
        with MclkSampler() as sampler:
            ramp_mclk(args.preload_seconds)
        print(f"preload {args.preload_seconds:.0f}s -> {sampler.report()}")
    print()
    for label, batch, hq, hkv, d, sq in SHAPES:
        gqa = hq // hkv
        scale = d**-0.5
        q, k_q8, v = make_inputs(batch, hq, hkv, d, sq, SEQ_K)
        base = static_config(sq, gqa, batch=batch)
        print(f"{label}  (gqa={gqa}, static rule: nc2={base.nc2} "
              f"kvsplit={base.kvsplit})")
        rows = []
        sampler = MclkSampler()
        sampler.__enter__()
        for nc2 in NC2_CANDIDATES:
            for kvsplit in KVSPLIT_CANDIDATES:
                try:
                    ms = time_config(q, k_q8, v, scale, nc2, kvsplit, args.reps)
                except RuntimeError as exc:  # invalid combination for this shape
                    print(f"    nc2={nc2} kvsplit={kvsplit:2d}  rejected: {exc}")
                    continue
                rows.append((ms, nc2, kvsplit))
                star = " *" if (nc2, kvsplit) == (base.nc2, base.kvsplit) else ""
                print(f"    nc2={nc2} kvsplit={kvsplit:2d}  {ms:8.3f} ms{star}")
        sampler.__exit__()
        print(f"    {sampler.report()}")
        if rows:
            rows.sort()
            best_ms, best_nc2, best_kv = rows[0]
            rule = next(
                (ms for ms, n, k in rows if (n, k) == (base.nc2, base.kvsplit)), None
            )
            if rule is not None:
                delta = 100.0 * (rule - best_ms) / rule
                print(f"    best: nc2={best_nc2} kvsplit={best_kv} "
                      f"({best_ms:.3f} ms) vs static rule {rule:.3f} ms "
                      f"-> {delta:+.1f}%\n")
            else:
                print(f"    best: nc2={best_nc2} kvsplit={best_kv} "
                      f"({best_ms:.3f} ms); static rule not measured\n")


    print("mclk is sampled per shape below; a median under 900 MHz means the "
          "numbers are cold-clock and must not be used to pin a record.")


if __name__ == "__main__":
    main()
