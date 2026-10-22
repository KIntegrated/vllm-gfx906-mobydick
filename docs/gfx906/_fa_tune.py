#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Measure gfx906 FA launch-config candidates per shape (P32-4/#63, slice 4).

Two modes:

* default — for a few representative shapes, time every (nc2, kvsplit)
  candidate and compare against `tuning.static_config`.
* `--sweep` — the (batch, seq_q) x kvsplit grid across both model geometries at
  TP=1 and per-rank TP=2, to answer whether the KV-split rule should key on
  batch x seq_q rather than seq_q alone.

Measurement discipline (docs/gfx906/dvfs-mi50.md), because these kernels are
3-6 ms bursts that never leave the 350 MHz idle state on their own:

* every shape gets its own preload, and a window whose mclk samples are not
  >= 900 MHz is **discarded and retried**, never averaged in;
* candidates are timed **interleaved** with a rotating order, so a slow window
  cannot masquerade as a slow candidate (the sequential-arm lesson);
* mclk is read from `/sys/class/drm/card*/device/pp_dpm_mclk`, not from
  `rocm-smi --showclocks` — that call blocks under load and reports DPM levels
  (it read a flat 350 MHz through a 30 s GEMM preload).

Usage:
    HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        .venv/bin/python docs/gfx906/_fa_tune.py [--sweep] [--reps 30]
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
from vllm.gfx906_fa.tuning import (
    FAConfig,
    TuningTable,
    cache_path,
    identity,
    static_config,
    workload_key,
)

SEQ_K = 65536
NC2_CANDIDATES = (1, 2, 8)
KVSPLIT_CANDIDATES = (8, 16, 32)

TABLE_SHAPES = [
    ("decode B=1 Sq=1", 1, 24, 4, 256, 1),
    ("decode B=4 Sq=2", 4, 24, 4, 256, 2),
    ("verify B=1 Sq=8", 1, 24, 4, 256, 8),
    ("verify B=2 Sq=8", 2, 24, 4, 256, 8),
]

# The GQA ratio is preserved under TP here (27B 24/4 -> 12/2, 35B 16/2 -> 8/1),
# but the grid shrinks -- which is exactly what a shape-blind rule can miss.
SWEEP_GEOMETRIES = (
    ("27B TP1", 24, 4, 256),
    ("27B TP2", 12, 2, 256),
    ("35B TP1", 16, 2, 256),
    ("35B TP2", 8, 1, 256),
)
SWEEP_BATCHES = (1, 2, 4, 8)
SWEEP_SQ = (2, 4, 8)


class MclkSampler:
    """Concurrent per-card mclk sampling (pp_dpm_mclk active level)."""

    _ACTIVE = re.compile(r"^\s*\d+:\s*(\d+)\s*Mhz\s*\*", re.M)

    def __init__(self, period: float = 0.05):
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

    def valid(self) -> bool:
        loaded = [s for s in self.samples if s >= 400] or self.samples
        if not loaded:
            return False
        return statistics.median(loaded) >= 900

    def report(self) -> str:
        if not self.samples:
            return "mclk unreadable"
        loaded = [s for s in self.samples if s >= 400] or self.samples
        share = 100.0 * sum(1 for s in loaded if s >= 900) / len(loaded)
        med = statistics.median(loaded)
        verdict = "" if med >= 900 else " BELOW 900"
        return f"mclk {med:.0f} MHz/{share:.0f}% (n={len(self.samples)}){verdict}"


def ramp_mclk(seconds: float) -> None:
    """Sustained fp16 GEMM so the memory clock is up before measuring."""
    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    deadline = time.time() + seconds
    while time.time() < deadline:
        a = a @ a
        a = a / a.abs().max().clamp(min=1e-3)
    torch.cuda.synchronize()


def make_inputs(batch, hq, hkv, d, sq, sk):
    q = torch.randn(batch, hq, sq, d, dtype=torch.float32, device="cuda")
    packed_row = (d // 32) * 34  # q8_0 row: D quants + D/32 fp16 scales
    k_q8 = torch.zeros(batch, hkv, sk, packed_row, dtype=torch.uint8, device="cuda")
    v = torch.randn(batch, hkv, sk, d, dtype=torch.float16, device="cuda")
    return q, k_q8, v


def time_candidates(q, k_q8, v, scale, nc2, candidates, reps, kv_max):
    """Time candidates interleaved, rotating the order each repetition."""
    timed = {kv: [] for kv in candidates}
    for rep in range(reps):
        shift = rep % len(candidates)
        order = candidates[shift:] + candidates[:shift]
        for kv in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fa.forward(q, k_q8, v, scale, kv_max=kv_max, nc2=nc2, kv_split=kv)
            end.record()
            torch.cuda.synchronize()
            timed[kv].append(start.elapsed_time(end))
    return {kv: statistics.median(ts) for kv, ts in timed.items()}


def measure_gated(batch, hq, hkv, d, seq_k, sq, nc2, candidates, reps,
                  preload_s, attempts=3):
    """Measure one shape, discarding and retrying any cold-clock window."""
    scale = d**-0.5
    for attempt in range(1, attempts + 1):
        q, k_q8, v = make_inputs(batch, hq, hkv, d, sq, seq_k)
        kv_max = torch.full((batch,), seq_k, dtype=torch.int32, device="cuda")
        ramp_mclk(preload_s)
        with MclkSampler() as sampler:
            times = time_candidates(q, k_q8, v, scale, nc2, candidates, reps,
                                    kv_max)
        ok, report = sampler.valid(), sampler.report()
        del q, k_q8, v, kv_max
        if ok:
            return times, report, attempt
    return None, report, attempts


def run_table(reps: int, preload_s: float) -> None:
    print(f"table mode, seq_k={SEQ_K}, reps={reps}")
    for label, batch, hq, hkv, d, sq in TABLE_SHAPES:
        gqa = hq // hkv
        base = static_config(sq, gqa, batch=batch)
        rows = []
        for nc2 in NC2_CANDIDATES:
            times, report, attempt = measure_gated(
                batch, hq, hkv, d, SEQ_K, sq, nc2, KVSPLIT_CANDIDATES, reps,
                preload_s,
            )
            if times is None:
                print(f"{label} nc2={nc2}: DISCARDED ({report})")
                continue
            for kv, ms in times.items():
                rows.append((ms, nc2, kv))
            print(f"{label} nc2={nc2} [{report}, try {attempt}] "
                  + "  ".join(f"y{kv}={v:.3f}" for kv, v in sorted(times.items())))
        if rows:
            rows.sort()
            best_ms, best_nc2, best_kv = rows[0]
            rule = next((ms for ms, n, k in rows
                         if (n, k) == (base.nc2, base.kvsplit)), None)
            if rule is not None:
                print(f"  best nc2={best_nc2} y={best_kv} {best_ms:.3f} ms vs "
                      f"rule {rule:.3f} ms -> {100 * (rule - best_ms) / rule:+.1f}%\n")


def write_record(device: str, dtype: str, hq: int, hkv: int, d: int,
                 seq_q: int, seq_k: int, batch: int, cfg: FAConfig) -> str:
    """Persist one measured winner for this (identity, shape)."""
    ident = identity(device, dtype, hq, hkv, d, 0)
    path = cache_path(ident)
    table = TuningTable.load(path, ident)
    table.records[workload_key(seq_q, seq_k, batch)] = cfg
    table.save(path)
    return str(path)


def run_sweep(reps: int, preload_s: float, seq_k: int, geometries, write: bool,
              min_gain: float) -> None:
    print(f"sweep mode, seq_k={seq_k}, reps={reps}, preload {preload_s:.0f}s "
          "per shape; cold windows discarded and retried"
          + (f"; recording winners above {min_gain:.1f}%" if write else ""))
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    recorded = 0
    discarded = 0
    rows_out = []
    for label, hq, hkv, d in geometries:
        gqa = hq // hkv
        print(f"\n{label}: Hq={hq} Hkv={hkv} D={d} gqa={gqa}")
        print("  B  Sq  B*Sq  rule    y=8     y=16    y=32   best  delta tries")
        for sq in SWEEP_SQ:
            for batch in SWEEP_BATCHES:
                base = static_config(sq, gqa, batch=batch)
                times, report, attempt = measure_gated(
                    batch, hq, hkv, d, seq_k, sq, base.nc2,
                    KVSPLIT_CANDIDATES, reps, preload_s,
                )
                if times is None:
                    discarded += 1
                    print(f"  {batch:2d} {sq:3d} {batch * sq:5d}  DISCARDED "
                          f"({report})")
                    continue
                best = min(times, key=lambda k: times[k])
                rule_ms = times[base.kvsplit]
                delta = 100.0 * (rule_ms - times[best]) / rule_ms
                flag = "" if best == base.kvsplit else " <-- rule off"
                rows_out.append((label, batch, sq, batch * sq, base.kvsplit,
                                 best, delta))
                if write and best != base.kvsplit and delta >= min_gain:
                    write_record(device, "float16", hq, hkv, d, sq, seq_k,
                                 batch, FAConfig(nc2=base.nc2, kvsplit=best,
                                                 clip=base.clip))
                    recorded += 1
                print(f"  {batch:2d} {sq:3d} {batch * sq:5d} y{base.kvsplit:<3d} "
                      f"{times[8]:8.3f} {times[16]:8.3f} {times[32]:8.3f}  "
                      f"y{best:<3d} {delta:+5.1f}%{flag}  {attempt}")
    print(f"\ndiscarded windows: {discarded}")
    if write:
        print(f"records written: {recorded}")
    if rows_out:
        print("\nsummary by batch*Sq (mean delta of the rule vs the measured "
              "best, and how often the rule was already best):")
        for product in sorted({r[3] for r in rows_out}):
            rows = [r for r in rows_out if r[3] == product]
            mean_delta = statistics.mean(r[6] for r in rows)
            hits = sum(1 for r in rows if r[4] == r[5])
            print(f"  B*Sq={product:3d}  mean gap {mean_delta:+5.1f}%  "
                  f"rule optimal in {hits}/{len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--preload-seconds", type=float, default=30.0,
                    help="preload for the first window; per-shape preloads use "
                         "a tenth of this (minimum 3 s)")
    ap.add_argument("--seq-k", type=int, default=SEQ_K)
    ap.add_argument("--write-records", action="store_true",
                    help="persist each measured winner that beats the static "
                         "rule by --min-gain")
    ap.add_argument("--min-gain", type=float, default=1.0)
    ap.add_argument("--only", default="",
                    help="comma-separated geometry labels, e.g. '27B TP1'")
    ap.add_argument("--sweep", action="store_true",
                    help="run the (batch, seq_q) x kvsplit sweep")
    args = ap.parse_args()

    per_shape_preload = max(3.0, args.preload_seconds / 10.0)
    with MclkSampler() as sampler:
        ramp_mclk(args.preload_seconds)
    print(f"initial preload {args.preload_seconds:.0f}s -> {sampler.report()}\n")
    if args.sweep:
        chosen = [g for g in SWEEP_GEOMETRIES
                  if not args.only or g[0] in
                  {x.strip() for x in args.only.split(",") if x.strip()}]
        run_sweep(args.reps, per_shape_preload, args.seq_k, chosen,
                  args.write_records, args.min_gain)
    else:
        run_table(args.reps, per_shape_preload)


if __name__ == "__main__":
    main()
