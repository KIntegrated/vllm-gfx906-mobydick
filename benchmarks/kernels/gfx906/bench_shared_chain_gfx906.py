#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""C5 gate probe: what is the shared-expert chain worth in GRAPHED serving?

The chain per MoE layer (Qwen3.5/3.8-35B-A3B, M=1 decode):

    gate_up, _ = shared_expert.gate_up_proj(x)   # MergedColumnParallelLinear
    act        = SiluAndMul()(gate_up)           # torch.ops._C.silu_and_mul
    out, _     = shared_expert.down_proj(act)    # RowParallelLinear

C5 proposes one fused chain kernel to "remove two launches per layer" (40
layers/step). Two things must hold for that to pay, and this probe measures
both, at the real per-rank shapes:

  1. **launches must still cost something.** Serving decode is graph-captured
     (C3's fold removed "one memset graph node per MoE layer"), so the chain is
     timed EAGER (launch-inclusive) and as a 40-copy CUDA GRAPH (launch-free).
     The difference is the launch share a fused kernel could remove — but only
     outside graphs.
  2. **the kernels must be away from the HBM floor.** A fused kernel cannot beat
     its own bandwidth floor, so (in-graph chain time − floor) bounds any
     kernel-side win, fusion or not.

TP shapes are measured separately because they change WHICH kernels run:
Qwen3MoeMLP does not set `disable_tp`, so at TP=2 gate_up is N-sharded
([512,2048]/rank) and down is K-sharded ([2048,256]/rank) — and both
`_llmm1_tiny_m` GEMV dispatch rules (shape[1]==2048 & m in {1,256,>=2048};
shape[1]==512 & m==2048) MISS those shapes. So production TP=2 runs LLMM1 for
both projections; the shipped S3 down-GEMV only fires at TP=1. (dense_gemv
cannot run at K=256 at all: kchunk must be 512/1024/2048/4096.)

Method notes: eager medians are block-of-10 CUDA-event medians; graph numbers
capture N sequential copies in one graph (same-stream capture preserves stream
order as dependencies, so copies do not overlap) and replay them.

mclk (docs/gfx906/dvfs-mi50.md): a memory-bound kernel cannot hold mclk up, so
a light duty-cycled compute heater keeps clocks at 800+ MHz; numbers are
labelled, and mclk is read from sysfs (`pp_dpm_mclk`) and gated — see the
Environment notes in DEVLOG-moe-c5-chain-fusion.md for why not `rocm-smi`.

Run: `PYTHONPATH=. .venv/bin/python -u
benchmarks/kernels/gfx906/bench_shared_chain_gfx906.py`
Env: LAYERS (40), GRAPH_REPS (20), HEATER (1), HEATER_SIZE (512),
     HEATER_GAP (0.0005 s), RPT (4)
"""
import glob
import os
import re
import statistics
import threading
import time

import torch

dev = "cuda"
torch.manual_seed(0)

HBM_BW = 822e9  # gfx906 in-context measured read BW (SYV-3 closed rung)

# (N, K, kernel production uses at that TP, tag)
SHAPES = {
    "tp1 (full, single GPU)": [
        (1024, 2048, "LLMM1", "gate_up"),
        (2048, 512, "gemv", "down"),
    ],
    "tp2 (per-rank, production)": [
        (512, 2048, "LLMM1", "gate_up"),
        (2048, 256, "LLMM1", "down"),  # gemv impossible at K=256
    ],
}


class MclkSampler:
    """Sample mclk (MHz) from sysfs for every DRM card — no subprocess.

    `rocm-smi --showclocks` BLOCKS uninterruptibly while the GPU is saturated by
    a graph-replay loop (its timeout does not fire: the child sits in D state),
    which silently produced "no samples". pp_dpm_mclk is a plain file read and
    names the active DVFS level with a `*`. All cards are sampled and reported;
    the one that ramps is the device under test.
    """

    GLOB = "/sys/class/drm/card*/device/pp_dpm_mclk"

    def __init__(self, period=0.2):
        self.period, self._stop = period, threading.Event()
        self.samples = {}
        self.fails = 0
        self._paths = sorted(glob.glob(self.GLOB))
        self._t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read_one(path):
        try:
            with open(path) as fh:
                for line in fh:
                    if "*" in line:
                        m = re.search(r"\(?(\d+)\s*Mhz", line)
                        if m:
                            return int(m.group(1))
        except Exception:
            return None
        return None

    def read_all(self):
        out = {}
        for p in self._paths:
            v = self._read_one(p)
            card = p.split("/")[4]
            if v is None:
                self.fails += 1
            else:
                out[card] = v
                self.samples.setdefault(card, []).append(v)
        return out

    def _run(self):
        while not self._stop.is_set():
            self.read_all()
            self._stop.wait(self.period)

    def __enter__(self):
        self.read_all()
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)
        return None  # never swallow exceptions

    def report(self):
        if not any(self.samples.values()):
            return f"mclk: NO SAMPLES (fails={self.fails}) — INVALID RUN"
        parts = []
        for card, vals in sorted(self.samples.items()):
            s = sorted(vals)
            p90 = s[int(0.9 * (len(s) - 1))]
            hot = "HOT" if p90 >= 800 else "cold"
            parts.append(f"{card} n={len(s)} med/p90/max="
                         f"{statistics.median(s):.0f}/{p90}/{s[-1]} {hot}")
        return ("mclk " + " | ".join(parts)
                + f" (fails={self.fails}; the ramped card is the device under test)")


class Heater:
    """Small duty-cycled matmul on its own stream: holds DVFS at 1000 MHz
    without occupying most CUs (a big continuous matmul inflates the timed
    kernels — a different error from the cold-clock one it prevents)."""

    def __init__(self, n=512, gap=0.0005):
        self.on = os.environ.get("HEATER", "1") == "1"
        self.n = int(os.environ.get("HEATER_SIZE", n))
        self.gap = float(os.environ.get("HEATER_GAP", gap))
        self._stop = threading.Event()
        a = torch.randn(self.n, self.n, dtype=torch.float16, device=dev)
        self.a = self.b = a
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            while not self._stop.is_set():
                torch.mm(self.a, self.b)
                time.sleep(self.gap)

    def __enter__(self):
        if self.on:
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.on:
            self._t.join(timeout=5)
            torch.cuda.synchronize()
        return None

    def label(self):
        if not self.on:
            return "heater=off (cold-clock risk)"
        return f"heater={self.n}x{self.n} gap={self.gap*1e3:.1f}ms (co-scheduled)"


def per_call_us(fn, warmup=20, iters=100):
    """Median of block-of-10 CUDA-event timings — EAGER: includes launch cost."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    blocks = []
    for _ in range(max(iters // 10, 1)):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            fn()
        e.record()
        torch.cuda.synchronize()
        blocks.append(s.elapsed_time(e) * 100.0)  # us/call
    blocks.sort()
    return statistics.median(blocks)


def graph_per_call_us(fn, copies, reps=20):
    """Capture `copies` sequential invocations in one CUDA graph, replay it.

    Same-stream capture records stream order as graph dependencies, so the
    copies do not overlap: this is launch-free per-op GPU time.
    """
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(copies):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / reps / copies  # us/call, launch-free


def main():
    from vllm import _custom_ops as ops

    layers = int(os.environ.get("LAYERS", "40"))
    reps = int(os.environ.get("GRAPH_REPS", "20"))
    rpt = int(os.environ.get("RPT", "4"))
    copies = max(layers, 40)

    print(f"device={torch.cuda.get_device_name(0)}  HBM_BW={HBM_BW/1e9:.0f} GB/s"
          f"  layers={layers}  graph_reps={reps}")

    with Heater() as heater, MclkSampler() as mclk:
        print(f"{heater.label()}\n")
        for family, shapes in SHAPES.items():
            print(f"=== {family} ===")
            blobs = {}
            for n, k, kern, tag in shapes:
                blobs[tag] = (torch.randn(n, k, dtype=torch.float16, device=dev),
                              torch.randn(1, k, dtype=torch.float16, device=dev),
                              n, k, kern)
            ngu, kgu = blobs["gate_up"][2], blobs["gate_up"][3]
            ndn, kdn = blobs["down"][2], blobs["down"][3]
            d = ngu // 2
            act_in = torch.empty(1, 2 * d, dtype=torch.float16, device=dev)
            act_out = torch.empty(1, d, dtype=torch.float16, device=dev)
            gu_w, gu_x = blobs["gate_up"][0], blobs["gate_up"][1]
            dn_w, dn_kern = blobs["down"][0], blobs["down"][4]

            def gate_fn():
                return ops.LLMM1(gu_w, gu_x, rpt)

            if dn_kern == "gemv":
                def down_fn():
                    return ops.dense_gemv_gfx906(dn_w, act_out, kdn)
                dn_name = f"dense_gemv kc{kdn}"
            else:
                def down_fn():
                    return ops.LLMM1(dn_w, act_out, rpt)
                dn_name = f"LLMM1 rpb{rpt}"

            def act_fn():
                torch.ops._C.silu_and_mul(act_in, act_out)

            def gate_act_fn():
                torch.ops._C.silu_and_mul(gate_fn(), act_out)

            def chain_fn():
                torch.ops._C.silu_and_mul(gate_fn(), act_out)
                return down_fn()

            assert gate_fn() is not None, "silent fallback — invalid run"

            e_gate = per_call_us(gate_fn)
            e_act = per_call_us(act_fn)
            e_down = per_call_us(down_fn)
            e_chain = per_call_us(chain_fn, iters=copies)

            g_gate = graph_per_call_us(gate_fn, copies, reps)
            g_gate_act = graph_per_call_us(gate_act_fn, copies, reps)
            g_chain = graph_per_call_us(chain_fn, copies, reps)
            g_act = g_gate_act - g_gate
            g_down = g_chain - g_gate_act

            f_gate = ngu * kgu * 2 / HBM_BW * 1e6
            f_down = ndn * kdn * 2 / HBM_BW * 1e6
            floor = f_gate + f_down
            print(f"  gate_up {ngu}x{kgu:<5} LLMM1 rpb{rpt}    eager {e_gate:6.2f}  "
                  f"graph {g_gate:6.2f} us   floor {f_gate:5.2f}")
            print(f"  act     1x{2*d:<7} silu_and_mul     eager {e_act:6.2f}  "
                  f"graph {g_act:6.2f} us")
            print(f"  down    {ndn}x{kdn:<5} {dn_name:<15} eager {e_down:6.2f}  "
                  f"graph {g_down:6.2f} us   floor {f_down:5.2f}")
            print(f"  chain   eager {e_chain:6.2f} us/layer   graph {g_chain:6.2f} "
                  f"us/layer   (graph parts sum {g_gate+g_act+g_down:6.2f})")
            print(f"  -> LAUNCH share a fused kernel could remove (eager only): "
                  f"{e_chain - g_chain:6.2f} us/layer = "
                  f"{(e_chain-g_chain)*layers:5.0f} us/step")
            print(f"  -> KERNEL headroom vs floor (graphed regime):            "
                  f"{g_chain - floor:6.2f} us/layer = "
                  f"{(g_chain-floor)*layers:5.0f} us/step")
            print(f"  -> per step ({layers} layers): eager {e_chain*layers:6.0f} us, "
                  f"graph {g_chain*layers:6.0f} us, floor {floor*layers:5.0f} us\n")

        print(f"=== {mclk.report()} ===")


if __name__ == "__main__":
    main()
