#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""M=1 top-k softmax router: dedicated gfx906 kernel vs generic topkGating.

S2 of the 2026-08-18 sprint (docs/gfx906/DEVLOG-moe-m1-sprint.md).
Measures per-call us for the decode shape (M=1, E=256, k=8, fp16) both
ways and checks bit-equality on random + tie-heavy inputs.
"""
import torch

from vllm import _custom_ops as ops

dev = "cuda"
N_ITERS = 200
WARMUP = 20


def time_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(N_ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / N_ITERS


def alloc():
    w = torch.empty(1, 8, dtype=torch.float32, device=dev)
    i = torch.empty(1, 8, dtype=torch.int32, device=dev)
    tei = torch.empty(1, 8, dtype=torch.int32, device=dev)
    return w, i, tei


def main():
    gating = (torch.randn(1, 256, device=dev) * 4).half()

    w, i, tei = alloc()
    t_fast = time_us(lambda: ops.moe_topk_softmax_m1_gfx906(
        w, i, tei, gating, True))

    w2, i2, tei2 = alloc()
    t_gen = time_us(lambda: ops.topk_softmax(w2, i2, tei2, gating, True))

    print(f"generic topkGating : {t_gen:7.2f} us/call")
    print(f"gfx906 m1 topk     : {t_fast:7.2f} us/call  "
          f"({100 * (t_gen - t_fast) / t_gen:.0f}% faster)")
    print(f"per step (x40)     : {40 * t_gen:7.1f} -> {40 * t_fast:6.1f} us")

    # bit-equal check across a spread of inputs
    cases = [gating]
    for seed in range(4):
        cases.append((torch.randn(1, 256, device=dev,
                                  generator=torch.Generator(device=dev)
                                  .manual_seed(seed)) * 2).half())
    cases.append(torch.full((1, 256), 0.5, device=dev, dtype=torch.half))
    ok = True
    for g in cases:
        w, i, tei = alloc()
        ops.moe_topk_softmax_m1_gfx906(w, i, tei, g, True)
        w2, i2, tei2 = alloc()
        ops.topk_softmax(w2, i2, tei2, g, True)
        if not (torch.equal(w, w2) and torch.equal(i, i2)
                and torch.equal(tei, tei2)):
            ok = False
            print(f"DIVERGENCE on {g[0, :8].tolist()}: "
                  f"{w.tolist()} vs {w2.tolist()}")
    print("bit-equal:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
