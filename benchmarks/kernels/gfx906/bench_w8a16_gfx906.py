# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""NH-2 — int8 in-kernel W8A16 (channel) GEMV/GEMM at Nemotron's dense shapes.

Probe (launch-regime evidence) for the NH-2 item in
docs/gfx906/ROADMAP.md: the CT W8A16 channel-dequant scheme currently
dequantizes the packed int8 to fp16 at load and serves it through the
unquantized GEMV family (LLMM1 rpb=4 at Nemotron's K=2688/4096). This
measures the new int8 in-kernel path (Triton GEMV M=1 / GEMM M>1,
dequant in-register, bias-128 convention) against the *actual*
production dispatch (rocm_unquantized_gemm_impl) at the exact
Nemotron-3.5-Lightning-30B dense-layer shapes:

  mamba in_proj  [10304, 2688] x23/step
  mamba out_proj [ 2688, 4096] x23/step
  GQA  o_proj    [ 2688, 4096] x 6/step
  GQA  q_proj    [ 4096, 2688] x 6/step
  GQA  k/v_proj  [  256, 2688] x12/step
  lm_head        [131072, 2688] x 1/step

The shipping gate is the serving A/B + PPL (DEVLOG-nemotron-h.md); this
probe is the kernel-level gate: if int8 loses at any shape it must show
up here before any serving time is spent.

Run (GPU idle, single card):
    source ~/env-rocm-7.14-gfx906.sh
    HIP_VISIBLE_DEVICES=0 .venv/bin/python \
        benchmarks/kernels/gfx906/bench_w8a16_gfx906.py
"""

import os

import torch
import triton

try:
    from triton.testing import do_bench
except ImportError:  # newer triton
    from triton import do_bench  # type: ignore

FLOOR_GBS = float(os.environ.get("W8A16_PROBE_BW", "798"))

# (label, N, K, calls per decode step)
SHAPES = [
    ("mamba in_proj", 10304, 2688, 23),
    ("mamba out_proj", 2688, 4096, 23),
    ("GQA o_proj", 2688, 4096, 6),
    ("GQA q_proj", 4096, 2688, 6),
    ("GQA k/v_proj", 256, 2688, 12),
    ("lm_head", 131072, 2688, 1),
]


def _log(*a):
    print(*a, flush=True)


def _make_weight(n: int, k: int, dev: str):
    """Random bias-128 int8 with ~10 % of bytes pinned to the 0x80
    (zero-weight) code, plus fp16 channel scales."""
    torch.manual_seed(0)
    words = torch.randint(
        0, 2**32, (n, k // 4), device=dev, dtype=torch.int64
    )
    w_u8 = words.to(torch.uint8).view(n, k)
    mask = torch.rand(n, k, device=dev) < 0.10
    w_u8 = torch.where(mask, torch.full_like(w_u8, 0x80), w_u8)
    w_u8 = w_u8.contiguous()
    scale = (torch.rand(n, 1, device=dev) * 0.05 + 0.005).half()
    return w_u8, scale


def _dequant_f16(w_u8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # bit-identical to the scheme's load-time dequant
    return ((w_u8.to(torch.float32) - 128.0) * scale.to(torch.float32)).half()


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / b.float().norm().item()


def main() -> int:
    if not torch.cuda.is_available():
        _log("NO GPU visible (set HIP_VISIBLE_DEVICES)")
        return 2
    dev = torch.cuda.current_device()
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a16_channel_dequant as w8,
    )
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    _log(
        f"# NH-2 W8A16 int8 in-kernel probe — {torch.cuda.get_device_name(dev)}, "
        f"torch {torch.__version__}, triton {triton.__version__}, "
        f"visible-dev={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}"
    )
    _log(f"  HBM floor reference: {FLOOR_GBS:.0f} GB/s")
    quick = os.environ.get("W8A16_PROBE_QUICK") == "1"
    shapes = SHAPES[:3] if quick else SHAPES

    m4 = os.environ.get("W8A16_PROBE_SKIP_M") != "1"
    hdr = (
        f"  {'shape':15s} {'N':>7s} {'K':>5s} {'x/step':>6s} | "
        f"{'M=1 cur':>9s} {'M=1 i8':>9s} {'i8/f16':>7s} {'i8 %floor':>9s} | "
        f"{'M=4 cur':>9s} {'M=4 i8':>9s} | {'M=4096 cur':>11s} {'M=4096 i8':>10s}"
    )
    _log("\n## per-shape times (us) — cur = rocm_unquantized_gemm_impl on the")
    _log("## dequanted fp16 weight (the actual production dispatch today)")
    _log(hdr)

    step_cur = step_i8 = 0.0
    bytes_i8 = 0.0
    errs = []
    for name, n, k, cd in shapes:
        w_u8, scale = _make_weight(n, k, dev)
        wf = _dequant_f16(w_u8, scale)
        x1 = (torch.randn(1, k, device=dev) * 0.5).half()
        t = {}
        try:
            t["c1"] = do_bench(lambda: rocm_unquantized_gemm_impl(x1, wf), rep=30) * 1e3
        except Exception as e:
            _log(f"  {name}: cur M=1 failed {type(e).__name__}: {e}")
            t["c1"] = float("nan")
        try:
            t["i1"] = do_bench(lambda: w8.w8a16_gemv(w_u8, scale, x1[0]), rep=30) * 1e3
        except Exception as e:
            _log(f"  {name}: i8 M=1 failed {type(e).__name__}: {e}")
            t["i1"] = float("nan")
        if m4:
            x4 = (torch.randn(4, k, device=dev) * 0.5).half()
            xbig = (torch.randn(4096, k, device=dev) * 0.5).half()
            try:
                t["c4"] = (
                    do_bench(lambda: rocm_unquantized_gemm_impl(x4, wf), rep=30) * 1e3
                )
            except Exception as e:
                _log(f"  {name}: cur M=4 failed {type(e).__name__}: {e}")
                t["c4"] = float("nan")
            try:
                t["i4"] = do_bench(lambda: w8.w8a16_gemm(w_u8, scale, x4), rep=30) * 1e3
            except Exception as e:
                _log(f"  {name}: i8 M=4 failed {type(e).__name__}: {e}")
                t["i4"] = float("nan")
            try:
                t["cb"] = (
                    do_bench(
                        lambda: rocm_unquantized_gemm_impl(xbig, wf), rep=20
                    )
                    * 1e3
                )
            except Exception as e:
                _log(f"  {name}: cur M=4096 failed {type(e).__name__}: {e}")
                t["cb"] = float("nan")
            try:
                t["ib"] = (
                    do_bench(lambda: w8.w8a16_gemm(w_u8, scale, xbig), rep=20) * 1e3
                )
            except Exception as e:
                _log(f"  {name}: i8 M=4096 failed {type(e).__name__}: {e}")
                t["ib"] = float("nan")
        # correctness
        ref1 = ((w_u8.to(torch.float32) - 128.0) * scale.to(torch.float32)) @ x1[
            0, :
        ].to(torch.float32)
        got1 = w8.w8a16_gemv(w_u8, scale, x1[0]).float()
        e1 = _relerr(got1, ref1)
        errs.append((name, "M=1", e1))
        if m4:
            ref4 = (
                ((w_u8.to(torch.float32) - 128.0) * scale.to(torch.float32))
                @ x4.to(torch.float32)
            )
            e4 = _relerr(w8.w8a16_gemm(w_u8, scale, x4).float(), ref4)
            errs.append((name, "M=4", e4))
            ebig = _relerr(
                w8.w8a16_gemm(w_u8, scale, xbig).float(),
                (
                    ((w_u8.to(torch.float32) - 128.0) * scale.to(torch.float32))
                    @ xbig.to(torch.float32)
                ),
            )
            errs.append((name, "M=4096", ebig))
        floor_i8 = n * k * 1.0 / 1e9 / FLOOR_GBS * 1e6
        step_cur += cd * t["c1"]
        step_i8 += cd * t["i1"]
        bytes_i8 += cd * n * k
        _log(
            f"  {name:15s} {n:7d} {k:5d} {cd:6d} | "
            f"{t['c1']:9.1f} {t['i1']:9.1f} {t['i1'] / t['c1']:7.2f}x "
            f"{t['i1'] / floor_i8 * 100:8.0f}% | "
            + (
                f"{t['c4']:9.1f} {t['i4']:9.1f} | {t['cb']:11.1f} {t['ib']:10.1f}"
                if m4
                else "            -            |              -           -"
            )
        )
        del w_u8, wf, x1
        torch.cuda.empty_cache()

    _log("\n## correctness (rel. err vs fp32 reference; 0x80 zero-codes pinned)")
    worst = 0.0
    for name, m, e in errs:
        flag = "  OK" if e <= 5e-3 else "  ** FAIL **"
        _log(f"  {name:15s} {m:6s} {e:.3e}{flag}")
        worst = max(worst, e)

    _log(
        f"\n## decode step total (M=1): cur {step_cur:7.1f} us -> "
        f"i8 {step_i8:7.1f} us ({step_cur / max(step_i8, 1e-9):.2f}x, "
        f"{step_cur - step_i8:6.1f} us saved; int8 bytes/step "
        f"{bytes_i8 / 1e9:.2f} GB, floor {bytes_i8 / 1e9 / FLOOR_GBS * 1e6:6.1f} us)"
    )
    if worst > 5e-3:
        _log("\nVERDICT: CORRECTNESS FAIL — do not ship")
        return 1
    _log("\nVERDICT: see table; shipping gate is the serving A/B + PPL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
