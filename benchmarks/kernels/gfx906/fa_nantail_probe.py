#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""NaN-tail gate (plan_masked_fa.md step 2).

Question: does the FA compute kernel (gfx906_fa.forward) ever read K_q8 or
V rows AT OR BEYOND kv_max (the live seq_len)? If not, the gather may stop
writing V zeros beyond seq_len — the persistent kernel's correctness
precondition.

Method: run forward with zero tails, then re-run with the K_q8 and V tails
[seq_len, Sk) poisoned with NaN. If the kernel reads any poisoned row, the
output becomes NaN (softmax weight 0 x NaN = NaN; Q . NaN = NaN).
Gate PASS = outputs bit-equal across the poison.

K_q8 rows are 34 bytes per 32-value block: [scale fp16 (2B) | qs i8 (32B)].
Poison = scale NaN (0x7E00 LE), qs 0. V poison = fp16 NaN (0x7E00).

Cases: seq_len aligned / misaligned to the 32-row K tile (tail-tile
oob_check path), seq_len 1 row before Sk (worst-case tail), seq_len == Sk
(no tail), B=1 and B=2 ragged. Hkv in {2,4} exercises the GQA-packed
(ncols2 > 1) decode path at Hq=24.

Run (local venv, GPU 0):
  cd /local/git/vllm-gfx906-mobydick
  source ~/env-rocm-7.14-gfx906.sh
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -u \
      benchmarks/kernels/gfx906/fa_nantail_probe.py
"""
import torch

dev = "cuda"
torch.manual_seed(0)

Hq = 24
D = 256
BPR = (D // 32) * 34  # 272 bytes per q8_0 row (D=256)
SCALE = 0.04419417382415922  # 1/sqrt(D)

K_ZERO = torch.zeros(BPR, dtype=torch.uint8, device=dev)
K_NAN = torch.zeros(BPR, dtype=torch.uint8, device=dev)
K_NAN[1] = 0x7E  # fp16 NaN scale, little-endian byte 1


def run_case(fa, hkv, b, sls, sk):
    kb = torch.randn(b, hkv, sk, D, dtype=torch.float16, device=dev)
    vb = torch.randn(b, hkv, sk, D, dtype=torch.float16, device=dev)
    kq = fa.quantize_q8_0(kb)
    q = torch.randn(b, Hq, 1, D, dtype=torch.float32, device=dev)

    for i, sl in enumerate(sls):
        if sl < sk:
            kq[i, :, sl:sk, :] = K_ZERO
            vb[i, :, sl:sk, :] = 0
    kv_max = torch.tensor(sls, dtype=torch.int32, device=dev)

    o_clean = fa.forward(q, kq, vb, SCALE, kv_max)

    for i, sl in enumerate(sls):
        if sl < sk:
            kq[i, :, sl:sk, :] = K_NAN
            vb[i, :, sl:sk, :] = 0x7E00  # fp16 NaN
    o_poison = fa.forward(q, kq, vb, SCALE, kv_max)

    ok = torch.equal(o_clean, o_poison)
    finite = bool(torch.isfinite(o_poison).all())
    return ok, finite, o_clean


def main():
    from vllm import _gfx906_fa_C as fa

    cases = [
        # (hkv, seq_lens, Sk_pad)
        (2, [2176], 2560),         # aligned to 32 (no tail tile)
        (2, [2177], 2560),         # tail tile oob path
        (2, [2559], 2560),         # 1 row before Sk (worst-case tail)
        (2, [2560], 2560),         # full (no tail)
        (2, [32], 2560),           # tiny
        (4, [2177], 2560),         # Hkv=4 packed path
        (2, [1024, 2177], 2560),   # B=2 ragged
    ]
    fails = 0
    for hkv, sls, sk in cases:
        ok, finite, o = run_case(fa, hkv, len(sls), sls, sk)
        print(f"hkv={hkv} B={len(sls)} sls={sls} sk={sk}: "
              f"{'PASS' if ok and finite else 'FAIL'} "
              f"(bit_equal={ok}, finite={finite})")
        if not (ok and finite):
            fails += 1
            print(f"  clean[0,:4]   = {o[0, 0, 0, :4].tolist()}")
        del o
        torch.cuda.empty_cache()
    print("RESULT:", "FAIL" if fails else
          "PASS (no tail reads at/beyond kv_max)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
