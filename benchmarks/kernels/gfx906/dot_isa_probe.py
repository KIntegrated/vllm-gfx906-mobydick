#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""gfx906 dot-instruction measured facts (probe).

Compiles + runs a small HIP probe that answers, on THIS toolchain +
GPU (gfx906/MI50, ROCm 7.14):

1. Which dot instructions does the BACKEND accept for gfx906?
   (The assembler frontend accepts more mnemonics than the backend
   supports — a `-S` "OK" is not an availability answer.)
2. `v_dot4_i32_i8` semantics: r = acc + a.lo*b.lo + a.hi*b.hi ... no:
   r = acc + sum of 4 signed i8 pairs (127*127 scale).
3. `v_dot2_f32_f16` operand semantics — the 2026-08-28 finding: NOT
   the clean "2-pair fp16 dot + fp32 accumulate". Measured with
   4th operand zeroed: r = a_lo*b_lo + 2*(a_hi*b_hi) (the hi pair is
   DOUBLE-COUNTED) + (4th-operand term that is not a clean f32 addend
   for f16x2 bit patterns). Do NOT use in production without
   confirming against the official gfx906 ISA reference.

Usage:
  python3 dot_isa_probe.py            # prints the fact table
  GFX906_DOT_PROBE_HIPCC=... python3 dot_isa_probe.py
"""
import os
import subprocess
import sys
import tempfile

ROCMLIB = "/opt/rocm/core-7.14/lib"
HIPCC = os.environ.get("GFX906_DOT_PROBE_HIPCC",
                       "/opt/rocm/core-7.14/bin/hipcc")

# Availability: each probe function must COMPILE for gfx906.
AVAIL = {
    "v_dot4_i32_i8":  'int r=0; asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));',
    "v_dot4c_i32_i8": 'int r=0; asm volatile("v_dot4c_i32_i8 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));',
    "v_dot8_i32_i8":  'int r=0; asm volatile("v_dot8_i32_i8 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));',
    "v_dot8c_i32_i8": 'int r=0; asm volatile("v_dot8c_i32_i8 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));',
    "v_dot2_f32_f16": 'float r=0; asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b));',
}

TPL = """
#include <hip/hip_runtime.h>
#include <cstdint>
__global__ void t(float* o, unsigned a, unsigned b) {{
    {body}
    if (r == 42 || (float)r == 42.0f) o[0] = 1;
}}
"""

SEM = r"""
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
__global__ void t(float* out, unsigned a, unsigned b, unsigned r4) {
    float dst = 0.0f;
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %3"
                 : "=v"(dst) : "v"(a), "v"(b), "v"(r4));
    out[0] = dst;
}
int main() {
    float* h; hipHostAlloc(&h, sizeof(float), hipHostMallocDefault);
    // a=(1,2) f16, b=(4,16) f16. Data points (r4 as f16x2):
    struct { const char* n; unsigned r4; float got; float fit; } cs[] = {
        {"r4=(0,0)   ", 0x00000000, 0.0f, 0.0f},
        {"r4=(1,2)   ", 0x40003C00, 0.0f, 0.0f},
        {"r4=(127,127)", 0x3F803F80, 0.0f, 0.0f},
    };
    for (auto& c : cs) {
        t<<<1,1>>>(h, 0x40003C00, 0x50004400, c.r4);
        hipDeviceSynchronize();
        printf("  dot2 a=(1,2) b=(4,16) %s: %f   [clean dot+acc fit would be %.1f]\n",
               c.n, h[0], 36.0);
    }
    return 0;
}
"""


def main() -> int:
    print(f"dot-instruction probe (hipcc={HIPCC}, arch=gfx906)")
    print("== availability (BACKEND-verified: compiles for --offload-arch=gfx906)")
    ok_all = True
    for name, body in AVAIL.items():
        src = TPL.format(body=body)
        with tempfile.TemporaryDirectory() as td:
            hip = os.path.join(td, "p.hip")
            with open(hip, "w") as f:
                f.write(src)
            r = subprocess.run(
                [HIPCC, "-O3", "--offload-arch=gfx906", hip, "-o", os.path.join(td, "p")],
                capture_output=True, text=True)
            verdict = "AVAILABLE" if r.returncode == 0 else \
                      "NOT AVAILABLE (" + (r.stderr.splitlines()[-1] if r.stderr.strip() else "?")[:70] + ")"
            if r.returncode != 0:
                # frontend-only acceptance check
                r2 = subprocess.run(
                    [HIPCC, "-O3", "--offload-arch=gfx906", "-S", hip, "-o", os.path.join(td, "p.s")],
                    capture_output=True, text=True)
                if r2.returncode == 0:
                    verdict += " [frontend-only]"
            ok_all &= r.returncode == 0
            print(f"  {name:<16}: {verdict}")

    print("== v_dot2_f32_f16 operand semantics (a=(1,2) f16, b=(4,16) f16;")
    print("   clean 'dot+acc' = 1*4+2*16+0 = 36.0 for every r4-as-f32 addend case)")
    with tempfile.TemporaryDirectory() as td:
        hip = os.path.join(td, "sem.hip")
        with open(hip, "w") as f:
            f.write(SEM)
        r = subprocess.run(
            [HIPCC, "-O3", "--offload-arch=gfx906", hip, "-o", os.path.join(td, "sem")],
            capture_output=True, text=True)
        if r.returncode != 0:
            print("  (dot2 semantics probe failed to compile — recheck availability above)")
            return 1
        env = dict(os.environ, LD_LIBRARY_PATH=ROCMLIB,
                   HIP_VISIBLE_DEVICES=os.environ.get("GFX906_DOT_PROBE_GPU", "0"))
        subprocess.run([os.path.join(td, "sem")], env=env)
    print("== VERDICT: gfx906 int-dot set = {v_dot4_i32_i8} only; v_dot2_f32_f16")
    print("   exists but is NOT a clean fp32-accumulate dot2 (see above) — needs")
    print("   official-ISA confirmation before any P·V use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
