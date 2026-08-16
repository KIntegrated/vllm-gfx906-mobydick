// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) Nick — nick413@gmail.com
//
// Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
// (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
//
// q8_0_quantize.cuh — Q8_0 (block-32) quantization helpers shared by
// gfx906_fa_quant.cu (standalone quantize) and gfx906_fa_gather.cu
// (fused gather-and-quantize, P3-3a stage 2).
//
// Numerics contract: block of 32 fp16 values -> 34 bytes (fp16 scale +
// 32 int8). amax via shfl_xor tree (fixed offset order 16,8,4,2,1),
// d = amax/127, id = d>0 ? 1/d : 0, qi = clamp(rintf(v*id), -128, 127).
// Bit-exact with the original quantize_q8_0_dense_kernel — the fused
// kernel reuses this helper unchanged, so gather+quantize is bit-equal
// to quantize(gather(x)).

#pragma once

#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF2_OPERATORS__
#undef __HIP_NO_HALF2_OPERATORS__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

static constexpr int QK8_0_SZ = 32;
static constexpr int Q8_0_BYTES = 34;     // sizeof(__half) + 32 int8

// Quantize one 32-value block. One halfwave (32 lanes) per block: lane
// `lane_in` holds value `lane_in`; amax is reduced inside the halfwave
// with __shfl_xor width=32 (a 64-lane wavefront runs two blocks in
// parallel, independently).
static __device__ __forceinline__ void quantize_block_q8_0_halfwarp(
    const __half * __restrict__ x,   // 32 values (one block)
    uint8_t      * __restrict__ y,   // 34 bytes (fp16 scale + 32 int8)
    int lane_in_block                // 0..31
) {
    const float v = __half2float(x[lane_in_block]);
    const float absv = fabsf(v);

    float amax = absv;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o = __shfl_xor(amax, offset, 32);
        amax = fmaxf(amax, o);
    }

    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    float q = v * id;
    int   qi = (int)rintf(q);
    if (qi < -128) qi = -128;
    if (qi >  127) qi =  127;

    if (lane_in_block == 0) {
        __half d_h = __float2half(d);
        // Unaligned-safe scale store (34-byte blocks: dst+34 is not
        // 2-aligned).
        uint16_t d_bits = *reinterpret_cast<uint16_t*>(&d_h);
        y[0] = d_bits & 0xff;
        y[1] = (d_bits >> 8) & 0xff;
    }
    y[2 + lane_in_block] = (uint8_t)(int8_t)qi;
}
