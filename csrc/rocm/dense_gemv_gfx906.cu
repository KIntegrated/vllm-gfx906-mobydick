// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// M=1 W16A16 dense GEMV for gfx906 (Vega 20, no MFMA). Phase 3 P3-2(b).
//
//   out[1, N] = x[1, K] @ W[N, K]^T      (all fp16, fp32 accumulation)
//
// Design (row-parallel, same structure as LLGemm1_kernel which saturates
// HBM on the big rows):
//   - Block covers RPT weight rows; threads split the K-chunk in 8-half
//     (16B) units: thread t handles k = k0 + 8*t .. k0 + 8*t + 7.
//   - Each thread loads one 16B slice of x (shared across its RPT rows) and
//     RPT 16B weight rows, dots via __ockl_fdot2 (v_dot2_f32_f16) — the
//     same 16B-load / fp32-dot pattern as moe_q_gemm_gfx906.cu.
//   - KCHUNK=512/1024/2048/4096 → 64/128/256/512 threads
//     (1/2/4/8 wavefronts; KCHUNK=4096 exceeds MI50's 256-thread
//     workgroup limit — bench-only path, never used by model dispatch).
//     KSPLIT = K / KCHUNK.
//   - KSPLIT==1: cross-warp reduce in LDS, single fp16 store (no atomics).
//   - KSPLIT>1:  grid.y spans the K-chunks; each block reduces its chunk in
//     fp32, converts to fp16, and atomic-adds (packed 32/64-bit CAS) into a
//     pre-zeroed output. Precision: one extra fp16 rounding per chunk
//     (reorder-class change, A/B-diffed at integration).
//
// Measured on gfx906 (MI50), DEVLOG "P3-2(b)": the winning configuration
// is single-pass KCHUNK=K with RPT=2 for K=2048 rows of N==256 or N>=2048
// (qkv -23%, router -17%, in_proj/LM head -6% vs LLMM1 rpb=4). The v1
// K-split hypothesis for the small rows is falsified: at M=1 the CAS +
// zero_ + tiny-block overhead makes splits 2.4-4.2x slower than LLMM1; the
// 3.6-14x-floor rows (GDN-small, shared down) are launch/latency-bound,
// not CU-occupancy-bound, and no GEMM kernel closes them. K-split is kept
// as a supported path (RPT>=2 only) but no model shape uses it.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace dense_gemv_gfx906 {

// Packed 2-half atomic add via one 32-bit CAS loop (RPT=2 epilogue).
__forceinline__ __device__ void atomic_add_pk2_f16(half* addr, half2 v01) {
  unsigned* addr_u = reinterpret_cast<unsigned*>(addr);
  unsigned old = *addr_u;
  while (true) {
    union {
      unsigned u;
      half2 h;
    } cur, sum;
    cur.u = old;
    sum.h = __hadd2(cur.h, v01);
    unsigned prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// Packed 4-half atomic add via one 64-bit CAS loop (RPT=4 epilogue).
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// 8-wide fp32 dot: 4 x v_dot2_f32_f16 over (w.h2[i], a.h2[i]).
__forceinline__ __device__ float dot8_f32(const half2 (&w)[4],
                                          const half2 (&a)[4]) {
  float r = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++)
    r = __ockl_fdot2(w[i], a[i], r, true);
  return r;
}

template <int RPT, int KCHUNK>
__global__ void __launch_bounds__(KCHUNK / 8)
    dense_gemv_kernel(const half* __restrict__ x,   // [K]
                      const half* __restrict__ w,   // [N, K] row-major
                      half* __restrict__ out,       // [N], pre-zeroed if KSPLIT>1
                      const int N, const int K,
                      const int ksplit) {
  static_assert(KCHUNK == 512 || KCHUNK == 1024 || KCHUNK == 2048 ||
                    KCHUNK == 4096,
                "KCHUNK must be 512, 1024, 2048 or 4096");
  static_assert(RPT == 1 || RPT == 2 || RPT == 4, "RPT must be 1, 2 or 4");
  constexpr int THREADS = KCHUNK / 8;
  constexpr int WARPS = THREADS / 64;
  const int t = threadIdx.x;
  const int row0 = blockIdx.x * RPT;
  const int k0 = blockIdx.y * KCHUNK;

  // 8-half (16B) slice of x, shared across this thread's RPT rows.
  union {
    uint4 u;
    half2 h2[4];
  } xa;
  xa.u = *(const uint4*)(x + k0 + t * 8);

  float acc[RPT];
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int row = row0 + r;
    if (row >= N) {
      acc[r] = 0.0f;
      continue;
    }
    union {
      uint4 u;
      half2 h2[4];
    } wa;
    wa.u = *(const uint4*)(w + (int64_t)row * K + k0 + t * 8);
    acc[r] = dot8_f32(wa.h2, xa.h2);
  }

  // Reduce the K-split within this block (THREADS/64 wavefronts).
  #pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2) {
    #pragma unroll
    for (int r = 0; r < RPT; ++r)
      acc[r] += __shfl_xor(acc[r], mask);
  }

  if constexpr (WARPS == 1) {
    // Single wavefront: lane r < RPT holds row r's full sum.
    if (t < RPT) {
      const int row = row0 + t;
      if (row >= N) return;
      if (ksplit == 1) {
        out[row] = __float2half_rn(acc[t]);
      } else if constexpr (RPT == 4) {
        // One 64-bit CAS per block covering rows row0..row0+3.
        if (t == 0) {
          half2 h01 = __halves2half2(__float2half_rn(acc[0]),
                                     __float2half_rn(acc[1]));
          half2 h23 = __halves2half2(__float2half_rn(acc[2]),
                                     __float2half_rn(acc[3]));
          atomic_add_pk4_f16(out + row0, h01, h23);
        }
      } else if constexpr (RPT == 2) {
        // One 32-bit CAS per block covering rows row0..row0+1.
        if (t == 0) {
          half2 h01 = __halves2half2(__float2half_rn(acc[0]),
                                     __float2half_rn(acc[1]));
          atomic_add_pk2_f16(out + row0, h01);
        }
      }
      // RPT==1 with ksplit>1 is rejected by the launcher.
    }
  } else {
    // Multiple wavefronts: exchange per-row partials through LDS.
    __shared__ float red_smem[RPT][8];  // WARPS <= 8 (KCHUNK <= 4096)
    const int warp = t / 64;
    const int lane = t % 64;
    if (lane < RPT) red_smem[lane][warp] = acc[lane];
    __syncthreads();
    if (warp == 0 && lane < RPT) {
      const int row = row0 + lane;
      if (row >= N) return;
      float s = 0.0f;
      #pragma unroll
      for (int wp = 0; wp < WARPS; ++wp) s += red_smem[lane][wp];
      if (ksplit == 1) {
        out[row] = __float2half_rn(s);
      } else if constexpr (RPT == 4) {
        // One 64-bit CAS per block covering rows row0..row0+3.
        if (lane == 0) {
          float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
          #pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) {
            s1 += red_smem[1][wp];
            s2 += red_smem[2][wp];
            s3 += red_smem[3][wp];
          }
          half2 h01 = __halves2half2(__float2half_rn(s),
                                     __float2half_rn(s1));
          half2 h23 = __halves2half2(__float2half_rn(s2),
                                     __float2half_rn(s3));
          atomic_add_pk4_f16(out + row0, h01, h23);
        }
      } else if constexpr (RPT == 2) {
        // One 32-bit CAS per block covering rows row0..row0+1.
        if (lane == 0) {
          float s1 = 0.0f;
          #pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) s1 += red_smem[1][wp];
          half2 h01 = __halves2half2(__float2half_rn(s),
                                     __float2half_rn(s1));
          atomic_add_pk2_f16(out + row0, h01);
        }
      }
      // RPT==1 with ksplit>1 is rejected by the launcher.
    }
  }
}

}  // namespace dense_gemv_gfx906
}  // namespace vllm

// ---------------------------------------------------------------------------
// Entry point
//
//   weight: [N, K] fp16, row-major, contiguous
//   x:      [1, K] (or [K]) fp16, contiguous
//   kchunk: 512, 1024, 2048 or 4096 (must divide K); kchunk >= K = single pass
//   rpt:    rows per thread override (VLLM_GFX906_GEMV_RPT env); default
//           auto: 4 if N%4==0, 2 if N%2==0, else 1. RPT=1 forbids K-split.
//
// Returns out: [1, N] fp16. Pre-zeroed internally when K > kchunk.
// ---------------------------------------------------------------------------
torch::Tensor dense_gemv_gfx906(torch::Tensor weight, torch::Tensor x,
                                int64_t kchunk) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda());
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2);
  TORCH_CHECK(weight.scalar_type() == torch::kHalf);
  TORCH_CHECK(x.scalar_type() == torch::kHalf);
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous());
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TORCH_CHECK(x.size(0) == 1, "x must be [1, K] (M=1 only)");
  TORCH_CHECK(x.size(1) == K, "x/weight K mismatch");
  TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
  TORCH_CHECK(kchunk == 512 || kchunk == 1024 || kchunk == 2048 ||
                  kchunk == 4096,
              "kchunk must be 512, 1024, 2048 or 4096");
  TORCH_CHECK(K % kchunk == 0, "K must be divisible by kchunk");

  // Rows per thread: env override (micro-bench sweeps), else the
  // gfx906 (MI50)-measured rule: single-pass KCHUNK=2048 with RPT=2 wins
  // for N==256 (router) and N>=2048 (in_proj/qkv/LM head); RPT=4
  // elsewhere (RPT=2 is far worse on the 1024-row shared gate_up).
  int rpt = -1;
  if (const char* e = getenv("VLLM_GFX906_GEMV_RPT")) {
    rpt = atoi(e);
    TORCH_CHECK(rpt != 0,
                "VLLM_GFX906_GEMV_RPT must be 1, 2 or 4 (got 0)");
    if (rpt != 1 && rpt != 2 && rpt != 4) {
      TORCH_WARN_ONCE(
          "VLLM_GFX906_GEMV_RPT (", rpt, ") is not one of 1/2/4; using the "
          "default rule instead.");
      rpt = -1;
    }
  }
  if (rpt < 0) {
    if (kchunk == 2048 && (N == 256 || N >= 2048))
      rpt = 2;
    else if (kchunk == 512 && N == 2048)
      // shared-expert down_proj [2048, 512] (M=1 decode): RPT=2 measured
      // 5.6-5.7 us vs 6.7-7.7 us LLMM1 rpb4 and 8.0-8.2 us for RPT=4
      // (bench benchmarks/kernels/gfx906/bench_dense_gemv_gfx906.py).
      rpt = 2;
    else if (kchunk == 1024)
      // K=17408 down_proj (N=5120, ksplit=17): RPT=2 measured at 100.2%
      // of the HBM floor vs 116% for RPT=4 (bench /
      // benchmarks/kernels/gfx906/bench_dense_gemv_k5120.py).
      rpt = (N % 2 == 0) ? 2 : 1;
    else
      rpt = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  }
  TORCH_CHECK(N % rpt == 0, "N (", N, ") not divisible by RPT (", rpt, ")");
  if (rpt == 1)
    TORCH_CHECK(kchunk >= K, "RPT=1 requires kchunk >= K (no K-split)");

  const int ksplit = (int)(K / kchunk);
  auto out = torch::empty({1, N}, weight.options());
  if (ksplit > 1) out.zero_();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* wp = (const half*)weight.data_ptr();
  const half* xp = (const half*)x.data_ptr();
  half* op = (half*)out.data_ptr();

  #define LAUNCH(RPT, KC)                                                   \
    {                                                                       \
      dim3 grid((N + RPT - 1) / RPT, ksplit);                               \
      vllm::dense_gemv_gfx906::dense_gemv_kernel<RPT, KC>                   \
          <<<grid, KC / 8, 0, stream>>>(xp, wp, op, (int)N, (int)K, ksplit);\
    }
  #define LAUNCH_BY_RPT(KCVAL)                                              \
    do {                                                                    \
      if (rpt == 4)                                                         \
        LAUNCH(4, KCVAL)                                                    \
      else if (rpt == 2)                                                    \
        LAUNCH(2, KCVAL)                                                    \
      else                                                                  \
        LAUNCH(1, KCVAL)                                                    \
    } while (0)

  if (kchunk == 4096)
    LAUNCH_BY_RPT(4096);
  else if (kchunk == 2048)
    LAUNCH_BY_RPT(2048);
  else if (kchunk == 1024)
    LAUNCH_BY_RPT(1024);
  else
    LAUNCH_BY_RPT(512);
  return out;
}
