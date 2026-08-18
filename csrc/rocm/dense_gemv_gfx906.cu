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

// ---------------------------------------------------------------------------
// M<=4 variant (spec-decode draft steps, Phase 1 L1').
//
//   out[M, N] = x[M, K] @ W[N, K]^T
//
// Same row-parallel structure as the M=1 kernel, but each thread also
// holds the 8-half x slice for every M row (registers; x re-reads hit
// L2) and accumulates RPT*M fp32 partials. Weight traffic is
// M-invariant — the whole point: at M=4 the triton_matmul fallback
// costs ~7x the HBM floor (174 us for a 31.5 MB weight), while this
// stays at the M=1 weight-read speed.
//
// K-split (kchunk < K) uses the same packed-fp16 CAS epilogue as the
// M=1 kernel: per M row, one 32-bit CAS (RPT=2) or 64-bit CAS (RPT=4)
// over the RPT adjacent output rows. RPT>=2 required (per-M-row output
// addresses are N apart, so the RPT=1 packed CAS is impossible).
// ---------------------------------------------------------------------------

template <int RPT, int KCHUNK>
__global__ void __launch_bounds__(KCHUNK / 8)
    dense_gemv_m_kernel(const half* __restrict__ x,   // [M, K]
                       const half* __restrict__ w,   // [N, K]
                       half* __restrict__ out,       // [M, N], pre-zeroed
                       // if KSPLIT>1
                       const int M, const int N, const int K,
                       const int ksplit) {
  static_assert(KCHUNK == 512 || KCHUNK == 1024 || KCHUNK == 2048 ||
                    KCHUNK == 4096,
                "KCHUNK must be 512, 1024, 2048 or 4096");
  static_assert(RPT == 2 || RPT == 4, "RPT must be 2 or 4");
  constexpr int THREADS = KCHUNK / 8;
  constexpr int WARPS = THREADS / 64;
  const int t = threadIdx.x;
  const int row0 = blockIdx.x * RPT;
  const int k0 = blockIdx.y * KCHUNK;

  // x slices for all M rows (x is small: M*K*2B <= 40 KB, L2-resident;
  // every block re-reads its k-chunk of it).
  union {
    uint4 u;
    half2 h2[4];
  } xa[4];
  #pragma unroll
  for (int m = 0; m < 4; ++m)
    if (m < M) xa[m].u = *(const uint4*)(x + (int64_t)m * K + k0 + t * 8);

  float acc[RPT][4];
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int row = row0 + r;
    #pragma unroll
    for (int m = 0; m < 4; ++m) acc[r][m] = 0.0f;
    if (row >= N) continue;
    union {
      uint4 u;
      half2 h2[4];
    } wa;
    wa.u = *(const uint4*)(w + (int64_t)row * K + k0 + t * 8);
    #pragma unroll
    for (int m = 0; m < 4; ++m)
      if (m < M) acc[r][m] = dot8_f32(wa.h2, xa[m].h2);
  }

  // Flatten to acc_flat[r*4+m] for the reduction (lanes < RPT*4).
  float acc_flat[RPT * 4];
  #pragma unroll
  for (int r = 0; r < RPT; ++r)
    #pragma unroll
    for (int m = 0; m < 4; ++m) acc_flat[r * 4 + m] = acc[r][m];

  #pragma unroll
  for (int mask = 32; mask >= 1; mask /= 2)
    #pragma unroll
    for (int i = 0; i < RPT * 4; ++i) acc_flat[i] += __shfl_xor(acc_flat[i], mask);

  if constexpr (WARPS == 1) {
    // Lane i < RPT*4 holds (r=i/4, m=i%4)'s full sum.
    if (ksplit == 1) {
      if (t < RPT * 4) {
        const int r = t / 4, m = t % 4, row = row0 + r;
        if (m < M && row < N)
          out[(int64_t)m * N + row] = __float2half_rn(acc_flat[t]);
      }
    } else {
      // CAS packs the RPT adjacent rows of each out[m]; lane 0 gathers
      // every (r, m) sum via shfl and issues the M CAS ops.
      if (t == 0) {
        float s[RPT * 4];
        #pragma unroll
        for (int i = 0; i < RPT * 4; ++i) s[i] = __shfl(acc_flat[t], i);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
          if (m >= M) continue;
          if (row0 + RPT - 1 >= N) continue;  // ragged tail: RPT rows valid
          if constexpr (RPT == 4)
            atomic_add_pk4_f16(
                out + (int64_t)m * N + row0,
                __halves2half2(__float2half_rn(s[0 * 4 + m]),
                               __float2half_rn(s[1 * 4 + m])),
                __halves2half2(__float2half_rn(s[2 * 4 + m]),
                               __float2half_rn(s[3 * 4 + m])));
          else
            atomic_add_pk2_f16(
                out + (int64_t)m * N + row0,
                __halves2half2(__float2half_rn(s[0 * 4 + m]),
                               __float2half_rn(s[1 * 4 + m])));
        }
      }
    }
  } else {
    __shared__ float red_smem[RPT * 4][8];  // WARPS <= 8 (KCHUNK <= 4096)
    const int warp = t / 64;
    const int lane = t % 64;
    if (lane < RPT * 4) red_smem[lane][warp] = acc_flat[lane];
    __syncthreads();
    if (warp == 0) {
      if (ksplit == 1) {
        if (lane < RPT * 4) {
          const int r = lane / 4, m = lane % 4, row = row0 + r;
          float s = 0.0f;
          #pragma unroll
          for (int wp = 0; wp < WARPS; ++wp) s += red_smem[lane][wp];
          if (m < M && row < N) out[(int64_t)m * N + row] = __float2half_rn(s);
        }
      } else {
        if (lane == 0) {
          float s[RPT * 4];
          #pragma unroll
          for (int i = 0; i < RPT * 4; ++i) {
            s[i] = 0.0f;
            #pragma unroll
            for (int wp = 0; wp < WARPS; ++wp) s[i] += red_smem[i][wp];
          }
          #pragma unroll
          for (int m = 0; m < 4; ++m) {
            if (m >= M) continue;
            if (row0 + RPT - 1 >= N) continue;  // ragged tail
            if constexpr (RPT == 4)
              atomic_add_pk4_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(s[0 * 4 + m]),
                                 __float2half_rn(s[1 * 4 + m])),
                  __halves2half2(__float2half_rn(s[2 * 4 + m]),
                                 __float2half_rn(s[3 * 4 + m])));
            else
              atomic_add_pk2_f16(
                  out + (int64_t)m * N + row0,
                  __halves2half2(__float2half_rn(s[0 * 4 + m]),
                                 __float2half_rn(s[1 * 4 + m])));
          }
        }
      }
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
// ---------------------------------------------------------------------------
// M<=4 entry point (spec decode; see dense_gemv_m_kernel above).
//
//   weight: [N, K] fp16 row-major; x: [M, K] fp16, 1 <= M <= 4
//   Returns out: [M, N] fp16.
// ---------------------------------------------------------------------------
torch::Tensor dense_gemv_m4_gfx906(torch::Tensor weight, torch::Tensor x,
                                   int64_t kchunk) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda());
  TORCH_CHECK(weight.dim() == 2 && x.dim() == 2);
  TORCH_CHECK(weight.scalar_type() == torch::kHalf);
  TORCH_CHECK(x.scalar_type() == torch::kHalf);
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous());
  const int64_t M = x.size(0);
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TORCH_CHECK(M >= 1 && M <= 4, "M must be 1..4 (got ", M, ")");
  TORCH_CHECK(x.size(1) == K, "x/weight K mismatch");
  TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
  TORCH_CHECK(kchunk == 512 || kchunk == 1024 || kchunk == 2048 ||
                  kchunk == 4096,
              "kchunk must be 512, 1024, 2048 or 4096");
  TORCH_CHECK(K % kchunk == 0, "K must be divisible by kchunk");

  // RPT is 2 or 4 (the packed CAS epilogue needs adjacent rows); env
  // override for micro-bench sweeps, default 2 (the M=1 K=17408 winner).
  int rpt = 2;
  if (const char* e = getenv("VLLM_GFX906_GEMVM_RPT")) {
    const int v = atoi(e);
    if (v == 2 || v == 4) rpt = v;
  }
  TORCH_CHECK(N % rpt == 0, "N (", N, ") not divisible by RPT (", rpt, ")");

  const int ksplit = (int)(K / kchunk);
  auto out = torch::empty({M, N}, weight.options());
  if (ksplit > 1) out.zero_();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* wp = (const half*)weight.data_ptr();
  const half* xp = (const half*)x.data_ptr();
  half* op = (half*)out.data_ptr();

  #define LAUNCHM(RPT, KC)                                                \
    {                                                                     \
      dim3 grid(N / RPT, ksplit);                                         \
      vllm::dense_gemv_gfx906::dense_gemv_m_kernel<RPT, KC>               \
          <<<grid, KC / 8, 0, stream>>>(xp, wp, op, (int)M, (int)N,       \
                                        (int)K, ksplit);                  \
    }
  #define LAUNCHM_BY_RPT(KCVAL)                                           \
    do {                                                                  \
      if (rpt == 4)                                                       \
        LAUNCHM(4, KCVAL)                                                 \
      else                                                                \
        LAUNCHM(2, KCVAL)                                                 \
    } while (0)
  if (kchunk == 4096)
    LAUNCHM_BY_RPT(4096);
  else if (kchunk == 2048)
    LAUNCHM_BY_RPT(2048);
  else if (kchunk == 1024)
    LAUNCHM_BY_RPT(1024);
  else
    LAUNCHM_BY_RPT(512);
  return out;
}

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
