// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// C1 stage 2: M=1 fused top-k + moe_align_block_size + count_and_sort
// routing kernel for gfx906 (E=256, topk=8, block_size=1, fp16 gating).
// See docs/gfx906/DEVLOG-moe-c1-routing-fusion.md.
//
// The production M=1 decode routing chain is three kernels per MoE layer:
// topkGating (11.8 us in-graph/node), moe_align_block_size_kernel
// (2-block, 1024-thread CUB scan) and count_and_sort_expert_tokens. This
// kernel does all three in one 128-thread CTA (120 -> 40 graph nodes per
// 40-layer step):
//
//   Phase 1  warp 0: bit-exact copy of the S2 dedicated M=1 topk phase
//                    (moe_topk_gfx906.cu; bit-equal to the generic
//                    topkGating arithmetic: max/expf/sum butterflies,
//                    reciprocal scale, NaN clamp, lowest-index tie-break,
//                    renormalize).
//   Phase 2  all 128 threads: zero the 256 per-expert counts + rank slots.
//   Phase 3  lanes 0..7: LDS-atomic count of the 8 (token, expert) slots.
//   Phase 4  warp 0: 5-step shfl_down warp scan over the 256 counts
//                    (exclusive prefix = total - suffix, total == 8).
//   Phase 5  lanes 0..7: lane-atomic placement into sorted_token_ids /
//                    expert_ids (single-slot blocks at block_size=1).
//
// No grid work, no global atomics, no D2H: capture-safe. The topk outputs
// are bit-equal to ops.topk_softmax and the align outputs bit-equal to the
// two-kernel moe_align_block_size chain for the exact shape it serves
// (verified in tests/kernels/moe/test_moe_routing_fused_m1_gfx906.py).
// Within-expert slot order follows issuing-lane order (slots 0..7),
// matching the production single-warp global-atomic order.

#include <cmath>
#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../cuda_compat.h"

namespace vllm {
namespace moe_routing_fused_m1_gfx906 {

static constexpr int E = 256;      // experts
static constexpr int K_TOP = 8;    // top-k
static constexpr int VPT = 8;      // experts per active lane (topk phase)
static constexpr int LANES = 32;   // active topk lanes
static constexpr int THREADS = 128;

__global__ void __launch_bounds__(THREADS) moe_routing_fused_m1_gfx906_kernel(
    const __half* __restrict__ gating,          // [256]
    float* __restrict__ topk_weights,           // [8]
    int* __restrict__ topk_ids,                 // [8]
    int* __restrict__ token_expert_ids,         // [8]
    int* __restrict__ sorted_token_ids,         // [8]
    int* __restrict__ expert_ids,               // [8]
    int* __restrict__ num_tokens_post_pad,      // [1]
    const bool renormalize) {
  __shared__ int s_counts[E];
  __shared__ int s_base[E];
  __shared__ int s_rank[E];

  const int t = threadIdx.x;

  // ---- Phase 1: topk (S2 bit-exact phase; warp 0, lanes 0..31) ----
  if (t < LANES) {
    uint4 raw = *(const uint4*)(gating + VPT * t);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
    float p[VPT];
    #pragma unroll
    for (int i = 0; i < VPT / 2; ++i) {
      float2 f = __half22float2(h2[i]);
      p[2 * i] = f.x;
      p[2 * i + 1] = f.y;
    }

    float row_max = p[0];
    #pragma unroll
    for (int i = 1; i < VPT; ++i) row_max = fmaxf(row_max, p[i]);
    #pragma unroll
    for (int mask = LANES / 2; mask > 0; mask >>= 1) {
      row_max = fmaxf(
          row_max, VLLM_SHFL_XOR_SYNC_WIDTH(row_max, mask, LANES));
    }

    float row_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
      p[i] = expf(p[i] - row_max);
      row_sum += p[i];
    }
    #pragma unroll
    for (int mask = LANES / 2; mask > 0; mask >>= 1) {
      row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, LANES);
    }
    const float recip = 1.f / row_sum;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) p[i] *= recip;

    #pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (isnan(p[i]) || isinf(p[i])) p[i] = 0.f;

    float choice[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; ++i) choice[i] = p[i];

    float selected_sum = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < K_TOP; ++k_idx) {
      float mv = choice[0];
      float mv_p = p[0];
      int expert = VPT * t;
      #pragma unroll
      for (int i = 1; i < VPT; ++i) {
        if (choice[i] > mv) {
          mv = choice[i];
          mv_p = p[i];
          expert = VPT * t + i;
        }
      }
      #pragma unroll
      for (int mask = LANES / 2; mask > 0; mask >>= 1) {
        float ov = VLLM_SHFL_XOR_SYNC_WIDTH(mv, mask, LANES);
        float op = VLLM_SHFL_XOR_SYNC_WIDTH(mv_p, mask, LANES);
        int oe = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, LANES);
        if (ov > mv || (ov == mv && oe < expert)) {
          mv = ov;
          mv_p = op;
          expert = oe;
        }
      }
      if (t == 0) {
        topk_weights[k_idx] = mv_p;
        topk_ids[k_idx] = expert;
        token_expert_ids[k_idx] = k_idx;
        if (renormalize) selected_sum += mv_p;
      }
      if (k_idx + 1 < K_TOP && t == expert / VPT)
        choice[expert % VPT] = -10000.f;
    }

    if (t == 0 && renormalize) {
      const float denom = selected_sum > 0.f ? selected_sum : 1.f;
      const float scale = 1.0f / denom;
      #pragma unroll
      for (int i = 0; i < K_TOP; ++i) topk_weights[i] *= scale;
    }
  }
  __syncthreads();

  // ---- Phase 2: zero counts + rank (2 x 256 ints over 128 threads) ----
  #pragma unroll
  for (int i = 0; i < 2 * E / THREADS; ++i) {
    const int idx = t + i * THREADS;
    if (idx < E) s_counts[idx] = 0;
    else s_rank[idx - E] = 0;
  }
  __syncthreads();

  // ---- Phase 3: count tokens per expert (LDS atomics) ----
  if (t < K_TOP) {
    atomicAdd(&s_counts[topk_ids[t]], 1);
  }
  __syncthreads();

  // ---- Phase 4: exclusive prefix sum over the 256 counts (warp 0) ----
  if (t < LANES) {
    int lane_excl[8];
    int acc = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int c = s_counts[8 * t + j];
      lane_excl[j] = acc;
      acc += c;
    }
    int incl = acc;
    #pragma unroll
    for (int d = 1; d < LANES; d <<= 1) {
      int n = VLLM_SHFL_DOWN_SYNC(incl, d);
      if (t + d < LANES) incl += n;
    }
    const int excl = K_TOP - incl;  // total (8 slots) - suffix
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      s_base[8 * t + j] = excl + lane_excl[j];
    }
    if (t == 31) {
      num_tokens_post_pad[0] = K_TOP;
    }
  }
  __syncthreads();

  // ---- Phase 5: placement (lane-atomic rank, slot order 0..7) ----
  if (t < K_TOP) {
    const int e = topk_ids[t];
    const int pos = s_base[e] + atomicAdd(&s_rank[e], 1);
    sorted_token_ids[pos] = t;
    expert_ids[pos] = e;
  }
}

}  // namespace moe_routing_fused_m1_gfx906
}  // namespace vllm

void moe_routing_fused_m1_gfx906(torch::Tensor gating,
                                 torch::Tensor topk_weights,
                                 torch::Tensor topk_ids,
                                 torch::Tensor token_expert_ids,
                                 torch::Tensor sorted_token_ids,
                                 torch::Tensor expert_ids,
                                 torch::Tensor num_tokens_post_pad,
                                 bool renormalize) {
  TORCH_CHECK(gating.is_cuda() && topk_weights.is_cuda() &&
                  topk_ids.is_cuda() && token_expert_ids.is_cuda() &&
                  sorted_token_ids.is_cuda() && expert_ids.is_cuda() &&
                  num_tokens_post_pad.is_cuda(),
              "all tensors must be CUDA/HIP tensors");
  TORCH_CHECK(gating.scalar_type() == torch::kHalf,
              "gating must be half (fp16)");
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat,
              "topk_weights must be float32");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt &&
                  token_expert_ids.scalar_type() == torch::kInt &&
                  sorted_token_ids.scalar_type() == torch::kInt &&
                  expert_ids.scalar_type() == torch::kInt &&
                  num_tokens_post_pad.scalar_type() == torch::kInt,
              "id buffers must be int32");
  TORCH_CHECK(gating.size(0) == 1 && gating.size(1) == 256,
              "M=1, E=256 only, got [", gating.size(0), ", ",
              gating.size(1), "]");
  TORCH_CHECK(topk_weights.size(1) == 8 && topk_ids.size(1) == 8 &&
                  token_expert_ids.size(1) == 8,
              "topk=8 only");
  TORCH_CHECK(sorted_token_ids.size(0) == 8 && expert_ids.size(0) == 8,
              "align buffers must be sized 8 (wrapper convention for M=1)");
  TORCH_CHECK(gating.is_contiguous() && topk_weights.is_contiguous() &&
                  topk_ids.is_contiguous() &&
                  token_expert_ids.is_contiguous() &&
                  sorted_token_ids.is_contiguous() &&
                  expert_ids.is_contiguous() &&
                  num_tokens_post_pad.is_contiguous(),
              "all tensors must be contiguous");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gating));
  auto stream = at::cuda::getCurrentCUDAStream();
  vllm::moe_routing_fused_m1_gfx906::moe_routing_fused_m1_gfx906_kernel
      <<<1, vllm::moe_routing_fused_m1_gfx906::THREADS, 0, stream>>>(
          (const __half*)gating.data_ptr(),
          topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>(),
          token_expert_ids.data_ptr<int>(), sorted_token_ids.data_ptr<int>(),
          expert_ids.data_ptr<int>(),
          num_tokens_post_pad.data_ptr<int>(), renormalize);
}
