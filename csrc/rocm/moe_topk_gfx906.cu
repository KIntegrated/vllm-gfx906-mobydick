// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
//
// M=1 fused top-k softmax router kernel for gfx906 (Vega 20), E=256, k=8,
// fp16 gating input. Week of 2026-08-18 sprint, S2 (see
// docs/gfx906/DEVLOG-moe-m1-sprint.md).
//
// Replaces the generic topkGating<8, 256, 4, 16, 64> for the single-token
// decode shape: that kernel launches a (64, 4) block and runs the generic
// row-packing/loop machinery for ONE row — 17.9 us/call in-model (40
// launches/step = 713 us/step) for top-8-of-256 work that is ~3-5 us of
// latency. This kernel is one 64-lane wave; lanes 0-31 each own 8
// consecutive experts — the exact THREADS_PER_ROW=32 / VPT=8 partition the
// generic kernel uses — and reproduce its arithmetic in the same order:
//
//   1. per-thread max of 8 logits, XOR-butterfly (mask 16..1, width 32)
//   2. p[i] = expf(x[i] - row_max); sequential per-thread sum, butterfly sum
//   3. p[i] *= 1.f / row_sum   (full-row softmax, reciprocal multiply)
//   4. NaN/Inf p[i] -> 0.f
//   5. k=8 iterations: local argmax (i ascending, strict >), XOR-butterfly
//      argmax with lowest-expert-index tie-break, blank winner to -10000.f
//   6. renormalize: out[i] *= 1.0f / selected_sum (denom<=0 -> 1.f)
//
// with the same expf (not __expf) and the same width-32 shuffles
// (VLLM_SHFL_XOR_SYNC_WIDTH), so the fp32 topk_weights / int32 topk_ids /
// token_expert_ids outputs are bit-equal to the generic path for finite
// inputs — matching the generic's own contract. The Python-side dispatch
// (fused_topk_router.py) only routes the exact M==1 / E==256 / k==8 / half /
// softmax / renormalize / no-bias / no-padding decode shape, which is
// finite-logit by construction (no padding rows).
//
// ISA notes (measured, gfx906): width-32 __shfl_xor lowers through
// v_readlane/v_writelane + the E32 register file — expensive, but the
// measured GPU self-time is still 12.5 us vs the generic's 17.5 us because
// this kernel runs one wave with no row machinery. A width-64 variant
// (32 dummy -inf lanes) is NOT faster: it does 2x the per-lane work and
// width-64 shuffles lower through ds_bpermute + E32 here too. An
// LDS+barrier variant (no shuffles at all) measured 35.5 us — the 8x32
// scan pattern + 22 s_barriers is worse than the shuffle path. See the
// dev log for the full table.

#include <cmath>
#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../cuda_compat.h"

namespace vllm {
namespace moe_topk_gfx906 {

static constexpr int K_TOP = 8;
static constexpr int LANES = 32;  // active lanes (THREADS_PER_ROW)
static constexpr int VPT = 8;     // experts per active lane

__global__ void __launch_bounds__(64)
    topk_softmax_m1_gfx906_kernel(const __half* __restrict__ gating,  // [256]
                                  float* __restrict__ topk_weights,   // [8]
                                  int* __restrict__ topk_ids,         // [8]
                                  int* __restrict__ token_expert_ids, // [8]
                                  const bool renormalize) {
  const int t = threadIdx.x;
  if (t >= LANES) return;

  // One 16B load of this lane's 8 experts (same layout as the generic
  // kernel's BYTES_PER_LDG=16 half loads).
  uint4 raw = *(const uint4*)(gating + VPT * t);
  const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
  float p[VPT];
  #pragma unroll
  for (int i = 0; i < VPT / 2; ++i) {
    float2 f = __half22float2(h2[i]);
    p[2 * i] = f.x;
    p[2 * i + 1] = f.y;
  }

  // (1) row max: per-thread max then butterfly (identical order).
  float row_max = p[0];
  #pragma unroll
  for (int i = 1; i < VPT; ++i) row_max = fmaxf(row_max, p[i]);
  #pragma unroll
  for (int mask = LANES / 2; mask > 0; mask >>= 1) {
    row_max = fmaxf(row_max,
                    VLLM_SHFL_XOR_SYNC_WIDTH(row_max, mask, LANES));
  }

  // (2) softmax: expf, sequential sum, butterfly sum — identical order.
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
  // (3) reciprocal scale (identical: multiply by 1/sum, not divide).
  const float recip = 1.f / row_sum;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) p[i] *= recip;

  // (4) NaN/Inf clamp (identical guard from the generic kernel).
  #pragma unroll
  for (int i = 0; i < VPT; ++i)
    if (isnan(p[i]) || isinf(p[i])) p[i] = 0.f;

  float choice[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) choice[i] = p[i];

  // (5) top-k selection (identical local scan + butterfly tie-break).
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
      // generic kernel: k_idx * num_rows + row; num_rows == 1 here.
      token_expert_ids[k_idx] = k_idx;
      if (renormalize) selected_sum += mv_p;
    }
    if (k_idx + 1 < K_TOP && t == expert / VPT)
      choice[expert % VPT] = -10000.f;
  }

  // (6) renormalize (identical: scale = 1/denom, out *= scale).
  if (t == 0 && renormalize) {
    const float denom = selected_sum > 0.f ? selected_sum : 1.f;
    const float scale = 1.0f / denom;
    #pragma unroll
    for (int i = 0; i < K_TOP; ++i) topk_weights[i] *= scale;
  }
}

}  // namespace moe_topk_gfx906
}  // namespace vllm

void moe_topk_softmax_m1_gfx906(torch::Tensor topk_weights,
                                torch::Tensor topk_ids,
                                torch::Tensor token_expert_ids,
                                torch::Tensor gating, bool renormalize) {
  TORCH_CHECK(gating.is_cuda() && topk_weights.is_cuda() &&
                  topk_ids.is_cuda() && token_expert_ids.is_cuda(),
              "all tensors must be CUDA/HIP tensors");
  TORCH_CHECK(gating.scalar_type() == torch::kHalf,
              "gating must be half (fp16)");
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat,
              "topk_weights must be float32");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt &&
                  token_expert_ids.scalar_type() == torch::kInt,
              "topk_ids and token_expert_ids must be int32");
  TORCH_CHECK(gating.size(0) == 1, "M=1 only, got M=", gating.size(0));
  TORCH_CHECK(gating.size(1) == 256, "E=256 only, got E=", gating.size(1));
  TORCH_CHECK(topk_weights.size(1) == 8, "topk=8 only, got ",
              topk_weights.size(1));
  TORCH_CHECK(gating.is_contiguous() && topk_weights.is_contiguous() &&
                  topk_ids.is_contiguous() &&
                  token_expert_ids.is_contiguous(),
              "all tensors must be contiguous");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gating));
  auto stream = at::cuda::getCurrentCUDAStream();
  vllm::moe_topk_gfx906::topk_softmax_m1_gfx906_kernel
      <<<1, 64, 0, stream>>>(
          (const __half*)gating.data_ptr(),
          topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>(),
          token_expert_ids.data_ptr<int>(), renormalize);
}
