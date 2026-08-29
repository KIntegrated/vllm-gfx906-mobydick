#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points for asymmetric quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, Tensor b_g_idx, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // W4A16 fused MoE GEMM for gfx906 (Vega 20, no MFMA). Works on any ROCm
  // target; selected at runtime on gfx906.
  rocm_ops.def(
      "moe_gptq_gemm_gfx906(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk, int zero_offset) -> ()");
  rocm_ops.impl("moe_gptq_gemm_gfx906", torch::kCUDA,
                &moe_gptq_gemm_gfx906);

  // M=1 W16A16 dense GEMV for gfx906 (Phase 3 P3-2b). Works on any ROCm
  // target; selected at runtime on gfx906.
  rocm_ops.def(
      "dense_gemv_gfx906(Tensor weight, Tensor x, int kchunk) -> Tensor");
  rocm_ops.impl("dense_gemv_gfx906", torch::kCUDA, &dense_gemv_gfx906);

  // M<=4 W16A16 dense GEMM (GEMV-family) for gfx906 spec decode (L1').
  rocm_ops.def(
      "dense_gemv_m4_gfx906(Tensor weight, Tensor x, int kchunk) -> Tensor");
  rocm_ops.impl("dense_gemv_m4_gfx906", torch::kCUDA, &dense_gemv_m4_gfx906);

  // M=1 fused top-k softmax router for gfx906 (S2; E=256, topk=8).
  // Works on any ROCm target; selected at runtime on gfx906 for the
  // exact decode shape (M==1, half, softmax, no bias/padding).
  rocm_ops.def(
      "moe_topk_softmax_m1_gfx906(Tensor! topk_weights, Tensor! topk_ids, "
      "Tensor! token_expert_ids, Tensor gating, bool renormalize) -> ()");
  rocm_ops.impl("moe_topk_softmax_m1_gfx906", torch::kCUDA,
                &moe_topk_softmax_m1_gfx906);

  // M=1 fused moe_align + count_and_sort for gfx906 (C1 stage 1;
  // E=256, topk=8, block_size=1). One 128-thread CTA replaces the
  // two-kernel generic chain for the decode shape.
  rocm_ops.def(
      "moe_align_block_size_m1_gfx906(Tensor topk_ids, int num_experts, "
      "int block_size, Tensor! sorted_token_ids, Tensor! expert_ids, "
      "Tensor! num_tokens_post_pad) -> ()");
  rocm_ops.impl("moe_align_block_size_m1_gfx906", torch::kCUDA,
                &moe_align_block_size_m1_gfx906);

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
