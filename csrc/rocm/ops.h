#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void moe_gptq_gemm_gfx906(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor b_qzeros, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded,
                          int64_t top_k, int64_t block_size_m,
                          bool mul_topk_weight, int64_t output_topk,
                          int64_t zero_offset);

// M<=4 W16A16 dense GEMM (GEMV-family) for gfx906 spec decode (L1').
torch::Tensor dense_gemv_m4_gfx906(torch::Tensor weight, torch::Tensor x,
                                   int64_t kchunk);

// M=1 W16A16 dense GEMV for gfx906 (P3-2b; see dense_gemv_gfx906.cu).
// weight [N, K] row-major fp16, x [1, K] fp16, kchunk 512|1024|2048|4096 (divides K).
// Returns out [1, N] fp16.
torch::Tensor dense_gemv_gfx906(torch::Tensor weight, torch::Tensor x,
                                int64_t kchunk);

// M=1 fused top-k softmax router for gfx906, E=256, topk=8 (S2; see
// moe_topk_gfx906.cu). Bit-equal to the generic topkGating for the exact
// decode shape it serves (M==1, half, no bias/padding).
void moe_topk_softmax_m1_gfx906(torch::Tensor topk_weights,     // [1, 8] f32
                                torch::Tensor topk_ids,         // [1, 8] i32
                                torch::Tensor token_expert_ids, // [1, 8] i32
                                torch::Tensor gating,           // [1, 256] f16
                                bool renormalize);

// M=1 fused moe_align_block_size + count_and_sort for gfx906, E=256,
// topk=8, block_size=1 (C1 stage 1; see moe_align_m1_gfx906.cu).
// Buffers are the wrapper-convention sizes for this shape (8 / 8 / 1).
void moe_align_block_size_m1_gfx906(torch::Tensor topk_ids,          // [1, 8] i32
                                    int64_t num_experts,             // 256
                                    int64_t block_size,              // 1
                                    torch::Tensor sorted_token_ids,  // [8] i32
                                    torch::Tensor expert_ids,        // [8] i32
                                    torch::Tensor num_tokens_post_pad);  // [1] i32

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);
