# gfx906 -> upstream main merge strategy

Date: 2026-08-13

## Rationale (investigation)

Upstream main dropped gfx906 from aiter support: aiter is built with
AITER_ROCM_ARCH=gfx942;gfx950, and gating is `get_cdna_version() > 2` + a
separate RDNA4 path. The fork's own 0.26 build already compiled aiter only for
gfx942/950, so on gfx906 the fork's "rocm_aiter_mla_sparse" backend already
falls back to Triton/torch (reference_mla_sparse_prefill / use_fp16_sparse).

=> Most of main's changes to kvcache dtype / minimax / MoE / MLA prefill are
   vLLM structural refactors converging on the CDNA3+/RDNA4 substrate, plus
   real aiter-scope tightening. Not gfx906-targeted removals.

## Strategy: build on main's newer structures, keep only minimal gfx906 kernel deltas

ADOPT main's structures wholesale for the big refactors:
- FusedMoEFactory MoE layer (also gives bailing_moe_v3 / Ling-3 for free)
- MLA prefill chunking (plan_mla_context_chunks), gather kernel, minimax module layout
- aiter scoping to CDNA3+/RDNA4

RE-APPLY only the true gfx906 kernel deltas as thin adaptations on top:
- fp32->fp16 casts before tl.dot in Triton minimax/sparse kernels
- waves_per_eu / LDS-safe num_warps/num_stages launch params
- fp32-KV / fp16 IndexerKVDType IF still needed (verify; main's fp32 GEMV path may supersede)
- ExllamaLinearKernel priority, gptq_shuffle_awq_qweight, cp_gather_indexer_k_cache_fp16
- the gfx906 NCCL workaround

## Verification
Rebuild native ext + smoke test on gfx906 docker (Qwen3-0.6B / AWQ / MRv2 / MoE),
plus MiniMax-style fp32-KV decode to settle the IndexerKVDType question.
