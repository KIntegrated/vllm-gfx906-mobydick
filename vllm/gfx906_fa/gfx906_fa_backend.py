# SPDX-License-Identifier: Apache-2.0
# Copyright (C) Nick — nick413@gmail.com
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
# (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
"""
vLLM v1 attention backend for gfx906 (MI50) built on the Q8
FlashAttention kernel ported from llama.cpp-gfx906.

Registered as AttentionBackendEnum.CUSTOM (selected per-model via
`attention_config={"backend": "CUSTOM"}`; the legacy
VLLM_ATTENTION_BACKEND env var no longer gates this path):

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum, register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "gfx906_fa_backend.Gfx906FABackend",
    )

The KV cache layout matches TritonAttentionBackend:
    (num_blocks, 2, block_size, num_kv_heads, head_size)
so the backend can be switched without allocator changes.

Decode (LEGACY=1, the default) gathers K/V from the paged fp16 cache
into contiguous fp16 buffers with a fused HIP gather kernel, quantizes
K to Q8 on device, and runs the Q8 FA kernel.
"""

import os as _os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType

# Project modules (need to be importable from the vllm package)
from vllm.gfx906_fa.gfx906_fa_paged import (  # noqa: E402
    _pick_ncols1,
    forward_paged,
)
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
@dataclass
class Gfx906FAMetadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor      # [B+1] int32
    seq_lens: torch.Tensor             # [B]   int32
    block_table: torch.Tensor          # [B, max_num_blocks] int32
    slot_mapping: torch.Tensor         # [num_tokens] int64
    use_cascade: bool = False
    common_prefix_len: int = 0


class Gfx906FAMetadataBuilder(
    AttentionMetadataBuilder[Gfx906FAMetadata]
):
    # P3-3a M2: the LEGACY (inline-quant) decode path is FULL-capture-safe
    # (first FULL capture runs at profile_seq_lens=max_model_len, so
    # Sk-sized buffers allocate at capacity; metadata is runner-staged and
    # re-read live at replay). Verified: serving bench 53.09 t/s + 128/128
    # greedy probe identical to the Triton-FULL reference, plus
    # test_cudagraph_capture_replay_legacy_decode_path.
    # GFX906_FA_CG=never|always still overrides for experiments.
    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport:
        # RC2 (sub-plan): the LEGACY=0 Q8 K side-buffer stays consistent
        # only if every KV write goes through do_kv_cache_update. Prefix
        # caching (COW'd blocks) and FULL cudagraph capture (dummy/replay
        # writes) bypass that invariant → corrupt attention output.
        if _os.environ.get("GFX906_FA_LEGACY", "1") != "1":
            if getattr(vllm_config.cache_config,
                       "enable_prefix_caching", False):
                # Fail closed: this combination corrupts the attention
                # output (the side-buffer misses COW'd prefix blocks), so
                # refusing to start beats logging an error and serving
                # wrong tokens.
                raise RuntimeError(
                    "GFX906_FA_LEGACY=0 with prefix caching enabled: the "
                    "Q8 K side-buffer misses COW'd prefix blocks and the "
                    "attention output will be CORRUPT. Disable prefix "
                    "caching or use the default GFX906_FA_LEGACY=1.")
            logger.warning(
                "GFX906_FA_LEGACY=0 (Q8 side-buffer path) is experimental: "
                "inconsistent with FULL cudagraph decode capture and with "
                "prefix caching. Use the default LEGACY=1 unless you know "
                "why.")
        mode = _os.environ.get("GFX906_FA_CG", "decode").lower()
        if mode == "always":
            return AttentionCGSupport.ALWAYS
        if mode == "decode":
            # With spec decode every decode step has query len
            # 1 + num_speculative_tokens; declaring
            # UNIFORM_SINGLE_TOKEN_DECODE makes vLLM demote the whole
            # engine to PIECEWISE for spec runs (measured ~3x step cost
            # on gfx906, 2026-08-18). The Q8 paged kernel is q_len-
            # generic (seq_q + q_abs_offset inline causal) and the
            # LEGACY=1 inline-quant path is FULL-capture safe (P3-3a M2),
            # so UNIFORM_BATCH (the level that covers spec decodes) is
            # the correct declaration when spec tokens are configured.
            if vllm_config.num_speculative_tokens > 0:
                return AttentionCGSupport.UNIFORM_BATCH
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Gfx906FAMetadata:
        # FULL decode capture: metadata is runner-staged at capacity
        # (profile_seq_lens) and re-read live at replay — see
        # get_cudagraph_support (M2).
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Gfx906FAMetadata:
        return Gfx906FAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            use_cascade=(common_prefix_len > 0),
            common_prefix_len=common_prefix_len,
        )


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------
class Gfx906FABackend(AttentionBackend):
    accept_output_buffer: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]
    # Only fp16 KV is supported (Q8 quantization happens on the fly
    # in the kernel; the cache itself stays fp16).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "half",
    ]

    # KV writes are a separate call (triton_reshape_and_cache_flash)
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        # Must match the name in AttentionBackendEnum so vLLM can
        # resolve it via AttentionBackendEnum[name]; registered as
        # CUSTOM, hence the name.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Gfx906FAImpl"]:
        return Gfx906FAImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # Identical to TritonAttentionBackend, so backends can be
        # switched without re-allocating the KV cache.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["Gfx906FAMetadataBuilder"]:
        return Gfx906FAMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Kernel validated for 64/128; 256 (Qwen3.5/3.6) added.
        return head_size in (64, 128, 256)

    @classmethod
    def supports_sliding_window(cls) -> bool:
        # Per-row window cutoff in the tile kernel (window arg, Muse
        # Glimmer iRoPE 2048). Gated on q_abs_offset, which
        # forward_paged supplies for every windowed batch (decode
        # included).
        #
        # Validated envelope (tests/kernels/attention/test_gfx906_fa.py,
        # test_forward_sliding_window* x8): W in {64..L, 2048-serving},
        # head_dim 128 (GQA 16:1) and 256 (GQA 8:1), decode B=1/B=2 and
        # per-row prefill, on BOTH kernel copies (gather fattn-q8.cuh and
        # direct-paged fattn-q8-paged.cuh). The mask math itself is
        # shape-independent (absolute-position cutoff on k), but an
        # untested head_dim/window/batch combo landing on this backend
        # runs unverified — extend the tests before claiming new shapes.
        #
        # Phase C KV-scan clip scope (perf, not correctness): the clip
        # only fires on the direct-paged decode path (B>=2 dispatch), so
        # a workload mixing B=1 decode (gather path, no clip yet) with
        # B>=2 windowed decode gets the HBM savings on one class of
        # request and the full scan on the other — correct in both cases,
        # but the perf of "windowed decode" here depends on the dispatch
        # path a request happens to hit.
        return True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # gfx906: major=9, minor=0, patch=6 -> capability.to_int() -> 906.
        # ROCm does not always fill DeviceCapability correctly for gfx906,
        # so accept everything and rely on explicit backend selection.
        return True


# -----------------------------------------------------------------------------
# Impl
# -----------------------------------------------------------------------------
class Gfx906FAImpl(AttentionImpl):

    # ------------------------------------------------------------------
    # CLASS-LEVEL shared gather buffers (K_q8, V_fp16).
    #
    # All attention layers in one worker share these buffers — the kernel
    # reads the cache once per (seq, head, tok) and writes the contiguous
    # out per forward call. Nothing persists between forwards.
    #
    # Shared -> ONE buffer pair per worker, not per layer. That saves
    # N_layers x (K_buf + V_buf) VRAM (e.g. MiniMax: 60 layers x ~24 MiB
    # = 1.4 GB per sequence). Grow logic is the same as q_pad_buf.
    # ------------------------------------------------------------------
    _k_gather_buf: ClassVar[torch.Tensor | None] = None
    _v_gather_buf: ClassVar[torch.Tensor | None] = None
    # Keep-alive for gather-buffer generations whose base VA is baked
    # into a replaying CUDA graph — freeing one is a use-after-free on
    # replay (2026-08-19 init-time Memory Fault). Post-fix
    # (GFX906_FA_GATHER_EXACT=0) only generations that were current
    # during a stream capture are retired (tracked per generation in
    # _gather_buf_captured); eager-serving generations are freed, and
    # since Sk is a grow-only capacity, replacement happens only on
    # Hkv/D/B change — the set normally holds exactly the one
    # capture-time generation. Under EXACT (pre-fix policy) the sticky
    # _gather_captured latch retires on every Sk change — the unbounded
    # growth behind the 256k Qwen3.8-27B TP=2 prefill OOM (see
    # docs/gfx906/oom-256k-prefill.md, plan-gfx906-fa-fix.md §1).
    # Keyed by data_ptr: unique for every live tensor, and a retained
    # tensor is never freed, so its VA can never be reused and no entry
    # can ever be overwritten (a shape key could collide between two
    # same-shape generations, silently freeing a captured one).
    _gather_retired: ClassVar[dict[int, tuple[torch.Tensor, ...]]] = {}
    # Pre-fix sticky latch (EXACT mode only): any generation replaced
    # after the first capture is retired, whether or not it was baked.
    _gather_captured: ClassVar[bool] = False
    # Post-fix per-generation flag: True iff THIS generation's base VA
    # was current during a capture. Reset (not OR'd) at each allocation.
    _gather_buf_captured: ClassVar[bool] = False
    # Kill switch: GFX906_FA_GATHER_EXACT=1 restores the pre-fix
    # exact-match policy at the backend, for A/B. Default: the new
    # grow-only capacity policy. Read once at import here AND as
    # _GATHER_EXACT in gfx906_fa_paged.py — keep the two in sync;
    # flipping only one site produces a split (meaningless) A/B.
    # TEMPORARY A/B arm, NOT a permanent knob (plan-gfx906-fa-fix.md
    # §6): drop at the NEXT gather-lifecycle change — both read sites,
    # the _gather_exact branches in _ensure_gather_buffers, the
    # _GATHER_EXACT branch in forward_paged, and
    # test_gather_exact_killswitch_restores_old_policy — re-gated on
    # a serving A/B. Until then every lifecycle edit must touch BOTH
    # policies and keep them divergence-free.
    _gather_exact: ClassVar[bool] = (
        _os.environ.get("GFX906_FA_GATHER_EXACT", "0") == "1")
    # One-shot warning once more than one capture-baked generation has
    # been retired (capture-order coupling or repeated captures — see
    # _ensure_gather_buffers).
    _gather_retired_warned: ClassVar[bool] = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("GFX906_FA: alibi_slopes unsupported")
        if logits_soft_cap not in (None, 0, 0.0):
            raise NotImplementedError(
                "GFX906_FA: logits_soft_cap unsupported")
        if sinks is not None:
            raise NotImplementedError("GFX906_FA: sinks unsupported")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"GFX906_FA: attn_type={attn_type} unsupported"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        # Sliding-window size in tokens (0 for full-attention layers).
        # The FA kernel masks keys older than the per-row window when
        # window > 0; forward_paged passes q_abs_offset for every
        # windowed batch (decode included) so the per-row formula holds.
        # GFX906_FA_NO_WINDOW (truthy = anything except "0", matching the
        # sibling knobs) forces 0 for every layer — perf-only A/B arm that
        # separates the CUSTOM-vs-ROCM_ATTN kernel-family speedup from the
        # window-mask cost (numerically WRONG for windowed layers; warn,
        # like the GFX906_FA_LEGACY precedent).
        if _os.environ.get("GFX906_FA_NO_WINDOW", "0") != "0":
            self.sliding_window = 0
            logger.warning(
                "GFX906_FA_NO_WINDOW is set: sliding-window masking is "
                "disabled on all layers — output is WRONG for windowed "
                "layers beyond the window (perf A/B arm only).")
        else:
            self.sliding_window = sliding_window or 0
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = num_heads // num_kv_heads

        # ------------------------------------------------------------------
        # Q8_0 side-buffer for K.
        #
        # vLLM allocates the primary K/V cache in fp16 (shape from
        # get_kv_cache_shape). We keep a side-buffer in block_q8_0 format
        # in parallel so forward does not re-quantize every step.
        #
        # Allocation is lazy in the first do_kv_cache_update (that is when
        # kv_cache.shape is known). Live layout:
        #   [num_blocks, block_size, Hkv, (D/32)*34]  uint8
        # Size: num_blocks * block_size * Hkv * D * 34/32 bytes
        #       = K_fp16_bytes * (34/32) / 2 ~= 0.53 x K_fp16 bytes.
        # ------------------------------------------------------------------
        self._k_cache_q8: torch.Tensor | None = None
        self._legacy = _os.environ.get("GFX906_FA_LEGACY", "1") == "1"

        # ------------------------------------------------------------------
        # Pre-allocated buffers for forward_paged. The sizes would come
        # from vLLM max_num_seqs x max_model_len, but those are not
        # reachable in this context, so we use lazy grow.
        # ------------------------------------------------------------------
        self._q_pad_buf: torch.Tensor | None = None
        # Sq=1 decode: dedicated [B, Hq, 2, D] fp32 buffer. The growing
        # _q_pad_buf slice [:B, :Hq, :2, :D] is non-contiguous after a
        # prefill-sized grow (dim2=512) and costs a copy per layer; the
        # [:num_seqs] prefix slice of this exact-shape buffer is always
        # contiguous, so the decode path needs no copy at all.
        self._q_pad_decode_buf: torch.Tensor | None = None
        # Buffers referenced by captured CUDA graphs: freed-then-realloc
        # would leave the graphs pointing at freed VAs (use-after-free on
        # replay), so retired captured buffers stay alive here.
        self._q_pad_retired: list[torch.Tensor] = []
        self._q_pad_captured: bool = False
        # Level 3a: mask_buf removed — causal is inlined in the kernel.

    def fused_output_quant_supported(self, quant_key):
        return False

    # ------------------------------------------------------------------
    # KV cache write (separate step, as in the Triton backend with
    # forward_includes_kv_cache_update=False)
    # ------------------------------------------------------------------
    def _ensure_q8_sidebuffer(self, key_cache: torch.Tensor) -> None:
        """Lazy-allocate the Q8 side-buffer to match the K-cache size."""
        if self._k_cache_q8 is not None:
            return
        # key_cache shape: [num_blocks, block_size, Hkv, D]  fp16
        num_blocks, block_size, Hkv, D = key_cache.shape
        assert D % 32 == 0, f"D={D} must be multiple of 32"
        bytes_per_row = (D // 32) * 34
        self._k_cache_q8 = torch.empty(
            (num_blocks, block_size, Hkv, bytes_per_row),
            dtype=torch.uint8,
            device=key_cache.device,
        )
        # Zero explicitly — in Q8_0, zeros decode to 0.
        self._k_cache_q8.zero_()

    def _ensure_forward_buffers(
        self,
        num_seqs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Lazy/grow-allocate the q_pad buffer.

        Level 3a: mask_buf removed — causality is inlined in the kernel.
        For a 60K prefill this saves ~480 MB of fp16 mask.
        """
        ncols1 = _pick_ncols1(max_seqlen_q)
        Sq_pad = ((max_seqlen_q + ncols1 - 1) // ncols1) * ncols1

        # q_pad_buf is only READ on multi-token forwards (Sq=1 decode uses
        # q_pad_decode_buf in both the gather and direct-paged branches),
        # so decode batches must not grow its dim0: a capture at B=8 would
        # otherwise balloon every layer's prefill buffer to (8, Hq, 2048,
        # 128) fp32 = 268 MB (Muse 30B: x52 layers = 14 GiB, first-request
        # OOM on the inductor prefill buffer, 2026-08-26).
        #
        # Spec-decode note: a verify step carries num_speculative_tokens+1
        # rows per sequence, so max_seqlen_q > 1 and it DOES read
        # q_pad_buf (forward_paged's Sq=1 fast paths are gated on
        # max_seqlen_q == 1). That is correct, and its Sq_pad is bounded
        # by the spec depth (small: (n_spec+1) up to the ncols1 tile), so
        # the decode-time dim0 growth it triggers stays ~B x 128 KB/layer
        # — the 14 GiB pathology required a prefill-sized Sq_pad, which
        # only real prefill rows produce. If a spec-decode config ever
        # shows q_pad growth again, this comment is where to start.
        qpad_num_seqs = num_seqs if max_seqlen_q > 1 else 1

        # Q buffer: [B, Hq, Sq_pad, D] fp32 (the kernel takes fp32 q; the
        # caller passes dtype=torch.float32).
        # Capture-safety: a CUDA graph bakes in the VA of the buffer that
        # was current when it was captured. Growing by free-then-realloc
        # would leave captured graphs pointing at freed memory on replay,
        # so once any buffer has been current during a capture, retired
        # buffers are kept alive (they are decode-sized; prefill-sized
        # buffers only ever exist eagerly). No empty_cache() here: it is
        # illegal during capture, and the caching allocator already
        # reuses freed blocks for the next allocation.
        #
        # The capture-state poll runs only until the first capture is
        # detected (_q_pad_captured latches); steady state costs nothing.
        if self._q_pad_buf is None:
            self._q_pad_buf = torch.empty(
                (qpad_num_seqs, self.num_heads, Sq_pad, self.head_size),
                dtype=dtype, device=device,
            )
            self._q_pad_captured = torch.cuda.is_current_stream_capturing()
        elif (self._q_pad_buf.shape[0] < qpad_num_seqs
                or self._q_pad_buf.shape[2] < Sq_pad
                or self._q_pad_buf.dtype != dtype):
            capturing = torch.cuda.is_current_stream_capturing()
            cur = self._q_pad_buf
            new_shape = (
                max(qpad_num_seqs, cur.shape[0]),
                self.num_heads,
                max(Sq_pad, cur.shape[2]),
                self.head_size,
            )
            if self._q_pad_captured or capturing:
                self._q_pad_retired.append(cur)
            self._q_pad_buf = torch.empty(new_shape, dtype=dtype, device=device)
            self._q_pad_captured = self._q_pad_captured or capturing
        elif not self._q_pad_captured:
            # No grow: latch the flag the first time we serve a forward
            # during capture (the buffer VA is being baked into a graph).
            self._q_pad_captured = torch.cuda.is_current_stream_capturing()

        # Sq=1 decode buffer [B, Hq, 2, D] fp32 (Sq_pad=2 for Sq=1).
        # Grows dim0 only; capture-safe via the shared retired list.
        if max_seqlen_q == 1:
            if self._q_pad_decode_buf is None:
                self._q_pad_decode_buf = torch.empty(
                    (num_seqs, self.num_heads, 2, self.head_size),
                    dtype=torch.float32, device=device,
                )
                self._q_pad_captured = (
                    self._q_pad_captured
                    or torch.cuda.is_current_stream_capturing())
            elif self._q_pad_decode_buf.shape[0] < num_seqs:
                capturing = torch.cuda.is_current_stream_capturing()
                if self._q_pad_captured or capturing:
                    self._q_pad_retired.append(self._q_pad_decode_buf)
                self._q_pad_decode_buf = torch.empty(
                    (num_seqs, self.num_heads, 2, self.head_size),
                    dtype=torch.float32, device=device,
                )
                self._q_pad_captured = (
                    self._q_pad_captured or capturing)

    @classmethod
    def _ensure_gather_buffers(
        cls,
        num_seqs: int,
        num_kv_heads: int,
        max_seqlen_k: int,
        head_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K_q8 + V_fp16 gather buffers, shared by all layers in a worker.

        shape:
            K: [B, Hkv, Sk, (D/32)*34]  uint8
            V: [B, Hkv, Sk, D]          fp16

        Post-fix policy (default; GFX906_FA_GATHER_EXACT=0):
        the Sk dimension is a *capacity*, not an exact shape. Any buffer
        wide enough (>= Sk_pad) is reused. Replacement sizing depends on
        the old generation's fate: one being retired (capture-baked)
        keeps capacity per axis — max(needed, existing), the in-tree
        _q_pad_buf pattern; one being freed (never capture-baked) is
        replaced at exact need, so a freeable generation never inherits
        dead capacity (see the sizing comment at the replacement site).
        FULL capture runs at max_model_len
        (build_for_cudagraph_capture), so after the first capture the
        buffer spans every seq_len the engine can schedule and eager
        chunked prefill reuses it forever — no per-chunk realloc/retire
        churn (the unbounded _gather_retired growth that drained the
        ~2 GiB post-capture headroom in the 256k Qwen3.8-27B TP=2
        prefill; docs/gfx906/oom-256k-prefill.md,
        plan-gfx906-fa-fix.md). Width-wider-than-live is safe: the
        persistent gather's work is live-bounded (device-side seq_lens)
        and the FA kernel cuts each sequence at kv_max, never at Sk.

        A request for a smaller B than the current buffer returns a
        leading-dim slice [:B] — zero-copy, contiguous, and same base VA
        (a non-leading-dim slice would copy, which is why Hkv/D changes
        realloc instead). Because FULL-graph capture bakes the base VA of
        ONE generation into every captured batch size, this slice reuse
        is what keeps a single generation alive across the whole capture
        sweep (up to ~35 batch sizes at max_num_seqs=256).

        Capture-safety: the generation current during a stream capture is
        baked into the graph by VA. _gather_buf_captured tracks that
        *per generation* — reset to the capture state at each allocation,
        latched forward only within a generation's no-realloc path — so
        only graph-baked generations are retired into _gather_retired on
        replacement. Eager-serving generations are never baked (attention
        is a splitting op — vllm::unified_attention_with_output in the
        default splitting_ops — so eager forwards run outside any
        captured subgraph) and are simply freed; the caching allocator
        reuses the blocks. Bounded: post-fix replacement happens only on
        Hkv/D/B change, so the set normally holds exactly one generation.
        (splitting_ops=[] — attention inside piecewise graphs — is still
        safe, at one extra retained generation: such generations latch
        captured and are retired, not freed.)

        Pre-fix policy (GFX906_FA_GATHER_EXACT=1, A/B kill switch):
        exact-Sk match and the sticky _gather_captured latch — the old
        behavior byte-for-byte, unbounded retire growth included.
        """
        Sk_pad = ((max_seqlen_k + 31) // 32) * 32
        bytes_per_row = (head_size // 32) * 34
        capturing = torch.cuda.is_current_stream_capturing()

        cur = cls._k_gather_buf
        if cur is not None:
            fit = (cur.shape[1] == num_kv_heads
                   and cur.shape[3] == bytes_per_row
                   and cur.device == device
                   and cur.shape[0] >= num_seqs
                   and (cur.shape[2] == Sk_pad if cls._gather_exact
                        else cur.shape[2] >= Sk_pad))
            if fit:
                # Reuse; latch the capture flag within THIS generation's
                # lifetime only (the buffer VA may be getting baked into
                # a graph right now).
                if cls._gather_exact:
                    if not cls._gather_captured:
                        cls._gather_captured = capturing
                else:
                    if not cls._gather_buf_captured:
                        cls._gather_buf_captured = capturing
                if cur.shape[0] == num_seqs:
                    # Exact size: no view (the hot decode path calls this
                    # per FA layer per step; a fresh TensorImpl per call
                    # is measurable on short steps).
                    return cur, cls._v_gather_buf
                # A leading-dim slice is contiguous and has the same base
                # VA and row layout as the parent, so the kernel and every
                # graph captured against this base see identical addresses.
                return cur[:num_seqs], cls._v_gather_buf[:num_seqs]

        # First allocation, or a growth/shape change: retire-or-free the
        # old generation, then allocate.
        new_b = num_seqs
        new_sk = Sk_pad
        if cur is not None:
            keep = ((cls._gather_captured or capturing)
                    if cls._gather_exact
                    else cls._gather_buf_captured)
            if keep:
                # Graph-baked VA: must outlive every replay.
                cls._gather_retired[cur.data_ptr()] = (
                    cur, cls._v_gather_buf)
                if (not cls._gather_exact
                        and not cls._gather_retired_warned
                        and len(cls._gather_retired) > 1):
                    # Post-fix, ONE capture-baked generation is the norm
                    # (the FULL-capture sweep reuses a single base VA
                    # across every captured batch size). A second retire
                    # means either the sweep itself retired generations —
                    # capture-order coupling (plan §2.2b; capture-time OOM
                    # possible) — or repeated re-captures / Hkv-D flaps.
                    logger.warning(
                        "GFX906_FA: %d retired capture-baked gather "
                        "generations (expected <= 1; capture-order "
                        "coupling — see plan-gfx906-fa-fix.md §2.2b)",
                        len(cls._gather_retired))
                    cls._gather_retired_warned = True
                if not cls._gather_exact:
                    # The retired generation's block stays resident
                    # anyway, so keeping its capacity costs nothing: a
                    # FULL capture spans B=max_num_seqs x
                    # Sk=max_model_len and later needs never exceed it.
                    new_b = max(num_seqs, cur.shape[0])
                    new_sk = max(Sk_pad, cur.shape[2])
            # else, capacity mode + never capture-baked: the replacement
            # allocates at exact need (new_b/new_sk as initialized).
            # Grow-only max() on BOTH axes here would pin the
            # B-highwater x Sk-highwater product forever — 32-way
            # short-context decode then one 250k prefill leaves a
            # [32, 262144] standing buffer (~13 GB/rank at the arm-B
            # geometry) that no single request ever needed. Reallocation
            # frequency is unchanged (a replacement happens exactly when
            # the current buffer no longer fits, and this block is freed
            # either way); only the new allocation stops inheriting dead
            # capacity. In FULL modes this branch runs only before the
            # first capture (post-capture generations are capture-baked
            # and take the keep path), so capture-time sizing — the
            # validated arm-B path — is unaffected.
        cls._k_gather_buf = torch.empty(
            (new_b, num_kv_heads, new_sk, bytes_per_row),
            dtype=torch.uint8, device=device,
        )
        cls._v_gather_buf = torch.empty(
            (new_b, num_kv_heads, new_sk, head_size),
            dtype=torch.float16, device=device,
        )
        if cls._gather_exact:
            cls._gather_captured = cls._gather_captured or capturing
        else:
            # Per generation: the new buffer starts from its own state,
            # not the old generation's capture history.
            cls._gather_buf_captured = capturing
        b, v = cls._k_gather_buf, cls._v_gather_buf
        if b.shape[0] == num_seqs:
            return b, v
        return b[:num_seqs], v[:num_seqs]

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)

        # 1) Primary fp16 write — the vLLM-standard path for V (and for K
        #    in LEGACY mode).
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # 2) In parallel into the Q8 side-buffer for K (fast-path
        #    forward). Skipped in LEGACY mode — forward quantizes on the
        #    fly.
        # IMPORTANT (RC2, sub-plan plan-gfx906fa-serving.md): the Q8
        # side-buffer path (LEGACY=0) is inconsistent whenever the fp16
        # kv_cache holds data written OUTSIDE our do_kv_cache_update
        # (vLLM warmup / profile_run / dummy forwards, torch.compile
        # captures, COW'd prefix-cache blocks, graph replay writes).
        # _k_cache_q8 then lags the fp16 cache -> forward reads Q8=0 and
        # produces garbage. Until fixed, LEGACY=1 (inline-quantize) is
        # the default.
        if not self._legacy:
            self._ensure_q8_sidebuffer(key_cache)
            from vllm import _gfx906_fa_C as gfx906_fa
            if slot_mapping.dtype != torch.int64:
                slot_mapping = slot_mapping.to(torch.int64)
            gfx906_fa.reshape_and_cache_q8(
                key.contiguous() if not key.is_contiguous() else key,
                slot_mapping,
                self._k_cache_q8,
            )

    def fused_rope_kvcache_supported(self):
        return False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,      # [num_tokens, num_heads, head_size]
        key: torch.Tensor,        # [num_tokens, num_kv_heads, head_size]
                                 # (already in kv_cache via do_kv_cache_update)
        value: torch.Tensor,
        kv_cache: torch.Tensor,   # [num_blocks, 2, block_size, num_kv_heads, head_size]
        attn_metadata: Gfx906FAMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "GFX906_FA: output quantization unsupported")

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert not attn_metadata.use_cascade, (
            "GFX906_FA: cascade unsupported")

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Unbind KV cache: (..., 2, ...) → (K, V) each [num_blocks, block_size, Hkv, D]
        key_cache, value_cache = kv_cache.unbind(1)

        # query [num_tokens, Hq, D] fp16 (forward_paged casts it into the
        # fp32 q_pad buffer inside the copy_ — a standalone .float() was
        # an extra kernel per layer).
        q_actual = query[:num_actual_tokens]
        out_actual = output[:num_actual_tokens]

        # Lazy-grow the forward buffers (q_pad / mask).
        num_seqs = attn_metadata.seq_lens.shape[0]
        self._ensure_forward_buffers(
            num_seqs=num_seqs,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            device=query.device,
            dtype=torch.float32,
        )

        # Fused-gather output buffers (class-level, shared across layers).
        # Used by both the LEGACY=0 fused Q8 gather and the LEGACY=1
        # fused gather+quantize; without reuse every attention layer
        # allocates 24-200+ MiB per step on long contexts.
        k_gather_buf, v_gather_buf = self._ensure_gather_buffers(
            num_seqs=num_seqs,
            num_kv_heads=self.num_kv_heads,
            max_seqlen_k=attn_metadata.max_seq_len,
            head_size=self.head_size,
            device=query.device,
        )

        # forward_paged returns [num_tokens, Hq*D] float32.
        # Fast path: pass the Q8 side-buffer when present.
        out_flat = forward_paged(
            query=q_actual,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            scale=self.scale,
            key_cache_q8=self._k_cache_q8 if not self._legacy else None,
            q_pad_buf=self._q_pad_buf,
            q_pad_decode_buf=self._q_pad_decode_buf,
            k_gather_buf=k_gather_buf,
            v_gather_buf=v_gather_buf,
            window=self.sliding_window,
        )  # [num_tokens, Hq*D] fp32

        # Write the result into output in-place (it is either
        # [num_tokens, Hq, D] or [num_tokens, Hq*D], depending on the
        # caller). copy_ fuses the fp32->fp16 cast (a .to() first was
        # an extra kernel per layer).
        out_view = out_actual.view(num_actual_tokens, -1)
        out_view.copy_(out_flat)

        return output


# -----------------------------------------------------------------------------
# Auto-register as CUSTOM on import
# -----------------------------------------------------------------------------
def register() -> None:
    """Register Gfx906FABackend as AttentionBackendEnum.CUSTOM.

    Called automatically on module import (and available to user code).
    No-op off gfx906: the extension only exists in gfx906 builds, and
    registering the backend elsewhere would make a broken backend
    selectable via VLLM_ATTENTION_BACKEND=CUSTOM.
    """
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx906

    if not (current_platform.is_rocm() and on_gfx906()):
        return
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        f"{__name__}.Gfx906FABackend",
    )
    logger.info("GFX906_FA backend registered as AttentionBackendEnum.CUSTOM")


# Auto-register on module import
register()
