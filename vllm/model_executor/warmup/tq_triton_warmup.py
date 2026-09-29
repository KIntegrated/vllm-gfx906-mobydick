# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up TurboQuant (TQ) Triton kernels before inference starts.

Why this exists
---------------
`_tq_decode_stage1`, `_tq_full_dequant_kv` and `_tq_fused_store_mse` are
`@triton.jit` kernels whose compile keys depend on per-layer constants plus the
integer strides and pointer alignments of the tensors handed to them. On gfx906
the TQ path is first reached *during* inference (long-context continuation
prefill), so the JIT monitor reports ``Triton kernel JIT compilation during
inference`` and the compile lands in the middle of a latency-sensitive request.

This module compiles them ahead of time by *running* them on minimal tensors,
following the same shape of solution as `qwen_triton_warmup`: probe layers by
attribute (never import the concrete impl class), run a micro-tensor, then
synchronize.

Key-space rules (measured, not assumed)
---------------------------------------
The compile key covers, per parameter: dtype, pointer alignment, the
``divisibility`` of raw integer arguments, ``tl.constexpr`` values and the
launch options. For the TQ decode kernel the varying part was enumerated from
the on-disk cache of a full run (6 entries under one kernel name) by diffing
the persisted ``ASTSource``: exactly three binary dimensions, everything else
identical (the ``.llir``/``.amdgcn`` artefacts are byte-identical, so only the
specialization attributes differ).

1. 16-byte alignment of ``Block_table_ptr``. The continuation fast path passes
   ``block_table[i : i + 1].expand(q_len, -1)`` -- a *row slice*, so the base
   address is offset by ``i * row_bytes`` and is unaligned for odd ``i``.
2. 16-byte alignment of ``Seq_lens_ptr``. The same path passes
   ``_arange_cache[cached_len + 1 : seq_len + 1]``, an offset slice.
3. ``divisibility`` of ``stride_bt_b``. It is handed over in *bytes*, so it is
   divisible by 16 exactly when the block-table row width is (measured: row
   width 156 -> 624 B -> divisible; 157 -> 628 B -> not). The ``.expand`` form
   contributes a zero stride, which is divisible. The runner's row width is one
   larger than ``ceil(max_model_len / block_size)``, so it is read from the
   runner rather than recomputed.

Therefore the decode warmup sweeps that 8-way lattice. It cannot cover the
separate *hash* of a shape it never launches, but it does cover every
specialization combination the runtime has been observed to produce.

Further key-equivalence rules
-----------------------------
* Only the leading ``num_blocks`` dimension of the TQ cache is shrunk. Every
  other extent -- and hence every stride passed to the kernel -- is inherited
  from the layer's *real* cache by slicing, so integer specializations match
  the runtime ones.
* ``block_table`` row width comes from the runner (see above); the fallback is
  ``ceil(max_model_len / block_size) + 1``.
* The decode kernel is launched with ``seq_len == 0``. `_tq_decode_stage1`
  returns before touching memory in that case (``split_start >= split_end``),
  so the launch is compile-only. This holds for the unaligned and expanded
  forms too: the loaded sequence lengths are genuine zeros.
* The store kernel is launched with ``slot_mapping = -1``. Both store variants
  return immediately on a negative slot, so nothing is written; only the
  compile happens. This matters: a store warmup with a valid slot would
  scribble on the real cache's first blocks.
* The full-dequant kernel has no early return, so it does read the real cache
  block 0 -- but it only ever writes into scratch buffers allocated here, never
  into the cache.
"""

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _next_power_of_2(n: int) -> int:
    n = int(n)
    return 1 << (n - 1).bit_length() if n > 1 else 1


@dataclass(frozen=True)
class _TqSpec:
    """Everything a TQ Triton kernel compile key can depend on."""

    key: tuple
    head_dim: int
    num_heads: int
    num_kv_heads: int
    block_size: int
    table_row_width: int
    max_num_kv_splits: int
    scale: float
    mse_bits: int
    mse_bytes: int
    val_data_bytes: int
    key_packed_size: int
    value_quant_bits: int
    key_fp8: bool
    norm_correction: bool
    block_d: int


def _impl_of(module: object) -> Any:
    """Return the `AttentionImpl` behind a forward-context entry, if any."""
    impl = getattr(module, "impl", None)
    if impl is not None:
        return impl
    # Some layer types register the impl itself.
    if hasattr(module, "tq_config"):
        return module
    return None


def _is_tq_impl(impl: object) -> bool:
    """Attribute probe for the TurboQuant attention impl.

    Deliberately avoids importing the concrete class: these three attributes
    are set together in `TurboQuantAttentionImpl.__init__` and nowhere else.
    """
    return (
        hasattr(impl, "tq_config")
        and hasattr(impl, "num_kv_groups")
        and hasattr(impl, "max_num_kv_splits")
    )


def _table_row_width(max_model_len: int, block_size: int) -> int:
    if max_model_len <= 0 or block_size <= 0:
        return 1
    return max(1, math.ceil(max_model_len / block_size))


def _resolve_row_width(
    worker: object,
    runner: object,
    max_model_len: int,
    block_size: int,
) -> tuple[int, str]:
    """Row width of the runner's block table, taken from its definition point.

    `stride_bt_b` reaches the kernel in bytes, so the row width decides whether
    it is divisible by 16 -- i.e. it is part of the compile key. Recomputing it
    from `max_model_len` alone yields one less than the runner's value, which
    lands on the wrong side of that bit, so read the runner's own number first
    and only fall back to the formula (plus one) when it cannot be reached.
    """
    holders = (
        runner,
        getattr(runner, "input_batch", None),
        getattr(runner, "block_table", None),
        worker,
    )
    for holder in holders:
        if holder is None:
            continue
        tables = getattr(holder, "block_table", None)
        for table in (tables, holder):
            if table is None:
                continue
            for attr in ("max_num_blocks_per_req", "max_num_blocks"):
                value = getattr(table, attr, None)
                if isinstance(value, int) and value > 0:
                    return value, "runner.block_table.{}".format(attr)
    return _table_row_width(max_model_len, block_size) + 1, "fallback_plus1"


def _tq_layer_specs(
    static_forward_context: object,
    *,
    max_model_len: int,
    row_width: int,
) -> tuple[list[tuple[Any, Any, _TqSpec, torch.Tensor]], int]:
    """Collect (module, impl, spec, cache_view) for every distinct TQ layer."""
    if not isinstance(static_forward_context, dict):
        return [], 0

    targets: list[tuple[Any, Any, _TqSpec, torch.Tensor]] = []
    seen_keys: set[tuple] = set()
    n_tq_layers = 0

    for module in static_forward_context.values():
        impl = _impl_of(module)
        if impl is None or not _is_tq_impl(impl):
            continue
        cache = getattr(module, "kv_cache", None)
        if not isinstance(cache, torch.Tensor) or cache.dim() != 4:
            continue
        n_tq_layers += 1

        # The TQ backend allocates (num_blocks, num_kv_heads, block_size, slot)
        # and the impl views it transposed as (num_blocks, block_size, Hk, slot)
        # before every launch. Mirror that view exactly. After the transpose the
        # extents are (num_blocks, block_size, num_kv_heads, slot): dim 1 is the
        # page size and dim 2 the head count. Reading the page size off dim 2
        # used to hand the dequant kernel BLOCK_SIZE=num_kv_heads.
        cache_view = cache.transpose(1, 2)
        if int(cache_view.shape[2]) != int(impl.num_kv_heads):
            logger.warning(
                "TurboQuant Triton warmup: unexpected cache layout %s "
                "(num_kv_heads=%d); skipping this layer.",
                tuple(cache.shape),
                int(impl.num_kv_heads),
            )
            continue

        block_size = int(cache_view.shape[1])
        spec = _TqSpec(
            key=(
                int(impl.head_size),
                int(impl.num_heads),
                int(impl.num_kv_heads),
                block_size,
                row_width,
                int(impl.max_num_kv_splits),
                bool(impl.tq_config.key_fp8),
                int(impl.tq_config.key_mse_bits),
                int(impl.tq_config.key_packed_size),
                int(impl.tq_config.effective_value_quant_bits),
                bool(impl.tq_config.norm_correction),
            ),
            head_dim=int(impl.head_size),
            num_heads=int(impl.num_heads),
            num_kv_heads=int(impl.num_kv_heads),
            block_size=block_size,
            table_row_width=row_width,
            max_num_kv_splits=int(impl.max_num_kv_splits),
            scale=float(impl.scale),
            mse_bits=int(impl.tq_config.key_mse_bits),
            mse_bytes=int(impl._mse_bytes),
            val_data_bytes=int(impl._val_data_bytes),
            key_packed_size=int(impl.tq_config.key_packed_size),
            value_quant_bits=int(impl.tq_config.effective_value_quant_bits),
            key_fp8=bool(impl.tq_config.key_fp8),
            norm_correction=bool(impl.tq_config.norm_correction),
            block_d=_next_power_of_2(int(impl.head_size)),
        )
        if spec.key in seen_keys:
            continue
        seen_keys.add(spec.key)
        targets.append((module, impl, spec, cache_view))

    return targets, n_tq_layers


def _buffers_of(layer: Any, spec: _TqSpec, device: torch.device) -> dict:
    """Collect the per-layer TQ buffers, synthesizing stand-ins if missing.

    None of these participate in the compile key, so a stand-in is fine when
    `_ensure_on_device` could not run.
    """
    d = spec.head_dim
    pit = getattr(layer, "_tq_PiT", None)
    pi = getattr(layer, "_tq_Pi", None)
    if pit is None or pi is None:
        pi = torch.eye(d, dtype=torch.float32, device=device)
        pit = pi
    n_centroids = 1 if spec.key_fp8 else 2**spec.mse_bits
    centroids = getattr(layer, "_tq_centroids", None)
    if centroids is None:
        centroids = torch.zeros(n_centroids, dtype=torch.float32, device=device)
    midpoints = getattr(layer, "_tq_midpoints", None)
    if midpoints is None:
        midpoints = torch.zeros(
            max(1, n_centroids - 1), dtype=torch.float32, device=device
        )
    return {"Pi": pi, "PiT": pit, "centroids": centroids, "midpoints": midpoints}


def _decode_variants(spec: _TqSpec, device: torch.device):
    """One (block_table, seq_lens) pair per decode specialization combination.

    The three binary dimensions are described in the module docstring; each is
    produced by construction rather than by hoping it matches:

    * unaligned ``Block_table_ptr``: a view starting one int32 (4 bytes) into a
      freshly allocated buffer, mirroring a row slice;
    * unaligned ``Seq_lens_ptr``: the same trick on the sequence-length buffer;
    * zero ``stride_bt_b``: ``expand`` of a single row, which is what the
      continuation path passes.

    Every variant carries genuine zero sequence lengths, so the kernel returns
    before loading the block table and the cache is never read.
    """
    w = max(1, int(spec.table_row_width))
    variants = []
    for bt_unaligned in (False, True):
        for sl_unaligned in (False, True):
            for expanded in (False, True):
                rows = 2 if expanded else 1
                pad = 1 if bt_unaligned else 0
                bt_raw = torch.zeros(rows * w + pad, dtype=torch.int32, device=device)
                bt = bt_raw[pad:].view(rows, w)
                if expanded:
                    # Leading extent 1 -> expand gives a zero row stride.
                    bt = bt[:1].expand(rows, -1)
                sp = 1 if sl_unaligned else 0
                sl_raw = torch.zeros(rows + sp, dtype=torch.int32, device=device)
                sl = sl_raw[sp : sp + rows]
                name = "bt_{}_sl_{}_{}".format(
                    "un" if bt_unaligned else "al",
                    "un" if sl_unaligned else "al",
                    "expand" if expanded else "rows",
                )
                variants.append((name, rows, bt, sl))
    return variants


def _warm_decode(
    spec: _TqSpec,
    cache: torch.Tensor,
    device: torch.device,
    bufs: dict,
    rows: int,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        triton_turboquant_decode_attention,
    )

    query = torch.zeros(
        rows, spec.num_heads, spec.head_dim, dtype=torch.float32, device=device
    )

    triton_turboquant_decode_attention(
        query=query,
        kv_cache=cache[:2],
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=bufs["Pi"],
        centroids=bufs["centroids"],
        scale=spec.scale,
        mse_bits=spec.mse_bits,
        key_packed_size=spec.key_packed_size,
        value_quant_bits=spec.value_quant_bits,
        key_fp8=spec.key_fp8,
        norm_correction=spec.norm_correction,
        PiT=bufs["PiT"],
        max_num_kv_splits=spec.max_num_kv_splits,
    )


def _dequant_variants(spec: _TqSpec, device: torch.device):
    """One block table per ``_tq_full_dequant_kv`` specialization combination.

    The kernel key varies along two binary dimensions, and both of them are
    properties of the block table alone:

    * ``Block_table_ptr`` 16-byte alignment -- a view starting one int32 into
      a freshly allocated buffer, mirroring a row slice or an odd offset;
    * ``stride_bt_b`` divisibility by 16 -- a row width that is off the 16
      grid, or a single expanded row whose stride is zero (zero is divisible).

    The width used by the non-divisible branch is nudged off the 16 grid when
    the caller's own width happens to be a multiple of 16; the two widths
    differ by a single element and nothing reads the table (contents stay
    zero), so this cannot change behaviour -- only the specialization.
    """
    base = max(1, int(spec.table_row_width))
    variants = []
    for bt_unaligned in (False, True):
        for stride_div16 in (False, True):
            pad = 1 if bt_unaligned else 0
            if stride_div16:
                raw = torch.zeros(pad + base, dtype=torch.int32, device=device)
                bt = raw[pad:].view(1, base).expand(2, -1)
            else:
                w = base if base % 16 else base + 1
                raw = torch.zeros(pad + w, dtype=torch.int32, device=device)
                bt = raw[pad:].view(1, w)
            name = "bt_{}_stride_{}".format(
                "un" if bt_unaligned else "al",
                "div16" if stride_div16 else "off16",
            )
            variants.append((name, bt))
    return variants


def _warm_full_dequant(
    spec: _TqSpec,
    cache: torch.Tensor,
    device: torch.device,
    bufs: dict,
    block_table: torch.Tensor,
) -> None:
    """Compile ``_tq_full_dequant_kv`` for one block-table specialization.

    ``block_table`` is built by :func:`_dequant_variants`; this function only
    launches, so the caller decides which specializations get covered.
    """
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_full_dequant_kv,
        _use_fp8_e4b15,
    )

    hk = spec.num_kv_heads
    d = spec.head_dim
    alloc_len = spec.block_size
    kv = cache[:2]
    k_cached = torch.empty(1, hk, alloc_len, d, dtype=torch.float16, device=device)
    v_cached = torch.empty_like(k_cached)
    grid = (alloc_len, 1 * hk)

    _tq_full_dequant_kv[grid](
        kv,
        block_table,
        bufs["centroids"],
        k_cached,
        v_cached,
        k_cached.stride(0),
        k_cached.stride(1),
        k_cached.stride(2),
        v_cached.stride(0),
        v_cached.stride(1),
        v_cached.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        block_table.stride(0),
        HEAD_DIM=d,
        BLOCK_SIZE=spec.block_size,
        NUM_KV_HEADS=hk,
        MSE_BYTES=spec.mse_bytes,
        KPS=spec.key_packed_size,
        VQB=spec.value_quant_bits,
        VAL_DATA_BYTES=spec.val_data_bytes,
        MSE_BITS=spec.mse_bits,
        KEY_FP8=1 if spec.key_fp8 else 0,
        BLOCK_D=spec.block_d,
        NORM_CORRECTION=1 if spec.norm_correction else 0,
        FP8_E4B15=_use_fp8_e4b15(device.index or 0),
        num_warps=4,
    )


def _warm_store(
    spec: _TqSpec, cache: torch.Tensor, device: torch.device, bufs: dict
) -> None:
    from vllm.v1.attention.ops.triton_turboquant_store import (
        triton_turboquant_store,
    )

    n, h, d = 1, spec.num_kv_heads, spec.head_dim
    key = torch.zeros(n, h, d, dtype=torch.float16, device=device)
    value = torch.zeros_like(key)
    # slot < 0 => both store variants return before any store; compile only.
    slot_mapping = torch.full((n,), -1, dtype=torch.int32, device=device)

    triton_turboquant_store(
        key,
        value,
        cache[:2],
        slot_mapping,
        bufs["PiT"],
        bufs["midpoints"],
        mse_bits=spec.mse_bits,
        key_packed_size=spec.key_packed_size,
        value_quant_bits=spec.value_quant_bits,
        key_fp8=spec.key_fp8,
    )


class _TqCompileTap:
    """Record which TQ kernels compiled during a window.

    Observation only: any previous hook is preserved and forwarded verbatim.

    A compile served from the on-disk Triton cache still reaches the
    ``jit_post_compile_hook``: it is called at the end of ``_do_compile``, which
    runs whenever the in-process cache misses, and the on-disk cache only
    short-circuits the code-generation pipeline inside it. So a non-zero count
    here means "the kernel was reached and its key resolved", not necessarily
    "a fresh compilation happened" -- the two must not be conflated when
    reading the log.
    """

    def __init__(self) -> None:
        self.names: list[str] = []
        self._prev: Any = None
        self._knobs: Any = None

    def __enter__(self) -> "_TqCompileTap":
        try:
            from triton import knobs
        except Exception:  # noqa: BLE001
            return self
        self._knobs = knobs
        self._prev = getattr(knobs.runtime, "jit_post_compile_hook", None)
        tap = self

        def _hook(**kwargs: Any) -> Any:
            with contextlib.suppress(Exception):
                name = getattr(kwargs.get("fn"), "name", "")
                if name.startswith("_tq_"):
                    tap.names.append(name)
            if tap._prev is not None:
                return tap._prev(**kwargs)
            return None

        with contextlib.suppress(Exception):
            knobs.runtime.jit_post_compile_hook = _hook
        return self

    def __exit__(self, *exc: object) -> None:
        if self._knobs is not None:
            with contextlib.suppress(Exception):
                self._knobs.runtime.jit_post_compile_hook = self._prev


@torch.inference_mode()
def tq_triton_warmup(worker: "Worker") -> None:
    """Compile the TurboQuant Triton kernels before inference starts."""
    try:
        runner = getattr(worker, "model_runner", None)
        if runner is None or getattr(runner, "is_pooling_model", False):
            return

        device = getattr(worker, "device", None)
        if device is None:
            device = getattr(runner, "device", torch.device("cuda"))
        assert device is not None

        compilation_config = getattr(runner, "compilation_config", None)
        static_forward_context = getattr(
            compilation_config, "static_forward_context", None
        )
        if not static_forward_context:
            return

        vllm_config = getattr(worker, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
        cache_config = getattr(vllm_config, "cache_config", None)
        cfg_block_size = int(getattr(cache_config, "block_size", 0) or 0)

        row_width, row_width_src = _resolve_row_width(
            worker, runner, max_model_len, cfg_block_size
        )
        # One of decode's three boolean specialization dims is "is stride_bt_b
        #   divisible by 16". The non-expand branch of the synthetic block_table
        #   sets stride_bt_b to row_width, so the "not divisible" specialization
        #   is only reachable when row_width is not a multiple of 16. A probe
        #   that picks up a different quantity with the same value (for example
        #   the total block count, which is not the same thing) or lands on a
        #   multiple of 16 loses that key. Fix: bump row_width to the nearest
        #   non-multiple of 16. The synthetic tensor holds zero-length sequences
        #   and the kernel returns before reading the block table, so widening
        #   it is inert.
        if row_width % 16 == 0:
            row_width += 1
            row_width_src += "+mod16guard"

        targets, n_tq_layers = _tq_layer_specs(
            static_forward_context,
            max_model_len=max_model_len,
            row_width=row_width,
        )
        if not targets:
            logger.debug_once(
                "Skipping TurboQuant Triton warmup: no TurboQuant layer found."
            )
            return

        warmed = {"decode": 0, "dequant": 0, "store": 0}
        noted: list[str] = []
        noted_deq: list[str] = []
        failed = 0
        with _TqCompileTap() as tap:
            for module, impl, spec, cache in targets:
                # The _tq_* buffers are created lazily by the impl. The dummy
                # run has normally created them already; the call is idempotent.
                with contextlib.suppress(Exception):
                    impl._ensure_on_device(module, device)
                bufs = _buffers_of(module, spec, device)
                for name, rows, block_table, seq_lens in _decode_variants(spec, device):
                    try:
                        _warm_decode(
                            spec, cache, device, bufs, rows, block_table, seq_lens
                        )
                        warmed["decode"] += 1
                        noted.append(name)
                    except Exception:  # noqa: BLE001
                        failed += 1
                        logger.warning(
                            "TurboQuant Triton warmup: decode/%s failed for "
                            "head_dim=%d num_heads=%d num_kv_heads=%d "
                            "block_size=%d.",
                            name,
                            spec.head_dim,
                            spec.num_heads,
                            spec.num_kv_heads,
                            spec.block_size,
                            exc_info=True,
                        )
                # Same reasoning as the decode variants above: the dequant
                # key depends on the block table, so every reachable block
                # table specialization is compiled rather than one guess.
                for name, block_table in _dequant_variants(spec, device):
                    try:
                        _warm_full_dequant(spec, cache, device, bufs, block_table)
                        warmed["dequant"] += 1
                        noted_deq.append(name)
                    except Exception:  # noqa: BLE001
                        failed += 1
                        logger.warning(
                            "TurboQuant Triton warmup: dequant/%s failed for "
                            "head_dim=%d num_heads=%d num_kv_heads=%d "
                            "block_size=%d.",
                            name,
                            spec.head_dim,
                            spec.num_heads,
                            spec.num_kv_heads,
                            spec.block_size,
                            exc_info=True,
                        )
                try:
                    _warm_store(spec, cache, device, bufs)
                    warmed["store"] += 1
                    noted.append("store")
                except Exception:  # noqa: BLE001
                    failed += 1
                    logger.warning(
                        "TurboQuant Triton warmup: store failed for "
                        "head_dim=%d num_heads=%d num_kv_heads=%d "
                        "block_size=%d.",
                        spec.head_dim,
                        spec.num_heads,
                        spec.num_kv_heads,
                        spec.block_size,
                        exc_info=True,
                    )
        if device.type == "cuda":
            torch.accelerator.synchronize(device)

        logger.info(
            "TurboQuant Triton warmup: warmed %d specializations "
            "(decode=%d dequant=%d store=%d) across %d TQ layers; "
            "compiled_now=%d %s; failures=%d; "
            "row_width=%d (from %s) block_size=%d "
            "decode_variants=%s dequant_variants=%s.",
            sum(warmed.values()),
            warmed["decode"],
            warmed["dequant"],
            warmed["store"],
            n_tq_layers,
            len(tap.names),
            sorted(set(tap.names)),
            failed,
            row_width,
            row_width_src,
            targets[0][2].block_size,
            ",".join(noted),
            ",".join(noted_deq),
        )
    except Exception:  # noqa: BLE001
        # Warmup must never be able to block engine startup.
        logger.warning("TurboQuant Triton warmup failed.", exc_info=True)
