#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""LAB-10: the hybrid (mamba+attention) KV page geometry, per KV layout.

Dry-runs the hybrid block-size aligner for the alias and packed KV layouts and
prints, per arm: attention bytes/token, the mamba state page, the block size the
aligner derives, the unified page, and the mamba:attention split of one block.

Why this exists (issue #58): the packed cell shrinks the attention page ~22-23 %
but on `Nemotron-3.5-Lightning-30B-A3B` that bought only +0.4 % KV capacity,
while the 35B-A3B gained +22 %. The question is whether the mamba state is what
caps the win, and the numbers below answer the geometry half of it: in align
mode the aligner *pins* the per-layer mamba page to the attention page, and the
mamba page is per **block** (independent of block size), so a smaller attention
cell buys capacity by raising the block size, not by shrinking a page.

CPU-only (no weights, no GPU work): it needs the model's ``config.json`` only.
Validated against the boot logs of the LAB-9 A/B (see ``--expect``).

    env HF_HUB_OFFLINE=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \\
        .venv/bin/python docs/gfx906/_hybrid_kv_ratio.py <model_dir> \\
        [--mamba-cache-mode align] [--max-model-len 8192] [--expect 2096,2672]
"""

import argparse
import copy

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.gfx906_fa import gfx906_fa_backend as fa
from vllm.model_executor.models import ModelRegistry
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, get_kv_quant_mode


def build_config(path: str, mamba_mode: str, max_model_len: int) -> VllmConfig:
    model_config = ModelConfig(
        model=path,
        dtype="float16",
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    cache_config = CacheConfig(cache_dtype="auto", mamba_cache_mode=mamba_mode)
    parallel_config = ParallelConfig(tensor_parallel_size=1)
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
    )


def page_sizes(vc: VllmConfig) -> tuple[int, int]:
    """(attention bytes per token per layer, mamba state bytes per block)."""
    mc, cc, pc = vc.model_config, vc.cache_config, vc.parallel_config
    spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=mc.get_num_kv_heads(pc),
        head_size=mc.get_head_size(),
        dtype=mc.dtype,
        kv_quant_mode=get_kv_quant_mode(cc.cache_dtype),
    )
    attn_1tok = fa.Gfx906FABackend.customize_spec(spec).page_size_bytes
    model_cls, _ = ModelRegistry.resolve_model_cls(mc.architecture, model_config=mc)
    if hasattr(model_cls, "get_mamba_specs_from_config"):
        mamba_page = max(
            s.page_size_bytes for s in model_cls.get_mamba_specs_from_config(vc)
        )
    else:
        mamba_page = MambaSpec(
            shapes=model_cls.get_mamba_state_shape_from_config(vc),
            dtypes=model_cls.get_mamba_state_dtype_from_config(vc),
            block_size=-1,
        ).page_size_bytes
    return attn_1tok, mamba_page


def arm(vc: VllmConfig, mode: str) -> dict:
    """Run the aligner for one layout on a pristine config copy."""
    fa.set_packed_config_mode(mode)
    fa.reset_packed_mode()
    fa.set_packed_dry_run(True)
    try:
        attn_1tok, mamba_page = page_sizes(vc)
        dry = copy.copy(vc)
        dry.cache_config = copy.copy(vc.cache_config)
        current_platform._align_hybrid_block_size(dry, fa.Gfx906FABackend)
        cc = dry.cache_config
    finally:
        fa.set_packed_dry_run(False)
        fa.reset_packed_mode()

    attn_page = attn_1tok * cc.block_size
    padded = cc.mamba_page_size_padded or max(attn_page, mamba_page)
    return {
        "mode": mode,
        "attn_1tok": attn_1tok,
        "mamba_page": mamba_page,
        "block_size": cc.block_size,
        "mamba_block_size": cc.mamba_block_size,
        "attn_page": attn_page,
        "padded": padded,
        "tokens_per_gb": cc.block_size / padded * 1e9,
        "mamba_share": mamba_page / padded,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mamba-cache-mode", default="align")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument(
        "--expect",
        help="comma-separated block sizes the boot logs show (alias,packed); "
        "the run fails if the dry run disagrees",
    )
    args = ap.parse_args()

    vc = build_config(args.model, args.mamba_cache_mode, args.max_model_len)
    mc = vc.model_config
    off, on = arm(vc, "off"), arm(vc, "on")

    print(
        f"model={args.model}\n  arch={mc.architecture} hybrid={mc.is_hybrid} "
        f"mamba_cache_mode={args.mamba_cache_mode} max_model_len={args.max_model_len}"
    )
    print(
        f"{'arm':5} {'attn B/tok':>10} {'mamba B/blk':>11} {'block':>6} "
        f"{'attn B/blk':>10} {'page/blk':>10} {'tok/GB':>9} {'attn%':>6} {'mamba%':>7}"
    )
    for r in (off, on):
        print(
            f"{r['mode']:5} {r['attn_1tok']:10d} {r['mamba_page']:11d} "
            f"{r['block_size']:6d} {r['attn_page']:10d} {r['padded']:10d} "
            f"{r['tokens_per_gb']:9.1f} {100 * (1 - r['mamba_share']):6.2f} "
            f"{100 * r['mamba_share']:7.2f}"
        )

    cell = 1 - on["attn_1tok"] / off["attn_1tok"]
    pad_off = 100 * (off["attn_page"] - off["mamba_page"]) / off["mamba_page"]
    pad_on = 100 * (on["attn_page"] - on["mamba_page"]) / on["mamba_page"]
    gain = 100 * (on["tokens_per_gb"] / off["tokens_per_gb"] - 1)
    print(
        f"cell saving {100 * cell:.1f}%  mamba padding "
        f"{pad_off:.2f}% -> {pad_on:.2f}%  "
        f"block {off['block_size']} -> {on['block_size']}  "
        f"page-limited gain {gain:+.2f}%"
    )

    if args.expect:
        want = [int(x) for x in args.expect.split(",")]
        got = [off["block_size"], on["block_size"]]
        if got != want:
            print(f"MISMATCH: expected block sizes {want}, got {got}")
            return 1
        print(f"block sizes match the boot logs: {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
