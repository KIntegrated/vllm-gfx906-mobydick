# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Resolve the compressed-tensors scheme vLLM would pick for a checkpoint's config groups.

CPU-only, no GPU needed: builds the checkpoint's CompressedTensorsConfig and asks it for the
scheme of representative layer names. Useful when a quantised checkpoint fails to load and the
question is whether the fault is routing (no scheme / wrong scheme) or the loader.

Usage: .venv/bin/python tools/ct_scheme_probe.py <model_dir>
"""
import json
import sys
S = sys.argv[1] if len(sys.argv) > 1 else "."
qc=json.load(open(f"{S}/config.json"))["quantization_config"]
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
cfg = CompressedTensorsConfig.from_config({"quantization_config": qc, "config_groups": qc["config_groups"],
                                           "format": qc["format"], "ignore": qc["ignore"],
                                           "quant_method": qc["quant_method"], "version": qc["version"]})
print("  CT config built:", type(cfg).__name__, "| target schemes:", len(getattr(cfg, 'target_scheme_map', {}) or {}))
for group, spec in qc["config_groups"].items():
    w = QuantizationArgs(**spec["weights"])
    for name in ("model.language_model.layers.0.self_attn.q_proj", "model.language_model.embed_tokens",
                 "lm_head", "model.language_model.layers.0.linear_attn.in_proj_qkv"):
        try:
            sch = cfg._get_scheme_from_parts(weight_quant=w, input_quant=None,
                                             format=qc["format"], layer_name=name)
            extra = " ".join(f"{a}={getattr(sch,a)}" for a in ("strategy","num_bits","group_size","symmetric") if hasattr(sch,a))
            print(f"  {group:11s} {name.split('.')[-2]+'.'+name.split('.')[-1]:28s} -> {type(sch).__name__} {extra}")
        except Exception as e:
            print(f"  {group:11s} {name:28s} -> RAISED {type(e).__name__}: {str(e)[:60]}")
