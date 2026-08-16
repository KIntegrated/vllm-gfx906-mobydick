#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""128-token greedy A/B probe: prints generated token ids as JSON so two
runs (different FA configs) can be diffed host-side.

Usage: python3 greedy_probe.py <model>  (PROBE_BACKEND env selects the
attention backend, like ppl_probe.py; the GFX906_FA_NC2/KVSPLIT envs
select the FA decode config). Prints "TOKENS: {json}".
"""
import json
import os
import sys

PROMPTS = [
    (
        "The capital of France is Paris, and it is famous for the Eiffel "
        "Tower, the Louvre museum, and its historic role in the "
        "development of Western art, philosophy, and fashion over the last "
        "several centuries."
    ),
    (
        "Photosynthesis is the process by which green plants convert "
        "sunlight, water, and carbon dioxide into glucose and oxygen. The "
        "process takes place mainly in the leaves, where chlorophyll "
        "absorbs light energy."
    ),
    (
        "A compiler translates a high-level programming language into "
        "machine code through several stages: lexical analysis, parsing, "
        "semantic checks, optimization, and code generation. Optimizations "
        "such as loop unrolling and inlining can greatly improve runtime "
        "performance."
    ),
    (
        "In thermodynamics, entropy measures the number of microscopic "
        "configurations that correspond to a macroscopic state. The second "
        "law states that the entropy of an isolated system never "
        "decreases."
    ),
]


def main():
    model = sys.argv[1]
    from vllm import LLM, SamplingParams

    backend = os.environ.get("PROBE_BACKEND", "CUSTOM")
    extra = {}
    if backend != "CUSTOM":
        extra["attention_config"] = {"backend": "ROCM_ATTN"}
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.85,
        max_model_len=512,
        dtype="auto",
        max_num_seqs=1,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        **extra,
    )
    params = SamplingParams(max_tokens=128, temperature=0.0)
    outs = llm.generate(PROMPTS, params)
    res = []
    for o in outs:
        res.append(list(o.outputs[0].token_ids))
    print("TOKENS: " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
