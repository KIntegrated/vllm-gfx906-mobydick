#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P3-3a M2 correctness probe: FULL-decode capture of the CUSTOM Q8 FA path.

A) ROCM_ATTN (Triton) + FULL_DECODE_ONLY          -- reference
C) CUSTOM (Q8 FA, GFX906_FA_CG=decode) + FULL_DECODE_ONLY + GEMV on

128 greedy tokens, 2048-token prompt; must be identical (fp16-reorder-class
divergence far in is acceptable; early divergence/garbage is not).
"""
import gc
import os

os.environ.setdefault("GFX906_FA_CG", "decode")

MODEL = "/models/QuantTrio/Qwen3.5-35B-A3B-AWQ"
PP = 2048
TG = 128


def build_prompt(tok):
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt = ""
    toks = []
    while len(toks) < PP:
        prompt += filler
        toks = tok.encode(prompt)
    return tok.decode(toks[:PP])


def run_case(tag, llm, prompt):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=TG, temperature=0.0, ignore_eos=True)
    out = llm.generate([prompt], sp)
    toks = out[0].outputs[0].token_ids
    text = out[0].outputs[0].text
    print(f"CASE {tag} tokens={len(toks)}", flush=True)
    print(f"CASE {tag} ids={toks}", flush=True)
    print(f"CASE {tag} text={text[:300]!r}", flush=True)
    return toks


def main():
    from vllm import LLM

    tok_engine = LLM(model=MODEL, max_model_len=PP + TG,
                     gpu_memory_utilization=0.85, enforce_eager=True,
                     disable_log_stats=True)
    prompt = build_prompt(tok_engine.get_tokenizer())
    del tok_engine
    gc.collect()

    llm_a = LLM(model=MODEL, max_model_len=PP + TG,
                gpu_memory_utilization=0.85,
                attention_config={"backend": "ROCM_ATTN"},
                compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                    "max_cudagraph_capture_size": 8},
                max_num_seqs=32, disable_log_stats=True)
    a = run_case("A_triton_full", llm_a, prompt)
    del llm_a
    gc.collect()

    llm_c = LLM(model=MODEL, max_model_len=PP + TG,
                gpu_memory_utilization=0.85,
                compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                    "max_cudagraph_capture_size": 8},
                max_num_seqs=32, disable_log_stats=True)
    c = run_case("C_custom_full", llm_c, prompt)
    del llm_c
    gc.collect()

    first_diff = next((i for i in range(min(len(a), len(c))) if a[i] != c[i]),
                      None)
    print(f"PROBE first_diff_index={first_diff} "
          f"(len_a={len(a)} len_c={len(c)})", flush=True)
    if first_diff is None:
        print("PROBE RESULT: IDENTICAL", flush=True)
    else:
        print(f"PROBE RESULT: DIVERGE at token {first_diff}: "
              f"A={a[first_diff:first_diff+12]} C={c[first_diff:first_diff+12]}",
              flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
