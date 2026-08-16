#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P3-3a serving-correctness probe (CUSTOM Q8 FA vs Triton reference).

Generates 128 greedy tokens from a fixed 2048-token prompt with:
  A) ROCM_ATTN (Triton paged) + FULL_DECODE_ONLY   -- reference
  B) CUSTOM (gfx906 Q8 FA)  + PIECEWISE           -- new default path
Prints token ids + text for both; they must agree (fp16-reorder-class
divergence far in is acceptable, early divergence/garbage is not — that
is the V-cache stride-bug signature).
"""
import gc
import json
import os

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
    prompt = tok.decode(toks[:PP])
    return prompt


def run_case(tag, llm, prompt):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=TG, temperature=0.0, ignore_eos=True)
    out = llm.generate([prompt], sp)
    toks = out[0].outputs[0].token_ids
    text = out[0].outputs[0].text
    print(f"CASE {tag} tokens={len(toks)}", flush=True)
    print(f"CASE {tag} ids[:40]={toks[:40]}", flush=True)
    print(f"CASE {tag} text={text[:400]!r}", flush=True)
    return toks


def main():
    from vllm import LLM

    prompt_holder = {}

    def make_prompt(model):
        tok = LLM(model=model, max_model_len=PP + TG, gpu_memory_utilization=0.85,
                  enforce_eager=True, disable_log_stats=True).get_tokenizer()
        return build_prompt(tok)

    # Build the prompt once (cheap engine, discarded).
    tok_engine = LLM(model=MODEL, max_model_len=PP + TG, gpu_memory_utilization=0.85,
                     enforce_eager=True, disable_log_stats=True)
    prompt = build_prompt(tok_engine.get_tokenizer())
    del tok_engine
    gc.collect()

    results = {}

    llm_a = LLM(model=MODEL, max_model_len=PP + TG, gpu_memory_utilization=0.85,
                attention_config={"backend": "ROCM_ATTN"},
                compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                    "max_cudagraph_capture_size": 8},
                max_num_seqs=32, disable_log_stats=True)
    results["A_triton_full"] = run_case("A_triton_full", llm_a, prompt)
    del llm_a
    gc.collect()

    llm_b = LLM(model=MODEL, max_model_len=PP + TG, gpu_memory_utilization=0.85,
                compilation_config={"cudagraph_mode": "PIECEWISE",
                                    "max_cudagraph_capture_size": 8},
                max_num_seqs=32, disable_log_stats=True)
    results["B_custom_pw"] = run_case("B_custom_pw", llm_b, prompt)
    del llm_b
    gc.collect()

    a, b = results["A_triton_full"], results["B_custom_pw"]
    first_diff = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    print(f"PROBE first_diff_index={first_diff} "
          f"(len_a={len(a)} len_b={len(b)})", flush=True)
    if first_diff is None:
        print("PROBE RESULT: IDENTICAL", flush=True)
    else:
        print(f"PROBE RESULT: DIVERGE at token {first_diff}: "
              f"A={a[first_diff:first_diff+12]} B={b[first_diff:first_diff+12]}",
              flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
