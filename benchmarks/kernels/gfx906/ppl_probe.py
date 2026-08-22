# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""S3 PPL gate: fixed 12-prompt prompt-logprob PPL (recreated 2026-08-18
after /tmp wipe; the original /tmp/bench/ppl_probe.py prompt set was lost,
so absolute values are NOT comparable to the 6.6895-era numbers — the
A/B on a fixed build pair is the gate).

Protocol (per docs/gfx906/DEVLOG-moe-opt.md F3): prefill prompt logprobs
with k=20 top-k, actual-token lookup, residual-mass fallback when the
token misses top-k, PPL = exp(-sum(logprob)/n_tokens).

Usage: VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/kernels/gfx906/ppl_probe.py
"""
import math
import os

from vllm import LLM, SamplingParams

PROMPTS = [
    "The quick brown fox jumps over the lazy dog while the farmer watches "
    "from the porch, counting the chickens that wandered into the yard.",
    "In distributed systems, consistency comes at the cost of availability "
    "and latency; engineers choose where on that spectrum their product "
    "must live, and they document the tradeoff carefully.",
    "Photosynthesis converts light energy into chemical energy stored in "
    "glucose, a process that ultimately powers nearly every food chain on "
    "the planet and shapes the composition of the atmosphere.",
    "The compiler first parses the source into an abstract syntax tree, "
    "then walks that tree emitting instructions while keeping the register "
    "pressure within the limits of the target architecture.",
    "Marble statues weather slowly as acid rain etches their surfaces, "
    "turning sharp chiselled detail into soft shapes over the course of "
    "centuries of exposure to the open air.",
    "A well-tended garden rewards patience: the tomato seedlings that "
    "survive their first frost bear heavy fruit in late summer, and the "
    "herbs return stronger each spring.",
    "Quantum computers exploit superposition and entanglement to explore "
    "many candidate solutions at once, though error correction remains the "
    "principal obstacle to practical machines.",
    "The lighthouse keeper climbed the spiral staircase each evening, "
    "trimmed the wick, and watched the beam sweep the dark water until "
    "the first ferry lights appeared on the horizon.",
    "Battery capacity fades with age because the solid electrolyte "
    "interface layer thickens on the anode, trapping lithium ions and "
    "reducing the charge the cell can deliver.",
    "Good tests describe intent: a name like test_overflow_refunds_when "
    "total_exceeds_budget tells the reader what behavior is protected "
    "without opening the body of the function.",
    "The river braids and splits around gravel islands each spring, "
    "carrying snowmelt from the high valleys down to the delta where the "
    "marsh grass bends but does not break.",
    "Compilers for tensor workloads must reason about memory bandwidth as "
    "carefully as arithmetic intensity, because a kernel that fits in "
    "cache runs orders of magnitude faster than one that streams.",
]


def main():
    llm = LLM(
        model=os.environ.get(
            "BENCH_MODEL", "/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ"),
        max_num_seqs=8,
        max_model_len=int(os.environ.get("BENCH_MAXLEN", "32768")),
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        hf_overrides={},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20)
    outs = llm.generate(PROMPTS, sp)
    tokenizer = llm.get_tokenizer()

    total_lp = 0.0
    n_tok = 0
    n_miss = 0
    for prompt, o in zip(PROMPTS, outs):
        ids = tokenizer.encode(prompt)
        pl = o.prompt_logprobs
        for i in range(1, len(ids) - 1):  # position 0 has no conditioning
            entry = pl[i]
            lp_obj = entry.get(ids[i])
            if lp_obj is not None:
                total_lp += lp_obj.logprob
            else:
                mass = sum(math.exp(v.logprob) for v in entry.values())
                total_lp += math.log(max(1.0 - mass, 1e-12))
                n_miss += 1
            n_tok += 1

    ppl = math.exp(-total_lp / n_tok)
    print(f"PPL {ppl:.4f}  ({n_tok} tokens, {n_miss} top-20 misses)")


if __name__ == "__main__":
    main()
