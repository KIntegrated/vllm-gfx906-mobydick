# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P3-1 A/B: greedy decode output for fixed prompts (tag printed by caller)."""
import sys

from vllm import LLM, SamplingParams

TAG = sys.argv[1] if len(sys.argv) > 1 else "?"
MODEL = "/models/Qwen3.5-35B-A3B-AWQ"
PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "Once upon a time in a small village,",
]

def main():
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    outs = llm.generate(PROMPTS, sp)
    for i, o in enumerate(outs):
        toks = [t for t in o.outputs[0].token_ids]
        print(f"{TAG} prompt{i}: {toks}")
        print(f"{TAG} text{i}: {o.outputs[0].text!r}"[:400])

if __name__ == "__main__":
    main()
