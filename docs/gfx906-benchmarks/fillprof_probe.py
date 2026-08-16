# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Attribute the per-step fill/copy kernel pile to torch ops + python stacks.

Runs the bench model EAGER (no cudagraph) under the vLLM torch profiler so
every fill/copy launch has a correlation id -> cpu_op (+ python stack). The
eager path executes the same model code the graph captures; per-step
fill/copy counts should match the rocprofv3 kernel trace (~247/step).

Usage (env per the local-venv bench recipe; the LLM itself is built
enforce_eager with fastsafetensors):
  HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  HF_HUB_OFFLINE=1 /local/git/vllm-gfx906-mobydick/.venv/bin/python \
  fillprof_probe.py <model>

Outputs: a chrome trace (rank0.pt.trace.json.gz) under /tmp/bench/fillprof
correlating cpu_op -> kernel launches (args["External id"]); python
stacks via the "Python id" / "Python parent id" tree. Parse with
/tmp/bench/parse_fillprof.py (ijson streaming).
"""

import os
import time

MODEL = os.environ.get("PROBE_MODEL",
                       "/local/models/QuantTrio/Qwen3.5-35B-A3B-AWQ")
OUT = "/tmp/bench/fillprof"
N_STEPS = int(os.environ.get("PROBE_STEPS", "50"))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig

    llm = LLM(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,  # attribution needs per-op launches
        gpu_memory_utilization=0.95,
        max_model_len=3328,
        load_format="fastsafetensors",
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=OUT,
            torch_profiler_with_stack=True,
            torch_profiler_record_shapes=True,
        ),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=N_STEPS + 64,
                        detokenize=False)
    # prompt ~2048 tokens
    prompt = [7] * 2048
    # warmup: separate request so JIT/capture settle before the profile
    llm.generate([prompt], SamplingParams(temperature=0.0,
                                          max_tokens=32,
                                          detokenize=False))
    t0 = time.time()
    llm.start_profile()
    out = llm.generate([prompt], sp)
    llm.stop_profile()
    dt = time.time() - t0
    n = len(out[0].outputs[0].token_ids)
    print(f"PROBE done: {n} tokens in {dt:.2f}s ({n/dt:.1f} t/s eager+prof)")


if __name__ == "__main__":
    main()
