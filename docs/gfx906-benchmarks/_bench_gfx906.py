#!/usr/bin/env python3
"""gfx906 benchmark (0.23 vs 0.26 vs main). Usage: python3 /bench/_b.py <model>
Env: BENCH_PP (2048), BENCH_TG (256), BENCH_GPU_UTIL (0.85), BENCH_MAXLEN,
     BENCH_WARMUP (1=do untimed warmup), BENCH_SAMPLES (default 1).
Measures one pp-prefill + tg-decode request (after an untimed warmup).
Prints "BENCH: {json}". Robust to cross-version SamplingParams differences.
"""
import json, os, sys, time

WARMUP = os.environ.get("BENCH_WARMUP", "1") == "1"
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "1"))


def model_arg():
    if len(sys.argv) > 1 and sys.argv[1]:
        return sys.argv[1]
    m = os.environ.get("BENCH_MODEL", "")
    if m:
        return m
    raise SystemExit("BENCH: no model (argv[1] or BENCH_MODEL)")


def main():
    model = model_arg()
    pp = int(os.environ.get("BENCH_PP", "2048"))
    tg = int(os.environ.get("BENCH_TG", "256"))
    gpu_util = float(os.environ.get("BENCH_GPU_UTIL", "0.85"))
    maxlen = int(os.environ.get("BENCH_MAXLEN", str(pp + tg + 512)))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_util,
        max_model_len=maxlen,
        dtype="auto",
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()

    # Build a prompt encoding to exactly pp tokens.
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt = ""
    toks = []
    while len(toks) < pp:
        prompt += filler
        toks = tok.encode(prompt)
    prompt = tok.decode(toks[:pp])

    def gen_params(max_tokens):
        # enable_thinking may not exist across versions; probe harmlessly.
        try:
            return SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                  ignore_eos=True, enable_thinking=False)
        except TypeError:
            return SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                  ignore_eos=True)

    if WARMUP:
        llm.generate([prompt], gen_params(min(tg, 8)))
        print("BENCH warmup_pass done", flush=True)

    results = []
    for s in range(SAMPLES):
        t0 = time.time()
        outs = llm.generate([prompt], gen_params(tg))
        t1 = time.time()
        gen_text = outs[0].outputs[0].text
        n_out = len(tok.encode(gen_text)) if gen_text else 0
        elapsed = t1 - t0
        results.append({
            "sample": s, "out_tokens": n_out, "elapsed_s": round(elapsed, 3),
            "tokens_per_s": round(n_out / elapsed, 3) if elapsed else 0.0,
        })

    print("BENCH: " + json.dumps({
        "model": model, "pp": pp, "tg": tg, "maxlen": maxlen,
        "samples": results,
    }), flush=True)


if __name__ == "__main__":
    main()