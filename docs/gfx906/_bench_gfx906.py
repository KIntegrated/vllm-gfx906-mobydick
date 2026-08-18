#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
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

    # BENCH_EAGER=0 runs with cudagraphs ("serving mode"); numbers are NOT
    # comparable to the eager tables in the README. FULL_DECODE_ONLY + small
    # capture size: this bench is single-request decode-dominated.
    eager = os.environ.get("BENCH_EAGER", "1") == "1"
    extra = {}
    # This vLLM dropped VLLM_ATTENTION_BACKEND; force the backend via
    # attention_config (AttentionConfig.backend). On gfx906 the default
    # resolves to the CUSTOM (Q8 FA) backend.
    attn_backend = os.environ.get("BENCH_ATTN_BACKEND")
    if attn_backend:
        extra["attention_config"] = {"backend": attn_backend}
    if not eager:
        # Hybrid GDN model: cudagraph capture requires max_num_seqs <= number
        # of Mamba cache blocks. Single-request bench -> 32 is plenty.
        # BENCH_MAX_SEQS overrides (dense 27B needs 4: the GDN state pool is
        # ~72 MB/seq and 32 seqs OOMs the 1568-chunk prefill, 2026-08-18).
        # BENCH_CG_MODE overrides the cudagraph mode (P3-3a M0 needs
        # Triton in PIECEWISE for the mode-matched baseline).
        extra["max_num_seqs"] = int(os.environ.get("BENCH_MAX_SEQS", "32"))
        extra["compilation_config"] = {
            "cudagraph_mode": os.environ.get("BENCH_CG_MODE", "FULL_DECODE_ONLY"),
            "max_cudagraph_capture_size": 8,
        }
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_util,
        max_model_len=maxlen,
        dtype="auto",
        enforce_eager=eager,
        **extra,
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
        o = outs[0].outputs[0]
        n_out = len(o.token_ids)
        # token_ids-based: text re-encoding collapses on degenerate/garbage
        # output (e.g. '!!!!...') and undercounts.
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