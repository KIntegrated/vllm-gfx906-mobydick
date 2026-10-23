#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Long-context decode measurement: TTFT + decode rate, separately.

`_bench_gfx906.py`'s t/s is out_tokens / (prefill + decode), which at 16k
context measures prefill. This measures the two halves separately:

* TTFT   -- a 1-token request on the same prompt (prefill + one step);
* decode -- a TG-token request, minus its TTFT, so the rate is decode-only.

Both halves use identical prompts with prefix caching off, so the subtraction is
sound. mclk is sampled continuously and reported for the decode window (which is
what has to clear the 900 MHz gate; the prefill phase is expected to hold 800,
per dvfs-mi50.md).

Env: BENCH_ATTN_BACKEND / BENCH_KV_DTYPE (via VLLM_ATTENTION_BACKEND too),
BENCH_PP, BENCH_TG, BENCH_SAMPLES, BENCH_GPU_UTIL, BENCH_MAXLEN, BENCH_MODEL,
BENCH_FA_PACKED, BENCH_PRELOAD.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from _bench_gfx906 import _MclkSampler  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

PP = int(os.environ.get("BENCH_PP", "16384"))
TG = int(os.environ.get("BENCH_TG", "256"))
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "3"))
UTIL = float(os.environ.get("BENCH_GPU_UTIL", "0.93"))
MAXLEN = int(os.environ.get("BENCH_MAXLEN", str(PP + TG + 512)))
MODEL = os.environ.get("BENCH_MODEL", "/data/models/qwen/Qwen3.5-27B-AWQ")
TAG = os.environ.get("BENCH_TAG", "longctx")


def extra_kwargs() -> dict:
    out = {}
    if os.environ.get("BENCH_ATTN_BACKEND"):
        out["attention_backend"] = os.environ["BENCH_ATTN_BACKEND"]
    elif os.environ.get("VLLM_ATTENTION_BACKEND"):
        out["attention_backend"] = os.environ["VLLM_ATTENTION_BACKEND"]
    if os.environ.get("BENCH_KV_DTYPE"):
        out["kv_cache_dtype"] = os.environ["BENCH_KV_DTYPE"]
    if not out.get("attention_backend"):
        out["gfx906_fa_packed"] = os.environ.get("BENCH_FA_PACKED", "on")
    return out


def main() -> None:
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=UTIL,
        max_model_len=MAXLEN,
        max_num_seqs=1,
        dtype=os.environ.get("BENCH_DTYPE", "float16"),
        enforce_eager=os.environ.get("BENCH_EAGER", "0") == "1",
        **extra_kwargs(),
    )
    tok = llm.get_tokenizer()
    para = "The quick brown fox jumps over the lazy dog while the engineer measures. "
    text = para * (int(PP * 3.2) // len(para) + 1)
    prompt = tok.encode(text)[:PP]
    print(f"prompt_tokens={len(prompt)} tg={TG} samples={SAMPLES} tag={TAG}",
          flush=True)

    # Preload so the decode window starts warm.
    preload = float(os.environ.get("BENCH_PRELOAD", "20"))
    if preload > 0:
        a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
        end = time.time() + preload
        while time.time() < end:
            a = a @ a
            a = a / a.abs().max().clamp(min=1e-3)
        torch.cuda.synchronize()
        del a
        torch.cuda.empty_cache()

    sampler = _MclkSampler().start()
    rows = []
    for s in range(SAMPLES):
        t0 = time.time()
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1))
        ttft = time.time() - t0

        mclk_a, _ = sampler.median_between(t0, time.time())
        t1 = time.time()
        outs = llm.generate([prompt], SamplingParams(temperature=0.0,
                                                     max_tokens=TG))
        total = time.time() - t1
        n_out = len(outs[0].outputs[0].token_ids)
        decode_s = total - ttft
        rate = (n_out - 1) / decode_s if decode_s > 0 else 0.0
        mclk_d, nsamp = sampler.median_between(t1 + ttft, time.time())
        share = sampler.share_between(t1 + ttft, time.time())
        low = {
            card: round(
                sum(v for k, v in secs.items() if 400 <= k < 900)
                / max(1e-9, sum(v for k, v in secs.items() if k >= 400)),
                3,
            )
            for card, secs in share.items()
        }
        row = {
            "sample": s,
            "ttft_s": round(ttft, 3),
            "total_s": round(total, 3),
            "out_tokens": n_out,
            "decode_tps": round(rate, 3),
            "mclk_prefill_mhz": mclk_a,
            "mclk_decode_mhz": mclk_d,
            "mclk_decode_n": nsamp,
            "decode_800_share": low,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    sampler.stop()

    rates = [r["decode_tps"] for r in rows]
    ttfts = [r["ttft_s"] for r in rows]
    clocks = [r["mclk_decode_mhz"] for r in rows if r["mclk_decode_mhz"]]
    print(
        f"BENCHLONG: {{\"tag\": \"{TAG}\", \"pp\": {PP}, \"tg\": {TG}, "
        f"\"decode_tps_mean\": {statistics.mean(rates):.3f}, "
        f"\"ttft_s_mean\": {statistics.mean(ttfts):.3f}, "
        f"\"mclk_decode\": {clocks}, \"rows\": {json.dumps(rows)}}}",
        flush=True,
    )
    if clocks and min(clocks) < 900:
        print("BENCHLONG WARNING: a decode window ran below 900 MHz - INVALID",
              flush=True)


if __name__ == "__main__":
    main()
