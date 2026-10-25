#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Pay the first-long-prefill cost at startup, not on the user's first request.

The first prefill at a given long length costs ~9x the steady-state time, once
per server process (~39.9 s vs 4.35 s for an 11k-token prompt, measured at mclk
350, 800 and 1000 MHz alike, so it is a first-call cost and not a clock effect).
A 64-token warm-up does not absorb it; a warm-up at the served length does.

This is a DIAGNOSTIC as much as a warm-up. On the serving path measured
2026-10-25 it buys nothing: two identical 12.7k-token requests took 51.06 s and
50.95 s -- there is no first-call premium over HTTP at all, so the ~36 s -> 4 s
first-call effect is an in-process phenomenon. Run it to find out which regime
your build is in; do not expect it to move latency on its own.

    python docs/gfx906/_warmup_serve.py --model Qwen3.8-27B --tokens 16384

Exit code is 0 when the second call is materially faster than the first, 1 when
it is not (i.e. the warm-up did not do its job -- check that the prompt really
reached the target length, printed as `prompt_tokens`).
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

WORDS = (
    "the quick brown fox jumps over a lazy dog while seventeen engineers argue "
    "about cache layouts and memory bandwidth on aging accelerators "
).split()


def build_prompt(target_tokens: int) -> str:
    """Word salad long enough for the target; measured ~1.05 tokens/word."""
    words = int(target_tokens * 0.98) + 64
    return " ".join(WORDS[i % len(WORDS)] for i in range(words))


def post(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, time.monotonic() - start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokens", type=int, default=16384, help="prompt length")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--prompt-file", help="use this exact prompt text")
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": args.model,
        "prompt": (
            Path(args.prompt_file).read_text()
            if args.prompt_file
            else build_prompt(args.tokens)
        ),
        "max_tokens": 1,
        "temperature": 0.0,
    }

    times, seen = [], 0
    for attempt in range(2):
        out, dt = post(url, payload, args.timeout)
        seen = out.get("usage", {}).get("prompt_tokens", 0)
        times.append(dt)
        print(
            f"warm-up call {attempt + 1}: {dt:7.2f} s  "
            f"prompt_tokens={seen}  (target {args.tokens})",
            flush=True,
        )

    if seen < args.tokens * 0.95:
        print(
            f"WARNING: prompt reached only {seen} of {args.tokens} tokens; "
            "the warm-up did not cover the length you care about"
        )

    first, second = times
    print(f"first-call cost paid: {first - second:.2f} s moved off the user path")
    return 0 if second < first * 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
