#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Streaming TTFT probe: measures client-side time-to-first-token per endpoint.

Used for P32-10/#70 to separate three candidate causes of the serving path's
~2x TTFT against in-process generation: the client's own timer (queueing +
chat-template + one decode step), the chat endpoint's template/tokenisation, or
server-side prefill throughput. The engine log's own "Avg prompt throughput"
line is the ground truth to compare against.

Usage: _probe_ttft.py PORT PROMPT_FILE MODE [MAX_TOKENS] [REPS]
  MODE = chat | completions
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

port, prompt_file, mode = sys.argv[1], sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 2

text = open(prompt_file).read()
base = f"http://127.0.0.1:{port}"
if mode == "chat":
    url, payload = base + "/v1/chat/completions", {
        "model": "q27",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
else:
    url, payload = base + "/v1/completions", {
        "model": "q27",
        "prompt": text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

ttfts = []
for rep in range(reps):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            usage = obj.get("usage") or {}
            if usage.get("prompt_tokens"):
                prompt_tokens = usage["prompt_tokens"]
    ttfts.append(ttft)
    print(json.dumps({
        "mode": mode, "rep": rep, "max_tokens": max_tokens,
        "ttft_s": round(ttft, 3), "prompt_tokens": prompt_tokens,
    }), flush=True)

print(json.dumps({
    "mode": mode, "max_tokens": max_tokens,
    "ttft_mean_s": round(statistics.mean(ttfts), 3),
    "ttft_all": [round(t, 3) for t in ttfts],
}), flush=True)
