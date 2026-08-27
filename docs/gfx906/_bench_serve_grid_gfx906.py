#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""
Serving benchmark grid for the running Muse-Glimmer TP=2 server.

Mirrors _bench_gfx906.py prompt semantics (repetitive fox filler padded to
exactly pp tokens, raw /v1/completions) but over HTTP with streaming:
  prefill_tps = pp / TTFT
  decode_tps  = (out-1) / (t_end - TTFT)
  wall_tps    = out / (t_end - t0)   (harness-comparable total)
Prefix caching is ON on the server, so every prompt carries a per-request
tag in its first block (no shared prefix blocks -> prefills stay real).
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000"
MODEL = "Muse-Glimmer-30B"
SNAP = ("/local/cache/huggingface/hub/models--cyankiwi--Muse-Glimmer-30B-"
        "AWQ-INT4/snapshots/cba01edf73e0f0f4f013615cc01281ea04e79f85")

FILLERS = [
    "The quick brown fox jumps over the lazy dog. ",
    "A pale gold coin spins slowly in the cold museum light. ",
    "The tide pulls the driftwood logs in long silver arcs. ",
    "Every spring the river braids around gravel islands again. ",
]


def api(path, body=None, timeout=3600):
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def make_prompt(tok, pp, variant, tag):
    filler = FILLERS[variant % len(FILLERS)]
    # per-request tag in the FIRST block: with prefix caching ON on the
    # server, this guarantees no sample/concurrent request shares blocks
    # (the /flush_cache endpoint is not exposed on this build).
    head = f"[run {tag}] "
    prompt, toks = head, tok.encode(head)
    while len(toks) < pp:
        prompt += filler
        toks = tok.encode(prompt)
    return tok.decode(toks[:pp])


def run_one(tok, pp, tg, variant, tag):
    prompt = make_prompt(tok, pp, variant, tag)
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": tg,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    t0 = time.time()
    first, t_end, usage, actual_pp = None, t0, None, len(tok.encode(prompt))
    with api("/v1/completions", body) as f:
        for raw in f:
            raw = raw.strip()
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                if first is None:
                    first = time.time()
                t_end = time.time()
    t_end = max(t_end, time.time())
    out = usage["completion_tokens"] if usage else 0
    ttft = (first - t0) if first is not None else (t_end - t0)
    wall = t_end - t0
    return {
        "pp_actual": actual_pp,
        "out": out,
        "ttft_s": round(ttft, 3),
        "wall_s": round(wall, 3),
        "prefill_tps": round(actual_pp / ttft, 1) if ttft > 0 else 0.0,
        "decode_tps": round((out - 1) / (wall - ttft), 2)
        if wall > ttft and out > 1 else 0.0,
        "wall_tps": round(out / wall, 2) if wall > 0 else 0.0,
    }


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)

    grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        [2048, 256], [2048, 512], [8192, 256], [8192, 512],
        [16384, 256], [16384, 512],
    ]
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print("BENCH-SERVE: " + json.dumps({
        "model": MODEL, "server": BASE, "samples": samples,
        "grid": grid}), flush=True)

    for pp, tg in grid:
        for nreqs in ([1, 4] if (pp, tg) == (2048, 256) else [1]):
            rows = []
            for s in range(samples):
                t_wall0 = time.time()
                if nreqs == 1:
                    rows.append(run_one(tok, pp, tg, s, f"pp{pp}tg{tg}s{s}"))
                else:
                    def _worker(v, pp=pp, tg=tg, s=s):
                        return run_one(tok, pp, tg, v,
                                       f"pp{pp}tg{tg}s{s}r{v}")

                    with ThreadPoolExecutor(nreqs) as ex:
                        res = list(ex.map(_worker, range(nreqs)))
                    wall = time.time() - t_wall0
                    tot_out = sum(r["out"] for r in res)
                    rows.append({
                        "aggregate": True,
                        "out_total": tot_out,
                        "wall_s": round(wall, 3),
                        "wall_tps": round(tot_out / wall, 2),
                        "ttft_max_s": round(max(r["ttft_s"] for r in res), 3),
                        "prefill_tps_aggregate": round(
                            nreqs * pp / wall, 1),
                    })
                tag = f"pp{pp}/tg{tg}/B{nreqs} s{s}"
                print(f"BENCH-SERVE {tag}: {json.dumps(rows[-1])}",
                      flush=True)

    print("BENCH-SERVE-DONE", flush=True)


if __name__ == "__main__":
    main()
