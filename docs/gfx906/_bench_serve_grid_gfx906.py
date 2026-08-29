#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""
Serving benchmark grid for a running gfx906 TP=2 (or TP=1) server.

Model/snapshot are env-driven (Muse defaults):
  LC_MODEL=<served name> LC_SNAP=<snapshot path> \
      .venv/bin/python docs/gfx906/_bench_serve_grid_gfx906.py \
      '[[pp, tg], ...]' <samples>
Default grid is the long-context prefill sweep (see README "Long-context
performance"): [[32768,128],[65536,128],[112640,128]] x 2.

Mirrors _bench_gfx906.py prompt semantics (repetitive fox filler padded
to exactly pp tokens, raw /v1/completions) but over HTTP with streaming:
  prefill_tps = pp / TTFT
  decode_tps  = (out-1) / (t_end - TTFT)
  wall_tps    = out / (t_end - t0)   (harness-comparable total)
Each prompt carries a per-request tag in its first block: with prefix
caching ON on the server this keeps prefills real (no shared blocks);
with it OFF (the long-context benchmark config) the tag is harmless.

Prompt building is O(1)-encode (bulk filler, no per-char re-encode) and
BOS-safe: tokenizers that prepend a BOS on encode and render it as
literal text on decode (e.g. Muse's) would otherwise round-trip to
pp+1 tokens.
The (2048, 256) cell optionally runs a B=4 concurrent aggregate
(nreqs logic below) for decode-throughput restamps.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000"
MODEL = os.environ.get(
    "LC_MODEL", "Muse-Glimmer-30B")
SNAP = os.environ.get(
    "LC_SNAP",
    "/local/cache/huggingface/hub/models--cyankiwi--Muse-Glimmer-30B-"
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
    """Build a prompt of EXACTLY pp tokens (bulk filler, O(1) encodes).

    BOS-safe: some tokenizers (e.g. Muse's) prepend a BOS on encode and
    render it as literal text on decode — decoding a list that contains
    the BOS makes the round-trip grow by one, so strip it first.
    """
    filler = FILLERS[variant % len(FILLERS)]
    n_fill = len(tok.encode(filler))
    prompt = f"[run {tag}] "
    n = len(tok.encode(prompt))
    while n < pp:
        prompt += filler * max(1, (pp - n) // n_fill)
        n = len(tok.encode(prompt))
    toks = tok.encode(prompt)
    has_bos = bool(toks) and toks[0] == tok.bos_token_id
    body = toks[1:] if has_bos else toks
    return tok.decode(body[:pp - (1 if has_bos else 0)])


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
        [32768, 128], [65536, 128], [112640, 128],
    ]
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print("BENCH-SERVE: " + json.dumps({
        "model": MODEL, "server": BASE, "samples": samples,
        "grid": grid}), flush=True)

    for pp, tg in grid:
        for nreqs in ([1, 4] if (pp, tg) == (2048, 256) else [1]):
            rows = []
            for s in range(samples):
                t_wall0 = time.time()
                if nreqs == 1:
                    rows.append(run_one(
                        tok, pp, tg, s, f"pp{pp}tg{tg}s{s}"))
                else:
                    def _worker(v, pp=pp, tg=tg, s=s):
                        return run_one(
                            tok, pp, tg, v, f"pp{pp}tg{tg}s{s}r{v}")

                    with ThreadPoolExecutor(nreqs) as ex:
                        res = list(ex.map(_worker, range(nreqs)))
                    wall = time.time() - t_wall0
                    tot_out = sum(r["out"] for r in res)
                    rows.append({
                        "aggregate": True,
                        "out_total": tot_out,
                        "wall_s": round(wall, 3),
                        "wall_tps": round(tot_out / wall, 2),
                        "ttft_max_s": round(max(r["ttft_s"] for r in res)),
                        "prefill_tps_aggregate": round(
                            nreqs * pp / wall, 1),
                    })
                tag = f"pp{pp}/tg{tg}/B{nreqs} s{s}"
                print(f"BENCH-SERVE {tag}: {json.dumps(rows[-1])}",
                      flush=True)

    print("BENCH-SERVE-DONE", flush=True)


if __name__ == "__main__":
    main()
