# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Gate for *instruction-tuned* checkpoints (Gemma-4-*-it, Muse-Glimmer, …).

Why this exists (GEMMA4-1, 2026-09-15): those checkpoints answer only inside their own
chat template. Fed raw text they emit fluent garbage that looks like a broken
model/kernel — and the in-process prompt-logprob PPL probe cannot gate them at all:
raw text scores PPL 84261 (362/363 top-20 misses) and *templating the prompt makes it
worse* (PPL 1278491), because prompt-logprob PPL asks the model to predict the **user's**
tokens, which an instruct model was never trained to model. Both numbers are prompt-format
artifacts, not defects.

What to run instead: this script. It renders each prompt through the model's own chat
template (with the generation prompt) and reports, per prompt, the greedy continuation and
the **first-token top-k logprobs** — a confident first token (logprob ≈ 0) with a sensible
completion is the sanity signal, and running it under two configurations (e.g.
V1 vs V2, or two triton builds) gives a parity gate by comparing text and logprobs.

Usage:
    BENCH_MODEL=<ckpt> HIP_VISIBLE_DEVICES=0 .venv/bin/python \
        benchmarks/kernels/gfx906/ift_chat_gate.py

Reference mode (GEMMA4-1 follow-up, #24): with BENCH_SAVE_REF=<path> the run
writes a reference file; with BENCH_REF=<path> it *asserts* against one instead of
merely reporting. Assertions, in order of strength:

* the **top-1 token id** must match per prompt (hard failure, exit 1);
* its logprob must be within BENCH_REF_TOL nats (default 0.5 -- the measured
  INT8-W8A16 vs bf16 spread on this box is <=0.4);
* the continuation prefix is reported but only asserted with
  BENCH_REF_STRICT=1, because quantization legitimately changes surface text.

That is what makes this a gate rather than a probe: a build that silently changes
the model's first-token distribution now fails a command instead of looking fine.

Env: BENCH_MODEL (required), BENCH_TOP_K (default 5), BENCH_MAX_TOKENS (default 16),
BENCH_GPU_UTIL (default 0.90 — W8A16 27B checkpoints need ~0.95), BENCH_EAGER=1,
BENCH_TP (default 1 — bf16 27B needs 2).
ATTN_BACKEND (optional, forwarded to LLM(attention_backend=...)), and the usual
VLLM_* / VLLM_USE_V2_MODEL_RUNNER knobs. Prompts can be overridden with BENCH_PROMPTS
(one per line, newline-escaped with \\n as literal).
"""

from __future__ import annotations

import json
import os

from vllm import LLM, SamplingParams

PROMPTS = [
    "What is the capital of France? Answer with one word.",
    "Write a Python function that adds two numbers.",
]

EXTRA = [p for p in os.environ.get("BENCH_PROMPTS", "").splitlines() if p.strip()]


def main() -> None:
    model = os.environ["BENCH_MODEL"]
    top_k = int(os.environ.get("BENCH_TOP_K", "5"))
    max_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "16"))
    ab = os.environ.get("ATTN_BACKEND") or None

    llm = LLM(
        model=model,
        max_model_len=2048,
        max_num_seqs=int(os.environ.get("BENCH_MAX_SEQS", "2")),
        gpu_memory_utilization=float(os.environ.get("BENCH_GPU_UTIL", "0.90")),
        dtype=os.environ.get("BENCH_DTYPE", "float16"),
        seed=0,
        attention_backend=ab,
        enforce_eager=os.environ.get("BENCH_EAGER", "0") == "1",
        # bf16 27B needs both cards (54 GB of weights), quantized one-card
        # checkpoints (W8A16 ~27 GB) do not.
        tensor_parallel_size=int(os.environ.get("BENCH_TP", "1")),
    )
    tok = llm.get_tokenizer()
    if not getattr(tok, "chat_template", None):
        raise SystemExit(
            "this tokenizer has no chat template — for raw-text-tolerant models use "
            "docs/gfx906/_bench_gfx906.py (speed) or benchmarks/kernels/gfx906/"
            "ppl_probe.py (numerics) instead"
        )

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in (PROMPTS + EXTRA)]
    print(f"GATE runner={os.environ.get('VLLM_USE_V2_MODEL_RUNNER', 'default')} "
          f"n_prompts={len(prompts)} top_k={top_k}", flush=True)

    results = []
    for p, rendered in zip(PROMPTS + EXTRA, prompts):
        o = llm.generate([rendered], SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                                   ignore_eos=True, logprobs=top_k),
                         use_tqdm=False)
        out = o[0].outputs[0]
        first = (out.logprobs or [{}])[0] if out.logprobs else {}
        top = sorted(
            ((t, lp.logprob) for t, lp in first.items()), key=lambda kv: -kv[1]
        )[:top_k]
        print(f"  PROMPT {p[:60]!r}")
        print(f"      text : {out.text[:120]!r}")
        print(f"      top{top_k} : " + ", ".join(f"{t!r}={v:.2f}" for t, v in top), flush=True)
        results.append({
            "prompt": p,
            "top1": int(top[0][0]) if top else None,
            "top1_logprob": round(float(top[0][1]), 4) if top else None,
            "topk": [[int(t), round(float(v), 4)] for t, v in top],
            "text_head": out.text[:40],
        })

    ref_out = os.environ.get("BENCH_SAVE_REF")
    if ref_out:
        with open(ref_out, "w") as fh:
            json.dump({"model": os.path.basename(os.path.normpath(model)),
                       "top_k": top_k, "n": len(results),
                       "prompts": results}, fh, indent=1)
        print(f"wrote reference {ref_out}", flush=True)

    ref_in = os.environ.get("BENCH_REF")
    if not ref_in:
        return
    with open(ref_in) as fh:
        ref = json.load(fh)
    tol = float(os.environ.get("BENCH_REF_TOL", "0.5"))
    strict = os.environ.get("BENCH_REF_STRICT", "0") == "1"
    if ref.get("n") != len(results):
        raise SystemExit(
            f"reference has {ref.get('n')} prompts, this run has {len(results)} "
            "-- regenerate it or fix BENCH_PROMPTS")
    bad = []
    for got, want in zip(results, ref["prompts"]):
        if got["prompt"] != want["prompt"]:
            bad.append(f"prompt mismatch: {got['prompt'][:40]!r} vs {want['prompt'][:40]!r}")
            continue
        if got["top1"] != want["top1"]:
            bad.append(f"{got['prompt'][:40]!r}: top-1 {got['top1']} != reference "
                       f"{want['top1']}")
            continue
        dlp = abs((got["top1_logprob"] or 0.0) - (want["top1_logprob"] or 0.0))
        if dlp > tol:
            bad.append(f"{got['prompt'][:40]!r}: top-1 logprob moved {dlp:.2f} nats "
                       f"(tol {tol})")
        if got["text_head"] != want["text_head"]:
            msg = (f"{got['prompt'][:40]!r}: continuation differs "
                   f"({got['text_head'][:24]!r} vs {want['text_head'][:24]!r})")
            if strict:
                bad.append(msg)
            else:
                print(f"  note: {msg}", flush=True)
    if bad:
        print(f"GATE FAIL ({len(bad)} item(s)):", flush=True)
        for b in bad:
            print(f"  - {b}", flush=True)
        raise SystemExit(1)
    print(f"GATE PASS: {len(results)} prompts match {ref_in} "
          f"(top-1 exact, <= {tol} nats)", flush=True)


if __name__ == "__main__":
    main()
