# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""In-engine probe: per-step CPU wall time of the CPU ngram proposer and of
`_bookkeeping_sync` (the D2H region the CPU drafter runs after) as a
function of context length.

The spec-decode roadmap (L3) carried a ~5 ms/step "CPU ngram proposer
D2H serialization" cost (grows with context length) from the 27B
agentic-era cost model. This probe re-measures that on the current
build: NgramProposer.propose is monkeypatched with a wall timer, as is
GPUModelRunner._bookkeeping_sync; one long prompt is decoded and the
per-step times are bucketed by context length.

Usage (local 9B by default):
    HIP_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/gfx906/probe_ngram_cpu_engine.py
"""

import os
import time
from collections import defaultdict

import numpy as np

# --- monkeypatch BEFORE the model loads -----------------------------------
import vllm.v1.spec_decode.ngram_proposer as np_mod
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

_propose_times = defaultdict(list)   # ctx_bucket -> [wall ms]
_bookkeep_times = []
_orig_propose = np_mod.NgramProposer.propose


def _timed_propose(self, *args, **kwargs):
    t0 = time.perf_counter()
    out = _orig_propose(self, *args, **kwargs)
    dt = (time.perf_counter() - t0) * 1e3
    # args: (num_spec_tokens, sampled_token_ids, num_tokens_no_spec,
    #       token_ids_cpu, ...) — bucket by the max ctx in the batch,
    # quantized to 4096 so the summary stays small.
    ctx = int(np.max(args[2])) if len(args) > 2 and args[2] is not None else -1
    _propose_times[ctx // 4096 * 4096].append(dt)
    return out


np_mod.NgramProposer.propose = _timed_propose

# ngram_gpu: patch its propose too (same bookkeeping, no token_ids_cpu scan)
_orig_propose_gpu = NgramProposerGPU.propose


def _timed_propose_gpu(self, *args, **kwargs):
    t0 = time.perf_counter()
    out = _orig_propose_gpu(self, *args, **kwargs)
    _propose_times[-1].append((time.perf_counter() - t0) * 1e3)
    return out


NgramProposerGPU.propose = _timed_propose_gpu

_orig_bookkeep = GPUModelRunner._bookkeeping_sync


def _timed_bookkeep(self, *args, **kwargs):
    t0 = time.perf_counter()
    out = _orig_bookkeep(self, *args, **kwargs)
    _bookkeep_times.append((time.perf_counter() - t0) * 1e3)
    return out


GPUModelRunner._bookkeeping_sync = _timed_bookkeep
# ---------------------------------------------------------------------------

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    model = os.environ.get(
        "BENCH_MODEL",
        "/local/cache/huggingface/hub/models--cyankiwi--Qwen3.5-9B-AWQ-INT8-INT4"
        "/snapshots/763be420f16be619241ed1bd1ac6b79deb4a986a",
    )
    ctx = int(os.environ.get("PROBE_CTX", "32768"))
    ntok = int(os.environ.get("PROBE_NTOK", "512"))
    method = os.environ.get("PROBE_METHOD", "ngram")

    llm = LLM(
        model=model,
        max_model_len=ctx + ntok + 256,
        max_num_seqs=1,
        gpu_memory_utilization=0.95,
        dtype="auto",
        enforce_eager=True,  # eager: every CPU step is visible
        speculative_config={
            "method": method,
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 2,
        },
        seed=0,
    )
    tok = llm.get_tokenizer()
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt = ""
    while len(tok.encode(prompt)) < ctx:
        prompt += filler
    prompt = tok.decode(tok.encode(prompt)[:ctx])

    llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0.0), use_tqdm=False)

    print(f"PROBE-NGRAM model={os.path.basename(os.path.dirname(os.path.dirname(model)))} "
          f"method={method} ctx={ctx} ntok={ntok}", flush=True)
    t0 = time.perf_counter()
    outs = llm.generate(
        [prompt], SamplingParams(max_tokens=ntok, temperature=0.0, ignore_eos=True),
        use_tqdm=False,
    )
    dt = time.perf_counter() - t0
    out_toks = outs[0].outputs[0].token_ids
    import hashlib
    h = hashlib.md5(bytes(np.asarray(out_toks, dtype=np.int32).tobytes())).hexdigest()[:12]
    nprop = sum(len(v) for v in _propose_times.values())
    acc = (len(out_toks) - nprop) / max(1, nprop)
    print(f"PROBE-NGRAM total: {len(out_toks)} tok / {dt:.1f}s = {len(out_toks)/dt:.1f} t/s "
          f"steps={nprop} acc/step={acc:.3f} tokhash={h}", flush=True)

    # summarize
    n = len(_bookkeep_times)
    arr = np.asarray(_bookkeep_times)
    print(
        f"PROBE-NGRAM-CPU bookkeep_sync: n={n} mean={arr.mean():.3f} ms "
        f"p50={np.percentile(arr,50):.3f} p95={np.percentile(arr,95):.3f} ms",
        flush=True,
    )
    order = np.argsort(arr)[::-1][:8]
    for i in order:
        print(f"PROBE-NGRAM-CPU bookkeep_top: step={i} {arr[i]:.1f} ms", flush=True)
    for c in sorted(_propose_times):
        v = np.asarray(_propose_times[c])
        print(
            f"PROBE-NGRAM-CPU propose@ctx~{c}: n={len(v)} mean={v.mean():.3f} ms "
            f"p50={np.percentile(v,50):.3f} p95={np.percentile(v,95):.3f} max={v.max():.3f}",
            flush=True,
        )
    allp = np.concatenate([v for v in _propose_times.values()]) if _propose_times else np.zeros(1)
    if len(allp) > 8:
        for i in np.argsort(allp)[::-1][:8]:
            print(f"PROBE-NGRAM-CPU propose_top: idx={i} {allp[i]:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
