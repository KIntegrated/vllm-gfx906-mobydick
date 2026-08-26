# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Microbench: CPU ngram proposer (`NgramProposer.batch_propose`) cost as a
function of context length, batch size, and (min_n, max_n).

The spec-decode roadmap (L3) attributed ~5 ms/step to the CPU ngram
proposer on agentic contexts (grows with context length). This bench
reproduces that in isolation (no model load) so kernel/proposer
changes can be compared at fixed shapes before the serving A/B.

Usage:
    HIP_VISIBLE_DEVICES= .venv/bin/python benchmarks/kernels/gfx906/bench_ngram_cpu.py
"""

import argparse
import time

import numpy as np

from vllm.v1.spec_decode.ngram_proposer import NgramProposer


def make_vllm_config(min_n: int, max_n: int, k: int, max_model_len: int):
    # Minimal config stub: NgramProposer only reads speculative_config
    # (prompt_lookup_min/max, num_speculative_tokens), model_config
    # (max_model_len), scheduler_config (max_num_seqs), and
    # parallel_config (tensor_parallel_size).
    import types

    spec = types.SimpleNamespace(
        prompt_lookup_min=min_n,
        prompt_lookup_max=max_n,
        num_speculative_tokens=k,
    )
    model = types.SimpleNamespace(max_model_len=max_model_len)
    sched = types.SimpleNamespace(max_num_seqs=32)
    par = types.SimpleNamespace(tensor_parallel_size=1)
    return types.SimpleNamespace(
        speculative_config=spec,
        model_config=model,
        scheduler_config=sched,
        parallel_config=par,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # (min_n, max_n, k) configs: production (max=2 => min=2) + roadmap-era.
    configs = [(2, 2, 5), (2, 5, 5), (1, 3, 3)]
    seqs = [1, 4, 8]
    lens = [2048, 8192, 32768, 131072]
    # The scan only touches [:num_tokens], but k = min(k, max_model_len -
    # total_token) must stay positive: use 2x the largest ctx (like the
    # real 262144 limit vs 131k contexts). The constructor's JIT warmup
    # allocates (1024, max_model_len) int32 transiently (~1 GB here).
    max_model_len = 2 * max(lens)
    rng = np.random.default_rng(args.seed)

    # Repetitive token stream (agentic coding: heavy reuse) — realistic
    # draft-hit pattern, unlike i.i.d. random (which rarely drafts).
    base = rng.integers(0, args.vocab, size=4096, dtype=np.int32)

    print(
        f"{'cfg':>10} {'seqs':>4} {'ctx':>7} {'draft%':>7} "
        f"{'ms/step':>9} {'ms/seq':>8}"
    )
    for (min_n, max_n, k) in configs:
        proposer = NgramProposer(make_vllm_config(min_n, max_n, k, max_model_len))
        for nseq in seqs:
            for ctx in lens:
                rows = np.empty((nseq, max_model_len), dtype=np.int32)
                for s in range(nseq):
                    reps = ctx // 4096 + 1
                    rows[s, :ctx] = np.tile(base, reps)[:ctx]
                num_tokens_no_spec = np.full(nseq, ctx, dtype=np.int32)
                sampled = [[int(base[ctx % 4096])]] * nseq

                # warmup (numba caches are hot from the constructor)
                for _ in range(3):
                    proposer.propose(k, sampled, num_tokens_no_spec, rows)

                t0 = time.perf_counter()
                n_draft = 0
                for _ in range(args.reps):
                    drafts = proposer.propose(
                        k, sampled, num_tokens_no_spec, rows
                    )
                    n_draft += sum(len(d) for d in drafts)
                dt = (time.perf_counter() - t0) / args.reps
                n_draft /= args.reps
                print(
                    f"({min_n},{max_n},k{k}) {nseq:>4} {ctx:>7} "
                    f"{100*n_draft/(nseq*k):>6.1f}% "
                    f"{dt*1e3:>9.3f} {dt*1e3/nseq:>8.3f}"
                )


if __name__ == "__main__":
    main()
