# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Semantic A/B: CPU numba ngram proposer vs GPU ngram kernel, identical
inputs. Classifies every draft disagreement so the GPU proposer's
draft-quality divergence (spec-decode roadmap item 1: 0.428 vs 1.08
acc/step) can be pinned to a specific rule difference.

Usage:
    HIP_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/gfx906/compare_ngram_cpu_gpu.py
"""

import types

import numpy as np
import torch

from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramGPUKernel


def make_config(min_n: int, max_n: int, k: int, max_model_len: int):
    spec = types.SimpleNamespace(
        prompt_lookup_min=min_n,
        prompt_lookup_max=max_n,
        num_speculative_tokens=k,
    )
    model = types.SimpleNamespace(max_model_len=max_model_len)
    sched = types.SimpleNamespace(max_num_seqs=16)
    par = types.SimpleNamespace(tensor_parallel_size=1)
    return types.SimpleNamespace(
        speculative_config=spec,
        model_config=model,
        scheduler_config=sched,
        parallel_config=par,
    )


def main():
    max_model_len = 65536
    k = 5
    device = "cuda"
    rng = np.random.default_rng(0)
    vocab = 500  # small vocab => lots of accidental matches

    # (min_n, max_n) regimes incl. production (2,2).
    regimes = [(2, 2), (1, 3), (2, 5)]
    n_seqs = 400
    results = {r: {"match": 0, "diff": 0, "cases": []} for r in regimes}

    for (min_n, max_n) in regimes:
        cpu_p = NgramProposer(make_config(min_n, max_n, k, max_model_len))
        # The kernel class is @support_torch_compile-decorated; we only
        # need the raw (uncompiled) match logic, which takes min/max as
        # args and touches no self attributes — call it unbound.
        gpu_k = types.SimpleNamespace()

        for s in range(n_seqs):
            # Lengths 20..2000; mix i.i.d. and repetitive (agentic-like)
            L = int(rng.integers(20, 2000))
            if s % 2 == 0:
                ctx = rng.integers(0, vocab, size=L, dtype=np.int32)
            else:
                pat = rng.integers(0, vocab, size=rng.integers(5, 120), dtype=np.int32)
                ctx = np.tile(pat, L // len(pat) + 1)[:L]
            sampled = int(ctx[-1])  # the "newly sampled" token is the last one

            # CPU: context INCLUDES sampled; num_tokens counts it.
            rows = np.zeros((1, max_model_len), dtype=np.int32)
            rows[0, :L] = ctx
            num_tokens = np.array([L], dtype=np.int32)
            cpu_draft = cpu_p.propose(k, [[sampled]], num_tokens, rows)
            cpu_draft = cpu_draft[0]

            # GPU: base context EXCLUDES the sampled token; the kernel
            # scatters it in. Use the raw (uncompiled) kernel module.
            base = L - 1
            tok_gpu = torch.zeros(1, max_model_len, dtype=torch.int32, device=device)
            tok_gpu[0, :base] = torch.from_numpy(ctx[:base]).to(device)
            num_tokens_gpu = torch.tensor([base], dtype=torch.int32, device=device)
            sampled_gpu = torch.full((1, k + 1), -1, dtype=torch.int32, device=device)
            sampled_gpu[0, 0] = sampled
            count_gpu = torch.tensor([1], dtype=torch.int32, device=device)
            # Replicate NgramProposerGPU.propose()'s scatter + kernel call
            # (the raw, uncompiled kernel = the semantic core).
            write_pos = num_tokens_gpu.unsqueeze(1) + torch.arange(
                k + 1, device=device
            ).unsqueeze(0)
            wmask = (
                torch.arange(k + 1, device=device).unsqueeze(0) < count_gpu.unsqueeze(1)
            )
            wmask = wmask & (sampled_gpu != -1) & (write_pos < max_model_len)
            wp = write_pos.clamp_(max=max_model_len - 1).long()
            existing = tok_gpu.gather(1, wp)
            to_scatter = torch.where(wmask, sampled_gpu, existing)
            tok_gpu.scatter_(1, wp, to_scatter)
            n_tmp = (num_tokens_gpu + count_gpu).to(torch.int32)
            mask = (n_tmp >= min_n)
            gpu_draft = NgramGPUKernel._find_first_and_extract_all_n_parallel(
                gpu_k, tok_gpu, n_tmp, min_n, max_n, k
            )
            gpu_draft = gpu_draft[0].cpu().tolist()
            gpu_draft = [t for t in gpu_draft if t != -1]

            if cpu_draft == gpu_draft:
                results[(min_n, max_n)]["match"] += 1
            else:
                results[(min_n, max_n)]["diff"] += 1
                if len(results[(min_n, max_n)]["cases"]) < 5:
                    results[(min_n, max_n)]["cases"].append(
                        (L, min_n, max_n, cpu_draft, gpu_draft)
                    )

    for r, d in results.items():
        tot = d["match"] + d["diff"]
        print(f"regime min={r[0]} max={r[1]}: match={d['match']}/{tot} "
              f"diff={d['diff']}")
        for c in d["cases"]:
            L, mn, mx, cd, gd = c
            print(f"  L={L}: cpu={cd}")
            print(f"          gpu={gd}")


if __name__ == "__main__":
    main()
