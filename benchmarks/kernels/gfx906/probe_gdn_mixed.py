# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""W1 probe: 2-request mixed spec batch — GDN kernel routing + t/s.

Request A: repetitive prompt (ngram always matches -> spec decode).
Request B: unique-n-gram prompt (ngram never matches -> non-spec decode).
Most decode steps are MIXED batches (one spec + one non-spec decode seq),
the regime where the GDN metadata reclass pathology lives (no-draft seqs
run the chunk kernel as 1-token "prefills" instead of the per-seq
recurrent kernel).

Spies the GDN leaf kernel entry points (module-global wrap; must be
installed before the model runs) and reports per-path call totals:
  - chunk:       fla_chunk_gated_delta_rule (prefill path)
  - fused_seq:   fused_sigmoid_gating_delta_rule_update
                 (spec 2.1 with num_accepted_tokens; peeled decodes
                 2.2/2.3 without)
  - packed:      fused_recurrent_gated_delta_rule_packed_decode

Usage:
    HIP_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/gfx906/probe_gdn_mixed.py
Env: BENCH_MODEL, PROBE_NTOK (256), PROBE_PROMPT_TOK (2048)
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import collections

import torch

# --- spies BEFORE model load ---------------------------------------------
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as g

totals = collections.Counter()
fused_spec_calls = 0
fused_nonspec_calls = 0
step_comps = collections.Counter()  # (num_prefills, num_decodes, num_spec_decodes)


def _wrap(name):
    real = getattr(g, name)

    def spy(*a, **kw):
        global fused_spec_calls, fused_nonspec_calls
        totals[name] += 1
        if name == "fused_sigmoid_gating_delta_rule_update":
            if kw.get("num_accepted_tokens") is not None:
                fused_spec_calls += 1
            else:
                fused_nonspec_calls += 1
        return real(*a, **kw)

    setattr(g, name, spy)


_wrap("fla_chunk_gated_delta_rule")
_wrap("fused_sigmoid_gating_delta_rule_update")
_wrap("fused_recurrent_gated_delta_rule_packed_decode")


def _wrap_step_composition():
    """Record the per-layer per-step GDN batch composition (ground truth
    for which dispatch branch each step took)."""
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gl

    real = gl.QwenGatedDeltaNetAttention._forward_core

    def spy(self, mixed_qkv, b, a, core_attn_out):
        fc = get_forward_context()
        md = fc.attn_metadata[self.prefix]
        key = (md.num_prefills, md.num_decodes, md.num_spec_decodes)
        step_comps[key] += 1
        return real(self, mixed_qkv, b, a, core_attn_out)

    gl.QwenGatedDeltaNetAttention._forward_core = spy


_wrap_step_composition()
# ---------------------------------------------------------------------------


def main():
    import time

    from vllm import LLM, SamplingParams

    model = os.environ.get(
        "BENCH_MODEL",
        "/local/cache/huggingface/hub/models--cyankiwi--Qwen3.5-9B-AWQ-INT8-INT4"
        "/snapshots/763be420f16be619241ed1bd1ac6b79deb4a986a",
    )
    ntok = int(os.environ.get("PROBE_NTOK", "256"))
    ptok = int(os.environ.get("PROBE_PROMPT_TOK", "2048"))

    llm = LLM(
        model=model,
        max_model_len=ptok + ntok + 256,
        max_num_seqs=2,
        gpu_memory_utilization=0.95,
        dtype="auto",
        enforce_eager=True,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 2,
        },
        seed=0,
    )
    tok = llm.get_tokenizer()

    # A: repetitive (always drafts)
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt_a = ""
    while len(tok.encode(prompt_a)) < ptok:
        prompt_a += filler
    prompt_a = tok.decode(tok.encode(prompt_a)[:ptok])

    # B: unique 2-grams ("t0 t1 t2 ...") — never drafts
    b = " ".join(f"t{i}" for i in range(ptok * 3))
    prompt_b = tok.decode(tok.encode(b)[:ptok])

    sp = SamplingParams(max_tokens=ntok, temperature=0.0, ignore_eos=True)
    llm.generate([prompt_a, prompt_b], sp, use_tqdm=False)  # untimed warmup

    t0 = time.perf_counter()
    outs = llm.generate([prompt_a, prompt_b], sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    n_a = len(outs[0].outputs[0].token_ids)
    n_b = len(outs[1].outputs[0].token_ids)
    print(
        f"GDN-MIXED t/s total={(n_a + n_b) / dt:.1f} "
        f"A={n_a / dt:.1f} B={n_b / dt:.1f} ({n_a + n_b} tok / {dt:.1f}s)",
        flush=True,
    )
    for name in (
        "fla_chunk_gated_delta_rule",
        "fused_sigmoid_gating_delta_rule_update",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ):
        print(f"GDN-MIXED kernel {name}: {totals[name]}", flush=True)
    print(
        f"GDN-MIXED fused_seq split: spec(natt)={fused_spec_calls} "
        f"non-spec={fused_nonspec_calls}",
        flush=True,
    )
    print(
        "GDN-MIXED step composition (num_prefills, num_decodes, "
        "num_spec_decodes): count",
        flush=True,
    )
    for key in sorted(step_comps):
        print(f"GDN-MIXED   comp={key}: {step_comps[key]}", flush=True)
    print(f"GDN-MIXED outA[:80]={outs[0].outputs[0].text[:80]!r}", flush=True)
    print(f"GDN-MIXED outB[:80]={outs[1].outputs[0].text[:80]!r}", flush=True)


if __name__ == "__main__":
    main()
