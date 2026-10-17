# RECON — `0c33/mi50-vllm-lab` (2026-10-17)

**What it is.** An independent lab built on this fork (`v0.30.0` @ `6ccbb03`),
2× MI50 16 GB, both cards power-capped at 90 W, TP=2, Docker. It ships three
vLLM fixes, three custom HIP kernels, serve profiles and measured results:
DFlash2 k=7 **73.7 t/s** (from 10.2 t/s on `main`), MTP k=3 **65.1 t/s**
(49.9 stock), a **200k-token** agent profile via int8 KV, and a TP=2
host-memory all-reduce (**28 vs ~72 µs**). Cloned to `/local/git/mi50-vllm-lab`
(single squashed commit, no history).

The headline claim that matters here: **"DFlash2 was not dead, it had two
bugs"** — and one of them is exactly our open LiLiCorr blocker.

## The three fixes, checked against our `main` (v0.31.0)

### 1. Drafter fp16 residual-stream overflow — **NOT in our `main`; adopt**

The bf16-trained drafter runs fp16 on gfx906. Its residual stream passes
65504 by layer 2 → inf/NaN → candidate ids degrade to `0..top_k-1` →
**acceptance exactly 0** (upstream vllm#55250). The fix is load-time only:
scale `o_proj`, the up half of `gate_up_proj`, `down_proj` and the layer-0
residual by a power of two (`VLLM_DFLASH2_FP16_RESIDUAL_DIVISOR`, default 32),
and multiply the layer-0 residual on entry. Every branch reads the stream
through an RMSNorm, and the **final** norm feeds the candidates/selector, so
the outputs are unchanged; zero runtime cost.

**Why it transfers to LiLiCorr** (verified in our tree, not just assumed):
`DFlashQwen3Model.forward` ends with `self.norm(hidden_states, residual)` and
returns the normed hidden ([`qwen3_dflash.py`](../../vllm/model_executor/models/qwen3_dflash.py)),
so both consumers see normed input — the LM head (`compute_candidates`) and the
LiLiCorr correlator's `pass_hidden_proj` (a plain linear, which would *not* be
scale-invariant on an un-normed stream). `anchor_hidden` comes from the
target's hidden states and is unaffected. The grouped conv is linear in the
branch output, so it scales consistently.

This is the leading explanation for `DEVLOG-nemotron-bolt.md`'s **0 accepted
drafts at every position 0–6**: our LiLiCorr drafter is also bf16-trained and
is running fp16.

### 2. Non-causal sliding window dropped — **PRESENT in our `main`; adopt**

[`vllm/gfx906_fa/gfx906_fa_paged.py`](../../vllm/gfx906_fa/gfx906_fa_paged.py)
zeroes `window` for non-causal batches (`if non_causal: window = 0`, ~L435).
The KV-cache manager frees a sliding layer's blocks once they leave the
window, so past ~2–4k context the drafter attends over freed/placeholder
blocks and acceptance collapses to 0 again. The fix is Python-only: keep the
window for the clip, and express "no causal clip, left window only" with
`q_abs_offset = seq_len`, `window = nc_window + max_seqlen_q` — the existing
kernel mask then does the right thing. Their test poisons out-of-window blocks
with NaN.

Not the cause of our LiLiCorr failure (it runs `TRITON_ATTN` because the
custom FA has no attention sinks, and it fails at short context too), but it is
a real bug for any **CUSTOM-FA sliding-window drafter** (DFlash2 2048,
LiLiCorr 1024 if it ever moves back to CUSTOM, Muse).

### 3. `triton_attn` `_seq_lens_cpu` — **we already fixed it; adopt their form**

We fixed it as `seq_lens_cpu_upper_bound` (`49e17fdca7`); they use
`getattr(common_attn_metadata, "_seq_lens_cpu", None)`. **Theirs is safer**: the
only readers are the opt-in torch-SDPA fast paths, which skip when it is
`None`, whereas an upper bound can *enable* them with an over-estimated sequence
length. Switch to their form.

## The kernels (the perf half)

One extension, `gfx906_skinny.so`, three kernels, all env-gated
(`VLLM_GFX906_SKINNY8_LIB`, `VLLM_GFX906_HOSTAR_LIB`) so stock behaviour is
unchanged when absent:

| kernel | what | measured (their box) |
|---|---|---|
| `skinny.cu` | M≤8 fp16 GEMM; 16-byte coalesced weight rows, X staged in LDS, butterfly wave reduce | lm_head 124160×5120 M=7: 10.3 → 3.4 ms |
| `w4.cu` | M≤8 4-bit (Exllama shuffled, v2 zeros); dequant once per weight, fp32 accumulate, no fp16 atomics | 1.1–1.7× per layer at M=8 |
| `hostar.cu` | TP=2 all-reduce through pinned host memory, generation-flag handshake, graph-replay safe | 80 KB: 72 → 28 µs, bit-identical over 3000 calls |

Relevance: spec-decode verify steps are almost entirely M≤8 matmuls. Their
checkpoint is the **same** `cyankiwi/Qwen3.8-27B-AWQ-INT4` we use — and it is
compressed-tensors `pack-quantized`, int4, **group 32**, asymmetric, which is
exactly the `c.group_size == 32` case their `exllama.patch` dispatches to
`gfx906_skinny.w4gemm`. The skinny hook sits at the top of
`rocm_unquantized_gemm_impl` (the fork's own function, so it ports cleanly).
The host all-reduce hooks `cuda_communicator.py` (which already has the fork's
custom-AR guard).

**Not drop-in for v0.31.0.** The patches were generated against 0.30.0: our
`layers/utils.py` is the spliced fork impl (hook point exists), `exllama.py`
upstream changed, and `cuda_communicator.py` has the fork's guard. Each needs a
re-port, and the `.so` a rebuild. Expect the DFlash2/MTP numbers to carry.

## Other findings worth adopting

- **amdsmi must be imported before torch** (`gfx906-amdsmi-first.pth`).
  Otherwise amdsmi enumerates 0 GPUs and vLLM sees `device_count() == 0`.
  Different mechanism from our "amdsmi did not return valid GCN arch", but the
  same class — worth testing on a fresh boot.
- **Cold `torch.compile` inflates the profiled peak by ~1.2 GiB**, so a config
  that fits warm can OOM cold. Their profiles pin `--kv-cache-memory-bytes`
  (same class as our dense-27B 0.93 note).
- **int8 per-token-head KV + `TRITON_ATTN`** (128 softmax segments) reaches
  200k context on 16 GB cards (~16.6 KB/token/rank), with MTP k=3 and prefix
  caching on a hybrid target. TurboQuant k8v4 was rejected (V1 drafter rejects
  MTP k>1; full graphs fail on verify).
- Their DFlash2 handover closed as "dead for this family" is superseded: the
  two bugs above explain every symptom, and the checkpoint is fine.

## Recommended order

1. **Port the fp16 residual rescale** to `DFlash2Qwen3Model`/`LiLiCorr` and
   re-test the LiLiCorr drafter — this is our OPEN 0-acceptance gate. Gate:
   acceptance > 0 at position 0, exact target output preserved.
2. **Adopt the non-causal window fix** (Python-only) + their freed-block test.
3. **Switch `triton_attn` to `getattr(..., None)`** (small, safer).
4. **Evaluate the kernels on v0.31.0**: build `gfx906_skinny.so`, re-port the
   `utils`/`exllama`/`cuda_communicator` hooks, A/B MTP k=3 and DFlash2 k=7.
   This is the largest perf item (their MTP k=3: 49.9 → 65.1 t/s).
5. Test the amdsmi-first `.pth`; consider int8 KV for a 200k profile.

**Relation to our ROADMAP.** (1)/(2) resurrect the parked DFlash2 track
(`docs/gfx906/DEVLOG-dflash2.md`, DFL2-*), (4) is a kernel train that overlaps
our own M≤8 GEMV work (`SKINNY_M16`, `_gfx906_spec_gemv_m4`) — compare before
adopting rather than layering both.
