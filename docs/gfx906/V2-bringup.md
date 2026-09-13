# V2 model-runner bring-up on gfx906 (0.29.0 line)

**Status: open — but points 1 and 2 below are now ANSWERED (2026-09-13, boot
eefacc1e): V2 loads, serves and is numerically at parity on gfx906.**
V2 is upstream's default from 0.29.0 (#53183); the fork still pins V1 everywhere
(`VLLM_USE_V2_MODEL_RUNNER=0`) until the *perf* steps of this plan are signed off.

**Session A result (2026-09-13).** A1 (V2 eager) died at engine-core init with
`hipErrorLaunchFailure`, but the kernel logged a BACO reset in the same second —
i.e. the load-lottery family, not V2 (wedge #79). A2 (**V2 graph mode**) then
loaded clean, reached READY in ~300 s, returned a coherent completion and tore
down cleanly, and **V2 in-process PPL = 10.5516 — bit-identical to V1's and to
the 0.28.0 baseline** (Qwen3.8-27B-AWQ-INT4, fp16, 359 tokens, 0 top-20 misses).
So the Y16 "V2 forced on GDN ⇒ unsupported-by-design" record was the lottery:
V2 is viable here, and the remaining work is performance/spec-decode parity, not
bring-up.

*Runner identification trick for the logs* (the log does not print V1/V2): the
V1 runner logs `[gpu_model_runner.py:…]` for the encoder-cache line, V2 logs
`[encoder_runner.py:…]`. The smoke script should assert this tag.
Plan owner: whoever runs the sessions below. Gate for each step is stated in the
table; the house wedge rules apply to every session (canary first, one retry per
wedge, 2 consecutive failures = burst → stop + reboot, log every event in
`degradation.md`/`degradation_details.md`).

Why this exists: 0.28.0 parity is a **V1** property (see `CHANGELOG.md`
2026-09-13). V2 is *additive* work for the next upstream version, but it is a
prerequisite for anything that rides the V2-only speculator loops (DFL2-1's
n-gram chains, A3's fused draft metadata) and for dropping the V1 pin. Upstream's
own V2-unsupported list is `stock torch.compile`, sequence parallelism, and PP
with `external_launcher` — **none of our features**, so nothing upstream blocks
us.

## 1. What already rides shared code (verify, don't port)

Most gfx906 work sits in paths both runners drive, so the question is
*reachability and behaviour under V2*, not code movement:

| fork work | where it lives | V1/V2 reachability | how it gets verified |
|---|---|---|---|
| CUSTOM Q8 FA backend (decode+prefill, M2 clip, KVSPLIT shape-aware, FIX-H2, gather track) | `vllm/gfx906_fa/*`, `csrc/gfx906_fa/*` (shared `AttentionBackend` API) | shared: V2 calls the same `do_kv_cache_update` / metadata builder | FA suite under V2 + PPL parity + serving A/B |
| Fused-content KV layout port (#51718) | `gfx906_fa_backend.py` (`transpose(1,2).split(...)`, `supported_kv_cache_layouts=LBHNC`) | shared | engine logs `Using LBHNC KV cache layout`; V1 already green |
| Dense GEMV / max-ilp GEMM dispatch | `model_executor/layers/utils.py` | shared `Linear` dispatch | decode benches V1-vs-V2 |
| gfx906 W4A16 MoE kernels (+ asymmetric zp) | `fused_moe/oracle/int_wna16.py` (`GFX906_HIP`), `csrc/rocm/*` | shared | MoE decode bench + PPL |
| GDN/mamba ops, SYV-10 bounds port | `layers/mamba/ops/*`, `third_party/flash_linear_attention/ops/*`, `v1/attention/backends/gdn_attn.py` | shared | PPL + GDN-model serving A/B |
| CAT-1 draft-vocab shortlist | `model_executor/models/qwen3_5_mtp.py` (draft **model** class) | shared class, **but see 2.3** | `MTP draft-vocab shortlist ACTIVE` marker + per-step ms delta |
| max-ilp build flags, headroom advisory, plugin registration | build-time / `platforms/rocm.py` | runner-independent | implicit in the above |

## 2. Gaps that need work

1. **V2 init on gfx906 — unresolved, and the first question.** The Y16 record
   ("V2 forced on Qwen3.8-GDN ⇒ init wedge") is *one* wedge; the same
   `hipErrorLaunchFailure` signature also hit pristine paths on this box, so this
   is the load lottery until a fresh-boot retry says otherwise. Do that retry
   before writing any code for V2.
2. **MTP k=3 (production depth) under V2.** V2 has its own speculator stack
   (`gpu/spec_decode/mtp`) with `share_mtp_topk_indices`/`compact_topk_indices`
   and a draft buffer sized by `num_speculative_steps` (`gpu/states.py`). Check
   the draft width and the GDN state-slot sizing (`MambaSpec.num_speculative_blocks`
   on `mamba_cache_mode=align`) match V1's k+1, then A/B.
3. **CAT-1 under V2 — the silent-loss risk.** If V2's MTP drafts come from the
   *target's* shared top-k indices rather than the drafter's truncated
   `draft_lm_head`/`LogitsProcessor`, the shortlist is bypassed and the
   +4.8 %/+5.9 % and 361 MB-vs-2.54 GB win vanish without an error. The model now
   logs `MTP draft-vocab shortlist ACTIVE (…)` on first use
   (`logger.info_once`, added 2026-09-13): **a V2 run that never prints it is not
   using the shortlist.** Verify the marker plus a per-step ms delta.
4. **M3 host-`cu_seqlens` path.** V2 passes *full-length* host/device
   `query_start_loc` slices in `mamba_hybrid`, unlike V1's
   `[:num_reqs_padded+1]`; our host-path argument was written for V1's slicing.
   Re-audit + test with a full-length slice; then a V2 prefill A/B.
5. **Graph capture.** 0.29 changed the defaults ("widest uniform decode batch by
   default", memory-safe graph sizes) and V2 reserves graph memory differently
   (#53306, #53682). Re-tune the trimmed capture ladder per model, assert no
   eager fallback, and re-run the gather-buffer UAF/capture regression tests
   (`_gather_captured`, `GFX906_FA_CG`).
6. **KV/VRAM sizing.** Compare the logged KV pool and peak VRAM V1-vs-V2 at the
   same `--gpu-memory-utilization` for dense 27B and MoE 35B before trusting any
   V2 bench.
7. **A3 (fused draft metadata)** — V2-only accelerator, stripped and archived
   (`archive/a3-fused-draft` + `A3-REVIVAL.md`). Only after 1–6: re-add the
   opt-in, re-audit the no-op contract at the serving k, and gate at k=7 (k=4 was
   NEUTRAL).
8. **Flip the recipes** (`VLLM_USE_V2_MODEL_RUNNER` removal) model by model, only
   as each one passes.

## 3. Test and verification matrix

| level | what it catches | concrete run |
|---|---|---|
| unit / kernel | backend API drift, K/V split, capture buffers | `tests/kernels/attention/test_gfx906_fa.py` (90 tests) with V2 pinned; the gather-lifecycle/UAF regression test |
| in-process numerical | wrong attention/MoE/GDN numerics under V2's metadata construction | `benchmarks/kernels/gfx906/ppl_probe.py` with `VLLM_USE_V2_MODEL_RUNNER=1`, compared to the recorded V1 numbers (dense 27B, MoE 35B, Muse-Glimmer, Nemotron) — PPL is our only valid numerical gate (greedy token identity is not, on this stack) |
| serving A/B | end-to-end parity: decode t/s, TTFT, acceptance, capture behaviour | standard recipes, same boot, V1 vs V2: dense 27B (`max-seqs 4`), MoE 35B (`max-seqs 32`), then the agentic corpus at 64k/120k |
| spec decode | k=3 draft width/state sizing, acceptance, CAT-1 shortlist actually acting | MTP k=3 on the agentic corpus, V1-vs-V2; CAT-1 on/off within V2; assert the `shortlist ACTIVE` marker and draft ids ⊆ list ∪ control family |
| memory | V2 reservation/sizing drift | logged KV pool + peak VRAM, same util, both models |
| manual (human) | behavioural drift no numeric gate sees | tool-call round trip through the server with the `qwen3_coder`/`qwen3` parsers on real agentic prompts; visual check of the continuation shape on our own corpus bodies; confirm in the log which runner was used (the log does not echo it — the smoke should assert it) |

## 4. Order and session plan

1. **Session A (fresh boot, ~30 min)**: V2 init smoke on the simplest model —
   dense 27B, `VLLM_USE_V2_MODEL_RUNNER=1`, eager first, then graph mode. If it
   wedges: one retry; if the retry wedges → burst → stop + reboot (that is the
   evidence that matters for the arch question).
2. **Session B**: in-process PPL parity for dense 27B and MoE 35B under V2 vs the
   recorded V1 numbers; FA suite with V2 pinned.
3. **Session C**: serving A/B V1-vs-V2 (greedy) for both models, with a
   capture-fallback assertion.
4. **Session D**: V2 + MTP k=3 on the agentic corpus, then CAT-1 on/off inside V2
   (marker + per-step ms).
5. **Sessions E+**: Muse-Glimmer / Nemotron / Ornith-Gemma, graph-ladder tuning,
   then item 7 (A3) if the V2 path is staying.

Each session ends with: canary, VRAM back to the 10.9 MB baseline, no hung
holders (`rocm-smi --showpids`), and a one-paragraph record here or in the
related dev log.
