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
   **Code audit (2026-09-14, `gfx906/v2-bringup`): no silent-loss path exists.**
   V2's draft sampling goes through `BaseSpeculator.sample_draft` →
   `self.model.compute_logits(...)` (`speculator.py:364-388`), i.e. the draft
   model's own `compute_logits`, where the shortlist scatter lives — and the
   greedy branch `_greedy_sample_draft` (`:358-362`) uses `compute_logits` too,
   *except* under `use_local_argmax_reduction`, where it calls `get_top_tokens`
   (full `lm_head`): that combination already raises at drafter init via our
   CAT-1 guard (`qwen3_5_mtp.py`, `use_local_argmax_reduction` check), so it
   fails closed rather than silently degrading. The
   `share_mtp_topk_indices` route (target's top-k instead of the drafter's head,
   `mtp/speculator.py:30-34`) is gated on
   `index_share_for_mtp_iteration` + `set_skip_topk`/`compact_topk_indices` —
   DeepSeek-style MTP only; the Qwen3.5 drafter has neither, so it stays off.
   Live confirmation (marker + ms/step under V2) is the session below.
4. **M3 host-`cu_seqlens` path.** V2 passes *full-length* host/device
   `query_start_loc` slices in `mamba_hybrid`, unlike V1's
   `[:num_reqs_padded+1]`; our host-path argument was written for V1's slicing.
   Re-audit + test with a full-length slice; then a V2 prefill A/B.
   **Code audit + guard test (2026-09-14): the path is length-agnostic.**
   `forward_paged` consumes the host list with `for s in range(num_seqs)` where
   `num_seqs` comes from the padded query tensor, never from the list's length
   (`gfx906_fa_paged.py:482-490` and `:575-582`), so trailing entries are never
   read; V2's zero-length padded rows are already skipped by `if n > 0`. V2's
   builder passes `query_start_loc_cpu` unsliced
   (`v1/worker/gpu/attn_utils.py:296`) — fine under that bound. The doc's
   failure mode is now guarded by
   `test_forward_mixed_batch_pad_tile_clamp_and_host_cu`, which additionally
   runs the same case with a **full-length slice whose tail is garbage
   (`-12345`)** and asserts bit-identical output (V2 layout). V2 prefill A/B
   still outstanding.
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
   as each one passes. **Done 2026-09-14 for dense 27B and MoE 35B** (both at
   parity: agentic greedy/spec + MoE in-process 58.36 vs 57.86 t/s; see the
   session results above). `run_server.sh` now defaults to V2 with V1 one env
   override away, and the root README carries the per-model status. Still pinned
   to V1: Muse-Glimmer (checkpoint not local — only the GGUF), Nemotron 3.5
   Lightning (`primitive-ai/Nemotron-3.5-Lightning-30B-A3B-mixed-INT4-INT8`),
   Ornith (`cyankiwi/Ornith-1.5-35B-A3B-AWQ-INT4`), Gemma-4
   (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) — all three checkpoints verified
   present, parity runs queued.
   **Flip verification status:** the flipped `run_server.sh` was confirmed to
   *select* V2 with no runner env (1 bare `[model_runner.py:*]` line, 0
   `gpu_model_runner.py`), but the end-to-end smoke could not complete: the boot
   took two consecutive load wedges (#84 at 14:23 on both GPUs, #85 at 14:34 on
   GPU1) → **BURST, GPU work stopped, host reboot required** before the
   remaining V2 work (Nemotron/Ornith/Gemma-4 parity → A3 revival → the V2 CAT-1
   headline re-measure).

## 2b. Session C/D result (2026-09-14, branch `gfx906/v2-bringup`, boot eefacc1e)

V2 serving on the agentic corpus (dense 27B, TP=2, V1-pinned reference numbers
from the same boot; 2 reps/cell, mclk 1000; runner confirmed V2 by the bare
`[model_runner.py:*]` tag, which V1 never emits):

| arm | V2 @64k | V2 @120k | V2 acc | V1 @64k / @120k | V2 vs V1 |
|---|---|---|---|---|---|
| greedy | 20.37 (20.78/19.95) | 13.27 (13.28/13.25) | – | 19.91 / 13.25 | +2.3 % / +0.2 % |
| MTP k=3 | 33.62 (34.38/32.86) | 23.75 (24.15/23.36) | 2.05/1.93, 2.13/2.00 | 33.30 / 24.54 | +1.0 % / −3.2 % |
| MTP k=3 + CAT-1 | **42.60** (42.31/42.89) | **26.94** (26.97/26.91) | 2.44/2.49, 2.37/2.37 | 34.62 / 25.55 | **+23.1 % / +5.4 %** |

- **CAT-1 is active under V2** — `MTP draft-vocab shortlist ACTIVE (35251 ids)`
  logged (count 1 in that arm, 0 in the others), with graph capture on and **no
  eager fallback** in any arm, i.e. the scatter survives capture. Item 3's
  silent-loss risk is refuted live, matching the code audit above.
- **Spec decode works under V2** at k=3 with acceptance on par with V1 in the
  plain arm (2.05/1.93 vs V1's 2.05/2.05 at 64k; 2.13/2.00 vs 2.06/2.15 at
  120k) — the −3.2 % @120k is inside the arm spread and is the only
  below-parity cell.
- **Item 6 (KV/VRAM), same util + flags:** V2 `GPU KV cache size` 454,536 tokens
  (greedy, 14.2 GiB avail) vs V1 496,693; 386,513 (spec) vs V1 442,368 (13.48
  GiB avail) — V2 reserves 8.5–12.6 % more, and its graph capture costs
  1.69+0.80 GiB (greedy) / 2.15+0.87 GiB (spec) against V1's 0.71 GiB. On V2 the
  capture-ladder trimming therefore matters more, not less.
- **V2-CAT1-1 RESOLVED — the list's content is the mechanism (three-arm run,
  2026-09-14).** Same boot, same runner, same prompts, same k, only the draft
  list differs:

  | V2 arm | @64k | @120k | acc @64k | acc @120k |
  |---|---|---|---|---|
  | no list (plain k=3) | 33.62 | 23.75 | 2.05/1.93 | 2.13/2.00 |
  | **corpus-matched 35,251** | **42.60** | **26.94** | 2.44/2.49 | 2.37/2.37 |
  | mismatched 32,768 (control) | 35.22 | 22.80 | 1.72/1.71 | 1.66/1.62 |

  Reading: the ms saving is common to both lists (+4.8 %/−4.0 % for the
  mismatched one — the same ~2.5 ms/step the V1 controlled A/B measured), and on
  top of it the *matched* list acts as a good prior over what the target
  actually generates (+20 % acceptance), while the mismatched list is a bad prior
  (−15 %). So the V2 CAT-1 win is partly acceptance, it is content-dependent, and
  it is not a structural V2 artifact.
  **Exactness is established by audit**, not assumed: `gumbel_sample` caches the
  *masked* draft logits (`logits_cache=draft_logits`) and
  `rejection_sampler_utils.py` computes the acceptance ratio from that same cache
  (`draft_logit … / temp`), while the target keeps its own full `lm_head` — the
  ratio therefore uses exactly the distribution the draft was sampled from. At
  `temperature = 0` the draft path is a plain argmax
  (`gumbel_noised_argmax`, "or plain argmax at temp 0"), so the effect is the
  mask shaping *which* token the argmax lands on, not stochastic drafting (an
  earlier hypothesis, now discarded).
  Consequence for quoting: **V2 + matched CAT-1 at 42.60/26.94 is the fastest
  configuration measured on this box**, but it should be quoted as V2-specific
  (under V1 the same list showed no acceptance effect, so its V1 gain stays
  ~+4–5 %).
- (Superseded, kept for the record) **V2-CAT1-1 (acceptance anomaly).** In the CAT-1 arm V2's
  acceptance is ~20 % *higher* than V2's plain arm (2.44/2.49 vs 2.05/1.93 @64k;
  server-side `Mean acceptance length` 3.3–3.5 vs 3.0 independently agrees), while
  under V1 the same list showed no acceptance effect (controlled A/B: z = −0.83).
  Hypothesis: V2's draft sampling is seeded/stochastic, so restricting the draft
  head to a corpus-matched prior *shapes the draft distribution* (raising
  agreement), whereas V1's greedy drafts are unchanged by the restriction.
  Exactness should be unaffected — rejection sampling corrects any draft
  distribution as long as the target's logits are untouched, which they are (the
  target is a separate model with its own full `lm_head`). Disambiguating runs
  (queued in ROADMAP V2-CAT1-1): a *mismatched* control list of the same size
  (if it shows the same boost, the content is irrelevant and something structural
  is happening), plus a token-identity/PPL check with the shortlist active under
  V2. Corpus/list-shaping caveat for the *headline*: on V2, `34.62 → 42.60` mixes
  the list's ms saving with this acceptance effect, so quote the V2 CAT-1 number
  only with the caveat until V2-CAT1-1 lands.
- Wedge #83 (GPU1) hit the mtp3 arm's first launch; the retry passed and is
  recorded in `degradation.md`.

**MoE 35B parity (session E-1, in-process harness, same boot).** `docs/gfx906/_bench_gfx906.py`,
`BENCH_SAMPLES=4 BENCH_PP=2048 BENCH_TG=256 BENCH_MAX_SEQS=32`, single GPU, mclk 1000
in every sample: **V1 57.86 t/s** {58.13, 58.18, 57.04, 58.08} vs **V2 58.36 t/s**
{58.42, 58.36, 58.32, 58.33} → **+0.9 %**, and within 0.1 % of the recorded 58.43
reference. Runner identity confirmed the same way (bare `[model_runner.py:*]`
tags only in the V2 arm). No cudagraph fallback in either arm. Note the KV
direction is config-dependent here: in-process single-GPU V2 kept *more* KV
(130,944 vs 123,904 tokens) while reserving more for capture (0.35+0.05 vs
0.05 GiB) — unlike the TP=2 dense-27B serving case where V2 was smaller.

**KVLAYOUT-2 closed in this session too:** the three capture/lifecycle tests that
were skipped as "0.29 fused KV layout migration pending" were already written
against the fused-layout helpers (`_make_fused_cache`/`_kv_split`/
`_write_v_fused`) and pass unchanged — the skips were stale. The suite is now
**91 passed, 0 skipped** (`tests/kernels/attention/test_gfx906_fa.py`), including
the extended M3 test which now also asserts that a **full-length (V2-style) host
`cu_seqlens` slice with a garbage tail** gives bit-identical output.

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
