# max_model_len decode tax (N4) — persistent live-bounded gather+quantize — tax removed at 131k AND 262k, 2.57× at 262k

Branch: `gfx906/fa-masked-gather` (off `gfx906/main` @ `fed5851105`) · 2026-08-21/22
Model: `cyankiwi/Qwen3.8-27B-AWQ-INT4` (snapshot `63768c10`, local HF
cache) · 64 layers (48 GDN + 16 FA), Hq=24, Hkv=4 (2/GPU at TP=2), D=256
Plan: `plan_masked_fa.md` §2.2 design · Diagnosis:
`tp_decode_investigation.md` RESOLUTION / roadmap N4 · Reviews:
`plan_masked_fa_plan_rev_claude.md`, `plan_masked_fa_rev_glm5.md`,
`plan_masked_fa_rev_qwen.md`

## 2026-08-21/22 — N4 tax fixed: persistent grid-stride gather, all gates pass

**VERDICT:** SHIPPED · **GATE:** TP=2 serving A/B (PERSIST=0 vs =1 at
max_model_len 131072/262144, 1091-token prompt, tg=256 greedy, 3 reps,
util 0.93, capture sizes [1,2,3,4]).

## HYPOTHESIS

If the S8 −25% decode tax at max_model_len 262144 is the two-kernel
fallback's O(Sk_pad) work baked into the FULL-graph launch dims
(`gather_paged_kv_q8_kernel_v2` + `quantize_q8_0_dense_kernel`, both
launched at `Sk_pad = pad32(max_model_len)` with live `seq_lens` ignored
for the tail), then a single persistent grid-stride kernel with a fixed
capture-time grid and live-`seq_lens`-bounded work removes the tax at
every max_model_len — and 131k configs improve too (their gather was
frozen at 131k-wide for ~1.5k-live decode).

## What was done

- Kernel `gather_paged_kv_quant_persistent`
  (`csrc/gfx906_fa/gfx906_fa_gather.cu`) + launcher + pybind
  (`gfx906_fa.cpp`) + dispatch (`gfx906_fa_paged.py`, env
  `GFX906_FA_PERSIST`). Fixed 1024-WG grid (`GFX906_FA_PERSIST_GRID`),
  grid-stride over (seq,head,token) rows, per-seq register prefix
  (B ≤ 16), in-kernel q8_0 via `quantize_block_q8_0_halfwarp` (bit-equal
  to the dense/fused paths), V byte-copy; rows [seq_len, Sk) NOT written
  except `GFX906_FA_PERSIST_MARGIN` (default 128 = max D=256 FA tail-tile
  width) V-zero rows. One code path at every Sk; the > 65535 two-kernel
  fallback is the PERSIST=0 rollback path (kept in tree).
- Gates run, in order: NaN-tail (precondition for tail-write removal) →
  standalone bit-exact + timing probe → capture/replay probe (B=1..4) →
  suite test → PPL A/B → serving A/B. Probes in
  `benchmarks/kernels/gfx906/fa_{nantail,persist,capture_replay}_probe.py`.
- Build note: the tree's `.hip` files are untracked hipify artifacts; the
  `.cu` files are the build sources (hipify regenerates `.hip` into the
  build dir at configure time).
- Also: TP=1 in-model tax curve (`_bench_gfx906.py`, pp2048/tg256,
  4 samples, live ~1.5k): 0.95 util: 2816 → 25.02 t/s, 16384 → 23.87
  t/s (−4.6% already at 16k); uniform 0.93 rerun: 2816 → 25.13, 16384
  → 24.83 (−1.2%). The 0.95 run OOM'd at ≥ 65504 (known dense-27B
  warm-cache inductor mode, AGENTS.md — 0.93 is the working util); the
  0.93 ≥ 65504 points were never captured (platform launch-failure
  flake burst, then a stale-VRAM hold from a crashed run shrank the KV
  budget to 3.98 GiB) — not chased further: the serving A/B (THE gate)
  already establishes the in-model tax (P0 22.36 vs 15.91) and its
  removal, so the TP=1 curve is corroborating evidence only.

## Evidence FOR

- **NaN-tail gate PASS** (precondition): FA `forward` outputs bit-equal
  with the K_q8/V tail [sl, Sk) poisoned NaN — aligned/misaligned 32-row
  tiles (oob_check path), 1-row-before-Sk worst case, full, tiny, B=2
  ragged, GQA-packed (Hq=24/Hkv=2,4 → nc2=2). The FA kernel never reads
  rows at/beyond kv_max; tail contents are irrelevant to the output.
- **Standalone per-layer, Hkv=2, live 1536** (launch-regime evidence):
  | Sk_pad | two-kernel | persistent | speedup |
  |---|---|---|---|
  | 3328 | 52.4 µs | 27.8 µs | 1.9× |
  | 65504 | 740.3 µs | 26.2 µs | 28.2× |
  | 131072 | 1136.8 µs | 27.0 µs | 42.2× |
  | 262144 | 2257.1 µs | 26.7 µs | 84.5× |
  (Hkv=4: 101.5→55.2, 1157.8→56.4, 2269.9→56.0, 4508.4→54.0 µs. K and V
  bit-equal in-range; margin V rows verified zero. Persistent also beats
  the small-Sk fused V1 kernel (~40 µs at 3328 per devlog).)
- **Capture/replay bit-exact** at Sk_pad=262144 with live sl swept:
  B=1 sl ∈ {32,1536,2177,8192,262112,262144}; B=1..4 sl ∈
  {128,1536,262112,262144} — the full capture-size range. Grid frozen,
  seq_lens re-read at replay. Suite test:
  `test_persistent_gather_capture_replay_large_sk` (passes).
- **PPL gate**: dense 27B, 12 prompts / 359 tokens / 0 top-20 misses:
  PERSIST=0 → **10.5516**, PERSIST=1 → **10.5516** — identical.
- **Serving A/B (THE gate)**, TP=2, mean of 3 greedy reps:
  | arm | t/s (reps) | vs P0 |
  |---|---|---|
  | 131k, P0 | 22.36 (22.39/22.37/22.32) | — |
  | 262k, P0 | 15.91 (15.94/15.91/15.89) | −28.8% (the S8 tax) |
  | 131k, P1 | 40.86 (40.86/40.84/40.87) | **+82.7%** |
  | 262k, P1 | 40.89 (40.89/40.87/40.90) | **+157.0%** |
  P1 residual tax (131k vs 262k): +0.07% = noise. Step-time transfer:
  131k Δmeasured 20.2 ms vs standalone (1137−27)µs×16 = 17.8 ms;
  262k Δmeasured 38.4 ms vs (2257−27)µs×16 = 35.7 ms — in-model
  ≈ standalone (×1.07–1.14; the S8 session's 0.47 coefficient was
  build/harness-specific). Reproduction note: the S8 39.9/29.9 numbers
  were the mtp2 harness; plain-greedy absolute values differ, but the
  P0 tax (−28.8% here vs −25% there) and the P1 fix transfer.

## Evidence AGAINST / caveats

- **Greedy 12×128 hashes diverge between P0 and P1 (3/12 prompts)** —
  refuted as a PERSIST effect by controls: P0-vs-P0 diverges too
  (p11), and p00 alone (batch=1, max_num_seqs=1) produces 3 distinct
  outputs across 3 P0 runs, with the batch-12 "P0-only" and "P1-only"
  values cross-appearing under both configs at batch=1 (1dd4a52d under
  P0×1 + P1×2; b10a3f65 under P0×1 + P1×1 in batch-12). First 4 tokens
  identical across all runs; PPL/prefill identical ⇒ the nondeterminism
  is in the DECODE path of this hybrid model (GDN triton decode is the
  prime suspect), not the FA gather. Bit-exact greedy is not a valid
  gate instrument for this model family; PPL + kernel-level bit-exact
  gates carry the correctness evidence. (Future: if a bit-exact greedy
  gate is wanted, run it on a pure-FA model or isolate the GDN
  nondeterminism first.)
- **Long-live-context A/B not run** (live ~1.1k only). Expected from the
  traffic math: at live 128k / Sk 262k, P1 does 128k live rows
  (≈0.5 ms/layer bandwidth-bound) vs P0's 262k rows — P1 still wins,
  less dramatically; the mechanism (work = live, not Sk) is kernel-level
  verified. Follow-up if the serving population is long-context-heavy.
- **rocprofv3 (plan step 1) remains unusable** in-model (attach thread
  not exposed; direct launch dies hipErrorLaunchFailure at device init,
  both TP=1 and TP=2). Standalone probes + serving A/B substitute; the
  tax split (gather vs FA-compute) is confirmed by the S8 eager-no-gap /
  graph-gap evidence plus this A/B closing the graph gap.

## Platform flake, same nights (context, NOT this change)

4+ `hipErrorLaunchFailure` wedges during TP=2 weight-load (S4 residual,
historical ~1/3; cold GPUs at 31°C, clean VRAM between runs; recovered
by retry; one TP=1 probe also aborted once and passed on rerun). BACO
reset / power-cycle need root, which this user lacks — if a session hits
the burst, escalate for a reset. Also: a server that boots past /health
can still half-dead (shm-broadcast timeout) and serve empty
`finish_reason: stop` streams — check request output, not just /health.

## Interactions / superseded-by · Refrigerated residue

- **Supersedes** the single-bound bounded-capture design and its
  dispatch-fallback (tp_decode_investigation.md CORRECTION section): the
  kernel live-bounding is the recommended path; no bound/fallback code
  built, none needed — one graph serves every live Sk. Multi-tier
  capture (§2.3) not needed.
- **Roadmap N4: RESOLVED** (this branch; merge to `gfx906/main` after
  review). `tp_decode_investigation.md` RESOLUTION: fix lever = this
  kernel.
- **Default flip**: `GFX906_FA_PERSIST` default ON after the A/B gate
  (kill switch `=0`); margin_zeros stays 128 (defensive, ~64 KB/head,
  removable to 0 — the NaN gate passed).
- Refrigerated: (a) delete the > 65535 two-kernel fallback + V2
  `cached_version` machinery once PERSIST=0 has no users; (b)
  `GFX906_FA_PERSIST_GRID` sweep (512/2048) — 1024 is fine at every
  probed shape; (c) LEGACY=0 direct-paged revisit (COW gap, N2) is
  unaffected and still blocked as before.
