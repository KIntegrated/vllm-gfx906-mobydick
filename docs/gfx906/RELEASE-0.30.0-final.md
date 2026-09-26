# gfx906 fork — 0.30.0-final release snapshot (DRAFT — **not tagged**)

**Status: PREP IN PROGRESS — the release is BLOCKED on `REL30-1`** (a reproducible
V2 35B-MoE graph-capture page fault; see `ROADMAP.md` and
`degradation_details.md` 2026-09-26). This file is the release-notes draft; do not
tag `gfx906/v0.30.0-final` until the gates below are green and the blocker is
resolved or explicitly waived.

**What this is.** The fork's first release on the upstream **0.30.0** base:
branch `gfx906/v0.30.0`, cut from the 0.29.0-final `main` (`524ac6f2d6`) and
merged with upstream `v0.30.0` (`ced6857afa`) as merge commit **`8893a50e54`**
— 2017 files, **31 conflicted files**, per-file resolution record in the merge
commit body and `CHANGELOG.md` 2026-09-24.

## What this snapshot adds over 0.29.0-final

| item | change | measured |
|---|---|---|
| **Upstream 0.30.0 base** | 772 upstream commits; the `gfx906/v0.30.0` merge train (31 conflicts, ~11 live-code hand-merges) | build rc=0; op schemas verified |
| **GPTQ act-order adopted-out** | #54809 removed group/dynamic activation ordering stack-wide; the fork adopted it and re-ported its **M=1 4-bit max-ilp** dispatch onto the new kernel skeleton | both fork model families validated (below) |
| **MiniMax-M3 gfx906 re-port** | upstream's indexer/backend rewrite + the fork's fp16 casts / launch kwargs | 122 MiniMax-M3 kernel tests + routing/`_forward_mla`-fp16 gates (QSA-FN-14) |
| **`#47042` chunked-continuation guard** | restored on the `seq_lens_cpu`→`seq_lens_cpu_upper_bound` replacement | MLA-sparse suites green |
| **`upstream-merge` skill** | the merge procedure + lessons + `post-merge-sweep.sh` | — |

## Breaking changes the base brings

- **GPTQ group/dynamic activation ordering removed (#54809)** — `g_idx` is
  ignored; the Marlin/GPTQ/CPU/RDNA3 act-order kernels are gone. The validator
  now fails loudly instead of mis-dequantizing. No fork model uses act-order.
- **Env removals (#55353)**: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`,
  `VLLM_MM_HASHER_ALGORITHM`, the `use_fp4_indexer_cache` alias, the ROCm
  `CUDA_VISIBLE_DEVICES` fallback, and the `seq_lens_cpu` /
  `num_computed_tokens_cpu` attention-metadata properties. The fork keeps a
  deprecated-env read for the first via `get_from_deprecated_env_if_set` (fork
  decision; upstream removed it).
- Scale-out endpoints are opt-in (`--enable-scale-out`); the `all` Mamba cache
  mode is deprecated (fork uses `align`); YaRN is aligned with Transformers. None
  of these affect the fork's served models/recipes (see `CHANGELOG.md`
  2026-09-26 for the per-item applies / not-applicable check).

## Gates measured on this snapshot

Model runner is labelled per measurement: the merged base **defaults to V2**;
the fork's serve recipes pin **V1**.

- **PPL probe, dense**: Qwen3.8-27B-AWQ-INT4 **10.5472** (359 tokens, 0 top-20
  misses; band 10.5516 / 10.5472) — V2, eager.
- **PPL probe, MoE**: Qwen3.5-35B-A3B-AWQ **15.9840** (359 tokens, 0 top-20
  misses; C4 band 15.9361–16.0169) — V2, eager. Exercises the gfx906 MoE WNA16
  path (`_process_weights_gfx906` 10-tuple, `moe_gptq_gemm_gfx906`).
- **TP=2 Qwen3.8-27B-AWQ-INT4** (MTP k=3, filler, prefix caching OFF, bt4096,
  util 0.82, capture `[4,8,12,16]`, maxlen 131072) — **V1**:
  2k **79.1** · 64k **42.0** · 120k **40.7** t/s decode; TTFT 4.2 / 185.3 /
  442.2 s. Standard `vllm bench serve` (random 2048→256): **52.9 t/s** @
  TPOT 18.90 ms.
- **Suites**: `test_moe_wna16.py` 47 passed · MiniMax-M3 kernels 122 passed /
  13 skipped · MiniMax-M3 model+sparse 32 passed / 3 skipped · `test_config.py`
  (attention/indexer/cudagraph) 43 passed · `test_gptq.py` 2 passed ·
  `post-merge-sweep.sh` PASS.

## Gates NOT yet run (release blockers)

- **`_bench_gfx906.py` 35B house bench (the anchor number)** — **BLOCKED**:
  reproducible V2 graph-capture page fault (`REL30-1`). 0.29.0-final measured
  **59.77–59.86 t/s** on this bench.
- **`tests/kernels/attention/test_gfx906_fa.py`** (0.29: 104 passed) — not run.
- **Mamba suite** (0.29: 195 passed) and **`test_c4_layer0_quant.py`**
  (0.29: 12 passed / 3 skipped) — not run.
- GPU work was **stopped per the house burst rule** after a second failure
  (`hipErrorLaunchFailure` during weight load → `GPU reset(1)`); the host needs a
  reboot (no passwordless sudo in this session).

## What is *not* in this snapshot (unchanged from 0.29.0-final)

- The **Qwen3.8-Flash-Next / QSA** work stays on `gfx906/qsa-fn` (the merge-review
  sequencing). A 0.30.0 release therefore ships without it.
- **MiniMax-M3-AWQ** cannot be loaded here (needs > 32 GB); its gfx906 paths are
  gated model-free (QSA-FN-14) but not weight-validated.
- **kimi-k3** is not runnable here (its `kda.py` merge is reasoning-only).

## Next (before tagging)

1. Reboot the host; re-run the V2 35B house bench once (discriminates `REL30-1`
   host-state vs regression).
2. Run the FA / mamba / C4 suites on the refreshed boot.
3. If `REL30-1` persists: ship on the **V1 pin** (as the recipes already do),
   file the V2 follow-up under `DFL2-2`, and state the V1 basis in this document.
4. Bump the README header / release-snapshot list from the 0.29 line.
5. Promote `main` (fast-forward), push, tag `gfx906/v0.30.0-final`, publish the
   image.
