# 0.30.0 merge review — which conflicts are our live code, and which are dead weight

> Analysis for decisions (no code changed). Branch `gfx906/v0.29.0` →
> `upstream/releases/v0.30.0` (both fetched 2026-09-17). Related:
> [`ROADMAP.md`](ROADMAP.md) `UP-3` (the merge train), [`DEAD-ENDS.md`](DEAD-ENDS.md)
> (the one-pass dead index), `REFRIGERATOR.md`.
>
> **Correction to an earlier count:** this merge is **31 conflicted files**, not
> 148. The 148 came from `git merge-tree --write-tree --name-only | wc -l`, which
> prints `Auto-merging <path>` progress lines **on stdout** alongside the
> conflicted names (85 of the 149 lines). Trust the `CONFLICT` lines
> (`git merge-tree … 2>&1 | grep -c '^CONFLICT'` → 30) or filter
> `^Auto-merging`; every one of the 31 has both sides changed since the merge base
> (or is an add/add), while 116 of the "extra" paths exist on neither side.

## Method (re-runnable)

```bash
MB=$(git merge-base gfx906/v0.29.0 upstream/releases/v0.30.0)   # f5c3cc240b
git merge-tree --write-tree gfx906/v0.29.0 upstream/releases/v0.30.0 2>&1 \
  | grep '^CONFLICT' | sed 's/.*Merge conflict in //' | sort -u > conflicts.txt
# per file: our-side change / their-side change / our substantive commits
while read -r f; do
  echo "== $f"
  git diff --numstat $MB gfx906/v0.29.0 -- "$f"
  git diff --numstat $MB upstream/releases/v0.30.0 -- "$f"
  git log --no-merges --format='%h %s' gfx906/v0.29.0 --not upstream/releases/v0.30.0 -- "$f"
done < conflicts.txt
```

## Table 1 — the 31 conflicts, by class

Classes: **L** = our live gfx906 code (default-on or model-support; resolve by
hand) · **G** = our code that is **gated off by default** (archive candidate) ·
**C** = an upstream commit our fork carries (cherry-pick) or a fix upstream also
has (take theirs, drop our copy) · **D** = bindings/CI/test glue (mechanical).

| file | ours | theirs | class | our-side change (fork-only commits) → recommendation |
|---|---|---|---|---|
| `csrc/libtorch_stable/quantization/gptq/q_gemm.cu` | 178+/64− | 112+/423− | L | q_gemm M=1 max-ILP split + the 0.29-merge fixes; **kill-switch `VLLM_GFX906_QGEMM_M1_MAXILP` is default-ON** (AGENTS: build default). KEEP — highest-value hand-merge |
| `vllm/_custom_ops.py` | 241+/2− | 483+/155− | L | MoE M1/NPT default-on, dead S2-topk removal, Minimax-M3 bindings. KEEP |
| `vllm/model_executor/layers/utils.py` | 494+/60− | 50+/27− | L+G | fp32 router-gate GEMV (NH-3), `DENSE_GEMV`/`DOWN_GEMV` default-on, triton_matmul >2-D — **plus `SKINNY_M16` (default off)**. KEEP; split out the M16 hunk (see Table 2) |
| `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` | 513+/42− | 123+/152− | L | Nemotron-3.5 INT4/INT8 wna16 support + code-review fixes. KEEP |
| `tests/quantization/test_moe_wna16.py` | 460+/0− | 223+/25− | L | the wna16 tests for the above. KEEP |
| `vllm/model_executor/layers/mamba/mamba_mixer2.py` | 23+/0− | 155+/46− | **G** | **NH-4 fused grouped gated-norm, `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM` default OFF** ("pending the serving A/B gate"). ARCHIVE CANDIDATE |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | 76+/21− | 58+/17− | L | dense GEMV switches (default-on). KEEP |
| `vllm/models/minimax_m3/amd/model.py` | 128+/18− | 181+/14− | L | Minimax-M3-AWQ-INT4 gfx906 support (the train we are ON). KEEP |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` | 292+/79− | 373+/102− | L | Minimax-M3 sparse/indexer fixes (uses upstream's `VLLM_ROCM_MLA_SPARSE_*`). KEEP |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | 106+/0− | 24+/13− | L | SYV-4 port (`VLLM_GFX906_SORT_FREE_SMALL_K` default ON). KEEP |
| `vllm/model_executor/layers/quantization/compressed_tensors/.../compressed_tensors_moe_wna16.py` | 20+/2− | 14+/129− | L | asymmetric CT W4A16 MoE + wna16 review fixes. KEEP |
| `vllm/model_executor/kernels/linear/mixed_precision/exllama.py` | 59+/32− | 3+/31− | L | Minimax-M3-AWQ exllama path. KEEP (model-scoped) |
| `csrc/libtorch_stable/torch_bindings.cpp` | 13+/0− | 121+/22− | D | Minimax-M3/DSv4 kernel bindings. KEEP (additive) |
| `tests/test_config.py` | 58+/2− | 1120+/10− | L | DFlash2 spec-decode config + `VLLM_GFX906_SPEC_CG_SMALL` (default ON). KEEP, small |
| `vllm/config/attention.py` | 13+/1− | 27+/25− | D | Minimax-M3 attention config (rocm-gated). KEEP, small |
| `vllm/distributed/device_communicators/cuda_communicator.py` | 5+/4− | 126+/5− | D | force-disable custom all-reduce on this topology. KEEP, small |
| `vllm/model_executor/layers/quantization/auto_gptq.py` | 13+/2− | 7+/92− | D | rocm guard + `VLLM_ROCM_USE_SKINNY_GEMM` plumbing. KEEP, small |
| `vllm/models/kimi_k3/amd/kda.py` | 4+/2− | 44+/23− | L | kimi-k3 non-spec decode peel (model not runnable here). KEEP, small |
| `vllm/model_executor/models/deepseek_mtp.py` | 20+/6− | 16+/2− | D | MTP plumbing the Minimax train needed. KEEP, small |
| `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` | 6+/0− | 39+/0− | C | upstream "avoid sync in TRT-LLM ragged prefill" — 0.30.0 has its own version. TAKE THEIRS |
| `vllm/multimodal/utils.py` | 9+/4− | 11+/7− | C | upstream SHM prefix-covered-items fix. TAKE THEIRS |
| `vllm/v1/core/kv_cache_utils.py` | 0+/10− | 434+/38− | C | upstream "remove misleading mamba prefix-cache warning". TAKE THEIRS |
| `tests/v1/core/test_kv_cache_utils.py` | 0+/15− | 858+/33− | C | the test half of that fix. TAKE THEIRS |
| `vllm/v1/spec_decode/dflash.py` | 0+/15− | 11+/22− | C | upstream DFlash RoPE-layout fix. TAKE THEIRS |
| `vllm/_aiter_ops.py` | 2+/2− | 563+/187− | D | rocm guard/registration line. KEEP, small |
| `vllm/models/minimax_m3/amd/ops/index_topk.py` | 30+/0− | 1006+/306− | C? | our "port gfx906 minimax indexer fixes into 0.25" is a **backport**; 0.30.0 has the full implementation. TAKE THEIRS after verifying the sparse path still works |
| `tests/kernels/moe/test_zen_cpu_int8_moe.py` | 253+/0− | 251+/0− | C | add/add of the same upstream Zen test we cherry-picked. TAKE THEIRS |
| `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py` | 162+/0− | 394+/341− | C | upstream "[CPU][Zen] route int8 MoE through zentorch". TAKE THEIRS |
| `.buildkite/hardware_tests/cpu.yaml` | 3+/0− | 80+/23− | C | the CI hook of that cherry-pick. TAKE THEIRS |
| `.buildkite/test_areas/misc.yaml` | 7+/1− | 189+/24− | C | CI hook of an upstream PP-send fix. TAKE THEIRS |
| `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP2-PCP2-EP.yaml` | 1+/1− | 3+/1− | C | upstream CI-config tweak. TAKE THEIRS |

**Reading of Table 1:** only **one** of the 31 is our code that is *off by
default* (`mamba_mixer2.py`, NH-4). ~11 are our live feature code and need a real
hand-merge; ~10 are upstream commits we carry, where 0.30.0 already has its own
(or better) version and the right move is to **drop our copy** rather than
re-apply it; the rest is additive glue.

## Table 2 — fork-wide inventory of switches that are OFF by default

The interesting population for "move it to a dead branch": code reachable only
when an env flag is set, where the flag defaults off.

| flag | read at | status (docs/roadmap) | recommendation |
|---|---|---|---|
| `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM` | `mamba/mamba_mixer2.py` | NH-4, "default off pending the serving A/B gate" (`DEVLOG-nemotron-h.md`) — **the A/B was never run** | **archive + delete** from main (also removes one 0.30.0 conflict), or run the A/B if Nemotron TP=2 perf still matters |
| `VLLM_GFX906_FUSED_DRAFT` | `gfx906_fa/gfx906_fa_backend.py` | FD-1 **CLOSED** (NEUTRAL, stack-confounded); "the flag's only reader in-tree was A3's opt-in, stripped 2026-09-13" (`f8a9400789`); branches `archive/a3-fused-draft`, `archive/fd1-fused-draft-meta` exist | **delete the leftover flag read** (dead code by the roadmap's own record) |
| `VLLM_GFX906_SKINNY_M16` | `model_executor/layers/utils.py` + `csrc/rocm/dense_gemv_gfx906.cu` | W4 skinny M=5..16 GEMV variant, default off; the M=2..4 part is default ON and live | **decide**: either gate it for deletion (archive the kernel variant) or run the A/B the docstring implies; verify the M=2..4 path stays |
| `VLLM_GFX906_QUANT_LAYER0_MOE` | `quantization/auto_awq.py`, `quantization/c4_layer0_moe.py` | not in any dev log I find; fork-only file `c4_layer0_moe.py` | **verify then delete** (looks like a parked experiment) |
| `VLLM_GFX906_FA_STRICT` | `platforms/rocm.py` | a *policy* switch (raise instead of warn for non-CUSTOM FA) | **keep** — it is a fail-closed guard, not dead code |
| `VLLM_GFX906_MOE_BM` / `_M1` / `_NPT`, `GEMV_RPT`, `GEMVM_RPT`, `GEMV_I8_RPT`, `W8A16_INT8*`, `QGEMM_M1_MAXILP` | MoE/GEMV/HIP kernels | null-means-on semantics (default **on**) | keep; not dead weight |

Defaults already in the tree (`VLLM_GFX906_ALIGN_M1`, `DENSE_GEMV`, `DOWN_GEMV`,
`SORT_FREE_SMALL_K`, `SPEC_CG_SMALL`, `SPEC_GEMM`, `TOPK_SINGLE_GROUP` = `1`) are
live and should not be touched in this pass.

## Proposed plan (for a decision, not yet executed)

1. **Preserve first.** Create `archive/gfx906-dead-2` from `gfx906/v0.29.0` and
   make one commit per dead item (NH-4, FD-1 leftover, and whatever the
   verify-then-delete items turn out to be), so each is one `git revert` away —
   the pattern already used for `gfx906/preserve-dead-kernels` (S2 topk + C1
   stage-2, removed 2026-09-01) and the four `archive/*` branches.
2. **Delete from main** (and therefore from `gfx906/qsa-fn`, which is based on
   it): NH-4 + its flag; FD-1's leftover flag read; then the verify-then-delete
   pair. Expected merge effect: `mamba_mixer2.py` stops conflicting entirely and
   `utils.py` shrinks to its live hunks — i.e. the *only* off-by-default conflict
   site disappears.
3. **Do not** drop the live trains to make the merge smaller: MoE M1/NPT, GEMV
   switches, wna16/Nemotron, Minimax-M3, SYV-4, `SPEC_CG_SMALL`, the q_gemm
   max-ILP build and the Minimax backports are all default-on or active model
   support.
4. **For the ~10 upstream carries** (class C): take 0.30.0's version, then
   re-check the only ones that existed because a *model we serve* needed them
   (the Zen CPU path, the Mamba prefix-cache warning, DFlash RoPE) — if 0.30.0's
   own fix covers it, our cherry-pick retires on the spot.
5. **Verification for each deletion** (cheap, and the reason step 1 exists):
   `grep` for the flag in `docs/gfx906/*` for the recorded verdict; confirm no
   model in the supported set is documented as needing it (NH-4 off ⇒ Nemotron
   takes the generic path it already takes by default); keep the
   `_bench_gfx906.py` MoE reference number (58.40 t/s) and the PPL probe as the
   non-regression gate on the branch after each removal; then confirm the 0.30.0
   merge conflict count drops (expected 31 → 30, plus smaller `utils.py`).

## Open questions for Kevin

- NH-4: delete, or run the Nemotron TP=2 serving A/B it was gated on? (The code
  is small and gated, so deletion is cheap either way.)
- `SKINNY_M16` and `QUANT_LAYER0_MOE`: are these still wanted? Both are
  off-by-default experiments whose docs I cannot find; my default assumption is
  "archive them with the rest".
- Do we want the *upstream carries* retired wholesale on the merge (taking
  0.30.0's versions), or re-applied per model? That is ~10 of the 31 conflicts
  and would shrink the hand-merge to the ~11 live-code files.
