# 0.31.0 merge review — scope, conflicts, and what the fork has to re-port

> Analysis for decisions (no branch cut, no code changed). Target release:
> upstream **v0.31.0** (2026-10-05, branch tip `db9527a468`); v0.32.0 is only at
> rc2 (`7960eeb9de`), so 0.31.0 is the latest *release* and the right merge target.
> Related: [`ROADMAP.md`](ROADMAP.md) (the merge train index),
> [`MERGE-0.30.0-review.md`](MERGE-0.30.0-review.md) (the previous sheet),
> the `upstream-merge` skill ([`SKILL.md`](../../.agents/skills/upstream-merge/SKILL.md)),
> and [`AGENTS.md`](AGENTS.md) merge-train rule 6.

## Base decision — cut from the 0.30 train, not from `main`

Past trains were cut from `main` because `main` was the previous release's tip
(`gfx906/v0.29.0` @ `524ac6f2d6` for the 0.30 merge). **That invariant is broken
right now:** `main` (`6ccbb03119`, the 0.30.0-final snapshot) is **83 commits
behind** `gfx906/v0.30.0` (`57530039ef`), which carries the validated post-0.30
work (WHT-1/QSA clamp, GDN-1, MBT-1r, PLE/GQA fixes). Cutting from `main` would
either drop it or force a second merge.

Recommendation: `git merge --ff-only gfx906/v0.30.0` to promote `main` to the
0.30 line first (the closeout promotion the pattern already uses), then cut
`gfx906/v0.31.0` from `main`; or cut directly from `gfx906/v0.30.0`. Either way
the branch point is `57530039ef` — and that costs **6 extra conflicts** (the
qwen4_exp/PLE/rocm/runner files), because that work is real and collides with
upstream 0.31.0.

## Scope (re-runnable)

```bash
BASE=gfx906/v0.30.0            # 57530039ef
UP=upstream/releases/v0.31.0   # db9527a468 = tag v0.31.0
MB=$(git merge-base $BASE $UP) # f2aad6aa70 (2026-09-15)
git merge-tree --write-tree $BASE $UP 2>&1 | grep -c '^CONFLICT'   # 34
git diff --shortstat $MB $UP    # upstream side
git diff --shortstat $MB $BASE  # fork side
```

| quantity | value |
|---|---|
| upstream commits in 0.31.0 | **728** |
| upstream files / lines | 3155 files, +157 157 / −46 584 |
| fork commits since the shared base | 673 (425 files, +81 374 / −1 725) |
| files touched by **both** sides | 128 |
| conflicted files | **34** (28 if cut from the stale `main`) |
| class breakdown | **21 live hand-merges (L) · 13 upstream carries (C)** |

Past calibration: 0.28.0 = 3 upstream commits (trivial), 0.29.0 = 561 commits /
23 conflicts, 0.30.0 = 772 commits / 31 conflicts. 0.31.0 is the largest merge
yet, and this time the fork side is large too (425 files since the base).

Classes: **L** = our live gfx906 code (hand-merge) · **C** = an upstream commit we
carry, or a fix upstream now has (take theirs; `--theirs` only when the whole file
is a carry).

## Table 1 — the 34 conflicts

| file | ours | theirs | class | our-side change → recommendation |
|---|---|---|---|---|
| `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` | 508+/42− | 208+/111− | **L** | Nemotron/MiniMax wna16 backend + gates. Re-port onto upstream's **canonical N-first format (#52798)**, Humming (#56685) and Zen DA8W4 (#54024). Highest-effort file |
| `vllm/model_executor/layers/quantization/compressed_tensors/.../compressed_tensors_moe_wna16.py` | 19+/2− | 96+/101− | **L** | asymmetric CT W4A16 + review fixes. Re-express the zp layout in the N-first `shape_map`; backend transposition now lives in `convert_to_wna16_moe_kernel_format` |
| `csrc/libtorch_stable/moe/moe_wna16_utils.h` | 84+/124− | 0+/8− | **L** | fork WNA16 expert support. Upstream #57621 removed the unused `float22num2` helpers — check the fork's line-41 copy has no caller, then drop it; keep the kernel |
| `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` | 16+/2− | 49+/7− | **L** | keep the fork's asymmetric path; adopt #58051 (skip non-local experts), #57984 (swiglu clamp), #48606 (Quark W4A16) |
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | 113+/63− | 5+/9− | **L** | fork MoE hunk (MiniMax); small theirs side, hand-merge |
| `vllm/_custom_ops.py` | 241+/2− | 151+/161− | **L** | fork M1/NPT MoE align, GEMV, `gptq_shuffle_awq` bindings; upstream **removed AllSpark (#58001)** + new Whisper/CPU/multimem bindings. Hand-merge; drop the allspark wrappers |
| `vllm/model_executor/layers/utils.py` | 480+/44− | 25+/7− | **L** | `SKINNY_M16` / GEMV / C4 / `triton_matmul`. Must be re-ported onto upstream's ROCm GEMM changes — **#58566 (69 wasted contiguous copies), #53283 (wvSplitK), #58923 (CPU-tensor fallback)**. Perf-relevant, A/B after |
| `vllm/v1/worker/gpu/model_runner.py` | 156+/3− | 209+/144− | **L** | fork V2 GC-freeze / PLE graph-capture guards vs upstream's **MRV2 rewrite** (#43091, #58411, #58400, #54335, …). Keep V1 the default; re-port the guards |
| `vllm/models/qwen4_exp/amd/ple_layer.py` | 375+/31− | 92+/12− | **L** | fork PLE mmap host-table vs upstream's **official PLE n-gram CPU offload #57497** + #58114. **Decision point** (ROADMAP QSA-FN-11): adopt upstream's path or keep ours — do not keep both |
| `vllm/models/qwen4_exp/amd/ops/qsa.py` | 251+/13− | 0+/6− | **L** | fork QSA fp16 tiled indexer + width clamp; theirs is only the ruff docstring sweep. Keep ours |
| `vllm/models/qwen4_exp/config.py` | 140+/5− | 15+/4− | **L** | fork PLE split guard; keep it over upstream's Transformers-v5 naming churn |
| `vllm/platforms/rocm.py` | 228+/13− | 142+/49− | **L** | fork FA-COVER-1 lost-FA guard + CUSTOM-FA registration vs upstream ROCm refactors (#58983, #54988, #56343, #57229). Hand-merge the guard |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` | 278+/35− | 7+/45− | **L** | MiniMax fp16-sparse re-apply + the `#47042` guard (on `seq_lens_cpu_upper_bound`). Keep both on upstream's `#58002` dead-code removal / `#57252` |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | 108+/0− | 123+/43− | **L** | SYV-4 sort-free small-k (`VLLM_GFX906_SORT_FREE_SMALL_K`, default on). Adopt upstream FlashInfer-0.7 / MRV2 / XPU dispatch around it |
| `vllm/model_executor/models/qwen3_dflash.py` | 41+/1− | 101+/20− | **L** | fork `_dense_kv_rows` promotion; take upstream's #54373 RoPE-layout fix underneath |
| `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py` | 127+/60− | 106+/15− | **L** | fork W1 non-spec-peel; upstream #53300 NIXL state layout. Keep the peel logic |
| `tests/quantization/test_moe_wna16.py` | 476+/0− | 45+/10− | **L** | keep the fork suite; add upstream's N-first shape cases |
| `tests/model_executor/layers/test_rocm_unquantized_gemm.py` | 382+/3− | 20+/4− | **L** | keep the fork GEMV/fp32-mv tests on upstream's new impl |
| `tests/v1/attention/test_gdn_metadata_builder.py` | 79+/39− | 146+/0− | **L** | keep the fork spec-mixed-peel test alongside upstream's new builders |
| `tests/models/qwen4_exp/test_qsa_reference.py` | 37+/9− | 79+/19− | **L** | keep the fp16 arm; add the fp8-KV arm upstream brings (#55557, #58961) |
| `pyproject.toml` | 5+/0− | 66+/0− | **L/D** | fork CUSTOM-FA build glue + upstream packaging; additive both — keep both |
| `vllm/entrypoints/scale_out/token_in_token_out/protocol.py` | 16+/0− | 104+/26− | **C** | fork carries #53187; 0.31.0 has its own version — take theirs |
| `vllm/entrypoints/scale_out/token_in_token_out/serving.py` | 35+/3− | 80+/5− | **C** | same #53187 carry — take theirs |
| `tests/entrypoints/scale_out/token_in_token_out/test_generate_stream.py` | 357+/0− | 501+/1− | **C** | test half of #53187 — take theirs |
| `rust/src/server/src/routes/inference/generate.rs` | 268+/4− | 470+/8− | **C** | Rust half of #53187 — take theirs |
| `rust/src/server/src/routes/inference/generate/types.rs` | 13+/0− | 102+/0− | **C** | same — take theirs |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` | 47+/2− | 497+/103− | **C** | fork carries #57049/#57077 — take theirs |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py` | 64+/14− | 95+/24− | **C** | carries #57570/#57049/#57077 — take theirs |
| `tests/v1/kv_connector/unit/test_nixl_connector_hma.py` | 169+/0− | 255+/2− | **C** | test half of #57077 — take theirs |
| `tests/v1/attention/test_mla_backends.py` | 47+/1− | 6+/5− | **C** | carries #57285 — take theirs |
| `tests/v1/core/test_prefix_caching.py` | 95+/9− | 425+/99− | **C** | carries #57049 — take theirs |
| `tests/v1/engine/test_engine_args.py` | 43+/0− | 75+/0− | **C** | carries #55760/#55861; verify no fork-only hunk before taking theirs |
| `docker/Dockerfile.cpu` | 6+/0− | 54+/85− | **C** | carries #57871 — take theirs |
| `requirements/rubin-prerelease.txt` | 16+/6− | 16+/6− | **C** | carries #56545; 0.31.0 has its own bump — take theirs (CUDA/Rubin, not our stack) |

## Table 2 — release-notes breaking changes, checked against the fork

Source: the v0.31.0 release notes' "Breaking Changes & Deprecations" section.

| item | fork impact | action |
|---|---|---|
| `VLLM_PLE_CPU_OFFLOAD` removed → `--engram-config` (#57937) | **applies** — directly hits the fork's PLE work and ROADMAP **QSA-FN-11** ("adopt upstream's official PLE CPU offload", previously blocked on a 0.30.1-based line) | with #57497 landing here, switch the PLE path or record why the mmap stays; delete the env read |
| AllSpark INT8 W8A16 removed (#58001) | **applies (mechanical)** — `_custom_ops.py` is a conflict; `tests/kernels/quantization/test_allspark_gemm.py` and `benchmarks/kernels/benchmark_marlin.py` usages are upstream code | take the removal; confirm no fork caller and no dangling `torch.ops._C.allspark_*` |
| `transformers` upper bound in requirements (#59614) | **applies (build)** | verify our checkpoints load inside the bound; keep the fork's manual ROCm/gfx906 pins as a commented note in `requirements/build/rocm.txt` (skill lesson 11) |
| `--enforce-eager` also disables JIT kernel warmup (#58197/#58593) | **applies (measurement)** | eager bench/probe numbers change regime; note it next to any eager TTFT |
| per-request `mm_processor_kwargs`/`media_io_kwargs` gated (#58830) | **applies if any local client sends them per request** | audit the gfx906 serving clients/tester recipes; add `--trust-request-mm-kwargs` where used |
| `tokenizer_mode="slow"` removed (#58545) | **N/A** — only an upstream example file uses it | — |
| `--enable-mamba-fine-grained-prefix-cache` → `--enable-mamba-shared-prefix-checkpoint` (#57382) | **N/A for fork code** — all uses are upstream files + one upstream test | adopt the rename as it lands |
| `quantization="fp8"` → `fp8_per_tensor` shorthand (#53585), Quark online quant removed (#51800) | **N/A** — no fp8 gfx906 path | — |
| `return_assistant_tokens_mask` removed from `/render` (#57520) | **N/A** — hits are upstream code | — |
| `VLLM_XPU_ENABLE_XPU_GRAPH` removed (#51600) | **N/A** | — |
| comma form of `--collect-detailed-traces` removed (#55702) | **N/A** — fork profiling recipes use JSON config | — |
| `VLLM_BATCH_INVARIANT` breakable graphs (#57586/#56377) | **N/A** | — |
| MRV1 + PP>1 + async + structured output rejected (#56250) | **N/A** — the fork serves V1 on TP=2, no PP | — |
| ROCm: torch 2.13 / Triton 3.8 / AITER v0.1.23; ROCR host segfault fix (#57328); AITER w4a4 ASM off (#57055) | mostly **already our stack**; #57328 may bear on the degradation record | note beside `degradation.md`; no patch expected |

## The substantive re-port list (ordered by effort × risk)

1. **CT WNA16 N-first canonicalisation (#52798).** The largest semantic change
   for the fork: it rewrites `compressed_tensors_moe_wna16.py` (backend
   transposition deferred to `convert_to_wna16_moe_kernel_format`,
   `shape_map` → plain N-first tuples) and the `int_wna16` oracle. The fork's
   asymmetric zero-point support, its fail-closed act-order gate and the
   repack helpers must be re-expressed in N-first shapes. Gate: MoE PPL band +
   `tests/quantization/test_moe_wna16.py` + the Nemotron/Gemma-4 CT paths.
2. **`layers/utils.py` ROCm GEMM re-port.** #58566 / #53283 / #58923 change the
   `rocm_unquantized_gemm_impl` skeleton the fork's `SKINNY_M16` / GEMV dispatch
   sits on. Gate: house 35B bench + `test_rocm_unquantized_gemm.py`; A/B the new
   upstream path against the fork's.
3. **Qwen4Exp / PLE convergence.** Upstream now has the official PLE CPU
   offload (#57497) and fp8-KV QSA (#55557) with the profiling-KV fix (#58961);
   the fork has fp16 QSA (4.4× kernel win), the width clamp and the mmap PLE
   table. Decide QSA-FN-11, then keep exactly one PLE path.
4. **MRV2 guards.** Upstream rewrote the MRV2 worker; the fork's GC-freeze /
   PLE graph-capture guards in `v1/worker/gpu/model_runner.py` must be re-ported.
   The fork still pins `VLLM_USE_V2_MODEL_RUNNER=0` in every recipe.
5. **FA reuse-hygiene.** `platforms/rocm.py` (FA-COVER-1 lost-FA guard, CUSTOM
   registration) and `rocm_aiter_mla_sparse.py` (fp16-sparse re-apply, `#47042`
   guard) — both have upstream refactors underneath.
6. **Carry cleanups.** 13 files are upstream carries: take theirs; delete the
   fork's duplicate where upstream converged (the 0.29/0.30 merges did this for
   dflash2 and Zen CPU MoE).

## The procedure to replicate (from 0.28/0.29/0.30)

1. **Scope first** — this review sheet (`MERGE-0.31.0-review.md`), before touching
   the tree. Fix stale default/gate comments (AGENTS.md merge-train rule 6).
2. **Cut `gfx906/v0.31.0`** from the 0.30 line (promote `main` first if the
   "from main" invariant is to be kept), then `git merge --no-commit --no-ff
   v0.31.0`; resolve by hand. Do **not** merge first / onto an rc.
3. **Static sweep** — `.agents/skills/upstream-merge/scripts/post-merge-sweep.sh`
   (conflict markers, tree-wide ruff F821/F811, in-memory compile, duplicate-def
   scan, import smoke).
4. **Rebuild the extensions** (`setup.py build_ext --inplace`) and **verify the op
   schemas** changed; the in-tree `.so` carries pre-merge schemas otherwise.
5. **Validation ladder** — fork unit suites for the touched areas → PPL probe on
   a dense **and** an MoE model (recorded bands: Qwen3.8-27B 10.5472–10.5516 ·
   Qwen3.5-35B 15.94–16.02 · Nemotron-3.5 26.96–27.02) → TP=2 serving
   smoke/bench.
6. **Commit merge, then fixes, then docs — separately**; `git status` clean before
   claiming validation.
7. **Record**: one dated `CHANGELOG.md` entry, append the outcome here, update
   `ROADMAP.md`; tag `gfx906/v0.31.0-final` at closeout; promote `main`
   (fast-forward) as the 0.30 line did.

Merge-commit message shape (as 0.29/0.30): parents = fork tip + upstream tag;
body = upstream delta (commits/files/conflicts) + resolution themes + post-merge
consistency fixes + known follow-ups.

## Open questions to decide before the cut

- **QSA-FN-11 / PLE**: adopt upstream's `--engram-config` + #57497 and delete
  the bespoke mmap, or keep the mmap for one more release? This changes conflict
  #15 (`ple_layer.py`) from a hand-merge to a "take theirs".
- **Base**: promote `main` to `57530039ef` first (keeps the documented
  "cut from main" invariant) or cut from `gfx906/v0.30.0` directly?
- **AllSpark**: adopt the removal wholesale (mechanical) — confirm no local
  checkpoint or recipe used it.
- **`--trust-request-mm-kwargs`**: which local serving clients (if any) send
  per-request multimodal kwargs.

## Outcome (2026-10-16)

**Merged.** `main` was fast-forwarded to the 0.30 train tip (`8d050331fa`) and
pushed; `gfx906/v0.31.0` was cut from it and merged with `v0.31.0`
(`db9527a468`) as **`db60e2cbd9`** — 34 conflicted files, exactly the set
scoped above. The decisions in "Open questions" resolved as: base = the 0.30
train (promote-then-cut); PLE = the fork's mmap kept (QSA-FN-11 follow-up);
AllSpark = removal adopted.

Five semantic fixes were needed beyond the conflict hunks (separate commits,
see [`CHANGELOG.md`](CHANGELOG.md) 2026-10-16): the N-first CT WNA16 repack
(`36bd5a82d0`), REL30-1's fused M=1 align defaulting off (`427c445723`), the
`use_v2_model_runner` / `triton_matmul` dtype fixes + test adaptations
(`8c394235c8`), the W1 GDN `update_block_table` / MTP crash (`d4a182edfb`),
and the dead-helper fix from the self-review (`64c0922256`).

Gates run (all green): extension rebuild + op schemas; PPL dense 10.5472 /
MoE 15.9122 / Nemotron 27.0612 (0 top-20 misses each); 35B house bench
61.23 t/s and dense 27B 16.49 t/s (V1, graphs, mclk 1000); FA 104 passed;
mamba 915 passed / 33 skipped; MiniMax-M3 kernels 133 passed (4 need the
missing `aiter` package) and model 14 passed; canary 38.7 t/s; TP=2
Qwen3.8-27B MTP k=3 79.35 t/s @2k / 42.2 t/s @64k.

**#113 was retracted**: the mid-session "stall" was Triton 3.8 codegen +
autotune ahead of the GDN crash, not host degradation (`amdsmi` stays broken
across a reboot, so it is an environment property here).
