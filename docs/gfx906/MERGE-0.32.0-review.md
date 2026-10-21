# MERGE-0.32.0 — review sheet (COMPLETE, released 2026-10-21)

**Status:** **released.** All 59 conflicts resolved (`1e514c6fd0`), five
marker-less breakages fixed (`46aa74c262`, `aacf920d39`, `bf58123bb6`),
build + static sweep + PPL ×3 + suite gates green, Lightning and 35B serving
smoke green. Promoted to `main` at `5f41d480b9`; release snapshot:
`RELEASE-0.32.0-final.md`. Post-merge model fixes: `96db9355c3` (#68),
`5f41d480b9` (#69), `7fcd4205b1` (packed `auto` under MTP).

**Late finding (2026-10-21, image build): a sixth marker-less breakage.** The
from-scratch container build failed in the wheel stage on

```
csrc/rocm/moe_align_m1_gfx906.cu:39:10: fatal error: '../cuda_compat.h' file not found
```

The `csrc/libtorch_stable/* -> csrc/*` flattening moved that header to
`csrc/core/cuda_compat.h` (upstream includes it as `"core/cuda_compat.h"`); this
fork-only file kept the pre-merge relative path. Fixed in `9b85b0e9f4`,
verified by a local rebuild of that TU.

**The lesson is the gate, not the fix:** a green *incremental* `build_ext` did
not catch it, because the TU was already built and its object was reusable — the
include path only breaks on a clean compile. From now on the merge train's
definition of done includes **one from-scratch build of the serving image**, not
just `setup.py build_ext --inplace`. (Also fixed in the same pass: the build
wrapper piped `docker buildx` through `tee`, so `$?` reported `tee`'s status and
logged `rc=0` for a failed build — `set -o pipefail`, same class as the
capture-rc-before-substitution trap in AGENTS.md.)

**Source:** upstream tag `v0.32.0` (`5df63a7203`), merged into `main`'s tip.
**Merge base:** `3d5f4d4cd5`. **Sizes:** 822 fork commits / 840 upstream commits,
**59 conflicts**.

## 1. Structural change: `csrc/libtorch_stable/*` → `csrc/*`

Upstream flattened the stable-ABI source tree into `csrc/` (dual extensions
`_C_stable_libtorch` / `_moe_C_stable_libtorch` survive; only the *paths* moved).
The fork had **187 files** under the old prefix. Git renamed the untouched ones;
these needed hand work:

| item | resolution |
|---|---|
| `csrc/libtorch_stable/ops.h`, `.../torch_bindings.cpp` (UD: upstream moved them to `csrc/`) | Ported the fork's **4 op registrations** (26 lines) into `csrc/ops.h` / `csrc/torch_bindings.cpp`: `gptq_shuffle_awq_qweight`, `indexer_k_cache_fp16`, `cp_gather_indexer_k_cache_fp16` (+ the `region_stride/max_union_rows` schema fragment). Old files deleted. |
| `csrc/libtorch_stable/quantization/gptq/q_gemm_m1_maxilp.cu` (file-location conflict) | Renamed to `csrc/quantization/gptq/q_gemm_m1_maxilp.cu`. |
| `CMakeLists.txt` (2 hunks) | Took upstream's flattened source list **plus** our `csrc/quantization/gptq/q_gemm_m1_maxilp.cu`; kept `csrc/moe/moe_wna16.cu` **unconditional** (upstream moved it to CUDA-only, but the gfx906 WNA16 path needs it on ROCm); repointed the max-ilp `set_source_files_properties` rule from `${CMAKE_CURRENT_BINARY_DIR}/csrc/libtorch_stable/...` to `.../csrc/quantization/gptq/...`. **No `csrc/libtorch_stable` references remain in CMakeLists.** |
| `csrc/quantization/w8a8/int8/per_token_group_quant.cu` (ours listed it) | Upstream deleted the kernel; dropped the entry rather than resurrect a missing file. |

## 2. Conflicts resolved so far (13 of 51)

| file | class | resolution |
|---|---|---|
| `vllm/v1/core/kv_cache_utils.py` | L/C | Took upstream's new `_get_kv_cache_bytes_per_block(groups, layout)` signature (**generic KV cache layout, #57169**) and **dropped our `_glm5_next_tensor_layout` carve** — upstream generalised it into `KVCacheLayout`. Kept the `cast` import. |
| `vllm/v1/core/single_type_kv_cache_manager.py` | C | Theirs ×3: upstream's `cache_hit_alignment_tokens` / `reachable_block_mask(retention_interval=…)` / `is_retained_checkpoint` supersedes the #59146 prompt-end checkpoint fix we carried; also took the `KVCacheMetricsCollector` and `HiSparseResidentSpec` additions. |
| `vllm/v1/attention/backends/triton_attn.py` | L | Theirs ×3 (`max_3d_query_tokens` → `max_num_tokens_3d`); the fork's `num_par_softmax_segments` additions survive as non-conflicting context. |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | L | **Ours** — our V2-MAMBA-1 seed comment (the fix line is identical on both sides; ours records the wild-address failure mode). |
| `vllm/v1/worker/gpu_worker.py` | C | Theirs: `profile_run` now runs under `set_current_vllm_config`. |
| `vllm/_custom_ops.py` | L | **Ours** — `cpu_activation_lut_bf16` is a fork addition (`ff063e44e2`, MiniMax-M3-AWQ). |
| `vllm/config/compilation.py` | L+C | **Merged**: upstream's `adjust_cudagraph_sizes_for_spec_decode(uniform_decode_query_len)` (TP arg dropped) **plus** our `VLLM_GFX906_SPEC_CG_SMALL` sub-query-len capture restore. |
| `vllm/model_executor/layers/utils.py` | C | Theirs: `VLLM_BATCH_INVARIANT` early return; upstream's import list already carries `on_gfx906`. |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | L+C | **Merged**: our gfx906 bitwise-dequant + fp16-A block **plus** upstream's `VLLM_BATCH_INVARIANT → configs = None`. |
| `vllm/v1/attention/backends/gdn_attn.py` | L+C | **Merged** our decode-first invariant assert with upstream's `aiter_prefill_metadata`; then **theirs ×2**: upstream now builds `MambaPrefillCheckpointMetadata` (mamba prefill checkpoints) and assigns `non_spec_indices = prefill_indices = blk_table[~masks, 0]` unconditionally (which fixes the W1 crash differently). |
| `vllm/v1/worker/gpu/model_runner.py` | L | Union imports (`Iterable` + `contextmanager`); **kept** our `@_gc_freeze_around` (fork V2 GC-freeze guard, `2c5c8a0b52`). |
| `CMakeLists.txt` | L+C | See §1. |

## 3. Remaining conflicts (38)

CI/glue: `.buildkite/{intel_jobs/test-intel.yaml,scripts/hardware_ci/run-intel-ci-test.sh,test-amd.yaml,test_areas/compile.yaml,test_areas/lm_eval.yaml}`,
`docker/{Dockerfile,versions.json}`, `requirements/{common,cuda,rubin-prerelease}.txt`.

csrc: `csrc/hisparse_kernels.cu` (5), `csrc/moe/moe_wna16.cu` (5).

Fork-critical Python: `vllm/config/vllm.py` (2), `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` (3, 681 fork lines),
`.../compressed_tensors_moe_wna16.py`, `.../experts/zentorch_moe.py`,
`vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`.

Model paths: `deepseek_v2.py`, `nano_nemotron_vl.py`, `qwen3_5_mtp.py`, `qwen3_vl.py`,
`vllm/models/kimi_k3/amd/kda.py`, `vllm/models/minimax_m3/{amd,common}/**`, `vllm/models/qwen4_exp/amd/ple_layer.py`,
`vllm/v1/attention/backends/mla/{flashinfer_mla_sparse,index_group,rocm_aiter_mla_sparse}.py`,
`vllm/v1/hisparse/{coordinator,runtime}.py`.

Tests: `tests/quantization/test_moe_wna16.py`, `tests/models/qwen4_exp/test_config.py`,
`tests/v1/core/test_{prefix_caching,kv_cache_utils}.py`, `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`,
`tests/v1/attention/test_sparse_mla_backends.py`, `tests/v1/kv_connector/unit/test_hisparse_connector.py`,
`tests/distributed/test_mnnvl_alltoall.py`, `tests/entrypoints/serve/lora/test_serving_models.py`.

## 4. Porting opportunities (from the v0.32.0 notes)

1. **Generic KV cache layout (#57169) — highest value for us.** Upstream now has
   `vllm/v1/kv_cache_layout.py::KVCacheLayout` (`LBNHC`/`LBHNC`, aliases `NHD`/`HND`),
   resolved once by the engine (`resolve_kv_cache_layout`) and read through
   `cache_config.get_resolved_kv_cache_layout()` — including inside
   `_get_kv_cache_bytes_per_block`. Our E2 packed cell currently assumes LBHNC in
   `_ensure_packed_views` and publishes its mode out-of-band; it should become a
   layout the engine *resolves* (or at least be validated against
   `get_resolved_kv_cache_layout()`), which also kills the class of bug fixed in
   `ad69623e9c` (layout chosen from a stale flag rather than from the pool).
2. **`int4_per_token_head` / `int8_per_token_head` / `ultraquant_4bit` KV** are now
   first-class `CacheDType`s (#56198, #57057). **LAB-5 (#49)** wanted int8
   per-token-head KV for gfx906 — adopt upstream's implementation and only add the
   gfx906 kernel if the upstream one has no path here.
3. **Mamba prefill checkpoints** (#57329 Mamba2 internal checkpoints for NemotronH /
   Granite-H / Zamba2; #59759/#60797 with MTP + prefix caching). Our hybrid models
   (Lightning, Qwen3.5-35B) and our GDN sync-cadence work overlap this; the merge
   brings upstream's implementation (imported in §2's `gdn_attn.py`). **Gate it on
   the Lightning class** — it is the most likely free win of this merge.
4. **ROCm batch invariance (#52231, #60271).** Every backend checked, custom
   all-reduce under `VLLM_BATCH_INVARIANT=1`. Cheap experiment on MI50 and it makes
   our A/B methodology stricter (bit-identical replays).
5. **RDNA-side items to check against our `on_gfx1x` paths:** `ROCM_SEGMENTED_ATTN`
   (#59132) and gfx11 W4A16 tuning (up to 23 % TTFT, #56301). Both are RDNA-only
   upstream, so they are gated on whether the fork still ships a gfx1x path.
6. **`--mamba-ssm-cache-dtype` honored for KDA state (#59147)**, `--kv-cache-dtype-skip-layers`
   (#60023, #60142) — both interact with our packed cell / skip-quant reconcile
   (`page_size_padded`), which `_packed_spec` currently refuses. Re-check after the
   merge whether that refusal is still needed.
7. **Triton kernels off `tl.make_block_ptr` (#60502)** — our fork's Triton kernels
   still emit that deprecation warning on every run; the upstream rewrite is the
   template for ours.
8. **`ROCM_SEGMENTED_ATTN` (#59132, MERGED) — mine it for our custom FA.**
   An opt-in Triton backend spanning gfx11/gfx12 (RDNA3/3.5/4) that covers paged
   decode, prefill **and multi-token verification**, head sizes 64/128/256, GQA
   1-16, causal **and non-causal**, sliding windows and **attention sinks**, with
   FP8 KV gated to gfx12 (so its FP16 path is what concerns us — MI50 has no
   FP8). 4.4 k lines, no new compiler dependency. Three things are worth taking
   even though the backend itself is RDNA-only:
   - **Startup autotune + persisted launch-choice cache, shared between target and
     draft.** Our FA picks its tiles from fixed heuristics (`nc2`, the gfx906
     gather clip, per-file max-ilp). A one-time tuning pass persisted per
     `(D, GQA, window, batch-shape class)` is the cheapest structural win
     available to us and would also make our A/Bs reproducible across boots.
   - **Its segmented-decode algorithm** is a second opinion on the split-K /
     direct-paged route our M5 bake rejected at B=4 (`GFX906_FA_DIRECT_PAGED_Q8`);
     the "segment + in-kernel merge" structure is what our gather clip does
     implicitly. Worth reading before we revisit that route.
   - **Non-causal + sinks coverage** — FA-NONCAUSAL Stage 2 (symmetric ±window)
     is parked in our fork, and this PR is a working reference for the
     non-causal verify tile the DFlash assistant uses.
9. **`ultraquant_4bit` KV (#57057, MERGED) — portable shape, but INT4 not FP4.**
   Stores KV at 4 bits (FP4 E2M1 codes + UE8M0 power-of-two per-group-32 scales
   + **Hadamard-rotated keys**), ~2× density over fp8; decode runs a FlyDSL
   `D=256` kernel for GQA {6,8,16} on gfx950 with a Triton unified-attention
   fallback. **MI50 has no FP4 support**, so a gfx906 port must be **INT4**
   (symmetric int4 codes, per-group-32 scale — the same shape our Q8_0 path
   already uses) and the Triton fallback, not FlyDSL, is the carrier. The
   rotation is the interesting part and is cheap on this hardware (one
   `hipBLAS`-free elementwise pass per store, HBM-bound); it is what keeps
   outliers inside int4 range. Natural landing site: our **packed KV cell**
   (`K q8 | V fp16`, 784 B at D=256) becoming `K q8 | V int4` (~272 B), which
   would roughly double the capacity win LAB-9 measured (+22 % → ~+2×) and is a
   much bigger lever than LAB-5's int8 per-token-head plan (#49). Effort: real
   kernel work (store + dequant-into-fp16-tile + dot), so it belongs behind a
   capacity gate, not in this merge train.

## 5. Challenges / breaking changes that bite the fork

1. **QSA width validator removed.** `vllm/models/qwen4_exp/config.py` is gone and
   `indexer_qsa.py` uses `config.indexer_budget` directly, so upstream no longer
   hides selection widths outside {512, 2048}. Consequences:
   - our `_QSA_TUNED_TOPK_WIDTHS` / `qsa_uses_tuned_topk_width()` relaxation is
     **obsolete** (file deleted in this merge; `tests/models/qwen4_exp/test_config.py`
     must be reworked — its assertion has no subject any more);
   - **#37's LDS guard becomes load-bearing**: a config with a large indexer budget
     now reaches `top_k_per_row_decode` with `topK > 8192`, which gfx906 cannot
     serve — it will fail closed with our clear error instead of corrupting memory.
     That is the right behaviour, but it needs to be a documented gfx906 constraint
     (and ideally an early config-time rejection, not a kernel-time one).
2. **MRV2 is the ROCm default and the model-specific MRV1 exceptions were removed
   (#56360).** Our recipes pin `VLLM_USE_V2_MODEL_RUNNER=0` for Gemma-4 and
   Muse-Glimmer. Re-verify both after the merge — the V1 escape hatch is upstream's
   to withdraw (V1 is removed in 0.32 per the 0.30 preset note: *"Upstream removes V1
   in 0.32.0"* — check whether it actually happened).
3. **Removed:** cascade attention (`--disable-cascade-attn` only warns),
   `mamba_cache_mode="all"` (validation error now), the torch.compile SP/AsyncTP
   passes (+`PassConfig.enable_sp`/`fuse_gemm_comms`/`sp_min_token_num`), and the
   deprecated API surface (`wait_for_inflight_requests`, `AsyncLLM.pause_generation`,
   `vllm bench throughput --dataset`, tensorizer LoRA, the PyO3 rust tool parser).
   **Grep the fork for each** (skill step 2b) — we deleted the two SP/AsyncTP tests
   in this merge, but `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`-class env removals have
   bitten us before.
4. **`max_tokens` is now an upper bound, not a reservation (#57035).** Our benchmark
   clients (`_bench_gfx906.py`, `_bench_chat_serve.py`) and any timing comparison
   against 0.31 numbers must be re-read: the same request may now return fewer/more
   tokens relative to context, which changes t/s denominators.
5. **Python ≥ 3.11** (we run 3.12 — fine) and **transformers ≥ 5.16.1,<5.20.0**
   (pinset may need a bump; Muse-Glimmer / Bolt configs should be re-smoke-tested).
6. **ROCm dependency bumps:** AITER 0.1.24.post1, MoRI 1.2.3.post1 — take upstream's
   requirements, but our gfx906 AITER/MoRI usage is limited; verify nothing we rely
   on moved.

## 6. Not testable on this box (name them, do not claim them)

QSA / Qwen3.8-Flash-Next paths (no servable checkpoint), DeepSeek-V4/V4.1 and
Kimi-K3 AMD paths, AITER MegaMoE, RDNA (`on_gfx1x`) items, PP/PCP/DCP and KV-connector
changes, SM90/SM100-only kernels, Intel XPU/CPU backends, and the Rust frontend.

## 6b. Validation log (what the ladder actually found)

**Build** (`aacf920d39`) — three breakages, all in the `csrc` flattening's blast
radius, none of them with a conflict marker:

1. `csrc/cache_kernels.cu` lost its `cuda_utils` include (the flattening turned
   `../cuda_utils.h` into a nonexistent path) → `use of undeclared identifier
   cuda_utils` on our split-KV insert path.
2. Upstream's rewritten `csrc/cuda_utils.h` **silently dropped** the fork's
   `cuda_utils::ceil_div` *and* the `HOST_DEVICE_INLINE` macro. Restored with a
   conditional device qualifier (the header is also included from host-only TUs).
3. The ported op registrations aborted the whole extension at load:
   `aoti_torch_library_def API call failed`. **Diagnosis method worth reusing:**
   tag every `ops.def`/`ops.impl` with `std::fprintf(stderr, "REG line %d", __LINE__)`,
   rebuild, load — the last printed line names the failing statement. It localised
   to a def placed in the file's later fragment region; moving our two indexer
   defs/impls to the top of the `_C` fragment loads cleanly.

**Fast smoke test learned:** `torch.ops.load_library('vllm/_C_stable_libtorch*.so')`
catches registration breakage in ~2 s, long before any model load. Run it
immediately after every rebuild that touches bindings.

**Import smoke** — one more semantic conflict (no marker): upstream deleted
`vllm/models/qwen4_exp/config.py`, but `vllm/config/speculative.py` and
`vllm/config/vllm.py` still imported the fork's family helpers from it, so the
tree only failed at import (`ModuleNotFoundError`). Fixed in `bf58123bb6` by moving
the helpers to `vllm/models/qwen4_exp/family.py`, deliberately **not** deriving the
family set from `transformers` classes: the release pins `transformers >= 5.16.1`
and the venv still has an older one, so the set is a static tuple and
`QWEN4_EXP_CONFIG_CLASSES` degrades to `()`.

**PPL gates (in-process probe, 359 tokens):**

| arm | PPL | band | verdict |
|---|---|---|---|
| MoE `Qwen3.5-35B-A3B-AWQ` | **15.9555** | 15.94–16.02 | pass, 0 top-20 misses |
| dense `Qwen3.5-27B-AWQ` | **14.3750** | its own recorded baseline 14.3750 | pass, exact match, 0 misses |

The dense arm deliberately used `Qwen3.5-27B-AWQ`: the skill names
`Qwen3.8-27B-AWQ-INT4` (band 10.5472–10.5516) but that checkpoint is not on this
box, and `ROADMAP.md` already warns *"do not use Qwen3.5-27B, which reads
14.3750"* for that band. Against its own recorded baseline the arm is exact to
the last digit, which is the stronger signal (it exercises `gptq_shuffle` +
`ExllamaLinearKernel`).

**Tests:** `tests/config/test_qwen4_exp_family_match.py` **43 passed**;
`tests/platforms/` 33 passed with the same 15 `test_cpu.py` failures that fail on a
reverted tree on this host (pre-existing, host-specific).

**Dependency divergence to watch:** upstream now pins torch `2.15.0.dev` on CUDA
(`requirements/rubin-prerelease.txt`) and `transformers >= 5.16.1`; our ROCm venv
is torch `2.13.0+gfx906` and an older transformers. Two of upstream's 0.32 perf
items ("fusion passes restored under PyTorch 2.15", "static FP8 quant fusion on
recent torch") therefore do not reach this build, and the Qwen4Exp/PLE paths stay
unservable locally.

**P32-7 (#65) closed 2026-10-20.** v0.32.0 keeps V1 (`VLLM_USE_V2_MODEL_RUNNER`
honoured; only HiSparse raises and watermarking overrides) but removes the
model-specific MRV1 exceptions, so V2 is the default for every architecture.
Verified on the merged base with the env var unset: Muse-Glimmer + DFlash assistant
→ `Using V2 Model Runner`, drafter graphs 1/1, coherent output, 38.6–40.5 t/s
decode. Gemma-4 is not downloaded on this box, so its arm is covered by the
2026-09-15 templated V1-vs-V2 evidence rather than re-measured here.


### Gates + two new findings (2026-10-21)

**PPL gates complete** (three model classes, in-process probe): MoE 35B **15.9555**
(15.94-16.02), dense 27B **14.3750** (recorded baseline, exact), **Nemotron-3.5-
Lightning mixed INT4/INT8 26.9922** (26.96-27.02) -- all 0 top-20 misses.
`tests/quantization/test_moe_wna16.py`: 50 passed, 9 skipped (all CUDA/XPU-only
backends, justified). The stashed BOLT-1b `ckpt_*_proj_name` fix is **subsumed**
by upstream's own adoption plus our `None` filter; its one unique defensive bit
(getattr defaults) is folded in (`6fd3fc966e`) and the stash is dropped.

**Finding 1 -- Lightning at TP=2 needs `--enable-expert-parallel`** (as AGENTS.md
already says): the CT INT4 group-64 experts have `moe_intermediate_size 1856`, so
TP=2 shards to 928 and `928 % 64 != 0` fails in `create_weights`. With EP the
class loads, selects `GFX906_HIP` WNA16, runs on V2, sizes 3,042,304 KV tokens and
captures graphs -- so the earlier in-process `Could not find an active GPU backend`
was the documented inductor fork-worker hazard, not a model defect.

**Finding 2 -- the packed-KV auto gate now reads +0.0 % on the 35B.** The serving
smoke logs `packed KV layout left off under gfx906_fa_packed=auto: predicted KV
capacity +0.0 %` with block 1072 -> 1408 at `max_model_len 32768`, where 0.31 read
**+33.6 % (on)** with block 1056 -> 1376 at 8k. Either the upstream layout/block
changes moved the geometry or our gate's arithmetic no longer matches the merged
KV accounting (which now takes `get_resolved_kv_cache_layout()`). This must be
re-derived before trusting `auto` on the 0.32 line -- and it is exactly the
overlap P32-1 (#60) exists to remove.

## 7. Next steps (in order)

1. Resolve the remaining 38 conflicts (fork-critical Python first: `int_wna16.py`,
   `compressed_tensors_moe_wna16.py`, `moe_wna16.cu`, `hisparse_kernels.cu`,
   `config/vllm.py`).
2. `scripts/post-merge-sweep.sh` (F821/F811, duplicate `def`, import smoke).
3. ~~Rebuild + op schemas~~ **done** (`aacf920d39`).
4. ~~PPL gate dense + MoE~~ **done** (15.9555 / 14.3750).
5. TP=2 serving smoke beyond the V2 Muse arm; then the mamba-prefill-checkpoint
   (#62) and packed-layout (#60) re-checks from §4.
