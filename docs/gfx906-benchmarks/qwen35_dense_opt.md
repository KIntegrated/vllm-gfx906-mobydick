# Dense Qwen3.5/3.6 on gfx906 — porting the MoE optimization stack (handover)
Copyright Kevin Read <me@kevin-read.com>

Status: **complete (takeover session 2026-08-17; DEVLOG dense section,
"W4A16 dense kernel" + "down_proj GEMV LANDED" subsections are the
current state — read those first).** Everything actionable from this
doc has been resolved: budget measured (eager kernel map +
three-anchor inference; rocprofv3 unusable for dense full-model runs),
NC2=2 FA fix + GDN flip (`1a895e8a01`), GemmaRMSNorm fused-kernel
(`19c1d41cf5`), down_proj GEMV K=17408 (2026-08-17, serving
**23.85 t/s** record, was 23.55), W4A16 dense kernel investigated and
**rejected** (exllama gptq faster on all dense shapes; see DEVLOG). The
two FA-backend bugs affecting dense models were found, fixed and
committed (`b4873459f8`). Remaining low-priority items: §6b copy-pile
call-site attribution. Branch: `gfx906/moe-opt`.

Companion doc: `DEVLOG-moe-opt.md` (the MoE optimization history this builds
on; read its "PROBE PITFALLS" section before instrumenting anything).

---

## 1. Mission

The MoE optimization phases (custom grouped W4A16 GEMM, FA decode
parallelism, dense GEMV, fill/copy cuts, cudagraph serving mode) were built
for `QuantTrio/Qwen3.5-35B-A3B-AWQ`. Question: **which of those carry over
to the *dense* Qwen3.5/3.6 models?** Constraint from the user: quantized
dense models already run at decent speed (they use the pre-existing exllama
`gptq_gemm` fast path), so the interest is in the *rest* of the stack,
particularly quantized 27B.

## 2. Primary test model: `/data/models/qwen/Qwen3.5-27B-AWQ` (NFS)

`Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`, text config:

| fact | value |
|---|---|
| layers | 64 = **48 GDN linear-attn + 16 full-attn** (every 4th) |
| hidden / intermediate | 5120 / 17408 |
| full-attn heads | Hq=24, Hkv=4, head_dim=256 → **GQA ratio 6** (MoE model was ratio 8!) |
| GDN | 16 k-heads × 128 k-dim, **48 v-heads × 128 v-dim** (mamba state ≈ 3× the MoE model's per sequence) |
| attn_output_gate | true |
| quant | AWQ int4, gs=128, zp=true |
| not converted (fp16/bf16) | `visual`, `linear_attn.in_proj_a/b`, `self_attn.q/k/v_proj`, `model.layers.0.*`, `mtp`, **lm_head** |
| lm_head | standalone bf16 [248320, 5120] = **2.37 GiB** → decode GEMV floor ≈ **3.2 ms/step** @ 798 GB/s |
| fp16 q/k/v (16 FA layers) | 80 MiB/layer → 1.25 GiB traffic/step → floor ≈ 1.7 ms/step |

Quantized modules (per safetensors index): `gate/up/down_proj`, `o_proj`,
GDN `in_proj_qkv`/`in_proj_z`/`out_proj`. Note this differs from the MoE
model where GDN in_proj was fp16.

**Local quantized-27B alternatives** (user prefers /local):
- `/local/cache/huggingface/models--cyankiwi--Qwen3.8-27B-AWQ-INT4` —
  same arch; **compressed-tensors** W4A16 *asymmetric* gs=32, zp int8.
  Keeps ALL GDN layers + lm_head fp16 (ignore list). ~19.6 GiB on disk.
- `/local/cache/huggingface/hub/models--Lorbus--Qwen3.6-27B-int4-AutoRound`
  — **auto-round** sym gs=128, packing `auto_round:auto_gptq`; same arch.
  18 GiB on disk, complete.
- Both *should* route to `ExllamaLinearKernel` on gfx906 (uint4 / uint4b8
  are in `SUPPORTED_QUANT_TYPES`; the oracle puts Exllama first on gfx906,
  see `vllm/model_executor/kernels/linear/__init__.py:choose_mp_linear_kernel`),
  but **neither has been booted yet** — handover task §8.

## 3. How to run dense models successfully on this hardware (THE recipe)

At session start, every dense run OOMed — including at gpu_util 0.95/0.92
and with/without fastsafetensors. Root causes found (in order of impact):

1. **Over-provisioned sequence count.** `_b.py` used `max_num_seqs=32`
   (fine for the MoE model). Dense 27B's per-sequence GDN/mamba state is
   ~3× larger (48 layers × 48 v-heads), so 32 sequences blow the card.
   **Use `BENCH_MAXSEQS=8`** for the single-request bench.
2. **FA backend prefill buffer (bug, fixed in `b4873459f8`).**
   `gfx906_fa.cpp` applied the decode-only `KVSPLIT=16` to prefill,
   allocating `o_part=[B,Sq,Hq,16,D]` fp32 — 588 MiB at chunk 1568. Now
   clamped to 1 when `seq_q>2`.
3. **Chunked-prefill chunk size.** `BENCH_BATCHED_TOKENS=512` keeps the
   prefill activation peak (incl. the exllama weight-dequant transient,
   §6c) away from the ceiling.
4. **Visual tower.** `language_model_only=True` (`BENCH_TEXT_ONLY=1` in
   `_b.py`) drops the 0.86 GiB fp16 ViT. MTP was *already* never loaded
   (both model classes use `skip_prefixes=["mtp."]`).
5. **fastsafetensors: dropped for dense** (per user). It holds ~2.8 GiB
   live at init — not worth it here. Use the default safetensors loader
   (~25 s from NFS with warm page cache).
6. **KV pin.** `kv_cache_memory_bytes=6 GiB` (`BENCH_KV_MEM`). The memory
   profiler's auto-sizing over-committed ~0.7 GiB at util ≥ 0.92
   ("Available KV" did not shrink when util dropped 0.95→0.92 — the
   hybrid KV+mamba sizing seems cap-driven; not fully understood, §8).

Working invocation (serving mode, custom FA, 2 samples ≈ 5 min):

```bash
cd /local/git/vllm-gfx906-mobydick
source ~/env-rocm-7.14-gfx906.sh
export HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1
export BENCH_EAGER=0 BENCH_PP=2048 BENCH_TG=256 BENCH_MAXLEN=3328
export BENCH_GPU_UTIL=0.92 BENCH_CG_MODE=FULL_DECODE_ONLY BENCH_SAMPLES=2
export BENCH_TEXT_ONLY=1 BENCH_MAXSEQS=8 BENCH_BATCHED_TOKENS=512
export BENCH_KV_MEM=6442450944
.venv/bin/python /tmp/bench/_b.py /data/models/qwen/Qwen3.5-27B-AWQ
```

(`_b.py` grew `BENCH_TEXT_ONLY`, `BENCH_MAXSEQS`, `BENCH_BATCHED_TOKENS`,
`BENCH_KV_MEM`, `BENCH_ARCH` envs this session; send a fuller warm-up
request before timing if new shapes trigger Triton JIT.)

**FA extension rebuild (no docker needed):** `/tmp/bench/build_fa_local.py`
compiles the 4 `csrc/gfx906_fa` TUs with host hipcc 7.14 and installs
`vllm/_gfx906_fa_C.cpython-312-x86_64-linux-gnu.so` in ~2 min. The tree's
`.so` files are root-owned docker build artifacts (gitignored); the local
rebuild overwrites only the FA one. All 15 tests in
`tests/kernels/attention/test_gfx906_fa.py` pass on the rebuilt module.

## 4. Numbers so far (serving mode, FULL_DECODE_ONLY, CUSTOM FA)

| run | tok/s (pp=2048/tg=256, incl. prefill) |
|---|---|
| dense6 (recipe above, 2 samples) | **23.20 / 23.17** |
| under rocprofv3 tracer (1 sample) | 22.81 |

Decode-only tok/s not yet extracted (need tg=1 vs tg=256 split or a
decode-window profile). No eager-mode number yet. No correctness check vs
Triton yet (§8 — the dense model ran with silently-wrong attention before
the NC2 fix, so the first order of business is validating the current
output).

## 5. Bugs found and FIXED this session (commit `b4873459f8`)

Both affect any model whose geometry differs from the MoE 35B's, i.e. all
dense Qwen3.5/3.6 — and both were invisible on the MoE model:

1. **NC2 GQA head-packing silently wrong** (`gfx906_fa_launcher.cu`). The
   NC2>1 path packs `ncols2` consecutive Q heads per tile sharing ONE KV
   head (`K/V base = head0 / gqa_ratio`), which requires
   `gqa_ratio % nc2 == 0`. Dense 27B has ratio 6; the default `NC2=8`
   passed the `heads_q % nc2` check and read wrong KV heads for ~half the
   Q heads (silent garbage, no crash). Only NC2={1,8} are template-
   instantiated, so the fix **fails closed to NC2=1** for unsupported
   ratios and errors loudly on un-instantiated env values (the env knob
   `GFX906_FA_NC2=2/4` previously silently ran as 8 — pre-existing lie).
   Confirmed live: capture logs `gqa_ratio=6 not divisible by nc2=8,
   falling back to nc2=1`. Decode parallelism is still fine via
   KVSPLIT=16 (24 tiles × 16 = 384 blocks).
2. **KVSPLIT applied to prefill** (`gfx906_fa.cpp`) — see §3.2. Clamped
   `kv_split=1` for `seq_q>2`.

An adversarial review by the MoE agent caught that my first fix attempt
(clamp nc2→2) was a no-op because dispatch only instantiates {1,8}; the
committed version is the fail-closed one.

## 6. What carries over from the MoE stack to dense

### 6a. Already carried over / already active

- **Custom Q8 FA backend** — registered and selected on gfx906 regardless
  of model; now *correct* for ratio-6 GQA (this session's fixes). Biggest
  lever per the MoE history (attention was ~9–10% of the MoE decode step,
  2.7× kernel win). **Expected to be proportionally important for dense**
  (16 FA layers, Sk grows to 3328) — unquantified, §8.
- **Cudagraph serving mode** (`FULL_DECODE_ONLY`, `max_num_seqs ≤ mamba
  blocks`, capture ≤ 8) — works identically on the dense hybrid model
  (capture logs clean, ~14 s).
- **Fused gather+quant / V1 gather default, q_pad skip** — model-agnostic,
  active.
- **P3-1 tiny-m LLMM1 dispatch** (`_llmm1_tiny_m`) — generic; dense's fp16
  q/k/v have rows divisible by 4, so no F.pad pile there.
- **GDN decode kernels** — same code paths; already faster than llama.cpp
  per the MoE devlog.

### 6b. Proposed improvements (dense-specific gaps), ranked

| # | item | est. upside | uncertainty |
|---|---|---|---|
| 1 | **Correctness probe first**: greedy/PPL of CUSTOM FA (NC2=1 fallback) vs `BENCH_ATTN_BACKEND=ROCM_ATTN` on dense 27B | n/a (gate) | low — methodology exists in DEVLOG §P3-3a |
| 2 | **Decode budget profile** (§8) — everything below is guesswork until this exists | n/a | low |
| 3 | **Extend `dense_gemv_gfx906` to K=5120** (`_llmm1_tiny_m` gate hardcodes K==2048 = the MoE hidden size). Targets: lm_head [248320,5120] (3.2 ms/step floor) and q/k/v (1.7 ms/step floor), currently LLMM1 | 0.2–0.5 ms/step (MoE precedent: −6..−23% vs LLMM1 on matched shapes) | medium — kernel is templated on KCHUNK {512,2048,4096}; K=5120 needs a new kchunk or K-split; shape rules must be re-derived (the N==256∨N≥2048 rule was K=2048-specific; N=1024 was pathological there) |
| 4 | **NC2=2 template instantiation** in the FA launcher for ratio-6 GQA (dispatch2 + config-table check for ncols=NC1×2 at HD=256) | unknown; halves redundant K/V reads vs NC2=1 at decode; KVSPLIT already supplies parallelism, so maybe small | medium — new kernel instantiation needs validation; MoE agent suggested this as the proper fix option (a) |
| 5 | **`GFX906_GDN_EMPTY_CORE_OUT=1` as default on gfx906** — 48 zero-fills/step on dense (was 30 on MoE, ~94 µs/step there) | ~0.1–0.15 ms/step | low — the safety argument is documented in DEVLOG P3-4 (kernel stores every cell unconditionally); only needs a dense-model PPL point |
| 6 | **Memory sizing fix / understanding** — hybrid KV+mamba auto-sizing over-commits (util-insensitive "Available KV"); `max_num_seqs` mamba-state scaling deserves a note or guard | robustness | medium — mechanism not understood (§8) |
| 7 | **Prefill dequant transient (6c)** — chunked dequant or persistent fp16 mirror for big dense weights | robustness at long context / big batches | high — touches the exllama kernel structure |
| 8 | Try the two local 27B quants (§2) — validates the Exllama oracle routing for compressed-tensors/auto-round and gives local-model runs | coverage | low effort, unknown perf delta (gs=32 asymmetric may hit different exllama paths) |

### 6c. Noted hazard, not a regression

The exllama `gptq_gemm` prefill path (`MAX_Q_GEMM_ROWS=32`) dequantizes the
full weight matrix to fp16 + hipBLAS for M>32: **340 MiB transient per
gate_up call** on dense 27B (exactly 34816×5120×2 — fingerprint of 4 of the
5 OOMs this session). Pre-existing and byte-identical between `gfx906/main`
and HEAD (`git diff gfx906/main..HEAD -- vllm/.../auto_awq.py
csrc/libtorch_stable/quantization/gptq/` is empty), so **not** a MoE-branch
regression; but it sets the prefill memory ceiling on 32 GB.

### 6d. Does NOT carry over

- The MoE grouped-GEMM kernel (`moe_q_gemm_gfx906.cu`), routing pipeline,
  shared-expert gate fix, MoE prefill BM/NPT tuning — dense has no routed
  experts and no shared experts.
- aiter stays out on gfx906 (arch-excluded; DEVLOG P3-2(a)).

## 7. Dense vs MoE memory map (why the OOMs)

| component | MoE 35B-A3B | dense 27B-AWQ |
|---|---|---|
| weights (no fst) | ~22.4 GiB | 18.9 GiB (text-only) |
| KV+state pool | 1.37 GiB (fits easily) | 6–7.3 GiB needed (16 FA layers × 4 kv heads + 3× mamba state) |
| mamba state/seq | 30 layers × 32 v-heads | 48 layers × 48 v-heads (**~3×**) |
| prefill dequant transient | small dense layers | 340 MiB/layer (§6c) |
| verdict at 32 GiB | comfortable at util 0.95, seqs=32 | tight: needs the §3 recipe |

## 8. Open handover tasks (in suggested order)

1. **Decode kernel-time budget (rocprofv3).** The session's profiling
   attempts failed:
   - `-f csv` → CSV writer CHECK-crashes on process exit (`ring_buffer
     mmap failed`), empty CSVs.
   - SQLite invocation `rocprofv3 --kernel-trace -d <dir> --
     .venv/bin/python -u /tmp/bench/_b.py <model>` → DB written and
     complete (`agg_db.py`-readable) **but captured only the parent
     process (0 kernel dispatches)**; the spawned EngineCore child that
     runs the model was not instrumented on this run. The MoE agent's
     earlier `trace_fa` DB (275 MB, 1.08 M dispatches,
     `/tmp/bench/trace_fa/out/mi50-01/719850_results.db`) shows the
     method CAN work on this box — diff its launch env against ours.
     Candidate levers: `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-proc; was
     not combined with SQLite output this session), rocprofv3
     `--sys-trace`/child-attach flags, or the devlog's fallback: in-proc
     torch.profiler (`PROBE PITFALLS` §1–6 apply).
   - Aggregate with `/tmp/bench/agg_db.py <results.db> [window_ms]
     [nsteps]` (CSV: `agg_kt.py`). Under tracer, per-dispatch times
     inflate ~10–15% — shares are the reliable signal; grid columns are
     untrustworthy.
2. **Correctness probe** CUSTOM FA vs Triton (`BENCH_ATTN_BACKEND=
   ROCM_ATTN`): 128 greedy tokens + 12-prompt PPL (DEVLOG P3-3a method).
3. **Decode-only tok/s**: tg=1 vs tg=256 split (or profile window) for
   the §4 table.
4. **Eager-mode numbers** (BENCH_EAGER=1) for comparison with the devlog
   eager tables — should work now that the kv_split prefill bug is fixed;
   was never reached this session.
5. Items §6b #3–#8 (each after the budget in task 1 exists).
6. Optional: check out `gfx906/main` in a worktree and boot dense 27B
   there as the ultimate "no regression" proof (predicted outcome: same
   OOM class at seqs=32 since the gptq path is byte-identical; worktree
   needs copied/rebuilt `.so`s — DEVLOG "FULL MODEL RESULTS" notes how).

## 9. Files touched / created this session

- `csrc/gfx906_fa/gfx906_fa.cpp`, `csrc/gfx906_fa/gfx906_fa_launcher.cu`
  — the two fixes (committed `b4873459f8`); FA `.so` rebuilt via
  `/tmp/bench/build_fa_local.py`.
- `/tmp/bench/_b.py` — bench env knobs listed in §3.
- `/tmp/bench/run_dense{1..6}.sh`, `run_dense_prof{,2,3}.sh` — attempts
  (dense6 = the working recipe; prof3 = the parent-only DB).
- Logs: `/tmp/bench/dense{1..6}.log`, `dense_prof*.log`; DB:
  `/tmp/bench/dense_kt3/mi50-01/744838_results.db` (0 dispatches).

## 10. Context for the receiving agent

- Read `DEVLOG-moe-opt.md` first (esp. PROBE PITFALLS, P3-2b, P3-3a,
  "Local-venv bench environment").
- The MoE agent is active on the same tree/box — coordinate GPU use; one
  vLLM at a time on `HIP_VISIBLE_DEVICES=0`.
- Machine facts: MI50 32 GB (60 CUs), HBM ~798 GB/s measured, dot2 pipe
  ~20 TF practical; host ROCm is `/opt/rocm-7.14` (source
  `~/env-rocm-7.14-gfx906.sh`).
- A second MI50 is coming (64 GB total) — memory-pressure items (§6b #6,7)
  become less urgent, profiling stays.
