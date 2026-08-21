# TP=2 dense 27B serving on 2× MI50 — platform fixed (official amdgpu driver), decode parity at mtp2, context capacity 2×

Branch: `gfx906/tp2-dense-serving` (off `gfx906/main` @ `b1f164a46c`) · 2026-08-20/21
Model: `cyankiwi/Qwen3.8-27B-AWQ-INT4` (snapshot `63768c10`, `/local/cache/huggingface/hub/…` — the AGENTS.md `/data/models` path is stale)
Platform: 2× MI50 32GB (gfx906), 2 PCI-switch hops apart through the CPU
root complex (00:03.1→0a→0b, 00:03.2→0d→0e), same IOMMU group, `iommu=pt`.
Harnesses + full logs: `/local/tmp/tp2-debug/` (offline repro
`tp2_offline.py`, streaming bench `tp2_serve_bench2.py`, gz logs).

### S1 — bring-up: 27 s/step decode collapse, isolated to GPU-side RCCL P2P/IPC (2026-08-20/21)

**VERDICT:** SUPERSEDED by S4 (was: OPEN) · **GATE:** offline in-process
repro (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), 128-tok greedy; VLLM_TP2_DEBUG
instrumentation (commit `49c935332d`, later reverted).

Boots were clean at every config (exllama gptq W4 path, GFX906_FA CUSTOM
backend with NC2 auto-downgrade for gqa_ratio 6 per `1a895e8a01`, GDN
triton paths, in-tree qwen_triton_warmup) — 262144 max_model_len, 463k
KV pool. First real request: prefill fine, decode ~27 s/step (0.04 t/s),
rocm-smi alternating 99%/0% (serialized ranks), init ~505 s.

Isolation chain (each with evidence, logs in tp2-debug/):
- Not messaging: worker ENQ-RESP→engine DEQ-RESP ~1 ms; scheduler busy
  loop `input=0.000`. The 27 s is real GPU time surfacing at
  `async_copy_ready_event.synchronize()` (async scheduling on).
- Not our kernels: reproduces with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`;
  FA ratio-6/nc2=2 microbench 23-247 µs across Sk.
- Not config: eager vs graphs, custom AR on/off, fork/spawn — identical.
- Not RCCL-in-isolation: `all_reduce_perf` clean (P2P/direct, 17 µs /
  8.2 GB/s); torch.distributed AR loops (sync/pipelined/side-stream/
  copy-stream) 0.35-9 ms/step for 64 ARs. In-server transport was
  P2P/IPC (NCCL_DEBUG_SUBSYS=INIT) — same label as the fast probes.
- `NCCL_PROTO=Simple`: no effect. `NCCL_P2P_DISABLE=1`: stall gone
  (init 503→137 s, 6.9 t/s). `NCCL_P2P_LEVEL=PXB` (→SHM/direct): OK,
  6.6-12.5 t/s. PHB: stalls. Raw P2P primitives (peer copy 9.6-14.2 GB/s,
  cross-GPU flag poll 0 ms) healthy — an early "flag poll hangs" repro
  was a harness bug (setter on legacy default stream serializes streams).
- **HYPOTHESIS (S1)**: P2P/IPC under real serving load stalls — confirmed
  in class by S3/S4; mechanism = host driver, not flag-sync starvation.

Wedge hazard (recurring): SIGKILLing stalled runs leaves the driver
mid-P2P-op → next init wedges a GPU (hipErrorLaunchFailure / amdgpu
reset storm; BACO recovers). Always SIGTERM + wait (now in AGENTS.md).

### S3 — ACS exonerated; cross-stack GPU-hang ⇒ driver-level (2026-08-21 pm)

**VERDICT:** SUPERSEDED by S4 (diagnosis chain correct; PXB workaround
obsolete after driver fix) · **GATE:** ACS-kernel rerun + docker A/B.

- Custom kernel `6.8.12-acso` (pcie_acs_override): stall persists with
  P2P/IPC selected ⇒ ACS NOT the mechanism (BIOS ACS toggles and setpci
  on 0a/0d also ineffective — internal GPP bridges, see tp2-claude.md).
- Docker `mixa3607/vllm-gfx906:0.20.1-rocm-7.2.1-aiinfos`: TP=2
  **hard-hangs the GPU during weight load** (amdgpu "GPU Hang"); TP=1
  same image fine ⇒ version-independent ⇒ host-driver-level.

### S4 — official amdgpu driver fixes P2P/IPC; eager-vs-graph lesson (2026-08-21)

**VERDICT:** SHIPPED (platform fix: official AMD DKMS amdgpu 6.19.14, on
6.8.12-acso) · **GATE:** offline repro, default env, P2P/IPC.

Default env now runs clean: init 135-145 s (vs 505), both GPUs 100%
concurrent, zero stalls. Root cause: stock Ubuntu amdgpu mishandles
P2P/IPC on this dual-root-port topology (soft 27 s stall w/ RCCL 2.30.4;
hard hang w/ older RCCL). Residual flake: ~1/3 inits wedge GPU1 mid-load,
BACO recovers, retry succeeds (watch; SIGTERM teardown reduces it).

**Eager TP=2 ≈ 7 t/s is an artifact** (per-op launch overhead × ~96 AR/
token); graphs are mandatory. Early "comm-bound ceiling ~7-12 t/s" claims
were this artifact — clean graph-mode decode is 39.7-40.7 t/s.

### S5/S6 — serving matrix, MTP depth, ctx-length tax, chunk A/B (2026-08-21)

**VERDICT:** SUPERSEDED in part by S7 (comparison baselines) · **GATE:**
streaming bench (`tp2_serve_bench2.py`), 3 reps, graph mode, 131k ctx,
batch=1 greedy. Server recipe: `-tp 2 --gpu-memory-utilization 0.93
--max-num-seqs 4 --max-model-len 131072 --compilation-config
'{"cudagraph_capture_sizes":[1,2,3,4]}'`.

Measured (tg decode t/s): baseline ~31 (pp2k/tg128) … 34-39; mtp2 39.7 /
38.2 / 34.0 / 34.4; mtp3 38.6 / — / — / 37.8 (cells pp2k/tg128, pp2k/tg256,
pp8k/tg128, pp8k/tg256). MTP acceptance at TP=2: mtp2 mean length 2.49,
~74%; mtp3 adds 3rd-position acceptance 0.511, mean 2.97 — **mtp3 +10%
on the pp8k/tg256 cell (34.4→37.8)**, -3% on pp2k/tg128. Choose mtp2 for
short prompts, mtp3 for long ctx.

Also learned: debug logging costs ~25% t/s (29.8 instrumented vs 40.7
clean — never bench with VLLM_TP2_DEBUG on); a leftover profiler in the
bench script skews numbers; trimmed capture `[1,2,3,4]` captures in 3 s
vs 2+ min and frees VRAM (now default, AGENTS.md); first-gen warmup
26.6 vs steady 40.7 offline.

**max_model_len 262144 costs ~25% decode** (29.9 vs 39.9 t/s, matched
prompts, clean restarts, acceptance unchanged) — mechanism in S8.
**max-num-batched-tokens 8192: DEAD-END** — slower everywhere (pp 423 vs
483, tg 35.9 vs 38.6) and OOM on 32k prefill (inductor 279 MB chunk
buffer, free:0 at util 0.93). Prefill cold ~480 tok/s at default 2048
chunks; higher 8k-cell numbers are prefix-cache warm hits.

### S7 — CORRECTION: S5/S6 headline comparisons used the wrong TP=1 baseline

**VERDICT:** SHIPPED (correction) · **GATE:** n/a (bookkeeping).

S5/S6 compared TP=2 mtp2/mtp3 against the TP=1 **baseline** (25.14-25.60)
instead of the TP=1 **mtp2** record (39.74, DEVLOG-spec-decode.md ~line
756). Honest table:

| arm | TP=1 | TP=2 | TP=2 gain |
|---|---|---|---|
| baseline | 25.14-25.60 | ~31-39 (server-path streaming; needs clean same-harness A/B) | ~1.2-1.5×? |
| mtp2 | 39.74 | 39.7 | **1.00× (parity)** |
| mtp3 | not measured | 37.8 (long-ctx) | n/a |

TP=2's delivered value: context capacity (445-480k-token KV pool,
131072 std with 3.4× concurrency, 262144 bootable with 1.8×). The ≥1.5×
decode-speed session target was NOT met at matched MTP config — the AR
tax eats the spec-decode headroom.

### S8 — ctx-length decode tax root-caused: capture bakes pad32(max_model_len) (2026-08-21 late)

**VERDICT:** SHIPPED (diagnosis) / OPEN (fix lever) · **GATE:** eager A/B
at matched real context — `tp_decode_investigation.md` experiment #4.

Eager 131k vs 256k at identical ~1.5k prompts: no gap (19.5 vs 19.9 t/s);
graph mode shows the full -25% (39.9 vs 29.9). Mechanism, numbers, and
the capture-time-Sk-bound fix lever: `tp_decode_investigation.md`
RESOLUTION (cross-linked, also roadmap N4). Implication: even 131k
configs overpay — replays attend max_model_len-wide for short contexts;
fixing this could speed all decode (TP=1 included).
