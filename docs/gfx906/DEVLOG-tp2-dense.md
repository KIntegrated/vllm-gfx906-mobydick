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

### S9 — final-build restamps + live-context decode tax curve (2026-08-24, boot E)

**VERDICT:** SHIPPED (records) — README final numbers restamped on the
final build (rc2 image @ `7e4567053e`, post FA-gather-lifecycle + PERSIST
+ C2-V/W2/W4); **live O(Sk) decode tax quantified; MTP < greedy beyond
~20k ctx.**

**GATE:** serving, rc2 docker image, TP=2, maxlen 262144, util 0.82,
chunk 1024, capture [1,2,3,4]; cold prefill per rep (per-rep unique
prompt header defeats prefix-cache carryover); n=3 (2k/8k), n=2
(32k/64k). Boot E healthy: 0 resets through ~75 min, canary 56.2/56.7
t/s. (One isolated GPU0 wedge 13:00:53 at the 2nd launch — retry clean;
`degradation_details.md`.)

27B MTP k=2 (record line replaced; old 42.63 was short-ctx pre-final):

| live ctx | MTP t/s | greedy t/s | MTP/greedy |
|---|---|---|---|
| ~1.5k | **59.2** (58.6/59.6/58.8 interleaved) | 40.8 | 1.45× |
| ~6k | 44.9 | 38.1 | 1.18× |
| ~32k | 25.2 | 30.5 | 0.83× |
| ~64k | 16.6 | 24.1 | 0.69× |

- **Mechanism**: PERSIST removed the capture-baked pad32(max_model_len)
  width, but the live-bounded gather+quant and decode-attention work
  still scales O(Sk) and is latency-bound at these sizes (~12 ms/step at
  8k vs 2k; step ≈ 40 ms + ~1.7 µs/token). The PERSIST A/B only
  measured a 1091-token prompt — it could not see this. Crossover:
  MTP's 2.5 tok/step no longer beats greedy's 1× FA/draft overhead
  beyond ~20k live ctx. **Agentic ~60k-ctx work: run greedy (24.1 t/s)
  or accept 16.6 (MTP).** This also explains why boot D's 16.4 t/s
  agentic decode matched healthy physics (see degradation resolution) —
  only the short-ctx canary was the true degradation signal.
- Superseded cells: "28.83 @4k + 32.79 @131k TP=2" (mixed provenance;
  32.79 @131k physically impossible on this curve).
- 35B re-stamps (in-process, GPU0, final build): single 65.7/66.1 (8
  samples; record 67.39, band widened 65.3–67.0); MTP k=2 88.6 vs
  76.7 greedy (1.16×; record 89.9/76.2, 1.18×) — the pre-W4 re-measure
  debt is paid; N=8 192.9/194.0 (record 191.0, soak 189.9±0.4).
- Suites: 28/28 FA + 43/43 MoE GEMM (README line "15/15, 12/12" stale).
- Prefill (cold): ~470-525 t/s at 2k-32k, 357 @64k (attention growth).

> 2026-09-22: `--disable-custom-all-reduce` verified a **no-op** on this topology (PYNCCL is the only enabled AR backend with and without it; 3 arms within 0.25 %) — see `DEVLOG-spec-decode.md` (same session), which also measures the drafter-cudagraph knob at −7.9 % (graphs off) on the TP=2 MTP k=3 config.

---

### S10 — P2P-1 (custom all-reduce re-test): UNBLOCKED — UEFI/BAR state restored, platform gate now env-overridable (2026-09-28, boots 0/heal)

**VERDICT:** OPEN (measurement in flight) · **GATE:** interleaved same-boot
A-B-A ms/step at B=1 and B=4, fresh boot, mclk 1000.

**HYPOTHESIS (issue #3 / P2P-1):** now that PCIe P2P is live, engaging vLLM's
custom all-reduce (instead of PYNCCL) wins at B=4 / long context. Two prior
blockers — both now cleared — were: (a) the platform gate making the recipe
flag a no-op, (b) PCIe P2P not actually being up.

**Finding 1 — the flag cannot change anything on gfx906 (root-caused; source + live log).**

- `vllm/platforms/rocm.py:1240-1242` — `use_custom_allreduce()` returns True
  only for `gfx94`/`gfx95` (**MI300-only**).
- `vllm/config/parallel.py:1080-1081` — `if not
  current_platform.use_custom_allreduce(): self.disable_custom_all_reduce = True`
  → the platform gate **overrides the CLI flag**.
- `vllm/v1/worker/gpu_worker.py:1532` — `set_custom_all_reduce(not
  disable_custom_all_reduce)` → `_ENABLE_CUSTOM_ALL_REDUCE = False`.
- `cuda_communicator.py:143/169` — `CustomAllreduce` (`ca_comm`) and
  QuickReduce (`qr_comm`) are constructed **only** when that flag is true, so
  both are skipped and the backend list resolves to `['PYNCCL']`.
- Live corroboration (2026-09-28 06:04 server, launched **without** the flag):
  logged `disable_custom_all_reduce=True` and `Using ['PYNCCL'] all-reduce
  backends ... out of potential backends ['FLASHINFER_PCIE_IPC', 'FLASHINFER',
  'NCCL_SYMM_MEM', 'QUICK_REDUCE', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM',
  'PYNCCL']`.
- The one other P2P-shaped backend, FlashInfer PCIe IPC
  (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`, default 0), is **CUDA-only**
  (`flashinfer_pcie_ipc_all_reduce.py:69` — "requires the CUDA platform").

→ Deliverable (a) "remove the flag from the recipes" is a **no-op**; the
experiment must instead gate the platform check itself (planned env knob
`VLLM_GFX906_CUSTOM_AR`, default off).

**Finding 2 — the P2P precondition is not met (hard blocker).**

- `rocm-smi --showtopoaccess`: `0->1 False`, `1->0 False` (diagonal only).
- BARs are **256 MiB** at ~3 GiB: `pci 0000:04:00.0: BAR 0 [mem
  0xc0000000-0xcfffffff]`; root-port windows are 32-bit (`Memory behind
  bridge: dfc00000-dfdfffff`, prefetchable `c0000000-d01fffff`) → the firmware
  exposes **no high MMIO window**, i.e. Above-4G Decoding / the lowered MMIO
  high base from 2026-09-22 is not in effect.
- BAR history (`Detected VRAM RAM=32752M, BAR=`): the 32 GiB BAR exists only on
  **boot −12 (2026-09-22 15:32)** and **boot −11 (2026-09-22 18:47)**, is lost
  again on **boot −10 (2026-09-22 21:18)** and on **every** boot since
  (−9 … 0, 2026-09-24 → 09-28). Boot −25 carries a future-dated RTC
  (**Oct 18 10:09**) mid-sequence — the signature of a CMOS/UEFI reset — which
  matches the enablement failing to persist.
- Kernel side is intact: cmdline carries **no** `pci=nocrs`; the running
  `6.8.12-acso` (built 2026-09-22 16:15) is the build the p2pdma whitelist
  patch targets (`/local/git/linux-acs-override/build-debian.sh:41`).

→ Action: re-apply the UEFI settings (Above 4G Decoding = Enabled; MMIO high
base ~1 TiB and window ~1 TiB so both 32 GiB BARs sit below 2^44), then verify
`BAR=32768M` in the boot log **and** `showtopoaccess` True before measuring.

**Finding 3 — the P2P precondition is now met (2026-09-28 13:12 boot, verified).**

```
root bus resource [mem 0x10000000000-0x13fffffffff window]      # 1 TiB .. 5 TiB
pci 0000:04:00.0: BAR 0 [mem 0x10000000000 …] [size=32G]
pci 0000:07:00.0: BAR 0 [mem 0x11000000000 …] [size=32G]
amdgpu 0000:04:00.0: added peer-to-peer DMA memory 0x10000000000-0x107ffffffff
amdgpu 0000:07:00.0: added peer-to-peer DMA memory 0x11000000000-0x117ffffffff
Detected VRAM RAM=32752M, BAR=32768M          # was BAR=256M on every boot since 09-22 21:18
rocm-smi --showtopoaccess : 0->1 True, 1->0 True
hipDeviceCanAccessPeer = 1 both ways; enablePeerAccess clean
/local/tmp/x99wl/p2p_check : 4/64/256 MB both directions bad=0 -> P2P-CHECK: PASS
no "PCIe P2P access … is not supported by the chipset" lines
24003M of GTT memory ready per card           # 48 GB host; GTT = RAM/2 (no gttsize override)
```

Two UEFI changes are needed, not one: Above 4G Decoding **and** the MMIO high
base lowered to ~1 TiB. With only the first, the BARs land at ~56 TiB and the
44-bit DMA-mask gate in `amdgpu_device_is_peer_accessible()` still fails — both
32 GiB BARs must sit **below 2^44 (16 TiB)**. The 09-22 21:18 relapse (and the
future-dated RTC on boot −25) points at a weak CMOS battery; a fresh cell has
been fitted.

**Change 1 — the platform gate is now env-overridable (`rocm.py:1240`, default OFF).**

`VLLM_GFX906_CUSTOM_AR=1` → `use_custom_allreduce()` returns True on gfx906;
unset or `0` → False (production behaviour unchanged). Verified by direct call
on arch `gfx906:sramecc+:xnack+`: unset → False, `0` → False, `1` → True. The
downstream `parallel.py:1080` only forces `disable_custom_all_reduce=True` when
the platform says no, so this one knob flips the whole path
(`gpu_worker` → `_ENABLE_CUSTOM_ALL_REDUCE` → `cuda_communicator` constructs
`CustomAllreduce`/QuickReduce). Inits that could still refuse on gfx906: the
world-size / same-node gates pass at TP=2, and the explicit `_can_p2p` probe is
**skipped on ROCm** (`custom_all_reduce.py:249`) — so the peer IPC buffer open
is the real test, which is exactly what used to fault.

Note this is not foreclosed by S1: S1's stall was the **RCCL** P2P transport
(`NCCL_P2P_DISABLE=1` made it disappear), which is a different path from custom
AR's own IPC buffers.

**Change 2 — a latent upstream crash on the custom-AR path, found and fixed (`rocm.py:1013`).**

Reaching `CustomAllreduce.__init__` on gfx906 exposed a crash that had never
been reachable here: `current_platform.is_fully_connected()` →
`amdsmi_get_processor_handles()[i]` → `IndexError: list index out of range`.
On this host **amdsmi enumerates 0 processor handles** (verified directly in
and out of the worker's env — not a visibility-env artefact), so every index
raises and the engine dies (`EngineDeadError`). Every other amdsmi helper in
the class already guards `physical_device_id >= len(handles)`; this one did
not. Fixed by guarding and returning **False** (the conservative answer: the
result only gates the >2-GPU custom-AR / QuickReduce paths, where a wrong True
is the dangerous direction). Upstream-worthy as a stand-alone fix.

**Result — custom AR engages on gfx906 and wins at B=1 (+17.1% decode).**

| arm | env | AR backend | B=1 decode t/s | prefill agg t/s | ttft |
|---|---|---|---|---|---|
| A1 | `NCCL_P2P_DISABLE=1` | `['PYNCCL']` | **33.94 / 33.94** | 175.6 | 4.12 s |
| B | + `VLLM_GFX906_CUSTOM_AR=1` | `['CUSTOM', 'PYNCCL']` | **39.72 / 39.71** | 193.7 | 4.13 s |
| A2 | `NCCL_P2P_DISABLE=1` (repeat) | `['PYNCCL']` | **33.89 / 33.89** | 175.4 | 4.12 s |
| C | `VLLM_GFX906_CUSTOM_AR=1`, P2P live | `['CUSTOM', 'PYNCCL']` | **39.62 / 39.66** | 193.4 | 4.11 s |
| control (run 3) | P2P live, no knobs | `['PYNCCL']` | **33.92 / 33.92** | 174.4 / 175.5 | 4.2 / 4.1 s |

- **+17.1 % decode** (33.90 → 39.67 mean of the four baseline/treatment reps),
  **+10 % prefill aggregate** (175.4 → 193.5), TTFT unchanged. Arms differ in
  the all-reduce backend only. mclk 1000 MHz confirmed in every arm (20+ samples
  ≥1000 MHz each); warm-cache loads 190–260 s; peak GTT use during load 42 MiB of
  24 GiB (no aperture pressure on this workload).
- Arm C proves the knob does **not** need the RCCL workaround: custom AR runs
  with P2P live as well as with `NCCL_P2P_DISABLE=1`.

**Correction — the run-1 "baseline stalls with P2P live" observation was NOT the RCCL P2P path.**
Run 1 arm A1 (baseline, P2P live) sat at 0 % GPU util with `shm_broadcast`
warnings for >4 min; the run-3 control run the *same* config and was up in
200 s at 33.92 t/s. It was a one-off **first-real-load transient** on that boot
(the first load after the RAM upgrade + BAR change), not a regression from P2P.
Treat the first load after a hardware change as suspect, never as a datum.

**Blocking finding (unrelated to AR) — a mixed prefill+decode batch kills the engine.**
The B=4 gate cell is currently unmeasurable. With `--max-num-batched-tokens 4096`
and four 2048-token prompts, the scheduler admits the prefills in waves, so
steps contain prefills *and* decodes →
`AssertionError: GDN decode-first invariant violated: non-spec decodes not first`
(`vllm/v1/attention/backends/gdn_attn.py:385` — our own guard from `233e8f202b`)
→ `EngineDeadError`, HTTP 500 to all four requests. Reproduced in **all four
arms** (A1, B, A2, C) and via run 3's warmup; **no GPU event** in the journal —
pure software. The guard is behaving as designed (fail loudly rather than
silently mis-slice), so the defect is in what produces that batch order.
Filed as its own issue: it is a crash on ordinary concurrent traffic, not a
perf item.

**Next:** (a) the GDN mixed-batch crash (blocks the B=4 cell); (b) re-run the
B=4 cell once fixed, or measure decode-only B=4 with 4×512-token prompts so all
four prefills land in one step; (c) a quality/numerics gate for the AR swap
(bit-exactness or PPL) before the knob could ever default on.

**Refs:** issue #3 · `/local/tmp/4g-handover.md` (gates 1–3 + enablement
recipe) · `DEVLOG-spec-decode.md` (2026-09-22 no-op session) · `degradation.md`
#104/#105.
