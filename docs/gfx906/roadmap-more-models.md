# More model families on gfx906 — open roadmap

Copyright Kevin Read <me@kevin-read.com>

Completed onboarding and portability findings are recorded in
[`CHANGELOG.md`](CHANGELOG.md). This file contains only model work that is
still open or blocked. The general rule is that gfx906 dispatch is selected by
weight format and shape, not by model family: verify every shape before adding
a model-specific gate.

## Onboarding procedure

For a new model:

1. Confirm that the model loads and identify its attention/linear-attention
   kernels.
2. Run a shape spy and build a per-step kernel table; do not infer transfer
   from Qwen3.5 numbers.
3. Microbenchmark each candidate GEMV/GEMM shape before changing dispatch.
4. Use a greedy-hash gate for bit-equal changes, or a PPL/coherence gate when
   accumulation order changes.
5. Run graph and eager serving A/Bs, with the default chosen from the A/B.

The portable design notes and hardware constraints are in
`latency-hiding.md`, `lds-layout.md`, `dequant-instructions.md`, and
`README.md`. The active MoE kernel is useful to AWQ W4A16 checkpoints with
compatible group size, layout, and dimensions; BF16, FP8, FP4, MLA, and DSA
paths need separate validation.

## Ling-3.0-tiny (`BailingMoeV3ForCausalLM`)

**Status: open; load and baseline first.** The on-disk checkpoint is an
approximately 7.5B BF16 model that should fit one MI50, but it is not yet
measured on gfx906. It has 24 layers, hidden size 1536, E=128/topk=8 MoE
layers, sigmoid plus `noaux_tc` routing with expert bias, KDA linear attention,
and MLA-style full attention. These properties do not match the Qwen3.5 GDN,
standard GQA, or W4A16 paths.

### L1 — get it running

Load the BF16 checkpoint and verify the `BailingMoeV3` model, KDA linear
attention, and MLA decode path on gfx906. Check for CDNA-only intrinsics or
aiter assumptions before porting anything. Record a greedy probe and PPL
baseline.

### L2 — establish a profile

Collect shape and kernel profiles for the full decode step, including layer-0
dense work, KDA, MLA, routing, and shared expert. Produce a measured budget
before selecting an optimization.

### L3 — BF16 MoE expert GEMM

If the model runs and profiling justifies it, benchmark a new W16A16 grouped
skinny-GEMM family for E=128, hidden=1536, and expert intermediate size 512.
The candidate dimensions are K=1536 and K=512. The existing lane-column,
wave-per-K-slice, single-wave-epilogue design is a starting point, not proof
that the Qwen3.5 W4A16 kernel transfers.

### L4 — routing

The sigmoid/`noaux_tc`/bias configuration is handled by the generic routing
path. Consider an M=1 specialization only if the measured profile shows a
large routing gap; the Qwen3.5 E=256 top-k result is not a sufficient reason.

### L5 — attention

Treat KDA recurrent decode and MLA paged decode as separate workstreams. The
Qwen3.5 GDN and custom FA implementations provide methodology only; they do
not establish correctness or performance for these kernels.

**Stop rule:** if L1 finds a hard gfx906 blocker in MLA or KDA, park Ling and
do not build an expert kernel for an unservable model.

## DeepSeek-V4-Flash

**Status: blocked; no optimization work is open on the current machine.** The
model has 43 layers plus MTP, E=256/topk=6 MoE with FP4 experts, FP8 dense
weights, DSA sparse-indexer attention, compressed KV, and a hidden size of
4096. The expert memory estimate is approximately 140 GB at FP4 (and more at
FP8), beyond the 64 GB available across the two cards. TP=2 is not a reliable
sharding path on this machine and DP would replicate the model.

Revisit only if a smaller variant, a substantially smaller checkpoint, or a
working multi-card sharding path becomes available. At that point, validate
format conversion, the K=4096 W4A16 extension, DSA/MLA attention, and
sqrtsoftplus/topk-6 routing independently.

## Generic AWQ MoE queue

The next compatible AWQ W4A16 MoE checkpoint can benefit from the existing
expert kernel without a new kernel, but it remains an onboarding task rather
than an automatic support claim. Qwen3-30B-A3B-AWQ is a candidate: E=128,
topk=8, hidden=2048, and expert dimensions that may fit the M=1 tile. Its
routing (E=128), dense attention, and shared-expert dimensions still require
shape-specific measurement. Follow the procedure above and the open items in
`moe-decode-roadmap.md`.

## Muse-Glimmer-30B (`MuseGlimmerForCausalLM`) — post-onboarding follow-ups

**Status: onboarded + window-FA shipped (2026-08-27, `feat/muse-glimmer`,
not yet in main); LEGACY=0 (Q8 side-view read) validated, LEGACY=1
remains the serving default; the read pattern (gather vs direct-paged)
is an *orthogonal* auto-gate — direct-paged + Phase C clip fire on the
B≥2 decode dispatch in both LEGACY modes (README erratum 2026-08-27).**
Onboarding and all gate numbers: `DEVLOG-muse-glimmer.md`. Knobs:
`README.md` table. The three independent review files
(`muse_glimmer_opt2_code_rev_{qwen,claude,ds4}.md`) were deleted
2026-08-27 after their findings were folded into M1–M4 below; M3's
items carry the surviving qwen text inline.

### M0 — per-impl q_pad buffers: the first-prefill OOM root cause — FIXED 2026-08-27 (probe verification + bt4096 serving re-validation PENDING POST-REBOOT)

The boot J/K first-prefill OOM (>10.6 GiB/GPU transient; "scales with
the chunk"; bt2048 workaround) was NOT inductor: the 3-arm
attribution probe (DEVLOG-muse-glimmer round 4) put 2.75 GiB of the
3.785 GiB TP=1 transient on the custom FA path, and per-layer
`memory_allocated()` hooks showed the exact shape: **the q_pad buffer
was per-impl, and v1 creates one impl per attention layer** — 52
impls × 256 MiB `[num_seqs, Hq, Sq_pad, D]` fp32 (metadata pads
num_seqs to max_num_seqs=4) grown on each impl's first prefill call
and kept alive by the capture-latch retire policy = 13.3 GiB (TP=2:
16 heads/GPU → 6.7 GiB/GPU). Buffer ∝ Sq_pad ∝ chunk = the
"linear-in-chunk" signature. The gather buffers had the identical
bug fixed earlier (ClassVar pattern); q_pad was missed. **Fix
shipped** (Python-only, no rebuild): `_q_pad_buf` /
`_q_pad_decode_buf` / `_q_pad_retired` / `_q_pad_captured` →
ClassVar, `_ensure_forward_buffers` → @classmethod(num_heads,
head_size, …); the q_pad lifecycle test rewritten with the
class-state snapshot/restore pattern. Unit gate: 51/51.
**Probe verification PASS (boot L, 2026-08-28, post-reboot)**:
custom arm survives the 0.5 GiB-KV-cap 4097-token prefill,
transient 1.285 GiB (vs 3.785 pre-fix), 4.89 GiB free after (vs
0.00). **Serving re-validation PASS (boot L, 2026-08-28):** bt4096
TP=2 launch (only change vs the boot K recipe) — first real 8192
request cleared (cold 452 t/s), decode ~99 @8k/B=1 / 111.5 @2k /
46.7 B=4 aggregate, 8.7 GiB headroom/GPU; **bt2048 workaround
droppable, bt4096 is the new default** (README TP=2 row updated);
6 GiB cap kept (plenty of margin). This item subsumes the old
"reduce the custom FA prefill transient" lever — DONE.

### M1 — window clip on the gather path (B=1 decode) — DONE 2026-08-27/28 (unit 51/51, e2e +8.1%)

The Phase C clip fires only on the direct-paged (B≥2) dispatch; B=1
decode (the gather path — the model's B=1 hot path) and all prefill
scan the full KV. The gather path materializes `[B, Hkv, L, bpr]` K/V
in HBM before the FA kernel, so this is a gather-side change (per-row
gather start), not just a kernel loop start.

**Implementation (Option A, absolute-position layout):** the persistent
gather writes only rows `[kv_start, seq_len)` at absolute positions
(buffer index == absolute token position; a 128-row margin
`GATHER_CLIP_MARGIN` ≥ max nbatch_fa=128 covers the kernel's
nbatch_fa tile-boundary floor) and the FA kernel (fattn-q8.cuh)
starts its k-loop at the floored `kv_start` — no mask changes, rows
`[0, floor)` never written or read. `kv_start = max(0, q_abs + 1 -
window)` per seq (conservative: the first query row's window start;
later rows' windows are subsets — covers Sq=1 decode, ngram Sq=6,
prefill alike). Kill switch `GFX906_FA_GATHER_CLIP` (default 1).
Files: fattn-q8.cuh (kv_start param + floor block LOCKSTEP with the
paged Phase C), gfx906_fa_gather.cu (persistent kernel + launcher),
gfx906_fa_launcher.cu / gfx906_fa.cpp (signatures/bindings),
gfx906_fa_paged.py (kv_start math + dispatch). M1 v1 = persistent
sub-path only (B≤16, all Sk = all current-server traffic); the other
gather sub-paths pass kv_start=None (full gather, still correct).

**Gates: PASS.** Unit: 5 new bit-identity tests (clip ON vs OFF via
`_GATHER_CLIP` monkeypatch, direct dispatch forced off so B=2 really
uses gather): Sq=1/6 × B=1/2 at L=4353/W=2048 (unaligned start
2305/2300) + short-ctx inert case; full suite 51/51; rows [2305, L)
bit-identical, [0, 2305) skipped. E2E (boot L, record recipe,
pp8192/B=1/tg256): **6.042 vs 5.587 t/s = +8.1%**
(`GFX906_FA_GATHER_CLIP` 1 vs 0) — matches the micro-bench's −48% FA
time × ~17% FA share of the B=1 step at 8k. See
DEVLOG-muse-glimmer.md round 5.

### M2 — per-row (2D) prefill clip

Prefill rows need only `[q_abs+1-W, q_abs]`; today the per-sequence
`k0_base` covers the whole q-tile, so early rows in a prefill chunk
still scan (and the gather materializes) out-of-window keys. A
conservative per-q-tile start (smallest row's window start) is
implementable without per-row loops. Gate: bit-identity + pp4096
prefill/TTFT A/B.

### M3 — kernel hygiene batch (one rebuild)

From the (now-deleted) qwen review — items carry their text inline —
bundle into one build/test cycle:
- **#8**: device-side `k0_base = max(0, kv_start[sequence])` clamp in
  `fattn-q8-paged.cuh` — a negative start would walk the k-loop into
  pages before token 0 (illegal access / wedge, not a wrong number).
- **#10**: overflow-free cutoff `q_abs_row - k_pos_abs >= window` at
  all four LOCKSTEP sites (the current `k < q_abs_row - window + 1`
  overflows for absurd windows → UB + silently disabled mask). Must
  preserve the unaligned bit-identity tests.
- **#4b/c**: allocate only the meta buffer actually used (`o_meta` is
  dead when `kv_split > 1`); note the `o_part`/`o_meta_split` per-call
  cliff at `_DIRECT_PAGED_MAX_SQ=16` (~35 MiB/layer) — shared arena
  only if a real workload hits Sq>2 direct-paged.
- Test hardening: amplified-V window-boundary case (probe-B trick,
  ~400× discriminative, 3 lines).

### M4 — long-context split-K accuracy point (qwen #4a)

Direct-paged split-K stores unscaled fp16 partials per split; the
partial magnitude grows with keys-per-slice × |V|. Tests cover
L ≤ 512 (plus the 4353/2048 clip case). One accuracy point at
L=16k–32k, split 8 vs split 1 vs fp32 torch ref (cheap, no server)
closes the claim that the default `clamp(16/B,2,8)` is safe at
long context.

### M5 — default read-path decision after a bake

LEGACY=0 (Q8 pre-quantized read) is validated (46/46 suite,
default-config + prefix-cache smokes, clip +3.6% / KVSPLIT +1.8%
gates) but stays experimental: B=1 still runs gather (no clip, inline
quantize — LEGACY=0 changes nothing at B=1 until M1 lands), the
LEGACY=1 TP=2 serving records (boot K, 2026-08-27: 114.6/79.1/57.0
@2k/8k/16k) postdate it, and two e2e gaps are open: (a) the LEGACY=1 +
direct-paged + clip combination (what auto-gating selects at B≥2 under
default LEGACY) has never been separately e2e-gated — the clip A/B ran
LEGACY=0; (b) the boot K B=4 grid point ran at ctx ≤2k where the clip
is a no-op — a B=4 long-context (8k+) clip on/off e2e is missing. Flip
the default only after a serving bake on the target workload. Gate: B=1
+ B=4 A/B (8k ctx) with the degradation canary green, plus (a)+(b).

### Housekeeping — drop the legacy `~/env-rocm-7.14-gfx906.sh` sourcing

The machine has a single ROCm toolchain now (/opt/rocm is the default),
so `source ~/env-rocm-7.14-gfx906.sh` (PATH/LD_LIBRARY_PATH to
/opt/rocm) is presumably unnecessary — remove it from the ACTIVE
recipes: `/local/git/AGENTS.md` (single-card bench recipe),
`docs/gfx906/running.md` (×2), `docs/gfx906/README.md`,
`docs/gfx906/degradation_details.md`. Dev logs keep their lines
(historical record). Gate: run the single-card bench recipe once
verbatim WITHOUT the source and confirm the harness works (boot K,
2026-08-27).

### DONE 2026-08-28 (boot L) — personal skill: MI50 vLLM memory attribution

`/home/kread/.agents/skills/gfx906-mem-attribution/SKILL.md` written
after the hunt's verdict landed (round 4): 3-arm probe + per-layer
hooks + bisection recipe, the in-process fresh-compile env set, the
confirmed dead ends (snapshot/kineto frames), and interpretation
notes. The probe is persisted in-repo at
`docs/gfx906/_probe_mem_attribution_gfx906.py` (the original
`/local/tmp/muse/probe_oom_attribution.py` is reboot-volatile). The
original plan the skill superseded, for reference:

- the probe design: arms (custom / rocm-attn / eager) × TP 1/2,
  explicit small `kv_cache_memory_bytes`, PP sized so the first
  prefill chunk is the OOM site, in-process `max_memory_allocated`
  peak-transient measurement (TP=1 only; TP=2 workers are separate
  processes → survive/OOM + last-straw alloc instead), raw
  memory-snapshot dump for the attribution.
- the env traps found: `VLLM_USE_AOT_COMPILE=0` (AOT mode is ON by
  default on torch ≥2.10 and its out-of-process workers fail with
  “Could not find an active GPU backend” on this box), the
  inductor async-compile worker HSA failures (FORK pool default;
  SPAWN children also failed to init HSA on a wedged-adjacent boot —
  parent HSA works, children don’t), and how to instrument
  (`generate_and_run_autotune_block` patch + `os.fork` trace +
  worker initializer patch, all picklable/top-level for spawn).
- the interpretation matrix (which arm surviving/OOMing implicates
  what: our FA vs inductor/model shapes vs KV sizing) and the
  recording protocol (HYPOTHESIS/GATE/VERDICT in the dev log,
  degradation table if a wedge is involved).
- gate for the skill itself: the next time an OOM needs attributing,
  a fresh session following the skill should reach the probe-launch
  stage without re-deriving the env traps.
