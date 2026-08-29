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

**Status: onboarded + window-FA + M1 gather clip shipped, merged to
main 2026-08-28 (`feat/muse-glimmer`); LEGACY=0 (Q8 side-view read)
validated, LEGACY=1 remains the serving default; the read pattern
(gather vs direct-paged) is an *orthogonal* auto-gate — direct-paged +
Phase C clip fire on the B≥2 decode dispatch in both LEGACY modes
(README erratum 2026-08-27).**
Onboarding and all gate numbers: `DEVLOG-muse-glimmer.md`. Knobs:
`README.md` table. The three independent review files
(`muse_glimmer_opt2_code_rev_{qwen,claude,ds4}.md`) were deleted
2026-08-27 after their findings were folded into M1–M4 below; M3's
items carry the surviving qwen text inline.

### M0 — per-impl q_pad buffers: the first-prefill OOM root cause — DONE 2026-08-27/28 (fixed, probe-verified 1.285 GiB, bt4096 TP=2 re-validated)

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

### M2 — per-row (2D) prefill clip — DONE at q-tile granularity, in main since 2026-08-29 (was branch `feat/fa-m2-tile-clip`; log: DEVLOG-fa-attention.md M2 + 2026-08-29 review-fixes entries; 65/65 suite; kernel A/B 3.19×/2.81× windowed + 2.22×/1.96× causal-only at pp4096/full-context; e2e A/B +11.8 % wall / +14.8 % prefill @ pp16384 windowed, +0.73 % @ pp2048 full-attention-hybrid (GEMM-dominated, FA component 1.96–2.22×) — the causal cap is a general chunked-prefill win, not a window feature)

Prefill rows need only `[q_abs+1-W, q_abs]`; today the per-sequence
`k0_base` covers the whole q-tile, so early rows in a prefill chunk
still scan (and the gather materializes) out-of-window keys. A
conservative per-q-tile start (smallest row's window start) is
implementable without per-row loops. Gate: bit-identity + pp4096
prefill/TTFT A/B. Implemented as two bit-identical per-q-tile scan
bounds (window raise of `k0_base` + causal cap of `k_VKQ_max`, knob
`GFX906_FA_TILE_CLIP` default on) + the DIRECT_PAGED clip gate
extended to prefill; per-row granularity (within a 64-row tile) left
open — the residual is <= ncols1-1 rows of masked keys per tile, a
~1/32 effect of what M2 removes.

### M3 — kernel hygiene batch (one rebuild) — DONE on branch `feat/fa-m3-hygiene` 2026-08-28, review fixes 2026-08-29 (unmerged for review; branch now merged with main incl. M2; log: DEVLOG-fa-attention.md 2026-08-28 M3 entry + 2026-08-29 review-fixes section; 70/70 post-merge suite)

From the (now-deleted) qwen review — items carry their text inline —
bundle into one build/test cycle:
- **#8**: device-side `k0_base = max(0, kv_start[sequence])` clamp in
  `fattn-q8-paged.cuh` — a negative start would walk the k-loop into
  pages before token 0 (illegal access / wedge, not a wrong number).
- **#10**: overflow-free cutoff `q_abs_row - k_pos_abs >= window` at
  all four LOCKSTEP sites (hardening/clarity — the old form
  `k < q_abs_row - window + 1` cannot actually wrap for int32; the
  new form is wrapping-free by construction). Preserves the
  unaligned bit-identity tests; new window=INT_MAX bit-identity test.
  (DONE — see the header; the "overflows" premise was corrected in
  the devlog.)
- **#4b/c**: allocate only the meta buffer actually used. F2-verified
  2026-08-29 against the code: the kernel's only `dst_meta` write is
  guarded by `gridDim.y != 1` (gridDim.y == kv_split) in BOTH kernels,
  and the `[B, Sq, Hq, 2]` `o_meta` is a local tensor the caller never
  reads — so it is dead at `kv_split == 1` too, not just `kv_split > 1`
  (where the written/read meta is `o_meta_split`). Final form passes
  `nullptr` at `kv_split == 1` and allocates only `o_meta_split`.
  Note the `o_part`/`o_meta_split` per-call cliff at
  `_DIRECT_PAGED_MAX_SQ=16` (~35 MiB/layer) — shared arena only if a
  real workload hits Sq>2 direct-paged.
- Test hardening: amplified-V window-boundary case (probe-B trick,
  ~400× discriminative, 3 lines).
- **FA-Q8 P·V via `v_dot2_f32_f16` — REFUTED 2026-08-28 (ISA), not
  implemented.** The premise ("P·V currently accumulates in `half2`
  (`v_pk_mul_f16` + `v_pk_add_f16` = 2 instr per 2 MAC, fp16-acc)")
  is wrong against the built ISA: the compiler already emits
  `v_pk_fma_f16` for the P·V accumulate (1 instr per 2 MACs, full
  packed rate — dequant-instructions.md). `v_dot2_f32_f16` is also
  1 instr per 2 MACs (full rate at ILP≥2, latency-sensitive), so the
  rewrite buys ZERO instruction count / rate — only fp16→fp32
  accumulation, at the cost of restructuring the P·V row ownership
  for row-pair packing plus new test references. The item's own
  decision rule ("worth it only if P·V is measurably VOPC-issue-bound
  in the A/B") is not met. **Reframed: precision-only candidate
  (fp32 P·V accumulation); revisit only if a numerics gate demands
  it.** (The op itself is verified clean on this toolchain —
  `vdst = acc + a_lo·b_lo + a_hi·b_hi`, identical to the
  `__ockl_fdot2`/`amd_mixed_dot` production path in
  `dense_gemv_gfx906.cu` / `moe_q_gemm_gfx906.cu`; see
  `dequant-instructions.md`.) Benefits both LEGACY modes
  equally (does not re-open the M5 gate); worth it only if P·V is
  measurably VOPC-issue-bound in the A/B (the Q·K sdot4 side has no
  remaining instruction lever — dot4 is the only and the full-rate
  int dot on gfx906).

### M4 — long-context split-K accuracy point (qwen #4a)

Direct-paged split-K stores unscaled fp16 partials per split; the
partial magnitude grows with keys-per-slice × |V|. Tests cover
L ≤ 512 (plus the 4353/2048 clip case). One accuracy point at
L=16k–32k, split 8 vs split 1 vs fp32 torch ref (cheap, no server)
closes the claim that the default `clamp(16/B,2,8)` is safe at
long context.

### M5 — default read-path decision after a bake — **DONE 2026-08-28: keep LEGACY=1**

**Gate executed (DEVLOG-muse-glimmer rounds 6–9), decision: `GFX906_FA_LEGACY` stays `1`.** *(2026-08-28, round 10: M6 Part B closed the B=4 half of the re-open condition — LEGACY=0 + gather-routed B=4 @2k reached LEGACY=1 parity, 46.3 vs 46.7 t/s; the flip gate now hinges on the B=1 same-boot adjudication, still open.)*
Sequence: (1) round 6 — M1 clip ported to the fused Q8 gather, 57/57
unit bit-identity; (2) round 7 — e2e clip gates closed: the original
gap (a) ("LEGACY=1 + direct-paged + clip") proved VACUOUS (direct-
paged is reachable only under LEGACY=0), and the real gates —
GATHER_CLIP on/off at pp8192/tg256 on the default path — showed real
deltas: B=2 **+12.1%** (6.394 vs 5.706), B=4 **+22.1%** (6.081 vs
4.982); (3) round 8 — the G3 LEGACY=0 TP=2 bake caught + fixed a
latent capture-unsafe D2H sync in the direct-paged Sq>1 loops (first
production hit = LEGACY=0 + ngram spec + B≥2 under FULL capture);
(4) round 9 — **G3 executed on boot M (canary 38.9 t/s): LEGACY=0
LOSES to the LEGACY=1 control at every controlled point** — B=1
decode −2.5…−3.7 % (107.4 vs 111.5 @2k; ~96.5 vs ~99 @8k), B=4
@2k aggregate −27…−31 % (35.8/32.1 vs 46.7), prefill a wash (495
vs 496.9 @8k). Reading (corrected 2026-08-28): both arms ran the
*identical* FA-Q8 `v_dot4_i32_i8` kernel — the LEGACY flag selects
the KV-layout/gather strategy, not the FA dot — so the B=1 delta is
the gather/read path and the B=4 delta is direct-paged's strided
paged reads (memory access pattern, not an FA ALU gap); the
10%-leaner HBM read is invisible at these sizes. Per the flip rule
(a wash already keeps LEGACY=1; only a win justifies flipping),
**D1 is not executed**. LEGACY=0 remains an experimental
opt-in (zero-extra-KV-memory alias, COW-safe); the gate re-opens if
a future FA change closes the read-layout gap measured above (M6
territory — see below; NOT a Q8-dot compute gap, corrected 2026-08-28).
Boot L closed out with a 3rd wedge burst → boot M (this bake); boot
M's own 15:00Z weight-load wedge (isolated, self-recovered) made the
bake attempt 2.

### M6 — LEGACY=0 salvage: read-layout fix first, Q4-KV/dot8 as the ISA upside (M5 follow-up, re-opens the flip gate)

**Part B (fix candidate 2) DONE 2026-08-28 (DEVLOG-muse-glimmer
round 10): LEGACY=0 B≥2 now routes to the fused-Q8 gather by default
(`GFX906_FA_DIRECT_PAGED_Q8` default flipped 1 → 0; direct-paged is
opt-in =1).** Gate: M5-recipe TP=2 serving bake, B=4 @2k ngram
35.7 → **46.3 t/s** (+29.7 %, parity with the 46.7 LEGACY=1 control,
B=1/prefill unchanged, 60/60 suite). The in-process Sq=1 A/B was a
wash — the loss was specific to the spec-decode (Sq=6) serving
regime. Mechanism (review-softened 2026-08-28, F1): the strided
Q8-slice reads are the leading but UNCONFIRMED hypothesis — a pure
read-layout story cannot explain the Sq=1 wash (dilution would
predict ~−10…−18 % there), so the direct-paged Sq>1 machinery
(round-8 Q-pad/unpad fast paths, graph-capture interaction) is at
least a co-contributor; round-10 erratum in the devlog. Under the
production LEGACY=1 default the flip is a no-op.
M5's B=4 flip gate is now GREEN; the LEGACY flip itself still needs
the B=1 same-boot adjudication (107.2 boot M vs 111.5 boot L,
−3.9 % cross-boot; next two-card launch belongs on a fresh boot —
boot M at 3 wedge observations in ~2.5 h).

**Reframed 2026-08-28** after the ISA rate probe (`DEVLOG-fa-attention.md`
2026-08-28 entry; rates in `dequant-instructions.md`): the original
"close the Q8-dot compute gap" premise is wrong — `v_dot4_i32_i8` is
full-rate on gfx906 (25.9 T MAC/s, 4.44× fp32 FMA, 2× packed fp16;
AMD's 53 TOPS INT8 for MI50 is this instruction) and already in the
inner loop; the B=1 decode path is gather-HBM-bound (~2.7× at D=128,
NC2=8), so the old item (a) (per-block fp16-rescale batching) cannot
surface at B=1 and is deprioritized to hygiene. What the M5 deltas
actually measure — confirmed at code level:

- **(b′) aliased-Q8 read layout (the B=1 −2.5…−3.7 % and the B=4
  −27…−31 % share this root)**: the Q8 side view packs 4×34 B q8_0
  blocks into the first 136 B of every 256-B fp16 K row, so every
  consumer reads 136 B out of a 256-B stride (5 sectors fetched, ≤85 %
  efficiency; 34-B block strides break 16-B vector alignment; V2
  gather pays a uint2 tail every token). Fix candidates, in order:
  1. **Repack the side view into aligned planes** — contiguous
     16-B-aligned quants plane (128 B/row at D=128) + separate scale
     plane (8 B/row); write-path quantize and gather/direct-paged
     readers updated together. Converts the nominal 512→392 B/row
     (1.31×) into an actual gather win instead of today's net loss;
     grows with context (8k+ points gain the most). **Plan written
     2026-08-28: `plan_fa_part_A.md`** — re-aimed at the B=1 gap
     (per the round-10 F1 erratum the B=4 point is already fixed by
     Part B and its mechanism is unconfirmed); audited diff scope is
     intra-row only (row byte count `(D/32)×34` invariant → all
     strides/buffers/alias contract unchanged); gate = same-boot M5
     B=1 serving grid on a fresh boot.
  2. **B≥2: route LEGACY=0 through the fused-Q8 gather** (1:1 byte
     copy, half the fp16 gather's read) instead of direct-paged
     misaligned slices — recovers the −27…−31 % before the repack
     even lands. **DONE 2026-08-28** (round 10): gate PASS, default
     flipped — the remaining M6 items are the repack (1) and Q4-KV
     (c), which target the B=1 long-context gap, not B=4.
- **(c) Q4-KV via native `v_dot8_i32_i4` — the LEGACY=0-specific
  instruction-level upside — SHELVED 2026-08-28 (user decision):
  quality unproven — Q4 K *and* Q4 Q (or Q8→Q4 requant) quantization
  accuracy is unvalidated on this model family, and the 7-level
  q4_0 codebook (vs 127) roughly doubles the KQ quantization error
  with no measured PPL evidence that attention output absorbs it.
  Revisit only behind a dedicated accuracy gate (PPL probe bands on
  the 442-token set, Q4-KV vs Q8-KV arms) that must pass *before* any
  kernel work; the HBM-traffic arithmetic below is the motivation, not
  the justification.** (the P·V `v_dot2_f32_f16` rewrite in M3
  is instruction-level too but benefits BOTH LEGACY modes equally and
  does not re-open this gate; 2026-08-28 `dot_isa_probe.py`
  confirmed the i8 dot8 forms are assembler-rejected — dot8 = the i4
  variant only): measured 49.6 T MAC/s (8.52× fp32, 2× dot4) with
  packed-nibble operands on BOTH sides and zero unpack ALU in the dot
  loop. K q4_0-style = 72 B/row (vs 136 q8 / 256 fp16); with V→Q8
  (dot4, 136 B) or V→Q4 the row read drops to 208/144 B (2.5–3.6×
  leaner than LEGACY=1's 512) and the KQ/V ALU halves again. Costs:
  Q must be i4-packed too (per-forward Q8→Q4 requant or direct Q4
  quantize), a new write-path quantizer + layout, and the accuracy
  gate (PPL probe bands on the 442-token set; Q4 V is the risk item —
  llama.cpp ships q8/q4 KV caches as precedent). Large-context models
  (Qwen3.8-27B 256k, Muse-Glimmer 16k+) are the beneficiaries.

Old M6 text (pre-reframe, kept for the record): two inferred causes —
(a) per-block rescale tax around the (already-correct) dp4a dot, (b)
direct-paged strided reads — neither measured; plan file
`plan_fa_legacy0_impr_claude.md`. Non-blocking; re-opens the M5 flip
gate only on a green B=1 **and** B=4 serving A/B — a B=1-only win does
not justify a flip per the M5 rule.

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
