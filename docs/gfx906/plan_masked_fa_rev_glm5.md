# Adversarial review: `plan_masked_fa.md`

> Reviewer: GLM-5 (agent). Scope: attack the plan's factual claims, cost
> model, option analysis, and step ordering against the actual code and the
> installed ROCm/PyTorch stack. Sources checked line-by-line:
> `csrc/gfx906_fa/gfx906_fa_gather.hip` (all three kernels + both launchers),
> `csrc/gfx906_fa/gfx906_fa.cpp` (dispatch, `quantize_q8_0`,
> `gather_paged_kv_*`), `csrc/gfx906_fa/gfx906_fa_quant.hip`,
> `csrc/gfx906_fa/gfx906_fa_launcher.hip`, `csrc/gfx906_fa/kernel/fattn-q8.cuh`
> (K-loop), `vllm/gfx906_fa/gfx906_fa_paged.py`, `vllm/gfx906_fa/
> gfx906_fa_backend.py`, `vllm/v1/worker/gpu_model_runner.py` (capture +
> seq_lens refresh), `tests/kernels/attention/test_gfx906_fa.py`,
> `benchmarks/kernels/gfx906/bench_gfx906_fa_gather.py`,
> `/opt/rocm/include/hip/hip_runtime_api.h` (graph node types),
> `DEVLOG-fa-attention.md` / `DEVLOG-tp2-dense.md` S8,
> `tp_decode_investigation.md`, `moe-decode-roadmap.md` §9.1/9.3.

Copyright Kevin Read <me@kevin-read.com>

## Verdict

The plan's §0 re-diagnosis is **correct and code-verified** (the FA compute
kernel is live-bounded; the gather launch is the capture-frozen piece), and
its §1 path decision (build on LEGACY=1) is sound. But the plan fails as an
implementation plan on three counts:

1. **Its target design (Option 1, conditional graph nodes) is unbuildable on
   this stack, and that is verifiable today in 30 seconds** — the plan's own
   Q1 "blocking open question" has a definite answer: no. By the plan's own
   §2.3 logic, the plan as written has no viable primary design.
2. **Its cost model is wrong**: it attributes the frozen cost to block
   *dispatch*, but the out-of-range gather blocks are not trivial — they
   zero-write the entire V tail, and (on the >65535 fallback path that the
   131072/262144 configs actually take) a separate dense `quantize_q8_0`
   pass reads+writes the whole `Sk_pad`-wide K buffer. These memory-traffic
   terms dwarf dispatch by ~2 orders of magnitude.
3. **At the two `max_model_len` values that motivated the plan, it describes
   the wrong code path**: V1's grid cannot exist at `Sk_pad > 65535`
   (`gridDim.z` cap); the launcher auto-switches to V2 and the LEGACY path
   falls back to `gather_paged_kv_fp16` + whole-buffer `quantize_q8_0`. The
   plan's "primary fix" leaves the largest frozen term (the quantize)
   completely untouched at exactly those configs.

Meanwhile the option the plan **dismisses** (Option 2), done properly as a
persistent/grid-stride kernel with a small *fixed* grid and a live
`seq_lens`-driven loop bound — the same pattern the FA kernel already uses —
solves both the dispatch and the work problem with no tiers, no new graph
APIs, and no dispatcher changes. The recommendation of this review: rewrite
§2 around kernel-side live bounding and re-gate everything on a cost
decomposition micro-bench run *first*.

---

## What the plan gets right (verified, keep)

- **§0 correction of the investigation is accurate.**
  - FA dense grid `dim3((seq_q+NC1-1)/NC1, kv_split, batch*heads_q/NC2)` —
    no `Sk` dimension (`gfx906_fa_launcher.hip:217-221`); paged variant
    grid `z = batch*ntiles_z` likewise (`:549-553`).
  - K-loop bound `k_VKQ_max = KV_max[...]` is a live per-sequence value
    (`fattn-q8.cuh:961`), fed from `kv_max_tensor = seq_lens.to(int32)`
    (`gfx906_fa_paged.py:~587`) — and `seq_lens` here is the runner's
    **persistent int32 GPU buffer** (`gpu_model_runner.py:825-827`),
    refreshed in place every step (`:2252-2255`), so `.to(int32)` is a
    no-op returning the same tensor and the baked kernel arg is live at
    replay. Confirmed end-to-end in code, not just by the investigation's
    narrative.
  - `GFX906_FA_KVSPLIT` default 16, static (`gfx906_fa.cpp:84-90`).
  - Capture freeze `max_seq_len = self.max_model_len` for every
    `for_cudagraph_capture` build (`gpu_model_runner.py:2387-2391`), and
    `max_seqlen_k=attn_metadata.max_seq_len` flows into `Sk_pad` and the
    gather grid (`gfx906_fa_backend.py:~675-690`, `gfx906_fa_paged.py:~316`).
- **§1 (build on LEGACY=1)** — verified: `get_cudagraph_support` raises
  `RuntimeError` for LEGACY=0 + prefix caching (`gfx906_fa_backend.py:
  100-115`, R2); the test file has exactly 8 tests, all on the LEGACY path,
  two asserting `impl._legacy` directly (`test_gfx906_fa.py:300,397`); none
  exercise `forward_paged_direct`. The decision and its rationale hold.
- **§2.1's final position on grid freezing** (launch geometry baked at
  capture; live grid dims impossible inside one captured graph) is correct.
- **Honest pricing of Option 3** (short-context mitigation only, inherits
  the investigation's correction) — correct.

---

## Findings

### F1 — BLOCKER: Q1 is answerable today and the answer is "no" — Option 1 is unbuildable on this stack

Evidence (checked, not speculated):

- `/opt/rocm/include/hip/hip_runtime_api.h:1560-1577`: the
  `hipGraphNodeType` enum on this ROCm 7.14 install ends at
  `hipGraphNodeTypeBatchMemOp = 14`. **There is no conditional node type.**
  `grep -rn "Conditional" /opt/rocm/include/hip/` returns nothing.
  CUDA 12.3+'s `cudaGraphNodeTypeConditional` has no HIP equivalent here.
- Explicit graph construction APIs exist (`hipGraphAddNode:8782`,
  `hipGraphAddKernelNode:8857`, `hipGraphCreate:8546`), but there is no
  conditional node type to add, and `hipGraphNodeTypeHost` (CPU callback at
  replay) cannot skip sibling GPU nodes — graph topology is fixed.
- PyTorch is a custom `2.13.0+gfx906` build; `torch.cuda.CUDAGraph` exposes
  only capture/instantiate/replay. There is no hook to insert any node into
  a stream-captured graph — and note that even on CUDA, conditional nodes
  are *not* producible by stream capture; they must be constructed via the
  explicit graph API. So even a future ROCm with conditional nodes would
  not make Option 1 reachable "from the existing `CUDAGraphWrapper`/
  `torch.cuda.graph` capture flow" — it would require hand-building the
  entire FULL-step graph outside PyTorch. The plan's "cheap in VRAM and
  capture time" framing was wrong on the axis that matters (integration
  cost), not just the capability check.

Consequence: by the plan's own §2.3 ("without conditional nodes, Option 1
is not buildable at all and Option 3 … or accepting the … tax are the only
remaining choices"), the plan's recommended target design is dead. A
blocking prerequisite that can be refuted with a header grep should not
have been deferred to implementation step 1 — it invalidates §2.2's
recommendation, §3 steps 2-6, and the Q3 tier-threshold workstream in one
stroke. Record Option 1 as DEAD-END with this evidence so it is not
re-fancied later (house refrigeration rule).

### F2 — CRITICAL: the cost model ("dispatch") misidentifies the dominant frozen terms — they are memory traffic

The plan repeatedly describes out-of-range gather blocks as "trivial
early-exit blocks" whose only cost is their scheduling/dispatch slot
(§2.1 Lever A, §2.2 Option 2). That is factually wrong about what the
kernels do:

- **V1** (`gather_paged_kv_q8_kernel`, `gather_paged_kv_quant_kernel`): an
  out-of-range block writes a full zero V row before returning
  (`gfx906_fa_gather.hip:~120-133`: `if (tok_pos >= seq_len …) { zero V;
  return; }`). The store is the work; the block is not a no-op.
- **V2**: the V pass stores unconditionally — `val = tok_valid ? load :
  zeros; store(val)` (`gfx906_fa_gather.hip`, V pass `for (idx…)` with
  `make_uint4(0u,0u,0u,0u)` on the invalid path). Every call zero-fills
  the **entire** `B × Hkv × Sk_pad × D` V buffer tail.
- **`quantize_q8_0`** (the separate dense quant on the >65535 path, see
  F3): `N` is the product of *all* leading dims of the gathered K tensor —
  i.e. `B × Hkv × Sk_pad` rows — with no `seq_lens` awareness
  (`gfx906_fa.cpp:382-412`; grid `(N+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK`,
  `gfx906_fa_quant.hip:177`). It reads the full fp16 K buffer (including
  the never-written garbage tail) and writes a full q8 buffer.

Back-of-envelope per FA layer per step (B=1, Hkv=4/rank for TP=2 27B,
D=128; **launch-regime estimate, not a gate**; B/Hkv for this model not
re-verifiable this session — `config.json` on NFS was unreachable):

| term | 262144 | 131072 | Δ (262k−131k) |
|---|---|---|---|
| V-tail zero-fill writes | 268 MB | 134 MB | 134 MB |
| dense quantize (read fp16 + write q8) | 411 MB | 205 MB | 205 MB |
| V2 block dispatch (Sk/16 × Hkv WGs @ ~1-2 ns) | ~65k WGs ≈ 0.07-0.13 ms | ~33k WGs | ~0.03-0.07 ms |

Traffic Δ ≈ 340 MB/layer → ~3.4 GB over 10 FA layers ≈ **3.5-4 ms** at
~1 TB/s — the same order as the observed 8.4 ms/step gap (25.1→33.4 ms;
B=2 in the harness would close it). Dispatch Δ is **two orders of
magnitude smaller**. Consequences:

- Lever A's dismissal ("limited headroom — the dispatch count doesn't
  shrink") is inverted: bounding the *zero-fill* is most of the win, and it
  requires no grid change at all (see F4).
- Any grid-only fix (Option 1 tiers, or shrinking the launched grid) that
  leaves `quantize_q8_0` untouched keeps the single largest frozen term.
- The plan has **no magnitude budget at all** (the investigation's
  performance-impact table was dropped, not corrected): no expected ms/step
  recovered, no target for the serving A/B gate to confirm.

### F3 — CRITICAL: at 131072/262144 the plan describes the wrong code path; the "primary fix" misses the biggest cost there

The plan's §0/Q4 frame `launch_gather_paged_kv_q8` **V1** as "the default"
bug site and treat V2 as an opt-in (`GFX906_FA_GATHER_V=2`, Q2). But HIP
caps `gridDim.z` at 65535, and the code knows it:

- `launch_gather_paged_kv_q8`: `if (cached_version == 1 && Sk > 65535)
  cached_version = 2;` — **V1 is auto-replaced by V2 at
  `Sk_pad > 65535`** (`gfx906_fa_gather.hip`, launcher). So at both
  `max_model_len=131072` and `262144` — the exact configs whose regression
  motivated this plan — the gather already runs V2 with
  `gridDim.z = ceil(Sk_pad/16)` (8192 / 16384), not V1's 262144-cell grid
  (which is illegal and cannot launch).
- The fused gather+quant kernel hard-fails beyond the cap
  (`TORCH_CHECK(Sk <= 65535, …)` in `gather_paged_kv_quantized`,
  `gfx906_fa.cpp:~771`; launcher returns `hipErrorInvalidValue`), and the
  Python wrapper switches to the **two-kernel fallback**:
  `gather_paged_kv_fp16` (byte-generic V2) + separate
  `quantize_q8_0(K_bhsd)` over the full buffer
  (`gfx906_fa_paged.py`, `if _FUSED_QUANT and Sk_pad <= 65535:` branch).

So at the motivating configs, the frozen cost is (V2 gather grid + V-tail
zero-fill) **plus a whole-buffer dense quantize the plan never mentions**.
The plan's fix surface — "change `launch_gather_paged_kv_q8` (both V1 and
V2 variants)" — does not touch `quantize_q8_0`, whose grid is frozen at
`N = B×Hkv×pad32(max_model_len)` rows and whose traffic (F2) is the larger
of the two terms. Even a hypothetically-working tiered design cannot fix
the top tier: any tier above 65535 sits on the two-kernel path.

Also note for Q2: "V2 … just not default … worth A/B'ing on its own" is
moot — **V2 is already the active kernel at both 131072 and 262144** by the
auto-switch, and a V1-vs-V2 A/B is impossible at those sizes (V1 can't
launch). The sticky `static int cached_version` mutation (a >65535 call
flips the process to V2 permanently, including for later small-`Sk` calls
such as prefill chunks) also contaminates any in-process V1/V2 comparison;
pin via the env var, and expect the flip, when benching.

### F4 — MAJOR: Option 2 is a strawman; done properly it is the correct design, and it is the plan's own §0 pattern applied to gather

The plan's Option 2 keeps the `Sk_pad`-sized grid and adds a per-block
live early-return — and then correctly observes this doesn't reduce
dispatch. True, but that is not the persistent-kernel design. The real
pattern:

- Launch a **small, fixed grid** (e.g. enough WGs to fill 64 CUs; a
  constant, hence trivially capture-safe — the same graph serves every
  `max_model_len` and every live length).
- **Grid-stride loop over the (seq, head, token) space, bounded by a live
  value read from the persistent `seq_lens` tensor inside the kernel** —
  exactly the mechanism the plan spends §0 verifying in `fattn-q8.cuh:961`
  (`k_VKQ_max`), and exactly what the investigation's "masked early-exit
  … the actual fix" alternative asked for. Varying a data-dependent loop
  bound inside a captured kernel is ordinary, legal HIP.

This solves *both* problems the plan treats as competing: work is
`O(real_seq_len)` (live K/V copies + margin zeroing only) *and* dispatch is
a small constant — strictly less than today's V2 dispatch at any `Sk ≥
~26k`. No conditional nodes, no tiers, no dispatcher fallback, no batch-wide
cliff, no Q3 traffic-distribution data needed to pick thresholds.

Correctness detail the design must state: the FA kernel reads V only up to
the straddling tile — its loop bound is `k_VKQ_max` with one `oob_check`
tail tile (`fattn-q8.cuh:961-985`) — so at most `nbatch_fa` rows past
`seq_len` are ever touched. The V zero-fill therefore only needs to cover
`[seq_len, seq_len + nbatch_fa)` (the margin exists because the tail tile
*loads* V for masked rows and `0 × NaN = NaN`; garbage V would poison the
row). Rows beyond the margin can hold stale/garbage data forever. K tail is
already skipped (FA cuts it via `kv_max`). Two sequenced, independently
shippable steps fall out of this:

1. **Cheap intermediate (kernel-only, no graph implications):** keep the
   grids exactly as today; change the OOB handling in all three gather
   kernels from "zero the whole tail" to "zero only the `nbatch_fa` margin,
   return without writing beyond it", and live-bound the dense `quantize_q8_0`
   (or better: extend the fused gather+quant kernel to the V2 paged-block
   form valid at all `Sk`, deleting the separate quantize pass entirely).
   This removes the F2 traffic while leaving dispatch as the (small)
   residual — and it is a few-line change per kernel, not a kernel-design
   project.
2. **Full fix:** the persistent/grid-stride form above, which also removes
   the dispatch residual.

The plan rejected the correct design by mischaracterizing it ("Option 2
doesn't change grid size (can't, under a single capture)") — a persistent
kernel's grid is small and fixed *by design*; nothing needs to change at
capture.

### F5 — MAJOR: step ordering violates the house protocol it cites; the decisive experiment is missing from the front of the queue

`docs/gfx906/AGENTS.md`: micro-bench per shape *before* the model path;
serving A/B is the gate. The plan's order is: HIP capability spike (§3.1)
→ kernel changes (§3.2) → kernel tests (§3.3) → **then** micro-bench
(§3.4) → PPL → serving A/B. The cheapest, most decisive experiment — a
**cost decomposition** of one decode step at captured `Sk_pad` ∈ {131072,
262144} with ~1.5k live tokens, splitting gather / quantize / FA kernel
time — is absent entirely, yet it (a) validates or refutes §0's magnitude
claim (is the gap really all gather-side?), (b) measures the traffic vs
dispatch split that decides whether F4's step 1 alone suffices, and (c)
produces the per-`Sk` coefficients any tier/threshold choice (Q3) or win
estimate needs. Note `benchmarks/kernels/gfx906_fa/bench_gfx906_fa_gather.py`
is eager-only today — "extend" (§3.4) means building capture-replay support
into it; budget for that.

Related gap: Option 3 at a ≤65535 bound (e.g. 32768) moves the decode path
onto the **fused V1** kernel, whose frozen grid is `B×Hkv×Sk_pad` (≈131k
WGs at a 32768 bound, plus full-width zero-fill) — a residual the
investigation's Option-3 win table ignores. If Option 3 is shipped as the
stopgap, this residual should be measured, not assumed away.

### F6 — MINOR: factual and precision nits

- **§2.1 self-contradiction**: "grid dimensions computed from a live …
  scalar at replay time are allowed to vary …" is flatly asserted, then
  corrected two sentences later ("HIP graph capture bakes the kernel-launch
  parameters …"). The final position is right; the paragraph should state
  only the correct version.
- **Citation drift**: `gfx906_fa_gather.hip:90-118` for the early-exit is
  actually ~120-133; `gfx906_fa_paged.py:500-511` for the gather calls is
  ~449-530 (branchy); "several assert `impl._legacy`" = exactly two.
- **§2.1 "potentially tens of thousands of cells at max_model_len=262144"**
  — under-states V2 (16384 × B × Hkv cells, i.e. up to ~260k WGs at B=4)
  and over-states V1 (illegal at that `Sk`).
- **S8 wording**: `DEVLOG-tp2-dense.md` S8 still says "replays attend
  max_model_len-wide", the mechanism §0 supersedes. Per the house
  dev-log rules, add a one-line correction cross-ref there when this plan
  (or its successor) lands.
- **Unverified model geometry**: the 27B's Hkv-per-rank and FA-layer count
  ("~10") are still "needs confirming" (carried from the investigation);
  `config.json` was unreachable this session. F2's absolute numbers inherit
  that uncertainty — confirm before quoting them.

### F7 — MINOR: the §3.3 test assertion is incompatible with any tail-garbage-tolerating design

"Assert the conditional-tier gather produces **bit-identical** output to
today's single-grid gather" cannot hold for the designs this review
recommends (margin zeroing, persistent live-bounded kernel): beyond the
margin, V holds stale/garbage instead of zeros — *by design*, since FA
never reads it (`k_VKQ_max` bound). The correct assertions are (a) bitwise
equality on `[0, seq_len + nbatch_fa)` for K and V, and (b) end-to-end
equality of the FA output (post-attention), which is the actual contract.
Also beware `torch.equal` / NaN comparisons reading the garbage tail in
`_DOUBLE_CHECK`-style checks.

---

## Revised recommendation (what this plan should become)

1. **Step 0 (new, gates everything):** cost-decomposition micro-bench —
   captured `Sk_pad` ∈ {131072, 262144} (optionally 32768), ~1.5k live
   tokens, per-kernel times for gather / dense-quantize / FA, plus eager
   controls. Confirms §0's mechanism quantitatively and decides how much
   of the gap is traffic (F2) vs dispatch.
2. **Primary design (replaces Options 1+2):** kernel-side live bounding —
   margin-only V zeroing in all three gather kernels + live-bounded (or
   fused-and-paged) quantize as the shippable first commit; persistent
   fixed-grid gather as the follow-up if dispatch proves non-negligible in
   step 0's data. Capture-safe by construction; single FULL graph serves
   all context lengths; no tiers, no fallback cliff.
3. **Option 3** stays as the config-only stopgap for short-context traffic,
   with its residual fused-V1 dispatch/zero-fill priced (F5).
4. **Option 1: record as DEAD-END** (F1 header evidence) instead of
   "open question", so the conditional-node idea is not revived when
   someone greps the plan in six months.
5. Keep §1 (LEGACY=1) and the §3 gate ladder (kernel test → micro-bench →
   PPL/greedy → serving A/B with exact config recorded), re-anchored to
   the designs above, and fix the F6/F7 nits in passing.

## Cross-references

- `plan_masked_fa.md` — the reviewed plan.
- `tp_decode_investigation.md` — origin; RESOLUTION table (39.9/29.9 t/s
  graph vs ~19.5/19.9 eager) that any decomposition bench must reproduce.
- `DEVLOG-tp2-dense.md` S8 — carries the superseded "attend max_model_len-
  wide" wording (F6).
- `DEVLOG-fa-attention.md` — V1/V2 history incl. the V2 serving trap
  (56.9 vs 49.6 t/s at D=256, Sk=3328) and the fused-gather track.
- `csrc/gfx906_fa/gfx906_fa_gather.hip`, `gfx906_fa.cpp:382-412`,
  `gfx906_fa_quant.hip:177` — the code F2/F3 rest on.
