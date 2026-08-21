# TP=2 dense decode — max_model_len / context-length decode-speed investigation

Copyright Kevin Read <me@kevin-read.com>

**VERDICT:** OPEN · **GATE:** none run yet — this is a code-analysis
investigation, no serving A/B has confirmed the mechanism live. See
"What needs to be checked" for the decisive experiments.

## Origin

During TP=2 dense-27B serving work (`DEVLOG-tp2-dense.md` S5, 39.7 t/s
record at `--max-model-len 131072`), a session testing
`--max-model-len 262144` on the same server/model/flags measured
**29.9 t/s decode vs a 39.9 t/s control at 131072** — a ~25% regression
reproduced both ways (clean restart, acceptance unchanged). This matches
the already-documented finding in `DEVLOG-tp2-dense.md` line 194-195
("Where TP=2 still pays: 131k-context capacity... at 262144 max_model_len
boots... at ~7-12 t/s decode" — a larger gap than 29.9, but same
direction) and `DEVLOG-dense-decode.md` line 476 ("max-model-len 131072
was unbootable [under an older config]... the gather buffers are sized
`max_num_seqs × pad32(max_model_len)`").

Two independent hypotheses were floated before code was read:

1. **Buffer-size theory** — bigger `max_model_len` → bigger block
   tables / FA gather buffers → some fixed per-replay cost scales with
   buffer capacity.
2. **Frozen split-count theory** (informed by generic ROCm/CUDA FA
   knowledge, not gfx906-specific) — FULL cudagraph capture bakes
   Flash-Decoding-style KV-split counts computed from the worst-case
   `max_model_len`; every replay pays for the frozen (inflated) split
   count regardless of real per-request context length.

Both were investigated against the actual code paths active in this
deployment (confirmed via `/tmp/vllm-tp2-mtp3b.log`: `GFX906_FA` backend
registered as `AttentionBackendEnum.CUSTOM`, `--max-model-len 131072`
control run, MTP `num_speculative_tokens=3`).

## What was ruled out

- **Block-table H2D copy size** (`vllm/v1/worker/block_table.py`
  `BlockTable.copy_to_gpu` → `CpuGpuBuffer.copy_to_gpu`). Row width
  (`max_num_blocks_per_req`) does scale with `max_model_len`
  (`vllm/v1/kv_cache_interface.py` `max_num_blocks_per_req()`), but the
  extra bytes moved per step from doubling `max_model_len` are in the
  hundreds-to-low-thousands range at `max_num_seqs=4` — sub-microsecond
  even unpinned; this buffer is pinned + `non_blocking=True`. Cannot
  explain an ~8.4 ms/token regression (25.1 ms → 33.4 ms at 39.9 → 29.9
  t/s). **Ruled out on magnitude.**
- **Triton `kernel_paged_attention_2d`** (`vllm/v1/attention/ops/
  chunked_prefill_paged_decode.py`, the `ROCM_ATTN`/`rocm_attn.py`
  fallback backend). Its inner loop is `num_blocks =
  cdiv_fn(seq_len, BLOCK_SIZE)` (line ~159) — driven by the real
  per-request `seq_len` at kernel-launch time, not by
  `max_num_blocks_per_req`/`block_table.shape[1]`. `block_table_stride`
  is used only as a pointer-arithmetic multiplier for row indexing, at
  zero extra cost regardless of row width. **Not the mechanism — and
  not the active backend for this deployment anyway** (gfx906 always
  has `use_rocm_custom_paged_attention()` return `False`, see
  `vllm/platforms/rocm.py:412-413`, so this path is dead code on
  gfx906 hardware for the custom-kernel branch; `ROCM_ATTN` is only
  selected as a fallback below `CUSTOM`, and the log confirms `CUSTOM`
  was active).
- **Generic split-K/Flash-Decoding "frozen split count baked into the
  graph"** (`vllm/v1/attention/ops/triton_unified_attention.py`,
  `NUM_SEGMENTS_PER_SEQ`). This mechanism is real *in that file*, but
  that kernel belongs to `TRITON_ATTN`/`rocm_aiter_unified_attn.py`,
  which is not the backend in use here. **Right shape of theory, wrong
  kernel for this deployment.**
- **`max_seq_len` frozen to `max_model_len` at replay time.**
  `Gfx906FAMetadataBuilder.build()` (`vllm/gfx906_fa/gfx906_fa_backend.py`
  line ~153) sets `max_seq_len=common_attn_metadata.max_seq_len`, which
  for real (non-capture) steps is computed fresh every step
  (`vllm/v1/worker/gpu_model_runner.py:2393`:
  `self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()`).
  `optimistic_seq_lens_cpu` itself
  (`gpu_model_runner.py:2145-2148`) is `num_computed_tokens_cpu +
  num_scheduled_tokens` — genuinely live per-request data. Freezing to
  `self.max_model_len` (`gpu_model_runner.py:2391`) only happens when
  `for_cudagraph_capture=True`, i.e. the one-time capture/warmup pass,
  not real decode replays. **Ruled out**: there is no code path where
  a request with genuinely short real context is charged
  `max_model_len`-sized work at decode time.

## What was confirmed as the likely mechanism

`GFX906_FA` (`vllm/gfx906_fa/`) is a **gather-then-dense-attention**
design, structurally different from the two Triton kernels above:

1. Decode (`GFX906_FA_LEGACY=1`, the default) gathers K/V from the
   paged fp16 cache into a **dense** buffer of shape
   `[B, Hkv, Sk_pad, D]` via the fused HIP kernels
   `gather_paged_kv_quantized`/`gather_paged_kv_fp16`
   (`vllm/gfx906_fa/gfx906_fa_paged.py` `forward_paged`, `Sk_pad =
   pad32(max_seqlen_k)`, line ~317).
2. `max_seqlen_k` passed in is `attn_metadata.max_seq_len`
   (`gfx906_fa_backend.py:663,675,690`) — confirmed live, not frozen
   (see above).
3. The FA kernel then runs dense attention over the full `Sk_pad`
   width — there is no per-request early-exit/mask-driven skip the way
   `kernel_paged_attention_2d`'s `j == num_blocks - 1` tail masking
   works; the whole padded buffer is real compute.
4. **This cost is empirically linear in `Sk`, already measured and
   recorded**: `DEVLOG-fa-attention.md` line 282-283 — "FA kernel 3272
   (11.3 ≈ 327 µs/layer @ Sk~2176 ≈ 4.3× the Sk~500 72 µs — Sk-linear)."
   Multiplied across every layer (10 FA layers in the Qwen3.5-27B
   config referenced in that devlog; layer count for the TP=2 dense
   model under test needs confirming) and every decode step, a batch
   whose live `max_seq_len` grows 4x produces ~4x the FA-kernel time.

**Conclusion so far:** the ~25% decode regression at `--max-model-len
262144` is not caused by the config value itself (no code path charges
`max_model_len`-sized work at decode time for short-context requests).
It is much more likely caused by **the actual accumulated context
length reached by the 262144 test's requests being longer than the
131072 control's**, combined with this backend's `Sk`-linear
gather-then-dense design (which has no sparsity benefit within the
padded buffer, unlike the Triton paged kernels). This would be expected
behavior for this kernel architecture — "longer real context costs
more decode time" — not a `max_model_len`-triggered bug.

## Secondary candidate: gather-buffer realloc thrash (unconfirmed, likely minor)

`Gfx906FAImpl._ensure_gather_buffers` (`gfx906_fa_backend.py` line
~479) uses an **exact-match** realloc condition on `Sk_pad`
(`b.shape[2] != Sk_pad`, line ~526) rather than a monotonic grow-only
policy like its sibling `_ensure_forward_buffers`/`_q_pad_buf` (which
uses `shape[2] < Sq_pad`, only grows). This means every time the live
batch's `max_seq_len` crosses a 32-token `pad32` boundary — plausible
under MTP (`num_speculative_tokens=3`, variable accepted length) or a
mixed batch where requests finish/start — the K/V gather buffers
reallocate (`torch.empty`), not just resize a view.

This was flagged in code review as **R3**/**R8** in
`moe-decode-roadmap.md` §9.1/9.3 and is **already partially
addressed**: R3 capped the retired-buffer keep-alive list at
`_gather_retired_max = 4` (bounding the leak, not the realloc
frequency), and R8 made the class-level gather buffers shared on both
`LEGACY=0` and `LEGACY=1` paths (previously `LEGACY=1` allocated fresh
per call). Neither fix changes the exact-match shrink-and-regrow
behavior itself. Whether this thrash is a measurable contributor on
top of the expected `Sk`-linear cost, or is negligible next to it
(reallocation is a device `cudaMalloc`-class op, not free, but may be
dwarfed by the 327 µs/layer FA cost at realistic sizes) — **not yet
measured**.

## What needs to be checked

In priority order (cheapest/most decisive first), per the gate rule in
`docs/gfx906/AGENTS.md`: **serving wall-clock A/B is the gate**, not
kernel census.

1. **Sweep real context length at fixed `--max-model-len 262144`.**
   Measure ITL at ~2k, ~30k, ~120k tokens of actual conversation/prompt
   length, same server, same flags as the S5 config
   (`-tp 2 --gpu-memory-utilization 0.93 --max-num-seqs 4
   --max-model-len 262144 --compilation-config
   '{"cudagraph_capture_sizes":[1,2,3,4]}'`).
   - ITL flat across all three → something is still charging
     `max_model_len`-sized (or otherwise config-fixed) work regardless
     of real content; re-open the ruled-out theories above and check
     for a missed freeze point.
   - ITL rises with real length, offset upward vs. an equivalent
     131072-config sweep at matched real length → confirms the
     `Sk`-linear gather-attention story; the "regression" is really
     "the 262144 test ran with longer real context than the control,"
     which is expected, not a bug.
2. **Matched-length control.** Run the 131072 config and the 262144
   config at the *same* real accumulated context (e.g. both servers
   actually holding ~100k tokens of real conversation). If decode
   speed still differs at matched real `Sk`, something beyond the
   `Sk`-linear FA cost is at play — most likely candidate is #3 below,
   or KV-cache/`gpu_memory_utilization` budgeting differences between
   the two `max_model_len` values changing concurrency/preemption
   behavior.
3. **Gather-buffer realloc thrash isolation.** With `GFX906_FA_DEBUG=1`
   (master debug switch, `gfx906_fa_backend.py` R12) or a targeted
   counter, log `_ensure_gather_buffers` realloc events per step during
   an MTP-enabled decode run. Compare realloc frequency/cost at 131072
   vs 262144 configs at matched real context — if thrash is common and
   its cost is non-trivial relative to the 327 µs/layer FA-kernel
   floor, it's a distinct, fixable lever (grow-only capacity buffer
   with an exact-size view, same pattern already used for `_q_pad_buf`
   — noting R3's finding that a naive grow-only capacity buffer risks
   silent corruption here because the gather kernels address output
   from shapes, not strides; any fix must preserve exact-shape
   addressing).
4. **`--enforce-eager` at both max_model_len values** (supporting
   check, not primary — `GFX906_FA`'s `get_cudagraph_support` already
   declares `UNIFORM_SINGLE_TOKEN_DECODE`/`UNIFORM_BATCH`, so this
   mainly rules out any residual capture-vs-eager metadata-freezing
   difference not covered above). Gap collapses → capture-related
   after all despite the code-read above; gap persists → confirms the
   mechanism is the live-context-length-driven FA cost, not a
   graph-capture artifact.
5. **`rocprofv3 --kernel-trace` diff** on both configs at matched real
   context: confirm the FA/gather-kernel durations account for the
   full measured gap (per the `Sk`-linear coefficient from
   `DEVLOG-fa-attention.md`), rather than some other kernel growing.
   Per `docs/gfx906/AGENTS.md`, treat this as launch-regime evidence,
   not the gate — the serving A/B in #1/#2 is authoritative.

## Cross-references

- `DEVLOG-tp2-dense.md` — TP=2 serving record and the original
  131072-vs-262144 observation (S4/S5 sections).
- `DEVLOG-fa-attention.md` — source of the `Sk`-linear FA-kernel
  measurement (327 µs/layer @ Sk~2176 vs 72 µs/layer @ Sk~500) and the
  fused-gather buffer design/capture-safety notes.
- `DEVLOG-dense-decode.md` — earlier note on gather-buffer sizing vs
  `max_model_len` in a different (hybrid GDN+FA) deployment.
- `moe-decode-roadmap.md` §9.1/9.3 (R3, R8) — the gather-buffer
  retired-list cap and shared-buffer fixes that partially bear on the
  thrash candidate in this doc.

---
## RESOLUTION (2026-08-21 late, pi agent) — experiment #4 decisive: capture-baking CONFIRMED

**VERDICT:** CONFIRMED mechanism — FULL-cudagraph capture freezes
`max_seq_len = self.max_model_len` (`gpu_model_runner.py:2390`,
`for_cudagraph_capture` branch), so the GFX906_FA gather-then-dense
kernels get `Sk_pad = pad32(max_model_len)` baked into their launch
dims; **every decode replay pays dense attention proportional to
max_model_len regardless of live context**. The "ruled out" above was
correct for live/piecewise steps (metadata is fresh) but incomplete:
FULL graph replay re-executes capture-time launch dims.

Evidence (identical prompts ~1.5k real tokens, same offline harness):

| max_model_len | graph decode | eager decode |
|---|---|---|
| 131072 | 39.9 t/s (server) | 18.9-19.5 t/s |
| 262144 | 29.9 t/s (server) | 19.7-19.9 t/s |

Eager gap at matched real context: none (256k marginally faster = noise).
Graph gap: -25%, tracking pad32(max_model_len) doubling.

### Fix lever (open)

Capture-time Sk bound: capture with `max_seq_len = min(max_model_len,
GFX906_FA_CAPTURE_SK)` (env knob), accepting piecewise/eager fallback
for live contexts beyond the bound. Since dense gather must cover the
worst case at replay, a graph captured at 32k cannot serve 120k
contexts directly — the knob trades capture coverage for replay speed.
Even the 131k config overpays today (replays attend 131072-wide for
typical short contexts): a 32k capture bound could speed ALL decode,
not just recover the 256k tax. Needs the gather kernel's Sk handling
reviewed for a masked-early-exit alternative (dense design has none
today).
