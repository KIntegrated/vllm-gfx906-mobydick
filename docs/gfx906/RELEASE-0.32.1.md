# gfx906 fork — 0.32.1 release snapshot (2026-10-23)

**What this is.** A patch release on the 0.32.0 base. The 0.32.0 image
(`0.32.0-9b85b0e9f4`) predates the fixes below, and one of them is a capacity
regression a user cannot work around in that build.

## Headline: Mamba prefill checkpoints now default OFF

Upstream enables a Mamba2 prefill checkpoint unconditionally. On this hardware
and model class it is a bad trade:

| | blocks per request | KV pool at 16k |
|---|---|---|
| checkpoint ON (upstream default) | 4 Mamba groups × 3 + 7 attention = **19** | 2,345,498 tokens |
| checkpoint OFF (this release) | 4 × 2 + 7 = **15** | **2,970,965 tokens** |

**+21.0 % KV capacity** (`15/19 = 0.789` → −21.1 % predicted, −21.0 % measured),
for no measurable quality change: the feature buys at most one Mamba block of
prefix-cache tail recompute — ~0.4 % of a follow-up TTFT on Lightning — and
Lightning has **no MTP head**, so the sibling-resumption case the feature exists
for cannot occur on the model this fork actually serves.

`VLLM_GFX906_MAMBA_PREFILL_CKPT=1` restores upstream behaviour for models where a
prefix-cache tail resume is worth a block per group. Verified with no env var
set: four groups size at `required_blocks=2`, pool 2,963,319 tokens on a fresh
boot. Details: `docs/gfx906/DEVLOG-nemotron-h.md`, P32-3/#62.

**This is the reason for the release.** The 0.32.0 image contains neither the new
default nor the kill switch, so on that build the 21 % cannot be recovered by any
supported flag.

## Also in this release

- **FA launch-config records** (`vllm/gfx906_fa/tuning.py`,
  `tuning_warmup.py`): the launch parameters (`nc2`, KV split, gather clip) now
  have one definition each in the C++, pinned against no-GPU-work probes
  (`nc2_effective`, `kv_split_default`) over 96 shape combinations, recorded per
  shape at boot, and passed explicitly from Python. **Behaviour-neutral and
  verified as such** — dense-27B PPL 14.3750 (its recorded baseline, exact) after
  each step. Records live in `$VLLM_CACHE_ROOT/gfx906_fa/`; a box with none
  behaves exactly as before. Two Python-side rule bugs were found and fixed by
  the probes: the paged-direct KV split is batch-aware (`8/8/5/2/2`), and `nc2`
  downgrades `8 → 2 → 1` by GQA ratio and clamps to 1 above Sq=2.
- **`VLLM_GFX906_KV_SIZING_DEBUG=1`** logs per-group `required_blocks`, page
  bytes and pool size at KV-sizing time — how the Mamba cost above was
  attributed.
- Harnesses and tools: `_bench_longctx.py` (TTFT + decode rate separately,
  `BENCH_PROMPT_FILE` for byte-identical cross-path comparisons), `_probe_ttft.py`
  (streaming first-chunk TTFT), `_fa_tune.py` / `_fa_tune_serving_cells.py`
  (launch-config measurement, mclk-gated), `gfx906_perf.sh` (pin/release the DPM
  performance level; sudoers-friendly single-word arguments).

## Measurement corrections carried in the docs (not code)

Three findings that changed how this box must be measured, all recorded in
`docs/gfx906/DEVLOG-kv-formats.md` and `/local/git/AGENTS.md`:

1. **Engine-log throughput is not a measurement instrument** — it reported
   1121 tok/s prompt throughput while the client measured 39.2 s, and
   2.7/21.1/287.9/574.4 tok/s on a re-run of the same configuration.
2. **A slow first prefill is a first-call cost, not a clock effect** — the first
   11k-token prefill costs ~39.9 s once per process, then 4.35 s per call at mclk
   350, 800 *and* 1000 MHz (pinned). Pinning the clock does not help.
3. **Serving warm-up should cover the length being served** — our 64-token
   warm-ups leave ~35 s on the user's first long-context request.

## Known caveats

- The FA record lookup reads `$VLLM_CACHE_ROOT/gfx906_fa/`. A box carrying
  records written by an older build will use them; they only select launch
  parameters and are PPL-neutral, but note it when comparing against a clean box.
- Records remain **opt-in for serving**: the measured kernel-level gains (up to
  41 % on the KV-split decision at some shapes) do not survive to end-to-end at
  256-token contexts, and where they might (long context) the backend choice
  dominates.
- `gfx906-perf on` pins sclk to 1800 MHz as well as mclk to 1000 — real power and
  fan consequences; it is not a fix for the first-call cost above.

## Not re-tagged

`0.32.0` and its `0.32.0-*` tags are published and stay immutable; this release
adds `0.32.1-*` alongside them.
