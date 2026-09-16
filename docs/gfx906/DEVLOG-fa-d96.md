# FA-D96 — a 96-wide head-dim instantiation for the custom FA (ViT 72→96, text 96→96)

> Branch `gfx906/fa-d96` off `main` · model `cyankiwi/Qwen3.8-27B-AWQ-INT4` (VL,
> image prompt) + Phi-3-mini-class text shapes · date `2026-09-16` ·
> roadmap item `VIT-2` (broadened) · follows `FA-COVER-1` step 2.

**VERDICT:** `SHIPPED` as an **opt-in** (`GFX906_FA_PAD96=1`); the reviewed default
map stays `(64,128,256)` · **GATE:** image-prompt TTFT A/B, serving, TP=1 GPU0,
1024×1024 **fresh** image per rep, `--no-enable-prefix-caching`, 3 reps/arm +
encoder-cache control.

**Gate result: −1.2 % (5.133 → 5.073 s), and the A-B-A order control was lost to a
GPU reset burst — so the default does not move.** The kernel instantiation, the
table fix and the tests ship; the pad map is opt-in until a clean-boot re-read of
the gate. Standalone numbers below are **launch-regime evidence**, not the gate.

---

## HYPOTHESIS

The launcher instantiated `{64,128,256}` only, so the ViT's real head_dim 72 was
zero-padded to 128 (VIT-1) and the head_dim-96 text class (Phi-3-mini, FA-COVER-1
step 2) to 128 as well. Measured cost tracks the **padded** dim (`DEVLOG-vit1.md`:
head sizes 72/80/96/112/128 all 7.7–8.0 ms, D=64 3.2 ms). If a 96-wide
instantiation exists, the 96..127 q8_0 block (all zeros by construction) stops
being read and computed, and the ViT attention should drop ~20 % with
**bit-identical** output — the removed dims contribute 0 to QK, quantise to zero
q8_0 blocks, and add 0 to P·V.

## What was done

- **C++**: `head_dim == 96` dispatch + `gfx906_fa_launch_impl<96>` / `…_paged_impl<96>`
  in `csrc/gfx906_fa/gfx906_fa_launcher.cu` (the `(96,96)` tile-config rows were
  already in the inherited llama.cpp table).
- **Tile table**: the inherited `(96,96)` rows were **wrong for this kernel
  family** — see Evidence AGAINST #1. Retuned to `nbatch_K=96` and `nbatch_fa`
  mirroring D=128 (128 for ncols 2, 64 for 4..64), occupancy 2 for ncols ≤ 8 and
  1 above (D=128's gfx906 tuning split).
- **Kernel guards**: `static_assert(nbatch_K % 32 == 0)` added to **both** kernel
  copies (`fattn-q8.cuh`, `fattn-q8-paged.cuh`) so a table row that cannot work
  with the Q8 block layout fails at compile time instead of silently scoring
  fewer dims.
- **Python**: `_INSTANTIATED_HEAD_DIMS = (64, 96, 128, 256)` (the kernel whitelist)
  in `gfx906_fa_backend.py` + `gfx906_fa_mm_encoder.py`, with the *pad map* left on
  `_FALLBACK_HEAD_DIMS = (64, 128, 256)`; `GFX906_FA_PAD96=1` selects the 96 map
  (exact 96 native, 72/80 → 96). Both the map and the `GFX906_FA_PAD=0` kill
  switch now read the *active* map, so the kill switch keeps its pre-FA-D96
  semantics (96 falls back rather than being served by a dim the map lacks).
- **Tests**: new `test_head_dim_96_instantiated_gather_and_paged_vs_fp32_ref`
  (both entry points, causal, vs fp32 ref); pad-map/customize-spec/padded-head-dim
  tests parametrized over both pad targets. Suite **102 passed** (was 97).
- **Bench**: `benchmarks/kernels/gfx906/bench_fa_d96.py` (dispatcher-faithful,
  interleaved A-B-A-B passes, deciles, concurrent mclk sampling with a hard gate
  and an arm-symmetry check).

## Evidence — FOR

Launch-regime A/B (`bench_fa_d96.py`, GPU0, mclk noted per window, interleaved two
passes, second pass reported; all outputs **bit-identical** between arms):

| case | 128-pad | 96-pad | Δ | arms differ |
|---|---|---|---|---|
| ViT 1024×1024 (S=2304, H=16, D=72) | 7119.8 µs | 6277.3 µs | **−11.8 %** | 0.00e+00 |
| ViT 512×512 (S=576) | 795.5 µs | 730.7 µs | **−8.2 %** | 0.00e+00 |
| text decode (Sq=1, Sk=2048, 32/32 heads) | 84.7 µs | 80.1 µs | −5.4 % | 0.00e+00 |
| text verify (Sq=4, Sk=2048) | 10.6 µs | 10.5 µs | −1.3 % | 0.00e+00 |
| text prefill (Sq=512, Sk=512) | 608.6 µs | 539.6 µs | **−11.3 %** | 0.00e+00 |

Serving gate (dense 27B VL, image prompt, `prompt_sha1=d09bb2875f57` on every rep):

| arm (load) | fresh-image TTFT | mean | encoder-cache control |
|---|---|---|---|
| pad 128 (default), load 1 | 5.152 / 5.118 / 5.118 s | 5.129 | 4.142 s |
| pad 128 (default), load 2 | 5.176 / 5.114 / 5.119 s | 5.136 | 4.141 s |
| pad 96 (`GFX906_FA_PAD96=1`) | 5.110 / 5.054 / 5.055 s | **5.073** | 4.148 s |
| pad 128 (A-B-A order control) | — | — | **lost: load died, wedge #97 (GPU reset burst)** |

**Population: A, A, B** (not A-B-A). The 96 arm is 0.060 s / **−1.2 %** faster than
either 128 arm, and its three reps (spread 0.056 s) sit below all six 128-arm reps
(5.114–5.176 s) — but with the order control missing and the effect inside ~2 σ of
the 128 arm's own spread (sd ≈ 0.024 s), this is *directional*, not a gate pass.
The control (repeat image → 4.14 s) is the "assert the work under test runs" guard:
the ViT is ≈1.0 s of the 5.13 s TTFT, so a −11.8 % ViT attention predicts ≈ −0.12 s
and only −0.06 s materialised. The server log confirms
`Using CUSTOM (gfx906 FA) backend for ViT attention` on every arm.

**Consequence: `GFX906_FA_PAD96` stays opt-in.** The pre-registered expectation for
this item was −5 % TTFT at 1024×1024 (`ROADMAP.md` VIT-2); −1.2 % is below it, and
the repo rule is that a dispatch/default change ships on a measured gate, not on a
directionally-correct one.

## Evidence — AGAINST

1. **The inherited `(96,96)` tile row produced silently wrong numerics.** It
   carried `nbatch_fa=32, nbatch_K=48` (llama.cpp's CUDA tile table), but this Q8
   kernel consumes K in whole q8_0 blocks: `blocks_per_K_row = nbatch_K/32` = 1,
   so only **64 of 96 dims** were scored. Measured before the fix: rel err
   **0.2398** vs the fp32 reference on a D=96 paged-decode call (D=128 control:
   0.0022) — a plausible-looking but wrong result, on both entry points and
   independent of `kv_split`. The same table rows for D=40/80/112 are also
   non-multiples of 32 and cannot be instantiated without a fix (the new
   `static_assert` now says so at compile time rather than at runtime).
2. **The `nbatch_fa` column matters more than the K tile.** With the first-cut
   `nbatch_fa=32` the headline ViT case measured **+6.7 %** (7170 → 7648 µs) —
   i.e. the "25 % less arithmetic" win was invisible because the K-loop needed 4×
   more iterations, and a decode shape was **+277 %** (82 → 311 µs). Mirroring the
   D=128 `nbatch_fa` values flipped the same case to −11.8 % / −5.4 %. Lesson: a
   new instantiation inherits a *table row*, and an untuned row can invert the
   result it was meant to deliver.
3. The ViT win (−11.8 %) is smaller than the roadmap's −22…−29 % estimate: the
   call is not purely head-dim-arithmetic bound (Q/K quantisation and the fixed
   tile overhead do not shrink with the head dim). The TTFT consequence is
   therefore ~−2 %, not ~−5 %.

## Why it (partly) fails / caveats

- The measured TTFT win (−1.2 %) is far below the item's −5 % expectation: the ViT
  attention is ~19 % of the TTFT and the 96-wide kernel only removes ~12 % of that
  call, so the ceiling is ~−2.3 % even with a perfect arm read. **The item's own
  estimate was wrong** (it priced the kernel ratio, not the kernel's share).
- The default therefore does not move tonight: with the order control lost to wedge
  #97 the evidence is A-A-B, and the policy is not to flip a default on a miss.
- D=80 (and 40/112) still cannot be served by this kernel: `DV % nbatch_K == 0`
  and `nbatch_K % 32 == 0` have no common solution for 40/80/112 (a 96-pad is the
  nearest servable target, which is what the pad map already does).

## Interactions / superseded-by

- Supersedes VIT-2's "add a tile-config entry" framing: the entry existed, it was
  the *pair* (K tile, fa rows) that had to be constrained.
- Interacts with FA-COVER-1: the padding path is now the same mechanism for the
  ViT and the text class, so this closes the "head_size not supported" class for
  72/80/96 **and** removes 25 % of the padded arithmetic for it.
- The `nbatch_K % 32` assert makes the table self-guarding for future
  instantiations.

## Refrigerated residue

- **Promotion gate (one clean boot, 2 loads):** A-B-A on the dense 27B VL at
  1024×1024 with ~6 reps/arm (the effect is ~2 σ at 3 reps), **plus** the Phi-3-mini
  arm for the head_dim-96 text class (`GFX906_FA_PAD96=1` vs `0`, the same
  `_bench_gfx906.py` shape FA-COVER-1 used for 36.41/36.18 t/s): if both arms hold,
  flip `_pad96_enabled` to default-on and keep the env as the rollback.
- **fp16-K for the ViT / Pad-D96** (removes the ~2e-2 Q8 error, VIT-2 residue) is a
  separate accuracy-not-speed item.
- **`nbatch_fa` sweep for D=96** at ncols 2/4/8 (the column set by analogy with
  D=128) — the decode shape was the weakest case (−5.4 %).
- **Wedge budget for the promotion session:** the night's three loads produced
  wedge #96 (load lottery) + resets(2) and (3) in 8 min = a burst; the promotion run
  should start on a fresh boot with the ~6-rep A-B-A as its *first* arm pair.

## Search keys

`FA-D96` `VIT-2` `head_dim 96` `nbatch_K % 32` `GFX906_FA_PAD96`
