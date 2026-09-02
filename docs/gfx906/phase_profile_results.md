# MTP-1 kernel-level phase breakdown @120k — RESULTS (2026-09-02, boot R)

Method: CUDA-event forward hooks in a vLLM general plugin (`mtp1_phase_plugin.py` v5),
mode-NONE eager path, TP=2, util 0.85, 122880-token context, median aggregation.
Full process → skill `mi50-kernel-time-benchmarking`.

## VALID: Greedy @120k (run p9771, clean GPU)
Median extrapolated total = **78.4 ms/step**, vs measured wall 78.7 ms/token
(12.74 t/s) → **0.4% error**. This validates the measurement method.

| category | layers | ms/step | % of step |
|---|---:|---:|---:|
| **full_attn** (O(Sk)) | 16 | **57.3** | **73%** |
| mlp (dense GEMM) | 64 | 12.6 | 16% |
| lin_attn (GDN, O(1)/step) | 48 | 8.6 | 11% |

Per-call medians: full_attn 3579 µs, mlp 197 µs, lin_attn 178 µs.
Single stream per rank (no multi-stream). Both TP ranks agree within <0.1%.

**Interpretation:** at 120k context, **full attention dominates the decode step
(73%)**. This is exactly the O(Sk) growth that the MTP-1 sweep showed turning MTP
from a win (≤32k) into a loss (≥64k). GDN linear-attn stays flat and cheap (11%).

## INVALID: MTP @120k (run p11283, degraded GPU) — DO NOT USE
Reported total 297 ms/step, full_attn 90.6%. **Rejected** because:
- Window ran at **0.55 t/s vs the known 9.18 t/s** (16× slow) — far beyond MTP's real
  overhead (sweep showed MTP = 0.72× greedy, i.e. only ~1.4× slower per step).
- full_attn median 16.8 ms/call vs greedy's clean 3.59 ms/call for *identical* attention work.
- Run landed on GPU0 immediately after wedge #5 (16:43), in a soft-degraded state.
The MTP-vs-greedy crossover is already pinned by the validated sweep; only the MTP-side
kernel split is missing, and it needs a clean GPU0 to re-run.

## Why MTP loses at long context (synthesis of sweep + this breakdown)
1. Base decode step @120k ≈ 78 ms, **73% of it full attention** (O(S·k), k = KV length).
2. MTP depth-2 adds a drafter forward + an extra verify per accepted token. The drafter
   re-runs the same O(Sk) full-attention over the long KV, so its cost scales with context
   just like the target's.
3. At short ctx (≤32k) attention is cheap enough that the ~1.5× extra work still nets a win
   (acceptance 2.0). At ≥64k the attention-heavy drafter+verify cost exceeds the tokens it
   saves → net loss, widening to 0.72× at 120k.

## GPU0 wedge log today (5×, all mode-NONE + long-ctx config family)
boot-Q 20:29 · 08:54 · 10:56 (prefill@120k) · ~11:07 (weight load) · 16:43 (MTP weight load).
All recovered via BACO; VRAM back to ~10 MiB baseline each time, no zombie handles.
Only 2 logged "GPU reset begin" — the rest were soft BACO recoveries, i.e. a GPU that keeps
dipping into a degraded state. See `docs/gfx906/degradation.md` (uncommitted).

## Next steps (need user direction)
- **Re-run MTP arm on a clean/healthy GPU0** to get the valid MTP kernel split. Given 5 wedges,
  a reboot before the attempt is the safe choice.
- The relative finding (full attention dominates at long ctx → drives the MTP crossover) does
  NOT depend on the invalid MTP run and stands on the validated greedy breakdown + sweep.

## Artifacts
- Plugin: `/local/tmp/mtp1/mtp1_phase_plugin.py` (v5) — also in venv site-packages.
- Driver: `/local/tmp/mtp1/phase_driver.py` · launcher `run_phase.sh` + `mtp1phase@.service`.
- Sanity test: `/local/tmp/mtp1/event_pair_test.py` · summary: `/local/tmp/mtp1/phase_summary.py`.
- Raw JSON: `phase_greedy_p9771.json` (valid), `phase_mtp_p11283.json` (invalid).
