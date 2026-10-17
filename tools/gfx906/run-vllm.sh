#!/usr/bin/bash
# --- COPY OF THE GFX906 DEPLOYMENT LAUNCHER (API key removed) -------------
# This file is a snapshot of the launcher that boots this box, exported by
# tools/gfx906/export-run-vllm.sh so that the flags, knobs and measurement notes
# that took weeks to accumulate travel with the source tree.
#
#   Two differences from the script a box actually runs, plus one repo-only lint
#   suppression, and only these:
#   * `export VLLM_API_KEY=...` no longer carries a literal secret - the value
#     must come from the environment (VLLM_API_KEY=... ./run-vllm.sh).
#   * this header.
#   * the shellcheck disable line below, because this repo lints every .sh and the
#     live script's five findings are benign-by-design. Nothing else was edited:
#     the copy stays byte-identical to what the box runs.
#
# It is NOT a supported entry point. Paths (/ai/models/..., logs/, scripts/*.py)
# are those of the reference box, and the preflight canary it calls is not in this
# repo. Re-export with the exporter above; `--check` fails if the copy drifted.
# Docs: docs/gfx906/running.md   Measurement log: docs/gfx906/degradation.md
# --- END COPY NOTE --------------------------------------------------------
# shellcheck disable=SC2012,SC2086,SC2206
# DO NOT kill or signal this process group from an automation/agent session.
# This endpoint serves the LLM that the automation itself runs on: a restart
# drops the in-flight requests *including the agent's own*, and takes ~5-6 min
# to come back. Restarts are operator-initiated (Ctrl-C on the tty running this
# script, or a deliberate `kill -INT -<pgid>` by a human who is watching).
# Check health without touching it:
#   curl -s -o /dev/null -w '%{http_code}\n' localhost:9000/v1/models   # 401 = up
#   tail -5 logs/vllm.log
# Serve Qwen3.8-Flash-Next-AWQ-INT4 (arch Qwen4ExpForConditionalGeneration:
# 48 layers = 36 GDN + 12 full-attn, 512 experts / top-10, PLE+ngram,
# hyper-connections, 1 MTP layer) on 4x gfx906, 32 GiB/GPU.
#
# Baseline for every change below: logs/vllm.log (boot 2026-09-19 18:32) and
# vllm.log-mtp (failed spec attempt 18:30; it and the other pre-2026-09-22 boots
# are in logs/, keyed by their old names in logs/MANIFEST.md).
# Prior copy: scripts/legacy/run-vllm.orig
#
# A/B knobs are env-overridable. Defaults: MTP k=3 + the NCCL watchdog guard --
# the validated pair (~2.0x decode throughput, §10 of
# mtp-cudagraph-profile-crash.md). Examples:
#   SPEC= ./run-vllm
#       no speculation -- the always-boots fallback (§6); the watchdog guard
#       switches itself off together with $SPEC, so you get blocking NCCL waits
#       for nothing if you leave them on.
#   NCCL_BLOCKING_WAIT=0 ./run-vllm
#       drop the guard -- KNOWN TO CRASH with MTP (§1/§7), so it warns instead
#       of silently booting a dead engine.
#   SPEC='{"method":"mtp","num_speculative_tokens":1}' ./run-vllm
#       the other legal depth at TP=4 (ladder [2,4,6]).
#   MAXLEN=262144 KVBYTES=8233762304 ./run-vllm   # full native ctx
#   TEXT_ONLY=0 ./run-vllm                        # keep the ViT
# OPTIMIZED=1 turns on the whole measured-best set with one word, and leaves the
#   defaults alone: this script's defaults are the known-good configuration, the one
#   that has survived many reboots, and nothing about that changes when the flag
#   exists. `OPTIMIZED=1 ./run-vllm` is the fast box, `./run-vllm` is the safe box,
#   and an explicit variable still beats the bundle (`OPTIMIZED=1 PLE_PREFETCH=0` is
#   the bundle minus one knob) so a single-variable A/B is still expressible. Every
#   boot writes a manifest into its own log file listing what it was started with --
#   see the OPTIMIZED BUNDLE block below for what the set is and what each part is
#   worth, and the manifest block for why it exists.
# GRAPHMODE=piecewise|eager|full picks the CUDA-graph strategy (default piecewise).
# GRAPHMODE=full is the pre-09-20 behaviour and is silently wrong for this model --
# see the GRAPHMODE block below and mtp-cudagraph-profile-crash.md §13.
# DRAFTER_GRAPHS=0 (default) is the live experiment against the boot-time GC
#   SIGSEGV: a local patch (docs/patches/drafter-graphs-off.patch) gives the MTP
#   drafter no
#   CUDA graphs while the target keeps torch.compile + graphs. DRAFTER_GRAPHS=1
#   restores upstream exactly -- i.e. the 45.8 t/s config of §10 and its ~2-in-11
#   boot odds (measured: 3 of 11 boots came up, §14.8).
# BOOT_TRIES=<n> (default 1) relaunches a boot that dies BEFORE it ever serves.
#   The MTP+graphs failure is boot-time-only -- an engine that reaches "Application
#   startup complete" has served 5h30m+ -- so retrying is currently the only way to
#   run the fast configuration: 34.5 t/s median vs 15.9 eager (§14.9). At 3/11, eight
#   attempts cost about 17 minutes. It never restarts a server that was serving, and
#   refuses to start at all while something already answers on :9000.
# VRAM_GATE_TIMEOUT=<s> (default 300) and VRAM_GATE_MAX_USED=<bytes> (default
#   2 GiB) bound the between-attempts wait for the GPUs to actually drain. A boot
#   that segfaults leaves its ~22 GiB/GPU pinned until the driver sees every fd
#   close, and a retry issued into that window fails in request_memory() instead
#   of testing anything -- see the wait_vram_released() comment for the 2026-09-21
#   log evidence. VRAM_GATE_TIMEOUT=0 makes the gate a single probe.
# DRAFTER_EAGER=1 injects "enforce_eager":true into $SPEC. It is OFF by default
#   because on this build's MTP path that flag does nothing at all -- see the
#   DRAFTER_EAGER block below for the file:line proof and the §7 false positive.
# TVMFFI=disable (default) sets TVM_FFI_DISABLE_TORCH_C_DLPACK=1 so tvm_ffi does
#   not install its JIT-built C DLPack addon on torch.Tensor. TVMFFI=fast restores
#   upstream behaviour. Both are explained next to their blocks below.
# GC_FREEZE=1 (default 0 = off) ports the V1 runner's gc.freeze()+gc.disable()
#   guard onto this V2 runner, which has no guard at all. MEASURED 2026-09-21:
#   it is not a fix, it moves the crash. Five boots ran with it enabled and five
#   died, every one inside the guard's OWN restoring gc.collect() at the end of
#   profile_run -- "GC restored" appears zero times in every log it has ever been
#   enabled in. Upstream can afford that exit traversal because it wraps CUDA
#   graph capture (seconds); this guard wraps profile_run (five minutes of
#   Dynamo/Inductor allocation churn), so the one collection guaranteed to be
#   huge is the one that runs. Default exit is now park-not-walk: freeze() again
#   on the way out, so the churn lands in the permanent generation and no later
#   collection ever traverses it. GC_THAW=1 restores upstream's unfreeze+collect
#   (the variant that died 5/5). GC_DUMP=1 re-arms faulthandler inside the region
#   -- tvm_ffi's own SIGSEGV handler prints C frames only, which is why the Python
#   frame holding the corrupted object has never been named.
# MAXSEQS defaults to 3 (not 4) because MTP's graphs leave only 3.11x
# concurrency at 131,072/req -- see the MAXSEQS note below.
# k=2 is not an option at TP=4: docs/gfx906/running.md lists only k=1 and k=3 as
# legal there, anything else raises ValueError at startup.

# Without this, `vllm | tee` always exits 0 and hides a crashed engine.
set -o pipefail

# --- LOG_DIR: where the engine's stdout+stderr go ----------------------------
# 2026-09-22: the boot logs moved out of the top level into logs/. A relative
# LOG_DIR is resolved against this script's own directory, so the engine logs in
# the same place no matter what cwd it is launched from (and no matter whether
# the script later cds). Override with LOG_DIR=/some/where.
# Historical logs keep their pre-move names; logs/MANIFEST.md maps them.
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE=$LOG_DIR/vllm.log.$(date "+%Y%m%d-%H%M%S")

case "$LOG_DIR" in
  /*) ;;
  *) LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$LOG_DIR" ;;
esac
if ! mkdir -p "$LOG_DIR"; then
  echo "run-vllm: cannot create log directory $LOG_DIR" >&2
  exit 1
fi
export LOG_DIR

export MAXLEN="${MAXLEN:-147456}"   # was an unconditional export: a caller's MAXLEN= was clobbered

# --- OPTIMIZED BUNDLE: the measured-best set, behind one flag ---------------
# The defaults of this script are the known-good configuration: whatever has
# survived many reboots on this box. Everything with only a handful of boots
# behind it stays default-off and is documented at its own block. This flag is
# the single switch for the whole fast set, so a boot can be described as
#
#   OPTIMIZED=1 ./run-vllm
#
# instead of a wall of env vars that somebody has to re-derive from the docs --
# and so "revert it" is one word missing, not a diff of four.
#
# What the bundle turns on, and what each part is worth on THIS box. Every
# number is a same-day single-variable measurement (docs/prefill-io-wall.md,
# docs/decode-perf-suggestions.md §5f/§5g, boots A'/B/C/D of 2026-09-26):
#
#   PLE_RANDOM=1     MADV_RANDOM on the 99 GiB CPU-resident ngram table, set
#                    before the table is mmap'd: 1.63x cold prefill, and 28x
#                    less NVMe per prompt token. Cannot change an output: the
#                    hint says how to fetch, never which bytes.
#   PLE_PREFETCH=1   fadvise read-ahead for the NEXT prefill chunk's ngram rows,
#                    issued while the GPU works on the current one: 1.53x on top
#                    of PLE_RANDOM => 2.49x cumulative (426 -> 1,061 tok/s cold
#                    prefill; a 35k-token prompt 82 s -> 33 s).
#   GC_FREEZE=1      gc.freeze() + shadowed gc.collect() around profile_run and
#                    capture, plus a park-not-walk exit. Reliability history,
#                    NOT a throughput knob: the old +27% was clean-vs-contended
#                    (measured null, 0.04%, across boots A and A').
#   DRAFTER_GRAPHS=1 upstream-exact CUDA graphs for the MTP drafter.
#
# Decode is unchanged by the bundle: 49.46 / 76.67 / 101.71 t/s at B=1/2/3 with
# it, 50.32 / 76.56 / 100.30 without, i.e. inside the run-to-run spread (one
# contended B=2 run on this box has read 8.20 t/s, so spread is the norm).
#
# What the bundle deliberately does NOT touch, each because it is a measured
# loss or a proven boot-killer, and each echoed in the manifest so a log reader
# can see it was considered and excluded rather than forgotten:
#
#   NCCL_PROTO        any value. LL costs a FIXED ~+634 ms per 4096-token chunk,
#                     so it is +15.5% of prefill now and got worse as prefill
#                     got faster, for a +1.4% at B=1 that did not replicate.
#   NCCL_ENVELOPE     Tree+LL+MAX_NCHANNELS=1: 99 dead boots, this RCCL rejects
#                     the config at the first collective.
#   GRAPHMODE         already piecewise by default; full is wrong for this model.
#   BOOT_TRIES        already 1. The bundle does not make boots more retry-happy.
#
# Boot reliability of the bundle: 4 of 4 first-try on 2026-09-26. The historical
# 3-of-11 for MTP+graphs (mtp-cudagraph-profile-crash.md §14.8) predates the
# park-not-walk exit in the GC_FREEZE guard, so 4/4 is the current sample and it
# is thin. That thinness is exactly why this is a flag and not the default.
# BEGIN OPTIMIZED BUNDLE (markers: scripts/test-optimized-bundle.sh extracts this block)
OPT_BUNDLE=(PLE_RANDOM PLE_PREFETCH GC_FREEZE DRAFTER_GRAPHS)
declare -A OPT_SRC=()
OPTIMIZED="${OPTIMIZED:-0}"
case "$OPTIMIZED" in
  0|1) ;;
  *)
    printf 'run-vllm: OPTIMIZED=%s is not 0 or 1 -- using the known-good defaults (OPTIMIZED=0)\n' \
      "$OPTIMIZED" >&2
    OPTIMIZED=0
    ;;
esac
for _k in "${OPT_BUNDLE[@]}"; do
  if [ -n "${!_k+set}" ]; then
    # Set by the operator, including set-but-empty: it beats the bundle, and a
    # bare `PLE_PREFETCH= ./run-vllm` must not silently become 1.
    OPT_SRC[$_k]="explicit(${!_k:-<empty>})"
  elif [ "$OPTIMIZED" = 1 ]; then
    export "${_k}=1"
    OPT_SRC[$_k]=bundle
  else
    OPT_SRC[$_k]="default-off"
  fi
done
unset _k
# END OPTIMIZED BUNDLE

# SPEC is resolved near SPEC_DEFAULT below and passed as --speculative-config.
# It must NOT be assigned here: an unconditional `export SPEC=` clobbers a
# caller's `SPEC='{...}' ./run-vllm`, and ${SPEC-$SPEC_DEFAULT} then sees it as
# "set but empty", so the default never applies either. That is exactly how the
# 23:20 MTP test boot ran with speculative_config=None while the operator
# believed MTP was on (same failure class as the old `export TEXT_ONLY=0` line).

# MTP watchdog guard (mtp-cudagraph-profile-crash.md §7). ON by default
# whenever MTP is on -- the decision is taken below, next to $SPEC, because it
# has to follow $SPEC's value. Override: NCCL_BLOCKING_WAIT=0 (skip the guard,
# known to crash with MTP) or =1 (force it on, even without MTP).
#
# Why it exists: when PyTorch creates a NCCL/RCCL process group it also starts
# a watchdog thread that polls every queued collective with hipEventQuery.
# While the MTP speculator captures its CUDA graphs, that poll hits an event
# last recorded on the *capturing* stream, and ROCm answers
# hipErrorCapturedEvent ("operation not permitted on an event last recorded in
# a capturing stream"). c10/hip/HIPEvent.h:112 (Event::query) hard-checks
# anything other than hipErrorNotReady, and hipErrorCapturedEvent is handled
# nowhere in c10/ or c10d/, so the exception escapes the watchdog thread ->
# std::terminate -> all 4 TP workers abort ~1s after "Capturing model for
# speculator..." (observed 22:30:12, all 4 ranks, PG ID 2 = a tp/ep PYNCCL
# group).
#
# TORCH_NCCL_BLOCKING_WAIT=1 makes ProcessGroupNCCL skip watchdog_->start()
# (pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:909 + :974-981,
# "NO watchdog thread is created"), so the illegal query is never issued.
#
# Cost: NCCL work completion becomes blocking (slower startup and steady state)
# and NCCL hang detection / flight dumps are gone -- which is why the guard is
# tied to $SPEC rather than unconditional: with no drafter there is no
# speculator capture, no illegal query is ever issued, and the cost buys nothing.
# The guard does not fix the underlying defect (the speculator still enqueues
# collectives inside its capture region), so the 19:44-style gc.collect()
# segfault (§1) can in principle resurface; it did not in the 23:48 boot (§10).

export HIP_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export VLLM_API_KEY="${VLLM_API_KEY:?set VLLM_API_KEY in the environment - this copy ships without the key}"

# --- NCCL/RCCL envelope: DEFAULT OFF, DO NOT re-enable Tree+LL here -----------
# History: added 2026-09-22 as a "free ~5 ms/step" lever, copied from
# joe2gaan's TP4 profiles (/ai/engines/gfx906/localaiservers/qwen36-gfx906/
# deploy.sh:32-38, vnext-launch-runs/hf-moe122b-tp4/vllm_command.sh:4-7),
# which run NCCL_ALGO=Tree NCCL_PROTO=LL NCCL_MAX_NCHANNELS=1 and measured
# NCCL_PROTO=auto as ~2x slower on decode-sized messages (0.0957 vs
# 0.0462 ms at 1x5120).
#
# It does not work on this build. Every boot died at profile_run (99 of them
# on 2026-09-23, logs/vllm.log), with a hard RCCL config error rather than a
# slowdown:
#
#   torch.distributed.DistBackendError: NCCL error in
#     torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:3780, invalid usage
#   Error : no algorithm/protocol available for function AllGather with
#           datatype ncclInt8. NCCL_ALGO was set to Tree. NCCL_PROTO was set
#           to LL.
#     vllm/distributed/parallel_state.py:220 all_gather
#     vllm/distributed/parallel_state.py:750 _all_gather_out_place
#     .../device_communicators/cuda_communicator.py:382 all_gather
#
# Root cause: pinning ALGO *and* PROTO removes RCCL's fallback, and Tree+LL
# has no valid pairing for AllGather of int8 -- and torch maps bool tensors to
# ncclInt8, so this is hit by an ordinary bool all_gather on the MTP drafter's
# prefill path (speculator.propose -> autoregressive _prefill -> _run_model),
# i.e. it is unavoidable, not a rare op. Their config gets away with it because
# a TP2 dense 27B issues no bool/int8 allgather; ours does on every drafter
# step. Retrying cannot help, which is why the box "would not start again":
# the wrapper faithfully re-ran a configuration RCCL rejects (the 06:45 boot
# alone consumed ~50 min of BOOT_TRIES before NCCL_ENVELOPE=0).
#
# MEASURED 2026-09-25 (docs/decode-perf-suggestions.md 5d-5f), which corrects the
# attribution above: the killer was NCCL_ALGO=Tree ALONE, not the pair. 'strings
# -a /opt/rocm/lib/librccl.so.1 | grep ^ncclDevFunc_' shows AllGather has NO Tree
# kernel in any protocol, so Tree deletes every algo that can do AllGather;
# PROTO=LL was named in the rejection as a witness, not a cause. NCCL_PROTO=LL
# alone BOOTS (verified 21:42) but is a NET LOSS -- prefill -10% (494 -> 444
# tok/s, a 21 MB per-AR collective at 4096 tokens is 3.7x slower on LL) against
# decode +5% at B=1 and 0% at B>=2. Do not default it on.
#
# The remaining knob, NCCL_MAX_NCHANNELS=1, was never isolated and may be the
# whole benefit (and the whole risk: it also caps prefill bandwidth). If this
# is ever revisited, do it ONE variable at a time and with a throwaway boot:
#   NCCL_ENVELOPE=1 NCCL_CHANNELS_ONLY=1 ./run-vllm
if [ "${NCCL_ENVELOPE:-0}" = 1 ]; then
  if [ "${NCCL_CHANNELS_ONLY:-0}" = 1 ]; then
    : "${NCCL_MAX_NCHANNELS:=1}"
    export NCCL_MAX_NCHANNELS
    echo "run-vllm: NCCL channels-only max_nchannels=$NCCL_MAX_NCHANNELS"
  else
    : "${NCCL_ALGO:=Tree}"
    : "${NCCL_PROTO:=LL}"
    : "${NCCL_MAX_NCHANNELS:=1}"
    export NCCL_ALGO NCCL_PROTO NCCL_MAX_NCHANNELS
    echo "run-vllm: NCCL envelope algo=$NCCL_ALGO proto=$NCCL_PROTO max_nchannels=$NCCL_MAX_NCHANNELS -- KNOWN BAD on this build (RCCL: no algo/proto for AllGather/ncclInt8); boots die at profile_run"
  fi
else
  # An inherited NCCL_* is NOT "the RCCL default", and saying so would put a
  # falsehood in the boot log: boot D runs NCCL_PROTO=LL with NCCL_ENVELOPE=0.
  # Report what the calling shell handed us, so the log matches the process.
  if [ -n "${NCCL_PROTO:-}${NCCL_ALGO:-}${NCCL_MAX_NCHANNELS:-}" ]; then
    echo "run-vllm: NCCL INHERITED from the calling shell: algo=${NCCL_ALGO:-<unset>} proto=${NCCL_PROTO:-<unset>} max_nchannels=${NCCL_MAX_NCHANNELS:-<unset>} (NCCL_ENVELOPE=0, so these were not set here)"
  else
    echo "run-vllm: NCCL left at RCCL defaults (NCCL_ENVELOPE=0; Tree+LL rejected by this RCCL, see comment above)"
  fi
fi

export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export LD_LIBRARY_PATH=/ai/engines/gfx906/unverbraucht/rccl/build:/opt/rocm-7.2.0/lib:$LD_LIBRARY_PATH
export PYTORCH_ROCM_ARCH=gfx906
export ROCM_HOME=/opt/rocm-7.2.0
export ROCM_PATH=/opt/rocm-7.2.0
export USE_NUMPY=1
export USE_ROCM=1

# Removed from the previous version:
#   FORCE_CUDA=1          -> made the engine see CUDA_VISIBLE_DEVICES, which
#                            platforms/rocm.py:_sync_hip_cuda_env_vars then
#                            warns about ("...deprecated... use
#                            HIP_VISIBLE_DEVICES instead") every few seconds.
#                            HIP_VISIBLE_DEVICES above is the only one needed.
#   VLLM_ATTENTION_BACKEND -> "Unknown vLLM environment variable" (the real
#                            knobs are --gdn-prefill-backend /
#                            --kda-prefill-backend / --mm-encoder-attn-backend)
#   VLLM_FP8_DISABLED      -> "Unknown vLLM environment variable"
#
# They can still arrive from the *calling* shell: the 23:48 boot's own
# environment (/proc/619441/environ) contained FORCE_CUDA=1,
# VLLM_ATTENTION_BACKEND=EAGER and VLLM_FP8_DISABLED=1 -- exported in an
# interactive session, hence invisible in rc files and in the login shell's
# /proc/environ (a login-time snapshot). Visible effect: rocm.py:135 "Using
# CUDA_VISIBLE_DEVICES on ROCm is deprecated" spam plus three envs.py:2308
# "Unknown vLLM environment variable" warnings. VLLM_ATTENTION_BACKEND=EAGER is
# harmless only because that name is unknown *today*; if it ever becomes a real
# knob it would force eager attention. CUDA_VISIBLE_DEVICES goes too, since
# platforms/rocm.py:_sync_hip_cuda_env_vars mirrors FORCE_CUDA into it.
unset FORCE_CUDA CUDA_VISIBLE_DEVICES VLLM_ATTENTION_BACKEND VLLM_FP8_DISABLED

# Consumed by docs/gfx906/gfx906-blocking-sync.pth (must be installed into
# site-packages once, see the note at the bottom of this file). Sets HIP to
# blocking-sync before the first HIP queue is created, so TP workers stop
# busy-waiting (measured here without it: ~190% CPU per Worker_TP* = ~2 of 64
# host cores burned per worker, for the life of the process).
export VLLM_GFX906_HIP_LIB_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so.7
# export VLLM_GFX906_HIP_BLOCKING_SYNC=0   # opt out (lowest dispatch latency)

TP=4
# 131072 = the validated context ceiling on this model (oom-256k-prefill.md);
# 122880 is the production context the MTP depth-matrix was measured at and is
# also fine for the 4-slot target (4.7x vs 4.4x on the 575,345-token pool).
# The baseline ran the native 262144, which the
# engine sized at "Maximum concurrency for 262,144 tokens per request: 2.19x"
# on a 575,345-token KV pool -- so --max-num-seqs 4 was unreachable and the
# cudagraph size 8 could never fire. Observed KV use was 1.9-13.7%, so the
# native context cost nothing but scheduling headroom. Do NOT add YaRN:
# ROADMAP.md QSA-FN-2 says rope scaling degrades quality at all positions.
MAXLEN="${MAXLEN:-131072}"
# 3, not 4. With MTP k=3 on, the drafter's graphs cost 0.78 + 0.64 = 1.42 GiB/GPU
# (spec-free: 0.51), which leaves "Available KV cache memory: 6.06 GiB" =
# 407,385 tokens = "Maximum concurrency for 131,072 tokens per request: 3.11x",
# so a 4th slot is not reachable at the full context (mtp-cudagraph-profile-crash.md
# §10, boot 23:48). Consequence: the k=3 ladder becomes [4,8,12] and a 16-token
# (4-request) decode step falls back to eager. Raise back to 4 only together with
# a lower MAXLEN (122,880 -> 4.7x) or a KVBYTES pin; the spec-free config is
# unaffected either way (its ladder is [1,2,3,4] and KV was 4.7x there).
MAXSEQS="${MAXSEQS:-3}"
# Also the compile range endpoint (compile_ranges_endpoints: [4096]); changing
# it invalidates the AOT cache that currently reloads the graph in ~5-10s.
MBT="${MBT:-4096}"
# Unset by default: let vllm size the pool from --gpu-memory-utilization.
# Pin it with the numbers the engine printed for this exact boot, e.g.
#   KVBYTES=7003153511  # 6.52 GiB, fits inside --gpu-memory-utilization 0.95
#   KVBYTES=8233762304  # 7.67 GiB, fully utilizes the 31.98 GiB device
KVBYTES="${KVBYTES:-}"
# MTP k=3 is the default: ~2.0x decode throughput (median 45.0 vs 22.5 t/s
# spec-free, acceptance length ~2.7, §10). It is only safe paired with the
# watchdog guard near the top of this file -- without it the speculator's
# CUDA-graph capture is fatal: GC segfault in all 4 TP workers with a cold
# torch.compile cache (§1), NCCL-watchdog abort on hipErrorCapturedEvent with a
# warm one (§7). So the guard defaults to on exactly when $SPEC is non-empty.
# Consequence of this default: 1.42 GiB/GPU of captured graphs, hence MAXSEQS 3
# above rather than 4. Verify every MTP boot in the log for
# "Capturing model for speculator..." (speculator.py:151) followed by a second
# "Graph capturing finished" -- a boot without that line proves nothing about
# MTP (the 23:20 false positive, §9: `export SPEC=` used to clobber this very
# override).
SPEC_DEFAULT='{"method":"mtp","num_speculative_tokens":3}'
SPEC="${SPEC-$SPEC_DEFAULT}"
TEXT_ONLY="${TEXT_ONLY:-1}"

# Apply the watchdog guard (rationale: block near the top of this file).
if [ -z "${NCCL_BLOCKING_WAIT-}" ]; then
  if [ -n "$SPEC" ]; then NCCL_BLOCKING_WAIT=1; else NCCL_BLOCKING_WAIT=0; fi
fi
if [ "$NCCL_BLOCKING_WAIT" = 1 ]; then
  export TORCH_NCCL_BLOCKING_WAIT=1
elif [ -n "$SPEC" ]; then
  echo "run-vllm: WARNING: MTP ($SPEC) with NCCL_BLOCKING_WAIT=0 is a known" \
       "crashing combination on this build: the speculator's graph capture" \
       "kills all 4 TP workers at boot" \
       "(mtp-cudagraph-profile-crash.md §1/§7)." >&2
fi
SPEC="${SPEC-$SPEC_DEFAULT}"
TEXT_ONLY="${TEXT_ONLY:-1}"

# --- DRAFTER_GRAPHS: the drafter gets no CUDA graphs (local patch) ----------
# The experiment that replaces the dead "enforce_eager" flag below. Patched
# into v1/worker/gpu/model_runner.py:665-681 in BOTH trees (backups: *.pre-drafter-
# graphs.bak, diff: docs/patches/drafter-graphs-off.patch). It discriminates the two surviving
# suspects for the boot-time SIGSEGV in the cyclic GC (report §13.4 + §14):
# "the drafter captures graphs" vs "a second AOT artifact is loaded at all" --
# both artifacts are on disk as torch_compile_cache/<hash>/rank_N_0/
# {backbone,eagle_head}. In PIECEWISE mode the drafter's *decode* manager was
# already NONE (autoregressive/speculator.py:137-140); what this switch removes
# is the prefill capture that takes the target's mode verbatim (:129-134), which
# is the capture that was running when the 20:17 and 20:22 boots died.
# The target model is untouched: same compile, same graphs, same KV sizing.
# DRAFTER_GRAPHS=1 = upstream behaviour = the fast-but-flaky config of §10.
export GFX906_DRAFTER_GRAPHS="${DRAFTER_GRAPHS:-0}"
if [ "$GFX906_DRAFTER_GRAPHS" = 1 ]; then
    echo "run-vllm: DRAFTER_GRAPHS=1 -> drafter CUDA graphs RESTORED (upstream)"
else
    echo "run-vllm: drafter CUDA graphs OFF (model_runner.py local patch)"
fi

# --- DRAFTER_EAGER: a flag that does nothing here, kept as a trap warning ---
# "enforce_eager" inside --speculative-config is declared at
# config/speculative.py:377 (docstring: "Override the default enforce_eager from
# model_config") and it is read in exactly two places, both in the OLD proposer:
# v1/spec_decode/llm_base_proposer.py:451 and v1/spec_decode/extract_hidden_states.py:249.
# This build does not run that path for MTP -- every crash log we have points at
# v1/worker/gpu/spec_decode/autoregressive/speculator.py:151 -- and grep finds
# zero reads of speculative_config.enforce_eager anywhere under
# v1/worker/gpu/spec_decode/. Nor does anything propagate it to the draft
# ModelConfig: speculative.py:1258 hardcodes
# enforce_eager=self.target_model_config.enforce_eager.
# That is exactly why the 09-19 22:25 boot (§7) passed the flag, still logged
# "Capturing model for speculator...", and died in the NCCL watchdog: the run
# tested nothing (same mistake class as §9). Hence default OFF -- injecting it by
# default would make an ordinary boot look like a test of something.
# DRAFTER_EAGER=1 still injects it, for use against builds that do take the old
# path; the real lever here is DRAFTER_GRAPHS above.
# --- GC_FREEZE: port V1's gc.freeze()+gc.disable() guard to the V2 runner ------
# The V1 runner wraps CUDA graph capture in _freeze_gc()
# (v1/worker/gpu_model_runner.py:6617, gated by VLLM_ENABLE_CUDAGRAPH_GC). The V2
# runner that actually loads here (every boot logs "Using V2 Model Runner") has
# THREE bare gc.collect() calls and no guard at all, and both of our boot-time
# SIGSEGVs die inside one of them: profile_run()'s (reached from
# determine_available_memory() -- the 20:17/20:22 deaths) and capture_model()'s
# (the 19:44 death). So VLLM_ENABLE_CUDAGRAPH_GC is a NO-OP on this box (it gates
# only the V1 path) and the protection V1 has was simply never ported.
# docs/patches/gc-freeze-capture.patch ports it, env-gated; this variable is that gate. It is
# OFF by default ON PURPOSE, and since 2026-09-21 there is a harder reason than
# attribution: with the guard enabled, five of five boots died at the guard's own
# exit collection (vllm.log 22:00:45/22:06:43/22:12:53, vllm.log2 20:52:48,
# vllm.log.error4 20:39:27 -- identical stack, builtin_next/gen_iternext/
# gen_send_ex2 = contextmanager.__exit__ resuming _gc_freeze_guard, and the
# "GC restored" log line after that collect has never printed anywhere). Warm vs
# cold AOT cache made no difference. Freezing does not repair the corruptor; the
# only variant that cannot die is the one that never traverses, which is now the
# default exit path (see GFX906_GC_THAW). Skipped collects bias the KV memory
# estimate LOW (unreclaimed garbage counts as used) = smaller cache = the safe
# direction. Report sections 14-16.
export GFX906_GC_FREEZE="${GC_FREEZE:-0}"
export GFX906_GC_THAW="${GC_THAW:-0}"
export GFX906_GC_DUMP="${GC_DUMP:-0}"
if [ "$GFX906_GC_FREEZE" = 1 ]; then
    echo "run-vllm: GC_FREEZE=1 -> gc.freeze()+gc.disable()+shadowed gc.collect()" >&2
    echo "          around profile_run and capture_model; on exit:" >&2
    if [ "$GFX906_GC_THAW" = 1 ]; then
        echo "          GC_THAW=1 = upstream unfreeze()+collect() = the variant" >&2
        echo "          that segfaulted on all five boots that used it" >&2
    else
        echo "          GC_THAW=0 = re-freeze and leave it: nothing is ever" >&2
        echo "          traversed (boot-time garbage is never reclaimed either)" >&2
    fi
    if [ "$GFX906_GC_DUMP" = 1 ]; then
        echo "run-vllm: GC_DUMP=1 -> faulthandler re-armed inside the regions, so a" >&2
        echo "          segfault prints Python frames, not only tvm_ffi's C ones" >&2
    fi
fi

if [ -n "$SPEC" ] && [ "${DRAFTER_EAGER:-0}" = 1 ] \
   && ! printf '%s' "$SPEC" | grep -q enforce_eager; then
    SPEC=$(printf '%s' "$SPEC" | sed 's/^{/{"enforce_eager":true,/')
    echo "run-vllm: DRAFTER_EAGER=1 -> injected $SPEC (NOTE: no effect on this"
    echo "          build's MTP path -- see the block above; use DRAFTER_GRAPHS)" >&2
fi

# --- PLE_RANDOM: read-ahead on the CPU-resident ngram table -----------------
# The PLE ngram table is ~99 GiB of mmap'd safetensors (22 shards, 128 sub-shards,
# 320 B rows) gathered at RANDOM, i.e. far larger than this node's ~50 GiB of page
# cache, so every lookup is a cold random fault. The kernel reads a 128 KiB window
# around each faulted page and the next touch is in a different shard, so the
# window is discarded: measured **1.03 MB of NVMe per prefill token at ~28x read
# amplification**, with all 4 GPUs at 0% busy for ~18 of every 29 sampled seconds
# of a cold prefill (docs/prefill-io-wall.md).
#
# MADV_RANDOM sets VM_RAND_READ so do_sync_mmap_readahead() fetches the single
# faulted page. Measured on a real shard, cold cache, 300 scattered touches:
# 32.3 -> 1.0 MiB read (27.6x -> 0.9x), 43 -> 24 ms. It changes how many bytes the
# kernel fetches, never which bytes, so it cannot alter model output.
#
# It must be decided BEFORE the load: file_ra_state.ra_pages is latched at open(),
# so writing /sys/block/*/queue/read_ahead_kb at a live engine does nothing to an
# already-mmap'd table (that null result is what the first attempt measured).
#
#   PLE_RANDOM=1 ./run-vllm    opt in
#   ./run-vllm                 DEFAULT OFF = kernel read-ahead, the known-good path
export VLLM_GFX906_PLE_MADV_RANDOM="${PLE_RANDOM:-0}"
if [ "$VLLM_GFX906_PLE_MADV_RANDOM" = 1 ]; then
    echo "run-vllm: PLE_RANDOM=1 -> MADV_RANDOM on the PLE ngram table (read-ahead" >&2
    echo "          suppressed; ~28x less NVMe per prefill token, output unchanged)" >&2
else
    echo "run-vllm: PLE_RANDOM=0 -> kernel default read-ahead on the PLE table" >&2
    echo "          (known-good; ~1.03 MB NVMe per prefill token -- see PLE_RANDOM)" >&2
fi

# --- PLE_PREFETCH: read AHEAD for the same table ----------------------------
# PLE_RANDOM above removes most of the *volume*; what is left is *latency*: the
# gather is 16-65k independent 4 KiB faults per prefill chunk, issued by the one
# host thread that is also the critical path, at PLE layer 2 of 48 -- so the GPUs
# idle behind it (0% busy for ~62% of a cold prefill's wall clock).
#
# With this on, a worker thread computes the ngram ids for the NEXT prefill chunk
# on the host (same _compute_ngram_ids, CPU mirrors, so it cannot drift from the
# device path), resolves each shard's rows to file offsets via /proc/self/maps,
# and issues POSIX_FADV_WILLNEED for those page runs while the GPU works on the
# current chunk. fadvise is an asynchronous hint: the issuing thread never blocks
# on the device and never carries the GIL across an NVMe wait.
#
# Measured (scripts/probe-ple-prefetch.py, cold cache, subprocess per arm, rotated
# order, real shard, 1024 tokens x 16 ids, ON TOP OF PLE_RANDOM=1):
#   serial 77.2 ms | gather-ahead thread 76.5 ms (no win; 0.89x = a regression
#   without MADV_RANDOM) | fadvise-ahead 66.0 ms = 1.17x, no extra traffic.
#
# It is a hint about a file, not a cache of results: there is no key to get wrong
# and no output to corrupt. Any failure disables it for the life of the process
# and logs once. Graph capture is skipped, so boot does not hint dummy pages.
#
#   PLE_PREFETCH=1 ./run-vllm  opt in (use WITH PLE_RANDOM=1; that is what was
#                              measured -- the two are complementary, not equal)
#   ./run-vllm                 DEFAULT OFF = gather stays on the critical path
export VLLM_GFX906_PLE_PREFETCH="${PLE_PREFETCH:-0}"
if [ "$VLLM_GFX906_PLE_PREFETCH" = 1 ]; then
    echo "run-vllm: PLE_PREFETCH=1 -> fadvise read-ahead for the next prefill" >&2
    echo "          chunk's ngram rows (+17% measured on PLE_RANDOM alone)" >&2
    if [ "$VLLM_GFX906_PLE_MADV_RANDOM" != 1 ]; then
        echo "run-vllm: NOTE PLE_PREFETCH=1 without PLE_RANDOM=1 -- the +17% was" >&2
        echo "          measured with MADV_RANDOM on; alone it is not validated" >&2
    fi
else
    echo "run-vllm: PLE_PREFETCH=0 -> no read-ahead (known-good path)" >&2
fi

# --- TVMFFI: tvm_ffi's torch C-DLPack fast path -----------------------------
# apache-tvm-ffi 0.1.10 is imported in every TP worker (libtvm_ffi.so is mapped
# into the live worker, cf. /proc/<worker>/maps) and tvm_ffi/__init__.py:90 runs
# _optional_torch_c_dlpack at import time. On ROCm ONLY -- prefer_rocm_override,
# _optional_torch_c_dlpack.py:97-98, which deliberately skips the prebuilt
# torch_c_dlpack_ext 0.1.5 wheel that is installed here -- it ctypes.CDLL()s
#   /root/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so
# and installs that library's C function table as the torch.Tensor class
# attribute __dlpack_c_exchange_api__ (same file:169-176). The addon has an
# undefined symbol THPVariable_Wrap(at::TensorBase const&): it creates and wraps
# tensor PyObjects from C++, and torch.Tensor is exactly the kind of
# GC-traversable object our crash walks (update_refs -> _PyGCHead_NEXT).
# PROVEN: it is loaded; it patches torch.Tensor; its cache key is
#   `torch211-rocm` with no git hash, and the addon on disk (built 09-18 20:44)
#   predates the installed torch (09-19 06:44: dist-info + 115 files newer).
# NOT PROVEN: that it is the corruptor. libtvm_ffi.so is mapped in the eager
#   boots that never crash too, so mere presence is not the differentiator, and
#   a CPU-only stress (same addon, 200x300 DLPack round-trips + gc.collect per
#   iteration, /tmp/gc_dlpack_stress.py) did not reproduce. Cheap hypothesis
#   test, not a fix.
# WHY DISABLING COSTS NOTHING (measured 09-20 late, /tmp/dlpack_force.py):
#   torch 2.11.0a0 already defines torch.Tensor.__dlpack_c_exchange_api__ itself.
#   On non-ROCm builds tvm_ffi sees that and returns without doing anything
#   (_optional_torch_c_dlpack.py:98); only the ROCm branch forces the replacement.
#   Forcing that branch in-process (patch torch.cuda.is_available, no GPU context
#   needed) visibly swaps the pointer: 0x7ff25f580c20 inside libtorch ->
#   0x7ff25cb50800 inside the cached .so. So TVMFFI=disable does not remove a
#   feature, it declines an unnecessary downgrade of torch's own implementation
#   with a cached one of unverifiable provenance. The prebuilt wheel cannot be
#   the fallback either: torch_c_dlpack_ext 0.1.5 ships addons for torch 2.4-2.9
#   cpu/cuda only -- no rocm variant, nothing for 2.11 -- which is why the ROCm
#   branch is the JIT path by construction.
#   Against it: the installed struct is not ABI-incompatible in any way we can
#   see -- the header qword matches torch's (0x300000001) and both tables expose
#   the same 5 function pointers; and 10,000 DLPack round-trips driven *through
#   the replaced API* with a gc.collect() per round still did not crash. A CPU
#   probe cannot reach the graph/CUDA paths, so local repro is out of reach here:
#   the remaining informational step is a PYTHONMALLOC=debug boot (report §14.11).
#   WHAT SETTLED IT: upstream classified this ROCm override as a BUG and removed it in
#   tvm-ffi 0.1.12 (PR #585, "Prefer upstream PyTorch DLPack API in torch extension
#   loader"). Verified from the wheel, not the changelog: 0.1.12's copy of this file has
#   no prefer_rocm_override anywhere and its early return is unconditional
#   (`if _check_and_update_dlpack_c_exchange_api(torch.Tensor): return None`), so on our
#   torch it returns before ever dlopen()ing the addon. Our import-time env var is
#   therefore a functional backport of that fix onto 0.1.10 -- a bug workaround, not a
#   lost feature. Still NOT a proven fix for THIS segfault.
#   If the package is ever upgraded, the target is 0.1.12, NOT 0.1.14.post0: 0.1.12 has
#   #585 and predates tvm-ffi#697 (0.1.13 segfaults in TVMFFIEnvRegisterCAPI *via
#   xgrammar* -- and xgrammar 0.2.7 is what actually imports tvm_ffi here, mapped 5x in
#   every worker alongside libtvm_ffi 10x; torch itself references tvm_ffi zero times).
#   Constraints: xgrammar wants >=0.1.10 (no upper bound), tilelang >=0.1.10,~=0.1.0 --
#   only vllm pins ==0.1.10, so vLLM's own metadata is the sole blocker.
#   Verdict: principled default, zero measured cost, NOT a proven fix.
#
#   tvm_ffi is also the thing that PRINTS our crash: its
#   src/ffi/backtrace.cc:168 does a bare std::signal(SIGSEGV, ...), which is why
#   every crash we have ever read is a C-only backtrace with no Python stack --
#   it overwrites faulthandler's handler and there is no env knob to stop it.
# TVMFFI=fast restores upstream behaviour (addon installed, warning kept).
TVMFFI="${TVMFFI:-disable}"
if [ "$TVMFFI" != fast ]; then
    export TVM_FFI_DISABLE_TORCH_C_DLPACK=1   # real knob: that file:206
fi
_addon=$(ls -t /root/.cache/tvm-ffi/libtorch_c_dlpack_addon_*.so 2>/dev/null | head -1)
_torch_di=$(ls -d /root/.pyenv/versions/venv312/lib/python3.12/site-packages/torch-*.dist-info 2>/dev/null | head -1)
if [ -n "$_addon" ] && [ -n "$_torch_di" ] && [ "$_addon" -ot "$_torch_di" ]; then
    echo "run-vllm: NOTE: $_addon predates the installed torch; tvm_ffi picks it"
    echo "          by filename alone (no git hash) -- rm it to force a rebuild." >&2
fi

# Capture ladder: multiples of k+1 up to MAXSEQS*(k+1) (house rule, see
# docs/gfx906/_serve_qsa_flash_gfx906.sh). k=3 with MAXSEQS=3 -> [4,8,12];
# spec-free (step=1) -> [1,2,3]. Sizes above the ladder run eager.
if [ -n "$SPEC" ]; then
    k=$(printf '%s' "$SPEC" | sed -n 's/.*num_speculative_tokens"*: *\([0-9]*\).*/\1/p')
    step=$(( ${k:-3} + 1 ))
else
    step=1
fi
sizes=$(seq -s, "$step" "$step" $((MAXSEQS * step)))

# CUDA-graph mode.  Default is `piecewise`, not `full`, because of the PLE ngram
# embedding op (vllm::qwen4_exp_amd_ple_ngram_embedding, qwen4_exp/amd/ple_layer.py):
# its body does a device->host copy of the ngram ids and then gathers rows out of
# 128 mmap'd CPU tensors in Python.  A CUDA graph can replay the two memcpys and
# nothing at all for the host gather, so every graph that captures this op replays
# the capture-time PLE row forever -- silently wrong output (report §13.2, and the
# state this box served in on 09-20 14:52).  Two things keep the op eager:
#   * splitting_ops lists it.  A user-supplied list REPLACES the default
#     (config/compilation.py:1147-1163), so all 20 entries go on the command line:
#     the 19 the engine resolves by itself (17 from _attention_ops + the 2
#     kv-cache update ops that use_inductor_graph_partition=False appends) plus PLE.
#   * cudagraph_mode 1 (PIECEWISE) instead of (2,1)=FULL_AND_PIECEWISE: the decode
#     path's FULL graph is captured *outside* the piecewise fx structure
#     (config/compilation.py:1155-1162), so splitting_ops alone cannot keep an op
#     out of a full decode graph.
#   piecewise -> mode 3 + cudagraph_mode 1 + 20-entry splitting_ops  (the fix)
#   eager     -> --enforce-eager on top; known-good with MTP (report §13.4: 2/2)
#   full      -> pre-09-20 behaviour: frozen PLE output, and with
#                docs/patches/ple-ngram-race.patch applied it does not boot at all
#                (hipErrorStreamCaptureUnsupported, report §13.1)
GRAPHMODE="${GRAPHMODE:-piecewise}"
SPLITTING_OPS=$(cat <<'JSON' | tr -d '\n '   # flattened: the argv stays one line, so `ps`/`pgrep -af` stay greppable
[
    "vllm::unified_attention_with_output",
    "vllm::unified_mla_attention_with_output",
    "vllm::mamba_mixer2",
    "vllm::mamba_mixer",
    "vllm::short_conv",
    "vllm::qwen4_exp_compute_ple_ngram_ids",
    "vllm::qwen4_exp_ple_short_conv",
    "vllm::qwen4_exp_qsa_with_output",
    "vllm::linear_attention",
    "vllm::qwen_gdn_attention_core",
    "vllm::qwen_gdn_attention_core_fused_norm_packed",
    "vllm::gdn_attention_core_xpu",
    "vllm::olmo_hybrid_gdn_full_forward",
    "vllm::sparse_attn_indexer",
    "vllm::rocm_aiter_sparse_attn_indexer",
    "vllm::deepseek_v4_attention",
    "vllm::hpc_rope_norm_forward",
    "vllm::unified_kv_cache_update",
    "vllm::unified_mla_kv_cache_update",
    "vllm::qwen4_exp_amd_ple_ngram_embedding"
]
JSON
)
case "$GRAPHMODE" in
    piecewise)
        comp_cfg="{\"mode\": 3, \"cudagraph_mode\": 1, \"cudagraph_capture_sizes\": [$sizes], \"splitting_ops\": $SPLITTING_OPS}"
        graph_args=() ;;
    eager)
        comp_cfg="{\"mode\": 3, \"cudagraph_capture_sizes\": [$sizes]}"
        graph_args=(--enforce-eager) ;;
    full)
        comp_cfg="{\"mode\": 3, \"cudagraph_capture_sizes\": [$sizes]}"
        graph_args=() ;;
    *)
        printf 'GRAPHMODE must be piecewise|eager|full (got %s)\n' "$GRAPHMODE" >&2
        exit 2 ;;
esac
# `full` with the PLE race patch applied is a guaranteed boot failure, so say so
# before the 5-minute wait.  Globbed path: does not depend on the package name.
PLE_PY=$(ls /root/.pyenv/versions/venv312/lib/python3.12/site-packages/vl*/models/qwen4_exp/amd/ple_layer.py 2>/dev/null | head -1)
if [ "$GRAPHMODE" = full ] && [ -n "$PLE_PY" ] && grep -q 'copy_(ngram_ids, non_blocking=False)' "$PLE_PY"; then
    printf 'WARNING GRAPHMODE=full with the PLE ngram-race patch applied: capture_model dies with hipErrorStreamCaptureUnsupported at %s (report §13.1). Use GRAPHMODE=piecewise, or restore %s.pre-ngram-race.bak.\n' "$PLE_PY" "$PLE_PY" >&2
fi

# JIT monitor knobs (docs/host-launch-overhead.md section 2).
# The engine already warns when a Triton kernel compiles mid-traffic, but the
# warning names only the kernel -- not the config that missed. JIT_VERBOSE=1 adds
# --jit-monitor-verbose, which logs constexprs / signature / specialization key
# per late compile; that key list is the input for widening warmup, so collect it
# on a throwaway boot BEFORE changing any warmup code.
# JIT_MONITOR=error makes any late compile abort the boot instead of spiking one
# request: the way to enumerate every miss without waiting for traffic to reach
# each code path. Costs a boot per miss, so use it for a deliberate sweep only.
jit_args=()
if [ "${JIT_VERBOSE:-0}" = 1 ]; then
  jit_args+=(--jit-monitor-verbose)
  echo "run-vllm: JIT_VERBOSE=1 -- late Triton compiles will log their constexprs/signature/key"
fi
if [ "${JIT_MONITOR:-warn}" != warn ]; then
  jit_args+=(--jit-monitor-mode "$JIT_MONITOR")
  echo "run-vllm: JIT_MONITOR=$JIT_MONITOR -- a late compile is fatal (sweep mode)"
fi

# --- PREFLIGHT: validate the NCCL/RCCL environment BEFORE the model load ----
# Why: a boot spends ~6 minutes loading four AWQ shards before it touches a
# collective, so an environment that RCCL rejects costs 6 minutes per attempt.
# On 2026-09-23 the borrowed Tree+LL envelope cost 99 dead boots that way; the
# rejection ("no algorithm/protocol available for function AllGather with
# datatype ncclInt8") was visible in the first second of the first collective.
# scripts/preflight-nccl.py forms a 4-rank group with no model and runs the
# collectives THIS engine issues -- including the drafter's bool all-gather,
# which is the one the envelope died on. Budget ~60 s (measured: a 1-rank group
# alone took 22 s to form, most of it torch import + GPU context creation).
#
# BACKOUT: default OFF. With PREFLIGHT unset this block evaluates one test and
# changes nothing at all. PREFLIGHT=1 turns it on; even then only a real
# rejection stops the boot:
#     canary rc 0 = PASS          -> boot
#     canary rc 1 = REJECT        -> refuse to boot (exit 3) -- the only verdict
#     canary rc 2/3/other         -> warn and boot anyway (busy box, no port,
#                                    timeout: "no verdict" is not "bad env", and
#                                    a safety tool that blocks boots on a busy
#                                    box becomes an outage). PREFLIGHT_STRICT=1
#                                    makes no-verdict refuse too, for sweeps.
# Exit 3 is deliberately distinct from the engine's own failures so a preflight
# rejection is never mistaken for a crash, and never retried by BOOT_TRIES.
PREFLIGHT="${PREFLIGHT:-0}"
PREFLIGHT_STRICT="${PREFLIGHT_STRICT:-0}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-420}"
# An already-exported PREFLIGHT_PY wins, so tests and non-default venvs can point
# the canary at another interpreter without touching $PYTHON.
PREFLIGHT_PY="${PREFLIGHT_PY:-${PYTHON:-/root/.pyenv/versions/venv312/bin/python}}"
_pf_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-$_pf_dir/scripts/preflight-nccl.py}"

# run_preflight: echo the canary into $LOG_FILE and return its exit code.
run_preflight() {
    local rc
    if [ ! -f "$PREFLIGHT_SCRIPT" ]; then
        echo "run-vllm: PREFLIGHT=1 but $PREFLIGHT_SCRIPT is missing -- skipping" >&2
        return 2
    fi
    echo "run-vllm: NCCL preflight (PREFLIGHT=1) -- ~60 s, before the ~6 min load."
    echo "          Backout: PREFLIGHT=0 (the default)."
    timeout -k 15 "$PREFLIGHT_TIMEOUT" "$PREFLIGHT_PY" "$PREFLIGHT_SCRIPT" \
        --hidden-size "${PREFLIGHT_HIDDEN:-2560}" \
        --prefill-rows "${PREFLIGHT_PREFILL_ROWS:-2048}" \
        --iters "${PREFLIGHT_ITERS:-20}" \
        --devices "${PREFLIGHT_DEVICES:-0,1,2,3}" \
        --pg-timeout "${PREFLIGHT_PG_TIMEOUT:-120}" \
        2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    case "$rc" in
      124|137) rc=3 ;;   # `timeout` itself fired -> no verdict, not a rejection
    esac
    return "$rc"
}

# preflight_gate: turn the canary's exit code into a boot decision.
preflight_gate() {
    local rc="$1" why
    if [ "$rc" = 0 ]; then
        echo "run-vllm: preflight PASS -- the collective environment is valid, booting"
        return 0
    fi
    if [ "$rc" = 1 ]; then
        echo "run-vllm: preflight REJECT -- REFUSING TO BOOT (exit 3)." >&2
        echo "          The NCCL environment in effect refuses a collective this engine" >&2
        echo "          needs. The model load would have taken ~6 minutes to say the" >&2
        echo "          same thing, and BOOT_TRIES would have retried it pointlessly." >&2
        echo "          The NCCL_* variables used are printed above; they are the suspects." >&2
        echo "          Bypass: PREFLIGHT=0 ./run-vllm   (or fix the env it flagged.)" >&2
        exit 3
    fi
    # One string per branch, on one line. A line-continuation here parses as
    # `why="first half" second-half`, which bash reads as an assignment PLUS a
    # command named "second-half": the reason vanished from the message and,
    # under `set -u`, an unbound $why aborted the subshell with exit 1 instead of
    # the intended refusal exit 3. Caught by scripts/test-preflight-gate.sh.
    local why=""
    case "$rc" in
      2) why="inconclusive: no GPU context or no process group (a box whose VRAM is already held does this -- RCCL wants ~256 MiB per GPU for its communicator -- and so does a rendezvous port that could not be bound)" ;;
      3) why="timed out: a hang is NOT a verdict, it may be rendezvous rather than RCCL" ;;
      *) why="unexpected exit code $rc" ;;
    esac
    if [ "$PREFLIGHT_STRICT" = 1 ]; then
        echo "run-vllm: preflight $why -- and PREFLIGHT_STRICT=1, so refusing to boot." >&2
        exit 3
    fi
    echo "run-vllm: preflight $why -- booting anyway (this is not a statement about" \
         "the env)." >&2
    return 0
}

# PREFLIGHT_ONLY=1: run the canary and exit without booting anything. Handiest
# way to test an env change, and the only way to do it against the env THIS
# script exports (NCCL_BLOCKING_WAIT and friends are decided above). It refuses
# to allocate GPU memory next to a serving engine unless forced.
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
           http://localhost:9000/v1/models 2>/dev/null)
    if [ "$code" = 401 ] || [ "$code" = 200 ]; then
        if [ "${PREFLIGHT_FORCE:-0}" != 1 ]; then
            echo "run-vllm: PREFLIGHT_ONLY refused -- an engine is serving on :9000" >&2
            echo "          and the canary would allocate on its GPUs (only ~60 MiB" >&2
            echo "          free per GPU with 0.95 utilization). PREFLIGHT_FORCE=1 to" >&2
            echo "          override; a failed allocation there is clean, but it is" >&2
            echo "          the serving process that owns that memory." >&2
            exit 1
        fi
    fi
    run_preflight
    rc=$?
    echo "run-vllm: PREFLIGHT_ONLY done, canary exit $rc (0=PASS 1=REJECT 2=no verdict 3=timeout)"
    exit "$rc"
fi

spec_args=(); [ -n "$SPEC" ] && spec_args=(--speculative-config "$SPEC")
mm_args=();   [ "$TEXT_ONLY" = 1 ] && mm_args=(--language-model-only)
kv_args=();   [ -n "$KVBYTES" ] && kv_args=(--kv-cache-memory-bytes "$KVBYTES")

# Changes vs the previous version:
#   --speculative-config   -> moved ABOVE the pipe (it trailed `| tee` and was
#                             therefore never passed: speculative_config=None)
#   --async-scheduling     -> off. Crashed all 4 workers at 18:53:57 with
#                             `AssertionError: GDN decode-first invariant
#                             violated: non-spec decodes not first`
#                             (v1/attention/backends/gdn_attn.py:382, a V1
#                             scheduler invariant) under spec=None; called a
#                             GDN async-scheduling race in HANDOVER-dflash2.md:18.
#                             Measured no throughput effect (DEVLOG-spec-decode.md:334).
#   spec method            -> mtp, not ngram: the failed attempt ran
#                             {"method":"ngram"}, which config/vllm.py:1265-1273
#                             rejects together with async scheduling (only
#                             EAGLE/MTP/draft_model/ngram_gpu/dspark pass).
#                             prompt_lookup_max is ngram-only, and qwen4_exp_mtp
#                             normalizes to mtp (config/speculative.py:822).
#                             k=3 shipped at production context; k=4 is a
#                             dead-end on real payloads (DEVLOG-mtp-depth-matrix.md).
#   --generation-config    -> once, not twice ("Found duplicate keys"): `vllm`
#                             (no generation config) was shadowed by `auto`.
#   --compilation-config   -> capture ladder sized to the spec step (was
#                             [1,2,4,8]: 8 unreachable at max-num-seqs 4).
#   --tool-call-parser     -> stays qwen3_coder: chat_template.jinja emits
#                             XML-style function and parameter tags wrapped in
#                             a tool_call block, so the CDNA recipe's
#                             qwen3_xml parser would have misparsed it.
#   --block-size 64        -> not added: the engine raises the attention block
#                             to 784 tokens anyway to match the mamba page size.
#   --trust-remote-code    -> dropped: no custom code in the checkpoint (no
#                             *.py, auto_map=None) and the arch resolves
#                             natively. Server is bound to 0.0.0.0.
#   --enable-expert-parallel, prefix caching (27-43% hit rate on agent
#                             traffic), --max-num-batched-tokens and --dtype
#                             float16 -> unchanged. float16 stays explicit on
#                             purpose: gfx906 has no native bf16, and the
#                             "fused CUDA kernel requires a BF16 GDN model"
#                             fallback to the Triton GDN path is expected here,
#                             not a misconfiguration (ROADMAP.md QSA-FN-1/2:
#                             sparse attention 26.5ms fp16 vs 116.5ms bf16).
#   --language-model-only  -> added. VL checkpoint (vision_config, depth 27);
#                             the engine reserved a 16384-token encoder cache
#                             and profiled the ViT for a text-only workload.
# boot_once(): one launch of the engine. Body deliberately left at its original
# indentation so `git diff -w` of this file stays readable.
# NOTE the "> >(tee ...)" instead of "| tee": a backgrounded *pipeline* makes $!
# the PID of tee (the last element), not of vllm, which breaks the liveness check
# in the BOOT_TRIES loop below. Process substitution keeps $! the engine's own PID
# (verified on this bash 5.2.21: `jobs -rp` matches $!, and a failing child still
# propagates rc=1). stdout+stderr both still land in $LOG_DIR/vllm.log, as before.
boot_once() {
# Assembled into an array, and the array echoed to the log before it runs, so the
# log file states exactly what the engine was started with (see write_manifest:
# the same defect that hid the 09-22 knobs also hid the argv). Building it inside
# the function is deliberate -- "$@" must expand at CALL time, when the operator's
# pass-through arguments are in scope, and with BOOT_TRIES>1 every attempt logs
# the argv it actually used. Note ${var[*]} joins with spaces for reading: the
# quoting of --compilation-config's JSON is preserved in the real exec, not here.
local argv=(
    vllm serve /ai/models/hf.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4
    --served-model-name vllm-chat
    --host 0.0.0.0
    --port 9000
    --tensor-parallel-size "$TP"
    --dtype float16
    --max-model-len "$MAXLEN"
    --max-num-seqs "$MAXSEQS"
    --max-num-batched-tokens "$MBT"
    --gpu-memory-utilization 0.95
    --compilation-config "$comp_cfg"
    "${graph_args[@]}"
    --no-async-scheduling
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --generation-config auto
    --chat-template /ai/models/hf.co/froggeric/Qwen-Fixed-Chat-Templates/chat_template.jinja
    "${jit_args[@]}" "${spec_args[@]}" "${mm_args[@]}" "${kv_args[@]}" $@
)
printf 'run-vllm: engine argv: %s\n' "${argv[*]}" >> "$LOG_FILE"
"${argv[@]}" > >(tee -a $LOG_FILE) 2>&1
}

#    --enable-expert-parallel \
# --- BOOT_TRIES: retry a boot that dies before it ever starts serving -------
# Why this exists: the MTP+graphs GC use-after-free (report §13.4, §14) is a
# boot-time-only failure. Across the 17 boots in the surviving logs (report
# §14.8) MTP+graphs served 3 of 11 and every other configuration served every
# time, while an engine that *does* reach startup has been observed to serve
# 5h30m+ without reproducing. Paying ~4 min per failed attempt is far cheaper
# than running eager: 15.9 t/s median served vs 34.5 t/s median for the same
# model with MTP + piecewise graphs (report §14.9).
#
# What it will NOT do: restart anything that was already serving. The retry
# branch is reachable only when the child exits before the health endpoint ever
# answered -- which is the failure being routed around. Once the endpoint
# answers, this script just waits on the child and exits with its status, exactly
# like the plain launch did, so a mid-serving crash stays the operator's call.
# (401 counts as "up": the endpoint is key-gated, so an unauthenticated GET
# answers 401 from the moment the API is listening. 000 = connection refused.)
#
# Between attempts it waits for the crashed workers to release their ~21.9 GiB
# per GPU -- measured, not on a clock. A retry that starts while the previous
# attempt still holds VRAM dies of OOM and teaches nothing, which would look like
# "the bug is deterministic" and would send the next investigation down a false
# path. A fixed sleep is not that wait: on 2026-09-21 the segfault at 20:52:48
# still had ~21.8 GiB/GPU pinned 2 and 4 minutes later, so two of the eight
# BOOT_TRIES were consumed by
#   ValueError: Free memory on device cuda:3 (10.17/31.98 GiB) on startup is
#   less than desired GPU memory utilization (0.95)
# raised from request_memory() (v1/worker/utils.py:526) -- not the GC bug at all,
# which is how a 3-in-11 coin flip turned into "every retry dies". The driver
# keeps buffer objects alive until every fd on /dev/kfd and /dev/dri/renderD*
# closes, so what has to be waited for is the processes, not the wall clock.
#
# BOOT_TRIES=1 (default) = previous behaviour exactly: launch once, foreground,
# no polling, no pre-flight check, no drain gate.
BOOT_TRIES="${BOOT_TRIES:-1}"
HEALTH_URL="http://localhost:9000/v1/models"
VRAM_GATE_TIMEOUT="${VRAM_GATE_TIMEOUT:-300}"
VRAM_GATE_MAX_USED="${VRAM_GATE_MAX_USED:-2147483648}"

# Highest VRAM usage across all GPUs, in bytes. -1 if the probe is unusable,
# which the caller treats as "cannot gate, fall back to a blind wait".
vram_max_used() {
    rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(max(int(v["VRAM Total Used Memory (B)"]) for v in d.values()))
except Exception:
    print(-1)
'
}

# wait_vram_released [reap]
#   Poll until no GPU reports more than VRAM_GATE_MAX_USED in use, or until
#   VRAM_GATE_TIMEOUT seconds have passed. With "reap" -- which the caller passes
#   only after an attempt IT launched has died -- it first signals VLLM::
#   stragglers: a worker that outlived its engine is holding ~22 GiB with nothing
#   left to serve, and reaching here already means :9000 does not answer. Nothing
#   is killed before the first attempt, because a VLLM:: process at that point is
#   somebody else's mid-boot engine rather than our own corpse. Returns non-zero
#   only on timeout, and the boot is attempted anyway: a GPU held by an
#   unrelated workload is the operator's call to make, not a reason to spin.
wait_vram_released() {
    local reap="${1:-}" waited=0 used
    if [ "$reap" = "reap" ] && pgrep -f '^VLLM::' >/dev/null 2>&1; then
        echo "run-vllm: reaping VLLM:: stragglers still holding GPU memory" >&2
        pkill -TERM -f '^VLLM::' 2>/dev/null
        sleep 10
        pkill -KILL -f '^VLLM::' 2>/dev/null
        sleep 5
    fi
    while :; do
        used=$(vram_max_used)
        if [ "$used" = "-1" ]; then
            echo "run-vllm: rocm-smi probe unusable -- falling back to a 45 s wait" >&2
            sleep 45
            return 0
        fi
        if [ "$used" -lt "$VRAM_GATE_MAX_USED" ]; then
            if [ "$waited" -gt 0 ]; then
                echo "run-vllm: GPUs drained after ${waited} s" >&2
            fi
            return 0
        fi
        if [ "$waited" -ge "$VRAM_GATE_TIMEOUT" ]; then
            echo "run-vllm: GPU memory NOT released after ${waited} s ($((used / 1048576))" >&2
            echo "          MiB/GPU still in use) -- booting anyway. If this attempt" >&2
            echo "          dies in request_memory() that is the drain, not the GC bug." >&2
            return 1
        fi
        echo "run-vllm: $((used / 1048576)) MiB/GPU still held, waiting (${waited}/${VRAM_GATE_TIMEOUT} s)" >&2
        sleep 15
        waited=$((waited + 15))
    done
}

# --- BOOT MANIFEST: what this log file was booted WITH ----------------------
# BEGIN BOOT MANIFEST (marker: scripts/test-optimized-bundle.sh extracts this)
# Twice in one week a measurement was worth less than it should have been
# because the log did not record the launch environment:
#   * the 2026-09-22 boot that served 494 tok/s prefill and could not be
#     reproduced (426 on 2026-09-26, 15.9% apart) -- nobody can say which knobs
#     it had, because the log does not say;
#   * boot D on 2026-09-26, where an inherited NCCL_PROTO=LL reached all four
#     workers through the shell while the launcher was printing "NCCL left at
#     RCCL defaults".
# Both were the same defect: no manifest. So the launcher writes one to the log
# AND to the terminal before the engine starts. It lists every knob this script
# resolves, where each came from (bundle / explicit / default), and the NCCL
# variables it deliberately did not set -- because "absent" is a finding too.
# It is written to $LOG_FILE first, so it sits above the engine's own output
# rather than being interleaved with it.
write_manifest() {
    {
        echo "run-vllm: ---------------- BOOT MANIFEST $(date '+%F %T') ----------------"
        echo "run-vllm: launcher $(basename "${BASH_SOURCE[0]}") $(wc -l < "${BASH_SOURCE[0]}") lines, sha256 $(sha256sum "${BASH_SOURCE[0]}" | cut -c1-12)"
        echo "run-vllm: OPTIMIZED=$OPTIMIZED  (0 = known-good defaults, 1 = measured-best bundle)"
        for _k in "${OPT_BUNDLE[@]}"; do
            printf 'run-vllm:   %-15s = %-8s [%s]\n' "$_k" "${!_k:-0}" "${OPT_SRC[$_k]}"
        done
        echo "run-vllm: engine: TP=$TP maxlen=$MAXLEN maxseqs=$MAXSEQS max-batched=$MBT graphmode=$GRAPHMODE"
        echo "run-vllm: spec:   ${SPEC:-<none: no speculation>}"
        echo "run-vllm: nccl:   envelope=${NCCL_ENVELOPE:-0} blocking_wait=$NCCL_BLOCKING_WAIT"
        echo "run-vllm:         algo=${NCCL_ALGO:-<unset>} proto=${NCCL_PROTO:-<unset>} max_nchannels=${NCCL_MAX_NCHANNELS:-<unset>} (not set here; inherited from this shell if shown)"
        echo "run-vllm: other:  BOOT_TRIES=$BOOT_TRIES PREFLIGHT=$PREFLIGHT TVMFFI=$TVMFFI TEXT_ONLY=$TEXT_ONLY"
        echo "run-vllm:         GC_THAW=${GC_THAW:-0} GC_DUMP=${GC_DUMP:-0} DRAFTER_EAGER=${DRAFTER_EAGER:-0}"
        echo "run-vllm: measured on this box: bundle = 0.942 ms/token cold prefill vs 2.346 defaulted;"
        echo "run-vllm: decode unchanged. docs/prefill-io-wall.md, docs/decode-perf-suggestions.md 5f/5g"
        echo "run-vllm: ---------------------------------------------------------------"
    } | tee -a "$LOG_FILE" >&2
}
# END BOOT MANIFEST

# The manifest goes out once per launcher run, before either boot path, so it is
# the first thing in the log file and cannot be confused with engine output.
write_manifest

if [ "$BOOT_TRIES" = 1 ]; then
    # "$@" here is mandatory: inside a function $@ is the FUNCTION's positional
    # params, so calling boot_once with no arguments would silently drop every
    # pass-through CLI argument the operator put after ./run-vllm .
    boot_once "$@"
    exit $?
fi

if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$HEALTH_URL")" = 401 ]; then
    echo "run-vllm: something is ALREADY serving on :9000 -- refusing to boot a" >&2
    echo "          second engine, because that one would fail and the retry loop" >&2
    echo "          would then paper over it. Stop the running server first." >&2
    exit 1
fi

# The NCCL preflight runs AFTER the "already serving" gate and BEFORE the model
# load: it allocates four GPU contexts, so it must never do that next to an
# engine that owns the VRAM. Failure here exits 3 and consumes no BOOT_TRIES.
if [ "$PREFLIGHT" = 1 ]; then
    run_preflight
    preflight_gate "$?"
fi

# Gate the first attempt too: an engine that is mid-crash does not answer on
# :9000, so the pre-flight above waves it through while its memory is still
# pinned. No reaping here, for the reason in wait_vram_released().
wait_vram_released

attempt=1
while :; do
    echo "run-vllm: boot attempt $attempt of $BOOT_TRIES" >&2
    boot_once "$@" &
    job=$!
    up=0
    while jobs -rp | grep -qx "$job"; do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$HEALTH_URL" 2>/dev/null)
        if [ "$code" = 401 ] || [ "$code" = 200 ]; then up=1; break; fi
        sleep 5
    done
    if [ "$up" = 1 ]; then
        echo "run-vllm: attempt $attempt is SERVING (pid $job) -- no further retries" >&2
        echo "run-vllm: from here on a crash is NOT retried; that is deliberate" >&2
        wait "$job"
        exit $?
    fi
    wait "$job"
    rc=$?
    echo "run-vllm: attempt $attempt died before serving (exit $rc)" >&2
    if [ "$attempt" -ge "$BOOT_TRIES" ]; then
        echo "run-vllm: giving up after $attempt attempts ($rc)" >&2
        exit "$rc"
    fi
    attempt=$((attempt + 1))
    echo "run-vllm: waiting for the crashed workers to release GPU memory" >&2
    wait_vram_released reap
done

# One-time setup for the HIP blocking-sync shim (docs/gfx906/running.md): with
# TP>=2 the workers busy-wait by default (hipDeviceScheduleAuto resolves to
# active-wait whenever the GPU count is below the CPU thread count, which is
# true of this 4-GPU / 64-thread box), so each worker pegs 1-2 host cores at
# ~100% for the life of the process. The flag can only be flipped before the
# process creates its first HIP queue, which is why the fix is a .pth file
# (executed at interpreter startup, before torch/vllm are imported) rather
# than something in this script. It does not survive a venv rebuild.
#
#   cp vllm-gfx906-mobydick/docs/gfx906/gfx906-blocking-sync.pth \
#      /root/.pyenv/versions/venv312/lib/python3.12/site-packages/
