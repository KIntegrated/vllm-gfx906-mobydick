# Qwen3.8-27B launch crashes — postmortem: NAS outage, not a GPU bug

Copyright Kevin Read <me@kevin-read.com>

Date: 2026-08-19. Machine: `mi50-01` (MI50 gfx906, kernel 6.8.0,
`iommu=pt amdgpu.noretry=0`). Two consecutive boots were killed while
starting `cyankiwi/Qwen3.8-27B-AWQ-INT4` (20 G, on NFS) in the local
vLLM build (`gfx906/main` @ `5d960a503c`).

## Verdict

**Neither crash is a GPU fault, a Qwen3.8 kernel bug, or a vLLM bug.**
Root cause: the NAS (`192.168.33.240:/volume2/ai` → `/data`, NFSv4,
**hard** mount) became unresponsive while vLLM was loading the 20 G
model through `np.memmap`'d safetensors. The amdgpu HMM page-pinning
path for the H2D weight copy blocked in uninterruptible D-state on NFS
reads; the GPU "fence timeout" and "unrecoverable state" log lines are
*downstream symptoms* of that I/O stall, not a GPU fault. The BACO GPU
reset **succeeded in both crashes** — the card was healthy. The power
cycles were forced by unkillable D-state threads (NFS I/O) that
survived the reset, plus the dead NFS share blocking unmount at
shutdown.

## Evidence (decisive): the hung-task stacks

Both boots produced hung-task dumps for `VLLM::EngineCore`, and both
bottom out in **NFS page reads**, not GPU waits:

```
Crash #1 (18:45:08, pid 170137)          Crash #2 (20:02:34, pid 34003)
task:VLLM::EngineCor state:D             task:VLLM::EngineCor state:D
  io_schedule                              io_schedule
  folio_wait_bit_common                    folio_wait_bit_common
  filemap_fault                            filemap_fault
  hmm_range_fault                          hmm_vma_fault / hmm_range_fault
  amdgpu_hmm_range_get_pages [amdgpu]      amdgpu_hmm_range_get_pages [amdgpu]
```

`amdgpu_hmm_range_get_pages` is the amdgpu HMM driver paging in host
pages for a DMA (H2D) copy. `filemap_fault → folio_wait_bit` means the
page is file-backed (the mmap'd safetensors file on NFS) and the kernel
is waiting for the NFSv4 READ to complete. On a hard mount that wait is
unbounded → D-state → unkillable.

Supporting facts:

- NAS "not responding" logs in **both** boots:
  `nfs: server 192.168.33.240 not responding, still trying ... timed out`
  (18:44:42–18:45:37 in boot -2; 20:03:37–20:04:29 in boot -1).
  These log lines *lag* the actual outage (they appear when RPC timeout
  backoff expires), which is why the NAS lines look later than the
  fence timeouts.
- **Zero** GPU page faults in either boot (`journalctl | grep -c
  "page fault"` = 0).
- **Zero** second fence timeouts after the reset in either boot — the
  reset GPU was fully functional.
- Both crashes fell inside the 20 G NFS weight-load window:
  - Crash #2: thread blocked since **T+10 s** (122 s hung-task warning
    at 20:02:34 → block start ~19:59:52; launch 19:59:42); fence
    timeout at T+46 s.
  - Crash #1: fence timeout at T+115 s (launch 18:39:47); thread
    blocked since ~18:43:06.

## Mechanism, step by step

1. vLLM loads safetensors with `np.memmap` — weights are a file-backed
   mapping over NFS.
2. Weight H2D copy: the ROCm/amdgpu HMM path calls
   `amdgpu_hmm_range_get_pages` to pin/page-in the host range for DMA.
3. Uncached page → kernel issues NFSv4 READ → NAS unresponsive → hard
   mount retries forever → faulting thread sleeps on the folio bit →
   **D-state**.
4. The in-flight H2D copy's GPU fence can never signal → driver logs
   `qcm fence wait loop timeout expired` after its wait budget.
5. Driver attempts queue preemption → fails (there is no runnable GPU
   work to preempt) → pessimistic "cp might be in an unrecoverable
   state" warning → **BACO cold reset** → *succeeds* (`GPU reset
   succeeded`, `VRAM is lost`).
6. The D-state thread is in NFS I/O, untouched by the GPU reset →
   survives; SIGKILL cannot reach D-state.
7. `systemd` poweroff waits on the unkillable thread and cannot unmount
   `/data` (NAS still dead) or `/local` (busy) → **shutdown hangs** →
   power cycle required.

## The two incidents

| | Crash #1 (boot -2) | Crash #2 (boot -1) |
|---|---|---|
| Launch | 18:39:47 (graphs, default) | 19:59:42 (`--enforce-eager`) |
| Block start | ~18:43:06 (from 122 s warning) | ~19:59:52 (T+10 s) |
| Fence timeout | 18:41:42 (T+115 s) | 20:00:28 (T+46 s) |
| BACO reset | succeeded 18:41:44 | succeeded 20:00:30 |
| NAS not-responding | 18:44:42–18:45:37 | 20:03:37–20:04:29 |
| Hung task | `VLLM::EngineCor:170137` D-state | `VLLM::EngineCor:34003` D-state |
| Shutdown | poweroff hang 18:45:49 → power cycle | SIGKILL 20:05:48 (no effect) → power cycle |

Eager and graph modes failed identically, which also retires the
initial hypothesis (CUDA-graph capture as the trigger) — the trigger
is the NFS load window, independent of graph mode.

## What this is *not*

- **Not the morning's 0.6B FA UAF** (`5d960a503c`): that was a real
  GPU memory bug with kernel-logged no-retry *write* page faults,
  SIGABRT (exit 134), no fence timeout, no D-state, fully recoverable.
  These crashes have zero page faults and a D-state I/O stack —
  different mechanism entirely. (Conversely, this journal data — 16:03
  kernel faults at `0x7584fc000000`–`0x75850c400000` bracketing the
  runtime-reported `0x7584ff900000`, `RW: 0x1` = write, unmapped page —
  independently *confirmed* the UAF diagnosis.)
- **Not proven to be innocent of Qwen3.8**: both runs died during
  weight load, before model kernels meaningfully executed. The
  D=256-head-dim serving question (gfx906 FA/gather at serving scale,
  3.8 block sizes, GDN params) remains open — see actions.

## Why Qwen3.8 specifically (correlation)

The 20 G AWQ checkpoint lives on NFS (`/data/cache/huggingface/...`),
so each load exposes the run to a multi-minute NFS stream. The same NAS
has served the 15 G 27B-AWQ load cleanly many times, and the MoE runs
from local `/local/models` are immune — exposure scales with load size
and duration. Two consecutive NAS outages during two consecutive
attempts makes the NAS itself the prime suspect.

## Actions

1. **Check the NAS** (Synology `192.168.33.240`, `/volume2/ai`):
   system/NFS logs for 2026-08-19 18:39–18:46 and 19:59–20:06, plus
   disk health for volume2 and the network path. If it is flaky, fix it
   before any further NFS weight loads.
2. **Copy the model to local disk** (`/local` or `/tmp`, root FS has
   ~231 G) and re-run the Qwen3.8 validation from local weights — this
   removes the failure class entirely and still answers the D=256
   question.
3. **Robustness**: hard NFS + mmap'd weight loading means *any* NAS
   hiccup during a load wedges the whole machine (D-state thread +
   blocked unmount). Options: keep serving weights on local disk
   (preferred), or mount the model share `soft,timeo=` so reads fail
   (load aborts cleanly) instead of hanging the box.
4. **Operational signature**: if a future GPU fence timeout is
   accompanied by D-state `VLLM::EngineCor` threads whose stack
   bottom is `amdgpu_hmm_range_get_pages`/`filemap_fault`, treat it as
   I/O, not GPU: check the NAS first; recover by power cycle.

## Current state (2026-08-19, post-reboot)

- Boot since 20:06:54; no amdgpu errors; NAS responsive again; `/data`
  is an `auto`/`nofail` fstab entry (remount after reboots as needed).
- Both `/tmp` trees were wiped by the reboots: `/tmp/bench/*`,
  `/tmp/fa-analysis.md` (docker handoff), `/tmp/HANDOVER-vllm-27b.md`
  are gone. The committed copies in the repo (`benchmarks/kernels/
  gfx906/`, `docs/gfx906/`) are the durable record.
- Qwen3.8-27B: **still unvalidated** (never got past weight load).
