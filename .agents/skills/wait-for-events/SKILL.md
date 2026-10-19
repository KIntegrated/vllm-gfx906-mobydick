---
name: wait-for-events
description: Wait for a detached long-running job (GPU bench, model load, server boot, multi-arm sweep) without blind `sleep`. Use when polling a background run for completion, for a known failure signature (e.g. RuntimeError, OOM, driver wedge), for VRAM to drain, or for a process to exit — and when a long sleep would risk a tool-abort or waste the window.
---

# Waiting for detached jobs (time-boxed, resumable)

## The rule

**Never `sleep N` to wait for a detached job.** A sleep is blind (no information
until it returns), abortable (an aborted tool call loses the wait *and* leaves you
guessing), and wasteful (a run that crashed at 40 s still costs the full N).

Instead: **(1) make the job print sentinels, (2) poll with
`docs/gfx906/_wait_for.py` in bounded windows, (3) branch on its exit code.**

## 1. Sentinels in the driver script

Every driver writes one line per state change, and one at the end:

```bash
run() {
  tag=$1
  echo "=== ARM $tag start $(date -u +%H:%M:%S) ==="
  timeout 2400 cmd ... > /local/tmp/$tag.log 2>&1
  rc=$?                      # capture BEFORE any command substitution
  echo "=== ARM $tag rc=$rc ==="
}
...
echo "=== DONE $(date -u +%H:%M:%S) ==="
```

`rc=$?` into a variable before `$(date)`/`$(...)` is mandatory — command
substitution resets `$?` (a documented false-"RC=0" trap in this repo).

## 2. Poll in bounded windows

```bash
.venv/bin/python docs/gfx906/_wait_for.py \
    --log /local/tmp/run.out \
    --until '=== DONE' \
    --fail 'RuntimeError|Traceback|D must be|HIP out of memory' \
    --timeout 240 --poll 10
```

Exit codes → next action:

| exit | meaning | do |
|---|---|---|
| 0 | success condition met | read the result files |
| 2 | still waiting at the deadline | **re-invoke the same command** |
| 3 | failure signature matched | the last lines are printed; diagnose now |
| 4 | log file does not exist yet | check the launch, then retry |

Heartbeat every poll shows elapsed time and the newest log line, so the
transcript records progress instead of a silent gap.

## 3. Other conditions

```bash
--idle 120                 # every log untouched for 120 s  (a silent hang)
--pid 12345                # that process exited
--vram-below 200000000     # teardown done: VRAM back near the ~10 MB baseline
```

Multi-arm sweeps: watch the driver's `.out` with `--until '=== B4 DONE'`, and add
`--fail` for the crash signatures you already know. Per-arm logs are then read
directly — the driver's `rc=` line says which arm failed.

## Why not just `wait`?

`wait` only works for children in the same shell. These sessions are launched
detached (`setsid nohup ... < /dev/null &`) so a tool abort cannot kill them;
that also means there is nothing to `wait` on — the log is the only channel.

## Related failure modes this does not fix

- Launching without `setsid` → the job dies with the tool abort. See the
  PROCESS-GROUP SAFETY section of the workspace AGENTS.md.
- `pkill -f <script>` matching your own command line when the script name
  appears in it; kill by PID instead.
- A resident `llama-server` holding VRAM → counts as "not drained"; check
  `rocm-smi --showpids` before blaming the run.
