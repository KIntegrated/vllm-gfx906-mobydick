#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Bounded wait for a detached run: success, failure, or "keep waiting".

The failure mode this replaces: `sleep 300` to wait for a detached GPU job. A
long sleep is (a) blind - it says nothing until it returns, (b) unreliable - the
tool call can be aborted mid-sleep and takes the information with it, and (c)
wasteful - a job that crashed after 40 s still costs the full sleep.

Instead every driver script should print sentinels (`=== ARM x rc=0 ===`,
`=== DONE ===`), and the agent calls this tool in short, bounded windows:

    python docs/gfx906/_wait_for.py --log /local/tmp/run.log \\
        --until '=== DONE' --fail 'RuntimeError|D must be' --timeout 240

Exit codes tell the caller what to do next:

    0  success condition met
    2  still waiting at the deadline -> re-invoke to keep waiting
    3  failure pattern matched (last lines are printed for the diagnosis)
    4  a required log file does not exist yet

Every call prints a heartbeat (elapsed, state, last log line), so the transcript
shows progress instead of a silent gap, and an aborted call loses nothing: the
job is detached and the next call re-derives the state from the log.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

EXIT_OK = 0
EXIT_WAITING = 2
EXIT_FAILED = 3
EXIT_NO_LOG = 4


def _last_line(path: str, limit: int = 110) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return "<unreadable>"
    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    if not lines:
        return "<empty>"
    return lines[-1][:limit]


def _tail(path: str, n: int) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [ln.rstrip()[:200] for ln in f.readlines()[-n:]]
    except OSError:
        return [f"<cannot read {path}>"]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _vram_used_bytes() -> int | None:
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.findall(r"Used Memory \(B\):\s*(\d+)", out)
    return max(int(x) for x in m) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", action="append", required=True,
                    help="log file to watch (repeatable)")
    ap.add_argument("--until", action="append", default=[],
                    help="regex meaning success (repeatable: any match wins)")
    ap.add_argument("--fail", action="append", default=[],
                    help="regex meaning failure (any match exits 3)")
    ap.add_argument("--idle", type=float, default=None,
                    help="success when every log has been idle this long")
    ap.add_argument("--pid", type=int, default=None,
                    help="success when this pid is gone")
    ap.add_argument("--vram-below", type=int, default=None,
                    help="success when max GPU VRAM used is below this")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--tail", type=int, default=12,
                    help="lines shown on failure")
    args = ap.parse_args()

    missing = [p for p in args.log if not os.path.exists(p)]
    if missing and len(missing) == len(args.log):
        print(f"NO_LOG {missing[0]}", flush=True)
        return EXIT_NO_LOG

    until = [re.compile(p) for p in args.until]
    fail = [re.compile(p) for p in args.fail]
    t0 = time.time()
    deadline = t0 + args.timeout

    while True:
        state = "waiting"
        hit: str | None = None
        for path in args.log:
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    body = f.read()
            except OSError:
                continue
            for pat in fail:
                m = pat.search(body)
                if m:
                    print(f"FAILED {path}: matched {pat.pattern!r}", flush=True)
                    for ln in _tail(path, args.tail):
                        print(f"  | {ln}", flush=True)
                    return EXIT_FAILED
            for pat in until:
                if pat.search(body):
                    hit = f"{path}: {pat.pattern!r}"
                    break
            if hit:
                break

        if hit:
            print(f"OK {hit} after {time.time() - t0:.0f}s", flush=True)
            return EXIT_OK

        idle_ok = False
        if args.idle is not None:
            ages = [time.time() - os.path.getmtime(p)
                    for p in args.log if os.path.exists(p)]
            idle_ok = bool(ages) and min(ages) >= args.idle

        pid_ok = args.pid is not None and not _pid_alive(args.pid)
        vram_ok = False
        if args.vram_below is not None:
            used = _vram_used_bytes()
            vram_ok = used is not None and used < args.vram_below

        if idle_ok or pid_ok or vram_ok or (not until and args.idle is None
                                            and args.pid is None
                                            and args.vram_below is None):
            state = "idle" if idle_ok else "pid-gone" if pid_ok else "vram-ok"
            print(f"OK {state} after {time.time() - t0:.0f}s", flush=True)
            return EXIT_OK

        if time.time() >= deadline:
            print(f"WAITING after {args.timeout:.0f}s "
                  f"(re-invoke to continue)", flush=True)
            for path in args.log:
                if os.path.exists(path):
                    print(f"  {os.path.basename(path)}: {_last_line(path)}",
                          flush=True)
            return EXIT_WAITING

        print(f"[{time.time() - t0:5.0f}s] {state} "
              f"{os.path.basename(args.log[0])}: {_last_line(args.log[0])}",
              flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
