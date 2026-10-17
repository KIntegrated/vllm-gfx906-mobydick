#!/usr/bin/env python3
"""The heap corruptor is `amdsmi_get_gpu_asic_info`, and it needs no engine.

Why this exists
---------------
This box has carried a "standing heap corruptor" for weeks: MTP+graph boots fail
~2-in-11, and on 2026-10-02 a **host-only** benchmark (torch CPU + numpy, no CUDA,
no NCCL, no engine, no mmap of the PLE table) segfaulted in setup and the identical
re-run was clean. Any theory whose load-bearing premise was "it must be the engine's
allocator use" could not explain that. `PYTHONMALLOC=debug` was the discriminator
that was supposed to answer it against the probe in seconds instead of against a
7-minute boot.

It answered on the first run, deterministically, at *import* time:

    Debug memory block at address p=...: API 'm'
        456 bytes originally requested
        The 8 pad bytes at tail=... are not all FORBIDDENBYTE (0xfd):
            at tail+0: 0x00 *** OUCH   ... (x8)
        Data at p: 41 4d 44 20 49 6e 73 74 ... 00 00 00 00   <- "AMD Inst..."
    Fatal Python error: _PyMem_DebugRawFree: bad trailing pad byte
    Current thread ...:
      File ".../vllm/platforms/rocm.py", line 205 in _query_gcn_arch_from_amdsmi
    Memory block allocated at (most recent call first):
      File ".../amdsmi/amdsmi_interface.py", line 1650        <- asic_info_struct = amdsmi_asic_info_t()

What the arms prove
-------------------
`measure`  Poisons 512 bytes of slack behind a real `amdsmi_asic_info_t`, calls the
           library, and reports which bytes changed. The slack has to be *poisoned*:
           the library writes **zeros**, so an unpoisoned tail would look like
           "nothing happened". This is the positive measurement, and it does not need
           the debug allocator.
`control`  Runs exactly one amdsmi call and lets the exit code speak: `handles` and
           `meminfo` exit 0, `asicinfo` aborts with the pad-byte fatal. Same process,
           same init, one call different -- so the attribution is to the function, not
           to amdsmi_init or to the handle query.
`import`   Imports `vllm.platforms.rocm`, i.e. the path every worker takes at boot
           (`rocm.py:249` runs `_GCN_ARCH = _get_gcn_arch()` at module level).

Read `measure`'s number as the ABI gap: the python binding's struct is 456 B
(amdsmi 6.3.3) and the installed native library writes far past it
(libamd_smi.so.26.2.1). The library returns rc=0 -- it reports success while
overrunning the caller's buffer.

Why it is probabilistic instead of always-fatal: the block comes from the pymalloc
pool (API 'm'), so the overflow lands in the adjacent 512-byte cell. If that cell is
free, nothing visible happens. If it holds a live object, you get a crash somewhere
else, minutes later, with no frame pointing here. That is the exact shape of both a
2-in-11 boot failure and a 1-in-N probe segfault.

Usage
-----
    python3 tools/gfx906/probe-amdsmi-overflow.py --arm measure
    PYTHONMALLOC=debug python3 tools/gfx906/probe-amdsmi-overflow.py --arm control
    PYTHONMALLOC=debug python3 tools/gfx906/probe-amdsmi-overflow.py --arm import

Exits 0 when no overflow is detected, 1 when one is. `control` re-execs itself under
`PYTHONMALLOC=debug` if it is not already set, because without the debug allocator
there is nothing to observe.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

SITE = Path("/root/.pyenv/versions/venv312/lib/python3.12/site-packages")
TREE = Path("/ai/engines/gfx906/unverbraucht/vllm-gfx906-mobydick")
SLACK = 512


def _reexec_debug() -> int:
    env = dict(os.environ, PYTHONMALLOC="debug")
    env.pop("PYTHONPATH", None)  # the pin outranks site-packages
    rc = subprocess.call([sys.executable, *sys.argv], env=env)
    # subprocess reports a signal as a negative return; sys.exit() would mask -6 to
    # 250 and hide the fact that the child was SIGABRT-ed by the debug allocator.
    return 128 - rc if rc < 0 else rc


def arm_measure() -> int:
    sys.path.insert(0, str(SITE))
    import amdsmi.amdsmi_wrapper as W
    from amdsmi import amdsmi_get_processor_handles, amdsmi_init

    amdsmi_init()
    handles = amdsmi_get_processor_handles()
    size = ctypes.sizeof(W.amdsmi_asic_info_t)
    buf = ctypes.create_string_buffer(b"\xAA" * (size + SLACK), size + SLACK)
    info = W.amdsmi_asic_info_t.from_buffer(buf)
    rc = W.amdsmi_get_gpu_asic_info(handles[0], ctypes.byref(info))
    raw = bytes(buf.raw)
    last = max((i for i, b in enumerate(raw) if b != 0xAA), default=-1)
    over = last + 1 - size

    print(f"python sizeof(amdsmi_asic_info_t) : {size} B")
    print(f"native rc                        : {rc}")
    print(f"last byte the call modified      : offset {last + 1}")
    print(f"written PAST the end of the      : {over} B")
    print(f"                                   (poison is 0xAA; the library writes 0x00,")
    print(f"                                    so an unpoisoned tail reads as 'clean')")
    if over > 0:
        print(f"VERDICT: OVERFLOW -- the native struct is >= {last + 1} B, "
              f"the binding allocates {size} B")
    # The struct view aliases a buffer we own; let neither get finalized twice.
    sys.stdout.flush()
    os._exit(1 if over > 0 else 0)


def arm_control(which: str) -> int:
    if os.environ.get("PYTHONMALLOC") != "debug":
        return _reexec_debug()
    sys.path.insert(0, str(SITE))
    import amdsmi
    import amdsmi.amdsmi_wrapper as W
    from amdsmi import amdsmi_get_processor_handles, amdsmi_init

    amdsmi_init()
    handles = amdsmi_get_processor_handles()
    if which == "handles":
        print(f"handles: {len(handles)} (no struct written)")
    elif which == "meminfo":
        total = amdsmi.amdsmi_get_gpu_memory_total(
            handles[0], amdsmi.AmdSmiMemoryType.VRAM
        )
        print(f"memory_total: {total} (struct written, smaller)")
    elif which == "asicinfo":
        info = amdsmi.amdsmi_get_gpu_asic_info(handles[0])
        print(f"asic_info market_name: {info['market_name']!r}")
    print("exited cleanly under the debug allocator")
    sys.stdout.flush()
    os._exit(0)


def arm_shim() -> int:
    """Apply the fork's pad, then make the call that used to abort.

    This is the green half of the control: same library, same binding, same debug
    allocator as `control --call asicinfo` (which aborts), with only the pad added.
    The shim is loaded by file path so the fix can be verified before it is
    deployed to the tree this interpreter would otherwise import.
    """
    import importlib.util

    sys.path.insert(0, str(SITE))
    abi_path = TREE / "vllm" / "platforms" / "amdsmi_abi.py"
    if not abi_path.exists():
        print(f"no shim at {abi_path}", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("amdsmi_abi_probe", abi_path)
    abi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abi)

    import amdsmi
    import amdsmi.amdsmi_wrapper as W
    from amdsmi import amdsmi_get_processor_handles, amdsmi_init

    if os.environ.get("PYTHONMALLOC") != "debug":
        return _reexec_debug()

    print("shim:", abi.install(W))
    amdsmi_init()
    handles = amdsmi_get_processor_handles()
    info = amdsmi.amdsmi_get_gpu_asic_info(handles[0])
    print(f"market_name={info['market_name']!r} "
          f"tgv={info['target_graphics_version']!r} "
          f"sizeof(struct)={__import__('ctypes').sizeof(W.amdsmi_asic_info_t)}")
    print("survived the debug allocator")
    return 0


def arm_import() -> int:
    sys.path.insert(0, str(SITE))
    import vllm.platforms.rocm as rocm  # noqa: F401  (the boot path)

    print(f"imported vllm.platforms.rocm, _GCN_ARCH={rocm._GCN_ARCH!r}")
    print("exited cleanly under the debug allocator")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["measure", "control", "shim", "import"],
                    default="measure")
    ap.add_argument(
        "--tree",
        type=Path,
        default=TREE,
        help="vLLM source tree to load the shim from (default: the main checkout)",
    )
    ap.add_argument(
        "--call",
        choices=["handles", "meminfo", "asicinfo"],
        default="asicinfo",
        help="which amdsmi call the control arm should make",
    )
    args = ap.parse_args()
    if args.arm == "measure":
        return arm_measure()
    if args.arm == "control":
        return arm_control(args.call)
    if args.arm == "shim":
        return arm_shim()
    return arm_import()


if __name__ == "__main__":
    sys.exit(main())
