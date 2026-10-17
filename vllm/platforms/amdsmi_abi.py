# SPDX-License-Identifier: Apache-2.0
"""Contain ``amdsmi_get_gpu_asic_info``'s native write inside our allocation.

The problem, measured on the box this exists for (2026-10-02)
------------------------------------------------------------
The ``amdsmi`` python binding allocates the asic-info struct itself::

    # amdsmi/amdsmi_interface.py:1650  (amdsmi 6.3.3)
    asic_info_struct = amdsmi_wrapper.amdsmi_asic_info_t()      # ctypes.sizeof -> 456
    amdsmi_wrapper.amdsmi_get_gpu_asic_info(handle, ctypes.byref(asic_info_struct))

The installed native library (``libamd_smi.so.26.2.1``) writes **812 bytes** into
that 456-byte object and returns ``rc = 0``. The binding and the library disagree
about the struct's size, so the caller's buffer is 356 bytes too small and the
library does not notice. On this box the same skew shows up in the decoded values,
which is how the mismatch was suspected before it was measured: ``market_name`` is
correct (``'AMD Instinct MI50/MI60'``) while ``device_id``/``asic_serial`` decode to
``'N/A'``, ``num_compute_units`` to ``0`` and ``target_graphics_version`` to
``'gfx0'``.

Why it survived weeks instead of crashing on every boot
-------------------------------------------------------
The block comes from the pymalloc pool, so the overflow lands in the *adjacent
512-byte cell*. A free cell is invisible; a live object becomes a crash somewhere
else, minutes later, with no frame pointing here. Observed consequences on this
box: ~2-in-11 failures of MTP + CUDA-graph boots, and a 1-in-N segfault of a
host-only benchmark that merely imports ``vllm``.

Why a pad and not a different arch source
-----------------------------------------
``vllm/platforms/rocm.py`` calls this at **module level** (``_GCN_ARCH =
_get_gcn_arch()``), so every process that imports vllm takes the overflow, and the
second call site (``get_device_name``) feeds tuned-kernel name lookups -- changing
what those calls *return* would change which tuned config is selected, which is a
serving-path behavior change. Appending slack changes no offset, no decoded value
and no return value: it only makes the allocation big enough to hold what the
library already writes. ``pad_asic_info_struct`` therefore preserves the prefix
layout byte-for-byte (asserted in the tests) and the watchdog turns any *future*
native growth into a log line instead of a silent corruption.

Measured with ``PYTHONMALLOC=debug`` (``Fatal Python error: _PyMem_DebugRawFree:
bad trailing pad byte``) and reproduced by ``tools/gfx906/probe-amdsmi-overflow.py``.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Measured native write extent on the box this was found on, in bytes.
MEASURED_NATIVE_WRITE = 812

#: Appended after the binding's fields. Generous relative to the measurement so a
#: larger struct in a future library build still lands inside our allocation -- and
#: so the watchdog below has untouched bytes to check.
TAIL_SLACK = 2048

#: Sentinel the watchdog poisons the slack with, and looks for after the call.
_SENTINEL = 0xA5

_PAD_ATTR = "_gfx906_tail_padded"


def _addr(obj: Any) -> int:
    """Address of a ctypes instance **or** of a ``ctypes.byref`` of one.

    The binding calls the native function with ``ctypes.byref(struct)``, and
    ``ctypes.addressof`` refuses a byref -- so an address helper that only handles
    instances silently does nothing on every real call.
    """
    try:
        return ctypes.addressof(obj)
    except TypeError:
        return ctypes.cast(obj, ctypes.c_void_p).value


def pad_asic_info_struct(wrapper: Any) -> Optional[tuple[int, int]]:
    """Rebind ``wrapper.amdsmi_asic_info_t`` to a tail-padded subclass.

    Returns ``(old_size, new_size)`` when the pad was applied, ``None`` when there
    was nothing to do (no such struct in this ``amdsmi`` build, or already padded).

    The subclass appends one trailing field, so every original field keeps its
    offset and the values the binding decodes are unchanged.
    """
    orig = getattr(wrapper, "amdsmi_asic_info_t", None)
    if orig is None or getattr(orig, _PAD_ATTR, False):
        return None

    pack = getattr(orig, "_pack_", None)
    native_size = ctypes.sizeof(orig)

    class PaddedAmdsmiAsicInfo(orig):  # type: ignore[misc,valid-type]
        # ctypes *appends* a subclass's _fields_ to the base's, so declaring only
        # the slack field puts it exactly where the original struct ended and keeps
        # every inherited offset. Re-listing the base's fields here would duplicate
        # them (and the test asserting the slack's offset catches that).
        _fields_ = [("_gfx906_tail_slack", ctypes.c_ubyte * TAIL_SLACK)]
        _gfx906_tail_padded = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Poison the slack so the watchdog can tell "the library wrote here"
            # from "nothing wrote here". The library writes zeros, so an
            # unpoisoned tail would read as containment either way.
            try:
                ctypes.memset(
                    ctypes.addressof(self) + native_size, _SENTINEL, TAIL_SLACK
                )
            except Exception:  # pragma: no cover - poisoning is best effort
                pass

    if pack is not None:
        # Layout must not shift: the binding set _pack_ = 1 on the original, and a
        # subclass that forgets it would repad the appended field.
        PaddedAmdsmiAsicInfo._pack_ = pack

    old_size, new_size = native_size, ctypes.sizeof(PaddedAmdsmiAsicInfo)
    if new_size <= MEASURED_NATIVE_WRITE:
        logger.warning(
            "amdsmi asic-info struct padded to %d B, still below the measured "
            "native write of %d B; raise TAIL_SLACK",
            new_size,
            MEASURED_NATIVE_WRITE,
        )

    wrapper.amdsmi_asic_info_t = PaddedAmdsmiAsicInfo
    return old_size, new_size


_reported = False


def watch_asic_info_write(wrapper: Any) -> bool:
    """Wrap the native call to measure how far the library writes past the struct.

    Padding *contains* the overflow; this *watches* it. The write past the
    binding's 456-byte struct is expected and contained, so it is reported once at
    info level, not as a warning per call -- a warning that fires on every lookup
    is a warning that gets filtered. It escalates to a warning only when the write
    approaches the end of the pad, which is the case where the pad has stopped
    working and the corruption is back.
    """
    inner = getattr(wrapper, "amdsmi_get_gpu_asic_info", None)
    struct = getattr(wrapper, "amdsmi_asic_info_t", None)
    if (
        inner is None
        or struct is None
        or not getattr(struct, _PAD_ATTR, False)
        or getattr(inner, "_gfx906_watchdog", False)
    ):
        return False

    native_size = ctypes.sizeof(struct) - TAIL_SLACK

    def wrapped(processor_handle: Any, info: Any) -> int:
        rc = inner(processor_handle, info)
        try:
            tail_off = native_size
            raw = (ctypes.c_ubyte * TAIL_SLACK).from_address(
                _addr(info) + tail_off
            )
            overrun = 0
            for byte in raw:
                if byte != _SENTINEL:
                    overrun += 1
                else:
                    break
            # NOTE: a *contiguous* run from the start of the slack, so this is a
            # lower bound -- a library that happens to write a 0xA5 byte mid-write
            # stops the count early (on this box: 340 reported vs 356 measured by
            # last-modified-offset). Cheap and monotone, which is what an escalation
            # check needs; exact extent is tools/gfx906/probe-amdsmi-overflow.py's job.
            global _reported
            if overrun and not _reported:
                _reported = True
                if overrun > TAIL_SLACK - 64:
                    logger.warning(
                        "amdsmi_get_gpu_asic_info wrote %d B past the binding's "
                        "%d B struct -- only %d B of the %d B pad left. Raise "
                        "TAIL_SLACK in vllm/platforms/amdsmi_abi.py before this "
                        "overflow escapes its allocation again.",
                        overrun,
                        native_size,
                        TAIL_SLACK - overrun,
                        TAIL_SLACK,
                    )
                else:
                    logger.info(
                        "amdsmi_get_gpu_asic_info writes %d B past the binding's "
                        "%d B struct (rc=%d); contained by the %d B pad. "
                        "vllm/platforms/amdsmi_abi.py",
                        overrun,
                        native_size,
                        rc,
                        TAIL_SLACK,
                    )
        except Exception:  # the watchdog must never break the call
            logger.debug("amdsmi asic-info watchdog could not inspect the tail",
                         exc_info=True)
        return rc

    wrapped._gfx906_watchdog = True  # type: ignore[attr-defined]
    wrapper.amdsmi_get_gpu_asic_info = wrapped
    return True


def install(wrapper: Any) -> Optional[str]:
    """Pad the struct and arm the watchdog. Safe to call more than once.

    Returns a short description of what was done, or ``None`` if this ``amdsmi``
    build has nothing to patch.
    """
    padded = pad_asic_info_struct(wrapper)
    watched = watch_asic_info_write(wrapper)
    if padded is None and not watched:
        return None
    if padded is not None:
        old, new = padded
        return (
            f"amdsmi asic-info struct padded {old} B -> {new} B "
            f"(native write measured at {MEASURED_NATIVE_WRITE} B); "
            f"watchdog {'armed' if watched else 'not armed'}"
        )
    return "amdsmi asic-info watchdog armed (struct already padded)"


__all__ = [
    "MEASURED_NATIVE_WRITE",
    "TAIL_SLACK",
    "install",
    "pad_asic_info_struct",
    "watch_asic_info_write",
]
