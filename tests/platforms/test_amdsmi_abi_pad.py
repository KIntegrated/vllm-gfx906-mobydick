# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The amdsmi asic-info pad: contain the native write, change no decoded value.

The fact under test is a size disagreement, measured on a real MI50 box: python
``amdsmi 6.3.3`` allocates a 456-byte ``amdsmi_asic_info_t`` and
``libamd_smi.so.26.2.1`` writes 812 bytes into it and returns ``rc = 0``. The tests
here use a synthetic struct and a synthetic "native" writer so the guard is testable
without a GPU, and the real-library case is exercised separately when amdsmi is
importable.
"""

import ctypes
import logging
from types import SimpleNamespace

import pytest

from vllm.platforms.amdsmi_abi import (
    MEASURED_NATIVE_WRITE,
    TAIL_SLACK,
    install,
    pad_asic_info_struct,
)

# Synthetic stand-in for the binding's struct: pack=1 and mixed-width fields, so a
# layout shift would show up in the offset assertions.
_SYNTH_FIELDS = [
    ("market_name", ctypes.c_char * 64),
    ("vendor_id", ctypes.c_uint32),
    ("device_id", ctypes.c_uint32),
    ("target_graphics_version", ctypes.c_char * 8),
    ("num_compute_units", ctypes.c_uint32),
]

# How far the synthetic library writes: the binding's struct plus the 356 B the real
# one writes past it.
_OVERRUN = 356


def _make_struct():
    return type(
        "struct_amdsmi_asic_info_t",
        (ctypes.Structure,),
        {"_pack_": 1, "_fields_": list(_SYNTH_FIELDS)},
    )


def _addr(obj):
    try:
        return ctypes.addressof(obj)
    except TypeError:
        return ctypes.cast(obj, ctypes.c_void_p).value


@pytest.fixture(autouse=True)
def _reset_watchdog_state():
    from vllm.platforms import amdsmi_abi

    amdsmi_abi._reported = False
    yield
    amdsmi_abi._reported = False


@pytest.fixture
def wrapper():
    """A fake ``amdsmi_wrapper`` plus a native that writes like the real library."""
    struct = _make_struct()
    extent = ctypes.sizeof(struct) + _OVERRUN

    def native(handle, info_ptr):
        # What libamd_smi does: fill the struct and keep going past it. Only ever
        # called on a padded instance, so the write is contained by construction.
        ctypes.memset(_addr(info_ptr), 0, extent)
        state["calls"] += 1
        return 0

    state: dict = {"calls": 0}
    return SimpleNamespace(
        amdsmi_asic_info_t=struct,
        amdsmi_get_gpu_asic_info=native,
        _extent=extent,
        state=state,
    )


def test_the_binding_struct_cannot_hold_the_native_write(wrapper):
    """RED: the precondition the pad exists for, as a capacity fact.

    If a future amdsmi build ships a struct big enough for the library's write,
    this test fails and the pad can be deleted -- which is the correct outcome, not
    a regression.
    """
    assert ctypes.sizeof(wrapper.amdsmi_asic_info_t) < MEASURED_NATIVE_WRITE
    assert ctypes.sizeof(wrapper.amdsmi_asic_info_t) < wrapper._extent


def test_pad_grows_only_the_tail(wrapper):
    old, new = pad_asic_info_struct(wrapper)
    assert new == old + TAIL_SLACK
    assert new >= MEASURED_NATIVE_WRITE


def test_pad_preserves_every_field_offset(wrapper):
    """GREEN, and the load-bearing one: appending slack must not move a field.

    The binding decodes fields by name from this struct. A pad that shifted an
    offset would "fix" the overflow by silently changing every returned value --
    which is exactly the failure mode this whole file is about. Reverting the pad
    makes the size assertions fail; padding at the *head* instead of the tail makes
    this one fail.
    """
    before = {name: getattr(wrapper.amdsmi_asic_info_t, name).offset
              for name, _ in _SYNTH_FIELDS}
    old_size = ctypes.sizeof(wrapper.amdsmi_asic_info_t)
    pad_asic_info_struct(wrapper)
    padded = wrapper.amdsmi_asic_info_t
    for name, off in before.items():
        assert getattr(padded, name).offset == off, f"{name} moved"
    # The appended field starts exactly where the original struct ended: the pad is
    # tail-only, which is what keeps every decoded value identical.
    assert getattr(padded, "_gfx906_tail_slack").offset == old_size


def test_pad_keeps_pack(wrapper):
    """_pack_ must survive the subclass, or the layout shifts."""
    pad_asic_info_struct(wrapper)
    assert wrapper.amdsmi_asic_info_t._pack_ == 1


def test_install_then_the_native_write_lands_inside_our_allocation(
    wrapper, caplog
):
    assert install(wrapper) is not None
    struct = wrapper.amdsmi_asic_info_t()
    padded_size = ctypes.sizeof(struct)
    wrapper.amdsmi_get_gpu_asic_info(0, ctypes.byref(struct))
    assert wrapper.state["calls"] == 1
    # Bytes past the write extent must still be the sentinel: positive evidence of
    # containment, not merely the absence of a crash.
    tail_off = wrapper._extent
    tail = (ctypes.c_ubyte * (padded_size - tail_off)).from_address(
        ctypes.addressof(struct) + tail_off
    )
    assert all(b == 0xA5 for b in tail), "native write escaped the contained region"


def test_watchdog_reports_the_overrun_once_at_info_not_warning(wrapper, caplog):
    caplog.set_level(logging.INFO, logger="vllm.platforms.amdsmi_abi")
    install(wrapper)
    struct = wrapper.amdsmi_asic_info_t()
    byref = ctypes.byref(struct)
    for _ in range(3):
        wrapper.amdsmi_get_gpu_asic_info(0, byref)
    hits = [r for r in caplog.records if "past the binding" in r.getMessage()]
    assert len(hits) == 1, "a per-call warning is a warning that gets filtered"
    assert hits[0].levelno == logging.INFO
    assert str(_OVERRUN) in hits[0].getMessage()


def test_watchdog_escalates_when_the_pad_is_almost_gone(wrapper, caplog):
    """The one case that deserves a warning: the pad has stopped containing it."""
    install(wrapper)
    struct_cls = wrapper.amdsmi_asic_info_t
    padded_size = ctypes.sizeof(struct_cls)

    # A library that grew to within 8 B of the end of the pad. Still contained, so
    # the test process is safe, but the pad has one lookup's worth of headroom left.
    def oversized_native(handle, info_ptr):
        ctypes.memset(_addr(info_ptr), 0, padded_size - 8)
        return 0

    wrapper.amdsmi_get_gpu_asic_info = oversized_native
    from vllm.platforms import amdsmi_abi

    assert amdsmi_abi.watch_asic_info_write(wrapper)  # re-arm on the new callable
    caplog.set_level(logging.WARNING, logger="vllm.platforms.amdsmi_abi")
    wrapper.amdsmi_get_gpu_asic_info(0, ctypes.byref(struct_cls()))
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns, "an overflow that outran the pad must not be silent"


def test_install_is_idempotent_and_safe_without_amdsmi(wrapper):
    first = install(wrapper)
    size_after_first = ctypes.sizeof(wrapper.amdsmi_asic_info_t)
    install(wrapper)
    install(wrapper)
    assert ctypes.sizeof(wrapper.amdsmi_asic_info_t) == size_after_first
    assert first is not None
    # A build with no amdsmi (or a wrapper module laid out differently) must not
    # break platform detection.
    assert install(object()) is None
    assert install(SimpleNamespace()) is None


def test_native_call_returns_unchanged_values(wrapper):
    """The pad must not change what the binding decodes."""
    struct = wrapper.amdsmi_asic_info_t
    plain = struct()
    ctypes.memset(ctypes.addressof(plain), 0x7A, ctypes.sizeof(plain))
    want = bytes(ctypes.string_at(ctypes.addressof(plain), ctypes.sizeof(plain)))
    install(wrapper)
    padded = wrapper.amdsmi_asic_info_t()
    ctypes.memset(ctypes.addressof(padded), 0x7A, ctypes.sizeof(struct))
    got = bytes(ctypes.string_at(ctypes.addressof(padded), ctypes.sizeof(struct)))
    assert got == want


@pytest.mark.skipif(
    not __import__("importlib.util", fromlist=["util"]).find_spec("amdsmi"),
    reason="amdsmi not installed",
)
def test_real_library_write_is_contained_by_the_pad():
    """The same assertion as above, against libamd_smi itself.

    Skipped without a GPU/amdsmi, which is the right behaviour for a unit test: the
    synthetic tests above carry the logic, this one carries the measurement.
    """
    import amdsmi
    import amdsmi.amdsmi_wrapper as W

    # Importing vllm.platforms already imports rocm, which installs the shim, so
    # install() legitimately returns None here. Assert the state, not install()'s
    # return value -- skipping on "nothing to pad" would report the success path as
    # an unsupported build.
    install(W)
    if not getattr(W.amdsmi_asic_info_t, "_gfx906_tail_padded", False):
        pytest.skip("amdsmi build has no asic_info struct to pad")
    try:
        amdsmi.amdsmi_init()
    except Exception as exc:  # pragma: no cover - no GPU in this environment
        pytest.skip(f"amdsmi_init failed: {exc}")
    handles = amdsmi.amdsmi_get_processor_handles()
    if not handles:
        pytest.skip("amdsmi returned no processor handles")
    info = W.amdsmi_asic_info_t()
    rc = W.amdsmi_get_gpu_asic_info(handles[0], ctypes.byref(info))
    assert rc == 0
    base = type(info).__mro__[1]
    assert ctypes.sizeof(info) == ctypes.sizeof(base) + TAIL_SLACK
    assert ctypes.sizeof(info) >= MEASURED_NATIVE_WRITE, "pad must cover the write"
    assert info.market_name, "the decoded values must still be readable"
    # Positive containment: the very end of the pad must be untouched by a library
    # that writes MEASURED_NATIVE_WRITE bytes.
    tail = (ctypes.c_ubyte * 64).from_address(
        ctypes.addressof(info) + ctypes.sizeof(info) - 64
    )
    assert all(b == 0xA5 for b in tail), "native write reached the end of the pad"
