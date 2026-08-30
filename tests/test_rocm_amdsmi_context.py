# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Guards for the with_amdsmi_context cleanup contract.

On gfx906-native ROCm 7.14 builds, amdsmi_init() can return success with
0 processor handles after torch import (the state ROCmPlatform's
get_device_name GCN-arch fallback is designed for), in which case
amdsmi_shut_down() raises AMDSMI_STATUS_NOT_INIT. The wrapper must not
let the cleanup failure mask a query that succeeded — a wrapped call
whose function returned normally must return its value, not crash.
This crashed a vLLM worker (Nemotron-H, TP=2 + EP, first mamba SSD
decode config lookup) on 2026-08-30.
"""

from unittest.mock import patch

import vllm.platforms.rocm as rocm


def test_wrapped_result_survives_shut_down_failure():
    def boom():
        raise RuntimeError("32 | AMDSMI_STATUS_NOT_INIT")

    with (
        patch.object(rocm, "amdsmi_init", create=True),
        patch.object(rocm, "amdsmi_shut_down", side_effect=boom, create=True),
    ):
        assert rocm.with_amdsmi_context(lambda: "AMD_GFX906")() == "AMD_GFX906"


def test_wrapped_failure_still_propagates_when_shut_down_also_fails():
    def broken():
        raise ValueError("underlying query failed")

    def boom():
        raise RuntimeError("32 | AMDSMI_STATUS_NOT_INIT")

    with (
        patch.object(rocm, "amdsmi_init", create=True),
        patch.object(rocm, "amdsmi_shut_down", side_effect=boom, create=True),
    ):
        try:
            rocm.with_amdsmi_context(broken)()
        except ValueError as error:
            assert str(error) == "underlying query failed"
        else:
            raise AssertionError("wrapped failure did not propagate")
