# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""The QSA selection width must be rejected at construction, not mid-serving.

`_C.top_k_per_row_decode` stages ``2 * topK * int32`` of dynamic shared memory and
does not opt in to a larger dynamic allocation, so gfx906's 64 KiB per-block LDS
caps it at topK 8192 (UPR-3/#37 fails closed inside the op). v0.32.0 removed
upstream's config-time {512, 2048} width check, so an over-wide `indexer_budget`
is now a legal configuration -- it must be refused where the indexer is built.
"""

import pytest

from vllm.models.qwen4_exp.family import (
    device_lds_bytes,
    max_decode_topk,
    validate_selection_width,
)

GFX906_LDS = 65536  # 64 KiB per block / per CU
MI300_LDS = 160 * 1024


@pytest.mark.parametrize(
    ("lds", "expected"),
    [(GFX906_LDS, 8192), (MI300_LDS, 20480), (49152, 6144)],
)
def test_max_decode_topk_follows_the_device_budget(lds: int, expected: int) -> None:
    assert max_decode_topk(lds) == expected


@pytest.mark.parametrize("width", [512, 2048, 8151, 8192])
def test_widths_within_the_budget_are_accepted(width: int) -> None:
    validate_selection_width(width, GFX906_LDS)


@pytest.mark.parametrize("width", [12288, 16384, 65536])
def test_widths_above_the_budget_are_rejected(width: int) -> None:
    with pytest.raises(ValueError, match="decode top-k budget"):
        validate_selection_width(width, GFX906_LDS)


def test_rejection_names_the_budget_and_the_fix() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_selection_width(65536, GFX906_LDS)
    message = str(excinfo.value)
    assert "topK <= 8192" in message
    assert "indexer_budget" in message


def test_the_same_width_is_legal_on_a_bigger_lds_device() -> None:
    """Width 16384 fits a 160 KiB-LDS device and not gfx906."""
    validate_selection_width(16384, MI300_LDS)
    with pytest.raises(ValueError, match="decode top-k budget"):
        validate_selection_width(16384, GFX906_LDS)


def test_device_lds_bytes_reads_the_current_device() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a device")
    assert device_lds_bytes() > 0
