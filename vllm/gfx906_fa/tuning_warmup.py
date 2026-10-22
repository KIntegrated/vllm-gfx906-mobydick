# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Record the gfx906 FA launch config for the shapes this engine will run.

The tuning table (``tuning.py``) exists so the launch parameters have a place to
live per shape; this is the step that makes a boot state what it chose, and
skips the work when every shape already has a record. Nothing here measures a
kernel -- the recorded values are the static rules, which are probe-verified
against the C++ (``nc2_effective`` / ``kv_split_default``) -- so a cold boot and
a warm boot must produce identical inference.
"""

from __future__ import annotations

import logging
import os

from vllm.gfx906_fa.tuning import (
    TuningTable,
    cache_path,
    gqa_ratio,
    identity,
    static_config,
    workload_key,
)

logger = logging.getLogger(__name__)

# Shapes the engine is expected to present: decode (Sq=1), MTP verify (Sq=k+1
# padded to 2/4/8) at single and batched concurrency, plus one prefill-sized
# shape. Kept small on purpose: this records *what the rules choose*, it is not
# a measurement sweep (that is ``docs/gfx906/_fa_tune.py``).
SHAPES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 1),
    (2, 4),
    (4, 1),
    (8, 1),
    (1568, 1),
)
SEQ_K_BUCKETS: tuple[int, ...] = (4096, 65536)

_done = False


def reset_for_tests() -> None:
    """Allow a test to exercise the cold path more than once."""
    global _done
    _done = False


def warmup_launch_config(
    *,
    device: str,
    dtype: str,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int = 0,
) -> None:
    """Record the static config for the standard shape set, once per process."""
    global _done
    if _done:
        return
    _done = True
    if os.environ.get("VLLM_GFX906_FA_TUNING_RECORDS", "1") == "0":
        return
    try:
        gqa = gqa_ratio(num_heads, num_kv_heads)
    except ValueError:
        # Non-MHA/GQA geometry (e.g. an unusual MLP-ish attention variant):
        # nothing sensible to record, and this must never break a boot.
        return

    ident = identity(device, dtype, num_heads, num_kv_heads, head_size, block_size)
    path = cache_path(ident)
    table = TuningTable.load(path, ident)
    recorded = 0
    for seq_q, batch in SHAPES:
        for seq_k in SEQ_K_BUCKETS:
            key = workload_key(seq_q, seq_k, batch)
            if key in table.records:
                continue
            table.records[key] = static_config(seq_q, gqa, batch=batch)
            recorded += 1
    if recorded:
        table.save(path)
    logger.info(
        "gfx906 FA launch-config records: %d shapes (%d recorded, %d reused) "
        "for %s %s Hq=%d Hkv=%d D=%d at %s",
        len(table.records),
        recorded,
        len(table.records) - recorded,
        device,
        dtype,
        num_heads,
        num_kv_heads,
        head_size,
        path,
    )
