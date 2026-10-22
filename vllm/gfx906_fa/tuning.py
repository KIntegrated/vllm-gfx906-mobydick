# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Shape-keyed FA launch-configuration records for gfx906.

The gfx906 FA kernels choose three launch parameters that trade parallelism
against traffic: GQA head packing (`nc2`), the KV split factor, and whether the
gather path clips its bounds check. Those choices currently live in
``csrc/gfx906_fa/gfx906_fa.cpp`` as process-wide environment defaults
(``GFX906_FA_NC2``, ``GFX906_FA_KVSPLIT``, ``GFX906_FA_GATHER_CLIP``), baked
from hand measurements at a handful of shapes.

This module is the Python half of moving them to a per-shape table with a
static fallback, following the cache design upstream uses for ROCm segmented
attention (vLLM PR #59132): a tuning *identity* (device/dtype/head geometry)
keys a persisted record set, and a lookup returns a tuned config or defers to
the static rule. Nothing here measures a kernel; it records what a boot chose
and what is still untuned, which is the prerequisite for measuring anything.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RECORDS_VERSION = 1


@dataclass(frozen=True)
class FAConfig:
    """Launch configuration for one FA workload shape."""

    nc2: int
    kvsplit: int
    clip: bool


def gqa_ratio(num_heads: int, num_kv_heads: int) -> int:
    """Query heads per KV head."""
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError(
            f"num_heads={num_heads} is not a multiple of num_kv_heads={num_kv_heads}"
        )
    return num_heads // num_kv_heads


def static_config(
    seq_q: int, gqa: int, *, batch: int = 1, direct: bool = False
) -> FAConfig:
    """The rule the C++ defaults encode today.

    Mirrors ``gfx906_fa_effective_nc2`` for nc2 (the C++ default 8, the
    ``8 -> 2 -> 1`` GQA downgrade, and the decode-only clamp: head packing is
    only validated at the decode tile, so prefill keeps nc2=1),
    ``fa_kv_split_default`` (32 for verify-sized query blocks, 16 for decode),
    the paged-direct path's batch-aware decode clamp (``fa_paged_kv_split_default``:
    8/8/5/2/2 for B=1/2/3/4/8, measured on MI50 — that path's grid is
    batch*heads_q blocks, so its decode optimum is not the gather path's 16),
    and the gather clip default. Any change here must be matched there, or the
    fallback silently diverges from the kernels that actually run;
    ``tests/gfx906_fa/test_tuning.py`` pins the kv_split half against the C++
    ``kv_split_default`` probe.

    Not modelled: ``fa_apply_kv_split_budget`` (a byte-budget clamp on the
    partial buffers that can force 1 for real prefill) and any explicit
    ``GFX906_FA_KVSPLIT`` override — both are runtime decisions, not defaults.
    """
    # Head packing needs the GQA ratio to divide evenly: 8, else the packed-by-2
    # fallback if the ratio is even, else unpacked. Only at the decode tile --
    # prefill keeps the legacy unpacked path.
    if gqa % 8 == 0:
        nc2 = 8
    elif gqa % 2 == 0:
        nc2 = 2
    else:
        nc2 = 1
    if seq_q > 2:
        nc2 = 1
    if direct and seq_q < 4:
        kvsplit = max(2, min(8, 16 // max(1, batch)))
    else:
        kvsplit = 32 if seq_q >= 4 else 16
    return FAConfig(nc2=nc2, kvsplit=kvsplit, clip=True)


def workload_key(seq_q: int, seq_k: int, batch: int) -> str:
    """Key a workload by the dimensions the launch choice depends on."""
    return f"sq{seq_q}-sk{seq_k}-b{batch}"


def identity(
    device: str,
    dtype: str,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
) -> dict[str, Any]:
    """Identify the tuning universe a record set is valid for."""
    return {
        "version": RECORDS_VERSION,
        "device": device,
        "dtype": dtype,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "block_size": block_size,
    }


def digest(ident: dict[str, Any]) -> str:
    """Stable hash of an identity (key order must not matter)."""
    blob = json.dumps(ident, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def cache_dir() -> Path:
    root = os.environ.get("VLLM_CACHE_ROOT") or os.path.expanduser("~/.cache/vllm")
    return Path(root) / "gfx906_fa"


def cache_path(ident: dict[str, Any]) -> Path:
    return cache_dir() / f"{digest(ident)}.json"


@dataclass
class TuningTable:
    """Records for one identity. A mismatched identity yields an empty table."""

    ident: dict[str, Any]
    records: dict[str, FAConfig]

    @classmethod
    def load(cls, path: Path, ident: dict[str, Any]) -> TuningTable:
        """Load records for `ident`, discarding a foreign or stale file."""
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls(ident=ident, records={})
        if data.get("identity") != ident:
            return cls(ident=ident, records={})
        records = {key: FAConfig(**cfg) for key, cfg in data.get("records", {}).items()}
        return cls(ident=ident, records=records)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "identity": self.ident,
            "records": {key: asdict(cfg) for key, cfg in self.records.items()},
        }
        with _locked(path):
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp.replace(path)

    def select(self, workload: str, fallback: FAConfig) -> tuple[FAConfig, bool]:
        """Return the tuned config for `workload`, or `fallback` if untuned.

        The bool says which one was used, so a caller can report coverage
        instead of silently inheriting the static rule.
        """
        tuned = self.records.get(workload)
        return (tuned, True) if tuned is not None else (fallback, False)

    def untuned(self, workloads: list[str]) -> list[str]:
        return [w for w in workloads if w not in self.records]


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialise writers so TP rank 0 cannot truncate another rank's save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    with open(lock, "w") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:  # non-POSIX: best effort, records are advisory
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            except ImportError:
                pass


def consistent_across_ranks(local: dict[str, Any], group: Any) -> bool:
    """Whether every rank in `group` derived the same identity.

    Upstream's segmented-attention tuner refuses to share records when ranks
    disagree (and tunes them independently instead); same hazard here: a
    per-rank record set keyed by a different geometry would hand one rank
    another rank's launch config.
    """
    import torch.distributed as dist

    try:
        gathered: list[dict[str, Any]] = [None] * dist.get_world_size(group)  # type: ignore[list-item]
        dist.all_gather_object(gathered, local, group=group)
    except Exception:
        return True  # no collective available: do not block on it
    return all(item == local for item in gathered)
