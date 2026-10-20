# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Qwen4Exp family matching, independent of one ``model_type`` spelling.

The family presents as ``qwen4_exp``, ``qwen4_exp_text``, ``qwen4_exp_vision``
depending on the checkpoint, and ``SpeculativeConfig`` rewrites a family
checkpoint's ``model_type`` to ``qwen4_exp_mtp``. Anything that must recognise
*the family* rather than one spelling has to go through these helpers: an exact
comparison against a single spelling silently misses the rest, and a missed match
in the compilation guard leaves the PLE n-gram lookup -- a blocking device->host
copy into pinned memory -- inside a full cudagraph capture, which kills engine
init with ``hipErrorStreamCaptureUnsupported``.

This module exists because upstream v0.32.0 deleted
``vllm/models/qwen4_exp/config.py`` (Qwen4Exp configs now come from
``transformers``), while the fork's family-match helpers were introduced in
`e3b6542393` and are still used by ``vllm/config/speculative.py``. It is
deliberately dependency-free: the strings are the stable contract, and the
installed ``transformers`` may predate the Qwen4Exp config classes.
"""

from __future__ import annotations

from typing import Any

try:  # transformers >= 5.16.1 defines the family's config classes
    from transformers import (  # noqa: F401
        Qwen4ExpConfig,
        Qwen4ExpTextConfig,
    )

    QWEN4_EXP_CONFIG_CLASSES: tuple[type, ...] = (
        Qwen4ExpConfig,
        Qwen4ExpTextConfig,
    )
except ImportError:  # pragma: no cover - local venv may pin an older transformers
    QWEN4_EXP_CONFIG_CLASSES = ()

QWEN4_EXP_MODEL_TYPES: tuple[str, ...] = (
    "qwen4_exp",
    "qwen4_exp_text",
    # SpeculativeConfig rewrites a family checkpoint's model_type to this.
    "qwen4_exp_mtp",
)


def is_qwen4_exp_model_type(model_type: object) -> bool:
    """Whether a ``model_type`` string belongs to the Qwen4Exp family."""
    return isinstance(model_type, str) and model_type in QWEN4_EXP_MODEL_TYPES


def is_qwen4_exp_config(hf_config: object) -> bool:
    """Whether a resolved HF config belongs to the Qwen4Exp family.

    ``model_type`` is the primary signal; the declared ``architectures`` entry is
    the fallback for a config that omits or re-spells it. Only a known container
    type is iterated, so an odd ``architectures`` value in a third-party config
    cannot raise here -- this runs for every model that is built, not just this
    family's.
    """
    model_type = getattr(hf_config, "model_type", None)
    if is_qwen4_exp_model_type(model_type):
        return True
    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, str):
        return architectures.startswith("Qwen4Exp")
    if not isinstance(architectures, list | tuple | set | frozenset):
        return False
    return any(
        isinstance(arch, str) and arch.startswith("Qwen4Exp") for arch in architectures
    )


def uses_ngram_embedding(hf_config: object) -> bool:
    """Whether this config builds the host-resident PLE n-gram lookup.

    Mirrors the model state's own gate (``ple_layer_ids``), which is also what
    decides whether the PLE layers -- and therefore
    ``vllm::qwen4_exp_amd_ple_ngram_embedding`` -- exist at all. Callers that
    reason about that op must ask this as well as the family question: a family
    checkpoint without ``ple_layer_ids`` never emits the op, so splitting it out
    of the graph would be pure cost.
    """
    return bool(getattr(hf_config, "ple_layer_ids", None))


def needs_ple_ngram_split(hf_config: object, *, is_rocm: bool) -> bool:
    """Whether this config's PLE n-gram lookup must stay out of the captured graph.

    All three questions are necessary: the family, whether the checkpoint uses
    the host-resident memory at all, and whether this is the platform whose
    implementation cannot be captured. Only the ROCm path gathers the mmap'd host
    table inside a single op with a *blocking* device->host copy, which HIP
    rejects inside a stream capture; the CUDA path prefetches through streams
    over a device-side table and registers no such op.
    """
    return (
        is_rocm and is_qwen4_exp_config(hf_config) and uses_ngram_embedding(hf_config)
    )


def model_type_of(config: Any) -> str | None:
    """Convenience accessor used by callers that only need the string."""
    value = getattr(config, "model_type", None)
    return value if isinstance(value, str) else None

def _torch_device_props():
    import torch

    return torch.cuda.get_device_properties(torch.cuda.current_device())


def device_lds_bytes() -> int:
    """Per-block LDS budget of the current device, in bytes.

    The decode selection kernel (`_C.top_k_per_row_decode`) stages
    ``2 * topK * int32`` of dynamic shared memory without opting in to a larger
    dynamic allocation, so the request is capped by the device's per-block LDS.
    PyTorch exposes the opt-in field on some builds only, hence the fallback: on
    gfx906 both read 64 KiB, which is also the hardware's per-CU limit.
    """
    props = _torch_device_props()
    return int(
        getattr(props, "shared_memory_per_block_optin", None)
        or props.shared_memory_per_multiprocessor
    )


def max_decode_topk(lds_bytes: int | None = None) -> int:
    """Largest selection width the decode kernel can serve on this device.

    Mirrors the guard added to the op itself (UPR-3/#37). Kept here as well so a
    configuration that cannot run is rejected at construction with a message that
    names the budget, instead of aborting a serving step later.
    """
    if lds_bytes is None:
        lds_bytes = device_lds_bytes()
    return max(1, int(lds_bytes) // (2 * 4))


def validate_selection_width(width: int, lds_bytes: int | None = None) -> None:
    """Raise if a QSA selection width cannot be served by the decode kernel."""
    limit = max_decode_topk(lds_bytes)
    if width > limit:
        lds = lds_bytes if lds_bytes is not None else device_lds_bytes()
        raise ValueError(
            f"QSA selection width {width} exceeds this device's decode top-k "
            f"budget (topK <= {limit} for {lds} B of per-block LDS): the kernel "
            "stages 2*topK*int32 of dynamic shared memory and cannot serve a "
            "wider row. Lower indexer_budget (or raise indexer_compress_ratio) "
            "so that indexer_budget // indexer_compress_ratio <= "
            f"{limit}."
        )
