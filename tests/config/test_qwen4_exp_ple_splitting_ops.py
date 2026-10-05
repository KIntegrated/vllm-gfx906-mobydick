# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Qwen4Exp PLE n-gram lookup split.

The PLE lookup gathers over a *host-resident* n-gram table: it copies the ids
device->host with a blocking copy so the host can read the mmap'd shards, then
copies the result back through pinned buffers. HIP refuses that inside a stream
capture, so the op has to stay outside the captured graph pieces. vLLM does that
by appending the op to ``splitting_ops`` for Qwen4Exp models.

That guard used to compare ``hf_config.model_type == "qwen4_exp"`` exactly, which
silently missed checkpoints that spell it ``qwen4_exp_text`` (e.g.
``logic65/Whittle-Qwen-3.8-35B-A3B``). The op then stayed in the captured graph
and engine init died with ``hipErrorStreamCaptureUnsupported``. These tests pin
the family-wide match for every spelling we know of.
"""

import json
import shutil
from pathlib import Path

import pytest

from vllm.config import ModelConfig, VllmConfig

PLE_NGRAM_OP = "vllm::qwen4_exp_amd_ple_ngram_embedding"

# The oldest spelling in the family, kept as the "existing behaviour" control.
_FAMILY_SPELLINGS = ["qwen4_exp", "qwen4_exp_text"]

_WHITTLE = Path(
    "/biglocal/cache/hf/hub/models--logic65--Whittle-Qwen-3.8-35B-A3B"
    "/snapshots/926d1dc370db65c9b905b5578989600954bbc2b7"
)

pytestmark = pytest.mark.skipif(
    not (_WHITTLE / "config.json").is_file(),
    reason="needs a cached Qwen4Exp checkpoint's config.json to build a valid "
    "ModelConfig (set up the Whittle snapshot to run these)",
)


def _checkpoint_dir_for(tmp_path: Path, model_type: str) -> Path:
    """A config dir that is the real checkpoint's config.json with one spelling."""
    target = tmp_path / model_type
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(
        json.dumps({**_read_config(), "model_type": model_type})
    )
    for name in ("tokenizer_config.json", "tokenizer.json", "vocab.json"):
        source = _WHITTLE / name
        if source.is_file():
            shutil.copy(source, target / name)
    return target


def _read_config() -> dict:
    return json.loads((_WHITTLE / "config.json").read_text())


@pytest.mark.parametrize("model_type", _FAMILY_SPELLINGS)
def test_ple_lookup_is_split_out_of_the_captured_graph(tmp_path, model_type):
    model_dir = _checkpoint_dir_for(tmp_path, model_type)
    model_config = ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        tokenizer_mode="auto",
        trust_remote_code=False,
    )
    assert model_config.architecture == "Qwen4ExpForCausalLM"

    vllm_config = VllmConfig(model_config=model_config)

    splitting_ops = vllm_config.compilation_config.splitting_ops
    assert splitting_ops is not None
    assert PLE_NGRAM_OP in splitting_ops, (
        f"{PLE_NGRAM_OP} must be split out for model_type={model_type!r}; "
        "otherwise the blocking device->host copy inside the PLE lookup runs "
        "inside a cudagraph capture and engine init fails"
    )


def test_repeated_configs_do_not_duplicate_the_split(tmp_path):
    """The append must not accumulate across VllmConfig builds (class-level list)."""
    model_dir = _checkpoint_dir_for(tmp_path, "qwen4_exp_text")
    seen = []
    for _ in range(3):
        model_config = ModelConfig(
            model=str(model_dir),
            tokenizer=str(model_dir),
            tokenizer_mode="auto",
            trust_remote_code=False,
        )
        vllm_config = VllmConfig(model_config=model_config)
        seen.append(list(vllm_config.compilation_config.splitting_ops))

    assert all(count == 1 for ops in seen for count in [ops.count(PLE_NGRAM_OP)])
