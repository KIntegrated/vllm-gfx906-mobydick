# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp family predicate: every spelling, and nothing outside it.

A checkpoint's ``model_type`` is not one string: the family declares
``qwen4_exp`` (wrapper and vision), ``qwen4_exp_text`` (the text sub-config), and
``SpeculativeConfig`` rewrites it to ``qwen4_exp_mtp`` for the drafter. Anything
matching the family by exact string silently misses the other spellings; the
concrete failure that motivated this was Whittle-Qwen-3.8-35B-A3B
(``qwen4_exp_text``) bypassing the compilation guard, which left the PLE n-gram
lookup's blocking device->host copy inside a full cudagraph capture and killed
engine init with ``hipErrorStreamCaptureUnsupported``.
"""

from types import SimpleNamespace

import pytest

from vllm.models.qwen4_exp.config import (
    QWEN4_EXP_MODEL_TYPES,
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
    is_qwen4_exp_config,
    is_qwen4_exp_model_type,
)

# Not-a-family controls: the neighbouring generation (qwen3_5), a longer string
# that shares the prefix, and a non-string.
NON_FAMILY_MODEL_TYPES = ["qwen3_5", "qwen3_5_text", "qwen4_exp_extra", "", "qwen4"]


def test_every_declared_spelling_is_accepted():
    """Adding a config class with a new ``model_type`` must extend the family."""
    declared = {
        config_cls.model_type
        for config_cls in (Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig)
    }
    assert declared <= set(QWEN4_EXP_MODEL_TYPES)
    for model_type in declared:
        assert is_qwen4_exp_model_type(model_type), model_type


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text", "qwen4_exp_mtp"])
def test_drafter_spelling_is_in_the_family(model_type):
    """``qwen4_exp_mtp`` is a rewrite of the same checkpoint, not a new family."""
    assert is_qwen4_exp_model_type(model_type)


@pytest.mark.parametrize("model_type", NON_FAMILY_MODEL_TYPES)
def test_non_family_spellings_are_rejected(model_type):
    assert not is_qwen4_exp_model_type(model_type)


@pytest.mark.parametrize("model_type", NON_FAMILY_MODEL_TYPES)
def test_config_level_check_rejects_non_family(model_type):
    config = SimpleNamespace(model_type=model_type, architectures=["LlamaForCausalLM"])
    assert not is_qwen4_exp_config(config)


def test_config_level_check_accepts_the_family_via_model_type():
    config = SimpleNamespace(model_type="qwen4_exp_text", architectures=[])
    assert is_qwen4_exp_config(config)


def test_config_level_check_falls_back_to_architectures():
    """A config that omits/re-spells ``model_type`` is still the family."""
    config = SimpleNamespace(model_type=None, architectures=["Qwen4ExpForCausalLM"])
    assert is_qwen4_exp_config(config)


def test_config_level_check_tolerates_a_missing_model_type():
    config = SimpleNamespace()
    assert not is_qwen4_exp_config(config)
    assert not is_qwen4_exp_config(None)
