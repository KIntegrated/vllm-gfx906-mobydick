# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the async-scheduling / GDN ordering guard.

The guard exists because of a specific failure on this box: with
`--async-scheduling` the V2 model runner sorts batch rows by *planned* token
count while the GDN metadata builder counts decodes by `query_len == 1` and
peels them off the front of the batch. When a chunked prefill's final chunk is
1 token the two predicates disagree, and the peel writes recurrent state for
the wrong request. It killed all four TP workers ~40 minutes into a boot.

The behaviour that matters here is that the failure becomes a startup log line
(default path) or a startup exception (explicit path) instead of an engine death
40 minutes later -- so these tests pin the *decision*, not the prose:

* a GDN model under the V2 runner is flagged, with a reason that says why;
* a model without the marker, an unknown model, and the V1 runner are untouched
  (a guard that fires broadly is a guard that gets turned off globally);
* `VLLM_GFX906_ALLOW_ASYNC_GDN=1` disables the guard, because the roadmap's next
  async experiment has to be runnable without editing code;
* and the helper is actually called from `VllmConfig.__post_init__` twice --
  once to refuse, once to default off. A correct predicate nobody calls is the
  most likely way for this guard to stop working silently.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest

vllm_config_module = pytest.importorskip(
    "vllm.config.vllm",
    reason="vllm.config.vllm is not importable in this build",
)

GDN_LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]
PURE_ATTN_LAYER_TYPES = ["full_attention"] * 4


def _cfg(layer_types=None, *, text_layer_types=None, use_v2=True, model=True):
    """A stand-in for VllmConfig carrying only what the predicate reads."""
    hf_config = SimpleNamespace(layer_types=layer_types)
    hf_text_config = SimpleNamespace(layer_types=text_layer_types)
    model_config = (
        None
        if not model
        else SimpleNamespace(hf_config=hf_config, hf_text_config=hf_text_config)
    )
    return SimpleNamespace(model_config=model_config, use_v2_model_runner=use_v2)


@pytest.fixture(autouse=True)
def _guard_enabled_by_default(monkeypatch):
    monkeypatch.delenv(vllm_config_module.ENV_ALLOW_ASYNC_GDN, raising=False)


def test_gdn_model_under_v2_runner_is_flagged():
    reason = vllm_config_module.gdn_async_ordering_risk(
        _cfg(text_layer_types=GDN_LAYER_TYPES)
    )
    assert reason is not None
    assert "linear-attention" in reason


def test_pure_attention_model_is_untouched():
    assert (
        vllm_config_module.gdn_async_ordering_risk(
            _cfg(text_layer_types=PURE_ATTN_LAYER_TYPES)
        )
        is None
    )


def test_composite_config_is_read_from_the_text_sub_config():
    # A VL/audio wrapper keeps layer_types on text_config and has none at the
    # top level. Reading only hf_config would silently miss every such model --
    # including the one this guard was written for.
    cfg = _cfg(layer_types=None, text_layer_types=GDN_LAYER_TYPES)
    assert vllm_config_module.gdn_async_ordering_risk(cfg) is not None


def test_model_without_layer_types_is_untouched():
    # "unknown" must not disable a feature: the guard would then fire on models
    # it has no evidence about, and the first false positive is what teaches
    # people to set the escape hatch permanently.
    assert vllm_config_module.gdn_async_ordering_risk(_cfg()) is None


def test_v1_model_runner_is_untouched():
    # The sort predicate that disagrees with the builder lives in the V2 runner.
    assert (
        vllm_config_module.gdn_async_ordering_risk(
            _cfg(text_layer_types=GDN_LAYER_TYPES, use_v2=False)
        )
        is None
    )


def test_no_model_config_is_untouched():
    assert vllm_config_module.gdn_async_ordering_risk(_cfg(model=False)) is None


def test_escape_hatch_disables_the_guard(monkeypatch):
    monkeypatch.setenv(vllm_config_module.ENV_ALLOW_ASYNC_GDN, "1")
    assert (
        vllm_config_module.gdn_async_ordering_risk(
            _cfg(text_layer_types=GDN_LAYER_TYPES)
        )
        is None
    )


def test_escape_hatch_must_be_exactly_one(monkeypatch):
    monkeypatch.setenv(vllm_config_module.ENV_ALLOW_ASYNC_GDN, "0")
    assert (
        vllm_config_module.gdn_async_ordering_risk(
            _cfg(text_layer_types=GDN_LAYER_TYPES)
        )
        is not None
    )


def test_tuple_layer_types_are_understood():
    # HF configs are not always lists.
    cfg = _cfg(layer_types=tuple(GDN_LAYER_TYPES))
    assert vllm_config_module.gdn_async_ordering_risk(cfg) is not None


def test_helper_is_wired_into_config_validation():
    """The predicate has to be *called* from the async-scheduling resolution.

    Two call sites are expected inside `VllmConfig.__post_init__`: one in the
    explicitly-enabled branch (refuse), one in the `async_scheduling is None`
    branch (default off with a warning). Zero or one means the guard has quietly
    stopped protecting one of the two ways async can turn on.
    """
    tree = ast.parse(inspect.getsource(vllm_config_module))
    calls_by_function: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "gdn_async_ordering_risk"
            ):
                calls_by_function[node.name] = calls_by_function.get(node.name, 0) + 1
    assert calls_by_function.get("__post_init__") == 2, calls_by_function
