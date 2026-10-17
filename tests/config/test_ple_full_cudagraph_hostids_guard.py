# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the PLE / full-cudagraph guard in ``vllm/config/vllm.py``.

Background, because the guard's *reason* is what decides whether it still applies.
The guard downgrades ``FULL`` to ``PIECEWISE`` for Qwen4Exp because the host-resident
PLE n-gram lookup does a blocking D2H inside the forward, which HIP cannot capture
(``hipErrorStreamCaptureUnsupported``). It did this unconditionally for the whole
model family and never consulted ``VLLM_GFX906_PLE_HOSTIDS`` -- the knob whose entire
purpose is to delete that D2H -- so the configuration that makes full graphs legal was
silently denied the thing it exists to buy. ``logs/vllm.log.20261015-173705`` is that
boot: ``GRAPHMODE=full`` requested, ``cg_mode=FULL`` never once printed, and the only
witness was a ``logger.warning_once`` line.

This file pins four things, in descending order of how much they cost to lose:

1. ``HOSTIDS=1`` **plus** ``PLE_HOST_STAGE_UNDER_FULL`` keeps FULL. That is the fix.
2. ``HOSTIDS=1`` **without** it downgrades. This is the dangerous pairing rather than
   the harmless one -- the graph captures the PLE op, staging is then declined by
   choice, and "declined under a captured graph" means the same rows replay every
   step while ``stale_steps`` stays 0 because nothing calls it a stale replay. Boot
   ``logs/vllm.log.20261008-083018`` measured 59.21 tok/s that way and it looked like
   a +19 % win. A guard that keyed on ``HOSTIDS`` alone would hand this out.
3. The predicate is still called from ``__post_init__``, and the keep-FULL branch is
   ordered before the escape hatch, which is ordered before the downgrade. A correct
   predicate nobody calls protects nothing; a correct predicate called in the wrong
   order is silently overridden by the hatch.
4. The downgrade keeps the exact substring the box's gates grep
   (``downgrading cudagraph_mode to PIECEWISE``), and the keep-FULL message must not
   contain it -- a message that echoes the pattern a gate greps turns a prediction
   into a finding, which is a mistake this project has now made twice
   (``vllm.py:2276``'s own warning, and the MoE witness guard).

The predicate is loaded from source rather than imported: this repo's ``vllm`` package
needs compiled extensions to import, and the guard is pure ``os.environ`` logic, so
importing the world to test twelve lines of it would make the test unrunnable in a
source-only worktree. The ``exec`` here runs the shipped text of the function, so a
mutation of the shipped text is a mutation of what is tested -- see the red controls.
"""

import ast
import os
from pathlib import Path

import pytest

CONFIG_SRC = Path(__file__).resolve().parents[2] / "vllm" / "config" / "vllm.py"

ENV_MODE = "VLLM_GFX906_PLE_HOSTIDS"
ENV_STAGE = "VLLM_GFX906_PLE_HOST_STAGE_UNDER_FULL"


def _source() -> str:
    return CONFIG_SRC.read_text(encoding="utf-8")


def _constants(source: str) -> dict[str, str]:
    """The env-var names the predicate reads, taken from the shipped assignments.

    Keyed by the *identifier* in ``vllm/config/vllm.py`` (``ENV_PLE_HOSTIDS``), whose
    *value* is the environment variable (``VLLM_GFX906_PLE_HOSTIDS``). Reading the
    value off disk rather than hardcoding it is the point: if someone renames the
    variable the runtime reads, this test stops matching and says so, instead of
    quietly testing a knob nothing sets.
    """
    wanted = {"ENV_PLE_HOSTIDS": ENV_MODE, "ENV_PLE_HOST_STAGE_UNDER_FULL": ENV_STAGE}
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            # Keyed by the identifier, because this dict is the namespace the
            # extracted function is exec'd in: its body names ENV_PLE_HOSTIDS, not
            # the string that name holds.
            found[target.id] = ast.literal_eval(node.value)
    assert found == wanted, (
        "the guard's env-var names moved or changed value; the tests and the "
        f"runtime would be talking about different knobs: {found}"
    )
    return found


def _load(text: str, consts: dict[str, str]):
    namespace: dict = {"os": os, **consts}
    exec(text, namespace)  # noqa: S102 - the text is read off disk, see module docstring
    return namespace["ple_hostids_full_cudagraph"]


@pytest.fixture(scope="module")
def predicate_and_source():
    source = _source()
    tree = ast.parse(source)
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "ple_hostids_full_cudagraph"
        ),
        None,
    )
    if fn is None:
        pytest.fail("ple_hostids_full_cudagraph is gone from vllm/config/vllm.py")
    fn_src = ast.get_source_segment(source, fn)
    return _load(fn_src, _constants(source)), fn_src, source


# ---------------------------------------------------------------- 1. the fix


def test_stage_on_mode1_keeps_full(predicate_and_source):
    pred, _, _ = predicate_and_source
    keep, why = pred({ENV_MODE: "1", ENV_STAGE: "stage"})
    assert keep is True, why
    assert "pinned" in why, why  # says *why* the D2H is gone, not just that it is


@pytest.mark.parametrize("value", ["stage", "STAGE", " stage ", "eager", "0-off"])
def test_mirrors_the_runtime_which_only_compares_against_off(
    predicate_and_source, value
):
    """model_state.py tests ``== "off"`` and nothing else.

    If the guard started accepting a narrower set of values than the runtime acts
    on, the pair would disagree about a booted configuration and the log would
    print a reason that is not true of the running engine.
    """
    pred, _, _ = predicate_and_source
    keep, why = pred({ENV_MODE: "1", ENV_STAGE: value})
    assert keep is True, f"{value!r} is not 'off', so the runtime stages: {why}"


# --------------------------------------------- 2. the pairings that must downgrade


def test_mode1_without_staging_downgrades_and_says_what_it_would_do(
    predicate_and_source,
):
    """The default of ``PLE_HOST_STAGE_UNDER_FULL`` is ``off``.

    So this is the shape of ``HOSTIDS=1`` typed on its own, and the message has to
    name the failure it is preventing -- a frozen embedding replayed every step --
    not just say "not supported".
    """
    pred, _, _ = predicate_and_source
    for env in ({ENV_MODE: "1"}, {ENV_MODE: "1", ENV_STAGE: "off"}):
        keep, why = pred(env)
        assert keep is False, f"{env} must not keep FULL"
        assert "stale_steps" in why and "frozen" in why, why
        assert ENV_STAGE in why, why  # names the knob that fixes it


@pytest.mark.parametrize("stage", ["off", "stage", "unset"])
def test_mode2_never_keeps_full(predicate_and_source, stage):
    """Mode 2 exists to compare against the device gather, so the read stays."""
    pred, _, _ = predicate_and_source
    env = {} if stage == "unset" else {ENV_STAGE: stage}
    env[ENV_MODE] = "2"
    keep, why = pred(env)
    assert keep is False, f"mode 2 kept FULL with {stage=}: {why}"
    assert "device gather" in why, why


@pytest.mark.parametrize("mode", [None, "0", "", "3", "yes"])
def test_every_other_mode_value_downgrades(predicate_and_source, mode):
    """Including values this project has not invented yet.

    An unrecognised mode must not be read as the one mode that is safe under FULL,
    the same way ``ple_hostids_mtp_conflict`` refuses an unknown mode even when its
    override is set.
    """
    pred, _, _ = predicate_and_source
    env = {ENV_STAGE: "stage"}
    if mode is not None:
        env[ENV_MODE] = mode
    keep, why = pred(env)
    assert keep is False, f"mode={mode!r} kept FULL: {why}"


def test_the_real_environment_is_not_read_when_one_is_passed(predicate_and_source):
    """The guard is injectable so it can be answered both ways.

    ``ple_hostids_async_conflict`` established this: a guard that can only be asked
    once, in the environment the test happens to run in, is a guard that has never
    been observed to pass.
    """
    pred, _, _ = predicate_and_source
    os.environ[ENV_MODE] = "1"
    os.environ[ENV_STAGE] = "stage"
    try:
        keep, _ = pred({})  # empty environ, not the process one
        assert keep is False, "passed an empty environ, got the process one"
    finally:
        del os.environ[ENV_MODE], os.environ[ENV_STAGE]


# ------------------------------------------------ 3. the wiring, not just the logic


def test_the_predicate_is_called_from_post_init_in_the_right_order(
    predicate_and_source,
):
    _, _, source = predicate_and_source
    tree = ast.parse(source)
    vllmconfig = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VllmConfig"
    )
    post_init = next(
        n
        for n in vllmconfig.body
        if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
    )
    body = ast.get_source_segment(source, post_init) or ""
    assert "ple_hostids_full_cudagraph()" in body, (
        "the config guard no longer calls the predicate -- a correct predicate "
        "nobody calls protects nothing"
    )
    keep_at = body.index("if keep_full:")
    hatch_at = body.index("VLLM_GFX906_QWEN4_EXP_ALLOW_FULL_CUDAGRAPH")
    down_at = body.index("downgrading cudagraph_mode to PIECEWISE")
    assert keep_at < hatch_at < down_at, (
        "ordering drifted: the escape hatch or the downgrade now runs ahead of the "
        "staging answer, so a legal configuration gets overridden"
    )


def test_downgrade_message_survives_and_keep_message_does_not_echo_it(
    predicate_and_source,
):
    """Both halves matter, and they are the same lesson from opposite directions.

    ``scripts/verify-serving-arm.sh`` and ``steady-probe.sh`` grep
    ``downgrading cudagraph_mode``; if that sentence is reworded the gate reports a
    clean boot that was downgraded. Conversely the keep-FULL sentence must not
    contain the string, or the gate finds its own needle in a log that downgraded
    nothing -- which is precisely how ``vllm.py:2276`` once made ``grep -c
    hipErrorStreamCaptureUnsupported`` return 3 on a boot with zero HIP errors.
    """
    _, _, source = predicate_and_source
    assert source.count("downgrading cudagraph_mode to PIECEWISE") == 1, (
        "the substring gates grep for vanished, or was duplicated into a second "
        "message that a gate would also match"
    )
    keep_block = source.split("if keep_full:")[1].split("elif (")[0]
    assert "downgrading cudagraph_mode" not in keep_block, keep_block


# --------------------------------------------------------------- red controls
# A green test suite that cannot go red has proved nothing. Each control mutates
# the shipped text in the one way the corresponding claim denies, and requires the
# predicate to answer differently.


def test_red_control_removing_the_staging_check_would_pass_the_dangerous_pair(
    predicate_and_source,
):
    _, fn_src, source = predicate_and_source
    mutated = fn_src.replace(
        "    if stage_off:", "    if False:  # RED CONTROL", 1
    )
    assert mutated != fn_src, "the mutation did not apply; the source moved"
    pred = _load(mutated, _constants(source))
    keep, _ = pred({ENV_MODE: "1"})  # HOSTIDS=1, staging at its default
    assert keep is True, (
        "the control did not flip: the staging check was never load-bearing, so "
        "the test above is not testing what it claims"
    )


def test_red_control_removing_the_mode_check_would_pass_the_differential_mode(
    predicate_and_source,
):
    _, fn_src, source = predicate_and_source
    mutated = fn_src.replace(
        '    if mode != "1":', "    if False:  # RED CONTROL", 1
    )
    assert mutated != fn_src, "the mutation did not apply; the source moved"
    pred = _load(mutated, _constants(source))
    keep, _ = pred({ENV_MODE: "2", ENV_STAGE: "stage"})
    assert keep is True, (
        "the control did not flip: mode 2 was never actually refused here"
    )
