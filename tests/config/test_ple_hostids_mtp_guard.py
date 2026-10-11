# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the VLLM_GFX906_PLE_HOSTIDS / speculative-decoding guard.

This file used to justify the guard with a measurement, and the measurement said
what it said: on 2026-10-01 22:20:51 (boot ``vllm.log.20261001-215123``, MTP with
``num_speculative_tokens=3``, differential mode) all four TP workers reported the
same mismatch, and inverting the n-gram hash (``scripts/invert-mismatch-2220b.py``
-- four heads are residues of one 64-bit value under four ~2e7 coprimes, so CRT
recovers it) produced a host query of ``[760, 1156, 2640, 343]`` against a device
buffer whose row 1 agreed and whose row 2 did not. The conclusion drawn was "no
single compute produces that array", and the guard was written to refuse on it.

The conclusion was wrong, and a guard's reason is not decorative: it is what the
next reader uses to decide whether the guard still applies. **``docs/ple-
multipass-verdict.md``** has the full case. In short:

* the drafter builds no PLE module (``mtp.py:8``; ``mtp.py:314-322`` passes
  ``input_ids=None``, which ``model.py:296`` would raise on, and that string is in
  zero logs from this box), so there is no second author for the buffer,
* boot #14 measured **one PLE op call per step** (14.4 comparisons/s against 15.0
  engine steps/s) and **8** gate-skips in 45 minutes where the multi-pass story
  predicts ~25,000 per worker,
* ``[0, 2, 2, 2]``, the qsl the mismatch reported, cannot be a pass of a
  3-request step and is the padded form of an earlier B=1 step
  (``input_batch.py:159``), and
* the mismatch disappeared when ``18c119f010`` made ``ple_query_start_loc``
  monotone -- the workspace scatter's ``searchsorted()`` was undefined on a
  non-monotone buffer, so positions landed in the wrong row. The rows were
  **misplaced**, not authored by different passes. Boot #14: 55,921 comparisons,
  0 mismatches.

What the guard is for now is the thing that remains true: ``HOSTIDS`` under a
drafter has never been booted, and the two modes concede different things when the
override opens them. Mode 2 compares and cannot take the staged branch, so its
worst outcome is a wrong verdict. Mode 1 serves the host's rows to the model, so a
step that should have been refused ships whatever the pinned buffer already held --
which under a captured CUDA graph is the previous step's rows. So both modes are
refused by default and both can be opened with ``VLLM_GFX906_ALLOW_HOSTIDS_MTP=1``,
and the tests below pin the parts that must not drift:

* an unknown mode value is refused even with the override, because the override
  permits a named risk;
* the message no longer states the false mechanism as fact, and says so, with the
  pointer to the verdict doc (``test_the_reason_corrects_the_old_mechanism``) --
  this is the regression guard for the narrative itself;
* the two modes do not read the same paragraph about what they risk;
* the dead route stays unadvertised: dropping ``--speculative-config`` was measured
  closed on both counts (no drafter means no host token source, so every step
  refuses with ``no-draft-source``, and drafter-off boots do not compile:
  ``ConstraintViolationError: input_ids specialized to 4096``,
  ``logs/vllm.log.20261001-234242`` / ``logs/vllm.log.20261002-065905``);
* and the predicate is still called from ``VllmConfig.__post_init__``, because a
  correct predicate nobody calls protects nothing.
"""

import ast
import inspect

import pytest

vllm_config_module = pytest.importorskip(
    "vllm.config.vllm",
    reason="vllm.config.vllm is not importable in this build",
)


@pytest.fixture(autouse=True)
def _knob_off_by_default(monkeypatch):
    monkeypatch.delenv(vllm_config_module.ENV_PLE_HOSTIDS, raising=False)
    monkeypatch.delenv(vllm_config_module.ENV_ALLOW_HOSTIDS_MTP, raising=False)


def _conflict(spec_tokens, *, knob="1", environ=None):
    env = {} if environ is None else dict(environ)
    if knob is not None:
        env[vllm_config_module.ENV_PLE_HOSTIDS] = knob
    return vllm_config_module.ple_hostids_mtp_conflict(spec_tokens, environ=env)


def _override():
    return {vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: "1"}


def test_mtp_with_the_knob_on_is_refused():
    reason = _conflict(3)
    assert reason is not None
    assert vllm_config_module.ENV_PLE_HOSTIDS in reason
    # The way out has to be in the message, because the refusal is the only thing
    # an operator sees, and an unnamed way out becomes a deleted knob.
    assert vllm_config_module.ENV_ALLOW_HOSTIDS_MTP in reason


def test_differential_mode_is_refused_without_the_override():
    # Mode 2 is the arm that HAS been run under MTP (boot #14), and it is still
    # refused by default: it earns its override by never serving what it compared,
    # which is a claim about the code that deserves an opt-in rather than a default.
    assert _conflict(3, knob="2") is not None


def test_one_speculative_token_still_needs_the_override():
    # k=1 is not a smaller risk than k=3 here; the question was never "how many
    # passes", it was "has this combination been booted", and it has not.
    assert _conflict(1) is not None
    assert _conflict(1, knob="2") is not None


def test_the_reason_corrects_the_old_mechanism():
    """The guard must not keep teaching a mechanism this tree does not have.

    A refusal text is documentation that gets pasted into tickets and boot notes.
    If it still asserted "the buffer is a patchwork of passes" as fact, the next
    reader would re-derive a false conclusion from it -- which is exactly how the
    wrong arm got specified in the first place. So the text quotes the old reason
    only to refute it, and names where the refutation lives.
    """
    reason = _conflict(3)
    assert reason is not None
    assert "FALSE on this tree" in reason
    assert "patchwork of passes" in reason, "the old reason must be named to be killed"
    assert "docs/ple-multipass-verdict.md" in reason
    assert "18c119f010" in reason, "the fix that made the cited mismatch disappear"
    # And the thing that is actually open, stated as unmeasured rather than wrong.
    assert "has not been measured on this hardware" in reason


def test_the_reason_says_why_and_names_the_way_out():
    reason = _conflict(3)
    assert reason is not None
    # An operator reading only this line must learn which mode is dangerous, the
    # override that opens it, and the route that does not work.
    assert "--speculative-config" in reason
    assert vllm_config_module.ENV_ALLOW_HOSTIDS_MTP in reason


def test_mode_one_and_mode_two_get_different_advice():
    # The modes fail differently: mode 1 serves, mode 2 only looks. One message for
    # both would have to be wrong about one of them.
    serving, diff = _conflict(3), _conflict(3, knob="2")
    assert "SERVES" in serving
    assert "never booted" in serving
    assert "only compares" in diff
    assert "wrong verdict" in diff
    assert serving != diff


def test_message_does_not_advertise_the_dead_route():
    # The first version of this guard told the operator to drop
    # --speculative-config. That route was measured closed on both counts, and a
    # remedy is part of the guard: asserting the correction is what keeps the
    # regression out of the next boot.
    for reason in (_conflict(3), _conflict(3, knob="2")):
        assert reason is not None
        assert "Do NOT drop --speculative-config" in reason
        assert "no-draft-source" in reason
        assert "ConstraintViolationError" in reason
        assert "Run the differential on a boot without" not in reason


def test_override_opens_both_modes_and_records_what_mode_one_concedes(caplog):
    """The override is one switch for two very different concessions.

    Mode 2 opening quietly is fine -- it is boot #14's arm, run and measured. Mode 1
    opening must leave a line in the boot log, because "the knob was on" is not
    evidence of what was accepted by turning it on, and the next reader of a FULL
    boot needs the sentence that says a served-row cliff was being watched for.
    """
    assert _conflict(3, knob="2", environ=_override()) is None
    caplog.clear()
    assert _conflict(3, knob="1", environ=_override()) is None
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "SERVED to the model" in text
    assert "never booted" in text
    assert "PREVIOUS step's rows" in text

    # Mode 2 with the override says nothing: it is not the arm with the cliff.
    caplog.clear()
    assert _conflict(3, knob="2", environ=_override()) is None
    quiet = "\n".join(r.getMessage() for r in caplog.records)
    assert "SERVED to the model" not in quiet


def test_unknown_mode_is_refused_even_with_the_override():
    # The override permits a named risk. A mode value that is neither 1 nor 2 names
    # nothing -- the code path is unknown -- so a typo must not become a green light
    # for the never-booted serving arm.
    for bad in ("02", "true", "3", "1x"):
        reason = _conflict(3, knob=bad, environ=_override())
        assert reason is not None, f"mode {bad!r} was let through by the override"
        assert "neither 1" in reason


def test_no_speculative_decoding_is_untouched():
    # No drafter, so this guard has nothing to say. This is NOT the configuration to
    # run the knob in -- the module docstring and the guard's own text both explain
    # that a drafter-off boot compares nothing and does not compile.
    assert _conflict(0) is None
    assert _conflict(0, knob="1") is None
    assert _conflict(0, knob="2") is None


def test_the_guard_is_silent_when_the_knob_is_off():
    # The overwhelmingly common case: an MTP boot with no experimental knob.
    assert _conflict(3, knob=None) is None
    assert _conflict(3, knob="0") is None
    assert _conflict(3, knob="") is None


def test_escape_hatch_must_be_exactly_one():
    # Anything but "1" keeps the protection, including a typo of it -- in either
    # mode, since the override now opens both.
    for knob in ("1", "2"):
        for value in ("0", "true", "yes", ""):
            assert (
                _conflict(
                    3,
                    knob=knob,
                    environ={vllm_config_module.ENV_ALLOW_HOSTIDS_MTP: value},
                )
                is not None
            ), f"{knob=} {value=}"


def test_helper_is_wired_into_config_validation():
    """The predicate has to be *called* from config validation.

    A guard whose predicate is correct and whose call site was dropped in a rebase
    is invisible until the wrong-answer boot it was written for happens again.
    """
    tree = ast.parse(inspect.getsource(vllm_config_module))
    calls_by_function: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "ple_hostids_mtp_conflict"
            ):
                calls_by_function[node.name] = calls_by_function.get(node.name, 0) + 1
    assert calls_by_function.get("__post_init__") == 1, calls_by_function
