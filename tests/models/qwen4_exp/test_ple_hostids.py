# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential tests for the host-side PLE ngram id path (D1).

``ple_hostids.compute_ngram_ids_host`` is a numpy reimplementation of
``Qwen4ExpNGramEmbedding._compute_ngram_ids``. That makes it a change whose
failure mode is a *wrong answer*, not a crash: an id off by one selects a
different row of a 20-million-row table, the model receives a plausible
embedding of the wrong ngram, and nothing downstream notices. So this file does
not test "does it run". It tests "does it agree, bit for bit, with the function
it replaces", by calling that function.

The structure of the evidence:

* **the oracle is production code.** ``_device_ids`` calls the real
  ``_compute_ngram_ids`` on CPU tensors -- the same arithmetic the GPU runs, with
  no local copy of the math anywhere in this file.
* **the cases the semantics hinge on are named**, not left to a fuzzer to
  stumble into: eos segment boundaries, the drafter's ``-1`` sentinel (which is
  what makes the remainder's *floor* definition load-bearing), int64 wraparound
  in ``token * multiplier``, multi-token spec-decode rows, zero-length requests,
  and a packed workspace far wider than the tokens in it (the real one is
  ``max_total_tokens`` wide and eos-filled, and the host path shortens each row
  to the request's last token -- a shortcut that is only legal if every term
  looks backwards).
* **red controls.** A differential test that cannot fail is worse than no test,
  because it reads as proof. ``_blind_variant`` and ``_trunc_variant`` are
  deliberate near-copies of the host path that ignore segment boundaries and
  truncate instead of flooring respectively. The suite requires that the
  fixtures catch both -- and, to keep the controls honest, requires that each
  control *agrees* on a fixture where its shortcut is harmless.

Everything runs on CPU tensors: no GPU, no model, no checkpoint. Nothing here is
timing-based -- one decode run on this box is not evidence -- and the
performance claim is measured in production by T7 with ``VLLM_GFX906_PLE_PROF``.
"""

import numpy as np
import pytest
import torch

ple = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_layer",
    reason="qwen4_exp AMD model code is not importable in this build",
)
hostids = pytest.importorskip(
    "vllm.models.qwen4_exp.amd.ple_hostids",
    reason="host-side PLE id module is not importable in this build",
)

Embedding = ple.Qwen4ExpNGramEmbedding

EOS = 151645
VOCAB = 151936
NGRAM_VOCAB_BASE = 20_000_000


# --------------------------------------------------------------------- buffers
def _real_buffers(ngram_size: int, heads_per_ngram: int, seed: int = 0):
    """The hash buffers a real Qwen4Exp PLE layer registers, unchanged.

    These are the values production will use, so agreement on them is the case
    that matters. They are also the *benign* case for overflow: with a ~1.5e5
    vocabulary the products stay inside int64, which is precisely why a suite
    made only of real configs would never notice a wraparound mistake. Hence
    ``_extreme_buffers``.

    ``seed`` only picks the layer id, so the fuzzer varies the hash values the
    same way a second PLE layer would; seed 0 is layer 2, the layer this model
    actually builds.
    """
    layer_id = 2 if seed == 0 else seed % 5
    multipliers = Embedding._make_layer_multipliers(
        ngram_size=ngram_size,
        unigram_vocab_size=VOCAB,
        seed=1234,
        ple_dense_layer_id=layer_id,
    )
    sizes, offsets, _total = Embedding._make_vocab_layout(
        ngram_vocab_size_base=NGRAM_VOCAB_BASE,
        ngram_heads=(ngram_size - 1) * heads_per_ngram,
        ple_dense_layer_id=layer_id,
    )
    return (
        np.array(multipliers, dtype=np.int64),
        np.array(sizes, dtype=np.int64),
        np.array(offsets, dtype=np.int64),
    )


def _extreme_buffers(ngram_size: int, heads_per_ngram: int, seed: int = 7):
    """Multipliers drawn from the full unsigned 64-bit range, so some are
    negative once reinterpreted as int64 and every product overflows.

    Not a configuration the model ships: a stress fixture. ``token *
    multiplier`` is the one expression whose result depends on two's-complement
    wrap behaviour, and ``bitwise_xor`` of two such products is the one
    expression whose result depends on both operands' bit patterns. An
    implementation that promoted to float64, or used Python ints and masked
    afterwards, passes every real-config case in this file and is wrong.
    """
    rng = np.random.default_rng(seed)
    heads = (ngram_size - 1) * heads_per_ngram
    multipliers = rng.integers(
        0, np.iinfo(np.uint64).max, ngram_size, dtype=np.uint64
    ).view(np.int64)
    sizes = np.array(
        [NGRAM_VOCAB_BASE + 3 * i + (1 if i % 2 else 2) for i in range(heads)],
        dtype=np.int64,
    )
    offsets = (np.cumsum(sizes) - sizes).astype(np.int64)
    return multipliers.astype(np.int64), sizes, offsets


def _negative_mixed_buffers(ngram_size: int, heads_per_ngram: int, seed: int = 11):
    """Deterministically negative hash values.

    ``mixed`` is a xor of ``token * multiplier`` terms, so its sign is not something
    a test should hope for. With every multiplier negative and every token
    positive, the bigram mix for shift 0 alone is negative, which makes the
    floor-versus-truncate difference unavoidable rather than probabilistic.
    """
    rng = np.random.default_rng(seed)
    heads = (ngram_size - 1) * heads_per_ngram
    multipliers = -(2 * rng.integers(1, 10**6, ngram_size) - 1).astype(np.int64)
    sizes = np.array(
        [NGRAM_VOCAB_BASE + 5 * i + 2 for i in range(heads)], dtype=np.int64
    )
    offsets = (np.cumsum(sizes) - sizes).astype(np.int64)
    return multipliers, sizes, offsets


# --------------------------------------------------------------- the two paths
def _device_ids(
    input_ids,
    query_start_loc,
    ngram_context,
    *,
    multipliers,
    vocab_sizes,
    head_offsets,
    eos_token_id,
    ngram_size,
    heads_per_ngram,
    packed_width=None,
):
    """Call the function being replaced.

    Deliberately no local copy of the arithmetic: the oracle *is* the production
    code. CPU tensors, so it runs anywhere.
    """
    num_reqs = max(int(query_start_loc.size) - 1, 1)
    if packed_width is None:
        packed_width = int(np.diff(query_start_loc).max(initial=0))
    packed = torch.full(
        (num_reqs, max(int(packed_width), 1)), eos_token_id, dtype=torch.int64
    )
    return Embedding._compute_ngram_ids(
        torch.from_numpy(np.ascontiguousarray(input_ids, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(query_start_loc, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(ngram_context, dtype=np.int64)),
        multipliers=torch.from_numpy(np.ascontiguousarray(multipliers, dtype=np.int64)),
        vocab_sizes=torch.from_numpy(np.ascontiguousarray(vocab_sizes, dtype=np.int64)),
        head_offsets=torch.from_numpy(
            np.ascontiguousarray(head_offsets, dtype=np.int64)
        ),
        packed=packed,
        positions=torch.arange(int(np.asarray(input_ids).size), dtype=torch.int64),
        eos_token_id=eos_token_id,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
    ).numpy()


def _host_ids(
    input_ids,
    query_start_loc,
    ngram_context,
    *,
    out=None,
    **kwargs,
):
    return hostids.compute_ngram_ids_host(
        input_ids,
        query_start_loc,
        ngram_context,
        out=out,
        **kwargs,
    )


def _assert_equal(host, device, *, what: str) -> None:
    host = np.asarray(host)
    device = np.asarray(device)
    assert host.shape == device.shape, (
        f"{what}: shape {host.shape} != device {device.shape}"
    )
    if host.size and not np.array_equal(host, device):
        bad = np.argwhere(host != device)
        first = tuple(int(v) for v in bad[0])
        pytest.fail(
            f"{what}: {bad.shape[0]} of {host.size} ids differ, first at "
            f"token/head {first}: host {int(host[first])} "
            f"device {int(device[first])}"
        )


# -------------------------------------------------------------------- fixtures
def _case(tokens_per_req, contexts, *, ngram_size=3):
    """Build ``(input_ids, query_start_loc, ngram_context)`` from per-request rows."""
    ctx_len = ngram_size - 1
    toks: list[int] = []
    qsl = [0]
    for row in tokens_per_req:
        toks.extend(int(t) for t in row)
        qsl.append(len(toks))
    ctx = np.asarray(contexts, dtype=np.int64).reshape(len(tokens_per_req), ctx_len)
    return (
        np.asarray(toks, dtype=np.int64),
        np.asarray(qsl, dtype=np.int64),
        np.ascontiguousarray(ctx),
    )


def _kw(buffers, ngram_size: int, heads: int, eos: int = EOS) -> dict:
    return dict(
        zip(("multipliers", "vocab_sizes", "head_offsets"), buffers),
        eos_token_id=eos,
        ngram_size=ngram_size,
        heads_per_ngram=heads,
    )


# -------------------------------------------------------------- named cases
@pytest.mark.parametrize("ngram_size,heads", [(3, 8), (2, 4), (4, 3), (5, 2)])
def test_matches_device_on_real_buffers(ngram_size, heads):
    """The production case (n=3, h=8) plus the shapes the layer could be built
    with, all on the hash buffers a real layer computes."""
    kwargs = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    ctx_len = ngram_size - 1
    cases = {
        "fresh-decode-single": _case(
            [[42]], [np.full(ctx_len, EOS)], ngram_size=ngram_size
        ),
        "decode-with-context": _case(
            [[9001]], [np.arange(100, 100 + ctx_len)], ngram_size=ngram_size
        ),
        "spec-verify-1-plus-k": _case(
            [[7, 8, 9, 10], [11]],
            [np.arange(50, 50 + ctx_len), np.arange(90, 90 + ctx_len)],
            ngram_size=ngram_size,
        ),
        "mixed-prefill-chunk-and-decode": _case(
            [list(range(200, 208)), [77]],
            [np.arange(60, 60 + ctx_len), np.full(ctx_len, EOS)],
            ngram_size=ngram_size,
        ),
        "empty-request": _case(
            [[], [77, 78]],
            [np.arange(30, 30 + ctx_len), np.arange(40, 40 + ctx_len)],
            ngram_size=ngram_size,
        ),
        "duplicate-tokens": _case(
            [[5, 5, 5, 5]], [np.array([5] * ctx_len)], ngram_size=ngram_size
        ),
    }
    for what, (toks, qsl, ctx) in cases.items():
        label = f"n={ngram_size} h={heads} {what}"
        _assert_equal(
            _host_ids(toks, qsl, ctx, **kwargs),
            _device_ids(toks, qsl, ctx, **kwargs),
            what=label,
        )


def test_eos_inside_prompt_resets_the_segment():
    """A token must be blind to everything at or before the nearest preceding eos.

    ``_shift_precompute``/``_shift_apply`` implement that with a cummax of eos
    positions behind a ``position_in_segment >= shift`` gate. This fixture puts
    an eos *between* two real tokens inside one step's row, so the third token's
    bigram must be (eos, token) and its trigram (eos, eos, token) rather than
    reaching back past the boundary.
    """
    ngram_size, heads = 3, 8
    kwargs = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    toks, qsl, ctx = _case([[101, EOS, 102, 103]], [[EOS, 100]])
    host = _host_ids(toks, qsl, ctx, **kwargs)
    _assert_equal(host, _device_ids(toks, qsl, ctx, **kwargs), what="eos in prompt")

    # The fixture must be doing something. Same row with the eos replaced by a
    # real token must give the third token different ids, or this test is
    # silently exercising the plain path.
    plain, _, _ = _case([[101, 104, 102, 103]], [[EOS, 100]])
    other = _host_ids(plain, qsl, ctx, **kwargs)
    assert not np.array_equal(host[3], other[3]), (
        "the eos fixture changed no id at all: it is not exercising the segment "
        "logic it claims to"
    )
    assert not np.array_equal(host[2], host[3]), (
        "tokens either side of an eos are identical: the segment boundary has "
        "no effect, so the fixture is inert"
    )


def test_sentinel_and_negative_ids_use_floor_remainder():
    """The drafter's ``PADDING_SENTINEL`` (-1) reaches the id math, where
    ``torch.remainder`` floors: non-negative for a positive divisor, even when
    the dividend is negative.

    A host version using C-style truncation agrees everywhere except here, and
    agrees on a *wrong* id. The fold in ``MmapShardedNGramEmbedding.forward``
    does not rescue it: that fold only catches ids outside the table, and a
    truncated remainder lands inside it.
    """
    ngram_size, heads = 3, 8
    kwargs = _kw(_negative_mixed_buffers(ngram_size, heads), ngram_size, heads)
    toks, qsl, ctx = _case([[-1, -1000, 5, EOS, 0]], [[-1, EOS]])
    host = _host_ids(toks, qsl, ctx, **kwargs)
    device = _device_ids(toks, qsl, ctx, **kwargs)
    _assert_equal(host, device, what="negative ids / sentinel")
    assert (host >= 0).all(), (
        "head offsets are non-negative and the remainder floors, so no id may "
        "be negative; a negative id means a truncating remainder"
    )

    # Red control, in both directions: the truncating copy must disagree here
    # (so the fixture discriminates) and agree on all-positive ids (so the
    # control is a faithful copy rather than something that differs at random).
    trunc = _trunc_variant(toks, qsl, ctx, **kwargs)
    assert not np.array_equal(trunc, device), (
        "the negative-id fixture does not distinguish floor from truncating "
        "remainder; strengthen it -- this test is the only thing pinning "
        "remainder semantics for the sentinel path"
    )
    # The harmless case for the control must use *positive* multipliers: with
    # the negative ones above, `mixed` is negative even for positive tokens,
    # which is why that fixture discriminates in the first place.
    positive = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    benign, b_qsl, b_ctx = _case([[101, 102, 103]], [[100, 99]])
    assert np.array_equal(
        _trunc_variant(benign, b_qsl, b_ctx, **positive),
        _device_ids(benign, b_qsl, b_ctx, **positive),
    ), (
        "the truncating control differs even with a non-negative dividend, so "
        "it is not a faithful copy and cannot serve as a control"
    )


def test_int64_overflow_matches_device():
    """``token * multiplier`` overflows int64 with adversarial multipliers; both
    implementations must wrap identically."""
    ngram_size, heads = 3, 8
    kwargs = _kw(_extreme_buffers(ngram_size, heads), ngram_size, heads)
    toks, qsl, ctx = _case(
        [[1 << 20, (1 << 20) + 1], [(1 << 19) - 3]],
        [[1 << 18, 1 << 19], [EOS, 1 << 17]],
    )
    _assert_equal(
        _host_ids(toks, qsl, ctx, **kwargs),
        _device_ids(toks, qsl, ctx, **kwargs),
        what="overflowing multipliers",
    )
    # Pin that the fixture overflows at all. Every token multiplies every
    # multiplier, so max|x| * max|m| is a product the code really computes.
    product = int(np.abs(toks).max()) * int(np.abs(kwargs["multipliers"]).max())
    assert product > (1 << 62), (
        f"largest product is 2^{product.bit_length()}, which fits in int64: the "
        "overflow fixture is not overflowing and proves nothing"
    )


def test_wide_packed_workspace_does_not_change_ids():
    """The production workspace is ``max_total_tokens`` wide and eos-filled, so a
    request's row is mostly padding after its last token. Both the segment scan
    and the shifted gathers look backwards only, so padding cannot influence an
    id -- which is the property the host shortcut (cut the row at the request's
    last token) relies on.

    If the device function ever gains a forward-looking term, this fails instead
    of the host path silently diverging.
    """
    ngram_size, heads = 3, 8
    kwargs = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    toks, qsl, ctx = _case([[9, EOS, 11]], [[EOS, 8]])
    tight = _device_ids(toks, qsl, ctx, packed_width=3, **kwargs)
    wide = _device_ids(toks, qsl, ctx, packed_width=4096, **kwargs)
    _assert_equal(tight, wide, what="tight vs wide workspace (device self-check)")
    _assert_equal(
        _host_ids(toks, qsl, ctx, **kwargs), wide, what="host vs wide workspace"
    )


def test_out_is_written_in_place_like_the_pinned_buffer():
    """In production ``out`` is a numpy view of ``_pinned_ngram_ids``, so the ids
    must land in the caller's buffer with no allocation on the step path, and
    reusing the workspace must be safe (nothing accumulates).
    """
    ngram_size, heads = 3, 8
    kwargs = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    num_heads = (ngram_size - 1) * heads
    toks, qsl, ctx = _case([[5, 6], [7]], [[1, 2], [3, 4]])
    pinned = torch.full((4, num_heads), -12345, dtype=torch.int64)
    view = pinned[:3].numpy()  # exactly `_pinned_ngram_ids[:T].numpy()`
    out = _host_ids(toks, qsl, ctx, out=view, **kwargs)
    assert out is view, "compute_ngram_ids_host must return the caller's buffer"
    assert (pinned[:3].numpy() == out).all(), (
        "ids did not land in the caller's pinned memory"
    )
    assert (pinned[3:].numpy() == -12345).all(), (
        "the write ran past the rows this step owns"
    )
    device = _device_ids(toks, qsl, ctx, **kwargs)
    _assert_equal(out, device, what="pinned-view out")
    _assert_equal(
        _host_ids(toks, qsl, ctx, out=view, **kwargs),
        device,
        what="second call reusing the same workspace",
    )


def test_empty_step_does_not_raise():
    """A step with no rows must not raise and must not produce rows.

    If the device function cannot even build its ``[num_reqs, ...]`` grid at
    zero requests, that is recorded rather than papered over: the engine never
    presents an empty step to PLE, and the caller's fallback covers it either
    way.
    """
    ngram_size, heads = 3, 8
    kwargs = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    toks = np.zeros(0, dtype=np.int64)
    qsl = np.array([0], dtype=np.int64)
    ctx = np.zeros((0, ngram_size - 1), dtype=np.int64)
    host = _host_ids(toks, qsl, ctx, **kwargs)
    assert host.shape == (0, (ngram_size - 1) * heads)
    try:
        device = _device_ids(toks, qsl, ctx, **kwargs)
    except RuntimeError as exc:  # pragma: no cover - documented above
        pytest.skip(f"device path rejects a zero-request step ({exc})")
    _assert_equal(host, device, what="empty step")


@pytest.mark.parametrize(
    "case",
    [
        "context-width",
        "qsl-end",
        "qsl-decreasing",
        "qsl-nonzero-start",
        "multipliers-short",
        "vocab-sizes-short",
        "out-dtype",
        "out-shape",
        "ngram-size",
        "heads",
        "context-rows",
        "context-ndim",
    ],
)
def test_refuses_layouts_it_cannot_prove(case):
    """Each refusal is a layout the host path has not been shown to reproduce,
    and the caller's contract is to fall back to the device path on
    ``ValueError``.

    ``context-width`` is the one that matters: the device code names a token's
    row index with ``columns + ngram_size - 1``, which is only true when the
    context is exactly ``ngram_size - 1`` wide. At any other width the two paths
    read different windows, both look healthy, and the ids are wrong.
    """
    ngram_size, heads = 3, 8
    kwargs = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    args = [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([0, 2, 3], dtype=np.int64),
        np.array([[1, 2], [3, 4]], dtype=np.int64),
    ]
    extra: dict = {}
    if case == "context-width":
        args[2] = np.array([[1, 2, 9], [3, 4, 9]], dtype=np.int64)
    elif case == "qsl-end":
        args[1] = np.array([0, 2, 2], dtype=np.int64)
    elif case == "qsl-decreasing":
        args[1] = np.array([0, 3, 1], dtype=np.int64)
    elif case == "qsl-nonzero-start":
        args[1] = np.array([1, 2, 3], dtype=np.int64)
    elif case == "multipliers-short":
        kwargs["multipliers"] = kwargs["multipliers"][:1]
    elif case == "vocab-sizes-short":
        kwargs["vocab_sizes"] = kwargs["vocab_sizes"][:1]
    elif case == "out-dtype":
        extra["out"] = np.zeros((3, (ngram_size - 1) * heads), dtype=np.float64)
    elif case == "out-shape":
        extra["out"] = np.zeros((2, (ngram_size - 1) * heads), dtype=np.int64)
    elif case == "ngram-size":
        kwargs["ngram_size"] = 1
    elif case == "heads":
        kwargs["heads_per_ngram"] = 0
    elif case == "context-rows":
        args[2] = args[2][:1]
    elif case == "context-ndim":
        args[2] = np.zeros(ngram_size - 1, dtype=np.int64)
    with pytest.raises(ValueError):
        _host_ids(*args, out=extra.get("out"), **kwargs)


# ------------------------------------------------------------- red controls
def _row_and_segment(toks, qsl, ctx, req, *, ngram_size, eos_token_id):
    """The shared per-request setup: the row, its token indices, and
    ``position_in_segment``. Returns ``None`` for a zero-length request.
    """
    ctx_len = ngram_size - 1
    start, end = int(qsl[req]), int(qsl[req + 1])
    if end == start:
        return None
    row = np.concatenate((ctx[req], toks[start:end]))
    idx = np.arange(ctx_len, ctx_len + (end - start), dtype=np.int64)
    eos_pos = np.where(row == eos_token_id, np.arange(row.size, dtype=np.int64), -1)
    previous_eos = np.concatenate(
        (np.array([-1], dtype=np.int64), np.maximum.accumulate(eos_pos[:-1]))
    )
    return row, idx, idx - previous_eos[idx] - 1, start, end


def _blind_shifts(row, idx, position_in_segment, *, ngram_size, eos_token_id):
    """Segment-blind shifts: the tempting shortcut."""
    shifted = [row[idx]]
    for shift in range(1, ngram_size):
        shifted.append(row[np.clip(idx - shift, 0, None)])
    return shifted


def _gated_shifts(row, idx, position_in_segment, *, ngram_size, eos_token_id):
    """What the production code does."""
    shifted = [row[idx]]
    for shift in range(1, ngram_size):
        source = idx - shift
        shifted.append(
            np.where(
                (source >= 0) & (position_in_segment >= shift),
                row[np.clip(source, 0, None)],
                eos_token_id,
            )
        )
    return shifted


def _mix(
    shifted,
    *,
    ngram_size,
    heads_per_ngram,
    multipliers,
    vocab_sizes,
    head_offsets,
    remainder,
):
    """Head layout and hash mix, with the remainder rule injected."""
    heads = (ngram_size - 1) * heads_per_ngram
    out = np.empty((shifted[0].size, heads), dtype=np.int64)
    with np.errstate(over="ignore"):
        for ngram in range(2, ngram_size + 1):
            a = (ngram - 2) * heads_per_ngram
            b = a + heads_per_ngram
            mixed = shifted[0] * multipliers[0]
            for i in range(1, ngram):
                mixed = np.bitwise_xor(mixed, shifted[i] * multipliers[i])
            out[:, a:b] = (
                remainder(mixed[:, None], vocab_sizes[None, a:b])
                + head_offsets[None, a:b]
            )
    return out


def _truncating_remainder(dividend, divisor):
    """C-style remainder: the sign follows the dividend."""
    quotient = np.abs(dividend) // divisor
    quotient = np.where(dividend < 0, -quotient, quotient)
    with np.errstate(over="ignore"):
        return dividend - quotient * divisor


def _variant(
    input_ids,
    query_start_loc,
    ngram_context,
    *,
    blind_shifts,
    truncating_remainder,
    **kwargs,
):
    """A near-copy of the host path with exactly one shortcut switchable.

    Kept in the test file on purpose: it is a control, not production code, and
    it must never be reachable from the engine.
    """
    ngram_size = kwargs["ngram_size"]
    heads = kwargs["heads_per_ngram"]
    num_heads = (ngram_size - 1) * heads
    out = np.empty((int(input_ids.size), num_heads), dtype=np.int64)
    for req in range(len(query_start_loc) - 1):
        setup = _row_and_segment(
            input_ids,
            query_start_loc,
            ngram_context,
            req,
            ngram_size=ngram_size,
            eos_token_id=kwargs["eos_token_id"],
        )
        if setup is None:
            continue
        row, idx, pis, start, end = setup
        shift_fn = _blind_shifts if blind_shifts else _gated_shifts
        rem_fn = _truncating_remainder if truncating_remainder else np.remainder
        out[start:end] = _mix(
            shift_fn(
                row,
                idx,
                pis,
                ngram_size=ngram_size,
                eos_token_id=kwargs["eos_token_id"],
            ),
            ngram_size=ngram_size,
            heads_per_ngram=heads,
            multipliers=kwargs["multipliers"],
            vocab_sizes=kwargs["vocab_sizes"],
            head_offsets=kwargs["head_offsets"],
            remainder=rem_fn,
        )
    return out


def _blind_variant(*args, **kwargs):
    return _variant(*args, blind_shifts=True, truncating_remainder=False, **kwargs)


def _trunc_variant(*args, **kwargs):
    return _variant(*args, blind_shifts=False, truncating_remainder=True, **kwargs)


def test_the_test_itself_can_fail():
    """Red control for the whole approach.

    Both shortcuts must be caught by the named fixtures, and each must agree on
    a fixture where its shortcut is harmless. Without this, "every differential
    test passed" would also be consistent with fixtures too weak to detect the
    thing being tested -- the same mistake as a probe that cannot pass.

    The no-shortcut variant is run through the same machinery and must agree
    with the device function, which pins that the control harness is a faithful
    copy and not a strawman.
    """
    ngram_size, heads = 3, 8
    real = _kw(_real_buffers(ngram_size, heads), ngram_size, heads)
    negative = _kw(_negative_mixed_buffers(ngram_size, heads), ngram_size, heads)

    eos_case = _case([[101, EOS, 102, 103]], [[EOS, 100]])
    fuzzish = _case([[EOS, 5, 6, EOS, 7], [8]], [[EOS, EOS], [9, EOS]])
    negative_case = _case([[-1, 7, 5]], [[EOS, EOS]])
    benign_case = _case([[101, 102, 103]], [[100, 99]])

    for what, case, kwargs in [
        ("eos-in-prompt", eos_case, real),
        ("two-eos-rows", fuzzish, real),
    ]:
        device = _device_ids(*case, **kwargs)
        assert np.array_equal(
            _variant(*case, blind_shifts=False, truncating_remainder=False, **kwargs),
            device,
        ), f"the control harness disagrees with the device on {what}"
        assert not np.array_equal(_blind_variant(*case, **kwargs), device), (
            f"{what} does not catch segment-blind shifts"
        )

    device = _device_ids(*negative_case, **negative)
    assert not np.array_equal(_trunc_variant(*negative_case, **negative), device), (
        "the negative fixture does not catch a truncating remainder"
    )
    benign_device = _device_ids(*benign_case, **real)
    assert np.array_equal(_trunc_variant(*benign_case, **real), benign_device), (
        "the truncating control differs even on a non-negative dividend"
    )
    assert np.array_equal(
        _variant(*benign_case, blind_shifts=False, truncating_remainder=False, **real),
        benign_device,
    ), "the control harness disagrees with the device on the benign fixture"


# ----------------------------------------------------------------------- fuzz
@pytest.mark.parametrize("seed", list(range(24)))
def test_random_agreement(seed):
    """Fuzz over shapes, ngram sizes, eos placement, and all three buffer
    flavours.

    eos is drawn into the token stream deliberately: real prompts contain the
    eos id and the drafter's padding is negative, so the segment and floor
    paths are exercised by construction rather than by luck. ``packed_width`` is
    drawn wider than needed to keep testing the backwards-only property.
    """
    rng = np.random.default_rng(seed)
    ngram_size = int(rng.integers(2, 6))
    heads = int(rng.integers(1, 9))
    num_reqs = int(rng.integers(1, 5))
    flavour = (
        _real_buffers
        if seed % 3 == 0
        else _extreme_buffers
        if seed % 3 == 1
        else _negative_mixed_buffers
    )
    kwargs = _kw(flavour(ngram_size, heads, seed=seed), ngram_size, heads)

    qlens = rng.integers(0, 6, num_reqs)
    while qlens.sum() == 0:
        qlens = rng.integers(0, 6, num_reqs)
    qsl = np.concatenate(([0], np.cumsum(qlens))).astype(np.int64)
    total = int(qlens.sum())
    tokens = rng.integers(0, VOCAB, total).astype(np.int64)
    for pos in rng.choice(total, size=max(1, total // 3), replace=False):
        tokens[int(pos)] = EOS if rng.random() < 0.7 else -1
    ctx = rng.integers(0, VOCAB, (num_reqs, ngram_size - 1)).astype(np.int64)
    for r in range(num_reqs):
        for c in range(ngram_size - 1):
            if rng.random() < 0.3:
                ctx[r, c] = EOS
    ctx = np.ascontiguousarray(ctx)

    _assert_equal(
        _host_ids(tokens, qsl, ctx, **kwargs),
        _device_ids(
            tokens,
            qsl,
            ctx,
            packed_width=int(qlens.max()) + int(rng.integers(0, 33)),
            **kwargs,
        ),
        what=f"seed {seed} n={ngram_size} h={heads} qlens={list(qlens)}",
    )
