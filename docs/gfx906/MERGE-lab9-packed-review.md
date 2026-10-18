# Self-review — LAB-9 packed KV layout, ready-to-merge check

Verdict: **mergeable as an opt-in feature.** The layout is numerically
identical to the alias path, the flag-off path is byte-identical, the suite is
green in both modes, and attribution is in place. Two defects were found in this
review and fixed before this document (`page_size_padded` fail-closed, README
attribution); the remaining items are notes, not blockers.

## What the review checked

**Correctness surface**
- `customize_spec` guards fail closed for every unsupported shape: asymmetric
  K/V head dims, head dims needing padding, non-2-byte KV dtypes, and (added in
  this review) a `page_size_padded` hint from the skip-quant path. That last one
  was silently dropped before — the packed branch always wrote
  `page_size_padded=None`, so a model combining `--kv-cache-dtype-skip-layers`
  with the flag would have lost the page reconciliation without a word. It now
  raises, with a test.
- The cell width is re-derived independently in the pool (`customize_spec` →
  `state_content_bytes`) and in the impl (`_ensure_packed_views`), and the impl
  asserts they agree ("customize_spec() and the pool allocator are out of
  lockstep"). A mismatch — e.g. the flag set in one process but not another —
  is therefore a loud failure, not silent corruption. This is the right
  fail-closed shape for a layout switch driven by an env var.
- `_ensure_packed_views` caches on `(data_ptr, shape, stride)` and keeps
  `self._packed_src` alive; retaining the source tensor is what makes the
  pointer identity safe, since the pool address cannot be reused while the
  reference is held. Worth a comment in the code if this file is refactored.
- The 5D `get_kv_cache_shape` return carries a leading `2` (the legacy K||V
  marker) while the packed pool is sized from `state_content_bytes`. **No
  in-tree caller uses that shape** (`grep -rn get_kv_cache_shape vllm/` → none;
  only our own tests), so nothing double-counts today. The hazard is
  documented in the function; keep it that way if a future allocator starts
  reading it, since `numel(shape) != pool bytes` under the packed layout.

**Numerics**
- Kernel A/B, 8 shapes (`benchmarks/kernels/gfx906/fa_packed_layout_bench.py`,
  interleaved per rep): outputs **bit-identical** (`max|Δ| = 0`), latency
  **−2.9…−9.8 %**.
- PPL, 35B, 12 prompts: 15.9584 (off) vs 15.9635 (on), **0 top-20 misses** both.
- The two non-causal tests (bidirectional batch vs torch ref; freed-block
  sliding window) now run on **both** layouts in the default run, so the packed
  read path has behavioural coverage, not just byte-layout coverage.

**Behaviour and config**
- Model-level decode at B=1/16k is neutral (57.1 → 57.2 t/s); prefix caching
  behaves identically (counters unchanged in both arms — note: "no change",
  not "reuse verified", since the hit counter read 0 in both).
- Hybrid models: works, but block size 2096 → 2672 and capacity +0.4 % — hence
  opt-in and excluded from the planned default (#58 / LAB-10).

**Hygiene**
- Flag-off path returns the spec object untouched (`is` identity asserted in a
  test); default stays off.
- Suites: 143 passed with the flag off; 103 passed + 40 skipped with it on (the
  skips are the alias-internals tests and the padding suite, each with a stated
  reason).
- Attribution: author preserved on the feature commit (`phoenixclyde`,
  `cherry-pick -x`), PR #43 credited in the commit body and now in the README's
  "Ported optimizations" section; the bench and the tests are ours.

## Open, non-blocking

1. The flag is an **env var**, so it does not appear in config dumps — promoting
   it to a `CacheConfig` field would make bug reports self-describing. Same for
   the planned default.
2. `bench_w4.py` (ported with the kernels) is still unrun.
3. Hybrid support / the aligner work is #58, explicitly measure-first.
