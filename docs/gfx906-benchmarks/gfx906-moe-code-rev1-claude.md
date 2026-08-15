# Code review — gfx906 MoE phase 1 + phase 2 changes

Reviewer: Claude Sonnet 4.6 · date: 2026-08-15  
Scope: all code changes from commit `85eacaeed9` (phase 1 kernel) through
`c2f6454ce2` (phase 2 close), covering:
- `csrc/rocm/moe_q_gemm_gfx906.cu`
- `vllm/model_executor/layers/fused_moe/experts/gfx906_w4a16_moe.py`
- `tests/kernels/moe/test_gfx906_moe_gemm.py`
- `benchmarks/kernels/gfx906/bench_moe_gemm_gfx906.py`

Severity levels: **BUG** (silent wrong output or crash), **RISK** (wrong under
specific conditions, or correctness depends on an unverified assumption),
**PERF** (measurable inefficiency), **MINOR** (style, robustness, or future
maintenance concern).

---

## BUG-1. `atomic_add_pk2_f16` writes to an unaligned address when `n` is not 4-aligned

**File:** `moe_q_gemm_gfx906.cu:57–71`

```c
void atomic_add_pk2_f16(half* addr, half2 v01) {
  unsigned* addr_u = reinterpret_cast<unsigned*>(addr);
  unsigned old = *addr_u;
  ...
  unsigned prev = atomicCAS(addr_u, old, sum.u);
```

`atomicCAS` on a 32-bit word requires 4-byte alignment. `addr = c + out_row *
size_n + n` where `n = offset_n + t * N_PER_THREAD`. With `N_PER_THREAD=2`,
each thread owns 2 consecutive halves. Thread `t` writes to column
`n = offset_n + t*2`. `offset_n = blockIdx.y * BLOCK_KN_SIZE * 2`. Since
`BLOCK_KN_SIZE=256` and each half is 2 bytes, `offset_n` is always a multiple
of `256*2*2 = 1024 bytes` — 4-byte aligned. Thread `t` then writes to
`offset_n + t*2` halves = `offset_n + t*4` bytes from the row base. That is
4-byte aligned for all `t`. However, this relies on `c` (the output tensor)
being 4-byte aligned at row starts, which is guaranteed by PyTorch for
contiguous tensors. **But the `size_n` stride is also part of the address** —
`out_row * size_n * 2` bytes. If `size_n` is odd (not 2-byte aligned), the row
base is misaligned. The TORCH_CHECK only verifies `c.dim() == 2` and
`c.scalar_type() == kHalf`; it does not check `size_n % 2 == 0`. For the Qwen
model shapes (N=1024, N=2048) this is never hit in practice, but it is a latent
bug for other expert sizes.

**Fix:** Add `TORCH_CHECK(size_n % 2 == 0, ...)` in the entry point, or add an
assertion before the NPT=2 dispatch that `size_n % 4 == 0` (to guarantee both
the 2-wide CAS and a possible future 4-wide path stay aligned).

---

## BUG-2. `loadN_zeros` may read beyond the qzeros word boundary when `n % 8 + N > 8`

**File:** `moe_q_gemm_gfx906.cu:94–100`

```c
template <int N>
__forceinline__ __device__ void loadN_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[N]) {
  uint32_t d = qzeros_row[n / 8] >> ((n & 0x07) * 4);
  #pragma unroll
  for (int i = 0; i < N; ++i) zeros[i] = (int)((d >> (4 * i)) & 0xF);
}
```

The comment above the function says "requires `n % 8 <= 8 - N`". When
`N_PER_THREAD=2`, `N=2`, so the requirement is `n % 8 <= 6` — i.e. the two
nibbles must both live in the same `uint32_t` word. Thread `t` owns column
`n = offset_n + t*2`. Since `offset_n` is a multiple of `BLOCK_KN_SIZE * 2 =
512` (= 512 columns, 64 uint32_t words), `n % 8 = (t*2) % 8`. For `t=3`,
`n % 8 = 6`; two nibbles at positions 6 and 7 — both in the same word. For
`t ∈ {0,1,2,3}` within each group of 4, the two nibbles span at most positions
6–7: safe. But the comment's constraint only holds because of the specific
`offset_n` alignment. For `N_PER_THREAD=4` at `n % 8 = 4`, nibbles 4,5,6,7
are all in the same word: fine. **If `offset_n` or `t`-stride ever changes
(e.g. a future 128-thread variant that changes N_PER_THREAD=3, or a caller
with an odd column offset), this silently reads garbage from the next word.**
The constraint is not enforced by code — only by the comment.

**Fix:** Add a `static_assert(N_PER_THREAD <= 4)` (already present as a
`static_assert` in the kernel) and a `static_assert` or runtime assertion that
`(n % 8) + N_PER_THREAD <= 8` holds, or restructure `loadN_zeros` to safely
span two uint32_t words when needed. At minimum, promote the comment to a
`static_assert` that fires if `N_PER_THREAD` is ever set to a value that can
violate the boundary.

---

## RISK-1. NPT=2 epilogue writes a 32-bit CAS to a 32-bit slot that overlaps with the NPT=4 epilogue's 64-bit CAS slot — mixing the two on the same output tensor is undefined

**File:** `moe_q_gemm_gfx906.cu:57–71, 334–344` and `gfx906_w4a16_moe.py:380–384`

`select_n_per_thread` returns 4 for `block_size_m < 8` and 2 for `block_size_m
>= 8`. Both paths write to the same output tensor `c`. The NPT=4 path uses a
64-bit CAS on aligned pairs of halves; the NPT=2 path uses a 32-bit CAS on
pairs of halves. **If both grid configs execute on the same output tensor
concurrently** (which they don't — there is only one kernel launch per GEMM
call), this would be a race. Currently they are never concurrent: there is
exactly one kernel dispatch per call with a single `block_size_m`. Safe as-is.

However, there is a subtler concern: NPT=4 writes 4 consecutive halves (columns
`n, n+1, n+2, n+3`) via one 64-bit CAS; NPT=2 writes 2 halves (columns
`n, n+1`) via one 32-bit CAS. The 64-bit CAS naturally covers an aligned
4-half group; the 32-bit CAS covers an aligned 2-half group. These do not
interfere within one call, but the NPT=4 path uses `blockIdx.y *
BLOCK_KN_SIZE * 4` for `offset_n` and the NPT=2 path uses `blockIdx.y *
BLOCK_KN_SIZE * 2`. The grid.y is set accordingly in `launch_moe_gemm_q4`.
All consistent. But if someone calls `dispatch_moe_gemm_q4` twice on the same
non-zeroed output with different `block_size_m` values (once BM=4/NPT=4 and
once BM=8/NPT=2), the 32-bit and 64-bit CAS patterns cover different halves
of the same 64-bit slot — **the 64-bit read in one call will see uninitialized
data written by the 32-bit CAS of another**. The Python `apply()` always
zero-fills and dispatches a single kernel, so this cannot happen today. But it
is a correctness invariant that is not enforced by the C++ API.

**Fix:** Document in the C++ entry-point comment that `c` must be zeroed and
that a single `block_size_m` must be used per call. Add a TORCH_CHECK that
`c` is contiguous (currently not checked).

---

## RISK-2. `dot22_8_f` assumes 16-byte alignment of `a_ptr`; `block_a` pad of 8 halves achieves this only when `BLOCK_KN_SIZE` is a multiple of 8

**File:** `moe_q_gemm_gfx906.cu:43–54`

```c
v.u = *(const uint4*)a_ptr;
```

`uint4` load requires 16-byte alignment. `a_ptr = &block_a[m][a_off]` where
`a_off = (k - offset_k) + 8 * j`. The LDS array is declared as
`__shared__ half block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + LDS_PAD]` with
`LDS_PAD=8`. Each row is `(256+8)*2 = 528` bytes. Row 0 starts at the LDS
base (aligned). `block_a[m][a_off]` byte-offset from row start = `a_off * 2`.
`a_off = 8*j` (j ∈ 0..3), so `a_off * 2 = 16*j` — **always 16-byte aligned
within the row**. The row base for `m > 0` is at offset `m * 528` bytes from
LDS base; 528 is not a multiple of 16 (`528 / 16 = 33.0`). So row 1 starts at
byte offset 528, which is `528 % 16 = 0` — actually fine, since
`528 = 33 * 16`. Row m starts at `m * 528` bytes = `m * 33 * 16` — always
16-byte aligned. **The alignment is correct.** However, this is non-obvious
and fragile: it depends on `(BLOCK_KN_SIZE + LDS_PAD) * sizeof(half)` being
divisible by 16, i.e. `(256 + 8) * 2 = 528 = 33 * 16`. If `LDS_PAD` is ever
changed to a value where `(BLOCK_KN_SIZE + LDS_PAD) % 8 != 0`, the uint4 load
silently becomes misaligned (undefined behaviour on GCN).

**Fix:** Add `static_assert((BLOCK_KN_SIZE + LDS_PAD) % 8 == 0, "row stride must be 16-byte aligned for ds_read_b128");` near the `LDS_PAD` definition.

---

## RISK-3. `prep_zero_scale_fp16` uses a bit-trick that is only valid for `zero ∈ [0, 15]`

**File:** `moe_q_gemm_gfx906.cu:113–115`

```c
z1u.u = (uint16_t)(0xE400 | zero);  // half(-1024 - zero) bit trick
```

This constructs a fp16 encoding of `(-1024 - zero)` by ORing `0xE400` (fp16
for -1024) with the low nibble `zero`. This works only when `zero ∈ [0, 15]`
because: the fp16 significand of -1024 is `0x400` (exact power of 2), so bits
[3:0] of the 10-bit mantissa are 0, and ORing a 4-bit value there is safe.
For `zero = 0x00..0x0F` this is exact.

The risk: if `zero_offset = 1` (GPTQ-v1 path), `zeros[i] + zero_offset` could
be 16 for a stored zero of 15. `prep_zero_scale_fp16` receives
`(uint32_t)(zeros[i] + zero_offset)`, so `zero=16` is passed. Then
`0xE400 | 16 = 0xE410` — fp16 `0xE410` is `-1040.0`, not the correct
`-1024 - 16 = -1040`. Wait: fp16 `0xE410` — sign=1, exp=28 (`-1024` uses
biased exp 25, so `0xE400` is sign=1, exp=28, mant=0 → value = -2^10 = -1024).
Mantissa `0x010` = 16 in the mantissa, value = -1*(1 + 16/1024)*2^10 = -1040.
This happens to be correct **only by coincidence** because the fp16 encoding of
-1024 has its low 4 mantissa bits zero and the mantissa is linear. But it
fails for `zero >= 32` (would require a carry into the exponent) — not reachable
for 4-bit zeros. More dangerous: **this technique is not documented to work in
general** and will silently produce wrong results if the mantissa encoding of
the bias changes (e.g. if someone extends zero range to 5-bit or changes the
bias constant).

**Fix:** Replace the bit-trick with
`z1 = __float2half_rn(-1024.0f - (float)zero)` — cleaner, equally fast on
gfx906 (one `v_cvt_f16_f32` instruction), and obviously correct for any zero.

---

## RISK-4. `_empty_topk_w` cached attribute is on the Python object, not tied to the device — will silently fail on multi-device or re-init scenarios

**File:** `gfx906_w4a16_moe.py:197–205`

```python
if not hasattr(self, "_empty_topk_w"):
    self._empty_topk_w = torch.empty(0, dtype=torch.float32,
                                     device=hidden_states.device)
```

The empty tensor is created once on whatever device `hidden_states` is on at
first call and reused forever. If the object is ever called with
`hidden_states` on a different device (model parallel, device reassignment,
or tests that move tensors), the stale `_empty_topk_w` is on the wrong device,
causing a device mismatch error at the C++ kernel entry. This is latent: in the
current single-GPU serving path it never triggers.

**Fix:** Use a class-level or module-level `torch.empty(0, ...)` created once,
or check `self._empty_topk_w.device == hidden_states.device` before reusing.
Simplest: `torch.empty(0, dtype=torch.float32, device=hidden_states.device)`
unconditionally — it's a zero-element allocation, essentially free.

---

## PERF-1. `bench_moe_gemm_gfx906.py` benchmark uses the old heuristic (BM=16 for `EM > 512`) instead of the updated heuristic (BM=8 for `EM > 512`)

**File:** `benchmarks/kernels/gfx906/bench_moe_gemm_gfx906.py:67, 101`

```python
bm = 16 if EM > 512 else (4 if EM > 32 else 1)
```

The `gfx906_w4a16_moe.py` heuristic (post P2-1c) uses BM=8 for `EM > 512`.
The benchmark file was not updated. Benchmark results for `M >= 65` (EM >= 520)
therefore measure the phase-1 BM=16 kernel, not the shipped BM=8/NPT=2 kernel.
The P2-0 micro-bench table in the devlog was taken before P2-1c landed; this
bug means the benchmark cannot reproduce the shipped performance without a
manual BM override.

**Fix:** Update both lines to `bm = 8 if EM > 512 else (4 if EM > 32 else 1)`
and add a comment cross-referencing `gfx906_w4a16_moe.py`.

---

## PERF-2. `bench_moe_gemm_gfx906.py` bandwidth calculation uses a stale (and wrong) roofline constant

**File:** `benchmarks/kernels/gfx906/bench_moe_gemm_gfx906.py:27`

```python
PEAK_F16 = 29.5e12
```

As documented in the devlog (P2-0 / P2-1), the datasheet figure is 26.8 TFLOPS
(MI50, not MI60), and the measured `v_dot2_f32_f16` peak is ~20 TFLOPS. All
`%peak` figures reported by this script are therefore ~48% too optimistic
(29.5 / 20 ≈ 1.48). The devlog already acknowledges this but the script was
not updated. Anyone running the benchmark sees inflated %peak numbers.

**Fix:** Set `PEAK_F16 = 20e12` (measured practical peak) or add a comment
explaining both figures. Also update the docstring which says "MI60 roofline
(29.5 TFPOPS fp16 peak)" — it should say MI50.

---

## PERF-3. The main K-loop stores weight chunks into `b_w[4][N_PER_THREAD]` before consuming them, preventing the compiler from scheduling the load and compute in parallel

**File:** `moe_q_gemm_gfx906.cu:271–313`

```c
uint32_t b_w[4][N_PER_THREAD];
while (k < end_k) {
    ...
    for (int j = 0; j < 4; ++j) {
      // load b_w[j][...]
    }
    b_ptr += 4 * size_n;
    for (int j = 0; j < 4; ++j) {
      // consume b_w[j][...]
    }
    k += 32;
}
```

All four weight chunks are loaded first, then all four are consumed. The devlog
documents that double-buffered software prefetch was tried and failed due to
register pressure. However, the current structure does the same thing within
one iteration: it loads `b_w[0..3]` fully before beginning any dequant/dot.
This does give the compiler 4 independent `global_load_dwordx2/4` ops before
the first `waitcnt`, which is the correct latency-hiding structure for a
single-stage pipeline. The DEVLOG ISA analysis confirms `s_waitcnt vmcnt(3)` is
emitted after the four loads. **This is correct and intentional.** Not a bug,
but documenting why the structure is correct is valuable (the comment says
"single-stage weight prefetch" without explaining why the 4-load grouping is
what provides the overlap).

**No fix required.** Consider a one-line comment: "4 loads before first waitcnt
gives the HBM pipeline 4 in-flight requests to hide latency."

---

## MINOR-1. `test_gfx906_moe_gemm.py` does not test the NPT=2 path explicitly

**File:** `tests/kernels/moe/test_gfx906_moe_gemm.py:133–151`

The test cases include `block_m ∈ {1, 4, 16}` — but not `block_m=8`, which is
the shipped default for prefill (EM > 512). The NPT=2 code path (selected for
BM≥8) is not exercised by any parametrized case. An NPT=2-specific bug
(e.g. in `atomic_add_pk2_f16` or the NPT=2 epilogue branch) would not be
caught.

**Fix:** Add a test case with `(M=64, ..., block_m=8)` or directly
parametrize `block_m` to include 8. One extra entry in `_CASES`:
`(64, 1024, 2048, 1024, 512, 8)`.

---

## MINOR-2. `test_gfx906_moe_gemm.py` uses `topk_w` as `float16` but passes `.float()` to the kernel — the reference accumulates in `float32` via a different path

**File:** `tests/kernels/moe/test_gfx906_moe_gemm.py:75, 110, 122`

```python
topk_w = torch.rand(M, TOPK, dtype=torch.float16)
...
ops.moe_gptq_gemm_gfx906(..., topk_w.view(-1).float(), ...)
...
h *= topk_w[tok, rows[:, 1]].float().unsqueeze(1)
```

The kernel receives fp32 `topk_weights`. The reference multiplies by
`topk_w[tok, rows[:, 1]].float()`. Since `topk_w` was created as fp16 and then
cast to fp32, both kernel and reference see the same fp32 values (from the same
source fp16). No precision discrepancy here. However, the test creates
`topk_w` as fp16 while the production code uses `topk_weights.view(-1).float()`
from a fp32 source (vLLM's topk weights are fp32). This mismatch does not
affect correctness of the test (the cast path is consistent) but is
unnecessarily confusing.

**Fix:** Create `topk_w` as `float32` directly:
`topk_w = torch.rand(M, TOPK, device=dev, dtype=torch.float32)` and drop the
`.float()` conversions.

---

## MINOR-3. `gfx906_w4a16_moe.py` checks `c.is_contiguous()` for `hidden_states` but not for `w1_out` or `output` before passing them as `c` to the kernel

**File:** `gfx906_w4a16_moe.py:174, 208–225`

```python
assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
```

`w1_out` and `output` are slices of workspace tensors obtained via
`_resize_cache`. These are expected to be contiguous, but the kernel's C++
entry point does not check contiguity of `c` (the output tensor). The atomic
epilogue computes `c + out_row * size_n + n` using the raw data pointer and
assumes row stride = `size_n` elements. A non-contiguous `c` (e.g. a view with
a different stride) silently produces wrong output.

**Fix:** Add `TORCH_CHECK(c.is_contiguous(), "c must be contiguous")` in
`moe_gptq_gemm_gfx906`. Or add a Python-side assertion after `_resize_cache`.

---

## MINOR-4. `select_n_per_thread` calls `getenv` on every kernel dispatch

**File:** `moe_q_gemm_gfx906.cu:380–384`

```c
static int select_n_per_thread(int block_size_m) {
  if (block_size_m < 8) return 4;
  const char* e = getenv("VLLM_GFX906_MOE_NPT");
  return (e && e[0] == '4') ? 4 : 2;
}
```

`getenv` is called on every kernel dispatch (~80 per forward pass in the
serving path). `getenv` is thread-safe on Linux via `glibc` but involves a
lock and a linear scan of the environment array. In a launch-bound path where
~1500 dispatches happen per decode step, this is a small but real CPU overhead.

**Fix:** Cache the result in a `static` local:

```c
static int select_n_per_thread(int block_size_m) {
  if (block_size_m < 8) return 4;
  static int cached = [] {
    const char* e = getenv("VLLM_GFX906_MOE_NPT");
    return (e && e[0] == '4') ? 4 : 2;
  }();
  return cached;
}
```

---

## MINOR-5. `_dequant_ref` in the test handles MoeWNA16 scale/zp layout by checking `s.shape[1] == N` — this heuristic is fragile

**File:** `tests/kernels/moe/test_gfx906_moe_gemm.py:56–62`

```python
def _dequant_ref(w, s, z, q):
    if s.shape[1] == N:  # MoeWNA16 scales/zp: [E, N, G]
        ...
```

`N` here is the module-level constant `N13=1024`. This accidentally works for
all test cases because the test is designed so that N13/N2 always differ from
the groups dimension `G = K//GS`. But for a model where `N == K//GS` (e.g.
N=16, group_size=16*K), the heuristic misidentifies the layout and silently
uses the wrong dequant path, causing the reference to diverge from the kernel
without the assert tripping.

**Fix:** Pass the `layout` parameter down to `_dequant_ref` and branch on it
explicitly, rather than inferring from shape.

---

## Summary table

| ID | Severity | File | Short description |
|----|----------|------|-------------------|
| BUG-1 | **BUG** | kernel .cu:57 | `atomic_add_pk2_f16` alignment depends on `size_n % 2 == 0`; no check |
| BUG-2 | **BUG** | kernel .cu:94 | `loadN_zeros` cross-word read; safe today by alignment, not by code |
| RISK-1 | RISK | kernel .cu, moe.py | NPT=2 / NPT=4 CAS sizes differ; correctness invariant unenforced in C++ API |
| RISK-2 | RISK | kernel .cu:43 | `uint4` LDS load: 16B alignment holds now but has no `static_assert` guard |
| RISK-3 | RISK | kernel .cu:113 | `0xE400 | zero` bit-trick undocumented; breaks for `zero >= 32` (not reachable today) |
| RISK-4 | RISK | moe.py:197 | `_empty_topk_w` cached per-object, wrong if device changes |
| PERF-1 | PERF | bench .py:67,101 | Benchmark uses old BM=16 heuristic; shipped code uses BM=8 |
| PERF-2 | PERF | bench .py:27 | `PEAK_F16 = 29.5e12` is wrong (MI50, and measured peak is 20 TF) |
| PERF-3 | — | kernel .cu:271 | Single-stage 4-load grouping is correct; comment would help |
| MINOR-1 | MINOR | test .py:133 | No test case with `block_m=8`; NPT=2 path not covered |
| MINOR-2 | MINOR | test .py:75 | `topk_w` created as fp16, confusing vs fp32 production path |
| MINOR-3 | MINOR | moe.py:174 | No contiguity check on `c` (output tensor) in kernel entry |
| MINOR-4 | MINOR | kernel .cu:380 | `getenv` called per-dispatch; should be cached |
| MINOR-5 | MINOR | test .py:56 | `_dequant_ref` layout heuristic uses shape comparison, not `layout` param |

Priority order for fixes: BUG-1 → RISK-3 → MINOR-1 (NPT=2 test coverage)
→ PERF-1/PERF-2 (benchmark accuracy) → RISK-2/RISK-4 → MINOR-3 → MINOR-4.
BUG-2 and RISK-1 are latent (no current trigger path) but worth a
`static_assert` or doc comment before any API extension.
