# gfx906 Dequant Instruction Notes

> Source: <https://skyne98.github.io/wiki-gfx906/> — copyright Oleksii Halahan.

## High-value instruction families for qdq work

- **Dot instructions** (`v_dot4_*`, `v_dot8_*`, `v_dot2_f32_f16`): use when
  data is already packed/quantized (or conversion cost is amortized).
- **SDWA instructions** (`*_sdwa`): best for byte/word extraction directly
  inside ALU/convert op (helps i8 dequant).
- **Bitfield/pack ops** (`v_bfe_*`, `v_lshl_or_b32`, `v_perm_b32`,
  shifts/ands): core tools for nibble/byte unpack and repack (especially
  int4/int8 layouts).
- **Packed fp16 ops** (`v_pack_b32_f16`, `v_cvt_pkrtz_f16_f32`,
  `v_pk_*_f16`): useful bridge path when dequantizing into fp16 or doing
  fp16 pre/post transforms.
- **Wave data movement** (`v_mov_b32_dpp`, `ds_bpermute_b32`,
  `ds_permute_b32`): useful for lane remap/reorder without global memory
  traffic.

## Measured dot-instruction rates (2026-08-28, MI50, ROCm 7.14 clang 23)

SCEV-proof probe (rotating dot outputs feed the next dot's input
operands — integer recurrences cannot be closed-formed by the
compiler); 960×256 threads fill 60 CUs, best-of-7 `hipEvent`; native
ISA emission verified by `-S` dump (`v_dot4_i32_i8`, `v_dot8_i32_i4`,
`v_dot2_f32_f16`, `v_pk_fma_f16` all present). Probe kept at
`/local/tmp/dotrate/dotrate2.cu` (numbers below are the record).

| op | GMAC/s | vs fp32 FMA | reading |
|---|---|---|---|
| fp32 FMA (`v_mac_f32`) | 5824 | 1.00× | 1 MAC/lane/cycle baseline |
| packed half2 FMA (`v_pk_fma_f16`) | 13210 | 2.27× | full packed rate, 2 MAC/cyc |
| `v_dot2_f32_f16` | 3349 | 0.58× | latency-chained probe floor; the MoE harness (DEVLOG-moe-opt P2-0, ILP≥2) reached ~10 T MAC/s ⇒ full rate but latency-sensitive |
| `v_dot4_i32_i8` (`sdot4`) | **25877** | **4.44×** | **FULL RATE — 4 int8 MAC/lane/cyc** (AMD's 53 TOPS INT8 figure for MI50 is exactly this instruction) |
| `v_dot8_i32_i4` (`sdot8`) | **49600** | **8.52×** | **FULL RATE — 8 i4 MAC/lane/cyc, native packed-nibble operands, no unpack ALU** |
| i8→half2 expand + 2×`fdot2` | 1002 | 0.17× | expansion (~2.5 VALU ops/elem) kills it — 25× slower than raw dot4 |
| i4→i8 unpack + 2×`sdot4` | 1375 | 0.24× | unpack not amortizable; moot vs native `sdot8` |

Consequences for quantized kernels on this chip:

- **There is no int8 dot gap.** `v_dot4_i32_i8` is the fastest Q8
  inner-loop instruction available (2× the packed-fp16 MAC rate); any
  "exact Q8 via fp16 dot2" scheme loses (expansion row above).
- **`v_dot8_i32_i4` is the one genuinely faster dot** (2× dot4's MAC
  rate at half the operand bytes) — but it requires i4 data on BOTH
  operands (a Q4 format change, not an instruction swap).
- Instruction throughput is clock-scaled (probe ran ~1.5 GHz effective);
  ratios are clock-independent.

## Practical limits and caveats

- The gfx906 int-dot set is `{v_dot4_i32_i8}`. Measured 2026-08-28
  (`benchmarks/kernels/gfx906/dot_isa_probe.py`, backend-verified via
  compile-to-object, which runs the llvm-mc validation that `-S` skips):
  `v_dot4c_i32_i8` = "instruction not supported on this GPU (gfx906)",
  `v_dot8_i32_i8` / `v_dot8c_i32_i8` = "invalid instruction" (all three
  pass ISel but are rejected by the assembler — `-S` output is NOT an
  availability answer). The table's `v_dot8_i32_i4` row is the *i4*
  variant (packed-nibble operands); it exists but is irrelevant to Q8's
  i8×i8 dot.
- gfx906 dot instructions are available, but `v_mfma*` instructions are not
  listed for this target.
- SDWA selects byte/word sublanes (BYTE_0..3, WORD_0..1, DWORD), not
  arbitrary bitfields.
- DPP/DS lane ops are wave-level operations; they are not global cross-wave
  data movement.
- clamp behavior matters for integer dot/arith overflow paths; enable only
  when required.

## v_dot2_f32_f16: clean fp16-2-pair dot, fp32 accumulate (2026-08-28)

Runtime-verified (same probe): `v_dot2_f32_f16 vdst, a, b, vacc` computes
**vdst = vacc + a_lo·b_lo + a_hi·b_hi** — the same op the production
dense-GEMV / MoE Q-GEMM gfx906 kernels use via `__ockl_fdot2` /
`amd_mixed_dot` (`csrc/rocm/dense_gemv_gfx906.cu`,
`csrc/rocm/moe_q_gemm_gfx906.cu`). Raw inline asm and the builtin agree
bit-for-bit in the probe.

Lesson recorded from the probe's false-alarm round: the apparent
"double-counted hi pair" / "weird 4th operand" readings were all wrong
hand-derived f16 constants (0x3C00 is 1.0 not 1.5; 0x3800 is 0.5; 0x4600
is 6.0; 0x5000 is 32.0). Verify f16 bit patterns with a runtime
`__ushort_as_half` conversion, never by hand.

**Open opportunity — FA-Q8 P·V.** The attention P·V inner loop still
accumulates with `half2` (`v_pk_mul_f16` + `v_pk_add_f16`: 2 instructions
per 2 MACs, fp16 accumulate). A `v_dot2_f32_f16` rewrite is 1 instruction
per 2 MACs with fp32 accumulate — half the P·V instruction count, better
rounding. (P·V reads both operands from LDS in the current kernel, so the
inner loop is LDS-load + VOPC-issue — the instruction halving is the
lever.) The Q·K side has no equivalent move: the dot is already the
full-rate `v_dot4_i32_i8`, and the per-block dequant ALU (scale mult +
cvt + fp32 FMA per 4-lane result) is intrinsic to consuming int dots in
the fp32 softmax — no instruction upgrade exists for it. Gating before
any attempt: kernel-level A/B (`bench_gfx906_fa_decode.py`) + PPL
invariance (fp16-acc → fp32-acc changes outputs), so this is a roadmap M3
candidate, not a drop-in (see `roadmap-more-models.md`).
