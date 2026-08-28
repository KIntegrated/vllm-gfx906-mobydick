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

- dot4c/dot8c are not available on gfx906; only use dot4/dot8 forms.
- gfx906 dot instructions are available, but `v_mfma*` instructions are not
  listed for this target.
- SDWA selects byte/word sublanes (BYTE_0..3, WORD_0..1, DWORD), not
  arbitrary bitfields.
- DPP/DS lane ops are wave-level operations; they are not global cross-wave
  data movement.
- clamp behavior matters for integer dot/arith overflow paths; enable only
  when required.
