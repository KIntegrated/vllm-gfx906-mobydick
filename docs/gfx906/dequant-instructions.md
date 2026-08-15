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
