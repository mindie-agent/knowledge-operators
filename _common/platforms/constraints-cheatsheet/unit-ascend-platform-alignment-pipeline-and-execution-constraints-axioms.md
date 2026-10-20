---
category: axiom
conditions:
  dtype:
  - bfloat16
  - float16
  - float32
  - int8
  - fp8_e4m3fn
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
  - Ascend950DT
  - Ascend950PR
domain_path: _common/platforms/constraints-cheatsheet
evidence_strength: software_contract
id: unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/constraints_cheatsheet.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend_patterns.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/index.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/constraints_cheatsheet.md
  source_type: compiler_and_hardware_constraint_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: 'Ascend NPU mandatory hardware/compiler pre-flight constraints: UB/L0 1/3 rule,
  512B row alignment, grid physical-core binding and scalar-fallback bans'
triggers:
  error_signatures:
  - aiv_scalar_ratio is abnormally high because int64 or int32 comparisons and pointer
    offsets triggered scalar fallback
  - a matmul kernel with BLOCK_N * dtype_bytes not divisible by 512B or row-major
    large-matrix launch suffers 25-35% L2 bank conflict
  - a 2D or 3D grid launch exceeds the physical AI Core count or 65535 hardware block
    limit without persistent grid-stride looping
  - tl.swizzle2d or in-kernel return/break/continue/lambda expressions fail lowering
    or produce undefined behavior on Ascend
  keywords:
  - constraints_cheatsheet
  - aiv_scalar_ratio
  - care_padding
  - grouped_launch_diagonal
  - tl.swizzle2d
  - 512B alignment
  - BLOCK_N
  - TRITON_ALL_BLOCKS_PARALLEL
  - grid-stride loop
  - axis borrowing transpose
---
# Ascend NPU mandatory hardware/compiler pre-flight constraints: UB/L0 1/3 rule, 512B row alignment, grid physical-core binding and scalar-fallback bans

**Condition.** An Ascend operator (Triton-Ascend, DSL, or Ascend C) is designed or reviewed for `Ascend910B` (A2), `Ascend910_93` (A3), or `Ascend950` (A5). Every kernel must satisfy five mandatory correctness constraints (**C1–C5**) and seven performance invariants (**P1–P7**) before compilation or profiling.

**Constraint.**

- **Five mandatory correctness constraints (`C1–C5`).**
  - **C1 — UB single-tile ceiling and `1/3` multi-operand rule**: Raw UB per AIV is `192 KB` on A2/A3 (`98,304` `fp16`/`bf16` elements, `49,152` `fp32` elements, `196,608` `fp8`/`int8` elements) and `248 KB` on A5. Because binary/ternary vector ops and matmul staging hold `A`, `B`, and `C` tiles simultaneously (and `multibuffer=True` halves capacity), size `BLOCK_SIZE * dtype_bytes <= UB_limit / 3`.
  - **C2 — Hardware grid extent ceiling**: `grid` must be a `1D`, `2D`, or `3D` tuple (never a bare integer or 4D tuple) with total blocks `x * y * z <= 65535`.
  - **C3 — 512 B Cube row-width alignment**: For any Cube / matmul operator (`CV` class), `BLOCK_N * dtype_bytes` must be an exact multiple of **`512 B`** (`BLOCK_N` a multiple of `256` for `fp16`/`bf16`, `128` for `fp32`, `512` for `fp8`/`int8`), and `BLOCK_M`, `BLOCK_N`, `BLOCK_K` must be powers of 2 (standard autotune space: `BLOCK_M, BLOCK_N in {128, 256}`, `BLOCK_K = 256`, `GROUP_SIZE in {8, 16, 32}`). For pure Vector (`VV` class) operators, the trailing axis must be a multiple of **`32 B`**; when a short trailing axis cannot divide `32 B` (e.g., shape `[2048, 3]`), apply **axis-borrowing transpose** so DMA transfers remain coalesced.
  - **C4 — Ban `tl.swizzle2d`**: `tl.swizzle2d` has undefined behavior on Ascend NPU; replace it with `grouped_launch_diagonal` for L2 cache locality.
  - **C5 — Physical-core Grid binding**: Unlike GPU logical grids, Ascend grids bind directly to physical AI Cores (`20–24` on A2, `40–48` AIC / `80–96` AIV on A3, `28–36` AIC / `56–72` AIV on A5). Prefer a `1D` persistent grid with `grid_size <= num_ai_cores` and a grid-stride loop over work items; launching far more blocks than physical cores requires software time-slicing (`TRITON_ALL_BLOCKS_PARALLEL=1`) and degrades scheduling efficiency.
- **Seven mandatory performance & lowering constraints (`P1–P7`).**
  - **P1 & P2 — Eliminate `int64` / `int32` CMP/ADD scalar fallbacks**: Comparing or adding `int64` or `int32` index vectors (such as `tl.arange()` results) degrades to the Scalar unit and spikes `aiv_scalar_ratio`. Cast index ranges to `.to(tl.float32)` before vector comparisons/additions, and clamp negative pointer offsets via `tl.maximum(offset, 0)` + mask.
  - **P3 — Decouple load/compute serialization with `care_padding=False`**: When out-of-bounds lanes are masked downstream, pass `care_padding=False` to `tl.load()` so MTE2 DMA does not serialize with Vector padding checks.
  - **P4 & P5 — Contiguous memory and persistent scheduling**: Ensure non-contiguous input tensors call `.contiguous()` before kernel launch so MTE2 vectorization does not degrade to scalar gathers.
  - **P6 — Diagonal tiling for large matrices (`M, N >= 512`)**: Naive row-major block scheduling across all AI Cores causes `25–35%` L2 cache bank conflicts on shared `A`/`B` panels; use diagonal block scheduling (`grouped_launch_diagonal`).
  - **P7 — FP32 accumulation for `fp16`/`bf16` reductions and dots**: Always accumulate reductions (`tl.sum`) and matrix multiplications (`tl.dot`) in `fp32` and cast once before `tl.store`.
- **Forbidden syntax in Ascend kernel bodies**: Never use early `return`, `break`, `continue`, `lambda` functions, chained Python booleans (`a and b and c`), or direct scalar indexing `tensor[i]` inside a vectorized kernel; express all control flow with bitwise mask tensors (`mask1 & mask2`) and `tl.where(cond, x, y)`.

**Rationale.** Violating `C1–C5` causes immediate compilation failures, UB overflow traps, or silent Cube tile corruption; violating `P1–P7` forces vector loops onto the single-issue Scalar pipeline or serializes MTE2 and VEC pipelines.

**Evidence.** Codified in `TritonAscendOps.td`, `hardware_consts.py`, and the pre-generation checklists in `https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/constraints_cheatsheet.md` and `ascend_patterns.md`.
