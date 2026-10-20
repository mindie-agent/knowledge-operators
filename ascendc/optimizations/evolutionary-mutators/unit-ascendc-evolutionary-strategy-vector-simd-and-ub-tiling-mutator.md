---
category: mutator
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend310P
  - Ascend910B
  - Ascend910_93
  - Ascend950PR
  - Ascend950DT
  workloads:
  - VectorKernels
  - RegBaseSimdVf
  - Normalization
  - Quantization
  - SortingTopK
  - Reduction
domain_path: ascendc/optimizations/evolutionary-mutators
evidence_strength: reproduced_empirical
id: unit-ascendc-evolutionary-strategy-vector-simd-and-ub-tiling-mutator
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
- unit-ascendc-simt-and-regbase-vector-programming-sop
provenance:
  attribution: Distilled from the 161 strategy cards in CANNBot-Knowledge contrib/ascendc/optimizations/evolution_strategies/cards
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/a1_fp32_intermediate.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p81_resident_ub_tbuf_reuse.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p84_vector_compute_efficiency_patterns.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p96_hardware_sort_replacement.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p100_gathermask_replacement.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/r1_vf_fusion.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/r2_register_reuse.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/index.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-discovered-evolution-strategies-and-cross-core-pipeline-mutator
- unit-ascendc-regbase-and-simt-thread-warp-programming-guide-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Systematic evolutionary mutator codifying the A1-A8 accuracy cards, D1-D5
  dtype/dispatch cards, P1-P114 pipeline/memory/vectorization cards, and R1-R8 Ascend
  950 RegBase VF optimization cards for automated Ascend C kernel refinement.
tier: L4
title: Ascend C Evolutionary Strategy Cards (A1-A8, D1-D5, P1-P114, R1-R8) Vector
  SIMD & UB Tiling Mutator
triggers:
  error_signatures:
  - Accuracy strategies (A1-A8) must be preserved as hard invariants during performance
    mutations (P1-P114, R1-R8)
  - RegBase VF kernels on Ascend 950 (R1-R8) must cap live RegTensor count <= 32 to
    avoid register spill/fill and normalize loop bounds for Hardware Loop encoding
  - Scalar-to-Vector rewrites (P94-P101) must replace element-wise GetValue/SetValue
    and scalar bubble/selection sort with GatherMask, BroadCast, and hardware Sort/Mrgsort
  keywords:
  - evolution_strategies_cards
  - a1_fp32_intermediate
  - a2_welford
  - p34_weight_resident
  - p81_tbuf_reuse
  - p94_scalar_to_vector
  - p96_hardware_sort
  - p100_gathermask
  - r1_vf_fusion
  - r2_register_reuse
---
## Preconditions & Signals

Trigger this mutator during evolutionary or iterative Ascend C optimization when a kernel exhibits scalar bottlenecks, excessive UB buffer allocations, sub-optimal vector instruction selection, or RegBase `VF` register spills:

1. **Scalar-in-Vector-Loop Anti-Patterns (`P94`–`P101`)**:
   - The kernel uses scalar `for` loops with `GetValue`/`SetValue` for strided extraction (`RoPE`, box unpacking), element-wise math, reductions, or sorting/TopK (`bubble sort` / `selection sort`).
2. **Repeated Parameter Reloads & UB Temp Fragmentation (`P34`, `P35`, `P41`, `P81`, `P85`, `P89`)**:
   - Small loop-invariant tensors (`gamma`, `beta`, `scale`, `offset`, `cos/sin`) are reloaded from GM on every tile iteration, or high-level APIs (`SoftMax`, `AscendDequant`) each allocate disjoint temporary `TBuf` regions.
3. **Ascend 950 RegBase `VF` Inefficiency (`R1`–`R8`)**:
   - Separate `__simd_vf__` functions materialize intermediate vectors back to UB between adjacent ops (`R1`), live `RegTensor` count exceeds 32 causing stack/UB spills (`R2`), or inner loops fail Hardware Loop canonicalization (`R4`).

## Mutation Moves (Ranked by Expected Gain)

### Mutation 1: Scalar-to-Vector Structural Rewrites (`P94`–`P101`, `P111`)
- **Target**: Any kernel containing `GetValue` / `SetValue` inside an element/row loop.
- **Transformation**:
  1. **Strided Unpack / Deinterleave (`P100`, `P101`)**: Replace `for (i) { even[i] = x.GetValue(2*i); odd[i] = x.GetValue(2*i+1); }` or host-side `x[:, 0].contiguous()` column splits with a single contiguous `DataCopy` + `GatherMask` (built-in patterns `1`/`2` for odd/even, or custom mask for 4-tuple box coordinates `[x1, y1, x2, y2]`).
  2. **Scalar Sort / TopK $\rightarrow$ Two-Level Hardware `Sort` + `Mrgsort` (`P96`, `P111`)**: Replace scalar selection/bubble sort with hardware `Sort32` / `Sort` (processing 32-aligned tiles) followed by 4-way hardware `Mrgsort` merge trees in UB across tiles (`Pattern C` two-level merge tiling when $N > \text{tileSize} \times \text{coreNum}$).
  3. **Per-Row Scalar Loop $\rightarrow$ `Brcb` + Batched 2D Vector Ops (`P97`, `P99`)**: Replace `for (r = 0; r < rows; ++r) Muls(dst[r*D], src[r*D], scale.GetValue(r), D)` with `Brcb` (broadcasting `scale[rows]` to `[rows, 8]` blocks) + batched `Mul` across all `rows * D` elements in a single call, and use zero-copy sub-tensor slicing `ubTensor[offset]` (`P97`).

### Mutation 2: Resident UB/TBuf Pinning & Shared Temp Arena (`P34`, `P35`, `P41`, `P81`, `P85`, `P89`)
- **Target**: Multi-iteration Vector kernels and composite pipelines.
- **Transformation**:
  1. **Pin Loop-Invariant Weights & Accumulators (`P34`, `P35`, `P41`, `P81`)**: Load small invariant parameters (`gamma`, `weight`, `quant_scale`) into a persistent `TBuf<TPosition::VECCALC>` once in `Init()` / prologue, pre-cast to `FP32` once, and reuse across all tile iterations. Keep cross-tile state (`softmaxMax/Sum`, `gradWeight`, recurrent state) in resident `TBuf` rather than round-tripping through `TQue` or GM.
  2. **Share High-Level API Temp Buffer & Multi-Phase Zones (`P85`, `P89`)**: Query `GetSoftMaxMaxTmpSize` / `GetAscendDequantMaxTmpSize` on Host (`max(tmpSize_stage1, tmpSize_stage2)`), allocate a single shared `sharedTmpBuf`, and pass its `LocalTensor<uint8_t>` into each high-level API call across non-overlapping stages.

### Mutation 3: Ascend 950 RegBase `__simd_vf__` Fusion, Register Reuse & Dual-Issue (`R1`–`R8`)
- **Target**: Ascend 950 (`ARCH 3510`) RegBase vector kernels.
- **Transformation**:
  1. **Fuse Adjacent `__simd_vf__` Blocks (`R1`) & Cap Live `RegTensor <= 32` (`R2`)**: Merge sequential `__simd_vf__` functions that share the same loop bounds so intermediates stay in vector registers (`vreg`) without UB store/load round-trips. Apply in-place register reuse and live-range shortening so live `RegTensor` count never exceeds 32.
  2. **Expose Instruction Dual-Issue (`R3`) & Canonicalize Hardware Loops (`R4`)**: Unroll independent register chains 2x inside `__simd_vf__` to feed A5's dual-issue Vector pipes, and keep loop induction variables strictly canonical (`for (int i = 0; i < tripCount; ++i)` with loop-invariant `tripCount` and no internal early-exit `break`) so the compiler emits zero-overhead Hardware Loops.
  3. **Use `UnalignRegForLoad/Store` (`R5`) & `Mutex` Sync (`R8`)**: Handle non-32B-aligned register loads via `UnalignRegForLoad` / `UnalignRegForStore`, and use lightweight `Mutex` synchronization instead of coarse `SetFlag`/`WaitFlag` where supported.

## Guardrails & Verification (The `A1`–`A8` & `D1`–`D5` Accuracy Invariants)

1. **Never Violate `A1`–`A8` Accuracy Cards During Performance Rewrites**:
   - **`A1` (FP32 Intermediate)**: Keep reductions, norms, softmax, and composite transcendental chains in `FP32` for `FP16`/`BF16` inputs.
   - **`A2` (Welford)** & **`A5` (Numerical Safety)**: Use single-pass Welford for training BatchNorm mean/var and guard `NaN`/`Inf` via `Compare + Select`.
   - **`A3` (Rounding Mode)** & **`A8` (Quant Clamping)**: Use `RoundMode::CAST_RINT` (round-to-nearest-even) on `FP32 -> FP16/BF16/INT8` casts and explicitly clamp quantized outputs to `[quant_min, quant_max]` via `Maxs`/`Mins`.
   - **`A6` (Newton-Raphson `rsqrt`)**: When hardware `Rsqrt` precision in `FP16` is insufficient, run one Newton-Raphson refinement step $y_{1} = y_0 (1.5 - 0.5 x y_0^2)$ or compute `Rsqrt` in `FP32`.

## Fallbacks

- **Hybrid Scalar/Vector Dispatch (`P98`)**: When dynamic runtime length $N$ is tiny (e.g., $N < 16$ where Vector setup + mask + alignment padding costs more than scalar execution), dispatch to a lightweight scalar loop for $N < N_{\text{crossover}}$ and the full vectorized pipeline for $N \ge N_{\text{crossover}}$.
