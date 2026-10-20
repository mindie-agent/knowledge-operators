---
category: sop
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
  - ReduceSum
  - ReduceMean
  - ReduceMax
  - ReduceMin
  - ReduceProd
  - ArgMax
  - ArgMin
  - LayerNormReduction
domain_path: Reduction/tree-and-argreduce
evidence_strength: reproduced_empirical
id: unit-ascendc-wholereduce-blockreduce-and-argreduce-sop
inherits_from:
- unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/optimizations/reductions,
    optimizations/vector, apis/simd_api/adv_api/reduction_operations, and evolution
    strategy cards P55, P68, P109, P110, R6
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/reductions/pairwise_sum_with_tail_folding.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/vector/lane_accumulation_then_final_reduce.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p68_low_latency_reduction.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p109_reduction_multi_axis_tiling.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p110_reduction_ub_accumulator_reuse.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/math/math/arg_max_950.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/reductions/pairwise_sum_with_tail_folding.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-argreduce-tree-reduction-and-atomic-contention-mutator
- unit-ascendc-vector-simd-softmax-activation-sort-and-gather-api-contract
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Ascend C reduction operators, covering multi-axis
  A/R/ARA dimension collapsing, low-latency Add/Max tree folding + BlockReduce/WholeReduce
  instruction combos, non-last-axis vertical vector accumulation, and ArgMax/ArgMin
  index extraction.
tier: L2
title: Ascend C WholeReduce, BlockReduce, Pairwise Tree Reduction & Multi-Axis AR/ARA
  Tiling SOP
triggers:
  error_signatures:
  - WholeReduceSum/Max/Min reduces each 256B repeat into 1 scalar; BlockReduceSum/Max/Min
    reduces each 32B datablock into 1 scalar (8 scalars per 256B repeat)
  - Reduction instructions have 2-5x higher pipeline latency than Vector Add/Max;
    multi-block reductions must fold across repeats with Vector Add/Max down to a
    single 256B repeat before calling WholeReduce/BlockReduce
  - Non-last-axis reductions (ARA / RA patterns) should reduce across rows via vertical
    Vector Add/Max without transposing, followed by final horizontal reduction or
    contiguous writeback
  keywords:
  - whole_reduce_sum
  - block_reduce_sum
  - whole_reduce_max
  - block_reduce_max
  - pair_reduce_sum
  - multi_axis_reduction_tiling
  - ar_ara_collapse
  - low_latency_reduction
---
## Preconditions & Signals

Apply this SOP when designing or optimizing any reduction operator (`sum`, `mean`, `amax`, `amin`, `prod`, `all`, `any`, `norm`, `argmax`, `argmin`) in Ascend C:

1. **Multi-Axis Shape Normalization to Alternating `A/R` Axes (`P109`)**:
   - Arbitrary $N$-dimensional input shapes and `dim` lists must first be collapsed by merging adjacent retained axes (`A`) and adjacent reduced axes (`R`) into one of four canonical forms:
     - **`R`** (All-Reduce: $[R_0] \rightarrow [1]$)
     - **`AR`** (Last-Axis Reduce: $[A_0, R_0] \rightarrow [A_0]$)
     - **`RA`** (First-Axis Reduce: $[R_0, A_0] \rightarrow [A_0]$)
     - **`ARA`** (Middle-Axis Reduce: $[A_0, R_0, A_1] \rightarrow [A_0, A_1]$)
2. **Hardware Reduction Instruction Hierarchy (`WholeReduce*`, `BlockReduce*`, `PairReduce*`)**:
   - `WholeReduceSum/Max/Min(dst, src, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride)`: Reduces all active elements in each `256B` repeat (`64` FP32 or `128` FP16) down to **1 element** per repeat (`WholeReduceMax/Min` optionally emits `(val, idx)` pairs when `isOutputIndex` / `vcmax` mode is selected).
   - `BlockReduceSum/Max/Min`: Reduces each `32B` datablock (`8` FP32 or `16` FP16) down to **1 element**, producing **8 elements** (`32B` for FP32, `16B` for FP16) per `256B` repeat.
   - `PairReduceSum`: Sums adjacent pairs `(2i, 2i+1)`, halving the element count per repeat.

## Procedure

### Step 1: Canonical `AR` (Last-Axis) Low-Latency Reduction Pipeline (`P68`, `R6`)
1. **Stage 1 — Inter-Repeat Vector Folding (`Add` / `Max` / `Min`)**:
   - Because `WholeReduceSum` has $2\text{--}5\times$ higher instruction latency than `Add`, never call `WholeReduceSum` directly on $K \gg 1$ repeats!
   - Instead, fold a row of length $R_0$ into a single `256B` (`64` FP32 elements) accumulator `laneAcc[64]` using standard Vector `Add` (with 2–4 independent accumulators `acc0..acc3` to hide 4-cycle V-pipe RAW latency, plus masked tail folding for $R_0 \bmod 64 \ne 0$).
2. **Stage 2 — Final `256B -> 1 Scalar` Horizontal Reduction (`P68`)**:
   - Once the row is folded into `laneAcc[64]` (`1` repeat of `FP32`), reduce `64 -> 8 -> 1` via:
     ```cpp
     BlockReduceSum<float>(blockAcc, laneAcc, 1, 64, 1, 1, 8); // 64 -> 8 floats (1 datablock)
     PipeBarrier<PIPE_V>();
     WholeReduceSum<float>(scalarOut, blockAcc, 8, 1, 1, 1, 8); // 8 -> 1 float
     ```
   - When reducing multiple rows `[M_tile, 64]` simultaneously (`M_tile = 8`), call `WholeReduceSum` with `repeatTimes = 8` so all 8 row results pack contiguously into a single `32B` datablock (`8 floats`) ready for aligned `DataCopy` to GM (`P56`).

### Step 2: Canonical `RA` & `ARA` (Non-Last-Axis) Vertical Vector Accumulation (`P110`)
1. **Zero-Transpose Vertical Accumulation**:
   - For $[A_0, R_0, A_1] \rightarrow [A_0, A_1]$, **never** transpose $[R_0, A_1]$ into $[A_1, R_0]$!
   - Tile $A_1$ into column panels `tileA1` (aligned to `64` FP32 / `128` FP16 elements) and keep a persistent FP32 UB accumulator `accBuf[tileA1]` (`P110`).
   - Stream rows $r \in [0, R_0)$ for the current panel `tileA1` from GM into UB and execute contiguous Vector `Add(accBuf, accBuf, rowBuf, tileA1)` (or `Max`/`Min`) across $r$. Every lane $j \in [0, \text{tileA1})$ accumulates its own independent column reduction with **zero** horizontal reduction instructions (`WholeReduce` is not needed at all!).
   - After exhausting $r \in [0, R_0)$, cast `accBuf[tileA1]` from `FP32` to output dtype and burst-write `[tileA1]` directly to GM.

### Step 3: All-Reduce (`R`) & Large-$R_0$ Two-Stage Cross-Core Reduction
1. **Intra-Core Lane Accumulation + Cross-Core Workspace Merge**:
   - Partition a huge $R_0$ across all $N_{\text{AIV}}$ cores (512B-aligned chunks per core).
   - Each core reduces its chunk into a single FP32 value (or `tileA1` vector for split-`R0` `RA` reduction) and writes to `workspaceGM[core_id]`.
   - Call `SyncAll()`; Core 0 loads `workspaceGM[0 : activeCores]` into UB and executes a single final reduction pass to write the deterministic scalar/vector output to GM (`P9`).

### Step 4: `ArgMax` / `ArgMin` (`arg_max_950`) Index Extraction
1. **For `RA` / `ARA` (`axis != -1`)**:
   - Maintain `bestVal[tileA1]` and `bestIdx[tileA1]` (`int32_t`) in UB across $r \in [0, R_0)$; update via `Compare(..., CMPMODE::GT/LT)` + `Select` (`Duplicate` scalar $r$ into `candIdx[tileA1]` or increment `candIdx` by `1` each row).
2. **For `AR` (`axis == -1`)**:
   - Use lane-wise accumulation (`lane_accumulation_then_final_reduce.md`) to fold tiles into `bestVal[64]` and `bestIdx[64]`, then run `WholeReduceMax<float>(..., isOutputIndex=true)` on `bestVal[64]` to obtain the winning lane $k^* \in [0, 64)$ and read `bestIdx.GetValue(k^*)`.

## Verification & Checks

1. **Neutral Identity Element Initialization for Tail Masking**:
   - Verify that tail padding lanes and initial UB accumulators use the exact mathematical identity element: `0.0f` for `Sum`, `1.0f` for `Prod`, `$-\infty$` for `Max`, `$+\infty$` for `Min`, `true` (`1`) for `All`, and `false` (`0`) for `Any`.
2. **Ascend 950 Vector Core `ReduceMin` / `ReduceMax` NaN Semantics (`950_reducemin_vector_core.md`)**:
   - Check whether the operator contract requires IEEE 754 `NaN`-propagating reduction (`max(x, NaN) = NaN`); if hardware `ReduceMin`/`ReduceMax` ignores `NaN`, pre-detect `NaN` via `Compare(mask, x, x, CMPMODE::NE)` and propagate `NaN` explicitly (`A5`).

## Failure Modes & Fallbacks

- **Symptom**: `ReduceMean` on `FP16` tensor with $R_0 = 65536$ returns `Inf` or `0.0`.
  - **Root Cause**: Accumulating sum in `FP16` (overflows at `65504`) or multiplying each element by `1.0 / R_0` in `FP16` before summing (underflows to `0`).
  - **Fix**: Upcast to `FP32` before summing (`A1`), divide the final `FP32` sum by `float(R_0)`, and cast back to `FP16` only once at the end.
