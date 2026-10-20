---
category: mutator
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
  - Ascend950PR
  - Ascend950DT
  workloads:
  - ArgReduce
  - Isamax
  - Isamin
  - Icamin
  - BevPool
  - ScatterReduce
  - TreeReduction
domain_path: ascendc/optimizations/reduction-and-atomic
evidence_strength: reproduced_empirical
id: unit-ascendc-argreduce-tree-reduction-and-atomic-contention-mutator
inherits_from:
- unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
provenance:
  attribution: Distilled from CANNBot-Knowledge controlled_code_perturbation/argreduce,
    atomicupdate, and ops/ascendc/optimizations/reductions/vector empirical records
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/argreduce/isamax_optimization_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/argreduce/isamin_pipeline_compute_a2_id_0.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/argreduce/icamin_pipeline_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/atomicupdate/bev_pool_optimization_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/atomicupdate/scatter_reduce_vectorization_compute_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/reductions/pairwise_sum_with_tail_folding.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/vector/lane_accumulation_then_final_reduce.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/argreduce/index.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-wholereduce-blockreduce-and-argreduce-sop
- unit-ascendc-evolutionary-strategy-vector-simd-and-ub-tiling-mutator
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Controlled code perturbation mutator for Ascend C ArgReduce (isamax, isamin,
  icamin), pairwise tree reduction, and atomic-update (bev_pool, scatter_reduce) kernels,
  replacing per-tile horizontal reductions and contended GM atomics with lane-wise
  UB accumulation and two-stage deterministic merge.
tier: L4
title: Ascend C ArgReduce (isamax/isamin/icamin), Tree Reduction & Atomic Contention
  Mutator
triggers:
  error_signatures:
  - ArgReduce kernels (isamax, isamin, icamin) must accumulate per-lane best values
    and indices in Vector registers/UB across tiles before running a single WholeReduceMax/Min
    on the final accumulator
  - First-occurrence tie-breaking in ArgMax/ArgMin requires deterministic lane/core
    index ordering when merging partial extrema
  - High-contention atomic updates (bev_pool, scatter_reduce) must pre-aggregate in
    UB or sort/group buckets before issuing GM SetAtomicAdd
  keywords:
  - argreduce
  - isamax
  - isamin
  - icamin
  - atomicupdate
  - bev_pool
  - scatter_reduce
  - lane_accumulation_then_final_reduce
  - pairwise_sum_with_tail_folding
  - deterministic_workspace_reduce
---
## Preconditions & Signals

Trigger this mutator when optimizing ArgReduce (`isamax`, `isamin`, `icamin`, `argmax`, `argmin`), long-vector reduction (`sum`, `mean`, `amax`, `amin`), or atomic scatter-add (`bev_pool`, `scatter_reduce`, `index_add`) kernels on Ascend NPUs:

1. **Per-Tile Horizontal Reduction Bottleneck (`isamax`, `isamin`, `icamin`)**:
   - The kernel invokes `WholeReduceMax` / `WholeReduceMin` on every input tile of length $L_{\text{tile}}$ inside the tile loop and updates the best `(val, idx)` via Scalar `GetValue` comparisons, causing heavy V-pipe-to-S-pipe stalls (`~43 cycles` per scalar readback).
2. **Pairwise Sum & Dependency Chain Stalls on Long Reductions**:
   - Long-vector `ReduceSum` chains accumulate sequentially into a single 64-element register block (`RAW` hazard every cycle) or suffer catastrophic FP16/FP32 rounding drift on large $N$.
3. **GM Atomic Contention & `512B` Cache-Line Thrashing (`bev_pool`, `scatter_reduce`)**:
   - Multiple AIV cores concurrently issue `SetAtomicAdd` to the same or adjacent `512B` GM cache lines (`P64`), serializing MTE3 writes.

## Mutation Moves (Ranked by Expected Gain)

### Mutation 1: Lane-Wise UB Accumulation Followed by Single Final Horizontal ArgReduce (`lane_accumulation_then_final_reduce`)
- **Target**: `isamax`, `isamin`, `icamin`, `argmax`, and `argmin` tile loops.
- **Transformation**:
  1. Allocate resident UB accumulators `bestValVec[W]` (`W = 64` or `tileLength`) initialized to $-\infty$ (`isamax`) or $+\infty$ (`isamin`/`icamin`) and `bestIdxVec[W]` initialized to `[0, 1, ..., W-1]`.
  2. For complex inputs (`icamin`), deinterleave real and imaginary components via `GatherMask` (`P100`) and compute $| \Re(z) | + | \Im(z) |$ in Vector pipes.
  3. On each tile $t$, compute candidate magnitudes `candValVec`, compare `Compare(cmpMask, candValVec, bestValVec, CMPMODE::GT)` (strict `>` to preserve first-occurrence tie-breaking!), and update `bestValVec` and `bestIdxVec` in-place via `Select` without any horizontal reduction or Scalar readback.
  4. After all tiles assigned to the core have been folded into the `W`-element accumulator, execute `WholeReduceMax` / `WholeReduceMin` *once* on `bestValVec[W]`, resolve the winning lane index in UB, and write the core's `(best_val, best_idx)` pair to GM workspace for final cross-core merge.

### Mutation 2: Pairwise Tree Folding with Multi-Accumulator Unrolling (`pairwise_sum_with_tail_folding` + `P68`)
- **Target**: Long-axis sum/mean/norm reductions.
- **Transformation**:
  1. Unroll the tile accumulation loop 2x–4x using 2–4 independent UB accumulator slices (`acc0`, `acc1`, `acc2`, `acc3`) so back-to-back `Add` instructions do not stall on same-register RAW dependencies.
  2. Fold non-power-of-two tail lengths into the lower half of the buffer (`Add(buf[0 : rem], buf[0 : rem], buf[half : half + rem])` with tail mask) until the active length is a power of two, then apply binary halving (`Add` with halving length down to 64 elements) before a single `WholeReduceSum` + `BlockReduceSum` (`P68`).

### Mutation 3: Bucket-Sorted Run-Length Pre-Aggregation in UB (`bev_pool`, `scatter_reduce`, `P9`)
- **Target**: Atomic scatter-update kernels (`bev_pool`, `scatter_reduce`, `index_add`).
- **Transformation**:
  1. **Run-Length UB Pre-Aggregation (`bev_pool`)**: When points/features within the same voxel/pillar interval share identical destination coordinates, accumulate all points belonging to the same interval into a single resident UB feature vector `voxelAcc[C]` and flush to GM *once* per interval via plain `DataCopy` (or a single `SetAtomicAdd` at interval boundary boundaries across cores).
  2. **Vectorized Feature Striding (`scatter_reduce`)**: When feature dimension $C$ is large or small-D rows can be merged (`P3`), vectorize the inner feature loop across full 256B vector width; if determinism is required (`P9`), partition by destination bucket index across cores so each core owns disjoint output rows and writes deterministically without atomics.

## Guardrails & Verification

1. **First-Occurrence Tie-Breaking Invariant in `isamax` / `isamin`**:
   - BLAS `isamax` / `isamin` and PyTorch `argmax` require returning the *smallest* index when multiple elements share the exact maximum/minimum value.
   - Use strict inequality (`CMPMODE::GT` for max, `CMPMODE::LT` for min) when updating `bestValVec` and `bestIdxVec`, and during final cross-lane / cross-core reduction, break value ties by `min(idx_a, idx_b)`.
2. **Tail Lane Masking in ArgMin / ArgMax**:
   - On the final non-aligned tile, pad invalid tail lanes with $+\infty$ (for min) or $-\infty$ (for max) before updating `bestValVec`, never `0.0f` (since `0.0f` beats negative maxima or positive minima!).

## Fallbacks

- When reduction length $N \le 64$, skip multi-tile lane accumulation and run a single masked `WholeReduceMax` / `WholeReduceSum` directly on the input tile.
