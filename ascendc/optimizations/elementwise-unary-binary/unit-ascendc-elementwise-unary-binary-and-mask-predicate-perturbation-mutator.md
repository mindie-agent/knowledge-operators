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
  - ElementwiseUnary
  - ElementwiseBinary
  - MaskPredicate
  - Arange
  - Cast
  - FastGeluGrad
  - Gcd
  - IsInf
  - CompareSelect
domain_path: ascendc/optimizations/elementwise-unary-binary
evidence_strength: reproduced_empirical
id: unit-ascendc-elementwise-unary-binary-and-mask-predicate-perturbation-mutator
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
- unit-ascendc-simt-and-regbase-vector-programming-sop
provenance:
  attribution: Distilled from CANNBot-Knowledge controlled_code_perturbation/elementwise_unary,
    elementwise_binary, and maskpredicate empirical optimization records
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_unary/arange_vectorization_compute_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_unary/cast_compute_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_binary/addcdiv_parallel_kernel_a2_id_0.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_binary/fast_gelu_grad_memory_a2_id_0.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/maskpredicate/greater_equal_pipeline_memory_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/maskpredicate/is_inf_compute_a2.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_unary/index.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-foreach-metadata-batching-and-layout-transform-perturbation-mutator
- unit-ascendc-evolutionary-strategy-vector-simd-and-ub-tiling-mutator
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Evolutionary and controlled code perturbation mutator for Ascend C elementwise
  unary, elementwise binary, and mask-predicate kernels, covering 1D dimension collapse,
  UB buffer reuse, Vector Counter mode, Compare+Select predicate synthesis, and pipeline
  double buffering.
tier: L4
title: Ascend C Elementwise Unary, Binary & Mask-Predicate Controlled Perturbation
  Mutator
triggers:
  error_signatures:
  - Elementwise operators without cross-element dependencies must collapse contiguous
    dimensions to 1D and size UB tiles to saturate Vector 256B repeat strides
  - Mask predicate kernels (equal, greater_equal, is_inf) must replace per-element
    Scalar branching with vectorized Compare + Select / SelectWithBytesMask pipelines
  - Iterative scalar-heavy integer kernels (e.g., gcd) must vectorize loop steps or
    use hybrid scalar-vector dispatch when tile length exceeds vector crossover
  keywords:
  - elementwise_unary
  - elementwise_binary
  - maskpredicate
  - compare_select
  - select_with_bytes_mask
  - arange_vectorization
  - fast_gelu_grad
  - dimension_collapse
  - vector_counter_mode
---
## Preconditions & Signals

Trigger this mutator when profiling or inspecting an Ascend C elementwise unary (`arange`, `cast`, `erfinv`, `expm1`, `logit`), elementwise binary (`add_custom`, `addcdiv`, `addcmul`, `fast_gelu_grad`, `gcd`, `xlogy`), or mask-predicate (`equal`, `greater_equal`, `is_inf`, `isnan`, `logical_and`) kernel reveals any of the following signals:

1. **High Scalar Ratio (`scalar_ratio > 0.25`) on Simple Elementwise Loops**:
   - Host tiling preserves multi-dimensional loop nests over contiguous tensors instead of collapsing to 1D (`P112`), or the device kernel uses `Normal` mask mode with manual main-block/tail-block scalar branching around every Vector API call (`P67`).
2. **UB Capacity Spill or `BUFFER_NUM=1` Serialization in Multi-Term Expressions**:
   - Composite elementwise gradients (`fast_gelu_grad`, `addcdiv`, `addcmul`) allocate 4–6 distinct `TBuf<VECCALC>` temporaries simultaneously, halving `tileLength` or preventing `BUFFER_NUM=2` ping-pong overlap.
3. **Scalar `GetValue`/`SetValue` in Sequence Generation or Predicate Evaluation**:
   - `arange` or `is_inf` / `equal` populates UB via scalar `for` loops or unpacks 1-bit predicate masks element-by-element on the Scalar unit (`S-pipe`).

## Mutation Moves (Ranked by Expected Gain)

### Mutation 1: 1D Dimension Collapse, Dual-Track Tail Partitioning & Vector Counter Mode (`P112` + `P11` + `P67`)
- **Target**: Host tiling and outer device loop of contiguous unary/binary kernels (`add_custom`, `addcdiv`, `cast`).
- **Transformation**:
  1. Collapse all contiguous dimensions into a single 1D total element count $N$.
  2. Partition $N$ across all available AIV cores using dual-track core sizing (`formerNum` cores handle `tileNum_former` aligned blocks + `tail_former`; `tailNum` cores handle `tileNum_tail` aligned blocks) so every core receives 32B-aligned GM base offsets (`512B`-aligned when $N$ is large, `P66`).
  3. Switch Vector API invocations from manual `SetVectorMask<T, MaskMode::NORMAL>` + repeat/tail split to `SetVectorMask<T, MaskMode::COUNTER>(calCount)` once at the top of `Compute()`, letting hardware handle repeat count and tail lanes automatically.

### Mutation 2: In-Place UB Live-Range Reuse & Algebraic Chain Compression (`fast_gelu_grad`, `addcmul`, `addcdiv`)
- **Target**: `Compute(int32_t progress)` in multi-step unary/binary expressions where live ranges of intermediate UB tensors do not overlap.
- **Transformation**:
  1. Construct an interference graph of UB temporaries (`tmp0`, `tmp1`, `tmp2`). Reuse the input queue buffer (`xLocal` after its last read) or fold sequential operations (`Muls` $\rightarrow$ `Adds` $\rightarrow$ `Exp` $\rightarrow$ `Mul`) into a 2-buffer ping-pong (`bufA`, `bufB`) inside a single `TBuf<VECCALC>`.
  2. Reclaim the freed UB footprint to double `tileLength` (e.g., from `4096` to `8192` or `16384` elements) and enable `BUFFER_NUM = 2` on `TQue<QuePosition::VECIN, 2>` and `TQue<QuePosition::VECOUT, 2>`.
  3. Fuse algebraic constants (`1.702f * 0.5f`, reciprocal scalars) on Host or before the tile loop so inner vector chains execute fewer instructions (`P69`).

### Mutation 3: Doubling-Vector `arange` & Vectorized `Compare + Select` Predicate Synthesis (`arange`, `is_inf`, `greater_equal`, `equal`)
- **Target**: Sequence initialization (`arange`) and boolean mask/predicate outputs (`uint8`/`bool`).
- **Transformation**:
  1. **Vectorized `arange`**: Initialize only the first 64/128 elements `[0, 1, ..., 63]` once in a resident UB seed buffer during `Init()`; inside the tile loop, generate full tiles via `Muls` + `Adds` (`seed * step + tile_base_scalar`) using 256B vector instructions instead of scalar `SetValue` loops.
  2. **Vectorized `is_inf` / Predicate Output**: Compute `Abs(xLocal)` in Vector pipes, run `CompareScalar(maskLocal, absLocal, +inf, CMPMODE::EQ)` (or `Compare` for `equal`/`greater_equal`), and materialize `uint8` `0/1` outputs via `Select(outFp16, maskLocal, onesResident, zerosResident, SELMODE::VSEL_TENSOR_TENSOR_MODE)` followed by `Cast<uint8_t, half>` rather than scalar bit extraction.

## Guardrails & Verification

1. **FP16 Overflow Guard in `is_inf` and `fast_gelu_grad`**:
   - When input is `BF16` or when `Exp(1.702 * x)` can overflow `FP16` (`> 65504`), upcast to `FP32` in UB (`A1`), evaluate the transcendental/comparison expression in `FP32`, and cast back to the target output dtype with `RoundMode::CAST_RINT` (`A3`).
2. **Non-32B Tail Write Safety**:
   - When the final core's tail tile is not a multiple of 32 bytes (`16` FP16/BF16 elements or `32` uint8/bool elements), use `DataCopyPad` on both `GM -> UB` and `UB -> GM` so adjacent cores' GM outputs are never overwritten by 32B block rounding.

## Fallbacks

- **Broadcast Binary Fallback (`P113`)**: When inputs have mismatched shapes (broadcast binary), do *not* blindly flatten to 1D; apply `DimensionCollapse` to merge only matching contiguous axes and use `Brcb` (when trailing dim is 1) or UB tile reuse along broadcast axes.
