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
  - GatherV2
  - ScatterElements
  - IndexSelect
  - IndexAdd
  - MaskedSelect
  - UnsortedSegmentSum
  - Unique
domain_path: elementwise/gather-scatter-dsl
evidence_strength: reproduced_empirical
id: unit-ascend-gather-v2-scatter-elements-and-index-operator-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-simt-and-regbase-vector-programming-sop
provenance:
  attribution: Adapted from CANNBot-DSL gather_v2 and scatter_elements samples and
    CANNBot-Knowledge ops/ascendc/guides/layouts and operators/direct/nn/index documentation
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/gather_v2/gather_v2.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/scatter_elements/scatter_elements.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/layouts/gather_last_axis_modes_and_signed_index_normalization.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/layouts/index_validation_prefix_counts_and_piece_ownership.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/index/gather_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/index/scatter_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/math/conversion/masked_select_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/gather_v2/gather_v2.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-vector-simd-softmax-activation-sort-and-gather-api-contract
- unit-ascendc-argreduce-tree-reduction-and-atomic-contention-mutator
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for implementing and optimizing indexing, gather,
  scatter, and compaction operators (gather_v2, scatter_elements, gather_950, scatter_950,
  unsorted_segment_sum_950, unique_950, masked_select_950) in CANNBot-DSL and Ascend
  C.
tier: L2
title: Ascend GatherV2, ScatterElements, MaskedSelect, Unique & Index Operator SOP
triggers:
  error_signatures:
  - CANNBot-DSL gather_v2 normalizes negative indices once when negative_index_support=True,
    outputs zero on out-of-bounds or empty index axes, and preserves NaN payload/-0.0/complex
    bit patterns via integer-view transfers
  - CANNBot-DSL scatter_elements supports reduction in {'none', 'add', 'mul'} along
    axis, skips out-of-bounds indices to prevent illegal GM writes, and preserves
    raw bit patterns on 'none' replacement
  - Last-axis gather in Ascend C must separate UB-resident table lookup (using Vector
    Gather with byte offsets) from large-axis GM random access
  keywords:
  - gather_v2
  - scatter_elements
  - masked_select
  - unsorted_segment_sum
  - unique_950
  - negative_index_support
  - bit_pattern_preservation
  - gather_last_axis_modes
---
## Preconditions & Signals

Apply this SOP when building or optimizing index-driven gather, scatter, segment reduction, and stream-compaction operators (`gather_v2`, `scatter_elements`, `gather_950`, `scatter_950`, `unsorted_segment_sum_950`, `unique_950`, `masked_select_950`):

1. **Three Structural Regimes by Post-Axis Slice Size ($S_{\text{inner}} = \prod_{k > \text{axis}} \text{shape}[k]$)**:
   - **Regime 1 — Large Contiguous Inner Slice ($S_{\text{inner}} \cdot \text{sizeof}(T) \ge 32\text{B}$, e.g., Embedding Lookup `[V, D]`)**: Each index selects a contiguous row of $S_{\text{inner}}$ elements; optimization is dominated by burst `DataCopy` / `DataCopyPad` and small-D multi-row coalescing (`P3`).
   - **Regime 2 — Last-Axis Gather with UB-Resident Source Row ($S_{\text{inner}} == 1$ and axis length $A$ fits in UB)**: Load the entire source row `x[outer, 0:A]` into UB once and extract indexed elements at full Vector speed via hardware `Gather` (`gather_last_axis_modes_and_signed_index_normalization.md`).
   - **Regime 3 — Last-Axis Gather/Scatter with Huge $A$ ($S_{\text{inner}} == 1$ and $A$ exceeds UB)**: Use SIMT thread-coalesced global loads/atomics on Ascend 950 or tiled range-filtering passes in UB.
2. **Signed Index Normalization & Bit-Pattern Fidelity (`gather_v2`, `scatter_elements`)**:
   - Floating-point vector pipelines may canonicalize signaling `NaN` payloads or flush subnormals/negative zero; pure movement modes (`gather_v2`, `scatter_elements(reduction="none")`) must move payloads via same-bitwidth unsigned integer views (`uint8`, `uint16`, `uint32`, `uint64`).

## Procedure

### Step 1: `gather_v2` Canonical Shape Lowering, Index Normalization & Bit-Preserving Transfer
1. **3D/4D Canonical Shape Collapse (`gather_v2.py`, `gather_950.md`)**:
   - Normalize `axis` and `batch_dims` ($0 \le \text{batch\_dims} \le \text{axis} < \text{rank}(x)$). Collapse dimensions into four canonical extents:
     $$B = \prod_{k < \text{batch\_dims}} x.\text{shape}[k], \quad M_{\text{outer}} = \prod_{\text{batch\_dims} \le k < \text{axis}} x.\text{shape}[k], \quad A = x.\text{shape}[\text{axis}], \quad S_{\text{inner}} = \prod_{k > \text{axis}} x.\text{shape}[k]$$
     and collapse `indices.shape[batch_dims:]` into $K_{\text{idx}}$. Output shape is `x.shape[:axis] + indices.shape[batch_dims:] + x.shape[axis + 1:]`.
2. **Single-Addition Negative Index Normalization & Out-of-Bounds Zeroing**:
   - When `negative_index_support=True`, add $A$ **once** to negative indices (`idx_norm = idx < 0 ? idx + A : idx`).
   - Check validity as $0 \le \text{idx\_norm} < A$ (unsigned comparison `(uint64_t)idx_norm < (uint64_t)A`). If out-of-bounds or if $A == 0$, output exact `0` without issuing an out-of-bounds GM/UB read.
3. **Integer-View Transfer for Bit-Exact Fidelity**:
   - Reinterpret `float16`/`bfloat16` as `uint16`, `float32` as `uint32`, `complex64` as `uint64` (or pair of `uint32`), and `complex128` as `2 x uint64` so `NaN` payloads and `-0.0` bits pass through untouched.

### Step 2: `scatter_elements` & `unsorted_segment_sum` Conflict-Safe Execution
1. **Out-of-Bounds Guard & Reduction Modes (`scatter_elements.py`, `scatter_950.md`)**:
   - Clone `data` to the output buffer first (`scatter_elements` does not mutate `data` in-place).
   - Validate $0 \le \text{indices}[p] < \text{data.shape}[\text{axis}]$ (negative indices are *not* normalized in `scatter_elements`; out-of-bounds indices are skipped in device code to prevent illegal memory access `507015`/`507035`).
   - For `reduction="none"`, write via integer views to preserve bit patterns; for `reduction="add"` and `"mul"`, partition work across cores along the **non-axis outer/inner coordinates** $(M_{\text{outer}}, S_{\text{inner}})$ so each core owns disjoint output lines and applies sequential updates along `indices` in UB without cross-core race conditions!
2. **Segment Reduction (`unsorted_segment_sum_950`)**:
   - When `num_segments` is small and $S_{\text{inner}}$ fits in UB, keep segment accumulators in UB per core and reduce across cores via workspace or `SetAtomicAdd<float>()`; skip negative `segment_ids` (no-op per contract).

### Step 3: Stream Compaction (`masked_select_950`) & Deduplication (`unique_950`)
1. **Two-Pass Prefix-Count `MaskedSelect` (`index_validation_prefix_counts_and_piece_ownership.md`)**:
   - **Pass 1 (Per-Core Count)**: Each AIV core counts active mask bits in its assigned 1D input slice using `GatherMask(..., rsvdCnt)` and writes `coreCount[core_id]` to GM workspace.
   - **Pass 2 (Prefix Sum & Compact Write)**: After `SyncAll()`, each core computes its global output write offset $\text{outOffset} = \sum_{c < \text{core\_id}} \text{coreCount}[c]$, re-runs `GatherMask` in UB chunks, and writes compacted valid elements contiguously to `outGM + outOffset`.
2. **Parallel Radix/Merge Sort + Adjacent Diff for `Unique` (`unique_950`)**:
   - Sort input elements (with original indices if `return_inverse=True`) using two-level `Sort32` + `Mrgsort` (`P111`), mark boundary transitions $x_{\text{sorted}}[i] \ne x_{\text{sorted}}[i-1]$, and compact unique elements via `GatherMask`.

## Verification & Checks

1. **Duplicate Index Determinism in `scatter_elements(reduction="none")`**:
   - When multiple positions in `indices` map to the same output coordinate under `reduction="none"`, partitioning across cores along the index dimension creates a data race; always partition across cores along the independent non-scatter coordinates $(M_{\text{outer}}, S_{\text{inner}})$.
2. **Empty Index Axis ($A == 0$ or `indices.numel() == 0`)**:
   - Verify that `gather_v2` with $A == 0$ and non-empty `indices` fills the output with zeros rather than dividing by zero (`idx % A`) or dereferencing a null `x` pointer.

## Failure Modes & Fallbacks

- **Symptom**: `gather_v2` flips signaling `NaN` (`0x7f800001`) to quiet `NaN` (`0x7fc00001`) on `float32` inputs.
  - **Root Cause**: Using floating-point `Select` or `Adds(..., 0.0f)` in Vector pipes to zero out-of-bounds positions, which triggers IEEE 754 NaN quietening.
  - **Fix**: Reinterpret the buffer as `uint32_t` / `int32_t` in UB and apply bitwise `And` with a `0xFFFFFFFF` / `0x00000000` validity mask.
