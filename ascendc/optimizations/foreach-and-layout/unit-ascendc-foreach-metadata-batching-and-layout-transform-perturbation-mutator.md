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
  - ForeachOps
  - ForeachErfc
  - ForeachLerpScalar
  - ForeachMulList
  - Reshape
  - Transpose
  - Permute
  - StridedSlice
domain_path: ascendc/optimizations/foreach-and-layout
evidence_strength: reproduced_empirical
id: unit-ascendc-foreach-metadata-batching-and-layout-transform-perturbation-mutator
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
- unit-ascendc-simt-and-regbase-vector-programming-sop
provenance:
  attribution: Distilled from CANNBot-Knowledge controlled_code_perturbation/elementwise_foreach
    and layouttransform empirical records and evolution strategies X3, P100, P114,
    P127
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_foreach/foreach_lerp_scalar_vectorization_compute_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_foreach/foreach_mul_list_compute_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/layouttransform/reshape_data_movement_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p114_transpose_nd_tiling_with_alignment.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p127_conversion_pattern_specialization_routing.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x3.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/elementwise_foreach/index.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-elementwise-unary-binary-and-mask-predicate-perturbation-mutator
- unit-ascendc-datacopypad-elimination-ub-bank-conflict-and-l2-tiling-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Controlled code perturbation mutator for Ascend C multi-tensor Foreach operators
  and Layout-Transform (reshape, transpose, permute, strided_slice) kernels, eliminating
  per-tensor launch/scalar overhead and replacing generic gather with aligned DMA
  + GatherMask/Vnchwconv routes.
tier: L4
title: Ascend C Foreach Tensor-List Metadata Batching & Layout-Transform Perturbation
  Mutator
triggers:
  error_signatures:
  - Foreach multi-tensor kernels must amortize per-tensor metadata lookup and avoid
    host-side FP32 upcast of FP16/BF16 tensor lists (End-to-End Dtype Passthrough)
  - Layout-transform kernels (transpose, permute, reshape copy) must route through
    specialized contiguous-burst, Brcb/GatherMask, or 2D tile-transpose paths rather
    than generic element-wise gather
  keywords:
  - elementwise_foreach
  - layouttransform
  - foreach_mul_list
  - foreach_lerp_scalar
  - foreach_erfc
  - transpose_tiling
  - gather_mask_transpose
  - dtype_passthrough
  - conversion_routing
---
## Preconditions & Signals

Trigger this mutator when optimizing multi-tensor `foreach_*` kernels (`foreach_erfc`, `foreach_lerp_scalar`, `foreach_minimum_scalar`, `foreach_mul_list`, `foreach_norm`) or layout/data-movement kernels (`reshape`, `transpose`, `permute`, `strided_slice`, `cat`):

1. **Multi-Tensor `foreach` Imbalance & Wrapper Upcast Overhead**:
   - Tensor lists contain heterogeneous tensor sizes (e.g., 1D bias vectors mixed with 2D weight matrices); naive 1-tensor-per-core assignment leaves most AIV cores idle while one core processes the largest tensor.
   - Host wrapper converts FP16/BF16 tensor lists to FP32 via `at::to(kFloat)` before launching the kernel (`disc_x3`), tripling GM traffic and launch count.
2. **Layout-Transform Generic Gather Trap (`P127`)**:
   - `transpose`, `permute`, or non-contiguous `reshape` computes per-element source coordinates in Scalar/Vector loops and issues unaligned or single-element DMA/Gather reads instead of specializing by axis permutation class.

## Mutation Moves (Ranked by Expected Gain)

### Mutation 1: End-to-End Dtype Passthrough & Virtual Chunk Flattening for `foreach_*` (`disc_x3` + `P58`)
- **Target**: Multi-tensor `foreach_*` host tiling and device kernel loop (`foreach_mul_list`, `foreach_lerp_scalar`, `foreach_erfc`).
- **Transformation**:
  1. **End-to-End Dtype Passthrough (`disc_x3`)**: Pass original `FP16`/`BF16` tensor pointers directly into the kernel via `DynamicTensorList` descriptor arrays; perform `Cast` to `FP32` inside UB (`TBuf<VECCALC>`), execute the arithmetic (`lerp`, `erfc`, `mul`), and cast back to `FP16`/`BF16` before `DataCopy` to GM.
  2. **Virtual Chunk Load Balancing**: On Host, partition the tensor list into uniform virtual work chunks `(tensor_idx, chunk_offset, chunk_len)` across all active AIV cores using a greedy bin-packing / prefix-sum cursor so large tensors split across multiple cores and tiny tensors batch onto a single core.
  3. **Batch Small Tensors in UB**: When consecutive small tensors share the same core, coalesce their UB slots and hoist `SetVectorMask` outside the inner tensor-chunk loop.

### Mutation 2: Conversion Pattern Specialization Routing (`P127` + `P114` + `P100`)
- **Target**: `transpose`, `permute`, `reshape`, and `strided_slice` kernels.
- **Transformation**:
  1. **Collapse Invariant Axes First**: Merge any adjacent dimensions whose relative order is preserved by the permutation (`[d0, d1, d2, d3]` with perm `(0, 2, 1, 3)` $\rightarrow$ 3D `[B, M, N]` perm `(0, 2, 1)`).
  2. **Route 1 — Trailing Axis Preserved (`perm[-1] == rank - 1`)**: The innermost dimension $D_{\text{last}}$ is contiguous in both source and destination! Issue contiguous burst `DataCopy` (or `DataCopyPad` when $D_{\text{last}}$ is not 32B-aligned) with multi-row strided DMA descriptors (`srcStride` / `dstStride`), completely bypassing UB vector permutation.
  3. **Route 2 — Small-Channel Transpose (`[C, N] -> [N, C]` with $C \le 16$ or Interleaved Unpack, `P114` + `P100`)**: Load contiguous `[C, N_tile]` blocks into UB and use `GatherMask` (or `Vnchwconv` $16 \times 16$ hardware transpose blocks) to transpose in UB at full Vector bandwidth before burst-writing `[N_tile, C]` to GM.
  4. **Route 3 — General Last-Axis Swap (`[M, N] -> [N, M]`)**: Tile into $32\text{B} \times 32\text{B}$ (e.g., $16 \times 16$ FP16/BF16) sub-blocks in UB, transpose in-UB via `Vnchwconv` / `Transpose`, and write contiguous rows to GM (`P26`).

### Mutation 3: Coalesced Large-Tile DMA & Loop Unrolling for Pure Data-Movement (`reshape_data_movement_a2`)
- **Target**: Pure or near-pure copy kernels (`reshape`, `cat`, `slice`).
- **Transformation**:
  1. Maximize UB `tileLength` up to the double-buffered UB ceiling (`~80 KB` per buffer on 192 KB UB) so each `DataCopy` transfers tens of kilobytes per descriptor rather than small 1–4 KB fragments (`disc_x95`: minimize DMA descriptor count).
  2. Eliminate redundant intermediate `VECIN -> VECCALC -> VECOUT` copies when no type conversion is needed—bind `TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2>` (`P22`) to copy directly `GM -> UB -> GM`.

## Guardrails & Verification

1. **Bit-Pattern Preservation on Pure Copy / Reshape**:
   - When moving data without arithmetic (`reshape`, `transpose`, `cat`, `gather_v2`), process elements via same-bitwidth integer views (`uint16_t` for FP16/BF16, `uint32_t` for FP32) wheneverVector instructions touch the data, so quiet/signaling `NaN` payloads and `-0.0` bit patterns are never altered.
2. **`DynamicTensorList` Pointer Alignment in `foreach_*`**:
   - Each tensor in a `foreach` list has an independent GM base address and tail length; never assume tensor $k+1$ is contiguous with tensor $k$ in GM, and apply `DataCopyPad` independently on each tensor's tail chunk.

## Fallbacks

- If trailing dimension $D_{\text{last}} < 32\text{B}$ in a non-last-axis permutation and `DataCopyPad` stride constraints are exceeded, pack multiple inner rows into UB using a precomputed `Gather` byte-offset table (`P33`) resident in UB.
