---
category: axiom
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
  - VectorSimd
  - SoftMax
  - SwiGLU
  - SortTopK
  - Gather
  - GatherMask
  - Brcb
domain_path: ascendc/apis/vector-activation-sort
evidence_strength: software_contract
id: unit-ascendc-vector-simd-softmax-activation-sort-and-gather-api-contract
inherits_from:
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  attribution: Distilled from CANNBot-Knowledge ops/ascendc/apis/simd_api/basic_api/memory_vector_compute,
    adv_api, and appendix/mask_interface_summary software contracts
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/general_description_and_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/appendix/mask_interface_summary.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/aux_data_structures/binary_repeat_params.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/adv_api/sort_operations/get_sort_len.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/adv_api/tensor_transform/broadcast.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/general_description_and_constraints.md
  source_type: public_software_contract_review
  supersedes: []
related:
- unit-ascendc-evolutionary-strategy-vector-simd-and-ub-tiling-mutator
- unit-ascendc-wholereduce-blockreduce-and-argreduce-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Foundational Ascend C software contract for basic Vector SIMD arithmetic/masking,
  Gather/GatherMask/Brcb data permutation, Select predicate muxing, and high-level
  Vector APIs (SoftMax, FlashSoftMax, SwiGLU, GeLU, Sort/Sort32/Mrgsort, TopK).
tier: L1
title: Ascend C Vector SIMD, Mask/Broadcast, SoftMax/Activation, Sort/TopK & Gather/GatherMask
  API Contract
triggers:
  error_signatures:
  - Every Vector instruction operates on 256B per repeat (128 half/bfloat16 or 64
    float/int32 elements, max repeatTimes = 255) with 32B-aligned UB operands
  - Gather byte offsets in srcOffsetLocal must be non-negative multiples of sizeof(T);
    GatherMask requires explicit rsvdCnt output reference to report extracted element
    count
  - Sort32 / Sort and Mrgsort require 32-element aligned UB buffers padded with -inf
    (descending) or +inf (ascending) on invalid tail slots
  keywords:
  - vector_simd_api
  - binary_repeat_params
  - unary_repeat_params
  - set_vector_mask
  - gather
  - gather_mask
  - brcb
  - broadcast
  - softmax_flash_v2
  - sort32
  - mrgsort
  - select
---
## Core Axioms & Invariants

### 1. Vector Execution Width (`256B`), Mask Modes & Repeat Strides
1. **Single-Repeat Geometry (`256 Bytes = 8 Datablocks of 32B`)**:
   - Each Vector repeat processes `256 bytes`: `128` elements for `half`/`bfloat16_t`/`int16_t`/`uint16_t` (`8 blocks x 16 elems`), or `64` elements for `float`/`int32_t`/`uint32_t` (`8 blocks x 8 elems`).
   - Hardware `repeatTimes` is an `uint8_t` in `[1, 255]`. For element counts $> 255 \times \text{elemsPerRepeat}$ (`> 32640` FP16 or `> 16320` FP32), invoke the High-Level overload `Op(dst, src0, src1, calCount)` or chunk the loop into $\le 255$-repeat segments.
2. **`MaskMode::NORMAL` vs `MaskMode::COUNTER` (`mask_interface_summary.md`)**:
   - **`MaskMode::NORMAL`**: Mask is either a contiguous lane count `mask` $\in [1, 128]$ (for 16-bit) / $[1, 64]$ (for 32-bit) or a `uint64_t mask[2]` bitmask applied identically to every repeat in `repeatTimes`.
   - **`MaskMode::COUNTER`**: Call `SetVectorMask<T, MaskMode::COUNTER>(totalElementCount)` before the Vector instruction; hardware automatically executes $\lfloor N / W \rfloor$ full repeats and masks the trailing $N \bmod W$ lanes on the final repeat (`P67`). Reset to `SetMaskNorm()` before calling APIs that require `MaskMode::NORMAL`.
3. **`UnaryRepeatParams` & `BinaryRepeatParams` (`binary_repeat_params.md`)**:
   - `{dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride}`:
   - All stride parameters are measured in **`32B` datablocks**! Standard contiguous execution uses `blkStride = 1` (adjacent 32B blocks within a repeat) and `repStride = 8` (`8 x 32B = 256B` advance between repeats). Setting `src1RepStride = 0` repeats the same 256B `src1` vector across all repeats of `src0`.

### 2. Permutation, Broadcast & Predicate Selection Primitives
1. **`Gather(dstLocal, srcLocal, srcOffsetLocal, srcBaseAddr, mask, repeatTimes, dstRepStride)`**:
   - Every element in `srcOffsetLocal` (`LocalTensor<uint32_t>`) is a **byte offset** (`index * sizeof(T)`) relative to `srcLocal + srcBaseAddr`.
   - `srcLocal` must be 32B-aligned, and every active byte offset must satisfy `0 <= byteOffset < srcBufferBytes` and `byteOffset % sizeof(T) == 0`.
2. **`GatherMask(dstLocal, src0Local, src1Pattern, reduceMode, mask, GatherMaskParams, rsvdCnt)`**:
   - Built-in patterns (`src1Pattern = 1` extracts odd-indexed elements `0, 2, 4, ...`; `src1Pattern = 2` extracts even-indexed elements `1, 3, 5, ...`; `3..7` for 4-tuple/stride patterns) or custom bitmask tensor `src1Pattern` (`LocalTensor<uint16_t/uint32_t>`).
   - Writes compact matching elements contiguously into `dstLocal` and outputs the exact extracted element count into `uint32_t& rsvdCnt`.
3. **`Brcb(dstLocal, src0Local, repeatTimes, BrcbRepeatParams)` & `BroadCast`**:
   - `Brcb` broadcasts each scalar element $i \in [0, 8 \cdot \text{repeatTimes})$ of `src0Local` to fill an entire `32B` datablock (`16` `half`/`bfloat16_t` or `8` `float`/`int32_t` elements) in `dstLocal`.
   - High-level `BroadCast<T, dimNum, axis>(dst, src, dstShape, srcShape, sharedTmpBuffer)` supports general multi-dimensional broadcasting using host-queried `GetBroadCastMaxMinTmpSize`.
4. **`Compare` / `CompareScalar` + `Select`**:
   - `Compare(dstMask, src0, src1, CMPMODE::LT/GT/EQ/LE/GE/NE, ...)` writes packed 1-bit per lane predicate masks into `LocalTensor<uint8_t/uint16_t> dstMask` (whose address and repeat stride must respect mask alignment).
   - `Select(dst, mask, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, ...)` selects `src0[i]` where `mask` bit is `1` and `src1[i]` where `mask` bit is `0`.

### 3. High-Level Normalization, Activation & Sort/TopK APIs
1. **`SoftMax` & `FlashSoftMax` (`SoftMaxFlashV2` / `OnlineSoftMax`)**:
   - `SoftMax<T, isReuseSource, isDataFormatNZ>(dst, expSum, maxOut, src, sharedTmp, softmaxTiling)`: computes row-wise max, subtracted exp, sum, and normalization in one call; requires host `SoftMaxTilingFunc` and `GetSoftMaxMaxMinTmpSize`.
   - `SoftMaxFlashV2` maintains running `(max, sum)` state across KV tiles and outputs the rescaling factor $\exp(m_{\text{old}} - m_{\text{new}})$ for online FlashAttention updates.
2. **High-Level Activations (`SwiGLU`, `GeLU`, `FastGelu`, `Silu`, `Sigmoid`, `Tanh`)**:
   - Prefer fused high-level APIs (`SwiGLU`, `FastGelu`) or `BF16/FP16 -> FP32` cast + basic `Muls`/`Exp`/`Adds`/`Div` chains (`A1`) when strict FP32 intermediate accuracy is required.
3. **Hardware Sorting & TopK (`Concat`, `Sort` / `Sort32`, `Mrgsort`, `Extract`, `TopK`)**:
   - **`Sort32(dstLocal, srcLocal, indexLocal, repeatTimes)`**: Sorts 32-element blocks of `(value, index)` pairs in descending order using the hardware sort unit.
   - **`Mrgsort<T, isExhaustedSuspension>(dstLocal, MrgsortSrcList, params)`**: Merges up to 4 sorted streams in UB into a single sorted output stream; stream lengths must be multiples of 32 bytes / elements per platform contract.
   - **`Extract(dstValue, dstIndex, srcSorted, repeatTimes)`**: Unpacks interleaved sorted `(value, index)` structures back into separate `dstValue` and `dstIndex` UB tensors.

## Verification Checklist

- [ ] `SetMaskNorm()` is restored after any `MaskMode::COUNTER` block before invoking APIs that expect `MaskMode::NORMAL`.
- [ ] `Gather` index tensors are multiplied by `sizeof(T)` (byte offsets) and cast to `uint32_t` before calling `Gather`.
- [ ] Unfilled tail elements in `Sort32` / `TopK` input blocks are initialized to $-\infty$ (for descending Top-K) before `Concat` / `Sort32`.
