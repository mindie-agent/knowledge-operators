---
category: sop
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
  - Ascend950PR
  - Ascend950DT
  workloads:
  - MemoryBoundKernels
  - DataMovement
  - VectorKernels
  - MatmulL2Tiling
  - StridedDMA
domain_path: ascendc/optimizations/mte-memory-alignment
evidence_strength: reproduced_empirical
id: unit-ascendc-datacopypad-elimination-ub-bank-conflict-and-l2-tiling-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend950dt-measured-bandwidth-and-dt-vs-pr-overrides-measurement
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/guides/data_movement, guides/memory,
    and evolution strategy cards P52, P56, P64, P65, P66, P74, P93, P105, P107, X95
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/data_movement/dma_optimization_units_alignment_and_batched_outputs.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/data_movement/datacopypad_rows_and_padding.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/memory/alignment_units_layout_stride_and_padding_values.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p65_ub_bank_conflict_avoidance.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p66_gm_512b_alignment.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x95.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/data_movement/dma_optimization_units_alignment_and_batched_outputs.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-datacopy-datacopypad-tpipe-tque-and-flag-sync-api-contract
- unit-ascendc-cube-l1-residency-l0-pingpong-and-fixpipe-fusion-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for MTE2/MTE3 memory subsystem optimization
  in Ascend C, covering DataCopyPad minimization via row coalescing and interior/tail
  splitting, 512B GM address alignment, UB bank-conflict padding, DMA descriptor minimization,
  and L2-aware serpentine tiling.
tier: L2
title: Ascend C DataCopyPad Elimination, UB Bank-Conflict Avoidance & 512B/L2 Cache
  Tiling SOP
triggers:
  error_signatures:
  - GM addresses aligned to 512 bytes maximize HBM burst efficiency and avoid cross-core
    512B cache-line write serialization
  - UB on 910B/910C is organized into 48 banks (16 bank groups x 3 banks, 4 KB per
    bank); power-of-two stride accesses that hit the same bank group must be padded
    by 32B
  - DataCopyPad has lower MTE2/MTE3 throughput than aligned DataCopy; multi-row merging
    or interior/boundary loop splitting should restrict DataCopyPad strictly to unaligned
    tails
  keywords:
  - datacopypad_elimination
  - ub_bank_conflict
  - gm_512b_alignment
  - l2_cache_tiling
  - serpentine_traversal
  - selective_l2_disable
  - dma_descriptor_count
  - batched_datacopy_output
---
## Preconditions & Signals

Apply this SOP when `msprof` or Roofline analysis shows a kernel is memory-bound (`mte2_ratio > 0.65` or `mte3_ratio > 0.60`) yet achieved HBM bandwidth is far below the hardware ceiling (`< 50%` of `~1.6 TB/s` on 910B or `< 1.0 TB/s` on 950DT):

1. **High DMA Descriptor Count (`disc_x95`) & `DataCopyPad` Overuse**:
   - The kernel issues one `DataCopy` or `DataCopyPad` per short row (`blockCount = 1`, `blockLen` small) inside a loop instead of coalescing contiguous rows or using multi-block strided DMA (`blockCount > 1`), making MTE command-issue latency the bottleneck rather than byte volume.
2. **UB Bank Conflict Stalls (`vec_bank_group_cflt` / `ub_read_conflict`, `P65`)**:
   - Vector instructions read two source operands (`src0`, `src1`) or write `dst` at UB offsets separated by exact multiples of `16 * 32B = 512B` (or `4096B`), causing concurrent reads to collide on the same UB bank group.
3. **512B GM Address Conflict & L2 Thrashing (`P64`, `P66`, `P93`, `P105`, `P107`)**:
   - Multiple cores simultaneously read the exact same GM address (`k=0` tile of $A$ or $B$) or write to adjacent non-512B-aligned tail segments, or row-major tile traversal evicts shared $B$ panels from L2 cache on every $M$-row switch.

## Procedure

### Step 1: Minimize DMA Descriptor Count & Restrict `DataCopyPad` to True Boundaries (`disc_x95`, `P56`)
1. **Coalesce Small-D Contiguous Rows (`P3`, `P56`)**:
   - When rows of length $D$ are contiguous in GM (`row_stride == D`), never copy 1 row per `DataCopyPad`. Load $R$ rows at once as a single contiguous chunk of $R \times D$ bytes whenever $R \times D \bmod 32 == 0$ (e.g., if $D = 24\text{B}$, $R = 4$ rows $\implies 96\text{B} = 3 \times 32\text{B}$ aligned!), or load a single multi-row `DataCopyPad(ub, gm, {blockCount=R, blockLen=D_bytes, srcStride=0, dstStride=pad_blocks}, padParams)` descriptor.
   - Similarly, accumulate multiple output rows in UB and emit a single batched `UB -> GM` `DataCopy` (`P56`).
2. **Interior Aligned Fast-Path vs Boundary Tail Path**:
   - Split the tile loop into `0 .. numAlignedTiles - 1` (using full-speed aligned `DataCopy`) and a single final tail step `numAlignedTiles` (using `DataCopyPad` only for the remaining unaligned bytes).

### Step 2: UB Bank-Conflict Avoidance (`P65`)
1. **Understand UB Bank Geometry**:
   - On Ascend 910B/910C, the 192 KB UB is partitioned into 48 banks (16 bank groups $\times$ 3 banks, `4 KB` per bank, `32B` width per bank access). Addresses separated by multiples of `16 * 32B = 512B` (and especially `4 KB`) map to the same bank group.
2. **Apply +32B Odd-Block Stride Padding & Offset Staggering**:
   - When allocating `TBuf<VECCALC>` slices for multi-input Vector instructions (`Add(dst, src0, src1)` or `MulAddDst`), stagger the starting offset of `src1` or `dst` by an odd multiple of `32B` (`+32B` or `+64B`), or pad the 2D UB row stride from $256\text{B} \times k$ to $256\text{B} \times k + 32\text{B}$ so simultaneous `src0`, `src1`, and `dst` accesses land in distinct bank groups.

### Step 3: 512B GM Alignment & Cross-Core K-Axis Staggering (`P64`, `P66`, `P93`)
1. **512B Per-Core GM Partition Alignment (`P64`, `P66`)**:
   - MTE2/MTE3 and L2 cache lines operate on `512B` (`16 x 32B`) aligned granularity. Round per-core GM chunk sizes `singleCoreLen` to multiples of `512B` (`256` FP16/BF16 elements or `128` FP32 elements) so two cores never write to the same `512B` GM line concurrently.
2. **K-Axis Staggered Access in Multi-Core Matmul (`P93`)**:
   - When multiple AIC cores read the same $A$ or $B$ matrix along the $K$ loop, offset the starting $K$ tile index by `core_idx`:
     $$k_{\text{actual}} = (k_{\text{step}} + \text{block\_idx}) \bmod K_{\text{tiles}}$$
     so all cores do not hammer the exact same GM address at $k_{\text{step}} = 0$.

### Step 4: L2 Cache Serpentine Traversal & Selective L2 Disable (`P52`, `P74`, `P105`, `P107`)
1. **L2-Aware Tile Grouping & Serpentine Traversal (`P105`, `P107`)**:
   - Group output tiles into $M_{\text{group}} \times N$ swizzle bands sized so $M_{\text{group}}$ tiles of $A$ plus 1 tile of $B$ fit within L2 cache.
   - Traverse $N$-tiles in alternating serpentine (boustrophedon) order (`left-to-right` on even $M$-bands, `right-to-left` on odd $M$-bands) so the trailing $B$ tiles of band $g$ remain hot in L2 for the leading $B$ tiles of band $g+1$.
2. **Selective L2 Cache Bypass for Streaming Cold Data (`P52`, `P74`)**:
   - When streaming massive single-use KV caches or elementwise tensors that far exceed L2 capacity (e.g., $> 192\text{ MB}$) alongside reusable weights, call `SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)` on the cold streaming GM tensors to prevent evicting hot reusable weights from L2.

## Verification & Checks

1. **`DataCopyPad` `dstStride` vs UB Row Stride Match**:
   - Verify that when `DataCopyPad` pads a row of `blockLen` bytes up to `alignedRowBytes` in UB, subsequent Vector instructions step by `alignedRowBytes / sizeof(T)` elements per row, while reductions mask out the `alignedRowBytes - blockLen` padded tail bytes.
2. **Staggered K-Loop Accumulator Initialization (`P93`)**:
   - When staggering $k_{\text{actual}} = (k_{\text{step}} + \text{block\_idx}) \bmod K_{\text{tiles}}$, verify that L0C accumulator initialization (`cInit = true` / `isL0CAccum = false`) is keyed on `k_step == 0` (the *first step executed*, regardless of `k_actual`), never on `k_actual == 0`!

## Failure Modes & Fallbacks

- **Symptom**: Non-deterministic corrupt bytes at the boundary between Core $i$'s end address and Core $i+1$'s start address in GM.
  - **Root Cause**: Core $i$'s MTE3 write is not 32B-aligned or uses `DataCopy` instead of `DataCopyPad` on its tail, overwriting the first few bytes of Core $i+1$'s output.
  - **Fix**: Enforce 32B/512B alignment on all non-final cores' partition sizes (`singleCoreBytes % 512 == 0`) so only the very last active core can ever have an unaligned tail, and use `DataCopyPad` on that tail.
