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
  - Matmul
  - GroupedMatmul
  - QuantMatmul
  - FlashAttentionCube
  - Conv2dIm2col
domain_path: ascendc/optimizations/cube-l1-l0-pipeline
evidence_strength: reproduced_empirical
id: unit-ascendc-cube-l1-residency-l0-pingpong-and-fixpipe-fusion-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend910b-910c-and-310p-memory-model-and-target-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/optimizations/matmul guides
    and evolution strategy cards P18, P46, P53, P70, P86, P87
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/matmul/independent_l1_l0_double_buffering.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/matmul/last_wave_2d_subdivision.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/matmul/edge_tile_size_redistribution.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/matmul/dequant_bank_padding_and_l1_copy_stride.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/matmul/merged_scale_loading.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/matmul/software_prefetch_task_identity.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/matmul/independent_l1_l0_double_buffering.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-cannbot-dsl-matmul-and-catlass-gemm-epilogue-sop
- unit-ascendc-matmul-impl-iterate-gettensorc-and-gemm-api-contract
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Ascend C Cube pipeline optimization, covering
  independent L1/L0 ping-pong buffering, L1 resident matrix reuse, last-wave 2D tile
  subdivision, edge-tile redistribution, merged scale loading with bank padding, and
  FixPipe inline epilogue fusion.
tier: L2
title: Ascend C Cube L1 Residency, Independent L1/L0 Ping-Pong, Last-Wave Subdivision
  & FixPipe Fusion SOP
triggers:
  error_signatures:
  - Independent L1 and L0 double buffering requires separate producer-consumer event
    pairs (MTE2<->MTE1 for L1 and MTE1<->M for L0A/L0B); dual addresses alone do not
    guarantee overlap
  - Last-wave 2D subdivision must distinguish global last wave from per-core final
    iteration and synchronize only active participant cores
  - Dequant scale loading in QuantMatmul should merge multi-row/group scale transfers
    and pad UB/L1 dequant tables to avoid bank conflicts
  keywords:
  - independent_l1_l0_double_buffering
  - last_wave_2d_subdivision
  - edge_tile_size_redistribution
  - dequant_bank_padding
  - merged_scale_loading
  - fixpipe_fusion
  - mte2_preload
  - l1_resident_reuse
---
## Preconditions & Signals

Apply this SOP when optimizing Cube-bound or MTE2/MTE1-stalled GEMM, GroupedMatmul, QuantMatmul, or FlashAttention Cube pipelines in Ascend C:

1. **False Double-Buffering Serialization (`aic_mte1_ratio` and `aic_mte2_ratio` Non-Overlapping)**:
   - Two L1 addresses (`A1_ping`, `A1_pong`) and two L0 addresses (`A2_ping`, `A2_pong`) are allocated, yet MTE2 waits for the entire K-step MMAD to finish before loading the next L1 tile because synchronization flags are placed inside the same lockstep loop body (`independent_l1_l0_double_buffering.md`).
2. **Severe Wave Quantization Tail (`last_wave_2d_subdivision.md`, `edge_tile_size_redistribution.md`)**:
   - When total output tiles $T_{\text{total}}$ leaves a small remainder $R = T_{\text{total}} \bmod N_{\text{AIC}}$ in the final wave, $N_{\text{AIC}} - R$ Cube cores sit idle during the entire K-loop of the last wave, or narrow edge tiles (`tailM < 32`) degrade MTE2 burst efficiency.
3. **Quantized Matmul Scale & Bank Stall (`merged_scale_loading.md`, `dequant_bank_padding_and_l1_copy_stride.md`)**:
   - Per-tile scale loads issue tiny MTE2 descriptors inside the K-loop, or power-of-two stride access in UB/L1 dequant tables hits bank conflicts.

## Procedure

### Step 1: Independent L1 and L0 Ping-Pong Buffering (`independent_l1_l0_double_buffering`)
1. **Decouple L1 Step Ratio ($K_{\text{L1}} = r \cdot K_{\text{L0}}$, $r \ge 2$)**:
   - Size L1 tiles (`stepKa`, `stepKb`) as a multiple $r \in \{2, 4\}$ of the L0 MMAD step (`baseK`). An L1 slot remains live across $r$ consecutive `MTE1` (`LoadData` L1 $\rightarrow$ L0) sub-steps and must only be released (`SetFlag<HardEvent::MTE1_MTE2>(l1_slot_event)`) after the $r$-th L0 load completes.
2. **Bidirectional Per-Slot Event Discipline**:
   - Maintain two independent event pairs per memory level:
     - **L1 Level**: `MTE2 -> MTE1` (L1 data ready for L0 load) and `MTE1 -> MTE2` (all $r$ L0 sub-steps finished reading L1 slot; safe for MTE2 overwrite).
     - **L0 Level**: `MTE1 -> M` (L0A/L0B sub-tile ready for `Mmad`) and `M -> MTE1` (`Mmad` finished consuming L0A/L0B slot; safe for next L1-to-L0 load).
   - Pre-seed the "`free to write`" credits (`MTE1_MTE2` and `M_MTE1`) before entering the K loop (or guard via `step_idx >= 2`), and drain remaining in-flight slots after the loop without issuing out-of-bounds MTE2 prefetches (`software_prefetch_task_identity.md`).

### Step 2: L1 Resident Reuse & 7-Buffer Partitioning (`P18`, `P53`, `P83`)
1. **Pin Invariant Left/Right Operands in L1**:
   - In FlashAttention (`Q` tile across KV loop) or small-M / repeated-B GEMM, load the invariant operand into a dedicated resident L1 partition once per outer task (`depthA1` / resident `TBuf<TPosition::A1>`) and reuse it across all inner N/K iterations without re-issuing `GM -> L1` MTE2 transfers.
2. **L1 Multi-Buffer Partitioning (`512 KB` Budget)**:
   - Partition the 512 KB L1 explicitly between resident buffers (e.g., `Q_L1`) and ping-pong streaming buffers (`K_L1_ping/pong`, `V_L1_ping/pong`) so every slice is 512B-aligned and zero dynamic L1 reallocation occurs mid-loop.

### Step 3: Last-Wave 2D Subdivision & Edge-Tile Size Redistribution
1. **Edge-Tile Size Redistribution (`edge_tile_size_redistribution`)**:
   - When `M % baseM` or `N % baseN` produces a tiny trailing edge tile (e.g., `baseM = 128`, `tailM = 16`), redistribute the last two tiles into two balanced medium tiles (e.g., `64 + 80`, both aligned to fractal 16) on Host so neither the penultimate nor the final tile suffers severe MTE2/Cube underutilization.
2. **Last-Wave 2D Sub-Block Splitting (`last_wave_2d_subdivision`)**:
   - When the global final wave has $R < N_{\text{AIC}}$ parent tiles, subdivide each parent `(baseM, baseN)` tile into $p \times q$ smaller 2D sub-tiles `(subM, subN)` (or split along $K$) across all $N_{\text{AIC}}$ cores.
   - Distinguish the **global last wave** from a core's **local last iteration**: derive `(parent_id, sub_m_id, sub_n_id)` from the global wave remainder index, clamp effective `(valid_M, valid_N)` against the true matrix boundary, and skip cores assigned empty out-of-bound sub-tiles.

### Step 4: FixPipe Inline Epilogue & Quantized Scale Coalescing (`P70`, `merged_scale_loading`)
1. **FixPipe Inline Bias, Cast & Quantization (`P70`, `disc_x43`)**:
   - Fuse bias addition (`SetBias`), L0C `FP32 -> FP16/BF16` rounding, or `INT32 -> FP16/INT8` dequantization (`deq_scale`) directly inside the `FIXPIPE` hardware transfer (`L0C -> GM` or `L0C -> UB` on 950) so `L0C` accumulators never spill to GM workspace as raw FP32/INT32.
2. **Merged Scale Loading & Bank-Conflict Padding (`merged_scale_loading`, `dequant_bank_padding_and_l1_copy_stride`)**:
   - Coalesce per-N/per-group quantization scales across multiple N-tiles or K-groups into a single 32B/512B-aligned MTE2 burst outside the inner K loop.
   - When staging dequantized tiles in UB/L1 with power-of-two row strides (e.g., `256B` or `512B` multiples that alias onto the same UB/L1 bank group), pad the row stride by one 32B block (`+32B`) and set `dstStride = 1` in `DataCopy` / `LoadData` to eliminate bank conflicts.

## Verification & Checks

1. **Credit Balance Audit**:
   - Every `SetFlag` on `MTE2_MTE1`, `MTE1_MTE2`, `MTE1_M`, and `M_MTE1` across prologue, steady-state K loop, and epilogue must have an exact 1:1 matching `WaitFlag` on every reachable control-flow path (including $K_{\text{iters}} = 1$).
2. **Arch35 FP32/B32 `ifTranspose=true` Guard**:
   - On Ascend 950 (`ARCH 3510`), verify that 32-bit (`fp32`/`b32`) `LoadData` from L1 to L0B does not rely on `ifTranspose=true` where hardware constraints forbid it; pre-transpose in UB/L1 or use ND-to-NZ MTE2 format conversion (`P24`).

## Failure Modes & Fallbacks

- **Symptom**: Kernel hangs on the last tile when `K` is small (`num_k_tiles == 1`).
  - **Root Cause**: Unconditional 2-stage prefetch prologue issues a second `DataCopy` and `WaitFlag` for tile `k=1` that does not exist.
  - **Fix**: Guard prefetch launch with `if (k + 1 < num_k_tiles)` and bound prologue depth by `min(2, num_k_tiles)`.
- **Symptom**: `MatmulImpl` on 910B (`c220`) corrupts output when draining multi-fragment `INT32` accumulator in `NZ` format (`disc_x11`).
  - **Root Cause**: Hardware defect on `c220` multi-fragment `NZ` L0C drain.
  - **Fix**: Route `MatmulImpl` Cube output format to `CubeFormat::ND` for `INT32` multi-row drain.
