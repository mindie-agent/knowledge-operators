---
schema: mindie-graded-unit/1
id: unit-ascendc-mc2-and-catlass-cube-vector-pipeline-workflow
revision: 1
tier: L3
category: workflow
domain_path: fusion/mc2-catlass
evidence_strength: software_contract
title: 'AscendC MC2 collective-compute overlap and CATLASS Cube-Vector rotating-workspace optimization workflow'
conditions:
  kernel_language:
  - ascendc
  - catlass
  op_family:
  - mc2
  - matmul
  - grouped_matmul
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-catlass-and-tilelang-ascend-operator-design-and-tuning-sop
- unit-ascend-grouped-matmul-swiglu-and-mc2-fusion-sop
provenance:
  source_type: public_software_contract_review
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-mc2-best-practice/SKILL.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  supporting_source_revisions:
  - be5882ef29fa9943305085652d2025a038ba491c
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-mc2-best-practice/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-mc2-best-practice/references/overview.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-mc2-best-practice/references/profiling_mc2.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-mc2-best-practice/references/a2_hccl_client.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-mc2-best-practice/references/a5_shmem.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/catlass-op-perf-tune/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/catlass-op-perf-tune/references/knowledge-base.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/optimization/mc2_best_practice.md
  supersedes: []
triggers:
  error_signatures:
  - an MC2 MatmulAllReduce or AllGatherMatmul kernel fails accuracy or hangs when tileCnt > 1 is enabled before verifying tileCnt = 1
  - MC2 benchmark latency looks fast only because warm L2 cache was not flushed between multi-rank profiling iterations
  - a CATLASS MatmulActivation or MatmulEpilogue kernel writes the entire M by N matrix to GM before the AIV epilogue starts
  - a CATLASS SwiGLU or GeGLU epilogue with a rotating workspace produces wrong values when N > L1_N
  - raising UB_STAGES in a dual-input CATLASS epilogue hangs at runtime because HardEvent V_MTE2 exceeds 8 event IDs
  keywords:
  - MC2
  - MatmulAllReduce
  - MatmulReduceScatter
  - AllGatherMatmul
  - MatmulAlltoAll
  - Hccl
  - HCCL_SERVER_TYPE_AICPU
  - blaze-shmem
  - apace
  - URMA
  - tileCnt
  - MmadAtlasA2PreloadAsync
  - GemmIdentityBlockSwizzle
  - UB_STAGES
  - rotating workspace
---

# AscendC MC2 collective-compute overlap and CATLASS Cube-Vector rotating-workspace optimization workflow

## Trigger and scope

Use this four-stage workflow when developing or tuning (1) a custom Ascend C `MC2` operator (`MatmulAllReduce`, `MatmulReduceScatter`, `AllGatherMatmul`, `MatmulAlltoAll`, or MoE `DistributeDispatch`/`DistributeCombine`) that overlaps Cube Matmul with multi-rank communication, or (2) a CATLASS fused Cube-Vector GEMM whose epilogue (`SwiGLU`, `GeGLU`, bias, dequant, residual) is serialized by a full-matrix GM round-trip or suboptimal tiling.

The four stages are strictly ordered: **route and architecture classification -> serial accuracy baseline (`tileCnt = 1` / golden gate) -> single-variable pipeline and tile tuning -> L2-flushed multi-rank / `msprof` verification**.

## Stage 1 — Classify the hardware route and pipeline bottleneck

Select the exact hardware route from the target SOC and operator pattern before writing code:

| Pattern & SOC | Execution route | Core synchronization & memory contract |
| --- | --- | --- |
| **MC2 Route A**: Dense Matmul + Collective on Ascend 910B / 910_93 | `Hccl` device client (`HCCL_SERVER_TYPE_AICPU`) + `AscendC::Matmul<..., CFG_MDL>` | Host `Mc2Init`/`Mc2Tiling` sets `"AICPU"`; AIC/AIV calls `hccl.AllReduce`/`ReduceScatter`/`AllGather`/`AlltoAll` -> `Commit(handleId)` -> `Wait(handleId)` -> `Finalize()` (`tileCnt <= 16`). |
| **MC2 Route B**: Dense Matmul + One-Sided Put on Ascend 950 | `blaze-shmem` + `apace` (`ROCE_SYMMETRIC` URMA engine, `COMM_BUFF_SIZE >= 64 MB`) | Call `SyncAll<false>()` before communication; AIC finishes tile via `WaitIterateBatch()`, AIV issues `put_nbi_warp` + `signal_op_warp(SIGNAL_ADD)` and peers wait via `WaitSignal(peer, tileIdx + 1)`. |
| **MC2 Route C**: MoE Token Dispatch / Combine across ranks | Direct AIV MTE window (`GetWindowsInAddr(rankId)` / `GetWindowsOutAddr(rankId)`) | AIV copies tokens directly between local UB and remote rank HBM IPC windows via `DataCopyPad`, synchronized with `CrossCoreSetFlag<0x0, PIPE_MTE3>(8)` / `CrossCoreWaitFlag<0x0>(8)`. |
| **CATLASS CV Fusion**: GEMM + Vector epilogue on 910B / 910_93 / 950 | `MmadAtlasA2PreloadAsync` + rotating L2 workspace | Replace `MatmulActivation`/`MatmulEpilogue` full `[M, N]` HBM write-back with per-tile `CrossCoreSetFlag<0x2, PIPE_FIX>(flag)` callback handoff into an L2-resident ring buffer. |

## Stage 2 — Lock the serial accuracy baseline (`tileCnt = 1` / Golden Gate)

Never debug multi-tile overlap and numerical correctness simultaneously:

1. **MC2 serial gate**: hardcode `tileCnt = 1` (single main tile covering `M`, `tailM = 0`) so compute and collective run serially (`Matmul -> Commit -> Wait -> Finalize` for compute-first; `Prepare -> Commit -> Wait -> Matmul -> Finalize` for comm-first). Verify multi-rank accuracy against unfused `torch.matmul` + `torch.distributed` collectives (`rtol=1e-2, atol=1e-2` for fp16/bf16).
2. **CATLASS baseline gate**: run `python3 scripts/run_golden.py` across all target `(M, N, K)` shapes and record baseline `msprof op` kernel duration before modifying `DispatchPolicy` or tile shapes.

## Stage 3 — Tune pipeline overlap and tile geometry one variable at a time

### 3A. MC2 `tileCnt` splitting and pipeline scheduling

- **Axis split rule**: split the M-axis (`B, S`) into `tileCnt` chunks of `tileM` plus `tailM` (`tailNum`) for `MatmulAllReduce`, `MatmulReduceScatter`, and `AllGatherMatmul` (never split K when fusing AllReduce/ReduceScatter, which would multiply collective volume). For `MatmulAlltoAll`, split along the N-axis (or 2D M/N tiles).
- **Hard invariants**:
  - `tileCnt <= 16` (AICPU queue capacity ceiling).
  - Every per-tile GM byte offset (`tileM * N * sizeof(T)` or `tileM * K * sizeof(T)`) must be **512 B aligned** for Cube L1 DMA and **32 B aligned** for UB DMA.
  - Set `context->SetBlockDim(aivNum)` on host and enforce `tiling.usedCoreNum <= aivNum` so no core reads out-of-bounds tiling descriptors.
  - **Compute-first (`MatmulAllReduce` / `MatmulReduceScatter`)**: in loop `i = 0..tileCnt-1`, compute Matmul tile `i` into `cGm + i * tileOffset`, then immediately call `hccl.Commit(handleId[i])`; defer `hccl.Wait(handleId[i])` until after all tiles (and `tailM`) are committed, then call `hccl.Finalize()`.
  - **Comm-first (`AllGatherMatmul`)**: commit all `tileCnt` `AllGather` handles during `Init`, compute the local rank's own shard (`rankId * rankM`) immediately without waiting, then in loop `i` call `hccl.Wait(handleId[i])` immediately before computing shard `i`.
- Sweep `tileCnt` over `{2, 4, 8, 16}` one value at a time; pick the setting that minimizes multi-rank makespan while passing accuracy.

### 3B. CATLASS five-priority optimization ladder

Apply in strict priority order, testing one change per iteration:

1. **Priority 1 — Async callback + rotating L2 workspace**: replace `MmadAtlasA2` + full-GM epilogue with `MmadAtlasA2PreloadAsync` (`DispatchPolicyMmadAtlasA2<2, true>`). Size the rotating workspace to fit L2:
   $$\text{workspace\_bytes} = L1_M \times L1_N \times \text{aic\_cores} \times \text{WORKSPACE\_STAGES} \times \text{sizeof}(C)$$
   with `WORKSPACE_STAGES = 2..4`. Inside the callback, AIC fires `CrossCoreSetFlag<0x2, PIPE_FIX>(flag)`; both AIV sub-blocks wait via `CrossCoreWaitFlag<0x2>(flag)` and each processes `L1_M / 2` rows.
   - **Cross-N-half gated epilogue rule (`SwiGLU`, `GeGLU`, `ReGLU`)**: because output column `j` combines `C[:, j]` and `C[:, j + N/2]`, a single-tile rotating slot corrupts outputs whenever `N > L1_N`. Schedule loops over the **output** shape `[M, N/2]`, compute both `block_coord` (for `C[:, :N/2]`) and `block_coord_b` (for `C[:, N/2:]`) into paired rotating slots (`2 * WORKSPACE_STAGES`), and signal AIV only after both halves are written.
2. **Priority 2 — `L1TileShape` and `L0TileShape` byte and wave fitting**:
   - Enforce `L0_M == L1_M`, `L0_N == L1_N`, `L0_K <= L1_K`, `L0C = L0_M * L0_N * 4B <= 128 KB` (`128x256` max on 910B/910_93), `L0A = L0_M * L0_K * 2B <= 64 KB`, `L0B = L0_K * L0_N * 2B <= 64 KB`, and `L1 <= 512 KB` with 2-stage ping-pong.
   - When `M <= 512` and `(M / 128) * (N / 256) < aic_cores`, shrink `L1_M` from `128` to `64` (`L1<64, 256, 256>`, `L0<64, 256, 64>`) to double the M-block count and fill all AIC cores.
3. **Priority 3 — `BlockScheduler` swizzle**: test `GemmIdentityBlockSwizzle<3, 0>` (when `M >= N`) vs `<3, 1>` (when `M < N`).
4. **Priority 4 — `UBTileShape` capacity fill**: scale AIV `UBTileShape` to utilize the 192 KB UB (`16384` elements for single-input unary epilogues; `8192` for dual-input/SwiGLU epilogues).
5. **Priority 5 — `UB_STAGES` event-ID guard**: `HardEvent::V_MTE2` has only **8 hardware event IDs** (`0..7`). A dual-input epilogue uses `2 * UB_STAGES` event IDs per pipeline, so `UB_STAGES >= 4` (or `>= 2` when combined with extra scratch queues) exhausts hardware event IDs and hangs at runtime. Keep `UB_STAGES = 1` for dual-input epilogues unless event ID count is verified `<= 8`.

## Stage 4 — Verify with L2-flushed multi-rank profiling and rollback guards

1. **MC2 L2-flush multi-rank benchmark**: repeating small/medium shapes in a loop leaves weights in the ~192-200 MB L2 cache, masking `MTE2` contention with RDMA/SIO traffic. Between benchmark iterations inside the `torch_npu.profiler` / `msprof` loop, invoke a dummy vector kernel (`heavy_add_kernel` over ~1 GB buffer) outside the measured op to flush L2. Parse `op_summary_*.csv` across all ranks, compute the mean of the **last 5** target `Aicore Kernel` durations on each rank, and take $\max_{\text{rank}}(\text{mean\_last\_5})$ as the operator latency.
2. **Single-variable rollback**: revert any step that fails accuracy, hangs under a 180 s timeout, or fails to reduce `aicore_time(us)` / multi-rank makespan.
