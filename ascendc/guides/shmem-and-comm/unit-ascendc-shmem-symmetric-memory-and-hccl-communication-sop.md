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
  - ShmemSymmetricMemory
  - BlazeMatmulComm
  - CocComputationOverCommunication
  - DeviceHcclAllReduce
  - ReduceScatter
  - AllGather
domain_path: ascendc/guides/shmem-and-comm
evidence_strength: reproduced_empirical
id: unit-ascendc-shmem-symmetric-memory-and-hccl-communication-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend910b-910c-and-310p-memory-model-and-target-axioms
- unit-ascendc-sync-audit-and-pipe-hazard-elimination-sop
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/guides/shmem, guides/blaze,
    guides/synchronization, and apis/simd_api/adv_api/hccl_communication documentation
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/shmem/hardware_topology_and_measured_capacity_boundaries.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/shmem/blaze_sample_ring_credit_and_flag_budget.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/shmem/blaze_sample_scale_put_completion_chain.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/shmem/coc_matmul_tiles_accumulators_and_workspace_handoffs.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/synchronization/apace_reduce_scatter_schedule_and_credit_audit.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/adv_api/hccl_communication/hccl_kernel/hccl_usage.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/shmem/hardware_topology_and_measured_capacity_boundaries.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-moe-dispatch-gmm-combine-and-alltoallv-routing-sop
- unit-ascendc-host-tilingkey-workspace-and-ub-partitioning-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Ascend C device-initiated SHMEM symmetric
  memory kernels (Blaze/SWAT/COC compute-communication overlap) and device-side HCCL
  collective communication (OpMC2Def, AllReduce, ReduceScatter, AllGather).
tier: L2
title: Ascend C SHMEM Symmetric Memory, Blaze/COC Matmul-Comm Fusion & HCCL Kernel
  API SOP
triggers:
  error_signatures:
  - SHMEM symmetric memory offsets and capacity bounds must be identical across all
    participating ranks; put/signal completion requires explicit scale+payload ordering
    and ring credit tracking
  - Device-side HCCL (Hccl<HCCL_SERVER_TYPE_AICPU/AICORE>) requires OpMC2Def registration
    on Host and paired Prepare / Commit / Wait tile handshakes on Device
  - Blaze / COC Split-K and ReduceScatter pipelines must reset cross-rank sentinel
    flags before reuse and account for rank accumulation order in FP16/BF16 tolerance
    checks
  keywords:
  - shmem_symmetric_memory
  - blaze_swat_scheduler
  - coc_matmul_comm
  - hccl_kernel_api
  - op_mc2_def
  - reduce_scatter
  - all_gather
  - ring_credit_flag_budget
  - scale_put_completion
---
## Preconditions & Signals

Apply this SOP when building or debugging multi-NPU compute-communication overlap (`MC2` / `COC` / `Blaze`) using either **SHMEM symmetric memory** (direct AIC/AIV peer-to-peer RMA puts/gets over HCCS/RoCE) or **Ascend C Device HCCL APIs** (`Hccl<...>`):

1. **SHMEM Symmetric Heap vs Device HCCL Selection**:
   - **SHMEM Symmetric Memory (`guides/shmem/*`, `guides/blaze/*`)**: Used for ultra-low-latency tile-grained producer-consumer pipelines (e.g., MXFP8/BF16 GEMM + AllGather / ReduceScatter / AllToAll in Blaze & SWAT schedulers) where AIC/AIV cores write tiles and completion flags directly into peer ranks' symmetric GM windows.
   - **Device HCCL (`Hccl` + `OpMC2Def`)**: Used when orchestrating standard collective primitives (`AllReduce`, `ReduceScatter`, `AllGather`, `AlltoAllV`) asynchronously from AICore via `HcclContext` and AICPU/HCCS engines.
2. **Hardware Topology & Symmetric Capacity Boundaries (`hardware_topology_and_measured_capacity_boundaries.md`)**:
   - Symmetric heap allocations must fit within the configured SHMEM heap quota and maintain identical virtual offset layouts across all ranks in the team (`bootstrap_teams_and_public_interface_layers.md`).

## Procedure

### Step 1: SHMEM Bootstrap, Symmetric Offset Layout & Ring Credit Management
1. **Host Bootstrap & Team Lifecycle (`blaze_host_bootstrap_lifetime_and_failure.md`)**:
   - Initialize the SHMEM runtime and rank team once per process group and reuse the context across invocations (`apace_multirank_startup_and_context_reuse_scope.md`); never re-bootstrap SHMEM inside the per-step operator launch loop.
2. **Symmetric Offset & Ring-Buffer Credit Protocol (`blaze_sample_symmetric_offsets_and_capacity.md`, `blaze_sample_ring_credit_and_flag_budget.md`)**:
   - Partition the symmetric window into fixed-size tile slots plus a dedicated 32B/64B-aligned flag/sentinel array.
   - Maintain a strict two-way credit loop per ring channel:
     - **Producer Rank**: Wait until remote slot credit is `EMPTY`, issue remote `put` for payload (and scale if MX-quantized), issue a memory fence / quiet so payload lands before the flag, and set remote slot flag to `READY(step_seq)`.
     - **Consumer Rank**: Poll/wait until local slot flag equals `READY(step_seq)`, invalidate data cache (`dcci`) before reading symmetric GM into UB/L1, consume the tile, and clear/acknowledge the slot credit back to `EMPTY`.

### Step 2: Quantized Scale + Payload Put Completion Chain (`blaze_sample_scale_put_completion_chain.md`)
1. **Two-Tensor Atomic Publication (MXFP8 / INT8 Tiles + Scales)**:
   - When a tile consists of both quantized payload (`MXFP8`/`INT8`) and group scales (`E8M0`/`FP32`), never signal `READY` to the peer rank after only the payload `put` completes!
   - Either pack payload and padded scale into a single contiguous symmetric transfer window (`blaze_sample_mx_rank_scale_padding.md`) or chain the completion events (`payload_put_done && scale_put_done -> signal_ready_flag`) so the remote consumer never dequantizes new payload with stale scales.

### Step 3: COC / Blaze Matmul-Communication Tile Overlap (`coc_matmul_tiles_accumulators_and_workspace_handoffs`)
1. **`AllGather + Matmul` (Prologue Overlap)**:
   - Launch tile 0 computation on the local rank's shard immediately while SHMEM/HCCL prefetches remote shard 1 into symmetric workspace; overlap `shard k` GEMM on AIC with `shard k+1` remote transfer.
2. **`Matmul + ReduceScatter` / `AllReduce` (Epilogue Overlap)**:
   - Schedule Matmul output tiles in rank-destination order (`SWAT` / rank-chunk order) so as soon as AIC drains all tiles belonging to destination rank $r$, AIV/SHMEM immediately transmits or reduces chunk $r$ while AIC computes chunk $r+1$.
   - Apply `bias` on **exactly one designated rank** (or after `ReduceScatter` completion, `blaze_sample_rank_accumulation_and_bias.md`)—never add full `bias` on every rank before summing across $P$ ranks!

### Step 4: Ascend C Device HCCL API (`Hccl` & `OpMC2Def`)
1. **Host `OpMC2Def` Registration (`op_mc2_def_intro.md`)**:
   - Register communication groups in `OpDef` via `this->MC2().HcclGroup("group")` and configure server type (`HCCL_SERVER_TYPE_AICPU` or `HCCL_SERVER_TYPE_AICORE`).
2. **Device `Hccl` Tile Handshake (`hccl_usage.md`)**:
   - Instantiate `Hccl<HCCL_SERVER_TYPE_AICPU>` on Device, call `hccl.Init(contextGM, tilingGM)`, issue `HcclHandle handle = hccl.AllReduce<true>(sendBuf, recvBuf, count, dtype, op, repeat)` (with `prepare` flag), `hccl.Commit(handle)` when the tile is ready in GM, and `hccl.Wait(handle)` before reading `recvBuf` into UB/L1.

## Verification & Checks

1. **Data Cache Coherency (`dcci`) on Symmetric GM Reads**:
   - Whenever an AIC/AIV core polls a symmetric GM flag or reads a peer-written symmetric GM buffer, call `DataCacheCleanAndInvalid` (`dcci`) on the cache line so the core does not read stale L2/D-cache lines.
2. **Cross-Rank Floating-Point Accumulation Order (`shmem_comparison_formulas_and_reference_limits.md`)**:
   - Ring `ReduceScatter` accumulates ranks in ring order `(r + 1, r + 2, ..., r + P)`, which differs per rank unless canonicalized; verify accuracy against a reference that matches the ring reduction tree or use FP32 intermediate accumulation.

## Failure Modes & Fallbacks

- **Symptom**: Hang on the second invocation of a SHMEM / Blaze kernel in the same process.
  - **Root Cause**: Sentinel/completion flags in symmetric memory were left at `1` from the first launch and matched the second launch's wait condition prematurely (or failed monotonic sequence check).
  - **Fix**: Use monotonically increasing epoch/sequence counters (`epoch * max_steps + step + 1`) for symmetric flags or explicitly zero-reset consumed flags before kernel exit (`blaze_local_splitk_sentinels_and_atomic_output.md`).
