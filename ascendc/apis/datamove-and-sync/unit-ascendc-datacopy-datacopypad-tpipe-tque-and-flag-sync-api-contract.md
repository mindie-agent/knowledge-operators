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
  - DataMovement
  - QueueManagement
  - Synchronization
  - AtomicOperations
  - CacheControl
domain_path: ascendc/apis/datamove-and-sync
evidence_strength: software_contract
id: unit-ascendc-datacopy-datacopypad-tpipe-tque-and-flag-sync-api-contract
inherits_from:
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  attribution: Distilled from CANNBot-Knowledge ops/ascendc/apis/simd_api/basic_api
    and guides/synchronization software contracts
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/data_move_guide/gm_ub_data_move.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_pipe/t_pipe_intro.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_que/t_que_intro.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/system_sync_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/atomic_operations/atomic_add.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/cache_control/data_cache_clean_and_invalid.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/data_move_guide/gm_ub_data_move.md
  source_type: public_software_contract_review
  supersedes: []
related:
- unit-ascendc-datacopypad-elimination-ub-bank-conflict-and-l2-tiling-sop
- unit-ascendc-507015-illegal-address-sync-hang-and-validation-fault-workflow
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Foundational Ascend C software contract for basic data movement (DataCopy,
  DataCopyPad, ND2NZ), resource management (TPipe, TQue, TQueBind, TBuf), intra/inter-core
  synchronization (SetFlag/WaitFlag, PipeBarrier, SyncAll, CrossCoreSetFlag/WaitFlag),
  atomics, and D-cache flushing.
tier: L1
title: Ascend C DataCopy, DataCopyPad, TPipe, TQue/TBuf & Intra/Cross-Core Sync API
  Contract
triggers:
  error_signatures:
  - Standard DataCopy requires both byte length and UB strides to be multiples of
    32B (1 datablock); DataCopyPad specifies blockLen in bytes while dstStride in
    UB remains in 32B datablocks
  - TQue EnQue/DeQue manages implicit MTE2->V and V->MTE3 synchronization for single-path
    queues; manual SetFlag/WaitFlag<HardEvent> must never consume the same event credit
    twice across loop iterations
  - SetAtomicAdd<T>() must be paired with SetAtomicNone() immediately after the atomic
    MTE3 DataCopy completes
  keywords:
  - datacopy
  - datacopypad
  - tpipe
  - tque
  - tque_bind
  - tbuf
  - set_flag
  - wait_flag
  - pipe_barrier
  - cross_core_set_flag
  - set_atomic_add
  - data_cache_clean_and_invalid
---
## Core Axioms & Invariants

### 1. `DataCopy` vs `DataCopyPad` Parameter Units (`gm_ub_data_move.md`)

| API | Struct Type | `blockCount` | `blockLen` Unit | `srcStride` Unit | `dstStride` Unit | Padding / Alignment Rule |
|---|---|---|---|---|---|---|
| `DataCopy(dst, src, DataCopyParams)` | `DataCopyParams` | Number of blocks (`uint16_t`) | **32B datablocks** (`1` = `32 bytes`) | **32B datablocks** | **32B datablocks** | Total transfer per block must be a multiple of `32B` (`512B` for L1 fractals) |
| `DataCopyPad(ubDst, gmSrc, DataCopyExtParams, DataCopyPadExtParams<T>)` | `DataCopyExtParams` | Number of blocks (`uint16_t`) | **Bytes** (`uint32_t`, arbitrary byte length) | **Bytes** (in GM) | **32B datablocks** (in UB) | Each block of `blockLen` bytes is automatically rounded up to `32B` in UB; optional `leftPadding`/`rightPadding` fills up to `32 / sizeof(T)` elements with `paddingValue` |
| `DataCopyPad(gmDst, ubSrc, DataCopyExtParams)` | `DataCopyExtParams` | Number of blocks (`uint16_t`) | **Bytes** (`uint32_t`, exact bytes written to GM) | **32B datablocks** (in UB) | **Bytes** (in GM) | Reads 32B-aligned UB blocks and writes only `blockLen` valid bytes to GM without overwriting adjacent GM memory |
| `DataCopy(l1Dst, gmSrc, Nd2NzParams)` | `Nd2NzParams` | `ndNum` matrices | `nValue` rows $\times$ `dValue` elements | `srcNdMatrixStride`, `srcDValue` (elements) | `dstNzC0Stride`, `dstNzNStride` ($C_0$-blocks/rows) | Converts `ND` in GM to `FRACTAL_NZ` (`[ceil(D/C0), N, C0]`) in L1 on-the-fly during MTE2 load |

### 2. Resource Management: `TPipe`, `TQue`, `TQueBind` & `TBuf`
1. **Single Active `TPipe` Owner**:
   - Only one `TPipe` instance may manage physical on-chip memory (`UB`, `L1`, `L0A`, `L0B`, `L0C`) at a time (or retrieve the shared pointer via `GetTPipePtr()` when mixing with `MatmulImpl`).
2. **`TQue` Lifecycle (`AllocTensor -> EnQue -> DeQue -> FreeTensor`)**:
   - `TQue<QuePosition::VECIN, BUFFER_NUM>`: `AllocTensor<T>()` acquires a free UB slot (waiting on `V -> MTE2` release); `DataCopy` issues `GM -> UB`; `EnQue` records `MTE2 -> V` ready; `DeQue<T>()` waits on `MTE2 -> V` before Vector reads; `FreeTensor` records `V -> MTE2` release.
   - `TQue<QuePosition::VECOUT, BUFFER_NUM>`: `AllocTensor<T>()` waits on `MTE3 -> V`; Vector writes result; `EnQue` records `V -> MTE3`; `DeQue<T>()` waits on `V -> MTE3`; `DataCopy` issues `UB -> GM`; `FreeTensor` records `MTE3 -> V`.
   - **Invariant**: Every `AllocTensor` must be matched by exactly one `FreeTensor`, and `EnQue` / `DeQue` depth must never exceed `BUFFER_NUM`.
3. **`TQueBind` & `TBuf`**:
   - `TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM>` binds input and output directions to the same physical UB buffer for pure copy/in-place passthrough.
   - `TBuf<TPosition::VECCALC>` allocates a static scratch buffer via `Get<T>()` with **zero automatic synchronization**; the caller must guard overwrites with `PipeBarrier<PIPE_V>()` or `SetFlag`/`WaitFlag`.

### 3. Intra-Core, Cross-Core & Global Synchronization (`system_sync_overview.md`)
1. **Intra-Core Pipeline Events (`SetFlag<HardEvent>` / `WaitFlag<HardEvent>`)**:
   - Valid `HardEvent` directions include `MTE2_V`, `V_MTE2`, `V_MTE3`, `MTE3_V`, `MTE2_MTE1`, `MTE1_MTE2`, `MTE1_M`, `M_MTE1`, `M_FIX`, `FIX_M`, `S_V`, `V_S`, `S_MTE2`, `MTE2_S`, `MTE3_S`, `MTE3_MTE2`.
   - Each `event_t` ID (`EVENT_ID0` .. `EVENT_ID7` via `AllocEventID<HardEvent>()`) is a single binary credit per `(HardEvent, event_id)` channel:
     - **No Double-Set Without Wait**: Issuing two `SetFlag<E>(id)` calls without an intervening `WaitFlag<E>(id)` overflows/wedges the hardware event counter (`debug_markers_must_not_setflag_a_shared_functional_event...`).
     - **No Double-Wait**: Issuing `WaitFlag<E>(id)` twice for a single `SetFlag<E>(id)` immediately deadlocks the core (`intra_core_event_credits_consumed_per_call_site_work_loop_boundary_double_consume_self_deadlocks.md`).
2. **Same-Pipe Ordering (`PipeBarrier<PIPE_*>`)**:
   - `PipeBarrier<PIPE_V>()` enforces sequential completion between back-to-back Vector instructions when UB source/destination buffers alias or overlap in-place (`alias_hazards_and_vector_recurrence.md`).
   - `PipeBarrier<PIPE_ALL>()` drains all pipelines on the current core (use only for debugging/triage; replace with directed `SetFlag`/`WaitFlag` in production).
3. **Cross-CoreAIC/AIV Handshake (`CrossCoreSetFlag` / `CrossCoreWaitFlag` & `SyncAll`)**:
   - `CrossCoreSetFlag<mode, pipe>(flagId)` and `CrossCoreWaitFlag<mode>(flagId)` synchronize AIC and AIV sub-cores within the same MIX group (`mode = 0x2`) or across all cores on the device (`mode = 0x0`).
   - `SyncAll()` performs a global barrier across all active cores; when using software `SyncAll(ubWorkspace, gmSyncWorkspace, usedCoreNum)` with non-full core participation, every participating core `0 .. usedCoreNum - 1` must call `SyncAll` the exact same number of times.

### 4. Atomic Operations & D-Cache Coherency
1. **GM Atomic Write Scope (`SetAtomicAdd<T>()`)**:
   - Call `SetAtomicAdd<float>()` (or `half`, `bfloat16`, `int32_t`, `int16_t`, `int8_t` per hardware support; see `atomic_operations_interface_boundary_value_summary.md`) before `DataCopy`/`DataCopyPad` from `UB -> GM`, and **always** restore `SetAtomicNone()` immediately after the MTE3 transfer is enqueued.
   - Pre-initialize the target GM buffer to `0` before the first atomic add, and insert `PipeBarrier<PIPE_MTE3>()` when changing atomic state between consecutive MTE3 stores if required.
2. **Scalar D-Cache Flush (`DataCacheCleanAndInvalid`)**:
   - When the Scalar unit (`S-pipe`) reads or writes a `GlobalTensor` buffer that is also accessed by `MTE2`/`MTE3` or another core, call `DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gmAddr)` (or `ENTIRE_DATA_CACHE`) to prevent stale Scalar D-cache reads or lost Scalar writes.

## Verification Checklist

- [ ] Every `DataCopy` length and stride is checked in `32B` datablock units (`DataCopyPad` length in bytes, UB stride in `32B` datablocks).
- [ ] Every `AllocTensor` has a matching `FreeTensor` and every `AllocEventID` has a matching `ReleaseEventID` (or balanced `SetFlag`/`WaitFlag` across all loop exits).
- [ ] Every `SetAtomicAdd<T>()` is followed by `SetAtomicNone()` before the kernel exits or writes non-atomic output tensors.
