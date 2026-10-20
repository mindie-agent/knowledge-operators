---
schema: mindie-graded-unit/1
id: unit-ascendc-sync-audit-and-pipe-hazard-elimination-sop
revision: 1
tier: L2
category: sop
domain_path: ascendc/sync
evidence_strength: software_contract
title: 'AscendC synchronization audit SOP: SetFlag/WaitFlag pairing, branch symmetry, double-buffer loop invariants, PipeBarrier scope and MIX_CV cross-core hazards'
conditions:
  kernel_language: ascendc
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
provenance:
  source_type: public_software_contract_review
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-sync-audit/SKILL.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  supporting_source_revisions:
  - be5882ef29fa9943305085652d2025a038ba491c
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-sync-audit/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-sync-audit/references/sync-hazards.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-sync-audit/references/sync-primitives.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/debugging/sync_audit_checklist.md
  supersedes: []
triggers:
  error_signatures:
  - aclrtSynchronizeStream hangs or returns 507015 after adding manual SetFlag/WaitFlag or TBuf double buffering
  - a kernel passes single-tile tests but corrupts iteration 1+ or the final MTE3 write-back when tileCount >= 2
  - a conditional tail branch or early return causes a hang in WaitFlag, DeQue or CrossCoreWaitFlag
  - PipeBarrier<PIPE_V> is used between MTE2 DataCopy and Vector compute instead of HardEvent::MTE2_V
  - SetAtomicAdd or SetAtomicNone is called immediately after MTE3/Fixpipe without PipeBarrier<PIPE_FIX> or PipeBarrier<PIPE_MTE3>
  - a MIX_CV kernel deadlocks or reads stale workspace slots when CrossCoreSetFlag mode, pipe or flagId conflicts with Matmul
  keywords:
  - AscendC
  - sync audit
  - SetFlag
  - WaitFlag
  - HardEvent
  - MTE2_V
  - V_MTE3
  - MTE3_MTE2
  - MTE1_M
  - M_FIX
  - PipeBarrier
  - PIPE_ALL
  - PIPE_FIX
  - CrossCoreSetFlag
  - CrossCoreWaitFlag
  - SetAtomicAdd
  - double buffer
  - SYNC-01
  - SYNC-14
---

# AscendC synchronization audit SOP: SetFlag/WaitFlag pairing, branch symmetry, double-buffer loop invariants, PipeBarrier scope and MIX_CV cross-core hazards

## Use when

A custom Ascend C kernel on Ascend 910B, 910_93 or 950 uses explicit `SetFlag`/`WaitFlag`, manual `TBuf` ping-pong, `PipeBarrier`, atomic GM stores, or `MIX_CV` `CrossCoreSetFlag`/`CrossCoreWaitFlag` and must be audited for deadlock, cross-iteration WAR/RAW races, or pipeline serialization (`SYNC-01`..`SYNC-14`).

## 1. Audit `SetFlag`/`WaitFlag` pairing and `HardEvent` direction (`SYNC-01`, `SYNC-02`, `SYNC-10`)

Every intra-core cross-pipe handoff requires an exact `(HardEvent, eventId)` match between producer and consumer pipes:

| Dataflow path | Required `HardEvent` | Invalid substitute |
| --- | --- | --- |
| GM -> UB (`DataCopy`/`DataCopyPad`) -> Vector compute | `HardEvent::MTE2_V` | `V_MTE2` (reversed), `PipeBarrier<PIPE_V>` (wrong pipe) |
| Vector compute -> UB -> GM (`DataCopy`/`DataCopyPad`) | `HardEvent::V_MTE3` | `MTE3_V` (reversed), `PipeBarrier<PIPE_V>` |
| UB -> GM (`MTE3`) -> next GM -> UB (`MTE2`) reusing same UB slot | `HardEvent::MTE3_MTE2` | Omitting back-pressure (WAR overwrite) |
| GM -> L1 (`MTE2`) -> L0A/B (`MTE1`) -> Cube (`M`) -> Fixpipe (`FIX`) | `MTE2_MTE1` -> `MTE1_M` -> `M_FIX` | `MTE2_M` (skips `MTE1`), missing `FIX_M` on L0C reuse |
| Vector compute -> Vector compute on same UB buffer | `PipeBarrier<PIPE_V>()` | `HardEvent::V_V` (does not exist in hardware) |

Hardware event IDs (`EVENT_ID0`..`EVENT_ID3`, or `0..7` where supported) are 1-bit latches per `(HardEvent, eventId)`:

- **Unpaired `WaitFlag` (`SYNC-01a`)**: blocks forever (`507015` timeout).
- **Unpaired `SetFlag` (`SYNC-01b`)**: leaves the latch set so a later loop consumes a stale token (silent RAW race), or two consecutive `SetFlag` calls on the same `(HardEvent, eventId)` without an intervening `WaitFlag` hang the producer.
- **Event ID exhaustion (`SYNC-10`)**: never exceed hardware ID capacity or reuse the same `(HardEvent, eventId)` across two live overlapping slots without waiting first. Prefer `TPipe::AllocEventID<HardEvent::...>()` / `ReleaseEventID` when mixing `TQue` and manual `TBuf`.

## 2. Audit divergent branches and early exits (`SYNC-03`)

When a conditional branch (`if (isTail)`, `if (validLen == 0)`, `continue`, `return`) executes on only one side of a producer/consumer pair, flag counts diverge:

```cpp
// DEFECT (SYNC-03): early continue skips WaitFlag/FreeTensor after SetFlag/AllocTensor
SetFlag<HardEvent::MTE2_V>(evtId);
if (validLen == 0) {
    continue; // leaves MTE2_V latched; next iteration double-sets evtId and deadlocks
}
WaitFlag<HardEvent::MTE2_V>(evtId);
```

**Invariant.** Every control-flow path between `SetFlag` and `WaitFlag` (or `AllocTensor` and `FreeTensor`) must guard both producer and consumer with the identical predicate or consume the token before branching. In multi-core kernels calling `SyncAll()`, idle cores (`blockIdx >= activeCores`) must still reach `SyncAll()` or be excluded from `blockDim`.

## 3. Audit loop-carried double buffering and ring indices (`SYNC-04`, `SYNC-05`, `SYNC-06`, `SYNC-14`)

Manual `TBuf` N-buffer loops (`BUFFER_NUM = 2`) fail in four characteristic ways:

1. **Index underflow at `loop == 0` (`SYNC-04`)**: `(loop - 1) % BUFFER_NUM` evaluates to `-1` in signed integers on iteration 0. Always index slots as `uint32_t ping = loop % BUFFER_NUM` (`EVENT_ID0` / `EVENT_ID1`).
2. **Pre-loop prime vs post-loop drain (`SYNC-05`)**: prime `MTE3_MTE2` before the loop so the first two iterations can issue `WaitFlag` unconditionally, then drain both slots after the loop:

```cpp
for (int32_t i = 0; i < 2; ++i) {
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i)); // prime slots
}
for (int32_t step = 0; step < numSteps; ++step) {
    event_t slot = static_cast<event_t>(step & 1);
    WaitFlag<HardEvent::MTE3_MTE2>(slot);          // wait until prior MTE3 store finished
    DataCopy(ubIn[slot], gmIn[step * tileLen], tileLen);
    SetFlag<HardEvent::MTE2_V>(slot);
    WaitFlag<HardEvent::MTE2_V>(slot);
    Compute(ubOut[slot], ubIn[slot], tileLen);
    SetFlag<HardEvent::V_MTE3>(slot);
    WaitFlag<HardEvent::V_MTE3>(slot);
    DataCopy(gmOut[step * tileLen], ubOut[slot], tileLen);
    SetFlag<HardEvent::MTE3_MTE2>(slot);           // release slot for step + 2
}
for (int32_t i = 0; i < 2; ++i) {
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i)); // drain final MTE3 stores
}
```

3. **Missing post-loop `MTE3` drain (`SYNC-06`)**: reusing UB scratch in a subsequent phase before the final `MTE3` write completes corrupts GM unless guarded by `MTE3_MTE2` drain or `PipeBarrier<PIPE_MTE3>()`.
4. **Ring-index skew (`SYNC-14`)**: the buffer index `ubBuf[slot]` and synchronization token `slot` must use the identical modulo expression (`step % N`). Using `step % 2` for buffers and fixed `EVENT_ID0` for flags serializes the pipeline or races on slot 1.

## 4. Audit `PipeBarrier` scope and atomic-state switches (`SYNC-07`, `SYNC-08`, `SYNC-13`)

- **Cross-pipe `PipeBarrier<PIPE_V>` misuse (`SYNC-07`)**: `PipeBarrier<PIPE_V>()` only orders Vector ALU ops against earlier Vector ALU ops. Placing it between `DataCopy(ub, gm)` (`PIPE_MTE2`) and `Add(ub, ...)` (`PIPE_V`) provides **zero** synchronization; `MTE2` and `V` require `HardEvent::MTE2_V`.
- **`PipeBarrier<PIPE_ALL>()` inside inner loops (`SYNC-08`)**: drains all pipes (`SCALAR`, `V`, `M`, `MTE1..3`, `FIX`), collapsing double buffering into serial execution. Restrict `PIPE_ALL` to kernel teardown or coarse phase boundaries.
- **Atomic state race (`SYNC-13`)**: `SetAtomicAdd<T>()` and `SetAtomicNone()` write a scalar config register read by `PIPE_MTE3` and `PIPE_FIX`. Because scalar ops execute ahead of async DMA/Fixpipe queues, calling `SetAtomicNone()` immediately after `DataCopy(gm, ub)` or `copy_matrix_cc_to_gm` disables atomic accumulation while the write is still in flight:

```cpp
SetAtomicAdd<float>();
PipeBarrier<PIPE_MTE3>();                 // ensure state is active before MTE3 issues
DataCopy(gmOut[offset], ubOut, tileLen);
PipeBarrier<PIPE_MTE3>();                 // mandatory: wait for MTE3 write to finish
SetAtomicNone();                          // safe to restore state
```

For Cube L0C->GM atomic writes, use `PipeBarrier<PIPE_FIX>()` before `SetAtomicNone()`.

## 5. Audit `MIX_CV` cross-core synchronization (`SYNC-11`, `SYNC-12`)

In `MIX` kernels (`KERNEL_TYPE_MIX_AIC_1_2`), AIC and AIV synchronize via `CrossCoreSetFlag<mode, pipe>(flagId)` and `CrossCoreWaitFlag<mode>(flagId)`:

- **`flagId` range and collision (`SYNC-12`)**: `flagId` must be in `[0, 15]` (4-bit counter, max 15). High-level `AscendC::Matmul` claims internal flags; never share `flagId` values with an active `Matmul` object.
- **Mode and 1:2 AIC:AIV ratio (`SYNC-11`)**:
  - `mode = 0`: global across all cores.
  - `mode = 2`: within the same Cube-Vector group (1 AIC + 2 AIVs on 910B/910_93). When AIVs signal AIC in `mode = 2`, **both** AIV sub-blocks (`GetSubBlockIdx() == 0` and `1`) must call `CrossCoreSetFlag<0x2, PIPE_MTE3>(flagId)` to unblock one `CrossCoreWaitFlag<0x2>(flagId)` on AIC.
  - `mode = 4` (Ascend 950 directed AIC<->single-AIV): pairs AIC with a specific `aiv_id` (`0` or `1`) for split L0C->UB Fixpipe transfers.
- **Producer `pipe`**: must match the pipe publishing shared data (`PIPE_FIX` for AIC Fixpipe writes; `PIPE_MTE3` for AIV MTE3 writes).

## Verification checklist

```bash
grep -nE 'SetFlag|WaitFlag|PipeBarrier|CrossCoreSetFlag|CrossCoreWaitFlag|SetAtomic' op_kernel/*.cpp op_kernel/*.h
```

1. Every `(HardEvent, eventId)` has `count(SetFlag) == count(WaitFlag)` on every path, including pre-loop primes and post-loop drains.
2. Zero `PipeBarrier<PIPE_V>` calls appear as the sole barrier between `DataCopy` and Vector compute.
3. Every `SetAtomicNone()` is preceded by `PipeBarrier<PIPE_MTE3>()` or `PipeBarrier<PIPE_FIX>()`.
4. Every `MIX_CV` `CrossCoreWaitFlag<0x2>(id)` is matched by `CrossCoreSetFlag<0x2, pipe>(id)` on **both** AIV sub-blocks with `id <= 15`.
