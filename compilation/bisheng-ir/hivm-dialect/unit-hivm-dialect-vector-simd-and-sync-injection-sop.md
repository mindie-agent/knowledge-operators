---
category: sop
conditions:
  compiler: bishengir
  kernel_language:
  - triton
  - bishengir
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/bisheng-ir/hivm-dialect
evidence_strength: software_contract
id: unit-hivm-dialect-vector-simd-and-sync-injection-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-bisheng-ir-compilation-pipeline-and-pass-lowering-contract
- unit-hivm-dialect-dma-macro-matmul-and-fixpipe-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/vector_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/vector_scalar_lowering.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/sync_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/sync_pipe_sync.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/sync_block_sync.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/sync_sync_injection.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/apis/ir/hivm/vector_overview.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'HIVM Vector SIMD and synchronization SOP: PIPE_V ops, OTF broadcast/transpose,
  ImplByScalarOpInterface avoidance and InjectSync/FFTS rules'
triggers:
  error_signatures:
  - HIVMLowerToLoopsPass lowers hivm.hir.vcmp, vadd, vreduce or vcumsum into 10x-100x
    slower scf.for scalar loops
  - OTF transpose fails because the innermost dimension is included in the permutation
  - Event ID exhaustion (more than 8 concurrent events per pipepair) or unmatched
    set_flag/wait_flag across branches
  - cross-core FFTS sync hangs because set_ffts_base_addr is missing or SyncBlockMode
    mismatches core domain
  keywords:
  - hivm.hir.vadd
  - hivm.hir.vcmp
  - hivm.hir.vsel
  - hivm.hir.vreduce
  - hivm.hir.vbrc
  - hivm.hir.vtranspose
  - ImplByScalarOpInterface
  - shouldLowerToScalarLoops
  - hivm.hir.set_flag
  - hivm.hir.wait_flag
  - hivm.hir.pipe_barrier
  - hivm.hir.sync_block
  - GraphSyncSolver
  - FFTS
---
# HIVM Vector SIMD and synchronization SOP: PIPE_V ops, OTF broadcast/transpose, ImplByScalarOpInterface avoidance and InjectSync/FFTS rules

## 1. HIVM Vector SIMD Operations and On-the-Fly (OTF) Broadcast/Transpose Rules

All `HIVM_VectorOp` operations execute on `PIPE_V` in UB (`#hivm.address_space<ub>`) with `DestinationStyleOpInterface` syntax:

```mlir
hivm.hir.<op> ins(%src0, %src1 : ...) outs(%dst : ...)
  [temp_buffer(%tmp : memref<..., #hivm.address_space<ub>>)]
  [broadcast = array<i64: ...>] [transpose = array<i64: ...>]
```

- **Operation Catalog (`HIVMVectorOps.td`)**:
  - **Unary**: `vexp`, `vabs`, `vln`, `vrelu`, `vrsqrt`, `vsqrt`, `vtanh`, `vsin`, `vcos`, `verf`, `vrec`, `vnot`.
  - **Binary**: `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vor`, `vand`, `vxor`, `vmod`, `vmodui`, `vpow`, `vmulext`.
  - **Ternary & Selection**: `vsel` (vector select; `i1` condition or mask tensor).
  - **Cast & Compare & Shift**: `vcast` (with rounding/overflow modes), `vcmp` (`EQ`, `NE`, `LT`, `LE`, `GT`, `GE`), `vshl`, `vshr`.
  - **Reduction & Scan/Sort**: `vreduce` (`sum`, `prod`, `max`, `min`, `any`, `all`, `xori`, `ori`, `andi`, `max_with_index`, `min_with_index`), `vcumsum`, `vcumprod`, `vsort`.
  - **Data Movement & Special**: `vbrc` (broadcast), `vtranspose`, `vinterleave`, `vdeinterleave`, `vflip`, `vpad`, `vconcat`, `vgather`, `varange`, `vmulextended`.
- **On-the-Fly (`BroadcastableOTF` / `TransposableOTF`) Fusion Contract**:
  - `InlineOTFBroadcast` folds standalone `hir.vbrc` into the consumer's `broadcast = [...]` attribute only when the broadcast dimension $d$ satisfies $0 \le d < \text{rank}(\text{dst})$ and $\text{shape}(\text{src})[d] \in \{1, \text{shape}(\text{dst})[d]\}$.
  - `TransposableOTF` folds transpose into the consumer only when `transpose` is a valid permutation of `[0 .. rank-1]` **and the last (innermost) dimension is unchanged** (`transpose[-1] == rank - 1`). Innermost-axis transposes require a standalone `hir.vtranspose` with UB `$temp_buffer` and 32 B aligned axes.

## 2. Eliminate the Five `ImplByScalarOpInterface` Scalar Lowering Traps

When `ImplByScalarOpInterface::shouldLowerToScalarLoops()` returns `true`, `HIVMLowerToLoopsPass` replaces the `PIPE_V` SIMD instruction with nested `scf.for` + `arith.*` scalar instructions on `PIPE_S`, causing a **10x–100x throughput collapse**:

| Trap Group | Affected HIVM Ops | Exact Scalar Lowering Trigger (`ShouldLowerToScalarLoops.cpp`) | Mandatory Prevention / Rewrite |
| --- | --- | --- | --- |
| **1. `i64` Elementwise** | `vadd`, `vsub`, `vmul`, `vmin`, `vmax`, `vabs`, `vshl`, `vshr`, `vinterleave`, `vdeinterleave` | Element type is `i64`, or execution mode is `isSIMTVF()` | Keep vector compute in `i32`/`f32`/`f16`/`bf16`; widen to `i64` only for scalar base pointers |
| **2. Integer Inequality `vcmp`** | `hivm.hir.vcmp` | Integer operand **and** (`dtype != i32` **or** `compare_mode not in {EQ, NE}`) | Only `i32` `==` / `!=` runs on Vector SIMD! Cast `i32` indices/values via `vcast` to `f32` before `<, <=, >, >=` (`vcmp`) |
| **3. Extended Multiply `vmulext`** | `hivm.hir.vmulext` | `i32` or `i64` operands (**always** lowers to scalar `arith.mului_extended`) | Replace `vmulext` with `vmulextended` or `f32`/`i32` standard arithmetic |
| **4. Innermost-Axis Scan** | `hivm.hir.vcumsum`, `vcumprod` | `cumDims.size() == 1` **and** (`dtype == i64` **or** scan axis is the last flattened dimension) | Transpose `[Rows, Cols] -> [Cols, Rows]` in UB before `vcumsum` so accumulation runs across contiguous rows, then transpose back |
| **5. `vreduce` Index & Rank** | `hivm.hir.vreduce` | On 910B/910_93 (Mem-based): `sum/prod/max/min/xori` with `i64`; `max/min_with_index` with `i16/i32/i64` or flattened `rank > 2`. On 950 (Reg-based): `max/min_with_index` when missing aligned `StridedLayoutAttr` | Cast inputs to `f32` before `max/min_with_index` (argmax/argmin), flatten to 2D before reduction, and enforce 32 B row stride |

## 3. Audit Intra-Core Pipe Sync and Cross-Core FFTS Synchronization

HIVM manages concurrency via two synchronization mechanisms injected by `InjectSync` (heuristic) or `GraphSyncSolver` (`--enable-hivm-graph-sync-solver=true`, optimal DAG solver):

1. **Intra-Core Pipeline Synchronization (`set_flag` / `wait_flag` / `pipe_barrier`)**:
   - Coordinates `PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_FIX` within one core using 8 hardware Event IDs per pipe pair (`#hivm.event<EVENT_ID0..7>`).
   - **Invariants**: Every `hir.set_flag <PIPE_SRC, PIPE_DST, EVENT_IDk>` must execute **1-to-1** with `hir.wait_flag <PIPE_SRC, PIPE_DST, EVENT_IDk>` across all control-flow paths (`scf.if`, `scf.for`). Never reuse an `EVENT_ID` before its matching `wait_flag` completes, and never exceed 8 simultaneously live events on the same `(PIPE_SRC, PIPE_DST)` pair.
   - Use `--hivm-compile-args=bishengir-sync-mode=BARRIERALL` (inserts `hir.pipe_barrier <PIPE_ALL>` between every op) to bisect whether a numerical corruption bug is caused by missing pipeline synchronization.
2. **Cross-Core FFTS Synchronization (`sync_block` / `sync_block_set` / `sync_block_wait`)**:
   - Uses the Fast Flag Transfer System (`FFTS`) hardware register (`hir.set_ffts_base_addr` added by `AddFFTSAddr` pass on 910B+).
   - `SyncBlockMode` must match the participating core domain: `ALL_CUBE (0)`, `ALL_VECTOR (1)`, `ALL_SUB_VECTOR (2)`, `BARRIER_CUBE (3)`, `BARRIER_VECTOR (4)`, `ALL (5)`. Joining an `ALL_VECTOR` barrier from an AIC function or letting a subset of cores early-return before `hir.sync_block` causes a deterministic hardware hang (`507015`).
