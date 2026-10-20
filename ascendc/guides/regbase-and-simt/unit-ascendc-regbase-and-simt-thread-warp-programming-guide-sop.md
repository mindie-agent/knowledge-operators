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
  - RegBaseVector
  - SimtKernels
  - HybridSimdSimt
  - CrossGenMigration2201To3510
domain_path: ascendc/guides/regbase-and-simt
evidence_strength: reproduced_empirical
id: unit-ascendc-regbase-and-simt-thread-warp-programming-guide-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascendc-simt-and-regbase-vector-programming-sop
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/guides/regbase, guides/cross_gen_migration_guide,
    and apis/simt_api documentation
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/regbase/api_families_masks_and_memory_contracts.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/cross_gen_migration_guide/a3_to_a5_ub_residency_is_the_port_decision.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/cross_gen_migration_guide/arch22_to_arch35_porting_playbook.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simt_api/simd_simt_hybrid_programming_intro/programming_model.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simt_api/warp_functions/warp_functions_intro.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/regbase/api_families_masks_and_memory_contracts.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-evolutionary-strategy-vector-simd-and-ub-tiling-mutator
- unit-ascendc-vector-simd-softmax-activation-sort-and-gather-api-contract
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Ascend C RegBase (__simd_vf__ register-based
  vector programming), SIMT thread/warp programming (warp vote/shfl/reduce, cooperative_groups,
  atomic/fence contracts), and DAV_2201-to-DAV_3510 (A2/A3 to A5) cross-generation
  kernel migration.
tier: L2
title: Ascend C RegBase (__simd_vf__) & SIMT Thread/Warp Hybrid Programming Guide
  SOP
triggers:
  error_signatures:
  - On Ascend 950 (ARCH 3510 / DAV_3510), Vector Core introduces 32 vector registers
    (RegTensor) in __simd_vf__ alongside 32-thread/warp SIMT execution (up to 1024
    threads per AIV)
  - Address-space conversions (_cvta_generic_to_ubuf/_global) and memory fences (asc_syncthreads,
    asc_threadfence_block) must bound every shared UB exchange between SIMD and SIMT
    phases
  - Cross-generation migration from c220 (2201) to A5 (3510) must first decide whether
    the tensor fits in A5's enlarged 256 KB UB before choosing L1/L2 or RegBase/SIMT
    paths
  keywords:
  - regbase
  - simd_vf
  - simt_api
  - warp_shfl
  - asc_syncthreads
  - cooperative_groups
  - arch22_to_arch35_migration
  - unalign_reg_for_load
---
## Preconditions & Signals

Apply this SOP when developing or migrating Ascend C Vector kernels on Ascend 950 (`ARCH 3510`) using the four-level progressive programming hierarchy (`L1` high-level API $\rightarrow$ `L2` RegBase `__simd_vf__` $\rightarrow$ `L3/L4` SIMT thread/warp and SIMD/SIMT hybrid):

1. **Choosing Between SIMD (`LocalTensor`), RegBase (`__simd_vf__`), and SIMT (`__simt_vf__`)**:
   - **Buffer-Based SIMD (`LocalTensor`)**: Best for coarse bulk vector pipelines where UB-to-UB reads/writes are amortized or high-level APIs (`SoftMax`, `Sort`) are used.
   - **RegBase (`__simd_vf__` / `RegTensor`)**: Best for multi-step elementwise/norm/dequant arithmetic chains on Ascend 950 where keeping intermediates in the 32 vector registers (`vreg`) eliminates UB load/store round-trips and unlocks dual-issue (`R1`–`R4`).
   - **SIMT (`__simt_vf__` / Thread-Warp Model)**: Best for data-dependent branching, irregular index scatter/gather, small-channel histograms, or thread-cooperative reductions where per-lane scalar/vector logic is awkward in SIMD masks (`R7`).
2. **A2/A3 (`DAV_2201`) to A5 (`DAV_3510`) Porting Decision (`a3_to_a5_ub_residency_is_the_port_decision.md`)**:
   - Ascend 950 expands UB from `192 KB` to `256 KB` per AIV core and introduces direct `L0C <-> UB` FixPipe paths and hardware MXFP8/MXFP4 support. Always evaluate whether a multi-pass A2/A3 kernel can collapse into a single UB-resident pass on A5 before porting low-level loop structure verbatim.

## Procedure

### Step 1: RegBase (`__simd_vf__`) Programming Discipline (`L2`)
1. **Register Lifecycle & Mask Contracts (`api_families_masks_and_memory_contracts.md`)**:
   - Declare register kernels with `__simd_vf__` and bind vector registers via `RegTensor<T>` alongside predicate mask registers (`CreateMask` / `CmpMask`).
   - Load from UB into `RegTensor` via `Copy(dstReg, srcUb, ...)` (or `UnalignRegForLoad` when UB offset is not 32B-aligned, `R5`), perform fused arithmetic (`Add`, `Mul`, `Fma`, `Cast`, `Deintlv`, `Intlv`) entirely in registers, and store final results back to UB via `Copy(dstUb, srcReg, ...)`.
2. **Avoid Register Spill (`<= 32` Live `RegTensor`) & Software-Loop Degradation**:
   - Keep live `RegTensor` variables within the 32-register hardware file (`R2`); scope temporary `RegTensor` objects inside inner blocks or overwrite dead registers in-place.
   - Keep loop bounds inside `__simd_vf__` simple compile-time or invariant scalar counts so the compiler generates zero-overhead Hardware Loops (`R4`).

### Step 2: SIMT Thread/Warp & Cooperative Groups Programming (`L3/L4`)
1. **Thread Hierarchy & Builtin Coordinates**:
   - Organize SIMT execution into 32-thread warps (`warpSize = 32`, up to 32 warps = 1024 threads per AIV block).
   - Index threads via builtin variables (`threadIdx.x`, `blockIdx.x`, `blockDim.x`) or `cooperative_groups` (`thread_block`, `thread_block_tile<32>`, `coalesced_group`) and `laneid()` / `lanemask_lt()`.
2. **Warp-Level Primitives (`warp_vote`, `warp_shfl`, `warp_reduce`)**:
   - Use `asc_activemask()`, `asc_all(pred)`, `asc_any(pred)`, and `asc_ballot(pred)` for warp-wide predicate consensus.
   - Exchange register values across lanes without touching UB using `asc_shfl`, `asc_shfl_up`, `asc_shfl_down`, and `asc_shfl_xor`.
   - Use `asc_reduce_add`, `asc_reduce_max`, and `asc_reduce_min` for single-instruction warp reductions before writing one per-warp summary to UB.
3. **Half2 / BFloat16x2 & FP8 SIMD-Within-SIMT Intrinsics**:
   - Process paired 16-bit elements per thread using `half2` / `bfloat16x2` intrinsics (`_haddx2`, `_hmulx2`, `_hcmadd`, `h2exp`, `_bfloat1622float2`) and hardware FP8/HIF8 conversion intrinsics (`_asc_cvt_float2_to_fp8x2`, `_e4m3x22float2`, `_float22hif82_rn`) so each thread issues 32-bit/64-bit coalesced memory transactions.

### Step 3: SIMD + SIMT Hybrid Synchronization & Address-Space Conversion
1. **Explicit Address-Space Qualifiers (`__gm__`, `__ubuf__`, `__local__`)**:
   - Use `_is_ubuf(ptr)` / `_is_global(ptr)` predicates and explicit address-space conversions (`_cvta_ubuf_to_generic`, `_cvta_generic_to_ubuf`, `_cvta_generic_to_global`) when passing UB or GM pointers into SIMT helper functions.
2. **Intra-Block & Memory Fence Protocol**:
   - Synchronize SIMT threads within an AIV block via `asc_syncthreads()` (`__syncthreads()`) and enforce memory ordering on UB/GM writes via `asc_threadfence_block()` or `asc_threadfence()`.
   - When transitioning between a SIMD `TQue` / `DataCopy` phase and a SIMT `__simt_vf__` phase sharing UB, insert the required `MTE2_V` / `V_MTE3` event flags outside the SIMT launch and `asc_syncthreads()` at the SIMT boundary so no warp reads partially DMA-loaded UB lines.

### Step 4: Cross-Generation Migration (`DAV_2201` $\rightarrow$ `DAV_3510` / Ascend 950)
1. **Replace Deprecated `c220` Interfaces (`2201_to_3510_arch_changes.md`)**:
   - Replace deprecated `_sync` C-APIs and `asc_create_iter_reg` / `asc_bfloat162e1m2x2` calls with their CANN 9.0/9.1 RegBase and SIMT equivalents (`deprecated_interface.md`).
   - Audit `LoadData` `ifTranspose` on 32-bit types (`fp32`/`b32`): on `ARCH 3510`, pre-transpose in UB/L1 rather than relying on `LoadData(..., ifTranspose=true)` for `B32` L1-to-L0B transfers.

## Verification & Checks

1. **Warp Divergence Audit on `asc_shfl_*` and `asc_syncthreads()`**:
   - Verify that `asc_syncthreads()` is reached by all threads in the block (never inside a divergent `if (tid < N)` branch unless the condition is uniform across the entire block).
2. **Cache Coherency on SIMT Global Loads/Stores**:
   - Use `asc_ldcg` / `asc_stcg` / `asc_stwt` and `asc_dcci_single` / `asc_dcci_entire` appropriately when SIMT threads read/write global memory flags across cores.

## Failure Modes & Fallbacks

- **Symptom**: RegBase `__simd_vf__` kernel runs slower than standard UB `LocalTensor` SIMD on Ascend 950.
  - **Root Cause**: Allocating $> 32$ live `RegTensor` variables inside a mega-fused `__simd_vf__` causes compiler register spilling to local stack memory.
  - **Fix**: Split the mega `__simd_vf__` at its narrowest live-variable cut-point or reuse dead `RegTensor` variables in-place (`R2`).
