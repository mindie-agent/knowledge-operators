---
schema: mindie-graded-unit/1
id: unit-ascendc-simt-and-regbase-vector-programming-sop
revision: 1
tier: L2
category: sop
domain_path: ascendc/simt-regbase
evidence_strength: software_contract
title: 'AscendC SIMT (__simt_vf__) and RegBase (RegTensor/MaskReg) vector programming SOP: paradigm selection, 40 KB UB reserve, scatter deduplication and LocalMemBar'
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
provenance:
  source_type: public_software_contract_review
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-simt-best-practices/SKILL.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  supporting_source_revisions:
  - be5882ef29fa9943305085652d2025a038ba491c
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-simt-best-practices/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ascendc-regbase-best-practice/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/ops-design-paradigms/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/programming/simt_best_practices.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/programming/regbase_best_practice.md
  supersedes: []
triggers:
  error_signatures:
  - an Ascend950 kernel containing __simt_vf__ crashes with 507035 or corrupts data because UB allocations exceeded total_UB - 40 KB
  - a RegBase __vf__ kernel reads stale UB values in GetValue or Vld immediately after Vst because LocalMemBar<VEC_STORE, VEC_LOAD> is missing
  - a SIMT scatter or index_put kernel suffers severe GM write amplification or multi-core races on duplicate indices
  - a multi-step elementwise kernel writes and reloads UB after every Vector API call instead of keeping intermediates in RegTensor registers
  keywords:
  - AscendC
  - SIMT
  - __simt_vf__
  - LAUNCH_BOUND
  - RegBase
  - __vf__
  - RegTensor
  - MaskReg
  - LocalMemBar
  - VEC_STORE
  - VEC_LOAD
  - 40 KB UB reserve
  - scatter deduplication
  - GatherMask
  - UnpackB8
  - Interleave
  - Ascend950
---

# AscendC SIMT (`__simt_vf__`) and RegBase (`RegTensor`/`MaskReg`) vector programming SOP: paradigm selection, 40 KB UB reserve, scatter deduplication and `LocalMemBar`

## Use when

Designing or optimizing a Vector-side Ascend C kernel on Ascend 950 (c310 / DaVinci V350, where hardware SIMT `__simt_vf__` and register-level `__vf__` `RegTensor` are native) or choosing between tile-based Vector APIs, RegBase vector functions, and SIMT thread blocks.

## 1. Select the execution paradigm from data-access regularity and branching

Do not default to SIMT for contiguous elementwise math or to tile-based Vector APIs for fine-grained irregular indexing:

| Paradigm | Best-fit operator structure | Hardware mechanism & key advantage |
| --- | --- | --- |
| **Tile-based Vector API** (`Add`, `Mul`, `Exp`, `ReduceSum` on `LocalTensor`) | Contiguous bulk tensor math, standard row/column broadcast, whole-tile reductions | Amortizes scalar issue overhead across large UB tiles (`16 KB..64 KB`) with hardware repeat counters. |
| **RegBase Vector Function** (`__aicore__ __vf__`, `RegTensor<T>`, `MaskReg`) | Multi-step fusible elementwise chains (`>= 3` ops), loop-carried register accumulators, structured stride/bit manipulation (`GatherMask`, `UnpackB8`, `Interleave`, `Deinterleave`) | Keeps intermediates in the 32 vector registers (`V0..V31`, 256 B per register on c310), eliminating intermediate UB read/write round-trips between ops. |
| **SIMT Thread Block** (`__simt_vf__`, `@LAUNCH_BOUND(N)`, `Simt::GetThreadIdx()`) | Irregular index gather/scatter (`index_put`, `embedding_bag`, CSR sparse), data-dependent per-element branching, small-bucket sorting/histograms | Executes 32-thread warps (up to 1024 threads/block) with per-thread scalar/vector predicates and direct UB/GM addressing. |
| **Hybrid DMA + RegBase/SIMT** | Irregular or multi-step compute over large GM tensors | Host/AICore schedules double-buffered `DataCopyPad` (`GM <-> UB`) while `__vf__` or `__simt_vf__` processes UB tiles. |

## 2. Enforce the 40 KB SIMT UB reserve and warp-aligned thread geometry

1. **Mandatory 40 KB high-address UB reserve**: whenever a translation unit invokes any `__simt_vf__` function, the hardware reserves the **top 40 KB of UB** for SIMT thread register spill stacks and context frames:
   $$\text{UB}_{\text{usable}} = \text{UB}_{\text{total}} - 40\text{ KB} \quad (248\text{ KB} - 40\text{ KB} = 208\text{ KB on Ascend 950})$$
   All `TPipe::InitBuffer` queues (`TQue`) and `TBuf` allocations must fit within `UB_usable` (minus the standard stack/headroom margin). Allocating beyond `UB_usable` overwrites active SIMT thread stacks and triggers `507035` or silent corruption.
2. **Compile-time `@LAUNCH_BOUND` vs runtime `blockDim.x`**:
   - Declare `@LAUNCH_BOUND(MAX_THREADS)` (`MAX_THREADS <= 1024`, multiple of warp size `32`, typically `256` or `512`) so the compiler bounds per-thread register usage.
   - Choose runtime `threads = min(MAX_THREADS, ceil(work_items / 32) * 32)` so small tail workloads do not launch idle warps.
3. **1D grid-stride vs 2D row/column mapping**:
   - Use a 1D grid-stride loop (`for (int64_t i = tid; i < N; i += blockDim.x)`) for flat elementwise/gather workloads so arbitrary `N` is covered without tail-launch branches.
   - Use a 2D mapping (core/block = outer row, `tid` = inner column stride) for row-wise reductions or per-row TopK, ensuring adjacent `tid` lanes read adjacent memory addresses (`base + tid`).

## 3. Coalesce SIMT memory accesses and deduplicate random scatter writes

- **Stage through UB before irregular inner access**: when multiple threads reuse or permute data within a bounded window, copy the window from GM to UB via `DataCopyPad`, synchronize (`WaitFlag<HardEvent::MTE2_V>`), run the `__simt_vf__` kernel on UB, and copy back (`V_MTE3`).
- **Hoist integer division and modulo**: integer `/` and `%` inside a SIMT loop are multi-cycle instructions; precompute reciprocals or split flat indices into outer/inner loops when the stride is constant.
- **Eliminate random-scatter GM write amplification (`index_put` / `scatter` without `accumulate`)**:
  - **Multi-core target-range partitioning**: when duplicate indices may appear across the index tensor in overwrite mode (`accumulate = false`), partitioning by *source* index causes multi-core write races on GM. Instead, partition the **destination** row range `[rowStart, rowEnd)` across cores (`coreId`) and have each core filter for ` rowStart <= idx[i] < rowEnd`.
  - **Reverse scan + write-visited deduplication**: when the last write wins (`accumulate = false`) and duplicate indices target the same destination row, scan `i` **backward** from `numIndices - 1` down to `0` and maintain a UB bitmask/boolean table `visited[dstRow - rowStart]`. Write only on the first hit (`!visited[...]`), turning $K$ redundant GM row writes into a single write.

## 4. Manage RegBase `RegTensor`/`MaskReg` lifecycles and `LocalMemBar` hazards

Inside an `__aicore__ __vf__` function (or `@pto.vf_inline` in CANNBot DSL):

1. **Keep fusible chains in `RegTensor`**: load once from UB (`Vl` / `vld`), chain all arithmetic/comparison/select ops (`Vadd`, `Vmul`, `Vcmp`, `Vsel`, `Vconv`) in `RegTensor<T>` and `MaskReg`, and store once (`Vst` / `vst`) at the end of the basic block:
   ```cpp
   __aicore__ __vf__ void FusedScaleGate(ub_addr8_t dstUb, ub_addr8_t xUb, ub_addr8_t gUb, int32_t len) {
       for (int32_t i = 0; i < len; i += VectorLen<float>()) {
           RegTensor<bfloat16_t> xBf16 = Vld<bfloat16_t>(xUb + i * sizeof(bfloat16_t));
           RegTensor<bfloat16_t> gBf16 = Vld<bfloat16_t>(gUb + i * sizeof(bfloat16_t));
           RegTensor<float> x = Vconv<float, RoundMode::NONE>(xBf16);
           RegTensor<float> g = Vconv<float, RoundMode::NONE>(gBf16);
           MaskReg pos = Vcmp<CmpMode::GT>(g, 0.0f);
           RegTensor<float> gRelu = Vsel(pos, g, 0.0f);
           RegTensor<float> out = Vmul(x, gRelu);
           Vst(dstUb + i * sizeof(bfloat16_t), Vconv<bfloat16_t, RoundMode::RINT>(out));
       }
   }
   ```
2. **Enforce `LocalMemBar<VEC_STORE, VEC_LOAD>` on UB store-to-load forwarding**: the Vector Store unit (`VEC_STORE`) and Vector/Scalar Load units (`VEC_LOAD`) have decoupled pipelines into UB. Whenever a `__vf__` block stores a `RegTensor` to UB (`Vst` / `Tensor.store`) and subsequently reads that same UB address via `Vld`, `Vl`, or scalar `GetValue` (for example after `Vcmax` or `Vcadd` row reduction), insert an explicit memory barrier:
   ```cpp
   Vst(reduceUb, vMaxReg);
   Reg::LocalMemBar<MemBarType::VEC_STORE, MemBarType::VEC_LOAD>(); // or pto.vmem_bar("vst_vld")
   float scalarMax = GetValue<float>(reduceUb, 0);
   ```
   Omitting `LocalMemBar<VEC_STORE, VEC_LOAD>` causes `GetValue`/`Vld` to read stale UB contents from the previous iteration.
3. **Bound live `RegTensor` pressure (`<= 32` hardware vector registers)**: a double-width `fp32` `RegTensor` derived from `bf16` consumes two register slots (`Even`/`Odd`). If a loop body keeps more than ~24 register slots live simultaneously, the compiler spills registers back to UB, erasing the RegBase bandwidth advantage. Split independent chains or narrow intermediates to `bf16` where numerically safe (e.g., `ReLU` or sign-bit masks before `fp32` accumulation).

## Verification checklist

1. Verify `sum(TQue + TBuf) + headroom <= UB_total - 40 KB` whenever `__simt_vf__` is present.
2. Audit every `Vst` -> `Vld`/`GetValue` pair on overlapping UB addresses for `Reg::LocalMemBar<MemBarType::VEC_STORE, MemBarType::VEC_LOAD>()` (or `pto.vmem_bar("vst_vld")`).
3. Test SIMT scatter/gather kernels on duplicate indices, `N % 32 != 0` warp tails, and 3 repeated runs to confirm byte-identical determinism.
