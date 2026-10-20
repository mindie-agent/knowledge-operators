---
category: axiom
conditions:
  compiler: bishengir
  kernel_language:
  - triton
  - bishengir
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/bisheng-ir/pipeline-and-arch
evidence_strength: software_contract
id: unit-bisheng-ir-compilation-pipeline-and-pass-lowering-contract
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/compilation_pipeline/overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/compilation_pipeline/ascend_passes.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/compilation_pipeline/pass_dependencies.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/architecture/data_layout.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/architecture/memory_hierarchy.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/apis/ir/compilation_pipeline/overview.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: 'AscendNPU-IR / BiShengIR compilation pipeline contract: TTIR-to-Linalg 12-pass
  adapter, HFusion/HIVM lowering chains and SIMD/SIMT/MIX backend paths'
triggers:
  error_signatures:
  - bishengir-compile fails between kernel.ttir.mlir, kernel.ttadapter.mlir and kernel.npuir.mlir
  - pass ordering violation between InferFuncCoreType, InferHIVMMemScope, PlanMemory
    and InjectSync
  - DiscreteMaskAccessConversion or TritonToUnstructured lowers non-contiguous masks
    into scalar loops
  - SplitMixKernel fails or CVPipelining cannot pair Cube and Vector sub-kernels
  keywords:
  - BiShengIR
  - AscendNPU-IR
  - ttir_to_linalg
  - buildHFusionPipelines
  - buildLowerHIVMPipelines
  - ConvertHFusionToHIVM
  - SplitMixKernel
  - DiscreteMaskAccessConversion
  - AutoBlockify
  - FractalSharedEncodingAttr
  - ND2NZ
---
# AscendNPU-IR / BiShengIR compilation pipeline contract: TTIR-to-Linalg 12-pass adapter, HFusion/HIVM lowering chains and SIMD/SIMT/MIX backend paths

## 1. Dual Compilation Paths and Three Backend Lowering Targets

AscendNPU-IR lowers kernels through two primary frontend paths that converge at the HIVM/TritonGPU hardware mapping layer:

1. **BiShengIR Path (`TTIR -> TTAdapter/Linalg -> HFusion -> HIVM -> LLVM/CCE`)**: Default path for Ascend NPU optimization. `ttir_to_linalg()` converts Triton IR (`tt.*`) into structured Linalg/HFusion IR (`kernel.ttadapter.mlir`), then runs `buildHFusionPipelines`, `buildConvertToHIVMPipeline`, `buildHIVMTensorOptimizations`, and `buildLowerHIVMPipelines` to emit `kernel.npuir.mlir` (`hivm.hir.*`).
2. **Direct TritonGPU Path (`TTIR -> TTGIR -> LLVM`)**: Converts `tt.*` via `ConvertTritonToTritonGPU` into `ttg.*` (`#ttg.blocked`, `#ttg.mma`, `#ttg.shared`) and lowers via `ConvertTritonAscendGPUToLLVM`.

At the backend boundary of `buildLowerHIVMPipelines`, execution splits into three target paths:

| Backend Path | Lowering Sequence | Workload Affinity | Governing Passes & Encodings |
| --- | --- | --- | --- |
| **SIMD** | `HIVM -> TritonGPU -> LLVM` | Cube-dense GEMM / `tl.dot` and structured vector blocks | `ConvertHIVMToTritonGPU` (`pure-simt=false`), `FractalSharedEncodingAttr` for Cube L1/L0 fractal tiles |
| **SIMT** | `HIVM -> AscendDPX -> LLVM` | Irregular gather/scatter, fine-grained elementwise/reduction on Ascend 950 | `ArithToHIVMLLVMConversionPass`, `GPUToDPX` (`use-dpx=true`), `InsertMemSemanticForSimtVF` |
| **MIX (CV)** | `HIVM -> SplitMixKernel -> AIC + AIV kernels` | Cube + Vector fused operators (FlashAttention, MLA, fused GEMM epilogue) | `SplitMixKernel`, `MarkRealCoreType`, `InsertWorkSpaceForMixCV`, `InsertCVTightCoupledBuffer`, `CVPipelining` (`enable-mixed-cv=true`) |

## 2. The 12-Step `ttir_to_linalg()` Ascend Adapter Sequence

After TTIR frontend optimization (`TritonCombineOps -> TritonReorderBroadcast -> TritonRewriteTensorPointer -> TritonRewriteTensorDescriptorToPointer -> TritonLoopInvariantCodeMotion -> TritonLoopAwareCSE`), `ttir_to_linalg()` executes a strict 12-pass sequence:

1. **`AutoBlockify`**: Merges logical blocks onto physical cores (`auto_blockify_size > 1`, active only when `TRITON_ALL_BLOCKS_PARALLEL=1`), rewriting `GetProgramIdOp` and bounding grid dimensions above `65535` when blocks have no inter-block ordering dependency.
2. **DAG Affinity Optimization (`add_auto_scheduling=True`)**: For mixed Cube+Vector kernels, executes `dag-sync` (inserts `hivm.sync_block_set`/`wait` and GM/UB transfers based on `SharedMemoryAliasAnalysis`) $\to$ `dag-scope` (partitions ops into `scope.scope @aic` and `@aiv`) $\to$ `dag-ssbuf` (maps vector intermediates into shared storage buffers).
3. **`TritonToStructured` (Pass 1)**: Decomposes pointer/mask integer division and modulo (`div`/`rem`) into multi-dimensional `PtrState`/`MaskState` (`RewriteAddPtrOp`, `RewriteLoadOp`, `RewriteStoreOp`, `RewriteFor`, `RewriteWhile`). Constraint: the original iteration extent must be divisible by the split axis; `while` loops containing inner `if` blocks cannot be structured.
4. **`DiscreteMaskAccessConversion`**: Must run **before** `TritonToUnstructured`. Rewrites non-contiguous masked stores to `load -> select -> store`, masked loads to full `load -> select(other)`, and masked `atomic_add` to `select -> atomic_add`. Controlled by `compile_on_910_95` and `force_simt_template`.
5. **`TritonToAnnotation`**: Lowers `al.compile_hint` (`triton::AnnotationOp`) to `annotation.mark`.
6. **`TritonToUnstructured`**: Expands discrete-axis `Load`/`Store`/`AtomicRMW`/`AtomicCAS` into nested `scf.for` + `scf.if` scalar memory loops.
7. **`TritonToHIVM`**: Lowers `al.sync_block_all`/`set`/`wait` directly into HIVM block-sync ops.
8. **`TritonToHFusion`**: Lowers `tt.histogram` into hardware-accelerated `hfusion.histogram`.
9. **`TritonToLLVM`**: Lowers `tt.elementwise_inline_asm` into `llvm.inline_asm` mapped to CCE intrinsics.
10. **`BubbleUpOperation`**: Hoists `tensor.extract` / `tensor.extract_slice` above loop bodies when indices are loop-invariant.
11. **`TritonToStructured` (Pass 2)**: Linearizes pointer/mask expressions introduced by passes 4–10.
12. **`TritonToLinalg`**: Final conversion of remaining `tt.*` ops into Linalg/HFusion structured ops (`enable_nd2nz_on_vector`, `enable_select_analysis`, `compile_on_910_95`).

## 3. Mandatory Pass Ordering Invariants Across HFusion and HIVM

`buildHFusionPipelines` and `buildLowerHIVMPipelines` enforce strict producer-consumer pass contracts:

| Prerequisite Pass | Dependent Pass | Architectural Invariant |
| --- | --- | --- |
| `ConvertGenericToNamedOp` | `HFusionOpFusion` $\to$ `AutoSchedule` | Fusion and schedule generation require canonical named Linalg/HFusion ops (`ComposeMultiReduce -> DecomposeMulti -> HFusionInlineBrc`). |
| `InferFuncCoreType` | `InferHIVMMemScope` | Address-space assignment (`#hivm.address_space<ub/cbuf/ca/cb/cc>`) depends on whether the enclosing function targets `CUBE`, `VECTOR`, or `CUBE_AND_VECTOR`. |
| `InferHIVMDataLayout` | `InsertConvertLayout` $\to$ `PropagateConvertLayout` | Fractal layout conversions (`ND <-> NZ/ZZ/ZN`) require inferred operand layout attributes before `ConvertLayoutToTranspose`. |
| `MarkStrideAlign` | `EnableStrideAlign` $\to$ `LiftLowestStride` | Stride alignment padding (32 B UB, 512 B L1) must be marked on subviews before buffer allocation sizes are adjusted. |
| `MarkMultiBuffer` | `EnableMultiBuffer` $\to$ `PlanMemory` | Double-buffered tensors must be marked before `PlanMemory` computes physical `hivm.hir.pointer_cast` offsets; otherwise ping-pong buffers alias or overflow UB/L1. |
| `PlanMemory` | `InjectSync` / `GraphSyncSolver` | Hazard analysis between `PIPE_MTE2`, `PIPE_V`, `PIPE_M`, `PIPE_FIX`, and `PIPE_MTE3` requires physical buffer offset aliasing from `PlanMemory`. |
| `SplitMixKernel` | `CVPipelining` | Mixed Cube-Vector kernels must be split into AIC and AIV sub-functions with workspace/tightly-coupled buffers before software pipelining. |

## 4. Hardware Address Spaces and Data Layout Contracts in IR

Every shaped memref in HIVM carries an explicit `#hivm.address_space<...>` and data layout encoding:

- **Address Spaces**: `gm` (1, Global Memory), `cbuf` (2, L1 Cube Buffer, 512 KiB), `ca` (3, L0A, 64 KiB), `cb` (4, L0B, 64 KiB), `cc` (5, L0C accumulator, 128 KiB on 910B/910_93, 256 KiB on 950), `ub` (6, Unified Buffer, 192 KiB / `1,572,864` bits on 910B/910_93, 248 KiB / `2,031,616` bits usable on 950).
- **Cube Fractal Layouts**:
  - `ND`: Row-major standard tensor layout in GM and UB.
  - `NZ` (Fractal_NZ): Column-major outer tiles of $16 \times 16$ micro-tiles (for 16-bit elements: $[N_1, M_1, M_0=16, N_0=16]$, or $[K_1, M_1, M_0=16, K_0=32]$ for int8), required in L1 (`cbuf`) before L0A/L0B transfer.
  - `ZZ` / `ZN`: L0A (`ZZ`, row-major $16 \times 16$ micro-tiles) and L0B (`ZN`, column-major $16 \times 16$ micro-tiles) layouts consumed by Cube `mmadL1`, producing `NZ` in L0C (`cc`).
  - `last_pass.mlir` corresponds to the IR snapshot immediately after `InjectSync` (`--bishengir-print-ir-after=hivm-inject-sync`) and before final LLVM lowering.
