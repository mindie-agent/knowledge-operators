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
domain_path: compilation/bisheng-ir/hivm-dialect
evidence_strength: software_contract
id: unit-hivm-dialect-dma-macro-matmul-and-fixpipe-contract
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-bisheng-ir-compilation-pipeline-and-pass-lowering-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/dma_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/dma_fixpipe.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/dma_padding_quantization.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/macro_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hivm/macro_mmad_l1.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/apis/ir/hivm/dma_overview.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: 'HIVM dialect DMA, Fixpipe and Cube Macro operation contract: pipeline assignment,
  ND/NZ layout conversion and UnitFlag semantics'
triggers:
  error_signatures:
  - illegal HIVM DMA address space combination on Ascend910B for UB-to-L1 copy or
    L0C-to-UB fixpipe
  - misaligned ND2NZ or L1 descriptor stride in hivm.hir.nd2nz or hivm.hir.mmadL1
  - UnitFlag deadlock in Split-K or zero-trip Cube loops when unit_flag_mode is misconfigured
  keywords:
  - hivm.hir.load
  - hivm.hir.store
  - hivm.hir.copy
  - hivm.hir.fixpipe
  - hivm.hir.nd2nz
  - hivm.hir.nz2nd
  - hivm.hir.l12ub
  - hivm.hir.mmadL1
  - hivm.hir.batchMmadL1
  - hivm.hir.matmul
  - hivm.hir.mix_matmul
  - hivm.hir.mix_group_matmul
  - PIPE_FIX
  - UnitFlag
---
# HIVM dialect DMA, Fixpipe and Cube Macro operation contract: pipeline assignment, ND/NZ layout conversion and UnitFlag semantics

## 1. HIVM DMA Operations and Hardware Pipeline Mapping (`HIVMDMAOps.td`)

All HIVM operations live in `::mlir::hivm` with the `hivm.hir.` prefix and implement `DestinationStyleOpInterface` (`ins(...) outs(...)`). Each DMA op binds statically or dynamically to a hardware pipeline (`PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_FIX`, `PIPE_V`):

| Operation | Source $\to$ Dest Address Space | Hardware Pipe | Layout Conversion | On-the-Fly Attributes & Capabilities | SOC Support |
| --- | --- | --- | --- | --- | --- |
| `hivm.hir.load` | `gm` $\to$ `ub` | `PIPE_MTE2` | None (`ND -> ND`) | Left+right `pad_mode` (`Pad_Null`, `Pad_FirstElem`, `Pad_Value`), `pad_value`, `init_out_buffer`, cache `eviction` | All Ascend SOCs |
| `hivm.hir.store` | `ub` $\to$ `gm` | `PIPE_MTE3` | None (`ND -> ND`) | Hardware atomic writeback (`atomic_kind`: `NONE`, `ADD`, `MAX`, `MIN`, `AND`, `OR`, `XOR`, `XCHG`, `CAS`) | All Ascend SOCs |
| `hivm.hir.nd2nz` | `gm` $\to$ `cbuf` (L1) | `PIPE_MTE2` | `ND` $\to$ `NZ` | Converts row-major GM tiles into $16 \times 16$ fractal L1 blocks; supports `init_out_buffer` | All Ascend SOCs |
| `hivm.hir.nz2nd` | `cbuf` (L1) $\to$ `gm` | `PIPE_MTE3` | `NZ` $\to$ `ND` | Converts fractal L1 tiles back to row-major GM | All Ascend SOCs |
| `hivm.hir.l12ub` | `cbuf` (L1) $\to$ `ub` | `PIPE_MTE1` | `NZ` $\to$ `ND` | Moves L1 staging buffers into Vector UB | All Ascend SOCs |
| `hivm.hir.copy` | `ub` $\to$ `ub` | `PIPE_V` | None | Left-only padding; intra-UB vector register/buffer copy | All Ascend SOCs |
| `hivm.hir.copy` | `gm` $\to$ `cbuf` (L1) | `PIPE_MTE2` | None | Contiguous GM-to-L1 DMA without `ND2NZ` reformatting | All Ascend SOCs |
| `hivm.hir.copy` | `ub` $\to$ `cbuf` (L1) | `PIPE_MTE3` | None | Direct Vector UB to Cube L1 transfer | **Ascend 950 only** |
| `hivm.hir.fixpipe` | `cc` (L0C) $\to$ `gm` / `cbuf` | `PIPE_FIX` | `NZ2ND` / `NZ2DN` / `NZ2NZ` | `pre_quant` (`NO_QUANT`, `F322F16`, `F322BF16`, `S322I8`), `pre_relu` (`NORMAL_RELU`, `LEAKY_RELU`, `P_RELU`), `dual_dst_mode` | All Ascend SOCs |
| `hivm.hir.fixpipe` | `cc` (L0C) $\to$ `ub` | `PIPE_FIX` | `NZ2ND` / `NZ2DN` / `NZ2NZ` | Direct Cube accumulator to Vector UB transfer (zero GM workspace) | **Ascend 950 only** |
| `hivm.hir.gather_load` / `scatter_store` | `gm` $\leftrightarrow$ `ub` | `PIPE_MTE2` / `PIPE_MTE3` | None | Indexed SIMD gather/scatter with UB index tensor; `indirect_load`/`indirect_store` for pointer tables | All Ascend SOCs |

**Invariant**: On Ascend 910B and 910_93 (`Atlas A2`/`A3`), there is **no** hardware datapath for `ub -> cbuf` (`hir.copy`) or `cc -> ub` (`hir.fixpipe`); Cube-to-Vector handoff must stage through GM workspace (`cc -> gm` via `PIPE_FIX`, cross-core sync, then `gm -> ub` via `PIPE_MTE2`).

## 2. Cube Macro Operations and Cross-Pipeline Traits (`HIVMMacroOps.td`)

Unlike single-pipe operations (`SinglePipeOpTrait`), Cube matrix multiply operations carry `MacroOpTrait` and `MacroOpPipeTrait<"PIPE::PIPE_IN, PIPE::PIPE_OUT">`, spanning multiple internal hardware pipelines:

1. **Local MMAD (`HIVM_LocalMmadOp`: `MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">`)**:
   - `hivm.hir.mmadL1`: Loads fractal `NZ` tiles from L1 (`cbuf`) into L0A (`ca`, converted to `ZZ`) and L0B (`cb`, converted to `ZN`) via `PIPE_MTE1`, and executes Cube MAC accumulation into L0C (`cc`, `NZ` layout) on `PIPE_M`.
   - `hivm.hir.batchMmadL1`: Batched L1-to-L0C matrix multiply across a leading batch dimension.
   - Key operands: `$a` (`cbuf`), `$b` (`cbuf`), `$init_condition` (`i1` flag controlling whether L0C initializes to zero on `k == 0` or accumulates `C += A * B`), `$real_m`, `$real_k`, `$real_n` (dynamic unpadded extents), `$per_channel_bias`, `$c` (`cc` output).
2. **Global MMAD (`HIVM_GlobalMmadOp`: `MacroOpPipeTrait<"PIPE::PIPE_MTE2, PIPE::PIPE_MTE3">`)**:
   - `hivm.hir.matmul`: Full GM-to-GM matrix multiply macro encapsulating GM $\to$ L1 $\to$ L0A/L0B $\to$ L0C $\to$ GM.
   - `hivm.hir.mix_matmul`: GM-to-GM matrix multiply fused with Vector epilogue operations.
   - `hivm.hir.mix_group_matmul`: Grouped GEMM macro for MoE expert workloads with ragged token counts per expert.

## 3. UnitFlag Hardware Synchronization in Cube Loops

Local MMAD ops (`hir.mmadL1`, `hir.batchMmadL1`) support hardware `UnitFlag` coupling between `PIPE_M` (Cube compute) and `PIPE_FIX` (`fixpipe` L0C drain) via `$unit_flag_cond` (`i1`) and `$unit_flag_mode` (`UnitFlagArrayAttr`):

- `DISABLED (0)`: Standard software `set_flag`/`wait_flag` synchronization between `PIPE_M` and `PIPE_FIX`.
- `RESERVED (1)`: Hardware-reserved state.
- `ENABLED_WITHOUT_UPDATE (2)`: Checks unit flag without advancing the state counter (intermediate K-loop steps).
- `ENABLED_WITH_UPDATE (3)`: Signals `PIPE_FIX` immediately upon completion of the final K-tile so L0C drain overlaps the next tile's setup.
- **Zero-trip invariant**: When a dynamic K-loop or Split-K partition can execute zero iterations, `unit_flag_cond` and `init_condition` must remain balanced across all branches; otherwise `PIPE_FIX` hangs waiting for a `PIPE_M` token that never fires.
