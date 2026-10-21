---
category: sop
conditions:
  cann_versions:
  - '>=8.0.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - CATLASS C++ template GEMM, BatchedMatmul, GroupedMatmul, and MLA/Quantized epilogue
    kernels
domain_path: operators/skill-playbooks/ascendc-and-catlass
evidence_strength: software_contract
id: unit-agent-skills-catlass-gemm-grouped-matmul-and-epilogue-fusion-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/catlass-operator-design/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/catlass-operator-code-gen/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/catlass-operator-performance-optim/SKILL.md
  source_digest: 382229f1f815bc5042236a639b9e898955bbd0f9b903a1a35559ce3aeadf0467
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/catlass-operator-design/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Developer SOP from `Ascend/agent-skills` (`catlass-operator-design`, `code-gen`,
  `dev`, `performance-optim`) for building high-performance Cube kernels using CATLASS
  `Kernel` -> `BlockMmad` / `BlockEpilogue` -> `TileCopy` hierarchy.
tier: L2
title: CATLASS Template Library Operator Design, BlockMmad Tiling, and Epilogue Fusion
  SOP
triggers:
  error_signatures:
  - catlass operator design skill
  - catlass operator code gen skill
  keywords:
  - catlass operator design skill
  - catlass operator code gen skill
  - catlass operator performance optim
  - catlass blockmmad blockepilogue blockswizzle
  - catlass
  - block-mmad
  - block-epilogue
  - block-swizzle
  - grouped-matmul
  - l1-l0-pingpong
---
## 1. CATLASS Four-Layer Architectural Hierarchy (`catlass-operator-design`)

**CATLASS** (CANN Templates for Linear Algebra Subroutines) is the Ascend equivalent of CUTLASS, structuring Cube + Vector operators into four composable C++ template layers:
1. **`Device` / `Adapter` Layer**: Host launch wrapper, workspace allocation, and `Tiling` parameter dispatch.
2. **`Kernel` Layer (`catlass/gemm/kernel/`)**: Orchestrates multi-core grid scheduling (`BlockScheduler` / `BlockSwizzle`), AIC/AIV split-core coordination (`AscendC::GetBlockIdx()`), and the outer loop over output tiles.
3. **`Block` Layer (`BlockMmad`, `BlockEpilogue`)**:
   - **`BlockMmad`**: Manages the `GM -> L1 -> L0A/L0B -> L0C` mainloop with multi-stage double buffering / preload (`DispatchPolicy`).
   - **`BlockEpilogue`**: Manages `L0C -> UB -> GM` vector post-processing (bias add, dequantization, `SwiGLU` / `GELU`, per-token dynamic quantization, and residual add) on AIV cores.
4. **`Tile` & `Basic` Layer (`TileCopy`, `TileMmad`)**: Wraps hardware-specific `DataCopy` (`ND -> NZ` on-the-fly conversion in MTE1) and `Mmad` instructions (`16x16x16` for `FP16`/`BF16`, `16x16x32` for `INT8`).

## 2. CATLASS Performance Optimization Levers (`catlass-operator-performance-optim`)

1. **L1 & L0 Tile Shape Selection (`L1TileShape`, `L0TileShape`)**:
   - Standard compute-bound tile: `L1TileShape = GemmShape<128, 256, 256>` (or `<128, 128, 256>`), `L0TileShape = GemmShape<128, 256, 64>`.
   - Verify L1 residency: $2 \times (M_{L1} \times K_{L1} + K_{L1} \times N_{L1}) \times \text{sizeof}(T) \le 512\text{ KB}$.
   - Verify L0A/L0B residency: $2 \times M_{L0} \times K_{L0} \times \text{sizeof}(T) \le 64\text{ KB}$ (`L0A`) and $2 \times K_{L0} \times N_{L0} \times \text{sizeof}(T) \le 64\text{ KB}$ (`L0B`).
2. **L2 Cache Locality via `BlockSwizzle`**:
   - Configure 2D swizzling (`GemmIdentityBlockSwizzle<3>` or `<4>`) so adjacent AIC cores process neighboring tiles sharing the same $A$ rows or $B$ columns in the `192 MB` L2 cache, raising L2 hit rate from `< 40%` to `> 80%`.
3. **AIC-AIV Asynchronous Epilogue Overlap**:
   - When fusing Vector epilogues (`SwiGLU`, `Dequant`), use `CrossCoreWaitFlag` / `CrossCoreSetFlag` with ping-pong workspace slots so `AIC` starts the next `BlockMmad` tile immediately while `AIV` processes the finished `L0C` tile.

## Verification Checklist

- Confirm inner dimensions ($M, N, K$) or `DataCopyPad` policies handle non-16-aligned matrix shapes cleanly without out-of-bounds MTE1 reads.
