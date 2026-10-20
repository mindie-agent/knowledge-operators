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
  - MatmulImpl
  - BmmChainedMatmul
  - ManualMmad
  - LoadData2d3d
  - FixPipeCopy
domain_path: ascendc/apis/cube-and-matmul
evidence_strength: software_contract
id: unit-ascendc-matmul-impl-iterate-gettensorc-and-gemm-api-contract
inherits_from:
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend910b-910c-and-310p-memory-model-and-target-axioms
provenance:
  attribution: Distilled from CANNBot-Knowledge ops/ascendc/apis/simd_api/adv_api/cube_compute
    and basic_api/cube_compute_isasi software contracts
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/matmul_config.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/matmul_policy.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/get_matmul_compute_space/matmul_get_tmp_buf_size.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute_isasi/cube_compute_load/data_copy_gm_to_l1_nd2_nz.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/appendix/cube_instruction_theoretical_perf_summary.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/matmul_config.md
  source_type: public_software_contract_review
  supersedes: []
related:
- unit-ascendc-cube-l1-residency-l0-pingpong-and-fixpipe-fusion-sop
- unit-cannbot-dsl-matmul-and-catlass-gemm-epilogue-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Foundational Ascend C software contract for the high-level MatmulImpl API
  (MatmulType, MatmulConfig, REGIST_MATMUL_OBJ, SetTensorA/B, Iterate/IterateAll,
  GetTensorC, End), host MatmulApiTiling, and low-level Cube primitives (LoadData,
  LoadData3D, Mmad, Fixpipe).
tier: L1
title: Ascend C MatmulImpl, MatmulConfig/Policy, Iterate/GetTensorC & Low-Level Mmad
  API Contract
triggers:
  error_signatures:
  - REGIST_MATMUL_OBJ must be invoked once per kernel launch outside inner tile loops,
    and on ARCH 3510 (Ascend 950) Mix kernels must initialize TPipe before REGIST_MATMUL_OBJ
    correctly
  - In MIX (AIC+AIV) kernels where AIV invokes MatmulImpl asynchronously, Iterate<false>()
    / IterateAll<false>() must be paired with GetTensorC<false>() to avoid synchronous
    AIC/AIV stalls
  - Low-level Mmad requires A in FRACTAL_Z (RowMajor 16xC0 fractals in L0A) and B
    in FRACTAL_ZN (ColMajor C0x16 fractals in L0B), with cInit=true on the first K
    step and cInit=false on subsequent steps
  keywords:
  - matmul_impl
  - regist_matmul_obj
  - matmul_type
  - matmul_config
  - iterate_all
  - get_tensor_c
  - matmul_api_tiling
  - mmad
  - load_data
  - fixpipe
---
## Core Axioms & Invariants

### 1. High-Level `MatmulImpl` Type & Configuration Contract (`matmul::MatmulImpl`)
1. **`MatmulType<TPosition, CubeFormat, Dtype, isTrans, LayoutMode, isIbShare>`**:
   - `A_TYPE`: `TPosition::GM` or `TPosition::A1` (`L1`) or `TPosition::VECCALC`/`TSCM`; `CubeFormat::ND` or `CubeFormat::NZ`; `Dtype` (`half`, `bfloat16_t`, `float`, `int8_t`, `int4b_t`, `fp8_e4m3fn_t`); `isTrans` (`true`/`false`); optional `isIbShare = true` when two AIV cores in a 1:2 MIX group share the same L1 operand (`P71`).
   - `B_TYPE`: `TPosition::GM` or `TPosition::B1` (`L1`); `CubeFormat::ND`, `CubeFormat::NZ`, or `CubeFormat::NZ_TRANS`.
   - `C_TYPE`: `TPosition::GM`, `TPosition::VECIN` (`UB`), or `TPosition::CO1` (`L0C`); `CubeFormat::ND` or `CubeFormat::NZ`; `Dtype` (`float`, `half`, `bfloat16_t`, `int32_t`).
   - `BIAS_TYPE`: `TPosition::GM` or `TPosition::C1` (`L1`) or `TPosition::VECIN`.
2. **`MatmulConfig` Modes (`CFG_NORM`, `CFG_MDL`, `CFG_PRELOAD`)**:
   - `CFG_NORM`: Basic Normal scheduling (splits single-core work into `baseM x baseN` outputs, iterating $K$ per output tile).
   - `CFG_MDL` (`GetMDLConfig`): Model-driven L1 residency scheduling that reuses L1 $A$/$B$ panels across multiple `baseM`/`baseN` steps before evicting L1 (`P46`, `P86`).
   - `doMTE2Preload`: Enables M/N-direction MTE2 software preloading across output tiles (requires `CFG_MDL` / `SpecialMDL`, full-K or sufficient L1 double-buffer budget).
3. **Registration & Lifecycle Sequence**:
   - **Registration**: Call `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm0, &tiling0, mm1, &tiling1)` **once** during operator `Init()` (never inside a loop!). When registering two chained `MatmulImpl` instances (`qkMatmul`, `pvMatmul` in FlashAttention), register both in a single `REGIST_MATMUL_OBJ` macro call so their L1/L0/workspace allocations are coordinated.
   - **Per-Task Call Order**:
     1. Optional runtime overrides: `mm.SetOrgShape(orgM, orgN, orgK)` $\rightarrow$ `mm.SetSingleShape(singleM, singleN, singleK)` $\rightarrow$ `mm.SetTail(tailM, tailN, tailK)`.
     2. Bind operands: `mm.SetTensorA(aTensor, isTransA)`, `mm.SetTensorB(bTensor, isTransB)`, optional `mm.SetBias(biasTensor)` / `mm.SetQuantVector(deqScale)`.
     3. Execute:
        - **Full-Output Mode**: `mm.IterateAll(dstC, isAtomicAdd)` (computes all `baseM x baseN` sub-tiles of `singleM x singleN` and writes to `dstC`).
        - **Step-by-Step Sub-Tile Mode**: `while (mm.Iterate()) { mm.GetTensorC(ubOrGmSlice, isAtomicAdd, enSequentialWrite); }`.
        - **Async MIX Mode (`P63`)**: `mm.Iterate<false>()` (or `mm.IterateAll<false>(dstC)`) followed by `mm.GetTensorC<false>(...)` / `mm.WaitIterateAll()` so the AIV caller overlaps Vector work while AIC computes.
     4. Cleanup: `mm.End()` after the loop completes.

### 2. Host `MatmulApiTiling` & `MultiCoreMatmulTiling` Contract (`P104`, `P117`, `X40`)
1. **Explicit Buffer Space & Split Constraints (`P117`)**:
   - Configure `tiling.SetAType(...)`, `SetBType(...)`, `SetCType(...)`, `SetBiasType(...)`, `SetOrgShape(...)`, `SetShape(...)`, and `SetBufferSpace(l1Size, l0cSize, ubSize)`.
   - When chaining two `MatmulImpl` instances (`QK` and `PV`) that share L1 concurrently, partition `l1Size` explicitly via `SetBufferSpace` and constrain `depthA1` / `depthB1` and `SetFixSplit(baseM, baseN, baseK)` so combined resident + streaming L1 buffers never exceed `512 KB`.
2. **Preventing the Single-Core Small-Shape Trap (`P104`, `X40`)**:
   - By default, `MatmulApiTiling::GetTiling()` may set `singleCoreM = M, singleCoreN = N` (`usedCoreNum = 1`) when the entire matrix fits in one core's L1/L0C!
   - For multi-core `MultiCoreMatmulTiling` (or per-core `MatmulApiTiling` invoked in a multi-core kernel), always bound `singleCoreM <= ceil(M / mBlocks)` and `singleCoreN <= ceil(N / nBlocks)` before calling `GetTiling()`.

### 3. Low-Level Cube Primitives (`LoadData`, `LoadData3D`, `Mmad`, `Fixpipe`)
1. **`GM -> L1` (`DataCopy`) & `L1 -> L0A/L0B` (`LoadData` / `LoadData3D`)**:
   - `L0A` requires `FRACTAL_Z` (`RowMajor` within each $16 \times C_0$ fractal: $C_0 = 16$ for FP16/BF16/FP32, $C_0 = 32$ for INT8/FP8, $C_0 = 64$ for INT4/FP4).
   - `L0B` requires `FRACTAL_ZN` (`ColMajor` within each $C_0 \times 16$ fractal).
   - `LoadData2DParams` (`startIndex`, `repeatTimes`, `srcStride`, `sid`, `dstGap`, `ifTranspose`): moves fractals from `A1/B1` (`L1`) to `A2` (`L0A`) or `B2` (`L0B`) on `PIPE_MTE1`.
   - `LoadData3DParamsV2Pro`: expands 2D/3D spatial feature maps from `L1` into `im2col` fractals in `L0A` on-the-fly (`P129`).
2. **`Mmad(l0cDst, l0aSrc, l0bSrc, MmadParams)` (`PIPE_M`)**:
   - `MmadParams`: `{m, n, k, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal}`.
   - **First K-Slice vs Subsequent K-Slices**:
     - On $k_{\text{step}} == 0$ (and no L0C bias preload): set `cmatrixInitVal = true` (`cInit = true`) to zero-initialize the L0C accumulator during the first MMAD.
     - On $k_{\text{step}} > 0$ (or when bias was preloaded into L0C via `DataCopy` / `LoadData`): set `cmatrixInitVal = false` (`cInit = false`) so `Mmad` accumulates into existing L0C values!
3. **`Fixpipe(dstGmOrUb, srcL0c, FixpipeParamsV220 / FixpipeParamsC310)` (`PIPE_FIX`)**:
   - Drains `L0C` (`FP32` or `INT32`) to `GM` (or directly to `UB` on Ascend 950 / `L1` for bias/dequant paths), applying inline type conversion (`FP32 -> FP16/BF16`), `RELU`/clipping, or `deq_scale` quantization (`INT32 -> FP16/INT8`).
   - Synchronize `M -> FIX` (`SetFlag<HardEvent::M_FIX>` / `WaitFlag<HardEvent::M_FIX>`) before `Fixpipe`, and `FIX -> M` before reusing the `L0C` buffer for the next output tile.

## Verification Checklist

- [ ] `SetOrgShape` reflects the physical GM matrix stride while `SetSingleShape` / `SetTail` reflects the current core's valid tile size.
- [ ] `cmatrixInitVal` (`cInit`) is `true` only on the first $K$-iteration and `false` on all subsequent $K$-iterations.
- [ ] On `c220` (910B), `MatmulImpl` with `INT32` output and multiple fragments uses `CubeFormat::ND` for `C_TYPE` rather than `CubeFormat::NZ` (`disc_x11`).
