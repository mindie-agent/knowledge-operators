---
category: sop
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
  - HostTiling
  - OpDefRegistration
  - TilingKeyTemplateDispatch
  - UbMemoryPlanning
  - WorkspaceAllocation
domain_path: ascendc/guides/tiling-and-ub-planning
evidence_strength: reproduced_empirical
id: unit-ascendc-host-tilingkey-workspace-and-ub-partitioning-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/guides/registration, guides/memory,
    and apis/utils_api documentation
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/host_metadata_runtime_attrs_and_graph_dtypes.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/memory/double_buffer_slot_counts_and_resident_parameters.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/utils_api/tiling_template_programming/get_tpl_tiling_key.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/utils_api/prototype_register_management/op_ai_core_def/launch_with_zero_ele_output_tensors.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/apis/simd_api/basic_api/kernel_tiling/get_tiling_data_with_struct.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/host_metadata_runtime_attrs_and_graph_dtypes.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-datacopy-datacopypad-tpipe-tque-and-flag-sync-api-contract
- unit-ascendc-bisheng-ccec-and-161001-compilation-error-triage-workflow
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Ascend C host-side OpDef registration, PlatformAscendc
  hardware resource querying, structured TilingKey template selection, system + user
  workspace calculation, and 32B-aligned UB memory budget planning.
tier: L2
title: Ascend C Host TilingKey Dispatch, OpDef Registration, Workspace Sizing & UB
  Partitioning SOP
triggers:
  error_signatures:
  - Total UB allocations (TQue + TBuf + high-level API tmpSize) per AIV core must
    never exceed platform UB capacity (192 KB on 910B/910C, 256 KB on 950) and every
    buffer size must be rounded up to a 32B multiple
  - Host TilingData struct layout and TilingKey decimal/bitfield encoding must match
    device GET_TILING_DATA_WITH_STRUCT / TILING_KEY_IS macros 1:1
  - System workspace size from PlatformAscendc::GetLibApiWorkSpaceSize() must be added
    to user workspace size in GetWorkspaceSizes()
  keywords:
  - tiling_data_def
  - tiling_key_is
  - ascendc_tpl_sel_param
  - platform_ascendc
  - get_lib_api_workspace_size
  - ub_partitioning
  - op_def_registration
  - zero_element_tensor
---
## Preconditions & Signals

Apply this SOP when authoring or auditing the Host-to-Device boundary (`op_host/*.cpp`, `*_tiling.h`, `op_kernel/*.cpp`) of an Ascend C operator:

1. **Dynamic Hardware Resource Querying (`platform_ascendc::PlatformAscendc`)**:
   - Never hardcode `coreNum = 24` or `ubSize = 196608`; query `GetCoreNumAic()`, `GetCoreNumAiv()`, and `GetSocMemSize(platform_ascendc::CoreMemType::UB, ubSize)` from `PlatformAscendc(context->GetPlatformInfo())`.
2. **Multi-Dimensional `TilingKey` Explosion vs Template Selection**:
   - Hand-written nested `if/else` chains for `(dtype, format, split_mode, broadcast_mode, is_aligned)` drift out of sync between Host `SetTilingKey` and Device `TILING_KEY_IS`.
3. **Zero-Element Tensor & Optional Input Handling**:
   - Passing empty tensors (`numel == 0`) or `nullptr` optional `aclTensor*` inputs into a kernel without host short-circuiting or explicit `LaunchWithZeroEleOutputTensors` configuration causes `561002` / `507035` runtime faults.

## Procedure

### Step 1: `OpDef` Prototype Registration & Format/Dtype Binding (`op_host/*_def.cpp`)
1. **Input/Output Parameter & Attribute Contracts**:
   - Declare all inputs, optional inputs (`.ParamType(Option::OPTIONAL)`), dynamic tensor lists (`.ParamType(Option::DYNAMIC)`), outputs, and attributes (`Attr("...").AttrType(OPTIONAL/REQUIRED)`) in `OpDef`.
   - Ensure attribute order, names, and C++ types (`int64_t`, `float`, `bool`, `std::vector<int64_t>`) match the `RuntimeAttrs` extraction indices in `TilingFunc` 1:1 (`host_metadata_runtime_attrs_and_graph_dtypes.md`).
2. **AICore Config Flags & Zero-Element Policy**:
   - Configure `DynamicCompileStaticFlag(true)`, `DynamicFormatFlag(true)`, and `DynamicShapeSupportFlag(true)` as required by the operator contract.
   - If the kernel must execute even when an output tensor has 0 elements (e.g., to initialize side-effect buffers or scalar counters), explicitly enable `.LaunchWithZeroEleOutputTensors(true)` on `OpAICoreDef`; otherwise handle `numel == 0` gracefully in Host tiling (`SetBlockDim(0)` or no-op tiling key).

### Step 2: Structured `TilingKey` Encoding & Template Programming (`ASCENDC_TPL_SEL_PARAM`)
1. **Orthogonal Decimal or Macro-Driven `TilingKey` Design (`D2`, `D3`)**:
   - Encode orthogonal compile-time specialization axes into a single `uint64_t` `TilingKey` using either `ASCENDC_TPL_SEL_PARAM` / `GET_TPL_TILING_KEY(dType, layoutMode, tileRegime, tailFlag)` or fixed decimal place values:
     $$\text{TilingKey} = 1000 \cdot \text{dtypeId} + 100 \cdot \text{algoId} + 10 \cdot \text{bufferNum} + \text{isAligned}$$
2. **Prune Unreachable Template Instantiations**:
   - Register only reachable combinations in the template selector table so `bisheng` does not compile hundreds of dead kernel instantiations (avoiding build timeouts and binary bloat).
   - On Device, use `GET_TILING_DATA_WITH_STRUCT(MyTilingData, tilingData, tiling)` inside specialized template functions so each branch binds only its required tiling struct fields.

### Step 3: UB Budget Equation & High-Level API Temp Sizing (`double_buffer_slot_counts_and_resident_parameters`)
1. **Exact UB Accounting Formula**:
   - Before choosing `tileLength` and `BUFFER_NUM` ($\in \{1, 2\}$), compute total UB consumption:
     $$\text{UB}_{\text{total}} = \sum_{q \in \text{Queues}} \text{bufferNum}_q \cdot \operatorname{AlignUp32}(L_{\text{tile}} \cdot \text{sizeof}(T_q)) + \sum_{r \in \text{Resident TBuf}} \operatorname{AlignUp32}(B_r) + \max_{s \in \text{Stages}} \operatorname{AlignUp32}(\text{tmpSize}_s)$$
   - Query high-level API scratch requirements on Host via `GetSoftMaxMaxMinTmpSize`, `GetBroadCastMaxMinTmpSize`, `GetAscendDequantMaxMinTmpSize`, etc., and reserve `UB_safe = ubSize - 2048` (leaving 2 KB headroom for compiler/system stack).
2. **Dynamic Tile-Length Down-Scaling & Buffer-Num Fallback (`P2`, `P8`)**:
   - If $\text{UB}_{\text{total}}(L_{\text{tile}}, \text{bufNum}=2) > \text{UB}_{\text{safe}}$, halve $L_{\text{tile}}$ iteratively down to minimum alignment (`32B` / `256B`); if a single row $D$ still exceeds double-buffered UB, fall back to `BUFFER_NUM = 1` or column-tiled two-pass execution.

### Step 4: Workspace Sizing & Multi-Core BlockDim Determination
1. **System + User Workspace Summation**:
   - Always compute `size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();` (or `MatmulGetTmpBufSize` when using `MatmulImpl` workspace) and set:
     ```cpp
     size_t* workspaces = context->GetWorkspaceSizes(1);
     workspaces[0] = sysWorkspaceSize + AlignUp512(userWorkspaceBytes);
     ```
   - On Device, offset user workspace pointers past `GetSysWorkSpacePtr()` (or use `GetUserWorkspace(workspace)`).
2. **Trim `BlockDim` to Actual Active Cores**:
   - Set `context->SetBlockDim(actualActiveCores)` where `actualActiveCores = min(maxPhysicalCores, totalTasks)`. Never launch 48 AIV cores when `totalTasks == 4` unless unused cores are explicitly guarded by `if (GetBlockIdx() >= activeCores) return;` at kernel entry.

## Verification & Checks

1. **Physical Storage Shape vs Logical View Shape (`561103`)**:
   - In `TilingFunc`, inspect both `GetStorageShape()` and `GetOriginShape()`; never assume a transposed or sliced view tensor has contiguous physical storage matching its logical view dimensions.
2. **32B Alignment of Every `InitBuffer` Call**:
   - Verify that every byte count passed to `pipe.InitBuffer(queOrBuf, num, lenBytes)` satisfies `lenBytes > 0 && lenBytes % 32 == 0`.

## Failure Modes & Fallbacks

- **Symptom**: `ACL_ERROR_RT_PARAM_INVALID` or silent no-op when launching kernel on a shape not covered by `TILING_KEY_IS`.
  - **Root Cause**: Host `TilingFunc` emits a `TilingKey` value that has no matching `if (TILING_KEY_IS(...))` branch in `op_kernel/*.cpp`, causing the kernel entry to return without writing outputs!
  - **Fix**: Return `ge::GRAPH_FAILED` from `TilingFunc` if no valid `TilingKey` matches, and assert exhaustive `TilingKey` coverage between Host and Device.
