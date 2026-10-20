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
domain_path: compilation/bisheng-ir/hfusion-and-hacc
evidence_strength: software_contract
id: unit-hfusion-hacc-and-triton-custom-dialects-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-bisheng-ir-compilation-pipeline-and-pass-lowering-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hfusion/overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/hacc/overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/other_dialects/index.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/triton_dialects/overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ascend_ext_overview.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/apis/ir/hfusion/overview.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'HFusion, HACC, Auxiliary Dialects and Triton-Ascend Extension SOP: fusion
  scheduling, host-device binding, scope/annotation/dpx and ascend_ext APIs'
triggers:
  error_signatures:
  - hfusion-op-fusion fails to fuse elementwise and reduction ops or spills intermediate
    tensors to GM
  - hacc.arg_type or hacc.tiling_function binding mismatch causes host-device kernel
    launch marshaling failure
  - scope.scope @aic and @aiv partitioning fails during SplitMixKernel
  - al.fixpipe, al.gather_out_to_ub or bl.subview fails lowering due to SOC or 32-byte
    alignment constraints
  keywords:
  - hfusion
  - hacc
  - hacc.function_kind
  - hacc.target_device_spec
  - scope.scope
  - annotation.mark
  - memref_ext.alloc_workspace
  - ascend_dpx
  - triton.language.extra.cann.extension
  - triton.extension.buffer.language
  - compile_hint
  - index_select_simd
---
# HFusion, HACC, Auxiliary Dialects and Triton-Ascend Extension SOP: fusion scheduling, host-device binding, scope/annotation/dpx and ascend_ext APIs

## 1. HFusion Dialect (`hfusion.*`) Operations and Fusion Transforms

HFusion (`::mlir::hfusion`) is the tensor-level structured fusion layer between Linalg/TTAdapter and HIVM. Its 42 operations follow Linalg Destination-Passing Style (`ins(...) outs(...)`):

- **Elementwise & Cast (6 ops)**: `hfusion.elemwise_unary` (`unary_fn = <sqrt|rsqrt|exp|log|abs|relu|...>`), `hfusion.elemwise_binary`, `hfusion.compare`, `hfusion.select`, `hfusion.cast` (with rounding modes), `hfusion.bitcast`.
- **Reduction & Scan (3 ops)**: `hfusion.reduce_with_index` (`<max|min>`, returning both value and index tensors), `hfusion.cumsum`, `hfusion.cumprod`.
- **Matrix Multiply (2 ops)**: `hfusion.matmul_mx` (microscaling FP8/FP4 block-scaled matmul with `$sa`, `$sb` scale tensors), `hfusion.group_matmul` (MoE grouped GEMM).
- **Data Movement & Memory (20 ops)**: `load`, `store`, `broadcast`, `transpose`, `concat`, `pad`, `interleave`, `deinterleave`, `flip`, `gather`, `arange`; `gather_load`, `scatter_store`, `indirect_load`, `indirect_store`, `gatherT`, `index_put`, `scatterT`, `atomic_cas`, `atomic_xchg`.
- **Special Ops (11 ops)**: `print`, `assert`, `barrier`, `sort`, `histogram`, `embedding_gather`, `is_inf`, `is_nan`, `is_finite`, `mulext`, `symbolic_dim`.
- **Key HFusion Passes**:
  - `InferFuncFusionKind` + `HFusionOpFusion` + `AutoSchedule`: Classifies functions into `PURE_ELEMWISE`, `LAST_AXIS_PBR` (pointwise-broadcast-reduction), ` SHALLOW_CV`, or `ANY_PB`, then tiles and fuses producer-consumer chains into a single loop nest before bufferization (`CacheIO`/`ReCacheIO`).
  - `OutlineVectorFunction` + `MergeVecScope`: Extracts vector subgraphs inside mixed kernels into `@aiv` vector functions so Cube and Vector schedules can be optimized independently.

## 2. HACC Dialect (`hacc.*`) Host-Device ABI and Target Specification

HACC (`::mlir::hacc`) annotates `func.func` and kernel parameters with host-device execution contracts and hardware DLTI entries (`#hacc.target_device_spec<...>`):

1. **Function Kind (`#hacc.function_kind<HOST|DEVICE>`)**: Separates CPU host functions (`tiling_function`, `infer_output_shape`, `infer_workspace_size`, `get_tiling_struct_size`) from NPU device entry kernels.
2. **Kernel Argument Classification (`#hacc.arg_type<...>`)**: Every device kernel parameter carries one of 12 roles (`input`, `output`, `input_and_output`, `workspace`, `tiling_data`, `tiling_key`, `ffts_base_addr`, `overflow_status`, etc.) so the CANN runtime marshals pointers and scalar tiling structs in exact ABI order.
3. **Target Device Spec (`DeviceSpecEnum`)**: Encodes 14 hardware constants in the module header (`aic_core_num`, `aiv_core_num`, `ub_size`, `l1_size`, `l0a_size`, `l0b_size`, `l0c_size`, `ub_align_size = 32`, `l1_align_size = 512`, etc.) consumed by `PlanMemory` and `AutoSchedule`.

## 3. Auxiliary Dialects (`scope`, `annotation`, `symbol`, `memref_ext`, `mathext`, `hmap`, `ascend_dpx`)

| Dialect | Key Operations / Attributes | Role in Lowering Pipeline |
| --- | --- | --- |
| **`scope`** | `scope.scope` (`tcore_type = #hivm.tcore_type<CUBE\|VECTOR>`) | Delimits Cube (`@aic`) vs Vector (`@aiv`) regions inside a mixed function prior to `SplitMixKernel` |
| **`annotation`** | `annotation.mark %val {key = val}` | Passes user/compiler hints (`multi_buffer`, `buffer_size`, `stride_align`) to `PlanMemory` and `EnableMultiBuffer` |
| **`symbol`** | `symbol.symbolic_int`, `symbol.bind_symbolic_shape` | Preserves dynamic shape relationships (`S0`, `S1`) across Linalg/HFusion passes until `DropSymbols` |
| **`memref_ext`** | `memref_ext.alloc_workspace` | Allocates inter-core GM workspace buffers bound to `#hacc.arg_type<workspace>` by `hivm-bind-workspace-arg` |
| **`mathext`** | `mathext.ilogb`, `mathext.ldexp`, `mathext.divfhp` | NPU high-precision floating-point intrinsics |
| **`hmap`** | `hmap.all_to_allv` | Multi-NPU collective communication primitive for uneven MoE token dispatch |
| **`ascend_dpx`** | Thread index, warp shuffle/reduce, SIMT atomics | Final virtual SIMT ISA before LLVM lowering on Ascend 950 (`GPUToDPX`) |

## 4. Triton-Ascend Python Extension APIs (`al.*` and `bl.*`)

When standard `tl.*` primitives cannot express explicit Cube-Vector pipelining, UB buffer reuse, or SIMD indexed DMA, import the Ascend extension modules:

```python
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
from triton.language.extra.cann import libdevice
```

- **Cube-Vector Scope & Cross-Core Sync**:
  - `with al.scope(core="cube"): ...` and `with al.scope(core="vector"): ...` lower to `scope.scope`.
  - `al.sync_block_set(sender="cube", receiver="vector", event_id=0)` / `al.sync_block_wait(...)` / `al.sync_block_all()` lower via `TritonToHIVM` to `hivm.hir.sync_block_set`/`wait`/`sync_block`.
  - `al.sub_vec_id()` (`0` or `1`) and `al.sub_vec_num()` (`2`) partition Vector work across the 2 AIV sub-cores paired with each AIC Cube core (`al.parallel(..., bind_sub_block=True)`).
- **Explicit Memory, Fixpipe & SIMD Gather/Scatter**:
  - `al.fixpipe(src_l0c, dst_ub, dma_mode=al.FixpipeDMAMode.NZ2ND, pre_quant=..., pre_relu=...)`: Direct L0C $\to$ UB transfer (**Ascend 950 only**).
  - `al.index_select_simd(src_ptr, dim, index, shape, ...)`: Zero-copy GM $\to$ UB SIMD row gather along `dim`, avoiding discrete scalar load loops when the trailing dimensions after `dim` are contiguous and 32 B aligned.
  - `al.gather_out_to_ub` / `al.scatter_ub_to_out` / `al.index_put`: Lower to `hivm.hir.gather_load` / `scatter_store`.
  - `al.get_element(tensor, [idx, ...])` and `al.insert_slice(ful, sub, offsets, sizes, strides)` / `al.extract_slice(...)`: Explicit sub-tensor slicing without full-tile `tl.where` masks.
  - `bl.alloc(shape, dtype, address_space)` + `bl.to_buffer(tensor)` / `bl.to_tensor(buf)` / `bl.subview(...)`: Direct buffer manipulation in UB/L1; `bl.alloc` forbids `int1` and `bl.subview` requires 32 B aligned byte offsets.
