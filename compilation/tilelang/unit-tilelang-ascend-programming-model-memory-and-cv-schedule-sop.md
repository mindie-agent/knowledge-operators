---
category: sop
conditions:
  kernel_language: tilelang
  operator_family:
  - gemm
  - attention
  - elementwise
  - reduction
  - cv_fusion
  soc:
  - Ascend910B
  - Ascend910_93
domain_path: compilation/tilelang
evidence_strength: software_contract
id: unit-tilelang-ascend-programming-model-memory-and-cv-schedule-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-hivm-dialect-dma-macro-matmul-and-fixpipe-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-programming-model-guide/Ascend910/instructions.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-api-best-practices/Ascend910/references/api-kernel-memory.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-api-best-practices/Ascend910/references/api-compute.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-api-best-practices/Ascend910/references/api-schedule-sync.md
  source_locator: cann/cannbot-skills@f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:ops/tilelang-op-implement/SKILL.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: TileLang-Ascend Programming Model, Memory Hierarchy & CV Schedule SOP
triggers:
  error_signatures:
  - undefined Variable v_thread when using T.if_then_else inside T.Parallel
  - Illegal instruction (unaligned UUB addresses) when setting cross flag on FIX pipe
    in Vector scope
  - DataCopyPad silent data corruption on non-innermost multi-dimensional T.copy slice
  - L0C capacity overflow segmentation fault when block_M * block_N > 32768 float32
    elements
  keywords:
  - tilelang
  - TL_ASCEND_AUTO_SYNC
  - TL_ASCEND_MEMORY_PLANNING
  - TL_ASCEND_AUTO_CV_COMBINE
  - TL_ASCEND_AUTO_CV_SYNC
  - T.alloc_shared
  - T.alloc_fragment
  - T.gemm_v0
  - T.Parallel
  - DataCopyPad
  - set_cross_flag
---
# TileLang-Ascend Programming Model, Memory Hierarchy & CV Schedule SOP

## 1. Developer vs. Expert vs. Hybrid Programming Modes

TileLang-Ascend provides three programming abstractions over Ascend NPU Cube (AIC) and Vector (AIV) pipelines:

| Dimension | Developer Mode | Expert Mode | Hybrid Mode (Recommended Default) |
| :--- | :--- | :--- | :--- |
| **Memory Allocation** | `T.alloc_shared` (inferred as L1 or UB) / `T.alloc_fragment` (inferred as L0A/L0B/L0C) | Explicit `T.alloc_L1`, `T.alloc_ub`, `T.alloc_L0A`, `T.alloc_L0B`, `T.alloc_L0C` | `T.alloc_shared` / `T.alloc_fragment` |
| **Compute Primitives** | `T.Parallel` + symbolic expressions + `T.gemm_v0` / `T.reduce_*` | Explicit `T.tile.xxx` buffer-level SIMD + `T.mma` / `T.gemm_v0` | Developer structure + Ascend-specific `T.tile.xxx` primitives |
| **AIC/AIV Scope** | Compiler-separated (`TL_ASCEND_AUTO_CV_COMBINE: True`) | Manual `with T.Scope("C"):` / `with T.Scope("V"):` | Automatic (`AUTO_CV_COMBINE: True`) or manual `T.Scope` (`AUTO_CV_COMBINE: False`) |
| **Intra/Inter-Core Sync** | Compiler-inserted (`TL_ASCEND_AUTO_SYNC` + `TL_ASCEND_AUTO_CV_SYNC`) | Manual `T.barrier_all`, `T.set_flag`/`T.wait_flag`, `T.set_cross_flag`/`T.wait_cross_flag` | Automatic via `pass_configs` |
| **CV Data Handoff** | On-chip direct `T.copy(buf_c, buf_v)` with `threads=2` (compiler synthesizes GM staging) | Explicit GM `workspace_idx=[...]` + manual `vid` halving (`block_M // 2`) | Prefer `threads=2` on-chip syntax; fall back to explicit `workspace` for complex multi-stage custom pipelines |

### 1.1 The Four Ascend `pass_configs` Switches

```python
import tilelang

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # "tl.ascend_auto_sync": auto intra-core barrier/event insertion
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # "tl.ascend_memory_planning": lifetime-based on-chip buffer reuse
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # "tl.ascend_auto_cv_combine": auto Cube/Vector scope splitting
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,     # "tl.ascend_auto_cross_core_sync": auto set_cross_flag/wait_cross_flag
}
```

- **Pure Vector Operators** (elementwise, softmax, norm): Enable `TL_ASCEND_AUTO_SYNC: True` and `TL_ASCEND_MEMORY_PLANNING: True`; omit `AUTO_CV_COMBINE` and `AUTO_CV_SYNC`.
- **Developer GEMM & CV Fusion Operators** (FlashAttention, Dequantize-GEMM): Enable all four switches (`True`). Never mix `TL_ASCEND_AUTO_CV_COMBINE: True` with manual `with T.Scope("C"/"V"):`.
- **Developer Mode `workspace` & `vid` Elimination (`threads=2` → `vid` removal → `workspace` removal)**:
  1. Replace `T.Kernel(block_num, is_npu=True) as (cid, vid)` with `T.Kernel(block_num, threads=2, is_npu=True) as (cid)` to auto-parallelize across the 2 Vector cores per Cube core.
  2. Remove `workspace_idx=[...]` from `@tilelang.jit` and `workspace_*` args from `@T.prim_func`.
  3. Drop `vid * (block_M // 2)` offsets and merge `L0C -> workspace -> UB` into a single `T.copy(C_L0, c_ub)`.

---

## 2. On-Chip Memory Capacities, Alignment & Fractal Constraints

### 2.1 Ascend 910B / 910C Storage Hierarchy & Alignment

| Memory Level | Developer API | Expert API | Capacity (Bytes) | Alignment | Valid `T.copy` Paths |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Memory (GM)** | `T.Tensor` | `T.Tensor` | HBM | Unaligned (`DataCopyPad`) | `GM↔L1`, `GM↔UB`, `L0C→GM` |
| **Unified Buffer (UB)** | `T.alloc_shared` | `T.alloc_ub` | `196,608` (192 KB) | `32 B` | `GM↔UB`, `UB↔UB` (no `UB↔L1`) |
| **L1 Buffer** | `T.alloc_shared` | `T.alloc_L1` | `524,288` (512 KB) | `32 B` | `GM→L1`, `L1→L0A`, `L1→L0B` |
| **L0A / L0B** | Internal (`gemm_v0`) | `T.alloc_L0A`/`L0B` | `65,536` (64 KB) each | `512 B` | `L1→L0A`, `L1→L0B` |
| **L0C** | `T.alloc_fragment` | `T.alloc_L0C` | `131,072` (128 KB) | `64 B` | `L0C→GM` (FixPipe) |

- **L0C Capacity Trap**: `block_M * block_N * sizeof(accum_dtype) > 131072` (`> 32768` FP32 elements, e.g., `256 x 256`) causes a silent runtime segfault. Cap `block_M * block_N <= 32768` for FP32 accumulators.
- **Cube Fractal Minimum Tile Sizes (`T.gemm_v0` / `T.mma`)**:
  - `L0A`: `16 * (32B / sizeof(dtype))` (`512B`); `L0B`: `(32B / sizeof(dtype)) * 16` (`512B`); `L0C`: fixed `16 x 16` (`256` elements).
  - Minimum `[block_M, block_K, block_N]`: `int8`/`uint8` requires `[16, 32, 16]`; `fp16`/`bf16` requires `[16, 16, 16]`; `int32`/`fp32` requires `[16, 8, 16]`.

### 2.2 `T.copy` Dynamic Tail Slicing & `DataCopyPad` Stride Invariant

1. **Dynamic Tail `BufferRegion` Slicing (No Host Zero-Padding)**:
   ```python
   valid_m, valid_n = T.min(block_M, M - m_start), T.min(block_N, N - n_start)
   T.copy(X[b, m_start:m_start + valid_m, n_start:n_start + valid_n], a_ub[0:valid_m, 0:valid_n])
   T.copy(b_ub[0:valid_n, 0:valid_m], Y[b, n_start:n_start + valid_n, m_start:m_start + valid_m])
   ```
2. **Hardware Limitation of `T.copy` Multi-Dimensional Slicing**:
   - `T.copy` lowers to AscendC `DataCopyPad`, which requires **contiguous elements within each row** and supports only inter-row stride (`srcGap`/`dstGap`).
   - If the last sliced dimension with `extent > 1` is **not** the physical innermost dimension of the GM tensor (e.g., `x[b, m:m+M, n:n+N, k:k+1]` on `[B, M, N, K]` with `K > 1`), column elements are strided by `K`, causing **silent data corruption**.
   - **Fix**: Loop over non-contiguous axes only when the inner `T.copy` slice covers the full innermost axis (`:`), apply a compatible zero-copy metadata view, or block-DMA contiguous chunks into UB before local reorder.

---

## 3. Compute Primitives, `T.Parallel` Constraints & Pipeline Synchronization

### 3.1 `T.Parallel` SIMD Lowering Rules vs. `T.tile.xxx` Primitives

`T.Parallel` lowers to buffer-level SIMD instructions (`ascend_lower_parallel`), **not** SIMT per-thread execution:
- **Supported Loops**: 1D `T.Parallel(N)`, 2D `T.Parallel(M, N)`, or `for i in range(M): for j in T.Parallel(N):` with compile-time constant extents (`IntImm`). 3D+ `T.Parallel` is rejected at compile time.
- **Forbidden Inside `T.Parallel`**:
  - Per-element `if-else` / `T.if_then_else(...)`: Fails compilation (`undefined Variable v_thread`) or degrades the tile loop into a serial scalar loop. Replace with branchless math (e.g., `max(x, 0) + ln(1 + exp(-|x|))`) or buffer-level `T.tile.compare(mask_ub, a_ub, thresh, mode)` + `T.tile.select(dst_ub, mask_ub, true_ub, false_ub, "VSEL_CMPMASK_SPR")`.
  - Per-element cast/reinterpret (`T.int8(expr)`, `tir.reinterpret`): Use buffer-level `T.tile.cast(dst_ub, src_ub, mode, count)` or `T.reinterpretcast`.
- **Key `T.tile.xxx` Constraints**:
  - `T.tile.compare` requires **256B-aligned** byte count (`64` FP32 / `128` FP16 elements). On `int32`, only `"EQ"` works (`LT/LE/GT/GE` degrade to `vcmpv_eq`); cast `int32 -> float32` before relational compare.
  - `T.tile.broadcast(dst_2d, src_2d)` requires 2D source `[M, 1]` or `[1, N]` matching `[M, N]` (1D `[M]` raises dimension mismatch).
  - Prefer fused compound ops: `T.tile.mul_add_dst(acc, x, w)`, `T.tile.axpy(dst, src, scale)`, `T.tile.relu(out, x)`, and `T.tile.rsqrt(inv, x)`.

### 3.2 Scheduling (`T.Pipelined`, `T.Persistent`) & Cross-Core Flag Pipes

- **Intra-Core vs. Inter-Core `T.Pipelined(range, num_stages=N)`**: Mutually exclusive on the same loop. Inter-core CV pipelining requires `TL_ASCEND_AUTO_CV_COMBINE: True` and `TL_ASCEND_AUTO_CV_SYNC: True`.
- **Wave Scheduling**: Use `for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], core_num, cid):` or launch `min(block_num, 24)` physical cores to bound workspace within L2 (`~201 MB`).
- **Manual Cross-Core Sync Pipe Rule**: In `T.set_cross_flag(pipe, flag)`, Vector cores (`UB -> GM`) must use `"MTE3"` and Cube cores (`L0C -> GM`) must use `"FIX"`. Using `"FIX"` in Vector scope triggers `Illegal instruction (unaligned UUB addresses)`.
