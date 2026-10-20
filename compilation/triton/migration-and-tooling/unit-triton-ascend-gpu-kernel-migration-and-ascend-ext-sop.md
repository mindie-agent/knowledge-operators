---
category: sop
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/migration-and-tooling
evidence_strength: software_contract
id: unit-triton-ascend-gpu-kernel-migration-and-ascend-ext-sop
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-triton-ascend-operator-domain-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ascend_ext_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/migration/flash_attention_migration.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/migration/fused_matmul_migration.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/migration/rope_migration.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/migration/fused_swiglu_migration.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/migration/softcap_migration.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/apis/ascend_ext_overview.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Triton GPU-to-Ascend Kernel Migration Patterns and Ascend Extension API SOP
triggers:
  error_signatures:
  - 1. Ten-Step GPU-to-Ascend Kernel Migration Checklist
  - 2. Ascend Compiler Launch Kwargs Selection Matrix
  - 3. Ascend Extension API (`al.*` & `bl.*`) Reference SOP
  - 3.1 Cube-Vector Explicit Scope & Cross-Core Synchronization
  keywords:
  - unit-triton-ascend-gpu-kernel-migration-and-ascend-ext-sop
---
# Triton GPU-to-Ascend Kernel Migration Patterns and Ascend Extension API SOP

## 1. Ten-Step GPU-to-Ascend Kernel Migration Checklist

When porting a GPU/CUDA Triton kernel to Ascend NPU (`910B`, `910_93`, `910_95`), execute these 10 structural transformations in order:

1. **Device & Library Imports**:
   - Replace `cuda` with `npu`, `torch.cuda.synchronize()` with `torch.npu.synchronize()`, and `from triton.language.extra import libdevice` with `from triton.language.extra.cann import libdevice`.
2. **Autotune Config Cleanup**:
   - Remove `num_warps`, `num_stages`, `cache_modifier`, and `evict_policy` from `triton.Config`. Restrict `BLOCK_*` candidates to powers of 2 (e.g., `32, 64, 128` for Cube; `512..8192` for Vector) to guarantee 32-byte DMA alignment.
3. **3D/2D Launch Grid -> 1D Persistent Core-Clamped Loop**:
   - Replace `grid = (cdiv(seq, BM), num_heads, batch)` with `grid = (AICORE_NUM,)` (or `VECTOR_NUM` for pure vector kernels) and decode `(start_b, start_h, start_m)` inside `for block_idx in range(pid, NUM_BLOCKS, AICORE_NUM):`.
4. **Loop Early-Exit Elimination**:
   - Convert `if start_m * BLOCK_M >= seq_len: return` (or `continue`) inside the persistent loop into a positive `if start_m * BLOCK_M < seq_len:` guard block.
5. **Block Pointer Dimensionality Expansion (2D -> 3D)**:
   - In multi-head vector kernels like `RoPE`, merge the `head` axis into `tl.make_block_ptr` (`shape=(len, head, DIM)`, `block_shape=(BLOCK_M, head, DIM // 2)`, `order=(2, 1, 0)`) with `head: tl.constexpr` so a single DMA burst loads all heads, while keeping total tile bytes within UB limits (`<= 140 KB` on 910B/A3, `<= 200 KB` on 910_95).
6. **Pointer Increment Chain -> Base + Offset Re-Calculation**:
   - Rewrite GPU loop-carried pointer updates (`a_ptrs += BLOCK_K`) into loop-invariant base pointers (`a_base = a_ptr + offs_m[:, None] * K + offs_k[None, :]`) plus loop-indexed offsets (`a_ptrs = a_base + k * BLOCK_K`), and remove `% M` modulo wrapping in favor of explicit `offs_m < M` predicates.
7. **Cube Bias Pre-Loading & Explicit Broadcast**:
   - In Fused MatMul, load `bias` **before** the `K` loop and add `tl.broadcast_to(b[None, :], (BLOCK_M, BLOCK_N))` to the `fp32` accumulator so bias addition fuses into the Cube pipeline without Vector round-trips.
8. **CV Pipelining Reordering in Attention**:
   - Move `v = load_if(v_block_ptr, ...)` and `pv = tl.dot(p.to(dtype), v)` **ahead** of vector reductions (`p_sum = tl.sum(p, 1)` and `alpha = tl.math.exp(m - m_new)`), then combine `acc = acc * alpha[:, None] + pv` so Cube `MMAD` overlaps with Vector `PIPE_V`.
9. **Online Softmax `propagate_nan` & Precomputed Mask**:
   - Add `propagate_nan=True` to `tl.max(s, 1, propagate_nan=True)` and `propagate_nan=tl.PropagateNan.ALL` to `tl.maximum(m, ...)`. Replace complex in-kernel dynamic mask generators with precomputed `[batch, max_q, max_k]` mask tensors loaded via `tl.make_block_ptr`.
10. **Fused Backward QKV & `break_vf` Hints**:
    - When fusing `bwd_kv` and `bwd_q` into `bwd_qkv_kernel`, initialize `dq` in `torch.float32` with `reset_to_zero=["dq_ptr"]`, accumulate `dq` via vector `tl.atomic_add`, and insert `al.compile_hint(ds, "break_vf")` before `tl.dot(ds, k)` to split overly long Vector fusion chains that would overflow UB.

---

## 2. Ascend Compiler Launch Kwargs Selection Matrix

Select launch/config kwargs strictly according to the kernel's execution unit profile:

| Kernel Profile | Required Ascend Compiler Launch Kwargs | Rationale |
|---|---|---|
| **CV Mixed Kernel** (contains `tl.dot` + Vector post/softmax) | `multibuffer=True`, `limit_auto_multi_buffer_of_local_buffer="no-limit"`, `enable_mixed_cv=True` (910_95), `enable_auto_bind_sub_block=True`, `sync_solver=True`, `enable_flatten=False`, `set_workspace_multibuffer=2` | Binds 1 Cube + 2 Sub-Vector cores, double-buffers cross-core workspace, and prevents `enable_flatten` from collapsing 2D Cube tiles. (Exception: Fused MatMul fwd with Cube-fused bias can set `enable_flatten=True` for the trailing cast/store.) |
| **Pure Cube Kernel** (`bwd_x` / `bwd_w` GEMM without Vector fusion) | `enable_auto_bind_sub_block=False`, `multibuffer=True` | Avoids unnecessary sub-vector core binding overhead. |
| **Pure Vector Kernel** (elementwise, norm, rope, `bwd_preprocess`) | `multibuffer=True`, `limit_auto_multi_buffer_of_local_buffer="no-limit"` (verify per operator; omit on tiny single-pass kernels) | Enables MTE2/MTE3 ping-pong in UB without Cube workspace allocation. |

> **UB Overflow Fallback Sequence**: If compilation fails with `ub overflow`, back off in this exact order:
> 1. Reduce `BLOCK_M` / `BLOCK_N` / `BLOCK_K`.
> 2. Reduce `set_workspace_multibuffer` from `2` to `1`.
> 3. Ensure `limit_auto_multi_buffer_of_local_buffer="no-limit"` is set.
> 4. Insert `al.compile_hint(tensor, "break_vf")` between heavy vector stages and `tl.dot`.

---

## 3. Ascend Extension API (`al.*` & `bl.*`) Reference SOP

```python
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
from triton.language.extra.cann import libdevice
```

### 3.1 Cube-Vector Explicit Scope & Cross-Core Synchronization
- **`al.scope(core_type=...)` & Sub-Vector Partitioning**: Partition explicit Cube and Vector blocks when manual orchestration is required, using `al.sub_vec_id()` (`0` or `1`) and `al.sub_vec_num()` (`2`) or `al.parallel(..., bind_sub_block=True)`.
- **`al.sync_block_set` / `al.sync_block_wait` / `al.sync_block_all`**: Producer core signals completion via `sync_block_set`; consumer core waits via `sync_block_wait`. Inside Vector VF blocks, `al.debug_barrier` inserts `SYNC_IN_VF` (`VV_ALL`, `VST_VLD`) pipeline barriers.

### 3.2 `al.fixpipe` (Direct `L0C -> UB` on Ascend 910_95)
- Moves Cube accumulator tiles directly from `L0C` to `UB` (on `910_95`) or `GM` with hardware format conversion (`FixpipeDMAMode.NZ2ND` / `NZ2DN` / `NZ2NZ`), pre-quantization (`FixpipePreQuantMode.F322F16` / `F322BF16` / `S322I8`), and fused ReLU (`FixpipePreReluMode`).

### 3.3 Vector Slice, Cast & SIMD Gather/Scatter Extensions
- **Sub-Tensor Manipulation**: `al.extract_slice`, `al.insert_slice`, `al.get_element`, `al.sort` (last dim only), `al.flip`.
- **Hardware Saturation Cast**: `al.cast(x, tl.int8, overflow_mode="saturate")` replaces multi-instruction `tl.maximum(tl.minimum(...))` clamping.
- **High-Throughput UB Indexed Memory Ops**: `al.gather_out_to_ub` (GM -> UB indexed read), `al.index_select_simd` (zero-copy SIMD row gather GM -> UB), `al.scatter_ub_to_out` / `al.index_put` (UB -> GM indexed write).
- **Buffer Programming Model (`bl.*`)**: `bl.alloc(shape, dtype, address_space)` allocates explicit `UB`/`L1`/`L0C` buffers; `bl.to_buffer` and `bl.to_tensor` provide zero-copy views; `bl.subview` slices buffers on 32-byte aligned boundaries.
