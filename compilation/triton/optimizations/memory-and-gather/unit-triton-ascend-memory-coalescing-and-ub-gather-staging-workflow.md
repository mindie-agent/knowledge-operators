---
category: workflow
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/optimizations/memory-and-gather
evidence_strength: reproduced_empirical
id: unit-triton-ascend-memory-coalescing-and-ub-gather-staging-workflow
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/discrete_access_ub_staging.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/window_slab_gather.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/window_2d_ub_staging.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/device_side_gather.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/continuous_copy_aggregation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/techniques/care_padding.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/optimizations/patterns/discrete_access_ub_staging.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Triton-Ascend Memory Coalescing, UB Gather Staging, Window Slab Gather, and
  Layout Optimization Workflow
triggers:
  error_signatures:
  - 1. Diagnosing Discrete Memory Access Bottlenecks (`OPT_12`, `OPT-9`)
  - 2. Stage-Then-Gather in Unified Buffer (`discrete_access_ub_staging`)
  - 2.1 Contiguous UB Preload + On-Chip `tl.gather`
  - 2.2 AoS-to-SoA Wrapper Transposition for Fixed-Channel Records
  keywords:
  - unit-triton-ascend-memory-coalescing-and-ub-gather-staging-workflow
---
# Triton-Ascend Memory Coalescing, UB Gather Staging, Window Slab Gather, and Layout Optimization Workflow

## 1. Diagnosing Discrete Memory Access Bottlenecks (`OPT_12`, `OPT-9`)

When CANN profiler shows high `aic_mte2_ratio` (`> 50%`) alongside low achieved HBM bandwidth (`< 200 GB/s`), inspect the kernel for four discrete-access anti-patterns:
1. **Indirect HBM Gather on the Hot Path**: `tl.load(x_ptr + idx_vec * stride, mask=mask)` where `idx_vec` is a non-affine vector of indices.
2. **Interleaved AoS Stride-$C$ Loads**: Reading fixed fields from `[N, 3]` or `[N, 4]` records via `tl.load(ptr + offs * 3 + c)`.
3. **Sliding-Window Redundant Reloads**: Loading `start_w + kw` from HBM inside `for kw in range(KERNEL_W):` when adjacent output windows overlap.
4. **Inner-Loop Strided Transpose Loads**: Reading column-major tiles (`stride > 1` on the innermost dimension) instead of contiguous row-major `tl.make_block_ptr` + on-chip `tl.trans`.

---

## 2. Stage-Then-Gather in Unified Buffer (`discrete_access_ub_staging`)

### 2.1 Contiguous UB Preload + On-Chip `tl.gather`
When the source lookup range `M` fits in UB (`M <= 32768` for `fp32`), replace scattered global loads with a single wide DMA transfer into UB followed by on-chip `tl.gather`:
```python
# ❌ BEFORE: Scattered HBM reads (degrades to per-lane scalar/narrow MTE2 requests)
val = tl.load(x_ptr + idx * stride_x, mask=mask, other=0.0)

# ✅ AFTER: Contiguous UB staging + on-chip tl.gather (or al.gather_out_to_ub)
x_ub = tl.load(x_ptr + tl.arange(0, BLOCK_SRC), mask=tl.arange(0, BLOCK_SRC) < M, other=0.0)
safe_idx = tl.where(mask, idx, 0).to(tl.int32)
val = tl.gather(x_ub, safe_idx, axis=0)
val = tl.where(mask, val, 0.0)
```

### 2.2 AoS-to-SoA Wrapper Transposition for Fixed-Channel Records
When an operator repeatedly accesses fixed channels from `[N, C]` (`C in {2, 3, 4}`), never stage `[N, C]` and extract scalars via `vec[0]` (unsupported on Ascend). Transpose once on the host (`coord_soa = coord.t().contiguous()`, shape `[C, N]`) so each channel loads contiguously via `tl.load(coord_ptr + c * N + offs, mask=mask)`.

---

## 3. Window Slab Gather for Pooling & Sliding-Window Reductions (`window_slab_gather`)

### 3.1 Geometry Invariants for 1D/2D/3D Sliding Windows
When a program owns `BLOCK_OW` adjacent output columns along the innermost contiguous spatial dimension $W$ with stride `STRIDE_W` and window width `KERNEL_W`:
- **Contiguous Input Slab Length (`tl.constexpr`)**:
  $$\text{W\_SLAB\_LEN} = \text{next\_pow2}(\text{STRIDE\_W} \times (\text{BLOCK\_OW} - 1) + \text{KERNEL\_W})$$
- **Slab Base Column**: `w_abs_min = ow_pid * BLOCK_OW * STRIDE_W - PAD_W`
- **On-Chip Gather Indices for Tap `kw`**: `lane_offsets + kw`, where `lane_offsets = STRIDE_W * tl.arange(0, BLOCK_OW).to(tl.int32)`

### 3.2 Slab Load + On-Chip Gather Rewrite
```python
# Load overlapping input columns for BLOCK_OW windows once per (kd, kh) row:
col = w_abs_min + tl.arange(0, W_SLAB_LEN).to(tl.int32)
slab = tl.load(x_ptr + row_base + col, mask=(col >= 0) & (col < in_w), other=0.0).to(tl.float32)
lane_offsets = STRIDE_W * tl.arange(0, BLOCK_OW).to(tl.int32)

for kw in range(KERNEL_W):
    tap_vals = tl.gather(slab, lane_offsets + kw, axis=0)
    acc += tap_vals
```
- **Guardrails**: Enforce `FULL_W_TILE` (`out_w % BLOCK_OW == 0`) on unmasked slab branches, promote `fp16`/`bf16` slabs to `tl.float32` before `tl.gather` if hardware gather faults on 16-bit types, and do **not** apply `window_slab_gather` on Ascend 950 `force_simt_only=True` kernels without A/B verification.

---

## 4. Load Instruction Reordering, `care_padding`, and Copy Aggregation

1. **Manual `tl.load` Reordering (`OPT_19`)**:
   - BiSheng IR preserves user `tl.load` order. When an expensive memory path (such as `weight` `GM -> L1 -> L0B` with 2D fractal transpose in `GroupedMatmulSwigluQuant`, or `V` tile load in `FlashAttention` / `PagedAttention`) is blocked behind earlier compute or loads, move the independent `tl.load` statement **above** the preceding compute/load so `MTE2` prefetches in parallel (`+6.2%` measured on `GroupedMatmulSwigluQuant`).
2. **`care_padding=False` on Un-Multibuffered Masked Loads**:
   - By default, `tl.load(ptr, mask=mask)` on Ascend allocates a scratch UB buffer to zero-pad masked elements (`MTE2 -> scratch -> PIPE_V`). Passing `care_padding=False` when downstream math masks out invalid lanes anyway eliminates the extra `PIPE_V` copy. (Note: do not stack `care_padding=False` onto elementwise kernels that already have `multibuffer=True`, as double buffering already hides the padding overhead.)
3. **Output Pre-Initialization & Continuous Copy Aggregation (`OPT_25`, `OPT_32`)**:
   - For operators where large regions of the output tensor take a constant default value (`Pad` with `mode='constant'`, `IndexPut` with sparse updates), call `out.fill_(default_val)` on the host and restrict the Triton kernel strictly to the active interior/source elements.
   - For `Split` / `Chunk` / `Slice` / `Cat`, aggregate adjacent contiguous sub-tensors into a single wide transfer rather than launching one kernel per slice.
