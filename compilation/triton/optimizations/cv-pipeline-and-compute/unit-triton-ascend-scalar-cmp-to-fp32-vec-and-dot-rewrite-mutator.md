---
category: mutator
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/optimizations/cv-pipeline-and-compute
evidence_strength: reproduced_empirical
id: unit-triton-ascend-scalar-cmp-to-fp32-vec-and-dot-rewrite-mutator
inherits_from:
- unit-triton-ascend-cv-pipeline-vector-degradation-and-dot-workflow
- unit-triton-ascend-hardware-and-compiler-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/vec_cmp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/dot_matmul_rewrite.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/redundant_boundary_operation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/mask_only_affine_addressing.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/optimizations/patterns/vec_cmp.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L4
title: Triton-Ascend Integer-to-FP32 Vector Compare, Manual-Loop-to-Dot, and Redundant
  Boundary Mutator
triggers:
  error_signatures:
  - '1. Mutator A: Named Integer Relational Compare -> `fp32` Vector Compare (`vec_cmp`)'
  - 1.1 Trigger Signals & Safety Guardrails
  - 1.2 Before / After Code Mutation
  - '2. Mutator B: Manual $K$-Reduction Loop -> Tiled Cube `tl.dot` (`dot_matmul_rewrite`)'
  keywords:
  - unit-triton-ascend-scalar-cmp-to-fp32-vec-and-dot-rewrite-mutator
---
# Triton-Ascend Integer-to-FP32 Vector Compare, Manual-Loop-to-Dot, and Redundant Boundary Mutator

## 1. Mutator A: Named Integer Relational Compare -> `fp32` Vector Compare (`vec_cmp`)

### 1.1 Trigger Signals & Safety Guardrails
- **Trigger**: An explicit `int32` or `int64` relational comparison (`<, >, <=, >=`) is assigned to a named boolean tensor that feeds `tl.where(cond, a, b)` or compound mask arithmetic on a Vector Core path, and `hivm` IR shows `ShouldLowerToScalarLoops` generating a scalar `scf.for` loop.
- **Safety Guardrails**:
  1. Both integer operands must lie strictly within the exact integer range of IEEE_754 `fp32`: **`[-16777216, 16777216]` (`[-2^24, 2^24]`)**.
  2. **Leave inline `tl.load`/`tl.store` `mask=` arguments untouched** (`tl.load(ptr + offs, mask=offs < N)` already takes the compiler's hardware MTE mask fast path).
  3. **Never apply to Cube `tl.load`/`tl.store` survival masks** (e.g., `row_hi` masks in `GroupedMatmulSwigluQuant`, where `fp32`-derived MTE descriptors hang the AI Core with `507014`).

### 1.2 Before / After Code Mutation

```python
# ❌ BEFORE: Named i32/i64 relational comparison feeding tl.where lowers to scalar loop on PIPE_S
cols = tl.arange(0, BLOCK_N).to(tl.int32)
in_window = (cols >= win_start) & (cols < win_end)
x_centered = tl.where(in_window, x - mean, 0.0)

# ✅ AFTER: Cast operands in [-2^24, 2^24] to tl.float32 so VCmp executes as 256B SIMD on PIPE_V
cols_f32 = tl.arange(0, BLOCK_N).to(tl.float32)
win_start_f32 = win_start.to(tl.float32)
win_end_f32 = win_end.to(tl.float32)
in_window = (cols_f32 >= win_start_f32) & (cols_f32 < win_end_f32)
x_centered = tl.where(in_window, x - mean, 0.0)
```

---

## 2. Mutator B: Manual $K$-Reduction Loop -> Tiled Cube `tl.dot` (`dot_matmul_rewrite`)

### 2.1 Trigger Signals
- **AST Signal**: A kernel computes $C_{m,n} = \sum_k A_{m,k} B_{k,n}$ (or attention $Q K^T$) using a per-dimension scalar loop (`for d in range(D): scores += q[:, None] * k[None, :]`) or row-wise vector `tl.sum(a * b, axis=-1)`, leaving Cube utilization `aic_mac_ratio < 10%`.
- **Guardrail**: Ensure effective `M >= 16` (or head group `GH >= 8` padded to `16`); keep `M == 1` on the Vector reduction path.

### 2.2 Before / After Code Mutation

```python
# ❌ BEFORE: Manual K-reduction loop runs entirely on Vector Core (Cube idle, 0.05x-0.2x speedup)
@triton.jit
def manual_k_reduce_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K):
        a_col = tl.load(A_ptr + offs_m * stride_am + k * stride_ak, mask=offs_m < M, other=0.0)
        b_row = tl.load(B_ptr + k * stride_bk + offs_n * stride_bn, mask=offs_n < N, other=0.0)
        acc += a_col[:, None].to(tl.float32) * b_row[None, :].to(tl.float32)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

```python
# ✅ AFTER: Tiled Cube tl.dot with matching native operand dtypes + fp32 accumulator (>10x speedup)
@triton.jit
def tiled_cube_dot_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

    a_base = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_base = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_valid = (offs_k + k) < K
        a = tl.load(a_base + k * stride_ak, mask=(offs_m[:, None] < M) & k_valid[None, :], other=0.0).to(A_ptr.dtype.element_ty)
        b = tl.load(b_base + k * stride_bk, mask=k_valid[:, None] & (offs_n[None, :] < N), other=0.0).to(B_ptr.dtype.element_ty)
        acc = tl.dot(a, b, acc)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(C_ptr.dtype.element_ty), mask=c_mask)
```

---

## 3. Mutator C: Redundant Boundary `tl.where` & Index Clamp Deletion (`OPT_26`, `mask_only_affine_addressing`)

### 3.1 Trigger Signals
1. **`OPT_26` (Redundant Post-Load Reset)**: `x = tl.load(ptr + offs, mask=mask, other=0.0)` already sets out-of-bounds lanes to `0.0`, yet subsequent zero-preserving math (`y = x * scale` or `acc += tl.where(mask, x, 0.0)`) applies a second `tl.where(mask, ..., 0.0)` before reduction or masked store.
2. **`mask_only_affine_addressing` (Redundant Pointer Clamp)**: A masked load clamps its pointer offset (`safe_offs = tl.minimum(offs, N - 1); x = tl.load(ptr + safe_offs, mask=offs < N, other=0.0)`), breaking provable affine pointer analysis in `bishengir` and injecting unnecessary vector `VMin` instructions.

### 3.2 Before / After Code Mutation

```python
# ❌ BEFORE: Clamped non-affine pointer + redundant post-load tl.where
safe_offs = tl.minimum(tl.maximum(offs, 0), N - 1)
x = tl.load(x_ptr + safe_offs, mask=mask, other=0.0).to(tl.float32)
y = x * w_scalar
y_masked = tl.where(mask, y, 0.0)  # redundant: 0.0 * w_scalar == 0.0
row_sum = tl.sum(y_masked, axis=0)

# ✅ AFTER: Pure affine pointer with mask= predicate + direct zero-preserving reduction
x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
row_sum = tl.sum(x * w_scalar, axis=0)
```
> **Exception**: Never delete post-math `tl.where(mask, ...)` when the intervening operation maps `0.0` to a non-zero value (e.g., `tl.exp(0.0) == 1.0`, `tl.cos(0.0) == 1.0`, or `+ bias`).
