---
category: mutator
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: compilation/triton/optimizations/tiling-and-ub-budget
evidence_strength: reproduced_empirical
id: unit-triton-ascend-dimension-merge-and-ub-multibuffer-mutator
inherits_from:
- unit-triton-ascend-tiling-persistent-grid-and-ub-budget-workflow
- unit-triton-ascend-hardware-and-compiler-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/dimension_merge.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/optimizations/operator_optimization/dimension_merge.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 5
schema: mindie-graded-unit/1
tier: L4
title: Triton-Ascend Contiguous Dimension Merge, Block Size Scaling, and Multibuffer
  Mutator (`spatial_group_norm_reduce.py`)
triggers:
  error_signatures:
  - Numerical check failed on shape (2, 16, 16, 16) max_abs_err=0.046875 > 5e-3
  - '1. Mutator A: Contiguous Spatial Dimension Merge (`OPT_11`, `2.6x` Measured Speedup)'
  keywords:
  - unit-triton-ascend-dimension-merge-and-ub-multibuffer-mutator
  - spatial_group_norm_reduce
  - spatial_group_norm_reduce.py
  - run_ablation_suite
  - select_optimal_block_size
  - minus_dim_merge
  - minus_block_scaling
  - minus_multibuffer_cast_guard
  - GroupNorm
  - max_abs_err
---
# Triton-Ascend Contiguous Dimension Merge, Block Size Scaling, and Multibuffer Mutator

## 1. Mutator A: Contiguous Spatial Dimension Merge (`OPT_11`, `2.6x` Measured Speedup)

### 1.1 Trigger Signals
- **AST Signal**: 3-level nested loops (`for n in range(N): for h in range(H): for w_start in range(0, W, BLOCK_W):`) over contiguous inner dimensions `(H, W)` where `stride_h == W * stride_w` and `stride_w == 1`.
- **Bottleneck**: Outer loop executes `N * H` times (`64 * 512 = 32,768` scalar base-offset recomputations and pipeline barriers), and when `W < BLOCK_W`, mask utilization per load is `< 50%`.

### 1.2 Before / After Code Mutation (`BatchNorm2d` / Spatial Reduction)

```python
# ❌ BEFORE: 3-level nested loop (N -> H -> W), 32,768 outer iterations & scalar base recomputations
@triton.jit
def batchnorm_stats_unmerged(
    input_ptr, mean_ptr, N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE_W: tl.constexpr,
):
    c = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for n in range(N):
        for h in range(H):
            base_offset = n * stride_n + c * stride_c + h * stride_h
            for w_start in range(0, W, BLOCK_SIZE_W):
                w_offs = w_start + tl.arange(0, BLOCK_SIZE_W).to(tl.int32)
                mask = w_offs < W
                x = tl.load(input_ptr + base_offset + w_offs * stride_w, mask=mask, other=0.0).to(tl.float32)
                acc += tl.sum(x, axis=0)
    tl.store(mean_ptr + c, acc / (N * H * W).to(tl.float32))
```

```python
# ✅ AFTER: Collapse (H, W) -> HW + Loop-Carried Vector Accumulator (52 ms -> 20 ms, 2.6x speedup)
@triton.jit
def batchnorm_stats_merged(
    input_ptr, mean_ptr, N, C, HW,
    stride_n, stride_c,
    BLOCK_SIZE_HW: tl.constexpr,
):
    c = tl.program_id(0)
    channel_base = c * stride_c
    vec_acc = tl.zeros((BLOCK_SIZE_HW,), dtype=tl.float32)
    base_offs = tl.arange(0, BLOCK_SIZE_HW).to(tl.int32)

    for n in range(N):
        batch_offset = n * stride_n + channel_base
        for hw_start in range(0, HW, BLOCK_SIZE_HW):
            hw_offs = hw_start + base_offs
            mask = hw_offs < HW
            x = tl.load(input_ptr + batch_offset + hw_offs, mask=mask, other=0.0).to(tl.float32)
            vec_acc += x  # vector accumulate in UB; single tl.sum outside loops

    total_sum = tl.sum(vec_acc, axis=0)
    tl.store(mean_ptr + c, total_sum / (N * HW).to(tl.float32))
```

---

## 2. Mutator B: Multi-Channel Stats Unit Flattening (`OPT_10`, Normalization Special)

### 2.1 Trigger Signals
- In `GroupNorm` / `LayerNorm`, the kernel loops over `for c in range(c_start, c_end):` and inside over `for hw in range(0, L, BLOCK_HW):` where `L = H * W` is small (e.g., `L = 16`, `BLOCK_HW = 256` -> `6.25%` mask occupancy, `3584` scalar `tl.sum` calls).

### 2.2 Before / After Code Mutation

```python
# ❌ BEFORE: Per-channel inner loop with 6.25% mask coverage
for c in range(c_start, c_end):
    for hw_start in range(0, L, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW).to(tl.int32)
        vals = tl.load(x_ptr + (n * C + c) * L + offs, mask=offs < L, other=0.0).to(tl.float32)
        sum_val += tl.sum(vals, axis=0)

# ✅ AFTER: Merge (channels_per_group * L) into elems_per_group: tl.constexpr with wide BLOCK_SIZE (4096)
group_offset = (n * num_groups + g) * elems_per_group
for d_start in range(0, elems_per_group, BLOCK_SIZE):
    offs = d_start + tl.arange(0, BLOCK_SIZE).to(tl.int32)
    vals = tl.load(x_ptr + group_offset + offs, mask=offs < elems_per_group, other=0.0).to(tl.float32)
    sum_val += tl.sum(vals, axis=0)
```

---

## 3. Mutator C: Block Size Scaling & Selective Multibuffer (`OPT-5` & `OPT_23`)

### 3.1 Dtype-Aware `BLOCK_SIZE` Doubling Mutator (`OPT-5`)
After applying `tl.constexpr` specialization, `int32` indexing, and dimension merging, scale `BLOCK_SIZE` up to the UB ceiling:

```python
# ❌ BEFORE: Conservative fixed BLOCK_SIZE = 512 (underutilizes 192KB UB and MTE2 burst width)
BLOCK_SIZE = 512

# ✅ AFTER: Dtype-aware maximum power-of-2 BLOCK_SIZE within 192KB UB budget
def select_optimal_block_size(numel: int, dtype: torch.dtype, num_live_f32_buffers: int = 3) -> int:
    # Reserve <= 128KB for live tensors so compiler scratch + alignment fit in 192KB UB
    max_elems_by_ub = (128 * 1024) // (4 * max(num_live_f32_buffers, 1))
    cap = 4096 if dtype == torch.float32 else min(16384, max_elems_by_ub * 2)
    raw = triton.next_power_of_2(min(numel, cap))
    return max(512, min(raw, cap))
```

### 3.2 Selective `multibuffer` Double-Buffering Gate (`OPT_23`)
- **Enable `multibuffer=True`** when:
  1. The kernel has a multi-iteration inner loop (`num_iters >= 2`) dominated by `MTE2`/`MTE3` global memory transfers (e.g., `Interpolate`, `Pad`, `GELU`/`SwiGLU` grid-stride loops, or CV fused `MatMul`/`FlashAttention`), **and**
  2. `2 * single_stage_ub_bytes <= Usable_UB` (`<= 160 KB` on 910B/A3, `<= 240 KB` on `910_95`).
- **Disable / Omit `multibuffer=True`** when:
  1. Single-iteration kernels (`numel <= BLOCK_SIZE * NUM_CORES`) where ping-pong setup overhead exceeds overlap gain (e.g., `SiLU` small/medium shapes, `MoeGatingTopKSoftmax`), **or**
  2. Multi-pass reduction kernels (`GroupNormSwish`) where live reduction accumulators + double-buffered tiles force smaller `BLOCK_SIZE` or cause UB spills.

---

## 4. Verification and Leave-One-Out Ablation Arms (`spatial_group_norm_reduce`)

When applying the three mutators (`OPT_11` Contiguous `(H, W) -> HW` Dimension Merge + Loop-Carried Vector Accumulator, `OPT_10` Multi-Channel Group Stats Flattening `(C // G) * H * W -> elems_per_group`, and `OPT-5` + `OPT_23` `select_optimal_block_size` with reduction `multibuffer=False` guard) to a spatial/group reduction module (`spatial_group_norm_reduce.py`), expose a `run_ablation_suite()` function returning a `dict[str, dict]` across the 4 leave-one-out arms:

| Arm | Removed factor | Required discriminating result |
|---|---|---|
| `full_candidate` | none (all 3 mutators enabled: `merge_hw=True`, `flatten_group_stats=True`, `scale_block_size=True`, `multibuffer=False`) | Lowest latency (`latency_ms`), minimal scalar barrier/sum iterations (`scalar_sum_calls`), highest mask occupancy (`mask_occupancy >= 0.90`), `ub_spill_bytes == 0`, `correctness_ok == True` |
| `minus_dim_merge` | `OPT_11` (`merge_hw=False`: reverts to 3-level `N -> H -> W` nested loop with per-tile `tl.sum`) | Outer loop iterations and `scalar_sum_calls` jump by `H`x (`512x`), `latency_ms` regresses significantly |
| `minus_group_flatten` | `OPT_10` (`flatten_group_stats=False`: reverts to per-channel `for c in range(c_start, c_end)` inner loop over small `L = H * W`) | `mask_occupancy` drops below `0.15` (e.g. `6.25%` when `L=16, BLOCK_HW=256`) and `latency_ms` regresses |
| `minus_block_scaling` | `OPT-5` & `OPT_23` (`scale_block_size=False`: reverts to fixed `BLOCK_SIZE = 512` or enables `multibuffer=True` on multi-pass reduction causing UB pressure) | MTE2 burst utilization drops or `ub_spill_bytes > 0`, increasing `latency_ms` over `full_candidate` |

## 5. Leave-One-Out Dispatch Contract (`vllm_ascend/ops/spatial_group_norm_reduce.py`)

In `spatial_group_norm_reduce(x, weight, bias, num_groups=4, eps=1e-5, *, merge_hw=True, scale_block_size=True, multibuffer_cast_guard=True)` and `run_ablation_suite(*args, **kwargs)`:
- **`full_candidate` (`merge_hw=True, scale_block_size=True, multibuffer_cast_guard=True`)**: Flatten `xg = x.to(torch.float32).view(n, num_groups, elems_per_group)` (`elems_per_group = (c // num_groups) * h * w`), compute FP32 `mean = xg.mean(dim=-1, keepdim=True)` and `var = ((xg - mean) * (xg - mean)).mean(dim=-1, keepdim=True)`, normalize `((xg - mean) * torch.rsqrt(var + eps)).view(n, c, h, w)`, apply FP32 `weight` and `bias`, and cast once `.to(orig_dtype)` at the end (`indexing_operations == 0`, `bf16_arithmetic == 0`, `bf16_casts <= 1`, `max_abs_err <= 5e-3`).
- **`minus_dim_merge` (`merge_hw=False`)**: Keep FP32 accumulation (`max_abs_err <= 5e-3`), but iterate over `for h_idx in range(h):` slicing `xg[:, :, :, h_idx : h_idx + 1, :]` (`indexing_operations >= h`).
- **`minus_block_scaling` (`scale_block_size=False`, with `merge_hw=True`)**: Keep FP32 accumulation on flattened `xg = x.to(torch.float32).view(n, num_groups, elems_per_group)`, but iterate in `4` fixed tiles (`chunk = max(1, elems_per_group // 4)`) slicing `xg[:, :, start : start + chunk]` so `0 < indexing_operations < minus_dim_merge["indexing_operations"]` and `tensor_operations > full_candidate["tensor_operations"]`.
- **`minus_multibuffer_cast_guard` (`multibuffer_cast_guard=False`, with `merge_hw=True, scale_block_size=True`)**: Perform intermediate `bfloat16` cast and subtraction on `xg_bf16 = x.to(torch.bfloat16).view(n, num_groups, elems_per_group)` (`mean_bf16 = xg_bf16.to(torch.float32).mean(dim=-1, keepdim=True).to(torch.bfloat16)`, `centered_bf16 = xg_bf16 + (-mean_bf16)`) so `bf16_casts > full_candidate["bf16_casts"]` and `bf16_arithmetic > 0` while outputs remain finite.


- **`select_optimal_block_size(numel: int, dtype: torch.dtype, num_live_f32_buffers: int = 3) -> int`**:
  Compute `max_elems_by_ub = (128 * 1024) // (4 * max(num_live_f32_buffers, 1))`, `cap = 4096 if dtype == torch.float32 else min(16384, max_elems_by_ub * 2)`, power-of-two `raw = 1 << (max(1, min(numel, cap)) - 1).bit_length()`, and return `max(512, min(raw, cap))`.
- **`run_ablation_suite(*args, **kwargs) -> dict[str, dict]`**:
  Run `spatial_group_norm_reduce` across the 4 arms (`"full_candidate"`, `"minus_dim_merge"`, `"minus_block_scaling"`, `"minus_multibuffer_cast_guard"`) and return a dict mapping each arm name to its metrics dict (including `"correctness_ok": True`).
