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
domain_path: compilation/triton/playbooks/norm-rope-activation
evidence_strength: software_contract
id: unit-triton-ascend-norm-rope-activation-and-transform-playbooks-sop
inherits_from:
- unit-triton-ascend-operator-domain-constraints-contract
- unit-triton-ascend-hardware-and-compiler-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/normalization__groupnormswish.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/normalization__adaptiveinstancenorm2dbackward.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__rotarymul.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/elementwise__activation.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__interpolate.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__pad.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/guides/operator_playbooks/normalization__groupnormswish.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Triton-Ascend GroupNormSwish, AdaIN Backward, RotaryMul, Activation, Interpolate,
  and Pad Playbooks SOP
triggers:
  error_signatures:
  - 1. Normalization Playbooks (`GroupNormSwish` & `AdaIN2DBackward`)
  - 1.1 `GroupNormSwish` (Forward Fused, `1.25x–1.40x` Geomean, 50/50 Pass)
  - 1.2 `AdaptiveInstanceNorm2DBackward` (Backward Multi-Granularity Split)
  - 2. `RotaryMul` (RoPE) & Elementwise Activations Playbook
  keywords:
  - unit-triton-ascend-norm-rope-activation-and-transform-playbooks-sop
---
# Triton-Ascend GroupNormSwish, AdaIN Backward, RotaryMul, Activation, Interpolate, and Pad Playbooks SOP

## 1. Normalization Playbooks (`GroupNormSwish` & `AdaIN2DBackward`)

### 1.1 `GroupNormSwish` (Forward Fused, `1.25x–1.40x` Geomean, 50/50 Pass)
1. **Single-Kernel Two-Pass Fused Skeleton**:
   - Launch `grid = (min(N * num_groups, VEC_CORE_NUM),)` with `for gid in range(pid, total_groups, num_cores):`.
   - **Pass 1**: Accumulate `sum_acc` and `sq_acc` in `tl.float32` scalars across `for d_start in range(0, elems_per_group, BLOCK_SIZE):`, then compute `mean`, `var = (sq_acc - sum_acc * sum_acc / elems_per_group) / elems_per_group`, and `rstd = 1.0 / tl.sqrt(var + eps)`.
   - **Pass 2**: Reload tiles, normalize, apply per-channel affine `w, b`, compute `out = x_norm * tl.sigmoid(swish_scale * x_norm)`, and store.
2. **Mandatory Scalar Channel Load & `D % 16 != 0` Block Shrinking**:
   - Declare `channels_per_group: tl.constexpr`, `elems_per_group: tl.constexpr`, and `BLOCK_SIZE: tl.constexpr` (`+38%` launch specialization speedup).
   - Set `MAX_BLOCK = 2048` (`fp32` or `D == 1`) or `4096` (`fp16`/`bf16`). Only when `16 < D < MAX_BLOCK and D % 16 != 0`, shrink `block_size = ((D - 1) // 16) * 16` so `block_size < D`, guaranteeing every block falls inside a single channel and `w = tl.load(weight_ptr + group_base + (d_start // D))` executes as a **scalar load** (zero vector integer division). Never call `tl.reshape`.

### 1.2 `AdaptiveInstanceNorm2DBackward` (Backward Multi-Granularity Split)
- Split into **Kernel 1 (`reduce_kernel`)** and **Kernel 2 (`apply_kernel`)**:
  - Kernel 1 reduces spatial tiles into `torch.float32` buffers `partial_s1`, `partial_s2` (`per-(N, C)`) and `grad_weight`, `grad_bias` (`per-C`) via `tl.atomic_add`.
  - Kernel 2 reads the globally reduced `partial_s1/s2` to compute `grad_input` in `fp32` before casting to output dtype.

---

## 2. `RotaryMul` (RoPE) & Elementwise Activations Playbook

### 2.1 `RotaryMul` (`[B, H, S, D]`, `1.02x–1.52x` Geomean, 50/50 Pass)
1. **Zero Flat-1D Coordinate Modulo (`L1.1`)**: Never flatten `B * H * S * D` and decode `(b, h, s, d)` via `%` and `//` per element (causes `hivm` scalarization to `memref<1xf16>` loads, `19.8%` vector utilization).
2. **2D Position-Tiled Vectorization + Broadcast Strides (`L1.2`–`L1.6`)**:
   - On the host, compute broadcast strides for `r1`/`r2`: `_bc_stride = tuple(0 if t.shape[i] == 1 and in_shape[i] > 1 else t.stride(i) for i in range(4))`.
   - Fix `TILE_S = 16` (or `BLOCK_ROWS in {32, 64}` over flattened `[total_rows, D]` in multi-head-per-program layout) and pass `MAX_HALF_D = D // 2` as `tl.constexpr`. Add `tl.assume(stride_in_d == 1)` at kernel entry.
   - In `interleave` mode, load the contiguous `D`-tile once into registers, split/combine even and odd lanes via direct algebraic rotation (`out_e = x_e * r1_e - x_o * r2_e`, `out_o = x_o * r1_o + x_e * r2_o`) and `tl.interleave`/`tl.join` without materializing intermediate buffers.

### 2.2 Elementwise Activations (`GELU`: `0.80x`, `SiLU`: `1.27x`, `SwiGLU`: `1.31x`)
- Always `reshape(-1)` to 1D and use a core-clamped 1D grid-stride loop (`grid = (min(triton.cdiv(numel, BLOCK), NUM_VCORE),)`).
- Dtype-aware `BLOCK_SIZE`: for large tensors (`>= 4M` elements), use `8192` on `fp16`/`bf16` but cap at `4096` on `fp32` to prevent UB spilling. Enable `multibuffer=True, unit_flag=True` on `GELU` and `SwiGLU`, but **omit** them on `SiLU`.
- For `F.gelu` on NPU, always use the sigmoid approximation formula `x / (1.0 + tl.exp(-1.5957691 * 0.044715 * (x / 0.044715 + x * x * x)))` to match `torch_npu` bit-exact across both `'none'` and `'tanh'` modes.

---

## 3. Tensor Transform Playbooks (`Interpolate` & `Pad`)

### 3.1 `Interpolate` (`nearest`, `bilinear`, `bicubic`, `area` — `1.3834x` Geomean, 73/73 Pass)
Never use a single monolithic kernel; dispatch in `_route()` across specialized kernels (`all with multibuffer=True`):
1. **`nearest` `2x` Upsample & `0.5x` Downsample**:
   - `2x` upsample (`H_out == 2 * H_in, W_out == 2 * W_in`): load contiguous input row and duplicate via `vals.reshape([BLOCK_IN, 1]) * tl.full([1, 2], 1, dtype)` (broadcast beats gather).
   - `0.5x` downsample: use regular stride-2 vector load (`offs * 2`).
2. **`bilinear` `0.5x` Downsample (`align_corners=False`) -> `2x2` Average Pool (`100x` faster than strided gather)**:
   - Contiguously load `2 * BLOCK_W` elements from row `2*h` and row `2*h + 1`, apply `tl.reshape(row, [BLOCK_W, 2])`, and sum (`(tl.sum(r0, 1) + tl.sum(r1, 1)) * 0.25`).
3. **`bilinear` Upsample & General Paths -> Host-Precomputed `X` Coordinates**:
   - Precompute `(x0, x1, xlambda)` tables once on the host (`_precompute_bilinear_x_coords`) and pass as pointers; for upsampling (`H_out > H_in`), use 2D vertical tiling (`BLOCK_H = 2, MAX_KH = 3`) so 3 loaded input rows serve 2 output rows.
4. **`bicubic` & `align_corners` Precision Rules**:
   - For `align_corners=False`, fix negative coordinate truncation (`tl.cast(src_y, tl.int32)` truncates toward zero instead of `floor`) via `y0 = tl.where(src_y < y0.to(tl.float32), y0 - 1, y0)`, and compute cubic polynomial weights with nested `tl.where(at < 1.0, w1, tl.where(at < 2.0, w2, 0.0))`.
   - For `align_corners=True` on `fp16`/`bf16`, compute `scale = (in - 1) / (out - 1)` using NPU `vdiv` (`torch.tensor(..., device="npu")`) to match CANN `1 ULP` division rounding; for `bicubic` `align_corners=True`, precompute 4-tap indices/weights on host and accumulate the 16 taps sequentially in scalar order to match PyTorch C++.

### 3.2 `Pad` (`constant`, `reflect`, `replicate`, `circular` — `1.68x` Geomean, 51/51 Pass)
- Squeeze leading size-1 dimensions on the host and scale `BLOCK_SIZE` cap up to **`16384`** (`1.08x` at `512` -> `1.25x–1.68x` at `16384`).
- For `mode == "constant"`, **never** branch inside the kernel with `tl.where(in_bounds, x, fill_val)`; execute `output.fill_(value)` first and launch a pure contiguous `copy_kernel` over the interior region.
- For `reflect`/`replicate`/`circular`, cast coordinates to `tl.float32` before `< 0` or `>= in_dim` boundary comparisons.
