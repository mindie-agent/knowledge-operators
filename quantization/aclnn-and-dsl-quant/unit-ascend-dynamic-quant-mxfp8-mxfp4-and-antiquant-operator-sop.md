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
  - DynamicQuant
  - DynamicMxQuant
  - DynamicDualLevelMxQuant
  - SwigluMxQuant
  - DequantSwigluQuant
  - AntiMxQuant
  - AscendQuant
domain_path: quantization/aclnn-and-dsl-quant
evidence_strength: reproduced_empirical
id: unit-ascend-dynamic-quant-mxfp8-mxfp4-and-antiquant-operator-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/operators/aclnn/nn/quant,
    operators/direct/nn/quant, and apis/simd_api/adv_api/quantization documentation
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/quant/dynamic_mx_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/quant/dynamic_mx_quant_with_dual_axis_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/quant/dynamic_dual_level_mx_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/quant/swiglu_mx_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/quant/anti_mx_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/quant/dequant_swiglu_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/quant/dynamic_quant_950.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/quant/dynamic_mx_quant_950.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-add-rmsnorm-dynamic-quant-rope-cache-and-grad-sop
- unit-cannbot-dsl-matmul-and-catlass-gemm-epilogue-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Ascend quantization and dequantization operators,
  covering per-token INT8/FP8 DynamicQuant, Ascend 950 OCP MXFP8/MXFP4 block/dual-axis/dual-level
  quantization, fused SwiGLU-Quant / Dequant-SwiGLU-Quant, and AntiQuant / AntiMxQuant.
tier: L2
title: Ascend Dynamic Quant, Dual-Axis/Block MXFP8/MXFP4 Quant, SwiGLU-Quant & AntiQuant
  SOP
triggers:
  error_signatures:
  - OCP MX quantization (DynamicMxQuant, SwigluMxQuant, AntiMxQuant on Ascend 950)
    uses 32-element micro-blocks sharing an E8M0 power-of-two exponent scale
  - Dual-axis MX quantization (dynamic_mx_quant_with_dual_axis_950, swiglu_mx_quant_with_dual_axis_950)
    emits both row-wise and column-wise quantized outputs and E8M0 scales in one launch
    for forward+backward training GEMMs
  - In-place Abs(x) before quantization overwrites the sign bits of the source buffer
    if src and absBuf alias in UB
  keywords:
  - dynamic_quant_950
  - dynamic_mx_quant_950
  - dynamic_mx_quant_with_dual_axis_950
  - dynamic_dual_level_mx_quant_950
  - swiglu_mx_quant_950
  - dequant_swiglu_quant_950
  - anti_mx_quant_950
  - ascend_dequant
  - e8m0_scale
---
## Preconditions & Signals

Apply this SOP when implementing, integrating, or debugging quantization and dequantization kernels across Ascend 910B/910C and Ascend 950 (`ARCH 3510`):

1. **Per-Token vs Per-Block vs Microscaling (`MX`) Quantization**:
   - **Per-Token Dynamic Quant (`dynamic_quant`, `dynamic_quant_950`)**: Computes one FP32 scale per token row ($s_i = \max_d |x_{i,d}| / Q_{\max}$, with $Q_{\max} = 127$ for `INT8`, `7` for `INT4`, or `448`/`57344` for `FP8`) and optional asymmetric `offset`.
   - **OCP Microscaling Quant (`dynamic_mx_quant_950`, `anti_mx_quant_950`)**: Partitions the quantization axis into 32-element micro-blocks; each 32-element block shares an 8-bit unsigned power-of-two exponent `E8M0` scale, quantizing elements to `MXFP8` (`e4m3fn`/`e5m2`) or `MXFP4` (`e2m1fn_x2`).
   - **Dual-Axis & Dual-Level MX Quant (`dynamic_mx_quant_with_dual_axis_950`, `dynamic_dual_level_mx_quant_950`, `grouped_dynamic_mx_quant_950`)**: Used in FP8/FP4 training and MoE where a tensor must be quantized along both the $K$-axis (for forward GEMM) and the $M$/$N$-axis (for backward weight-gradient GEMM) in a single UB pass.
2. **Fused Activation + Quantization (`dequant_swiglu_quant_950`, `swi_glu_quant_950`, `swiglu_mx_quant_950`, `swiglu_mx_quant_with_dual_axis_950`)**:
   - Fuses post-GEMM dequantization (when input is `INT32` from W8A8 GEMM), SwiGLU activation ($\operatorname{SiLU}(x_{\text{gate}}) \odot x_{\text{up}}$), and dynamic INT8/FP8/MXFP8 quantization of the intermediate FFN activation.

## Procedure

### Step 1: Per-Token & Per-Block Dynamic Quantization (`dynamic_quant_950`, `dynamic_block_quant_950`)
1. **Non-Destructive Max-Abs Reduction in UB**:
   - Load `x` into UB, cast `FP16`/`BF16` to `FP32`, and optional multiply by resident `smooth_scales[D]` (`P41`).
   - Compute `Abs(absBuf, xFp32)` into a **separate** `absBuf` (never overwrite `xFp32` in-place with `Abs`, which destroys negative signs!).
   - Reduce `maxAbs = ReduceMax(absBuf)` across the token row (or $1 \times 128$ / $128 \times 128$ block in `dynamic_block_quant_950`).
2. **Reciprocal Multiplication, Rounding & Saturation Clamping (`A3`, `A8`)**:
   - Compute `scale = maxAbs / Q_max` (write `scale` to GM in `FP32`) and `invScale = (maxAbs == 0.0f) ? 0.0f : (Q_max / maxAbs)`.
   - Multiply `scaled = xFp32 * invScale`, round using `Cast(..., RoundMode::CAST_RINT)` (round-to-nearest-even), and clamp strictly to `[quant_min, quant_max]` (`[-128, 127]` for `INT8`, `[-8, 7]` for packed `INT4`).

### Step 2: Ascend 950 Microscaling (`MXFP8` / `MXFP4`) & Dual-Axis Quantization
1. **32-Element Micro-Block `E8M0` Scale Derivation (`dynamic_mx_quant_950`)**:
   - Reshape the quantization axis into groups of 32 elements (`pad` tail with zeros if $D \bmod 32 \ne 0$).
   - For each 32-element group, find the maximum biased exponent among the 32 elements, subtract the target format's maximum normal exponent ($E_{\max} = 8$ for `e4m3fn`, $15$ for `e5m2`, $1$ for `e2m1`), and store the shared exponent as an 8-bit `UE8M0` scale (or use hardware MX quantization instructions in `vf(mode="simd")` / SIMT `_asc_cvt_float2_to_fp8x2`).
   - In `dynamic_dual_level_mx_quant_950`, combine a global/per-tensor FP32 scale with per-32-element `E8M0` micro-scales to preserve dynamic range for `MXFP4`.
2. **Single-Pass Dual-Axis MX Quantization (`dynamic_mx_quant_with_dual_axis_950`)**:
   - Tile the 2D input $[M, K]$ into $32 \times 32$ tiles in UB.
   - Within the same UB residency window:
     1. Quantize each row of 32 elements along $K$ to emit `(y_row_mx, scale_row_e8m0)`.
     2. Transpose the $32 \times 32$ tile in UB (via `Vnchwconv` / `Transpose`) and quantize each column of 32 elements along $M$ to emit `(y_col_mx, scale_col_e8m0)`.
   - This eliminates a full GM read pass and a standalone transpose kernel during FP8/MXFP8 training.

### Step 3: Fused `DequantSwiGLUQuant` & `SwiGLUMxQuant`
1. **Split-Half vs Interleaved Gate/Up Layout**:
   - Support both concatenated `[..., 2*H]` (`gate = x[..., :H]`, `up = x[..., H:]`) and interleaved `[..., H, 2]` layouts (`activate_left = true/false`).
   - Load matching tiles of `gate` and `up` into UB (applying `AscendDequant` or FP32 `deq_scale` + `bias` if input is `INT32`), evaluate $\text{act} = \frac{x_{\text{gate}}}{1 + \exp(-x_{\text{gate}})} \odot x_{\text{up}}$ in `FP32`, and immediately pass `act` into the UB dynamic/MX quantizer without writing the unquantized FP16/BF16 intermediate to GM.

### Step 4: Hardware-Accelerated Dequantization (`AscendAntiQuant`, `AntiMxQuant`)
1. **Weight-Only & Activation Dequantization (`ascend_anti_quant.md`, `anti_mx_quant_950.md`)**:
   - Use hardware `AscendDequant` or `Brcb` scale/offset broadcast + `Cast<float, int8_t>` + `Add`/`Mul` (`P49`) to dequantize `INT8`/`INT4` tensors to `FP16`/`BF16`.
   - For `anti_mx_quant_950`, broadcast each `E8M0` scale across its 32-element `MXFP8`/`MXFP4` micro-block in UB and reconstruct `FP16`/`BF16`/`FP32`.

## Verification & Checks

1. **In-Place `Abs` Source Corruption Check (`abs_inplace_overwrites_signed_source_before_quant.md`)**:
   - Audit every dynamic quantization kernel to ensure `Abs(absLocal, xLocal)` never uses `absLocal == xLocal`; otherwise all quantized outputs become non-negative!
2. **Dual-Axis Scale Packing & Padding Alignment**:
   - Verify whether downstream `QuantBatchMatmulMX` expects paired scales (`[..., ceil(K/64), 2]`) and pad odd 32-element tail groups with neutral `E8M0` scale.

## Failure Modes & Fallbacks

- **Symptom**: `SwiGLUMxQuant` produces `NaN` on negative `gate` inputs with large magnitude ($x_{\text{gate}} < -88$).
  - **Root Cause**: Computing $\operatorname{SiLU}(x) = x / (1 + \exp(-x))$ where $\exp(-x)$ overflows FP32 to $+\infty$, or computing in FP16 where $\exp(-x)$ overflows at $x < -11$.
  - **Fix**: Always compute `SiLU` in `FP32` (`1 / (1 + +inf) == 0.0f` is well-defined in IEEE FP32 as long as `x * sigmoid(x)` is evaluated as `x * (1.0f / (1.0f + exp(-x)))`, never `x / (1 + exp(-x))` when `x = -inf`).
