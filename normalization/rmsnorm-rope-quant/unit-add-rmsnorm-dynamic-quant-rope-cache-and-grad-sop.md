---
category: sop
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
  - Ascend950PR
  - Ascend950DT
  workloads:
  - RmsNorm
  - AddRmsNormDynamicQuant
  - RmsNormDynamicMxQuant
  - RmsNormGrad
  - ApplyRotaryPosEmb
domain_path: normalization/rmsnorm-rope-quant
evidence_strength: reproduced_empirical
id: unit-add-rmsnorm-dynamic-quant-rope-cache-and-grad-sop
inherits_from:
- unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  attribution: Adapted from CANNBot-DSL rms_norm sample and CANNBot-Knowledge direct/ACLNN
    norm and RoPE operator guides
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/rms_norm/rms_norm.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/norm/add_rms_norm_dynamic_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/norm/rms_norm_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/transformer/posembedding/apply_rotary_pos_emb_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/norm/rms_norm_dynamic_mx_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/nn/norm/rms_norm_grad_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/rms_norm/rms_norm.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-mhc-pre-post-sinkhorn-knopp-and-engram-gate-fusion-sop
- unit-ascend-dynamic-quant-mxfp8-mxfp4-and-antiquant-operator-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for implementing and optimizing RMSNorm, AddRmsNormDynamicQuant,
  RmsNormDynamicMxQuant, RmsNormGrad, and ApplyRotaryPosEmb (RoPE) across CANNBot-DSL,
  Ascend C direct invoke, and ACLNN.
tier: L2
title: AddRmsNormDynamicQuant, RmsNorm (DSL/Grad/MXQuant) & Rotary Position Embedding
  SOP
triggers:
  error_signatures:
  - RmsNorm and AddRmsNormDynamicQuant must accumulate x^2 mean and rstd = rsqrt(mean(x^2)
    + eps) in FP32 even when x/gamma/y are FP16 or BF16
  - CANNBot-DSL rms_norm adaptively selects full-row UB residency vs column-tiled
    two-pass reduction based on UB capacity and trailing K normalization axes
  - AddRmsNormDynamicQuant fuses residual addition x = x1 + x2, RMSNorm, optional
    smooth_scales, and per-token INT8/FP8 dynamic quantization in a single UB residency
    pass
  keywords:
  - rms_norm
  - add_rms_norm_dynamic_quant
  - rms_norm_dynamic_mx_quant
  - rms_norm_grad
  - apply_rotary_pos_emb
  - rope
  - rstd_fp32
  - full_row_vs_col_split
---
## Preconditions & Signals

Apply this SOP when developing or optimizing RMSNorm variants (`rms_norm`, `add_rms_norm_dynamic_quant`, `rms_norm_dynamic_mx_quant`, `rms_norm_grad`) and Rotary Position Embedding (`apply_rotary_pos_emb`) on Ascend 910B/910C and Ascend 950 (`ARCH 3510`):

1. **Normalization Axis & Dtype Contract**:
   - Normalize over the trailing $K$ dimensions of `x` matching `gamma.shape` ($D = \prod_{k} \text{gamma.shape}[k]$).
   - Support `bfloat16`, `float16`, and `float32` for `x`, `gamma`, and `y`, while keeping `rstd` (`1 / sqrt(mean(x^2) + eps)`) unconditionally in `float32`.
2. **Fused Residual + Norm + Dynamic Quantization (`add_rms_norm_dynamic_quant`, `rms_norm_dynamic_mx_quant`)**:
   - Avoids 2–3 extra GM round-trips by keeping the residual sum $r = x_1 + x_2$ and normalized activations $u = \operatorname{RMSNorm}(r) \odot \gamma$ in UB while computing either per-token symmetric INT8/FP8 scale (`max(abs(u)) / Q_max`) or per-32-element MXFP8 (`E8M0`) scale.
3. **RoPE Interleaved vs Split-Half Pairing (`apply_rotary_pos_emb`)**:
   - Rotates query/key head dimensions using either adjacent-element pairs $(x_{2i}, x_{2i+1})$ or half-split pairs $(x_i, x_{i + D/2})$ with `cos` and `sin` tables.

## Procedure

### Step 1: Adaptive Full-Row vs Column-Split RMSNorm (`cannbot-dsl/samples/rms_norm`)
1. **Regime A — Full-Row UB Residency ($D$ fits in UB with double buffering)**:
   - Load multiple rows $[M_{\text{tile}}, D]$ of `x` and pre-load `gamma[D]` once into a persistent UB buffer (`ResidentGamma`).
   - Cast `x` to FP32 in UB, compute $s = \frac{1}{D}\sum_{j=0}^{D-1} x_j^2$, compute `rstd = rsqrt(s + eps)`, multiply `y = cast_out(x_fp32 * rstd * gamma_fp32)`, and store `y` and `rstd` to GM in a single pass.
2. **Regime B — Column-Tiled Two-Pass Reduction (Large $D$ exceeding UB budget)**:
   - **Pass 1 (Variance Accumulation)**: Stream column tiles $d \in [0, D)$ through UB, accumulating partial sum of squares $\sum x_d^2$ into an on-chip FP32 scalar/vector accumulator per row without writing intermediate FP32 tiles back to GM; finalize `rstd`.
   - **Pass 2 (Normalize & Scale)**: Stream column tiles of `x` and `gamma` a second time, multiply by the finalized row `rstd` and `gamma` tile, and write `y` to GM.

### Step 2: Fused `AddRmsNormDynamicQuant` & `RmsNormDynamicMxQuant`
1. **Single-Pass UB Pipeline (`add_rms_norm_dynamic_quant_950`)**:
   - Load `x1` and `x2` into UB, compute $x_{\text{out}} = x_1 + x_2$, and immediately write $x_{\text{out}}$ to GM (for the residual stream) while keeping its FP32 copy in UB.
   - Compute `rstd = rsqrt(mean(x_fp32^2) + eps)` and normalized activation $h = x_{\text{fp32}} \cdot \text{rstd} \cdot \gamma$ (plus optional `smooth_scale` multiplication).
   - For per-token dynamic quantization: reduce $m_i = \max_d |h_{i,d}|$, compute $\text{scale}_i = m_i / 127.0$ (or FP8 max), and quantize $q_{i,d} = \operatorname{round\_to\_even}(h_{i,d} / \text{scale}_i)$ in UB.
   - For MX quantization (`rms_norm_dynamic_mx_quant_950`): partition each row into 32-element blocks, compute shared `E8M0` exponent scales per 32-element block, and emit `fp8_e4m3fn` / `fp8_e5m2` + paired scale tensor.

### Step 3: RMSNorm Backward (`rms_norm_grad_950`) & RoPE (`apply_rotary_pos_emb_950`)
1. **Two-Output Gradient Reduction (`rms_norm_grad`)**:
   - For each row, compute $\text{dot}_i = \sum_d dy_{i,d} \cdot x_{i,d} \cdot \gamma_d$ in FP32, then compute input gradient:
     $$dx_{i,d} = \text{rstd}_i \cdot \left(dy_{i,d} \cdot \gamma_d - \frac{x_{i,d} \cdot \text{rstd}_i^2}{D} \cdot \text{dot}_i\right)$$
   - Accumulate partial weight gradients $d\gamma_d = \sum_i dy_{i,d} \cdot x_{i,d} \cdot \text{rstd}_i$ in FP32 within each AIV core across its assigned rows before performing a single cross-core reduction (or deterministic workspace reduction) to GM.
2. **Vectorized RoPE (`apply_rotary_pos_emb_950`)**:
   - Broadcast `cos` and `sin` across the head dimension per token position. For adjacent-pair RoPE, use hardware even/odd gather or paired complex-multiply vector instructions to compute $(-x_{2i+1}, x_{2i})$ without scalar lane loops; for half-split RoPE, slice the UB buffer at offset $D/2$ (`D/2` must be 32B-aligned or handled via aligned vector register shifts) and compute $x \odot \cos + \text{rotate\_half}(x) \odot \sin$ in FP32 before rounding to BF16/FP16.

## Verification & Checks

1. **Order of Rounding vs `gamma` Multiplication**:
   - Check whether the target model contract specifies `BF16(x_fp32 * rstd) * BF16(gamma)` or `BF16(x_fp32 * rstd * FP32(gamma))`—align the kernel's cast point with the reference contract to prevent 1-ULP BF16 divergence.
2. **Non-32B-Aligned Hidden Size $D$**:
   - When $D \times \text{sizeof(dtype)}$ is not a multiple of 32 bytes, use `DataCopyPad` on GM load/store and zero-fill the padded tail elements in UB before computing $\sum x_j^2$ and $\max |h_j|$, while still dividing the variance sum by the exact logical $D$ (never by $D_{\text{aligned}}$).

## Failure Modes & Fallbacks

- **Symptom**: `AddRmsNormDynamicQuant` outputs `INT8` saturation (`-128` or `127` everywhere) or `NaN` scale on padded rows.
  - **Root Cause**: Uninitialized padding lanes in UB participate in `ReduceMax(abs(h))`, or an all-zero input row yields `max_abs = 0` and `0 / 0 = NaN` during quantization division.
  - **Fix**: Mask tail lanes to `0.0f` before `ReduceMax`, and clamp the quantization divisor (`inv_scale = max_abs == 0 ? 0.0f : 127.0f / max_abs`).
- **Symptom**: Severe atomic contention in `rms_norm_grad` when batch*seq $M$ is large and hidden size $D$ is small.
  - **Root Cause**: Issuing `SetAtomicAdd` to GM `dgamma[D]` on every row inside the inner loop across 48 AIV cores.
  - **Fix**: Keep a persistent FP32 `dgamma_acc[D]` buffer in UB per core, accumulate across all rows assigned to that core, and flush to GM only once at kernel epilogue.
