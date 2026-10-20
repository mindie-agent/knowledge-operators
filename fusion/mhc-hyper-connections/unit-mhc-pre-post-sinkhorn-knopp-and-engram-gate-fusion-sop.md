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
  - MhcSinkhorn
  - EngramGate
  - EngramGateFusion
  - BlockAttnResUpdateRmsNorm
domain_path: fusion/mhc-hyper-connections
evidence_strength: reproduced_empirical
id: unit-mhc-pre-post-sinkhorn-knopp-and-engram-gate-fusion-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascendc-simt-and-regbase-vector-programming-sop
- unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
provenance:
  attribution: Adapted from CANNBot-DSL engram_gate and block_attn_res_update_rms_norm
    samples and CANNBot-Knowledge MhcSinkhorn and EngramGateFusion 950 direct operator
    guides
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/engram_gate/engram_gate.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/block_attn_res_update_rms_norm/block_attn_res_update_rms_norm.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/transformer/mhc/mhc_sinkhorn_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/transformer/attention/engram_gate_fusion_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/engram_gate/engram_gate.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-flash-kda-and-mega-recurrent-replayssm-dsl-sop
- unit-add-rmsnorm-dynamic-quant-rope-cache-and-grad-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Manifold-Constrained Hyper-Connections (MhcSinkhorn),
  EngramGate / EngramGateFusion, and Block-Attention Residual Update + RMSNorm fusion
  in CANNBot-DSL and Ascend C on Ascend 950.
tier: L2
title: MHC Sinkhorn-Knopp, EngramGate Fusion & Block-Attention Residual Update RMSNorm
  SOP
triggers:
  error_signatures:
  - MhcSinkhorn on Ascend 950 dispatches across 4 Direct Routes keeping iterative
    row/column Sinkhorn-Knopp normalization resident in UB in FP32
  - EngramGate and EngramGateFusion compute dual-path RMSNorm, weighted dot-product
    gate with signed-sqrt sigmoid, depthwise dilated causal conv1d + SiLU (in full
    fusion), and residual update
  - block_attn_res_update_rms_norm updates partial_block in-place, merges online-softmax
    block scores on-chip without writing intermediate h to GM, and outputs RMSNorm(h)
    * gamma
  keywords:
  - mhc_sinkhorn
  - engram_gate
  - engram_gate_fusion
  - block_attn_res_update_rms_norm
  - sinkhorn_knopp
  - hyper_connections
  - signed_sqrt_sigmoid
  - online_softmax_residual
---
## Preconditions & Signals

Apply this SOP when implementing or optimizing residual/stream hyper-connection and gating operators on Ascend 950 (`ARCH 3510`), including Manifold-Constrained Hyper-Connections (`MhcSinkhorn`), `engram_gate` / `engram_gate_fusion`, and `block_attn_res_update_rms_norm`:

1. **Multi-Stream Hyper-Connection Sinkhorn Normalization (`MhcSinkhorn`)**:
   - Alternates row and column normalization over small stream matrices ($n \times n$, e.g., $n \in \{4, 8\}$) for $K$ iterations so the routing matrix converges to doubly stochastic form without round-tripping to GM between iterations.
2. **Engram Residual Gate & Causal Conv1D Fusion (`engram_gate`, `engram_gate_fusion`)**:
   - **`engram_gate` (CANNBot-DSL)**: Fuses dual-path RMSNorm on `x` and `key`, weighted dot-product score, signed-sqrt sigmoid gate activation, optional `image_mask` zeroing, and gated `value` residual addition into `x`.
   - **`engram_gate_fusion` (Ascend C Direct Invoke on 950)**: Extends `engram_gate` with a second RMSNorm on the gated value, depthwise dilated causal Conv1D + SiLU, gated residual, and continuous conv state output.
3. **Block Attention Residual Update + RMSNorm (`block_attn_res_update_rms_norm`)**:
   - Fuses residual block update `p = partial_block + float32(delta)`, pseudo-query scoring, online-softmax accumulation with prior blocks (`numerator`, `logit_max`, `exp_sum`), and final RMSNorm $\times \gamma$ in a single Vector pass.

## Procedure

### Step 1: Manifold-Constrained Hyper-Connections (`MhcSinkhorn` on Ascend 950)
1. **UB-Resident Iterative Sinkhorn-Knopp Loop**:
   - Load each token's $n \times n$ hyper-connection routing matrix into UB once, cast to `float32`, exponentiate $P^{(0)} = \exp(M / \tau)$, and run all $K$ alternating row-normalization and column-normalization steps entirely inside UB registers/buffers before writing the normalized matrix back to GM.
2. **Direct Route Selection & Tail Guarding**:
   - Select among the 4 Direct Routes in `mhc_sinkhorn_tiling.cpp` based on matrix extent and batch size; pad the inner row stride in UB to 32-byte alignment while masking active columns $j < n$ during row-sum and column-sum reductions so padding lanes never pollute Sinkhorn denominators.

### Step 2: EngramGate (`engram_gate`) & EngramGateFusion (`engram_gate_fusion`)
1. **CANNBot-DSL `engram_gate` Pipeline**:
   - Given `x`, `key`, `value`, `weight`, optional `image_mask`, `eps=1e-6`, and `clamp_value=1e-6`:
   - Accumulate $\operatorname{mean}(x^2)$ and $\operatorname{mean}(\text{key}^2)$ in FP32 to normalize $\hat{x}$ and $\hat{k}$, compute the weighted dot product $s = \sum_d w_d \hat{x}_d \hat{k}_d$, apply the signed-sqrt transform $\operatorname{sign}(s)\sqrt{\max(|s|, \text{clamp\_value})}$ followed by $\sigma(\cdot)$, zero the gate where `image_mask` is active, and output $y = x + g \cdot \text{value}$.
2. **Ascend 950 Direct `engram_gate_fusion` Dual-Route Architecture**:
   - **Weight Dtype Normalization**: When the four weight tensors are not already contiguous `FP32` (e.g., `BF16`/`FP16`/`FP64` combinations), convert them on NPU into a contiguous `FP32` compute image before launching the kernel.
   - **Strict-Fast AIV Vector Route vs SIMT General Route**: Dispatch aligned production hidden dimensions (e.g., `128`-aligned head/channel slices in `fused_causal_conv1d_128dim.h`) to the strict-fast AIV Vector VF pipeline, and route arbitrary strides/layouts to the SIMT general fallback.

### Step 3: Block-Attention Residual Update + RMSNorm (`block_attn_res_update_rms_norm`)
1. **Single-Pass On-Chip Online-Softmax & RMSNorm Fusion**:
   - For each token independently along hidden dimension $D$ in FP32:
     $$\begin{aligned}
     p &= \text{partial\_block} + \operatorname{float32}(\text{delta}) \\
     \text{score} &= \frac{\sum_d p_d \cdot \text{pseudo\_query}_d}{\sqrt{\operatorname{mean}(p^2) + \text{score\_eps}}} \\
     m &= \max(\text{logit\_max}, \text{score}), \quad a = \exp(\text{logit\_max} - m), \quad b = \exp(\text{score} - m) \\
     h &= \frac{a \cdot \text{numerator} + b \cdot p}{a \cdot \text{exp\_sum} + b}
     \end{aligned}$$
   - Keep $h$ strictly in UB (never write $h$ to GM), compute $y = \operatorname{RMSNorm}(h, \text{rms\_eps}) \odot \gamma$, write $y$ to output GM, and write $p$ back in-place to `partial_block` while leaving `numerator`, `logit_max`, and `exp_sum` unmodified.

## Verification & Checks

1. **In-Place Mutation Contract (`block_attn_res_update_rms_norm`)**:
   - Verify that `partial_block` is the *only* input tensor mutated in-place; `numerator`, `logit_max`, `exp_sum`, `delta`, `pseudo_query`, and `gamma` must remain bit-identical after kernel execution.
2. **Signed-Sqrt Clamp Stability (`engram_gate`)**:
   - Verify that `clamp_value` (`1e-6`) is applied inside `sqrt(max(abs(s), clamp_value))` before restoring the sign of $s$ so near-zero dot products neither produce NaN gradients nor lose sign polarity.

## Failure Modes & Fallbacks

- **Symptom**: Sinkhorn matrix rows/columns drift away from `1.0` after 10+ iterations on BF16 inputs.
  - **Root Cause**: Casting intermediate Sinkhorn iterates back to BF16 between row and column normalization steps or failing to add epsilon to zero-sum rows.
  - **Fix**: Keep all $K$ Sinkhorn iterations in FP32 inside UB and downcast to BF16/FP16 only on final GM writeback.
- **Symptom**: Memory bandwidth bottleneck in `block_attn_res_update_rms_norm` when hidden size $D$ is large.
  - **Root Cause**: Materializing the intermediate combined vector $h$ to GM between the online-softmax merge and the RMSNorm reduction.
  - **Fix**: Retain $p$, `numerator`, and $h$ in UB across both reductions so GM traffic is strictly limited to reading inputs once and writing `(partial_block, y)` once.
