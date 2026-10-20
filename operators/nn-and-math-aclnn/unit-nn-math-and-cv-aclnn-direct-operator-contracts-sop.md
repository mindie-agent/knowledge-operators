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
  - Conv2dConv3d
  - ApplyAdamW
  - CrossEntropyLoss
  - RoiAlign
  - GridSampler
  - VoxelConv
  - PointNetSA
  - GruLstm
domain_path: operators/nn-and-math-aclnn
evidence_strength: reproduced_empirical
id: unit-nn-math-and-cv-aclnn-direct-operator-contracts-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
provenance:
  attribution: Adapted from CANNBot-DSL voxel_conv and pointnet_sa samples and CANNBot-Knowledge
    ops/ascendc/operators/direct and aclnn documentation
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/voxel_conv/voxel_conv.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/pointnet_sa/pointnet_sa.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/conv/conv_2d_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/conv/depthwise_conv_2d_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/optim/apply_adam_w_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/loss/cross_entropy_loss_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/rnn/lstm_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/cv/objdetect/roi_align_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/voxel_conv/voxel_conv.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-discovered-evolution-strategies-and-cross-core-pipeline-mutator
- unit-ascendc-interpolation-grid-sample-and-fused-composite-perturbation-mutator
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for NN, Math, and 3D/CV operators across CANNBot-DSL
  (VoxelConv, PointNet Set Abstraction), Ascend C Direct-Invoke (Conv2D/3D, DepthwiseConv2D,
  GRU/LSTM, ApplyAdamW, CrossEntropyLoss, RoiAlign), and ACLNN.
tier: L2
title: NN, Math & 3D/CV ACLNN/Direct/DSL Operator Contracts (Conv, Optim, Loss, VoxelConv,
  PointNetSA) SOP
triggers:
  error_signatures:
  - CANNBot-DSL VoxelConv uses explicit memory hierarchy staging and Load3D hardware
    instructions for NCHW input/output and OIHW weights in FP16/BF16
  - CANNBot-DSL PointNet Set Abstraction (pointnet_sa) fuses shared MLP (1x1 conv
    over points[K, N_per_group, D_in]) with symmetric group max-pooling in a single
    kernel
  - Fused optimizer kernels (apply_adam_w_950) and loss kernels (cross_entropy_loss_950)
    must maintain FP32 master moments/log-sum-exp accumulators even when gradients/logits
    are BF16/FP16
  keywords:
  - voxel_conv
  - pointnet_sa
  - conv_2d_950
  - depthwise_conv_2d_950
  - apply_adam_w_950
  - cross_entropy_loss_950
  - roi_align_950
  - gru_950
  - lstm_950
  - load3d
---
## Preconditions & Signals

Apply this SOP when implementing or optimizing multimodal 3D perception (`VoxelConv`, `PointNet++ Set Abstraction`), 2D/3D Convolution (`conv_2d_950`, `depthwise_conv_2d_950`, `conv_3d_backprop_filter_950`), RNNs (`gru_950`, `lstm_950`), fused optimizers (`apply_adam_w_950`), and losses (`cross_entropy_loss_950`):

1. **Multimodal 3D Perception Kernels in CANNBot-DSL (`voxel_conv`, `pointnet_sa`)**:
   - **`voxel_conv`**: Implements the VoxelNet convolutional middle-layer 2D projection $C[N, C_o, H_o, W_o] = \operatorname{Conv2D}(x[N, C_i, H_i, W_i], \text{filter}[C_o, C_{iG}, K_h, K_w])$ in `float16`/`bfloat16` (`NCHW` activations, `OIHW` weights) using explicit `L1 -> L0` `Load3D` (`im2col` on-the-fly) hardware instructions.
   - **`pointnet_sa`**: Implements PointNet++ Set Abstraction $\text{feat}[K, D_{\text{out}}] = \max_{j \in \text{group}} (W[D_{\text{out}}, D_{\text{in}}] \cdot \text{points}[K, j, D_{\text{in}}])$ in `float16`/`bfloat16`, fusing the shared-weight MLP ($1 \times 1$ conv) with symmetric max-pooling over the `N_per_group` neighbor axis.
2. **Ascend 950 Direct NN/Math/CV Operators (`operators/direct/*`)**:
   - Covers `conv_2d_950`, `depthwise_conv_2d_950`, `conv_3d_backprop_filter_950`, `gru_950`, `lstm_950`, `apply_adam_w_950`, `cross_entropy_loss_950`, `adaptive_avg_pool_3d_950`, and `roi_align_950`.

## Procedure

### Step 1: CANNBot-DSL `VoxelConv` (`Load3D` Pipeline) & `PointNetSA` (MLP + Max-Pool Fusion)
1. **`VoxelConv` Explicit Memory & `Load3D` Staging (`voxel_conv.py`)**:
   - Tile spatial output $(H_o, W_o)$ into $M$-blocks and output channels $C_o$ into $N$-blocks per group.
   - Stage the corresponding input spatial halo window $[H_{\text{band}}, W_{\text{band}}]$ from `NCHW` GM into L1, invoke hardware `Load3D` (`LoadData3D`) to expand the $K_h \times K_w$ sliding window into L0A fractals on-the-fly without GM `im2col` materialization, and multiply against `OIHW` filter tiles in L0B.
2. **`PointNetSA` Fused Cube MLP + Vector Group Max-Pooling (`pointnet_sa.py`)**:
   - View `points[K, N_per_group, D_in]` as a 2D matrix `[K * N_per_group, D_in]` for the Cube GEMM against `weight[D_out, D_in]^T`.
   - Instead of writing the expanded `[K, N_per_group, D_out]` MLP features back to GM and launching a separate reduction kernel, drain each `[K_tile, N_per_group, D_out_tile]` tile to UB/workspace and immediately apply Vector max-reduction along the `N_per_group` dimension to emit compact `feat[K, D_out]`.

### Step 2: Standard vs Depthwise Convolution Routing (`conv_2d_950`, `depthwise_conv_2d_950`, `X22`)
1. **Standard/Grouped Conv2D & Conv3D (`conv_2d_950`, `conv_3d_backprop_filter_950`)**:
   - For $1 \times 1$ stride-1 Conv2D (`P130`/`X19`), bypass `LoadData3D` / `im2col` completely and treat `NCHW` input `[C_i, H*W]` directly as the GEMM $B$ operand.
   - For general $K_h \times K_w$, use `LoadData3D` band windowing in L1 (`P129`) or group-flattened $K_{\text{flat}} = C_i \cdot K_h \cdot K_w$ block-diagonal packing (`X90`) when group count $G$ is large and $C_i / G$ is small.
2. **Depthwise Conv2D (`depthwise_conv_2d_950`, `X22`)**:
   - When `groups == C_in` (`C_iG == 1`), **never** route through `im2col + Cube GEMM` (which wastes $15/16$ of every Cube fractal column). Route directly to the AIV Vector sliding-window pipeline using `Brcb` weight broadcast + `MulAddDst` (`X22`, `X23`, `X25`).

### Step 3: Fused Optimizer (`apply_adam_w_950`), Loss (`cross_entropy_loss_950`) & RNN (`lstm_950`, `gru_950`)
1. **Multi-Tensor In-Place `ApplyAdamW` (`apply_adam_w_950`)**:
   - Load `var`, `m` (first moment), `v` (second moment), and `grad` in lockstep 1D tiles with `BUFFER_NUM = 2`.
   - Upcast `BF16`/`FP16` gradients/params to `FP32` in UB (`A1`), apply decoupled weight decay $\theta \leftarrow (1 - \eta \lambda)\theta$, update $m \leftarrow \beta_1 m + (1 - \beta_1) g$ and $v \leftarrow \beta_2 v + (1 - \beta_2) g^2$, compute bias-corrected step $\Delta \theta = \hat{\eta} m / (\sqrt{v} + \epsilon)$, and write updated `var`, `m`, and `v` back to GM in a single pass.
2. **Online Streaming `CrossEntropyLoss` with Inline Target Capture (`cross_entropy_loss_950`, `X1`)**:
   - Stream vocabulary dimension $V$ in tiles through UB; maintain online `(m, l)` log-sum-exp state in FP32 and capture `logits[target_idx]` inline when the tile covering `target_idx` is already in UB (`X1`), avoiding a separate scalar/DMA readback.
3. **RNN / LSTM / GRU (`lstm_950`, `gru_950`, `X15`, `X16`)**:
   - Batch input-to-hidden GEMM $X W_{ih}^\top$ across all $T$ timesteps into a single Cube Matmul call (`X15`), and partition the per-timestep recurrent gate activations (`sigmoid`, `tanh`) and hidden-to-hidden $h_{t-1} W_{hh}^\top$ across cores along `(batch, hidden_slice)` (`X16`).

## Verification & Checks

1. **NPU Conv2D 12-Significant-Bit Reference Behaviour (`P147`)**:
   - When validating custom Conv2D kernels against `torch_npu` `F.conv2d` FP32 baselines on 910B (`DAV_2201`), remember that default `F.conv2d` on NPU rounds FP32 inputs to `HF32` / `BF16` (`12-sig-bit` RNE) unless `torch.npu.conv.allow_hf32 = False` is set!
2. **`RoiAlign` Bilinear Boundary & Empty-Bin Guard (`roi_align_950`)**:
   - In `roi_align_950`, verify that sampling points with coordinates $<-1.0$ or $> H, W$ are zeroed before bilinear Gather and that pooling bins count $\ge 1$ (`max(roi_bin_grid_h, 1)`).

## Failure Modes & Fallbacks

- **Symptom**: Severe Cube underutilization (`mac_ratio < 0.08`) on Grouped Conv2D with small channels per group ($C_i / G = 4$).
  - **Root Cause**: Padding each group's $K_g = 4 \cdot K_h \cdot K_w$ up to a multiple of 16/32 separately and launching $G$ tiny GEMMs.
  - **Fix**: Flatten groups along the reduction axis or use Vector batched spatial FMA (`X23`, `X90`).
