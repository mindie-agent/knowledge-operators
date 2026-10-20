---
category: mutator
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
  - Ascend950PR
  - Ascend950DT
  workloads:
  - FlashAttention
  - GroupedMatmul
  - QuantMatmul
  - LinearRecurrentSSM
  - Conv2dIm2col
  - DeconvCol2im
  - RnnLstmGru
domain_path: ascendc/optimizations/evolutionary-mutators
evidence_strength: reproduced_empirical
id: unit-ascendc-discovered-evolution-strategies-and-cross-core-pipeline-mutator
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascendc-sync-audit-and-pipe-hazard-elimination-sop
provenance:
  attribution: Distilled from the 89 discovered strategies (disc_x1 to disc_x98) and
    advanced cards (P102 to P147) in CANNBot-Knowledge contrib/ascendc/optimizations/evolution_strategies
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x9.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x10.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x12.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x83.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x85.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p103_cube_neumann_triangular_solve.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p128_fp32_via_bf16_hilo_3gemm.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p129_fused_img2col_loaddata3d.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/index.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-evolutionary-strategy-vector-simd-and-ub-tiling-mutator
- unit-ascendc-cube-l1-residency-l0-pingpong-and-fixpipe-fusion-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Advanced structural mutator distilling empirically discovered Ascend C evolution
  strategies (X1-X98 and P102-P147), covering algebraic deferred scaling, Cube Neumann/divide-and-conquer
  triangular inversion, FP32-via-BF16/FP16 Hi/Lo GEMM decomposition, fused LoadData3D
  im2col, and decoupled AIC/AIV CV software pipelines.
tier: L4
title: Ascend C Discovered Evolution Strategies (X1-X98 & P102-P147) Cross-Core CV
  & Algebraic Mutator
triggers:
  error_signatures:
  - On 910B (DAV_2201), Cube Mmad does not natively consume FP32 inputs; high-precision
    FP32 GEMM on Cube requires BF16 Hi/Lo 3-GEMM (P128/X17) or FP16 Hi/Lo 4-GEMM (X37)
  - On 910B2C, same-launch MTE3 GM write followed by MTE2 GM read across cores without
    explicit cross-core sync/cache coherence is unsafe (P145)
  - Triangular linear solvers (I + T)^-1 in chunk-scan SSM/KDA must replace O(C^2)
    scalar forward substitution with Cube Neumann expansion (P103) or divide-and-conquer
    block inversion (P135)
  keywords:
  - discovered_strategies_x1_x98
  - p103_cube_neumann_solve
  - p128_fp32_via_bf16_3gemm
  - p129_loaddata3d_im2col
  - p132_chunk_scan_three_stage
  - p135_delta_rule_inverse
  - x9_deferred_scaling
  - x10_device_group_schedule
  - x12_quant_gemv_vec
  - x85_deferred_rescaling_softmax
---
## Preconditions & Signals

Trigger this mutator when standard local vector/tiling tweaks plateau and the operator requires an **algorithmic or cross-engine (`AIC <-> AIV`) structural transformation** (`disc_x1`–`disc_x98`, `P102`–`P147`):

1. **Vector-Simulated Matrix / Triangular / Recurrent Chains (`P102`, `P103`, `P120`, `P132`–`P136`, `X15`, `X44`)**:
   - Kernel simulates dot products, GEMM, or triangular forward substitution $(I + T)^{-1}$ on AIV Vector/Scalar units instead of migrating matrix contractions to Cube (`AIC`).
2. **Quantized Matmul / Decode GEMV Engine Mismatch (`X9`, `X10`, `X11`, `X12`, `X13`, `X14`)**:
   - Small-$M$ ($M \in [1, 8]$) weight-quantized GEMV launches heavy Cube pipelines with poor L0B utilization, or dequantizes `INT8 -> FP16` inside the $K$ loop on every step.
3. **Materialized `im2col` / `col2im` / Workspace Round-Trips (`P129`, `P130`, `X18`, `X19`, `X26`, `X29`, `X34`, `X96`, `X98`)**:
   - Convolution or transposed convolution materializes expanded `im2col` buffers or weight-repack buffers in GM (`70%+` of total runtime).

## Mutation Moves (Ranked by Expected Gain)

### Mutation 1: Algebraic Deferred Scaling & Shape-Adaptive Dual-Engine Routing (`X9`, `X10`, `X11`, `X12`, `X13`, `X14`, `X87`)
- **Target**: Weight-quantized Matmul (`y = x @ ((w + offset) * scale) + bias`) and Grouped Matmul.
- **Transformation**:
  1. **Deferred Scaling & Zero-Point Factoring (`X9`, `X12`)**: Factor per-channel `scale[n]` outside the $K$-reduction loop:
     $$y_{m,n} = \text{scale}_n \cdot \left(\sum_k x_{m,k} w_{k,n} + \text{offset}_n \sum_k x_{m,k}\right) + \text{bias}_n$$
     so the inner $K$ loop performs pure integer/FP16 multiply-accumulate without per-K dequantization.
  2. **Small-$M$ Pure Vector Streaming vs Large-$M$ Mix-Kernel Cube (`X12`, `X13`, `X14`)**: Route $M \le 8$ (GEMV) to a pure AIV Vector streaming kernel (broadcasting $x_{m,k}$ across $N$-lanes via `Brcb`), and route $M > 8$ to a pairwise `AIV-Producer (dequant to L1/workspace) -> AIC-Consumer (Cube MMAD)` MIX kernel (`X14`).
  3. **Device-Side Group Scheduling + `ND-int32` Drain (`X10`, `X11`, `X87`)**: For `GroupedMatmul` with device `group_list`, parse `(group_idx, m_block_idx)` directly on each AIC core without host sync (`X10`), and drain `INT32` L0C accumulators in `ND` format (`X11`, `X87`) to avoid `c220` multi-fragment `NZ` hardware bugs.

### Mutation 2: Linear-Recurrent / Chunk-Scan 3-Stage CV Pipeline & Divide-and-Conquer Inversion (`P103`, `P132`–`P136`, `X15`, `X82`, `X83`)
- **Target**: Chunk-scan SSM / Delta-Rule / KDA / RNN / LSTM / GRU kernels.
- **Transformation**:
  1. **Hoist Non-Recurrent Projections to Batched Cube GEMM (`X15`)**: In RNN/LSTM/GRU/SSM, hoist input projections $X W_{ih}$ across all $T$ timesteps into a single `[B*T, H_in] @ [H_in, 4*H]` Cube GEMM before entering the sequential time loop, and split the recurrent loop across cores along `batch * hidden_slice` (`X16`).
  2. **Block Divide-and-Conquer or Neumann Triangular Solver (`P103`, `P135`)**: To invert the $C \times C$ unit lower-triangular matrix $(I + T)^{-1}$ ($T$ strictly lower triangular, $C=64$):
     - Partition into $16 \times 16$ diagonal blocks $A_{ii}$ (solved via short vector forward substitution or 4-step Neumann product $(I - T)(I + T^2)(I + T^4)(I + T^8)$ on Cube, `P103`) and compute off-diagonal blocks via Cube GEMM $A_{ij} = -A_{ii} \sum_{k=j}^{i-1} T_{ik} A_{kj}$ (`P135`).
  3. **Decoupled Dual-Stream AIC/AIV Software Pipeline (`P133`, `X83`)**: Decouple the inter-chunk state chain from intra-chunk precomputation using 6-flag directional AIC/AIV handshakes so Cube computes chunk $c+1$'s $K K^\top / Q K^\top$ while Vector updates chunk $c$'s recurrent state $S_c$ Resident in UB (`P136`).

### Mutation 3: High-Precision FP32-via-BF16/FP16 Multi-GEMM & Fused `LoadData3D` Convolution (`P128`–`P130`, `X17`–`X20`, `X29`, `X34`, `X96`, `X98`)
- **Target**: FP32-input GEMM on 910B Cube and 2D/3D Conv / ConvTranspose kernels.
- **Transformation**:
  1. **FP32-via-BF16 3-GEMM (`P128`/`X17`) or FP16 4-GEMM (`X37`)**: Since 910B Cube `Mmad` does not natively consume `FP32` without `HF32` truncation, split $x = x_{\text{hi}} + x_{\text{lo}}$ into high and low `BF16` (or scaled `FP16`) components in UB, run 3 (or 4) Cube `Mmad` accumulations into `FP32` L0C (`A_hi @ B_hi + A_hi @ B_lo + A_lo @ B_hi`), and drain `FP32` from L0C to achieve near-IEEE-FP32 accuracy at Cube throughput.
  2. **On-the-Fly `LoadData3D` `im2col` & Identity $K$-Order (`P129`, `P130`, `X96`, `X98`)**: Eliminate GM `im2col` materialization by loading spatial feature bands into L1 and expanding sliding windows on-the-fly into L0A via hardware `LoadData3D` (`P129`). Order the GEMM $K$-axis as `(ci, kh, kw)` so native `[Co, Ci, Kh, Kw]` weights are already contiguous without a separate weight-repack kernel (`X98`), or for $1 \times 1$ Conv flip GEMM operands directly (`P130`/`X19`).
  3. **Stride-1 Tap-Combination & Phased `col2im` (`X29`, `X34`)**: For `ConvTranspose`, replace per-pixel scatter `col2im` with phased row reductions directly over the GEMM intermediate matrix in UB.

### Mutation 4: FlashAttention Deferred Rescaling, MLA Concat-K & Workspace Zero-Init (`P124`, `P125`, `P137`, `P139`, `X1`, `X85`)
- **Target**: FlashAttention, MLA, and Log-Softmax/Cross-Entropy kernels.
- **Transformation**:
  1. **Inline Target Capture (`X1`) & Deferred-Rescaling Online Softmax (`X85`)**: In Cross-Entropy / NLL fused softmax, capture `target_logit` inline during the streaming tile scan (`X1`). In FlashAttention where a stable reference max $m_{\text{ref}}$ is known or bounded, freeze rescaling across interior tiles and rescale $O$ only when the running max excursions past threshold (`X85`).
  2. **MLA `nope + rope` Single-GEMM K-Concatenation (`P124`) & Compact Top-K Cube (`P137`)**: Concatenate `Q_nope | Q_rope` and `K_nope | K_rope` as consecutive $K$-slices accumulating into the *same* L0C tile (`cInit=false` on the `rope` slice, `P124`). For token-sparse Top-K attention, gather the $K$ selected KV tokens into a compact workspace buffer on AIV so AIC executes dense Cube `Mmad` rather than falling back to Vector dot products (`P137`).

## Guardrails & Verification

1. **Multi-Slot Workspace Full Zero-Initialization (`P139`)**:
   - When using 2-slot or 3-slot CV ping-pong workspace buffers in GM, initialize *all* $k_{\text{slots}}$ buffers for padded tail rows (`ZeroWorkspaceRows`), never just slot 0—otherwise uninitialized `NaN` values in slot 1 corrupt `L0C`/`UB` rescaling on odd iterations.
2. **Host `at::empty` vs `at::zeros` Discipline (`X39`)**:
   - Replace `at::zeros` with `at::empty` *only* for workspace/output regions that are provably 100% overwritten by the kernel before any read or atomic add.

## Fallbacks

- On `910B2C` when a single kernel launch would require Core A to write GM via `MTE3` and Core B to read that same GM region via `MTE2` without hardware-coherent cross-core sync, split into multi-launch pure AIV/AIC stages (`P145`).
