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
  - MotionCompensation
  - UpsampleBicubic2dAaGrad
  - GridSample
  - ComplexMatMul
  - FusedComposite
  - AddRmsNormDynamicQuant
  - FlashAttentionLargeHeadDim
domain_path: ascendc/optimizations/interpolation-and-composite
evidence_strength: reproduced_empirical
id: unit-ascendc-interpolation-grid-sample-and-fused-composite-perturbation-mutator
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascendc-tiling-pipe-double-buffer-and-sync-sop
- unit-ascendc-simt-and-regbase-vector-programming-sop
provenance:
  attribution: Distilled from CANNBot-Knowledge controlled_code_perturbation/interpolation,
    contraction, and fusedcomposite empirical records and discovered strategies X93,
    X94
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/interpolation/motion_compensation_vectorization_compute_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/interpolation/upsample_bicubic2d_aa_grad_tiling_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/contraction/complex_mat_mul_vectorization_compute_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/fusedcomposite/add_rms_norm_dynamic_quant_tiling_a2_id_0.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/fusedcomposite/flash_attention_score_with_large_head_dim_optimization_a2.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x94.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/controlled_code_perturbation/interpolation/index.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-elementwise-unary-binary-and-mask-predicate-perturbation-mutator
- unit-ascendc-discovered-evolution-strategies-and-cross-core-pipeline-mutator
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Controlled code perturbation mutator for Ascend C spatial interpolation (motion_compensation,
  upsample_bicubic2d_aa_grad, grid_sample), complex contraction (complex_mat_mul),
  and fused composite kernels, applying separable Gather hoisting, GatherMask complex
  deinterleaving, and UB zone time-multiplexing.
tier: L4
title: Ascend C Interpolation (MotionCompensation/Upsample), Contraction & Fused-Composite
  Perturbation Mutator
triggers:
  error_signatures:
  - Separable 2D interpolation (bilinear/bicubic/motion_compensation) must hoist horizontal
    Gather/interpolation to input rows before vertical combination to halve Gather
    count and read each input row once
  - Complex matrix multiplication (complex_mat_mul) must decompose into 4 real GEMMs
    or 3-GEMM Gauss-Karatsuba on Cube rather than scalar/vector complex loops
  - Fused composite kernels must partition UB into non-overlapping lifetime zones
    or multi-stage phases so large head dimensions or multi-norm chains do not spill
    to GM
  keywords:
  - interpolation
  - motion_compensation
  - upsample_bicubic2d_aa_grad
  - complex_mat_mul
  - fusedcomposite
  - separable_gather_hoisting
  - ub_zone_reuse
  - deinterleave_gathermask
---
## Preconditions & Signals

Trigger this mutator when optimizing 2D/3D spatial interpolation (`motion_compensation`, `upsample_bicubic2d_aa_grad`, `upsample_bilinear2d`, `grid_sample`), complex-valued contractions (`complex_mat_mul`), or memory-constrained fused composite operators (`add_rms_norm_dynamic_quant`, `flash_attention_score_with_large_head_dim`):

1. **Redundant 4-Corner Gather & Per-Pixel Scalar Indexing in Interpolation (`motion_compensation`, `upsample_*`)**:
   - The baseline kernel loops over output rows and output column tiles, issuing 4 separate `Gather` instructions (`gA, gB, gC, gD`) and re-reading overlapping input rows $y_0, y_1$ from GM for consecutive output rows (`disc_x94`).
2. **Interleaved Complex Arithmetic Bottleneck (`complex_mat_mul`)**:
   - Complex tensors stored as interleaved `(real, imag)` pairs are multiplied via scalar loops or inefficient lane shifts instead of deinterleaving via `GatherMask` and dispatching planar GEMM/vector FMA.
3. **UB Exhaustion on Large-D Fused Composite Kernels (`fusedcomposite`)**:
   - Large head dimensions ($D \in \{256, 512\}$) or multi-stage norm+quant chains exceed the 192 KB / 256 KB UB limit when all stage buffers are allocated statically without zone reuse (`P42`, `P85`).

## Mutation Moves (Ranked by Expected Gain)

### Mutation 1: Separable Horizontal Gather Hoisting & Input-Row Window Caching (`disc_x94` + `motion_compensation`)
- **Target**: Bilinear/bicubic interpolation and `motion_compensation` kernels where interpolation weights separate as $h_{y} \cdot (w_0 \cdot I[y, x_0] + w_1 \cdot I[y, x_1])$.
- **Transformation**:
  1. **Hoist Horizontal Gather to Input Rows**: Instead of performing 4 Gathers (`(y0,x0), (y0,x1), (y1,x0), (y1,x1)`) per output row, compute the horizontal interpolation $H[y] = w_0 \odot \operatorname{Gather}(I[y], x_0) + w_1 \odot \operatorname{Gather}(I[y], x_1)$ (only 2 Gathers per unique input row $y$) and keep a rolling 2-row (bilinear) or 4-row (bicubic) cache of $H[y]$ in UB.
  2. **Precompute Horizontal Offset & Weight Tables Once**: Since horizontal coordinates $(x_0, x_1)$ and weights $(w_0, w_1)$ depend only on the output column $x$ (invariant across batch, channel, and height $H_{\text{out}}$), precompute and pin the byte-offset table and weight vectors in resident UB buffers during `Init()` (`P33`, `P34`).
  3. **Invert Backward Scatter to Gathered Output Footprint (`upsample_bicubic2d_aa_grad`)**: Partition backward `grad_input` rows across AIV cores so each core owns disjoint `grad_input` rows and accumulates overlapping `grad_output` filter taps locally in UB without cross-core GM atomic contention.

### Mutation 2: `GatherMask` Deinterleaving & Planar Cube/Vector Contraction (`complex_mat_mul`)
- **Target**: `complex_mat_mul` and complex-valued contraction kernels.
- **Transformation**:
  1. Load interleaved complex tiles `[..., 2]` into UB and split into contiguous planar `(A_real, A_imag)` and `(B_real, B_imag)` buffers using two `GatherMask` calls (`mask = 1` for odd/even extraction, `P100`).
  2. Execute planar contractions:
     $$C_{\text{real}} = A_{\text{real}} B_{\text{real}} - A_{\text{imag}} B_{\text{imag}}, \qquad C_{\text{imag}} = A_{\text{real}} B_{\text{imag}} + A_{\text{imag}} B_{\text{real}}$$
     using native Cube `MatmulImpl` (for matrix dimensions) or full-width 256B Vector `Mul`/`Sub`/`Add` (for batched small contractions), then re-interleave `(C_real, C_imag)` on output writeback.

### Mutation 3: On-Chip UB Zone Time-Multiplexing & Large-Head-Dim D-Splitting (`fusedcomposite`, `P42`, `P85`)
- **Target**: Fused composite kernels (`add_rms_norm_dynamic_quant`, `flash_attention_score_with_large_head_dim`).
- **Transformation**:
  1. **Union/Zone Aliasing Over Disjoint Stages (`P42`/`P85`)**: Allocate a single large `TBuf<VECCALC>` arena and bind non-overlapping stage views (`ZoneA` for residual add + variance reduction; `ZoneB` aliasing dead scratch tensors for dynamic quantization scale + rounding).
  2. **Head-Dimension Sub-Tiling for Large $D$ (`flash_attention_score_with_large_head_dim`)**: When $D > 256$ overflows L1/UB under standard $[B_r, D]$ tiling, split $D$ into $D_{\text{sub}} = 128$ K-slices during $Q K^\top$ accumulation in L0C (`isL0CAccum = true`) and stream $V$ in matching $D_{\text{sub}}$ column panels during $P V$, keeping online softmax state $(m, \ell)$ shared across panels.

## Guardrails & Verification

1. **Boundary Coordinate Clamping in Interpolation**:
   - Clamp source coordinates $\hat{x} = \min(\max(x_{\text{src}}, 0), W_{\text{in}} - 1)$ and $\hat{y} = \min(\max(y_{\text{src}}, 0), H_{\text{in}} - 1)$ (or apply zero-padding mask for `padding_mode="zeros"`) *before* scaling by `sizeof(T)` to form `Gather` byte offsets; an unclamped negative coordinate produces an unsigned wrap-around offset and triggers `507035` UB read fault.
2. **Alignment of `Gather` Source Base Address**:
   - The base UB tensor passed to `Gather(dst, src, offsetLocal, ...)` must be 32B-aligned, and every byte offset in `offsetLocal` (`uint32_t`) must be a multiple of `sizeof(T)`.

## Fallbacks

- When input width $W_{\text{in}}$ in `motion_compensation` or `grid_sample` is too large for an entire input row to reside in UB, partition the output into horizontal spatial bands (`band_W`) and clamp/shift the Gather offset table relative to each band's minimum source $x_{\min}$.
