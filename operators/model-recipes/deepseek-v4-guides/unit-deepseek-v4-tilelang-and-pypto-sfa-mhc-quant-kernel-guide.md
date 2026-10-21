---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - TileLang-Ascend and PyPTO custom operators for DeepSeek-V4
domain_path: operators/model-recipes/deepseek-v4-guides
evidence_strength: reproduced_empirical
id: unit-deepseek-v4-tilelang-and-pypto-sfa-mhc-quant-kernel-guide
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_tilelang_operator_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_pypto_operator_guide.md
  source_digest: 2e0bc1aa4dd2a2010d9f606015c0892a28a190f2421bf5e139e7fbb912c20622
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v4/deepseek_v4_tilelang_operator_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Developer guide for DeepSeek-V4 TileLang-Ascend and PyPTO operators covering
  Sparse Flash Attention with anchor blocks, mHC Sinkhorn-Knopp fusion, INT8 GEMM
  dequant epilogue, and dynamic Activation Quantization.
tier: L2
title: 'DeepSeek-V4 TileLang and PyPTO Operator Guide: Sparse Flash Attention, mHC,
  and INT8 GEMM'
triggers:
  error_signatures:
  - deepseek v4 tilelang operator guide
  - deepseek v4 pypto operator guide
  keywords:
  - deepseek v4 tilelang operator guide
  - deepseek v4 pypto operator guide
  - tilelang sparse flash attention anchor
  - tilelang mhc sinkhorn int8 gemm
  - pypto mpmd tile graph deepseek v4
  - deepseek-v4
  - tilelang
  - pypto
  - sparse-flash-attention
  - mhc
  - int8-gemm
  - activation-quant
---
## 1. TileLang-Ascend Compiler Primitives (`deepseek_v4_tilelang_operator_guide.md`)

TileLang-Ascend lowers declarative tile programs into AscendC via four key compiler passes:
1. **Automatic Hardware Pipeline Synchronization**: Inserts `set_flag` / `wait_flag` barriers across `MTE1`, `MTE2`, `MTE3`, `Cube`, and `Vector` pipes automatically.
2. **Memory Planning & UB/L1 Reuse**: Packs non-overlapping tile lifetimes into shared L1/L0/UB SRAM addresses.
3. **Automatic Cube/Vector (`CV`) Instruction Splitting**: Splits mixed matrix-vector loops into decoupled AIC and AIV instruction streams connected by workspace/UB queues.
4. **`T.Parallel` Loop Vectorization**: Maps 1D/2D elementwise and reduction loops over UB tiles to vectorized AscendC intrinsics.

## 2. Four Core DeepSeek-V4 TileLang Operators

1. **Sparse Flash Attention (`SFA` with Attention Anchor Blocks)**:
   - **8-Stage Tile Flow**: (1) Load query tile to L1; (2) Gather sparse KV indices and construct valid block mask; (3) Cube $Q K^\top$ score GEMM; (4) Vector Online Softmax update ($m_{\text{new}}, l_{\text{new}}$); (5) Cube $P V$ context accumulation; (6) Rescale running output; (7) **Attention Anchor Computation** (fusing mandatory initial sink/anchor tokens alongside dynamically indexed sparse blocks); (8) Write final normalized output from UB to GM.
2. **Manifold-Constrained Hyper-Connections (`mHC` matching `TileKernels`)**:
   - **6-Stage UB Flow**: (1) Multi-stream feature scaling; (2) Pre-transform tensor $H^{\text{pre}}$; (3) Post-transform tensor $H^{\text{post}}$; (4) Exponential normalization $M^{(0)} = \exp(\tilde{H}^{\text{res}})$; (5) Iterative row/column Sinkhorn-Knopp normalization in UB; (6) Contracted stream write-back.
3. **`INT8` General Matrix Multiplication (`INT8 GEMM`)**:
   - Executes `INT8 x INT8 -> INT32` in L0C via Cube `Mmad`, transfers `INT32` tiles to UB, converts to `FP32`, multiplies by per-token and per-channel dequantization scales (`x_scale * w_scale`), casts to `BF16`, and stores to GM.
4. **Dynamic Activation Quantization (`Activation Quantization`)**:
   - Computes per-row $\max(|x_i|)$ via `T.reduce_max` in UB, derives `scale = max_val / 127.0`, quantizes `round(x / scale)` to `INT8`, and writes both `x_int8` and `scale` in one GM pass.

## 3. DeepSeek-V4 PyPTO Mega-Kernel Guide (`deepseek_v4_pypto_operator_guide.md`)

- Combines `LightningIndexer`, `Compressor`, and pre-attention Norm/RoPE/Quant operations into an MPMD (`Multiple Program Multiple Data`) tile dependency graph where Cube and Vector cores coordinate via data-driven tile production/consumption without host launch gaps.

## Verification Checklist

- Confirm UB tile sizes in TileLang (`block_M`, `block_N`, `block_K`) keep total double-buffered UB footprint within `192 KB` on Atlas A2/A3 (`256 KB` on Ascend 950).
