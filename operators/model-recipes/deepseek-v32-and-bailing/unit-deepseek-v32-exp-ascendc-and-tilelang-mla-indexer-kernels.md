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
  - DeepSeek-V3.2-Exp AscendC and TileLang custom attention/indexer kernels
domain_path: operators/model-recipes/deepseek-v32-and-bailing
evidence_strength: reproduced_empirical
id: unit-deepseek-v32-exp-ascendc-and-tilelang-mla-indexer-kernels
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_ascendc_operator_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_tilelang_operator_guide.md
  source_digest: 4561ac12c522796dd1182e762f97f4fac8884b0a14568bf172da9b64d2531fa2
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_ascendc_operator_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Operator implementation guide for DeepSeek-V3.2-Exp in AscendC and TileLang
  covering multi-level TensorIR lowering, BiSheng AscendC codegen, LightningIndexer
  tiling, and Sparse Flash Attention CV pipelining.
tier: L2
title: 'DeepSeek-V3.2-Exp AscendC and TileLang Operator Guide: MLA Absorption, Indexer,
  and SFA'
triggers:
  error_signatures:
  - deepseek v3.2 exp ascendc operator guide
  - deepseek v3.2 exp tilelang operator guide
  keywords:
  - deepseek v3.2 exp ascendc operator guide
  - deepseek v3.2 exp tilelang operator guide
  - tilelang ascend lowering bisheng codegen
  - sfa topk 2048 ascendc tiling
  - deepseek-v3.2-exp
  - ascendc
  - tilelang
  - lightning-indexer
  - sfa
  - mla-tiling
  - cv-pipeline
---
## 1. DeepSeek-V3.2-Exp AscendC Operator Design (`deepseek_v3.2_exp_ascendc_operator_guide.md`)

1. **`LightningIndexer` AscendC Kernel Architecture**:
   - **Compute Formulation**: Given query $Q^{\text{idx}} \in \mathbb{R}^{T \times 64 \times 128}$, key cache $K^{\text{idx}} \in \mathbb{R}^{S \times 1 \times 128}$, and head weights $W^{\text{idx}} \in \mathbb{R}^{T \times 64}$, computes `ReLU(Q_idx @ K_idx^T)` on Cube (`AIC`), weights and reduces across the 64 heads on Vector (`AIV`), and selects the top `k=2048` indices per token.
   - **Cube-Vector Ping-Pong Tiling**:
     - Tiles the key sequence dimension $S$ into chunks of $B_S = 512$ or $1024$ so `AIC` continuously streams `[64, 128] x [128, B_S]` GEMMs while `AIV` applies `ReLU`, head-weight multiplication, cross-head reduction, and running Top-$k$ candidate maintenance in double-buffered UB.
2. **`Sparse Flash Attention` (`SFA`) AscendC Kernel Architecture**:
   - Uses the `[T, 2048]` sparse index table to gather non-contiguous latent KV rows (`512` nope + `64` rope) into L1 buffer tiles, then executes absorbed MLA FlashAttention (`Q_nope @ K_nope^T + Q_rope @ K_rope^T -> Online Softmax -> P @ V_c`) across the `2048` selected tokens.

## 2. TileLang-Ascend Lowering & Compilation Pipeline (`deepseek_v3.2_exp_tilelang_operator_guide.md`)

TileLang-Ascend compiles Python tile kernels into NPU binaries via a 6-step workflow:
1. **Multi-Level TensorIR Lowering**: Transforms high-level `T.Kernel` / `T.gemm` / `T.copy` / `T.Parallel` AST into Ascend-specific TensorIR (`L1`, `L0A`, `L0B`, `L0C`, `UB` memory scopes, Cube/Vector split, and pipe barrier injection).
2. **AscendC Codegen**: Emits C++ AscendC source code (`GlobalTensor`, `LocalTensor`, `TPipe`, `DataCopy`, `Mmad`).
3. **BiSheng Compilation (`bisheng`)**: Compiles generated AscendC code into a shared library (`.so`).
4. **Runtime Loading & Python Binding**: Loads `.so` via `torch_npu` runtime and wraps the entry point as a standard Python callable compatible with PyTorch tensors.

## Verification Checklist

- Ensure the gathered `2048` sparse KV indices per token are sorted or grouped by physical page block ID before issuing `DataCopy` from GM to L1/UB to maximize HBM burst coalescing.
