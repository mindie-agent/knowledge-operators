---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - 'DeepSeek-V4 AscendC attention kernels: Compressor, LightningIndexer, and SparseAttnSharedKV'
domain_path: operators/model-recipes/deepseek-v4-guides
evidence_strength: reproduced_empirical
id: unit-deepseek-v4-ascendc-compressor-indexer-and-sparse-attn-sharedkv
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_ascendc_operator_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/ops/ascendc/README.md
  source_digest: 07473b71ff5827b30350d5ebce9bfe512cd98ff1e7db5fab5308887f76cfff82
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v4/deepseek_v4_ascendc_operator_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: AscendC operator design and tiling guide for DeepSeek-V4 (`Compressor`, `LightningIndexer`
  with Ascend 950 histogram Top-K, and `SparseAttnSharedKV` with decoupled AICPU metadata
  scheduling).
tier: L2
title: 'DeepSeek-V4 AscendC Fused Operators: Compressor, LightningIndexer 950 Histogram,
  and SparseAttnSharedKV'
triggers:
  error_signatures:
  - deepseek v4 ascendc operator guide
  - npu_sparse_attn_sharedkv_metadata aicpu scheduler
  keywords:
  - deepseek v4 ascendc operator guide
  - npu_sparse_attn_sharedkv_metadata aicpu scheduler
  - sparseattnsharedkv compressor lightningindexer
  - ascend 950pr dt histogram topk lightning indexer
  - deepseek-v4
  - ascendc
  - compressor
  - lightning-indexer
  - sparse-attn-sharedkv
  - aicpu-scheduler
  - fp8
---
## 1. DeepSeek-V4 AscendC Attention Operator Trio (`deepseek_v4_ascendc_operator_guide.md`)

DeepSeek-V4 Attention combines Sliding Window Attention, Compressed KV Attention, and Top-$k$ Sparse Attention under a shared latent KV representation (`SharedKV`). Three custom AscendC kernels implement this pipeline on Atlas A3 and Ascend 950PR/DT:

### A. `Compressor` Fused Kernel
- **Mathematical Role**: Compresses contiguous blocks of historical KV latent tokens into coarse summary representations using strided window pooling/linear projection and writes them into a compressed KV cache pool.
- **Tiling & Offset Write-Back (`错位写入`)**:
  - Splits sequence blocks across AIC/AIV cores by `(batch, kv_head, block_idx)`.
  - Uses offset-aligned MTE3 write-back so partial boundary blocks at sequence tails write directly to their strided destination slots without read-modify-write hazards or host padding.

### B. `LightningIndexer` Enhanced for Compressed KV & Ascend 950 Histogram Top-K
- **Compressed-Mask Adaptation**: Aligns indexer score masking with the compressed KV stride so invalid tail slots in partially filled compression windows are masked to `-inf` prior to Top-$k$.
- **Ascend 950PR/DT Histogram Top-K Acceleration**:
  - On Atlas A3, Top-$k$ (`k=2048` out of `S <= 128K`) uses vectorized bitonic/radix selection in UB.
  - On Ascend 950PR/DT, `LightningIndexer` leverages hardware **Histogram instructions** to bucket FP16/BF16 score distributions in a fast pass, narrow the threshold bin containing the $k$-th largest element, and extract exact top-$k$ indices in a single refinement pass—eliminating multi-pass sorting bottlenecks.

### C. `SparseAttnSharedKV` (`SAS`) & Decoupled `AICPU Scheduler`
- **Unified Multi-Scenario Coverage**: Handles Sliding Window, Compressed KV, Top-$k$ Sparse KV, and their arbitrary combinations in a single kernel supporting both `BF16` and `FP8` (on 950PR/DT).
- **Decoupled `AICPU Scheduler` (`npu_sparse_attn_sharedkv_metadata`)**:
  - Because variable sequence lengths and dynamic sparse block lists create value-dependent core workloads, computing load-balanced tile schedules on `AICore` Either stalls Vector units or breaks static graph capture.
  - Extracting schedule generation into an `AICPU` operator (`metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(...)`) produces a compact tile-task descriptor consumed by `torch.ops.custom.npu_sparse_attn_sharedkv(..., metadata=metadata)`.

## Verification Checklist

- Always execute `npu_sparse_attn_sharedkv_metadata` once per attention layer group and reuse the returned `metadata` tensor across layers sharing the same sparse mask topology.
