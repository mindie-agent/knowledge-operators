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
  - Dynamic Sparse Attention (DSA) index selection via aclnnLightningIndexerV2
domain_path: operators/transformer-fused/attention-mla-indexer
evidence_strength: software_contract
id: unit-ops-transformer-aclnn-lightning-indexer-v2-and-sparse-fia-contract
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md
  source_digest: 64897a4be74f125154d851d1c93facdaf86df32e2ece3a0a10c8da02ab50a912
  source_locator: cann/ops-transformer@4891cf6a210bf08a4a9bb26f7c1e2277ed43c860:attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Specification of `aclnnLightningIndexerV2` in `cann/ops-transformer` covering
  tensor layouts (`BSND`/`TND`/`PA_BSND`), head-weighted ReLU score reduction, Top-K
  sparse index output, and `SFA` handoff.
tier: L2
title: 'cann/ops-transformer Contract: aclnnLightningIndexerV2 and Sparse Flash Attention'
triggers:
  error_signatures:
  - aclnnLightningIndexerV2 contract
  - lightning indexer v2 ops transformer
  keywords:
  - aclnnLightningIndexerV2 contract
  - lightning indexer v2 ops transformer
  - sparse_count 2048 lightning indexer
  - paged key index cache block_table
  - ops-transformer
  - lightning-indexer-v2
  - sparse-flash-attention
  - dsa
  - paged-kv
  - topk-selection
---
## 1. `aclnnLightningIndexerV2` Mathematical & Interface Contract

`aclnnLightningIndexerV2` computes token-to-token importance scores for Dynamic Sparse Attention (`DSA` in DeepSeek-V3.2-Exp / DeepSeek-V4) and outputs the top-`sparseCount` (`k = 2048`) key token indices per query token:
$$S_{t, s} = \sum_{h=0}^{H_q - 1} w_{t, h} \cdot \text{ReLU}\big(\langle q_{t, h}, k_{s, g(h)} \rangle\big), \quad I_t = \text{TopK}_{s \le t}(S_{t, s}, k)$$

### Input & Output Tensor Constraints
1. **`query`**: Shape `[B, S_q, H_q, D]` (`BSND`) or `[T_q, H_q, D]` (`TND`), dtype `BF16` / `FP16` (and `FP8` on Ascend 950PR/DT), where typically $H_q = 64, D = 128$.
2. **`key`**:
   - Contiguous mode: `[B, S_kv, H_kv, D]` or `[T_kv, H_kv, D]`.
   - Paged cache mode (`blockTable` provided): `[num_blocks, block_size, H_kv, D]` (`PA_BSND`), where $H_{kv} = 1$ (MQA index key).
3. **`weights`**: Head importance weights of shape `[B, S_q, H_q]` or `[T_q, H_q]` matching `query` token dimensions.
4. **`actualSeqLengthsQuery` & `actualSeqLengthsKey`**:
   - Per-request lengths in `BSND` or cumulative prefix-sum lengths in `TND`.
5. **`sparseCount` (`k`)**: Number of selected key indices per query token (must satisfy `sparseCount <= 2048` and alignment constraints, typically `2048`).
6. **`sparseIndicesOut` (Output)**: `INT32` tensor of shape `[B, S_q, H_kv, sparseCount]` or `[T_q, H_kv, sparseCount]` containing the selected key positions in `[0, S_kv - 1]`. When actual valid keys $S_{\text{valid}} < \text{sparseCount}$, trailing slots are filled with `-1` or masked by `actualSeqLengthsKey` in downstream `SparseFlashAttention`.

## 2. Tiling & Workspace Sizing Rules

- Uses two-phase invocation: first call `GetWorkspaceSize` to compute temporary buffer size for cross-core Top-$k$ reduction when $S_{kv}$ is split across multiple AIV sub-blocks, then call `aclnnLightningIndexerV2` on the target ACL stream.
- When `layoutQuery="TND"` and `blockTable` is active (Paged KV), `actualSeqLengthsQuery` is cumulative `[T_1, T_1+T_2, ...]` while `actualSeqLengthsKey` gives each sequence's KV context length.

## Verification Checklist

- Ensure `H_q % H_kv == 0` (`64 % 1 == 0`) and `D == 128` (32-byte aligned in `BF16`/`FP16`).
- Verify `sparseIndicesOut` dtype is `torch.int32` before passing it directly into `npu_sparse_flash_attention`.
