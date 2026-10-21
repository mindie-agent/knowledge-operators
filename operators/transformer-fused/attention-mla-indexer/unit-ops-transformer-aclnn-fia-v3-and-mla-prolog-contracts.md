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
  - aclnnFusedInferAttentionScoreV3 and aclnnMlaPrologV2/V3 in cann/ops-transformer
domain_path: operators/transformer-fused/attention-mla-indexer
evidence_strength: software_contract
id: unit-ops-transformer-aclnn-fia-v3-and-mla-prolog-contracts
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/attention/mla_prolog_v3/docs/torchapi_mla_prolog.md
  source_digest: cbfc50be1e2eef6c7f9b366a780b1bf7937c1e38eb38f7979c060a5baa05a598
  source_locator: cann/ops-transformer@4891cf6a210bf08a4a9bb26f7c1e2277ed43c860:attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Software contract and parameter constraints for `aclnnFusedInferAttentionScoreV3`
  (`FIA V3`) and `aclnnMlaProlog` (`V2/V3/WeightNz`) in `cann/ops-transformer` across
  Prefill, Decode, PagedAttention, and MLA.
tier: L2
title: 'cann/ops-transformer Contract: aclnnFusedInferAttentionScoreV3 and aclnnMlaPrologV2/V3'
triggers:
  error_signatures:
  - aclnnFusedInferAttentionScoreV3 contract
  - aclnnMlaPrologV2WeightNz aclnnMlaPrologV3
  keywords:
  - aclnnFusedInferAttentionScoreV3 contract
  - aclnnMlaPrologV2WeightNz aclnnMlaPrologV3
  - fia v3 sparse_mode actual_seq_lengths_kv
  - mla prolog fused qkv down up projection rope
  - ops-transformer
  - fia-v3
  - mla-prolog
  - mla-preprocess
  - paged-attention
  - tnd-layout
  - weight-nz
---
## 1. `aclnnFusedInferAttentionScoreV3` (`FIA V3`) Execution Contract

`aclnnFusedInferAttentionScoreV3` is the unified two-phase CANN attention operator (`GetWorkspaceSize` + `Execute`) covering FlashAttention Prefill, PagedAttention Decode, MLA Absorbed Attention, and `INT8`/`FP8` KV Cache Quantization (`C8`):
1. **Supported Input Layouts (`inputLayout`)**:
   - Contiguous KV: `BSH`, `BSND`, `BNSD`, `TND` (variable-length packed sequences), `NTD`.
   - Paged Attention (`blockTable != nullptr`): `PA_BSND`, `PA_BNSD`, `PA_BSH`, `TND`.
2. **`actualSeqLengths` vs. `actualSeqLengthsKv` Invariant**:
   - In `BSH`, `BSND`, `BNSD`: each entry is the **per-batch sequence length** $S_b$.
   - In `TND`: each entry MUST be the **cumulative prefix sum** $\sum_{i=0}^b S_i$ (`int64` array), with the final element equal to $T_{\text{total}}$.
3. **`sparseMode` Masking Semantics**:
   - `sparseMode=0`: Default mask (uses `pseShift` / `attenMask` over `[preTokens, nextTokens]`).
   - `sparseMode=2`: Left-Up aligned causal mask (`2048 x 2048` compressed lower-triangular boolean/uint8 mask).
   - `sparseMode=3`: Right-Down aligned causal mask (required for **chunked prefill** and **MTP speculative verification** where $S_q < S_{kv}$ and the last query token aligns with the last key token).
   - `sparseMode=4`: Band mask within `[preTokens, nextTokens]`.
4. **MLA Absorbed Mode (`queryRope` & `keyRope`)**:
   - Accepts separate `query` (`nope`, `D=512`) + `queryRope` (`D=64`) and `key` (`kv_c` cache, `D=512`) + `keyRope` (`k_pe` cache, `D=64`) tensors so $Q_{\text{nope}} K_{\text{nope}}^\top + Q_{\text{rope}} K_{\text{rope}}^\top$ is accumulated on-chip in L0C without concatenating 576-dim tensors in HBM.

## 2. `aclnnMlaProlog` (`V2` / `V3` / `WeightNz`) Fused Pre-Attention Contract

`MlaProlog` fuses the entire DeepSeek MLA pre-attention subgraph (`10+` small ops) into a single kernel:
- **Fused Pipeline**:
  1. Down-projects hidden state $x$ into $q_c = x W^{DQ}$ and $(kv_c, k_{\text{pe}}^{\text{raw}}) = x W^{DKV}$.
  2. Applies `RMSNorm` (and optional dynamic quantization) to $q_c$ and $kv_c$.
  3. Up-projects $q_c$ via $W^{UQ}$ and absorbs $W^{UK}$ into $q_{\text{nope}}$ (`[B, S, N, 512]`).
  4. Applies RoPE (`cos`, `sin`) to $q_{\text{rope}}$ (`[B, S, N, 64]`) and $k_{\text{pe}}$ (`[B, S, 1, 64]`).
  5. Writes $kv_c$ and $k_{\text{pe}}$ directly into Paged KV Cache slots (`slotMapping`).
- **`WeightNz` Acceleration**: Pre-converting `W_dq`, `W_uq_uk`, and `W_dkv_kr` weights to `FRACTAL_NZ` layout (`torch_npu.npu_format_cast(..., 29)`) eliminates MTE1 format conversion overhead on AICore.

## Verification Checklist

- Always pass `sparseMode=3` with the shared `2048 x 2048` compressed causal mask during chunked prefill and MTP multi-token verification.
