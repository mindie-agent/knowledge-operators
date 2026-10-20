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
  - FlashAttentionScore
  - FusedInferAttentionScore
  - IncreFlashAttention
  - BlockSparseAttention
  - MC2AllGatherMatmul
  - MC2MatmulReduceScatter
  - MhcPrePost
  - NormRopeConcat
domain_path: operators/transformer-aclnn
evidence_strength: reproduced_empirical
id: unit-transformer-aclnn-and-direct-attention-mc2-gmm-moe-operator-contracts-sop
inherits_from:
- unit-ascend-fused-infer-attention-and-flash-mla-c8-sop
- unit-ascend-grouped-matmul-swiglu-and-mc2-fusion-sop
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
provenance:
  attribution: Adapted from CANNBot-Knowledge ops/ascendc/operators/aclnn/transformer
    and operators/direct/transformer documentation
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/attention/fused_infer_attention_score_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/attention/flash_attention_score_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/attention/block_sparse_attention_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/mc2/all_gather_matmul_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/mc2/allto_all_quant_matmul_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/posembedding/norm_rope_concat_950.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/attention/fused_infer_attention_score_950.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-moe-dispatch-gmm-combine-and-alltoallv-routing-sop
- unit-mhc-pre-post-sinkhorn-knopp-and-engram-gate-fusion-sop
- unit-ascendc-shmem-symmetric-memory-and-hccl-communication-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure and operator contract guide for the CANN Transformer
  ACLNN and Direct-Invoke families across Attention, MC2 (Matmul + Collective Communication),
  Grouped Matmul, MoE, MHC (Manifold Hyper-Connections), and Fused Norm/RoPE Cache
  on Ascend 910B/910C and Ascend 950.
tier: L2
title: Transformer ACLNN & Direct Operator Contracts (Attention, MC2, GMM, MoE, MHC
  & PosEmbedding) SOP
triggers:
  error_signatures:
  - ACLNN two-stage execution (aclnn*GetWorkspaceSize + aclnn*) requires exact layout/dtype
    alignment between aclTensor descriptors and underlying physical storage shapes
  - MC2 operators (AllGatherMatmul, MatmulReduceScatter, AlltoAllMatmul, AlltoAllQuantMatmul)
    require valid HCCL group handles and symmetric rank shape contracts across the
    communication domain
  - Transformer MHC ACLNN suite (mhc_pre_sinkhorn, mhc_post, mhc_pre_backward, mhc_post_backward)
    and posembedding suite (norm_rope_concat, qkv_rms_norm_rope_cache) enforce strict
    stream/head dimension alignment on 910B/910C and 950
  keywords:
  - aclnn_flash_attention_score
  - aclnn_fused_infer_attention_score
  - aclnn_block_sparse_attention
  - all_gather_matmul
  - matmul_reduce_scatter
  - allto_all_quant_matmul
  - mhc_pre_sinkhorn
  - mhc_post
  - norm_rope_concat
  - qkv_rms_norm_rope_cache
---
## Preconditions & Signals

Apply this SOP when selecting, integrating, or customizing CANN `ops-transformer` ACLNN and Direct-Invoke operators across Ascend 910B/910C (`DAV_2201`) and Ascend 950 (`DAV_3510`):

1. **Attention Family Selection (`FlashAttentionScore` vs `FusedInferAttentionScore` vs `BlockSparseAttention` vs `NsaCompress`)**:
   - **Training Forward & Backward**: `aclnnFlashAttentionScore` / `aclnnFlashAttentionVarLenScore` + `aclnnFlashAttentionScoreGrad` / `aclnnFlashAttentionUnpaddingScoreGrad` (consumes `softmax_max`, `softmax_sum`, and `attention_out` saved from forward).
   - **Inference Unified Prefill + Decode (`FIA` / `IFA`)**: `aclnnFusedInferAttentionScore` (supports Paged KV Cache, MLA absorbed `576/512` dims, INT8/FP8 KV cache quantization, and FlashDecode split-KV via `aclnnAttentionUpdate`).
   - **Sparse & Linear Attention**: `aclnnBlockSparseAttention` (+ `Grad`), `aclnnNsaCompress` / `aclnnNsaCompressAttentionInfer`, and `aclnnChunkGatedDeltaRule` + `aclnnFusedCausalConv1d`.
2. **MC2 Compute-Communication Fusion Suite (`operators/aclnn/transformer/mc2/*`)**:
   - Covers `AllGatherMatmul`, `MatmulReduceScatter`, `MatmulAllReduce`, `AlltoAllMatmul`, `AlltoAllQuantMatmul`, and `AlltoAllAllGatherBatchMatMul` across TP and EP groups.
3. **MHC & Fused Norm/RoPE/KV-Cache Prologues (`mhc/*`, `posembedding/*`)**:
   - Covers `mhc_pre_sinkhorn`, `mhc_post`, `mhc_pre_backward`, `mhc_post_backward`, `norm_rope_concat`, and `qkv_rms_norm_rope_cache`.

## Procedure

### Step 1: Attention ACLNN & Direct Contract Compliance
1. **Layout & `actualSeqLengths` / `actualSeqLengthsKv` Representation**:
   - Match the `inputLayout` string (`"BNSD"`, `"BSND"`, `"BSH"`, `"TND"`, `"NTD"`) to the exact tensor dimension order.
   - In `"TND"` varlen mode, pass cumulative prefix-sum sequence lengths (`cu_seqlens`, shape `[B]` or `[B+1]` per operator contract) in `actualSeqLengths` / `actualSeqLengthsKv`; in non-`"TND"` padded modes, pass per-batch valid lengths.
2. **Sparse Mode & Causal Mask Alignment (`sparseMode` $\in \{0..4\}$)**:
   - `sparseMode = 0`: default mask using full `attenMask` shape.
   - `sparseMode = 2` (left-up causal) vs `sparseMode = 3` (right-down causal, required for decode/chunked prefill with KV cache) vs `sparseMode = 4` (band/sliding window between `preTokens` and `nextTokens`): pass the compressed `2048 x 2048` boolean/uint8 lower-triangular mask tensor rather than materializing a full `[B, N, S_q, S_kv]` mask in GM.
3. **Split-KV & `AttentionUpdate` Merge**:
   - When `FusedInferAttentionScore` runs in split-KV FlashDecode mode (or multi-source KV mode), combine partial attention outputs $\{O^{(s)}\}$ and log-sum-exp states $\{\text{LSE}^{(s)}\}$ via `aclnnAttentionUpdate` in FP32.

### Step 2: MC2 (`AllGatherMatmul`, `MatmulReduceScatter`, `AlltoAllQuantMatmul`) Integration
1. **Symmetric Shape & Tile Granularity Contract**:
   - In `aclnnAllGatherMatmul`, input `x1` on each rank has shape `[M / TP, K]`, weight `x2` has shape `[K, N]`, and output `y` has shape `[M, N]` (with optionalGathered `gather_out` `[M, K]`). Ensure `M / TP` is aligned to Cube tile granularity (or padded equally across all TP ranks) because collective RMA transfers require identical shard sizes across ranks.
   - In `aclnnMatmulReduceScatter`, input `x1` has `[M, K]`, weight `x2` has `[K, N]`, and scattered output `y` has `[M / TP, N]`.
   - In `aclnnAlltoAllQuantMatmul` (`_950`), fuse expert/sequence All-to-All permutation with INT8/FP8/MXFP8 quantized GEMM by providing per-rank send/receive split sizes and matching scale tensors.

### Step 3: MHC (`mhc_pre_sinkhorn`, `mhc_post`) & Fused Norm-RoPE-Cache (`norm_rope_concat`, `qkv_rms_norm_rope_cache`)
1. **MHC Forward & Backward Stream Contracts (`mhc/*`)**:
   - Use `mhc_pre_sinkhorn` before the residual sub-layer to compute doubly stochastic Sinkhorn stream mixing weights $H_{\text{res}}, H_{\text{pre}}$ and mix the $n$-stream residual state into the sub-layer input.
   - Use `mhc_post` after the sub-layer to project and merge the sub-layer output back into the $n$-stream residual state, and pair with `mhc_pre_backward_950` / `mhc_post_backward_950` during training so stream mixing intermediates are recomputed/reduced in UB rather than saved in GM.
2. **Fused Norm + RoPE + Concat / Cache Writeback (`norm_rope_concat_950`, `qkv_rms_norm_rope_cache`)**:
   - Fuse Q/K RMSNorm, RoPE rotation on the rotary sub-dimension $D_r$, concatenation with the non-rotary sub-dimension $D_{\text{nope}}$ (`norm_rope_concat`), and direct scatter into the Paged KV Cache (`qkv_rms_norm_rope_cache`) in a single kernel launch.

## Verification & Checks

1. **`aclTensor` Physical Storage Shape vs View Shape (`561103`)**:
   - Before invoking any `aclnn*GetWorkspaceSize`, verify whether input tensors are non-contiguous PyTorch views (e.g., `q, k, v = qkv.chunk(3, dim=-1)` where `storage_offset` and `stride` differ from a contiguous tensor). Only operators that explicitly declare strided view support in their `OpParamDef` can consume non-contiguous views directly; otherwise fuse the split into `qkv_rms_norm_rope_cache` or materialize contiguous slices.
2. **`softmax_max` / `softmax_sum` 8-Element Per-Row Stride in `FlashAttentionScore`**:
   - Note that CANN `FlashAttentionScore` stores `softmax_max` and `softmax_sum` with trailing dimension `8` (replicated across an entire 32B datablock of 8 `float32` elements per `(B, N, S)` row for hardware `Brcb` compatibility); never pass a `[B, N, S, 1]` buffer when `[B, N, S, 8]` is required by the ACLNN contract.

## Failure Modes & Fallbacks

- **Symptom**: `EZ9999` / `561002` parameter validation error in `aclnnFlashAttentionScore` when `sparseMode = 3`.
  - **Root Cause**: Passing a per-head or non-`2048x2048` `attenMask` tensor when `sparseMode` $\in \{2, 3, 4\}$ mandates the compressed `[2048, 2048]` `bool`/`uint8` mask shape.
  - **Fix**: Allocate a single static `[2048, 2048]` upper-right (`logical_not(tril)`) boolean mask tensor on NPU and share it across all layers with `sparseMode = 3`.
