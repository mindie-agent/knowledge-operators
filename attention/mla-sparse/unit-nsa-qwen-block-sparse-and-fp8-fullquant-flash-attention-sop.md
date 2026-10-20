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
  - SparseFlashAttention
  - QwenSparseAttn
  - QuantBlockSparseAttn
  - FlashAttnFP8FullQuant
  - FlashAttn
domain_path: attention/mla-sparse
evidence_strength: reproduced_empirical
id: unit-nsa-qwen-block-sparse-and-fp8-fullquant-flash-attention-sop
inherits_from:
- unit-ascend-fused-infer-attention-and-flash-mla-c8-sop
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  attribution: Adapted from CANNBot-DSL sparse and quantized Flash Attention samples
    and CANNBot-Knowledge attention guides
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/sparse_flash_attention/sparse_flash_attention.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/qwen_sparse_attn/qwen_sparse_attn.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/quant_block_sparse_attn/quant_block_sparse_attn.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_attn_fp8_fullquant/flash_attn_fp8_fullquant.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_attn/flash_attn.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/transformer/attention/sparse_flash_attention_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/qwen_sparse_attn/qwen_sparse_attn.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-mixed-quant-sparse-flash-mla-and-kvcache-dsl-sop
- unit-lightning-indexer-prolog-epilogue-and-qsa-stem-indexer-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for token-level and block-level Sparse Flash
  Attention (SFA/NSA, Qwen Sparse Attention, Quant Block-Sparse Attention) and FP8/BF16
  Flash Attention in CANNBot-DSL and Ascend C.
tier: L2
title: NSA/SFA, Qwen Block-Sparse, Quant Block-Sparse & FP8 Full-Quant Flash Attention
  SOP
triggers:
  error_signatures:
  - Sparse Flash Attention (SFA) uses MLA-absorb mode (V_tilde = K_tilde) over token-level
    sparse_indices with continuous TND or Paged KV cache
  - Qwen Sparse Attention fixes block_size=128 with Q in TND [T, N1, 128], K/V in
    PA_BBND [P, 128, N2, 128], and AICPU single-Query metadata filtering
  - Quant Block Sparse Attention and FlashAttn FP8 FullQuant apply per-token/per-block
    or per-head FP8 descale factors before online softmax and PV accumulation
  keywords:
  - sparse_flash_attention
  - qwen_sparse_attn
  - quant_block_sparse_attn
  - flash_attn_fp8_fullquant
  - flash_attn
  - block_sparse
  - mla_absorb
  - paged_attention
  - gqa_packing
---
## Preconditions & Signals

Apply this SOP when implementing or tuning token-sparse, block-sparse, or FP8 full-quantized Flash Attention kernels on Ascend NPUs:

1. **Token-Level vs Block-Level Sparse Selection**:
   - **Token-Sparse MLA-Absorb (`sparse_flash_attention`)**: Selects arbitrary KV tokens per Query via `sparse_indices`, using MLA-absorb semantics where selected Key serves as Value ($\widetilde{V} = \widetilde{K}$, or split Nope/RoPE where $Q K^\top$ uses full `nope + rope` while $P V$ accumulates only the `nope` slice).
   - **Block-Sparse Paged Attention (`qwen_sparse_attn` & `quant_block_sparse_attn`)**: Selects 128-token KV blocks via `sparse_block_idx` (`int32[N2, T, topK]`) and `sparse_block_count` (`int32[N2, T]`), leveraging contiguous 128-token DMA bursts per physical page in `PA_BBND` `[P, 128, N2, 128]`.
2. **Dense & FP8 Full-Quant Flash Attention (`flash_attn`, `flash_attn_fp8_fullquant`)**:
   - Supports `BNSD`, `BSND`, and variable-length `TND` layouts, GQA (`N1 % N2 == 0`), Paged KV cache, `mask_mode` in `{0: full, 3: right-context causal, 4: sliding window}`, and FP8 (`e4m3fn`/`e5m2`) Q/K/V with descale factors.

## Procedure

### Step 1: Token-Sparse MLA-Absorb Flash Attention (`sparse_flash_attention`)
1. **Separate Score Dimension (`D_qk`) vs Value Dimension (`D_v`)**:
   - When RoPE is enabled in MLA-absorb mode, compute $S = (Q_{\text{nope}} \widetilde{K}_{\text{nope}}^\top + Q_{\text{rope}} \widetilde{K}_{\text{rope}}^\top) \cdot \text{scale}$, then accumulate $O = \operatorname{softmax}(S) \widetilde{K}_{\text{nope}}$.
   - Never overwrite or alias the $\widetilde{K}_{\text{nope}}$ L1 tile with $\widetilde{K}_{\text{rope}}$ before the $P \widetilde{K}_{\text{nope}}$ Cube stage completes.
2. **GQA Query-Group Packing (`N1 / N2`)**:
   - Pack the $G = N_1 / N_2$ Query heads sharing the same KV head into a single Cube $M$-tile (`[G, D]`) so each gathered sparse KV token tile is loaded once from GM into L1 and reused across all $G$ Query heads in L0A/L0B.

### Step 2: Block-Sparse Paged Attention (`qwen_sparse_attn`)
1. **AICPU Single-Query Metadata Filtering & Page Mapping (`qwen_sparse_attn_metadata.py`)**:
   - Pre-filter `sparse_block_idx` (`[N2, T, topK]`) and `sparse_block_count` (`[N2, T]`) on AICPU: discard out-of-causal-bound blocks, deduplicate or sort valid logical blocks, translate logical blocks through `block_table` into physical page indices in `PA_BBND` `[P, 128, N2, 128]`, and balance work across AIC cores.
2. **Main Kernel Execution (`qwen_sparse_attn.py`)**:
   - Validate inputs and metadata ABI via `qwen_sparse_attn_check.py` (`Q` contiguous `TND` `[T, N1, 128]`, `T <= 16384`, `N1 % N2 == 0`, `group <= 128`).
   - Support `BF16`, `FP16`, and `FP8 E4M3` operands; apply partial causal mask only on the boundary block containing `query_pos` while treating strictly interior sparse blocks as unmasked full `128`-token tiles.

### Step 3: FP8 Quantized Block-Sparse & Full-Quant Flash Attention (`quant_block_sparse_attn`, `flash_attn_fp8_fullquant`)
1. **Algebraic Factoring of FP8 Descale Factors**:
   - Given $\hat{Q}_i = Q_i^{\text{fp8}} \cdot s^Q_i$, $\hat{K}_j = K_j^{\text{fp8}} \cdot s^K_j$, and $\hat{V}_j = V_j^{\text{fp8}} \cdot s^V_{\text{head}(j)}$:
   - Execute the Cube $Q^{\text{fp8}} (K^{\text{fp8}})^\top$ MMAD directly in native FP8 hardware units (accumulating into FP32 in L0C).
   - Fuse the effective score scale $\tilde{\alpha}_{i,j} = s^Q_i \cdot s^K_j \cdot \text{softmax\_scale}$ into the Vector online-softmax scaling step rather than dequantizing $Q$ and $K$ in UB prior to Cube MMAD.
   - Convert online softmax probabilities $P_{i,j}$ to FP8 (or keep in BF16/FP16 per kernel path) for the $P V^{\text{fp8}}$ Cube MMAD, and apply $s^V_{\text{head}(j)}$ once per row/block during L0C-to-UB accumulation or final normalization.

### Step 4: Dense Flash Attention Multi-Layout & Mask Engine (`flash_attn`)
1. **Unified Layout Stride Lowering (`BNSD` / `BSND` / `TND`)**:
   - Lower `BNSD`, `BSND`, and varlen `TND` (via `cu_seqlens_q` / `cu_seqlens_k`) into per-head base pointers + `(row_stride, head_stride)` in `flash_attn_metadata.py` after running `flash_attn_validation.py`.
2. **Causal (`mask_mode=3`) & Sliding Window (`mask_mode=4`) Block Skipping**:
   - Compute exact `[kv_block_start, kv_block_end)` loop bounds per Query tile so fully masked KV tiles are skipped before issuing GM-to-L1 DMA, and only boundary tiles invoke Vector mask predicate generation.

## Verification & Checks

1. **Logical Token Identity under GQA & Paged Cache**:
   - Verify that when $G = N_1 / N_2$ Query heads are packed into one $M$-tile, the causal mask comparator uses the single shared token coordinate $t_q$ (not the packed row index $m \in [0, G)$), whereas when multiple Query tokens $T_{\text{tile}}$ are packed, each row uses its true token position $t_q$.
2. **FP8 Overflow & Descale Placement Audit**:
   - Verify via `flash_attn_fp8_fullquant_checker.py` and `qwen_sparse_attn_check.py` that `LSE` is always computed and stored in `float32` and that FP8 conversion of attention weights $P$ occurs after subtracting the tile maximum $m_{\text{new}}$ (so $P \in [0, 1]$ never overflows FP8 range).

## Failure Modes & Fallbacks

- **Symptom**: Severe accuracy degradation on boundary blocks in `qwen_sparse_attn` or `quant_block_sparse_attn`.
  - **Root Cause**: Applying a top-left causal mask (`col <= row_in_tile`) instead of right-down/logical-position causal mask (`block_logical_start + col <= query_logical_pos`) when sparse blocks are non-contiguous.
  - **Fix**: Compare each KV token's global logical position (`logical_block_id * 128 + offset_in_block`) against the Query token's logical position.
- **Symptom**: CV pipeline deadlock in Flash Attention when a Query tile skips all KV blocks or hits a single-block window.
  - **Root Cause**: Hardcoding a 2-stage or 3-stage ping-pong prologue/epilogue that assumes `num_kv_blocks >= 2`.
  - **Fix**: Bound the software pipeline prologue by `min(pipe_depth, num_valid_kv_blocks)` and drain exactly the launched slot count.
