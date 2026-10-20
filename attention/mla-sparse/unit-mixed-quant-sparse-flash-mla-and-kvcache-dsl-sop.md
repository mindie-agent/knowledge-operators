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
  - FlashMLA
  - MixedQuantSparseFlashMLA
  - SparseFlashMLA
  - KvCompressEpilog
  - PagedKVCache
domain_path: attention/mla-sparse
evidence_strength: reproduced_empirical
id: unit-mixed-quant-sparse-flash-mla-and-kvcache-dsl-sop
inherits_from:
- unit-ascend-fused-infer-attention-and-flash-mla-c8-sop
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  attribution: Adapted from CANNBot-DSL MLA sample implementations and CANNBot-Knowledge
    direct MLA operator guides
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/mixed_quant_sparse_flash_mla_metadata/mixed_quant_sparse_flash_mla_metadata.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/sparse_flash_mla/sparse_flash_mla.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_mla_with_kvcache/flash_mla_with_kvcache.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/kv_compress_epilog/kv_compress_epilog.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/transformer/attention/mla_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-nsa-qwen-block-sparse-and-fp8-fullquant-flash-attention-sop
- unit-lightning-indexer-prolog-epilogue-and-qsa-stem-indexer-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for implementing and scheduling Flash MLA with
  Paged KV Cache, Non-Quantized Sparse Flash MLA (SMLA), Mixed-Quant Sparse Flash
  MLA (MQSMLA with FP8 original KV + FP4 compressed KV), and KvCompressEpilog in CANNBot-DSL
  and Ascend C.
tier: L2
title: Mixed-Quant Sparse Flash MLA, Sparse Flash MLA & KV Cache Compression DSL SOP
triggers:
  error_signatures:
  - Flash MLA with KV Cache expects Query TND [T, N, 576] (N in {64, 96}), Paged KV
    cache PA_NZ or PA_BBND with page_size=128 and num_kv_heads=1, and Value head_dim=512
  - Mixed-Quant Sparse Flash MLA consumes BF16 Query, FP8 original KV, and optional
    FP4 compressed KV scheduled via AICPU int32[1024] metadata
  - KvCompressEpilog packs [rope bf16 128B][nope fp8 (d-64)B][scale][pad] in-place
    per slot_mapping with 64-element FP8 E4M3 group scales
  keywords:
  - flash_mla_with_kvcache
  - mixed_quant_sparse_flash_mla
  - sparse_flash_mla
  - kv_compress_epilog
  - int32_1024_metadata
  - sinks
  - mla_absorb
  - paged_kv_cache
  - flash_decode_split
---
## Preconditions & Signals

Apply this SOP when designing, optimizing, or debugging Multi-head Latent Attention (MLA) inference kernels on Ascend 910B/910C and Ascend 950 (`ARCH 3510`), covering dense Paged KV Cache MLA, non-quantized Sparse Flash MLA (SMLA), Mixed-Quant Sparse Flash MLA (MQSMLA), and KV cache compression writeback (`KvCompressEpilog`):

1. **MLA Head Geometry Split (`576 = 512 + 64`)**:
   - Query has `head_dim = 576` (`512` latent Nope + `64` RoPE) in `TND` layout `[T, N, 576]` where $N \in \{64, 96\}$ (or absorbed query groups), while the absorbed Value head dimension is `512` and KV head count is `1`.
2. **Five SMLA Modes & Two MQSMLA Modes**:
   - **Non-Quantized SMLA (`sparse_flash_mla`)**: Supports `SWA` (sliding window only, no sparse indices, no compressed KV), `HCA` (dense compressed KV + window), `CSA` (sparse compressed KV + window), `ORI_SPARSE` (sparse original KV only), and `ORI_CMP_SPARSE` (joint sparse original KV + sparse compressed KV) with per-head FP32 Softmax `sinks`.
   - **Mixed-Quant SMLA (`mixed_quant_sparse_flash_mla`)**: Supports `ORI_SPARSE` (BF16 Query vs FP8 original KV selected by sparse indices and page table) and `ORI_CMP_SPARSE` (joint attention over selected FP8 original KV + selected FP4 compressed KV).
3. **Decoupled AICPU Metadata Planning (`int32[1024]`)**:
   - Both SMLA general branch and MQSMLA consume an `int32[1024]` metadata buffer generated asynchronously on the current NPU stream by an AICPU planner (`sparse_flash_mla_metadata` / `mixed_quant_sparse_flash_mla_metadata`) without copying device tensors back to Host CPU.

## Procedure

### Step 1: Dense Flash MLA with Paged KV Cache (`flash_mla_with_kvcache`)
1. **Input & Page Table Contracts**:
   - Accept `query` in `TND` `[T, N, 576]` (`float16` or `bfloat16`), `kv_cache` in `PA_NZ` or `PA_BBND` with `page_size = 128` and `num_kv_heads = 1`, and output `O` in `NTD` `[N, T, 512]` plus `LSE` in `float32`.
   - Support `mask_mode = 0` (unmasked) and `mask_mode = 3` (right-down causal alignment).
2. **Split-KV & Online Softmax Pipeline**:
   - Partition long KV sequences across AIC cores via `flash_mla_with_kvcache_metadata.py` when decode batch $\times$ query groups underutilizes the 32/24 AIC cores.
   - Maintain per-tile running maximum $m_i$ and exponential sum $\ell_i$ in `float32` on AIV; delay final division $O \leftarrow O / \ell$ until the epilogue or cross-split reduction stage so partial outputs and `LSE` remain algebraically composable.

### Step 2: Sparse Flash MLA (`sparse_flash_mla`) Dual-Branch & Sink Softmax Architecture
1. **High-Performance Branch vs General Metadata Branch**:
   - In `sparse_flash_mla()`, dispatch uniform/structured decode shapes directly to `sparse_flash_mla_high_perf.py` (which bypasses AICPU metadata).
   - For ragged/complex multi-source KV combinations (`SWA`, `HCA`, `CSA`, `ORI_SPARSE`, `ORI_CMP_SPARSE`), invoke `sparse_flash_mla_metadata` on AICPU to partition continuous `(batch * kv_head, query_group)` ranges across active AIC cores into `int32[1024]`.
2. **Softmax Sink Initialization (`sinks`)**:
   - When `sinks` (`float32[N_q]`) is provided, initialize each Query head's online softmax running state with $m_0 = \text{sinks}[h]$ and $\ell_0 = 1.0$ (representing $\exp(\text{sinks}[h] - m_0)$) before accumulating any original or compressed KV blocks. Never apply `sinks` after the final normalization division, and ensure empty KV rows still normalize cleanly against $\ell_0$.

### Step 3: Mixed-Quant Sparse Flash MLA (`mixed_quant_sparse_flash_mla`)
1. **AICPU Whole-Row vs Selective FlashDecode (`FD`) Planning**:
   - Run `mixed_quant_sparse_flash_mla_metadata` on the current NPU stream using Query prefix sums (`cu_seqlens_q`) and valid TopK lengths of original/compressed KV.
   - Default to whole-row core assignment; when batch/query row count is small (e.g., selective 72-row FlashDecode criteria), emit split-KV slot descriptors into `int32[1024]` metadata and designate reduction slots to merge partial `(O_partial, lse_partial)` into the caller-preallocated output tensor.
2. **Joint FP8 Original KV + FP4 Compressed KV Pipeline**:
   - In the AICore MIX kernel, gather sparse KV tokens from the paged cache via page table + sparse indices.
   - Dequantize FP8 original KV and optional FP4 (`E2M1` + group scale) compressed KV on-the-fly in UB/L1 before or during QK/PV Cube MMAD, accumulating into a shared FP32 online-softmax state $(m, \ell, O_{\text{acc}})$ across both KV sources.

### Step 4: In-Place KV Cache Compression Epilogue (`kv_compress_epilog`)
1. **Packed Row Memory Layout**:
   - Compress incoming `bfloat16` KV activations (`64 < d <= 8192`, `d % 64 == 0`) and scatter-write in-place into `uint8` KV cache rows indexed by `slot_mapping` (`int32`/`int64`):
     ```text
     [rope bf16: 128B (first 64 elements)] [nope fp8_e4m3: (d - 64)B] [group_scales] [tail_padding]
     ```
2. **Group-64 Dynamic FP8 Quantization & Scale Modes**:
   - Preserve the 64-element RoPE segment in exact `bfloat16` (`128 B`).
   - Partition the Nope segment (`d - 64` elements) into 64-element groups; compute per-group max-abs in FP32 and quantize to `fp8_e4m3`.
   - Support both `quant_mode = 0` (`bfloat16` scale) and `quant_mode = 1` (`e8m0` power-of-two exponent scale, with optional `round_scale=True` upward power-of-two rounding).

## Verification & Checks

1. **Metadata ABI & Stream Ordering**:
   - Verify that `sparse_flash_mla_metadata` and `mixed_quant_sparse_flash_mla_metadata` execute on the active NPU stream with zero host-to-device/device-to-host sync (`is_batch_consistency=True` must be rejected explicitly on unsupported paths rather than silently ignored).
2. **Delayed PV & Multi-Slot State Integrity**:
   - When overlapping Cube $P_{i} V_{i}$ of tile $i$ with Vector online softmax of tile $i+1$, verify that separate UB/L0C slots hold the rescaling factor $\alpha_i = \exp(m_{i-1} - m_i)$ and new score tile $S_{i+1}$ so delayed PV rescaling never reads overwritten running maxima.
3. **KvCompressEpilog Bit-Exactness**:
   - Validate that negative `slot_mapping` entries (padding tokens) are skipped without out-of-bounds GM writes, and that `rope` bytes (`0..127`) match the input `bfloat16` bit-for-bit.

## Failure Modes & Fallbacks

- **Symptom**: NaN or systematic output bias on Query rows where all sparse KV indices are masked out (`valid_topk == 0`).
  - **Root Cause**: Dividing $O_{\text{acc}} = 0$ by $\ell = 0$ when `sinks` is absent or uninitialized, or computing $\exp(-\infty - (-\infty)) = \text{NaN}$ in online softmax rescale.
  - **Fix**: Guard empty rows so $\ell = 0 \implies O = 0, \text{LSE} = -\infty$, and keep `m_prev` finite-guarded (`m_new == -inf ? 0 : exp(m_prev - m_new)`).
- **Symptom**: High AIC idle time in decode with small batch size ($T \le 32$) in `mixed_quant_sparse_flash_mla`.
  - **Root Cause**: Whole-row scheduling assigns at most $T$ tasks, leaving remaining AIC cores idle while long sparse TopK lists serialize on a few cores.
  - **Fix**: Enable the selective FlashDecode (`FD`) split plan in `mixed_quant_sparse_flash_mla_metadata` so long TopK KV lists are partitioned across all AIC cores and reduced deterministically.
