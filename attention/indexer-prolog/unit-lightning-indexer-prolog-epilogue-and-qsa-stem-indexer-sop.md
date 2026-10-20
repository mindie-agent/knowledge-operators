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
  - QuantLightningIndexer
  - QuantSparseLightningIndexer
  - IndexerPrologueK
  - IndexerPrologueQw
  - AttnPrologue
  - AttnEpilogue
  - QuantMlaProlog
  - QsaIndexer
  - StemIndexer
domain_path: attention/indexer-prolog
evidence_strength: reproduced_empirical
id: unit-lightning-indexer-prolog-epilogue-and-qsa-stem-indexer-sop
inherits_from:
- unit-ascend-fused-infer-attention-and-flash-mla-c8-sop
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  attribution: Adapted from CANNBot-DSL Lightning Indexer, QSA/Stem Indexer, and MLA/Indexer
    Prologue/Epilogue samples
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/quant_lightning_indexer_dsl/quant_lightning_indexer_dsl.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/quant_sparse_lightning_indexer_dsl/quant_sparse_lightning_indexer_dsl.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/indexer_prologue_k/indexer_prologue_k.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/indexer_prologue_qw/indexer_prologue_qw.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/attn_prologue/attn_prologue.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/attn_epilogue/attn_epilogue.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/qsa_indexer/qsa_indexer.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/stem_indexer/stem_indexer.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/transformer/attention/mla_prolog_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/quant_lightning_indexer_dsl/quant_lightning_indexer_dsl.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-mixed-quant-sparse-flash-mla-and-kvcache-dsl-sop
- unit-nsa-qwen-block-sparse-and-fp8-fullquant-flash-attention-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for sparse attention indexers (MXFP4 QLI/QSLI,
  QSA Indexer, Stem Indexer) and fused MLA/Indexer prologues and epilogues (IndexerPrologueK,
  IndexerPrologueQw, AttnPrologue, QuantMlaProlog, AttnEpilogue) in CANNBot-DSL and
  Ascend C.
tier: L2
title: Lightning Indexer (QLI/QSLI), Indexer/MLA Prologue, Attn Epilogue & QSA/Stem
  Indexer SOP
triggers:
  error_signatures:
  - QLI and QSLI consume MXFP4 (signed E2M1 + UE8M0 group-32 scale) Q/K tensors with
    g=32 Query heads, 1 Key head, and D=128 on Ascend 950PR/950DT
  - IndexerPrologueK compiles a 2-kernel (Cube GEMM + Vector RMSNorm/RoPE/MXFP4/scatter)
    AOT pipeline keyed by static (H, D, Dr) + storage mode with dynamic T and cache
    strides
  - AttnPrologue and AttnEpilogue fuse MXFP8 Quant32 (group-32 E8M0 scale), RoPE/inverse-RoPE,
    and 2-stage Cube projections on Ascend 950
  keywords:
  - quant_lightning_indexer_dsl
  - quant_sparse_lightning_indexer_dsl
  - indexer_prologue_k
  - indexer_prologue_qw
  - attn_prologue
  - attn_epilogue
  - quant_mla_prolog
  - qsa_indexer
  - stem_indexer
  - mxfp4_e2m1
  - mxfp8_e8m0
---
## Preconditions & Signals

Apply this SOP when building or optimizing the Sparse Attention Indexer pipeline and MLA Prologue/Epilogue fusion kernels on Ascend 950 (`ARCH 3510`):

1. **Two-Stage MXFP4 Lightning Indexer (`QLI` -> `QSLI`)**:
   - **QLI (`quant_lightning_indexer_dsl`)**: Scans visible MXFP4 (`E2M1` + group-32 `E8M0` scale) Key context ($g=32$ Query heads, $1$ Key head, $D=128$, 4 scale groups of 32 elements) to compute weighted correlation scores $R_{t,h,j} = \sum_{a=0}^3 S^Q_{t,h,a} S^K_{j,a} \langle Q^{(4)}_{t,h,a}, K^{(4)}_{j,a} \rangle$, selecting TopK tokens and optionally emitting 8-token candidate blocks for `QSLI`.
   - **QSLI (`quant_sparse_lightning_indexer_dsl`)**: Evaluates only the candidate 8-token logical blocks $c_{t,0}, \dots, c_{t,C_t-1}$ ($\Omega_t = \bigcup_{p=0}^{C_t-1} \{8c_{t,p} \dots 8c_{t,p}+7\} \cap \mathcal{V}_t$) rather than scanning the full $S_2$ context.
2. **QSA & Stem Block Indexers (`qsa_indexer`, `stem_indexer`)**:
   - **QSA Indexer**: Computes $s_{t,j} = \operatorname{BF16}(\sum_{h=0}^3 \operatorname{ReLU}(Q_{t,h} \cdot K_j))$ over 4:1 compressed Key blocks using FP32 accumulation, encodes BF16 scores to `uint16` for hardware TopK, and expands selected blocks + causal tail into raw token indices.
   - **Stem Indexer**: Computes block scores $S_{b,h,i,j} = \frac{Q_{b,h,i} \cdot K_{b,\lfloor h/G \rfloor, j}^\top}{(B_s / T_s)^2} + \text{bias}_{b,\lfloor h/G \rfloor, j}$ (with $B_s=128, T_s=16 \implies D_{\text{block}}=2048$, scale $1/64$) and concatenates $\text{Sink}_i \Vert \operatorname{TopK}(S_i, \widehat{K}_i) \Vert \text{Window}_i$.
3. **Fused Indexer & MLA Prologues / Epilogues**:
   - `indexer_prologue_k`, `indexer_prologue_qw`, `attn_prologue`, `quant_mla_prolog`, and `attn_epilogue` fuse GEMM projections, RMSNorm, RoPE / inverse-RoPE, and dynamic MXFP8/MXFP4 group-32 quantization into minimal kernel launches.

## Procedure

### Step 1: Indexer Key & Query-Weight Prologues (`indexer_prologue_k`, `indexer_prologue_qw`)
1. **`IndexerPrologueK` Two-Kernel Same-Stream AOT Pipeline**:
   - **Kernel 1 (Cube)**: Compute BF16 GEMM $\text{latent}[T, H] \times W_k^\top$.
   - **Kernel 2 (Vector)**: Fuse FP32 RMSNorm $\times \text{norm\_weight}$ (round to BF16) $\rightarrow$ adjacent-pair RoPE on the last $D_r$ dims (round to BF16) $\rightarrow$ group-32 MXFP4 (`signed E2M1` + `UE8M0` scale) quantization $\rightarrow$ paged cache scatter-write via `cache_index[T]`.
   - Cache the compiled AOT pipeline by static `(H, D, Dr, storage_mode)` while keeping `T` and all cache geometry axes (block count, slot count, dim 0/1 strides) as dynamic `Dim` contract variables so varying batch/cache sizes never trigger recompilation.
2. **`IndexerPrologueQw` Split-K & FRACTAL_NZ Pre-Formatting**:
   - Convert weights `wqb` and `ww` once at load time via `to_nz()` to `FRACTAL_NZ`; keep activations in `ND` and handle short-tail $T$ inside L1 without host padding.
   - Fuse MXFP8 $Q$ GEMM $\rightarrow$ tail RoPE $\rightarrow$ MXFP4 (`E2M1` + `E8M0`) quantization alongside BF16 $W$ GEMM $\times \text{softmax\_scale}$ (`dim=5120, q_lora=1280, N=32, D=128, Dr=64`), using deterministic workspace Split-K reduction for $T \in [1, 3072]$.

### Step 2: MXFP4 Lightning Indexer (`QLI` & `QSLI`) Execution & AICPU Scheduling
1. **AICPU `LI` (Compute Range) & `LD` (Cross-Core TopK Merge) Metadata**:
   - In `quant_lightning_indexer_metadata_dsl`, group Queries by up to 6 rows and partition Keys into 256-token tiles; derive causal visible ranges from `cu_seqlens`, `seqused`, `cmp_ratio`, and `cmp_residual_k`.
   - In `quant_sparse_lightning_indexer_metadata_dsl`, schedule each Query independently by partitioning its valid candidate prefix `candidate_block_length[t]` into 64-block tiles (`64 * 8 = 512` candidate tokens per tile).
   - Emit `LI` slot coordinates for local scoring/TopK and `LD` merge descriptors whenever a Query's tiles span multiple AIC cores.
2. **On-Chip MXFP4 Group-32 Dot Product & TopK Merge**:
   - Multiply 4 groups of 32 `E2M1` elements per head with their `E8M0` scales $S^Q_{t,h,a} S^K_{j,a}$, weight across the $g=32$ Query heads, select local TopK per tile, and execute the `LD` cross-core reduction pass for split rows.

### Step 3: Fused MLA Prologue (`attn_prologue`, `quant_mla_prolog`) & Epilogue (`attn_epilogue`)
1. **`attn_prologue` / `quant_mla_prolog` Single-MIX-Kernel Pipeline**:
   - For production geometry $(H, R, N, D, D_r) = (5120, 1280, 64, 512, 64)$, fuse:
     1. Down-projection $q_a = \text{Deq}(x) @ \text{Deq}(W_{qa})^\top$ (`[T, 1280]`) and $kv = \text{Deq}(x) @ \text{Deq}(W_{kv})^\top$ (`[T, 512]`).
     2. Vector RMSNorm + `Quant32` (group-32 `E8M0` MXFP8 dynamic quantization) on $q_a \rightarrow (q_r, \text{descale}_{qr})$ using `vf(mode="simd")`.
     3. Up-projection $q_{\text{proj}} = \text{Deq}(q_r) @ \text{Deq}(W_{qb})^\top$ (`[T, 64*512]`), tail RoPE on $Q$ and $KV$, and paged KV cache writeback.
2. **`attn_epilogue` Two-Stage MXFP8 Projection Fusion**:
   - Given attention output $O$ with 64 heads of dim 512 grouped into `NG=8` groups of 8 heads (`F=4096, O_LORA=1024, DIM=5120`):
   - Apply `inverse_rope` in BF16, quantize every 32 elements to OCP MXFP8 (`Q(A)`), run group GEMM $Y_{t,g} = \operatorname{MXMatmul}(Q(A_{t,g}), W_{oA}[g]^\top)$ into internal workspace, re-quantize concatenated groups to MXFP8, and compute $Out_t = \operatorname{MXMatmul}(Q(\text{concat}(Y_t)), W_{oB}^\top)$.

## Verification & Checks

1. **SIMD Vector Mode Contract (`vf(mode="simd")`)**:
   - Verify that all register vector intervals in CANNBot-DSL prologues (including input padding helpers such as `_MlaInputPadUnit.pad` in `quant_mla_prolog.py`) use `vf(mode="simd")` rather than legacy `raw`/`raw_reg`.
2. **UINT16 Score Encoding for TopK (`qsa_indexer`)**:
   - Verify that BF16 $\rightarrow$ `uint16` order-preserving encoding in `qsa_indexer` maps non-negative ReLU sums monotonically and masks out-of-range causal positions before TopK selection.

## Failure Modes & Fallbacks

- **Symptom**: Recompilation storm on every decode step in `IndexerPrologueK`.
  - **Root Cause**: Passing `T` or KV cache physical block count as compile-time Python ints instead of dynamic `Dim` symbols.
  - **Fix**: Key the AOT cache strictly on `(H, D, Dr, storage_mode)` and bind `T` and cache strides as dynamic `Dim` parameters.
- **Symptom**: Missing TopK indices or uninitialized output rows when `candidate_block_length[t] == 0` in `QSLI`.
  - **Root Cause**: Skipping zero-workload Queries entirely in `quant_sparse_lightning_indexer_metadata_dsl` without scheduling output buffer initialization (`-1` padding).
  - **Fix**: Retain explicit zero-workload output-initialization descriptors in the AICPU metadata schedule.
