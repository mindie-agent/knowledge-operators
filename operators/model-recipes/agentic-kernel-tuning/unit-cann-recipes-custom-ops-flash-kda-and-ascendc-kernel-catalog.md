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
  - Custom AscendC, TileLang, PyPTO, and cannbot_dsl (Flash KDA) kernels in cann-recipes-infer/ops
domain_path: operators/model-recipes/agentic-kernel-tuning
evidence_strength: reproduced_empirical
id: unit-cann-recipes-custom-ops-flash-kda-and-ascendc-kernel-catalog
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-triton-ascend-constexpr-and-specialization-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/ops/cannbot_dsl/flash_kda.py
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/ops/ascendc/README.md
  source_digest: e66d8c48bd1e32c9c59a9945310e87db5bca206b019a7eb876d910c25d30cb47
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:ops/cannbot_dsl/flash_kda.py
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Technical specification of the custom operators in `cann-recipes-infer/ops/`,
  highlighting `ops/cannbot_dsl/flash_kda.py` (chunkwise Key-Delta / Gated DeltaNet
  attention with WY representation) and `ops/ascendc`.
tier: L2
title: 'CANN-Recipes-Infer Custom Operator Catalog: Flash KDA (cannbot_dsl), AscendC,
  PyPTO, and TileLang'
triggers:
  error_signatures:
  - ops cannbot_dsl flash_kda.py
  - ops ascendc readme cann recipes infer
  keywords:
  - ops cannbot_dsl flash_kda.py
  - ops ascendc readme cann recipes infer
  - flash kda key delta attention kernel
  - wy representation householder chunk deltanet
  - flash-kda
  - cannbot-dsl
  - kimi-kda
  - gated-deltanet
  - ops-ascendc
  - chunk-linear-attention
---
## 1. `Flash KDA` Chunkwise Linear/Delta Attention (`ops/cannbot_dsl/flash_kda.py`)

Recurrent linear attention mechanisms (`Gated DeltaNet` in `Qwen3-Next` and `KDA` in `Kimi-Linear`) update a matrix state $S_t \in \mathbb{R}^{D_k \times D_v}$ token-by-token via delta rule transitions:
$$S_t = S_{t-1} (\text{Diag}(\alpha_t) - \beta_t k_t k_t^\top) + \beta_t k_t v_t^\top$$
Evaluating this recurrence sequentially during prefill is completely Vector-bound and underutilizes the NPU Cube unit. `ops/cannbot_dsl/flash_kda.py` implements **Chunkwise Parallel `Flash KDA`**:
1. **Chunk Decomposition & `WY` Representation**:
   - Partitions sequence length $T$ into chunks of size $B_c = 64$.
   - Within each chunk, accumulates the product of generalized Householder transformations $(\text{Diag}(\alpha_t) - \beta_t k_t k_t^\top)$ into compact $[B_c, D]$ matrices $W$ and $U$ via forward substitution on the chunk lower-triangular matrix $(I + \text{tril}(\beta K K^\top \odot \Gamma))^{-1}$.
2. **Cube-Accelerated Intra-Chunk & Inter-Chunk GEMMs**:
   - Converts the sequential token recurrence into matrix-matrix multiplications (`chunk_K @ S_prev`, `Q @ K^T` intra-chunk causal block, and state update `S_next = Decay * S_prev + K_tilde^T @ U`) executed on the Ascend Cube (`AIC`) engine.
3. **On-Chip Pipelining**:
   - Keeps the recurrent state tile $S \in \mathbb{R}^{D_k \times D_v}$ (`[128, 128]` `FP32` accumulator) resident in on-chip SRAM across consecutive chunks of the same sequence, writing only final output tokens $O$ and final state $S_T$ to HBM.

## 2. `cann-recipes-infer/ops` Multi-DSL Operator Organization (`ops/ascendc/README.md`)

`cann-recipes-infer/ops/` organizes model-specific custom kernels across five DSL tiers according to performance sensitivity and development velocity:
- `ops/ascendc/`: Hand-tuned C++ AscendC kernels for flagship attention (`SparseAttnSharedKV`, `Compressor`, `LightningIndexer`), `mHC`, and HSTU ragged attention.
- `ops/pypto/` & `ops/pypto_python/`: MPMD tile-graph mega-kernels (`DeepSeek-V3.2-Exp`, `DeepSeek-V4`, `Ling-3.0-Flash`).
- `ops/tilelang/`: Declarative TileLang-Ascend kernels (`SFA`, `mHC`, `INT8 GEMM`, `ActQuant`).
- `ops/cannbot_dsl/`: High-level DSL kernels such as `flash_kda.py`.

## Verification Checklist

- Maintain the recurrent state accumulator $S$ in `FP32` across chunks in `flash_kda.py`; accumulating delta-rule state updates in `BF16` causes rapid numerical decay over `S > 4096`.
