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
  - PyPTO MPMD fused mega-kernels for DeepSeek-V3.2-Exp and Ling-3.0-Flash (Bailing-V3)
domain_path: operators/model-recipes/deepseek-v32-and-bailing
evidence_strength: reproduced_empirical
id: unit-pypto-mpmd-megakernel-deepseek-v32-exp-and-bailing-v3-guide
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_pypto_operator_guide.md
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/integration/ling-3.0-flash/bailing_v3_pypto_operator_guide.md
  source_digest: 1393e38e135528113215a528fa298de6b7bd2890278de57fb970f8f23028cf71
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_pypto_operator_guide.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Architectural and programming guide for CANN PyPTO (`Parallel Tile Operation`)
  covering Tensor/Tile abstraction, MPMD dataflow scheduling without global barriers,
  automatic SRAM management, and DeepSeek/Bailing mega-kernels.
tier: L2
title: PyPTO MPMD Mega-Kernel Programming Guide for DeepSeek-V3.2-Exp and Ling-3.0-Flash
  (Bailing-V3)
triggers:
  error_signatures:
  - deepseek v3.2 exp pypto operator guide
  - bailing v3 pypto operator guide
  keywords:
  - deepseek v3.2 exp pypto operator guide
  - bailing v3 pypto operator guide
  - pypto parallel tile operation pto instruction
  - mpmd vs spmd megakernel ascend
  - pypto
  - mpmd
  - megakernel
  - pto-instruction
  - deepseek-v3.2-exp
  - bailing-v3
  - sram-reuse
---
## 1. PyPTO (`Parallel Tile Operation`) Architecture & Design Principles

Manual AscendC mega-kernels that fuse 5–10 heterogeneous stages (e.g., `RMSNorm -> QKV LoRA -> QK Norm -> RoPE -> KV Cache Update -> Indexer -> Attention`) suffer from combinatorial synchronization complexity and poor cross-generation portability. **PyPTO** solves this through four mechanisms:
1. **Declarative Tensor/Tile Programming Paradigm**:
   - Developers express multi-operator subgraphs as mathematical operations on `Tensor` and `Tile` objects (`reshape`, `slice`, `matmul`, `rmsnorm`, `rope`, `softmax`) while configuring tile shapes and SRAM residency declaratively.
2. **Automatic On-Chip SRAM Allocation & DMA Generation**:
   - The compiler automatically generates `GM <-> L1/UB` DMA read/write instructions and reuses SRAM buffers across producer-consumer tiles without manual `TPipe` offset arithmetic.
3. **Portable `PTO` Instruction IR**:
   - Lowers tile programs into platform-independent **Parallel Tile Operation (`PTO`)** instructions; retargeting from Atlas A2/A3 (`910B/910C`) to `Ascend 950` maps PTO instructions to the target hardware backend without kernel source rewrites.
4. **`MPMD` (`Multiple Program Multiple Data`) Dataflow Runtime**:
   - Unlike SPMD kernels that enforce global synchronization across all cores between phases, PyPTO's MPMD runtime constructs a fine-grained tile dependency DAG based on producer-consumer data relationships. Different AIC and AIV cores execute different operator stages concurrently, eliminating tail-effect idle bubbles.

## 2. DeepSeek-V3.2-Exp & Bailing-V3 Mega-Kernel Fusion Cases

1. **Pre-Attention Projection + RoPE + Cache Write (`Prolog` Mega-Kernel)**:
   - Fuses input `RMSNorm`, `q_a_proj` / `kv_a_proj`, latent `RMSNorm`, `q_b_proj`, per-head `QK Norm`, `RoPE` rotation, and paged KV cache scatter write into a single PyPTO MPMD kernel.
2. **Indexer + Sparse Attention Co-Scheduling**:
   - Pipelines `LightningIndexer` score evaluation and `SFA` attention tiles inside one MPMD task graph so early query blocks begin sparse attention on AIC/AIV cores while later query blocks are still completing Top-$k$ index selection.

## Verification Checklist

- Use PyPTO's visual trace & SRAM analyzer IDE view to verify zero unexpected intermediate `GM` spill occurs between fused producer and consumer tiles.
