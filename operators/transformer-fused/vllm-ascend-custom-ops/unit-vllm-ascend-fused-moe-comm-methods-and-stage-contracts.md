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
  - vLLM-Ascend FusedMoE execution across TP, EP, DP, and FlashComm/MC2 modes
domain_path: operators/transformer-fused/vllm-ascend-custom-ops
evidence_strength: software_contract
id: unit-vllm-ascend-fused-moe-comm-methods-and-stage-contracts
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/fused_moe/fused_moe.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/fused_moe/moe_comm_method.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/fused_moe/moe_mlp.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/fused_moe/eplb.py
  source_digest: 9fda3f1c830e1fce471f907185a2f9def89c80ac7f0aae580651c1107ea15c25
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/fused_moe/fused_moe.py
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Code contract for `vllm_ascend/ops/fused_moe/` (`fused_moe.py`, `moe_comm_method.py`,
  `moe_mlp.py`, `eplb.py`) covering communication strategy selection (`AllGather`,
  `MC2`, `AlltoAllv`, `FusedMC2`), `unified_apply_mlp`, and EPLB expert remapping.
tier: L2
title: 'vLLM-Ascend FusedMoE Architecture: AllGather, MC2, AlltoAllv, and EPLB Stage
  Contracts'
triggers:
  error_signatures:
  - vllm_ascend ops fused_moe fused_moe.py
  - vllm_ascend moe_comm_method mc2 alltoallv
  keywords:
  - vllm_ascend ops fused_moe fused_moe.py
  - vllm_ascend moe_comm_method mc2 alltoallv
  - vllm_ascend eplb redundant expert routing
  - unified_apply_mlp npu_grouped_matmul vllm_ascend
  - vllm-ascend
  - fused-moe
  - moe-comm-method
  - mc2
  - alltoallv
  - eplb
  - moe-mlp
---
## 1. `vllm_ascend/ops/fused_moe` Modular Pipeline Architecture

`vllm-ascend` structures MoE execution around a 4-stage contract (`moe_stage_contracts.py` & `moe_comm_method.py`):
1. **Stage 1 — Router Gating (`select_experts`)**:
   - Supports `softmax` (`npu_moe_gating_top_k_softmax`), `sigmoid` with `e_score_correction_bias` (`npu_moe_gating_top_k`), grouped top-$k$ (`num_expert_group`, `topk_group`), and custom Triton/Ascend renormalization gates.
   - When **EPLB (`eplb.py`)** is enabled, maps logical `topk_ids` through the dynamic `log2phy` expert table so overloaded hot experts are balanced across redundant physical expert replicas.
2. **Stage 2 — Pre-MLP Communication & Token Permutation (`MoECommMethod.prepare`)**:
   - Dynamically selects one of four communication backends based on `(ep_size, dp_size, batch_tokens, graph_mode)`:
     - **`AllGatherCommImpl`**: Used for small EP or TP-dominant prefill where gathering tokens + local filtering beats all-to-all overhead.
     - **`MC2CommImpl` / `FusedMC2`**: Invokes `torch_npu.npu_moe_distribute_dispatch_v2` for low-latency decode (`aclgraph`-compatible, zero host sync, fused `W8A8` dynamic quant).
     - **`AlltoAllvCommImpl`**: Used for large-token prefill across nodes where bulk HCCL `all_to_all_single` bandwidth dominates.
3. **Stage 3 — Grouped Expert MLP (`moe_mlp.py`: `unified_apply_mlp`)**:
   - Unquantized (`BF16`/`FP16`): `npu_grouped_matmul(w13)` -> `npu_swiglu` -> `npu_grouped_matmul(w2)`.
   - `W8A8` Dynamic Quantized: Fuses `npu_dynamic_quant` + quantized `npu_grouped_matmul` (or `npu_grouped_matmul_swiglu_quant`) + second quantized `GMM` with `per_token_scale`.
   - `W4A8` / `W4A16`: Uses group-wise weight dequantization scales (`weight_scale`, `weight_offset`) inside `npu_grouped_matmul`.
4. **Stage 4 — Post-MLP Combine & Shared Expert Accumulation (`MoECommMethod.finalize`)**:
   - Executes `npu_moe_distribute_combine_v2` (for `MC2`) or `npu_moe_finalize_routing` + `AlltoAllv` / `ReduceScatter`, adding the overlapped shared expert output stream.

## Verification Checklist

- Confirm `expert_map` / `log2phy` resides on NPU and uses vectorized indexing so EPLB expert remapping never breaks `aclgraph` capture.
