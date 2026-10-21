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
  - Fused MoE expert GEMM + SwiGLU + Requantization and Token Permutation in cann/ops-transformer
domain_path: operators/transformer-fused/moe-mc2-gmm-and-rope
evidence_strength: software_contract
id: unit-ops-transformer-gmm-swiglu-quant-v2-and-moe-init-routing-v2-contract
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md
  source_digest: ff4aa25cbe22041539765d86e52be5fd03cbcc76cc15b0b929d1a010903f9496
  source_locator: cann/ops-transformer@4891cf6a210bf08a4a9bb26f7c1e2277ed43c860:gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Software contract for `aclnnGroupedMatmulSwigluQuantV2` (fused GMM1 + Dequant
  + SwiGLU + Dynamic Quant for A8W8/A8W4/MXFP8/MXFP4) and `aclnnMoeInitRoutingV2`
  in `cann/ops-transformer`.
tier: L2
title: 'cann/ops-transformer Contract: aclnnGroupedMatmulSwigluQuantV2 and aclnnMoeInitRoutingV2'
triggers:
  error_signatures:
  - aclnnGroupedMatmulSwigluQuantV2 contract
  - aclnnMoeInitRoutingV2 contract
  keywords:
  - aclnnGroupedMatmulSwigluQuantV2 contract
  - aclnnMoeInitRoutingV2 contract
  - gmm swiglu quant v2 mxfp8 mxfp4
  - expert_tokens_num_type drop_pad_mode moe_init_routing_v2
  - ops-transformer
  - grouped-matmul-swiglu-quant-v2
  - moe-init-routing-v2
  - mxfp8
  - mxfp4
  - a8w8
  - a8w4
---
## 1. `aclnnGroupedMatmulSwigluQuantV2` Fused Pipeline Contract

In quantized MoE FFN layers, executing `GMM1 (gate_up) -> Dequant -> SwiGLU -> DynamicQuant` as four separate kernels materializes the `2 * intermediate_size` intermediate tensor in HBM three times. **`aclnnGroupedMatmulSwigluQuantV2`** fuses the entire four-step sequence on-chip (`AIC` L0C -> `AIV` UB -> `GM` quantized output):
1. **Supported Quantization Modes Across Hardware**:
   - **Atlas A2 / Atlas A3 (`910B` / `910C`)**:
     - `A8W8` (`x`: `INT8` per-token, `weight`: `INT8` per-channel, output `y`: `INT8` + per-token `y_scale`: `FP32`, where $Q\_scale_i = \max(|S_i|) / 127$).
     - `A8W4` (`x`: `INT8`, `weight`: `INT4` per-group/per-channel with `weightAssistMatrix`).
   - **Ascend 950PR & 950DT**:
     - Adds native **`MXFP8`** (`FLOAT8_E4M3FN` with `emax=8`, or `FLOAT8_E5M2` with `emax=15`) and **`MXFP4`** (`FLOAT4_E2M1` with `emax=2`) microscaling quantization using block-32 shared exponents:
       $$\text{shared\_exp} = \lfloor \log_2(\max_i(|S_i|)) \rceil - \text{emax}$$
     - Accepts `aclTensorList` for `weight`, `weightScale`, and `weightAssistMatrix` to support heterogeneous/flexible expert tensor lists.
2. **Determinism Guarantee**:
   - `aclnnGroupedMatmulSwigluQuantV2` is **deterministic by default** across runs.

## 2. `aclnnMoeInitRoutingV2` Token Permutation Contract

`aclnnMoeInitRoutingV2` sorts `[N_tokens, K]` top-$k$ expert assignments into contiguous expert buckets before `GMM`:
1. **`activeNum` & `dropPadMode`**:
   - Dropless mode (`dropPadMode=0`, `activeNum = N_tokens * K`): Sorts all routed token copies without dropping tokens; outputs `expandedX` of shape `[N_tokens * K, H]`.
   - Drop/Pad capacity mode (`dropPadMode=1`, `expertCapacity` specified): Caps each expert at `expertCapacity` tokens and pads under-filled experts to fixed `[numExperts, expertCapacity, H]` shape.
2. **`expertTokensNumType` Invariant**:
   - `0`: Outputs **cumulative prefix sums** (`cumsum`) of tokens per expert (`[c_0, c_0+c_1, ...]`).
   - `1`: Outputs **per-expert token counts** (`count`: `[c_0, c_1, ...]`).
   - `2`: Key-value pair layout.
   - Must match the `groupListType` (`0` = cumsum, `1` = count) passed to `aclnnGroupedMatmulV4` / `aclnnGroupedMatmulSwigluQuantV2`.

## Verification Checklist

- For `aclnnGroupedMatmulSwigluQuantV2`, verify the `gate` and `up` halves of `weight` (`2 * intermediate_size`) match the expected interleaved or half-split SwiGLU layout required by the target platform.
