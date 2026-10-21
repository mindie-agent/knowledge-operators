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
  - Large-EP MoE collective-compute fusion via MC2 DispatchV2 and CombineV2
domain_path: operators/transformer-fused/moe-mc2-gmm-and-rope
evidence_strength: software_contract
id: unit-ops-transformer-mc2-moe-distribute-dispatch-and-combine-v2-contract
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV2.md
  - https://gitcode.com/cann/ops-transformer/blob/4891cf6a210bf08a4a9bb26f7c1e2277ed43c860/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV2.md
  source_digest: c7f1a778dbc3ac6d59ab1ec7630e7ea66d4ada46514dff1e5c4cc99524382ed9
  source_locator: cann/ops-transformer@4891cf6a210bf08a4a9bb26f7c1e2277ed43c860:mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV2.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Software contract for `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineV2`
  in `cann/ops-transformer` covering HCCS/RDMA two-tier communication, fused dynamic
  quantization, shared-expert overlap, and buffer sizing.
tier: L2
title: 'cann/ops-transformer Contract: aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2'
triggers:
  error_signatures:
  - aclnnMoeDistributeDispatchV2 contract
  - aclnnMoeDistributeCombineV2 contract
  keywords:
  - aclnnMoeDistributeDispatchV2 contract
  - aclnnMoeDistributeCombineV2 contract
  - npu_moe_distribute_dispatch_v2 ep_world_size
  - mc2 moe dispatch dynamic quant combine
  - ops-transformer
  - mc2
  - moe-distribute-dispatch-v2
  - moe-distribute-combine-v2
  - ep
  - hccs
  - rdma
---
## 1. `aclnnMoeDistributeDispatchV2` Fused Collective-Compute Contract

In large-scale Expert Parallelism (`EP16` to `EP256`), separate `npu_moe_init_routing + dynamic_quant + HCCL AlltoAllv` launches incur severe host synchronization (transferring per-rank token counts to CPU for `AlltoAllv` split sizes) and redundant HBM round-trips. **`aclnnMoeDistributeDispatchV2`** fuses all three into a single AICore-driven `MC2` kernel:
1. **Zero Host-Sync On-Device Metadata Exchange**:
   - Exchanges per-rank token routing histograms directly between NPUs over HCCS / RoCE using AICore notify registers and RDMA/SDMA descriptors—eliminating `.cpu()` split-size synchronization and making `EP` fully capturable inside `aclgraph` / `torchair`.
2. **Fused Quantization Mode (`quantMode`)**:
   - `quantMode=0`: No quantization (`BF16`/`FP16` dispatch).
   - `quantMode=1` / `2`: Fuses per-token dynamic `INT8` quantization (`expandXOut` in `INT8` + `dynamicScalesOut` in `FP32`) prior to network transmission, cutting inter-die/inter-node `AlltoAll` traffic by **50%** compared to `BF16`.
3. **Outputs Delivered Ready for `GroupedMatmul`**:
   - `expandXOut`: Contiguously grouped by local expert ID `[0 .. localExpertNum - 1]`.
   - `expertTokenNumsOut`: `INT64` tensor of token counts (or prefix sum according to `expertTokenNumsType`) consumed directly as `group_list` by `npu_grouped_matmul`.
   - `expandIdxOut` & `epRecvCountsOut`: Routing bookkeeping tensors passed unchanged into `aclnnMoeDistributeCombineV2`.

## 2. `aclnnMoeDistributeCombineV2` Contract

- Reverses the `DispatchV2` permutation across `groupEp` (and optional `groupTp`), converts expert outputs back to original token order, multiplies each expert output by its router weight `expertScales` (`[B*S, K]`), sums over the $K$ selected experts, and optionally fuses shared expert output addition (`sharedExpertX`).

## 3. Topology & Parameter Invariants

- **`moeExpertNum % epWorldSize == 0`**: Each rank hosts `localExpertNum = moeExpertNum / epWorldSize` experts.
- **Atlas A3 Single-Server vs. Multi-Server**: On Atlas A3 (`16` dies per node over full-mesh HCCS), single-node `EP16` uses direct one-hop HCCS DMA; multi-node `EP32+` uses hierarchical two-stage dispatch (inter-node RoCE RDMA + intra-node HCCS forwarding).

## Verification Checklist

- Confirm `HCCL_BUFFSIZE` is sized adequately (typically `>= 256` or `512` MB) when running `EP >= 32` with large prefill/decode batches so `MC2` symmetric IPC windows do not overflow.
