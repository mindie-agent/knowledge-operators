---
schema: mindie-graded-unit/1
id: unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence
revision: 3
tier: L2
category: sop
domain_path: normalization/rmsnorm
evidence_strength: single_observation
title: 'Ascend gated RMSNorm Bottleneck & Fastpath Dispatch Triage SOP'
conditions:
  soc:
  - Ascend910B
  - Ascend910B4
  dtype: bfloat16
  activation: sigmoid
  target_file: vllm_ascend/ops/layernorm.py
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  source_type: public_task_start_metadata_plus_same_family_observation
  source_locator: https://github.com/vllm-project/vllm-ascend/tree/f02da291ec0bdbe31af248deffa69f7a97e8f259
  source_url: https://github.com/vllm-project/vllm-ascend
  source_digest: 94687a1df029594699981125fb9a29d273e18f5307c6776107e42b3c4b614f9f
  evidence_refs:
  - public-task-metadata:sha256:94687a1df029594699981125fb9a29d273e18f5307c6776107e42b3c4b614f9f
  - original-task-instruction:sha256:c747db1c6139860c9b6d650501e18c5659322b0de7dfae201eb9a8478d21eb45
  - fixed-case-observation:sha256:57c4859ac0096806477c4192fada6a68787caad97a70e6e9433d2d56dfa67c7d
  - identity-scope-correction:sha256:36b7f1bac3bccc506f3696bd915ed5dd0768c3a586efccc75ba687453a274b00
  supersedes:
  - mindie://operators/units/unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence@2
triggers:
  error_signatures:
  - gated RMSNorm falls back to unoptimized PyTorch instead of the Ascend fast path
  - Ascend fused gated RMSNorm path does not take effect
  - gated RMSNorm runs as separate RmsNorm, Sigmoid and Mul kernels
  keywords:
  - gated RMSNorm
  - AscendFusedRMSNormGated
  - GDN
  - KDA
  - RmsNorm
  - Sigmoid
  - Mul
  - Ascend910B
  - Ascend910B4
  - fastpath
  - dispatch
---
# Ascend gated RMSNorm Bottleneck & Fastpath Dispatch Triage SOP

## Use when

A bf16 sigmoid-gated RMSNorm in `vllm_ascend/ops/layernorm.py` on Ascend910B/910B4 needs triage: did the intended fast path load, is the current implementation a bottleneck, and which unit applies next? Inherited L1 alignment and measurement rules are premises.

## Triage procedure

1. **Freeze the source and symbol contract.** Record repository revision, target path and SHA-256, exact entry point, and the dispatch/registry entry expected to select the native path. Read the candidate checkout, not a cached import; a class name does not prove registration or execution.
2. **Inventory executed stages.** Use the msprof kernel view to list per-step tasks. Separate `RmsNorm`/`Sigmoid`/`Mul` tasks indicate a composite chain; one fused or native task indicates fusion. Record every active step and target call count, and check for silent PyTorch fallback.
3. **Check the fastpath gates before calling it a bottleneck.** Confirm bf16 input, sigmoid activation, residual-free input and declared output dtype. Inspect contiguous and strided gate views; record the dispatch decision, because shape equality alone does not prove the same kernel ran. A strided-gate fallback is a dispatch defect, not a performance result.
4. **Freeze the oracle before timing.** Compute the reference as `x * rsqrt(mean(x^2) + eps) * weight * sigmoid(gate)` with fp32 reduction and intermediate. Record tolerance, max absolute/relative error, returned dtype and input preservation. Gate 1 must pass for both layouts.
5. **Measure only after Gates 1 and 4.** Use synchronized before/after blocks and excluded warmups, at least three blocks per case. Report per-case block envelopes and keep host dispatch and kernel views separate. A correct composite chain slower than its matched native alternative is a validated bottleneck; without that baseline it is a hypothesis.
6. **Dispose and route.** Choose one:
   - native fast path loaded, registered and exercised: no mutator; record dispatch evidence and stop;
   - native path exists but composite chain is selected: apply the L4 residual-free mutator, including `required_dispatch_registration` and `input_preservation_out_dtype`;
   - Gate 1 fails, or the transformation should have removed the bottleneck but did not: escalate to the L3 five-stage workflow and its L4 escalation trigger;
   - case outside the declared shape/dtype/activation envelope: do not reuse this SOP as a performance claim; collect new evidence.

## Output record

Write a triage record: source identity, symbol/registration path, executed stages per case, both gate-view shapes, oracle result, timing blocks, disposition and every unknown condition. Keep rejected records.

## Claim boundaries

Dispatch evidence is not fusion evidence; a fused task is not a speedup. A fixed-case numerical pass is not full-model correctness, and one shape is not a workload envelope. The L4 envelope and L3 workflow carry their own conditions; this L2 unit only chooses which applies.
