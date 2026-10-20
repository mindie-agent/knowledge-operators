---
schema: mindie-graded-unit/1
id: unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence
revision: 2
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
  - mindie://operators/units/unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence@1
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

A bf16 sigmoid-gated RMSNorm is on the critical path in `vllm_ascend/ops/layernorm.py` on Ascend910B or Ascend910B4, and the task is to decide whether the intended Ascend fast path actually loaded, whether the current implementation is a bottleneck, and which unit to apply next. Apply the L1 memory/tiling axioms and the L1 four-gate/four-view contract as inherited premises.

## Triage procedure

1. **Freeze the source and symbol contract.** Record the repository revision, the target file path and SHA-256, the exact class/function entry point, and the dispatch table or registry entry that is supposed to select the native path. Read the candidate checkout, not a cached import. A class name is not evidence that a kernel was registered or executed.
2. **Observe the executed stage inventory.** Use the msprof kernel view to list the per-step tasks. Separate `RmsNorm`, `Sigmoid` and `Mul` tasks indicate a composite chain. One fused or native task indicates a fused path. A single step is not enough: record every active step and its target call count, and check for a silent fallback to PyTorch.
3. **Check the fastpath gates before calling it a bottleneck.** Confirm bf16 input, sigmoid activation, residual-free input, and the declared output dtype. Inspect both a contiguous gate view and a strided gate view. The contiguous and strided dispatch decisions must be explicit; shape equality alone does not prove the same kernel ran. If a strided gate falls back, that is a dispatch defect, not a performance result.
4. **Freeze the oracle before any timing.** Compute the reference as `x * rsqrt(mean(x^2) + eps) * weight * sigmoid(gate)` with fp32 reduction and an fp32 intermediate. Record tolerance, max absolute/relative error, returned dtype, and an input-preservation check. Gate 1 must pass for both the contiguous and strided cases; a passing contiguous case does not cover the strided case.
5. **Measure only after Gates 1 and 4.** Use synchronized before/after blocks, excluded warmups, and at least three blocks per case. Report per-case block envelopes and the host dispatch and kernel views separately. A composite chain that is numerically correct but slower than its native alternative is a validated bottleneck; a chain without a matched baseline is only a hypothesis.
6. **Dispose and route.** Choose one:
   - the native fast path is already loaded, registered and exercised: no mutator, record the dispatch evidence and stop;
   - the native path exists but the composite chain is selected: apply the L4 residual-free mutator, including `required_dispatch_registration` and `input_preservation_out_dtype`;
   - Gate 1 fails, or the transformation should have removed the bottleneck but did not: escalate to the L3 five-stage workflow and its L4 escalation trigger;
   - the case is outside the declared shape/dtype/activation envelope: do not reuse this SOP as a performance claim; collect new evidence.

## Output record

Write a triage record containing: source identity, symbol and registration path, executed stage inventory per case, both gate-view shapes, oracle result, timing blocks, the chosen disposition, and every condition that remains unknown. Keep the raw failed or rejected records; do not replace them with a later interpretation.

## Claim boundaries

Dispatch evidence is not fusion evidence. A fused task is not a speedup. A fixed-case numerical pass is not full-model correctness. A single shape is not a workload envelope. The L4 measured envelope and the L3 workflow carry their own conditions; this L2 unit only decides which one applies.
