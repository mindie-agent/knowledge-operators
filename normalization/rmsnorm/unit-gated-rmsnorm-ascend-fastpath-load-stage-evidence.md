---
category: sop
conditions:
  declared_hardware:
    npu_count: 1
    npu_model: Ascend910
  repository: vllm-project/vllm-ascend
  task_start_base_commit: f02da291ec0bdbe31af248deffa69f7a97e8f259
  topic: gated-rmsnorm-gdn-kda-fastpath
content_digest: sha256:052c3021547f44142562355cd6388ff0a3e7ca28d66570a7ec0fa0784eebee8d
domain_path: normalization/rmsnorm
evidence_strength: single_observation
id: unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence
inherits_from: []
provenance:
  evidence_refs:
  - public-task-metadata:sha256:94687a1df029594699981125fb9a29d273e18f5307c6776107e42b3c4b614f9f
  - original-task-instruction:sha256:c747db1c6139860c9b6d650501e18c5659322b0de7dfae201eb9a8478d21eb45
  - fixed-case-observation:sha256:57c4859ac0096806477c4192fada6a68787caad97a70e6e9433d2d56dfa67c7d
  - identity-scope-correction:sha256:36b7f1bac3bccc506f3696bd915ed5dd0768c3a586efccc75ba687453a274b00
  source_digest: 94687a1df029594699981125fb9a29d273e18f5307c6776107e42b3c4b614f9f
  source_locator: https://github.com/vllm-project/vllm-ascend/tree/f02da291ec0bdbe31af248deffa69f7a97e8f259
  source_type: public_task_start_metadata_plus_same_family_observation
  source_url: https://github.com/vllm-project/vllm-ascend
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'Ascend gated RMSNorm: verify the intended fast path loaded, then bound the
  claim to the observed stage inventory and numerics'
triggers:
  error_signatures:
  - gated RMSNorm falls back to unoptimized PyTorch instead of the Ascend fast path
  - Ascend fused gated RMSNorm path does not take effect
  keywords:
  - gated RMSNorm
  - AscendFusedRMSNormGated
  - GDN
  - KDA
  - RmsNorm
  - Sigmoid
  - Mul
  - Ascend910
---
# Ascend gated RMSNorm fast-path load and stage-evidence procedure

## Use when

The consumer is checking the original exposed-Dev gated-RMSNorm dispatch task in `vllm-project/vllm-ascend` at task-start revision `f02da291ec0bdbe31af248deffa69f7a97e8f259` with declared `Ascend910` x1 hardware. The question is whether the intended Ascend gated-RMSNorm path actually loaded and whether the observed stage inventory and numerics justify the specific claim being made.

All four canonical conditions are required and exact: repository, full task-start base commit, declared hardware (`Ascend910`, count 1), and the normalized topic token `gated-rmsnorm-gdn-kda-fastpath`. A topic-only observation, a consumer-intent question, a different base revision, or a different NPU model/count does not satisfy the version/environment gate. Missing or unknown facts remain unknown; they do not match.

## Procedure

1. Bind task-start facts before looking for answers. Confirm the public repository, the full task-start base commit, the declared NPU model/count, and the task topic. Keep dtype and the specific CANN version out of the required gate because the admitted task-start metadata does not supply them. Do not turn shape, seed, tolerance, kernel names, fix/candidate/gold identities, oracle values, or measurement values into query conditions.
2. Confirm the intended path loaded. Inspect the class, module, registry/op registration, and source bytes from the actual candidate checkout. A Python class named `AscendFusedRMSNormGated` is dispatch evidence, not kernel-fusion evidence.
3. Inspect the mechanism. The observed forward path is `torch_npu.npu_rms_norm(x, weight, eps)` followed by `sigmoid(gate)` multiply. That is a multi-operation chain; it can still launch separate kernels.
4. Inspect the per-step kernel inventory. Record whether each active step contains separate `RmsNorm`, `Sigmoid`, and `Mul` tasks, one fused task, or another inventory. Three separate tasks support a stage-chain/dispatch claim. A single fused task or explicit fusion schedule is required for a fusion claim.
5. Compare numerics only after the fixed case is known. Record the reference formula, tolerance, max absolute/relative errors, finite flags, and input-preservation flags as evidence for that case. These values are not retrieval prerequisites and do not generalize to other shapes, seeds, activations, weights, epsilon values, or residual paths.
6. Classify the claim. Keep these separate: (a) intended path/source loaded; (b) three-kernel stage chain executed; (c) fixed-case final output is numerically close; (d) single fused kernel; (e) controlled performance benefit; (f) full-model correctness; (g) native TaskClient attachment identity; (h) P1/P2 installed-product acceptance.
7. Record the gaps. If the inventory is three kernels, do not call it a single fused kernel. If there is no controlled baseline/candidate comparison, do not claim a speedup. If only one operator case is tested, do not claim full-model correctness. If the attachment is synthetic, do not claim native identity closure.

## Claim boundaries

The source-recorded fixed-case observation is one sigmoid-gated, weight-ones case with three separate kernels per step; exact shape, seed, tolerance, errors, and flags are recorded only in the evidence_scope annex. It is the separate execution lane's evidence, not a re-run for this revision. It does not establish a single fused kernel, throughput or performance benefit, full-model or application accuracy, full 0day support, or native DSH/P1/P2/four-native formal acceptance.

This unit is a new L2 SOP. It is not `unit-residual-free-rmsnorm-fastpath` (L4 measured mutator) and not `unit-rmsnorm-five-stage-ablation-workflow` (L3 five-stage workflow). It does not inherit their measured claims, and it does not turn the fixed-case observation into a general measured gain.

## Identity caveat

Execution, profiler, HostQueue, and cleanup evidence can be real while the TaskClient attachment identity is synthetic. The observed source used a synthetic fixture attachment, not a verified native DSH session. Keep numerical/profiler/lifecycle evidence separate from native attachment identity. No native DSH or P1/P2 acceptance is claimed; if identity is in scope, require an actual DSH attach, resume, and TaskClient identity verification.

## Limits

- One fixed public operator case; no other shapes, seeds, activations, weights, epsilon values, or residual paths.
- Three kernels per recorded step; single-kernel fusion is not shown.
- No controlled performance baseline or throughput measurement.
- No full-model or application-model correctness.
- No tensor-level reference recomputation from persisted raw tensors.
- No SiLU, residual, non-unit-weight, other-shape, epsilon, or ACLGraph capture/replay coverage.
- dtype and specific CANN version are absent from task-start metadata and must not be inferred from posterior measurement.
- Native DSH TaskClient identity and P1/P2 installed-product acceptance are not established.
- The exact task-start base commit scopes this procedure to the source task; it is not a numeric guarantee for every artifact at that revision.
