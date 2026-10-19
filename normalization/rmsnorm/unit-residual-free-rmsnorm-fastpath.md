---
category: mutator
conditions:
  ascend_rt_visible_devices: '0'
  base_commit: f02da291ec0bdbe31af248deffa69f7a97e8f259
  factor_id: native_residual_free_fastpath
  gold_patch_sha256: f01efd3a242a3b183e1431d9df6aefb700ba067314c41ad7891c8c249babf2ec
  machine: aarch64
  measure_sha256: bc77c926270222f5ab051e1d94abb070df6d39bb66ace05963a83f3f29cd4efe
  oracle_manifest_sha256: f343d8a964b8b20a168efcc9b8380e7c5afecffd5a1cd048d0a7cb93c2bd6bf2
  platform: Linux-5.10.0-282.0.0.184.oe2203sp3.aarch64-aarch64-with-glibc2.35
  python_version: 3.12.13
  source_root_tree_sha256: efb31c0470b80b8a4f955bcde7825c90cccd1ea1886215f925830901749aa715
  target_file: vllm_ascend/ops/layernorm.py
  task_id: fused-rmsnorm-gate-dispatch-performance
content_digest: sha256:655a8c56c81d6aa337201dbc895c9760826d0f1b7dc7087e9301479943dd4ef3
domain_path: normalization/rmsnorm
evidence_strength: reproduced_empirical
id: unit-residual-free-rmsnorm-fastpath
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/c4-assembly-health-ablation.json
  source_digest: 67ca5ed641213ab4d8db8bc0d23faf848a58b37185de8f15b3678076d7e210f3
  source_locator: mindie-003-execution/c4-measured/assembly-health/ablation_receipt.json
  source_type: ablation_receipt
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L4
title: Residual-free RMSNorm dispatch under recorded C4 development conditions
triggers:
  error_signatures: []
  keywords:
  - rmsnorm
  - residual
  - dispatch
  - fastpath
---
Retain the `native_residual_free_fastpath` factor together with the required
`required_dispatch_registration` and `input_preservation_out_dtype` glue when
applying this measured candidate to the recorded source tree. The glue is
required for dispatch and correctness; it is not an independently measured
performance factor.

The C4 receipt binds distinct baseline, full-candidate, leave-one-out, and final
runs under identical recorded conditions. Baseline: `stage1-1cd637fea06e4e9d890e441b690129dc-baseline`.
Full candidate: `stage1-1cd637fea06e4e9d890e441b690129dc-full`. Independent final:
`stage1-1cd637fea06e4e9d890e441b690129dc-final`. The final combination
passed correctness and performance checks and reports a p50 speedup of
1.724170128087756 relative to the recorded baseline. Reverting the
factor reports a p50 ratio of 0.992708146668242; this supports the
factor's conditional contribution for these samples.

No factor was pruned: `pruned_combination_verified` is false and
`pruning_verifications` is empty. The independent final run verifies the
retained combination, not a pruning result. Conditions include the exact base
commit, source tree, measurement program, patch, and oracle manifest hashes.
Read the mechanically redacted receipt in
`docs/evidence/c4-assembly-health-ablation.json` and its provenance record before
reusing this claim.

This is evidence from a development C4 case. It does not establish a full-model
serving gain, heldout generalization, another shape or hardware configuration,
or a product-level productivity gain. A new target needs its own correctness
and matched baseline/candidate measurements. Repository integration is a local
draft: the reviewed contract/2 runtime pin and formal publication remain pending.
