---
category: mutator
conditions:
  activation: sigmoid
  ascend_rt_visible_devices: '0'
  base_commit: f02da291ec0bdbe31af248deffa69f7a97e8f259
  cann: 9.1.0
  driver: 25.2.2
  dtype: bfloat16
  factor_id: native_residual_free_fastpath
  firmware: 7.7.0.10.220
  framework:
    torch: 2.10.0+cpu
    torch_npu: 2.10.0.post4
  gold_patch_sha256: f01efd3a242a3b183e1431d9df6aefb700ba067314c41ad7891c8c249babf2ec
  hostname: <redacted:local-path:0e4201aad4a5>
  input_shape:
  - 1x37x4x128
  - 1x193x8x128
  machine: aarch64
  measure_sha256: bc77c926270222f5ab051e1d94abb070df6d39bb66ace05963a83f3f29cd4efe
  non_factors:
  - required_dispatch_registration
  - input_preservation_out_dtype
  oracle_manifest_sha256: f343d8a964b8b20a168efcc9b8380e7c5afecffd5a1cd048d0a7cb93c2bd6bf2
  platform: Linux-5.10.0-282.0.0.184.oe2203sp3.aarch64-aarch64-with-glibc2.35
  python_executable: /usr/local/python3.12.13/bin/python3
  python_version: 3.12.13
  soc: Ascend910B4
  source_root_tree_sha256: efb31c0470b80b8a4f955bcde7825c90cccd1ea1886215f925830901749aa715
  stage_timeout_seconds: 240.0
  target_file: vllm_ascend/ops/layernorm.py
  task_id: fused-rmsnorm-gate-dispatch-performance
  workload:
    exact_set:
    - input_shape: 1x37x4x128
      samples_per_shape: 36
      workload_case_sha256: 2869c075070aff7c08049c50d772f27fa67ecdf9a019a90154191ee9ac00ae2a
    - input_shape: 1x193x8x128
      samples_per_shape: 36
      workload_case_sha256: fa2291e2cfb790aa2c30dab849999118d004246e00aa46517c1b54708f0c4b2e
  workload_case_sha256:
  - 2869c075070aff7c08049c50d772f27fa67ecdf9a019a90154191ee9ac00ae2a
  - fa2291e2cfb790aa2c30dab849999118d004246e00aa46517c1b54708f0c4b2e
content_digest: sha256:64f02c7298d8238d28d9884d4bc599d2d0bd27160be76e80f76841dc5f674d0c
domain_path: normalization/rmsnorm
evidence_strength: reproduced_empirical
id: unit-residual-free-rmsnorm-fastpath
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/c4-case-block-v2-ablation.json
  - docs/evidence/c4-case-block-v2-stage-acceptance.json
  source_digest: 9893d83c5b58b03692fab48ae7ad904ce64c19d8a2e4739a3ec70a225f429dd4
  source_locator: mindie-003-execution/c4-case-block-assembly-v2/ablation_receipt.json
  source_type: ablation_receipt
  supersedes:
  - mindie://operators/units/unit-residual-free-rmsnorm-fastpath@049e64317e28aeaad88e2df1649661935270acadef82ba3dc97e22bf42cbdbef
revision: 4
schema: mindie-graded-unit/1
tier: L4
title: Residual-free RMSNorm fastpath for the complete measured Ascend910B4 workload
triggers:
  error_signatures: []
  keywords:
  - rmsnorm
  - residual
  - dispatch
  - fastpath
---
Retain `native_residual_free_fastpath` together with
`required_dispatch_registration` and `input_preservation_out_dtype` under the
complete measured workload and environment declared in conditions. Dispatch
registration and dtype preservation are correctness requirements, not separately
measured performance factors.

The workload is one exact set: bfloat16/sigmoid cases 1x37x4x128 with a contiguous
gate and 1x193x8x128 with a strided gate. Each shape is paired with its exact case
hash and 36 kernel samples. A subset, an extra case or a swapped shape/hash pair
does not satisfy this rule. The measured environment is Ascend910B4, CANN 9.1.0,
driver 25.2.2, firmware 7.7.0.10.220, torch 2.10.0+cpu and torch_npu 2.10.0.post4.

Use the v2 ablation receipt's explicit comparison identity. Kernel durations are
the original synchronized host-monotonic intervals, grouped into three timing
blocks per case after two warmup rounds. Each comparison first computes the
arithmetic mean within each case and then the geometric mean of the case ratios.
For the retained final run versus its recorded baseline, the aggregate ratio is
1.6767473221869322. The two per-case observed block-envelope lower bounds are
1.5638748801806301 and 1.7387598131467064. For final versus the full candidate,
the aggregate ratio is 1.097599970887633; its lower bounds are
1.0868596255956195 and 1.0809065218997271. These are observed envelopes, not
confidence intervals. The stages are separate runs; there is no paired-block or
independent between-run replication claim.

The full candidate versus baseline geometric-mean ratio is 1.5276488398873962.
The native gate separately records its own baseline and estimator, producing a
different ratio. Those two valid comparison records must not be treated as an
identity or substituted for one another. Do not gate heterogeneous cases on a
pooled IQR. The maximum within-case, within-block relative IQR is descriptive.

Correctness, execution, measurement and ablation gates pass in the derived v2
assembly. No factor was pruned; the separate final run verifies the retained
combination. The original producer aggregate remains failed because its dtype
summary compared an observed set with a singleton list. The derived execution
proof corrects that comparison using original wrapper captures. Raw baseline
and leave-one-out Oracle candidate verdicts remain rejected; their measurements
are valid comparisons with passed correctness, not accepted candidate solutions.

Read `docs/evidence/c4-case-block-v2-ablation.json`,
`docs/evidence/c4-case-block-v2-gates.json` and
`docs/evidence/c4-case-block-v2-stage-acceptance.json`. The source and redacted
file hashes are in `docs/evidence/c4-revision4-provenance.json`. Previous runtime
environment captures, correction evidence and all revision-3 documents remain
unchanged. This revision changes workload identity and statistical interpretation
of the recorded samples; it does not claim another measurement campaign.

This is the exposed development task. It establishes no heldout generalization,
other shape or hardware benefit, full-model serving gain, or product productivity
gain. Different conditions require their own correctness and matched measurements.
