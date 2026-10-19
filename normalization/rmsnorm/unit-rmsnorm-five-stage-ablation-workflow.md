---
category: workflow
conditions:
  activation: sigmoid
  base_commit: f02da291ec0bdbe31af248deffa69f7a97e8f259
  cann: 9.1.0
  driver: 25.2.2
  dtype: bfloat16
  factor_id: native_residual_free_fastpath
  firmware: 7.7.0.10.220
  framework:
    torch: 2.10.0+cpu
    torch_npu: 2.10.0.post4
  soc: Ascend910B4
  target_file: vllm_ascend/ops/layernorm.py
  workload:
    exact_set:
    - input_shape: 1x37x4x128
      samples_per_shape: 36
      workload_case_sha256: 2869c075070aff7c08049c50d772f27fa67ecdf9a019a90154191ee9ac00ae2a
    - input_shape: 1x193x8x128
      samples_per_shape: 36
      workload_case_sha256: fa2291e2cfb790aa2c30dab849999118d004246e00aa46517c1b54708f0c4b2e
content_digest: sha256:f72c21fd52649a8b086010eda2049fbd200276d4b22badee963f0d2aa088fc67
domain_path: normalization/rmsnorm
evidence_strength: single_observation
id: unit-rmsnorm-five-stage-ablation-workflow
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/l3-ablation-workflow.json
  - docs/evidence/c4-case-block-v2-ablation.json
  - docs/evidence/c4-case-block-v2-stage-acceptance.json
  - docs/evidence/c4-case-block-v2-gates.json
  - docs/evidence/c4-revision4-provenance.json
  source_digest: 559ec6571463148494c090be4f3ac768268ca73104119ffe1791d4b125e55b2c
  source_locator: docs/evidence/l3-ablation-workflow.json
  source_type: run_manifest
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Observed five-stage RMSNorm ablation workflow with preserved reporting failure
triggers:
  error_signatures: []
  keywords:
  - observed
  - five-stage
  - rmsnorm
  - ablation
  - workflow
  - with
  - preserved
  - reporting
  - failure
---
One recorded RMSNorm campaign executed source, baseline, full candidate, leave-one-out and final retained stages. For each stage, pre/post environment captures bind to that stage's own wrapper PID and process-start identity and bracket its measurement intervals; the five stages used different process IDs. Ten captures validate five such pairs. Keep the workload-case hash with every sample: each of the two shapes has 36 samples arranged in three 12-sample blocks. Compare a full candidate with its baseline, remove the named performance factor while retaining required dispatch and output-correctness changes, and compare the leave-one-out evidence before deciding whether to retain that factor. The recorded final stage separately measured the retained combination within this same campaign. Its samples already existed before the later reporting correction; the correction did not launch a new final measurement. The original remote producer exited 1: its dtype proof compared a set with a singleton list and rejected equal bfloat16 values. Preserve that failure, the raw oracle/proof statuses, and the successful lease release, descendant drain and slot cleanup. A separate zero-NPU reanalysis corrected the comparison and produced derived correctness, execution, measurement and ablation acceptance. The reviewed case/block analysis retained native_residual_free_fastpath and verified the existing final-stage result; no factor was pruned. This is a workflow observation from one campaign. Two shapes, three sample blocks, and later reanalysis do not create additional independent campaigns. The block envelope is descriptive and must not be called a confidence interval or matched pairing across stages. A new hardware, software, source or workload condition needs new evidence before a runtime or performance conclusion.
