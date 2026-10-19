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
  input_shape:
  - 1x37x4x128
  - 1x193x8x128
  machine: aarch64
  measure_sha256: bc77c926270222f5ab051e1d94abb070df6d39bb66ace05963a83f3f29cd4efe
  oracle_manifest_sha256: f343d8a964b8b20a168efcc9b8380e7c5afecffd5a1cd048d0a7cb93c2bd6bf2
  platform: Linux-5.10.0-282.0.0.184.oe2203sp3.aarch64-aarch64-with-glibc2.35
  python_version: 3.12.13
  soc: Ascend910B4
  source_root_tree_sha256: efb31c0470b80b8a4f955bcde7825c90cccd1ea1886215f925830901749aa715
  target_file: vllm_ascend/ops/layernorm.py
  task_id: fused-rmsnorm-gate-dispatch-performance
  workload_case_sha256:
  - 2869c075070aff7c08049c50d772f27fa67ecdf9a019a90154191ee9ac00ae2a
  - fa2291e2cfb790aa2c30dab849999118d004246e00aa46517c1b54708f0c4b2e
content_digest: sha256:049e64317e28aeaad88e2df1649661935270acadef82ba3dc97e22bf42cbdbef
domain_path: normalization/rmsnorm
evidence_strength: reproduced_empirical
id: unit-residual-free-rmsnorm-fastpath
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/c4-environment-ablation.json
  - docs/evidence/c4-environment-runtime-index.json
  - docs/evidence/c4-environment-stage-acceptance.json
  source_digest: 40c444ff848a9a83f3632f63b414c2c93f4c8924f7e71556ca9f8e727a30f7b9
  source_locator: mindie-003-execution/c4-environment/assembly-health/ablation_receipt.json
  source_type: ablation_receipt
  supersedes:
  - mindie://operators/units/unit-residual-free-rmsnorm-fastpath@9c4ee3f0e570d1eca504f722090460096541887bac052182fe477d4645ed3626
revision: 3
schema: mindie-graded-unit/1
tier: L4
title: Residual-free RMSNorm fastpath under the measured Ascend910B4 environment
triggers:
  error_signatures: []
  keywords:
  - rmsnorm
  - residual
  - dispatch
  - fastpath
---
Retain `native_residual_free_fastpath` together with
`required_dispatch_registration` and `input_preservation_out_dtype` when using
the measured candidate under the exact conditions declared above. The latter
two changes provide required dispatch and correctness; neither is an
independently measured performance factor.

This revision binds the new independent C4 run `stage1-034f88c590b746a4b381830336a1e24a`. The actual
runtime was Ascend910B4 with CANN 9.1.0, driver 25.2.2, firmware 7.7.0.10.220,
torch 2.10.0+cpu and torch_npu 2.10.0.post4. Five measurement processes each have
matching captures before and after measurement; all ten captures lie outside
the kernel and warm tiny-service timing intervals. The frozen bfloat16/sigmoid
cases are 1x37x4x128 with a contiguous gate and 1x193x8x128 with a strided gate.
Exact case hashes bind shape and layout together. Each arm has 36 kernel
samples per case from three AB/BA/AB timing blocks.

The baseline comparison is `stage1-034f88c590b746a4b381830336a1e24a-baseline`, the full
candidate is `stage1-034f88c590b746a4b381830336a1e24a-full`, and the separate
final retained-combination verification is
`stage1-034f88c590b746a4b381830336a1e24a-final`. Relative to
the recorded baseline, the full candidate's p50 ratio is
1.563466019727391, the leave-one-out ratio is
1.017449577990265, and the final ratio is
1.696208537370911. This supports the
factor's conditional contribution for these samples. Correctness, execution,
measurement and ablation gates pass in the derived assembly. No factor was
pruned; the final run verifies the retained combination.

The original producer aggregate remains `failed`: its dtype summarizer
compared an observed set with a singleton list. The derived execution proof
corrects only that comparison using the original wrapper captures, with a
real bfloat16 positive and a dtype-only negative check. Raw source, baseline
and leave-one-out Oracle verdicts remain rejected; full and final remain
accepted. Baseline and leave-one-out are valid measured comparisons with
passed correctness, not accepted candidate solutions. Read the separate
correction and stage-acceptance evidence; no raw failed result was rewritten.

The new evidence is in `docs/evidence/c4-environment-ablation.json`,
`docs/evidence/c4-environment-runtime-index.json`,
`docs/evidence/c4-environment-stage-acceptance.json`,
`docs/evidence/c4-environment-correction-validation.json` and
`docs/evidence/c4-environment-gates.json`. Original versus mechanically redacted
hashes and the admission/review split are recorded in
`docs/evidence/c4-environment-provenance.json`.

This is the same exposed development task, not an independent heldout task.
It does not establish another shape, runtime, hardware target, full-model
serving gain or product-level productivity gain. A changed target needs its
own correctness and matched measurements. The original C4 result and revision
2's unknown CANN/driver premises remain historical evidence; this new run does
not fill those historical unknowns. Local applicability and C6 acceptance do
not establish formal Feed integration or a published runtime contract.
