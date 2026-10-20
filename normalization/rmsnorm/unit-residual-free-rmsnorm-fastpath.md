---
schema: mindie-graded-unit/1
id: unit-residual-free-rmsnorm-fastpath
revision: 7
tier: L4
category: mutator
domain_path: normalization/rmsnorm
evidence_strength: reproduced_empirical
title: 'Residual-free gated RMSNorm native fastpath for Ascend910B'
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
  source_type: ablation_receipt
  source_locator: mindie-003-execution/c4-case-block-assembly-v2/ablation_receipt.json
  source_digest: 9893d83c5b58b03692fab48ae7ad904ce64c19d8a2e4739a3ec70a225f429dd4
  evidence_refs:
  - docs/evidence/c4-case-block-v2-ablation.json
  - docs/evidence/c4-case-block-v2-stage-acceptance.json
  supersedes:
  - mindie://operators/units/unit-residual-free-rmsnorm-fastpath@6
triggers:
  error_signatures:
  - residual-free gated RMSNorm falls back to separate RmsNorm, Sigmoid and Mul operations
  - the native gated RMSNorm path is not registered or is not selected for bf16 sigmoid input
  - a strided gate takes the contiguous fast path or the candidate mutates its inputs
  keywords:
  - rmsnorm
  - residual-free
  - gated RMSNorm
  - dispatch
  - fastpath
  - native_residual_free_fastpath
  - required_dispatch_registration
  - input_preservation_out_dtype
  - contiguous
  - strided
  - Ascend910B
---
# Residual-free gated RMSNorm native fastpath

## Applicability

Residual-free gated RMSNorm in `vllm_ascend/ops/layernorm.py` on Ascend910B/910B4, bf16, sigmoid; a residual input is out of scope.

Signature: composite chain `npu_rms_norm(x, weight, eps)` (or equivalent), `sigmoid(gate)`, multiply; msprof may show separate tasks or extra GM-to-UB traffic. Confirm executed kernels, call count and fallback; transform only when the chain dominates the boundary, both gate layouts are covered and the oracle passes.

## Transformation

Apply all three.

- **`native_residual_free_fastpath`**: use the installed build's exported native op. Reduce in fp32, apply epsilon in fp32, multiply the normalized fp32 value by `sigmoid(gate)`, and cast once to `out_dtype`.
- **`required_dispatch_registration`**: register the op in the target dispatch path (namespace, import, dtype/activation branch). Verify the entry and nonzero msprof call count; unselected registration is a fallback. Keep it in every arm.
- **`input_preservation_out_dtype`**: for contiguous gate `1x37x4x128`, pass after the 32 B UB check; for strided gate `1x193x8x128`, pass declared strides or a `gate.contiguous()` copy, never assume stride 1. Do not mutate inputs or use in-place sigmoid/multiply; return exactly `out_dtype`.

## Verification and ablation

- **Gate 1 — OracleFreezer**: freeze the fp32 reference first; cover both cases: tolerance, exact `out_dtype`, unchanged inputs, finite outputs.
- **Gate 2 — timing**: exclude warmups, synchronize `acl.rt.synchronize_stream()` around each block, collect three 12-sample blocks per case with hashes, report per-case means, geometric ratio and envelope; never pool shapes into one IQR or call it a confidence interval.
- **Gate 3 — AblationGate**: remove only `native_residual_free_fastpath` in the leave-one-out arm, keeping both prerequisites. Compare candidate vs arm and arm vs baseline with Gates 1/2 repeated; retain only if correctness passes and the delta exceeds noise in both comparisons.
- **Gate 4 — execution**: record module SHA-256, registration entry, target call count and fallback status. Timing is invalid if the native symbol did not run.

## Reference empirical envelope

Ascend910B4, bf16, sigmoid, `vllm_ascend/ops/layernorm.py`; cases `1x37x4x128` (contiguous gate) and `1x193x8x128` (strided gate), 36 samples per case in three blocks of 12 after two warmups, host-monotonic interval. Final retained vs baseline **1.6767473221869322x**, envelope lower bounds 1.5638748801806301 and 1.7387598131467064; full candidate vs baseline 1.5276488398873962x; final retained vs full candidate 1.097599970887633x, envelope lower bounds 1.0868596255956195 and 1.0809065218997271.

The factor was retained; prerequisites stayed in every arm. The envelope is descriptive, not a confidence interval; stages were separate runs without paired-block or replication claims. Evidence: `docs/evidence/c4-case-block-v2-ablation.json`, `...-gates.json`, `...-stage-acceptance.json`. Other `conditions.soc` values need their own oracle, baseline and ablation; no other shape, dtype, activation, residual path or end-to-end gain is established.
