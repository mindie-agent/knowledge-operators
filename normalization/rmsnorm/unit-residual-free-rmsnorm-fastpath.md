---
schema: mindie-graded-unit/1
id: unit-residual-free-rmsnorm-fastpath
revision: 6
tier: L4
category: mutator
domain_path: normalization/rmsnorm
evidence_strength: reproduced_empirical
title: 'L4 mutator: residual-free gated RMSNorm native fastpath for Ascend910B'
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
  - mindie://operators/units/unit-residual-free-rmsnorm-fastpath@3
  - mindie://operators/units/unit-residual-free-rmsnorm-fastpath@5
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
# L4 mutator: residual-free gated RMSNorm native fastpath

## 1. Applicability & Bottleneck Pattern

Apply only to a residual-free gated RMSNorm in `vllm_ascend/ops/layernorm.py` on Ascend910B/Ascend910B4 with bf16 input and sigmoid activation. A residual input means this mutator is not applicable; route to a residual-aware unit.

The bottleneck signature is a composite chain instead of one native operator: `npu_rms_norm(x, weight, eps)` (or an equivalent reduction) followed by `sigmoid(gate)` and a multiply. The msprof kernel view may show separate `RmsNorm`, `Sigmoid` and `Mul` tasks; the host view may show three launches per call; the memory view may show extra GM-to-UB round trips for the intermediate. A Python class name, a registered function name or a single fast step does not prove fusion. Confirm the executed kernel inventory, the target call count and the absence of a fallback before transforming code.

The bottleneck is worth removing when the composite chain is a material share of the operator boundary, both the contiguous and strided cases are covered, and the oracle already passes. A chain that is already at the host-boundary floor, or whose native alternative has not been validated for the same dtype and gate layout, is not a candidate.

## 2. Code Transformation Operator

Apply these three factors together. They are the transformation, not optional tuning knobs.

### `native_residual_free_fastpath`

Replace the composite Python chain with the native residual-free gated RMSNorm op exposed by the installed CANN/torch_npu build. Compute the sum and sum-of-squares in fp32, apply epsilon in fp32, multiply the normalized fp32 value by `sigmoid(gate)`, then cast once to the declared `out_dtype`. Use the symbol actually exported by the installed build; do not copy a guessed op name.

```python
# sketch only; resolve the real registered symbol from the candidate build
if x.dtype is torch.bfloat16 and activation == "sigmoid" and residual is None:
    gate_arg = gate if gate.is_contiguous() else gate.contiguous()
    return native_gated_rmsnorm(x, weight, gate_arg, eps, out_dtype=x.dtype)
```

### `required_dispatch_registration`

Register the native op in the dispatch path that the target file uses, and make the Python forward reach it. This includes the custom-op namespace/registration, the import that activates it, and the dtype/activation branch that selects it. Verify by reading the registration entry and by observing a nonzero target call count in msprof. A registered symbol that is never selected is a fallback, not a fast path. Keep this factor in every ablation arm; it is a correctness prerequisite.

### `input_preservation_out_dtype`

Handle both gate layouts without mutating inputs:

- **Contiguous gate** (`1x37x4x128`): pass the gate view directly after the 32 B UB alignment check.
- **Strided gate** (`1x193x8x128`): pass the declared strides to a strided native entry point, or materialize a contiguous gate copy with `gate.contiguous()` and pass the copy. Never assume stride 1 and never reinterpret a strided view as contiguous. Materializing the gate is allowed; mutating it is not.
- Do not use in-place sigmoid, in-place multiply, or write the normalized value back into `x`. Do not change `weight` or `gate`. The returned dtype must equal the declared `out_dtype` exactly; an internal fp32 result is not a reason to return fp32.
- Apply the inherited L1 alignment, tail-mask and repeat-stride rules to both layouts. A candidate that passes only the contiguous case is incomplete.

The patch is minimal: one native call, one dispatch registration, one layout-preserving gate argument. Do not bundle unrelated changes; a bundle cannot be attributed by Stage 4.

## 3. Verification & Ablation Invariants

**Gate 1 — OracleFreezer.** Freeze the fp32 reference before the patch. Cover the contiguous and strided cases. Require max absolute/relative error within tolerance, returned dtype exactly `out_dtype`, all inputs unchanged after the call, and finite outputs. A dtype pass or an input-preservation pass alone is not a numerics pass.

**Gate 2 — synchronized timing.** Warm up at least two rounds and exclude them. Call `acl.rt.synchronize_stream()` before and after each timed block. Collect at least three blocks of at least twelve samples per case and keep the case hash. Compute per-case means and the geometric mean of case ratios; report the per-case block envelope. Do not pool the two shapes into one IQR and do not call the envelope a confidence interval.

**Gate 3 — AblationGate.** Remove only `native_residual_free_fastpath` in the leave-one-out arm. Keep `required_dispatch_registration` and `input_preservation_out_dtype` in every arm; they are correctness prerequisites, not measured performance factors. Compare full candidate versus leave-one-out and leave-one-out versus baseline, with Gate 1 and Gate 2 repeated. Retain the factor only when correctness passes and the conditional delta is positive above the recorded noise floor in both comparisons. Do not prune on one aggregate.

**Gate 4 — execution.** Record the loaded module SHA-256, the registration entry, the target call count and the fallback status. A candidate whose native symbol did not run is an execution failure and its timing is not evidence.

## 4. Reference Empirical Envelope

The recorded campaign used Ascend910B4, bf16, sigmoid, target `vllm_ascend/ops/layernorm.py`, with two cases: `1x37x4x128` with a contiguous gate and `1x193x8x128` with a strided gate. Each case contributed 36 kernel samples in three blocks of 12 after two warmup rounds; timing was the original synchronized host-monotonic interval, and the estimator was the geometric mean of per-case arithmetic-mean ratios.

- Final retained versus baseline: **1.6767473221869322x** (reported as 1.68x). The per-case observed block-envelope lower bounds were 1.5638748801806301 and 1.7387598131467064.
- Full candidate versus baseline: 1.5276488398873962x.
- Final retained versus full candidate: 1.097599970887633x, with envelope lower bounds 1.0868596255956195 and 1.0809065218997271.
- The factor was retained; correctness prerequisites stayed in every arm. The envelope is descriptive and is not a confidence interval; the stages are separate runs, with no paired-block or between-run replication claim.

Raw samples, case hashes, block groups and comparison identities are in `docs/evidence/c4-case-block-v2-ablation.json`. Gate results are in `docs/evidence/c4-case-block-v2-gates.json`; stage acceptance is in `docs/evidence/c4-case-block-v2-stage-acceptance.json`.

The measured envelope is Ascend910B4. The transformation applies to the other soc values in `conditions`, but each such target needs its own oracle, synchronized baseline and ablation before any speedup is claimed. No other shape, dtype, activation, residual path, full-model serving gain or product productivity gain is established here.
