---
schema: mindie-graded-unit/1
id: unit-rmsnorm-five-stage-ablation-workflow
revision: 4
tier: L3
category: workflow
domain_path: normalization/rmsnorm
evidence_strength: reproduced_empirical
title: 'Five-stage Hermes operator optimization, leave-one-out ablation, and distillation workflow'
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
  source_type: run_manifest
  source_locator: docs/evidence/l3-ablation-workflow.json
  source_digest: 559ec6571463148494c090be4f3ac768268ca73104119ffe1791d4b125e55b2c
  evidence_refs:
  - docs/evidence/l3-ablation-workflow.json
  - docs/evidence/c4-case-block-v2-ablation.json
  - docs/evidence/c4-case-block-v2-stage-acceptance.json
  - docs/evidence/c4-case-block-v2-gates.json
  - docs/evidence/c4-revision4-provenance.json
  supersedes:
  - mindie://operators/units/unit-rmsnorm-five-stage-ablation-workflow@3
triggers:
  error_signatures:
  - an operator optimization has no frozen oracle, no synchronized baseline or no leave-one-out attribution
  - routine optimization stalls and the task never escalates to a code-transformation mutator
  - a final retained result is claimed without a separate retained verification run
  keywords:
  - Hermes
  - five-stage
  - operator optimization
  - leave-one-out
  - ablation
  - distillation
  - RMSNorm
  - escalation
  - reviewed_change
  - experiment-ledger
---
# Five-stage Hermes operator optimization, leave-one-out ablation, and distillation workflow

Run in order for Ascend910B/910B4 bf16 sigmoid RMSNorm on `vllm_ascend/ops/layernorm.py`. Each stage has an entry, artifact and exit gate; never skip one.

## 1. Source/symbol contract + oracle freeze

Record commit, target path and SHA-256, entry point and dispatch entries; confirm the imported module. Capture shape, dtype, stride, seed, epsilon, activation, residual and `out_dtype` for both cases. Freeze the fp32 reference and inputs before editing. Gate 1 must pass on the baseline for both cases (tolerance, exact dtype, unchanged inputs, finite outputs), else stop with an oracle or environment failure.

## 2. Synchronized block baseline timing

Follow inherited L1 Gate 2: exclude warmups, synchronize around each block, keep host-monotonic and case hash per sample, collect three 12-sample blocks per case, compute within-case means and the geometric ratio, and report block envelope and noise floor. Exit with run ID, timing hash, counts.

## 3. Candidate synthesis and L4 escalation

Synthesize the smallest candidate for the bottleneck; keep `required_dispatch_registration` and `input_preservation_out_dtype` from the start. Run Gates 1/2 for both cases plus the four-view check. Escalate to L4 on **stall** (attempts exhausted with no candidate above noise, or a structural gap) or **gate failure** (Gate 1 numerics/dtype/input mutation, or Gate 2 unstable timing, needs a different transformation); record failing gate, invariant, target symbol and L4 id. Never ablate a Gate-1 failure.

## 4. Leave-one-out factor ablation

Decompose into named factors; assign roles from the record. Prerequisites are `required_dispatch_registration` and `input_preservation_out_dtype`; `native_residual_free_fastpath` is a performance mutator. Build one leave-one-out arm per performance factor, keeping every prerequisite and removing one factor; run Gates 1/2 and compare candidate vs arm and arm vs baseline. Retain only if correctness passes and delta is positive above noise in both comparisons; do not prune on one aggregate. Exit with disposition (`retain`/`prune`/`required`), per-case evidence, comparison identities; failed verdicts stay visible.

## 5. Final retained verification and distillation

Re-run the retained combination separately: Gates 1/2, four-view check, final speedup as geometric mean of per-case ratios vs baseline; report envelope and timing hash. Use `knowledge_distill` to produce a `reviewed_change` with complete units, base generation and review reference, and append one ledger row (task id, conditions, dispositions, run IDs, timing hashes, result, refs). Preserve raw records; apply via the reviewed-change path. Publication needs matching conditions and a separate check.

## Invariants

Gate order is fixed: 1 before 2, 1/2 before 3, 3 before 4, 4 before 5. A Gate 1 failure is correctness failure; a Gate 2 failure is unmeasured; neither is a speedup or ablatable. Unknown conditions never match; missing measurements and rejected verdicts stay visible. Inherited L1 alignment constrains candidates; the four-view contract defines timing and attribution.
