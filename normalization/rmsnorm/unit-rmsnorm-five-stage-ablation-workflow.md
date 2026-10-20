---
schema: mindie-graded-unit/1
id: unit-rmsnorm-five-stage-ablation-workflow
revision: 3
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
  - mindie://operators/units/unit-rmsnorm-five-stage-ablation-workflow@2
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

Run the five stages in order for an operator optimization on Ascend910B/Ascend910B4, bf16, sigmoid, targeting `vllm_ascend/ops/layernorm.py`. Each stage has an entry condition, a persisted artifact, and an exit gate. Do not skip a gate because an earlier stage looked convincing.

## Stage 1 — Source & Symbol Contract + Oracle Freeze

Entry: a task checkout and a target symbol are known.

1. Record the repository, base commit, target file path and SHA-256, the class/function entry point, and every dispatch-table or registry entry that can select the path. Confirm the module actually imported by the runtime, not a sibling copy.
2. Capture the input contract: shapes, dtypes, strides, seed, epsilon, activation, residual presence, and declared `out_dtype` for both a contiguous and a strided case.
3. Compute a reference before editing anything. Use the declared formula with fp32 reduction/intermediate, and freeze the reference output plus an unmodified copy of every input.
4. Run Gate 1 on the unmodified baseline: max absolute/relative error within tolerance, returned dtype exactly `out_dtype`, inputs unchanged, finite outputs.

Exit: Gate 1 passes on the baseline for both cases, and the frozen oracle bytes and hashes are persisted under the run directory. If Gate 1 cannot pass on the baseline, stop and report an oracle or environment failure; do not optimize.

## Stage 2 — Synchronized Block Baseline Timing

Entry: Stage 1 exit.

1. Run at least two warmup rounds and exclude them.
2. Call `acl.rt.synchronize_stream()` immediately before and after every timed block. Record `host_monotonic_ns` per block and keep the case hash with every sample.
3. Collect at least three blocks of at least twelve samples per case. Compute the arithmetic mean within each case, then the geometric mean of the per-case ratios.
4. Report the per-case block envelope and the within-case noise floor. Do not pool different shapes into one IQR and do not call the envelope a confidence interval.

Exit: a baseline record with run ID, timing hash, per-case block means, sample counts and noise floor. Gate 2 fails on a missing synchronization boundary, missing warmups, fewer than three blocks, or a pooled-IQR estimator.

## Stage 3 — Candidate Synthesis & L4 RSI Escalation Trigger

Entry: Stage 2 exit.

1. Synthesize the smallest candidate that removes the named bottleneck. Prefer one named transformation over a bundle; keep the correctness prerequisites (`required_dispatch_registration`, `input_preservation_out_dtype`) in the candidate from the start.
2. Run Gate 1 and Gate 2 on the candidate for both cases, then the four-view check from the inherited measurement unit.
3. Escalate to an L4 reusable mutator when either trigger holds:
   - **stall:** routine L3 optimization has exhausted its planned attempts and no candidate beats the baseline above the noise floor, or the remaining gap is structural rather than a parameter change;
   - **gate failure:** a candidate fails Gate 1 (numerics, dtype or input mutation) or Gate 2 (unstable or non-synchronized timing) in a way that requires a different code transformation, not another tuning pass.
4. On escalation, write the escalation record: failing gate, the invariant that must be preserved, the target file/symbol, and the L4 unit id to apply. On no escalation, continue with the candidate.

Exit: a candidate (or an explicit escalation record), its Gate 1/2 artifacts, and the four-view summary. Do not ablate a candidate that has not passed Gate 1.

## Stage 4 — Leave-One-Out Factor Ablation

Entry: Stage 3 candidate passes Gate 1 and Gate 2.

1. Decompose the candidate into named factors and label each factor as a correctness prerequisite or a performance mutator. The correctness prerequisites are `required_dispatch_registration` and `input_preservation_out_dtype`. Performance mutators include `native_residual_free_fastpath`, but the label is assigned from the recorded factor role, not from its name.
2. Build one leave-one-out arm per performance factor. Every arm keeps all correctness prerequisites. Remove exactly one performance factor per arm.
3. For each arm run Gate 1 and Gate 2 and compare full candidate versus leave-one-out, and leave-one-out versus baseline. Keep the per-case ratios.
4. Retain a factor only if correctness passes and the conditional delta is positive above the recorded noise floor in both comparisons. Do not prune a factor on one aggregate; do not report a prerequisite as a measured performance factor; do not infer a factor's role from its position in the patch.

Exit: an ablation record per factor with disposition (`retain`, `prune`, `required`), per-case evidence, and the exact comparison identities. Failed or rejected candidate verdicts remain visible; they are not deleted by a later correction.

## Stage 5 — Final Retained Verification & Hermes `knowledge_distill`

Entry: Stage 4 exit with at least one retained performance factor.

1. Re-run the retained combination in a separate final stage: Gate 1, Gate 2, the four-view check, and the final speedup as the geometric mean of per-case mean ratios versus the recorded baseline. Report the per-case block envelope and the final timing hash.
2. Call Hermes `knowledge_distill` with the final artifact set. The decision must produce a `reviewed_change` containing the complete unit bytes, the exact base generation and the review reference, and one appended row in `experiment-ledger.jsonl` containing the task id, observed conditions, factor dispositions, stage run IDs, timing hashes, final result and evidence refs.
3. Preserve the raw stage records and any rejected oracle or proof status. A reporting correction does not create a new measurement campaign and does not rewrite raw bytes.
4. Apply the reviewed change through the runtime's reviewed-change path. Promotion is not automatic: a unit is published only when its declared conditions match the measured envelope and the separate publication check passes.

Exit: final retained verification passed, `reviewed_change` written, `experiment-ledger.jsonl` appended, and the promoted unit's conditions and limits state the measured envelope rather than a general speedup.

## Invariants and failure handling

- Gate order is fixed: Stage 1 before Stage 2, Stage 1/2 before Stage 3, Stage 3 before Stage 4, Stage 4 before Stage 5.
- A candidate that fails Gate 1 is a correctness failure; a candidate that fails Gate 2 is unmeasured. Neither is a speedup and neither is ablated.
- Unknown conditions remain unknown and never match. A missing measurement, a rejected verdict and a failed producer state stay visible in the record.
- The inherited L1 alignment/tiling axioms constrain every candidate; the inherited L1 four-gate/four-view contract defines the timing and attribution evidence.
