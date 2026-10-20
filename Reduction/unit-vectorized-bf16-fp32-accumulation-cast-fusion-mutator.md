---
schema: mindie-graded-unit/1
id: unit-vectorized-bf16-fp32-accumulation-cast-fusion-mutator
revision: 1
tier: L4
category: mutator
domain_path: Reduction
evidence_strength: hypothesis
title: 'L4 mutator: vectorized bf16 reduction with fp32 accumulation and fused store cast'
conditions:
  soc:
  - Ascend910B
  - Ascend910B1
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  output_dtype:
  - bfloat16
  - float16
  op_family:
  - reduction
  - norm
  reduction_axis:
  - last
  tail_alignment: 32B
  register_pressure: headroom
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
provenance:
  source_type: ablation_design_receipt
  source_locator: docs/evidence/l4-vectorized-reduction-cast-fusion-ablation.json
  source_digest: d3db32071a28dd317956aa49494e81b3aeaf1da8159aa1bdb39232ef81844a6b
  evidence_refs:
  - docs/evidence/l4-vectorized-reduction-cast-fusion-ablation.json
  - docs/evidence/c4-case-block-v2-ablation.json
  - docs/evidence/c4-case-block-v2-stage-acceptance.json
  experiment_status: not_run_source_only
  supersedes: []
triggers:
  error_signatures:
  - a bf16 or fp16 reduction casts a partial sum back to half between stages
  - the vectorized load improves the aligned prefix but the tail drives the error or NaN
  - fp32 partials spill to GM or an intermediate fp32 output tensor is materialized
  - split-K atomics make repeated runs of the same input return different bytes
  - a candidate is faster on one shape but regresses the workspace or warm-server guard
  keywords:
  - vectorized reduction
  - fp32 accumulation
  - cast fusion
  - split-K
  - UB alignment
  - tail mask
  - bfloat16
  - float16
  - Ascend910B
  - Ascend910_93
  - Triton-Ascend
  - leave-one-out
  - guard workload
---
# L4 mutator: vectorized bf16 reduction with fp32 accumulation and fused store cast

## Applicability

Ascend 910B/910_93, bfloat16 or float16 input, a last-axis reduction (`sum`, `mean`, `amax`, `amin`, `l2norm`, `logsumexp`, variance) or a normalized reduction whose output dtype is declared. The row base and row stride are 32 B UB aligned and the vector tile plus fp32 accumulator fit the UB/register budget with headroom.

Apply only when the observed facts match every condition. `tail_alignment: 32B` and `register_pressure: headroom` are observed facts, not assumptions. A missing fact is `condition_unknown` and the unit is excluded by default. A mismatch is the fallback, not an invitation to lower precision.

## Signals

- `msprof` shows the reduction kernel plus a separate cast or an intermediate GM tensor, or `ub_to_gm`/HBM traffic exceeds the input plus declared output.
- The aligned case is flat or faster while a tail case (`row_bytes % 32 != 0`) diverges, returns NaN/Inf, or changes with a tail-only control.
- Repeated execution of the same input is not byte-identical, which points at split-K atomic order rather than noise.
- A local speedup appears with a workspace high-water-mark increase, graph-shape failure, or warm-server regression.

## Transformation

Keep the tail and fallback branches in every arm. Apply the factors in this order; the ablation receipt records them separately.

1. **`vectorized_bf16_load_fp32_accumulate`** — load the tile through the vectorized 32 B UB path, widen bf16/fp16 lanes to fp32 before the first accumulation, and keep the running sum, square sum, min/max and every partial in fp32. Do not cast a partial back to half between levels.
2. **`single_store_cast_fusion`** — combine the final fp32 reduction (including the true denominator for `mean` and epsilon only where the formula requires it), then cast once to the declared output dtype at the store. Do not materialize an intermediate fp32 GM output or return fp32 as an unrequested dtype.
3. **`two_stage_splitk_tail_guard`** — when the reduction crosses the tile boundary, split the aligned prefix deterministically across stages, combine stage partials in fp32 in a fixed order, and mask the tail with the correct identity (`0` for sum/mean/l2norm, `-inf` for amax/logsumexp, `+inf` for amin). Prefer a deterministic two-stage partial buffer over unordered atomics. A padded lane must never enter the denominator or the maximum.

The factors are separable for leave-one-out. Remove `vectorized_bf16_load_fp32_accumulate` for the load arm; remove `single_store_cast_fusion` for the store arm; remove `two_stage_splitk_tail_guard` for the tail arm, which must fail the correctness gate rather than merely slow down. Keep the two correctness prerequisites (the fp32 accumulator and the masked tail) in every arm.

## Fallback boundary

- **Reduction tail is unaligned.** If the tail cannot be masked with the correct identity, or the row stride is not 32 B compliant, do not take the fused path. Use the inherited L2 reduction SOP's explicit `DataCopyPad`/masked-tail path, or materialize an aligned view, and keep the fallback visible. Never read or count an uninitialized tail.
- **Register pressure spills.** If the fp32 accumulator plus vector tile spills, the UB tile no longer fits, or the compiler reports a local-memory spill, abandon the fused path. Do not lower the accumulator to half precision or shrink the tile until correctness fails silently.
- **Static shape without a tile-boundary crossing.** If the workload is a fixed small shape with no meaningful tail or split, the split-K factor is not applicable; retain the simpler vectorized path and record the boundary.
- **No evidence of the fused kernel running.** If the symbol is not selected, the target call count is zero, or the fallback composite ran, the timing is invalid and the candidate is not the claimed transformation.

## Verification and ablation

- **Gate 1 — OracleFreezer.** Freeze the fp32 reference and every input. Cover an aligned shape, a `N mod 16` tail and a `N mod 16 == 1` tail. Require finiteness, max-abs and relative error inside the pre-registered tolerance, exact `out_dtype`, unchanged inputs and repeatability.
- **Gate 2 — synchronized timing.** Exclude at least two warmups, synchronize immediately before and after each block, keep case identity and hashes per sample, and report per-case block envelopes. Never pool aligned and tail cases into one IQR or call the envelope a confidence interval.
- **Gate 3 — leave-one-out.** Compare full candidate vs each removed-factor arm and each arm vs baseline with Gates 1 and 2 repeated. Distinguish a correctness prerequisite from a performance factor. Do not prune a result inside the noise envelope, and do not reuse the full candidate's run as the final-combination verification.
- **Gate 4 — system guards and hidden retention.** Re-run the final retained combination on frozen hidden inputs. Classify each item as `both_pass`, `opt_regression`, `both_fail`, `opt_improvement` or `evaluation_error`; only comparable `both_pass` items enter a speedup ratio. Measure every guard workload (aligned, unaligned tail, tile-boundary crossing, workspace high-water mark, graph capture/replay shape stability, warm-server repeated call). A local gain never overrides a guard regression.
- **Provenance.** This unit is a design-level hypothesis. The linked receipt records factors, arms, guards and fallback boundaries with `experiment_status: not_run_source_only`; it establishes no measured speedup. Promote to `reproduced_empirical` only after the complete final combination, hidden inputs and guards are independently verified.
