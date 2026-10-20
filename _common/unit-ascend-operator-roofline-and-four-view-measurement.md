---
schema: mindie-graded-unit/1
id: unit-ascend-operator-roofline-and-four-view-measurement
revision: 1
tier: L1
category: measurement
domain_path: _common
evidence_strength: reproduced_empirical
title: 'Ascend operator four-gate measurement and four-view roofline contract'
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
  - float32
inherits_from: []
provenance:
  source_type: measurement_contract
  source_locator: docs/evidence/c4-case-block-v2-gates.json
  evidence_refs:
  - docs/evidence/c4-case-block-v2-gates.json
  - docs/evidence/c4-case-block-v2-stage-acceptance.json
  supersedes: []
triggers:
  error_signatures:
  - a speedup is claimed without synchronized before/after timing or without warmup exclusion
  - one pooled IQR is used across different shapes while per-case block behavior is ignored
  - a correctness prerequisite is counted as a measured performance factor
  - kernel duration is reported as end-to-end operator latency
  keywords:
  - Ascend operator
  - measurement
  - OracleFreezer
  - synchronize_stream
  - AblationGate
  - leave-one-out
  - roofline
  - msprof
  - HBM
  - UB
  - four-view
---
# Ascend operator four-gate measurement and four-view roofline contract

Use this contract for every operator latency, speedup or bottleneck claim on Ascend 910B / 910_93. A number is admissible only when its gate, clock, view and workload identity are named. This unit supplies the measurement premises; the consuming unit supplies the operator-specific conditions.

## Four gates

**Gate 1 — OracleFreezer (correctness).** Freeze the reference output and an unmodified copy of every input before examining a candidate. Compute the reference in fp32 from the declared formula and tolerance. Pass only when max absolute and relative error are within tolerance, the returned dtype equals the declared `out_dtype` exactly, and every input tensor is byte/value unchanged after the call. Correctness is a prerequisite; it is not a performance factor and cannot be ablated away.

**Gate 2 — Synchronized block timing (measurement).** Call `acl.rt.synchronize_stream()` immediately before and immediately after each timed block. Exclude at least two warmup rounds. Record at least three blocks per case with at least twelve samples per block, and keep the case identity with every sample. Compute the arithmetic mean within each case and then the geometric mean of the per-case candidate/reference ratios. Report the within-case block envelope (the minimum and maximum block-mean ratio). Never pool heterogeneous cases into one IQR, never call the envelope a confidence interval, and never treat samples from separate stage runs as paired blocks. Use `host_monotonic_ns` for the block interval; use msprof for device time separately.

**Gate 3 — AblationGate (attribution).** Decompose the candidate into named factors. Leave out exactly one performance factor per arm while retaining all correctness prerequisites (`required_dispatch_registration`, `input_preservation_out_dtype`). Compare full candidate versus leave-one-out and leave-one-out versus baseline, with Gate 1 and Gate 2 repeated for each arm. Retain a factor only when correctness passes and the conditional delta is positive above the recorded noise floor; report each case separately. Do not prune on one aggregate and do not present a prerequisite as a measured performance mutator.

**Gate 4 — ExecutionGate (artifact identity).** Before trusting any timing, record the loaded module path and SHA-256, the dispatch/registration state, the target call count and the absence of a silent fallback. A candidate whose intended symbol did not execute is an execution failure, not a slow operator.

## Four aligned views

1. **Kernel view (msprof).** Per-kernel task duration, op summary and the device timeline. This is device execution time, not end-to-end operator latency.
2. **Host view.** Host dispatch overhead, launch gaps and CPU-op time around the operator boundary, measured with the same synchronized-block discipline. Small operators can be host-bound even when the kernel is fast.
3. **Memory and roofline view.** Count HBM traffic and UB traffic for the operator. Arithmetic intensity is FLOPs divided by bytes moved; the roofline ridge point is peak FLOP/s divided by peak byte/s. Below the ridge the operator is memory-bound; above it, compute-bound. A theoretical roofline bound is not a measured speedup.
4. **End-to-end operator boundary view.** The full call from host entry to synchronized completion, including dispatch, copies and kernels. Report it alongside the kernel-only number; never substitute one for the other.

All four views use the same workload identity, dtype, shape, case hash and clock domain. A view measured on a different run or shape does not complete another view. Failed captures, rejected oracle verdicts and missing measurements stay visible.

## Claim rule

A speedup claim needs Gate 1, Gate 2 and Gate 4 on the same candidate and workload, plus the Gate 3 attribution that names the retained factor. State the measured envelope, the case set and the clock. Do not generalize beyond the declared conditions; a missing or unknown condition does not match.
