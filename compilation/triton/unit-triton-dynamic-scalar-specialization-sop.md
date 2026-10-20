---
schema: mindie-graded-unit/1
id: unit-triton-dynamic-scalar-specialization-sop
revision: 2
tier: L2
category: sop
domain_path: compilation/triton
evidence_strength: single_observation
title: Diagnose Triton recompilation across dynamic input lengths
conditions:
  kernel_language: triton
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
provenance:
  source_type: source_review
  source_locator: https://github.com/triton-lang/triton/blob/v3.2.0/python/triton/runtime/jit.py
  evidence_refs:
  - https://triton-lang.org/main/python-api/generated/triton.jit.html
  - https://github.com/triton-lang/triton/blob/v3.2.0/python/triton/runtime/jit.py
  supersedes:
  - mindie://operators/units/unit-triton-dynamic-scalar-specialization-sop@1
triggers:
  error_signatures:
  - changing token count causes a new Triton compilation
  - first request at a new input length has a latency spike
  keywords:
  - Triton
  - recompilation
  - do_not_specialize
  - constexpr
  - BLOCKS_PER_CORE
  - dynamic shape
  - warmup
---
# Dynamic-length recompilation SOP

Use the inherited L1 constexpr/specialization contract.

1. Repeat an identical launch, then change only the input length. Inspect compilation activity separately from kernel timing and confirm that the serving path calls this kernel.
2. Trace length-derived launch arguments into the JIT signature. In Triton 3.2, `constexpr` values enter the compilation key separately from ordinary scalar specialization. Adding `do_not_specialize` while retaining `tl.constexpr` does not remove that constant.
3. For a varying loop count or per-core work count, use a runtime scalar and list it in `do_not_specialize`. Keep tile extents needed by `tl.arange` static. Preserve masks, alignment and per-core ownership while changing specialization.
4. Warm the finite tile, dtype and mode combinations still used by the workload. Check the installed backend's JIT behavior before selecting the warmup set.
5. Verify numerical results on boundary lengths, repeated launches and graph replay where used. Compare compilation activity and cold/warm latency separately. If compilation fails, repair the concrete error before judging performance.

For an Ascend UB allocation error, inspect intermediate tensor sizes and tiling before repeating the measurement. Record the affected shape and backend version so a later attempt with changed premises can be evaluated again.
