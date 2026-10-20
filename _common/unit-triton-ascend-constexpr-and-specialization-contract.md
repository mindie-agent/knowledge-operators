---
schema: mindie-graded-unit/1
id: unit-triton-ascend-constexpr-and-specialization-contract
revision: 1
tier: L1
category: axiom
domain_path: _common
evidence_strength: reproduced_empirical
title: 'Triton-Ascend constexpr and runtime scalar specialization contract'
conditions:
  kernel_language: triton
inherits_from: []
provenance:
  source_type: source_review_and_reproduced_case
  source_locator: https://github.com/triton-lang/triton/blob/v3.2.0/python/triton/runtime/jit.py
  evidence_refs:
  - https://triton-lang.org/main/python-api/generated/triton.jit.html
  - https://github.com/triton-lang/triton/blob/v3.2.0/python/triton/runtime/jit.py
  supersedes: []
triggers:
  error_signatures:
  - a Triton kernel recompiles when only an input length or per-core work count changes
  - do_not_specialize is applied but a length-derived argument still creates a new compilation
  keywords:
  - Triton
  - Triton-Ascend
  - constexpr
  - do_not_specialize
  - specialization key
  - recompilation
  - runtime scalar
---
# Triton-Ascend constexpr and specialization contract

**Condition.** Triton-Ascend kernels whose launch arguments vary with input length, shape or per-core work.

**Constraint.**
- **`tl.constexpr` is compile-time.** A constexpr is baked into the compiled kernel and enters the specialization key; changing it produces a distinct compilation regardless of `do_not_specialize`.
- **`do_not_specialize` covers runtime scalars.** It stops the JIT specializing an ordinary scalar on its value or divisibility. It cannot remove a constexpr, so convert a length-derived work-partition scalar to a runtime scalar first, then list it in `do_not_specialize`.
- **Keep true constants static.** Tile extents used by `tl.arange`, dtype and layout stay constexpr. Preserve masks, alignment and per-core ownership when moving a value out of the key.
- **Check the installed backend's JIT key** before promising cache reuse.

**Rationale.** The key contains the arguments the backend treats as specialized; applying `do_not_specialize` to a constexpr turns the wrong knob and leaves the compile count unchanged.

**Evidence.** Triton 3.2 JIT source and API contract; reproduced on the recorded length-varying kernel case.
