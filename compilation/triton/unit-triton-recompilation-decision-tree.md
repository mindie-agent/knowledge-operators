---
schema: mindie-graded-unit/1
id: unit-triton-recompilation-decision-tree
revision: 1
tier: L3
category: workflow
domain_path: compilation/triton
evidence_strength: reproduced_empirical
title: Triton recompilation decision tree
conditions:
  kernel_language: triton
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-triton-dynamic-scalar-specialization-sop
provenance:
  source_type: reproduced_case_decision_record
  source_locator: https://github.com/mindie-agent/mindie-agent/blob/main/docs/knowledge-context-and-exploration.md#72
  evidence_refs:
  - https://github.com/mindie-agent/mindie-agent/blob/main/docs/knowledge-context-and-exploration.md#72
  - https://github.com/triton-lang/triton/blob/v3.2.0/python/triton/runtime/jit.py
  supersedes: []
triggers:
  error_signatures:
  - a Triton-Ascend serving path recompiles on the same input length
  - first requests at new input lengths trigger repeated Triton compilation
  keywords:
  - Triton
  - Triton-Ascend
  - recompilation
  - specialization key
  - do_not_specialize
  - constexpr
  - dynamic shape
  - warmup
---
# Triton recompilation decision tree

**Trigger.** A Triton-Ascend kernel recompiles when the same length repeats, or only after a length or shape change. Confirm the serving path launches it and separate compile time from kernel time.

**First discriminating observation.** Repeat an identical launch at the same length. If compilation still appears, go to Branch A. Otherwise rerun at a new length and inspect the JIT specialization key and length-derived arguments. If a new key appears, go to Branch B.

**Branch A — same-length recompilation.** Check the cache directory, cache-key inputs, process/container boundary and backend environment; repair the miss or isolation. Do not change kernel arguments for a cache-path problem.

**Branch B — new-length specialization.** If a length-derived work-partition argument is `tl.constexpr`, `do_not_specialize` alone cannot help: convert it to a runtime scalar and list it in `do_not_specialize`, keeping `tl.arange` tile extents static. Verify compile-count convergence before timing.

**Warmup.** Warm only finite legal tile, dtype and mode variants the workload uses. Never enumerate an unbounded length-derived scalar; enumeration hides the problem.

**Verify.** Keep existing tiling, numerics and per-core ownership. Check boundary lengths, repeated launches and graph replay; compare compile activity and cold/warm latency separately.

**Completion and exploration exit.** The workflow ends when same-length repeats stop compiling, or new lengths reuse one specialization without changing numerics. If the key is stable yet compilation or correctness still changes, root cause is likely compiler internals or layout mechanics: stop this tree and open a targeted exploration with that hypothesis instead of retuning `do_not_specialize`.
