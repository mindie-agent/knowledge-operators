---
category: sop
conditions:
  cann_versions:
  - '>=8.2.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - Triton-Ascend kernel authoring, code review, and performance optimization
domain_path: operators/skill-playbooks/triton-and-tilelang
evidence_strength: software_contract
id: unit-agent-skills-triton-operator-design-review-and-npu-optimization
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/triton-operator-design/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/triton-operator-code-review/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/triton-operator-performance-optim/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/vector-triton-ascend-ops-optimizer/SKILL.md
  source_digest: 00d8f84986a5cf6f4d61c04da376461c12089c12a9c35f1b6958576b2ee9e04d
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/triton-operator-design/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Comprehensive playbook from `Ascend/agent-skills` (`triton-operator-design`,
  `code-review`, `performance-optim`, `vector-triton-ascend-ops-optimizer`) for authoring
  and tuning Triton kernels on Ascend NPU.
tier: L2
title: Triton-Ascend Operator Design, Code Review, and Vector/Cube Optimization Playbook
triggers:
  error_signatures:
  - triton operator design skill
  - triton operator code review skill
  keywords:
  - triton operator design skill
  - triton operator code review skill
  - vector triton ascend ops optimizer
  - triton ascend ub overflow persistent kernel
  - triton-ascend
  - operator-design
  - code-review
  - vector-optimizer
  - ub-overflow
  - grid-tiling
---
## 1. GPU Triton vs. Triton-Ascend Architectural Shift (`triton-operator-design`)

On NVIDIA GPUs, Triton maps each `tl.program_id` to a warp-scheduled thread block (`num_warps`, `num_stages`) with SIMT scalar/tensor registers. On **Ascend NPU (`910B`/`910C`)**, BiSheng-Triton compiles each program instance onto **SIMD Vector (`AIV`, `48` cores) and Cube (`AIC`, `24` cores)** engines backed by a `192 KB` Unified Buffer (`UB`):
1. **Persistent Grid Sizing (`Grid <= Physical Cores`)**:
   - Launching a 1D grid of `100,000` tiny blocks incurs massive task scheduling overhead on NPU.
   - **Rule**: Cap grid size at `num_vector_cores` (`48` on 910B, `96` per die-pair on A3) or a small multiple (`48..192`) and loop over work tiles (`for tile_idx in range(pid, num_tiles, num_programs):`) inside the kernel.
2. **Contiguous Vectorized Inner Dimension (`BLOCK_SIZE` Alignment)**:
   - The innermost axis loaded via `tl.load` / `tl.store` must have stride `1` and size aligned to `32` bytes (`16` `fp16`/`bf16` or `8` `fp32` elements), ideally `256..4096` elements per tile so Vector SIMD instructions run at full width (`256B` per cycle) and MTE2/MTE3 achieves burst DMA coalescing.
   - Avoid scattered index loads (`tl.load(ptr + random_indices)`) inside the inner loop; transpose or sort outside when possible.

## 2. Triton-Ascend Code Review & Optimization Rules (`vector-triton-ascend-ops-optimizer`)

1. **Preventing UB Overflow (`ub overflow` Compilation Error)**:
   - Every live `tl.tensor` inside a Triton kernel resides in UB (`192 KB`).
   - If `BLOCK_M * BLOCK_N * sizeof(float32)` plus temporary expressions exceeds `192 KB`, compilation fails with `ub overflow`.
   - **Fix**: Split large 2D reductions into a tiled inner loop (`for col_start in range(0, N, BLOCK_N):`) maintaining online statistics (`max_val`, `sum_exp`) in 1D `[BLOCK_M]` accumulators instead of loading `[BLOCK_M, N]` at once.
2. **`FP32` Accumulation for Reductions & Transcendentals**:
   - Always cast `fp16`/`bf16` tiles to `tl.float32` (`x = tl.load(...).to(tl.float32)`) before `tl.sum`, `tl.max`, `tl.exp`, `tl.log`, or `tl.rsqrt`.
3. **`tl.dot` Cube Lowering Requirements**:
   - `tl.dot(a, b)` lowers to the Cube unit only when `BLOCK_M`, `BLOCK_N`, and `BLOCK_K` are multiples of `16` (for `fp16`/`bf16`) and `>= 16`. Never use `tl.dot` on `[1, K] x [K, N]` vectors; either pad $M$ to `16` or use vectorized multiply-accumulate when $M < 16$.

## Verification Checklist

- Ignore GPU-only autotune knobs (`num_warps`, `num_ctas`) on NPU; tune `(BLOCK_M, BLOCK_N, BLOCK_K, multibuffer/num_stages)` against physical AIV/AIC core counts (`48` / `24`).
