---
category: workflow
conditions:
  cann_versions:
  - '>=8.2.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - Migrating GPU Triton or PyTorch vector operators to Triton-Ascend and TileLang-Ascend
domain_path: operators/skill-playbooks/triton-and-tilelang
evidence_strength: software_contract
id: unit-agent-skills-gpu-to-npu-triton-and-tilelang-vector-migration-workflow
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/tilelang-vector-ascend-ops-migration/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/simple-vector-triton-gpu-to-npu/SKILL.md
  source_digest: 31804e5caff56705bc51a685bc3eb8ebbe7cc9a0abe6d9cc975cbd22175473ac
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/tilelang-vector-ascend-ops-migration/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Step-by-step migration workflow from `Ascend/agent-skills` (`simple-vector-triton-gpu-to-npu`
  and `tilelang-vector-ascend-ops-migration`) for porting GPU Triton/PyTorch kernels
  to Triton-Ascend and TileLang-Ascend.
tier: L3
title: GPU-to-NPU Triton and TileLang Vector Operator Migration and Optimization Workflow
triggers:
  error_signatures:
  - tilelang vector ascend ops migration
  - simple vector triton gpu to npu
  keywords:
  - tilelang vector ascend ops migration
  - simple vector triton gpu to npu
  - migrate gpu triton kernel to tilelang ascend
  - tl.libdevice replacement ascend npu
  - tilelang-migration
  - triton-gpu-to-npu
  - vector-ops
  - tl-parallel
  - t-copy
  - simd-vectorization
---
## 1. GPU-to-NPU Triton Vector Migration Workflow (`simple-vector-triton-gpu-to-npu`)

When porting an upstream GPU Triton kernel (`fla`, `vllm`, `sglang`, `xformers`) to Ascend NPU:
1. **Step 1 — Replace CUDA-Specific Intrinsics & Device Calls**:
   - Replace `tl.extra.cuda.libdevice.*` / `tl.math.*` CUDA PTX helpers with standard `tl.exp`, `tl.log`, `tl.cos`, `tl.sin`, `tl.sqrt`, `tl.rsqrt`, `tl.erf`, or polynomial approximations supported by BiSheng-Triton.
   - Replace `.cuda()` and `torch.cuda.current_device()` with `.npu()` and `torch.npu.current_device()`.
2. **Step 2 — Refactor Scalar/Mask Branches & Non-Contiguous Loads**:
   - GPU SIMT threads tolerate per-lane divergent `tl.where` and strided loads across warps; Ascend AIV SIMD cores require contiguous vector blocks.
   - Refactor 1D elementwise/reduction kernels into 2D `(row_block, col_block)` tiles where `col_block` is contiguous in memory and a multiple of `64` or `128`.
3. **Step 3 — Retune Grid & Block Configs for `48` AIV Cores + `192 KB` UB**:
   - Strip hardcoded `@triton.autotune` configs that request `BLOCK_SIZE >= 16384` with `num_warps=16` (which overflow `192 KB` UB) and replace with NPU-validated tile sizes (`BLOCK_M in {8, 16, 32, 64}`, `BLOCK_N in {128, 256, 512, 1024}`).

## 2. PyTorch / Triton to TileLang-Ascend Vector Migration (`tilelang-vector-ascend-ops-migration`)

When migrating a vector/reduction operator into **TileLang-Ascend** (`@tilelang.jit` / `T.prim_func`):
1. **Explicit UB Allocation & DMA Copy (`T.alloc_ub` + `T.copy`)**:
   - Unlike Triton where `tl.load` implicitly allocates UB, TileLang requires explicit on-chip buffer declarations:
     ```python
     A_ub = T.alloc_ub((block_M, block_N), dtype="float16")
     Acc_ub = T.alloc_ub((block_M, block_N), dtype="float32")
     T.copy(A_gm[bx * block_M, by * block_N], A_ub)
     ```
2. **Vector Computation via `T.Parallel` & Intrinsic Primitives**:
   - Express elementwise loops with `for i, j in T.Parallel(block_M, block_N):` and reductions with `T.reduce_sum` / `T.reduce_max` over UB buffers.
   - Always promote `float16`/`bfloat16` UB buffers to `float32` UB buffers before reduction or nonlinear math, then cast back to output UB buffer before `T.copy(Out_ub, Out_gm[...])`.
3. **Double Buffering (`T.Pipelined`)**:
   - Wrap the inner tile loop in `for k in T.Pipelined(num_k_tiles, num_stages=2):` so `MTE2` (`GM -> UB`), `Vector` (`UB compute`), and `MTE3` (`UB -> GM`) overlap across consecutive tiles.

## Verification Checklist

- Verify migrated kernels across both aligned shapes (multiples of `128`) and prime/unaligned shapes (`M=17, N=259`) against the PyTorch `FP32` reference (`torch.testing.assert_close`).
