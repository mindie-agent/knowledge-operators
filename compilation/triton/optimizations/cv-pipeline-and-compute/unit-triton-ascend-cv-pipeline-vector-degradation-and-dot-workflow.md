---
category: workflow
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/optimizations/cv-pipeline-and-compute
evidence_strength: reproduced_empirical
id: unit-triton-ascend-cv-pipeline-vector-degradation-and-dot-workflow
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/techniques/cv_pipeline_optimization.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/techniques/scalar_degradation_avoidance.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/cv_fusion_pingpong.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/chained_matmul_buffer_dtype.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/cube_mte3_decoupling.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/hilo_split_matmul.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/optimizations/techniques/cv_pipeline_optimization.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Triton-Ascend Cube-Vector Pipelining, Scalar Degradation Avoidance, and Chained
  MatMul Workflow
triggers:
  error_signatures:
  - 1. Cube-Vector (CV) Mixed Pipeline Optimization (`OPT_33`, `OPT_35`)
  - '1.1 Hardware Dataflow: A2/A3 (Mem-Based) vs. 910_95 (Reg-Based)'
  - 1.2 CV Instruction Reordering & Helper Encapsulation (`+10–35%` Speedup)
  - 1.3 Ordered 8-Step CV Precision Rollback Protocol
  keywords:
  - unit-triton-ascend-cv-pipeline-vector-degradation-and-dot-workflow
---
# Triton-Ascend Cube-Vector Pipelining, Scalar Degradation Avoidance, and Chained MatMul Workflow

## 1. Cube-Vector (CV) Mixed Pipeline Optimization (`OPT_33`, `OPT_35`)

### 1.1 Hardware Dataflow: A2/A3 (Mem-Based) vs. 910_95 (Reg-Based)
Each AI Core pairs `1` Cube Core with `2` Sub-Vector Cores (`1:2` ratio). Always launch CV mixed kernels with `grid = (num_aicore,)`:
- **Ascend 910B / 910_93 (Mem-Based FFTS)**: Cube writes `L0C --PIPE_FIX--> GM workspace --PIPE_MTE2--> UB` (`2` GM touches per cross-core transfer). Therefore, **always compute segmented GM bandwidth before fusing or de-mixing Cube and Vector stages**: if an epilogue is lightweight, fusing into Cube (`FixPipe`) or keeping an `fp16` GM handoff between two dedicated kernels can beat a tightly coupled MIX loop if de-mixing doubles intermediate dtype width (`int32` vs `fp16`).
- **Ascend 910_95 (Reg-Based)**: Supports direct `L0C --al.fixpipe--> UB` zero-GM transfers and register-level `SetFlag`/`WaitFlag` across `16` `event_id` channels (`0..15`), requiring `512 B` tail-axis alignment.

### 1.2 CV Instruction Reordering & Helper Encapsulation (`+10–35%` Speedup)
1. **Schedule `tl.dot` Ahead of Independent Vector Reductions**:
   In Attention loops, issue `v = tl.load(V_bp, ...)` and `pv = tl.dot(p.to(v.dtype), v)` **before** `p_sum = tl.sum(p, 1)` and `alpha = tl.math.exp(m - m_new)`, then combine `acc = acc * alpha[:, None] + pv`. This allows Cube `MMAD` (`pv`) and Vector `PIPE_V` (`p_sum`, `alpha`) to execute concurrently.
2. **Helper Function Encapsulation for `auto-multi-buffer`**:
   In `PagedAttention` prefill, wrap the single-KV-block body in a `@triton.jit` helper function with strict op order (`early V-load -> QK dot -> exp -> propagate_nan -> separated scale/sub -> 3-arg tl.dot(p, v, acc)`). Inlining the helper or delaying `V` load prevents `bishengir` `auto-multi-buffer` from double-buffering `K`/`V` (`90.7 us -> 54.5 us`).
3. **Breaking Overlong Vector Fusion Chains (`break_vf`)**:
   In backward pass kernels (`bwd_qkv`), insert `al.compile_hint(ds, "break_vf")` immediately before `tl.dot(ds, k)` to prevent the compiler from fusing all upstream vector ops into one giant UB live range.

### 1.3 Ordered 8-Step CV Precision Rollback Protocol
If adding the 7 CV launch kwargs (`enable_mixed_cv=True`, `sync_solver=True`, `limit_auto_multi_buffer_of_local_buffer="no-limit"`, `set_workspace_multibuffer=2`, `enable_auto_bind_sub_block=True`, `multibuffer=True`, `enable_flatten=False`) causes numerical drift, roll back **one item at a time** in this exact order:
`1. remove enable_mixed_cv` -> `2. sync_solver=False` -> `3. remove no-limit` -> `4. remove set_workspace_multibuffer=2` -> `5. enable_auto_bind_sub_block=False` -> `6. multibuffer=False` -> `7. remove propagate_nan` -> `8. restore original instruction order`.

---

## 2. Diagnosing & Eliminating HIVM Scalar Lowering (`OPT-4`, `OPT_27`)

When a Vector kernel suffers a `10x–100x` slowdown (`aiv_scalar_ratio > 30%` or `hivm` IR contains `scf.for` step-1 loops generated by `ShouldLowerToScalarLoops.cpp`), check the exact hardware lowering matrix:

| Operation Class | Lowered as Vector SIMD (`PIPE_V`) | Degrades to Scalar Loop (`PIPE_S`) | Mandatory Avoidance Rule |
|---|---|---|---|
| **Arithmetic (`+,-,*,min,max,abs,<<,>>`)** | `fp16`, `fp32`, `int8`, `int16`, `int32` | **`int64` always degrades**; `VMulExt` (`i32/i64`) always degrades | Cast `tl.program_id` and loaded indices `.to(tl.int32)` immediately; promote only final 64-bit base pointers. |
| **Comparisons (`VCmp`)** | `fp16`/`fp32` (`==,!=,<,>,<=,>=`); `int32` (`==,!=` only) | **`int32` (`<,>,<=,>=`)** and **all `int8`/`int16`/`int64` compares** (when stored into a named boolean outside inline `tl.load`/`tl.store` `mask=`) | Cast `int32`/`int64` operands to `tl.float32` before `<, >, <=, >=` when values lie in `[-2^24, 2^24]`. **Warning**: Never use `fp32` comparison for Cube `MTE` load/store masks in `GroupedMatmul` (causes `507014` hang). |
| **Reductions (`sum,prod,max,min`)** | **910_95**: all dtypes (including `i64`); **910B/A3**: `fp16`, `bf16`, `fp32`, `int32` | **910B/A3**: `int64` degrades | Cast `int64` values to `tl.float32` or `tl.int32` before `tl.sum`/`tl.max` on 910B/A3. |
| **Index Reductions (`argmax,argmin`)** | **910_95**: 32B-aligned with `StridedLayoutAttr`; **910B/A3**: `fp16`/`bf16`/`fp32` with rank `<= 2` | **910B/A3**: `int16`/`int32`/`int64` always degrade; rank `> 2` degrades | On 910B/A3, cast integer tensors `.to(tl.float32)` before `tl.argmax`/`tl.argmin` and flatten to `<= 2D`. |
| **Scans (`cumsum,cumprod`)** | Non-last axis (`axis != -1`) on `fp16`/`fp32`/`int32` | **Last axis (`axis == -1`)** or `int64` on any axis | Transpose tile first (`tl.trans(x)`), run `tl.cumsum(..., axis=0)` (or non-last axis), then `tl.trans` back. |

---

## 3. Chained MatMul Dtype & Hi/Lo Split `fp32` MatMul (`OPT-6`, `OPT-8`, `hilo_split_matmul`)

1. **Chained `tl.dot` Intermediate Buffer Dtype (`OPT-6`)**:
   - In multi-stage `tl.dot` pipelines ($Q \cdot K^T \to P \cdot V$), keep the accumulator in `tl.float32`, but cast the intermediate matrix $P$ to `v.dtype` (`fp16`/`bf16`) before the second `tl.dot(p.to(v.dtype), v, acc)` so the second matmul runs at full Cube throughput (`16x` higher IPC than `fp32` dot).
2. **Hi/Lo Split `fp32` MatMul (`Split2` / `Split3`) for Strict `fp32` Contracts**:
   - When inputs are `fp32` and native Cube `tf32` mode fails `rtol=1e-6`, split each `fp32` tile into `hi = x.to(tl.float16)` (or `bfloat16`) and residual `lo = (x - hi.to(tl.float32)).to(tl.float16)`, then accumulate:
     $$\text{acc} \mathrel{+}= \text{tl.dot}(a_{\text{hi}}, b_{\text{hi}}) + \text{tl.dot}(a_{\text{hi}}, b_{\text{lo}}) + \text{tl.dot}(a_{\text{lo}}, b_{\text{hi}})$$
3. **Cube / MTE3 Decoupling to Eliminate Cross-Program Atomic Contention (`OPT-8`)**:
   - Never split a reduction dimension across programs if every program must `tl.atomic_add` a `[BLOCK_M, BLOCK_D]` Cube tile to the same output token address. Either align the parallel grid with disjoint output tokens, or write per-split partials to a contiguous workspace `[NUM_SPLITS, M, D]` and merge in a lightweight Vector combine kernel.
