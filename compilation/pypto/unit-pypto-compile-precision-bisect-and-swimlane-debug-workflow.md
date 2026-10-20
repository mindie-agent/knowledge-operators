---
category: workflow
conditions:
  kernel_language: pypto
  operator_family:
  - attention
  - matmul
  - moe
  - fused_layer
  - reduction
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/pypto
evidence_strength: reproduced_empirical
id: unit-pypto-compile-precision-bisect-and-swimlane-debug-workflow
inherits_from:
- unit-pypto-tile-operator-design-api-and-execution-constraints-sop
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-perf-tune/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-perf-tune/shared/optimization_catalog.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-precision-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-develop/references/error-code-troubleshooting.md
  source_locator: cann/cannbot-skills@f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:ops/pypto-op-perf-tune/SKILL.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: PyPTO Error-Code Troubleshooting, Precision Bisect & Three-Phase Swimlane Performance
  Tuning Workflow
triggers:
  error_signatures:
  - 'Errcode: FC0000 ERR_PARAM_INVALID or FC0001 ERR_PARAM_DTYPE_UNSUPPORTED'
  - 'Errcode: FC1000 ERR_CONFIG_TILE or FC1001 ERR_CONFIG_ALIGNMENT'
  - 'Errcode: F62014 SYMBOL_NOT_FOUND or F63001 COMPILE_CODE_FAILED'
  - pypto.view followed by pypto.reshape(inplace=True) causes 96%+ elements out of
    tolerance
  keywords:
  - pypto
  - swimlane.json
  - sg_set_scope
  - vec_nbuffer_setting
  - cube_l1_reuse_setting
  - cube_nbuffer_setting
  - NONE_CACHEABLE
  - set_cache_policy
  - combine_axis
---
# PyPTO Error-Code Troubleshooting, Precision Bisect & Three-Phase Swimlane Performance Tuning Workflow

## 1. Error-Code Triage (`Errcode: Fxxxxx!`) & Compile/Runtime Diagnostics

When a PyPTO operator raises a `RuntimeError` or fails during graph compilation/execution, first inspect `stderr` (for frontend validation errors) and `pypto-log*.log` (for backend codegen/machine errors):

| Error Code / Prefix | Component | Root Cause | Mandatory Remediation |
| :--- | :--- | :--- | :--- |
| **`FC0000`** (`ERR_PARAM_INVALID`) | `VECTOR` | Rank out of supported `2..4`D range, multi-axis broadcast, or mismatched tensor dimensions | Normalize tensor rank to 2D/3D and decompose multi-axis broadcasts into single-axis steps |
| **`FC0001`** (`ERR_PARAM_DTYPE_UNSUPPORTED`) | `VECTOR` | Calling `sum`/`topk`/`softmax`/`sin`/`cos`/`sigmoid` on `DT_FP16`/`DT_BF16`, or mixed operand dtypes | Explicitly `pypto.cast` inputs to `DT_FP32` and wrap scalar literals in `pypto.Element(dtype, val)` |
| **`FC1000`** (`ERR_CONFIG_TILE`) | `VECTOR` | Missing `set_vec_tile_shapes` before op, or TileShape rank/size mismatch after `reshape`/`sum(keepdim=False)` | Call `pypto.set_vec_tile_shapes` matching current tensor rank immediately before the op |
| **`FC1001`** (`ERR_CONFIG_ALIGNMENT`) | `VECTOR` | Last dimension of TileShape or tensor slice is not `32B`-aligned (`16` FP16/BF16 or `8` FP32 elements) | Pad/concat narrow last axis to `>= 32B` alignment and pass `valid_shape` to clip logical elements |
| **`F62014`** (`SYMBOL_NOT_FOUND`) / **`F63001`** (`COMPILE_CODE_FAILED`) | `CODEGEN` | Missing PTO-ISA headers (`DivAlgorithm` check in `${PTO_TILE_LIB_CODE_PATH}/include/pto/`), parallel compile race, or invalid symbolic expression | Verify `PTO_TILE_LIB_CODE_PATH`, enable serial compilation when debugging codegen logs, and ensure `view` shapes are concrete `int` |
| **`F7XXXX`** | `MACHINE` | On-chip L1/UB/L0 buffer exhaustion or invalid tile divisibility (`L1 % L0 != 0`) | Reduce `mL1`/`nL1`/`kL1` or `vec_tile_shapes` and enforce `L0 <= L1 && L1 % L0 == 0` |

---

## 2. Priority-Ordered Precision Debugging & Bisect Workflow (`pypto-precision-debug`)

Whenever an operator fails accuracy verification (`--run-mode npu`), apply the 8 user-code workarounds in strict priority order before suspecting a framework bug:

| Priority | Failure Pattern | Workaround / Fix | Underlying Compiler Mechanism |
| :--- | :--- | :--- | :--- |
| **1 (P0)** | Non-standard decorator | Use `@pypto.frontend.jit` | Avoids legacy graph builder lowering defects |
| **2 (P0)** | `pypto.view` followed by `pypto.reshape(..., inplace=True)` (`96%+` elements wrong) | Set `inplace=False` on `reshape` after `view`, or hoist `reshape(inplace=True)` to function entry | `inplace=True` after `view` drops `view` offset and `valid_shape` metadata |
| **3 (P1)** | Accuracy corruption when `unroll_list` has multiple factors (`[4, 2, 1]`) | Force `unroll_list=[1]` in `pypto.loop(...)` | Bypasses `RegisterCopy` pass register offset bug under loop unrolling |
| **4 (P1)** | Non-deterministic/corrupted results in nested `pypto.loop` | Set `submit_before_loop=True` on inner `pypto.loop` | Submits child loop tasks before parent advances, preventing parallel overwrite |
| **5 (P1)** | Tail-block corruption on non-divisible dynamic shapes | Pass explicit `valid_shape=[actual_m, actual_n]` to `view`/`reshape` | Prevents reading garbage padding or writing past logical tensor bounds |
| **6 (P2)** | Random `NaN` or run-to-run drift in accumulators | Initialize via `pypto.zeros` or `if pypto.is_loop_begin(idx):` | `pypto.tensor(...)` allocates uninitialized memory |
| **7 (P2)** | Compiler pass eliminates or reorders math | Append `+ 0.0` (`result = compute(...) + 0.0`) | Acts as an algebraic barrier preventing unsafe pass fusion/reordering |
| **8 (P2)** | Boundary pass bug when trailing axis is `1` | Reshape/concat to avoid trailing axis `== 1` | Avoids degenerate `valid_shape` propagation edge cases |

- **Midpoint Bisect (`pypto-precision-compare`)**: If all 8 workarounds fail, write intermediate tensors to an output buffer at the kernel midpoint and compare against PyTorch Golden intermediates to isolate the exact offending op.

---

## 3. Three-Phase Swimlane Performance Tuning Workflow (`pypto-op-perf-tune`)

Performance tuning requires **100% passing accuracy** (`--run-mode npu`). Collect hardware swimlane traces via `debug_options={"runtime_debug_mode": 1}` on `@pypto.frontend.jit` (emits `swimlane.json` and `dyn_topo.txt`; **remove `debug_options` before delivery**).

### 3.1 Phase 1: Out-of-the-Box Frontend Tuning (`PHASE_FRONTEND`)

1. **Hoist & Eliminate Redundant Reshapes (`[F-4]`)**: Move `pypto.reshape(..., inplace=True)` on original function inputs to kernel entry before all loops; replace input `squeeze`/`unsqueeze` with entry-level `reshape(inplace=True)`. Never use `inplace=True` on views or output tensors.
2. **Loop & Granularity Optimization (`[F-1..F-8]`)**:
   - Replace `pypto.loop` on static axes with Python `for i in range(N):` (`[F-5]`).
   - Block large outer dynamic loops (`[F-7]`) and apply `unroll_list=[8, 4, 2, 1]` only on the innermost loop (`[F-8]`).
   - Fuse `transpose + matmul` into `pypto.matmul(..., a_trans=True/b_trans=True)` (`[F-13]`) and replace `concat` with `pypto.assemble` (`[F-14]`).
   - For binary ops with a contiguous `[M, 1]` broadcast operand (from `sum`/`amax(keepdim=True)` or GM), set `pypto.experimental.set_operation_options(combine_axis=True)` at JIT entry (`[F-15]`).

### 3.2 Phase 2: Swimlane-Driven Deep Tuning (`PHASE_SWIMLANE`)

Inspect `swimlane.json` metrics (Core Utilization, Bubble Rate, and per-`leafHash` `total(us)`):
1. **Core Filling (`[S-1, S-2]`, Core Utilization `< 50%`)**: Identify `NOT FULL` subgraphs (`used_cores < physical_cores`) and shrink `mL0/mL1` or `nL0/nL1` (keeping `L1 % L0 == 0`) to increase parallel task count until all AIC/AIV cores are saturated.
2. **Load Balancing & L2-Optimal Matmul Core Layout (`[S-3, S-11, S-13]`)**:
   - Sort subgraphs by `total(us)` and tune the bottleneck subgraph's `TileShape` until the gap between the top two subgraphs is `< 20%`.
   - For `M=1` decode Matmul, use asymmetric 3-element K tiling `k_tile = [kL0, kAL1, kBL1]` with `kAL1 = K` so activation A stays in L1 while weight B streams in `kBL1 = 256` chunks (`[S-11]`).
   - For large square Matmul (`M=N=K=6144`), add an outer M/N blocking loop satisfying `nDim * nL1 = mDim * mL1` (`[S-13]`, `+31%` throughput).
3. **Subgraph Fusion & Buffer Reuse (`[S-4..S-10]`, Bubble Rate `> 10%`)**:
   - **Vector Manual Subgraph Scope (`[S-4]`, `sg_set_scope`)**: Group adjacent, directly dependent AIV ops at the same loop level (with no interleaved Cube ops) into a single `sg_set_scope` block.
   - **Vector Auto N-Buffer (`[S-5]`)**: For short AIV tasks (`< 10 us`), set `pass_options={"vec_nbuffer_setting": {-2: 1, -1: 2}}` (must include `-2: 1`).
   - **Cube L1Reuse & CubeNBuffer (`[S-6, S-7]`)**: When AIC cores are full and M/N tiles reload weights, set `pass_options={"cube_l1_reuse_setting": {-1: 2, 0: 8}}` and `{"cube_nbuffer_setting": {-1: 2}}` (never pass empty `{}`).
   - **Stitch & Schedule (`[S-9, S-10]`)**: Tune `runtime_options={"stitch_function_max_num": 128, "device_sched_mode": 1}` (`0..3`).

### 3.3 Phase 3: In-Core Instruction & Cache Tuning (`PHASE_INCORE`)

1. **Batch L2 Cache Bypass for Large Weights (`[I-2]`)**: In fused layers with multiple large weights (`qkv_weight`, `o_weight`, `gate_weight`, `up_weight`, `down_weight`) read once without reuse, call `weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)` on **all** weights simultaneously (`-19.1%` latency) to prevent L2 activation eviction. Never set `NONE_CACHEABLE` on small activations, outputs, or a single weight in isolation.
2. **Degenerate Matmul Reshape (`[I-1]`) & De-Dependency (`[I-3]`)**: Reshape extreme Matmuls (e.g., `(884736, 16) x (16, 16)`) via Vector preprocessing (`500 us -> 40 us`), and duplicate small shared bias tensors across `psgId` branches in `dyn_topo.txt` (`[I-3]`) to break 1-to-many serialization.
