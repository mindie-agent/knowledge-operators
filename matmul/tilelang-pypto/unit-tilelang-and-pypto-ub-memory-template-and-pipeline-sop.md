---
schema: mindie-graded-unit/1
id: unit-tilelang-and-pypto-ub-memory-template-and-pipeline-sop
revision: 1
tier: L2
category: sop
domain_path: matmul/tilelang-pypto
evidence_strength: software_contract
title: 'TileLang-Ascend and PyPTO operator SOP: DMA capacity maximization, branchless vectorization, three-phase swimlane tuning and Stage-5 memory ledger'
conditions:
  kernel_language:
  - tilelang
  - pypto
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-catlass-and-tilelang-ascend-operator-design-and-tuning-sop
provenance:
  source_type: public_software_contract_review
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-perf-optimization/SKILL.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  supporting_source_revisions:
  - be5882ef29fa9943305085652d2025a038ba491c
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-perf-optimization/Ascend910/instructions.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-perf-optimization/Ascend910/references/performance-antipatterns.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-perf-optimization/Ascend950/instructions.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-perf-tune/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-perf-tune/shared/optimization_catalog.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-memory-template/SKILL.md
  supersedes: []
triggers:
  error_signatures:
  - a TileLang wrapper calls permute().contiguous() or forces a 2D reshape and degrades dim != -1 cases by 10x to 30x
  - T.if_then_else inside T.Parallel degrades to a serial scalar loop in ascend_lower_parallel
  - reduce_sum(dim=0) inside a multi-row inner loop produces NaN or per-iteration reduction overhead
  - a PyPTO operator inplace-reshapes an output tensor and produces all-zero outputs due to broken slice indexing
  - PyPTO vec_nbuffer_setting is configured without -2:1 or cube_nbuffer_setting uses an empty dict {} and regresses latency
  keywords:
  - TileLang
  - PyPTO
  - PTO-ISA
  - TL_ASCEND_MEMORY_PLANNING
  - TL_ASCEND_AUTO_CV_COMBINE
  - T.if_then_else
  - mul_add_dst
  - runtime_debug_mode
  - merged_swimlane.json
  - set_cube_tile_shapes
  - set_vec_tile_shapes
  - sg_set_scope
  - vec_nbuffer_setting
  - cube_l1_reuse_setting
  - NONE_CACHEABLE
  - valid_shape
  - MEMORY.md
---

# TileLang-Ascend and PyPTO operator SOP: DMA capacity maximization, branchless vectorization, three-phase swimlane tuning and Stage-5 memory ledger

## Use when

A TileLang-Ascend or PyPTO operator on Ascend 910B/910_93 (`Ascend910`, `NpuArch=2201`) or Ascend 950 (`Ascend950`, `NpuArch=3510`) is being audited for DMA/Vector antipatterns or tuned across frontend, swimlane and in-core stages. This SOP governs TileLang DMA/Vector lowering rules and the PyPTO three-phase tuning and `MEMORY.md` workflow.

## 1. Route by platform and enforce the multi-version UB/L1 memory ledger

Never mix `Ascend910` (`NpuArch=2201`: UB `196,352 B` / 192 KiB, L1 `524,032 B` / 512 KiB, L0C 128 KiB, L2 ~201 MB) and `Ascend950` (`NpuArch=3510`: UB 248 KiB after 8 KiB system reserve, or ~208 KiB when any `T.SimtVF` block reserves 40 KiB, L0C 256 KiB) branches.

Inspect generated Ascend C source via `func.get_kernel_source()` (`IS_ASCEND_AIC` = Cube, `IS_ASCEND_AIV` = Vector, both = CV mixed). Enable compiler buffer reuse while sizing the explicit worst-case footprint:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}
```

- **Pure-Vector `AUTO_CV_COMBINE` hazard**: if `get_kernel_source()` contains only `IS_ASCEND_AIV` and the kernel uses `T.alloc_var`, verify whether `TL_ASCEND_AUTO_CV_COMBINE: True` splits variable definitions and uses across cores; if so, disable `TL_ASCEND_AUTO_CV_COMBINE`.
- **Execution order (`[ORDER-PLAN]`)**: apply layout-changing edits first, buffer-count/pipeline edits (`T.Pipelined` / double buffer) second, tile-budget sizing third, and `pass_configs` last—one change per `msprof op` verification cycle.

## 2. Eliminate the five TileLang DMA and Vector lowering antipatterns

Every DMA call and tile step pays a fixed setup cost ($\text{eff} = \text{useful\_bytes} / (\text{useful\_bytes} + \text{fixed\_cost})$). Enforce the five-link chain:

1. **Zero wrapper copies via 3D logical decomposition (`wrapper_copy_bytes == 0`)**: forbid `x.permute(...).contiguous()`, `.to(dtype)` or forced 2D `TORCH_CHECK(x.dim() == 2)` in wrappers (which cause 10x-30x regressions on `dim != -1`). Decompose arbitrary shapes into `[outerSize, dimSize, innerSize]` and index GM directly with `rowSize` and `inputRowSize` strides.
2. **2D strided DMA over row-by-row `DataCopyPad` loops**: replace `for r in range(rows): DataCopyPad(...)` with a single 2D `DataCopyExtParams` (`blockCount = numRows`, `blockLen = rowSize * sizeof(T)`, `srcStride`) to avoid $N\times$ DMA setup overhead.
3. **Branchless `T.tile` vectorization over `T.if_then_else` and row loops**:
   - `T.if_then_else` inside `for i, j in T.Parallel(...)` cannot be vectorized by `ascend_lower_parallel` and degrades to a serial scalar loop. Rewrite with branchless algebraic identities (e.g., $\ln(1 + e^x) = \max(x, 0) + \ln(1 + e^{-|x|})$ via `T.tile.max`, `T.tile.abs`, `T.tile.exp`).
   - Replace `for row in range(block_M): T.tile.sub(acc[row, :], ...)` with `T.tile.broadcast(max_2d, max_ub, axis=1)` followed by whole-tile `T.tile.sub`.
   - Fuse compound ops: `mul + add -> T.tile.mul_add_dst` (or `T.tile.axpy`), `max(x, 0) -> T.tile.relu`, `sqrt + div -> T.tile.rsqrt`.
4. **Loop-carried `[rows, block_S]` accumulation and orthogonal-axis transposition**:
   - Never call `T.reduce_sum` inside a multi-row inner tile loop or use `T.reduce_sum(..., dim=0)` on `[rows, 1]` (NaN risk). Accumulate into `accum [rows, block_S]` with `T.tile.add` inside the loop, then execute two `T.reduce_sum(..., dim=-1)` calls outside (`[rows, block_S] -> [rows] -> [1]`).
   - When an inner loop has a sequential scan dependency along `L` (`v[r, i] = f(a[r, i], v[r, i-1])`) across independent rows `r`, transpose `[Rows, L] -> [L, Rows]` so `Rows` is contiguous and vectorize with `T.tile.compare`, `T.tile.select`, and `T.tile.min`.
5. **Fixed-core launch and pipeline staging**: launch `min(block_num, physical_cores)` cores with an in-core serial loop (`T.ceildiv(block_num, core_num)`), never `T.Kernel(block_num)` when `block_num > physical_cores`. Enable `T.Pipelined(loop_k, num_stages=2)` (with `TL_ASCEND_AUTO_CV_COMBINE` and `TL_ASCEND_AUTO_CV_SYNC` on mixed CV kernels); never nest `T.Pipelined` loops.

## 3. Execute the PyPTO three-phase single-variable tuning workflow

PyPTO tuning requires a passing accuracy baseline (`--run-mode npu`), sets `@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1})` during collection (`merged_swimlane.json`, `bubble_analysis.log`, `machine_runtime_operator_trace.json`), changes **one parameter per iteration** with accuracy re-verification, and **removes `debug_options` before final sign-off**.

### Phase 1: `PHASE_FRONTEND` (Code structure and initial TileShapes)

- **Input `reshape(inplace=True)` hoisting (`F-4`)**: hoist `pypto.reshape` (and replace `pypto.squeeze`/`unsqueeze` with `pypto.reshape(..., inplace=True)`) on **function input parameters** outside all loops. **Never** apply `inplace=True` to intermediate or output tensors—inplace-reshaping an output tensor breaks slice write indexing and yields all-zero outputs.
- **Loop and TileShape contracts (`F-5`, `F-8`, `F-9`, `F-10`, `F-15`)**: replace static-extent `pypto.loop` with Python `range`; place `unroll_list` only on the innermost dynamic loop; set `pypto.set_cube_tile_shapes` with `L0 <= L1` and `L1 % L0 == 0` (for decode `M = 1`, use 3D K-config `[kL0, kAL1=K, kBL1=256]` to keep `A` in L1); set `pypto.set_vec_tile_shapes(64, 512)` (**always reset `set_vec_tile_shapes` to the target shape after `reshape` and before `assemble`**); enable `pypto.experimental.set_operation_options(combine_axis=True)` for contiguous tail-1 broadcasts.

### Phase 2: `PHASE_SWIMLANE` (Core occupancy, subgraph fusion and L2 layout)

- **Core filling and load balance (`S-1`..`S-3`)**: inspect per-`leafHash` core usage (`analyze_core_usage.py`); if `NOT FULL`, shrink `nL0`/`nL1` (`256 -> 128 -> 64`) to increase task count before attempting subgraph fusion. Balance bottleneck subgraphs until the top-two `total(us)` gap is `< 20%`.
- **AIV and AIC subgraph fusion (`S-4`..`S-8`)**:
  - Wrap contiguous, directly dependent AIV ops at the same loop level (no intervening Cube ops) in `sg_set_scope` (`S-4`).
  - For short AIV tasks (`< 10 us`), set `pass_options={"vec_nbuffer_setting": {-2: 1, -1: 2}}` (`-2: 1` is mandatory, `S-5`).
  - Once AIC cores are full, apply `cube_l1_reuse_setting: {-1: 2, 0: 8}` (`S-6`) and `cube_nbuffer_setting: {-1: 2}` (`S-7`; **never** pass empty `{}` auto-mode).
- **L2-optimal Matmul core layout (`S-13`)**: for large GEMMs with low L2 hit rate, add an outer M/N loop so per-wave core counts `mDim, nDim` and L1 tiles `mL1, nL1` satisfy $n_{\text{Dim}} \cdot n_{L1} = m_{\text{Dim}} \cdot m_{L1}$.

### Phase 3: `PHASE_INCORE` (Instruction, cache policy and tail optimization)

- **Degenerate small-shape Matmul (`I-1`)**: preprocess `[M, 16] x [16, 16]` shapes with Vector `concat`/`reshape` into standard Cube tiles.
- **Batch `NONE_CACHEABLE` on streaming weights (`I-2`)**: in fused multi-weight operators, apply `w.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)` to **all** streaming weight matrices simultaneously so single-use weights do not evict activations from L2. Never set `NONE_CACHEABLE` on a single weight in isolation or on inputs/outputs.
- **Sub-loop overlap and tail masking (`I-8`, `I-9`)**: pass `submit_before_loop=True` on inner loops to overlap DMA and compute; pass `valid_shape` to `pypto.view` on partial tail blocks to skip zero-padded compute.

## 4. Maintain the PyPTO Stage-5 `custom/<op>/MEMORY.md` contract

In multi-module PyPTO development, Stage 5 implementation agents own `custom/<op>/MEMORY.md` (seeded from `SPEC.md`, `DESIGN.md`, and `module_interfaces.yaml`): update `active_module`, `modules_pypto_verified`, `current_staged_file`, and `next_mandatory_step` every turn, cross-check `Golden function inventory` and `Module contracts` after each Phase $M_k$ boundary verification, and log every fix in the `Development & debug log`.
