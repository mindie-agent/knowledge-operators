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
domain_path: compilation/triton/optimizations/tiling-and-ub-budget
evidence_strength: reproduced_empirical
id: unit-triton-ascend-tiling-persistent-grid-and-ub-budget-workflow
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/techniques/tiling_and_grid_910_95.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/tiling_optimization.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/operator_optimization/vector_core_partition.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/intermediate_ub_overflow.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/patterns/live_length_tiling.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/optimizations/techniques/tiling_and_grid_910_95.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Triton-Ascend Tiling Geometry, Persistent Grid Scheduling, and UB Budget Fitting
  Workflow
triggers:
  error_signatures:
  - 1. Platform-Specific UB Capacity & Alignment Accounting (`OPT_28`)
  - 2. Engine-Aware Grid Scheduling & Persistent Core Clamping (`OPT_29`, `OPT_41`)
  - 'Step 1: Select Physical Core Count by Execution Engine'
  - 'Step 2: Clamp 1D Grid & Balance Work Across Cores'
  keywords:
  - unit-triton-ascend-tiling-persistent-grid-and-ub-budget-workflow
---
# Triton-Ascend Tiling Geometry, Persistent Grid Scheduling, and UB Budget Fitting Workflow

## 1. Platform-Specific UB Capacity & Alignment Accounting (`OPT_28`)

Before selecting `BLOCK_SIZE`, `BLOCK_M`, `BLOCK_N`, or `BLOCK_K`, compute peak on-chip memory against the target chip's usable UB ceiling:

| Platform | Hardware UB | Usable UB (`bishengir`) | With `multibuffer=True` (2x) | Tail-Axis DMA Alignment |
|---|---|---|---|---|
| **Ascend 910B / 910_93 (A2/A3)** | `192 KB` (`1,572,864 bits`) | `~160–176 KB` (reserve `16–32 KB` scratch) | `~80–88 KB` per stage | Vector: `32 B` (`16` fp16, `8` fp32); CV Mixed: `512 B` |
| **Ascend 910_95 (A5)** | `256 KB` | `248 KB` (`2,031,616 bits`, `8 KB` reserved) | `124 KB` per stage | Vector: `32 B`; CV Mixed: `512 B` |

$$\text{Required UB} = \left(\sum_{t \in \text{Inputs} \cup \text{Outputs} \cup \text{Live Intermediates}} \text{numel}(t) \times \text{sizeof}(t)\right) \times \text{multibuffer\_copies} \le \text{Usable UB}$$

- **Scalar Reduction Scratch Rule**: Every scalar buffer (`mean`, `var`, `max`, `sum`) occupies `32 B` in UB due to hardware alignment.
- **L0C Accumulator Ceiling (Cube)**: `BLOCK_M * BLOCK_N * sizeof(acc_dtype) <= 64 KB` (`16384` elements in `fp32`/`int32`, e.g., `128 x 128` or `64 x 256`; `128 x 256 = 32768` elements hits the `128 KB` hardware ceiling and fails compilation).

---

## 2. Engine-Aware Grid Scheduling & Persistent Core Clamping (`OPT_29`, `OPT_41`)

### Step 1: Select Physical Core Count by Execution Engine
- **Pure Vector Kernel (zero `tl.dot`)**: `NUM_CORES = props["num_vectorcore"]` (`40` or `48` on A2/A3; `64` or `72` on `910_95`).
- **Cube / CV Mixed Kernel (contains `tl.dot`)**: `NUM_CORES = props["num_aicore"]` (`20` or `24` on A2/A3; `32` or `36` on `910_95`).

### Step 2: Clamp 1D Grid & Balance Work Across Cores
Never launch `grid = (total_tiles,)` when `total_tiles > NUM_CORES` (causes wave scheduling overhead or `coreDim > 65535` runtime failure) or fixed `grid = (NUM_CORES,)` when `total_tiles < NUM_CORES` (spawns empty programs):
```python
grid = (min(total_tiles, NUM_CORES),)
```
Inside the kernel, choose between two load-balancing loops:
- **Interleaved Grid-Stride Loop** (best when tiles are independent):
  ```python
  for task_id in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
      row_id = task_id // NUM_COL_TILES
      col_id = task_id - row_id * NUM_COL_TILES  # avoid % on vector/loop paths
  ```
- **Contiguous Uniform Block Splitting** (best when adjacent tiles share L2 cache lines, e.g., `RotaryMul` or `Pad`):
  ```python
  pid = tl.program_id(0)
  base = total_tiles // NUM_CORES
  rem = total_tiles - base * NUM_CORES
  start = base * pid + tl.minimum(pid, rem)
  end = start + base + tl.where(pid < rem, 1, 0)
  for task_id in range(start, end):
      ...
  ```

---

## 3. High-Yield Tiling Transformations

### 3.1 Single-Row -> Multi-Row Batching (`BLOCK_ROWS` / `OPT_28`)
- **Symptom**: Each program processes `1` row of a short row-length tensor (`N <= 512`), resulting in poor SIMD fill and high per-row scalar setup overhead.
- **Transformation**: Introduce `BLOCK_ROWS: tl.constexpr` in `{2, 4, 8, 16, 32}` (must be a power of 2 if `tl.sort` is called) so each program loads a 2D tile `[BLOCK_ROWS, BLOCK_N]` with 2D mask `(row_offs[:, None] < M) & (col_offs[None, :] < N)`.

### 3.2 Live-Length Tiling & Exact-Tile Fast Path
- **Live-Length Tiling**: Size `BLOCK_N = min(MAX_BLOCK, triton.next_power_of_2(actual_N))` rather than a static upper bound `4096` when `actual_N = 64` (prevents `98%` masked empty lanes from dominating vector execution).
- **Exact-Tile No-Boundary Fast Path**: When host dispatch proves `M % BLOCK_M == 0 and N % BLOCK_N == 0`, pass `EVEN_MN: tl.constexpr = True` to elide boundary mask generation and `tl.where` predication on the hot loop.

### 3.3 L2 Stripe-Capacity Tile Widening & N-Strip Chunking
- When a GEMM right-hand matrix or dequantized workspace `[K, N]` exceeds L2 capacity (`> 60 MB` on `96 MB` L2), first widen `BLOCK_N` from `128` to `256` (`+17–38%` MTE2 reduction), and if `[K, N]` still exceeds `60 MB`, chunk $N$ on the host into `<= 48 MB` L2-resident strips consumed across all $M$-tiles.

### 3.4 Resolving Intermediate UB Overflow via `al.extract_slice` / `al.insert_slice`
- **Symptom**: A kernel with large accumulator updates (`result = acc * alpha[:, None] + update`) fails with `ub overflow` because `acc`, `alpha`, `update`, and `result` are simultaneously live in UB, yet shrinking `BLOCK_M` degrades Cube `tl.dot` efficiency.
- **Workflow**: Keep `BLOCK_M` unchanged for `tl.dot`, and slice the epilogue/update along `BLOCK_M` into `num_slices` (`2` or `4`) using `al.extract_slice` and `al.insert_slice` so only `BLOCK_M // num_slices` rows of intermediate buffers are live at once:
  ```python
  slice_m = BLOCK_M // 4
  for s in range(4):
      off = s * slice_m
      acc_s = al.extract_slice(acc, (off, 0), (slice_m, HEAD_DIM), (1, 1))
      alpha_s = al.extract_slice(alpha, (off,), (slice_m,), (1,))
      upd_s = al.extract_slice(update, (off, 0), (slice_m, HEAD_DIM), (1, 1))
      res_s = acc_s * alpha_s[:, None] + upd_s
      acc = al.insert_slice(acc, res_s, (off, 0), (slice_m, HEAD_DIM), (1, 1))
  ```
