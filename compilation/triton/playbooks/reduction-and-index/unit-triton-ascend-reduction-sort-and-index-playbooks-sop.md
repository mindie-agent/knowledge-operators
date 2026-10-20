---
category: sop
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/playbooks/reduction-and-index
evidence_strength: software_contract
id: unit-triton-ascend-reduction-sort-and-index-playbooks-sop
inherits_from:
- unit-triton-ascend-operator-domain-constraints-contract
- unit-triton-ascend-hardware-and-compiler-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/math_compute__sort.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/reduction_amax_large.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/reduction_sum_fused.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/index_computation__index_gather.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/index_computation__indexput_scatter.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/index_computation__embeddingdensebackward.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/index_histogram.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/guides/operator_playbooks/math_compute__sort.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Triton-Ascend Reduction, Sort/TopK, Index/Gather, Scatter, EmbeddingBackward,
  and Histogram Playbooks SOP
triggers:
  error_signatures:
  - 1. Reduction Playbooks (`amax`, `amin`, `sum`, `mean`, `prod`, Fused Reductions)
  - 1.1 Shape-Regime Dispatch Matrix
  - 2. `Sort` / `TopK` Playbook (`2.04x` Geomean, 31/31 Pass)
  - 2.1 Critical `tl.sort` Contract & Toolchain Boundary
  keywords:
  - unit-triton-ascend-reduction-sort-and-index-playbooks-sop
---
# Triton-Ascend Reduction, Sort/TopK, Index/Gather, Scatter, EmbeddingBackward, and Histogram Playbooks SOP

## 1. Reduction Playbooks (`amax`, `amin`, `sum`, `mean`, `prod`, Fused Reductions)

### 1.1 Shape-Regime Dispatch Matrix
Select the reduction skeleton strictly by the ratio between the non-reduction dimension $M$ and the reduction dimension $N$:

| Regime | Typical Shape `(M, N)` | Grid Mapping & Inner-Loop Strategy |
|---|---|---|
| **Small / Medium $N$ (`N <= 4096`)** | `(1024, 512)` to `(512, 4096)` | Map $M$ across cores (`grid = (min(cdiv(M, BLOCK_M), VEC_CORE_NUM),)`), load `[BLOCK_M, BLOCK_N]` (`BLOCK_N = next_pow2(N)`), and reduce in one pass via `tl.max`/`tl.sum(x, axis=1)`. |
| **Fused Elementwise + Large $N$ (`M >= VEC_CORE_NUM`, `N >= 8192`)** | `(1000, 8192)` | Set `grid = (VEC_CORE_NUM,)` (e.g., `40` cores, `BLOCK_M = 25`, `SUB_BLOCK_M = 25` with zero tail blocks). **Loop-Carried Vector Accumulator**: maintain a 2D register tile `acc = tl.zeros([SUB_BLOCK_M, BLOCK_N], tl.float32)`, accumulate `acc += tl.where(mask, val, 0.0)` inside `for n_start in range(0, N, BLOCK_N):`, and call `tl.sum(acc, axis=1)` **once after the loop** (`91.7 us -> 47.6 us`). |
| **Extreme $M \ll N$ (`M <= 16, N >= 65536`)** | `(16, 262144)` | **Map Reduction Axis $N$ Across Cores**: set `grid = (cdiv(N, BLOCK_N),)` (`BLOCK_N = 8192`) with secondary UB sub-tiling `for n_sub in range(0, BLOCK_N, SUB_BLOCK_N):` (`SUB_BLOCK_N = 512` or `1024`), reduce locally across `SUB_BLOCK_N`, and merge across programs into pre-initialized output via `tl.atomic_max` / `tl.atomic_min` / `tl.atomic_add`. |

---

## 2. `Sort` / `TopK` Playbook (`2.04x` Geomean, 31/31 Pass)

### 2.1 Critical `tl.sort` Contract & Toolchain Boundary
- **Always call `tl.sort(X, dim=1, descending=DESCENDING)`** (standard Triton bitonic hypercube sort in `triton.language.standard`). Never call `triton.language.extra.cann.extension.sort` (raises `MLIRCompilationError`).
- **`BLOCK_ROWS` Power-of-2 Invariant**: For 2D `tl.sort(X, dim=1)` on shape `[BLOCK_ROWS, L]` (`L = next_pow2(n)`), **`BLOCK_ROWS` must be a power of 2** (`1, 2, 4, 8, ...`). Non-power-of-2 `BLOCK_ROWS` fails in `hivm.hir.bitcast` alignment.
- **Normalize Target Dimension & Directional Padding**: If `dim != x.ndim - 1`, `permute` target `dim` to the last dimension on the host, sort, and `permute` back. Pad out-of-bounds columns (`col_offs >= n`) with `other = -float("inf") if DESCENDING else float("inf")`, and slice `[:, :n]` after sorting.

### 2.2 Segmented Dispatch by Last-Dimension Length `L = next_pow2(n)`
1. **Small/Medium `L` (`L <= 1024`, or `L <= 4096` when supported by `PlanMemory`)**:
   - Single-kernel `tl.sort(X, dim=1, descending=DESCENDING)` with `BLOCK_ROWS` chosen as the largest power of 2 satisfying `L * BLOCK_ROWS <= 8192` (`fp32`) or `16384` (`fp16`/`bf16`).
2. **Medium-Large `L` (`2048 <= L <= 8192` when `tl.sort` hits `PlanMemory Failed`)**:
   - Single-kernel in-UB software bitonic sort (`BLOCK_ROWS` sized so `8` live buffers of `[BLOCK_ROWS, L]` fit in `< 50 KB` UB), or 2-half `tl.sort` (using negation trick `Y = -tl.sort(-X, dim=0, descending=True)` for ascending order) + bitonic merge.
3. **Very Large `L` (`L == 16384` `fp16` Three-Stage Sort-Merge, or `L > 8192` Recursive Bitonic)**:
   - Partition into `L_TILE` chunks, sort each chunk with alternating bitonic directions (`_bitonic_chunk_directions` -> `[True, False, True, False, ...]`), then execute cross-tile elementwise `min`/`max` merge stages followed by within-tile bitonic merge.

---

## 3. `Index` / `Gather`, `IndexPut` / `Scatter`, `EmbeddingDenseBackward` & `Histogram`

### 3.1 `Index` / `Gather` (`index-read`)
- Specialize `DIM: tl.constexpr` (`0, 1, 2, 3`) on the host; compute loop bounds from `index.shape` (never `x.shape`).
- When gathering along the last dimension (or after transposing `dim` to `-1` when `gather_gain > 2 * transpose_cost`) and `T <= UB_LIMIT` (`<= 32768`), **preload the entire source row into UB once** (`row = tl.load(x_ptr + pid_m * T + tl.arange(0, UB_LIMIT), mask=...)`) and extract elements in UB via `tl.gather(row, idx * T + t_offs, mask=mask)`.

### 3.2 `IndexPut` / `Scatter` & `EmbeddingDenseBackward` (`index-accumulate`)
- **Three-Stage Decomposition for `EmbeddingDenseBackward`**:
  1. `freq_kernel`: Count occurrences of each index via `tl.atomic_add(freq_ptr + idx, 1.0, mask=mask)` when `scale_grad_by_freq=True`.
  2. `embedding_grad_kernel`: Allocate `out_fp32 = torch.zeros(..., dtype=torch.float32)`, iterate over indices in the block, and issue **vector `tl.atomic_add(out_ptr + row_idx * D + d_offs, val * scale, mask=d_mask)`** along contiguous embedding chunks `BLOCK_D`. Never extract scalars with `tl.get_element` for scalar atomic adds.
  3. `padding_zero_kernel`: Zero out the `padding_idx` row in a separate lightweight kernel outside the hot scatter loop, then cast `out_fp32` back to input dtype.

### 3.3 `Index Histogram` (Large-Scale Expert/Bin Counting, `19x` Speedup)
- Never scan $N$ elements inside each of $M$ bin programs (`O(N * M)` complexity = $524288 \times 365 \approx 1.9 \times 10^8$ ops).
- **Pre-Sort in `float32` + Per-Bin Binary Search (`O(N log N + M log N)`)**:
  1. Flatten indices and cast to `torch.float32` before calling `sorted_indices, _ = torch.sort(indices_flat)` (casting `int32 -> float32` routes `torch.sort` to Vector Core SIMD instead of AI CPU fallback, preserving exactness since indices $\ll 2^{23}$).
  2. Launch `histogram_kernel` with `grid = (num_bins,)`; each program runs a `~19`-step binary search (`while left <= right:`) to locate the lower and upper bounds of `bin_id` in `sorted_indices`.
