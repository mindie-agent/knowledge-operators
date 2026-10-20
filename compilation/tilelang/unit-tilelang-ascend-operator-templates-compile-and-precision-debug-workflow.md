---
category: workflow
conditions:
  kernel_language: tilelang
  operator_family:
  - attention
  - linear_attention
  - gemm
  - layout_transform
  - reduction
  soc:
  - Ascend910B
  - Ascend910_93
domain_path: compilation/tilelang
evidence_strength: reproduced_empirical
id: unit-tilelang-ascend-operator-templates-compile-and-precision-debug-workflow
inherits_from:
- unit-tilelang-ascend-programming-model-memory-and-cv-schedule-sop
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/tilelang/guides/chunk_gdn_kda_task_partition.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/tilelang/guides/kda_gate_factorization_and_numerical_stability.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/tilelang/guides/chunk_triangular_block_inverse.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-op-develop/Ascend910/instructions.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/tilelang-perf-optimization/Ascend910/instructions.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/tilelang/guides/chunk_gdn_kda_task_partition.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: TileLang-Ascend Operator Templates, Host-Pure Gate, Performance Audit & Precision
  Debug Workflow
triggers:
  error_signatures:
  - 'TVMError: Memory allocation failed for buffer required X available Y'
  - 'InternalError: Check failed: pb->value != 0 : Divide by zero'
  - 'error: Source and Dest dimension must match in T.tile.broadcast'
  - aclnnInplaceRandom or aclnnCopy triggered from host wrapper or NPU test input
    setup
  - KDA gate factorization exp(-g_j) FP32 overflow or backward upper-triangular NaN
  keywords:
  - tilelang
  - chunk_gated_delta_rule
  - KDA
  - GDN
  - triangular_block_inverse
  - HOST-METADATA-AUDIT
  - REORDER-COST-AUDIT
  - get_kernel_source
  - msprof
---
# TileLang-Ascend Operator Templates, Host-Pure Gate, Performance Audit & Precision Debug Workflow

## 1. Host-Side Purity Gate & Data Reorder Cost Audit (`tilelang-op-develop`)

### 1.1 Five Mandatory Host-Side Buffer Prohibitions (`[HOST-METADATA-AUDIT]`)

All core math, padding, casting, and layout reordering must execute inside the `@tilelang.jit` kernel. Host Python wrappers must **never** trigger implicit `aclnn` calls on NPU tensors:
1. **No Data Pointer Rebinding**: Never rebind an input/output tensor to a newly allocated tensor (`data_ptr` change).
2. **No Physical Host Reordering**: Never call `.contiguous()`, `x_perm.reshape(-1)` on non-contiguous views, or `y = y[:, :, :, :S]; y.reshape(shape)` (which triggers `aclnnCopy`). Clip output tails directly inside the kernel via `T.copy` with `BufferRegion` valid extents.
3. **No Host In-Place Mutation**: Never run `x[:] = ...`, `x.add_(...)`, or `torch.mul(..., out=x)` outside the kernel.
4. **No Implicit `aclnn` Ops**: Never call `F.pad` (`aclnnPad`), `torch.cat`/`stack`, `.to(dtype)` (`aclnnCast`), or `.clone()` on NPU tensors in the wrapper.
5. **CPU-Only Test Input Setup**: Generate random inputs, inject `NaN`/`Inf` masks, cast dtypes, and compute Golden outputs on **CPU**, then run one H2D `.npu()` copy before launch and one D2H `.cpu()` copy for check. Calling `torch.rand`/`randint` on NPU fails with `aclnnInplaceRandom` in stripped benchmark runners.

### 1.2 Six-Step Data Reorder & `[REORDER-COST-AUDIT]` Gate

Whenever an operator performs layout permutation, gather/scatter, or multi-dtype adaptation:
1. Pass strides/shapes as Python compile-time constants to `@tilelang.jit` and accumulate offsets with `T.alloc_var("int32", init=...)` in-kernel; never pack strides into a runtime `int32` GM tensor (avoids an extra `GM -> UB` read).
2. Inspect `func.get_kernel_source()` for every supported dtype. Reject any path where `GetValue`/`SetValue` expands scalar GM accesses over `numel`, or row-by-row `DataCopyPad` loops issue hundreds of thousands of short DMAs.
3. Aggregate contiguous suffix records into 2D block `T.copy` transfers (`GM -> UB`), reorder strictly inside UB, and write back coalesced blocks (`UB -> GM`).
4. Verify the 5-item DMA capacity chain via `msprof op --kernel-name="main_kernel"`:
   - ① Zero host full-tensor copies (`forward()` copy count `= 0`)
   - ② 2D strided DMA instead of row-by-row `CopyIn` loop
   - ③ Effective tile utilization `actual_tileSize / TILE_SIZE >= 80%`
   - ④ Double buffering (`BUFFER_NUM >= 2` / `T.Pipelined(..., num_stages=2)`) after `[DB-ANALYSIS]`
   - ⑤ Active core utilization `usedCoreNum / physicalCoreNum >= 90%`

---

## 2. Production Operator Templates: Chunked GDN / KDA & Triangular Block Inverse

### 2.1 Chunked GDN & KDA Task Partitioning (`chunk_size = 64`)

Linear attention operators (Gated DeltaNet / GDN with scalar gate $g \in [B, T, H]$ and KDA with per-channel vector gate $g \in [B, T, H, K]$) decompose sequence recurrences into chunk-local matrix ops (`BT = 64`) and inter-chunk state recurrence:

| Stage | Recurrence Dependency | Parallelizable Axes (`cid`) | Tile Sizes & Core Type |
| :--- | :--- | :--- | :--- |
| **Stage 1 (`cumprod` / `cumsum` + `k_cumdecay` / `A` matrix)** | Intra-chunk only (`BT = 64`) | `B * H * num_chunks` | `BT=64, BK=64`; Vector-only (GDN `cumsum`) or CV Fusion (KDA per-channel `gemm_v0`) |
| **Stage 2 (`chunk_gated_delta_rule_fwd_h` state recurrence)** | Sequential across chunks `i_t = 0 .. NT-1` | `N * H * ceil(V / BV)` (`N` = batch or varlen sequences) | `BT=64, BK=64, BV=64`; `T.serial(NT)` outer loop over chunks, updating `b_h` (`[BK, BV]` FP32 in L1/UB) |
| **Stage 3 (`WY` representation & `fwd_o` output)** | Independent per chunk once `h` is materialized | `B * H * num_chunks * ceil(V / BV)` | `BT=64, BK=64, BV=64`; Cube `Q @ K^T` + causal mask + `P @ V` |

- **Varlen Sequence Mapping (`cu_seqlens`)**: Precompute a `chunk_indices` pair table `[total_chunks, 2]` mapping each logical chunk `i_n` to `(seq_idx, local_chunk_idx)`. Inside the kernel, read `bos = cu_seqlens[seq_idx]`, `eos = cu_seqlens[seq_idx + 1]`, `T_len = eos - bos`, and clip tail chunks with `actual_len = T.min(BT, T_len - i_t * BT)`.

### 2.2 KDA Gate Factorization & Numerical Stability (`exp(g_i - g_j)`)

For KDA where $g \in \mathbb{R}^{B \times T \times H \times K}$ is cumulative log-decay (`g <= 0`, non-increasing within a chunk, down to `-50`):
- **Never** factor $\exp(g_i - g_j) = \exp(g_i) \cdot \exp(-g_j)$ relative to chunk start `g[0]`: $\exp(-g_j)$ overflows FP32 (`> 10^{21}`) and loses precision against $\exp(g_i) \approx 10^{-22}$.
- **Causal Forward (`i >= j`) — Midpoint Pivot $p = \text{BT}/2 - 1$ (`row 31` for `BT = 64`)**:
  - Diagonal (`TL` `0..31`, `BR` `32..63`): Compute exact $\Gamma_{i,j,k} = \exp(g_{i,k} - g_{j,k})$ in Vector UB (with `i > j` mask) and reduce over `K`.
  - Off-diagonal (`BL`, `i in 32..63`, `j in 0..31`): Factor through pivot $g_{p,k}$ (`row 31`):
    $$g_{i,k} - g_{j,k} = (g_{i,k} - g_{p,k})_{\le 0} + (g_{p,k} - g_{j,k})_{\le 0}$$
    Scale `q_scaled[i, k] = q[i, k] * exp(g[i, k] - g[p, k])` and `k_scaled[j, k] = k[j, k] * exp(g[p, k] - g[j, k])`. Both factors stay in $(0, 1]$, enabling overflow-free Cube `T.gemm_v0(q_scaled, k_scaled, S_BL, transpose_B=True)`.
- **Strict-Upper Backward (`i < j`, `dA`)**: Only `i < j` entries are valid where $g_i - g_j \ge 0$, so midpoint pivot $p = 32$ overflows! Use **row-min stabilization** on `q` (`m_i = max_k(g_{i,k})`) and **col-max stabilization** on `k` (`M_j = min_k(g_{j,k})`), rescaling `dA[i, j]` by $\exp(m_i - M_j)$ after GEMM before masking `i < j`.

### 2.3 Chunk Triangular Block Inverse $(I + A)^{-1}$ (`BT = 64 -> 4 x 16x16`)

To invert $L = I + \text{strictLower}(A)$ of shape `64 x 64` within Cube fractal limits (`16 x 16`):
1. **Diagonal `16 x 16` Blocks ($T_{11..44}$)**: Invert each `16 x 16` block in **Vector UB** via 15-step serial row recurrence (or Neumann doubling $T = (I - A + A^2)(I + A^4)(I + A^8)$) using broadcast + `T.tile.mul` + `T.reduce_sum`.
2. **Off-Diagonal `16 x 16` Blocks ($T_{21..43}$)**: Compute in **Cube L1/L0C** via `T.gemm_v0` (up to 9 `16 x 16` GEMMs):
   - $T_{21} = -T_{22} A_{21} T_{11}$, $\quad T_{32} = -T_{33} A_{32} T_{22}$, $\quad T_{31} = -T_{33} (A_{31} T_{11} + A_{32} T_{21})$
   - $T_{43} = -T_{44} A_{43} T_{33}$, $\quad T_{42} = -T_{44} (A_{42} T_{22} + A_{43} T_{32})$, $\quad T_{41} = -T_{44} (A_{41} T_{11} + A_{42} T_{21} + A_{43} T_{31})$

---

## 3. Compilation, Autotune & Precision Troubleshooting Checklist

| Symptom / Error Message | Root Cause | Fix Action |
| :--- | :--- | :--- |
| `TVMError: Memory allocation failed for: ...` | UB (`192KB`) or L1 (`512KB`) exceeded | Enable `TL_ASCEND_MEMORY_PLANNING: True`, shrink `block_M`/`block_N`, or reuse UB buffers |
| `Check failed: pb->value != 0 : Divide by zero` | `n_num = N // block_N == 0` when `block_N > N` | Compute grid counts with `T.ceildiv(N, block_N)` instead of `//` |
| `Source and Dest dimension must match` | 1D `[M]` passed to 2D `T.tile.broadcast` | Allocate reduction buffer as 2D `[M, 1]` or `[1, N]` |
| Autotune `get_configs` / `supply_prog` error | `get_configs` gets `(key_args, key_kwargs)`; `supply_prog` has only inputs | Use `def get_configs(key_args, _kw=None): M, N, K = key_args` and index only input `params` |
| Position-sensitive `NaN`/`Inf` mismatch | `equal_nan=True` masks `NaN` diffusion | Assert exact `isnan()`, `isposinf()`, `isneginf()` masks before checking `atol`/`rtol` on finite elements |
