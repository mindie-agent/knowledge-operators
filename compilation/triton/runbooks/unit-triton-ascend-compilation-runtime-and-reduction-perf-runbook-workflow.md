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
domain_path: compilation/triton/runbooks
evidence_strength: reproduced_empirical
id: unit-triton-ascend-compilation-runtime-and-reduction-perf-runbook-workflow
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-triton-ascend-reduction-sort-and-index-playbooks-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/index.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/elemwise_broadcast_2d.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/reduction_amax_small.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/reduction_amax_medium.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/reduction_amin_large.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/reduction_mean_large.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/reduction_prod_small.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/runbooks/performance/reduction_weighted_swiglu.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/runbooks/performance/index.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Triton-Ascend Compilation/Runtime Failure Triage and Shape-Driven Reduction
  Performance Runbook
triggers:
  error_signatures:
  - 1. Compilation & ACL Runtime Error Diagnostic Triage Table
  - 2. Shape-Driven Empirical Performance Runbook (12 Golden Cases on 40-Core Ascend
    NPU)
  keywords:
  - unit-triton-ascend-compilation-runtime-and-reduction-perf-runbook-workflow
---
# Triton-Ascend Compilation/Runtime Failure Triage and Shape-Driven Reduction Performance Runbook

## 1. Compilation & ACL Runtime Error Diagnostic Triage Table

| Error Signature | Root Cause in Triton-Ascend / BiSheng IR | Immediate Remediation |
|---|---|---|
| `ub overflow, requires N bits while 1572864 bits available` (or `[ConvertLinalgRToBinary] IR Dump After PlanMemory Failed`) | Peak live tensors in UB (multiplied by `2` when `multibuffer=True`) exceed `192 KB` (`910B`/`A3`) or `248 KB` (`910_95`); or in-kernel multi-page gather / `tl.sort` on `L >= 2048` exceeds UB allocator limits. | 1. Shrink `BLOCK_M`/`BLOCK_N`/`BLOCK_K` or set `multibuffer=False`.<br>2. Slice oversized intermediate updates with `al.extract_slice`/`al.insert_slice`.<br>3. Insert `al.compile_hint(x, "break_vf")` before `tl.dot`.<br>4. For `tl.sort` with `L >= 2048`, switch to chunked/recursive bitonic merge. |
| `cc overflow` / `L0C overflow` | Cube accumulator tile `BLOCK_M * BLOCK_N * 4B` reaches or exceeds `128 KB` (`32768` elements, e.g., `128 x 256`). | Cap `BLOCK_M * BLOCK_N <= 16384` (`128 x 128` or `64 x 256` -> `64 KB`). |
| `cannot align N axis for hivm.hir.bitcast memref<1x2x2x...>` | `tl.sort(X, dim=1)` called on `[BLOCK_ROWS, L]` where `BLOCK_ROWS` or `L` is not a power of 2. | Force `BLOCK_ROWS = 1 << (br.bit_length() - 1)` and `L = next_pow2(n)`. |
| `Both operands must be same dtype. Got fp32 and fp16` | `tl.load(..., other=0.0)` promoted a masked `fp16`/`bf16` tile to `fp32` before `tl.dot`. | Explicitly cast `.to(q.dtype)` after `tl.load` before feeding `tl.dot`. |
| `RuntimeError: grid should be less than 65536!` | 1D launch grid exceeds hardware `uint16` block limit (`65535`) on main or auxiliary kernel (`RoPE`/norm). | Clamp **every** kernel's grid to `grid = (min(NUM_CORES, total_tasks),)` with a persistent in-kernel stride loop. |
| `ACL stream synchronize failed: 507014` (AICore timeout / hang) | 1. Cube `tl.dot` called with `M = 1`.<br>2. Non-affine store (`rows // GH`) degraded to elementwise scalar scatter.<br>3. 3D masked load in split-KV reduce.<br>4. `fp32`-derived MTE mask on Cube load/store. | 1. Route `M = 1` to pure Vector `tl.sum` or group `GH >= 8`.<br>2. Fix `BLOCK_QO = 1` for provably affine stores.<br>3. Replace 3D masked load with `for s in range(num_splits):`.<br>4. Keep MTE load/store masks in `int32`. **Important**: A `507014` hang leaves the NPU device wedged at `99%` AICore utilization—switch to a clean device (`ASCEND_RT_VISIBLE_DEVICES`) before re-benchmarking! |
| `ACL error 507034` (Vector Core timeout) | Single program executes an excessive serial loop (`kv_len // BLOCK_KV` or `N * E` scalar atomic loop) with oversized `BLOCK`. | Vectorize inner `E` dimension (`tl.atomic_add` on vector `arange`), enable split-KV, or shrink `BLOCK` to match `next_pow2(dim)`. |
| `ACL error 507035` (Memory / illegal instruction) | Out-of-bounds gather index in `tl.gather(row, idx, 0)`, runtime `if dim == X` branch, or `tl.gather` on unsupported 16-bit dtype. | Clamp gather indices `safe_idx = tl.where(mask, idx, 0)`, specialize `DIM: tl.constexpr`, and promote `slab` to `tl.float32` before `tl.gather` if needed. |

---

## 2. Shape-Driven Empirical Performance Runbook (12 Golden Cases on 40-Core Ascend NPU)

| Workload Pattern | Representative Shape | Optimal Tiling & Grid Configuration | Measured Latency & Key Mechanism |
|---|---|---|---|
| **1. Tiny Reduction (`reduction_amax_small`)** | `numel < 1000` (1D/2D) | **`grid = (1,)`**, single-block reduction | **`2.16 us`** (vs `3.51 us` multi-core). Multi-core launch/sync overhead exceeds compute for `< 1000` elements. |
| **2. Medium 1D Reduction (`reduction_amin_small`)** | `(65536,)` 1D | **`grid = (8,)`** (moderate parallelism `< num_cores`) + atomic/stage merge | **`2.21 us`**. Balances per-core chunk size against cross-core scheduling overhead. |
| **3. Massive 1D Reduction (`reduction_amin_large`)** | `(4194304,)` (`4M` 1D) | **`grid = (32,)`**, 2-level sub-tiling (`BLOCK_N`, `SUB_BLOCK_N`) filling UB with zero tail block + loop-carried vector `tl.minimum(acc, x)` | **`9.61 us`**. Defers `tl.min(acc, axis=0)` until after the chunk loop. |
| **4. Medium-M, Large-N 2D (`reduction_amax_medium`)** | `(1000, 10000)` (`10M`) | **`grid = (40,)` (`== AI Core count`)**, `BLOCK_M = 25`, loop-carried vector `acc = tl.maximum(acc, x)` | **`25.73 us`**. Exact core-count grid + post-loop `tl.max(acc, 1)` eliminates inner-loop horizontal reductions. |
| **5. Medium-M, Huge-N 2D (`reduction_amin_medium`)** | `(64, 524288)` | Maximize reduction tile **`BLOCK_SIZE_N = 16384`** within UB capacity | Minimizes reduction loop trip count and MTE2 pipeline flushes. |
| **6. Large Row-Mean with Sub-Tiling (`reduction_mean_large`)** | `(1000, 8192)` last-axis mean | **`grid = (40,)`**, `BLOCK_M = 25, SUB_BLOCK_M = 25` (**no tail sub-block**), `BLOCK_N = 512` | **`16.00 us`**. Ensuring `BLOCK_M % SUB_BLOCK_M == 0` avoids tail-block predication penalties. |
| **7. First-Axis Mean (`reduction_mean_medium`)** | `(1024, 1024)` axis-0 mean | **`grid = (32,)`** (`<= num_cores`, zero tail blocks) + 2D loop-carried vector accumulation | **`9.98 us`**. Coalesces column reads across rows before final `axis=0` sum. |
| **8. First-Axis Product (`reduction_prod_small`)** | `~100K` elements, axis-0 `prod` | **`grid = (16,)`** (`< num_cores`), custom `@triton.jit def _mul(a, b): return a * b` with `tl.reduce(acc, axis=0, combine_fn=_mul)` | **`2.15 us`**. Triton has no built-in `tl.prod` in older releases; use `tl.reduce` with `_mul` and loop-carried product accumulator `acc *= x`. |
| **9. Fused Elementwise + Sum (`reduction_sum_fused`)** | `(1000, 8192) + (8192,)` | **`grid = (40,)`**, `BLOCK_M = 25, SUB_BLOCK_M = 25, BLOCK_N = 256` (zero tail block) | **`47.58 us`** (vs `91.69 us` at `grid=20` and `79.00 us` at non-multiple `grid>40`). |
| **10. Huge-M, Medium-N Sum (`reduction_sum_large`)** | `(65536, 1024)` | Prioritize filling UB along the reduction axis (**`BLOCK_SIZE_N = 1024`**) | **`685.65 us`**. Single-pass along $N$ avoids reloading or re-reducing rows. |
| **11. 2D Broadcast Division (`elemwise_broadcast_2d`)** | `(M_small, N_large)` broadcast | Do **not** split the small broadcast axis; preload once outside the loop, fix `NUM_BLOCKS = 40`, and sub-tile `SUB_M` inside the kernel | Maximizes broadcast operand reuse in UB across `SUB_M` rows. |
| **12. 3D Weighted SwiGLU Bwd (`reduction_weighted_swiglu`)** | `(16, 1024, 2048) x 3` | **Flatten leading dims `(B, M) -> BM = 16384`** on host, `BLOCK_BM = 8, SUB_BLOCK_BM = 8, BLOCK_N = 512` (`grid = (2048,)`) | **`1091.26 us`**. 3D -> 2D reshape simplifies indexing and maximizes `BLOCK_N = 512` reduction width in UB. |
