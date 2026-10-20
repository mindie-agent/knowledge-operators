---
category: axiom
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/constraints
evidence_strength: software_contract
id: unit-triton-ascend-hardware-and-compiler-constraints-contract
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-triton-ascend-constexpr-and-specialization-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/global_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/blocksize_and_boundary_policy.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/api_support_matrix.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/data_type_matrix.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/guides/constraints/global_constraints.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Triton-Ascend Hardware, Memory, BlockSize, API, and Data Type Constraints Contract
triggers:
  error_signatures:
  - 1. Universal Global Constraints (`GC-1` to `GC_15`)
  - 2. Unified Buffer (UB) Capacity, BlockSize Policy & Boundary Handling
  - 2.1 UB Hardware Capacity and Safe Working Budget
  - 2.2 BlockSize Selection & Tail Masking Contract
  keywords:
  - unit-triton-ascend-hardware-and-compiler-constraints-contract
---
# Triton-Ascend Hardware, Memory, BlockSize, API, and Data Type Constraints Contract

## 1. Universal Global Constraints (`GC-1` to `GC_15`)

Every Triton-Ascend operator must satisfy the cross-category hardware and compiler contract (`GC-1` through `GC_15`):

| Rule ID | Constraint | Architectural Root Cause & Enforcement |
|---|---|---|
| **GC-1** | Dynamic core count | Query `num_vectorcore` (`40`/`48` on 910B/A3) and `num_aicore` (`20`/`24` Cube) via `get_device_properties(dev)` cached at module load. Never hardcode `8`, `24`, or `48`. |
| **GC-2** | 1D Persistent Grid | Clamp 1D grid to `min(total_tiles, num_cores)` (`<= 65535`) and loop inside kernel (`for idx in range(pid, total_tiles, num_progs):`). Avoid raw 2D/3D grids exceeding physical cores. |
| **GC-3** | Power-of-2 `tl.constexpr` tiles | `BLOCK_SIZE`, `BLOCK_M`, `BLOCK_N`, `BLOCK_K` must be powers of 2 passed as `tl.constexpr`. Runtime lengths force dynamic-shape loads that degrade to scalar loops. |
| **GC-4** | `int32` lane offsets + `int64` base | Cast `tl.program_id` and `tl.arange` to `tl.int32`; promote only row base pointers (`row_idx.to(tl.int64) * stride`) when `numel > 2^31 - 1`. Never run `%` on `int64` vectors; use `x - (x // M) * M`. |
| **GC-5** | Contiguous & host flatten | Ensure `.contiguous()` inputs (unless strided layout is explicitly handled in-kernel) and flatten independent batch prefixes on the host. |
| **GC-6** | Pure Triton forward path | `forward()` may only compute shapes, allocate outputs, and launch Triton kernels. Never compute intermediate math via `torch.*` on host/device. |
| **GC-7** | No `continue` / `break` / loop `return` | BiSheng IR rejects irregular loop exits (`continue`, `break`, early `return`). Use forward `if` guards or `tl.where` predication. |
| **GC-8** | Kernel `tl.cdiv` vs Host `triton.cdiv` | Inside `@triton.jit`, call `tl.cdiv(a, b)`; on Python host, call `triton.cdiv(a, b)` or `(a + b - 1) // b`. |
| **GC-9** | Dummy tensor for optional ptrs | Never pass `None` for dereferenced optional pointers without `tl.constexpr` guards; pass a small dummy tensor or split kernels. |
| **GC_10** | Promote `fp16`/`bf16` to `fp32` | Cast `fp16`/`bf16` to `tl.float32` before `tl.sum`, variance, `exp`/`log`, or multi-step nonlinear math, then cast back before `tl.store`. |
| **GC_11** | Strip GPU launch kwargs | Remove `num_warps`, `num_stages`, `cache_modifier`, and `evict_policy` (ignored or rejected on Ascend). |
| **GC_12** | Vector `int` compare degradation | Vector `int32` comparisons inside complex expressions can degrade to scalar loops in `hivm`; cast coordinates to `tl.float32` before comparison when scalar stalls occur. |
| **GC_13** | Avoid non-pow2 `tl.reshape` | `tl.reshape` requires `tl.constexpr` power-of-2 shapes; prefer direct 1D/2D pointer arithmetic. |
| **GC_14** | No stride-swapped fake transpose | Do not transpose tiles by swapping strides in pointer arithmetic before `tl.dot`; use `tl.trans` or `tl.make_block_ptr(..., order=(...))`. |
| **GC_15** | Per-operator compiler hints | Test `multibuffer=True, unit_flag=True` per operator: helpful on multi-input fused loops, but can regress small single-pass or reduction kernels. |

---

## 2. Unified Buffer (UB) Capacity, BlockSize Policy & Boundary Handling

### 2.1 UB Hardware Capacity and Safe Working Budget
- **Ascend 910B / 910_93 (A2/A3)**: Physical UB is `192 KB` (`1,572,864 bits`) per Vector Core. After reserving `~12–20 KB` for `bishengir-compile` temporary scratch and alignment padding, the **safe static tensor live-range budget is `<= 140–160 KB`** (or `<= 80 KB` when `multibuffer=True` double-buffers UB).
- **Ascend 910_95 (A5)**: Physical UB is `256 KB` (`2,097,152 bits`). Note that on `910_95`, `multibuffer` defaults to `False` and must be explicitly set to `True` when software pipelining is desired.
- **Peak Live-Range Estimation**:
  $$\text{UB}_{\text{bytes}} = \sum_{\text{live tensors}} (\text{BLOCK\_M} \times \text{BLOCK\_N} \times \text{sizeof(dtype)}) \times \text{multibuffer\_factor}$$
  Every `fp16`/`bf16` tile promoted to `fp32` consumes $2\times$ additional UB (`4 bytes/elem`), and `tl.where` masks or intermediate reductions keep source and destination buffers simultaneously live.

### 2.2 BlockSize Selection & Tail Masking Contract
- **1D Elementwise / Vector Kernels**: Choose `BLOCK_SIZE` in `{512, 1024, 2048, 4096, 8192}` (or autotune up to `16384`/`32768` for `fp16`/`bf16`). For `fp32` tensors with multiple live intermediates, cap `BLOCK_SIZE <= 4096` to avoid `ub overflow`.
- **2D Row-Wise / Reduction Kernels**:
  - Single-pass row fits in UB (`N <= 4096` for `fp32`): set `BLOCK_N = triton.next_power_of_2(N)` and `BLOCK_M = max(1, min(32, 4096 // BLOCK_N))`.
  - Large row (`N > 4096`): set `BLOCK_M = 1`, `BLOCK_N = 2048` or `4096`, and loop over column chunks (`for col_start in range(0, N, BLOCK_N)`).
- **Cube (`tl.dot`) Kernels**:
  - `BLOCK_M`, `BLOCK_N`, `BLOCK_K` must be multiples of `16` (`fp16`/`bf16`/`fp32`) or `32` (`int8`), typically chosen from `{16, 32, 64, 128}`.
  - **Cube `M=1` Forbidden Zone**: Never invoke `tl.dot` with effective row count `M = 1` (causes `ACL stream synchronize failed 507014` or severe scalar degradation); either group heads so `M >= 8` (padded to `16`) or dispatch a pure Vector dot-product (`tl.sum(a * b, axis=-1)`).
- **Boundary Handling**:
  - Always guard out-of-bounds lanes with `mask = offs < limit` and explicit neutral `other` (`0.0` for sum/dot, `-1.0e9` or `-3.0e38` for max/softmax—never `float("-inf")` on full-masked rows where `exp(-inf - (-inf))` yields `NaN`).
  - With `tl.make_block_ptr`, pass `boundary_check=(0, 1)` and `padding_option="zero"`, and still apply score-level masking before `softmax` so zero-padded keys do not contribute $e^0 = 1$ weight.

---

## 3. Triton-Ascend API Support & Data Type Matrix

| Category | Supported Standard `tl.*` APIs | Ascend Restrictions, Prohibitions & Extensions (`al.*` / `libdevice`) |
|---|---|---|
| **Memory & Pointers** | `tl.load`, `tl.store`, `tl.make_block_ptr`, `tl.advance`, `tl.max_contiguous`, `tl.multiple_of` | `care_padding=False` is an Ascend extension on `tl.load`. `tl.make_tensor_descriptor` works only on `910_95`. Vector-base cross-page gathers (`ptr + vec_idx[:, None] * stride`) evict `tl.dot` from Cube. |
| **Linear Algebra** | `tl.dot(a, b, acc, input_precision=...)`, `tl.trans` | Supports `fp16`, `bf16`, `fp32` (`tf32` mode), and `int8 -> int32`. Does **not** support `fp64` or `uint` types. 3D batched `tl.dot` requires `BLOCK_B` outer loop or supported 3D shape. |
| **Math & Activations** | `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`, `tl.cos`, `tl.sin`, `tl.sqrt`, `tl.rsqrt`, `tl.abs`, `tl.maximum`, `tl.minimum`, `tl.where`, `tl.Clamp` (float only) | **Warning**: `tl.sigmoid` has systematic ULP drift on some quantization/norm paths—use `x / (1.0 + tl.exp(-x))` when golden uses `1/(1+exp(-x))`. `tl.round` is round-half-to-even; for PyTorch quant round-half-away-from-zero, use `tl.where(x >= 0, x + 0.5, x - 0.5)` + `al.cast(..., overflow_mode="saturate")`. Use `tl.math.tanh` for softcap (never manual `exp` ratio). |
| **Reductions & Scans** | `tl.reduce`, `tl.sum`, `tl.max`, `tl.min`, `tl.argmax`, `tl.argmin`, `tl.cumsum`, `tl.cumprod`, `tl.sort`, `tl.flip` | Online softmax reductions on NPU require `propagate_nan=True` in `tl.max` and `propagate_nan=tl.PropagateNan.ALL` in `tl.maximum`. `al.sort` / `tl.sort` operates only along the last dimension. |
| **Atomics** | `tl.atomic_add`, `tl.atomic_max`, `tl.atomic_min`, `tl.atomic_cas`, `tl.atomic_xchg`, `tl.atomic_and`, `tl.atomic_or`, `tl.atomic_xor` | Vector `tl.atomic_add(ptr + tl.arange(0, BLOCK_E), vals, mask=...)` is strongly preferred over scalar loops. For `fp16`/`bf16` gradient accumulation, allocate an `fp32` workspace buffer for `tl.atomic_add` and cast to output dtype in a final pass. |
| **Data Types** | `int1`, `int8`, `int16`, `int32`, `int64`, `fp16`, `bf16`, `fp32` | **`fp64` and unsigned integers (`uint8/16/32/64`) are unsupported on Vector/Cube arithmetic paths.** Bitwise `&`, `|`, `^`, `<<`, `>>` require integer operands (`int8`/`int16`/`int32`). |
