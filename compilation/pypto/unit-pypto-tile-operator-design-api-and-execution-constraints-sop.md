---
category: sop
conditions:
  kernel_language: pypto
  operator_family:
  - attention
  - matmul
  - reduction
  - elementwise
  - fused_layer
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/pypto
evidence_strength: software_contract
id: unit-pypto-tile-operator-design-api-and-execution-constraints-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-develop/references/execution-constraints.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-design/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/pypto-op-develop/SKILL.md
  source_locator: cann/cannbot-skills@f305e200aa4b1e1fed73cd21ff1f083b8df25a1e:ops/pypto-op-develop/references/execution-constraints.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: PyPTO Tile-Based Operator Design, API Semantics & Dynamic-Shape Execution Constraints
  SOP
triggers:
  error_signatures:
  - 'RuntimeError: Input tensor is not a valid torch tensor type when calling pypto.from_torch'
  - operand1 dim[0] = -1, must be > 0 when passing multi-DYN tensor to pypto.matmul
  - 'View(): incompatible function arguments or Cannot convert symbols to int on SymbolicScalar
    shape'
  - tile shape not set or silent NaN when reshape(inplace=True) is applied to output
    parameter
  keywords:
  - pypto
  - pypto.frontend.jit
  - set_vec_tile_shapes
  - set_cube_tile_shapes
  - SymbolicScalar
  - pypto.view
  - pypto.assemble
  - valid_shape
  - submit_before_loop
---
# PyPTO Tile-Based Operator Design, API Semantics & Dynamic-Shape Execution Constraints SOP

## 1. `@pypto.frontend.jit` Entry Contract & Parameter Rules

PyPTO compiles declarative tile graphs into multi-core Ascend AIC/AIV subgraphs:
1. **Standard JIT Decorator & Signature Order**:
   - Every kernel must be decorated with `@pypto.frontend.jit`. Tensor parameters must appear **before** non-tensor scalar parameters, and every tensor parameter must carry an explicit `pypto.Tensor([dim0, ...], pypto.DT_XXX)` annotation (empty `pypto.Tensor()` or `pypto.Tensor([], dtype)` is forbidden).
   - **Never call `pypto.from_torch(x)` before invoking a `@pypto.frontend.jit` kernel**: the C++ launcher (`LaunchKernelTorch` -> `TorchTensorConverter::ParseTensorData`) requires raw `torch.Tensor` inputs (`pypto.Tensor.data_ptr` is an `int` rather than a callable method, raising `RuntimeError: Input tensor is not a valid torch tensor type`).
2. **In-Place Output Writeback (No Return Value)**:
   - `@pypto.frontend.jit` kernels do not support `return`. Results must be written into output tensor parameters via `out[:] = ...`, `out.move(...)`, or `pypto.assemble(tile, offsets, out)`. Rebinding `out = result` only mutates a local Python reference and leaves the caller's output buffer unchanged.
3. **Uninitialized Local Tensors**:
   - `pypto.tensor(shape, dtype, name)` allocates uninitialized memory containing garbage values. Either initialize via `pypto.zeros`/`pypto.ones`/`pypto.full` (after calling `pypto.set_vec_tile_shapes`) or guard the first loop iteration with `if pypto.is_loop_begin(idx): acc[:] = tile` before accumulating `else: acc[:] = acc + tile`.

---

## 2. TileShape Configuration & Per-Operation Hardware Constraints

### 2.1 Mandatory `set_vec_tile_shapes` and `set_cube_tile_shapes` Rules

- **Vector TileShape (`pypto.set_vec_tile_shapes(*dims)`)**:
  - Must be set before any Vector operation (`zeros`, `ones`, `add`, `mul`, `cast`, `sum`, `amax`, `reshape`, `assemble`), with `1..4` positive dimensions matching the rank of the target tensor.
  - **Last-Dimension Alignment**: Must satisfy `32B` alignment (`16` elements for FP16/BF16, `8` elements for FP32). Prefer covering the entire last dimension; if splitting a large last axis, split at `512B`-aligned boundaries. Never split along a reduction axis.
  - **Reshape / Reduction Reset Invariant**: After `pypto.sum(..., keepdim=False)` or `pypto.reshape(x, new_shape)`, you **must** immediately call `pypto.set_vec_tile_shapes(...)` matching the new rank/shape before subsequent operations or `pypto.assemble`.
- **Cube TileShape (`pypto.set_cube_tile_shapes(m_tile, k_tile, n_tile)`)**:
  - Mandatory prerequisite before every `pypto.matmul(a, b, out_dtype=...)` call (for 3D/4D matmul, both `set_cube_tile_shapes` and `set_vec_tile_shapes` must be set).
  - Format: `m_tile = [mL0, mL1]`, `k_tile = [kL0, kL1]` (or `[kL0, kAL1, kBL1]` for asymmetric L1 residency in `M=1` decode), `n_tile = [nL0, nL1]`.
  - Invariant: `L0 <= L1`, `L1 % L0 == 0`, `L1 <= axis_length`, and `16`-element alignment for FP16/BF16. Recommended baseline: `[128, 128], [64, 256], [256, 256]`.

### 2.2 Operation-Specific Dtype, Broadcast & Layout Restrictions

| Category | Operations | Key Constraints & Required Workarounds |
| :--- | :--- | :--- |
| **Elementwise Arithmetic** | `add`, `sub`, `mul`, `div`, `neg`, `abs` | Rank `2..4`D; **single-axis broadcast only** (multi-axis broadcast must be decomposed into sequential steps); identical input dtypes; use `pypto.Element(dtype, val)` for typed scalars. |
| **Transcendental / Activations** | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `sigmoid`, `relu` | `sin`, `cos`, `sigmoid` support **only `DT_FP32`** (cast FP16/BF16 to `DT_FP32` first). `rsqrt` requires positive epsilon guard (`x <= 0` yields `Inf`/`NaN`). |
| **Comparison & Selection** | `eq`, `gt`, `maximum`, `minimum`, `where`, `clip` | Use `pypto.maximum`/`pypto.minimum` for tensors and `sym.max(other)`/`sym.min(other)` for `SymbolicScalar` (never Python built-in `min`/`max` inside kernel). `where` requires `DT_BOOL` condition and FP16/BF16/FP32 branches. |
| **Reductions & Sorting** | `sum`, `amax`, `amin`, `topk` | `sum` and `topk` support **only `DT_FP32`**. `amax`/`amin` require TileShape `<= 64KB`, last axis `32B`-aligned, and second-last axis `<= 255`. `topk` operates only on the last axis with `k <= TileShape[-1]` and `< 22KB`. |
| **Views, Reshape & Assemble** | `view`, `reshape`, `transpose`, `concat`, `assemble`, `pad` | **No Cyclic Read-Write**: Reading a tensor via `view` and writing back to the **same** tensor via `assemble` in the same graph creates a cycle error. `reshape(..., inplace=True)` is allowed **only** on original function input parameters or loop inputs (never on function output tensors, which silently produces `NaN`/zeros). |

---

## 3. Dynamic-Shape Execution Pattern: "2D Reshape + Nested `pypto.loop` + Concrete Tile"

### 3.1 Why Direct Multi-Dynamic-Axis Operations Fail

1. **`pypto.matmul` Requires Compile-Time Concrete Tile Shapes**: Passing a 4D tensor with `pypto.DYN` axes (`[pypto.DYN, N, pypto.DYN, D]`) directly to `pypto.matmul` fails with `operand1 dim[0] = -1, must be > 0` because Cube tiling requires concrete integer dimensions at graph construction time.
2. **`pypto.view` Shape Parameter Must Be Concrete Python `int`**:
   - In `pypto.view(tensor, shape, offsets, valid_shape=...)`, `shape` accepts **only Python `int` constants** (passing a `SymbolicScalar` raises `View(): incompatible function arguments`, and Python slicing `q[b_off:b_off+1]` raises `Cannot convert symbols to int`).
   - Dynamic `SymbolicScalar` values are permitted **only** in `offsets`, `valid_shape`, and `pypto.loop(trip_count)` bounds!

### 3.2 Canonical "2D Wrapper Reshape + Nested Symbolic Loop + Concrete `view`/`assemble`" Pattern

When an operator has 2 or more dynamic axes (e.g., `B` and `S` in `[B, N, S, D]`):
1. **Wrapper Layer**: Reshape 4D tensors `[B, N, S, D]` to 2D `[B * N * S, D]` in Python before calling the `@pypto.frontend.jit` kernel, and pass `B`, `S` as symbolic/scalar parameters or derive them from shape.
2. **Kernel Layer**: Annotate the 2D tensor as `pypto.Tensor([pypto.DYN, D], dtype)` and iterate over dynamic axes using nested `pypto.loop` calls (with `unroll_list=[1]` during initial development to prevent compile-path explosion):
   ```python
   @pypto.frontend.jit
   def attn_stage_kernel(
       q_2d: pypto.Tensor([pypto.DYN, D], pypto.DT_BF16),
       out_2d: pypto.Tensor([pypto.DYN, D], pypto.DT_BF16),
       b_size: int,
       s_size: int,
   ):
       s_tiles = (s_size + S_TILE - 1) // S_TILE
       for b_idx in pypto.loop(b_size, name="B_LOOP", idx_name="b_idx"):
           for s_idx in pypto.loop(s_tiles, name="S_LOOP", idx_name="s_idx", unroll_list=[1]):
               row_off = (b_idx * s_size + s_idx * S_TILE) * N_HEADS
               actual_rows = ((s_size - s_idx * S_TILE).min(S_TILE)) * N_HEADS
               # shape=[S_TILE * N_HEADS, D] is 100% concrete Python int!
               q_tile = pypto.view(
                   q_2d,
                   [S_TILE * N_HEADS, D],
                   [row_off, 0],
                   valid_shape=[actual_rows, D],
               )
               pypto.set_vec_tile_shapes(S_TILE * N_HEADS, D)
               res_tile = compute_on_concrete_tile(q_tile)
               pypto.assemble(res_tile, [row_off, 0], out_2d)
   ```
3. **Loop Dependency & Two-Pass Backward Rule**:
   - `pypto.loop` dispatches iterations across multiple cores in parallel by default. Whenever loop iterations carry cross-iteration data dependencies, set `submit_before_loop=True`.
   - For multi-output backward kernels accumulating along orthogonal axes (e.g., FlashAttention `dQ` accumulated over `S_kv` vs. `dK`/`dV` accumulated over `S_q`), split into **two separate loop passes** (Pass 1 outer `S_q` / inner `S_kv` for `dQ`; Pass 2 outer `S_kv` / inner `S_q` for `dK, dV`) to avoid cross-loop write races without serializing the outer loop.
