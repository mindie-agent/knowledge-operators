---
schema: mindie-graded-unit/1
id: unit-triton-ascend-hivm-lowering-grid-overflow-and-intrinsics-sop
revision: 1
tier: L2
category: sop
domain_path: compilation/triton
evidence_strength: software_contract
title: 'Triton-Ascend lowering SOP: UB budget, CoreDim ceiling, int1 storage and GPU-intrinsic replacement'
conditions:
  kernel_language: triton
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  source_type: public_software_contract_review
  source_locator: https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/debug_guide/debugging.md
  source_revision: c22984baa55e1506eadd7c72955d9bf669b1c959
  evidence_refs:
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/debug_guide/debugging.md
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api/Programming_Model/program_id.md
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api/Atomic_Ops/atomic_add.md
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api/Comparing_Ops/eq.md
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  - https://github.com/vllm-project/vllm-ascend/issues/3548
  supersedes: []
triggers:
  error_signatures:
  - ub overflow, requires N bits while M bits available
  - unsupported AST node type Break or Continue
  - unsupported type bool, uint8, uint64 or fp64
  - coreDim=... can't be greater than UINT16_MAX
  - invalid CoreDim
  - failed to legalize operation 'tt.advance'
  - unsupported stride pattern for transpose
  - misaligned memory access
  - a CUDA-only import or inline PTX reaches the Triton-Ascend compiler
  keywords:
  - Triton-Ascend
  - BiShengIR
  - HIVM
  - ub overflow
  - UB
  - CoreDim
  - '65535'
  - grid
  - int1
  - bool
  - atomic_add
  - tl.dot
  - 32-byte alignment
  - 512-byte alignment
  - msprof
  - Ascend910B
  - Ascend910_93
  - Ascend950
---
# Triton-Ascend lowering SOP: UB budget, CoreDim ceiling, int1 storage and GPU-intrinsic replacement

Use this SOP when a Triton-Ascend kernel fails after the frontend accepts it: the failure is in BiShengIR/HIVM lowering, UB allocation, launch geometry or an unsupported GPU-specific primitive. The unit is a diagnostic and repair order, not a performance claim. The inherited alignment axiom covers the 32 B UB and 512 B L1 foundation for the 910B/910_93 family; 950 capacity and tail figures are read from the 950 target specification. Record the SOC, CANN/`triton-ascend` revision, kernel source hash, shape, dtype and launch grid before changing anything.

## 1. Diagnose BiShengIR/HIVM lowering and UB allocation failures

Reproduce with dumps enabled and the cache disabled. A cache hit can suppress the dump and make the failure look intermittent.

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
export TRITON_CACHE_DIR="$PWD/.triton_probe"
export MLIR_ENABLE_DUMP=1
python3 repro_kernel.py 2> mlir_dump.log
find "$TRITON_CACHE_DIR" -maxdepth 4 -type f | sort
```

Read the stage files in order:

```text
kernel.ttir.mlir        frontend/TTIR accepted the kernel
kernel.ttadapter.mlir    Ascend passes produced Linalg/UB buffers
kernel.npuir.mlir        BiShengIR/HIVM output; also written on failure
```

Interpret the first missing file. Missing `ttir` is a Python/Triton frontend error. `ttir` present and `ttadapter` missing is an unsupported operation or type. `ttadapter` present and `npuir` missing is a BiShengIR/HIVM pass failure; use `MLIR_ENABLE_DUMP=1` and the debug log's `cmd_list` to find the failing pass.

`ub overflow` values are in **bits**, not bytes: 910B/910_93 has 1572864 bits (192 KiB), and 950 has 2031616 bits (248 KiB, with 8 KiB reserved by the compiler). Convert the reported `requires` and `available` values by dividing by 8 before comparing with a tile budget.

The common repairs, in order:

- Reduce the live tile footprint: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, the number of simultaneously live intermediate tensors, or the epilogue scratch. A double-buffered path must fit the smaller half-budget; the sum of all concurrent buffers is what the planner charges.
- Remove large index tensors. A broadcast `cols[:, None] + rows[None, :]` materialised solely to compute a mask or address can dominate UB. Compute the mask from 1-D vectors or fold the index arithmetic into the pointer expression.
- Check dtype promotion. bf16 comparisons and reductions promote to fp32; int8 accumulation can widen; an fp16 division promotes to fp32. A tile that fits in input dtype may not fit after promotion.
- Disable or reduce multibuffering when the error appears only after an auto-multibuffer pass. Do not disable it permanently without profiling; record the change and the resulting occupancy.
- `unsupported AST node type: Break` / `Continue`: Triton-Ascend does not lower Python loop-control statements inside a kernel. Replace them with a bounded `range`, a positive `if` guard, or a `tl.where` selection. Do not rewrite a data-dependent early exit as an unbounded loop.
- `unsupported type: bool`, `uint8`, `uint64`, `fp64`: convert to a supported storage type (`int8`, `int32`, `int64`, `fp32`) on the host or at the kernel boundary. `tl.load`/`tl.store` support int8/int16/int32/int64/fp32, not the unsigned family or fp64, on the recorded backend.
- `failed to legalize operation 'tt.advance'`: rebuild the block pointer inside the loop instead of advancing a complex pointer across branches.
- `unsupported stride pattern for transpose`: express the transpose through the layout `order`; do not fake it by swapping strides.
- `misaligned memory access`: check the base address and the innermost tile byte width. Vector paths need 32 B; Cube/L1 descriptors need 512 B; a 950 CV fused tail needs 512 B. `tl.multiple_of` is a hint, not a repair.

For `tl.dot`, separate the Cube contract from the Vector contract. The Cube micro-tile is 16x16, so the tile should satisfy `M >= 16`, `N >= 16`, and a positive `K` that is a multiple of 16 for 16-bit inputs (a 32 B multiple for the chosen dtype). An M or N tile below 16 leaves most of the micro-tile idle. A pure Vector kernel has no 16x16 micro-tile; its contract is the 32 B UB tail. If the profile shows the kernel running as `mix_aic` while the source has no `tl.dot`, the compiler chose a mixed path for another reason; do not assume the dot contract applies. Confirm the executed core type in `msprof` before changing the tile.

**Verification.** Recompile every shape in the workload, not only the failing one. Inspect each `memref.alloc` in `kernel.ttadapter.mlir` against the UB budget and confirm that no large temporary survives across the loop. A compile pass with an uninitialised or silently clamped tile is not a repair. If the error is a frontend AST/type error, confirm the generated TTIR exists before moving to a later stage.

## 2. Prevent grid dimension overflow and 32-bit pointer offset overflow

Triton-Ascend maps the first launch dimension to the NPU CoreDim. The ceiling is `UINT16_MAX = 65535`, independent of the SOC generation. A grid proportional to tokens or blocks reaches it silently at a large batch or long sequence.

A launch-side clamp is only a bound; it is valid only when the kernel also covers the blocks that no longer fit in the first grid dimension. A clamp that drops blocks is a correctness bug:

```python
MAX_GRID_X = 65535
grid = (min(triton.cdiv(n_elements, BLOCK), MAX_GRID_X),)
# The kernel must loop over the logical blocks beyond grid_x; the loop below does that.
```

Prefer a core-bounded persistent grid, which covers every block and also avoids launching thousands of tiny blocks below the ceiling:

```python
@triton.jit
def kernel(ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK)
    for block_idx in range(pid, num_blocks, num_programs):
        offs = block_idx * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(ptr + offs, mask=mask, other=0.0)
        tl.store(ptr + offs, x, mask=mask)
```

Launch with `grid = (min(cdiv(n_elements, BLOCK), num_cores),)`, where `num_cores` comes from `driver.active.utils.get_device_properties(device_id)["num_vectorcore"]` for a vector kernel or `["num_aicore"]` for a kernel containing `tl.dot`. For a 2-D tile grid, flatten the logical block index and recover the coordinates with scalar division; do not let the first dimension grow with `M` or `N`.

`TRITON_ALL_BLOCKS_PARALLEL=1` can change how blocks are dispatched, but it does not prove that every block was covered. Treat it as an experiment, not as a substitute for the stride loop.

The second overflow is the pointer offset. `pid * stride` is computed in int32 unless widened. A tensor with more than `2**31 - 1` bytes (2 GiB) can wrap the linear offset even when each dimension is small. Widen the final byte offset, not every index:

```python
# Keep tile coordinates in int32; widen the address that can exceed 2 GiB.
base = (pid.to(tl.int64) * BLOCK_M).to(tl.int64) * stride_m
offs = base + tl.arange(0, BLOCK_N).to(tl.int64)
```

`tl.arange` is int32 on the recorded backend. Avoid int64 modulo: the support matrix limits int64 `%` to the `-2**24 .. 2**24` range. Use int32 scalar division for `pid_m`/`pid_n`, then widen the final linear offset. If the backend cannot legalise the int64 address path, split the tensor into sub-2 GiB chunks; do not accept wraparound.

**Verification.** Check `grid_x` in the generated launch metadata and the runtime error text. For the pointer path, print `numel * element_size`, the maximum linear offset, and the base pointer's `% 32` before the run. Exercise a shape above 2 GiB or a synthetic offset that crosses the int32 boundary. Confirm the kernel reads and writes the expected rows; a finite output from the wrong address is still a failure.

## 3. Handle int1/bool load and store limitations

Triton predicates are `int1`, but the Ascend memory path does not store a one-bit element. The frontend may lower a boolean load through int8 and tag it `was_bool_to_int8`; a boolean store is not universally supported and can fail with `unsupported type: bool`. Keep predicates in `int1` for `tl.where` and masks, but use `int8` at the GM boundary.

```python
# GM mask is int8; reconstruct the predicate in UB.
mask_i8 = tl.load(mask_ptr + offs, mask=offs < N, other=0).to(tl.int8)
active = mask_i8 != 0
y = tl.where(active, x, 0.0)
tl.store(out_ptr + offs, y, mask=offs < N)
```

When writing a mask:

```python
tl.store(mask_out_ptr + offs, active.to(tl.int8), mask=offs < N)
```

Prefer `int8`; the recorded support matrix rejects `uint8` for `tl.load`/`tl.store` even though the hardware description sometimes names it. Use `uint8` only when the installed version's support matrix explicitly accepts it. A boolean tensor loaded as int8 and then used directly in arithmetic is an integer 0/1, not a predicate; convert with `!= 0` before `tl.where`, `&`, or `|`.

An `int1` output that is genuinely a tensor result has a separate tail constraint: the vector data-movement contract requires the I1 output tail axis to be 16-aligned. If the mask width is not a multiple of 16, pad the store tile or write int8 bytes and reconstruct on the host.

**Verification.** Check the stored bytes are exactly 0 and 1, count the true entries against the reference, and exercise a mask whose tail is not 8/16-aligned. Inspect the TTIR for `was_bool_to_int8` and confirm the predicate is converted back before it feeds `tl.where`. A mask that loads correctly but is used as an integer multiplier produces plausible but wrong numbers on the false lanes.

## 4. Replace unsupported GPU-specific intrinsics and atomics

A GPU Triton kernel can contain `tl.extra.cuda` imports, inline PTX, CUDA libdevice symbols, cache modifiers, or warp-level primitives. The Ascend backend has no PTX path and no CUDA L1/L2 cache semantics. Replace these before debugging the kernel's math:

- `tl.extra.cuda` or CUDA libdevice: use `triton.language.extra.cann.libdevice` for the documented math functions, `tl.math` for the standard operations, or a pure-Triton expression. Do not assume a CUDA symbol exists under the same name.
- Inline PTX / `tl.inline_asm_elementwise`: remove it unless the installed backend explicitly documents an Ascend implementation. A frontend that parses the call is not proof that BiShengIR lowers it.
- `cache_modifier`, `eviction_policy`, `volatile`: these are ignored or rejected because the NPU has no corresponding cache level. Remove them rather than tuning them.
- Warp shuffles / `num_warps` / `num_stages`: the NPU has no CUDA warp. Use `tl.sum`/`tl.max` reductions, shared UB, or the Ascend pipe/sync primitives; remove the warp/stage configuration from the kernel and autotune config.
- `tl.atomic_add` with int64 or fp64: not supported. `sem` is limited to `"acq_rel"` and `scope` to `"gpu"`; `atomic_and/or/xor/xchg/cas` are not supported inside a loop. `atomic_add` does not support the multi-core add-and-save-intermediate pattern.

Replace fine-grained global atomics with a core-local accumulator and a deterministic two-stage reduction:

```python
# Stage 1: each program writes a private fp32 partial; no global RMW.
partial = tl.sum(tile.to(tl.float32), axis=0)
tl.store(partial_ptr + pid * BLOCK + tl.arange(0, BLOCK), partial)

# Stage 2: one program (or a fixed tree) reduces the partials in a fixed order.
```

For a histogram or scatter, build a per-core private histogram in UB, then merge the per-core histograms in a second kernel. A global `tl.atomic_add` over every bin is a serialised read-modify-write and can be orders of magnitude slower; the recorded histogram case fell from a usable baseline to a small fraction after introducing global atomics. For a contiguous output segment, a vector atomic is acceptable only when the dtype and memory-order constraints hold and the reduction is genuinely commutative; it is not a determinism guarantee.

**Verification.** Search the source for `tl.extra.cuda`, `inline_asm`, `num_warps`, `num_stages`, `cache_modifier`, and unsupported atomic dtypes. Confirm the compiled TTIR contains no unresolved external CUDA symbol. Run the same input at least three times and compare output bytes: an atomic-order-dependent result is non-deterministic even when each individual value is within tolerance. Count the global atomic operations per program before and after; a reduction that still issues one atomic per element has not been rewritten.

## 5. Inspect compiled HIVM/AscendC artifacts and profile AIV versus AIC

Use a dedicated cache directory and keep the compile artifacts with the measurement. The default cache is under the process home; an isolated directory prevents a previous kernel from masking the current one.

```bash
export TRITON_CACHE_DIR="$PWD/.triton_artifacts"
export TRITON_KERNEL_DUMP=1
export TRITON_DEBUG=1
export MLIR_ENABLE_DUMP=1
python3 run_case.py
```

If the pipeline fails after TTAdapter, reproduce it manually to isolate the pass:

```bash
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-auto-multi-buffer=True \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```

Replace `--target` and the multibuffer flags with the options documented for the actual SOC; a 910 command is not a 950 command. The purpose of the manual compile is to isolate the failing pass, not to define a portable build recipe.

Then profile the same case:

```bash
msprof --application="python3 run_case.py" \
  --output=./prof_case --ai-core=on --aic-metrics=PipeUtilization --l2=on
python3 - <<'PY'
import glob
import pandas as pd

path = sorted(glob.glob("prof_case/**/op_summary_*.csv", recursive=True))[-1]
df = pd.read_csv(path)
keep = [c for c in [
    "op_name", "op_type", "task_id",
    "aicore_time(us)", "aiv_time(us)",
    "aic_mac_time(us)", "aic_mte2_time(us)", "aic_fixpipe_time(us)",
    "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)",
    "aiv_scalar_time(us)", "aic_scalar_time(us)",
] if c in df]
print(path)
print(df[keep].to_string(index=False))
PY
```

Read `aicore_time` and `aiv_time` as separate core clocks. A pure-vector kernel can have `aicore_time = 0`; that does not mean the kernel is idle. Use `max(aicore_time, aiv_time)` when estimating core activity, and read the AIC and AIV MTE2 columns separately: they are different memory paths.

Use the pipe ratios as relative duty cycles, not as absolute latency. PMU collection can inflate kernel duration by roughly 2x to 16x; take the absolute timing from a separate run without PMU. A ratio above 1 indicates an inconsistent MIX-kernel counter, not a saturated pipe.

Typical readings:

- `aiv_mte2` high with low achieved bandwidth: the access pattern or tile shape is poor, not necessarily HBM-bound. Widen the contiguous segment and reduce short transactions.
- `aiv_mte3` high: the writeback dominates; fuse the epilogue or remove an intermediate GM tensor.
- `aiv_vec` high: the Vector ALU is the limit; reduce transcendental work, vectorise the tail, or remove a scalar loop.
- `aiv_scalar` high: address arithmetic, branches or `tl.where` scalarisation dominate. Check `cols < N` comparisons and cast the compared operand to fp32 where the backend would otherwise lower the comparison to scalar.
- `aic_mac` high: the Cube is compute-bound; tiling and dtype changes will not create free time.
- `aic_mte2` high with low `aic_mac`: the Cube is starved on input; change L1 reuse, layout or prefetch before changing the MAC tile.

**Verification.** Keep the `kernel.ttir.mlir`, `kernel.ttadapter.mlir`, `kernel.npuir.mlir`, the `bishengir-compile` command and the `msprof` output with the case record. Confirm the executed core type (`aic`, `aiv`, `mix`) matches the contract you used for the tile. If the artifact is missing, the cache was hit or the compile was skipped; rerun with `TRITON_DISABLE_CACHE=1`. If a ratio is above 1 or the pipeline columns are absent, record `pipeline signal unavailable` instead of inventing a bottleneck.

**Exit criteria.** The repair is complete only when: every workload shape compiles; no UB/L1 budget is exceeded; `grid_x` is below the CoreDim ceiling and every logical block is covered; bool/int1 values round-trip as int8; no CUDA-only intrinsic remains; the correctness gate passes on the boundary shapes; and the profiler artifact exists for the same source hash. A kernel that compiles but silently drops a tail block, wraps a 32-bit offset, or changes the reduction precision has not met the exit criteria.
