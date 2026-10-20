---
category: sop
conditions:
  compiler: bishengir
  kernel_language:
  - triton
  - bishengir
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/bisheng-ir/pipeline-and-arch
evidence_strength: software_contract
id: unit-bisheng-ir-bufferization-memory-planning-and-multibuffer-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-bisheng-ir-compilation-pipeline-and-pass-lowering-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/memory_management/overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/memory_management/bufferization.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/memory_management/memory_planning.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/memory_management/multi_buffer.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/apis/ir/memory_management/extra_buffer.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/ir/ir_analysis_workflow.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/apis/ir/memory_management/overview.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: 'BiShengIR memory subsystem SOP: bufferization, linear-scan PlanMemory, multibuffer
  ping-pong, extra_buffer and last_pass.mlir diagnosis'
triggers:
  error_signatures:
  - ub overflow, requires N bits while 1572864 bits available
  - ub overflow, requires N bits while 2031616 bits available
  - hivm-enable-multi-buffer doubles live UB footprint and triggers ub overflow
  - extra_buffer allocation for vreduce, vtranspose or vsort exceeds remaining UB
    capacity
  - high aiv_scalar_ratio in msprof is misclassified as scalar-bound when MTE2 overlaps
    scalar execution in last_pass.mlir
  keywords:
  - hivm-plan-memory
  - pointer_cast
  - hivm-mark-multi-buffer
  - hivm-enable-multi-buffer
  - hivm-alloc-extra-buffer
  - memrefext.alloc_workspace
  - tightly_coupled_buffer
  - last_pass.mlir
  - bufferization
  - ub overflow
---
# BiShengIR memory subsystem SOP: bufferization, linear-scan PlanMemory, multibuffer ping-pong, extra_buffer and last_pass.mlir diagnosis

## 1. Trace the Eight-Stage Tensor-to-Physical-Offset Transformation Chain

When diagnosing on-chip capacity overflows, buffer aliasing, or workspace allocation errors in BiShengIR, inspect the eight memory management stages in order:

1. **Bufferization (`hivm-clone-tensor-empty`, `CloneSCFIfYieldOperand`, One-Shot Bufferize)**: Eliminates SSA tensor value semantics, converting `tensor.empty` and DPS `outs(...)` into logical `memref.alloc()` buffers. Yield operands inside `scf.if`/`scf.for`/`scf.while` are cloned when their lifetimes conflict with loop-carried aliases.
2. **Memory Scope Inference (`hivm-infer-mem-scope`)**: Attaches `#hivm.address_space<ub|cbuf|ca|cb|cc|gm>` to every `memref.alloc` based on the enclosing function's core type and consuming operations.
3. **Stride & Size Alignment (`hivm-align-alloc-size`, `hivm-mark-stride-align`, `hivm-enable-stride-align`, `hivm-lift-lowest-stride`)**: Pads innermost contiguous dimensions and allocation sizes to hardware boundaries (32 B in UB, 512 B in L1 `cbuf` and Cube descriptors).
4. **Extra Scratch Buffer Allocation (`hivm-alloc-extra-buffer`, `hivm-outline-alloc-in-VF`)**: Queries `ExtraBufferOpInterface` on complex vector ops (`hir.vreduce`, `hir.vtranspose`, `hir.vsort`, `hir.vcumsum`, `hir.vgather`, `hir.vpad`) and allocates explicit `$temp_buffer` memrefs in UB.
5. **Multi-Buffer Marking & Expansion (`hivm-mark-multi-buffer` $\to$ `hivm-enable-multi-buffer`)**: Identifies loop-carried DMA buffers (`annotation.mark` with `multi_buffer` or `--enable-auto-multi-buffer=True`) and expands single allocations into $K$-way (typically 2-way ping-pong) physical buffers indexed by `loop_iter % 2`.
6. **Tightly-Coupled & Global Workspace Buffers (`hivm-insert-cv-tight-coupled-buffer`, `insert-workspace-for-mix-cv`, `hivm-bind-workspace-arg`)**:
   - On Ascend 910B/910_93, mixed Cube-Vector kernels allocate GM workspace (`memref_ext.alloc_workspace`) with host-inferred size (`hivm-insert-infer-workspace-size-func`).
   - On Ascend 950, Cube-to-Vector handoff can use tightly-coupled on-chip buffers (`L0C -> UB` via `fixpipe` or `UB -> L1` via `copy`) without a GM round-trip.
7. **Linear-Scan Memory Planning (`hivm-plan-memory`)**: Replaces `memref.alloc()` with `hivm.hir.pointer_cast(byte_offset)` in each address space (`local-mem-plan` for UB/L1/L0; `global-workspace-plan` for GM workspace).
8. **Backend Lowering (`lower-memref-ext`, `StripMemRefAddressSpace`)**: Lowers planned offsets to hardware base-register intrinsics.

## 2. Diagnose `PlanMemory` In-Place Reuse and `ub overflow` Bit Budgets

`hivm-plan-memory` reuses physical offsets across buffers whose live intervals do not overlap:

- Disjoint sequential buffers receive identical offsets (e.g., two sequential `pointer_cast(0)` allocations).
- **Loop-invariant vs loop-local lifetime rule**: A buffer allocated outside a `scf.for` loop and read/written across iterations remains live for the entire loop and **cannot** share an offset with any buffer allocated inside the loop body.
- **Hidden UB consumers**: Total UB demand equals:
  $$\text{UB}_{\text{total}} = \sum \text{Live Tile Buffers} + \text{Stride-Align Padding} + \text{MultiBuffer Duplicates} + \text{ExtraBuffer Scratch}$$
- **Bit-unit error reporting**: `ub overflow, requires R bits while A bits available!` reports sizes in **bits**:
  - Ascend 910B / 910_93: $A = 1,572,864\text{ bits} = 196,608\text{ B} = 192\text{ KiB}$.
  - Ascend 950: $A = 2,031,616\text{ bits} = 253,952\text{ B} = 248\text{ KiB}$ (256 KiB physical UB minus 8 KiB reserved by compiler; further reduced by 40 KiB to 208 KiB when any SIMT VF block is active).

### Repair Order for `ub overflow`

1. Inspect `kernel.ttadapter.mlir` / pre-`PlanMemory` IR for wide promoted tensors (`bf16`/`fp16` promoted to `f32`, or `i64` index tensors from `tl.arange` broadcasting).
2. Check whether `hivm-alloc-extra-buffer` inserted a large `$temp_buffer` for `vreduce` or `vtranspose`; shrink tile dimensions or split the reduction axis.
3. If the overflow is between $1\times$ and $2\times$ of the single-stage footprint, disable or scope down multi-buffering (`al.multibuffer(tensor, 1)` or `--enable-auto-multi-buffer=False` on the spilled buffer) before halving `BLOCK_M`/`BLOCK_N`.

## 3. Extract and Audit `last_pass.mlir` Before Interpreting `msprof` Ratios

Follow the IR-first analysis workflow (`ir_analysis_workflow`):

```bash
export TRITON_DISABLE_CACHE=1
export TRITON_CACHE_DIR="$PWD/.triton_ir_dump"
export MLIR_ENABLE_DUMP=1
python3 run_kernel.py 2> full_ir_dump.log
# Extract the final HIVM stage after sync injection (last_pass.mlir):
bishengir-compile "$TRITON_CACHE_DIR"/*/kernel.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync \
  2> last_pass.mlir
```

Audit four structural checkpoints in `last_pass.mlir` and cross-verify with `msprof`:

1. **Scalar Degradation (`scf.for` + `arith.*` inside vector regions)**:
   - Caveat: High `aiv_scalar_ratio` (`> 20%`) in `msprof` is **not** a bottleneck when `last_pass.mlir` shows scalar address/loop setup overlapping long `hir.load` (`PIPE_MTE2`) waits.
   - Only when `last_pass.mlir` shows `HIVMLowerToLoopsPass` expanding vector ops into element-by-element `scf.for` loops (`arith.cmpi`, `arith.addi` on `i64` or `i32` inequality) on the critical path is scalar lowering the true bottleneck.
2. **Memory Access Alignment & Discrete Fallback**:
   - Check whether `hir.load`/`hir.store` became `scf.for` scalar loads via `TritonToUnstructured` or carries unaligned `StridedLayoutAttr` padding copies.
3. **Multi-Buffer & Pipeline Overlap**:
   - Verify that `scf.for` K-loops alternate between two `hivm.hir.pointer_cast` offsets (`event_id = EVENT_ID0` and `EVENT_ID1`) rather than serializing `set_flag`/`wait_flag` on a single `EVENT_ID0`.
4. **Barrier Density**:
   - Count `hivm.hir.pipe_barrier <PIPE_V>` and `pipe_barrier <PIPE_ALL>`. Independent vector chains separated by `pipe_barrier` indicate missed `GraphSyncSolver` optimization (`--enable-hivm-graph-sync-solver=true`).
