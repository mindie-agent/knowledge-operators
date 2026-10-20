---
schema: mindie-graded-unit/1
id: unit-triton-gpu-to-ascend-kernel-migration-and-verification-workflow
revision: 1
tier: L3
category: workflow
domain_path: compilation/triton
evidence_strength: reproduced_empirical
title: 'Triton GPU-to-Ascend migration workflow: launch audit, UB tiling, numerical parity and steady-state profiling'
conditions:
  kernel_language: triton
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-triton-ascend-hivm-lowering-grid-overflow-and-intrinsics-sop
provenance:
  source_type: reproduced_migration_case_and_runbook_review
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/main/knowledge/ops/triton/guides/migration/index.md
  source_revision: 7ea0efbeb79f2559f121fa8514a9fe5c82245807
  supporting_source_revisions:
  - c22984baa55e1506eadd7c72955d9bf669b1c959
  - d2a6661365ef65ca738cb66299449b2740c857eb
  evidence_refs:
  - docs/evidence/l4-triton-constexpr-runtime-scalar-ablation.json
  - https://gitcode.com/cann/cannbot-knowledge/blob/main/knowledge/ops/triton/guides/migration/index.md
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api/index.md
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/debug_guide/debugging.md
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api/Programming_Model/program_id.md
  - https://gitcode.com/Ascend/AscendNPU-IR
  - https://github.com/vllm-project/vllm-ascend/issues/3548
  supersedes: []
triggers:
  error_signatures:
  - a GPU Triton kernel still contains num_warps, num_stages or warp-shuffle logic
  - a 1D grid grows with batch times sequence length and overflows CoreDim
  - an unaligned pointer or row stride causes misaligned DMA on the NPU
  - the migrated kernel passes a small aligned shape but fails a prime tail or strided view
  - fp16 or bf16 accumulation drifts from the fp32 reference
  - thousands of tl.atomic_add calls replace a deterministic reduction
  - a speedup is claimed from kernel time without warmup, synchronization or a matched baseline
  keywords:
  - Triton migration
  - GPU to Ascend
  - Triton-Ascend
  - num_warps
  - num_stages
  - warp shuffle
  - persistent grid
  - UB tiling
  - 32-byte alignment
  - atomic_add
  - numerical parity
  - msprof
  - aiv_time
  - aicore_time
  - Ascend910B
  - Ascend910_93
  - Ascend950
---
# Triton GPU-to-Ascend migration workflow: launch audit, UB tiling, numerical parity and steady-state profiling

## Trigger and scope

Use this workflow when a Triton kernel that runs on a GPU must run on an Ascend 910B, 910_93 or 950 NPU. The trigger is a migration, not a native Ascend numerical bug: the source already has a GPU execution path, a reference result, and a workload shape set. The workflow covers launch-model audit, tile and reduction restructuring, numerical parity, and steady-state profiling.

Out of scope: changing the mathematical contract to make a shape fit, treating a successful compile as numerical parity, and claiming a model-level or serving-level speedup from one operator timing. A kernel whose reference is unknown, or whose output dtype is not declared, must be frozen before Stage 1.

## Facts to freeze before profiling

Record these facts before editing or timing. A missing fact stays `unknown`; do not infer it from a model name.

- SOC and die count, device count, CANN/`torch_npu`/`triton-ascend` revision, driver/firmware, and the compiler target.
- Kernel source hash, launch grid, `num_warps`, `num_stages`, autotune configs, and every `tl.constexpr` that changes with shape.
- Input shape, stride, storage offset, dtype and output dtype; whether the innermost dimension and row stride are 32 B aligned; whether a view is contiguous or strided.
- Reference implementation and its precision (CPU fp32, GPU fp64/fp32, or a native `torch_npu` op), tolerance, epsilon, and the exact output dtype.
- Graph/eager mode, warmup count, sample count, device clock, and the matched baseline hash.
- Whether the kernel uses `tl.dot`, pure Vector work, or a mix; whether it launches a token-proportional grid; whether it uses global atomics.

Record the device facts and tensor geometry on the actual NPU:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 - <<'PY'
import torch
import torch_npu
import triton.runtime.driver as driver

props = driver.active.utils.get_device_properties(torch.npu.current_device())
print("aicore", props.get("num_aicore"), "vectorcore", props.get("num_vectorcore"))
for name, t in {
    "x": torch.randn(4096, 1000, dtype=torch.bfloat16, device="npu"),
    "w": torch.randn(1000, dtype=torch.bfloat16, device="npu"),
}.items():
    print(name, t.dtype, tuple(t.shape), t.stride(),
          "base_mod32", t.data_ptr() % 32,
          "row_bytes", t.shape[-1] * t.element_size(),
          "row_bytes_mod32", (t.shape[-1] * t.element_size()) % 32)
PY
```

If the device properties do not expose the core counts, use the documented SOC table only as a hypothesis and confirm with the runtime or profiler. The 910B family has 20 or 24 AI cores depending on the model; 910_93 has 20 or 24; each AI core has one Cube and two Vector cores. Do not hard-code one model's count into a kernel that will run on another.

## Stage 1 — Audit GPU-centric launch assumptions and map them to AIC/AIV

Audit the source before changing tiles. The goal is to separate assumptions that are merely ineffective on the NPU from assumptions that change correctness or launch validity.

```bash
grep -nE 'num_warps|num_stages|tl\.extra\.cuda|inline_asm|shfl|shuffle|atomic_add|grid *=' migrated_kernel.py
```

| GPU assumption | Ascend consequence | First action |
| --- | --- | --- |
| `num_warps` tunes thread-level parallelism | No CUDA warp; the value may be ignored or rejected | Remove it from the kernel/autotune config; tune the tile and core count instead |
| `num_stages` is a software pipeline depth | No CUDA pipeline stage; multibuffer is a compiler/pipe decision | Remove it; use the backend's multibuffer control only after profiling |
| `tl.shfl_*` / warp shuffle reduction | No warp shuffle; the lane group does not exist | Replace with `tl.sum`/`tl.max` over the tile axis or an explicit UB reduction |
| Grid is `(B*S, ...)` or `(cdiv(tokens, BLOCK),)` | Grows with workload; CoreDim ceiling is 65535 and small blocks add launch gaps | Use a core-bounded persistent grid; see the L2 SOP |
| Pointer arithmetic assumes a flat 32-bit offset | A tensor above 2 GiB can wrap the linear offset | Widen the final byte offset to int64; keep tile coordinates int32 |
| A one-dimensional launch maps to one "thread block" | The NPU maps the first grid dimension to CoreDim and schedules AI cores, not CUDA blocks | Recompute ownership and tail coverage for the persistent loop |
| A kernel with `tl.dot` is tuned like a vector kernel | Cube and Vector have different tiles and core counts | Use `num_aicore` for dot kernels and `num_vectorcore` for pure Vector kernels |
| A pure Vector kernel is expected to use Cube | `aicore_time` can be zero; only `aiv_time` is meaningful | Profile the executed core type before diagnosing |

Map the work to the engine that actually executes it:

- **AIC / Cube.** `tl.dot`, matrix multiply, and Cube-side epilogue work. Use `props["num_aicore"]` (20/24 on the recorded families). A tile with M or N below 16 wastes the 16x16 Cube micro-tile; a K tile that is not a multiple of 16 for 16-bit inputs forces padding. On 950, a CV-fused tail has a 512 B alignment contract in addition to the 32 B UB contract.
- **AIV / Vector.** Elementwise, reduction, normalization, softmax, gather/scatter, and the vector half of a fused kernel. Use `props["num_vectorcore"]` (40/48 on the recorded families). A pure Vector kernel does not need `tl.dot` and should not be forced into a Cube path just to raise `aic_mac_ratio`.

Remove GPU-only primitives before measuring: `tl.extra.cuda`, inline PTX, CUDA libdevice symbols, cache modifiers, and warp-level operations have no Ascend lowering. Replace math with `tl.math` or `triton.language.extra.cann.libdevice`; replace warp reductions with tile reductions. A frontend that accepts an unknown call is not proof that BiShengIR lowers it.

**Stage 1 verification.** The audit is complete when the source no longer contains a warp/stage knob that is being used as a tuning variable, the launch grid has a proven bound, every pointer offset has a proven width, and each kernel has a declared engine (AIC, AIV, or mixed). Compile the migrated kernel once on the target and record the executed core type; if the compiler selected a mixed path unexpectedly, stop and explain why before tuning.

## Stage 2 — Restructure tiles for UB capacity, 32 B DMA and deterministic reductions

Use the hardware budget, not the GPU shared-memory budget. On 910B/910_93 the UB is 192 KiB and L1 is 512 KiB; L0A and L0B are 64 KiB each and L0C is 128 KiB. On 950 the UB is 248 KiB (8 KiB reserved by the compiler), L1 is 512 KiB, L0A/L0B are 64 KiB, and L0C is 256 KiB. A tile that fits one level does not necessarily fit the next.

For a matmul-like kernel, write the byte budget explicitly:

```text
L1  = L1_M * L1_K * sizeof(A) * stages_A
    + L1_K * L1_N * sizeof(B) * stages_B          <= 512 KiB
L0A = L0_M * L0_K * sizeof(A) * stages_A         <= 64 KiB
L0B = L0_N * L0_K * sizeof(B) * stages_B         <= 64 KiB
L0C = L0_M * L0_N * sizeof(accum) * stages_C     <= 128 KiB (A2/A3) or 256 KiB (950)
UB  = epilogue inputs + outputs + scratch        <= 192 KiB (A2/A3) or 248 KiB (950)
```

For fp16 input with fp32 accumulation and two input stages, `L1 <128,256,256>` costs `(128*256 + 256*256) * 2 * 2 = 384 KiB`, and `L0 <128,256,64>` costs 32 KiB for L0A, 64 KiB for L0B, and 128 KiB for a single fp32 L0C. That is a candidate, not a universal optimum; changing the dtype, transpose or stage count changes the budget. `m0 = m1` and `n0 = n1` are constraints of the Common template, not of every CATLASS kernel.

For a vector kernel, the budget is simpler but the alignment is stricter:

- Every innermost tile byte width is a multiple of 32 B. For bf16/fp16, 16 elements; for fp32, 8 elements. `BLOCK_N = 250` is legal as an extent but not as a 32 B tile boundary; use a 32 B multiple and mask the logical tail.
- The row stride, not only the allocation size, must be 32 B aligned. A padded allocation with an unpadded stride still starts every row after the first at a misaligned address.
- A Cube/L1 descriptor base and stride are 512 B aligned. If a view does not satisfy that, re-tile, pad, or stage through UB; do not rely on a compiler hint to repair a misaligned descriptor.
- A `tl.dot` M or N tile below 16, or a K tile that is not a multiple of 16 for 16-bit inputs, is a Cube-efficiency problem even when the output is numerically correct. Restructure the row/token loop into the M dimension rather than launching one row per program.
- Keep `BLOCK_M`, `BLOCK_N`, `HEAD_DIM` and the K tile static. Only the logical extent may be a runtime scalar; see the inherited constexpr contract.

Replace fine-grained global atomics with a deterministic reduction when many blocks write the same output:

```python
# Bad: one global read-modify-write per block on the same address.
tl.atomic_add(out_ptr + index, block_sum)

# Better: per-program private partials, then one fixed-order reduction.
tl.store(partial_ptr + pid, block_sum)
# Stage 2 reads partial_ptr[0:num_programs] and reduces in a fixed order.
```

For a split-K matmul, write one fp32 partial per `(pid_m, pid_n, k_split)` and reduce them in a fixed order. For a scatter-add, use a vector atomic only for a contiguous output segment and only when a commutative, non-deterministic result is acceptable; otherwise use a private per-core histogram or a two-stage scatter. The NPU does not support the GPU pattern of many blocks atomically accumulating and saving an intermediate in one pass.

**Stage 2 verification.** Recompute the byte budget from the actual buffers, not from the tile names. Compile every dtype/transpose/layout instance in the workload. Check `kernel.ttadapter.mlir` for `memref.alloc` sizes and confirm the largest concurrent set fits UB. Run the aligned, prime-tail and strided-view shapes through the Stage 3 oracle before timing. A fast kernel with an unaligned row stride or an unsupported atomic is not a completed migration.

## Stage 3 — Verify numerical parity on aligned, unaligned, high-dynamic-range and strided inputs

Do not compare only a friendly aligned shape. A migration can preserve the aligned prefix and fail exactly at the tail, the stride, or the low-precision accumulator.

Freeze one fp32 reference and all inputs before running the candidate. The case set must include:

1. **Aligned:** the innermost extent and every tile boundary are multiples of 32 B, and the logical extent is a multiple of the tile.
2. **Prime/unaligned tail:** `N` or `M` is prime or one element past a tile boundary; the last tile is partial.
3. **High dynamic range:** values spanning small and large magnitudes, plus a near-zero reference region, so a relative-error check does not hide an absolute error.
4. **Strided view:** a non-unit innermost stride or a storage offset that still satisfies the declared base alignment; if it does not satisfy 32 B, the expected result is the explicit contiguous fallback, not a silent fast path.
5. **Small and large:** `M < BLOCK_M`, `M = BLOCK_M`, a large `M` whose logical block count exceeds 65535, and a batch/sequence shape that changes the launch grid.
6. **Dtype:** every declared input and output dtype, with the exact output dtype checked.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 - <<'PY'
import torch
import torch_npu

def check(name, out, ref, dtype):
    torch.npu.synchronize()
    out_f, ref_f = out.float(), ref.float()
    finite = bool(torch.isfinite(out_f).all().item())
    abs_err = (out_f - ref_f).abs()
    rel_err = abs_err / ref_f.abs().clamp_min(1e-6)
    print(name, "finite", finite, "dtype", out.dtype,
          "max_abs", abs_err.max().item(), "max_rel", rel_err.max().item())
    assert finite, "NaN/Inf"
    assert out.dtype == dtype, "output dtype drift"

# check("aligned", y, ref_aligned, torch.bfloat16)
# check("prime_tail", y_tail, ref_tail, torch.bfloat16)
PY
```

For each case require: finiteness; max-abs and relative error inside the pre-registered tolerance; exact output dtype; input tensors unchanged (hash or compare before/after); and repeatability across at least three runs. For a reduction or normalization, the accumulator is fp32 and epsilon is applied in fp32; for softmax/attention, use the correct mask identity (`-inf` for masked logits) and `propagate_nan=True` where the reference propagates NaN. For a strided view, record which kernel actually ran; a contiguous fallback that returns the right answer is acceptable only if it was the intended path.

**Stage 3 verification.** A case that passes only after padding the reference, skipping the assertion, or comparing on CPU is not a pass. If the aligned case passes and the prime tail fails, repair the mask/identity before touching performance. If the high-dynamic-range or near-zero case drifts, repair the accumulator or the cast order before touching the tile. Record the failing artifact and the exact shape; do not average a failing case into a mean error.

## Stage 4 — Warm up and profile steady-state latency with MTE2/Vector/Scalar attribution

Only after Stage 3 passes, measure the migrated kernel. Separate cold compilation, first-call latency and steady-state latency; a compile-time win is not a kernel win.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 - <<'PY'
import time
import torch
import torch_npu

# x, w, out are prepared for one frozen case.
for _ in range(2):
    run_case(x, w, out)
torch.npu.synchronize()

for block in range(3):
    torch.npu.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(12):
        run_case(x, w, out)
    torch.npu.synchronize()
    t1 = time.perf_counter_ns()
    print("block", block, "mean_us", (t1 - t0) / 12 / 1000)
PY
```

Then capture the pipe breakdown on the same source hash and case:

```bash
msprof --application="python3 bench_case.py --shape 4096x1000 --dtype bf16" \
  --output=./prof_migrated --ai-core=on --aic-metrics=PipeUtilization --l2=on
python3 - <<'PY'
import glob
import pandas as pd

path = sorted(glob.glob("prof_migrated/**/op_summary_*.csv", recursive=True))[-1]
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

Attribute the leading pipe before choosing an optimization:

- `aiv_mte2` high and achieved bandwidth low: the access pattern or transaction size is poor; widen contiguous segments, coalesce rows, and reduce short DMA.
- `aiv_mte3` high: writeback dominates; fuse the epilogue, remove an intermediate GM tensor, or reduce the output stride.
- `aiv_vec` high: Vector ALU work dominates; remove transcendental work, vectorize the tail, or reduce redundant casts.
- `aiv_scalar` high: address arithmetic or branches dominate; hoist index math, replace per-element conditionals with masks, and check `tl.where` scalarisation.
- `aic_mac` high: Cube compute-bound; a different tile will not create free time.
- `aic_mte2` high and `aic_mac` low: Cube starvation; change L1 reuse, layout or prefetch.

Read `aicore_time` and `aiv_time` as separate core clocks; a pure-vector kernel can have `aicore_time = 0`, so use `max(aicore_time, aiv_time)` for core activity. PMU ratios are relative duty cycles; PMU collection can inflate kernel duration by roughly 2x to 16x. Take absolute latency from a separate run without PMU. A ratio above 1 is a counter inconsistency, not a saturated pipe. Do not merge AIC and AIV MTE2 columns.

**Stage 4 verification.** Report per-case block envelopes, not one pooled IQR across shapes. Confirm the baseline and candidate ran on the same device, clock, dtype and graph mode. Keep cold compile, first-call and steady-state values separate. If the kernel time improves but the end-to-end operator or serving latency does not, report the kernel-local result and stop there.

## Verification and exit criteria

A migration is complete only when all of the following hold:

1. **Launch audit:** no GPU warp/stage knob is used as a tuning variable; no CUDA-only intrinsic remains; the grid is bounded and every logical block is covered.
2. **Budget:** every tile instance fits UB/L1/L0 for its dtype and stage count; no `ub overflow` on any workload shape.
3. **Alignment:** every DMA path is 32 B compliant; every L1 descriptor is 512 B aligned; every `tl.dot` M/N/K tile satisfies the Cube contract or has a measured reason to pad.
4. **Parity:** aligned, prime-tail, high-dynamic-range, strided-view and small/large cases pass the frozen fp32 oracle with the exact output dtype, unchanged inputs and repeatability.
5. **Reduction:** no unsupported int64/fp64 atomic and no unbounded global RMW; a deterministic two-stage path is used where determinism is required.
6. **Timing:** at least two warmups, synchronized before/after blocks, at least three blocks, per-case envelopes, and a matched baseline on the same device and clock.
7. **Attribution:** the profiler artifact exists for the same source hash; the leading MTE2/Vector/Scalar/Cube pipe is named; kernel time is not reported as end-to-end latency.
8. **Claim boundary:** a compile pass is not parity; parity is not a speedup; a kernel speedup is not a model speedup. The result is local to the recorded SOC, shape, dtype, stack and clock.

If any criterion fails, retain the failing artifact and return to the stage that owns it: Stage 1 for launch/intrinsic errors, Stage 2 for budget/alignment/atomic errors, Stage 3 for numerical errors, Stage 4 for a timing or attribution claim. Do not proceed to performance tuning while Stage 3 is failing, and do not promote a migration result that has only an aligned-shape pass.
