---
schema: mindie-graded-unit/1
id: unit-triton-constexpr-to-runtime-scalar-specialization-mutator
revision: 1
tier: L4
category: mutator
domain_path: compilation/triton
evidence_strength: reproduced_empirical
title: 'Triton-Ascend dynamic-shape specialization to a runtime-scalar persistent grid'
conditions:
  kernel_language: triton
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
  workload: dynamic_shape
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-triton-dynamic-scalar-specialization-sop
provenance:
  source_type: ablation_design_receipt
  source_locator: docs/evidence/l4-triton-constexpr-runtime-scalar-ablation.json
  evidence_refs:
  - docs/evidence/l4-triton-constexpr-runtime-scalar-ablation.json
  experiment_status: not_run_source_only
  supersedes: []
triggers:
  error_signatures:
  - a Triton-Ascend serving path compiles a new kernel when the batch or token count changes
  - the number of Triton cache entries grows with distinct M, seq_len or num_tokens values
  - do_not_specialize was added but the length-derived argument is still declared tl.constexpr
  - coreDim is reported above UINT16_MAX or the launch creates thousands of tiny blocks
  - a non-multiple-of-BLOCK_M tail reads out of bounds, returns NaN/Inf or changes the denominator
  - a bf16 or fp16 reduction drifts from the fp32 reference after the persistent-loop rewrite
  keywords:
  - Triton-Ascend
  - dynamic shape
  - constexpr
  - do_not_specialize
  - specialization key
  - recompilation
  - persistent grid
  - coreDim
  - '65535'
  - masked tail
  - 32-byte alignment
  - fp32 accumulation
  - RMSNorm
  - Ascend910B
  - Ascend910_93
  - Ascend950
---
# Triton-Ascend dynamic-shape specialization to a runtime-scalar persistent grid

## Applicability

Use this mutator on an Ascend 910B, 910_93 or 950 Triton-Ascend kernel whose launch shape changes with the workload: `M`, `seq_len`, `num_tokens`, a per-core block count, or another extent derived from the batch. The observed symptom is a new compilation for each new extent and, at large extents, a grid that approaches the CoreDim ceiling or launches many tiny blocks.

The workload must have a **static tile geometry**: `BLOCK_M`, `BLOCK_N` and `HEAD_DIM` are powers of two, and every innermost tile byte width is a multiple of 32 B. For bf16/fp16 that means a multiple of 16 elements; for fp32, a multiple of 8. Keep those tile extents as `tl.constexpr`. Demote only the extents that vary.

A kernel whose `tl.arange` extent itself changes, whose innermost dimension is not 32 B aligned, or whose output must preserve a bitwise-fixed reduction order is outside this mutator without the fallbacks below. The inherited alignment axiom states the 32 B UB and 512 B L1 foundation for the 910B/910_93 family; the 950 capacity numbers in this unit come from the 950 target specification, so do not treat the inherited axiom as a 950 UB/L0C capacity claim. A missing observed condition is `condition_unknown`; do not infer it from a class name or a dispatch entry.

## Signals

- The first request at a new token count shows a multi-second compile spike; steady-state latency for the same shape is normal. The number of Triton cache entries tracks the number of distinct lengths rather than the number of kernels.
- `do_not_specialize` is present but the argument is still annotated `tl.constexpr`, so the specialization key still contains the extent. The compile count does not fall.
- The launch grid is `(cdiv(M, BLOCK_M),)` and grows with tokens; at a large enough `M` the runtime reports a CoreDim overflow, and below that line the timeline is full of short blocks with launch gaps.
- The last block is partial. A non-multiple-of-`BLOCK_M` shape reads wrong values, returns NaN/Inf, or changes its result when the tail is padded differently.
- After widening the reduction, the result is finite but drifts from the fp32 reference on high-dynamic-range or near-zero inputs. This points at a bf16/fp16 partial rather than a wrong mask.
- `msprof` shows a large `aiv_scalar_time` share from per-block index arithmetic, or a large launch gap between tiny kernels, before any arithmetic bottleneck.

## Transformation

Keep the boundary mask and the fp32 accumulator in every arm. The three factors are separable for leave-one-out: factor 1 changes the specialization key, factor 2 changes the launch geometry, and factor 3 changes the tail/precision contract. Apply them in this order; the linked receipt records them separately.

1. **`demote_dynamic_extent_from_constexpr`** — Remove `tl.constexpr` from every extent that varies: `M`, `seq_len`, `num_tokens`, `BLOCKS_PER_CORE`, and any per-core work count derived from them. Pass the extent as an `int32` runtime scalar, or `int64` only when the value can exceed `2**31 - 1`. Register the extents the kernel actually consumes with `do_not_specialize`; the annotation and the JIT option are two separate changes and both are required.

   ```python
   @triton.jit(do_not_specialize=["M", "seq_len", "num_tokens"])
   def rmsnorm_kernel(
       x_ptr, w_ptr, out_ptr,
       M, N, seq_len, num_tokens,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
   ):
       ...
   ```

   Keep the tile extents static and 32 B compliant:

   ```python
   # bf16/fp16: 16 elements = 32 B; fp32: 8 elements = 32 B
   BLOCK_M: tl.constexpr = 64       # 64 * 2 B = 128 B
   BLOCK_N: tl.constexpr = 256      # 256 * 2 B = 512 B
   HEAD_DIM: tl.constexpr = 128     # 128 * 2 B = 256 B
   ```

   Do not demote `BLOCK_M`, `BLOCK_N` or `HEAD_DIM` merely because a test shape is irregular. If a real tile shape changes, use a small finite bucket set and warm each bucket; do not let the number of specializations grow with the input.

   **Invariant.** The specialization key must be independent of `M`, `seq_len`, `num_tokens` and `BLOCKS_PER_CORE`. A `tl.constexpr` cannot be removed by `do_not_specialize` alone. Check the installed backend's JIT key rather than assuming GPU Triton behaviour.

2. **`fixed_core_persistent_grid_loop`** — Replace the token-proportional grid with a core-bounded grid and cover the remaining logical blocks inside the kernel. Query the physical core count from the device properties; do not hard-code a model number.

   ```python
   import torch
   import torch_npu
   import triton
   import triton.runtime.driver as driver

   def launch_rmsnorm(x, w, out, M, N, seq_len, num_tokens):
       props = driver.active.utils.get_device_properties(torch.npu.current_device())
       num_blocks = triton.cdiv(M, BLOCK_M)
       # A pure Vector kernel uses num_vectorcore; a kernel with tl.dot uses num_aicore.
       num_cores = props["num_vectorcore"]
       grid = (min(num_blocks, num_cores),)
       rmsnorm_kernel[grid](
           x, w, out, M, N, seq_len, num_tokens,
           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
       )
   ```

   ```python
   @triton.jit(do_not_specialize=["M"])
   def rmsnorm_kernel(...):
       pid = tl.program_id(0)
       num_programs = tl.num_programs(0)
       num_blocks = tl.cdiv(M, BLOCK_M)

       for bid in range(pid, num_blocks, num_programs):
           rows = bid * BLOCK_M + tl.arange(0, BLOCK_M)
           ...
   ```

   For a two-dimensional GEMM or attention tile grid, flatten the logical block index and recover the coordinates with scalar arithmetic:

   ```python
   num_m = tl.cdiv(M, BLOCK_M)
   num_n = tl.cdiv(N, BLOCK_N)
   num_blocks = num_m * num_n
   for bid in range(pid, num_blocks, num_programs):
       pid_m = bid // num_n          # keep this division scalar and int32
       pid_n = bid - pid_m * num_n
       ...
   ```

   For a kernel containing `tl.dot`, use `props["num_aicore"]` (the Cube cores) and keep the pure-vector path on `props["num_vectorcore"]`. On the recorded 910B family the ratio is 2 Vector cores per Cube core, but query it per device rather than encoding the ratio.

   **Invariant.** `grid_x <= 65535` is a hard launch condition, but `65535` is a ceiling, not a target. A persistent grid is bounded by the physical core count, so the same launch is valid from a small decode batch to a long prefill. The loop must cover every logical block exactly once: `pid + k * num_programs < num_blocks`. Skip the launch for an empty workload; a zero-sized grid is invalid.

   **Invariant.** `tl.num_programs(0)` is the actual grid size, not a compile-time model constant. If the installed backend cannot legalise it, pass the chosen core count as a warmed `constexpr`; that leaves a small, finite specialization set instead of one entry per token count.

3. **`masked_32b_tail_and_fp32_accumulator_guard`** — Keep an explicit boundary mask on every load and store, with the neutral identity for the operation, and keep the reduction accumulator in fp32 until the single final cast.

   ```python
   # Sketch: stride_x, stride_o and eps are defined by the real kernel.
   for bid in range(pid, num_blocks, num_programs):
       rows = bid * BLOCK_M + tl.arange(0, BLOCK_M)
       row_mask = rows < M
       ss = tl.zeros((BLOCK_M,), dtype=tl.float32)   # per logical row block

       # Pass 1: fp32 sum of squares over the full logical row.
       for n0 in range(0, N, BLOCK_N):
           cols = n0 + tl.arange(0, BLOCK_N)
           mask = row_mask[:, None] & (cols < N)[None, :]
           x = tl.load(
               x_ptr + rows[:, None] * stride_x + cols[None, :],
               mask=mask, other=0.0,
           ).to(tl.float32)
           ss += tl.sum(x * x, axis=1)

       mean_sq = ss / N
       rstd = 1.0 / tl.sqrt(mean_sq + eps)

       # Pass 2: normalize and store with the same boundary mask.
       for n0 in range(0, N, BLOCK_N):
           cols = n0 + tl.arange(0, BLOCK_N)
           col_mask = cols < N
           mask = row_mask[:, None] & col_mask[None, :]
           x = tl.load(
               x_ptr + rows[:, None] * stride_x + cols[None, :],
               mask=mask, other=0.0,
           ).to(tl.float32)
           w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
           y = x * rstd[:, None] * w[None, :]
           tl.store(out_ptr + rows[:, None] * stride_o + cols[None, :],
                    y.to(out_ptr.dtype.element_ty), mask=mask)
   ```

   Use `other=0.0` for `sum`, `mean` and `l2norm`; `-float("inf")` for `max` and `logsumexp`; `float("inf")` for `min`. For `mean`, divide by the true `N`, never by the padded tile width. For `logsumexp`, keep the row maximum and the exponentiated sum in fp32 and exclude padded lanes from the maximum.

   **Invariant.** A row accumulator belongs to one logical row block. Do not add one bid's row sum into the next bid's rows. When a reduction genuinely spans logical blocks (split-K or a cross-block sum), write a private fp32 partial per block and combine the partials in a fixed order; do not hide the cross-block combine in the row loop.

   **Invariant.** The 32 B contract applies to the tile width and the row stride, not only to the allocation size. A `BLOCK_N` of 16 bf16 elements is exactly 32 B; a row stride that is not 32 B aligned makes every row after the first start misaligned. Pad the row stride or pass real strides to a kernel that handles them.

   **Invariant.** For byte offsets above `2**31 - 1`, cast the final linear offset to `tl.int64` before adding it to the pointer. Do not cast every index to int64: Triton-Ascend's int64 arithmetic and modulo are more limited and slower than int32. Keep `bid`, `pid_m` and `pid_n` in int32 and widen only the byte address.

## Fallback boundary

- **Dynamic tile extent.** If `tl.arange` or an allocation needs a runtime extent, the tile geometry is not static. Keep a finite set of `constexpr` tile buckets and warm them; do not demote the tile extent.
- **Small logical grid.** If `cdiv(M, BLOCK_M)` is already at or below the physical core count, the direct grid may be simpler and avoids loop overhead. Retain the direct grid and record the crossover; do not force a persistent loop onto a tiny workload.
- **Bitwise-fixed reduction order.** A different persistent-loop assignment can change the order of partial additions. If the consumer requires bitwise identity, use a fixed two-stage partial buffer or restrict the schedule; a tolerance pass is not bitwise identity.
- **Pointer offset above 2 GiB.** If the int64 pointer path cannot be legalised or is unacceptably slow, split the tensor into chunks below the bound. Do not let an int32 offset wrap and do not silently drop the mask.
- **Runtime loop unsupported.** If the backend cannot legalise a runtime `range` bound or `tl.num_programs(0)`, pass the core count as a warmed constexpr. If that still fails, fall back to a finite bucket set. Never enumerate an unbounded `M` in a warmup table.
- **Unrepresentable identity.** If the tail cannot be masked with the correct neutral value, stage it through an aligned buffer and mask it explicitly, or do not take the fused path. An uninitialised tail is not neutral even when its value should be mathematically irrelevant.
- **Unaligned innermost extent.** If `HEAD_DIM` or the innermost tile is not a 32 B multiple, pad or stage it through an aligned UB buffer. Do not shrink the tile to a non-aligned width to make the byte count fit.
- **Graph capture or replay rejects the new grid.** If the graph engine cannot capture the core-bounded grid, keep eager execution for that path or capture the finite grid at the actual core count. Do not claim graph success from an eager-only run.

## Verification and ablation

Run the gates in order; a later gate does not repair an earlier failure.

- **Gate 1 — correctness on frozen inputs.** Freeze the fp32 reference and every input. Cover at least: an aligned `M` (multiple of `BLOCK_M`), `M = BLOCK_M - 1`, `M = BLOCK_M + 1`, a prime `M`, a large `M` whose logical block count exceeds 65535, a high-dynamic-range input, and a strided view with a 32 B compliant base. For each case require finiteness, max-abs and relative error inside the pre-registered tolerance, exact output dtype, unchanged input and repeatability. A tail case that passes only when padded is a failure.

- **Gate 2 — compile-count convergence.** Use an isolated cache per arm and per shape sweep:

  ```bash
  rm -rf cache_probe
  for M in 1 7 13 101 1009 65536; do
    TRITON_CACHE_DIR="$PWD/cache_probe" \
    python3 bench_rmsnorm.py --m "$M"
  done
  # The cache layout is backend-versioned; count the metadata files it actually emits.
  find cache_probe -type f \( -name '*.json' -o -name 'kernel.ttir.mlir' \) | wc -l
  ```

  For the full candidate, varying `M` must not create a new specialization. Count compiled kernels or cache metadata, not Python function calls. If the count grows, inspect the JIT signature: a remaining `tl.constexpr`, a length-derived value folded into a tile constant, or a changing `num_warps`/launch option will still specialize. Use `TRITON_ALWAYS_COMPILE=1` only for compile-time measurement, never for cache-count evidence.

- **Gate 3 — synchronized timing, cold and warm separately.** Exclude at least two warmups, call `torch.npu.synchronize()` immediately before and after each timed block, keep at least three blocks of twelve samples per case, and keep case identity on every sample. Report cold compile time, first-call latency and steady-state per-case envelopes separately. A compile-count win is not a steady-state speedup; a steady-state win with a worse cold path is not automatically a serving win.

- **Gate 4 — leave-one-out.** Run `full_candidate`, `minus_demote`, `minus_persistent_grid` and `minus_tail_guard` with Gates 1-3 repeated. The expected discriminations are:
  - `minus_demote`: correctness may pass, but compile count must grow with `M`; this isolates factor 1.
  - `minus_persistent_grid`: the large-`M` case must hit the CoreDim ceiling or the tiny-block regime; this isolates factor 2.
  - `minus_tail_guard`: the prime/unaligned tail or the low-precision accumulation case must fail correctness; this isolates factor 3 and is not allowed to pass as a small timing win.
  Do not reuse the full candidate's run as the minus-factor run. Do not prune a timing difference inside the measured noise envelope.

- **Gate 5 — guards and hidden retention.** Re-run the final retained combination on frozen hidden shapes. Cover the dynamic sweep, graph capture/replay, warm-server repeated call, workspace high-water mark, fallback selection and a shape that takes the direct-grid branch. Classify each item as `both_pass`, `opt_regression`, `both_fail`, `opt_improvement` or `evaluation_error`; only comparable `both_pass` items enter a speedup ratio. A local gain never overrides a guard regression.

**Provenance and claim boundary.** The recompilation storm and unaligned-tail failure were reproduced in the recorded user cases; the CoreDim ceiling and the UB/int1 contracts were reviewed against the Triton-Ascend and CANNBot software contracts. The controlled three-factor campaign, hidden-input retention run and synchronized speedup measurement were **not** executed for this candidate; the linked receipt records `experiment_status: not_run_source_only` and no measurement claims. This unit therefore prescribes a transformation and its verification contract, not a measured speedup. Promote it only after the full candidate, every leave-one-out arm and the guard set are independently verified.
