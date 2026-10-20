---
category: sop
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
  - Ascend950PR
  - Ascend950DT
  workloads:
  - MatmulA16W16
  - MatmulStreamK
  - BatchMatmul
  - QuantBatchMatmulMX
  - MatmulMixQuantMXA8W4
  - QuantBatchMatmulHif8Tt
domain_path: matmul/dsl-and-catlass
evidence_strength: reproduced_empirical
id: unit-cannbot-dsl-matmul-and-catlass-gemm-epilogue-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend950dt-measured-bandwidth-and-dt-vs-pr-overrides-measurement
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  attribution: Adapted from CANNBot-DSL matmul, batch_matmul, and quant_matmul samples
    and CANNBot-Knowledge matmul optimization guides
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/matmul/matmul.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/matmul/matmul_streamk.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/batch_matmul/batch_matmul.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/quant_matmul/quant_batch_matmul_mx.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/quant_matmul/quant_batch_matmul_mxa8w4.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/quant_matmul/quant_batch_matmul_hif8_tt.py
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/direct/nn/matmul/quant_matmul_950.md
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/matmul/matmul_streamk.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-cube-l1-residency-l0-pingpong-and-fixpipe-fusion-sop
- unit-ascendc-matmul-impl-iterate-gettensorc-and-gemm-api-contract
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for implementing non-quantized and quantized
  Matmul / BatchMatmul kernels in CANNBot-DSL and Ascend C, covering L2-aware sliding-window
  scheduling, Stream-K (DPSK) 3-phase MIX reduction, rank 2-6 broadcast BatchMatmul,
  and MXFP8/MXFP4/MXA8W4/HIF8 quantized pipelines.
tier: L2
title: CANNBot-DSL Matmul (Slide-Window, Stream-K, BatchMatmul, MX/MXA8W4/HIF8 Quant)
  & Ascend C Matmul SOP
triggers:
  error_signatures:
  - Stream-K (DPSK) matmul uses a single MIX kernel where AIC writes DP tiles to output
    GM and SK partial tiles to workspace GM, synchronized via a 3-phase cross-block
    barrier before AIV workspace reduction
  - QuantBatchMatmul MX (MXFP8/MXFP4) uses one E8M0 scale per 32 K-axis elements packed
    in pairs of shape [..., ceil(K/64), 2]
  - MXA8W4 mixed-precision matmul runs an AIV weight prologue to unpack/prepare FP4
    weights before AIC MX matmul execution
  keywords:
  - matmul
  - matmul_streamk
  - batch_matmul
  - quant_batch_matmul_mx
  - quant_batch_matmul_mxa8w4
  - quant_batch_matmul_hif8_tt
  - slide_window
  - stream_k_dpsk
  - mxfp8_mxfp4_scale_pair
  - l1_l0_pingpong
---
## Preconditions & Signals

Apply this SOP when building or tuning dense or quantized GEMM / BatchMatmul kernels in CANNBot-DSL or Ascend C on Ascend 910B/910C and Ascend 950 (`ARCH 3510`):

1. **Scheduling Regime Selection (`matmul.py` vs `matmul_streamk.py`)**:
   - **Adaptive Sliding-Window (`matmul.py`)**: When total output tiles $M_{\text{tiles}} \times N_{\text{tiles}} \gg N_{\text{AIC}}$, use L2-cache-friendly 2D sliding-window / swizzle scheduling with independent L1 and L0A/L0B ping-pong buffering.
   - **Hybrid DP + Stream-K (`matmul_streamk.py`)**: When $M, N$ are small (underutilizing $N_{\text{AIC}}$ or creating a severe last-wave tail) and $K$ is large, use Data-Parallel + Stream-K (`DPSK`) hybrid scheduling in a single MIX kernel.
2. **High-Rank Broadcast BatchMatmul (`batch_matmul.py`)**:
   - Computes $C[c_{\text{batch}}, M, N] = A[a_{\text{batch}}, M, K] @ B[b_{\text{batch}}, N, K]^\top + \text{bias}$ across ranks $2 \dots 6$ with NumPy/PyTorch right-aligned size-1 broadcast semantics, zero-copy transpose view deduction, and optional `hf32=True` (TF32 fast mode for FP32 inputs).
3. **Ascend 950 Quantized Matmul Variants (`quant_matmul/`)**:
   - **OCP MX (`quant_batch_matmul_mx.py`)**: Unified `MXFP8` (`float8_e4m3fn`, `float8_e5m2`) and `MXFP4` (`float4_e2m1fn_x2`) with paired `E8M0` scales along $K$.
   - **Mixed `MXA8W4` (`quant_batch_matmul_mxa8w4.py`)**: `FP8` activations $\times$ `FP4` weights using AIV weight prologue + AIC MX matmul.
   - **Per-Tensor `HIF8` (`quant_batch_matmul_hif8_tt.py`)**: `GM -> L1 -> L0 -> L0C -> FIXPIPE (deq_scale) -> GM`.

## Procedure

### Step 1: L1/L0 Ping-Pong & Sliding-Window Matmul (`matmul.py`)
1. **Decoupled L1 and L0 Double Buffering**:
   - Tile outer loops over $(M_0, N_0)$ with a 2D sliding window (window width $W \in \{4, 8\}$) so concurrent AIC cores share $A$ or $B$ tiles in L2 cache.
   - Stage bigger $K_{\text{L1}}$ chunks from GM into L1 `PING`/`PONG` buffers via MTE2 while stepping smaller $K_{\text{L0}}$ sub-tiles from L1 into L0A/L0B `PING`/`PONG` via MTE1. Never couple L1 buffer lifetime 1:1 to a single L0 sub-step when $K_{\text{L1}} > K_{\text{L0}}$.
2. **Software Prefetch Slot Identity**:
   - Track each prefetched tile by its complete work-item coordinate `(mn_tile_idx, k_slice_idx, base_addr, valid_M, valid_N, valid_K)`. Bound initial prefetch launch by `min(prefetch_depth, num_tasks)` so short $K$ loops never wait on unlaunched producers.

### Step 2: Stream-K (`DPSK`) Single-MIX-Kernel Pipeline (`matmul_streamk.py`)
1. **Work Partitioning (`DP` Tiles vs `SK` Tiles)**:
   - Partition full waves of output tiles as Data-Parallel (`DP`) tiles written directly by AIC via FixPipe to output GM.
   - Partition remaining tail output tiles across all AIC cores along the $K$ dimension as Stream-K (`SK`) splits; each AIC writes its partial FP32/BF16 sum for an `SK` split to `workspace` GM.
2. **Three-Phase Cross-Block Barrier & AIV Register Reduction**:
   - **Phase 1**: AIC cores finish computing and flushing `SK` partial tiles from L0C to `workspace` GM.
   - **Phase 2**: Execute cross-block synchronization between AIC and AIV cores.
   - **Phase 3**: Launch `GetTaskRatio() * k_splits` AIV sub-blocks (each AIV handles an $M$-segment of one `mn_tile` reduction), accumulate partial sums in vector registers inside a VF block, and write the reduced tile to output GM.

### Step 3: Rank 2–6 Broadcast `BatchMatmul` (`batch_matmul.py`)
1. **Host Stride Folding & Broadcast Deduction (`BatchMatmulTiling`)**:
   - Right-align `a_batch` and `b_batch` dimensions (ranks $2 \dots 6$); fold contiguous non-broadcast batch dimensions and set stride $= 0$ along any size-1 broadcast dimension.
   - Deduce transposed views (`A.T`, `B.T`) via stride inspection at zero copy cost rather than materializing contiguous transpose copies before the kernel.
2. **Stride Scheduler & Epilogue Bias (`BatchMatmulKernel`)**:
   - Decompose linear batch index $b_{\text{linear}}$ on device via precomputed magic-number divmod into per-dimension coordinates to compute $A$ and $B$ batch offsets, fuse `bias` addition in FixPipe / L0C, and enable `hf32=True` only when input dtype is `float32` and `~3e-4` relative tolerance is permitted.

### Step 4: MXFP8 / MXFP4 / MXA8W4 / HIF8 Quantized Matmul (`quant_matmul/`)
1. **Paired `E8M0` Scale Layout (`quant_batch_matmul_mx.py`)**:
   - Each 32 elements along $K$ share one `E8M0` scale; pack adjacent pairs of scales so the scale tensor has K-group dimension `ceil(K / 64)` and trailing dimension `2` (`[..., ceil(K/64), 2]`).
   - Stream data and paired `E8M0` scales through `GM -> L1 -> L0 -> L0C -> FIXPIPE -> GM` using ASWT host tiling.
2. **Mixed `MXA8W4` AIV-to-AIC Collaborative Pipeline (`quant_batch_matmul_mxa8w4.py`)**:
   - Use `MixQuantKernel` where AIV cores execute the weight prologue (unpacking/formatting `FP4` weights into L1/workspace) synchronized with AIC cores executing the MX matmul against `FP8` activations.
3. **Per-Tensor `HIF8` (`quant_batch_matmul_hif8_tt.py`)**:
   - Apply the per-tensor dequantization scale (`deq_scale`) directly in the `FIXPIPE` hardware stage when draining L0C to GM.

## Verification & Checks

1. **Static Tiling Upper Bound vs Runtime Shape Separation**:
   - Never conflate compile-time template upper bounds (`maxSingleCoreM/N/K`), `baseM/baseN/baseK` tile granularities, and runtime `SetOrgShape` / `SetSingleShape` / `SetTail` extents.
2. **Stream-K Determinism**:
   - Verify that AIV reduction in `matmul_streamk.py` accumulates the `k_splits` workspace slices in fixed deterministic split order `$s = 0 \dots k_{\text{splits}}-1$` rather than using non-deterministic cross-core atomic adds to GM.

## Failure Modes & Fallbacks

- **Symptom**: Stream-K kernel hangs or reads stale zeros from `workspace` GM during AIV reduction.
  - **Root Cause**: Missing cross-core barrier between AIC `FIXPIPE` writeback and AIV `MTE2` workspace read, or mismatch between AIC workspace slot indexing and AIV `GetTaskRatio() * k_splits` consumer indexing.
  - **Fix**: Enforce the 3-phase cross-block barrier after AIC FixPipe completion and verify workspace stride alignment.
- **Symptom**: Wrong output values on `MXFP8`/`MXFP4` matmul when $K$ is not a multiple of 64.
  - **Root Cause**: Unpaired odd `E8M0` scale group at the $K$ tail (`ceil(K / 32)` is odd) misaligns the `[..., ceil(K/64), 2]` scale pair layout.
  - **Fix**: Pad the odd trailing scale slot in the final `[..., ceil(K/64), 2]` pair with neutral scale (`0x7F` / zero-padded K data) and pass exact runtime $K$ to the kernel tail guard.
