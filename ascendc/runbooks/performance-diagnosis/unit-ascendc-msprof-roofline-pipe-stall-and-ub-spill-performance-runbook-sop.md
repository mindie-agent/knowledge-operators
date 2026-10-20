---
category: sop
conditions:
  kernel_language: ascendc
  soc:
  - Ascend950
  - Ascend310P
  - Ascend910B
  - Ascend910_93
  - Ascend950PR
  - Ascend950DT
  workloads:
  - PerformanceProfiling
  - MsprofOp
  - RooflineAnalysis
  - PipelineStallDiagnosis
  - KernelOnlyTiming
domain_path: ascendc/runbooks/performance-diagnosis
evidence_strength: reproduced_empirical
id: unit-ascendc-msprof-roofline-pipe-stall-and-ub-spill-performance-runbook-sop
inherits_from:
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-ascend950dt-measured-bandwidth-and-dt-vs-pr-overrides-measurement
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
provenance:
  attribution: Adapted from the 163 performance diagnosis runbooks in CANNBot-Knowledge
    ops/ascendc/runbooks/performance and guides/performance
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/performance/aclnn_timing_aclrtevent_kernel_only.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/performance/aclnn_task_prof_fallback_collection.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/performance/bandwidth_bound_compute_optimization_no_gain.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/performance/bandwidth_util_over_100pct_l2_working_set.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/performance/block_count_uses_all_available_vector_cores.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/performance/aclnn_timing_aclrtevent_kernel_only.md
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-ascendc-datacopypad-elimination-ub-bank-conflict-and-l2-tiling-sop
- unit-ascendc-cube-l1-residency-l0-pingpong-and-fixpipe-fusion-sop
- unit-ascendc-discovered-evolution-strategies-and-cross-core-pipeline-mutator
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Ascend C performance diagnosis using msprof
  and aclrtEvent, mapping hardware pipe utilization ratios (mac_ratio, vec_ratio,
  mte1/2/3_ratio, scalar_ratio, fixpipe_ratio) to concrete bottleneck classes and
  structural fixes.
tier: L2
title: Ascend C msprof Roofline Attribution, Pipe Stall Diagnosis & UB Spill Performance
  Runbook SOP
triggers:
  error_signatures:
  - Kernel-only latency must be measured via aclrtEvent on the active NPU stream (after
    warmup and L2 cache flush/steady-state separation) or msprof task/op profiling,
    never wall-clock host time
  - Optimizing Vector/Cube compute instructions on a kernel where mte2_ratio > 0.75
    and vec_ratio < 0.15 yields zero wall-clock gain unless GM traffic, DMA descriptor
    count, or L2 hit rate changes
  - Apparent bandwidth utilization > 100% of HBM peak indicates the working set fits
    inside the 192 MB / 128 MB L2 cache; benchmarks must size inputs or rotate buffers
    to reflect true HBM vs L2 regime
  keywords:
  - msprof_op
  - roofline_four_view
  - aclrtevent_kernel_timing
  - mte2_ratio
  - vec_ratio
  - mac_ratio
  - scalar_ratio
  - bandwidth_bound_no_gain
  - l2_working_set_artifact
---
## Preconditions & Signals

Apply this SOP before and after every Ascend C or CANNBot-DSL performance optimization to diagnose the true hardware bottleneck and avoid wasted optimization cycles (`bandwidth_bound_compute_optimization_no_gain.md`):

1. **Accurate Kernel-Only Timing (`aclnn_timing_aclrtevent_kernel_only.md`, `aclnn_task_prof_fallback_collection.md`)**:
   - Host `std::chrono` or `time.perf_counter()` around `aclnn*GetWorkspaceSize + aclnn*` measures host tiling, workspace allocation, and launch queue overhead rather than device execution time.
   - Always measure device kernel duration using `aclrtRecordEvent(start, stream)` $\rightarrow$ `aclnn*(...)` $\rightarrow$ `aclrtRecordEvent(stop, stream)` $\rightarrow$ `aclrtSynchronizeEvent(stop)` $\rightarrow$ `aclrtEventElapsedTime(&ms, start, stop)` after $\ge 5$ warmup iterations, or collect PMU counters via `msprof op` / `msprof --application=...`.
2. **Beware the L2 Cache Working-Set Illusion (`bandwidth_util_over_100pct_l2_working_set.md`)**:
   - When total input + output bytes are smaller than the L2 cache (`192 MB` on 910B, `128 MB` on 950) and the benchmark loops over the *same* tensor pointers, warmup pulls all tensors into L2 (`~4–6 TB/s`), yielding apparent HBM utilization $> 100\%$. Rotate across a ring of distinct buffer allocations (`total ring size > 2 x L2`) when benchmarking cold-HBM bandwidth.

## Procedure

### Step 1: Collect & Classify the Four-View Pipe Utilization Signature
Inspect the `msprof` Op Summary (`Task Duration (us)`, `Block Dim`, `mac_ratio`, `vec_ratio`, `scalar_ratio`, `mte1_ratio`, `mte2_ratio`, `mte3_ratio`, `fixpipe_ratio`, `icache_miss_rate`) and classify the dominant bottleneck:

| Pipe Signature | Bottleneck Class | Primary Root Causes | TargetSOP / Mutator |
|---|---|---|---|
| `mte2_ratio > 0.65` AND Achieved BW $> 75\%$ HBM peak | **True HBM Bandwidth Bound** | Algorithmic byte volume (extra passes, workspace round-trips, FP32 upcast on host) | Fuse passes in UB (`P69`), keep intermediates on-chip (`P35`/`P41`), use `dtype` passthrough (`disc_x3`). **Do not tune Vector compute instructions** (`bandwidth_bound_compute_optimization_no_gain.md`)! |
| `mte2_ratio > 0.60` AND Achieved BW $< 45\%$ HBM peak | **DMA Latency / Alignment / L2 Miss Bound** | Excessive small DMA descriptors (`disc_x95`), per-row `DataCopyPad`, unaligned 512B GM offsets, or L2 cache thrashing | Coalesce rows (`P3`/`P56`), split interior aligned `DataCopy` from tail `DataCopyPad`, apply serpentine L2 tiling (`P107`). |
| `scalar_ratio > 0.25` (or `vec/mac/mte` all $< 0.20$) | **Scalar / Launch Overhead Bound** | Scalar `GetValue`/`SetValue` loops, per-repeat `Normal` mask setup, heavy host/device init per task (`P54`), or single-core under-dispatch | Replace scalar loops with `GatherMask`/`Brcb` (`P94`–`P101`), switch to `MaskMode::COUNTER` (`P67`), expand `blockDim` to all 48 AIV / 24 AIC cores (`block_count_uses_all_available_vector_cores.md`). |
| `vec_ratio > 0.65` | **Vector Compute / Bank Conflict Bound** | Multi-pass horizontal `WholeReduce`, transcendental chain expansion, UB bank conflicts (`P65`), or RegBase register spills | Fold repeats with `Add` before a single `BlockReduce+WholeReduce` (`P68`), stagger UB offsets by `+32B` (`P65`), fuse `__simd_vf__` (`R1`/`R2`). |
| `mac_ratio < 0.35` AND `mte1_ratio` or `mte2_ratio` high on Cube | **Cube Starvation / False Ping-Pong** | Small `baseM/baseN` (`L0B` fill bottleneck, `X41`), synchronous `L1<->L0` flags, or materialized `im2col` GM round-trips | Decouple `L1`/`L0` double buffering, enlarge `baseM`/`singleCoreN` (`X11`/`X41`), use `LoadData3D` (`P129`) or L1 residency (`P53`). |
| All pipe ratios $< 0.25$ on MIX kernel | **Cross-Core AIC<->AIV Handshake Stall** | Lockstep per-chunk `CrossCoreWaitFlag` rendezvous (`X33`, `X83`) or small `Q/KV` tile granularity in GM workspace ring (`X38`) | Decouple AIC/AIV into a multi-slot software pipeline (`P14`, `P20`, `X83`) and widen workspace tile granularity (`X38`). |

### Step 2: Audit Multi-Core Occupancy & Wave Balance (`block_count_uses_all_available_vector_cores`)
1. **Check `Block Dim` in `msprof` Summary**:
   - On Ascend 910B (24 AIC / 48 AIV) or Ascend 950 (32 AIC / 64 AIV), verify that pure Vector kernels report `Block Dim = 48` (or `64` on 950) when problem size is large enough.
   - A common bug in host tiling queries `GetCoreNumAic()` (`24`) instead of `GetCoreNumAiv()` (`48`) for a Vector-only kernel, leaving 50% of the Vector cores completely idle!
2. **Audit Per-Core Cycle Skew**:
   - Inspect per-core cycle histograms in `msprof`: if Core 0 takes $3\times$ longer than Core 47 (e.g., due to triangular causal attention workload or ragged `TND` sequence lengths), switch from uniform row partitioning to cost-driven / zig-zag / greedy bin-packing partitioning (`P16`, `P58`).

### Step 3: Paired Baseline Acceptance Verification
1. **Before/After A/B Profiling on Identical Thermal & Frequency State**:
   - Run paired baseline vs optimized kernel measurements back-to-back in the same process (`bandwidth_bound_elementwise_paired_baseline_acceptance.md`) and verify that:
     - Total kernel duration (`us`) decreases by the expected margin;
     - The targeted hardware counter (`mte2_cycles`, `scalar_cycles`, `vec_cycles`, or `aic_mac_ratio`) shifts in the predicted direction;
     - All accuracy checks (`rtol`/`atol`, 3-run bitwise determinism) remain green.

## Verification & Checks

1. **Verify Roofline Feasibility**:
   - Compute $\text{T}_{\text{floor}} = \max\left(\frac{\text{Bytes}_{\text{GM}}}{\text{BW}_{\text{measured}}}, \frac{\text{FLOPs}}{\text{Peak}_{\text{Cube/Vec}}}\right)$. Once kernel time reaches $\ge 80\%$ of $\text{T}_{\text{floor}}$, further micro-optimizations without reducing algorithmic GM bytes (fusion) cannot yield $> 1.25\times$ speedup.
2. **Check `icache_miss_rate`**:
   - Excessive template unrolling (`#pragma unroll` on huge loops) can overflow the 32 KB / 64 KB instruction cache, spiking `icache_miss_rate`; bound unroll factors to `2` or `4`.
