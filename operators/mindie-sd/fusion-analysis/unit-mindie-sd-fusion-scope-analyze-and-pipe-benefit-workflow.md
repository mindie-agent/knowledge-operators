---
schema: mindie-graded-unit/1
id: unit-mindie-sd-fusion-scope-analyze-and-pipe-benefit-workflow
title: MindIE-SD fusion-scope-analyze structural boundary rules, PipeUtilization benefit ceiling, and post-hoc equal-work audit workflow
tier: L3
category: workflow
evidence_strength: reproduced_empirical
domain_path: operators/mindie-sd/fusion-analysis
inherits_from:
  - unit-ascend-operator-roofline-and-four-view-measurement
  - unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
  - unit-mindie-sd-custom-sparse-attention-catlass-ffn-and-triton-ops-sop
conditions:
  cann_versions:
    - ">=8.5.0"
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
  workloads:
    - Operator fusion scope identification, PipeUtilization go/no-go estimation, and post-fusion regression root-cause analysis
triggers:
  error_signatures:
    - "fused kernel wall clock is flat or slower than unfused baseline"
    - "PipeUtilization columns missing or N/A in op_summary"
    - "sum of independent candidate fusion speedups exceeds actual combined speedup"
  keywords:
    - fusion-scope-analyze
    - check_fusion_scope
    - memory_bound
    - PipeUtilization
    - mte2_ratio
    - mac_ratio
    - vec_ratio
    - cube_utilization
    - equal-work-busy-cycle
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/fusion-scope-analyze/SKILL.md
  source_digest: 0150cf3251d8afcc7ee720dee1e9d5939d17d2dc343b08c9d13129ce0b3e33fa
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/fusion-scope-analyze/SKILL.md
  supersedes: []
revision: 1
---
## Fusion Scope & Benefit Analysis Workflow

### Step 1: Validate Profiling Inputs (`kernel_details.csv` + `PipeUtilization`)
1. Require both operator execution sequence (`kernel_details.csv`: `Name`, `Start Time(us)`, `Duration(us)`, `Task Type`) and `PipeUtilization` metrics (`op_summary_*.csv`: `*_vec_ratio`, `*_mac_ratio`, `*_mte2_ratio`, `*_mte3_ratio`, `cube_utilization(%)`).
2. If any of the four pipe ratio families are missing or `N/A`, reject the capture fail-closed and re-collect with `--task-time=l1 --aic-mode=task-based --aic-metrics=PipeUtilization`.
3. Compute `memory_bound = mte2_ratio / max(mac_ratio, vec_ratio)` directly from the four pipe ratios (do not rely on a pre-populated `memory_bound` column).

### Step 2: Partition Structural Fusion Boundaries (Rules 1–6)
1. **Rule 1 (Function Boundary Candidate Group)**: Operators defined within the same model function (`norm`, `rope`, modulation helpers) form initial candidate groups.
2. **Rule 2 (Attention Family Hard Anchor)**: Attention kernels and all sparse/quantized variants (`FA`, `BSA`, `RainFusion`, `EagleQBSA`) are **strict non-fusable anchors**—never absorb or fuse across them.
3. **Rule 2b (Communication Conditional Anchor)**: `Task Type == COMMUNICATION` ops are boundaries **only** when serially dependent with zero compute overlap; concurrent comm-stream collectives do not break a compute fusion chain.
4. **Rule 3 (Anchor Partitioning)**: Partition the execution timeline by `norm` / `rope` / `FA` anchors; prioritize pure-Vector regions for single-pass fusion.
5. **Rule 4a/4b/4c (Cube-Head Wrapping)**:
   - **4a (Latency-Hiding Type)**: When $\Sigma\text{vec} < \text{Cube}$, use the **Cube op as the head operator** and wrap trailing Vector ops until $\Sigma\text{vec} \ge \text{Cube}$ or the next Cube op is reached.
   - **4b (GM Round-Trip Elimination Type)**: A small Cube op followed by a large Vector chain ($\Sigma\text{vec} \ge \text{Cube}$) is still eligible for fusion if their shapes/dtypes connect directly, eliminating intermediate GM read/write traffic.
   - **4c (No Cross-Dual-Cube Complex Fusion)**: Do not fuse `Cube + Vec + Vec + Cube` into a single mega-kernel when the middle Vector segment is smaller than both Cube ops; handle each Cube head separately.
6. **Rule 5 & 6 (Iterative Pure-Vector Fusion & Existing Carrier Check)**: After one pass, iteratively merge adjacent pure-Vector fusion units if their combined benefit exceeds the $0.5\%$ block threshold. If a region is already served by a single fused operator, verify whether that carrier is actually engaged before proposing a new unit.

### Step 3: Classify Pipe Bottleneck & Compute Two-Tier Go/No-Go Benefit
1. **Pipe Classification**:
   - **MTE-Saturated (`memory_bound > 1`, high `mte2/mte3_ratio`, low `vec_ratio`)**: **Fusable (Go)**—fusion eliminates intermediate GM round-trips.
   - **Vector-Saturated (`memory_bound < 1`, high `vec_ratio`, `mte2_ratio ~ 0`)**: **Reject (No-Go)**—reducing GM traffic or merging more Vector ops cannot speed up a compute-saturated Vector pipe.
   - **Vector-Constrained Single Pass**: Worth fusing only up to the Vector compute ceiling.
2. **Site-Overlap Deduplication (Anti-Double-Counting)**:
   - Before ranking candidates, audit whether Candidate A (e.g., cross-layer boundary fusion) consumes the target operator sites of Candidate B. Overlapping candidates **must never have their standalone speedups summed**; report Candidate B's residual value under Candidate A.
3. **Two-Tier Benefit Estimation (`L0` Traffic Ceiling $\to$ `L1` Critical-Stream Theoretical Floor)**:
   - Compute `L1` theoretical floor on the **critical stream only** (`Bytes / Measured_Effective_BW` and `FLOPs / Measured_Peak_Compute`), excluding operators where `duration_over_theoretical ~ 1.0`, and verify Amdahl propagation against the noise floor (`perf-gate`).

### Step 4: Post-Hoc Equal-Work Busy-Cycle Audit (When a Fused Kernel Measures Flat or Negative)
1. Never close a flat/negative fusion experiment as "fusion has no benefit" before running an **equal-work busy-cycle comparison**:
   - Verify the fused kernel actually executed on the critical stream and replaced the original kernels (`check_fusion_hit.py` + `kernel_details.csv`).
   - Compare total AI Core busy cycles (`aicore_time * core_count` and pipe active cycles) of the fused kernel against the sum of replaced unfused kernels on identical inputs.
2. If the fused kernel exhibits **higher AI Core busy cycles** for equal mathematical work (due to register spill, suboptimal tiling, redundant broadcast, or epilogue serialization), escalate to `operator-dev` to fix the kernel implementation rather than abandoning the fusion scope.
