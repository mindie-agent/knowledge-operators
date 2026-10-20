---
schema: mindie-graded-unit/1
id: unit-operator-memory-vs-compute-bottleneck-convergence-workflow
revision: 1
tier: L3
category: workflow
domain_path: Reduction
evidence_strength: software_contract
title: 'Operator bottleneck convergence: MTE2/MTE3, Vector/Cube, and host launch triage'
conditions:
  soc:
  - Ascend910B
  - Ascend910B1
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  - float32
  op_family:
  - reduction
  - elementwise
  - norm
  - matmul
  workload:
  - fixed_shape
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
provenance:
  source_type: public_profiling_contract_review
  source_locator: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devaids/auxiliarydevtool/atlasprofiling_16_0001.html
  source_url: https://www.hiascend.com/document
  evidence_refs:
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devaids/auxiliarydevtool/atlasprofiling_16_0001.html
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html
  - https://github.com/Ascend/ascendc-samples
  - https://github.com/Ascend/pytorch
  supersedes: []
triggers:
  error_signatures:
  - an operator is declared vector-bound or memory-bound from end-to-end latency alone
  - msprof shows low aicore_time and aiv_time but the operator is still slow
  - vectorization increases vector ALU time without reducing wall time
  - adding multibuffering or a larger tile makes the operator slower
  - a speedup is claimed after one iteration without checking that the bottleneck moved
  keywords:
  - operator optimization
  - bottleneck
  - MTE2
  - MTE3
  - Vector
  - Cube
  - host launch
  - tiling
  - multibuffer
  - redundant cast
  - msprof
  - aicore_time
  - pipe utilization
  - convergence
  - Ascend910B
  - Ascend910_93
---
# Operator bottleneck convergence: MTE2/MTE3, Vector/Cube, and host launch triage

## Trigger and scope

Use for a fixed-shape operator optimization on Ascend 910B/910_93 when the current result is slower than a matched baseline and the next action is not obvious. The workflow first decides whether the limit is host launch/tiling, MTE2 input movement, MTE3 output movement, Vector ALU, Cube MAC, scalar instruction issue, or a mixed profile. It then chooses one transformation family and iterates only while the bottleneck moves. It does not replace the correctness oracle or the L1 measurement contract.

Out of scope: changing the mathematical contract, tuning a kernel whose correctness gate fails, and claiming a model-level or serving-level speedup from an operator-local result.

## Facts to freeze before profiling

Record SOC and die count, CANN/`torch_npu`/framework revision, kernel source hash, shape and stride, dtype and output dtype, reduction axis or tile layout, graph/eager mode, device clock, warmup count, sample count, and the matched baseline hash. Record whether the baseline and candidate were built with the same compiler flags, tiling constants and environment. A missing fact stays `unknown`.

Confirm the device is healthy and the workload is actually on the NPU:

```bash
npu-smi info
ASCEND_RT_VISIBLE_DEVICES=0 python3 -c \
  "import torch, torch_npu; print(torch.npu.get_device_name(0)); print(torch.npu.is_available())"
```

## Stage 1 — Separate wall time from device time

Run the synchronized baseline defined by the inherited measurement L1. In parallel, capture one `msprof` run of the same case:

```bash
msprof --application="python3 bench_op.py --shape 4096x1000 --dtype bf16" \
  --output=./prof_op --ai-core=on --aic-metrics=PipeUtilization --l2=on --msproftx=on
```

Inspect the exported task table before selecting a transformation:

```bash
python3 - <<'PY'
import glob
import pandas as pd

path = sorted(glob.glob("prof_op/**/kernel_details*.csv", recursive=True))[-1]
df = pd.read_csv(path)
print(path, "rows=", len(df))
print("columns=", list(df.columns))
print(df.head(20).to_string())
PY
```

Use the exporter's actual column names. The classification needs, at minimum: operation name/type, start time, duration, block dimension, accelerator core, and the AIC/AIV pipeline columns when exported. If the run exports `op_summary_*.csv` instead, use the same fields from that table. If pipeline columns are absent, classify at the op-type level and record `pipeline signal unavailable`; do not fabricate zeroes.

Compute two ratios on the same workload:

```text
kernel_share = sum(device task durations) / synchronized wall time
gap_share    = (wall time - sum(device task durations)) / wall time
```

Discriminating observations:

- `kernel_share < 0.5`: host launch, Python dispatch, tiling setup, synchronization, or gaps between tasks dominate. Go to Branch A before changing a kernel.
- `0.5 <= kernel_share <= 0.8`: mixed host/device. Inspect the timeline for gaps and short tasks; a kernel change can still be swamped by host overhead.
- `kernel_share > 0.8`: device-bound. Go to Stage 2.
- Many tasks of very short duration, small `Block Dim`, or large inter-task gaps: launch/tiling overhead even if the total kernel time is not tiny.
- `op_type` of `aicpu`, `dsa`, or pure `communication`: exclude these from Vector/Cube classification; they have their own branch.

## Stage 2 — Classify the device-bound pipeline

Aggregate the nine sub-stages separately for AIC and AIV. Do not merge `aic_mte2_time` with `aiv_mte2_time`; they are different memory paths. For each aggregate, compute the stage-family totals:

```text
cube    = aic_mac_time + aic_fixpipe_time
vector  = aiv_vec_time
aic_mte = aic_mte1_time + aic_mte2_time
aiv_mte = aiv_mte2_time + aiv_mte3_time
scalar  = aic_scalar_time + aiv_scalar_time
```

Use a 10% mixed margin as a decision rule, not as a physical constant: if the top family leads the runner-up by less than 10% of the summed stage time, treat the profile as **mixed** and drill into the per-stage values. Otherwise name the leading family.

Then read the leading stage:

| Leading observation | Classification | First action family |
| --- | --- | --- |
| `aiv_mte2_time` high, `aiv_vec_time` low | AIV input-memory-bound (GM → UB) | larger UB tile, multibuffer/prefetch, reuse, coalescing, remove duplicate loads |
| `aiv_mte3_time` high | AIV output-memory-bound (UB → GM) | fuse epilogue, write once, coalesce, remove intermediate GM tensors |
| `aiv_vec_time` high, MTE low | Vector ALU-bound | vectorize the tail, remove scalar loops/branches, reduce transcendental work, increase ILP |
| `aic_mac_time` high | Cube MAC-bound | M/N/K tiling, dtype/layout, L1 reuse, 512 B tile alignment |
| `aic_mte2_time` high, `aic_mac_time` low | Cube starved on input/weights | L1 prefetch, weight layout, tile reuse, reduce GM traffic |
| `aiv_scalar_time` or `aic_scalar_time` high | Scalar instruction-bound | unroll/vectorize, replace per-element branches with masks |
| `op_type = mix_cv` | Cube and Vector overlap | inspect AIC and AIV totals side by side; do not collapse to one core |
| `op_type = mix_comm_aiv` | Collective + AIV fused | rank AIV stages only; separate communication wait from AIV work |
| no stage leads, or pipeline signal absent | mixed/unknown | stop pipeline tuning; use Stage 1 host evidence or a targeted microbenchmark |

A low `l2_cache_hit_rate` with high MTE2 and high HBM traffic supports input-memory-bound. A high L2 hit rate with high MTE2 points to UB tiling or prefetch depth rather than HBM bandwidth. Neither observation alone proves the bound.

## Stage 3 — Choose one transformation family

Apply exactly one family per iteration and keep the correctness gate from the inherited L2 SOP first.

- **A. Host launch/tiling (Branch A).** Fuse adjacent elementwise stages, increase work per launch, remove per-iteration `.item()`/CPU round-trips and host synchronizations, reduce Python dispatch, or move the fixed-shape path to a compiled/graph execution. Check task count and gap share before and after. Vectorizing a kernel that is not running long enough is not a fix.
- **B. AIV input-memory (MTE2).** Increase the UB tile or `BLOCK` only while it fits UB; add double buffering/`num_stages`; reuse one loaded tile across multiple reductions; coalesce rows; align the row stride to 32 B; remove a redundant cast copy that doubles GM traffic. Verify that the larger tile does not reduce occupancy or spill.
- **C. AIV output-memory (MTE3).** Fuse the epilogue, cast at the store, write the final tensor once, and remove intermediate GM tensors. A separate cast kernel that reads and writes the full tensor is a memory-bound stage, not free.
- **D. Vector ALU.** Vectorize the tail with masks, remove scalar inner loops and data-dependent branches, use vector reductions (`tl.sum`, `tl.max`) with the correct identity, and keep `float32` accumulation. Do not improve vector time by dropping accumulator precision.
- **E. Cube MAC.** Re-tile M/N/K, check dtype and layout, reuse L1 tiles, and keep 512 B L1 alignment. A cube-bound kernel is not fixed by adding more vector work.
- **F. Cube starvation.** Prefetch weights/activations, use the native NZ layout where required, increase L1 reuse, and remove redundant GM-to-L1 copies. Check `aic_mte2_time` before and after; a higher MAC time with lower wall time can still be a win if starvation fell.
- **G. Scalar issue.** Replace per-element conditionals with masks, unroll fixed loops, and move index arithmetic out of the inner loop. Recheck `aiv_vec_time`/`aic_mac_time` so the scalar reduction did not become a different bottleneck.
- **H. Redundant cast.** Count `Cast`/`TensorMove` tasks and the bytes they move. Remove duplicate `bf16 → fp32 → bf16` conversions by fusing the cast into the producer or consumer; keep exactly the `float32` accumulation the numerical contract requires. Measure the removed task time and GM traffic, not just the cast op count.
- **I. Mixed profile.** Change the leading family only, then re-profile. If the top two families are within the mixed margin, a small change may move the label without improving wall time; require the end-to-end gate.

## Stage 4 — Convergence loop

For each candidate, run the fixed order: correctness gate, synchronized timing, profile, attribution.

Convergence requires all of:

1. The correctness gate passes on every recorded case, including tail and dtype cases.
2. The candidate beats the matched baseline by more than the recorded noise envelope in two independent runs.
3. The targeted stage share or family total falls, or the end-to-end gain is explained by a measured reduction in that stage.
4. No new dominant stage appears that erases the gain.
5. The change is a single named factor, so a later leave-one-out comparison is possible.

Stop and retain when criteria 1–5 hold. Stop and revert when the change is inside noise, regresses, or moves the bottleneck without an end-to-end gain. After three failed iterations on the same family, stop tuning that family and escalate.

Do not chain unrelated changes to force a win. Do not prune the correctness prerequisites. Do not report a per-kernel improvement as an operator speedup, or an operator speedup as a model speedup.

## Stage 5 — Escalation and exit

Escalate to a structural change when the profile is stable and the leading family is already at its practical limit: for example, a transposition or layout change, a different reduction algorithm, a fused kernel, or a different precision contract that the task actually permits. Escalate to the measurement owner when the pipeline signal is missing, the clock or device changed, the baseline is not reproducible, or the workload is not the same shape.

Exit with a record: source hashes, baseline/candidate identities, stage totals before and after, the named factor, correctness artifacts, timing blocks, profiler path, and the disposition (`retain`, `revert`, `escalate`, or `unmeasured`). A retained result is local to the recorded SOC, dtype, shape, stack and clock; other conditions need their own baseline and convergence run.

## Claim boundaries

`msprof` stage time attributes work inside a kernel; it does not prove end-to-end causality. A leading MTE2 stage can coexist with a host-side bubble, and a mixed profile can hide a serialization. The inherited L1 alignment and measurement contracts remain in force. The workflow reduces the chance of optimizing the wrong resource; it does not convert an unmeasured hypothesis into evidence.
