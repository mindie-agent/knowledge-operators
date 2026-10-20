---
schema: mindie-graded-unit/1
id: unit-triton-ascend-mfu-roofline-and-camodel-simulator-sop
revision: 1
tier: L2
category: sop
domain_path: compilation/triton-mfu
evidence_strength: software_contract
title: 'Triton-Ascend MFU roofline breakdown, 2D max-shape search and CAModel simulator instruction-level diagnosis SOP'
conditions:
  kernel_language: triton
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-triton-ascend-hivm-lowering-grid-overflow-and-intrinsics-sop
provenance:
  source_type: public_software_contract_review
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/triton-mfu-analyzer/SKILL.md
  source_revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  supporting_source_revisions:
  - be5882ef29fa9943305085652d2025a038ba491c
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/triton-mfu-analyzer/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/triton-simulator-optimizer/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/triton-simulator-optimizer/references/bottleneck-diagnosis.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/triton-simulator-optimizer/references/msprof-simulator.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/tooling/mfu_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/tooling/mfu_max_shape_search.md
  supersedes: []
triggers:
  error_signatures:
  - a Triton-Ascend fp32 case reports ~10% MFU because the denominator used the fp16/bf16 nominal Cube peak
  - scaling to the largest shape lowers MFU below the unscaled baseline or silently runs the unscaled JSON file
  - a sparse or sliding-window reference traced with FlopCounterMode reports MFU > 100% because dense masked FLOPs inflated the numerator
  - an operator is claimed to be fp32 dot Cube-bound without simulator evidence while MMAD share is < 5% and WAIT_FLAG_DEVI is > 50%
  - msprof op simulator hangs on a large shape or atomic_add, or exits with code 134 after writing CSVs
  keywords:
  - Triton-Ascend
  - MFU
  - operator_mfu
  - FlopCounterMode
  - aic_cube_fops
  - MFU@nom
  - MFU@ach
  - msprof op simulator
  - instr_exe.csv
  - code_exe.csv
  - WAIT_FLAG_DEVI
  - MMAD
  - fixp_l0c2ub
  - MIX_AIC
---

# Triton-Ascend MFU roofline breakdown, 2D max-shape search and CAModel simulator instruction-level diagnosis SOP

## Use when

A Triton-Ascend operator on Ascend 910B, 910_93 or 950 requires (1) single-operator Model FLOPs Utilization (`operator_mfu`) measurement or upper-bound `(case, scale)` probing on NPU hardware, or (2) instruction-level bottleneck attribution via `msprof op simulator` before claiming a hardware limit or selecting a kernel optimization.

## 1. Define the semantic FLOPs numerator and dtype-calibrated denominator

Single-operator MFU follows the `msprof-analyze` `operator_mfu` contract:

$$\text{MFU} = \frac{\text{FLOPs}_{\text{semantic}}}{\text{kernelDuration}(\text{s}) \times \text{PeakFLOPS}_{\text{dtype}}}$$

1. **Semantic FLOPs numerator**: reference `FlopCounterMode` tracing and hardware PMU counters (`aic_cube_fops + aiv_vector_fops` from `--aic-metrics=ArithmeticUtilization`) have opposite biases:
   - Reference `FlopCounterMode` overcounts when the PyTorch reference expresses sparse, causal, or sliding-window attention as a dense matmul followed by a mask.
   - Hardware `aic_cube_fops` overcounts when the kernel computes padded tiles or masked blocks that are later discarded.
   - Take $\text{FLOPs}_{\text{semantic}} = \min(\text{FLOPs}_{\text{ref}}, \text{FLOPs}_{\text{hw}})$. Flag any case where $\text{FLOPs}_{\text{ref}} > \sum (\text{duration}_k \times \text{aic\_mac\_ratio}_k) \times \text{PeakFLOPS}$ or $\text{FLOPs}_{\text{ref}} / \text{FLOPs}_{\text{hw}} > 1.2$. Keep Cube and Vector FLOPs separate: dividing pure Vector FLOPs (`aic_mac_ratio == 0`) by Cube peak FLOPS is dimensionally invalid.
   - Trace FLOPs with `FakeTensor` only (`MFU_FLOPS_NO_REAL=1` + `RLIMIT_AS` cap); never fall back to real CPU tensor allocation on large `Sq * Sk` intermediate attention matrices.
2. **Dtype-calibrated peak denominator (`MFU@ach` vs `MFU@nom`)**: formula-derived peak (`2 * 16^3 * ai_core_cnt * freq`, e.g., `353.89 TFLOPS` on 24-core 910B2C) reflects fp16/bf16 Cube throughput. Under fp32, Cube throughput is ~1/4 (`~85 TFLOPS` measured via `peak_calib.py` on $4096^3$ GEMM vs `~314-320 TFLOPS` for fp16/bf16). Report both nominal `MFU@nom` and dtype-calibrated `MFU@ach`.
3. **Complete denominator coverage**: for hybrid implementations mixing `@triton.jit` and `aclnn`/PyTorch ops, detect the repeating forward launch period from `Task Start Time(us)` across the last $N$ iterations rather than filtering by kernel prefix; omitting `aclnn` projections inflates MFU above the ~90.4% empirical ceiling.

## 2. Probe peak MFU via 2D `(case, scale)` grid search and five validity gates

Peak MFU is **not** MFU at the maximum shape: ~2/3 of operators lose MFU when scaled because working sets spill from L2 into `aic_mte2` HBM bandwidth walls, and different cases can take distinct code branches. Search the 2D space `(case, scale)` where `case` includes the top-8 original cases by FLOPs and `scale` includes `s = 1` plus powers of two along **`seq_len` and `batch` only** (never uniformly scale all dimensions, which explodes memory as $s^{\text{ndim}}$ and alters `head_dim`). Identify `batch` and `seq_len` by grouping `(tensor, axis)` value vectors across cases while excluding each tensor's last axis (`head_dim`).

Every scaled point must pass five ordered gates:

| Gate | Criterion | Failure mode caught |
| --- | --- | --- |
| 1. Input byte cap | `input_bytes <= max_bytes` (e.g., 10 GB) | Device OOM |
| 2. Reference acceptance | `class Model` passes `FakeTensor` trace with `FLOPs > 0` | Semantically illegal scaled shape that `ModelNew` silently runs |
| 3. Target execution | `class ModelNew` completes on NPU without trap (`507015`/`507035`) | UB overflow, MLIR compile failure, kernel assert |
| 4. Single-run latency | `wall_ms <= max_ms` (e.g., 4000 ms; revert to prior `s` on breach) | Unusable PMU collection duration (PMU inflates runtime 2x-16x) |
| 5. HBM bandwidth sanity | `input_bytes / duration <= HBM_peak` (`1.3 TB/s` on 910B2C) | Silent failure to update hardcoded `.json` case files (`model.json` vs `<Op>.json`) |

Keep `msprof` collection concurrency `<= num_devices / 4` to avoid profiler-induced duration inflation.

## 3. Decompose multi-kernel MFU and run the fp32-vs-fp16 Cube-bound test

When an operator launches multiple kernels (e.g., QKV/O projections + attention), decompose MFU per kernel part (`flops_parts`). To test whether a low-MFU kernel is genuinely Cube compute-bound:

- Compare the **absolute TFLOPS** (unnormalized by peak) of that kernel on fp32 vs fp16/bf16 inputs.
- Because fp32 Cube throughput is ~1/4 of fp16, a true Cube-bound kernel drops to ~1/4 TFLOPS in fp32. If fp32 and fp16 achieve similar absolute TFLOPS, the Cube is waiting on Vector compute, `MTE2` memory transfers, or `MIX_AIC` cross-core sync—tuning tile sizes for Cube MAC utilization yields zero gain.

## 4. Run the full-kernel CAModel simulator gate (`msprof op simulator`)

Before declaring an operator "at hardware limit" or when end-to-end latency is unexplained, run instruction-level simulation:

1. **Full-kernel coverage gate**: AST-enumerate every `@triton.jit` kernel (`enumerate_kernels.py`, excluding `ZerosLike`/`Empty`). Never diagnose from a subset of kernels.
2. **Environment and minimal-shape driver**:
   ```bash
   export SOC_VERSION=Ascend910B1
   CANN_HOME="${ASCEND_HOME_PATH:-$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort -V | tail -1)}"
   source "$CANN_HOME/set_env.sh"
   export LD_LIBRARY_PATH="$CANN_HOME/aarch64-linux/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH"
   export TRITON_DEBUG=1 TRITON_ALLWAYS_COMPILE=1 TRITON_DISABLE_LINE_INFO=0

   msprof op simulator --application="python3 prof_small.py" --output=prof_out \
     --kernel-name=<kernel_func> --launch-count=1 --soc-version=${SOC_VERSION} --timeout=60
   ```
   Because `msprof op simulator` simulates **every** kernel launched by `forward()` instruction-by-instruction, `prof_small.py` must run a **single forward call** (zero warmup) with minimal shapes (`64..128` per axis), especially when `atomic_add` is present. Judge success by whether non-empty `*_instr_exe.csv` and `*_code_exe.csv` exist under `prof_out`, not by subprocess exit code (`134` during teardown is benign once CSVs are on disk).

## 5. Jointly attribute `instr_exe.csv` and `code_exe.csv` across Cube and Vector cores

Always parse all **four** tables (`core0.cubecore0` and `core0.veccore0` `*_instr_exe.csv` + `*_code_exe.csv`):

| Rule | `instr_exe.csv` & `code_exe.csv` signature | Root cause & optimization direction |
| --- | --- | --- |
| **1. Cube waiting on Vector** | Cube `WAIT_FLAG_DEVI > 50%`, `MMAD < 5%`; top `code_exe` line has `fixp_l0c2ub` + `wait_flag` | Intra-loop `dot -> Vector transform -> dot` serial dependency. Split into staged passes or materialize in workspace to decouple Cube/MTE3. |
| **2. `i32` scalar degradation** | Vector `SCALAR > 30%`, `CMPN`/`ADD`/`LD`/`ST` `call_count == BM * BN` | `i32` index comparison (`i_idx > j_idx`) lowered as scalar loop. Cast comparison operands to `fp32` vector compare + diagonal block-skip. |
| **3. HBM memory bound** | `MTE2` (`tds_2dsync`/`td::load`) or `MTE3` (`td::store`) dominant; `MMAD` and `VECTOR` low | GM bandwidth wall. Fuse multi-pass kernels, hoist loop-invariant loads, or enlarge tiles within UB budget. |
| **4. True Cube compute bound** | Cube `MMAD > 50%` and `WAIT_FLAG_DEVI` low | Legitimate Cube saturation. Enlarge dot tiles, use native bf16/fp16 dot if tolerance permits, or skip masked blocks. |
| **5. `MIX_AIC` CV sync / `BAR` wall** | Vector `BAR` ~712 cyc/inst (vs ~150 cyc/vector op), `WAIT_FLAG_DEVI` ~3400 cyc/call, Vector ALU `< 10%` | Compiler-inserted `pipe_barrier` per vector op + CV handshake. Check $\text{cost\_per\_iter} = \text{Duration} / \lceil \text{iters} / \text{cores} \rceil$: if constant (`4..10 us`) across tile sizes, minimize KV iterations and inner-loop vector ops; never split tiny shapes across extra `MIX_AIC` kernels (~4.5 us launch floor). |
