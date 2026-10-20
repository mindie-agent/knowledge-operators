---
category: measurement
conditions:
  dtype:
  - bfloat16
  - float16
  - float32
  - int8
  - fp8_e4m3fn
  - fp4_e2m1
  soc:
  - Ascend950
  - Ascend950DT
  - Ascend950PR
domain_path: _common/platforms/ascend950dt-observations
evidence_strength: reproduced_empirical
id: unit-ascend950dt-measured-bandwidth-and-dt-vs-pr-overrides-measurement
inherits_from:
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950dt_measured_observations.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950dt_vs_pr_overrides.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/target_ascend950dt.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/target_ascend950pr.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950dt_measured_observations.md
  source_type: empirical_hardware_measurement_and_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Ascend 950DT vs 950PR SKU overrides, platform_config INI resolution and measured
  bandwidth/compute methodology rules
triggers:
  error_signatures:
  - a roofline calculation uses a 64 MB copy bandwidth of 4874 GB/s (L2-resident artifact)
    instead of the >= 1024 MB HBM plateau (2.66-3.08 TB/s)
  - a conv2d fp32 efficiency report uses torch.matmul fp16 (430.4 TFLOP/s) or matmul
    fp32 (27.0 TFLOP/s without HF32) as the denominator instead of conv2d HF32 (176-183
    TFLOP/s)
  - a platform lookup keys on Short_SoC_version=Ascend950 instead of the full SoC_version
    (such as Ascend950DT_9582) and silently drops SKU core-count differences
  - a Sort or Reduce kernel splits <= 64 fp32 elements into multiple BlockReduceMax
    or short MrgSort calls and pays the 100-cycle setup floor per call
  keywords:
  - Ascend950DT
  - Ascend950DT_9582
  - Ascend950PR
  - platform_config
  - Short_SoC_version
  - allow_hf32
  - allow_tf32
  - HBM plateau
  - 430.4 TFLOP/s
  - MrgSort cycles
  - WholeReduceMax cycles
  - SKU table
---
# Ascend 950DT vs 950PR SKU overrides, platform_config INI resolution and measured bandwidth/compute methodology rules

**Condition.** A performance model, roofline calculator, or operator tuner runs on Ascend 950DT or Ascend 950PR (`Ascend950DT_*` / `Ascend950PR_*`, CANN 9.0/9.1) and needs SKU-accurate peak compute, HBM bandwidth, or primitive instruction cycle baselines.

**Constraint.**

- **950DT vs 950PR architectural identity and memory overrides.** Ascend 950DT (Decode/Training, 4 on-package memory modules) and Ascend 950PR (Prefill/Recommendation, 8 on-package memory modules) share the exact same `DAV_3510` AI Die microarchitecture and differ only in on-package memory configuration and SKU core harvesting:

| Parameter | Ascend 950DT (36 / 32 / 28 AIC SKUs) | Ascend 950PR (32 / 28 AIC SKUs) | Evidence Layer |
| --- | --- | --- | --- |
| On-package HBM capacity | Up to `144 GB` (`96 GB` SKU tier) | Up to `128 GB` (`112 GB` SKU tier) | `[SPEC p13 Table 3-1]` |
| On-package HBM spec bandwidth | `4.0 TB/s` (4 memory modules) | `1.6 TB/s` (`1.4 TB/s` SKU tier, 8 modules) | `[SPEC p13 Table 3-1]` |
| AI Core (AIC) count | `36 / 32 / 28` | `32 / 28` | `[SPEC p13 Table 3-1]` |
| Vector Core (AIV) count | `72 / 64 / 56` (`2 × AIC`) | `64 / 56` (`2 × AIC`) | `[SPEC p13 Table 3-1]` |
| L2 Cache capacity | `128 MB` | `128 MB` (`112 MB` SKU tier) | `[SPEC p15 Table 3-1]` |
| AI CPU (Linx816) | `8C16T / 6C12T` | `8C16T / 6C12T / 4C8T` | `[SPEC p14 Table 3-1]` |
| MXFP4 TFLOP/s (total / Cube) | `2007 / 1784 / 1561` (`1946 / 1730 / 1513` Cube) | `1784 / 1561` (`1730 / 1513` Cube) | `[SPEC p13-14]` |
| FP8 / HiF8 / INT8 TFLOP/s | `1034 / 919 / 804` (`973 / 865 / 756` Cube) | `919 / 804` (`865 / 756` Cube) | `[SPEC p13-14]` |
| BF16 / FP16 TFLOP/s | `547 / 486 / 425` (`486 / 432 / 378` Cube; `60 / 54 / 47` Vec) | `486 / 425` (`432 / 378` Cube; `54 / 47` Vec) | `[SPEC p13-14]` |
| TF32 (HF32) / FP32 Vec TFLOP/s | `273 / 243 / 212` (`243 / 216 / 189` Cube; `30 / 27 / 23` FP32 Vec) | `243 / 212` (`216 / 189` Cube; `27 / 23` FP32 Vec) | `[SPEC p13-14]` |

- **Mandatory `platform_config` resolution by full `SoC_version`.** Never infer core counts from `"950DT"` vs `"950PR"` or from `Short_SoC_version=Ascend950`. CANN 9.1.0 ships 68 `.ini` files under `aarch64-linux/data/platform_config/`, including 34 Ascend 950 variants (`25` `Ascend950DT_*` and `9` `Ascend950PR_*`), all 34 of which define `ai_core_cnt`, `cube_core_cnt`, `vector_core_cnt`, `l1_size`, `l2_size`, and `ub_size=253952`. For example, `Ascend950DT_9582.ini` (`sha256: 43883e3c7224b940bbc04bd7289f78ce01fa5500916d3656ae8ae84910a41e3b`) reports `ai_core_cnt=32`, `cube_core_cnt=32`, `vector_core_cnt=64`, `l1_size=524288` (512 KB), `l2_size=134217728` (128 MB), and `ub_size=253952` (248 KB).
- **Measured empirical baselines on `Ascend950DT_9582` and measurement guardrails.**
  - **`matmul` FP16 / BF16 (`8192³`, 30-run median)**: **`430.4 TFLOP/s`** (`2.555 ms`), matching the 28/32-core Cube envelope.
  - **`matmul` FP32 (`8192³`)**: **`27.0 TFLOP/s`** (`torch.npu.matmul.allow_tf32` reads `None` and does not enable HF32 on this build).
  - **`conv2d` FP16 (`1×1` large shape)**: **`353–364 TFLOP/s`**; **`conv2d` FP32 with `torch.npu.conv.allow_hf32=True`**: **`176–183 TFLOP/s`** vs **`26.9 TFLOP/s`** with `allow_hf32=False` (`6.70×` speedup). Never use `torch.matmul` FP16 (`430.4 TFLOP/s`) or `torch.matmul` FP32 (`27.0 TFLOP/s`) as the roofline ceiling for `conv2d`.
  - **HBM sustained bandwidth (`>= 1024 MB` buffers)**: **`2.66–3.08 TB/s`** (`copy` / `triad` / `read`) on `Ascend950DT_9582`, yielding an FP16 matmul ridge point of **`140–162 FLOP/byte`** (whereas `Ascend950PR` measures `~1.1 TB/s` sustained on its `1.6 TB/s` spec HBM). Never cite small-buffer bandwidth (`64 MB` copy measures `4874 GB/s`, `~1.8×` above HBM plateau due to `128 MB` L2 residency) as HBM bandwidth.
- **Measured Sort / Reduce and MTE2 instruction characteristics (`dav-c310`).**
  - **Per-AIV `DataCopy` MTE2 concurrency**: Despite `2 × 128 B` read AXI interfaces per AI Core, issuing two concurrent `DataCopy` streams within a single AIV yields only **`1.106×` (`~10%`)** speedup because the dual read channels are shared across the AIC+2×AIV subsystem; `128-bit` (`vec4`) SIMT loads are **`2.1×`** faster than `32-bit` loads (`64-bit` is `1.4×`).
  - **`WholeReduceMax<float>` vs `BlockReduceMax<float>` (64 elements, 4096× loop-amplified)**: `232.30 µs` (`~102 cyc/call`, `0.63 elts/cyc`) vs `225.18 µs` (`~99 cyc/call`, `0.65 elts/cyc`). Both are dominated by a **`~100-cycle` setup + writeback floor**; never tile reductions of `<= 64` `fp32` elements across multiple calls.
  - **`MrgSort` (`4` pre-sorted queues)**: `q=64` (256 output pairs) takes `~182 cyc/call` (`1.41 elts/cyc`), `q=256` (1024 pairs) takes `~601 cyc/call` (`1.70 elts/cyc`), and `q=1024` (4096 pairs) takes `~2276 cyc/call` (`1.80 elts/cyc`), following the linear cost model **`~0.55 cyc × output_pairs + ~42 cyc`**. Maximize queue length per `MrgSort` call and minimize call count.

**Rationale.** Using L2-resident 64 MB copy numbers (`4.87 TB/s`) or unverified roadmap figures (`4.0 TB/s`) as achieved HBM bandwidth distorts roof-line bottleneck attribution by `1.5–1.8×`. Similarly, assuming `MrgSort` scales logarithmically per call or that `Short_SoC_version` uniquely identifies core counts leads to sub-optimal tiling and wrong launch extents.

**Evidence.** Reproduced measurements on `Ascend950DT_9582` and `Ascend950PR_9589` (CANN 9.0/9.1) and specification tables in `https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950dt_measured_observations.md` and `ascend950dt_vs_pr_overrides.md`.
