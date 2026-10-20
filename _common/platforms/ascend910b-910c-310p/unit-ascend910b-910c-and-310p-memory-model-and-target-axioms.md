---
category: axiom
conditions:
  dtype:
  - bfloat16
  - float16
  - float32
  - int8
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend310P
  - Ascend950
domain_path: _common/platforms/ascend910b-910c-310p
evidence_strength: hardware_spec
id: unit-ascend910b-910c-and-310p-memory-model-and-target-axioms
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/npu_memory_model.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend_910b.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/target_ascend910b.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/target_ascend910c.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/target_ascend310p.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/npu_memory_model.md
  source_type: hardware_architecture_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Ascend 910B (A2), 910C (A3 dual-die) and 310P (dav-m200) target platform, memory
  model and cross-generation porting axioms
triggers:
  error_signatures:
  - ub overflow, requires 3072256 bits while 1572864 bits available when compiling
    a kernel for Ascend 910B or 910C with multibuffer enabled
  - an Ascend 950 SIMT kernel using Simt::WarpShflSync or ThreadBarrier fails to compile
    on Ascend 910B/910C (arch22 / dav-c220)
  - a CV fusion kernel on Ascend 910B or 910C attempts direct L0C->UB Fixpipe or UB->L1
    MTE3 transfer without staging through GM
  - a DAV_2002 / dav-m200 macro dump is treated as unique runtime proof of Ascend310P
    execution or bfloat16 support
  keywords:
  - Ascend910B1
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_9392
  - Ascend910C
  - Ascend310P
  - dav-c220
  - dav-m200
  - DAV_2201
  - DAV_2002
  - arch22
  - multibuffer
  - InsertLoadStoreForMixCV
  - 1572864 bits
---
# Ascend 910B (A2), 910C (A3 dual-die) and 310P (dav-m200) target platform, memory model and cross-generation porting axioms

**Condition.** A kernel or compiler lowering targets Ascend 910B (`Ascend910B1`..`B4`, single-die `dav-c220` / `DAV_2201` / `arch22`), Ascend 910C (`Ascend910_9392` / `93xx`, dual-die `dav-c220` / `DAV_2201` / `arch22`), or Ascend 310P (`dav-m200` / `DAV_2002`), or migrates operators across `dav-c220` and `dav-c310`.

**Constraint.**

- **Ascend 910B (A2) and Ascend 910C (A3) shared `dav-c220` (`DAV_2201` / `arch22`) architecture.**
  - **CANN build family**: Both 910B and 910C (`Ascend910_93xx`) belong to `ascend910b_list` and compile with `BUILD_MODE=c220`, `__NPU_ARCH__ == 2201`, `__CCE_AICORE__ == 220`, and arch directory `arch22`. In CANN 9.0+, CMake must include `${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake` (with fallback to `compiler/tikcpp/...` on CANN 8.x).
  - **Compute and die scaling**:
    - **910B (single die)**: `Ascend910B2` has `24 AIC / 48 AIV` (`64 GB` HBM2e); `Ascend910B3` / `B4` has `20 AIC / 40 AIV` (`64 GB` or `32 GB` HBM3e).
    - **910C (dual 910B dies per package)**: `Ascend910_9392` (`9391/9381/9382/9372/9362`) exposes `40–48 AIC / 80–96 AIV` total across 2 dies, `128 GB` HBM total, and `~1.6 TB/s` aggregate HBM bandwidth (`~64 MB` L2 per die).
  - **SIMD-only execution contract**: `dav-c220` has **no SIMT hardware path**—`threadIdx`, `Simt::WarpShflSync`, `Simt::WarpReduceAddSync`, `ThreadBarrier()`, `ThreadFence()`, `LAUNCH_BOUND`, and SIMT DCache/shared memory do not exist. Any A5 SIMT kernel (such as per-thread scatter-add or warp reduction) must be rewritten into `TPipe`/`TQue`/`DataCopy` SIMD pipelines (e.g., sort-then-segment-reduce or UB scratchpad reduction) for A2/A3.
  - **On-chip SRAM and structured sparsity deltas vs A5 (`dav-c310`)**:
    - **UB (`ub`)**: `192 KB` (`1,572,864 bits`) per AIV (`32 B` aligned). With `multibuffer=True` (default on A2/A3 in Triton-Ascend, whereas A5 defaults to `multibuffer=False`), effective per-buffer UB is halved to **`96 KB`**; exceeding `1,572,864 bits` triggers `ub overflow, requires ... bits while 1572864 bits available!`.
    - **L1 (`cbuf`)**: `512 KB` per AIC (`32 B` aligned); **L0A (`ca`) / L0B (`cb`)**: `64 KB` each (`512 B` aligned); **L0C (`cc`)**: **`128 KB`** per AIC (`512 B` aligned, half of A5's `256 KB`).
    - **BT Buffer**: **`1 KB` (`1024 B`, `64 B` aligned)** (quarter of A5's `4 KB`); **FP Buffer**: `7 KB` (`128 B` aligned).
    - **Structured 4:2 sparsity**: **Supported (`sparsity = 1`)** on `DAV_2201` (unlike `DAV_3510` where `sparsity = 0`).
- **Data-path separation on `dav-c220` vs `dav-c310`.**
  - On `dav-c220` (910B/910C), Cube (`GM -> L1 -> L0A/L0B -> L0C -> FIX -> GM`) and Vector (`GM -> UB -> V -> UB -> GM`) are strictly separated on chip: **there is no `L0C -> UB` Fixpipe path and no `UB -> L1` MTE3 path**. Every Cube-Vector (CV) fused operator on 910B/910C must stage intermediates through GM (`InsertLoadStoreForMixCV`: `L0C -> GM -> UB` and `UB -> GM -> L1`) synchronized via FFTS cross-core flags (`mode = 2`).
  - Compiler IR differences between `dav-c220` and `dav-c310`: `dav-c220` generates `PipeBarrier` for all pipes (whereas `dav-c310` skips `PIPE_V` barriers in register mode), scalar-degrades `i64` reductions and integer `argmax`/`argmin`, and canonicalizes `vcmp(NE)` into `vnot(vcmp(EQ))`.
- **Ascend 310P (`dav-m200` / `DAV_2002`) provenance and capability boundary.**
  - **Directional identity**: `Ascend310P -> DAV_2002 -> dav-m200` (`__CCE_AICORE__ == 200`, `__DAV_M200__ == 1`, `__DAV_CUBE__ == 1`, `__DAV_VEC__ == 1`) is a one-way mapping; `DAV_2002` is shared by `Ascend310` and `Ascend310P` and is distinct from `Ascend310B` (`DAV_3002` / `dav-m300`).
  - **Architecture & dtype constraints**: On `dav-m200`, Cube and Vector share a single core and Scalar unit (`__mix__(1,2)` expands to AIC-only without separate AIC/AIV cores or cross-core synchronization), `L0C` alignment is `64 B` (`FRACTAL_NZ`), and `bfloat16_t` (`DT_BF16`) is rejected by `dav-m200` cross-compilation (`fp16` and `fp32` only). Cross-compile receipts without device execution must never be claimed as runtime or precision evidence.

**Rationale.** Porting between A2/A3 (`dav-c220`) and A5 (`dav-c310`) or 310P (`dav-m200`) fails immediately when UB (`192 KB` vs `248 KB`), L0C (`128 KB` vs `256 KB`), BT (`1 KB` vs `4 KB`), `multibuffer` defaults, or CV staging paths (`GM` round-trip vs `L0C -> UB`) are conflated.

**Evidence.** Derived from CANN `NPUTargetSpec.td`, `HIVMDMAOps.td`, `hardware_consts.py`, and platform target specifications in `https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/`.
