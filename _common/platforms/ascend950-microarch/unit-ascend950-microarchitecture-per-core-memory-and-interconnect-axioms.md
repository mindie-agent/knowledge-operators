---
category: axiom
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
  - Ascend910_95
domain_path: _common/platforms/ascend950-microarch
evidence_strength: hardware_spec
id: unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950_microarchitecture.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950_per_core_memory.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950_scheduler_interconnect.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend_910_95.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/target_ascend950pr.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/common/platforms/concepts/ascend950_microarchitecture.md
  source_type: hardware_architecture_specification
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Ascend 950 (950PR/950DT / dav-c310) microarchitecture, per-core memory dual-accounting,
  NDDMA, BufferID and STARS2.0 interconnect axioms
triggers:
  error_signatures:
  - a SIMT or regbase kernel on Ascend 950 budgets 248 KB UB without subtracting the
    40 KB SIMT_UB_SIZE_BYTE DCache reservation and silently corrupts thread state
  - a pure AIV kernel on Ascend 950 attempts generic DataCopy between LocalTensor<A1>
    and LocalTensor<UB> without a Cube context and traps with aivec illegal instruction
    error 259
  - an Ascend 950 port assumes 4:2 structured sparsity acceleration from A2/A3 DAV_2201
    where DAV_3510 sets sparsity=0
  - an Ascend 950 kernel uses ToFloat<half> directly in regbase mode instead of reinterpreting
    via bfloat16_t
  - an Ascend 950 kernel treats CrossCoreSetFlag flagId 0..10 as the hardware ceiling
    rather than a per-Group API window over the STARS2.0 128K flag pool
  keywords:
  - Ascend950PR
  - Ascend950DT
  - Ascend910_95
  - dav-c310
  - DAV_3510
  - NDDMA
  - BufferID
  - get_buf
  - rel_buf
  - STARS2.0
  - HSCB
  - SIMT_UB_SIZE_BYTE
  - RegTensor
  - MaskReg
  - AddrReg
  - Aux Scalar
  - UB bank conflict
  - FixpipeC310
  - TPosition::A1
  - error 259
---
# Ascend 950 (950PR/950DT / dav-c310) microarchitecture, per-core memory dual-accounting, NDDMA, BufferID and STARS2.0 interconnect axioms

**Condition.** Target hardware is the Ascend 950 family (`Ascend950`, `Ascend950PR`, `Ascend950DT`, `Ascend910_95`, architecture `dav-c310` / `DAV_3510` / `arch35`, `__NPU_ARCH__ == 3510`, `__CCE_AICORE__ == 300`). These axioms govern chip die composition, per-core SRAM dual-accounting, 6-pipeline + dual Scalar execution, register-based SIMD/SIMT vector architecture, Cube MMAD shapes, UB↔L1 / L0C→UB tightly-coupled channels, NDDMA, BufferID synchronization, and STARS2.0 scheduling.

**Constraint.**

- **Package composition and UMA memory topology.** Each Ascend 950 package integrates **2 AI Dies + 2 IO Dies** connected via D2D Clink into a single Unified Memory Architecture (UMA) logical NPU. The AI Dies host the AI Core subsystems, L2 cache (128 MB total = 2 dies × 64 MB, 16 banks/die × 4 MB, 512 B cacheline with 4 × 128 B sectors, 5.28 TB/s read bandwidth), and Linx816 ARMv8-A AI CPUs (64 KB L1, 1 MB L2 per core, 4 MB L3 per cluster, hardware-coherent with AI Core L2). The IO Dies host UB 2.0 (72 lanes × 112 Gbps HiLink SerDes = 18 × x4 ports, 2016 GB/s bidirectional), UBoE (2 × 400 Gbps), URMA (Jetty SQE/CQE), CCU collective offload engine, PCIe 5.0 x16 (128 GB/s bidirectional), and DVPP (4× VPC + 4× JPEGE + 8× JPEGD).
- **Per-core SRAM hierarchy and UB dual-accounting contract.** Each AI Subsystem contains 1 AIC (Cube) + 2 AIV (Vector):
  - **L1 (`cbuf`)**: `512 KB` per AI Core (`32 B` DMA aligned, `512 B` fractal aligned).
  - **L0A (`ca`) / L0B (`cb`)**: `64 KB` each per AI Core (`512 B` aligned) + `4 KB` L0A/L0B MX buffer for `MXFP8`/`MXFP4` microscaling scales.
  - **L0C (`cc`)**: `256 KB` per AI Core (`512 B` aligned, doubled from `128 KB` on `dav-c220`).
  - **BT Buffer (BiasTable)**: `4 KB` (`4096 B`, `64 B` aligned; increased from `1 KB` on `dav-c220`) and **FP Buffer (FixPipe)**: `7 KB` (`128 B` aligned, 1024 entries for quant/dequant/bias/relu parameters).
  - **SSBuffer (Shared Scalar Buffer)**: `3 KB` per AI Core for direct intra-core AIC↔AIV scalar communication without GM round-trips.
  - **UB (`ub`) dual-accounting**: The whitepaper's `512 KB / AI Core` is the sum of two physical `256 KB` UB banks (`UB0` and `UB1`, one per AIV). Per AIV, physical capacity is `256 KB` (`262,144 B`), CANN platform config (`ub_size`) exposes **`248 KB` (`253,952 B`)** after an `8 KB` compiler reserve, and **SIMT / L3 regbase kernels reserve an additional `40 KB` (`SIMT_UB_SIZE_BYTE = 40,960 B`)** at the top of UB for SIMT DCache (`32–128 KB` configurable, `128 B` cacheline) and thread state, leaving **`~208 KB` (`212,992 B`)** effective UB. Always query `GetCoreMemSize(UB)` and subtract `40 KB` when SIMT/DCache is active.
- **Six execution pipelines plus dual independent Scalar units.** Each AI Core exposes 6 asynchronous pipelines (`PIPE_M` / CUBE, `PIPE_V` / VEC, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_FIX`) dispatched by an out-of-order **Scalar Unit** (32 KB I-cache on AIC / 16 KB on AIV, 32 KB D-cache, 5 instructions/cycle issue). In Regbase mode, an **Aux Scalar Unit** separately executes scalar instructions inside `__simd_vf__` functions while the main Scalar Unit handles scalar control outside `__simd_vf__`. Apparent Scalar-Vector serialization in `msprof` is caused by `GetValue()` RAW dependencies (`MTE2_S` / `S_V` / `V_S`), not hardware contention; prefer vector-pipe scalar broadcast (`Duplicate` or UB-resident scalar operand) over `GetValue()` detours.
- **Double-issue Register-Based SIMD and SIMT/SIMD hybrid execution.**
  - **SIMD Register File (`__simd_vf__`)**: Operands follow `GM -> UB (MTE2) -> RegTensor (VL = 256 B) -> VEC ALU (256 B × 2 double-issue) -> RegTensor -> UB -> GM (MTE3)`, with `MaskReg` (`VL / 8 = 32 B`), `AddrReg`, and `UnalignRegForLoad` / `UnalignRegForStore`. Keeping multi-step intermediates in `RegTensor` avoids UB read/write traffic and UB bank conflicts.
  - **UB multi-port bank group**: Each UB bank group has **2 read ports + 2 write ports**, supporting `2R0W` or `1R1W` per cycle (vs `1R` or `1W` on `dav-c220`). Dense sequential access (`stride = 1`) is `~22%` faster than strided access, while stride modulo-4 penalties are absorbed to `<= 6%`.
  - **SIMT warp model**: 4 warp schedulers per AIV, warp size `32`, max `2048` threads/AIV (`LAUNCH_BOUND` 512/1024/2048), `128 KB` register file per AIV (`32,768` × 4 B registers: `64 regs/thread` at 512 threads, `32 regs/thread` at 1024 threads, `16 regs/thread` at 2048 threads). SIMD↔SIMT mode switching occurs across or within VF boundaries by exchanging data in UB (`TQue::AllocTensor` -> `GetPhyAddr()` -> `Simt::VF_CALL`).
  - **Regbase intrinsic restrictions**: `ToFloat<T>` on `arch35` only instantiates for `bfloat16_t`, `fp8_e5m2_t`, and `fp8_e4m3fn_t` (not `half`); use `Ops::NN::AclnnUtil::IsRegbase()` for host router dispatch and strip `#if __CCE_AICORE__ == 220` guards when porting to `arch35`.
- **Cube MMAD native shapes, numerical modes, and no 4:2 sparsity.**
  - Native per-cycle Cube MMAD `[M, K, N]` shapes: `FP32` `[16, 4, 16]`, `TF32/HF32` `[16, 8, 16]`, `FP16/BF16` `[16, 16, 16]`, `HiF8, MXFP8, FP8_E5M2, FP8_E4M3, INT8` `[16, 32, 16]` (2× FP16 throughput), and `MXFP4 (E2M1/E1M2)` `[16, 64, 16]` (4× FP16 throughput).
  - Supports **IEEE 754 mode** and **Saturation mode** (clamping NaN/Inf to `±FP_MAX` or `0`).
  - **Structured 4:2 sparsity is NOT supported (`sparsity = 0`)** on `DAV_3510` (unlike `DAV_2201` on 910B/910C where `sparsity = 1`).
- **CV direct channels (`L0C -> UB` Fixpipe & `UB <-> L1`), NDDMA, and pure-AIV L1 gating.**
  - **Fixpipe `L0C -> UB` (`MoveToUb`)**: `PIPE_FIX` transfers Cube accumulators from `L0C` directly into `UB` or `L1` with on-the-fly `FP32 -> BF16/FP16/FP8/INT8` quantization, `PRELU/RELU`, `ROW_SPLIT`/`COLUMN_SPLIT` dual-destination split, and `NZ -> ND/DN` layout conversion. Requires a mixed AIC+AIV kernel context.
  - **`UB <-> L1` (`MoveToL1` via `PIPE_MTE3`, `L1 -> UB` via `PIPE_MTE1`)**: Hardware connects `UB` and `L1` directly, but `TPosition::A1/B1` lowers via `DataCopyUB2L1Impl` only in a **Cube/Mix context**. Calling generic `DataCopy(LocalTensor<A1>, LocalTensor<UB>)` inside a pure-AIV kernel emits an invalid opcode that faults at runtime with **aivec illegal-instruction error 259 (`subErrType 0x4`)**.
  - **NDDMA**: `PIPE_MTE2` includes a hardware 5D tensor address generator and internal cache that coalesces element-level global memory reads into `128 B` AXI transactions.
- **BufferID synchronization, STARS2.0 scheduler, and far-end atomics.**
  - **BufferID (`get_buf()` / `rel_buf()`)**: Mutex-style intra-core local memory occupancy synchronization that decouples buffer lifetime from pairwise pipeline event IDs.
  - **STARS2.0 scheduler**: Hardware scheduler supporting 8 Die-affinity Groups, 16 resource pools, 2048 task streams, ns-level High-Speed Control Bus (HSCB), and **128K 1-bit / 4096 32-bit hardware sync flags** (the `CrossCoreSetFlag` `flagId 0..10` limit with max count 15 is a per-Group software API window, not the silicon flag limit).
  - **Atomics and L2 hints**: Native HBM `atomicAdd` for `FP32` (`3.3 cyc/op` low fan-in, `15.9 cyc/op` high fan-in), `FP16`, `BF16`, `S32/U32/S64/U64` with L2 Reduce Atomic coalescing (`HA.FS009.02`), plus `__ldg` / `__stg` L2 cache hints (`L2_CACHE_HINT_NOTALLOC_CLEAN`, `L2_CACHE_HINT_NORMAL_FV`, `NON_CACHEABLE`).

**Rationale.** Treating the 512 KB AI Core UB figure as per-AIV budget, omitting the 40 KB SIMT DCache reserve, or calling `DataCopy` on `TPosition::A1` in a pure-AIV kernel causes immediate out-of-bounds corruption or runtime error 259. Assuming 4:2 sparsity or `dav-c220` GM-staging on `dav-c310` either miscompiles or leaves the 1.5–2× CV direct channel unused.

**Evidence.** Derived from the official Ascend 950 NPU Architecture Whitepaper (`ece3405e6a17fabdd462338fb94266558649a6407a2f28008403211387b3a927`), CANN 9.0/9.1 `Ascend950DT_9582.ini` / `NPUTargetSpec.td`, and verified `dav-c310` hardware probes in `https://gitcode.com/cann/cannbot-knowledge/tree/be5882ef29fa9943305085652d2025a038ba491c`.
