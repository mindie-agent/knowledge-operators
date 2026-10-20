---
schema: mindie-graded-unit/1
id: unit-catlass-and-tilelang-ascend-operator-design-and-tuning-sop
revision: 1
tier: L2
category: sop
domain_path: matmul/catlass-tilelang
evidence_strength: software_contract
title: 'CATLASS and TileLang-Ascend design SOP: template choice, L1/L0/UB tiling, L2 swizzle, AIC-AIV epilogue and Cube utilization'
conditions:
  kernel_language:
  - catlass
  - tilelang
  - ascendc
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  source_type: public_software_contract_review
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/architecture/00-hardware-arch.md
  source_revision: 502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea
  supporting_source_revisions:
  - c22984baa55e1506eadd7c72955d9bf669b1c959
  - d2a6661365ef65ca738cb66299449b2740c857eb
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/architecture/00-hardware-arch.md
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/custom-epilogue.md
  - https://gitcode.com/cann/cannbot-skills
  - https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  - https://gitcode.com/Ascend/AscendNPU-IR
  supersedes: []
triggers:
  error_signatures:
  - a CATLASS or TileLang GEMM is slower than the aclnn baseline despite a larger tile
  - the L1/L0/UB byte budget is exceeded or a tile is silently spilled
  - a 512 B L1 descriptor or 32 B UB DMA is misaligned
  - large-N GEMM or MLA loses L2 hit rate after a row-major block schedule
  - a fused epilogue writes the whole M by N fp32 matrix to GM and reads it back
  - AIC and AIV deadlock around CrossCoreSetFlag or CrossCoreWaitFlag
  - a SwiGLU or gated epilogue is wrong only when N exceeds one L1 N-tile
  - a large speedup is claimed while aic_mac_ratio is low or correctness fails
  keywords:
  - CATLASS
  - TileLang-Ascend
  - AscendC
  - GEMM
  - Grouped GEMM
  - MLA
  - FlashAttention
  - L1
  - L0A
  - L0B
  - L0C
  - UB
  - BlockSwizzle
  - Morton
  - L2 cache
  - CrossCoreSetFlag
  - epilogue fusion
  - aic_mac_ratio
  - Ascend910B
  - Ascend910_93
  - Ascend950
---
# CATLASS and TileLang-Ascend design SOP: template choice, L1/L0/UB tiling, L2 swizzle, AIC-AIV epilogue and Cube utilization

Use this SOP for a matrix-multiply family operator (plain GEMM, grouped GEMM, quantized GEMM, FlashAttention/MLA projection, or a fused GEMM+epilogue) on Ascend 910B, 910_93 or 950. It selects the implementation surface, computes the on-chip budget, schedules blocks for L2 reuse, places the epilogue, and verifies correctness and Cube utilization. The inherited alignment axiom supplies the 32 B UB and 512 B L1 foundation for 910B/910_93; the 950 UB/L0C capacities are read from the 950 target specification. This SOP does not replace the target version's official CATLASS or TileLang documentation.

## 1. Choose CATLASS C++ templates or TileLang-Ascend

Both surfaces can produce a correct GEMM. The choice is about how much explicit control the operator needs and how quickly the shape set must be explored.

| Decision | CATLASS C++ templates | TileLang-Ascend |
| --- | --- | --- |
| Best fit | High-throughput Cube GEMM, grouped GEMM, quantized GEMM, production epilogue fusion, explicit L1/L0A/L0B/L0C ping-pong | Fast FlashAttention/MLA/GEMM prototyping, a compact Python kernel, a shape family that is still being explored |
| Control surface | `ArchTag`, `DispatchPolicy`, `BlockMmad`, `BlockEpilogue`, `BlockScheduler`, `Kernel`; explicit workspace and tile parameters | `T.Kernel`, `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_ub`, `T.copy`; Developer mode by default, Expert `T.alloc_L1/L0A/L0B/L0C` when exact placement is needed |
| Build/runtime | C++/Ascend C build, template instantiation, Host tiling and workspace ABI | `@tilelang.jit`, `test_{op}.py`, generated kernel source inspection |
| Main risk | Template/version drift, workspace and dispatch-policy mismatch, long build iteration | The automatic memory-scope inference may place a buffer differently than expected; a 1-D `T.Kernel` cannot express a 3-D grid without `block_metadata` |

CATLASS candidates from the reviewed catalog include `BasicMatmul`, `OptimizedMatmul`, `PaddingMatmul`/`PaddingSplitkMatmul`, `SplitkMatmul`, `SingleCoreSlicekMatmul`, `MatmulFullLoadA`, `SmallMatmul`, the quantized variants (`12_quant_matmul`, `30_w8a16`, `32_w4a8`, `38_w4a4`), and the activation/epilogue examples. The names are navigation aids: confirm the symbol, template parameters and Host dispatch in the actual repository version. A DispatchPolicy that is "recommended" in an example is not a universal optimum.

TileLang candidates are the GEMM, FlashAttention, MLA and pipeline examples in the target platform branch. The platform routing is strict: Ascend910B/910_93 and Ascend950 use different instruction sets and pass configurations; do not mix branches or inherit a 950 example into a 910 kernel. `T.Kernel` on the reviewed Ascend path accepts a one-dimensional block grid; express a 2-D or 3-D logical grid through a flat task index or a precomputed `block_metadata` table.

```bash
# CATLASS: inspect the actual component names in the pinned source.
rg -n "ArchTag|DispatchPolicy|BlockMmad|BlockEpilogue|BlockScheduler" ./catlass/include ./catlass/examples | head -80

# TileLang: inspect the kernel surface and memory scopes.
rg -n "T\.Kernel|alloc_shared|alloc_fragment|alloc_ub|alloc_L1|alloc_L0|use_swizzle" ./examples ./custom | head -80
```

**Invariants.** Choose one surface per operator version and record it in the design. CATLASS selects `Catlass::Arch::AtlasA2` for 910B/910_93 and `Catlass::Arch::Ascend950` for the 950 family; do not mix a 910 CATLASS `ArchTag` with a 950 `CATLASS_ARCH`, do not mix `Ascend910` and `Ascend950` TileLang branches, and do not treat a Python prototype's performance as the production kernel's performance without re-running the target branch.

**Verification.** The selected surface must compile the full workload shape set, not one demo shape. For CATLASS, confirm the `DispatchPolicy` and `Kernel` actually call the selected `BlockEpilogue`; a template that compiles without invoking the epilogue is a plain GEMM. For TileLang, dump `get_kernel_source()` and confirm the expected L1/L0/UB scopes and the DMA path; a Python-level `T.copy` is not proof of a single large block DMA.

## 2. Size the L1/L0A/L0B/L0C/UB hierarchy from bytes, not tile names

The capacity hierarchy is independent per level:

| Level | 910B / 910_93 | 950 | Use |
| --- | ---: | ---: | --- |
| L1 | 512 KiB | 512 KiB | A/B staging and ping-pong |
| L0A | 64 KiB | 64 KiB | A micro-tile |
| L0B | 64 KiB | 64 KiB | B micro-tile |
| L0C | 128 KiB | 256 KiB | fp32/int32 accumulator and fixpipe source |
| UB | 192 KiB | 248 KiB (8 KiB reserved) | Vector epilogue, scales, bias, scratch |

Compute the budget for the actual dtype, accumulation type and buffer stages:

```text
L1  = L1_M * L1_K * sizeof(A) * stages_A
    + L1_K * L1_N * sizeof(B) * stages_B          <= 512 KiB
L0A = L0_M * L0_K * sizeof(A) * stages_A         <= 64 KiB
L0B = L0_N * L0_K * sizeof(B) * stages_B         <= 64 KiB
L0C = L0_M * L0_N * sizeof(accum) * stages_C     <= 128 KiB (A2/A3) or 256 KiB (950)
UB  = staged epilogue inputs + outputs + scratch <= 192 KiB (A2/A3) or 248 KiB (950)
```

A reviewed fp16 + fp32-accumulation starting point is `L1<128,256,256>` with two input stages and `L0<128,256,64>`: L1 is 384 KiB, L0A is 32 KiB, L0B is 64 KiB, and one fp32 L0C tile is 128 KiB. For int8 input the element width is smaller, so `L1<128,256,512>` and `L0<128,256,128>` become possible; verify the actual int8 promotion and accumulator path before adopting them. An fp16 `L1<128,256,512>` is already about 768 KiB with two stages and does not fit. `m0 = m1` and `n0 = n1` are constraints of the Common template, not of every kernel.

The 32 B and 512 B contracts apply at different levels:

- Every UB DMA base and byte count is 32 B aligned. A bf16/fp16 innermost tile is a multiple of 16 elements; fp32 is a multiple of 8.
- Every L1 tile base and descriptor stride is 512 B aligned. A `stride` that is not 512 B aligned may need padding, a different layout, or a staged copy; a source example using non-512 B stride is a performance case, not a proof of universal legality.
- On 950, a CV-fused tail has a 512 B contract in addition to the 32 B UB contract. A tile that is legal on 910B can still be illegal there.

TileLang maps the Developer abstractions to physical scopes through inference: `T.alloc_shared` is a dynamic shared scope that the compiler places in L1 or UB; `T.alloc_fragment` becomes L0A/L0B/L0C. Use `T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0A/L0B/L0C` when the placement must be exact. Regardless of the API name, compute the byte budget from the inferred or explicit shape. For a fused GEMM, add the epilogue's staged inputs, fp32 working buffers and double-buffer copies; a GEMM tile that fits L1 can still overflow UB in the epilogue.

Workspace is separate from on-chip capacity. For a multi-stage rotating epilogue, the reviewed formula is:

```text
workspace_bytes = L1_M * L1_N * aic_core_num * WORKSPACE_STAGES * sizeof(C_element)
```

`WORKSPACE_STAGES` is commonly 2. Use the kernel's actual `GetWorkspaceSize` or the equivalent formula; do not allocate `M*N*sizeof(C)` for a rotating workspace.

**Verification.** Write the byte total for every live buffer and every stage count into the design record. Compile the maximum-dtype, maximum-tile and tail case; inspect `memref.alloc` sizes in the generated IR or the TileLang kernel source. Confirm no buffer spills to GM and no UB overflow. A mathematical budget that fits but a compiler that spills is not a completed tile.

## 3. Swizzle the block schedule to keep the working set in L2

A row-major block order sends consecutive cores to far-apart columns, which increases the L2 working set and evicts the A/B tiles that the next block needs. Use the platform's block scheduler rather than a hand-written `bx = cid // n_num; by = cid % n_num` when the shape is large.

CATLASS provides the `GemmIdentityBlockSwizzle<Offset, Direction>` family (the generic block-swizzle surface). The reviewed default is `Offset = 3`, with `Direction = 0` for `M >= N` and `Direction = 1` for `M < N`. The source example selects `<3,0>` or `<3,1>` from the M/N relation; this is a candidate heuristic, not a proof that every shape improves. TileLang provides the equivalent `T.use_swizzle(idx, M, N, K, block_M, block_N, off=3)` for the flattened task index. A Morton/Z-order schedule is an alternative when the pinned platform exposes one, but it is not a drop-in replacement for the identity swizzle: verify the actual kernel mapping and the L2 working-set change before adopting it.

```python
# TileLang: keep the physical core count fixed and swizzle the logical task index.
# props comes from driver.active.utils.get_device_properties(torch.npu.current_device()).
core_num = props["num_aicore"]          # 20 or 24 on the recorded 910B/910_93 families
with T.Kernel(core_num, is_npu=True) as (cid, _):
    for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
        task = T.use_swizzle(i * core_num + cid, M, N, K, block_M, block_N, off=3)
        if task < m_num * n_num:
            bx = task // n_num
            by = task % n_num
            ...
```

```cpp
// CATLASS: confirm the selected ArchTag and DispatchPolicy in the pinned version.
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
```

The scheduler changes the order, not the tile math. Keep the same tile shape and the same core count when comparing a swizzled and a row-major arm, otherwise the L2 effect is confounded with occupancy.

**Verification.** Run an A/B on the same shape, dtype, tile and clock: row-major vs swizzled. Report `l2_cache_hit_rate`, HBM read/write traffic, kernel time and per-case block envelope. A swizzle that improves a large GEMM but regresses a small shape is a workload-boundary result, not a universal win. If the profiler cannot report L2 hit rate, record `L2 signal unavailable` and compare HBM traffic and time only.

## 4. Fuse the AIC-AIV epilogue without a whole-matrix GM round trip

A naive fused GEMM writes the full `[M,N]` fp32 accumulator to GM, launches an AIV epilogue that reads it back, and writes the final output. That costs `M*N*4` bytes of write plus `M*N*4` bytes of read, often outside L2 for a large N. The better structure is a small rotating workspace that stays L2-resident, with the AIC producing one tile and the AIV consuming that tile before it is evicted.

On 910B/910_93 there is no L0C-to-UB direct path. The Cube writes a tile to a GM workspace, signals the AIV, and the AIV loads the tile into UB, applying dequantization, activation, bias or residual, then stores the final output to GM. On 950, a fixpipe/L0C-to-UB path can carry the tile directly; use it only when the target branch and the kernel actually expose it.

```cpp
// A2/A3 sketch: the producer writes the C tile to the rotating workspace,
// then releases the consumer. The pipe argument must match the producer stage.
Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(stage_ready);
// Consumer:
Arch::CrossCoreWaitFlag(stage_ready);
```

The synchronization contract is as important as the math:

- `CrossCoreSetFlag` and `CrossCoreWaitFlag` must use the same mode ID and the same flag. A Set without a matching Wait, or a Wait on a flag that is never Set, is a deadlock.
- The flag count must be balanced across the prologue, main loop and drain. A consumer that returns early or a producer that skips an iteration leaves the token count unbalanced.
- `CrossCoreBarrier` is a core-type-domain barrier. An AIV-domain barrier cannot be joined by AIC work, and an AIC participant cannot satisfy an AIV-only barrier. Keep the two-stage reduction inside the domain that owns the barrier.
- Do not add a manual Wait on a flag that an internal `BlockMmad` already waits on. A double wait consumes the flag once and then blocks forever.
- A `HardEvent` type has a limited number of event IDs (for example, `V_MTE2` has 8 on the reviewed path). Too many `UB_STAGES` can exceed the ID budget and hang at runtime even though the code compiles. Count the IDs before raising the stage count; double-input epilogues often need `UB_STAGES=1`.

A cross-N-half gated epilogue (SwiGLU, GeGLU, ReGLU) needs special handling: the paired operand is at column `j + N/2`, which can live in a different N-block. A stock rotating workspace that stores one `[L1_M, L1_N]` C tile per slot will be correct only when `N <= L1_N`; for larger N it reads the wrong half. Schedule by the output shape and produce both C regions in one stage, or use a full-width/global source pointer and accept the extra traffic.

For TileLang CV fusion, use the platform's Developer or Expert CV pattern: `T.Scope("C"/"V")`, the matching `AUTO_CV_COMBINE`/`AUTO_CV_SYNC` pass configs, `threads=2` for the single-`cid` Developer path, or the workspace slot mode for the Expert path. Do not mix the two modes in one kernel, and do not assume a workspace slot exists when the Developer path eliminates it. For CATLASS, prefer the preload-async-with-callback policy (concatenate the two code spans `MmadAtlasA2` and `PreloadAsyncWithCallback`) with a multi-stage rotating workspace over `MatmulActivation`/`MatmulEpilogue` for a large-N production epilogue. The callback variant overlaps the epilogue with the next matmul; the example-level full-workspace kernels are correct but not the performance endpoint.

**Verification.** Test the epilogue at a shape where `N` exceeds one L1 N-tile, not only at `N <= L1_N`. For a gated epilogue, compare both halves against the fp32 reference and exercise `N` just above and just below the tile boundary. Run a deadlock timeout on the fused kernel; a hang is a flag/token failure, not a slow kernel. Check the output dtype and the quantized scale/bias order. Confirm the workspace size matches the rotating formula and that the final output is written once, not once per intermediate stage.

## 5. Verify correctness first, then Cube MAC utilization against the aclnn baseline

Correctness comes before any `aic_mac_ratio` or speedup claim. Freeze a reference from `torch_npu.npu_matmul`, `aclnnMatmul`, or a CPU fp32 implementation whose numerical order matches the operator contract. For a quantized GEMM, use an independent golden that mirrors the scale/zero-point and accumulation order; a final magnitude match does not prove an exact integer dot.

Cover at least: an aligned `M,N,K` case, a small-M case, a prime/unaligned tail, the real-network shapes, every declared dtype/transpose/layout instance, a grouped-GEMM uneven group, and a fused epilogue case with `N` larger than one L1 N-tile. Require the exact output dtype, finiteness, max-abs/relative error inside tolerance, unchanged inputs and repeatability. For MLA/FlashAttention, also check the mask identity, the online-softmax rescale and `propagate_nan` behavior against the reference.

Only after correctness passes, capture the Cube utilization:

```bash
msprof --application="python3 bench_gemm.py --m 4096 --n 8192 --k 4096 --dtype bf16" \
  --output=./prof_gemm --ai-core=on --aic-metrics=PipeUtilization --l2=on
python3 - <<'PY'
import glob
import pandas as pd

path = sorted(glob.glob("prof_gemm/**/op_summary_*.csv", recursive=True))[-1]
df = pd.read_csv(path)
keep = [c for c in [
    "op_name", "op_type", "task_id", "aicore_time(us)", "aiv_time(us)",
    "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
    "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
] if c in df]
print(path)
print(df[keep].to_string(index=False))
PY
```

`aic_mac_ratio` is the MAC/Cube duty cycle. It is the metric to use for "is the Cube busy"; `cube_utilization(%)` can be high while `aic_mac_ratio` is low, so the two names are not interchangeable. A ratio below roughly 10% with a matmul-shaped kernel indicates the Cube is mostly idle: check for a missing `tl.dot`, a row-by-row GEMV loop, a `BLOCK_M=1` schedule, or a serialized handshake. A ratio above roughly 55% indicates a compute-bound kernel; further tile changes will not create free time, and the next step is algorithm or precision work.

Read the other pipes with it:

- `aic_mte2` high: GM-to-L1 input movement; increase L1 reuse, change the layout or prefetch.
- `aic_mte1` high: L1-to-L0 movement; adjust the L0 tile or stage count.
- `aic_fixpipe` high: output/quantization movement; fuse or reduce the number of passes.
- `aiv_mte2`/`aiv_mte3` high in a fused kernel: the AIV is waiting on the workspace read/write; the epilogue is not hidden behind the Cube.
- All pipes below the threshold: dependency/latency-bound, not a single saturated resource.

PMU ratios are relative duty cycles. PMU collection can inflate kernel duration by roughly 2x to 16x, so take absolute latency from a separate run without PMU. A ratio above 1 indicates inconsistent MIX-kernel counters; record the anomaly instead of treating it as saturation.

**Verification and exit criteria.** The operator is complete only when: the correctness gate passes on every declared shape/dtype/layout; the selected template or TileLang branch actually executes the intended Cube/Vector path; the tile budget and 512 B/32 B alignment hold; the swizzle and epilogue were compared on a matched baseline; the fused path has no flag deadlock and no cross-N-half error; and `aic_mac_ratio` plus the absolute latency are reported for the same source hash and clock. A compile pass, a high `cube_utilization`, or a low kernel time with a failing correctness case is not a valid result.
