---
category: mutator
conditions:
  dtype:
  - bfloat16
  - float16
  - float32
  - fp8_e4m3fn
  - fp4_e2m1
  op_family:
  - grouped_matmul
  - moe
  - swiglu
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: moe/grouped-matmul
evidence_strength: reproduced_empirical
id: unit-moe-grouped-matmul-swiglu-finalize-and-dynamic-shape-selection-mutator
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-ascend-grouped-matmul-swiglu-and-mc2-fusion-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/moe_shape_selection/tokens_per_expert_drives_l1_bandwidth.md
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/grouped_matmul/group_matmul.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/grouped_matmul/README.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/gmm/grouped_matmul_finalize_routing_950.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/contrib/ascendc/optimizations/moe_shape_selection/tokens_per_expert_drives_l1_bandwidth.md
  source_type: reproduced_empirical_and_source_closed_optimization
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L4
title: MoE GroupedMatmul DSL, SwiGLU/FinalizeRouting MXA8W4 VCV resplit and tokens-per-expert
  L1 bandwidth mutator
triggers:
  error_signatures:
  - aic_l1_read_bw in fused MoE GMM is low and the engineer attempts to retune saturated
    L1/L0 tiles instead of diagnosing low tokens_per_expert = BS * topk / (epWorldSize
    * expertPerRank)
  - GroupedMatmul host wrapper pads or concatenates TensorList inputs on host before
    launch instead of slicing dynamic group boundaries in-kernel with tile_view
  - mixed MX-A8W4 (FP8_E4M3FN x FP4_E2M1) GroupedMatmulSwigluQuant or FinalizeRouting
    fails or stalls because FP4 weights are not unpacked in AIV UB->L1 before Cube
    MMA
  - GroupedMatmulFinalizeRouting overwrites sharedExpert output or suffers L1 overflow
    when bias is present and kbL1Size is not reduced from 512 to 256 at mL1Size >
    240
  keywords:
  - tokens_per_expert
  - aic_l1_read_bw
  - group_matmul
  - GroupedMatmulSwigluQuantV2
  - GroupedMatmulFinalizeRoutingV3
  - MXA8W4_NZNK
  - VCV resplit
  - tile_view
  - kbL1Size
  - dualDst Fixpipe
---
# MoE GroupedMatmul DSL, SwiGLU/FinalizeRouting MXA8W4 VCV resplit and tokens-per-expert L1 bandwidth mutator

## Applicability

Use when optimizing or diagnosing MoE `GroupedMatmul` (`group_type in {-1, 0, 2}`), `GroupedMatmulSwigluQuantV2` / `WeightNzV2`, or `GroupedMatmulFinalizeRoutingV3` across Ascend 910B (A2), 910_93 (A3), and 950 (A5).

## Signals

- **Signal 1 (`aic_l1_read_bw` / `aic_mac_ratio` drop on MoE GMM)**: On `arch22` (A3) fused MoE (`L1TileShape::M = 128`, `N = 256`, `K0 = 64`), `aic_l1_read_bw` and `aic_mac_ratio` appear low even though `L0C` (`128 × 256 = 32,768` elements) and `L0B` are already 100% saturated by the vendor tile constants.
- **Signal 2 (Host padding / TensorList materialization overhead in DSL GMM)**: Host code pads each expert's token count `m_i` to 16/128 multiples or calls `torch.cat`/`contiguous` on `TensorList` inputs before launching `group_matmul`.
- **Signal 3 (Mixed `MX-A8W4` `FP8_E4M3FN × FP4_E2M1` routing stall or L1 overflow on Ascend 950)**: `aclnn_grouped_matmul_swiglu_quant_weight_nz_v2` or `aclnn_grouped_matmul_finalize_routing_v3` is invoked with `x = FP8_E4M3FN`, `w = FP4_E2M1` (`5-D FRACTAL_NZ / C0_32`), and `E8M0` scales.

## Transformation

Apply the three orthogonal transformations according to the observed bottleneck signal:

1. **`diagnose_and_shift_tokens_per_expert_before_retiling`** —
   - **Rule**: On `arch22` (`Ascend910_93`) BF16 fused MoE where `L1TileShape::M = 128`, `aic_l1_read_bw` and `aic_mac_ratio` are **strictly monotonic in `tok/expert = BS * topk / (epWorldSize * expertPerRank)`**, NOT in tile fill ratio `fill = (tok/expert) / (ceil(tok/expert / 128) * 128)` or integer `m_tiles`:
     - ABBA controlled measurement: Arm A (`tok/expert = 96`, `fill = 0.750`, `m_tiles = 1`) -> `aic_mac_ratio = 0.4560`; Arm B (`tok/expert = 128`, `fill = 1.000`, `m_tiles = 1`) -> `0.5290`; Arm C (`tok/expert = 160`, `fill = 0.625` lowest fill, `m_tiles = 2`) -> **`0.5770` (highest MAC ratio)**.
     - Across hidden sizes `H = 1024` and `H = 3584` (3.5× apart) at constant `fill`, doubling `tok/expert` from `128` to `256` increases `aic_l1_read_bw` by **`+12.46 pp`** and **`+12.40 pp`** (reproducible within `0.06 pp`).
   - **Action**: Do not waste cycles altering `L1`/`L0` tile sizes when `L0C` (`128×256`) and `L0B` (`K0=64`) are already at hardware ceiling; adjust the deployment batch/EP sharding ratio (`BS / (epWorldSize * expertPerRank)`) so `tok/expert >= 160..256`.

2. **`in_kernel_tile_view_zero_pad_and_adaptive_sliding_window_gmm` (`cannbot-dsl/samples/grouped_matmul/group_matmul.py`)** —
   - Pass `x` and `weight` `TensorList` pointers directly to `GmmKernel` without host-side padding or `cat`/`stack` materialization across all 6 scenarios (`S1: group_type=-1`, `S2..S4: SPLIT_M group_type=0`, `S5..S6: SPLIT_K group_type=2`).
   - Inside `GmmKernel`, decode `group_list` (`type 0` cumsum or `type 1` count; skip `m_g == 0` or `k_g == 0` groups), slice the per-group tensor view to exact `[m_g, k_g, n_g]` bounds, and use `pto.tile_view` on boundary tiles so `DataCopyPad` / `Fixpipe` clip strictly to the dynamic group boundary.
   - Schedule blocks via adaptive sliding-window core assignment with **even-row N-direction snake reversal** to maximize L2 weight hit rate, and toggle L2 cache hints per group based on tail group size.

3. **`mxa8w4_vcv_weight_unpack_and_dual_dst_fixpipe_resplit` (`grouped_matmul_swiglu_quant_weight_nz_v2` R-004 & `GroupedMatmulFinalizeRoutingV3` R-003/R-004 on Ascend 950)** —
   - **Vector-Cube-Vector (VCV) `MXA8W4_NZNK` pipeline**: Because Cube cannot directly multiply `FP8_E4M3FN` activations by packed `FP4_E2M1` weights without unpacking, dispatch `sizeof(x) != sizeof(w)` to the `WeightQuantResplitController` (`MIX_AIC_1_2`, requiring `transposeWeight = true`, 5-D `NZ/C0_32` storage `(K/C0, N/16, 16, C0)`, and `K, N >= 32` aligned to 32; `N % 128 == 0` for SwiGLU):
     - **Stage V0 (AIV unpack)**: AIV loads packed `FP4_E2M1` weights from GM into UB, unpacks nibbles into `FP8_E4M3FN` in `__simd_vf__` registers (`gmm_fr_mx_a8w4_vf.h`), writes unpacked `FP8` weights into `L1` (`PIPE_MTE3`), and signals AIC via cross-core flags `8/24` and `9/25`.
     - **Dynamic L1 K-tile sizing with bias awareness**: Set `M_L1 = N_L1 = 256`, `mL1Size = ceil(m_e / ceil(m_e / 256))`, `nL1Size = min(256, N - nOffset)`. When `nL1Size <= 128` and `mL1Size <= 256` (no bias, `R-003`) or **`mL1Size <= 240` (with BF16 bias occupying `8 KiB` L1 in `R-004`)**, double `kbL1Size` from `256` to **`512`**, and set `kaDepth = min(ceil(nL1Size / (align16(mL1Size) * 2)), A_L1_bytes / (align16(mL1Size) * kbL1Size))`.
     - **Stage C -> V1 (Dual-Destination Fixpipe handoff)**: AIC executes MX MMA (`mx_m16n16k64` / `k32` with `E8M0` scales in `[E, N, ceil(K/64), 2]`) into `L0C` and issues a **dual-destination Fixpipe** transfer (`flags 4/20, 3`) that splits each M-tile in half across the two paired AIV sub-blocks (`AIV0` and `AIV1`).
     - **Stage V1 Epilogue**:
       - In `GroupedMatmulSwigluQuant`: AIV computes `SwiGLU(C[:, N/2:N], C[:, 0:N/2], beta=1)` and online 64-element shared-exponent `E8M0` MX quantization to `Y [M, N/2]` and `YScale [M, ceil((N/2)/64), 2]`.
       - In `GroupedMatmulFinalizeRouting`: Before GMM (`SyncAll`), AIV initializes `out [batch, N]` (`FP32`) by zeroing non-shared regions and writing `float(sharedInput) * sharedInputWeight` into `[sharedInputOffset : sharedInputOffset + sharedLen, N]`; after Fixpipe handoff, AIV multiplies valid rows by `logit[r]` (skipping negative `rowIndex[r]`) and accumulates into `out[rowIndex[r], :]` via hardware `FP32` `AtomicAdd`.

## Fallback boundary

- On Ascend 910B / 910_93 (`dav-c220`), `MX-A8W4` hardware `E8M0` MMA and `UB -> L1` / `L0C -> UB` paths are unavailable; use `W4A8` or `W4A16` N-dim packed `int4` antiquant routes with GM staging.
- Never extrapolate `tok/expert` `aic_l1_read_bw` gains directly as wall-clock speedup without synchronized end-to-end timing.

## Verification and ablation

1. Verify `group_matmul.py` across `S1..S6` with zero-size expert groups (`m_g = 0`) and unaligned `M_e` tails against a per-group `torch.matmul` FP32 reference.
2. On Ascend 950, verify `grouped_matmul_swiglu_quant_weight_nz_v2` and `GroupedMatmulFinalizeRoutingV3` on both homogeneous (`MXFP8×MXFP8`, `MXFP4×MXFP4`) and mixed `MX-A8W4` (`FP8_E4M3FN × FP4_E2M1`) inputs, including the `mL1Size in {240, 241, 256}` bias threshold boundary.
