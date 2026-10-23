---
schema: mindie-graded-unit/1
id: unit-mindie-sd-custom-sparse-attention-catlass-ffn-and-triton-ops-sop
title: MindIE-SD custom C++/AscendC/TIK/Triton/Catlass operators (RainFusion, EagleQBSA, LaPreprocess, mm_swiglu_mxquant) SOP
tier: L2
category: sop
evidence_strength: software_contract
domain_path: operators/mindie-sd/custom-ops
inherits_from:
  - unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
  - unit-triton-ascend-constexpr-and-specialization-contract
  - unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
conditions:
  cann_versions:
    - ">=8.5.0"
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
  workloads:
    - MindIE-SD and vLLM-Omni DiT custom attention, normalization, RoPE, and quantized FFN kernels
triggers:
  error_signatures:
    - "inferShape function does not exist"
    - "AI Core 507015"
    - "aclnnBlockSparseAttention"
  keywords:
    - RainFusionAttention
    - AdaBlockSparseAttention
    - EagleQuantBlockSparseAttention
    - EagleQBSA
    - EagleFfn
    - mm_swiglu_mxquant
    - LaPreprocess
    - ascend_laser_attention
    - NormRopeConcat
    - quant_four_over_six_a5
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/operator-dev/SKILL.md
  source_digest: 747d1489067e7d888c9c344a8535d6518e846d82a2bbbcd5264585158e1de186
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/operator-dev/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/sparse.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/quantization.md
  supersedes: []
revision: 1
---
## Standard Operating Procedure

1. **Enforce Custom OPP Runtime Registration Ordering (`ASCEND_CUSTOM_OPP_PATH`)**:
   - `MindIE-SD` packages compiled vendor operators under `mindiesd/ops/vendors/*` and prepends `ASCEND_CUSTOM_OPP_PATH` inside `mindiesd/env.py` on package import.
   - Always execute `import mindiesd` **before** initializing the NPU runtime or constructing any `torch.npu` tensors. Importing `mindiesd` after NPU context initialization causes CANN `aclnn` lookup to miss custom operator tiling/shape libraries and fail with `aclnnXxx ... inferShape function does not exist`.

2. **Dispatch Custom Block-Sparse & Quantized Attention Kernels**:
   - **`RainFusionAttention` (`rf_v2`) & `SparseBlockEstimate`**: Combines 3D `[t, h, w]` window token rearrangement, representative-token pooling (`SparseBlockEstimate` with `row_sparse`), first-frame sink (`first_frame_weight`), and block-sparse attention inside `csrc/ops/` and `csrc/plugin/`.
   - **`aclnnBlockSparseAttention` (`rf_v3`) & `EagleQuantBlockSparseAttention` (`EagleQBSA`)**:
     - `rf_v3` automatically probes V3 $\to$ V2 $\to$ V1 `aclnnBlockSparseAttention` entrypoints.
     - Under `precision="mix"`, `EagleQBSA` executes block-sparse attention with per-block INT8 Q/K and per-channel FP8 V inside a single fused kernel (avoiding the per-layer full-tensor rearrange/pooling overhead of external Python mask paths).
     - Under `precision="fp8"` (`RotateAttention`) and `precision="mxfp4"` (`MXAttention` on Ascend 950), apply online Hadamard rotation before FP8 or FP4 (E2M1 + E8M0 microscaling) attention execution.
   - **`LaPreprocess` (`ascend_laser_attention`)**: Pads Q, K, V sequence lengths to 256/128-token multiples and casts FP32 inputs to BF16/FP16 on device before invoking Atlas A2 Laser Attention.

3. **Integrate Catlass Fused Quantized FFN Kernels (`mm_swiglu_mxquant` / `EagleFfn`)**:
   - `mm_swiglu_mxquant` fuses `QuantMatMul (MXFP8) + SwiGLU + Output MXQuant` into a single Catlass kernel for SwiGLU/MoE DiT blocks (such as `MiniMax-H3`).
   - **Adaptive Weight Layout Caching (`AdaptLayoutCached`)**: Third-party serving engines (`vllm-omni`) store quantized weights in GEMM-ready `(K, N)` layout with `(c, N, 2)` scales and gate-first row ordering. Use C++ `AdaptLayoutCached` to detect `w.size(0) == k` and cache the transposed `w^T` once per weight tensor without runtime row-swapping.
   - **Mandatory Stream Synchronization Before Raw `aclrtLaunch`**: When `AdaptLayoutCached` or upstream preparation performs asynchronous transpose/copy operations, call `aclrtSynchronizeStream` (or bind to the exact caller ACL stream) before raw `aclrtLaunch` of the Catlass kernel; otherwise cross-stream races trigger `AI Core 507015` exceptions.

4. **Apply Ascend 950 (A5) `quant_four_over_six_a5` & Triton-Ascend Vertical Fusions**:
   - On Ascend 950, `W4A4_MXFP4_FOUROVERSIX_DYNAMIC` dispatches `quant_four_over_six_a5` to evaluate adaptive 4-vs-6 block scaling thresholds for MXFP4 dynamic quantization.
   - For Triton-Ascend vertical fusions (`NormRopeConcat`, `AdaLayerNorm`, `mmgelu`), keep tile sizes within UB capacity, use integer bitwise RNE for any intermediate BF16 rounding, and verify zero spillover in compiled `.ttir`/`.ttadapter` IR.
