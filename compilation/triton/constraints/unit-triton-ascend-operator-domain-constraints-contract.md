---
category: axiom
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/constraints
evidence_strength: software_contract
id: unit-triton-ascend-operator-domain-constraints-contract
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/attention_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/mla_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/paged_attention_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/matrix_compute_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/normalization_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/quantization_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/index_computation_constraints.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/constraints/elementwise_constraints.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/guides/constraints/attention_constraints.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Triton-Ascend Operator Domain Constraints Contract (Attention, MLA, MatMul,
  Norm, Quant, Index, Elementwise)
triggers:
  error_signatures:
  - 1. Attention Family Constraints (FlashAttention, GQA, PagedAttention, MLA)
  - 1.1 Sub-Family Routing & Core Bottleneck Philosophies
  - 2. Matrix Compute Constraints (MatMul, BMM, Grouped GEMM, Addmm)
  - 3. Normalization Constraints (`N1` to `N6`)
  keywords:
  - unit-triton-ascend-operator-domain-constraints-contract
---
# Triton-Ascend Operator Domain Constraints Contract

## 1. Attention Family Constraints (FlashAttention, GQA, PagedAttention, MLA)

### 1.1 Sub-Family Routing & Core Bottleneck Philosophies
Never mix optimization strategies across attention sub-families:
- **Standard FlashAttention / GQA (`flash_attention`, `gqa`)**: Contiguous KV; use online softmax (`m_i`, `l_i`, `acc`) in `fp32`, `tl.make_block_ptr` for DMA (`MTE2`), and early `V` load before `PV` dot so `biShengIR` triggers `auto-multi-buffer` CV pipelining.
- **Paged KV Cache Attention (`paged_attention`)**:
  - **Prefill (Compute-Bound)**: Split-Q persistent grid (`grid = (cube_core_num,)`), helper function encapsulating `early V-load -> exp -> propagate_nan -> scale/sub -> tl.dot(p, v, acc)`, and causal mask loaded via `aux_mask` table (`MTE2`) rather than arithmetic vector comparison.
  - **Large-Page Decode (Memory-Bound)**: GQA-grouped grid (`(num_kv_heads, batch_size)`) sharing each KV page across `G = H_q // H_kv` query heads (`BLOCK_M >= 16`); skip whole SWA dead-zone pages.
  - **Small-Page Batch Decode with Golden Cast Chain (Scalar-Pipe-Bound)**: Enforce `BLOCK_BS <= PAGE_SIZE` and `PAGE_SIZE % BLOCK_BS == 0` (never cross page boundaries in a single `block_ptr` tile). When the reference has explicit `einsum(...).float()` and `softmax(...).to(v.dtype)`, execute a **2-pass KV loop** inside the kernel (Pass 1 computes global `m_g, l_g`; Pass 2 recomputes `QK`, normalizes `p = (exp(s - m_g) / l_g).to(native)`, and runs `PV` dot). For `G == 1` (MHA decode), route to a pure Vector kernel (`tl.sum`) instead of padding `M=1` to `16` on Cube.
- **Multi-Head Latent Attention (`mla` — Matrix Absorption `ckv` as both K and V)**:
  - **Rule `F1` (Provable Affine Addressing > Everything)**: Fix `BLOCK_QO = 1` and group `GH` heads of the same request in one program (`rows = tl.arange(0, BLOCK_M)`, `h = h_group * GH + rows`, `q_row` scalar). Any row index derived from runtime `rows // GH` or `rows % GH` breaks affine analysis and degrades output stores into millisecond-scale elementwise scalar scatter (or hangs).
  - **Rule `F5` & `F8` (16-bit PV & LSE Base Contract)**: Try single `f16` PV dot (`p.to(tl.float16)`) first; enable `P_SPLIT` (`p_hi + p_lo` dual Cube dot) only if tight verification tolerances fail. Match the reference LSE logarithm base verbatim (`tl.log` base-$e$ vs `tl.log2` base-$2$; never multiply by `LOG2E` when reference uses natural `log`).
  - **Rule `F6` (Split-KV Reduce Loop)**: In the split-KV reduction kernel, never use 3D masked loads (causes Vector Core timeout); use a scalar loop `for s in range(num_splits):`.

---

## 2. Matrix Compute Constraints (MatMul, BMM, Grouped GEMM, Addmm)

1. **2D Tile Accumulator in `fp32`**: Initialize `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)` before the `K` loop and pass `acc` into `tl.dot(a, b, acc)`. Never allocate a 3D `[B, M, N]` tile inside a single program.
2. **2D Grid Launch Limit & Persistent L2 Swizzling**: Keep `grid = (num_tiles,)` or `(min(num_tiles, num_aicore),)` with 2D L2 swizzling (`GROUP_M = 8`) computed from `pid` inside the kernel.
3. **Pointer Base + Offset Re-Calculation**: Precompute `a_base = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak` outside the `K` loop and compute `a_ptrs = a_base + k * BLOCK_K * stride_ak` inside the loop instead of `a_ptrs += ...` (avoids long scalar dependency chains in BiSheng IR).
4. **Cube Bias Fusion**: Load `bias` **before** the `K` loop and add `tl.broadcast_to(b[None, :], (BLOCK_M, BLOCK_N))` to `acc` so `bishengir-compile` fuses bias into the Cube `FixPipe` instead of spilling to Vector Core.
5. **High-Precision `fp32` MatMul (`mm-fp32-split`)**: When native `tf32` precision on Cube is insufficient for strict `fp32` verification, decompose each `fp32` input into `hi + lo` `bf16`/`fp16` components (`x_hi = x.to(tl.bfloat16)`, `x_lo = (x - x_hi.to(tl.float32)).to(tl.bfloat16)`) and accumulate 3 or 4 Cube `tl.dot` terms in `fp32`.

---

## 3. Normalization Constraints (`N1` to `N6`)

1. **Forward Fused vs Backward Split (`N0` / `N2`)**:
   - **Forward Norm (`GroupNormSwish`, `RMSNorm`, `LayerNorm`)**: Always use a **single fused kernel** with Pass 1 (reduce `sum_acc`, `sq_acc` in `fp32` registers) followed by Pass 2 (normalize + affine + activation). Never split forward norm into two HBM kernels.
   - **Backward Multi-Granularity Norm (`AdaptiveInstanceNorm2DBackward`, `LayerNormBackward`)**: When `grad_input` requires per-`(N, C)` statistics across multiple spatial tiles while `grad_weight`/`grad_bias` require per-`C` reduction, split into **two kernels** (Kernel 1 writes `fp32` partial sums and accumulates per-channel gradients via `tl.atomic_add`; Kernel 2 reads global statistics to compute `grad_input`).
2. **Scalar Load for Per-Channel Weight/Bias (`N3`)**: Ensure `BLOCK_SIZE <= D` (spatial size per channel) so an entire block lies within one channel; load `w = tl.load(w_ptr + group_base + (d_start // D))` as a scalar. Never compute vector integer division `ch_vec = offs // D` for vector `tl.load` (degrades to 1024 scalar divisions).
3. **`tl.constexpr` Length Specialization (`N5`)**: Pass `channels_per_group`, `elems_per_group`, and `BLOCK_SIZE` as `tl.constexpr` (+38% measured speedup on `GroupNormSwish`).

---

## 4. Quantization Constraints (`Q1` to `Q8`)

1. **Multi-Kernel Host Dispatch (`L2.1`)**: Split kernels along structural axes (`static` vs `dynamic`, `group` vs `non-group`, `smooth` vs `no-smooth`) on the host; never pile runtime `if` branches inside a single monolithic quantization kernel.
2. **Vector Scale Division for Bit-Exact Dynamic Quant (`L3.2`)**: In `DynamicQuant` and `SwigluQuant`, scalar float division `max_abs / 127.0` or `int_scale / max_abs` deviates by `1–2 ULP` from PyTorch `aclnnDiv`, causing off-by-1 errors after rounding. Store `max_abs` to a buffer, reload a vector tile (`BLOCK_ROWS >= 2`), compute vector division `max_vec / 127.0`, and store `scale`.
3. **Round-Half-Away-From-Zero & Hardware Saturation (`L3.3` / `L3.4`)**:
   - Replace `tl.round` (round-half-to-even) with `q = tl.where(q >= 0, q + 0.5, q - 0.5)` followed by `al.cast(q, tl.int8, overflow_mode="saturate")`.
   - For `INT4` (`torch.quint4x2`), explicitly clamp in `int32` via `tl.minimum(tl.maximum(q.to(tl.int32), -8), 7)` before packing `((q_odd & 0x0F) << 4) | (q_even & 0x0F)`.
4. **Manual Sigmoid in `SwigluQuant` (`L3.1`)**: Always write `gate = x / (1.0 + tl.exp(-x))`; never call `tl.sigmoid(x)` in `SwigluQuant`.

---

## 5. Index, Gather, Scatter & Elementwise Activation Constraints

1. **Compile-Time `DIM` / `NDIM` Specialization (`G3`)**: For `Index`, `Gather`, `IndexPut`, and `Scatter` with `ndim <= 4`, dispatch specialized kernels per `DIM: tl.constexpr` on the host. Never decode ND coordinates with a runtime `for d in range(NDIM)` loop inside the kernel.
2. **UB Preload + `tl.gather` vs Vector `tl.atomic_add` (`G1` / `G5`)**:
   - **Read (`Index`/`Gather`)**: Align parallelism with outer dimensions, preload source rows into UB when `ROW_SIZE <= UB_LIMIT`, and extract via `tl.gather` or `al.gather_out_to_ub`. Only transpose on host if `gather_gain > 2 * transpose_cost`.
   - **Write/Accumulate (`IndexPut`/`Scatter`/`EmbeddingDenseBackward`)**: Along wide contiguous trailing dimensions (`E >= 64`), issue vector `tl.atomic_add(out_ptr + base + tl.arange(0, BLOCK_E), vals, mask=e_mask)` into an `fp32` buffer; handle `padding_idx` zeroing in a separate lightweight post-kernel.
3. **Elementwise Activations (`GELU`, `SiLU`, `SwiGLU`)**:
   - Flatten to 1D (`x.reshape(-1)`) with a dynamically clamped 1D grid-stride loop (`min(cdiv(numel, BLOCK), num_vectorcore)`).
   - **Ascend `F.gelu` Golden Contract**: `torch_npu` `F.gelu` (both `'none'` and `'tanh'`) matches the sigmoid approximation `x / (1.0 + tl.exp(-1.5957691 * 0.044715 * (x / 0.044715 + x * x * x)))`; using exact `erf` fails `F.gelu` verification on NPU.
