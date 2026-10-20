---
category: sop
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/playbooks/attention-and-mla
evidence_strength: software_contract
id: unit-triton-ascend-attention-mla-and-gqa-playbooks-sop
inherits_from:
- unit-triton-ascend-operator-domain-constraints-contract
- unit-triton-ascend-hardware-and-compiler-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/flash_attention__flash_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/mla__multi_head_latent_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/paged_attention__paged_kv_cache_attention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__sparseflashattention.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/linear_recurrent__fused_recurrent_gated_delta_rule.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/guides/operator_playbooks/flash_attention__flash_attention.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Triton-Ascend Attention, MLA, Paged KV Cache, Sparse FA, and Linear Recurrent
  Playbooks SOP
triggers:
  error_signatures:
  - 1. FlashAttention & Composite MHA (`flash_attention`, `fa-mha`) Playbook
  - 1.1 High-Leverage Structural Rules (`L1.1` to `L1.16`)
  - 2. Multi-Head Latent Attention (`mla` — Matrix Absorption) Playbook
  - 2.1 Five Canonical Skeletons & Selection Criteria
  keywords:
  - unit-triton-ascend-attention-mla-and-gqa-playbooks-sop
---
# Triton-Ascend Attention, MLA, Paged KV Cache, Sparse FA, and Linear Recurrent Playbooks SOP

## 1. FlashAttention & Composite MHA (`flash_attention`, `fa-mha`) Playbook

### 1.1 High-Leverage Structural Rules (`L1.1` to `L1.16`)
1. **Pre-Transpose Projection Weights (`L1.12`, `182x` linear speedup)**: When the operator includes `nn.Linear` projections (`y = x @ W^T`, `W` shape `[out_features, in_features]`), **never** pass `W` directly with `stride(0), stride(1)` (causes strided inner-dimension gather). Pre-transpose `W_t = W.t().contiguous()` once on the host and re-tile the projection GEMM to `BLOCK_M=128, BLOCK_N=256, BLOCK_K=max(256, 512 // elem_size)` so each `A` tile row spans `>= 512B`.
2. **Hoist Input Transforms Out of the KV Loop (`L1.13` + `L1.15`)**: Hoist `RoPE` / QK-norm into a separate lightweight pre-kernel (clamped to `grid = (min(NUM_CORES, total_tasks),)` so `B * H * S >= 65536` never overflows the 1D grid limit). Once hoisted, increase the `scores` tile budget from `8192` to **`16384` elements (`128 x 128`, `64 KB` in `fp32`)**, cutting KV loop iterations and cross-core handshakes by $4\times$ (`1.72x -> 2.10x` geomean).
3. **In-Place Static Even/Odd Partner Indexing for Interleaved RoPE (`L1.10`)**: Never slice `x[..., ::2]` and `tl.reshape` inside JIT; compute static partner offsets with `offs_d = tl.arange(0, BLOCK_D)`, `is_odd = (offs_d % 2) == 1`, `partner = tl.where(is_odd, offs_d - 1, offs_d + 1)`, and `x_rot = tl.where(is_odd, x_pair, -x_pair)`.
4. **Native Dtype `tl.dot` + `fp32` Accumulator (`L1.1` & `L1.2`)**: Keep both `tl.dot(q, k_t)` operands in their native dtype (even for `fp32` inputs) and explicitly `.to(q_dtype)` after `tl.load(..., other=0.0)` so `other=0.0` does not promote one operand to `fp32`.

---

## 2. Multi-Head Latent Attention (`mla` — Matrix Absorption) Playbook

### 2.1 Five Canonical Skeletons & Selection Criteria
- **Skeleton 1 (`mla-paged` Main / Decode)**: `ckv` acts as both $K$ and $V$:
  $$O = \text{softmax}\left(\text{sm\_scale} \cdot (Q_{\text{nope}} \cdot \text{CKV}^T + Q_{\text{pe}} \cdot K_{\text{pe}}^T)\right) \cdot \text{CKV}$$
  - **Mandatory Affine Rule (`BLOCK_QO = 1`)**: Launch `grid = (num_requests, num_splits)`. Inside each program, process `GH` heads of a single query token (`rows = tl.arange(0, BLOCK_M)` where `BLOCK_M = GH >= 8`, `h = h_group * GH + rows`, `q_row` scalar). Every Q/O/LSE load and store is provably affine.
  - **In-Kernel Page Table Mapping**: Map virtual KV index `j` to physical cache row via `(pages[j // (page_size * H)] * page_size + (j // H) % page_size) * H + j % H`. Decouple `BLOCK_KV` from `page_size` and size `BLOCK_KV` to fit `< 140 KB` UB (leaving `>= 12 KB` compiler scratch margin).
- **Skeleton 2 (`mla-paged` Small-$M$ Vector Reduction Path)**: When `H == 1` or `GH == 1` (`M = 1` Cube forbidden zone), eliminate `tl.dot` completely (`grid = (total_q * H,)`) and compute QK and PV via `tl.sum(q_vec[None, :] * ckv_tile, axis=1)` and `tl.sum(p[:, None] * ckv_tile, axis=0)`. Never index `tl.dot(...)[0]`.
- **Skeleton 3 (`mla-sparse` Dual-Pool)**: Shrink KV loop bounds to active window/topk spans while keeping tail masks (`other=0.0`, score mask `-1.0e9`) for `-1` sparse slots.

---

## 3. Paged KV Cache Attention (`paged_attention`) Playbook

| Sub-Pattern | Bottleneck | Canonical Skeleton & Key Optimizations |
|---|---|---|
| **Sub-Pattern A: Batch Small-Page Decode (`BS <= 64`, Golden Cast Chain)** | `PIPE_S` per-page scalar overhead | Persistent 1D grid; **G==1 dispatches to pure Vector kernel (`+24.3%`)**, **G>=2 uses Cube `tl.dot` (`BLOCK_G=16`)**; **2-Pass KV loop** replicating `dot.to(native).to(f32)` and post-normalization `p = (exp(s - m) * inv_l).to(native)` before `PV` dot; pre-round `fp16` `scale` on host (`_scale_for_dtype`); hoist `inv_l = 1.0 / l` outside Pass 2 (`+7.1%`). |
| **Sub-Pattern B: Large-Page Decode (`SWA` / `GQA`, Clean `fp32` Golden)** | HBM bandwidth (`MTE2`) | `grid = (num_kv_heads, batch)`; load KV via `tl.make_block_ptr(..., boundary_check=(0, 1), padding_option="zero")` (`251.8 us -> 17.6 us`); skip whole pages in the SWA dead zone `[global_win, kv_len - 1 - local_win)`. |
| **Sub-Pattern C: Split-Q Persistent Prefill** | Cube `MMAD` & CV overlap | `grid = (cube_core_num,)`; load causal mask from precomputed `aux_mask` table (`MTE2`, `0.84x -> 1.00x`); wrap single-KV-block in a `@triton.jit` helper with strict op order: **early `V` load -> `QK` dot -> `exp` -> `propagate_nan` -> separated scale/sub -> 3-arg `tl.dot(p, v, acc)`** (`90.7 us -> 54.5 us`, `1.72x` overlap). |

---

## 4. Sparse FlashAttention (`SparseFlashAttention` — Gen3 Route2) Playbook

- **Bottleneck**: Sparse KV gather (`MTE2`/`MTE3`), not Cube compute. Never fuse gather and attention into a single kernel (`25–60x` degradation).
- **Two-Kernel T1-Chunked Architecture (`BLOCK_T1 = 16`)**:
  1. **`gather_kv_kernel` -> Contiguous Workspace (`pos_ws`, `k_ws`, `kr_ws`, `v_ws`)**: Vectorize gather with `BLOCK_N = 64`. When unique source rows in a tile span `< min(BS, 4)`, use **span-select deduplicated broadcast** (load contiguous `[src_lo, src_lo + BS)` window once and broadcast via `tl.where(src_safe == src_lo + i, ...)`). Clamp span window strictly to `<= 4`.
  2. **Adaptive Routing (`gathered_attn_kernel` vs `split_attn_kernel` + `split_combine_kernel`)**: Keep `q_main` (`BLOCK_D=128`) and `q_rope` (`BLOCK_DR=64`) separate without host concatenation; replace 2D boolean `tl.where` masks on softmax with a 1D `fp32` additive bias (`scores = acc * SCALE + bias[None, :]` where `bias = tl.where(row_ok, 0.0, -2.0e35)`).

---

## 5. Linear Recurrent (`FusedRecurrentGatedDeltaRule`) Playbook

- **Single Fused Persistent Kernel**: Launch `grid = (B * VH,)`, keeping the `fp32` recurrent state `S[BK, BV]` (`BK = next_pow2(K)`, `BV = next_pow2(V)`) resident in UB across `for t in range(T):`.
- **Rank-1 Delta Update**: Compute `kv = tl.sum(S * k[:, None], axis=0)`, `resid = (v - kv) * beta`, `S += k[:, None] * resid[None, :]`, and `o_t = tl.sum(S * (q * scale)[:, None], axis=0)`.
- **Critical State Writeback Rule (`L1.3`)**: Masked 2D stores of `S[BK, BV]` silently drop elements on Ascend (`20/25` case failures). Always write the final state to a **flat 1D buffer `[B * VH * BK * BV]`** (`flat_offs = offs_k[:, None] * BV + offs_v[None, :]`) and `.view(B, VH, BK, BV)[:, :, :K, :V]` on the host.
