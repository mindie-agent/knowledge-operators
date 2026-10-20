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
domain_path: compilation/triton/playbooks/matmul-and-moe
evidence_strength: software_contract
id: unit-triton-ascend-matmul-quant-and-moe-playbooks-sop
inherits_from:
- unit-triton-ascend-operator-domain-constraints-contract
- unit-triton-ascend-hardware-and-compiler-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__groupedmatmulswigluquant.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__quantmatmul.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/quantization__weightquantbatchmatmul.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__moegatingtopksoftmax.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__moecomputeexperttokens.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/guides/operator_playbooks/matrix_compute__groupedmatmulswigluquant.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: public_software_contract_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Triton-Ascend QuantMatMul, WeightQuantBMM, GroupedMatmulSwigluQuant, and MoE
  Playbooks SOP
triggers:
  error_signatures:
  - 1. `GroupedMatmulSwigluQuant` Playbook (`1.3202x` vs PyTorch, 60/60 Pass)
  - 1.1 Persistent Two-Kernel Architecture (`K1` Cube + `K2` Vector)
  - 2. `WeightQuantBatchmatmul` Playbook (`0.9133x` vs Native, `8.90x` over Baseline)
  - 3. `QuantMatmul` (`int8 x int8`) Playbook (`0.9282x` vs `npu_quant_matmul`)
  keywords:
  - unit-triton-ascend-matmul-quant-and-moe-playbooks-sop
---
# Triton-Ascend QuantMatMul, WeightQuantBMM, GroupedMatmulSwigluQuant, and MoE Playbooks SOP

## 1. `GroupedMatmulSwigluQuant` Playbook (`1.3202x` vs PyTorch, 60/60 Pass)

### 1.1 Persistent Two-Kernel Architecture (`K1` Cube + `K2` Vector)
1. **Single Persistent Launch Across Experts (`L1.1`–`L1.4`, `+37.4%` primary lever)**:
   - Never loop `for e in range(E): kernel[grid](...)` or call `group_list[e].item()` per expert.
   - Copy `gl_cpu = group_list.cpu().tolist()` once (`1` D2H sync), filter empty experts (`E_eff`), and pack all per-expert metadata into a **single `int64` tensor** (`1` H2D upload):
     `meta = [w_delta(E_eff) | wsd_delta(E_eff) | tile_lo(E_eff + 1) | row_lo(E_eff) | row_hi(E_eff)]`
     Note that `tile_lo` has `E_eff + 1` elements (including sentinel `total_tiles`), so `row_lo` starts at offset `3 * E_eff + 1` and `row_hi` at `4 * E_eff + 1`. Convert pointer deltas to **element units** (`(ptr_e - base_ptr) // elem_size`).
2. **`K1` (`gmm_dequant_persistent_kernel` on Cube)**:
   - Global tile loop `for tile_idx in range(pid, total_tiles, NUM_CUBE_CORES):` with unrolled `tl.constexpr` scalar prefix scan (`for i in range(E_C - 1): e += (tile_idx >= bound).to(tl.int32)`—never vectorize expert lookup).
   - **L0C & UB Hardware Bounds**: Keep `BLOCK_M * BLOCK_N <= 16384` (`64 x 256` or `128 x 128` -> `64 KB` in `int32`; `128 x 256 = 32768` overflows `128 KB` `L0C`) and `2 * (BM * BK + BK * BN) <= 192 KB` (`BLOCK_K = 256`). Issue `weight` load **before** `x` load to hide `MTE1` (`L1 -> L0B` 2D transpose) latency (`+6.2%`).
   - **Row Boundary & Bit-Exact Dequant Chain**: Mask `x`, `xs`, and `y` using per-expert `row_hi[e]` (never global `M`). Freeze the epilogue rounding chain: `((acc_i32.to(f32) * ws).to(fp16).to(f32) * xs).to(fp16)` written to GM `y_fp16`.
3. **`K2` (`swiglu_dquant_kernel` on Vector)**:
   - Set `BLOCK_N = next_power_of_2(HALF_N)` to cover the full half-width in one tile.
   - **Critical `npu_swiglu` Formula**: `gate = y[:, :HALF_N]`, `up = y[:, HALF_N:]`, and **`sw = (gate / (1.0 + tl.exp(-gate))) * up` (strictly `sigmoid(gate) * up` without an extra `gate *` factor!)**. Round-half-away-from-zero + `int32` clamp `[-128, 127]`, and store scalar `out_s = max_abs / 127.0` using the 1-element vector broadcast trick `r1 = tl.arange(0, 1); tl.store(out_s_ptr + row + r1, val + r1 * 0.0)`.

---

## 2. `WeightQuantBatchmatmul` Playbook (`0.9133x` vs Native, `8.90x` over Baseline)

1. **Group vs. Non-Group Kernel Split (`L1.1`–`L1.4`)**:
   - Split into dedicated `group` and `non_group` kernels. Never materialize `[G, N]` group scales into `[K, N]` on the host; pass `[G, N]` directly, choose `BLOCK_K <= group_size` with `group_size % BLOCK_K == 0`, and load a single `[1, BLOCK_N]` row via scalar `g = k_start // group_size` (`4.4x` speedup over 2D vector gather).
   - Match dequant domain to `x.dtype`: `fp16` dequantizes in `fp16` before `tl.dot`; `bf16` dequantizes in `fp32 -> bf16 -> fp32`.
2. **Decoupled Dequant Workspace + Pure GEMM (`L1.10`–`L1.12`)**:
   - In fused kernels with large `M`, dequantization sits on the `tl.dot` critical path and repeats `M // BLOCK_M` times, starving Cube (`Cube ratio < 15%`).
   - Decouple into `dequant_weight_kernel` (writes `w_fp[K, N]`) + `matmul_kernel` (`BM=128, BN=256, BK=128`) whenever:
     - `not is_group and M >= 256`, **or**
     - `is_group and (group_size <= 64 or x.dtype == torch.bfloat16 or M >= 512)`.
3. **N-Strip L2 Chunking for Large Workspaces (`L1.13`, `0.54x -> 1.48–1.75x` on Large-$N$)**:
   - When `K * N * sizeof(low_dtype) > 60 MB` (`> 2/3` of `96 MB` L2 cache), materializing the full `[K, N]` workspace causes every M-tile in `matmul_kernel` to re-read `w_fp` from HBM.
   - Partition $N$ on the host into L2-resident strips of width `chunk_n = (48_MB // (K * elem_bytes)) // 256 * 256`. For each strip, launch `dequant_weight_kernel` into a `48 MB` workspace `[K, chunk_n]` and immediately launch `matmul_kernel` across **all M-tiles** into `out[:, n_start:n_end]` (disjoint N-slices, zero `atomic_add`).

---

## 3. `QuantMatmul` (`int8 x int8`) Playbook (`0.9282x` vs `npu_quant_matmul`)

- **Mandatory `int32` Accumulator & Post-Processing Order (`L1.1`–`L1.6`)**:
  - Accumulate `acc = tl.dot(x1_i8, x2_i8, acc_i32)` in `tl.int32` (`BLOCK_K = 256`).
  - If `bias.dtype == torch.int32` and `pertoken_scale is None`, compute `acc_f = (acc + bias_i32[None, :]).to(tl.float32) * scale[None, :]`.
  - Otherwise, strictly follow: `acc_f = acc.to(tl.float32) * scale[None, :]` -> `* pertoken_scale[:, None]` -> `+ bias` -> `+ offset` -> optional clamp -> `.to(out_dtype)`.
- **Dynamic Tile & Clamped Grid**: Use `128 x 128 x 256` when `not do_clamp and M * N > 32768`, but fall back to `64 x 128 x 256` or `64 x 64 x 256` when `do_clamp` is active (clamp materializes two full-size `fp32` buffers in UB). Decode 1D tile index without `%` (`rem = tile - bb * TILES_PER_BATCH`).

---

## 4. MoE Gating & Token Counting Playbooks

### 4.1 `MoeGatingTopKSoftmax` (`sort-topk`, 50/50 Pass)
- **Single-Pass In-Register Top-K (`NUM_EXPERTS <= 1536`)**: Load the entire expert row once with `offs = tl.arange(0, NUM_EXPERTS)`, compute `fp32` softmax in registers, and run `for i in range(k):` in registers (`best_val = tl.max(curr, 0)`, `best_idx = tl.argmax(curr, 0)`, mask out selected expert via `curr = tl.where(offs == best_idx, -float("inf"), curr)`). Never spill intermediate softmax vectors to GM temp buffers (`tl.argmax` after GM round-trip is unreliable and $3\times$ slower).
- **Large Expert Adaptive Chunking (`NUM_EXPERTS > 1536`)**: Use `CHUNK_E = 2048` or `4096` with single-pass online softmax + per-chunk top-$K$ candidates merged in registers (`0.46x -> 0.66x`). Pass `finished` as a dummy `torch.bool` tensor when `None`, and do **not** enable `multibuffer=True, unit_flag=True` (regresses latency on this kernel).

### 4.2 `MoeComputeExpertTokens` (`1.7764x` vs PyTorch, 50/50 Pass)
- **Two-Stage Contention-Free Architecture**:
  1. **Phase 1 (`moe_count_kernel`, Expert-Parallel `grid = (num_expert,)`)**: Each program owns **one** `expert_id`, scans `sorted_expert_for_source_row` in `BLOCK_SIZE = 1024` vector chunks, accumulates `count += tl.sum(((tokens == expert_id) & mask).to(tl.int32))`, and writes once to `counts[expert_id]`. **Never** parallelize across tokens with `tl.atomic_add` (`6x` slower due to atomic contention).
  2. **Phase 2 (`prefix_sum_kernel`, `grid = (1,)`)**: Single-program unrolled scalar scan (`for eid in range(num_expert): prefix += count; store`) for `num_expert <= 64` (avoids `tl.cumsum` lowering issues). Allocate `counts` with `torch.empty` instead of `torch.zeros`.
