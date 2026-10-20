---
schema: mindie-graded-unit/1
id: unit-streamk-matmul-and-mxa8w4-quant-bmm-tile-scheduling-mutator
revision: 1
tier: L4
category: mutator
domain_path: matmul/streamk-mxquant
evidence_strength: hypothesis
title: 'Stream-K DPSK load-balanced Matmul and MXA8W4 bit-permuted group-quantized BatchMatmul tile scheduling mutator'
conditions:
  op_family:
  - matmul
  - quant_batch_matmul
  - streamk_matmul
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  - fp8_e4m3fn
  - int4
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-catlass-and-tilelang-ascend-operator-design-and-tuning-sop
provenance:
  source_type: ablation_design_receipt
  source_locator: docs/evidence/l4-streamk-and-mxa8w4-matmul-ablation.json
  source_revision: 4519e5372a8b9306f7d823688c0cb8141ee34fb1
  supporting_source_revisions:
  - f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  - be5882ef29fa9943305085652d2025a038ba491c
  evidence_refs:
  - docs/evidence/l4-streamk-and-mxa8w4-matmul-ablation.json
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/matmul/matmul_streamk.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/matmul/quant_matmul/quant_batch_matmul_mxa8w4.py
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/cannbot-dsl-best-practice/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/aiss-tiling-solver/SKILL.md
  experiment_status: not_run_source_only
  supersedes: []
triggers:
  error_signatures:
  - a GEMM with total_tiles % aic_num != 0 suffers wave quantization where most AIC cores sit idle on the last wave
  - Stream-K global barrier polling hangs because sync_gm is read without SINGLE_CACHE_LINE cache invalidation
  - W4A8 or MXA8W4 weight dequantization on AIV bottlenecks the Cube pipeline due to lookup-table or multi-step float conversion
  - host-side K-padding copies the entire activation and weight tensors when K is not a multiple of bk
  keywords:
  - Stream-K
  - DPSK
  - wave quantization
  - sliding window L2
  - SINGLE_CACHE_LINE
  - DataCacheCleanAndInvalid
  - MXA8W4
  - QuantBatchMatmul
  - E2M1
  - E8M0
  - fp8_e4m3fn
  - bit permutation
  - deq_scale
  - DualParam
  - Ascend950
---

# Stream-K DPSK load-balanced Matmul and MXA8W4 bit-permuted group-quantized BatchMatmul tile scheduling mutator

## Applicability

Use this mutator for (1) dense `bf16`/`fp16` GEMMs where the output tile count $\text{total\_tiles} = \lceil M / b_m \rceil \times \lceil N / b_n \rceil$ is not divisible by the physical AIC core count (`aic_num`), causing tail-wave underutilization, or (2) group-quantized `MXA8W4` BatchMatmul (`X` in `fp8_e4m3fn`, `W` packed `int4x2` `fp4_e2m1` with group size $G = 32$ and `E8M0` uint8 scales, `bf16` output) on Ascend 950 (or Ascend 910B/910_93 for the Stream-K and Roofline tiling factors).

**Evidence boundary.** Distilled from `matmul_streamk.py` and `quant_batch_matmul_mxa8w4.py` in `cannbot-dsl` (`4519e5372a8b9306f7d823688c0cb8141ee34fb1`). No paired hardware timing campaign or leave-one-out execution was run in this distillation pass; `docs/evidence/l4-streamk-and-mxa8w4-matmul-ablation.json` records `experiment_status: not_run_source_only` and `evidence_strength: hypothesis`.

## Signals

- **Tail-wave quantization signal**: $\text{rem} = \text{total\_tiles} \bmod \text{aic\_num} \neq 0$ (e.g., 37 tiles on 36 AICs runs 2 full waves at 51% occupancy), or row-major tile ordering degrades L2 hit rate on large $M, N$.
- **Stream-K reduction signal**: routing all tiles through a workspace adds unnecessary GM traffic for unsplit tiles, or polling a global atomic counter hangs without `SINGLE_CACHE_LINE` cache invalidation.
- **MXA8W4 weight prologue signal**: AIV `int4 -> fp8` unpacking stalls AIC (`aic_mte1_ratio` / `WAIT_FLAG` high), explicit vector multiplication by scale factors burns VF cycles, or `K % bk != 0` triggers a full-tensor GM padding copy on the host.

## Transformation

Keep all five correctness prerequisites (`fp32` Stream-K partial accumulation, `SINGLE_CACHE_LINE` cache invalidation during barrier spin, deterministic ordered partial reduction, `deq_scale = 64.0` + `bias * 2^-6` bit-permutation compensation, and 64B/32B K-tail alignment narrowing) active in every ablation arm. Apply the three performance factors separately:

1. **`dpsk_streamk_two_tile_wave_split_and_sliding_window_l2_order`** (`matmul_streamk.py`) —
   - **Hybrid Data-Parallel + Stream-K (DPSK) partition**: given $\text{total\_tiles} = \lceil M / b_m \rceil \times \lceil N / b_n \rceil$ and $\text{rem} = \text{total\_tiles} \bmod \text{aic\_num}$:
     $$\text{sk\_tiles} = \begin{cases} 0 & \text{if } \text{rem} == 0 \\ \text{rem} + \text{aic\_num} & \text{if } \text{total\_tiles} > \text{aic\_num} \text{ and } \text{rem} > 0 \\ \text{rem} & \text{otherwise} \end{cases}, \quad \text{dp\_tiles} = \text{total\_tiles} - \text{sk\_tiles}$$
     Partition the $\text{total\_sk\_iters} = \text{sk\_tiles} \times \lceil K / b_k \rceil$ K-iterations evenly across all `aic_num` cores (`iters_per_core = total_sk_iters // aic_num`, with the first `total_sk_iters % aic_num` cores taking `+1` iteration), and assign the remaining `dp_tiles` in full-tile waves (`t = sk_tiles + core_id + wave * aic_num`).
   - **Sliding-window serpentine L2 ordering & 4-stage L1 pipeline**: map logical tile index `t` to `(m_idx, n_idx)` in column panels of width `WINDOW_W = 4` (`window_id = t // (4 * num_m)`, `m_idx = (t % (4 * num_m)) // 4`, `n_idx = window_id * 4 + (t % 4)`) with serpentine row reversal on odd rows (`if m_idx % 2 == 1: n_idx = num_n - 1 - n_idx`), and stage K iterations through a 4-buffer L1 (`a_l1`, `b_l1`) + 2-buffer L0 (`l0a`, `l0b`) software pipeline (`preload = min(total_k, 3)`).

2. **`three_phase_atomic_cross_core_sk_workspace_and_aiv_register_reduce`** (`matmul_streamk.py`) —
   - **Full-K direct Fixpipe bypass**: even within the `sk_tiles` region, any tile slice covering the full K axis (`k_start == 0` and `k_end == k_iters_per_tile`, plus all `dp_tiles`) writes directly from `L0C` to `C_gm` via `copy_matrix_cc_to_gm(..., quant_pre="F322BF16")`, bypassing the workspace.
   - **Split-K workspace & 3-phase cross-core barrier**: partial tiles write `f32` accumulators to `workspace_gm` at slot `t * aic_num + (core_id - first_core_for_tile(t))`. Then execute:
     1. **Phase 1 (AIC -> local AIV0)**: each AIC signals `pto.set_cross_flag("FIX", "MTE2", EVENT_CV_SYNC)` after its final Fixpipe write.
     2. **Phase 2 (Global AIV0 atomic barrier)**: AIV0 (`sub_id == 0`) waits on `EVENT_CV_SYNC`, adds `1` to `sync_gm[0]` via `SetAtomicAdd` + `copy_ub_to_gm` + `SetAtomicNone`, and spin-waits until `sync_gm[0] == aic_num`, issuing `pto.DataCacheCleanAndInvalid(sync_gm, 0, "SINGLE_CACHE_LINE")` inside every poll iteration so L2/UB never caches a stale counter.
     3. **Phase 3 (Parallel AIV register reduction)**: all `2 * aic_num` AIVs (`vid = core_id * 2 + sub_id`) stride across split tiles in `REDUCE_CHUNK = 16384`-element chunks, accumulating `num_partials` `f32` workspace slots in vector registers (`vadd(acc, acc, cur)` inside `@pto.vf_inline`) in deterministic core order before a single `vconv` (`RINT` for `bf16`) and `copy_ub_to_gm`.

3. **`mxa8w4_bit_permuted_nibble_unpack_and_dual_aiv_split_m_l1_ring`** (`quant_batch_matmul_mxa8w4.py`) —
   - **Makespan-aware Roofline tile selection (`MixQuantTiling`)**: choose `(bm, bn, bk)` by minimizing $\text{waves} \times (\max(T_{\text{mte2}}, T_{\text{vec}}, T_{\text{mte3}}, T_{\text{mte1}}, T_{\text{mac}}) + T_{\text{fix}})$, shrinking `bm/bn` from `128` to `64` when small `B * M * N` underfills cores (`tiles < num_cores // 2`) and doubling `bk` from `128` to `256` when $K \ge 256$ and `total_tiles >= 2 * num_cores`.
   - **Zero-LUT $2^{-6}$ bit-permuted `E2M1 -> E4M3` unpack on AIV**: an `fp4_e2m1` nibble `[s e1 e0 m0]` (bias 1) mapped into an `fp8_e4m3fn` byte as `[s 0 0 e1 e0 m0 0 0]` (bias 7) has exponent $(e - 1) = (e - 7) + 6$, making the byte's exact `fp8_e4m3fn` value $2^{-6} = 1/64$ of the `fp4_e2m1` value for every normal, subnormal (`0001` $\to$ $0.5 \times 2^{-6} = 2^{-7}$), and signed zero (`0000`/`1000`). Unpack two nibbles per byte in `@pto.vf_inline` using bitwise `vand` (`0x0F0F` / `0xF0F0`), sign extraction (`0x8080` / `0x0808`), shift (`vlsl` / `vlsr` by 2), `vor`, and `vzip(..., "B8")`.
   - **Zero-cost Fixpipe `deq_scale = 64.0` compensation**: eliminate per-element vector multiplication by $64.0$ by pre-scaling optional `bias` by $2^{-6} = 0.015625$ (`vmuls`) and passing `deq_scale = 64.0` to `copy_matrix_cc_to_gm(..., quant_pre="DEQF322BF16")` so $(X \cdot (W \cdot 2^{-6}) + \text{bias} \cdot 2^{-6}) \times 64 = X \cdot W + \text{bias}$.
   - **Dual-AIV `DualParam(0, 1)` split-M L1 ring and runtime K-tail narrowing**: AIV0 and AIV1 unpack `sub_n = bn // 2` rows each into a `k_l1`-deep L1 ring (`b_l1_ring [k_l1, bn, bk]`) synchronized with AIC via `mode = 2` `CrossCoreSetFlag`/`CrossCoreWaitFlag`. On a partial K tail (`tail_k = K - k_idx * bk < bk`), narrow L1 DMA width to $\lceil \text{tail\_k} / 64 \rceil \times 64$ and pass $k_{\text{actual}} = \lceil \text{tail\_k} / 32 \rceil \times 32$ to `mx_matmul`, avoiding any host GM K-padding.

## Fallback boundary

- **Divisible tile grid (`total_tiles % aic_num == 0`)**: `sk_tiles` is `0`; skip the Stream-K workspace and 3-phase barrier completely.
- **Non-Ascend950 targets**: `mx_m16n16k64` and `fp8_e4m3fn` require Ascend 950; on Ascend 910B/910_93 use the platform's `W4A8`/`W8A8` `QuantBatchMatmul` contract while retaining the DPSK scheduling and Roofline wave model.

## Verification and ablation

1. **Gate 1 — OracleFreezer**: verify against an `fp32` reference across `total_tiles % aic_num == 0`, `total_tiles % aic_num == 1`, `total_tiles < aic_num`, `K % bk != 0` tails, subnormal/signed-zero `E2M1` nibbles, with/without `f32` bias, and 3 byte-identical repeats.
2. **Gate 2 — Leave-one-out ablation**: evaluate `full_candidate`, `minus_dpsk_wave_split_and_l2_window`, `minus_direct_full_tile_bypass_and_reg_reduce`, and `minus_bit_permuted_unpack_and_k_tail_narrowing` per `docs/evidence/l4-streamk-and-mxa8w4-matmul-ablation.json`.
