---
schema: mindie-graded-unit/1
id: unit-sparse-mla-and-lightning-indexer-cube-vector-pingpong-mutator
revision: 1
tier: L4
category: mutator
domain_path: attention/sparse-mla-indexer
evidence_strength: hypothesis
title: 'DeepSeek/Qwen Sparse Attention, MLA Prologue and MXFP4 Lightning Indexer Cube-Vector ping-pong and online softmax mutator'
conditions:
  op_family:
  - attention
  - sparse_attention
  - lightning_indexer
  - mla_prologue
  soc:
  - Ascend950
  - Ascend910B
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  - fp4_e2m1
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-ascend-fused-infer-attention-and-flash-mla-c8-sop
provenance:
  source_type: ablation_design_receipt
  source_locator: docs/evidence/l4-sparse-mla-lightning-indexer-ablation.json
  source_revision: 4519e5372a8b9306f7d823688c0cb8141ee34fb1
  supporting_source_revisions:
  - f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  - be5882ef29fa9943305085652d2025a038ba491c
  evidence_refs:
  - docs/evidence/l4-sparse-mla-lightning-indexer-ablation.json
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/sparse_flash_attention/sparse_flash_attention.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/quant_lightning_indexer_dsl/quant_lightning_indexer_dsl.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/attn_prologue/attn_prologue.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/qwen_sparse_attn/qwen_sparse_attn.py
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/cannbot-dsl-best-practice/SKILL.md
  experiment_status: not_run_source_only
  supersedes: []
triggers:
  error_signatures:
  - sparse attention gathers KV tokens one by one into L1 and serializes AIC QK/PV matmuls with AIV online softmax
  - p_ub probabilities are written in ND layout and require an extra L1 transpose or HBM copy before Cube PV matmul
  - scalar get_value() after a register-level vector UB store reads stale row_max or row_sum values because vmem_bar('vst_vld') is missing
  - attention prologue reloads input X from HBM for KV projection or suffers RMSNorm drift after split-K reduction
  - MXFP4 lightning indexer transfers full fp32 L0C tiles to a single AIV or overflows UB when sorting total_n > 4096 candidates
  keywords:
  - SparseFlashAttention
  - QwenSparseAttn
  - QuantLightningIndexer
  - AttnPrologue
  - MLA absorb
  - paired KV gather
  - Cube-Vector ping-pong
  - NZ fractal p_ub
  - online softmax
  - vmem_bar
  - vst_vld
  - MXFP4
  - E2M1
  - E8M0
  - vbitsort32
  - vmrgsort4
  - Ascend950
---

# DeepSeek/Qwen Sparse Attention, MLA Prologue and MXFP4 Lightning Indexer Cube-Vector ping-pong and online softmax mutator

## Applicability

Use this mutator for sparse MLA-absorb attention (`D = 512` Nope, `Dr = 64` RoPE, `G = H_q // H_kv <= 64`, `topk` sparse token indices), GQA sparse attention (`D = 128`, `SubH1 = min(H_q // H_kv, 16)`), MLA attention prologue (`Matmul_QA` + split-K RMSNorm + `Matmul_KV` + `INTERLEAVE_HALF` RoPE), or MXFP4 (`E2M1` + `E8M0` `[2, 2]` group scale) Lightning Indexer (`D = 128`, `H <= 64`, `N_BLOCK = 512`) on Ascend 950 (or Ascend 910B/910_93 via GM/L2 workspace fallback).

**Evidence boundary.** The three factors are distilled from the pinned `cannbot-dsl` reference implementations (`4519e5372a8b9306f7d823688c0cb8141ee34fb1`). No paired hardware timing campaign or leave-one-out execution was run in this distillation pass; the linked receipt records `experiment_status: not_run_source_only` and `evidence_strength: hypothesis`.

## Signals

- **Sparse KV gather & CV serialization signal**: AIV issues one `copy_ub_to_cbuf` (`MTE3 -> L1`) per sparse token index, `Q_nope`/`Q_rope` are reloaded across `s2_tile` chunks, or AIC `QK`/`PV` matmuls wait on AIV softmax without multi-slot L1/workspace ring overlap.
- **Online softmax & prologue signal**: online softmax materializes multiple intermediate UB buffers across separate VF calls; removing a barrier causes `get_value()` to read stale `row_max`/`row_sum`; or the MLA prologue reloads `X [M, K]` from HBM for `Matmul_KV` and shows split-K RMSNorm precision drift.
- **Lightning indexer signal**: `L0C -> UB` transfers full `fp32` scores to one AIV sub-block while the twin AIV sits idle; `ReLU` runs after widening to `fp32`; or `total_n > 4096` overflows UB in full-row `topk_sort` while skewed variable sequence lengths leave cores idle.

## Transformation

Keep all six correctness prerequisites (`pto.vmem_bar("vst_vld")`, `fp32` running max/LSE rescaling, negative/OOB sparse index masking to `-inf`, `s2_tile` tail masking, NZ fractal `p_ub` stride alignment, and Kahan split-K compensation) active in every ablation arm. Apply the three performance factors separately:

1. **`mla_absorb_paired_kv_gather_and_3slot_cv_pingpong`** —
   - **Outer L1 query residency**: preload `q_l1 [64, 512]` and `q_rope_l1 [64, 64]` once per `(b, s1, h2)` query task and keep them resident in L1 across all `topk` KV chunks (`s2_tile = 64`).
   - **Paired 2-token AIV UB gather (`_gather_kv_pair_to_l1`)**: instead of copying 1 token at a time from UB to L1, gather two sparse tokens `idx0, idx1` from `KV_gm` (`[total_kv, 1, 512]`) into a contiguous `[2, 512]` UB buffer `kv_pair_ub` (`MTE2`), issue `pto.pipe_barrier("mte2_mte3")`, and transfer the 2-row block into `k_nope_l1[slot, t:t+2, :]` via a single `copy_ub_to_cbuf` (`MTE3 -> L1`), cutting L1 transfer commands in half. Clamp negative or out-of-bounds indices (`idx < 0` or `idx >= s2_act`) to row 0 during gather and mask their attention scores to `-1e30` in softmax.
   - **Multi-slot Cube-Vector ping-pong**: allocate a 3-slot L1 KV ring (`k_nope_l1 [3, 64, 512]`, `k_rope_l1 [3, 64, 64]`, `MOD = 3` in MLA-absorb; or a 4-stage `MOD = 4` workspace ring in GQA sparse attention) and overlap Stage 0 (AIV paired KV gather into `slot`), Stage 1 (AIC `C = Q_nope @ K_nope^T + Q_rope @ K_rope^T` into `s_gm[slot]`), Stage 2 (AIV online softmax writing `p_l1[slot]`), and Stage 3 (AIC `O = P @ V_nope` into `o_gm[slot]`) using four directional cross-core event sets (`EVT_K_L1_READY`, `EVT_S_GM_READY`, `EVT_P_L1_READY`, `EVT_O_GM_READY`).
   - **Direct NZ fractal `p_ub` staging**: format the cast `bf16`/`fp16` probability tile in UB directly as NZ fractal `[s2_tile // 16, 64, 16]` (`[4, 64, 16]` for `64x64`, 16-column contiguous fractals) so `copy_ub_to_cbuf(..., pad_mode="None")` lands in `p_l1` ready for `extract_l1_to_l0b` without an L1 transpose.

2. **`register_online_softmax_vst_vld_and_compensated_prologue_rope`** —
   - **Register-level `_StableOnlineSoftmaxVector` with `vst_vld` fence**: fuse row-wise running maximum `m_new = max(m_prev, row_max)` (`vcmax`), rescaling factor $\alpha = \exp((m_{\text{prev}} - m_{\text{new}}) \cdot \text{scale})$, unnormalized weight $p = \exp((s - m_{\text{new}}) \cdot \text{scale})$, denominator update $l_{\text{new}} = l_{\text{prev}} \cdot \alpha + \sum p$ (`vcadd`), and output rescaling $O \leftarrow O \cdot \alpha + O_{\text{cur}}$ in `fp32` vector registers inside `@pto.vf_inline`. Immediately after storing a reduced vector register (`row_max` or `row_sum`) to a UB tensor and before reading scalars via `.get_value(r)` for row-broadcast `vadds`/`vmuls`, emit `pto.vmem_bar("vst_vld")`:

     ```python
     s_max_ub.store(v_max, 0, repeat=1)
     pto.vmem_bar("vst_vld")  # mandatory before scalar get_value() reads UB
     for r in pto.range(0, G, 1):
         row_m = s_max_ub.get_value(r)
     ```

   - **L1-resident `X` prologue with Kahan split-K RMSNorm and fused tail RoPE (`attn_prologue.py`)**: load input activation `X [m_tile, k_tile]` into L1 once and reuse both L1 and `L0A` (`reuse_l0a`) across `Matmul_QA` (`W_qa`) and `Matmul_KV` (`W_ukv`, `D_NOPE = 512`). When accumulating split-K partials for `qa_gm` before RMSNorm, apply Kahan error compensation (`t = sum + y; c_err = (t - sum) - y; sum = t`) in `fp32` registers so multi-split reduction matches the single-pass oracle, then fuse `INTERLEAVE_HALF` RoPE (`even = x[2i] * cos - x[2i+1] * sin`, `odd = x[2i+1] * cos + x[2i] * sin`) on the `Dr = 64` tail in UB.

3. **`mxfp4_lightning_indexer_split_m_fixpipe_and_streaming_topk`** —
   - **MXFP4 16x16x128 Cube MMA + split-M `F322BF16` Fixpipe handoff (`quant_lightning_indexer_dsl.py`)**: keep quantized `q_l1 [64, 128]` (`fp4_e2m1`) and `q_scale_l1` (`E8M0` uint8 `[2, 2]` group scales) resident in L1 across all `N_BLOCK = 512` key blocks (`mx_m16n16k128`). Partition the `[H, N_PER_SUB]` (`[64, 256]`) `L0C` accumulator along M into two `[H_HALF, 256]` (`[32, 256]`) halves and transfer them directly from `L0C` to AIV0 and AIV1 UB (`s_ub`) with hardware `quant_pre="F322BF16"` and directed `pto.set_cross_flag("FIX", "V", flag, mode=4, aiv_id=0/1)`.
   - **BF16 ReLU pre-filter before `fp32` weighted head reduction**: execute `vmaxs(s_bf16, 0.0)` in `bf16` before `vconv_bf162f32` and `vmla(acc_f32, s_f32, w_h)`, halving vector width during ReLU.
   - **Dual-mode cached vs streaming TopK (`total_n == 4096` crossover) + AICPU dynamic queue**: when `total_n <= 4096`, cache up to 8 blocks of 512 scores in UB and run `topk_sort` (`vbitsort32` + `vmrgsort4`) once at the end; when `total_n > 4096`, run incremental streaming TopK after each 512-element block by sorting the current block and merging its top `K_ALIGN` candidates with the running `top_k_ub` via `vmrgsort4`. Dispatch variable-length `(b, m)` rows across cores through an AICPU-managed atomic task queue (`TASK_STATE_OFFSET`) so short causal rows never leave cores idle.

## Fallback boundary

- **Ascend 910B / 910_93 without L0C->UB Fixpipe or MXFP4 MMA**: stage `L0C` through an L2-resident GM workspace with `mode = 2` `CrossCoreSetFlag` and use the platform's supported `bf16`/`fp16`/`int8` MMA instructions.
- **Dimensions outside the L1 resident contract**: if `G > 64` or `D > 512` exceeds the 512 KiB L1 budget alongside the 3-slot KV ring, reduce `MOD` from 3 to 2 or tile along `G`.
- **Mandatory memory fence**: never strip `pto.vmem_bar("vst_vld")` between a vector UB store and scalar `get_value()`.

## Verification and ablation

1. **Gate 1 — OracleFreezer**: verify `Out` and `LSE` against the `fp32` reference (`rtol=1e-2, atol=1e-2` for `bf16`/`fp16`) across aligned `topk`, unaligned `topk % 64 != 0`, negative/OOB sparse index entries, `total_n <= 4096` and `total_n > 4096` TopK crossover, and skewed `actual_seq_lengths`.
2. **Gate 2 — Leave-one-out ablation**: run `full_candidate`, `minus_paired_gather_and_cv_pingpong`, `minus_register_softmax_and_prologue_reuse`, and `minus_mxfp4_split_m_and_streaming_topk` as defined in `docs/evidence/l4-sparse-mla-lightning-indexer-ablation.json`.
