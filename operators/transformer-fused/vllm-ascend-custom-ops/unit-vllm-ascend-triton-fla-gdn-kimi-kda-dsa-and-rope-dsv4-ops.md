---
category: sop
conditions:
  cann_versions:
  - '>=8.5.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - vLLM-Ascend custom Triton and AscendC operators for Qwen3-Next, Kimi-Linear, and
    DeepSeek-V3.2/V4
domain_path: operators/transformer-fused/vllm-ascend-custom-ops
evidence_strength: software_contract
id: unit-vllm-ascend-triton-fla-gdn-kimi-kda-dsa-and-rope-dsv4-ops
inherits_from:
- unit-triton-ascend-constexpr-and-specialization-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/triton/fla/chunk.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/gdn.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/kimi_kda.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/dsa.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/mla.py
  - https://gitcode.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/vllm_ascend/ops/rope_dsv4.py
  source_digest: 00c953f57f75d93dfc3aae27adf3dca54169bab87053b2a9e6fbf2693d0183f4
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/ops/triton/fla/chunk.py
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Operator wiring and kernel contract in `vllm_ascend/ops/` for Triton FLA
  chunked Gated DeltaNet (`gdn.py`, `triton/fla/chunk.py`), `kimi_kda.py`, Dynamic
  Sparse Attention (`dsa.py`), `mla.py`, and `rope_dsv4.py`.
tier: L2
title: 'vLLM-Ascend Custom Operators: Triton FLA/GDN, Kimi-KDA, DSA Sparse Attention,
  and DeepSeek-V4 RoPE'
triggers:
  error_signatures:
  - vllm_ascend ops triton fla chunk.py
  - vllm_ascend ops gdn.py kimi_kda.py
  keywords:
  - vllm_ascend ops triton fla chunk.py
  - vllm_ascend ops gdn.py kimi_kda.py
  - vllm_ascend ops dsa.py mla.py
  - vllm_ascend ops rope_dsv4.py
  - vllm-ascend
  - triton-fla
  - gdn
  - kimi-kda
  - dsa
  - mla
  - rope-dsv4
---
## 1. Gated DeltaNet (`gdn.py`) & Triton FLA Chunk Kernels (`ops/triton/fla/chunk.py`)

`vllm_ascend/ops/gdn.py` and `vllm_ascend/ops/triton/fla/` power hybrid linear-attention models (`Qwen3-Next`, `Bailing-Linear`):
1. **Prefill vs. Decode Split Dispatch**:
   - **Prefill Tokens (`seq_len > 1`)**: Dispatches to `chunk_gated_delta_rule` (`ops/triton/fla/chunk.py`), which computes cumulative decay gates (`chunk_local_cumsum`), WY Householder matrices (`prepare_wy_repr`), inter-chunk state propagation (`chunk_fwd_h`), and intra/inter-chunk output combination (`chunk_fwd_o`) over `BT = 64` token chunks.
   - **Decode Tokens (`seq_len == 1` or `1 + MTP`)**: Dispatches to `fused_sigmoid_gating_delta_rule_update` (and `causal_conv1d_update`), updating each sequence's recurrent state slot `ssm_state[state_indices]` in-place in a single kernel pass.
2. **`Kimi-KDA` (`ops/kimi_kda.py` & `kimi_kda_state.py`)**:
   - Implements Key-Delta Attention state management (`conv_state` + `kda_recurrent_state` indexed by request slot mapping) with mixed chunkwise prefill and single-step recurrent decode updates.

## 2. Dynamic Sparse Attention (`ops/dsa.py`), `ops/mla.py`, and `ops/rope_dsv4.py`

1. **`ops/dsa.py` (`Dynamic Sparse Attention`)**:
   - Orchestrates the DeepSeek-V3.2-Exp / DeepSeek-V4 sparse attention forward pass in `vllm-ascend`:
     - Projects indexer queries/keys/weights, writes indexer keys into paged `indexer_k_cache`, invokes `torch_npu.npu_lightning_indexer` (`topk=2048`), and passes the resulting `sparse_indices` into `npu_sparse_flash_attention` (`SFA`).
2. **`ops/mla.py` (`Multi-head Latent Attention`)**:
   - Integrates `torch_npu.npu_mla_prolog_v2` / `v3` for fused pre-attention down/up projection + RoPE + KV cache write, paired with `npu_fused_infer_attention_score` (`FIA`) in absorbed decode mode (`W_uk` absorbed into query, `W_uv` applied after attention).
3. **`ops/rope_dsv4.py` (`DeepSeek-V4 RoPE`)**:
   - Handles DeepSeek-V4 specific rotary position embedding layouts (interleaved vs. half-split rotary dimensions with YaRN frequency scaling) using custom Triton/AscendC kernels registered as `torch.ops.vllm` custom ops for full `aclgraph` compatibility.

## Verification Checklist

- Ensure all custom Triton-Ascend kernels in `ops/triton/fla/` use fixed `tl.constexpr` block sizes (`BT=64, BK=64, BV=64`) that fit within Ascend 910B/910C UB capacity (`192 KB`).
