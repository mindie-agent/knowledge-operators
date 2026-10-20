---
category: sop
conditions:
  kernel_language: cannbot_dsl
  soc:
  - Ascend950
  - Ascend950PR
  - Ascend950DT
  workloads:
  - FlashKDA
  - FlashKdaFwd
  - FlashKdaCp
  - MegaRecurrentKda
  - MegaRecurrentKdaReplaySsm
  - FusedRecurrentKdaSnapshot
  - CommitRecurrentKdaReplaySsm
domain_path: attention/kda-recurrent-ssm
evidence_strength: reproduced_empirical
id: unit-flash-kda-and-mega-recurrent-replayssm-dsl-sop
inherits_from:
- unit-ascend950-microarchitecture-per-core-memory-and-interconnect-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  attribution: Adapted from CANNBot-DSL FlashKDA and Recurrent KDA ReplaySSM sample
    implementations
  evidence_refs:
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_kda/flash_kda.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_kda_fwd/flash_kda_fwd.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_kda_cp_preprocess/flash_kda_cp_preprocess.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_kda_cp_main/flash_kda_cp_main.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/mega_recurrent_kda/mega_recurrent_kda.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/mega_recurrent_kda_replayssm/mega_recurrent_kda_replayssm.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/fused_recurrent_kda_replayssm/fused_recurrent_kda_replayssm.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/fused_recurrent_kda_snapshot/fused_recurrent_kda_snapshot.py
  - https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/commit_recurrent_kda_replayssm/commit_recurrent_kda_replayssm.py
  source_locator: https://gitcode.com/cann/cannbot-dsl/blob/4519e5372a8b9306f7d823688c0cb8141ee34fb1/samples/flash_kda/flash_kda.py
  source_type: empirical_kernel_and_runbook_distillation
  supersedes: []
related:
- unit-mixed-quant-sparse-flash-mla-and-kvcache-dsl-sop
- unit-mhc-pre-post-sinkhorn-knopp-and-engram-gate-fusion-sop
revision: 1
schema: mindie-graded-unit/1
status: active
summary: Standard operating procedure for Kimi Delta Attention (KDA) across 64-token
  Chunk Prefill (inference, training forward, and Context-Parallel split) and Speculative
  Decode / ReplaySSM verification, snapshot, and state commit in CANNBot-DSL on Ascend
  950.
tier: L2
title: FlashKDA (Chunk/Fwd/CP) & Mega/Fused Recurrent KDA ReplaySSM DSL SOP
triggers:
  error_signatures:
  - Chunk FlashKDA partitions sequences into 64-token chunks, solving lower-triangular
    (I + stril(diag(beta) * K_tilde * K_bar^T))^-1 within each chunk and propagating
    FP32 state S across chunks
  - Context-Parallel (CP) FlashKDA splits prefill into flash_kda_cp_preprocess (intra-chunk
    + intra-rank summary) and flash_kda_cp_main (post cross-rank prefix scan)
  - MegaRecurrentKda and ReplaySSM support controlled AIC quotas {8, 16, 24, 28, 32}
    (and {8, 24, 28, 32} for Commit) with AIV quota >= 2 * AIC
  keywords:
  - flash_kda
  - flash_kda_fwd
  - flash_kda_cp_preprocess
  - flash_kda_cp_main
  - mega_recurrent_kda
  - mega_recurrent_kda_replayssm
  - fused_recurrent_kda_replayssm
  - fused_recurrent_kda_snapshot
  - commit_recurrent_kda_replayssm
  - kimi_delta_attention
---
## Preconditions & Signals

Apply this SOP when implementing, scheduling, or debugging Kimi Delta Attention (KDA) linear-attention / state-space operators on Ascend 950 (`ARCH 3510`), covering both 64-token Chunk Prefill and speculative-decoding ReplaySSM recurrence:

1. **64-Token Chunk KDA Prefill Family**:
   - **`flash_kda` (Inference) & `flash_kda_fwd` (Training Forward)**: Fuse row-wise $Q/K$ L2 normalization, gate log-decay activation $g_t$, $\beta_t = \sigma(\beta_t^{\text{raw}})$, 64-token intra-chunk triangular inversion $A_c = (I + T_c)^{-1}$, and inter-chunk FP32 state recurrence $S_c \in \mathbb{R}^{D_v \times D_k}$ ($D_k=D_v=128$).
   - **`flash_kda_cp_preprocess` & `flash_kda_cp_main` (Context-Parallel Prefill)**: Split chunk KDA across CP ranks around an external cross-rank prefix scan.
2. **Recurrent Decode & Speculative ReplaySSM Family ($S \le 8$, $D=128$)**:
   - **Standalone Recurrent (`fused_recurrent_kda_snapshot` & `fused_recurrent_kda_replayssm`)**: Operate on pre-projected/convolved $Q/K/V$ to either save per-token full state snapshots or emit lightweight replay records (`replay_u`, `replay_k`, `replay_decay`) without mutating the read-only checkpoint state.
   - **Mega Fused Kernel (`mega_recurrent_kda` & `mega_recurrent_kda_replayssm`)**: Fuse QKV/decay/beta/output-gate projections, causal Conv1D + SiLU, Q/K L2 norm, recurrent KDA, RMSNorm, output gating, and output projection into a single mixed AIC/AIV kernel (`S=8, H=7168, D=128`, TP4/TP8/TP16).
   - **State Commit (`commit_recurrent_kda_replayssm`)**: Sequentially applies $S \leftarrow S \odot \text{decay} + U K^\top$ for the first `num_accepted_tokens[b]` tokens per batch.

## Procedure

### Step 1: 64-Token Chunk KDA & Triangular Solver (`flash_kda`, `flash_kda_fwd`)
1. **Persistent vs Staged Kernel Dispatch**:
   - Generate work distribution via `flash_kda_metadata.py` / `flash_kda_fwd_metadata.py` on AICPU.
   - Dispatch between `flash_kda_persistent.py` (keeping state $S$ resident on-chip across chunks of the same `(batch, head)`) and `flash_kda_staged.py` based on sequence length and core occupancy.
2. **Intra-Chunk Cumulative Decay & Strictly Lower-Triangular Inversion**:
   - Within each 64-token chunk, accumulate log-decay $G_t = \sum_{i=0}^t g_i$ in FP32, form $\gamma_t = \exp(G_t)$, and scale $\tilde{K} = \gamma \odot K$, $\bar{K} = \gamma^{-1} \odot K$, $\tilde{Q} = \text{scale} \cdot (\gamma \odot Q)$.
   - Compute $T_c = \operatorname{stril}(\operatorname{diag}(\beta) \tilde{K} \bar{K}^\top)$ and invert $A_c = (I + T_c)^{-1}$ by block forward substitution (e.g., $16 \times 16$ diagonal blocks + off-diagonal GEMM updates).
   - Compute $U_c = A_c \operatorname{diag}(\beta) V_c$ and $W_c = A_c \operatorname{diag}(\beta) \tilde{K}_c$, then update output $O_c$ and state $S_c$ across chunks in FP32. For `flash_kda_fwd`, additionally write out backward-pass activation caches.

### Step 2: Context-Parallel (CP) Two-Stage Prefill (`flash_kda_cp_preprocess`, `flash_kda_cp_main`)
1. **Stage 1 (`flash_kda_cp_preprocess`)**:
   - Schedule via `flash_kda_cp_preprocess_metadata.py`. Fuse $Q/K$ L2Norm, gate/beta activation, intra-chunk $(q^g_c, k^g_c, u_c, w_c, M_{qk,c}, \text{decay}_c)$ computation, and intra-rank recurrent state summary into 8 output tensors so the caller can perform a lightweight cross-rank prefix scan.
2. **Stage 2 (`flash_kda_cp_main`)**:
   - Schedule via `flash_kda_cp_main_metadata.py`. Consume the 8 precomputed chunk tensors plus this rank's prefix-scanned initial state $S_0$:
     $$\Delta_c = u_c + w_c S_{c-1}^\top, \qquad O_c = q^g_c S_{c-1}^\top + M_{qk,c} \Delta_c$$
     and propagate $S_c$ to produce the final rank output and state.

### Step 3: Speculative Decode ReplaySSM Verify, Snapshot & Commit
1. **Read-Only Checkpoint Verify (`fused_recurrent_kda_replayssm` & `mega_recurrent_kda_replayssm`)**:
   - For each candidate token $t \in [0, S)$, normalize $q_t = Q_t / \sqrt{\sum Q_{t,j}^2 + 10^{-6}}$ and $k_t = K_t / \sqrt{\sum K_{t,j}^2 + 10^{-6}}$, compute $\beta_t = \sigma(\beta_t^{\text{raw}})$ and $d_t = \exp(\text{lower\_bound} \cdot \sigma(e^{A_{\log}}(g_t + \text{dt\_bias})))$.
   - Advance $S_t$ in on-chip FP32 UB without modifying the GM checkpoint `recurrent_state`, and write compact replay records: `replay_u` (`FP16 [B, 8, N, 128]`), `replay_k` (`FP16 [B, 8, N, 128]`), and `replay_decay` (`FP32 [B, 8, N, 128]`).
   - Conversely, in `fused_recurrent_kda_snapshot` (`fused_recurrent_kda`), write full per-token state snapshots back to the state pool.
2. **Accepted-Token State Commit (`commit_recurrent_kda_replayssm`)**:
   - After speculative verification accepts `num_accepted_tokens[b]` tokens for batch $b$, run `commit_recurrent_kda_replayssm` to update `recurrent_state` in-place via $S \leftarrow S \odot \text{decay} + U K^\top$ over $t \in [0, \text{num\_accepted\_tokens}[b])$.
3. **Stream Core-Quota Adaptation (`mega_recurrent_kda`)**:
   - Read the active stream's AIC quota without changing the 20-parameter public signature; clamp quotas $> 32$ to `32`, validate membership in `{8, 16, 24, 28, 32}` (`{8, 24, 28, 32}` for Commit), and verify `AIV_quota >= 2 * AIC_quota`.

## Verification & Checks

1. **Sign & Scale Folding in CP Tensors**:
   - Verify in `flash_kda_cp_main` that `qg` already incorporates `scale`, `w` already incorporates the negative sign (`-w`), and `mqk` already incorporates the lower-triangular mask from `flash_kda_cp_preprocess`—never double-apply scale or negate `w` a second time.
2. **TP vs Operator Batch Dimension Mapping (`mega_recurrent_kda`)**:
   - Verify that operator input `[B, 8, 7168]` has $B = \text{model\_batch} \times \text{TP}$ and $N = 96 / \text{TP}$ (`TP=4 -> N=24`, `TP=8 -> N=12`, `TP=16 -> N=6`).

## Failure Modes & Fallbacks

- **Symptom**: Numerical blow-up (`Inf`/`NaN`) in $\gamma^{-1} = \exp(-G_t)$ within a 64-token chunk when gate decay $g_t$ is strongly negative.
  - **Root Cause**: Unshifted cumulative log-decay $G_t$ underflows $\gamma_t$ and overflows $\gamma_t^{-1}$ in exponentiation.
  - **Fix**: Stabilize intra-chunk relative decay differences $G_i - G_j$ for $i \ge j$ so exponents remain non-positive ($\le 0$) inside the lower-triangular mask.
- **Symptom**: Corrupted `recurrent_state` after rejected speculative tokens in ReplaySSM.
  - **Root Cause**: Updating `recurrent_state` in-place during the verify pass (`mega_recurrent_kda_replayssm` / `fused_recurrent_kda_replayssm`) instead of buffering `replay_u`, `replay_k`, and `replay_decay` for `commit_recurrent_kda_replayssm`.
  - **Fix**: Keep `recurrent_state` strictly read-only during verify and mutate it only in `commit_recurrent_kda_replayssm` up to `num_accepted_tokens[b]`.
