---
schema: mindie-graded-unit/1
id: unit-ascend-fia-gqa-host-tiling-sink-and-bn2s-layout-mutator
revision: 1
tier: L4
category: mutator
domain_path: attention/fia-mla
evidence_strength: reproduced_empirical
title: 'FIA GQA device tiling sink, BN2S head-fold view and split-KV fp32 LSE combine'
conditions:
  op_family:
  - attention
  - fia
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
  dtype:
  - bfloat16
  - float16
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
- unit-ascend-fused-infer-attention-and-flash-mla-c8-sop
provenance:
  source_type: ablation_design_receipt
  source_locator: docs/evidence/l4-fia-gqa-host-tiling-sink-ablation.json
  evidence_refs:
  - docs/evidence/l4-fia-gqa-host-tiling-sink-ablation.json
  experiment_status: not_run_source_only
  supersedes: []
triggers:
  error_signatures:
  - ACLGraph replay still performs a host traversal of actual_seq_lengths_kv or a per-step .tolist()/.item()
  - a Transpose or TensorMove of the query is scheduled before the Cube BMM1 and its bytes equal the query size
  - only a fraction of the AIC/AIV cores are busy when B * num_kv_heads is smaller than the core count
  - split-KV partial outputs are combined in bf16, or softmax_lse is not fp32
  - a padded zero-length request causes a stale block-table row or an empty required input in the graph
  keywords:
  - FIA
  - GQA
  - host tiling sink
  - device tiling
  - host-device sync
  - actual_seq_lengths_kv
  - ACLGraph
  - BN2S
  - BNSD
  - head fold
  - transpose elimination
  - split-KV
  - softmax_lse
  - fp32 combine
  - paged attention
  - Ascend910B
  - Ascend910_93
  - Ascend950
---

# FIA GQA device tiling sink, BN2S head-fold view and split-KV fp32 LSE combine

## Applicability

Ascend 910B, 910_93 or 950, FIA (`torch_npu.npu_fused_infer_attention_score` / `aclnnFusedInferAttentionScoreV2+`) on a GQA decode or chunked-prefill path, `bfloat16` or `float16` activations, a paged KV cache, and at least one of:

- the host still traverses `actual_seq_lengths_kv` or `seq_lens` to build per-step tiling metadata, or the ACLGraph replay path re-tiles on the host;
- the query is materialized through a `.transpose(...).contiguous()` GM copy before Cube BMM1;
- `B * num_kv_heads < num_cores` and a long KV leaves most AIC/AIV cores idle.

The factors are independent and are applied in the order below. The baseline is the current FIA path with the same shapes, dtype, KV cache layout, block size, graph mode and clock. A baseline whose correctness gate fails is not a valid baseline. C8 (`int8`) KV is a separate axis: the device tiling sink can coexist with C8, but the BN2S fold and the split-KV combine must be re-validated against the C8 dequantization path rather than assumed to compose.

The three factors are separable for leave-one-out. `device_tiling_metadata_sink` changes where the tiling metadata is computed. `gqa_group_head_fold_bn2s_view` changes the logical shape handed to Cube. `adaptive_kv_split_lse_fp32_combine` changes the work partition and the reduction. A change that mixes two of them into one edit cannot attribute the result.

**Evidence boundary.** The source-level mechanisms were reproduced in the pinned FIA tiling-sink and A5 FlashMLA worktrees: the host list is removed from the sink path, the metadata is built on device, and the A5 path selects FlashMLA when `kv_lora_rank=512`, `qk_rope_head_dim=64`, `num_kv_heads=1`. The combined three-factor ablation and paired timing campaign was **not run**. This unit is a design-level transformation with an ablation receipt, not a measured speedup.

## Signals

Record the signal that motivates each factor; a factor with no signal is not applied.

- **Host tiling signal.** The per-step path contains `.tolist()`, `.item()`, a Python loop over `seq_lens`, or a fresh `torch.tensor(...)` for the length buffer. Under ACLGraph, the host gap between replays grows with `num_reqs`, or the graph is re-captured when only the sequence lengths changed. A `requires_exact_host_seq_lens`-style branch is a hint that one path still needs the host mirror while the sink path does not.
- **BN2S signal.** `msprof` shows a `Transpose`/`TensorMove` task before the first Cube task, or `aic_mte2_time` is high while the query bytes alone do not explain it. A `.transpose(1, 2).contiguous()` in the Python forward is direct evidence.
- **Split-KV signal.** `aic_mac_time` and `aiv_time` are low while wall time is long, `Block Dim` is smaller than the core count, or `B * num_kv_heads` is visibly below `GetCoreNumAic()`. Long KV with a small batch is the characteristic shape.
- **Correctness prerequisite signal.** A folded-head candidate that improves timing but changes the output usually has a mask that was not expanded by `G`, or an inverse view that writes heads in the wrong order. A split-KV candidate that is finite but drifts has a bf16 LSE or a bf16 combine.

## Transformation

Keep the correctness prerequisites (mask expansion, fp32 LSE, tail mask, valid block table) in every arm. Apply the factors separately.

1. **`device_tiling_metadata_sink`** — Remove the per-step host traversal of the sequence lengths and build the tiling metadata on the device. Keep a persistent device buffer for `actual_seq_lengths_kv` (and the cumulative query offsets where the operator needs them), update it in place from the current batch, and call the device-tiling variant (`FIA` device-tiling sink metadata or `flash_mla_metadata`) with a preallocated output buffer. The host supplies only shape/`num_reqs`; it does not read the values back. For a padded full-graph batch, give every padded request a valid block-table row and clamp its device length to at least 1; a zero-length padded request must not become an empty required input. The sink path can then set the host-mirror requirement to false, while the legacy path keeps its exact host list.

   Implementation invariants: one metadata buffer per `(num_reqs, max_q_len, max_kv_len)`; all addresses stable across replays; the metadata is regenerated when the shape contract changes; the host list is not mutated during replay; the device length buffer is `int32` or `int64` exactly as the operator schema requires. A Python list is a host constant and cannot be updated by a device-side `copy_`; if the selected operator requires a host list, this factor is not applicable.

2. **`gqa_group_head_fold_bn2s_view`** — Fold the GQA group into the inner sequence axis instead of materializing a transposed copy (`BN2S`, batch/head-to-sequence fold). For a contiguous `BNSD` query `[B, N_q, S_q, D]` with `N_q = N_kv * G`, the memory order is already `[B, N_kv, G, S_q, D]`; view it as `[B, N_kv, G*S_q, D]` and hand that to Cube BMM1. The group is now a batch dimension: `G*S_q` rows per KV head, no `.transpose().contiguous()` and no GM copy. The K/V stay `[B, N_kv, S_kv, D]`. The output is viewed back to the declared layout. If the source layout is `BSND` or `NZ`, construct the equivalent strided view only if the head and sequence strides permit it; otherwise keep the copy outside the hot path and do not claim the factor.

   The mask must be expanded with the same fold: query row `g*S_q + s` belongs to group `g` and position `s`, so its causal window is the same as the ungrouped row `s`, and a chunked-prefill mask must be tiled `G` times rather than used as a `[S_q, S_kv]` block. The inverse view must restore the original head order; a fold that reorders `n = kv*G + g` and writes back as `g*N_kv + kv` is a silent head permutation.

3. **`adaptive_kv_split_lse_fp32_combine`** — When `B * N_kv < num_cores`, split the KV axis into `S` shards so that `B * N_kv * S >= num_cores`, capped by `ceil(S_kv / min_shard)` and by the workspace budget. Each shard computes a partial `out_s` and `lse_s`; the tail shard masks the KV positions beyond `S_kv`. Combine in fp32:

   ```text
   m   = max_s(lse_s)
   w_s = exp(lse_s - m)
   out = sum_s(w_s * out_s) / sum_s(w_s)
   lse = m + log(sum_s(w_s))
   ```

   `out_s` keeps the declared output dtype, but the weights, the denominator and the LSE are fp32. An empty shard returns `out = 0`, `lse = -inf`. The workspace is `S * (T*N*D*out_bytes + T*N*4)` plus alignment, at least 32 B aligned, and the combine order is fixed so repeated runs are byte-identical. The split is not applied when `S_kv` is short, when `B * N_kv >= num_cores`, or when the operator does not expose `softmax_lse`.

   Implementation invariants: every KV position belongs to exactly one shard; the union covers `[0, S_kv)`; the tail mask is applied before the combine; the LSE is fp32; the output dtype is exact; the combine is deterministic; the workspace is sized for the maximum shard count, not the current one.

The factors are separable. Remove factor 1 for the host-tiling arm; remove factor 2 for the transpose-copy arm; remove factor 3 for the unsplit arm. Keep the mask expansion, the fp32 LSE and the tail mask in every arm; they are correctness prerequisites, not performance factors.

## Fallback boundary

- **Host list is mandatory.** If the installed backend validates `actual_seq_lengths_kv` as `SymInt[]` and cannot consume a device tensor, the device tiling sink cannot remove the host list. Keep the host list, re-capture per shape, and record that the sink factor is not applicable. Do not mutate a captured list and call it a sink.
- **Non-viewable head/sequence strides.** If `N_q != N_kv * G`, or the query is `BSND`/`NZ` with a stride that cannot express `[B, N_kv, G*S_q, D]`, do not force a view. Materialize once outside the measured region, or leave the baseline copy in place. A view with a wrong stride is a correctness bug, not a speedup.
- **Mask cannot be expanded.** If the operator's mask is precomputed for `[S_q, S_kv]` and cannot be tiled by `G`, the fold changes semantics. Fall back to the ungrouped path.
- **Short KV or enough parallelism.** If `B * N_kv >= num_cores`, or `S_kv` is below the minimum profitable shard, split-KV adds workspace and combine cost without adding occupancy. Do not apply it.
- **No LSE / bf16 LSE.** If the operator does not expose `softmax_lse`, or the only available LSE is bf16, the fp32 combine cannot be formed. Do not substitute a bf16 combine and call it equivalent.
- **Bit-exact requirement.** Split-KV changes the reduction order. If the acceptance criterion is bit-exact against a single-pass reference, the split factor is out of scope; use tolerance-based acceptance and record the boundary.

## Verification and ablation

- **Gate 1 — OracleFreezer.** Freeze the fp32 reference and every input. Cover an aligned case, a KV tail (`S_kv % block_size != 0`), a zero-token/padded request, `G = 1`, `G > 1`, and a `B * N_kv < num_cores` case. Require finiteness, max-abs and relative error inside the pre-registered tolerance, exact output dtype, unchanged inputs, and three byte-identical repeats.
- **Gate 2 — synchronized timing.** Exclude at least two warmups, synchronize immediately before and after each block, keep case identity and input hashes per sample, and report per-case block envelopes. Measure the host gap separately from the device kernel; a kernel speedup that is offset by a host gap is not an end-to-end win. For ACLGraph, report replay time separately from capture time and verify the same batch is being replayed.
- **Gate 3 — leave-one-out.** Run at least four arms: full candidate, minus `device_tiling_metadata_sink`, minus `gqa_group_head_fold_bn2s_view`, minus `adaptive_kv_split_lse_fp32_combine`. For interaction effects, add the pairwise arms. Repeat Gates 1 and 2 for every arm. A factor that changes correctness is a prerequisite, not a performance factor; do not prune it to make the timing cleaner.
- **Gate 4 — system guards and hidden retention.** Re-run the retained combination on frozen hidden inputs and on every guard workload: `B*N_kv < cores`, `B*N_kv >= cores`, `S_kv` tail not a multiple of `block_size`, `G=1` and `G>1`, `num_reqs` at the graph's padded maximum, a zero-length padded request, workspace high-water mark, graph capture/replay shape stability, warm-server repeated call, and a C8 KV variant if C8 is in scope. Classify each hidden case as `both_pass`, `opt_regression`, `both_fail`, `opt_improvement` or `evaluation_error`; only comparable `both_pass` cases enter a speedup ratio. A local gain never overrides a guard regression.
- **Provenance.** This unit is a design-level mutator. The linked receipt records the factors, arms, guards and fallback boundaries with `experiment_status: not_run_source_only`; it establishes no measured speedup. Promote the unit only after the complete final combination, hidden inputs and guards are independently verified, and keep the failed arms in the receipt.
