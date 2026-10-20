---
schema: mindie-graded-unit/1
id: unit-ascend-grouped-matmul-swiglu-and-mc2-fusion-sop
revision: 1
tier: L2
category: sop
domain_path: fusion/moe-gmm
evidence_strength: reproduced_empirical
title: 'GroupedMatmul, SwiGLU and MC2 fusion SOP: split contracts, zero-token experts, NZ weights and Cube/Vector profiling'
conditions:
  op_family:
  - grouped_matmul
  - moe
  - swiglu
  - matmul
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  source_type: reproduced_operator_integration_and_public_contract_review
  source_locator: 'GroupedMatmul aclnn V4/V5 and grouped_matmul_swiglu_quant contracts; MoE GMM/SwiGLU/W8A8/W4A8/W4A16/MC2 integration transcripts'
  evidence_refs:
  - https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md
  - https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/GroupedMatmul算子设计介绍.md
  - https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul_swiglu_quant_v2/docs/torchapi_grouped_matmul_swiglu_quant.md
  - https://gitcode.com/cann/cann-recipes-infer
  - unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
  supersedes: []
triggers:
  error_signatures:
  - group_list_type=0 is passed to the kernel but the list contains per-expert counts, so every expert after the first reads the wrong segment
  - a zero-token expert reads the previous expert's tokens or produces NaN in the dynamic-quant scale
  - a small-M tail (M_e < 16) underfills the Cube tile and the last rows are stale or unmasked
  - GMM reads a K-dim-packed int4 weight and returns a layout-mismatch error or wrong values
  - the unfused path materializes separate gate and up tensors in GM and doubles the activation traffic
  - Cube is starved on weights while Vector is idle, or the SwiGLU epilogue makes the chain Vector-bound
  keywords:
  - GroupedMatmul
  - GMM
  - npu_grouped_matmul
  - split_item
  - group_list_type
  - group_list
  - MoE
  - SwiGLU
  - npu_swiglu
  - dynamic quant
  - W8A8
  - W4A8
  - W4A16
  - NZ
  - Fractal_NZ
  - QuantBatchMatmul
  - MC2
  - all_gather_base_mm
  - mm_reduce_scatter_base
  - alltoallv_gmm
  - gmm_alltoallv
  - msprof
  - PipeUtilization
---

# GroupedMatmul, SwiGLU and MC2 fusion SOP: split contracts, zero-token experts, NZ weights and Cube/Vector profiling

## Use when

A MoE FFN on Ascend 910B, 910_93 or 950 computes per-expert `gate_up` and `down` projections with `torch_npu.npu_grouped_matmul` (GMM), optionally fused with `npu_swiglu`, dynamic per-token quantization, or an MC2 collective. The SOP covers the tensor contract, the fused GMM1→SwiGLU→quant→GMM2 chain, zero-token and small-M tails, weight layout conversion, and the Cube/Vector/MTE2 profile. It does not cover the router or the dispatch/combine protocol except where the GMM consumes their `group_list`.

Freeze: SOC, CANN/`torch_npu` revision, expert count `E`, hidden size `K`, intermediate size `N`, token count `M`, per-expert counts, dtype (bf16/fp16/int8/int4/fp8), quantization scheme (W8A8, W4A8, W4A16, MXFP8), weight layout (`ND`/`NZ`), `split_item`, `group_list_type`, EP/TP topology, and whether MC2 is enabled. A missing fact is `unknown`; the GMM contract differs by path.

## 1. Pin the `split_item`, `group_type` and `group_list_type` contract

`npu_grouped_matmul` is a family of templates selected by `split_item`, `group_type`, `group_list_type` and the dtype tuple. Mixing the numbers is the most common silent error.

| Field | Value | Contract |
| --- | --- | --- |
| `split_item` | `0` / `1` | Output is a TensorList, one tensor per group. Use when each expert's result stays separate. |
| `split_item` | `2` | Single concatenated output, typically `x [M,K]`, `weight [E,K,N]`, `y [M,N]`; the common MoE GMM form. |
| `split_item` | `3` | Single-output template, used by quantized GMM examples where the weight is a TensorList or a single 3D tensor. The public contract does not distinguish `2` and `3` internally; follow the installed example for the dtype tuple. |
| `group_type` | `0` | M-axis grouping: `K` and `N` are shared, `M_i` may differ. The MoE forward case. |
| `group_type` | `2` | K-axis grouping: `M` and `N` are shared, `K_i` may differ. Typically a backward/weight-gradient case, not MoE forward. |
| `group_list_type` | `0` | Cumulative sum (`cumsum`): non-negative, monotonic non-decreasing, last value `<= M`. |
| `group_list_type` | `1` | Per-expert count: non-negative; `sum(counts) == M` for a full partition (some templates allow `<= M` when trailing rows are unused). The form returned by dispatch as `expert_token_num`. |
| `group_list_type` | `2` | `[group_idx, group_size]` pairs, shape `[E,2]`; used by some full-quant paths. Do not pass a count vector here. |

`group_list` is an `int64` device tensor. Some Python wrappers accept a host list and convert it internally, but the operator contract is the device tensor, and a graph-captured call needs the tensor form with a stable address. A host Python list must be converted once, outside the hot path, and the resulting tensor must keep its address for graph capture. The dtype is part of the contract: an `int32` count tensor is not silently widened by the operator.

```python
import torch, torch_npu
M, E = 4096, 128
counts = torch.tensor([0, 1, 7, 120] + [32] * (E - 4), dtype=torch.int64, device="npu")
assert int(counts.sum()) == M, "group_list_type=1 counts must sum to M"
cumsum = torch.cumsum(counts, dim=0)
assert bool((cumsum[1:] >= cumsum[:-1]).all()) and int(cumsum[-1]) == M
# both encodings describe the same partition; never mix them
torch_npu.npu_grouped_matmul(
    [x], [weight], group_list=counts, split_item=2,
    group_type=0, group_list_type=1,
)
```

**Invariants.** `sum(counts) == M` for a full partition (if the selected template allows unused trailing rows, record the gap explicitly); `cumsum` is monotonic and ends at `M` for type 0; `group_list.numel() == E`; `group_list.dtype == torch.int64`; the output partition matches the input partition; the `TilingData` group count equals the runtime list length; `split_item` and the output tensor list length agree (`0/1` -> length `E`; `2/3` -> length `1`).

**Verification.** Build the same partition in both encodings and compare the GMM output byte-for-byte. Then compare against a per-expert Python loop on a small case. A pass on type 0 with a type-1 list is not a contract pass; it is a coincidence of the first group.

## 2. Fuse GMM1 → SwiGLU → dynamic quantization → GMM2

The unfused chain materializes `gate_up [M, 2I]`, then splits it into `gate [M,I]` and `up [M,I]`, applies `silu(gate) * up`, quantizes, and calls GMM2. The two separate `gate` and `up` tensors are extra GM traffic and extra launches; the fused chain consumes the `[M,2I]` buffer once and writes only the quantized activation and its per-token scale.

The math is:

```text
gate_up = GMM1(x, W_gate_up)                  # [M, 2I]
gate, up = gate_up[..., :I], gate_up[..., I:]
h = silu(gate) * up                           # SwiGLU
h_q, h_scale = dynamic_quant(h, 127 / max(abs(h), eps))
y = GMM2(h_q, W_down, scale=h_scale)          # dequantize in the epilogue
```

`torch_npu.npu_swiglu` implements the split-and-gate form; use its declared `dim` and do not pre-split the tensor in Python. The unclamped conceptual form is `silu(gate) * up`; the A5 fused operator applies the declared `clamp_limit`, `glu_alpha` and `glu_bias`, so its exact value can differ from the plain form. On A5, `cann_ops_transformer.grouped_matmul_swiglu_quant` fuses GMM, dequant, SwiGLU and MXFP8 quantization; it requires `swiglu_mode=2` and a single `weight` tensor in FRACTAL_NZ, with `clamp_limit` (default 7.0), `glu_alpha` (default 1.702) and `glu_bias` (default 1.0). On A2/A3, keep GMM1 and GMM2 separate and put the SwiGLU and quant in the GMM1 epilogue or a UB-resident vector stage; do not write `gate` and `up` as separate GM tensors.

```python
gate_up = torch_npu.npu_grouped_matmul(
    [x], [w13], group_list=counts, split_item=2, group_type=0, group_list_type=1,
)
h = torch_npu.npu_swiglu(gate_up[0], dim=-1)
h_q, h_scale = torch_npu.npu_dynamic_quant(h)   # per-token int8
y = torch_npu.npu_grouped_matmul(
    [h_q], [w2], scale=[w2_scale], per_token_scale=[h_scale],
    group_list=counts, split_item=2, group_type=0, group_list_type=1,
)
```

**Invariants.** The `[M,2I]` buffer is consumed once; no standalone `gate`/`up` GM tensor appears in the allocation trace; the SwiGLU split axis is the last axis and the two halves are equal; the dynamic-quant scale is per token and fp32; the GMM2 `per_token_scale` length equals `M`; the dequantization is applied exactly once (in the GMM2 epilogue or after it, not both).

**Verification.** Compare the fused chain against the explicit gate/up reference on a small case. Count `TensorMove`/`DataCopy` tasks and GM allocations before and after; the fused path should remove the `gate` and `up` writes, not merely rename them. On A5, confirm the fused operator's `swiglu_mode=2`, clamp and alpha match the model's activation; a default clamp can change the output.

## 3. Handle zero-token experts and small-M tails without NaN or stale rows

A pure-EP decode can route zero tokens to some experts. `expert_token_num` may be all zeros on a rank, and the GMM is still called (the dispatch path has no `sum() > 0` guard). The kernel must handle an empty group without reading `x` and without producing a NaN that contaminates the combine.

- **Zero-token group.** For `group_list_type=1`, a zero count is legal. The group's output segment is empty; the kernel must not read the previous group's `x` or write into another group's output. For `group_list_type=0`, the cumulative value does not advance; the same rule applies.
- **Small-M tail.** `M_e < 16` underfills a 16-row Cube tile. Pad `M_e` to the tile with a defined value, mask the padded rows before the store, and keep the dynamic-quant scale computed only over the real rows. A 512 B-misaligned L1 tile base is out of contract even if the M tail is small.
- **Dynamic-quant scale.** Use `scale = 127.0 / max(abs(x), eps)` with `eps > 0`; define the all-zero row explicitly (`scale = 0` with a zero output, or `scale = 1`). A `0/0` gives `NaN`; a huge finite scale with a zero numerator gives `0`, which is safe, but a `NaN` in the scale propagates through GMM2 and the combine. The dequantization counterpart is `x = x_q * scale`.
- **Determinism.** The same input and counts must produce the same bytes. A split-K or atomic accumulation order that changes with the schedule is not a valid fix for a tail race.

```python
counts = torch.tensor([0, 0, 1, 3, 5, 7, 9, 11], dtype=torch.int64, device="npu")
assert int(counts.sum()) == 36
# reference: skip empty groups, compute only real rows
offset = 0
for e, n in enumerate(counts.tolist()):
    if n == 0:
        continue
    x_e = x[offset:offset + n]
    ...
    offset += n
```

**Invariants.** An empty group contributes no output rows; `sum(counts) == M`; padded rows are masked before the store; the dynamic-quant scale is finite on every real row; the zero-row policy is explicit and identical in the fused and unfused paths; the output offset for group `e` is `cumsum(counts)[e-1]`.

**Verification.** Run a case with a zero-token expert and a case with `M_e ∈ {1, 7, 15, 16, 17}`. Compare against the per-expert loop, check `torch.isfinite` on the output and the scales, and run the same input three times. A finite output with the wrong row mapping is still a failure; check the row offsets, not only the max error.

## 4. Convert weights to the layout the GMM kernel actually reads

GMM reads `FRACTAL_NZ` (`format=29`) more efficiently than `ND`; the conversion belongs in `process_weights_after_loading`, not in the hot path. The path-specific rules are:

- **Unquantized GMM:** cast the expert weight directly to NZ: `torch_npu.npu_format_cast(w, 29)`. Keep the original pointer only if the framework needs it; release the duplicate to avoid resident double storage.
- **W8A8 GMM:** the expert weight is transposed and made contiguous before the NZ cast (`transpose(1, 2).contiguous()` then `npu_format_cast(..., 29)`); the unquantized branch does not transpose.
- **W4A8:** weight is 4-bit packed two per int8, with a per-expert float bias/assist matrix and per-channel scale; the GMM takes the assist matrix as `bias` and outputs bf16. The per-expert `alpha` clip is part of the accuracy contract, not an optional tune.
- **W4A16:** the int4 antiquant GMM accepts **N-dim packed** weights, while compressed-tensors commonly ships **K-dim packed** weights. Repack after load: `unpack_from_int32(..., 4)` -> `transpose(1, 2)` -> `pack_to_int32` / `npu_convert_weight_to_int4pack`, and transpose the per-group scale/offset with the same axes. A K-dim-packed weight fed to the N-dim kernel is a layout mismatch, not a numerical tolerance issue.
- **`QuantBatchMatmul`:** the same NZ preference applies; a weight converted for GMM is not automatically valid for a different quantized matmul template. Convert per operator and keep the format tag with the tensor.

```bash
python3 - <<'PY'
import torch, torch_npu
# one expert weight [K,N]; the GMM wrapper supplies the expert axis
w = torch.randn(512, 256, dtype=torch.bfloat16, device="npu")
w_nz = torch_npu.npu_format_cast(w, 29)
print("format:", torch_npu.get_npu_format(w_nz), "shape:", tuple(w_nz.shape))
# the NZ tensor keeps the logical [K,N] view; the kernel reads the fractal storage
PY
```

**Invariants.** The format tag is the one the selected GMM template expects; the logical shape and the storage shape are both recorded; a transpose is applied only on the branch that requires it; the scale/offset axes match the weight axes after repacking; the original weight is released or retained by an explicit policy, not leaked.

**Verification.** Compare NZ and ND GMM outputs on the same weights (same math, different layout); then compare against the per-expert reference. Check `msprof` `aic_mte2_time` before and after the conversion. A faster kernel with a wrong `N`/`K` interpretation is a layout bug; verify the logical output shape before timing.

## 5. Fuse the collective with GMM only when the MC2 contract holds, then profile the chain

MC2 fuses a collective with the adjacent matmul in one operator so the communication and Cube work overlap. The forms are:

- **TP linear:** `npu_all_gather_base_mm` (AllGather + matmul) for the column-parallel forward, `npu_mm_reduce_scatter_base` (matmul + ReduceScatter) for the row-parallel forward. Requires TP > 1 and sequence parallelism; the fused matmul does not take a bias.
- **MoE EP:** `npu_alltoallv_gmm` (All2All-v + GMM) for fc1, `npu_gmm_alltoallv` (GMM + All2All-v) for fc2. Requires an EP All2All dispatcher and a valid `hcom` name; the send/recv counts must match the dispatch.

MC2 is not a drop-in for every shape. It is mutually exclusive with the framework's software CoC overlap, with some pipeline/overlap features, and some MC2 paths exclude A3. Check the installed operator's platform support and the feature-gate list before enabling it. If the collective and the matmul are not adjacent, or the EP dispatcher is not All2All, MC2 does not apply; keep the unfused path.

Profile the GMM+SwiGLU chain on the same case and clock:

```bash
msprof --application="python3 bench_moe.py --case decode --tokens 4096" \
  --output=/tmp/prof_gmm --aic-metrics=PipeUtilization,Memory,MemoryUB --l2=on
python3 - <<'PY'
import glob, pandas as pd
path = sorted(glob.glob("/tmp/prof_gmm/**/op_summary_*.csv", recursive=True))[-1]
df = pd.read_csv(path)
keep = [c for c in ["op_name","op_type","aicore_time(us)","aiv_time(us)",
                    "aic_mac_time(us)","aic_mte2_time(us)",
                    "aiv_vec_time(us)","aiv_mte2_time(us)","aiv_mte3_time(us)"] if c in df]
print(path)
print(df[keep].to_string(index=False))
PY
```

Read the columns by engine:

- `aic_mac_time` high: Cube-bound; re-tile M/N/K, check the 512 B L1 tile alignment and the NZ layout.
- `aic_mte2_time` high with `aic_mac_time` low: Cube starved on weights/activations; prefetch, increase L1 reuse, verify the NZ conversion actually ran.
- `aiv_vec_time` high: SwiGLU, dequant or quant epilogue is the limit; fuse the epilogue, reduce the transcendental work, keep the fp32 scale.
- `aiv_mte2_time` high: GM→UB input; `aiv_mte3_time` high: UB→GM output; remove intermediate `gate`/`up`/`h` GM tensors.
- `mix_comm_aiv` or a collective op: rank the AIV stages separately; do not attribute the communication wait to the GMM.

**Invariants.** MC2 is enabled only when the topology and feature gates hold; the `hcom` name matches the communication group; the fused matmul's bias policy is explicit; the EP send/recv counts are consistent with the GMM group list; the profile compares the same shape, dtype and clock as the baseline.

**Verification.** Compare MC2 and the unfused path numerically on the same input (tolerance, not bit-exact if the reduction order changes), then compare the synchronized latency. Check that the communication actually overlaps: a faster wall time with the same sum of kernel times is evidence of overlap; a faster kernel with a larger exposed collective is not. Record the `msprof` artifact and the per-stage totals. A GMM speedup that leaves the SwiGLU stage unchanged is a local result, not a chain speedup.

## Claim boundaries

This SOP is a contract-and-integration procedure. It does not assert that a specific `split_item`/`group_list_type` combination is supported on every CANN revision, that an NZ conversion is always faster, or that MC2 is available on every SOC/topology. The installed operator's page and the project's feature gates are authoritative. A per-expert equality pass is not a full-batch pass; a decode result is not a prefill result; a local GMM speedup is not an end-to-end MoE speedup. The inherited measurement contract still applies: correctness first, synchronized blocks, per-case envelope.
