---
schema: mindie-graded-unit/1
id: unit-ascend-fused-infer-attention-and-flash-mla-c8-sop
revision: 1
tier: L2
category: sop
domain_path: attention/fia-mla
evidence_strength: reproduced_empirical
title: 'FIA and FlashMLA five-step SOP: TND/BNSD layout, paged KV, C8 dequant, MLA split-RoPE and graph-stable tiling'
conditions:
  op_family:
  - attention
  - fia
  - flash_mla
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
provenance:
  source_type: reproduced_operator_integration_and_public_contract_review
  source_locator: 'FIA aclnn V2/V5 and flash_mla operator contracts; A5/A3 attention integration transcripts (a5-fia, fia_gqa, flash_mla, C8 KV cache)'
  evidence_refs:
  - https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md
  - https://gitcode.com/cann/ops-transformer/blob/master/attention/flash_mla/docs/torchapi_flash_mla.md
  - https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/SKILL.md
  - https://gitcode.com/cann/cann-recipes-infer
  - unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
  supersedes: []
triggers:
  error_signatures:
  - actual_seq_lengths_kv is passed as a Tensor on the torch.ops.npu/npugraph_ex path and fails schema validation
  - block_table first dimension does not equal num_reqs and Ascend950 FIA rejects the shape
  - a paged KV cache with a hole or an out-of-range block index reads another request's data
  - C8 int8 KV cache returns NaN or a systematically scaled output because the antiquant scale axis follows the wrong layout
  - MLA decode passes a merged 576-dim query to an operator that expects split nope/rope inputs
  - split-KV partial outputs are combined in bf16 and the result drifts at long context
  - ACLGraph replay reuses a host list or a stale metadata buffer after the sequence lengths change
  keywords:
  - FIA
  - FusedInferAttentionScore
  - flash_mla
  - FlashMLA
  - TND
  - NTD
  - BNSD
  - BSND
  - BSH
  - GQA
  - paged attention
  - block_table
  - block_size
  - actual_seq_lengths
  - actual_seq_lengths_kv
  - C8
  - int8 KV cache
  - antiquant_scale
  - antiquant_offset
  - per-channel
  - per-tensor
  - MLA
  - kv_lora_rank
  - qk_rope_head_dim
  - query_rope
  - key_rope
  - softmax_lse
  - split-KV
  - ACLGraph
---

# FIA and FlashMLA five-step SOP: TND/BNSD layout, paged KV, C8 dequant, MLA split-RoPE and graph-stable tiling

## Use when

An attention path on Ascend 910B, 910_93 or 950 uses `torch_npu.npu_fused_infer_attention_score` / `aclnnFusedInferAttentionScoreV2/V5` or `cann_ops_transformer.flash_mla`, with a paged KV cache, GQA, MLA latent KV, C8 (`int8`) KV cache, or a split-KV reduction. The SOP covers the host-side contract and the graph-capture contract; it does not replace the operator's own tiling diagnostics. An operator whose layout or quantization mode is not in the installed CANN's supported set is out of scope, and a path that only works in eager mode is not a graph-stable path.

Freeze these facts before choosing a branch: SOC, CANN/`torch_npu` revision, operator and version (`FIA v1/v2/v5`, `flash_mla`), `input_layout`, query/KV shapes and strides, `num_heads`, `num_kv_heads`, `head_dim`, `block_size`, `num_blocks`, max KV length, KV dtype, scale dtype, execution mode (eager / `torch.compile` / ACLGraph), and whether the length inputs are host lists or device tensors. A missing fact is `unknown`; do not infer it from another model.

## 1. Select the input-layout contract and validate the GQA ratio

Use one layout family per call and keep the layout through the whole path. The operator accepts a set of layouts; the model's KV cache layout and the attention layout must agree.

| Workload | Layout | Shape contract | When to use |
| --- | --- | --- | --- |
| Ragged prefill / chunked prefill | `TND` | query `[T, N_q, D]`, key/value `[T_kv, N_kv, D]`; `actual_seq_lengths` is the cumulative query offset with length `B`, non-decreasing, last element `T`; `actual_seq_lengths_kv` is cumulative for a TND key/value layout (last element equals the KV token count) and per-request for a non-TND layout | Variable-length prompts, no padding, no per-request loop |
| MLA prefill (non-absorb) | `NTD` or `TND_NTD` | `[N, T, D]` for the FA view; cumulative offsets as above | Avoids a TND transpose/bandwidth loss on the prefill path; confirm the installed operator supports it |
| Decode, paged `BnNBsD` cache | `BNSD` | query `[B, N_q, S_q, D]`, KV cache `[num_blocks, N_kv, block_size, D]` | Preferred paged decode layout |
| Decode, NZ/fractal cache | `BSND` | query `[B, S_q, N_q, D]`, cache viewed as `[num_blocks, N_kv, D/D0, block_size, D0]` | KV cache is `PA_NZ`; `D0` is 16 for bf16 and 32 for int8 |
| Decode, packed heads | `BSH` | query `[B, S_q, N_q*D]` | The model already stores heads packed and no per-head view is needed |

The basic GQA contract is `num_heads % num_kv_heads == 0`; `num_key_value_heads` must be passed explicitly from the KV tensor's head dimension. If it is omitted, the operator can degrade the KV head count to the query head count and produce a semantically wrong result without a shape error. Two additional constraints are path-specific and must be checked before capture:

- **MLA in torchair graph mode (observed on A2/A3):** the per-rank `num_heads / num_kv_heads` must be in `{32, 64, 128}`. This is a FIA/MLA shape contract, not a framework preference; re-check the installed operator on A5 before reusing the same set.
- **MLA `NO_QUANT` + rope + query/key head_dim 512:** `group_num` must be a power of two in `{1, 2, 4, 8, 16, 32, 64, 128}`. A value such as 5 fails with `561002` / `CheckFeatureMlaNoQuantShape`, often during graph capture before the first request.

```bash
python3 - <<'PY'
B, Nq, Nkv, D = 4, 32, 8, 128
assert Nq % Nkv == 0, "GQA requires num_heads % num_kv_heads == 0"
G = Nq // Nkv
q_lens = [7, 1, 33, 128]
kv_lens = [64, 1, 96, 256]
assert all(x >= 0 for x in q_lens) and sum(q_lens) > 0
assert all(x > 0 for x in kv_lens)
cu_q = [0]
for x in q_lens:
    cu_q.append(cu_q[-1] + x)
assert cu_q[-1] == sum(q_lens)
print("G =", G, "cu_q =", cu_q, "last =", cu_q[-1])
PY
```

**Invariants.** `T == sum(actual_seq_lengths)` on the TND path; `actual_seq_lengths` is non-decreasing and ends at `T`; `actual_seq_lengths_kv` has one entry per request, is non-decreasing with a last element equal to the KV token count on a TND key/value layout, and is each `<= max_kv_len` on a non-TND layout; `query.shape[0] == T` for TND; `query.shape[1] == num_heads` for BNSD/BSND; `block_table.shape[0] == B == len(actual_seq_lengths_kv)`; the KV cache's `N_kv` equals `num_kv_heads`.

**Verification.** Run a single-request ragged case and a multi-request case with different `q_lens` and `kv_lens`. For TND, assert the last cumulative offset equals the physical query token count. For decode, assert the KV head ratio and the layout-specific stride. A shape pass is not a correctness pass; keep the fp32 reference for step 4.

## 2. Build a valid paged block table and pin the length-input contract

The paged contract is small but strict:

- `block_table` is `int32`, shape `[B, max_blocks]`, contiguous, and every entry is a valid physical block index in `[0, num_blocks)`. The operator does not accept a hole; a logical page with no data must point to a real null/placeholder block, not `-1`.
- `block_size` is the number of tokens per physical block. The common supported set is `{16, 32, 64, 128}`; some C8/NZ paths restrict it further (for example 128 or 512), so read the installed operator's page before assuming a value.
- `max_blocks >= ceil(max_kv_len / block_size)`. A request with `kv_len = 0` still needs a valid block-table row when the graph shape includes it; clamp its device-side length to at least 1 and point it at a real placeholder block.
- `actual_seq_lengths_kv` is mandatory for paged attention. Its length must equal the actual batch, and on Ascend 950 the stricter check is `num_reqs == block_table.shape[0]`. A padded request added for graph capture must not leave the block table shorter than the padded request count.
- The host/device type depends on the backend: `torch.ops.npu` and `npugraph_ex`/`torch.compile` validate `actual_seq_lengths_kv` against a `SymInt[]` schema and require a **host Python list**; `torchair`/`ge_graph` accepts a **device Tensor**. The same split applies to cumulative query offsets in some MLA implementations. Bind the operator entry and the length form together at initialization; do not assume the two backends are interchangeable.

```bash
python3 - <<'PY'
import math
B, max_kv, block_size, num_blocks = 4, 4096, 128, 512
max_blocks = math.ceil(max_kv / block_size)
assert max_blocks <= num_blocks
table = [[(b * max_blocks + i) % num_blocks for i in range(max_blocks)] for b in range(B)]
assert all(0 <= x < num_blocks for row in table for x in row)
for b, row in enumerate(table):
    for i in range(max_blocks):
        assert row[i] >= 0
print("max_blocks =", max_blocks, "rows =", len(table))
PY
```

For a graph, allocate the block table at its maximum shape once and update its contents in place. A fresh `torch.tensor(...)` per step changes the address and forces a re-capture or a stale pointer. The same rule applies to the length buffers: use a persistent `int32`/`int64` buffer whose dtype matches the operator schema, copy the current values into it, and keep the buffer alive across replays.

**Invariants.** `block_table.dtype == torch.int32`; `block_table.shape == (B, max_blocks)`; every entry is in range; `max_blocks` covers the longest KV; `len(actual_seq_lengths_kv) == B`; the device length buffer has a stable address; padded requests have a valid block row and a length of at least 1.

**Verification.** Compare a paged decode against a dense reference at the same lengths. Then change one request's KV length and replay: if the output does not change, either the length buffer was not updated or the graph captured a stale metadata buffer. Capture and replay twice, and check the block-table pointer is identical across replays.

## 3. Apply the C8 (`int8`) KV dequantization contract on the correct axis

C8 stores the KV latent in `int8` and keeps the RoPE component in the model dtype. The quantization is not merely a dtype cast: the write path, the prolog, and the attention operator must use the same scale convention and the same layout.

**Scale direction.** The cache stores an unscaled value; the write-side operator receives the reciprocal of the scale, and the read-side dequantization receives the scale itself. A single per-tensor `ckv_scale` should drive both sides. Applying the scale twice, or applying it on one path and not the other, produces a systematically scaled output that still looks finite.

**Mode and shape.** The FIA inputs are `antiquant_scale` (or the split `keyAntiquantScale`/`valueAntiquantScale`) and the optional `antiquant_offset`. `keyAntiquantMode` and `valueAntiquantMode` select the scale layout: `0` per-channel/per-tensor, `1` per-token, `2` per-tensor + per-head, `3` per-token + per-head, `4`/`5` paged scale management, `6` per-token-group. The scale and offset must follow the **key/value layout**, not the query layout:

| `input_layout` | per-channel `keyAntiquantScale` | per-channel `valueAntiquantScale` |
| --- | --- | --- |
| `BSH` | `[H]` | `[H]` |
| `BNSD` | `[N, 1, D]` | `[N, 1, D]` |
| `BSND` | `[N, D]` | `[N, D]` |

For a per-tensor mode, the scale is a single scalar (or one scalar per head) and must broadcast across `D`; passing a per-channel `[N,D]` tensor for a per-tensor mode, or the reverse, produces a finite but wrongly scaled result. For per-token modes, the scale's sequence dimension must cover `maxBlockNumPerSeq * blockSize`; for a paged scale buffer it must be at least the block-table row capacity. `antiquant_offset` follows the same shape rule and is optional only when the operator's mode says it is. For the NZ int8 path, the fractal last dimension is 32 for the KV cache and 16 for `keyRope`; reusing the bf16 `D0 = 16` for the KV cache splits the tensor on the wrong axis.

**Packed C8 buffer.** When an MLA/SFA prolog consumes the cache as one buffer, the last dimension is `kv_lora_rank + 2*qk_rope_head_dim + scale_bytes`; for `kv_lora_rank=512`, `qk_rope_head_dim=64`, four fp32 scales per 16-element group, that is `512 + 128 + 16 = 656`. The separate `rope_cache` is then `None`, and every downstream `.narrow`/`.view` on it needs a `None` guard. The FIA v2 C8 path uses `query_quant_mode=3` and `dequant_scale_key/value=ckv_scale`; the non-C8 path passes `None` and `query_quant_mode=0`. FIA v1 cannot consume an int8 KV cache.

**Dynamic quantization.** For a per-token activation, the common formula is `scale = 127.0 / max(abs(x), eps)` and `x_q = round(x / scale)`; the zero-max row needs an explicit policy (`scale = 0` and a zero output, or `scale = 1`), never a division by zero that yields `NaN`. The dequantization counterpart is `x = x_q * scale`. A scale computed in fp16 and a quantized value rounded in a different mode can differ by one integer step; keep the scale in fp32 and use the operator's declared rounding mode.

```bash
python3 - <<'PY'
import torch
torch.manual_seed(0)
x = torch.randn(4, 512, dtype=torch.float32)
eps = 1e-6
scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
deq = q.float() * scale
print("max_abs_err =", (deq - x).abs().max().item())
assert torch.isfinite(deq).all()
# per-channel scale must broadcast on the D axis, not on the head axis
w = torch.randn(8, 128, dtype=torch.float32)
s = 127.0 / w.abs().amax(dim=0).clamp_min(eps)   # [D]
assert s.shape == (128,)
print("per-channel scale shape:", tuple(s.shape))
PY
```

**Invariants.** Cache dtype is `int8` (or the operator's explicit C8 dtype); RoPE cache is the model dtype; scale dtype is fp32/bf16 as declared; the scale's axes follow the KV layout; the write-side reciprocal and the read-side scale are paired; the packed last dimension matches the formula; `rope_cache is None` has guards; no `NaN` on a zero row.

**Verification.** Dequantize a small block and compare against the original bf16 values on the aligned prefix; then compare the full C8 attention output against the bf16 path. Check per-channel and per-token modes separately. A finite result with a constant scale factor is a scale-direction bug, not a tolerance issue.

## 4. Feed MLA split nope/rope correctly and combine split-KV outputs in fp32

MLA has two operator forms and they are not interchangeable:

- **FIA split form:** `query` is the NoPE part (`D = kv_lora_rank`, typically 512), `query_rope` is the RoPE part (`D = qk_rope_head_dim`, typically 64), `key` is the NoPE cache (`D = 512`), `key_rope` is the RoPE cache (`D = 64`), and `value` is the NoPE cache. Pass the four inputs separately; do not `cat` them into one 576-dim tensor for this path. The operator consumes the RoPE part as a separate input and includes it in the QK score; the model still applies RoPE before the call.
- **FlashMLA merged form:** `q` is already the merged 576-dim tensor, `k_cache` carries both Key (all 576 dims) and Value (first 512 dims), and there are no separate `v`, `q_rope` or `k_rope` inputs. FlashMLA does not perform KV compression or RoPE; the preprocessing must have done it. `block_size = 128`, `num_heads_kv = 1`, `head_dim_qk = 576`, `head_dim_v = 512`, `layout_q = TND`, `layout_kv = PA_BBND` or `PA_NZ`, `layout_out = NTD`.

`softmax_lse` is an fp32 tensor. Its shape is `[B, N, Q_S, 1]` for a non-TND layout and `[T, N, 1]` for TND; FlashMLA returns `[Q_N, Q_T]` when `return_softmax_lse=True` and an empty `(0,)` tensor otherwise. Keep the LSE in fp32 even when the output is bf16.

For a split-KV (or split-context) reduction, each shard produces `(out_i, lse_i)`. Combine in fp32 with the online-softmax rule, not by averaging:

```text
m    = max_i(lse_i)
w_i  = exp(lse_i - m)
den  = sum_i(w_i)
out  = sum_i(w_i * out_i) / den
lse  = m + log(den)
```

An empty shard contributes `out = 0` and `lse = -inf`; `exp(-inf) = 0`, so it is the neutral element and needs no special case downstream. A tail shard whose KV block is partially valid must be masked **before** the combine; masking only the LSE while leaving the output unmasked still pollutes the result. The combine order should be fixed so repeated runs are byte-identical; an unordered atomic reduction is not a deterministic combine.

The softmax scale is the model's declared value. If derived for the absorbed latent path, it is `(kv_lora_rank + qk_rope_head_dim) ** -0.5`; for the non-absorbed path use `(qk_nope_head_dim + qk_rope_head_dim) ** -0.5`. A YaRN `mscale` correction applies only when `rope_type != "default"`. The MLA KV-LoRA scale is applied on the query side, with the cache left unscaled, so the prefill and decode write paths do not disagree on the cache scale.

```bash
python3 - <<'PY'
import torch
# two split-KV shards, fp32 combine
out0 = torch.randn(8, 1, 512, dtype=torch.float32)
out1 = torch.randn(8, 1, 512, dtype=torch.float32)
lse0 = torch.randn(8, 1, 1, dtype=torch.float32)
lse1 = torch.randn(8, 1, 1, dtype=torch.float32)
m = torch.maximum(lse0, lse1)
w0, w1 = torch.exp(lse0 - m), torch.exp(lse1 - m)
out = (w0 * out0 + w1 * out1) / (w0 + w1)
lse = m + torch.log(w0 + w1)
assert lse.dtype == torch.float32 and out.shape == out0.shape
print("lse shape:", tuple(lse.shape), "dtype:", lse.dtype)
PY
```

**Invariants.** `query.shape[-1] == kv_lora_rank` and `query_rope.shape[-1] == qk_rope_head_dim` on the split path; `q.shape[-1] == kv_lora_rank + qk_rope_head_dim` on the merged path; `value` is the NoPE cache; `softmax_lse.dtype == torch.float32`; every shard has the same output shape and dtype; an empty shard is `(0, -inf)`; the tail block is masked before the combine.

**Verification.** Compare the split-KV result against a single full-KV call at the same lengths; compare the LSE against `torch.logsumexp` of the fp32 logits. Run a shard count of 1, 2 and a non-divisor (for example 3) so the tail shard is exercised. Check that changing the combine order does not change the bytes.

## 5. Make the A5/A3 tiling workspace and the ACLGraph replay address-stable

The A5 and A3 paths differ in where the tiling metadata comes from and how long it lives:

- **A5 FlashMLA / device tiling sink.** The metadata (`cu_seqlens_q`, `cache_seqlens`, the FlashMLA scheduling buffer) is produced on the device from device-side sequence lengths. Allocate one persistent set of buffers per `num_reqs`, copy the current lengths into them with `copy_`, and call the metadata builder with the preallocated output buffer. The A5 FlashMLA path requires `kv_lora_rank == 512`, `qk_rope_head_dim == 64` and `num_kv_heads == 1`; do not reuse an A2/A3 FIA v2 metadata buffer for it. The A5 workspace follows the arch35 contract: 32 B alignment, a 16 MB system reserve, and the framework accessor rather than a raw pointer.
- **A3 / A2 FIA device-tiling sink.** The device tiling path keeps the length metadata on the device and removes the per-step host traversal. The legacy path may still require a host list for `actual_seq_lengths_kv` on `npugraph_ex`; if so, keep the host list stable for the captured shape and do not mutate it during replay. The two backends require different types; bind the type at initialization.
- **Graph capture.** Capture once per `(num_reqs, max_q_len, max_kv_len, block_table shape)` tuple. All tensor inputs must have stable addresses: block table, length buffers, metadata buffer, cache, query and output. A Python list is a host constant; it cannot be updated by a device-side `copy_`. If the selected operator requires a host list, either re-capture when the list changes or use the device-tensor device-tiling variant. A graph that replays with an old list is not measuring the current batch.
- **Metadata lifetime.** FlashMLA metadata and FIA device tiling metadata are not shape-polymorphic. Regenerate them when any of the sequence lengths, head counts, head dims, mask mode or query layout changes. A padded request added for the graph must have a valid block-table row and a device length of at least 1; a zero-length padded request must not become an empty required input.

```bash
python3 - <<'PY'
import torch
# persistent graph buffers: allocate once, update contents in place
num_reqs, max_blocks = 8, 64
block_table = torch.zeros((num_reqs, max_blocks), dtype=torch.int32, device="npu")
cache_seqlens = torch.ones(num_reqs, dtype=torch.int32, device="npu")
cu_q = torch.zeros(num_reqs + 1, dtype=torch.int32, device="npu")
ptr0 = (block_table.data_ptr(), cache_seqlens.data_ptr(), cu_q.data_ptr())
# update in place; do not reallocate
cache_seqlens.copy_(torch.tensor([1, 7, 33, 128, 1, 1, 1, 1], dtype=torch.int32))
cu_q.copy_(torch.tensor([0, 1, 2, 5, 9, 10, 11, 12, 13], dtype=torch.int32))
ptr1 = (block_table.data_ptr(), cache_seqlens.data_ptr(), cu_q.data_ptr())
assert ptr0 == ptr1, "graph buffer address changed"
assert bool((cache_seqlens >= 1).all())
print("stable buffers:", ptr0, "cu_q last:", int(cu_q[-1]))
PY
```

A minimal capture/replay check:

```python
g = torch.npu.CUDAGraph()
# warm up on a side stream, then capture the attention call with static buffers
with torch.npu.graph(g):
    out = attention(query, key_cache, value_cache, block_table=block_table,
                    actual_seq_lengths_kv=cache_seqlens, ...)
for _ in range(2):
    cache_seqlens.copy_(new_lengths)   # in-place, same address
    g.replay()
    torch.npu.synchronize()
    # compare against the eager call for the same lengths
```

**Invariants.** All graph inputs have stable `data_ptr()` values across replays; length buffers are updated in place; padded requests have a valid block row and length `>= 1`; the metadata buffer is large enough for the captured `num_reqs`; the operator variant and the length type match the backend; the metadata is regenerated when the shape contract changes; the output dtype and shape are exact.

**Verification.** Capture, replay twice with different lengths, and compare each replay against the eager call. Change `num_reqs` or `max_kv_len` and confirm a new capture is required rather than a silent stale read. Check the A5 path on the A5 SOC and the A3 path on the A3 SOC separately; a pass on one does not establish the other. If the replay is faster but the output matches the previous batch, the graph is reading stale metadata; discard the timing.

## Claim boundaries

This SOP is an integration and correctness procedure. It does not assert that every layout, block size or C8 mode is supported on every CANN revision; the installed operator's page is authoritative. A shape pass is not a numerical pass, a paged decode pass is not a prefill pass, and a single-SOC graph pass is not a cross-generation guarantee. The inherited UB/L1 and measurement contracts remain in force; a latency number from a stale graph or a failed correctness case is not a measurement.
