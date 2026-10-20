---
schema: mindie-graded-unit/1
id: unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
revision: 1
tier: L1
category: axiom
domain_path: _common
evidence_strength: hardware_spec
title: 'Ascend 910B and 910_93 memory alignment, vector repeat and ACLGraph tiling axioms'
conditions:
  soc:
  - Ascend910B
  - Ascend910B1
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  - float32
inherits_from: []
provenance:
  source_type: hardware_architecture_specification
  source_locator: docs/evidence/c4-case-block-v2-gates.json
  evidence_refs:
  - docs/evidence/c4-case-block-v2-gates.json
  supersedes: []
triggers:
  error_signatures:
  - misaligned UB DMA, L1 tile or vector repeat causes wrong results or a runtime fault
  - bf16 or fp16 RMSNorm/LayerNorm/Softmax loses accuracy because the reduction is not fp32
  - a strided gate view silently takes the contiguous fast path
  - ACLGraph capture fails or replay changes the captured workspace
  keywords:
  - Ascend910B
  - Ascend910_93
  - alignment
  - UB
  - L1
  - vector repeat
  - fp32 accumulation
  - RMSNorm
  - LayerNorm
  - Softmax
  - contiguous
  - strided
  - ACLGraph
---
# Ascend 910B / 910_93 memory alignment and tiling axioms

Scope: Ascend 910B (A2) and Ascend 910_93 (A3 dual-die, Phy-ID 0..15), with bfloat16, float16 or float32 operands. These are hardware-contract axioms. A kernel or wrapper that violates one is outside this rule even if one observed output happens to look correct.

## DMA and buffer alignment

- **UB / GM DMA: 32B.** Every GM-to-UB and UB-to-GM transfer uses a 32-byte aligned base address and a byte count that is a multiple of 32B. Pad the final block to 32B; never let a vector read consume padding as data. A view whose storage offset breaks 32B alignment is copied to an aligned contiguous buffer, or handled by an explicit scalar/tail path, before DMA.
- **L1: 512B.** L1-resident cube tiles use 512-byte aligned base addresses and 512-byte aligned tile strides in the L1 descriptor. A tile whose offset or stride is not 512B aligned is re-tiled, padded, or staged through UB; a misaligned L1 descriptor is undefined.
- **Per-die ownership.** On Ascend 910_93 each die owns its UB, L1, stream and event state. Workspace and captured addresses are per-die unless an explicit cross-die (HCCL) transfer belongs to the operator contract. Do not share one captured workspace address across dies.

## Vector repeat and stride

- One vector repeat covers 256 B: 128 bf16/fp16 lanes or 64 fp32 lanes. `mask` selects the active lanes inside a repeat.
- `repeatTimes` is 0..255. The repeat stride is the advance between repeats in the unit declared by the API: one 256B repeat block for the vector repeat form, or 32B blocks for block-strided memory forms. Convert an element stride as `stride_bytes = stride_elements * sizeof(dtype)`. Stride 1 selects the next contiguous repeat block, stride 0 broadcasts, and stride greater than 1 interleaves.
- A strided tensor is described by its real stride plus a tail mask. Never reinterpret a strided view as contiguous and never over-read to fill the last repeat. The tail is masked or padded with defined values, and the output tail is written only for valid lanes.

## Reduction accumulation

- `RMSNorm`, `LayerNorm` and `Softmax` reductions over bf16/fp16 input accumulate the sum and sum-of-squares in fp32. Cast to the requested `out_dtype` once, at the final store.
- An fp16/bf16 accumulator is a different numerical contract: it can lose the mean or square mean and shift the normalized result. An unrequested fp32 output is `out_dtype` drift, not an accuracy improvement.
- Epsilon is applied in the same fp32 domain as the reduction. For gated RMSNorm, the sigmoid gate multiplies the normalized fp32 value before the final cast.

## Dispatch: contiguous versus strided

- The contiguous vector kernel requires a last-dimension stride of 1 and 32B (UB) / 512B (L1) compliant base and strides. Shape equality alone does not select it.
- A view with a non-unit last-dimension stride, a non-32B-aligned storage offset, or a gate tensor with a different layout takes the strided kernel or an explicit contiguous materialization. Record which path executed; a dispatch-table entry is not proof that the native path ran.

## ACLGraph capture invariants

- Capture requires a static workspace address and size, a fixed capture stream, and no host allocation, synchronization, shape-dependent branch or dynamic tiling inside the captured region.
- Allocate and align workspace before capture; warm up and synchronize outside the graph. Replay keeps the same input and output addresses and does not mutate captured inputs.
- A shape or dtype outside the captured specialization falls back to eager execution. Capturing a fallback branch or resizing workspace during replay invalidates the graph.
