---
category: workflow
conditions:
  framework:
  - vllm-ascend
  - any
domain_path: compilation/triton/optimizations/tiling-and-ub-budget
evidence_strength: reproduced_empirical
id: unit-triton-ascend-tiling-persistent-grid-and-ub-budget-workflow
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/optimizations/techniques/tiling_and_grid_910_95.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/optimizations/techniques/tiling_and_grid_910_95.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L3
title: Triton-Ascend Tiling Geometry, Persistent Grid Scheduling, and UB Budget Fitting
  Workflow (`spatial_group_norm_reduce.py`)
triggers:
  error_signatures:
  - Numerical check failed on shape (2, 16, 16, 16) max_abs_err=0.046875 > 5e-3
  - 1. Platform-Specific UB Capacity & Alignment Accounting (`OPT_28`)
  - 2. Engine-Aware Grid Scheduling & Persistent Core Clamping (`OPT_29`, `OPT_41`)
  keywords:
  - unit-triton-ascend-tiling-persistent-grid-and-ub-budget-workflow
  - spatial_group_norm_reduce
  - spatial_group_norm_reduce.py
  - GroupNorm
  - max_abs_err
  - BLOCK_SIZE
  - multibuffer
---
# Triton-Ascend Tiling Geometry, Persistent Grid Scheduling, and UB Budget Fitting Workflow (`spatial_group_norm_reduce.py`)

## Stage 1 — Clamp 1D Grid to Physical VectorCore Count (`grid = (min(total_tiles, NUM_CORES),)`)

1. Select `NUM_CORES = 48` (VectorCore count on Ascend 910B/A3) and clamp `grid = (min(total_tiles, NUM_CORES),)`.
   - **Limitation**: On `spatial_group_norm_reduce.py`, grid clamping alone leaves the 3-level nested loop (`N -> G -> H`) intact (`indexing_operations >= 128`) and retains per-row `bfloat16` rounding (`max_abs_err = 0.046875 > 5e-3`, `verdict: "refuted"`).

## Stage 2 — Single-Row to Multi-Row Batching (`BLOCK_ROWS = 4`) without Contiguous `(H, W)` Merge

1. Batch `BLOCK_ROWS = 4` rows per program while keeping `(H, W)` unmerged.
   - **Limitation**: Still slices along `H` (`indexing_operations > 0`) and fails the `max_abs_err <= 5e-3` numerical check due to intermediate `bfloat16` accumulation (`verdict: "refuted"`).

## Stage 3 — Enable `multibuffer=True` (`multibuffer_cast_guard=True`) on Multi-Pass Reduction

1. Enable `multibuffer=True` double-buffering on the GroupNorm reduction loop.
   - **Limitation**: Multi-pass reduction live accumulators overflow the double-buffered UB stage ceiling (`80 KB`), forcing intermediate `bfloat16` staging casts (`bf16_casts > 0`, `bf16_arithmetic > 0`) (`verdict: "refuted"`).

## Stagnation & RSI Escalation Rule

When all 3 stages above (`Stage 1`, `Stage 2`, `Stage 3`) fail to eliminate both the 3D indexing loop and the `bfloat16` accumulation drift (`3` consecutive `refuted` attempts), record the 3 failed `L3` stage attempts in `logs/experiment-ledger.jsonl` and re-query `knowledge_query` with `ledger_entries` + `bottleneck_summary` (or `mode="explore"`) to trigger RSI automatic switching and mount the `L4` dimension-merge mutator.
