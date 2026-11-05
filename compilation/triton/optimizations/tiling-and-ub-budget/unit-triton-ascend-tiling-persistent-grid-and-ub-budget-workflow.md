---
category: workflow
conditions:
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
  - Ascend950DT
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
revision: 6
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

1. Edit `spatial_group_norm_reduce.py` so `spatial_group_norm_reduce` clamps `grid = compute_persistent_grid(total_tiles)` (`NUM_CORES = 48`), then run `pytest -q tests/ut/ops/test_spatial_group_norm_reduce.py` to evaluate Stage 1 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Stage 2 — Single-Row to Multi-Row Batching (`BLOCK_ROWS = 4`) without Contiguous `(H, W)` Merge

1. Edit `spatial_group_norm_reduce.py` to batch `BLOCK_ROWS = 4` rows per step (`for h_start in range(0, height, 4)`) with fixed `BLOCK_SIZE = 512` while keeping `(H, W)` unmerged, then run `pytest -q tests/ut/ops/test_spatial_group_norm_reduce.py` to evaluate Stage 2 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Stage 3 — Enable `multibuffer=True` (`multibuffer_cast_guard=True`) on Multi-Pass Reduction

1. Edit `spatial_group_norm_reduce.py` to keep `multibuffer = True` double-buffering enabled on the reduction loop across `(c, h)` slices, then run `pytest -q tests/ut/ops/test_spatial_group_norm_reduce.py` to evaluate Stage 3 and record its `candidate_sha256` outcome in `logs/experiment-ledger.jsonl`.

## Verification & Follow-Up Retrieval

1. Always apply and test Stage 1, Stage 2, and Stage 3 as 3 separate incremental edits (running `pytest -q tests/ut/ops/test_spatial_group_norm_reduce.py` after each edit) so `logs/experiment-ledger.jsonl` records 3 distinct `candidate_sha256` evaluations.
2. Do **not** attempt to brute-force test capture hashes via standalone scripts or inspect parent directories outside the workspace (`..`): the knowledge routing engine gates deeper structural specifications strictly on `logs/experiment-ledger.jsonl` containing 3 distinct `candidate_sha256` entries produced by editing `vllm_ascend/ops/spatial_group_norm_reduce.py` and running `pytest -q tests/ut/ops/test_spatial_group_norm_reduce.py` for Stage 1, Stage 2, and Stage 3 in sequence.
3. If the Ascend 950DT check remains `refuted` after all 3 stage edits, call `knowledge_query` again with default `mode="reuse"` (`domain="operators"`, `conditions={"framework": "vllm-ascend", "soc": "Ascend950DT"}`, `token_budget=24000`, `goal`, and `bottleneck_summary`) to retrieve deeper structural remedies.
