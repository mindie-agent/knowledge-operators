---
category: sop
conditions:
  dtype:
  - bfloat16
  - float16
  - float32
  - int8
  - int32
  op_family:
  - moe
  - dispatch_combine
  - alltoallv_gmm
  - moe_init_routing
  - moe_token_permute
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: moe/dispatch-combine
evidence_strength: reproduced_empirical
id: unit-moe-dispatch-gmm-combine-and-alltoallv-routing-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-a5-950-ub-l1-cube-vector-architecture-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/moe/dispatch_combine_metadata.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/moe/moe_fusion_internal_metadata_and_overlap_conditions.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/moe/moe_init_routing.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/moe/moe_init_routing_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/moe/moe_token_permute.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/moe/moe_token_unpermute.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/mc2/allto_allv_grouped_mat_mul_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/mc2/moe_distribute_dispatch_950.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/mc2/moe_distribute_combine_add_rms_norm_950.md
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/operators/aclnn/transformer/moe/moe_init_routing.md
  source_type: operator_contract_and_source_closure_review
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: MoE Dispatch/Combine, AlltoAllvGroupedMatMul, MoeInitRoutingV3 and MoeTokenPermute/Unpermute
  end-to-end routing SOP
triggers:
  error_signatures:
  - Combine waits on maximum capacity A instead of actual recvCounts and hangs waiting
    for unproduced tokens
  - MoeInitRoutingV3 forces full-load route R-003 based only on small L=N*K without
    deducting sort, rowIdx, expert, and gather/quant UB buffers
  - AlltoAllvGroupedMatMul on Ascend 950 passes local expert count e > 32 and overflows
    the fixed 32-entry device expertTokenNum_ array
  - MoeTokenPermute double-argsort overflows FP32 exact integer representation when
    L = N * K >= 16777215
  - MoeTokenUnpermute accumulates BF16/FP16 top-K expert outputs without FP32 TBuf
    staging or fails to skip zero-probability experts before gather
  keywords:
  - MoeDistributeDispatchV4
  - moe_distribute_combine_add_rms_norm_v2
  - AlltoAllvGroupedMatMulV2
  - MoeInitRoutingV3
  - MoeTokenPermute
  - MoeTokenUnpermute
  - expandIdx
  - assistInfoForCombine
  - fullmesh_v1
  - fullmesh_v2
  - DataAsFlag
  - schedule_mode
---
# MoE Dispatch/Combine, AlltoAllvGroupedMatMul, MoeInitRoutingV3 and MoeTokenPermute/Unpermute end-to-end routing SOP

## Use when

Designing, debugging, or integrating MoE token routing, permutation, and distributed EP collective-compute fusion on Ascend 910B (A2), 910_93 (A3), or 950 (A5) using `MoeInitRoutingV3`, `MoeTokenPermute`, `MoeTokenUnpermute`, `MoeDistributeDispatchV4`, `moe_distribute_combine_add_rms_norm_v2`, or `AlltoAllvGroupedMatMulV2`.

## 1. Preserve Dispatch-to-Combine metadata alignment and window epoch invariants

1. **Co-ordered payload and source triple (`expandIdx` / `assistInfoForCombine`)**:
   - Every routed item carries a feature payload (`expandX`), source recovery triple `expandIdx = [srcRankId, srcTokenIdx, srcTopKIdx]` (`assistInfoForCombine`), and per-token dynamic scale (`dynamicScalesOut` / `expandScalesOut`). All three arrays must preserve identical valid-entry ordering.
   - Never include the packed 3-integer metadata header inside the logical hidden dimension `H` of `expandX`.
2. **Upper-bound capacity `A` vs actual received count**:
   - `A` (`axisMaxBS = globalBs / epWorldSize`) is the static allocation ceiling, whereas `epSendCounts` / `recvCounts` record the actual tokens transferred in the current step.
   - Never use `A` as the polling target in `WaitDispatch` or as the row extent for expert GEMM; polling `A` hangs indefinitely when fewer than `A` tokens route to the rank, and reading `A` rows ingests stale window memory.
3. **`MoeDistributeDispatchV4` & `moe_distribute_combine_add_rms_norm_v2` on Ascend 950**:
   - On Ascend 950 (`DAV_3510`), `MoeDistributeDispatchV4` supports `1024 <= H <= 8192`, `1 <= K <= 16` (`K <= 12` for `commAlg="fullmesh_v2"`), `HasTp=false`, and `quantMode in 0..4`.
   - **Route R-001 (`fullmesh_v1` / empty `commAlg`)**: Uses independent `32 B` status flags, `SyncAll`, and `16 MiB + 512 * aivNum^2` workspace.
   - **Route R-002 (`fullmesh_v2`)**: Uses `512 B` `DataAsFlag` blocks (`480 B` payload + `32 B` inline flag per `512 B` stride: `ceil(hOutSizeAlign / 480) * 512`).
   - `moe_distribute_combine_add_rms_norm_v2` (`16 MiB + aivNum * 10 KiB` workspace, `0 < Bs <= 512`) waits for `K + sharedExpertNum` status slots per token, aggregates normal experts in `FP32`, handles `zero` (skip), `copy` (`oriX`), and `const` (`softmax(alpha1, alpha2)` blending `oriX` and `constExpertV`) experts, and fuses `x = combine + residualX`, `rstd = 1 / sqrt(sum(x^2)/H + eps)`, and `y = BF16(BF16(x * rstd) * gamma)`.

## 2. Select and budget `MoeInitRoutingV3` routes (`R-001`..`R-005`)

`MoeInitRoutingV3` sorts `L = N * K` pairs `(expertIdx, rowIdx)` and emits `expandedXOut`, `expandedRowIdxOut` (`rowIdxType=0` scatter inverse map `out[sortedRow[i]] = i`; `rowIdxType=1` gather map `out[i] = sortedRow[i]`), `expertTokensCountOrCumsumOut` (`expertTokensNumType`: `0` cumsum, `1` count, `2` key-value), and `expandedScaleOut`.

1. **Strict UB admission for Full-Load (`R-003` on A2/A3, `R-002` on A5)**:
   - Never select the single-pass UB full-load route based solely on `L <= sortLoopMaxElement` (`floor((UB - aivNum * 32) / 32 / 32) * 32` on A2/A3; capped at `32 * 255 = 8160` on A5 after deducting `SIMT_DCACHE_SIZE = 65536 B`).
   - Full-load admission (`IsFullLoad()`) requires `!dropPadMode` AND positive remaining UB after deducting all four buffers:

     ```cpp
     int64_t sortSpace   = sortNum * sizeof(int32_t) * 4;
     int64_t rowIdxSpace = sortNum * sizeof(int32_t) * 3;
     int64_t expertSpace = AlignUp(expertNum * sizeof(int64_t), 32) * 2;
     int64_t gatherSpace = AlignUp(cols * inputXDtypeSize, 32) * perCoreTokens;
     int64_t remainUb    = ubSize - sortSpace - rowIdxSpace - expertSpace - 1024;
     ```

2. **Multi-core VBS/VMS merge and workspace sizing (`R-004` / `R-005`)**:
   - When `remainUb <= 0` or `L > sortLoopMaxElement`, stage through GM workspace (`Wsort = 24 * L`, `Wsync = 64 * coreNum`, `Wscatter = 4 * L`, `Wcount = 4 * actualExpertNum`, plus `4 * coreNum * H` for dynamic quant and `16 MiB` system reserve) using up to 4-way `MrgSort` ping-pong across `SyncAll` barriers.
3. **Dynamic quantization numerical invariant**:
   - For `quantMode = 1` (dynamic `int8`) or `quantMode = 13` (dynamic `int4` on A5, requiring even `H`), widen `BF16`/`FP16` inputs to `FP32` first, multiply by `smooth`, compute `Abs` and `ReduceMax` in `FP32` with `PipeBarrier<PIPE_V>()` between dependent steps, divide by `127.0f` (`7.0f` for `int4`), and convert via `RoundMode::CAST_RINT -> half -> int8`.

## 3. Enforce `MoeTokenPermute` and `MoeTokenUnpermute` invariants

1. **`MoeTokenPermute` double-argsort and split-D thresholds**:
   - On A2/A3 (`IsRegbase() == false`; A5 routes to `MoeInitRoutingV2`), `MoeTokenPermute` runs a single AIV kernel with `schedule_mode = 1` and three sequential phases separated by `SyncAll` and `TPipe::Destroy`:
     - **Phase S1**: Argsort `flatten(indices)` into GM `sortedIndices` by casting integer keys to `FP32` (`CAST_ROUND`), multiplying by `-1.0f`, padding `L % 32` tails with `MIN_FP32`, and sorting `32`-element blocks. Requires `L < 16,777,215` and `indices < 16,777,215` so keys stay within the exact `FP32` integer mantissa range (`2^24`).
     - **Phase S2**: Argsort `sortedIndices` in-place to obtain the inverse permutation `q` (`permutedTokensOut[q[i], :] = tokens[i / K, :]`).
     - **Phase G**: Whole-row double-buffered gather (`key 1/2/5/6`, `onceUbTokenNums <= 4095` due to `uint16_t` `DataCopyExtParams.blockCount`) when `UB - 2048 >= 2 * Align32(D * tokenBytes)`, or split-D gather (`key 3/4/7/8`, `oneTokenOnceMove = floor(floor((UB - 2048) / 2, 512) / tokenBytes)`) when a full row exceeds UB.
2. **`MoeTokenUnpermute` FP32 accumulation and zero-prob skip**:
   - Always accumulate weighted top-K expert rows (`Y[i, h] = sum_{j=0..K-1} valid(S[i*K+j]) * X[S[i*K+j], h] * P[i, j]`) in `FP32` (`TBuf` for 16-bit tokens, or directly in the output queue for `FP32` tokens).
   - Skip out-of-range indices (`s >= M`) and zero-probability slots (`P[i, j] == 0`) **before** issuing the GM->UB token gather DMA.

## 4. Overlap `AlltoAllvGroupedMatMulV2` on Ascend 950 (`MIX_AIC_1_2`)

1. **Rank-first GMM without redundant permute**:
   - `aclnnAlltoAllvGroupedMatMulV2` launches a `KERNEL_TYPE_MIX_AIC_1_2` kernel (`blockDim = min(AIC, AIV / 2)`) with `commMode in {"ccu", "ai_cpu"}`.
   - `AIV0` submits batched `AlltoAllV` transfers into `commOut` (`align512(A * H1 * 2)` bytes in workspace) ordered `rank -> expert`. While `AIV0` communicates, `AIC` executes the local shared-expert `mmY = mmX * mmWeight` in parallel.
   - After `AIV0`/`AIV1` signal cross-core flags `8` and `24`, `AIC` constructs a `(expert, rank)` offset table (`actualExpertNum * P <= 128`) and computes routing GMM directly on the rank-first `commOut` buffer. Set `permuteOutFlag = false` (`permuteOut = nullptr`) unless a downstream consumer explicitly requires an expert-first `[A, H1]` copy, saving `2 * 32 KB` AIV UB and a full HBM round-trip.
2. **Guard the 32-expert-per-rank device array limit**:
   - Although the V2 host doc mentions `e <= 48` (`e * P <= 256`), `QuantGroupedMatmul::Init` on device allocates a fixed `expertTokenNum_[32]` array (`MAX_EXPERT_PER_RANK = 32`). Enforce `e <= 32` local experts per rank to prevent device stack/buffer overflow.

## Verification

- Verify Dispatch-Combine round-trip with a synthetic rank/token/topK encoding `(srcRank * 100000 + srcToken * 100 + topK)` across empty experts (`recv == 0`), partial capacity (`sum(recv) < A`), and `O < L` truncated permutation.
- Profile `AlltoAllvGroupedMatMulV2` with `msprof` to confirm AIC shared-expert `aic_mac_time` overlaps AIV `AlltoAllV` communication before flags `8/24`.
