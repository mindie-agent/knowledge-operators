---
category: sop
conditions:
  cann_versions:
  - '>=8.0.0'
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
  workloads:
  - AscendC C++ operator design, implementation, code review, and optimization
domain_path: operators/skill-playbooks/ascendc-and-catlass
evidence_strength: software_contract
id: unit-agent-skills-ascendc-operator-design-codegen-and-review-sop
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-platform-alignment-pipeline-and-execution-constraints-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/ascendc-operator-design/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/ascendc-operator-code-review/SKILL.md
  - https://gitcode.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/ascendc-operator-performance-optim/SKILL.md
  source_digest: 6969add80b1156b3be50379188ce8e8c23e1d9e03d00b0425493a8fd2f3e47a8
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/ascendc-operator-design/SKILL.md
  source_type: code
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
summary: Engineering playbook from `Ascend/agent-skills` (`ascendc-operator-design`,
  `code-gen`, `code-review`, `performance-optim`, `precision-debug`) covering 32-byte
  alignment, UB budget equations, TPipe queues, and tail handling.
tier: L2
title: AscendC Operator Design, Code Generation, Review, and Performance Optimization
  Playbook
triggers:
  error_signatures:
  - ascendc operator design skill
  - ascendc operator code review skill
  keywords:
  - ascendc operator design skill
  - ascendc operator code review skill
  - ascendc operator performance optim skill
  - tpipe quePosition double buffer 32 byte alignment
  - ascendc
  - operator-design
  - code-review
  - tpipe
  - 32b-alignment
  - double-buffer
  - ub-budget
---
## 1. AscendC Operator Design Contract (`ascendc-operator-design`)

Every AscendC operator design document and host/kernel implementation must satisfy four hardware invariants:
1. **32-Byte Memory & UB Block Alignment**:
   - Every `DataCopy` transfer between `GM` and `UB`/`L1` and every `TPipe::InitBuffer` allocation must be a multiple of **32 bytes** (`16` `float16`/`bfloat16` elements or `8` `float32` elements).
   - Vector instructions (`Add`, `Mul`, `Exp`, `Muls`) operate on **256-byte** (`128` `fp16` or `64` `fp32`) vector repeats.
2. **Strict UB Capacity Equation (`<= 192 KB` on 910B/910C)**:
   $$\sum_{q \in \text{Queues}} (\text{BUFFER\_NUM}_q \times \text{tileLength}_q \times \text{sizeof}(T_q)) + \sum_{b \in \text{TBufs}} \text{size}(b) \le \text{UB\_MAX}$$
   - When `BUFFER_NUM = 2` (double buffering), each queue consumes $2\times$ its single-tile byte size.
3. **Host-Side Tiling (`TilingFunc`) Completeness**:
   - Compute `usedCoreNum`, `formerNum` (big-core count), `tailNum` (small-core count), `singleCoreLength`, `tileNum`, `tileLength`, and `lastTileLength` on the Host CPU and pass them via `TilingData` struct—never compute division-heavy dynamic tiling inside `AICore`.
4. **Non-Aligned Tail Handling (`DataCopyPad`)**:
   - When `lastTileLength * sizeof(T)` is not a multiple of 32 bytes, use `DataCopyExtParams` + `DataCopyPad` on `910B`/`910C` (rather than out-of-bounds `DataCopy`) to prevent overwriting adjacent GM tensors.

## 2. Mandatory Code Review Checklist (`ascendc-operator-code-review`)

1. **Symmetric Queue Lifecycle**:
   - Every `AllocTensor` must have a matching `EnQue`, and every `DeQue` must have a matching `FreeTensor` inside the same loop iteration; leaked `TQue` slots deadlock `TPipe` on iteration `BUFFER_NUM + 1`.
2. **No `BF16` Direct Vector Arithmetic**:
   - Ascend 910B/910C Vector units do not execute native `BF16` arithmetic (`Add`/`Mul`/`Exp`); always `Cast` `bfloat16_t -> float` (`RoundMode::CAST_NONE`) in a `TBuf<TPosition::VECCALC>` workspace and `Cast` `float -> bfloat16_t` (`RoundMode::CAST_RINT`) before `CopyOut`.
3. **Cross-Core Synchronization (`SyncAll` / `IBSet` / `IBWait`)**:
   - If a kernel reads GM written by another core in the same launch, allocate a global `syncWorkspace` (`ZeroInit`) and invoke `SyncAll()` between the write phase and read phase.
