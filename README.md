# Operators knowledge

The `graded-v2` branch of this repository is the `operators` domain graded knowledge source for MindIE Agent `0.0.3`. It stores **graded knowledge units** (`mindie-graded-unit/1`) for Ascend operator mechanisms, workflows and their measured applicability conditions. `main` remains unchanged and is not the graded contract/2 publication branch.

## Access

Install through the native MindIE Agent plugin for **Codex, Claude Code, Kimi or DSH** and follow the main installation guide: <https://github.com/mindie-agent/mindie-agent>. This repository does not copy host installers. Read-only query and explain do not require community sharing. The runtime can host other domains; the `0.0.3` release scope covers `operators` and `vllm-ascend`, not a hardcoded two-domain runtime. 0.0.3 plugins and manual contributions use `main`; queries and feedback identify units by `ID@revision`.

A unit is not a universal speedup claim. It states the conditions under which an observation or procedure holds, the evidence behind it, and the limits that remain. Read a unit only after comparing its declared conditions with facts your task actually observed; a missing or `unknown` condition does not match.

## Live inventory

Markdown units are the source of truth. [`index/routing-manifest.json`](index/routing-manifest.json) is their rebuildable inventory of IDs, revisions, paths, conditions and inheritance. Rebuild the routing index after changing a unit; reviewed changes use the runtime's existing `mindie-reviewed-change/1` application path.

| Unit | Revision | Tier / category | Evidence strength | Pinned reference |
| --- | --- | --- | --- | --- |
| [Ascend 910B/910_93 memory alignment, vector repeat and ACLGraph tiling axioms](_common/unit-ascend-910b-a3-memory-alignment-and-tiling-axioms.md) | 1 | L1 / `axiom` | `hardware_spec` | `mindie://operators/units/unit-ascend-910b-a3-memory-alignment-and-tiling-axioms@1` |
| [Ascend operator four-gate measurement and four-view roofline contract](_common/unit-ascend-operator-roofline-and-four-view-measurement.md) | 1 | L1 / `measurement` | `reproduced_empirical` | `mindie://operators/units/unit-ascend-operator-roofline-and-four-view-measurement@1` |
| [Ascend gated RMSNorm Bottleneck & Fastpath Dispatch Triage SOP](normalization/rmsnorm/unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence.md) | 2 | L2 / `sop` | `single_observation` | `mindie://operators/units/unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence@2` |
| [Five-stage Hermes operator optimization, leave-one-out ablation, and distillation workflow](normalization/rmsnorm/unit-rmsnorm-five-stage-ablation-workflow.md) | 3 | L3 / `workflow` | `reproduced_empirical` | `mindie://operators/units/unit-rmsnorm-five-stage-ablation-workflow@3` |
| [L4 mutator: residual-free gated RMSNorm native fastpath for Ascend910B](normalization/rmsnorm/unit-residual-free-rmsnorm-fastpath.md) | 6 | L4 / `mutator` | `reproduced_empirical` | `mindie://operators/units/unit-residual-free-rmsnorm-fastpath@6` |
| [Dynamic-length Triton recompilation SOP](compilation/triton/unit-triton-dynamic-scalar-specialization-sop.md) | 1 | L2 / `sop` | `single_observation` | `mindie://operators/units/unit-triton-dynamic-scalar-specialization-sop@1` |

The two L1 units live under `_common` and are inherited by every `normalization/rmsnorm` unit. The `operators` domain has no migrated `vllm-ascend` task corpus; shared knowledge is accessed through explicit domain routes, not copied here.

## Conditions and evidence boundaries

### L1 foundation, revision 1

The axiom unit states the hardware contracts for Ascend 910B (A2) and Ascend 910_93 (A3 dual-die, Phy-ID 0..15): 32 B UB DMA alignment, 512 B L1 alignment, vector repeat/stride rules, fp32 accumulation for RMSNorm/LayerNorm/Softmax reductions, contiguous-versus-strided dispatch, and ACLGraph workspace/stream capture invariants. Its conditions cover the Ascend 910B family plus Ascend 910_93 and bfloat16, float16 and float32.

The measurement unit defines the four gates (OracleFreezer correctness, synchronized block timing, AblationGate leave-one-out attribution, and execution/artifact identity) and the four aligned views (msprof kernel, host dispatch, HBM/UB traffic against the Roofline ridge point, and end-to-end operator boundary). Both L1 units are inherited premises, not operator-specific performance claims.

### L2 gated RMSNorm triage, revision 2

The L2 unit is a reusable bottleneck and fastpath dispatch SOP. Its conditions are the observed facts `soc` in `[Ascend910B, Ascend910B4]`, `dtype: bfloat16`, `activation: sigmoid`, and `target_file: vllm_ascend/ops/layernorm.py`; it no longer requires a task-start base commit or a specific driver/firmware pair. It triages the executed kernel inventory, the dispatch registration, the oracle and the timing before routing to the L4 mutator or the L3 workflow. A class name or dispatch entry is treated as load evidence, not as fusion or speedup evidence.

### L3 five-stage Hermes workflow, revision 3

The L3 unit is an executable five-stage workflow: source/symbol contract and oracle freeze; synchronized block baseline timing; candidate synthesis with an explicit L4 escalation trigger when routine optimization stalls or fails Gate 1/2; leave-one-out ablation that separates the correctness prerequisites `required_dispatch_registration` and `input_preservation_out_dtype` from performance factors such as `native_residual_free_fastpath`; and final retained verification with Hermes `knowledge_distill` into `reviewed_change` plus `experiment-ledger.jsonl`. Its conditions are the same four observed facts as the L2 unit, with no base-commit or driver/firmware lock.

### L4 residual-free gated RMSNorm mutator, revision 6

The L4 unit is an actionable code-transformation mutator. Its conditions are `soc` in `[Ascend910B, Ascend910B4]`, `dtype: bfloat16`, `activation: sigmoid`, and `target_file: vllm_ascend/ops/layernorm.py`. It applies `native_residual_free_fastpath` together with the correctness prerequisites `required_dispatch_registration` and `input_preservation_out_dtype`, handles contiguous and strided gates without mutating inputs, and carries the Gate 1-3 verification invariants.

The reference envelope was measured on Ascend910B4: bfloat16/sigmoid cases `1x37x4x128` with a contiguous gate and `1x193x8x128` with a strided gate, 36 samples per case in three blocks after two warmups. Final retained versus baseline is `1.6767473221869322x` (reported as 1.68x); the observed per-case block-envelope lower bounds are 1.5638748801806301 and 1.7387598131467064. The full candidate versus baseline is 1.5276488398873962x. The envelope is descriptive, not a confidence interval, and the stages are separate runs. Raw samples and comparison identities remain in [the ablation receipt](docs/evidence/c4-case-block-v2-ablation.json), [gate results](docs/evidence/c4-case-block-v2-gates.json) and [stage acceptance](docs/evidence/c4-case-block-v2-stage-acceptance.json). The measured speedup is not claimed for other shapes, SOCs, dtypes or full-model serving.

## Using these units

### Query with observed facts

`knowledge_query` compares the unit's conditions with facts your task observed. Conditions are conjunctive; pass only what you actually measured or read from the environment:

```python
knowledge_query(
    query="rmsnorm residual fastpath ablation",
    conditions={
        "soc": "Ascend910B4",
        "dtype": "bfloat16",
        "activation": "sigmoid",
        "target_file": "vllm_ascend/ops/layernorm.py",
    },
)
```

These four observed facts are the complete condition set for the L2/L3/L4 units and match both L1 foundation units. Omitting any one of them yields `condition_unknown` and the unit is excluded by default. That is the correct result: supply the facts you actually observed, never copy the unit's own answers into the query, and never use a gold patch, oracle verdict or held-out answer to force applicability. A larger `token_budget` is needed when the caller wants the full inherited L1 bodies injected; the matched/rejected status does not depend on the budget.

All declared conditions must hold for `matched`. Any explicit contradiction is `mismatch`; missing, `unknown` or incomparable facts remain `unknown` and do not match.

### Read the full body by revision

A query result carries a revision-specific `ref`. Read that revision:

```python
knowledge_explain(
    ref="mindie://operators/units/unit-residual-free-rmsnorm-fastpath@6",
)
```

The integer after `@` is the unit revision. Semantic edits advance it; location-only moves preserve it. Use the `ref` returned by the query.

### Optional local feedback

After using a unit in real work, record `up` or `down` against the exact observed revision:

```python
knowledge_feedback(
    ref="mindie://operators/units/unit-residual-free-rmsnorm-fastpath@6",
    rating="up",
    reason="Recorded after reusing the measured workload boundary in the observed task.",
)
```

Feedback is optional and descriptive. It binds the revision actually returned, not a newer revision. A private local reason is not published by default; a publishable reason must pass the deterministic privacy scan. Vote counts are not confidence, correctness proof or an automatic promotion gate.

### Explicit domain routes

Knowledge shared from another domain is read through an explicit `mindie://<domain>/units/<unit-id>@<revision>` route with that domain configured. An unknown or unconfigured domain fails instead of falling back. This repository neither imports another domain's units nor merges same-topic rules across domains.

## Publication and validation

On `graded-v2`, [`publication-contract.json`](publication-contract.json) declares `mindie-publication-contract/2`, the graded data formats and the immutable validator revision used by publication checks. Local content/Feed checks, remote readback of `graded-v2` and installed native acceptance are separate outcomes. `main` remains unchanged and is not the graded publication branch. The contract's presence and a passing local check do not assert that a remote merge or installed native acceptance has occurred; release and validation evidence record those facts separately. `graded-v2` may receive content updates within this compatibility version while keeping the same `publication-contract/2` bytes; formal assets, remote publication and paid native acceptance are separate results, and locally prepared content is not published proof.

Documentation and engineering changes follow the development review path. Unit, feedback and derived-index changes follow the content path. Neither path is an automatic semantic acceptance or promotion mechanism. Applicability remains a matter of matching declared conditions with observed task facts.

## Repository layout

```text
knowledge-operators/
├── publication-contract.json
├── AGENTS.md
├── README.md
├── _common/
│   ├── unit-ascend-910b-a3-memory-alignment-and-tiling-axioms.md
│   └── unit-ascend-operator-roofline-and-four-view-measurement.md
├── index/
│   ├── routing-manifest.json
│   └── shard-topology.json
├── normalization/
│   └── rmsnorm/
│       ├── unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence.md
│       ├── unit-residual-free-rmsnorm-fastpath.md
│       └── unit-rmsnorm-five-stage-ablation-workflow.md
├── docs/
│   └── evidence/
└── scripts/
```

Format, revision, condition, inheritance and reviewed-change rules are in the runtime's unit schema; this domain follows the same `mindie-graded-unit/1` contract. For repository boundaries and maintenance rules, see [`AGENTS.md`](AGENTS.md).
