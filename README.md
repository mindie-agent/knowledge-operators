# Operators knowledge

The `graded-v2` branch of this repository is the `operators` domain graded knowledge source for MindIE Agent `0.0.3`. It stores **graded knowledge units** (`mindie-graded-unit/1`) for Ascend operator mechanisms, workflows and their measured applicability conditions. `main` remains unchanged and is not the graded contract/2 publication branch.

## Access

Install through the native MindIE Agent plugin for **Codex, Claude Code, Kimi or DSH** and follow the main installation guide: <https://github.com/mindie-agent/mindie-agent>. This repository does not copy host installers. Read-only query and explain do not require community sharing. The runtime can host other domains; the `0.0.3` release scope covers `operators` and `vllm-ascend`, not a hardcoded two-domain runtime. 0.0.3 plugins select `graded-v2` and bind the exact digest of this domain's `publication-contract.json`; manual contributions start from a `graded-v2` checkout and PRs target `graded-v2`.

A unit is not a universal speedup claim. It states the conditions under which an observation or procedure holds, the evidence behind it, and the limits that remain. Read a unit only after comparing its declared conditions with facts your task actually observed; a missing or `unknown` condition does not match.

## Live inventory

[`index/routing-manifest.json`](index/routing-manifest.json) is the authoritative, live inventory. It records stable IDs, paths, tiers, evidence strength, conditions, content digests, inheritance and token estimates. The table lists the units present in the manifest at this check; the manifest remains authoritative. A new unit becomes current only when its unit Markdown and rebuilt manifest entries both exist. Reviewed changes use the runtime's `mindie-reviewed-change/1` plan: complete unit bytes, the exact base generation and a review reference. Applying a plan writes units and rebuilds the local index; it does not commit, push or publish.

| Unit | Revision | Tier / category | Evidence strength | Pinned reference |
| --- | --- | --- | --- | --- |
| [Ascend gated RMSNorm: verify the intended fast path loaded, then bound the claim to the observed stage inventory and numerics](normalization/rmsnorm/unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence.md) | 1 | L2 / `sop` | `single_observation` | `mindie://operators/units/unit-gated-rmsnorm-ascend-fastpath-load-stage-evidence@052c3021547f44142562355cd6388ff0a3e7ca28d66570a7ec0fa0784eebee8d` |
| [Residual-free RMSNorm fastpath](normalization/rmsnorm/unit-residual-free-rmsnorm-fastpath.md) | 4 | L4 / `mutator` | `reproduced_empirical` | `mindie://operators/units/unit-residual-free-rmsnorm-fastpath@64f02c7298d8238d28d9884d4bc599d2d0bd27160be76e80f76841dc5f674d0c` |
| [Five-stage RMSNorm ablation workflow](normalization/rmsnorm/unit-rmsnorm-five-stage-ablation-workflow.md) | 1 | L3 / `workflow` | `single_observation` | `mindie://operators/units/unit-rmsnorm-five-stage-ablation-workflow@f72c21fd52649a8b086010eda2049fbd200276d4b22badee963f0d2aa088fc67` |

These units live under the `normalization/rmsnorm` domain path. The `operators` domain has no migrated `vllm-ascend` task corpus; shared knowledge is accessed through explicit domain routes, not copied here.

## Conditions and evidence boundaries

### L4 residual-free RMSNorm fastpath, revision 4

The L4 rule applies only to the complete recorded bfloat16/sigmoid workload on Ascend910B4:

- input shapes `1x37x4x128` and `1x193x8x128`, 36 samples per shape, bound as a complete `exact_set`;
- one visible logical device (`ascend_rt_visible_devices: "0"`), CANN `9.1.0`, driver `25.2.2`, firmware `7.7.0.10.220`;
- `torch 2.10.0+cpu`, `torch_npu 2.10.0.post4`, Python 3.12.13 on aarch64;
- source commit `f02da291ec0bdbe31af248deffa69f7a97e8f259`, target `vllm_ascend/ops/layernorm.py`, with the exact source and workload identities recorded in the unit.

A subset of the recorded workload does not satisfy the `exact_set`: collection order does not matter, but a shape from one case cannot be paired with another case's sample count or case digest, and dtype must be the recorded bfloat16. The same rule does not establish gains on other shapes, models, workloads, full-model serving, unseen tasks or product development time.

Revision 4 uses the reviewed per-case/block analysis in [the ablation receipt](docs/evidence/c4-case-block-v2-ablation.json), [gate results](docs/evidence/c4-case-block-v2-gates.json) and [stage acceptance](docs/evidence/c4-case-block-v2-stage-acceptance.json). The reported block envelopes are descriptive, not confidence intervals or independent repeated campaigns. The source and redacted file identities remain in [the provenance record](docs/evidence/c4-revision4-provenance.json). The unit does not claim a new NPU run.

### L3 five-stage RMSNorm ablation workflow, revision 1

The L3 unit describes the source, baseline, full candidate, leave-one-out and final stages of the same RMSNorm campaign. It is a `single_observation` workflow, not a second measurement campaign.

The original producer's reporting failure and raw verdict records are preserved. The later case/block analysis corrects their interpretation without rewriting the raw records and without claiming another NPU run. The correction does not promote the L3 observation to `reproduced_empirical`, and it does not turn the L4 envelope into a general performance claim.

### L2 gated RMSNorm fastpath/load-stage evidence, revision 1

The L2 unit is a `single_observation` source/load-stage SOP. It asks the consumer to verify that the intended fast path actually loaded and to bound the claim to the observed stage inventory and numerics. Its conditions bind repository `vllm-project/vllm-ascend`, task-start base commit `f02da291ec0bdbe31af248deffa69f7a97e8f259`, topic `gated-rmsnorm-gdn-kda-fastpath`, and declared hardware (one Ascend910). It is not a second measurement campaign and does not upgrade the L4 `reproduced_empirical` evidence or reclassify the L3 workflow. Missing or unknown facts remain `unknown`.

## Using these units

### Query with observed facts

`knowledge_query` compares the unit's conditions with facts your task observed. Conditions are conjunctive; pass only what you actually measured or read from the environment:

```python
knowledge_query(
    query="residual-free RMSNorm fastpath bfloat16",
    conditions={
        "soc": "Ascend910B4",
        "dtype": "bfloat16",
        "cann": "9.1.0",
        "framework": {
            "torch": "2.10.0+cpu",
            "torch_npu": "2.10.0.post4",
        },
    },
)
```

This excerpt shows the shape of `conditions`; it is not the complete L4 condition set. The L4 unit also requires the exact workload group, source commit, target file and other recorded identities. Omitting them yields `condition_unknown` and the unit is excluded by default. That is the correct result: supply the facts you actually observed, never copy the unit's own answers into the query, and never use a gold patch, oracle verdict or held-out answer to force applicability.

All declared conditions must hold for `matched`. Any explicit contradiction is `mismatch`; missing, `unknown` or incomparable facts remain `unknown` and do not match.

### Read the full body by exact digest

A query result carries a digest-pinned `ref`. Read that exact revision:

```python
knowledge_explain(
    ref="mindie://operators/units/unit-residual-free-rmsnorm-fastpath@64f02c7298d8238d28d9884d4bc599d2d0bd27160be76e80f76841dc5f674d0c",
)
```

The hash after `@` is the bare 64-character content digest, not the revision number. After a semantic edit, an old reference fails explicitly; a location-only move keeps the digest and revision. Reading the body does not establish applicability.

### Optional local feedback

After using a unit in real work, record `up` or `down` against the exact observed revision:

```python
knowledge_feedback(
    ref="mindie://operators/units/unit-residual-free-rmsnorm-fastpath@64f02c7298d8238d28d9884d4bc599d2d0bd27160be76e80f76841dc5f674d0c",
    rating="up",
    reason="Recorded after reusing the measured workload boundary in the observed task.",
)
```

Feedback is optional and descriptive. It binds the revision actually returned, not a newer revision. A private local reason is not published by default; a publishable reason must pass the deterministic privacy scan. Vote counts are not confidence, correctness proof or an automatic promotion gate.

### Explicit domain routes

Knowledge shared from another domain is read through an explicit `mindie://<domain>/units/<unit-id>@<sha256>` route with that domain configured. An unknown or unconfigured domain fails instead of falling back. This repository neither imports another domain's units nor merges same-topic rules across domains.

## Publication and validation

On `graded-v2`, [`publication-contract.json`](publication-contract.json) declares `mindie-publication-contract/2`, the graded data formats and the immutable validator revision used by publication checks. Local content/Feed checks, remote readback of `graded-v2` and installed native acceptance are separate outcomes. `main` remains unchanged and is not the graded publication branch. The contract's presence and a passing local check do not assert that a remote merge or installed native acceptance has occurred; release and validation evidence record those facts separately. `graded-v2` may receive content updates within this compatibility version while keeping the same `publication-contract/2` bytes; formal assets, remote publication and paid native acceptance are separate results, and locally prepared content is not published proof.

Documentation and engineering changes follow the development review path. Unit, feedback and derived-index changes follow the content path. Neither path is an automatic semantic acceptance or promotion mechanism. Applicability remains a matter of matching declared conditions with observed task facts.

## Repository layout

```text
knowledge-operators/
├── publication-contract.json
├── AGENTS.md
├── README.md
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

Format, digest, condition, inheritance and reviewed-change rules are in the runtime's unit schema; this domain follows the same `mindie-graded-unit/1` contract. For repository boundaries and maintenance rules, see [`AGENTS.md`](AGENTS.md).
