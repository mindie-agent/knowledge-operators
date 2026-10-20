# Operators knowledge

This repository contains operator mechanisms, workflows and their measured
applicability conditions under `domain: "operators"`.

## Current content

The [routing manifest](index/routing-manifest.json) records two accepted local
units:

| Unit | Revision | Tier | Evidence strength |
| --- | --- | --- | --- |
| [Residual-free RMSNorm fastpath](normalization/rmsnorm/unit-residual-free-rmsnorm-fastpath.md) | 4 | L4 | `reproduced_empirical` |
| [Five-stage RMSNorm ablation workflow](normalization/rmsnorm/unit-rmsnorm-five-stage-ablation-workflow.md) | 1 | L3 | `single_observation` |

The L4 rule applies to the complete recorded bfloat16/sigmoid workload on
Ascend910B4: shapes `1x37x4x128` and `1x193x8x128`, CANN `9.1.0`, driver
`25.2.2`, firmware `7.7.0.10.220`, torch `2.10.0+cpu` and torch_npu
`2.10.0.post4`, together with the exact source and workload identities in the
unit. A subset of those cases does not satisfy its conditions. The L3 unit
describes the source, baseline, full candidate, leave-one-out and final stages
of that same campaign.

Revision 4 uses the reviewed per-case/block analysis in
[the ablation receipt](docs/evidence/c4-case-block-v2-ablation.json),
[gate results](docs/evidence/c4-case-block-v2-gates.json) and
[stage acceptance](docs/evidence/c4-case-block-v2-stage-acceptance.json).
The reported block envelopes are descriptive, not confidence intervals or
independent repeated campaigns. The source and redacted file identities remain
in [the provenance record](docs/evidence/c4-revision4-provenance.json).

The original producer's reporting failure and raw oracle verdicts are preserved.
The later analysis corrects their interpretation without rewriting the raw
records or claiming another NPU run. Historical revisions, including their
unknown conditions, remain in Git history and retained evidence.

Query applicability must come from the consuming task's observed facts. Missing
or unknown conditions do not authorize automatic reuse. These units do not
establish gains on other workloads, full-model serving, unseen tasks or product
development time.

## Publication status

[publication-contract.json](publication-contract.json) already declares
`mindie-publication-contract/2` and records the validator revision. Local content
and Feed checks are available; the final release validator pin, remote publication readback and
installed native consumption remain pending. The contract's presence alone is
not evidence that those release steps have completed.

This domain contains no migrated `vllm-ascend` task corpus. Shared knowledge is
accessed through explicit domain routes.
