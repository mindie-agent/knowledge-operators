# Operators domain boundaries

This repository contains operator knowledge and evidence under the `operators`
domain. Runtime code belongs to `mindie-agent/knowledge`.

- Keep stable unit IDs independent of paths. Validate units as
  `mindie-graded-unit/1` and rebuild `index/routing-manifest.json` and
  `index/shard-topology.json` after unit changes.
- Preserve declared conditions, evidence strength and source digests. Changes
  to the meaning, conditions or provenance of an existing unit require semantic
  review before promotion. Keep superseded revisions and their evidence intact.
- Keep raw execution facts separate from later analysis. Failed producer states,
  rejected oracle verdicts, missing measurements and unknown conditions remain
  visible. A reporting correction does not create a new measurement campaign.
- An unknown condition never matches itself. Use facts available from the
  consuming task to assess applicability; do not copy gold answers or private
  oracle identities into a query to force a match. Evidence identities and
  measured workload bounds do not establish broader applicability.
- Admit new sources with their actual provenance and privacy/lineage checks.
  Do not copy the `vllm-ascend` corpus or another domain's units into this domain.
  Reuse other domains through explicit routes; inheritance stays within one
  domain.
- `publication-contract.json` declares `mindie-publication-contract/2`. Before
  formal publication, bind the contract and CI to the same reviewed immutable
  validator revision, validate the intended tree, and verify the actual remote
  base. Local content acceptance, remote publication and installed native
  consumption are separate results.
