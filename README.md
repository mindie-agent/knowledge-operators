# Operators knowledge

This independent local repository holds operator mechanisms and their measured
conditions. Its first unit was admitted from the actual C4 RMSNorm receipt and
exported through the current graded knowledge pipeline. It contains no migrated
vllm-ascend task corpus.

The repository is a **local development draft**. There is no
`publication-contract.json`: an exact reviewed runtime commit implementing
contract/2 must be pinned before formal Feed integration or publication. The
recorded runtime base commit alone is not that implementation. No remote or CI
acceptance is claimed.

- `normalization/rmsnorm/unit-residual-free-rmsnorm-fastpath.md`: the L4 mutator,
  exact applicability conditions, measured run identities, and development limits.
- `index/routing-manifest.json` and `index/shard-topology.json`: generated routing
  metadata; the Markdown unit is authoritative.
- `docs/evidence/c4-assembly-health-ablation.json`: mechanically redacted measured
  receipt. Its original and redacted digests are distinguished in
  `docs/evidence/provenance.json`.

The claim concerns the recorded single development case. It does not establish
heldout generalization, a full-model serving gain, or product-level productivity.
Keep the raw source digest, failed or uncertain boundaries, and exact source
conditions when correcting or extending a unit.
