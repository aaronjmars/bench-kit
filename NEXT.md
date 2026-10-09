# Next

Updated 2026-10-09.

1. Paired Bayes interval for small-n pass rates (Bowyer et al. 2025), next to the paired t.
2. Converters into manifest + samples: Harbor job/trial `result.json`, and EVMbench-style per-audit `grade.json` results.
3. Noise band helper: given two manifests of the same config, report the A/A difference for METHOD.md.
4. Olympic mean and median helpers for timing benches, so they no longer need `from: external`.
5. Clustered intervals for rate metrics with few clusters: a cluster bootstrap or design-effect Wilson next to CR1 t.
6. Lint tolerance relative to the metric scale, so metrics with very small values are checked as tightly as large ones.
7. export-eee: carry sample transcripts into EEE `messages` for multi_turn and agentic runs (today an empty list).

Done in 0.3.0 (2026-10-09): significant-figure formatting for small values, schema-valid single_turn
EEE export, clustered intervals (`stats.cluster_by`), per-subject task counts and repeats.
