# bench-kit
A results standard for benchmark repos, with JSON Schemas, a renderer, small-sample stats, a linter and an Every Eval Ever exporter (STANDARD.md is the source of truth).

## Status
Updated 2026-10-09. State: active, v0.4.0. Next: see NEXT.md.

## Latest result
bench-kit is tooling, not a bench, so it has no runs. The standard it enforces is in STANDARD.md.

## How it works
Every bench repo keeps its own runner. After a run it writes `runs/<id>/manifest.json`
(schema `bench-manifest/1`) and `runs/<id>/samples.jsonl` (schema `bench-sample/1`), one row
per task x subject x repeat. bench-kit then:

- `bench-kit stats`: computes means, 95% intervals (t, or Wilson for 0/1), paired per-task
  comparisons with wins/ties/losses and a sign test, error counts and token totals from
  samples.jsonl into the manifest. The task is the unit; repeats are averaged within a task first.
  Optional: cluster-robust intervals (`stats.cluster_by`), per-subject task and repeat counts, and
  per-metric task subsets (`task_filter`, e.g. recall over positive tasks only).
- `bench-kit render`: regenerates the README latest block (between the `bench:latest` markers),
  RESULTS.md and runs/INDEX.md from all manifests.
- `bench-kit lint`: checks the repo against the standard, rules BL001-BL010 (`bench-kit rules`).
- `bench-kit export-eee <run> [--org NAME]`: writes Every Eval Ever 0.3.0 aggregate + samples
  files and validates them against the vendored EEE schemas.

## Run it
```bash
# from a bench repo checkout
uv tool install git+https://github.com/aaronjmars/bench-kit@v0.4.0   # or: pipx install git+https://...
bench-kit stats && bench-kit render && bench-kit lint
bench-kit export-eee <run-id> --out eee-export --org "Your Org"
```

In any repo's GitHub Actions CI:
```yaml
- uses: actions/checkout@v4
- uses: aaronjmars/bench-kit@v0.4.0
  with:
    args: --warn-only   # drop once the repo is migrated
```

Develop:
```bash
uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .
```

## Layout
- `STANDARD.md`: the standard (files, README shape, manifest and samples fields, stats rules, lint rules)
- `schema/`: `bench-manifest-1.schema.json`, `bench-sample-1.schema.json`, and `eee/` (vendored Every Eval Ever 0.3.0 schemas, MIT, see `schema/eee/SOURCE.md`)
- `src/bench_kit/`: `stats.py`, `render.py`, `lint.py`, `export_eee.py`, `cli.py`
- `action.yml`: composite action that installs bench-kit from the action checkout and runs `bench-kit lint`
- `tests/`: pytest, including hand-computed stats checks

## License

MIT, see LICENSE. The vendored Every Eval Ever schemas in schema/eee/ keep their own MIT license (see schema/eee/SOURCE.md).
