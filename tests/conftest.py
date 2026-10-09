import json
from pathlib import Path

import pytest

from bench_kit import render, stats

README = """# demo-bench
Measures demo things.

## Status
Updated 2026-10-09. State: active. Next: see NEXT.md.

## Latest result
<!-- bench:latest:start -->
<!-- bench:latest:end -->

## How it works
Runs tasks.

## Run it
make run

## Layout
runs/ run records
"""

METHOD = """# demo-bench method

## Changelog
### [1-A] - 2026-10-01
First frozen set.
"""


def manifest(**over) -> dict:
    m = {
        "schema_version": "bench-manifest/1",
        "id": "2026-10-09-ab",
        "bench": "demo-bench",
        "question": "Is B better than A?",
        "started_at": "2026-10-09T10:00:00Z",
        "finished_at": "2026-10-09T10:30:00Z",
        "revision": {"repo": "o/demo-bench", "commit": "abc1234", "dirty": False},
        "setup": "standard",
        "source_data": {"dataset_name": "demo", "version": "1-A", "n_planned": 3, "n_completed": 3},
        "subjects": [
            {"name": "A", "model_info": {"id": "claude-haiku-5-5", "developer": "anthropic"}},
            {"name": "B", "model_info": {"id": "claude-sonnet-5-5", "developer": "anthropic"}},
        ],
        "repeats": {"planned": 2, "completed": 2, "aggregation": "mean"},
        "metric_config": {
            "score": {
                "metric_name": "Score",
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0,
                "max_score": 5,
                "metric_parameters": {"from": "score"},
            },
            "solved": {"metric_name": "Solved", "lower_is_better": False, "score_type": "binary", "metric_parameters": {"from": "is_correct"}},
        },
        "primary_metric": "score",
        "evaluation_results": {},
        "comparisons": [{"baseline": "A", "candidate": "B", "metric_id": "score"}],
        "errors": {"n": 0, "of": 0, "counted_as": "zero"},
        "cost": {"usd": 1.5, "basis": "api", "prices_as_of": "2026-10-09"},
        "decision_rule": "SWITCH if CI excludes 0",
        "verdict": "INCONCLUSIVE",
        "status": "complete",
        "raw": {"kept": True, "location": "runs/2026-10-09-ab/"},
        "corrections": [],
    }
    m.update(over)
    return m


# per task means: A t1=1, t2=2, t3=3 ; B t1=2, t2=2, t3=5 (diffs 1, 0, 2)
SAMPLE_SCORES = {
    ("A", "t1"): [1.0, 1.0],
    ("A", "t2"): [1.5, 2.5],
    ("A", "t3"): [3.0, 3.0],
    ("B", "t1"): [2.0, 2.0],
    ("B", "t2"): [2.0, 2.0],
    ("B", "t3"): [4.0, 6.0],
}


def samples() -> list[dict]:
    rows = []
    for (subj, task), vals in SAMPLE_SCORES.items():
        for rep, v in enumerate(vals, 1):
            rows.append(
                {
                    "sample_id": task,
                    "subject": subj,
                    "repeat": rep,
                    "evaluation": {"score": v, "is_correct": v >= 2},
                    "token_usage": {"input_tokens": 10, "output_tokens": 5},
                    "error": None,
                }
            )
    return rows


def write_repo(root: Path, m: dict | None = None, rows: list[dict] | None = None, run_stats: bool = True) -> Path:
    m = m or manifest()
    rows = samples() if rows is None else rows
    (root / "README.md").write_text(README)
    (root / "METHOD.md").write_text(METHOD)
    (root / "NEXT.md").write_text("# Next\n\nNothing yet (2026-10-09).\n")
    d = root / "runs" / m["id"]
    d.mkdir(parents=True)
    if run_stats:
        m = stats.apply(m, rows)
    (d / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    render.render(root)
    return d


@pytest.fixture
def repo(tmp_path):
    write_repo(tmp_path)
    return tmp_path
