"""Per-metric task subsets (bench-kit 0.4.0): task_filter, metric_parameters.field and .missing.

The fixture is a generic find-the-bug bench: 4 positive tasks (each has known bugs, the per-row value is the
share found) and 2 clean controls (no bug; a row records whether the report raised a false alarm). The
blended per-task score mixes both, so the bench reports recall over positives and false_alarm over controls.
"""

import json

import pytest
from conftest import manifest, samples, write_repo
from jsonschema import Draft7Validator

from bench_kit import export_eee, lint, render, stats
from bench_kit.io import load_run, load_schema

POS = ["p1", "p2", "p3", "p4"]
CLEAN = ["n1", "n2"]


def recall_cfg(**params) -> dict:
    return {
        "metric_name": "Recall on positive tasks",
        "lower_is_better": False,
        "score_type": "continuous",
        "min_score": 0,
        "max_score": 1,
        "metric_parameters": {"from": "score", "missing": "zero", **params},
        "task_filter": {"field": "metadata.kind", "equals": "positive"},
    }


def false_alarm_cfg() -> dict:
    return {
        "metric_name": "False alarms on clean controls",
        "lower_is_better": True,
        "score_type": "continuous",
        "min_score": 0,
        "max_score": 1,
        "metric_parameters": {"from": "score", "field": "metadata.false_alarm", "missing": "excluded"},
        "task_filter": {"field": "metadata.kind", "equals": "clean"},
    }


def row(subj, task, rep, score, false_alarm=None, error=False, kind=True) -> dict:
    meta = {}
    if kind:
        meta["kind"] = "positive" if task in POS else "clean"
    if false_alarm is not None:
        meta["false_alarm"] = false_alarm
    return {
        "sample_id": task,
        "subject": subj,
        "repeat": rep,
        "evaluation": {"score": score, "is_correct": None if score is None else score >= 1},
        "error": {"source": "agent", "type": "Timeout", "message": "timed out"} if error else None,
        "metadata": meta,
    }


def bug_rows() -> list[dict]:
    rows = []
    # "quiet" reports nothing: no bug found, clean controls pass with an empty report
    for rep in (1, 2):
        rows += [row("quiet", t, rep, 0.0) for t in POS]
        rows += [row("quiet", t, rep, 1.0, false_alarm=0.0) for t in CLEAN]
    # "finder": recall per task p1 1, p2 (1 + errored 0) / 2, p3 0.5, p4 (0 + 1) / 2
    vals = {"p1": [1.0, 1.0], "p2": [1.0, None], "p3": [0.5, 0.5], "p4": [0.0, 1.0]}
    for t, vs in vals.items():
        for rep, v in enumerate(vs, 1):
            # the errored row carries no metadata at all: it takes its task's kind from the other rows
            rows.append(row("finder", t, rep, v, error=v is None, kind=v is not None))
    # n1: one false alarm, one trial with no report (no false_alarm value: left out); n2: one alarm, one clean
    rows += [row("finder", "n1", 1, 0.0, false_alarm=1.0), row("finder", "n1", 2, 0.0)]
    rows += [row("finder", "n2", 1, 0.0, false_alarm=1.0), row("finder", "n2", 2, 1.0, false_alarm=0.0)]
    return rows


def bug_manifest(**over) -> dict:
    m = manifest(
        source_data={"dataset_name": "demo-bugs", "version": "1-A", "n_planned": 6, "n_completed": 6},
        subjects=[
            {"name": "quiet", "model_info": {"id": "claude-haiku-5-5", "developer": "anthropic"}},
            {"name": "finder", "model_info": {"id": "claude-sonnet-5-5", "developer": "anthropic"}},
        ],
        metric_config={
            "recall": recall_cfg(),
            "false_alarm": false_alarm_cfg(),
            "reward": {
                "metric_name": "Blended reward",
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0,
                "max_score": 1,
                "metric_parameters": {"from": "score"},
            },
        },
        primary_metric="recall",
        comparisons=[
            {"baseline": "quiet", "candidate": "finder", "metric_id": "recall"},
            {"baseline": "quiet", "candidate": "finder", "metric_id": "false_alarm"},
            {"baseline": "quiet", "candidate": "finder", "metric_id": "reward"},
        ],
    )
    m.update(over)
    return m


def test_report_nothing_agent_gets_zero_recall_and_zero_false_alarm():
    out = stats.compute(bug_manifest(), bug_rows())
    q = out["evaluation_results"]["quiet"]
    assert q["recall"]["score"] == 0
    assert q["recall"]["uncertainty"]["num_samples"] == 4
    assert q["recall"]["subset"] == {"tasks": 4, "of": 6, "rows_zeroed": 0, "rows_excluded": 0, "filter": "metadata.kind = positive"}
    assert q["false_alarm"]["score"] == 0
    assert q["false_alarm"]["uncertainty"]["num_samples"] == 2
    # the blended score still gives the do-nothing agent 2 of 6
    assert q["reward"]["score"] == pytest.approx(1 / 3, abs=1e-4)
    assert "subset" not in q["reward"]


def test_filtered_values_hand_computed():
    out = stats.compute(bug_manifest(), bug_rows())
    f = out["evaluation_results"]["finder"]
    # recall per task [1, 0.5, 0.5, 0.5]: mean 0.625, sd 0.25, se 0.125, t(3) 3.182 -> [0.22725, 1.02275] clipped to 1
    assert f["recall"]["score"] == 0.625
    ci = f["recall"]["uncertainty"]["confidence_interval"]
    assert ci["lower"] == pytest.approx(0.22725, abs=1e-4) and ci["upper"] == 1
    assert f["recall"]["subset"]["rows_zeroed"] == 1  # the errored p2 trial counts as 0
    # false_alarm per task n1 1 (one row left out), n2 0.5
    assert f["false_alarm"]["score"] == 0.75
    assert f["false_alarm"]["uncertainty"]["num_samples"] == 2
    assert f["false_alarm"]["subset"]["rows_excluded"] == 1


def test_comparisons_use_only_filtered_tasks():
    comps = {c["metric_id"]: c for c in stats.compute(bug_manifest(), bug_rows())["comparisons"]}
    rec = comps["recall"]
    assert rec["uncertainty"]["num_samples"] == 4
    assert rec["diff"] == 0.625
    assert (rec["wins"], rec["ties"], rec["losses"]) == (4, 0, 0)
    assert rec["sign_test_p"] == 0.125
    assert rec["uncertainty"]["confidence_interval"]["lower"] == pytest.approx(0.22725, abs=1e-4)
    fa = comps["false_alarm"]
    assert fa["uncertainty"]["num_samples"] == 2
    assert fa["diff"] == 0.75
    assert (fa["wins"], fa["ties"], fa["losses"]) == (0, 0, 2)  # lower is better: more alarms lose
    assert comps["reward"]["uncertainty"]["num_samples"] == 6
    assert "unpaired" not in rec


def test_missing_rules():
    rows = bug_rows()
    m = recall_cfg()
    keep = stats.filter_tasks(rows, m)
    assert keep == set(POS)
    zero = stats.per_task(rows, "finder", m, "zero", keep)
    assert zero["p2"] == 0.5
    m_ex = recall_cfg(missing="excluded")
    assert stats.per_task(rows, "finder", m_ex, "zero", keep)["p2"] == 1.0
    # unset: errored rows follow errors.counted_as (0.3.0 behaviour)
    m_def = recall_cfg()
    del m_def["metric_parameters"]["missing"]
    assert stats.per_task(rows, "finder", m_def, "zero", keep)["p2"] == 0.5
    assert stats.per_task(rows, "finder", m_def, "excluded", keep)["p2"] == 1.0
    # missing zero also zeroes rows that are not errors
    fa_zero = {**false_alarm_cfg(), "metric_parameters": {"from": "score", "field": "metadata.false_alarm", "missing": "zero"}}
    assert stats.per_task(rows, "finder", fa_zero, "zero", stats.filter_tasks(rows, fa_zero))["n1"] == 0.5


def test_filter_by_in_and_sample_ids():
    rows = bug_rows()
    assert stats.filter_tasks(rows, {"task_filter": {"field": "metadata.kind", "in": ["positive", "clean"]}}) == set(POS + CLEAN)
    assert stats.filter_tasks(rows, {"task_filter": {"sample_ids": ["p1", "n2", "missing-task"]}}) == {"p1", "n2"}
    both = {"task_filter": {"field": "metadata.kind", "equals": "positive", "sample_ids": ["p1", "n2"]}}
    assert stats.filter_tasks(rows, both) == {"p1"}
    assert stats.filter_tasks(rows, {"metric_parameters": {"from": "score"}}) is None


def test_task_with_two_filter_values_is_an_error():
    rows = bug_rows()
    rows[0]["metadata"]["kind"] = "clean"  # quiet p1 repeat 1
    with pytest.raises(ValueError, match="more than one value"):
        stats.compute(bug_manifest(), rows)


def test_bad_options_are_errors():
    rows = bug_rows()
    for tf in ({"equals": "x"}, {"field": "metadata.kind"}, {"field": "metadata.kind", "equals": "a", "in": ["a"]}, {"foo": 1}):
        with pytest.raises(ValueError):
            stats.filter_tasks(rows, {"task_filter": tf})
    with pytest.raises(ValueError, match="missing"):
        stats.filter_tasks(rows, {"metric_parameters": {"missing": "drop"}})


def test_rate_metric_from_a_field_and_per_repeat():
    rows = bug_rows()
    for x in rows:
        if x["metadata"].get("kind") == "clean" and "false_alarm" in x["metadata"]:
            x["metadata"]["alarm"] = x["metadata"]["false_alarm"] > 0
    m = bug_manifest(
        metric_config={
            "alarm_rate": {
                "metric_name": "Alarm",
                "lower_is_better": True,
                "score_type": "binary",
                "metric_parameters": {"from": "is_correct", "field": "metadata.alarm", "missing": "excluded"},
                "task_filter": {"field": "metadata.kind", "equals": "clean"},
            }
        },
        primary_metric="alarm_rate",
        comparisons=[],
    )
    out = stats.compute(m, rows)["evaluation_results"]
    assert out["finder"]["alarm_rate"]["score"] == 0.75
    assert out["finder"]["alarm_rate"]["per_repeat"] == [1.0, 0.0]  # repeat 2: n1 has no value, n2 clean
    assert out["quiet"]["alarm_rate"]["uncertainty"]["confidence_interval"]["method"] == "wilson"
    rows[0]["metadata"]["alarm"] = 0.5
    rows[0]["metadata"]["kind"] = "positive"
    bad = bug_manifest(
        metric_config={"r": {**m["metric_config"]["alarm_rate"], "task_filter": {"sample_ids": ["p1"]}}}, primary_metric="r", comparisons=[]
    )
    with pytest.raises(ValueError, match="rate metrics"):
        stats.compute(bad, rows)


def test_filtered_and_clustered():
    rows = bug_rows()
    for x in rows:
        x["cluster"] = "a" if x["sample_id"] in ("p1", "p2", "n1") else "b"
    out = stats.compute(bug_manifest(stats={"cluster_by": "cluster"}), rows)
    rec = out["evaluation_results"]["finder"]["recall"]
    assert rec["uncertainty"]["num_samples"] == 4 and rec["uncertainty"]["num_clusters"] == 2


def test_no_options_means_no_subset_and_same_numbers():
    m = manifest()
    out = stats.compute(m, samples())
    assert all("subset" not in sd for res in out["evaluation_results"].values() for sd in res.values())
    assert out["evaluation_results"]["A"]["score"]["score"] == 2.0


def test_subset_run_lints_clean_and_renders_n(tmp_path):
    write_repo(tmp_path, bug_manifest(), bug_rows())
    assert [f.plain() for f in lint.lint(tmp_path) if f.level == "error"] == []
    results = (tmp_path / "RESULTS.md").read_text()
    assert (
        "- Metric subsets: recall: tasks where metadata.kind = positive, no value counts as 0; false_alarm: tasks where metadata.kind = clean"
        in results
    )
    assert "finder 0.625 [0.227, 1] (4 of 6 tasks; 1 row with no value counted as 0)" in results
    assert "false_alarm: quiet 0 [0, 0] (2 of 6 tasks), finder 0.75 [0, 1] (2 of 6 tasks; 1 row with no value left out)" in results
    assert "finder vs quiet on recall: diff +0.625 [0.227, 1.023], n=4 tasks" in results
    readme = (tmp_path / "README.md").read_text()
    assert "| finder | 0.625 [0.227, 1] (4 of 6 tasks; 1 row with no value counted as 0) | 4 |" in readme


def test_bl004_rechecks_filtered_numbers_and_counts(tmp_path):
    d = write_repo(tmp_path, bug_manifest(), bug_rows())
    data = json.loads((d / "manifest.json").read_text())
    data["evaluation_results"]["finder"]["recall"]["subset"]["tasks"] = 6
    data["comparisons"][0]["uncertainty"]["num_samples"] = 6
    data["evaluation_results"]["quiet"]["false_alarm"]["score"] = 0.5
    (d / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    render.render(tmp_path)
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL004"]
    assert any("finder.recall.subset.tasks is 6, samples give 4" in x for x in msgs)
    assert any("comparisons[0].num_samples is 6, samples give 4" in x for x in msgs)
    assert any("quiet.false_alarm.score is 0.5, samples give 0" in x for x in msgs)


def test_bl004_reports_conflicting_filter_values(tmp_path):
    rows = bug_rows()
    write_repo(tmp_path, bug_manifest(), rows)
    rows[0]["metadata"]["kind"] = "clean"
    (tmp_path / "runs" / "2026-10-09-ab" / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL004"]
    assert any("cannot compute stats" in x and "more than one value" in x for x in msgs)


def test_schema_accepts_options_and_rejects_bad_filters():
    v = Draft7Validator(load_schema("bench-manifest-1.schema.json"))
    m = stats.apply(bug_manifest(), bug_rows())
    assert list(v.iter_errors(m)) == []
    for tf in ({"equals": "x"}, {"field": "metadata.kind"}, {"field": "metadata.kind", "equals": "a", "in": ["a"]}, {}, {"sample_ids": []}):
        bad = bug_manifest()
        bad["metric_config"]["recall"]["task_filter"] = tf
        assert list(v.iter_errors(bad)), tf
    bad = bug_manifest()
    bad["metric_config"]["recall"]["metric_parameters"]["missing"] = "drop"
    assert list(v.iter_errors(bad))


def test_export_eee_carries_filter_as_text(tmp_path):
    d = write_repo(tmp_path, bug_manifest(), bug_rows())
    aggs = [p for p in export_eee.export(load_run(d), tmp_path / "eee") if p.suffix == ".json"]
    agg = json.loads(aggs[0].read_text())
    rec = next(r for r in agg["evaluation_results"] if r["evaluation_name"].endswith(":recall"))
    assert rec["metric_config"]["metric_parameters"]["task_filter"] == "metadata.kind = positive"
    assert rec["metric_config"]["metric_parameters"]["missing"] == "zero"
    assert rec["source_data"]["additional_details"]["n_tasks_in_subset"] == "4"
    assert rec["score_details"]["uncertainty"]["num_samples"] == 4
