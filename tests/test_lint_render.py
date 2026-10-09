import json
import re

from conftest import manifest, samples, write_repo

from bench_kit import cli, export_eee, lint, render, stats
from bench_kit.io import discover, load_run

EM = chr(0x2014)


def codes(root):
    return sorted({f.rule for f in lint.lint(root)})


def test_clean_repo_passes(repo):
    assert lint.lint(repo) == []


def test_render_fills_readme_results_index(repo):
    text = (repo / "README.md").read_text()
    assert "Run 2026-10-09-ab (2026-10-09)" in text
    assert "| A | 2 [" in text
    assert "W/T/L 2/1/0" in text
    res = (repo / "RESULTS.md").read_text()
    assert "## 2026-10-09-ab" in res and "Verdict: INCONCLUSIVE" in res
    assert "| [2026-10-09-ab](2026-10-09-ab/)" in (repo / "runs" / "INDEX.md").read_text()
    assert render.render(repo) == []  # idempotent


def test_bl003_stale_generated(repo):
    (repo / "RESULTS.md").write_text("# old\n")
    assert codes(repo) == ["BL003"]


def test_bl002_missing_section_and_date(tmp_path):
    write_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text().replace("## Layout", "## Files").replace("Updated 2026-10-09", "Updated today"))
    found = lint.lint(tmp_path)
    msgs = [f.message for f in found if f.rule == "BL002"]
    assert any("Layout" in m for m in msgs) and any("no YYYY-MM-DD" in m for m in msgs)


def test_bl001_schema_and_dir_mismatch(tmp_path):
    m = manifest(verdict="MAYBE")
    d = write_repo(tmp_path, m)
    data = json.loads((d / "manifest.json").read_text())
    data["id"] = "other-id"
    (d / "manifest.json").write_text(json.dumps(data))
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL001"]
    assert any("verdict" in m for m in msgs)
    assert any("differs from directory" in m for m in msgs)


def test_bl004_hand_edited_number(tmp_path):
    d = write_repo(tmp_path)
    data = json.loads((d / "manifest.json").read_text())
    data["evaluation_results"]["A"]["score"]["score"] = 2.5
    (d / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    render.render(tmp_path)
    found = [f for f in lint.lint(tmp_path) if f.rule == "BL004"]
    assert found and "samples give 2.0" in found[0].message


def test_bl004_row_count(tmp_path):
    rows = samples()[:-1]
    write_repo(tmp_path, rows=rows)
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL004"]
    assert any("subject B: 5 rows" in m for m in msgs)


def test_bl005_switch_needs_ci_excluding_zero(tmp_path):
    write_repo(tmp_path, manifest(verdict="SWITCH"))
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL005"]
    assert any("includes 0" in m for m in msgs)


def test_bl005_single_repeat(tmp_path):
    rows = [r for r in samples() if r["repeat"] == 1]
    m = manifest(verdict="WINNER:B", repeats={"planned": 1, "completed": 1})
    write_repo(tmp_path, m, rows)
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL005"]
    assert any("fewer than 2 repeats" in m for m in msgs)


def test_bl006_dash_and_suppression(repo):
    nxt = repo / "NEXT.md"
    nxt.write_text(f"# Next\n\nA {EM} B\n")
    found = [f for f in lint.lint(repo) if f.rule == "BL006"]
    assert found and found[0].line == 3
    nxt.write_text(f"# Next\n<!-- bench-lint: ignore BL006 quoting upstream text -->\nA {EM} B\n")
    assert [f for f in lint.lint(repo) if f.rule == "BL006"] == []


def test_repo_ignore_file(repo):
    (repo / "NEXT.md").write_text("# Next\n\nTODO later\n")
    assert "BL010" in codes(repo)
    (repo / ".bench-lint-ignore").write_text("BL010 NEXT.md planning notes\n")
    assert "BL010" not in codes(repo)


def test_bl007_bl008_bl009_warnings(tmp_path):
    m = manifest(
        subjects=[
            {"name": "A", "model_info": {"id": "gpt-latest"}, "harness": {"name": "omp"}},
            {"name": "B", "model_info": {"id": "claude-sonnet-5-5"}},
        ],
        revision={"repo": "o/r", "commit": "abc", "dirty": True},
        source_data={"dataset_name": "demo", "version": "2-A", "n_planned": 3, "n_completed": 3},
    )
    write_repo(tmp_path, m)
    found = lint.lint(tmp_path)
    assert {"BL007", "BL008", "BL009"} <= {f.rule for f in found}
    assert all(f.level == "warning" for f in found if f.rule in ("BL007", "BL008", "BL009"))


def test_cli_warn_only_exit_codes(tmp_path, capsys):
    write_repo(tmp_path, manifest(verdict="SWITCH"))
    assert cli.main(["lint", str(tmp_path), "--format", "github"]) == 1
    out = capsys.readouterr().out
    assert "::error file=runs/2026-10-09-ab/manifest.json,title=BL005::" in out
    assert cli.main(["lint", str(tmp_path), "--warn-only"]) == 0


def test_cli_stats_then_render(tmp_path):
    d = write_repo(tmp_path, run_stats=False)
    assert "BL004" in codes(tmp_path)
    assert cli.main(["stats", str(tmp_path)]) == 0
    assert cli.main(["render", str(tmp_path)]) == 0
    assert lint.lint(tmp_path) == []
    assert json.loads((d / "manifest.json").read_text())["comparisons"][0]["wins"] == 2


def test_export_eee_validates(tmp_path):
    d = write_repo(tmp_path)
    out = tmp_path / "eee"
    paths = export_eee.export(load_run(d), out)
    aggs = [p for p in paths if p.suffix == ".json"]
    assert len(aggs) == 2
    agg = json.loads(aggs[0].read_text())
    assert agg["schema_version"] == "0.3.0"
    assert agg["model_info"]["id"].startswith("anthropic/")
    der = agg["detailed_evaluation_results"]
    assert (out / der["file_path"]).exists() and der["total_rows"] == 6
    assert agg["source_metadata"]["source_organization_name"] == "unknown"


def test_export_eee_org(tmp_path):
    d = write_repo(tmp_path, manifest(source_organization={"name": "Acme Labs", "url": "https://example.com"}))
    agg = json.loads([p for p in export_eee.export(load_run(d), tmp_path / "a") if p.suffix == ".json"][0].read_text())
    assert agg["source_metadata"]["source_organization_name"] == "Acme Labs"
    assert agg["source_metadata"]["source_organization_url"] == "https://example.com"
    agg = json.loads([p for p in export_eee.export(load_run(d), tmp_path / "b", org="Other Org") if p.suffix == ".json"][0].read_text())
    assert agg["source_metadata"]["source_organization_name"] == "Other Org"
    assert "source_organization_url" not in agg["source_metadata"]


def test_latest_prefers_headline_and_sorts_by_utc(tmp_path):
    write_repo(tmp_path, manifest(id="2026-10-09-ab", headline=True))
    later = manifest(id="2026-10-10-other", started_at="2026-10-10T01:00:00+02:00")  # 2026-10-09T23:00Z
    d = tmp_path / "runs" / later["id"]
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(stats.apply(later, samples()), indent=2) + "\n")
    (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples()))
    runs = discover(tmp_path)
    assert [r.id for r in runs] == ["2026-10-10-other", "2026-10-09-ab"]
    assert render.latest_run(runs).id == "2026-10-09-ab"


# ---------------------------------------------------------------- v0.3.0


def test_num_and_money_keep_small_values():
    assert render.num(0.000412) == "0.000412"
    assert render.num(0.0123) == "0.0123"
    assert render.num(0.4902) == "0.49"
    assert render.num(0.12345) == "0.123"
    assert render.num(3.14159) == "3.142"
    assert render.num(-0.0004, True) == "-0.0004"
    assert render.num(0) == "0" and render.num(0.9996) == "1"
    assert render.money(0.00042) == "$0.00042"
    assert render.money(0.4) == "$0.40"
    assert render.money(0) == "$0.00"
    assert render.money(1234.5) == "$1,234.50"
    assert render.minutes(2) == "0.0333 min" and render.minutes(300) == "5.0 min"


def test_render_small_costs_everywhere(tmp_path):
    m = manifest(
        cost={"usd": 0.0021, "basis": "api", "per_subject": {"A": {"usd_per_task": 0.00042, "seconds_per_task": 3}}},
        seconds=12,
    )
    write_repo(tmp_path, m)
    readme = (tmp_path / "README.md").read_text()
    assert "| $0.00042 | 0.05 min |" in readme
    res = (tmp_path / "RESULTS.md").read_text()
    assert "Cost: $0.0021 total, api basis. Time: 0.2 min" in res
    assert not re.search(r"\$0\.00(?!\d)|0\.000(?!\d)", readme + res)
    assert lint.lint(tmp_path) == []


def test_clustered_run_lints_clean(tmp_path):
    rows = samples()
    for x in rows:
        x["cluster"] = "d1" if x["sample_id"] in ("t1", "t2") else "d2"
    write_repo(tmp_path, manifest(stats={"cluster_by": "cluster"}), rows)
    assert lint.lint(tmp_path) == []
    assert "n=3 tasks in 2 clusters" in (tmp_path / "RESULTS.md").read_text()


def test_bl004_cluster_spanning_two_values(tmp_path):
    rows = samples()
    for i, x in enumerate(rows):
        x["cluster"] = f"c{i}"
    write_repo(tmp_path, manifest(stats={"cluster_by": "cluster"}), rows, run_stats=False)
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL004"]
    assert any("more than one cluster value" in m for m in msgs)
    assert cli.main(["stats", str(tmp_path)]) == 1


def uneven_rows() -> list[dict]:
    return [x for x in samples() if not (x["subject"] == "B" and x["sample_id"] == "t3")]


def test_bl004_uneven_subjects_need_declaration(tmp_path):
    write_repo(tmp_path, rows=uneven_rows())
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL004"]
    assert any("subject B: 4 rows" in m for m in msgs)


def test_uneven_subjects_declared_pass_and_pair_shared_tasks(tmp_path):
    sd = {"dataset_name": "demo", "version": "1-A", "n_planned": 3, "n_completed": 3, "n_completed_by_subject": {"B": 2}}
    d = write_repo(tmp_path, manifest(source_data=sd), uneven_rows())
    assert lint.lint(tmp_path) == []
    c = json.loads((d / "manifest.json").read_text())["comparisons"][0]
    assert c["uncertainty"]["num_samples"] == 2 and c["unpaired"] == {"baseline": 1, "candidate": 0}
    res = (tmp_path / "RESULTS.md").read_text()
    assert "n=2 tasks shared (1 baseline-only and 0 candidate-only left out)" in res
    assert "(per subject, tasks x repeats: A 3 x 2, B 2 x 2)" in res


def test_bl004_per_subject_count_mismatch_and_unknown_subject(tmp_path):
    sd = {"dataset_name": "demo", "version": "1-A", "n_planned": 3, "n_completed": 3, "n_completed_by_subject": {"B": 3, "Z": 1}}
    write_repo(tmp_path, manifest(source_data=sd), uneven_rows())
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL004"]
    assert any("subject B: 4 rows, expected n_completed_by_subject x repeats.completed = 6" in m for m in msgs)
    assert any("'Z', which is not a subject" in m for m in msgs)


def test_uneven_repeats_declared_pass(tmp_path):
    rows = [x for x in samples() if not (x["subject"] == "B" and x["repeat"] == 2)]
    write_repo(tmp_path, rows=rows)
    assert any("subject B: 3 rows" in f.message for f in lint.lint(tmp_path) if f.rule == "BL004")
    rp = {"planned": 2, "completed": 2, "completed_by_subject": {"B": 1}}
    (tmp_path / "ok").mkdir()
    write_repo(tmp_path / "ok", manifest(repeats=rp), rows)
    assert lint.lint(tmp_path / "ok") == []


def test_bl005_uses_per_subject_repeats(tmp_path):
    rows = [x for x in samples() if not (x["subject"] == "B" and x["repeat"] == 2)]
    rp = {"planned": 2, "completed": 2, "completed_by_subject": {"B": 1}}
    write_repo(tmp_path, manifest(repeats=rp, verdict="WINNER:B"), rows)
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL005"]
    assert any("fewer than 2 repeats" in m for m in msgs)


def test_export_eee_single_turn_validates(tmp_path):
    from jsonschema import Draft7Validator

    from bench_kit.io import load_schema

    rows = samples()
    rows[0]["output"] = "The answer is 4."
    rows[1]["output"] = {"raw": ["4"], "reasoning_trace": "2 + 2"}
    d = write_repo(tmp_path, manifest(interaction_type="single_turn"), rows)
    assert lint.lint(tmp_path) == []
    paths = export_eee.export(load_run(d), tmp_path / "eee")
    inst = [json.loads(line) for p in paths if p.suffix == ".jsonl" for line in p.read_text().splitlines()]
    v = Draft7Validator(load_schema("eee/instance_level_eval.schema.json"))
    assert len(inst) == 12
    for row in inst:
        assert list(v.iter_errors(row)) == []
        assert row["interaction_type"] == "single_turn" and row["messages"] is None
    by_id = {r["sample_id"]: r["output"] for r in inst if r["model_id"].endswith("haiku-5-5")}
    assert by_id["t1#1"] == {"raw": ["The answer is 4."]}
    assert by_id["t1#2"] == {"raw": ["4"], "reasoning_trace": ["2 + 2"]}
    assert by_id["t2#1"] == {"raw": []}


def test_instance_output_accepts_list():
    out = export_eee.instance_output({"output": ["a", "b"]})
    assert out == {"raw": ["a", "b"]}


def test_v020_manifest_still_valid():
    from jsonschema import Draft7Validator

    from bench_kit.io import load_schema

    m = stats.apply(manifest(), samples())
    assert list(Draft7Validator(load_schema("bench-manifest-1.schema.json")).iter_errors(m)) == []
    assert m["schema_version"] == "bench-manifest/1"
