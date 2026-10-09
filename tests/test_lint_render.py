import json

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
