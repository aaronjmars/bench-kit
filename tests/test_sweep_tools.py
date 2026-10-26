import json
import subprocess

import pytest
from conftest import manifest, samples, write_repo

from bench_kit import cli, ledger, lint, runcheck, scrub

KEY = "sk-test-" + "a1b2c3d4" * 5
SHORTER = KEY[:20]


def codes(root):
    return sorted({f.rule for f in lint.lint(root)})


def git_init(root):
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


# ------------------------------------------------------------------ scrub


def test_scrub_env_glob_longest_first_and_json_escaped(tmp_path):
    run = tmp_path / "run"
    (run / "raw").mkdir(parents=True)
    url = f"https://host.example/v1?key={KEY}"
    (run / "raw" / "log.txt").write_text(f"calling {url}\nprefix {SHORTER}\n")
    slashy = "abc/def/ghi/jkl/" + "x" * 10
    (run / "raw" / "t.json").write_text(json.dumps({"auth": slashy}).replace("/", "\\/") + "\n")
    (run / "raw" / "blob.bin").write_bytes(b"\0\xff" + KEY.encode())
    env = {"DEMO_API_KEY": KEY, "DEMO_URL": url, "SHORT_TOKEN": "abc", "OTHER": "not-a-secret-at-all", "SLASHY_TOKEN": slashy}
    vals = scrub.values_from_env(["*_API_KEY", "DEMO_URL", "*_TOKEN"], env)
    assert {label for _, label in vals} == {"DEMO_API_KEY", "DEMO_URL", "SHORT_TOKEN", "SLASHY_TOKEN"}
    vals.append((SHORTER, "PREFIX"))
    res = scrub.scrub([run], vals)
    assert res.skipped_short == ["SHORT_TOKEN"]
    log = (run / "raw" / "log.txt").read_text()
    assert "calling <redacted:DEMO_URL>" in log and "<redacted:PREFIX>" in log and KEY not in log
    assert "<redacted:SLASHY_TOKEN>" in (run / "raw" / "t.json").read_text()
    assert (run / "raw" / "blob.bin").read_bytes().endswith(KEY.encode())  # binary left alone
    assert scrub.scrub([run], vals, check=True).hits == {}


def test_scrub_values_file(tmp_path):
    vf = tmp_path / "secrets.env"
    vf.write_text(f"# comment\nexport A_KEY='{KEY}'\n\nB_TOKEN=\nbare-value-0123456789\n")
    assert scrub.values_from_file(vf) == [(KEY, "A_KEY"), ("bare-value-0123456789", "secret")]


def test_scrub_cli_check_and_write(tmp_path, monkeypatch, capsys):
    f = tmp_path / "out.jsonl"
    f.write_text(json.dumps({"msg": f"key {KEY}"}) + "\n")
    monkeypatch.setenv("BK_TEST_API_KEY", KEY)
    assert cli.main(["scrub", str(f), "--env", "BK_TEST_*", "--check"]) == 1
    assert KEY in f.read_text()
    assert cli.main(["scrub", str(f), "--env", "BK_TEST_*"]) == 0
    assert cli.main(["scrub", str(f), "--env", "BK_TEST_*", "--check"]) == 0
    assert "<redacted:BK_TEST_API_KEY>" in f.read_text()
    out = capsys.readouterr().out
    assert KEY not in out
    assert cli.main(["scrub", str(f)]) == 1  # no values given


# ------------------------------------------------------------------ check-run

AUTH_LINES = [
    '{"type":"error","error":{"type":"authentication_error","message":"OAuth access token has been revoked."}}',
    "Error: 401 Unauthorized",
    "openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}",
    '{"error": {"message": "User not found.", "code": 401}, "status": 401}',
    "gh: Bad credentials (HTTP 401)",
    "Invalid API key. Please run /login",
    "codex: refresh token has expired, sign in again",
]
CLEAN_LINES = [
    '{"usage": {"input_tokens": 401, "output_tokens": 401, "total_tokens": 802}}',
    "processed 401 tasks, token budget ok",
    "the model said the key rotation design is invalid in two places",
]


@pytest.mark.parametrize("line", AUTH_LINES)
def test_auth_markers_found(line):
    assert runcheck.auth_hits(line, runcheck.compile_patterns())


@pytest.mark.parametrize("line", CLEAN_LINES)
def test_no_false_auth_on_ordinary_text(line):
    assert runcheck.auth_hits(line, runcheck.compile_patterns()) == []


def test_excerpt_masks_key():
    hits = runcheck.auth_hits(f"401 Unauthorized: invalid api key {KEY}", runcheck.compile_patterns(), "r.log")
    assert hits and KEY not in hits[0].excerpt and "<masked>" in hits[0].excerpt


def trial(root, name, text="all good", files=("result.json",)):
    d = root / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text(json.dumps({"exit_code": 0, "log": text}) + "\n")
    return d


def test_check_trial_auth_skip_only_require(tmp_path):
    t = trial(tmp_path, "t1", files=("result.json",))
    (t / "agent").mkdir()
    (t / "agent" / "transcript.jsonl").write_text('{"tool_output": "HTTP/1.1 401 Unauthorized from the target app"}\n')
    assert runcheck.check_trial(t).status == "auth-error"
    assert runcheck.check_trial(t, skip=["transcript.jsonl"]).status == "ok"
    assert runcheck.check_trial(t, only=["result.json"]).status == "ok"
    c = runcheck.check_trial(t, require=["result.json", "verifier/*.json"], skip=["agent/*"])
    assert c.status == "incomplete" and c.missing == ["verifier/*.json"]
    err = runcheck.sample_error(runcheck.check_trial(t))
    assert err["source"] == "infra" and err["type"] == "auth"
    assert runcheck.sample_error(runcheck.check_trial(t, skip=["agent/*"])) is None


def test_check_run_cli_exit_codes_and_streak(tmp_path, capsys):
    ok = trial(tmp_path, "ok")
    bad = [trial(tmp_path, f"bad{i}", text="AuthenticationError: invalid x-api-key") for i in range(3)]
    assert cli.main(["check-run", str(ok)]) == 0
    assert cli.main(["check-run", str(ok), "--require", "missing.txt"]) == 1
    assert cli.main(["check-run", str(bad[0]), str(ok)]) == 2  # auth error, streak reset by the clean trial
    assert cli.main(["check-run", *map(str, bad)]) == 3  # 3 in a row stops the sweep
    sf = tmp_path / "streak.json"
    assert cli.main(["check-run", str(bad[0]), "--streak-file", str(sf), "--stop-after", "2"]) == 2
    assert cli.main(["check-run", str(bad[1]), "--streak-file", str(sf), "--stop-after", "2"]) == 3
    assert cli.main(["check-run", str(ok), "--streak-file", str(sf), "--stop-after", "2"]) == 0
    assert json.loads(sf.read_text())["consecutive_auth_errors"] == 0
    capsys.readouterr()
    assert cli.main(["check-run", str(bad[0]), "--json"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["trials"][0]["status"] == "auth-error" and out["stop_sweep"] is False


def test_check_run_extra_pattern(tmp_path):
    t = trial(tmp_path, "t", text="provider said: SESSION_GONE")
    assert runcheck.check_trial(t).status == "ok"
    assert runcheck.check_trial(t, extra_patterns=[r"session_gone"]).status == "auth-error"


# ------------------------------------------------------------------ ledger


def test_ledger_from_run_merges_and_validates(tmp_path):
    rows = samples()
    rows[0]["error"] = {"source": "infra", "type": "auth", "message": "401 " + "x" * 500}
    rows[0]["evaluation"]["score"] = None
    write_repo(tmp_path, rows=rows)
    p = ledger.write(tmp_path, "sweep-1", runs=["2026-10-09-ab"], status="running")
    data = json.loads(p.read_text())
    assert p == tmp_path / "data" / "sweeps" / "sweep-1.json"
    assert data["schema_version"] == "bench-sweep/1" and data["bench"] == "demo-bench"
    assert data["counts"] == {"records": 12, "scored": 11, "errored": 1, "skipped": 0, "auth_errors": 1}
    errored = [r for r in data["records"] if r["status"] == "errored"][0]
    assert len(errored["error"]["message"]) == ledger.MSG_MAX
    started = data["started_at"]
    # a later call merges: same key replaced, new record added, started_at kept
    fixed = {"run_id": "2026-10-09-ab", "sample_id": rows[0]["sample_id"], "subject": rows[0]["subject"], "repeat": rows[0]["repeat"]}
    fixed.update(status="scored", score=1.0, error=None)
    extra = {"run_id": None, "sample_id": "t9", "subject": "C", "repeat": 1, "status": "skipped", "note": "endpoint stopped"}
    p = ledger.write(tmp_path, "sweep-1", records=[fixed, extra], status="halted", halted="3 consecutive auth errors")
    data = json.loads(p.read_text())
    assert data["started_at"] == started and data["status"] == "halted" and data["halted"].startswith("3 consecutive")
    assert data["counts"] == {"records": 13, "scored": 12, "errored": 0, "skipped": 1, "auth_errors": 0}
    assert data["runs"] == ["2026-10-09-ab"]
    assert codes(tmp_path) == ["BL013"]  # ledger is valid bench-sweep/1; the auth row still flags the run


def test_ledger_rejects_bad_input(tmp_path):
    write_repo(tmp_path)
    with pytest.raises(ValueError):
        ledger.write(tmp_path, "../x")
    with pytest.raises(ValueError):
        ledger.write(tmp_path, "s", runs=["no-such-run"])
    with pytest.raises(ValueError):
        ledger.write(tmp_path, "s", records=[{"sample_id": "t1", "subject": "A", "status": "done"}])


def test_ledger_cli_records_file_and_ignored_warning(tmp_path, capsys):
    write_repo(tmp_path)
    git_init(tmp_path)
    rec = tmp_path / "records.jsonl"
    rec.write_text(json.dumps({"run_id": "r1", "sample_id": "t1", "subject": "A", "repeat": 1, "status": "scored", "score": 1}) + "\n")
    assert cli.main(["ledger", "s1", "--root", str(tmp_path), "--records", str(rec), "--run", "2026-10-09-ab"]) == 0
    data = json.loads((tmp_path / "data/sweeps/s1.json").read_text())
    assert data["status"] == "finished" and data["runs"] == ["2026-10-09-ab", "r1"] and data["counts"]["records"] == 13
    (tmp_path / ".gitignore").write_text("data/\n")
    assert cli.main(["ledger", "s2", "--root", str(tmp_path), "--records", str(rec)]) == 2
    assert "gitignored" in capsys.readouterr().err


# ------------------------------------------------------------------ lint BL001 / BL012 / BL013


def test_bl012_ignored_ledger_dir_and_runs(tmp_path):
    write_repo(tmp_path)
    git_init(tmp_path)
    assert "BL012" not in codes(tmp_path)
    (tmp_path / ".gitignore").write_text("runs/*/raw/\n")  # raw transcripts ignored: fine
    assert "BL012" not in codes(tmp_path)
    (tmp_path / ".gitignore").write_text("/data/\n/runs/\n")
    found = [f for f in lint.lint(tmp_path) if f.rule == "BL012"]
    assert {f.path for f in found} == {"data/sweeps/", "runs/2026-10-09-ab/manifest.json", "runs/2026-10-09-ab/samples.jsonl"}
    assert all(f.level == "warning" for f in found)
    (tmp_path / ".bench-lint-ignore").write_text("BL012 * this repo keeps runs in an external store\n")
    assert "BL012" not in codes(tmp_path)


def test_bl012_skipped_outside_git(tmp_path):
    write_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("/data/\n")
    assert "BL012" not in codes(tmp_path)


def test_bl013_auth_rows_in_complete_run(tmp_path):
    rows = samples()
    rows[3]["error"] = {"source": "infra", "type": "auth", "message": "OAuth access token has been revoked"}
    write_repo(tmp_path, rows=rows)
    found = [f for f in lint.lint(tmp_path) if f.rule == "BL013"]
    assert len(found) == 1 and "1 row(s)" in found[0].message
    other = tmp_path / "other"
    other.mkdir()
    write_repo(other, manifest(status="partial"), rows=rows)
    assert "BL013" not in codes(other)


def test_bl001_bad_bench_sweep_ledger_and_foreign_format(tmp_path):
    write_repo(tmp_path)
    d = tmp_path / "data" / "sweeps"
    d.mkdir(parents=True)
    (d / "own-format.json").write_text(json.dumps({"header": {}, "records": [{"task": "x"}]}))
    assert codes(tmp_path) == []
    (d / "s1.json").write_text(json.dumps({"schema_version": "bench-sweep/1", "sweep_id": "other", "status": "done"}))
    msgs = [f.message for f in lint.lint(tmp_path) if f.rule == "BL001"]
    assert any("status" in m for m in msgs) and any("differs from file name" in m for m in msgs)
