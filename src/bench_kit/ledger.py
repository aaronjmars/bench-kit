"""Sweep ledgers (STANDARD.md section 11): one tracked file per sweep, `data/sweeps/<sweep-id>.json`.

Raw transcripts may stay out of git, but the per-trial verdicts of a sweep (what ran, what it
scored, what errored) are small and must be committed, so a paid sweep can be re-read after the
local run folders are gone. `write()` merges: call it after every trial (a crash then loses
nothing) and once more at sweep end with the final status.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft7Validator

from .io import discover, dump_json, load_schema

LEDGER_DIR = Path("data") / "sweeps"
SCHEMA = "bench-sweep/1"
STATUSES = ("running", "finished", "stopped", "halted")
MSG_MAX = 300


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ledger_path(root: Path, sweep_id: str, ledger_dir: Path = LEDGER_DIR) -> Path:
    if not sweep_id or "/" in sweep_id or sweep_id.startswith("."):
        raise ValueError(f"bad sweep id {sweep_id!r}")
    return Path(root) / ledger_dir / f"{sweep_id}.json"


def _key(r: dict) -> tuple:
    return (r.get("run_id"), r.get("sample_id"), r.get("subject"), r.get("repeat"))


def _short_error(err) -> dict | None:
    if not err:
        return None
    out = {k: err.get(k) for k in ("source", "type") if err.get(k) is not None}
    out["message"] = str(err.get("message") or "")[:MSG_MAX]
    return out


def records_from_samples(run_id: str, rows: Iterable[dict]) -> list[dict]:
    """One ledger record per sample row: verdict fields only, never outputs or transcripts."""
    out = []
    for row in rows:
        ev = row.get("evaluation") or {}
        err = _short_error(row.get("error"))
        rec = {
            "run_id": run_id,
            "sample_id": row.get("sample_id"),
            "subject": row.get("subject"),
            "repeat": row.get("repeat"),
            "status": "errored" if err else "scored",
            "score": ev.get("score"),
            "is_correct": ev.get("is_correct"),
            "error": err,
        }
        if row.get("cost_usd") is not None:
            rec["cost_usd"] = row["cost_usd"]
        out.append(rec)
    return out


def load_records(path: Path) -> list[dict]:
    """Records from a JSONL file the bench's own runner wrote (one object per line)."""
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e.msg}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{i}: not a JSON object")
            if rec.get("error"):
                rec["error"] = _short_error(rec["error"])
            out.append(rec)
    return out


def counts(records: list[dict]) -> dict:
    c = {"records": len(records), "scored": 0, "errored": 0, "skipped": 0, "auth_errors": 0}
    for r in records:
        if r.get("status") in ("scored", "errored", "skipped"):
            c[r["status"]] += 1
        if (r.get("error") or {}).get("type") == "auth":
            c["auth_errors"] += 1
    return c


def write(
    root: Path,
    sweep_id: str,
    runs: Iterable[str] = (),
    records: Iterable[dict] = (),
    status: str = "running",
    halted: str | None = None,
    note: str | None = None,
    bench: str | None = None,
    ledger_dir: Path = LEDGER_DIR,
) -> Path:
    """Create or update the ledger: add the sample rows of `runs` (run ids under runs/) and the given
    records, newest wins per (run_id, sample_id, subject, repeat). Writes atomically."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    root = Path(root)
    path = ledger_path(root, sweep_id, ledger_dir)
    old = json.loads(path.read_text()) if path.is_file() else {}
    merged = {_key(r): r for r in old.get("records", [])}
    run_ids = list(old.get("runs", []))
    want = list(runs)
    if want:
        found = {r.dir.name: r for r in discover(root)}
        for rid in want:
            run = found.get(rid)
            if run is None or run.samples is None:
                raise ValueError(f"run {rid} has no runs/{rid}/samples.jsonl")
            for rec in records_from_samples(rid, run.samples):
                merged[_key(rec)] = rec
            if bench is None:
                bench = run.manifest.get("bench")
            if rid not in run_ids:
                run_ids.append(rid)
    for rec in records:
        merged[_key(rec)] = rec
        rid = rec.get("run_id")
        if rid and rid not in run_ids:
            run_ids.append(rid)
    recs = sorted(merged.values(), key=lambda r: tuple(str(x) for x in _key(r)))
    data = {
        "schema_version": SCHEMA,
        "sweep_id": sweep_id,
        "bench": bench or old.get("bench") or root.resolve().name,
        "started_at": old.get("started_at") or now(),
        "updated_at": now(),
        "status": status,
        "halted": halted if halted is not None else (old.get("halted") if status in ("halted", "stopped") else None),
        "note": note if note is not None else old.get("note"),
        "runs": run_ids,
        "counts": counts(recs),
        "records": recs,
    }
    errs = sorted(Draft7Validator(load_schema("bench-sweep-1.schema.json")).iter_errors(data), key=lambda e: list(e.absolute_path))
    if errs:
        where = "/".join(str(x) for x in errs[0].absolute_path) or "(root)"
        raise ValueError(f"ledger would not match {SCHEMA}: {where}: {errs[0].message}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(dump_json(data))
    tmp.replace(path)
    return path


def ignored(root: Path, rel_paths: Iterable[str]) -> list[str] | None:
    """Which of these repo-relative paths git would ignore (tracked files never count as ignored).
    None when root is not in a git work tree or git is missing."""
    paths = [str(p) for p in rel_paths]
    if not paths:
        return []
    try:
        r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=root, input="\n".join(paths) + "\n", capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode not in (0, 1):
        return None
    return [x for x in r.stdout.splitlines() if x]
