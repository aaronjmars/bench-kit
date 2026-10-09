"""Loading manifests, samples and schemas from a bench repo."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent


def schema_dir() -> Path:
    """Schemas ship inside the wheel as bench_kit/schema; in a source checkout they live at repo root."""
    for cand in (PKG / "schema", PKG.parent.parent / "schema"):
        if (cand / "bench-manifest-1.schema.json").exists():
            return cand
    raise FileNotFoundError("bench-kit schema directory not found")


def load_schema(name: str) -> dict:
    return json.loads((schema_dir() / name).read_text())


@dataclass
class Run:
    dir: Path
    manifest: dict
    manifest_error: str | None = None
    samples: list[dict] | None = None
    samples_errors: list[tuple[int, str]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.manifest.get("id", self.dir.name)

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def samples_path(self) -> Path:
        return self.dir / "samples.jsonl"


def load_samples(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    rows, errs = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            errs.append((i, f"invalid JSON: {e.msg}"))
    return rows, errs


def load_run(d: Path) -> Run:
    mp = d / "manifest.json"
    try:
        manifest = json.loads(mp.read_text())
        err = None if isinstance(manifest, dict) else "manifest is not a JSON object"
        if err:
            manifest = {}
    except json.JSONDecodeError as e:
        manifest, err = {}, f"invalid JSON: {e.msg} (line {e.lineno})"
    run = Run(dir=d, manifest=manifest, manifest_error=err)
    sp = d / "samples.jsonl"
    if sp.exists():
        run.samples, run.samples_errors = load_samples(sp)
    return run


def started(run: Run) -> datetime:
    """started_at as an aware UTC datetime; no zone means UTC; unparseable sorts first."""
    raw = str(run.manifest.get("started_at") or "")
    try:
        t = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def discover(root: Path) -> list[Run]:
    """Every runs/<id>/manifest.json, newest first (started_at in UTC, then id)."""
    runs = [load_run(p.parent) for p in sorted((root / "runs").glob("*/manifest.json"))]
    runs.sort(key=lambda r: (started(r), r.id), reverse=True)
    return runs


def dump_json(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
