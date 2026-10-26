"""Check one finished trial before it is scored (STANDARD.md section 11).

Two checks:

1. Auth errors. A trial whose output shows a provider auth failure (HTTP 401, a revoked or
   expired token, an invalid API key, a lost login) says nothing about the subject. It must be
   recorded as errored (`error.type: "auth"`), never scored and never counted as a clean exit,
   and a sweep stops after a few of them in a row, since every later trial on the same login
   fails the same way.
2. Completeness. Every file the bench needs to judge the trial later is there (`--require`).

Patterns are generic provider wording. Agent transcripts of benches whose tasks call HTTP
services can quote the target's own 401 responses: point `--only` at the harness result and
error files for those benches, or `--skip` the transcript.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

AUTH_PATTERNS = [
    r"\b401\b[^\n]{0,60}?(unauthori[sz]ed|authentication_error|authentication failed|invalid[ _-]?(x-)?api[ _-]?key|revoked)",
    r"(unauthori[sz]ed|authentication_error)[^\n]{0,60}?\b401\b",
    r"[\"']?status(_?code)?[\"']?\s*[:=]\s*401\b",
    r"\bHTTP(/[\d.]+)?\s+401\b",
    r"(access|oauth|refresh|bearer|auth)[ _-]?token[^\n]{0,40}?(has been |was |is )?(revoked|expired|invalid)",
    r"token (has been|was) revoked",
    r"invalid[ _-]?(x-)?api[ _-]?key",
    r"incorrect api key",
    r"api[ _-]?key[^\n]{0,20}?(is )?(not valid|invalid|revoked|expired|disabled|rejected)",
    r"authentication[ _](error|failed)",
    r"invalid[ _]authentication",
    r"bad credentials",
    r"not logged in",
    r"please run /?login\b",
]
SECRET_LIKE = re.compile(r"[A-Za-z0-9_\-.+/=]{24,}")
MAX_BYTES = 50_000_000


@dataclass
class Hit:
    path: str
    line: int
    excerpt: str

    def text(self) -> str:
        return f"{self.path}:{self.line}: {self.excerpt}"


@dataclass
class TrialCheck:
    trial: str
    auth: list[Hit] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.auth:
            return "auth-error"
        return "incomplete" if self.missing else "ok"

    def as_dict(self) -> dict:
        return {
            "trial": self.trial,
            "status": self.status,
            "auth_errors": [h.__dict__ for h in self.auth],
            "missing": self.missing,
        }


def compile_patterns(extra: Iterable[str] = ()) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in [*AUTH_PATTERNS, *extra]), re.IGNORECASE)


def mask(text: str) -> str:
    """Excerpt safe to print: long token-like strings (keys, JWTs) become <masked>."""
    return SECRET_LIKE.sub("<masked>", text.strip())[:160]


def _match(rel: str, globs: Iterable[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in globs)


def _files(trial: Path, only: list[str], skip: list[str]):
    if trial.is_file():
        yield trial, trial.name
        return
    for p in sorted(trial.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(trial).as_posix()
        if only and not _match(rel, only):
            continue
        if skip and _match(rel, skip):
            continue
        yield p, rel


def auth_hits(text: str, pattern: re.Pattern, path: str = "", limit: int = 5) -> list[Hit]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        m = pattern.search(line)
        if m:
            start = max(0, m.start() - 40)
            hits.append(Hit(path, i, mask(line[start : m.end() + 60])))
            if len(hits) >= limit:
                break
    return hits


def check_trial(
    trial: Path,
    require: Iterable[str] = (),
    only: Iterable[str] = (),
    skip: Iterable[str] = (),
    extra_patterns: Iterable[str] = (),
) -> TrialCheck:
    """Scan a trial (a directory or one file) for auth-error markers and missing required files.
    require: globs relative to the trial dir, each must match at least one file."""
    trial = Path(trial)
    pattern = compile_patterns(extra_patterns)
    out = TrialCheck(str(trial))
    if not trial.exists():
        out.missing.append("(trial path)")
        return out
    for p, rel in _files(trial, list(only), list(skip)):
        try:
            if p.stat().st_size > MAX_BYTES:
                continue
            text = p.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out.auth += auth_hits(text, pattern, rel)
    if trial.is_dir():
        for g in require:
            if not any(p.is_file() for p in trial.glob(g)):
                out.missing.append(g)
    return out


class AuthStreak:
    """Consecutive auth-error trials, optionally kept in a small JSON file so a shell loop that calls
    `bench-kit check-run` once per trial still sees the streak. A clean trial resets it."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.count = 0
        self.last: list[str] = []
        if self.path and self.path.is_file():
            try:
                saved = json.loads(self.path.read_text())
                self.count = int(saved.get("consecutive_auth_errors", 0))
                self.last = list(saved.get("trials", []))
            except (ValueError, TypeError, AttributeError):
                self.count, self.last = 0, []

    def update(self, check: TrialCheck) -> int:
        if check.auth:
            self.count += 1
            self.last = [*self.last, check.trial][-20:]
        else:
            self.count, self.last = 0, []
        return self.count

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"consecutive_auth_errors": self.count, "trials": self.last}, indent=2) + "\n")


def sample_error(check: TrialCheck, source: str = "infra") -> dict | None:
    """The `error` object for this trial's samples.jsonl row(s), or None when the trial is clean."""
    if not check.auth:
        return None
    return {"source": source, "type": "auth", "message": "auth error: " + check.auth[0].excerpt}
