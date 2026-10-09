"""bench-lint: rules BL001-BL010 from STANDARD.md section 9."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft7Validator

from . import render, stats
from .io import Run, discover, load_schema

RULES = {
    "BL001": ("error", "manifest or samples fail schema, duplicate run id, or id differs from directory"),
    "BL002": ("error", "required file or README section missing, or Status has no date"),
    "BL003": ("error", "generated files out of date (run bench-kit render)"),
    "BL004": ("error", "samples disagree with manifest (row count or recomputed numbers)"),
    "BL005": ("error", "SWITCH or WINNER verdict not backed by stats"),
    "BL006": ("error", "em or en dash"),
    "BL007": ("warning", "model id is an alias, or subject version missing without reason"),
    "BL008": ("warning", "revision.dirty is true without a correction note"),
    "BL009": ("warning", "task-set version missing from METHOD.md changelog"),
    "BL010": ("warning", "leftover TODO, or raw data not kept without location or reason"),
}
REQUIRED_FILES = ["README.md", "METHOD.md", "NEXT.md", "RESULTS.md", "runs/INDEX.md"]
README_SECTIONS = ["Status", "Latest result", "How it works", "Run it", "Layout"]
DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")  # en dash, em dash
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
IGNORE_MARK = re.compile(r"bench-lint:\s*ignore\s+(BL\d{3})\s+\S")
ALIAS = re.compile(r"(^|[-/:@])(latest|default|auto|current)($|[-/:@])", re.I)
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", "dist", "build"}
TOL = 0.0015


@dataclass
class Finding:
    rule: str
    path: str
    message: str
    line: int | None = None

    @property
    def level(self) -> str:
        return RULES[self.rule][0]

    def annotation(self) -> str:
        cmd = "error" if self.level == "error" else "warning"
        loc = f"file={self.path}" + (f",line={self.line}" if self.line else "")
        return f"::{cmd} {loc},title={self.rule}::{self.message}"

    def plain(self) -> str:
        loc = self.path + (f":{self.line}" if self.line else "")
        return f"{loc}: {self.rule} {self.level}: {self.message}"


class Linter:
    def __init__(self, root: Path):
        self.root = root
        self.findings: list[Finding] = []
        self.runs: list[Run] = discover(root)
        self.repo_ignores = self._load_repo_ignores()

    # ---------------------------------------------------------------- helpers
    def rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(p)

    def add(self, rule: str, path: Path | str, msg: str, line: int | None = None):
        self.findings.append(Finding(rule, path if isinstance(path, str) else self.rel(path), msg, line))

    def _load_repo_ignores(self) -> list[tuple[str, str]]:
        f = self.root / ".bench-lint-ignore"
        out = []
        if f.exists():
            for line in f.read_text().splitlines():
                parts = line.split(None, 2)
                if len(parts) == 3 and parts[0].startswith("BL") and not line.lstrip().startswith("#"):
                    out.append((parts[0], parts[1]))
        return out

    def md_files(self) -> list[Path]:
        try:
            out = subprocess.run(
                ["git", "ls-files", "-co", "--exclude-standard", "*.md"], cwd=self.root, capture_output=True, text=True, check=True
            ).stdout.split("\n")
            files = [self.root / x for x in out if x]
        except (OSError, subprocess.CalledProcessError):
            files = list(self.root.rglob("*.md"))
        keep = []
        for f in files:
            parts = Path(self.rel(f)).parts
            if parts[:1] == ("runs",) and parts[-1] == "INDEX.md":
                keep.append(f)
            elif not (set(parts[:-1]) & SKIP_DIRS) and f.exists():
                keep.append(f)
        return sorted(keep)

    def suppressed(self, f: Finding) -> bool:
        for rule, glob in self.repo_ignores:
            if rule == f.rule and fnmatch.fnmatch(f.path, glob):
                return True
        p = self.root / f.path
        if p.name == "manifest.json":
            run = next((r for r in self.runs if r.manifest_path.resolve() == p.resolve()), None)
            if run and any(x.get("rule") == f.rule for x in run.manifest.get("lint_ignore") or []):
                return True
        elif p.suffix == ".md" and p.exists():
            if f.rule in IGNORE_MARK.findall(p.read_text()):
                return True
        return False

    # ---------------------------------------------------------------- rules
    def bl001(self):
        mv = Draft7Validator(load_schema("bench-manifest-1.schema.json"))
        sv = Draft7Validator(load_schema("bench-sample-1.schema.json"))
        seen: dict[str, Path] = {}
        for r in self.runs:
            if r.manifest_error:
                self.add("BL001", r.manifest_path, r.manifest_error)
                continue
            for e in sorted(mv.iter_errors(r.manifest), key=lambda e: list(e.absolute_path)):
                where = "/".join(str(x) for x in e.absolute_path) or "(root)"
                self.add("BL001", r.manifest_path, f"{where}: {e.message}")
            rid = r.manifest.get("id")
            if rid and rid != r.dir.name:
                self.add("BL001", r.manifest_path, f"id {rid!r} differs from directory name {r.dir.name!r}")
            if rid in seen:
                self.add("BL001", r.manifest_path, f"duplicate run id {rid!r} (also {self.rel(seen[rid])})")
            seen[rid] = r.manifest_path
            for line, msg in r.samples_errors:
                self.add("BL001", r.samples_path, msg, line)
            bad = 0
            for i, row in enumerate(r.samples or [], 1):
                errs = list(sv.iter_errors(row))
                if errs:
                    bad += 1
                    if bad <= 20:
                        self.add("BL001", r.samples_path, f"row {i}: {errs[0].message}", i)
            if bad > 20:
                self.add("BL001", r.samples_path, f"{bad - 20} more invalid rows not shown")

    def bl002(self):
        for f in REQUIRED_FILES:
            if not (self.root / f).exists():
                self.add("BL002", f, "required file missing")
        readme = self.root / "README.md"
        if not readme.exists():
            return
        text = readme.read_text()
        heads = re.findall(r"^##\s+(.+?)\s*$", text, re.M)
        for s in README_SECTIONS:
            if s not in heads:
                self.add("BL002", readme, f"README section '## {s}' missing")
        m = re.search(r"^##\s+Status\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
        if m and not DATE.search(m.group(1)):
            self.add("BL002", readme, "Status section has no YYYY-MM-DD date")
        if render.START not in text or render.END not in text:
            self.add("BL002", readme, "latest-result markers missing (bench:latest:start / end)")

    def bl003(self):
        if any(r.manifest_error for r in self.runs):
            return
        for path, content in render.outputs(self.root).items():
            if not path.exists() or path.read_text() != content:
                self.add("BL003", path, "out of date; run bench-kit render")

    def bl004(self):
        for r in self.runs:
            m = r.manifest
            if r.manifest_error or not m:
                continue
            mcfg = m.get("metric_config") or {}
            needs = [k for k, v in mcfg.items() if stats.computable(v)]
            if r.samples is None:
                if needs:
                    self.add("BL004", r.manifest_path, f"metrics {needs} come from samples but samples.jsonl is missing")
                continue
            if r.samples_errors:
                continue
            names = {s.get("name") for s in m.get("subjects", [])}
            extra = sorted({x.get("subject") for x in r.samples} - names)
            if extra:
                self.add("BL004", r.samples_path, f"subjects not in manifest: {extra}")
            sd = m.get("source_data") or {}
            rp = m.get("repeats") or {}
            if isinstance(sd.get("n_completed"), int) and isinstance(rp.get("completed"), int):
                want = sd["n_completed"] * rp["completed"]
                for s in sorted(names):
                    got = sum(1 for x in r.samples if x.get("subject") == s)
                    if got != want:
                        self.add("BL004", r.samples_path, f"subject {s}: {got} rows, expected n_completed x repeats.completed = {want}")
            try:
                fresh = stats.compute(m, r.samples)
            except (KeyError, TypeError, ValueError) as e:
                self.add("BL004", r.samples_path, f"cannot compute stats: {e}")
                continue
            have = m.get("evaluation_results") or {}
            for s, metrics in fresh["evaluation_results"].items():
                for mid, want_sd in metrics.items():
                    got_sd = (have.get(s) or {}).get(mid)
                    if not got_sd:
                        self.add("BL004", r.manifest_path, f"evaluation_results.{s}.{mid} missing; run bench-kit stats")
                        continue
                    for label, a, b in _pairs(got_sd, want_sd):
                        if not _close(a, b):
                            self.add("BL004", r.manifest_path, f"evaluation_results.{s}.{mid}.{label} is {a}, samples give {b}; run bench-kit stats")
            for i, (got_c, want_c) in enumerate(zip(m.get("comparisons") or [], fresh["comparisons"], strict=False)):
                if not stats.computable(mcfg.get(got_c.get("metric_id"), {})):
                    continue
                for label, a, b in _pairs(got_c, want_c, diff=True):
                    if not _close(a, b):
                        self.add("BL004", r.manifest_path, f"comparisons[{i}].{label} is {a}, samples give {b}; run bench-kit stats")
            e = m.get("errors") or {}
            for k in ("n", "of"):
                if e.get(k) != fresh["errors"][k]:
                    self.add("BL004", r.manifest_path, f"errors.{k} is {e.get(k)}, samples give {fresh['errors'][k]}")

    def bl005(self):
        for r in self.runs:
            m = r.manifest
            v = str(m.get("verdict") or "")
            if not (v == "SWITCH" or v.startswith("WINNER:")):
                continue
            if (m.get("repeats") or {}).get("completed", 0) < 2:
                self.add("BL005", r.manifest_path, f"verdict {v} with fewer than 2 repeats; use INCONCLUSIVE, KEEP or SNAPSHOT")
            pm = m.get("primary_metric")
            comps = [c for c in m.get("comparisons") or [] if c.get("metric_id") == pm]
            if v.startswith("WINNER:"):
                w = v.split(":", 1)[1]
                comps = [c for c in comps if w in (c.get("baseline"), c.get("candidate"))]
            if not comps:
                self.add("BL005", r.manifest_path, f"verdict {v} without a comparison on the primary metric")
                continue
            lib = bool(((m.get("metric_config") or {}).get(pm) or {}).get("lower_is_better"))
            for c in comps:
                ci = (c.get("uncertainty") or {}).get("confidence_interval")
                if not ci:
                    self.add("BL005", r.manifest_path, f"verdict {v}: comparison {c.get('candidate')} vs {c.get('baseline')} has no CI")
                    continue
                if ci["lower"] <= 0 <= ci["upper"]:
                    self.add(
                        "BL005",
                        r.manifest_path,
                        f"verdict {v}: CI [{ci['lower']}, {ci['upper']}] of {c.get('candidate')} vs {c.get('baseline')} includes 0",
                    )
                    continue
                cand_better = (ci["lower"] > 0) != lib
                winner = c.get("candidate") if cand_better else c.get("baseline")
                if v.startswith("WINNER:") and winner != v.split(":", 1)[1]:
                    self.add("BL005", r.manifest_path, f"verdict {v} but the comparison favours {winner}")

    def bl006(self):
        files = self.md_files() + [r.manifest_path for r in self.runs]
        for f in files:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if DASHES.search(line):
                    self.add("BL006", f, "em or en dash; use a plain hyphen or reword", i)

    def bl007(self):
        for r in self.runs:
            for s in r.manifest.get("subjects", []) or []:
                mid = (s.get("model_info") or {}).get("id", "")
                if mid and ALIAS.search(mid):
                    self.add("BL007", r.manifest_path, f"subject {s.get('name')}: model id {mid!r} looks like an alias; use an exact id")
                h = s.get("harness")
                if h and not h.get("version") and not s.get("version_unknown_reason"):
                    self.add("BL007", r.manifest_path, f"subject {s.get('name')}: harness version missing and no version_unknown_reason")

    def bl008(self):
        for r in self.runs:
            if (r.manifest.get("revision") or {}).get("dirty") and not r.manifest.get("corrections"):
                self.add("BL008", r.manifest_path, "revision.dirty is true; add a correction note explaining it")

    def bl009(self):
        method = self.root / "METHOD.md"
        text = method.read_text() if method.exists() else ""
        known = set(re.findall(r"^###\s*\[([^\]]+)\]", text, re.M))
        for r in self.runs:
            v = (r.manifest.get("source_data") or {}).get("version")
            if v and v not in known:
                self.add("BL009", r.manifest_path, f"task-set version {v} has no '### [{v}]' entry in METHOD.md changelog")

    def bl010(self):
        for f in self.md_files():
            if self.rel(f) in ("RESULTS.md", "runs/INDEX.md"):
                continue
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if re.search(r"\bTODO\b", line) and "bench-lint:" not in line:
                    self.add("BL010", f, "leftover TODO", i)
        for r in self.runs:
            raw = r.manifest.get("raw") or {}
            if raw.get("kept") is False and not (raw.get("location") or raw.get("reason")):
                self.add("BL010", r.manifest_path, "raw.kept is false without location or reason")

    def run(self) -> list[Finding]:
        for rule in sorted(RULES):
            getattr(self, rule.lower())()
        return [f for f in self.findings if not self.suppressed(f)]


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOL


def _pairs(got: dict, want: dict, diff: bool = False):
    key = "diff" if diff else "score"
    yield key, got.get(key), want.get(key)
    gci = (got.get("uncertainty") or {}).get("confidence_interval") or {}
    wci = (want.get("uncertainty") or {}).get("confidence_interval") or {}
    if wci or gci:
        yield "ci.lower", gci.get("lower"), wci.get("lower")
        yield "ci.upper", gci.get("upper"), wci.get("upper")
    if diff:
        for k in ("wins", "ties", "losses"):
            yield k, got.get(k), want.get(k)


def lint(root: Path) -> list[Finding]:
    return Linter(root).run()
