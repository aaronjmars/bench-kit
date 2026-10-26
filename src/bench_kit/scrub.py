"""Scrub secret values out of run files (STANDARD.md section 11).

The caller says which values are secret: names of environment variables (globs allowed, e.g.
`*_API_KEY`) and/or files of `NAME=value` lines. bench-kit never reads a secret store itself.
Each value found in a text file is replaced with `<redacted:NAME>`, longest value first, so a
value that contains another (a URL holding a key) is masked whole. Values shorter than
MIN_LEN are skipped, since they are too likely to hit ordinary text.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

MIN_LEN = 8
NAME = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


@dataclass
class ScrubResult:
    files_changed: list[Path] = field(default_factory=list)
    hits: dict[Path, list[str]] = field(default_factory=dict)  # file -> labels of values found
    skipped_short: list[str] = field(default_factory=list)


def values_from_env(patterns: Iterable[str], environ: Mapping[str, str] | None = None) -> list[tuple[str, str]]:
    """(value, label) for every environment variable whose name matches one of the patterns."""
    env = os.environ if environ is None else environ
    out = []
    for name in sorted(env):
        if any(fnmatch.fnmatchcase(name, p) for p in patterns) and env[name]:
            out.append((env[name], name))
    return out


def values_from_file(path: Path) -> list[tuple[str, str]]:
    """(value, label) from a file: `NAME=value` lines (an `export ` prefix and matching quotes are
    dropped) or bare values (labeled `secret`). Blank lines and `#` comments are skipped."""
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = NAME.match(s)
        name, value = (m.group(1), m.group(2)) if m else ("secret", s)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            out.append((value, name))
    return out


def _prepare(values: Iterable[tuple[str, str]], result: ScrubResult) -> list[tuple[str, str]]:
    keep = {}
    for value, label in values:
        if len(value) < MIN_LEN:
            result.skipped_short.append(label)
            continue
        keep.setdefault(value, label)
        escaped = json.dumps(value)[1:-1]  # how the value appears inside a JSON string
        for variant in (escaped, escaped.replace("/", "\\/")):  # some encoders also escape "/"
            if variant != value:
                keep.setdefault(variant, label)
    return sorted(keep.items(), key=lambda vl: len(vl[0]), reverse=True)


def _files(paths: Iterable[Path]):
    for root in paths:
        root = Path(root)
        if root.is_file():
            yield root
        elif root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    yield p


def scrub(paths: Iterable[Path], values: Iterable[tuple[str, str]], check: bool = False) -> ScrubResult:
    """Replace every value in the text files under paths. check=True only reports (no writes):
    use it as the completeness check after a scrub, it must find nothing."""
    result = ScrubResult()
    vals = _prepare(values, result)
    if not vals:
        return result
    for p in _files(paths):
        try:
            data = p.read_bytes()
            text = data.decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable
        if "\0" in text:
            continue
        new, found = text, []
        for value, label in vals:
            if value in new:
                found.append(label)
                new = new.replace(value, f"<redacted:{label}>")
        if found:
            result.hits[p] = found
            if not check:
                p.write_bytes(new.encode("utf-8"))
                result.files_changed.append(p)
    return result
