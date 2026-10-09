"""bench-kit command line."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, export_eee, lint, render, stats
from .io import discover, dump_json


def cmd_render(a) -> int:
    changed = render.render(a.root)
    for p in changed:
        print(f"wrote {p}")
    if not changed:
        print("up to date")
    return 0


def cmd_stats(a) -> int:
    runs = discover(a.root)
    if a.run:
        runs = [r for r in runs if r.dir.name in a.run]
    rc = 0
    for r in runs:
        if r.manifest_error:
            print(f"{r.manifest_path}: {r.manifest_error}", file=sys.stderr)
            rc = 1
            continue
        if r.samples is None:
            continue
        new = stats.apply(r.manifest, r.samples)
        text = dump_json(new)
        if r.manifest_path.read_text() != text:
            r.manifest_path.write_text(text)
            print(f"updated {r.manifest_path}")
    return rc


def cmd_lint(a) -> int:
    findings = lint.lint(a.root)
    gha = a.format == "github" or (a.format == "auto" and os.environ.get("GITHUB_ACTIONS") == "true")
    for f in findings:
        print(f.annotation() if gha else f.plain())
    errors = sum(1 for f in findings if f.level == "error")
    warns = len(findings) - errors
    print(f"bench-lint: {errors} error(s), {warns} warning(s)" + (" (warn-only)" if a.warn_only and errors else ""))
    return 1 if errors and not a.warn_only else 0


def cmd_export(a) -> int:
    runs = [r for r in discover(a.root) if r.dir.name == a.run]
    if not runs:
        print(f"run {a.run} not found under {a.root}/runs", file=sys.stderr)
        return 1
    for p in export_eee.export(runs[0], a.out, validate=not a.no_validate, org=a.org):
        print(f"wrote {p}")
    return 0


def cmd_rules(_a) -> int:
    for code, (level, desc) in sorted(lint.RULES.items()):
        print(f"{code} {level}: {desc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench-kit", description="Shared standard tooling for bench repos.")
    ap.add_argument("--version", action="version", version=f"bench-kit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("render", help="rebuild README latest block, RESULTS.md and runs/INDEX.md")
    p.add_argument("root", nargs="?", type=Path, default=Path("."))
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("stats", help="fill metrics, comparisons, errors and token usage from samples.jsonl")
    p.add_argument("root", nargs="?", type=Path, default=Path("."))
    p.add_argument("--run", action="append", help="only this run id (repeatable)")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("lint", help="check the repo against the standard (BL001-BL010)")
    p.add_argument("root", nargs="?", type=Path, default=Path("."))
    p.add_argument("--warn-only", action="store_true", help="report errors but exit 0")
    p.add_argument("--format", choices=["auto", "github", "plain"], default="auto")
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("export-eee", help="export one run to Every Eval Ever 0.3.0 files")
    p.add_argument("run", help="run id (directory under runs/)")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("eee-export"))
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--org", help="organization that ran the eval (EEE source_organization_name); overrides the manifest")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("rules", help="list lint rules")
    p.set_defaults(fn=cmd_rules)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
