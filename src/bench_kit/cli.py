"""bench-kit command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, export_eee, ledger, lint, render, runcheck, scrub, stats
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
        try:
            new = stats.apply(r.manifest, r.samples)
        except (KeyError, TypeError, ValueError) as e:
            print(f"{r.manifest_path}: cannot compute stats: {e}", file=sys.stderr)
            rc = 1
            continue
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


def cmd_ledger(a) -> int:
    records = []
    for f in a.records or []:
        records += ledger.load_records(f)
    try:
        path = ledger.write(a.root, a.sweep_id, runs=a.run or [], records=records, status=a.status, halted=a.halted, note=a.note)
    except (ValueError, OSError) as e:
        print(f"bench-kit ledger: {e}", file=sys.stderr)
        return 1
    rel = path.relative_to(a.root) if path.is_relative_to(a.root) else path
    print(f"wrote {rel}")
    hit = ledger.ignored(a.root, [str(rel)])
    if hit:
        print(f"warning: {rel} is gitignored, so this ledger would not be committed (BL012)", file=sys.stderr)
        return 2
    return 0


def cmd_check_run(a) -> int:
    streak = runcheck.AuthStreak(a.streak_file)
    checks, stop = [], False
    for t in a.trial:
        c = runcheck.check_trial(t, require=a.require or [], only=a.only or [], skip=a.skip or [], extra_patterns=a.pattern or [])
        checks.append(c)
        if streak.update(c) >= a.stop_after:
            stop = True
    streak.save()
    if a.json:
        out = {"trials": [c.as_dict() for c in checks], "consecutive_auth_errors": streak.count, "stop_sweep": stop}
        print(json.dumps(out, indent=2))
    else:
        for c in checks:
            print(f"{c.trial}: {c.status}")
            for h in c.auth:
                print(f"  auth error: {h.text()}")
            for m in c.missing:
                print(f"  missing: {m}")
        if stop:
            print(f"STOP SWEEP: {streak.count} consecutive auth errors (limit {a.stop_after}); fix the login or key, then resume")
    if stop:
        return 3
    if any(c.auth for c in checks):
        return 2
    return 1 if any(c.missing for c in checks) else 0


def cmd_scrub(a) -> int:
    values = scrub.values_from_env(a.env or [])
    for f in a.values_file or []:
        values += scrub.values_from_file(f)
    if not values:
        print("bench-kit scrub: no secret values (give --env NAME or --values-file FILE; empty variables are skipped)", file=sys.stderr)
        return 1
    res = scrub.scrub(a.path, values, check=a.check)
    for label in sorted(set(res.skipped_short)):
        print(f"warning: value of {label} is shorter than {scrub.MIN_LEN} characters, not scrubbed", file=sys.stderr)
    for p, labels in res.hits.items():
        verb = "still holds" if a.check else "scrubbed"
        print(f"{p}: {verb} {', '.join(sorted(set(labels)))}")
    if a.check:
        if res.hits:
            print(f"bench-kit scrub --check: secrets left in {len(res.hits)} file(s)")
            return 1
        print("bench-kit scrub --check: clean")
        return 0
    print(f"scrubbed {len(res.files_changed)} file(s) with {len({v for v, _ in values if len(v) >= scrub.MIN_LEN})} secret value(s)")
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

    p = sub.add_parser("lint", help="check the repo against the standard (BL001-BL013)")
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

    p = sub.add_parser("ledger", help="write or update the tracked sweep ledger data/sweeps/<sweep-id>.json")
    p.add_argument("sweep_id")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--run", action="append", help="add every sample row of this run id as a record (repeatable)")
    p.add_argument("--records", type=Path, action="append", help="add records from a JSONL file the bench's runner wrote (repeatable)")
    p.add_argument("--status", choices=ledger.STATUSES, default="finished")
    p.add_argument("--halted", help="why the sweep stopped early")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_ledger)

    p = sub.add_parser("check-run", help="flag auth errors (401, revoked or invalid key) and missing files in finished trials")
    p.add_argument("trial", nargs="+", type=Path, help="trial directory or file, in run order")
    p.add_argument("--require", action="append", help="glob, relative to the trial dir, that must match a file (repeatable)")
    p.add_argument("--only", action="append", help="scan only files matching this glob (repeatable)")
    p.add_argument("--skip", action="append", help="do not scan files matching this glob, e.g. an agent transcript (repeatable)")
    p.add_argument("--pattern", action="append", help="extra auth-error regex, case-insensitive (repeatable)")
    p.add_argument("--stop-after", type=int, default=3, help="consecutive auth-error trials that stop the sweep (default 3)")
    p.add_argument("--streak-file", type=Path, help="JSON file that keeps the streak across calls (one call per trial)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check_run)

    p = sub.add_parser("scrub", help="replace secret values in run files with <redacted:NAME>")
    p.add_argument("path", nargs="+", type=Path, help="files or directories to scrub")
    p.add_argument("--env", action="append", help="environment variable name or glob whose value is secret, e.g. '*_API_KEY' (repeatable)")
    p.add_argument("--values-file", type=Path, action="append", help="file of NAME=value lines or bare values (repeatable)")
    p.add_argument("--check", action="store_true", help="only report files that still hold a secret; exit 1 if any")
    p.set_defaults(fn=cmd_scrub)

    p = sub.add_parser("rules", help="list lint rules")
    p.set_defaults(fn=cmd_rules)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
