"""``disseqt judges <command>``: calibration, golden sets, templates and the judge cache."""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .cache import JsonDirStore
from .calibration.golden import (
    DEFAULT_DIR,
    GoldenError,
    export_for_labeling,
    import_labels,
    load_golden,
    save_golden,
    validate_items,
)
from .calibration.report import load_reports, write_report
from .calibration.split import split_of
from .config import KindConfig
from .templates import verify_lock

DEFAULT_CALIBRATION = ".disseqt/calibration"
DEFAULT_CACHE = ".disseqt/judge_cache"


def _out(text: str = "") -> None:
    print(text)


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _golden_path(arg: str, kind: str | None) -> Path:
    p = Path(arg)
    return p / f"{kind}.jsonl" if p.is_dir() and kind else p


def cmd_calibrate(ns: argparse.Namespace) -> int:
    from .api import make_judge
    from .calibration.run import calibrate

    try:
        items = load_golden(_golden_path(ns.golden, ns.kind))
        make = lambda n: make_judge(  # noqa: E731
            ns.model,
            require_calibration="off",
            cache_mode=ns.cache_mode,
            cache_dir=ns.cache_dir,
            kinds={ns.kind: KindConfig(samples=n)},
        )
        reports = calibrate(ns.kind, items, make, samples=tuple(ns.samples), positive=ns.positive,
                            report_root=ns.out)  # fmt: skip
    except (GoldenError, OSError, ValueError) as exc:
        _err(f"disseqt judges calibrate: {exc}")
        return 3
    code = 0
    for r in reports:
        path = write_report(r, ns.out)
        _out(f"{r.kind} n={r.samples} identity={r.identity_hash} "
             f"gating_allowed={r.gating_allowed} -> {path}")  # fmt: skip
        for f in r.failures:
            _out(f"  - {f}")
        for w in r.warnings:
            _out(f"  ! {w}")
        code = code or (0 if r.gating_allowed else 1)
    return code


def cmd_report(ns: argparse.Namespace) -> int:
    reports = load_reports(ns.dir)
    if ns.json:
        _out(json.dumps([r.to_dict() for r in reports], indent=2, sort_keys=True))
        return 0
    if not reports:
        _out(f"no calibration reports under {ns.dir}")
        return 0
    for r in reports:
        _out(f"{r.kind:26} n={r.samples} {r.identity_hash} gate={'yes' if r.gating_allowed else 'no '}"
             f" abstain={r.abstain_rate} items={r.counts.get('human_test')}")  # fmt: skip
        for f in r.failures:
            _out(f"    - {f}")
    return 0


def cmd_golden_validate(ns: argparse.Namespace) -> int:
    bad = 0
    for path in ns.paths:
        try:
            items = load_golden(path)
        except (GoldenError, OSError) as exc:
            _err(str(exc))
            bad += 1
            continue
        problems = validate_items(items, ns.kind)
        for p in problems:
            _err(f"{path}: {p}")
        bad += bool(problems)
        counts = {
            s: sum(1 for i in items if split_of(i.kind, i.id) == s)
            for s in ("dev", "validation", "test")
        }
        sources = {
            s: sum(1 for i in items if i.source == s) for s in ("human", "silver_code", "draft")
        }
        _out(f"{path}: {len(items)} items, splits {counts}, sources {sources}")
    return 1 if bad else 0


def cmd_golden_export(ns: argparse.Namespace) -> int:
    items = load_golden(ns.golden)
    if ns.split:
        items = [i for i in items if split_of(i.kind, i.id) == ns.split]
    n = export_for_labeling(items, ns.out, blind=not ns.show_labels)
    _out(f"wrote {n} rows to {ns.out}; each rater fills 'rater' and 'label' (disagreements need "
         "'adjudicated_label')")  # fmt: skip
    return 0


def cmd_golden_import(ns: argparse.Namespace) -> int:
    result = import_labels(ns.csv, min_raters=ns.min_raters)
    target = Path(ns.out)
    existing = {(i.kind, i.id): i for i in load_golden(target)} if target.exists() else {}
    for item in result.items:
        existing[(item.kind, item.id)] = item  # human labels replace drafts of the same id
    save_golden(target, existing.values())
    _out(f"imported {len(result.items)} human items into {target}")
    for line in (*result.needs_adjudication, *result.skipped):
        _out(f"  ! {line}")
    return 1 if result.needs_adjudication else 0


def cmd_templates_check(ns: argparse.Namespace) -> int:
    problems = verify_lock()
    for p in problems:
        _err(p)
    _out("templates match LOCK.json" if not problems else f"{len(problems)} template problem(s)")
    return 1 if problems else 0


def cmd_cache(ns: argparse.Namespace) -> int:
    store = JsonDirStore(ns.dir)
    files = store.list()
    if ns.action == "stats":
        size = sum(p.stat().st_size for p in files)
        _out(f"{len(files)} entries, {size} bytes in {ns.dir}")
        return 0
    if ns.action == "verify":
        corrupt = []
        for p in files:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or "response" not in data:
                    corrupt.append(str(p))
            except (OSError, ValueError):
                corrupt.append(str(p))
        for c in corrupt:
            _err(f"corrupt cache entry: {c}")
        _out(f"{len(files) - len(corrupt)} ok, {len(corrupt)} corrupt")
        return 1 if corrupt else 0
    cutoff = time.time() - ns.older_than_days * 86400
    old = [p for p in files if p.stat().st_mtime < cutoff]
    for p in old:
        _out(f"{'would remove' if ns.dry_run else 'removing'} {p}")
        if not ns.dry_run:
            p.unlink(missing_ok=True)
    _out(f"{len(old)} entries {'would be ' if ns.dry_run else ''}pruned")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="disseqt judges")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="measure a judge against a golden set")
    c.add_argument("--kind", required=True)
    c.add_argument(
        "--golden", default=str(DEFAULT_DIR), help="a .jsonl file or the golden directory"
    )
    c.add_argument("--model", required=True, help="provider:model, for example service:judge-1")
    c.add_argument("--samples", type=int, nargs="+", default=[3, 1])
    c.add_argument("--positive")
    c.add_argument("--out", default=DEFAULT_CALIBRATION)
    c.add_argument("--cache-mode", default="auto")
    c.add_argument("--cache-dir", default=DEFAULT_CACHE)
    c.set_defaults(fn=cmd_calibrate)

    r = sub.add_parser("report", help="show calibration reports")
    r.add_argument("--dir", default=DEFAULT_CALIBRATION)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_report)

    g = sub.add_parser("golden", help="golden set tools")
    gs = g.add_subparsers(dest="golden_command", required=True)
    v = gs.add_parser("validate")
    v.add_argument("paths", nargs="+")
    v.add_argument("--kind")
    v.set_defaults(fn=cmd_golden_validate)
    e = gs.add_parser("export", help="write a blind CSV for human labelling")
    e.add_argument("--golden", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--split", choices=("dev", "validation", "test"))
    e.add_argument("--show-labels", action="store_true")
    e.set_defaults(fn=cmd_golden_export)
    i = gs.add_parser("import", help="merge rater CSVs into human golden items")
    i.add_argument("csv", nargs="+")
    i.add_argument("--out", required=True)
    i.add_argument("--min-raters", type=int, default=2)
    i.set_defaults(fn=cmd_golden_import)

    t = sub.add_parser("templates", help="template tools")
    ts = t.add_subparsers(dest="templates_command", required=True)
    ts.add_parser("check").set_defaults(fn=cmd_templates_check)

    k = sub.add_parser("cache", help="judge response cache")
    k.add_argument("action", choices=("stats", "verify", "prune"))
    k.add_argument("--dir", default=DEFAULT_CACHE)
    k.add_argument("--older-than-days", type=float, default=90)
    k.add_argument("--dry-run", action="store_true")
    k.set_defaults(fn=cmd_cache)
    return ap


def main(argv: Sequence[str]) -> int:
    try:
        ns = build_parser().parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    return int(ns.fn(ns))


def register() -> None:
    from ..cli.main import register_subcommand

    register_subcommand("judges", lambda argv: main(argv), override=True)
