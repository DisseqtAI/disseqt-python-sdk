"""``disseqt scan`` — walk a source tree and hunt AI-security issues.

Dispatches char-bounded chunks to disseqt-go validators (defaults to the
six app-sec judges: BFLA, BOLA, RBAC, ShellInjection, DebugAccess,
IntellectualProperty; falls back to sql-injection / prompt-injection /
data-leakage / insecure-output until those judges are deployed).

Auth reads from the same env vars as every other CLI verb — see
``disseqt validate --help`` for the required set.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from ..scan import (
    APPSEC_VALIDATORS,
    DEFAULT_BATCH_CHARS,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_FILE_BYTES,
    DispatchStats,
    GitDiffError,
    ScanConfig,
    changed_files,
    collect_chunks,
    dispatch,
    load_config,
    meets_min_severity,
    resolve_batch_chars,
    to_json,
    to_markdown,
    to_sarif,
)
from ._common import _fail

SEVERITY_CHOICES = ("low", "medium", "high", "critical")
FORMAT_CHOICES = ("markdown", "sarif", "json")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _resolve_validators(cli_flag: tuple[str, ...], cfg: ScanConfig) -> tuple[str, ...]:
    if cli_flag:
        return cli_flag
    if cfg.validators:
        return tuple(cfg.validators)
    return APPSEC_VALIDATORS


def _render(findings: list, fmt: str) -> str:
    if fmt == "sarif":
        return to_sarif(findings)
    if fmt == "json":
        return to_json(findings)
    return to_markdown(findings)


@click.command("scan")
@click.argument(
    "scan_path", type=click.Path(exists=True, file_okay=True, dir_okay=True), default="."
)
@click.option(
    "--diff",
    "diff_spec",
    help="Scan only files changed between two git refs (e.g. main..HEAD).",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(FORMAT_CHOICES),
    default="markdown",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--min-severity",
    type=click.Choice(SEVERITY_CHOICES),
    default="low",
    show_default=True,
    help="Drop findings below this severity.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    help="Write report to a file instead of stdout.",
)
@click.option(
    "--fail-on-findings/--no-fail-on-findings",
    default=True,
    show_default=True,
    help="Exit non-zero when findings remain after --min-severity filtering.",
)
@click.option(
    "--max-file-bytes",
    type=int,
    default=DEFAULT_MAX_FILE_BYTES,
    show_default=True,
    help="Skip files larger than this many bytes.",
)
@click.option(
    "--max-chunk-chars",
    type=int,
    default=DEFAULT_MAX_CHUNK_CHARS,
    show_default=True,
    help="Split files into chunks no larger than this many characters.",
)
@click.option(
    "--batch-chars",
    type=int,
    default=None,
    help=f"Bundle chunks up to this many chars per request "
    f"(default {DEFAULT_BATCH_CHARS}; env DISSEQT_SCAN_CONTEXT_LIMIT).",
)
@click.option(
    "--validator",
    "validator_overrides",
    multiple=True,
    help="Override the default validator set. Repeat for multiple.",
)
def scan(
    scan_path: str,
    diff_spec: str | None,
    output_format: str,
    min_severity: str,
    output_path: str | None,
    fail_on_findings: bool,
    max_file_bytes: int,
    max_chunk_chars: int,
    batch_chars: int | None,
    validator_overrides: tuple[str, ...],
) -> None:
    """Scan SCAN_PATH for AI-security issues and print a report.

    Exit code: 0 = no findings, 1 = findings remain after filtering (only
    when --fail-on-findings is on), 2 = usage/config error.
    """
    _configure_logging()
    root = Path(scan_path).resolve()
    scan_root = root if root.is_dir() else root.parent

    cfg = load_config(scan_root)
    effective_min = min_severity if min_severity != "low" else (cfg.min_severity or min_severity)
    effective_max_file = cfg.max_file_bytes or max_file_bytes
    effective_max_chunk = cfg.max_chunk_chars or max_chunk_chars
    effective_batch_chars = resolve_batch_chars(batch_chars or cfg.batch_chars)
    validators = _resolve_validators(validator_overrides, cfg)

    only: list[str] | None = None
    if diff_spec:
        try:
            only = [str(p) for p in changed_files(diff_spec, cwd=scan_root)]
        except GitDiffError as exc:
            _fail(str(exc))
        if not only:
            click.echo("# disseqt scan\n\nNo changed files in diff range.", err=False)
            sys.exit(0)

    chunks = list(
        collect_chunks(
            root,
            only=only,
            max_file_bytes=effective_max_file,
            max_chunk_chars=effective_max_chunk,
        )
    )
    if not chunks:
        click.echo("# disseqt scan\n\nNo source files matched (nothing to do).", err=False)
        sys.exit(0)

    stats = DispatchStats()
    findings = list(
        dispatch(
            chunks,
            validators,
            batch_chars=effective_batch_chars,
            stats=stats,
        )
    )
    findings = [f for f in findings if meets_min_severity(f.severity, effective_min)]

    report = _render(findings, output_format)
    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
    else:
        click.echo(report)

    # Summary always to stderr so it doesn't corrupt piped JSON/SARIF output.
    click.echo(
        f"scanned {len(chunks)} chunk(s) across {len({c.file_path for c in chunks})} file(s); "
        f"{stats.batches_ok} batch(es) ok, {stats.batches_failed} failed; "
        f"{len(findings)} finding(s) after min-severity={effective_min}",
        err=True,
    )

    if fail_on_findings and findings:
        sys.exit(1)
    sys.exit(0)
