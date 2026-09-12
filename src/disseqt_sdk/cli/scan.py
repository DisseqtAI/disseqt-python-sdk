"""``disseqt scan`` — code/trace scanner CLI skeleton (Phase 4e).

The heavy lifting lives in the Go binary shipped by Phase 4d
(``disseqt tracescan`` + ``disseqt scan`` Go implementations). This
subcommand is intentionally a shell-out placeholder — it looks up the
binary on PATH (env var ``DISSEQT_SCAN_BIN`` overrides) and forwards
arguments. When the binary isn't installed, we emit a friendly message
so the SDK-only user isn't blocked from other subcommands.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # noqa: S404 — required for shell-out
import sys

import click

BIN_ENV = "DISSEQT_SCAN_BIN"


def _resolve_bin() -> str | None:
    override = os.environ.get(BIN_ENV)
    if override:
        return override if os.path.isfile(override) else None
    return shutil.which("disseqt-scan")


@click.command("scan", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("scan_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def scan(ctx: click.Context, scan_args: tuple[str, ...]) -> None:
    """Run the disseqt code/trace scanner.

    Forwards all arguments to the ``disseqt-scan`` binary (Phase 4d).
    Set ``DISSEQT_SCAN_BIN`` to point at a specific build.
    """
    binary = _resolve_bin()
    if binary is None:
        click.secho(
            "disseqt-scan binary not found on PATH — install the Phase 4d Go "
            f"binary or set ${BIN_ENV} to a build. See "
            "https://github.com/DisseqtAI/disseqt-go/tree/main/cmd/tracescan",
            fg="yellow",
            err=True,
        )
        raise SystemExit(127)
    proc = subprocess.run([binary, *scan_args], check=False)  # noqa: S603
    sys.exit(proc.returncode)
