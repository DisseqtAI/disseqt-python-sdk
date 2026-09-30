"""Resolve the ``--diff BASE..HEAD`` flag to a list of changed files."""

from __future__ import annotations

import subprocess  # noqa: S404 — required to shell out to git
from pathlib import Path


class GitDiffError(RuntimeError):
    """Raised when we can't produce a valid diff (bad refs, no git, etc.)."""


def parse_diff_range(spec: str) -> tuple[str, str]:
    """Split ``base..head`` into ``(base, head)``. Also accepts ``base...head``."""
    for sep in ("...", ".."):
        if sep in spec:
            base, _, head = spec.partition(sep)
            base, head = base.strip(), head.strip()
            if base and head:
                return base, head
    raise GitDiffError(f"invalid diff spec {spec!r}: expected BASE..HEAD (e.g. main..HEAD)")


def changed_files(spec: str, *, cwd: Path) -> list[Path]:
    """Return files changed between two git refs, as paths relative to ``cwd``."""
    base, head = parse_diff_range(spec)
    try:
        proc = subprocess.run(  # noqa: S603 — inputs are validated git refs
            ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base}..{head}"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GitDiffError("git is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise GitDiffError(f"git diff {base}..{head} failed: {stderr}") from exc
    return [cwd / line for line in proc.stdout.splitlines() if line.strip()]
