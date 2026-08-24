"""Prepare a release: bump pyproject, stamp the CHANGELOG, extract notes.

Used by .github/workflows/release.yml. Run from the repo root:

    python3 .github/scripts/prepare_release.py <patch|minor|major>

- bumps ``[project] version`` in pyproject.toml (the single source of truth
  — both packages resolve ``__version__`` from the installed metadata)
- retitles ``## [Unreleased]`` in CHANGELOG.md to ``## [X.Y.Z] - <today>``,
  keeping a fresh empty Unreleased section above it
- writes the new section's body to release_notes.md (not committed) for
  the GitHub release
- prints the new version to stdout (and nothing else)
"""

from __future__ import annotations

import datetime
import pathlib
import re
import subprocess
import sys

BUMPS = ("patch", "minor", "major")


def current_version_is_tagged(current: str) -> bool:
    """Refuse to bump past a version that was never released.

    If pyproject's version has no matching ``v<version>`` git tag, someone
    hand-bumped it in a feature branch (the pre-automation habit). Bumping
    again would skip that number entirely — this is exactly how 0.9.0
    jumped straight to 0.11.0 on 2026-08-24: a feature branch pre-bumped
    to 0.10.0, which was never tagged or published, and the next button
    press bumped past it.

    Fail-open when git itself is unavailable or errors for environmental
    reasons: the guard must never be the thing that breaks a release. Only
    a definitive "tag does not exist" (git exit code 1) blocks.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "-q", "--verify", f"refs/tags/v{current}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"warning: tag check skipped (git unavailable: {exc})", file=sys.stderr)
        return True
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        print(
            f"pyproject.toml is at {current}, but tag v{current} does not exist — "
            "the version was hand-bumped without being released. Bumping again "
            f"would silently skip {current}. Either release {current} first "
            f"(tag v{current} + GitHub release), or revert the hand-bump so the "
            "Release workflow owns the numbering, then re-run. Reminder: feature "
            "branches should only add [Unreleased] CHANGELOG entries — never "
            "touch the version.",
            file=sys.stderr,
        )
        return False
    print(
        f"warning: tag check inconclusive (git exited {result.returncode}); continuing",
        file=sys.stderr,
    )
    return True


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in BUMPS:
        print(f"usage: prepare_release.py <{'|'.join(BUMPS)}>", file=sys.stderr)
        return 2
    bump = sys.argv[1]

    pyproject = pathlib.Path("pyproject.toml")
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version = "(\d+)\.(\d+)\.(\d+)"$', text, flags=re.M)
    if match is None:
        print("pyproject.toml has no plain X.Y.Z version line", file=sys.stderr)
        return 1
    major, minor, patch = (int(g) for g in match.groups())
    if not current_version_is_tagged(f"{major}.{minor}.{patch}"):
        return 1
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    version = f"{major}.{minor}.{patch}"
    pyproject.write_text(
        text[: match.start()] + f'version = "{version}"' + text[match.end() :],
        encoding="utf-8",
    )

    changelog = pathlib.Path("CHANGELOG.md")
    log = changelog.read_text(encoding="utf-8")
    marker = "## [Unreleased]"
    if marker not in log:
        print("CHANGELOG.md has no '## [Unreleased]' section", file=sys.stderr)
        return 1
    today = datetime.date.today().isoformat()
    heading = f"## [{version}] - {today}"
    log = log.replace(marker, f"{marker}\n\n{heading}", 1)
    changelog.write_text(log, encoding="utf-8")

    # Everything between the new heading and the next "## [" is the notes.
    tail = log.split(heading, 1)[1]
    next_section = re.search(r"^## \[", tail, flags=re.M)
    notes = (tail[: next_section.start()] if next_section else tail).strip()
    if not notes:
        notes = "Maintenance release."
    pathlib.Path("release_notes.md").write_text(notes + "\n", encoding="utf-8")

    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
