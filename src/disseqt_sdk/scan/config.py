"""Auto-discovery for ``.disseqt-code-scan.yaml`` in the scan root.

YAML is optional — if pyyaml isn't installed we silently skip the file
rather than requiring an extra dep for the whole SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_FILENAME = ".disseqt-code-scan.yaml"


@dataclass(frozen=True, slots=True)
class ScanConfig:
    """User-supplied overrides that stack on top of CLI flags."""

    validators: list[str] = field(default_factory=list)
    min_severity: str | None = None
    max_file_bytes: int | None = None
    max_chunk_chars: int | None = None
    batch_chars: int | None = None
    extra_skip_globs: list[str] = field(default_factory=list)


def discover(root: Path) -> Path | None:
    """Return the path to ``.disseqt-code-scan.yaml`` in ``root`` if present."""
    candidate = root / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def load(root: Path) -> ScanConfig:
    """Load config from the scan root, returning defaults if missing."""
    path = discover(root)
    if path is None:
        return ScanConfig()
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        # No pyyaml → treat as no-config rather than crashing the scan.
        return ScanConfig()
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return ScanConfig()
    if not isinstance(raw, dict):
        return ScanConfig()
    return ScanConfig(
        validators=list(raw.get("validators") or []),
        min_severity=raw.get("min_severity"),
        max_file_bytes=raw.get("max_file_bytes"),
        max_chunk_chars=raw.get("max_chunk_chars"),
        batch_chars=raw.get("batch_chars"),
        extra_skip_globs=list(raw.get("extra_skip_globs") or []),
    )
