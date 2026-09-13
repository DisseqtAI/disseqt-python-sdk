"""``disseqt scan`` code scanner — walk source, dispatch to validators, render."""

from __future__ import annotations

from .collector import (
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_FILE_BYTES,
    CodeChunk,
    chunk_file,
    collect_chunks,
    iter_source_files,
)
from .config import ScanConfig
from .config import load as load_config
from .dispatcher import (
    APPSEC_VALIDATORS,
    DEFAULT_BATCH_CHARS,
    FALLBACK_VALIDATORS,
    ChunkBatch,
    DispatchStats,
    batch_chunks,
    dispatch,
    resolve_batch_chars,
)
from .formatters import to_json, to_markdown, to_sarif
from .git_diff import GitDiffError, changed_files
from .schema import SEVERITY_ORDER, CodeFinding, Severity, meets_min_severity

__all__ = [
    "APPSEC_VALIDATORS",
    "ChunkBatch",
    "CodeChunk",
    "CodeFinding",
    "DEFAULT_BATCH_CHARS",
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MAX_FILE_BYTES",
    "DispatchStats",
    "FALLBACK_VALIDATORS",
    "GitDiffError",
    "SEVERITY_ORDER",
    "ScanConfig",
    "Severity",
    "batch_chunks",
    "changed_files",
    "chunk_file",
    "collect_chunks",
    "dispatch",
    "iter_source_files",
    "load_config",
    "meets_min_severity",
    "resolve_batch_chars",
    "to_json",
    "to_markdown",
    "to_sarif",
]
