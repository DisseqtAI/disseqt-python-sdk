"""Batch code chunks and POST them to disseqt-go validators.

Each batch is sent to *every* configured validator. A batch is a bundle of
chunks whose combined text fits under ``batch_chars`` — grouping cuts the
per-request overhead vs one-request-per-chunk.

The validator returns a JSON envelope; we heuristically pull findings out
of it. The disseqt-go response shapes we accept today:

- ``{"data": {"findings": [...]}}``
- ``{"data": {"issues":   [...]}}``
- ``{"findings": [...]}``

Any other shape → we log a warning and skip. The individual finding dict
is fed to :func:`_finding_from_dict`, which maps common field aliases.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .collector import CodeChunk
from .schema import CodeFinding, Severity

log = logging.getLogger(__name__)

DEFAULT_BATCH_CHARS = 40_000
BATCH_CHARS_ENV = "DISSEQT_SCAN_CONTEXT_LIMIT"

# App-security judges wired on the disseqt-go feat branch. Once merged, the
# CLI defaults to these six. Until then, callers can override with --validator
# or via .disseqt-code-scan.yaml to hit whatever's currently registered.
APPSEC_VALIDATORS: tuple[str, ...] = (
    "bfla",
    "bola",
    "rbac",
    "shell-injection",
    "debug-access",
    "intellectual-property",
)

# Validators that exist on disseqt-go today (fallback when app-sec judges
# aren't deployed yet). Verified against
# disseqt-go/internal/validatorregistry/registry.go and cli/enums.py.
FALLBACK_VALIDATORS: tuple[str, ...] = (
    "sql-injection",
    "prompt-injection",
    "data-leakage",
    "insecure-output",
)

VALIDATOR_DOMAIN = "input-validation"
VALIDATOR_PATH = "/api/v1/sdk/validators/{domain}/{validator}"
BASE_ENV = "DISSEQT_BASE_URL"
DEFAULT_BASE = "https://api.disseqt.ai/realtime-validations"


def resolve_batch_chars(cli_value: int | None) -> int:
    """CLI flag beats env var beats default."""
    if cli_value:
        return cli_value
    env = os.environ.get(BATCH_CHARS_ENV)
    if env:
        try:
            return int(env)
        except ValueError:
            log.warning("ignoring non-integer %s=%r", BATCH_CHARS_ENV, env)
    return DEFAULT_BATCH_CHARS


@dataclass(frozen=True, slots=True)
class ChunkBatch:
    """A group of chunks that will be sent in one HTTP round-trip."""

    chunks: tuple[CodeChunk, ...]

    @property
    def total_chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)

    def as_prompt(self) -> str:
        """Concatenate chunks into a single prompt with file/line markers."""
        parts: list[str] = []
        for chunk in self.chunks:
            parts.append(
                f"--- FILE: {chunk.file_path} "
                f"(lines {chunk.start_line}-{chunk.end_line}, {chunk.language}) ---\n"
                f"{chunk.text}"
            )
        return "\n".join(parts)


def batch_chunks(
    chunks: Iterable[CodeChunk],
    *,
    batch_chars: int = DEFAULT_BATCH_CHARS,
) -> Iterator[ChunkBatch]:
    """Yield :class:`ChunkBatch` groups, each under ``batch_chars`` of text."""
    buf: list[CodeChunk] = []
    buf_size = 0
    for chunk in chunks:
        size = len(chunk.text)
        if buf and buf_size + size > batch_chars:
            yield ChunkBatch(tuple(buf))
            buf = []
            buf_size = 0
        buf.append(chunk)
        buf_size += size
    if buf:
        yield ChunkBatch(tuple(buf))


@dataclass(slots=True)
class DispatchStats:
    """Bookkeeping surfaced to the CLI for the final summary line."""

    total_batches: int = 0
    batches_ok: int = 0
    batches_failed: int = 0
    findings_by_validator: dict[str, int] = field(default_factory=dict)


# Type of the transport hook — split out so tests can inject a fake.
Transport = Callable[[str, str, dict[str, Any]], Any]


def _default_transport(method: str, path: str, body: dict[str, Any]) -> Any:
    # Lazy import — dispatcher is imported from cli/scan.py, and cli/_http
    # in turn touches cli/__init__.py which registers scan. Break the cycle
    # by resolving the HTTP layer at call time.
    from ..cli import _http  # noqa: PLC0415 — deliberate lazy import

    return _http.request(method, BASE_ENV, DEFAULT_BASE, path, json_body=body)


def _validator_path(validator: str, domain: str = VALIDATOR_DOMAIN) -> str:
    return VALIDATOR_PATH.format(domain=domain, validator=validator)


def _build_payload(batch: ChunkBatch, validator: str) -> dict[str, Any]:
    """Payload shape mirrors ``InputValidationRequest.to_input_data()``."""
    return {
        "input_data": {
            "llm_input_query": batch.as_prompt(),
        },
        "config_input": {
            "scan_mode": "code",
            "validator": validator,
            "chunk_count": len(batch.chunks),
        },
    }


def _coerce_severity(value: Any) -> Severity:
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"critical", "high", "medium", "low"}:
            return low  # type: ignore[return-value]
        if low in {"error", "err"}:
            return "high"
        if low in {"warn", "warning", "info"}:
            return "medium"
    return "medium"


def _find_chunk_for(file_path: str, batch: ChunkBatch) -> CodeChunk | None:
    for chunk in batch.chunks:
        if chunk.file_path == file_path:
            return chunk
    return None


def _finding_from_dict(
    raw: dict[str, Any],
    *,
    validator: str,
    batch: ChunkBatch,
) -> CodeFinding | None:
    """Map a disseqt-go finding dict onto :class:`CodeFinding`.

    We accept a few field aliases because the app-sec judges aren't fully
    frozen yet — pinning to one exact shape would break on the first rename.
    """
    file_path = raw.get("file_path") or raw.get("file") or raw.get("path") or ""
    # If the validator didn't tag a file, fall back to the first chunk's
    # file — better than dropping the finding entirely.
    if not file_path and batch.chunks:
        file_path = batch.chunks[0].file_path
    if not file_path:
        return None

    chunk = _find_chunk_for(file_path, batch)
    line_start = int(raw.get("line_start") or raw.get("start_line") or raw.get("line") or 1)
    line_end_raw = raw.get("line_end") or raw.get("end_line")
    line_end = int(line_end_raw) if line_end_raw is not None else None

    vuln = str(raw.get("vulnerability") or raw.get("title") or raw.get("message") or validator)
    vuln_type = str(raw.get("vulnerability_type") or raw.get("type") or validator)
    reason = str(raw.get("reason") or raw.get("description") or raw.get("detail") or vuln)

    snippet = raw.get("code_snippet") or raw.get("snippet")
    if not snippet and chunk is not None:
        snippet = _slice_snippet(chunk, line_start, line_end)

    return CodeFinding(
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        vulnerability=vuln,
        vulnerability_type=vuln_type,
        severity=_coerce_severity(raw.get("severity")),
        reason=reason,
        recommendation=raw.get("recommendation") or raw.get("fix"),
        code_snippet=snippet,
        validator=validator,
        raw=raw,
    )


def _slice_snippet(chunk: CodeChunk, line_start: int, line_end: int | None) -> str | None:
    """Return the lines of ``chunk`` between ``line_start`` and ``line_end``."""
    if line_start < chunk.start_line or line_start > chunk.end_line:
        return None
    lines = chunk.text.splitlines()
    if not lines:
        return None
    end = line_end if line_end is not None else line_start
    end = min(end, chunk.end_line)
    lo = max(0, line_start - chunk.start_line)
    hi = min(len(lines), end - chunk.start_line + 1)
    return "\n".join(lines[lo:hi]) or None


def _extract_findings_list(envelope: Any) -> list[dict[str, Any]]:
    """Pull the findings array out of a validator response, or ``[]``."""
    if not isinstance(envelope, dict):
        return []
    data = envelope.get("data")
    for holder in (data if isinstance(data, dict) else {}, envelope):
        for key in ("findings", "issues", "results"):
            val = holder.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def dispatch(
    chunks: Iterable[CodeChunk],
    validators: Sequence[str],
    *,
    batch_chars: int = DEFAULT_BATCH_CHARS,
    transport: Transport | None = None,
    stats: DispatchStats | None = None,
) -> Iterator[CodeFinding]:
    """POST batches to each validator and yield the parsed findings.

    Batch-level failures are logged and skipped so one flaky validator doesn't
    kill the whole scan.
    """
    stats = stats or DispatchStats()
    send: Transport = transport or _default_transport
    if not validators:
        return

    batches = list(batch_chunks(chunks, batch_chars=batch_chars))
    stats.total_batches = len(batches) * max(len(validators), 1)

    for batch in batches:
        for validator in validators:
            try:
                envelope = send(
                    "POST", _validator_path(validator), _build_payload(batch, validator)
                )
            except SystemExit:
                # _http._fail() raises SystemExit — treat as a batch-level
                # failure, not a hard-abort, so the scan keeps going.
                log.warning(
                    "validator %s failed for batch of %d chunks", validator, len(batch.chunks)
                )
                stats.batches_failed += 1
                continue
            except Exception as exc:  # noqa: BLE001 — anything the HTTP layer throws
                log.warning("validator %s errored: %s", validator, exc)
                stats.batches_failed += 1
                continue
            stats.batches_ok += 1
            for raw in _extract_findings_list(envelope):
                finding = _finding_from_dict(raw, validator=validator, batch=batch)
                if finding is None:
                    continue
                stats.findings_by_validator[validator] = (
                    stats.findings_by_validator.get(validator, 0) + 1
                )
                yield finding
