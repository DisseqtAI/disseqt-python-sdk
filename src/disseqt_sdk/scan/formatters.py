"""Render :class:`CodeFinding` collections as markdown / SARIF / JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .schema import SEVERITY_ORDER, CodeFinding

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
TOOL_NAME = "disseqt-scan"
TOOL_INFO_URI = "https://github.com/DisseqtAI/disseqt-python-sdk"

# SARIF only knows note < warning < error. Map our four-level severity.
SARIF_LEVEL: dict[str, str] = {
    "low": "note",
    "medium": "warning",
    "high": "error",
    "critical": "error",
}


def to_json(findings: Sequence[CodeFinding]) -> str:
    """Machine-readable JSON: a top-level ``findings`` array + counts."""
    counts: dict[str, int] = defaultdict(int)
    for f in findings:
        counts[f.severity] += 1
    payload: dict[str, Any] = {
        "tool": TOOL_NAME,
        "count": len(findings),
        "counts_by_severity": dict(counts),
        "findings": [f.to_dict() for f in findings],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def to_markdown(findings: Sequence[CodeFinding]) -> str:
    """Human-readable summary grouped by file, sorted by severity desc."""
    if not findings:
        return "# disseqt scan\n\nNo findings.\n"

    counts: dict[str, int] = defaultdict(int)
    by_file: dict[str, list[CodeFinding]] = defaultdict(list)
    for f in findings:
        counts[f.severity] += 1
        by_file[f.file_path].append(f)

    lines: list[str] = ["# disseqt scan", ""]
    lines.append(f"**{len(findings)} findings** across {len(by_file)} files.")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("| --- | --- |")
    for sev in ("critical", "high", "medium", "low"):
        if counts.get(sev):
            lines.append(f"| {sev} | {counts[sev]} |")
    lines.append("")

    for file_path in sorted(by_file):
        lines.append(f"## `{file_path}`")
        file_findings = sorted(
            by_file[file_path],
            key=lambda f: (-SEVERITY_ORDER.get(f.severity, 0), f.line_start),
        )
        for f in file_findings:
            loc = f"L{f.line_start}" if not f.line_end else f"L{f.line_start}-{f.line_end}"
            lines.append(f"- **[{f.severity.upper()}]** `{f.vulnerability_type}` at {loc}")
            lines.append(f"  - {f.reason}")
            if f.recommendation:
                lines.append(f"  - _Fix:_ {f.recommendation}")
            if f.code_snippet:
                fence = "```"
                lines.append(f"  {fence}")
                for snippet_line in f.code_snippet.splitlines():
                    lines.append(f"  {snippet_line}")
                lines.append(f"  {fence}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_sarif(findings: Sequence[CodeFinding]) -> str:
    """Minimal valid SARIF 2.1.0 for GitHub code-scanning upload."""
    rules_seen: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for f in findings:
        rule_id = f.vulnerability_type or f.validator or "unknown"
        if rule_id not in rules_seen:
            rules_seen[rule_id] = {
                "id": rule_id,
                "name": rule_id.replace("-", "_"),
                "shortDescription": {"text": f.vulnerability or rule_id},
                "fullDescription": {"text": f.reason or f.vulnerability or rule_id},
                "defaultConfiguration": {"level": SARIF_LEVEL.get(f.severity, "warning")},
                "properties": {"security-severity": _security_severity_score(f.severity)},
            }
        region: dict[str, Any] = {"startLine": max(1, f.line_start)}
        if f.line_end and f.line_end >= f.line_start:
            region["endLine"] = f.line_end
        if f.code_snippet:
            region["snippet"] = {"text": f.code_snippet}
        result: dict[str, Any] = {
            "ruleId": rule_id,
            "level": SARIF_LEVEL.get(f.severity, "warning"),
            "message": {"text": f.reason or f.vulnerability or rule_id},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.file_path},
                        "region": region,
                    }
                }
            ],
        }
        if f.recommendation:
            result["fixes"] = [{"description": {"text": f.recommendation}}]
        results.append(result)

    sarif: dict[str, Any] = {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": TOOL_NAME,
                        "informationUri": TOOL_INFO_URI,
                        "rules": list(rules_seen.values()),
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(sarif, indent=2, sort_keys=True)


def _security_severity_score(sev: str) -> str:
    """GitHub uses a 0-10 numeric score. Map our buckets to sane midpoints."""
    return {"low": "3.0", "medium": "5.5", "high": "7.5", "critical": "9.5"}.get(sev, "5.0")
