"""Unit tests for the ``disseqt scan`` code-scanner (Python side).

No real HTTP is fired — the dispatcher takes an injected ``transport``
callable, and the CLI tests stub the module-level HTTP request instead.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from disseqt_sdk.cli import cli
from disseqt_sdk.scan import (
    APPSEC_VALIDATORS,
    ChunkBatch,
    CodeChunk,
    CodeFinding,
    DispatchStats,
    ScanConfig,
    batch_chunks,
    changed_files,
    collect_chunks,
    dispatch,
    iter_source_files,
    meets_min_severity,
    resolve_batch_chars,
    to_json,
    to_markdown,
    to_sarif,
)
from disseqt_sdk.scan.git_diff import GitDiffError, parse_diff_range

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def test_meets_min_severity_ladder():
    assert meets_min_severity("critical", "low")
    assert meets_min_severity("medium", "medium")
    assert not meets_min_severity("low", "high")
    # Unknown severity → treated as below floor
    assert not meets_min_severity("unknown", "medium")


def test_code_finding_to_dict_drops_raw():
    f = CodeFinding(
        file_path="a.py",
        vulnerability="x",
        vulnerability_type="t",
        severity="high",
        reason="r",
        raw={"foo": "bar"},
    )
    d = f.to_dict()
    assert "raw" not in d
    assert d["severity"] == "high"


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def _write(tmp: Path, rel: str, body: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def test_iter_source_files_skips_ignored_dirs(tmp_path: Path):
    _write(tmp_path, "src/app.py", "print('hi')\n")
    _write(tmp_path, "src/app.min.js", "var x=1;\n")  # excluded glob
    _write(tmp_path, "node_modules/pkg/index.js", "x\n")  # excluded dir
    _write(tmp_path, "README.md", "# hi\n")  # unsupported ext
    _write(tmp_path, "src/svc.go", "package main\n")

    files = sorted(p.name for p in iter_source_files(tmp_path))
    assert files == ["app.py", "svc.go"]


def test_iter_source_files_respects_max_file_bytes(tmp_path: Path):
    _write(tmp_path, "big.py", "x = 1\n" * 10)
    files = list(iter_source_files(tmp_path, max_file_bytes=5))
    assert files == []


def test_iter_source_files_whitelist_only(tmp_path: Path):
    a = _write(tmp_path, "a.py", "x=1\n")
    _write(tmp_path, "b.py", "y=2\n")
    files = list(iter_source_files(tmp_path, only=[str(a)]))
    assert files == [a]


def test_chunk_file_line_alignment(tmp_path: Path):
    content = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"
    p = _write(tmp_path, "file.py", content)
    chunks = list(collect_chunks(tmp_path, max_chunk_chars=50))
    assert len(chunks) > 1
    # Chunks must be contiguous and cover every line exactly once.
    assert chunks[0].start_line == 1
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        assert curr.start_line == prev.end_line + 1
    assert chunks[-1].end_line == 20
    # Round-trip: concatenated chunks reconstruct the file.
    joined = "".join(c.text for c in chunks)
    assert joined == p.read_text(encoding="utf-8")


def test_collect_chunks_empty_file_skipped(tmp_path: Path):
    _write(tmp_path, "blank.py", "   \n\n")
    assert list(collect_chunks(tmp_path)) == []


# ---------------------------------------------------------------------------
# Batching + resolve_batch_chars
# ---------------------------------------------------------------------------


def _chunk(text: str, path: str = "a.py", start: int = 1) -> CodeChunk:
    end = start + max(0, text.count("\n"))
    return CodeChunk(file_path=path, language="python", start_line=start, end_line=end, text=text)


def test_batch_chunks_respects_char_cap():
    chunks = [_chunk("a" * 30), _chunk("b" * 30), _chunk("c" * 30)]
    batches = list(batch_chunks(chunks, batch_chars=50))
    # 30 + 30 > 50 → each new chunk starts a fresh batch
    assert len(batches) == 3
    assert all(isinstance(b, ChunkBatch) for b in batches)


def test_batch_chunks_bundles_when_under_cap():
    chunks = [_chunk("a" * 10), _chunk("b" * 10), _chunk("c" * 10)]
    batches = list(batch_chunks(chunks, batch_chars=100))
    assert len(batches) == 1
    assert batches[0].total_chars == 30


def test_resolve_batch_chars_precedence(monkeypatch):
    monkeypatch.delenv("DISSEQT_SCAN_CONTEXT_LIMIT", raising=False)
    assert resolve_batch_chars(None) == 40_000
    monkeypatch.setenv("DISSEQT_SCAN_CONTEXT_LIMIT", "77")
    assert resolve_batch_chars(None) == 77
    assert resolve_batch_chars(999) == 999  # CLI beats env


def test_resolve_batch_chars_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("DISSEQT_SCAN_CONTEXT_LIMIT", "not-a-number")
    assert resolve_batch_chars(None) == 40_000


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatch_parses_findings_and_updates_stats():
    chunks = [_chunk("if user_input: eval(user_input)\n", path="danger.py", start=42)]

    def fake_transport(method: str, path: str, body: dict):
        assert method == "POST"
        assert path.startswith("/api/v1/sdk/validators/input-validation/")
        assert body["input_data"]["llm_input_query"].startswith("--- FILE: danger.py")
        return {
            "data": {
                "findings": [
                    {
                        "file_path": "danger.py",
                        "line_start": 42,
                        "vulnerability": "Arbitrary code execution via eval()",
                        "vulnerability_type": "shell-injection",
                        "severity": "critical",
                        "reason": "eval on user input",
                        "recommendation": "use ast.literal_eval",
                    }
                ]
            }
        }

    stats = DispatchStats()
    findings = list(dispatch(chunks, ["shell-injection"], transport=fake_transport, stats=stats))
    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].vulnerability_type == "shell-injection"
    assert findings[0].code_snippet is not None
    assert stats.batches_ok == 1
    assert stats.findings_by_validator == {"shell-injection": 1}


def test_dispatch_swallows_transport_errors():
    chunks = [_chunk("x=1\n")]

    def broken(method, path, body):
        raise RuntimeError("connection refused")

    stats = DispatchStats()
    findings = list(dispatch(chunks, ["bfla"], transport=broken, stats=stats))
    assert findings == []
    assert stats.batches_failed == 1
    assert stats.batches_ok == 0


def test_dispatch_accepts_alternate_finding_shapes():
    chunks = [_chunk("secret = 'AKIA...'\n", path="s.py")]

    def transport(method, path, body):
        # No `data` wrapper, uses `issues` key + `file`/`type` aliases.
        return {
            "issues": [
                {
                    "file": "s.py",
                    "line": 1,
                    "type": "data-leakage",
                    "message": "hardcoded credential",
                    "severity": "HIGH",
                }
            ]
        }

    findings = list(dispatch(chunks, ["data-leakage"], transport=transport))
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].vulnerability_type == "data-leakage"


def test_dispatch_no_validators_is_noop():
    chunks = [_chunk("x=1\n")]
    findings = list(dispatch(chunks, [], transport=lambda *a, **k: {}))
    assert findings == []


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _sample_finding(**overrides):
    base = {
        "file_path": "src/app.py",
        "line_start": 10,
        "line_end": 12,
        "vulnerability": "Shell command injection",
        "vulnerability_type": "shell-injection",
        "severity": "high",
        "reason": "subprocess with shell=True on user input",
        "recommendation": "pass args as a list; don't set shell=True",
        "code_snippet": "subprocess.run(cmd, shell=True)",
        "validator": "shell-injection",
    }
    base.update(overrides)
    return CodeFinding(**base)


def test_to_json_produces_valid_json():
    doc = json.loads(to_json([_sample_finding()]))
    assert doc["count"] == 1
    assert doc["counts_by_severity"] == {"high": 1}
    assert doc["findings"][0]["file_path"] == "src/app.py"


def test_to_markdown_groups_by_file():
    md = to_markdown([_sample_finding(), _sample_finding(severity="critical", line_start=5)])
    assert "# disseqt scan" in md
    assert "src/app.py" in md
    assert "CRITICAL" in md  # highest severity first
    assert "shell-injection" in md


def test_to_markdown_empty_findings():
    assert "No findings" in to_markdown([])


def test_to_sarif_is_valid_schema_shape():
    doc = json.loads(to_sarif([_sample_finding()]))
    assert doc["version"] == "2.1.0"
    assert doc["$schema"].endswith("sarif-2.1.0.json")
    run = doc["runs"][0]
    assert run["tool"]["driver"]["name"] == "disseqt-scan"
    assert len(run["tool"]["driver"]["rules"]) == 1
    result = run["results"][0]
    assert result["ruleId"] == "shell-injection"
    assert result["level"] == "error"  # high → error
    loc = result["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "src/app.py"
    assert loc["region"]["startLine"] == 10
    assert loc["region"]["endLine"] == 12


def test_to_sarif_empty_findings_still_valid():
    doc = json.loads(to_sarif([]))
    assert doc["version"] == "2.1.0"
    assert doc["runs"][0]["results"] == []
    assert doc["runs"][0]["tool"]["driver"]["rules"] == []


def test_to_sarif_maps_severity_levels():
    for sev, expected in (
        ("low", "note"),
        ("medium", "warning"),
        ("high", "error"),
        ("critical", "error"),
    ):
        doc = json.loads(to_sarif([_sample_finding(severity=sev)]))
        assert doc["runs"][0]["results"][0]["level"] == expected


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------


def test_parse_diff_range_variants():
    assert parse_diff_range("main..HEAD") == ("main", "HEAD")
    assert parse_diff_range("main...HEAD") == ("main", "HEAD")


def test_parse_diff_range_rejects_bad_spec():
    with pytest.raises(GitDiffError):
        parse_diff_range("main")


def _init_git_repo(tmp: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp, check=True)


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not installed",
)
def test_changed_files_returns_diff(tmp_path: Path):
    _init_git_repo(tmp_path)
    _write(tmp_path, "a.py", "x=1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    _write(tmp_path, "b.py", "y=2\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add-b"], cwd=tmp_path, check=True)

    files = changed_files("HEAD~1..HEAD", cwd=tmp_path)
    assert [p.name for p in files] == ["b.py"]


# ---------------------------------------------------------------------------
# CLI end-to-end (with stubbed HTTP)
# ---------------------------------------------------------------------------


def test_cli_scan_help_lists_key_flags():
    result = CliRunner().invoke(cli, ["scan", "--help"])
    assert result.exit_code == 0
    for flag in ("--diff", "--format", "--min-severity", "--fail-on-findings", "--batch-chars"):
        assert flag in result.output


def test_cli_scan_no_files_exits_zero(tmp_path: Path):
    # No supported source files → clean exit even without --no-fail-on-findings.
    _write(tmp_path, "README.md", "# hi\n")
    result = CliRunner().invoke(cli, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "No source files" in result.output


def test_cli_scan_produces_sarif_when_requested(tmp_path: Path, monkeypatch):
    _write(tmp_path, "app.py", "eval(user_input)\n")

    def fake_request(method, base_env, default_base, path, json_body=None, **kwargs):
        return {
            "data": {
                "findings": [
                    {
                        "file_path": "app.py",
                        "line_start": 1,
                        "vulnerability": "eval on untrusted input",
                        "vulnerability_type": "shell-injection",
                        "severity": "critical",
                        "reason": "eval() on user input",
                    }
                ]
            }
        }

    monkeypatch.setattr("disseqt_sdk.cli._http.request", fake_request)

    result = CliRunner().invoke(
        cli,
        [
            "scan",
            str(tmp_path),
            "--format",
            "sarif",
            "--validator",
            "shell-injection",
            "--no-fail-on-findings",
        ],
    )
    assert result.exit_code == 0, result.output
    # CliRunner may mix stderr summary line into output; slice to the closing brace.
    body = result.output[: result.output.rindex("}") + 1]
    doc = json.loads(body)
    assert doc["version"] == "2.1.0"
    assert doc["runs"][0]["results"][0]["ruleId"] == "shell-injection"


def test_cli_scan_fail_on_findings_returns_1(tmp_path: Path, monkeypatch):
    _write(tmp_path, "app.py", "os.system(bad)\n")

    def fake_request(method, base_env, default_base, path, json_body=None, **kwargs):
        return {
            "data": {
                "findings": [
                    {
                        "file_path": "app.py",
                        "line_start": 1,
                        "vulnerability": "shell",
                        "vulnerability_type": "shell-injection",
                        "severity": "high",
                        "reason": "os.system with user data",
                    }
                ]
            }
        }

    monkeypatch.setattr("disseqt_sdk.cli._http.request", fake_request)

    result = CliRunner().invoke(
        cli,
        ["scan", str(tmp_path), "--format", "json", "--validator", "shell-injection"],
    )
    assert result.exit_code == 1
    assert '"vulnerability_type": "shell-injection"' in result.output


def test_cli_scan_min_severity_filters_out_low(tmp_path: Path, monkeypatch):
    _write(tmp_path, "app.py", "x = 1\n")

    def fake_request(method, base_env, default_base, path, json_body=None, **kwargs):
        return {
            "data": {
                "findings": [
                    {"file_path": "app.py", "line_start": 1, "severity": "low", "reason": "meh"}
                ]
            }
        }

    monkeypatch.setattr("disseqt_sdk.cli._http.request", fake_request)

    result = CliRunner().invoke(
        cli,
        [
            "scan",
            str(tmp_path),
            "--min-severity",
            "high",
            "--validator",
            "shell-injection",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    body = result.output[: result.output.rindex("}") + 1]
    doc = json.loads(body)
    assert doc["count"] == 0


def test_cli_scan_writes_output_file(tmp_path: Path, monkeypatch):
    _write(tmp_path, "app.py", "x = 1\n")
    monkeypatch.setattr(
        "disseqt_sdk.cli._http.request",
        lambda *a, **k: {"data": {"findings": []}},
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "scan",
            str(tmp_path),
            "--format",
            "json",
            "--output",
            str(out),
            "--validator",
            "bfla",
            "--no-fail-on-findings",
        ],
    )
    assert result.exit_code == 0
    doc = json.loads(out.read_text())
    assert doc["count"] == 0


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------


def test_scan_config_missing_returns_defaults(tmp_path: Path):
    from disseqt_sdk.scan.config import load

    cfg = load(tmp_path)
    assert isinstance(cfg, ScanConfig)
    assert cfg.validators == []
    assert cfg.min_severity is None


def test_appsec_validator_set_present():
    # The 6-judge default set is the top of the app-sec ladder.
    assert set(APPSEC_VALIDATORS) == {
        "bfla",
        "bola",
        "rbac",
        "shell-injection",
        "debug-access",
        "intellectual-property",
    }
