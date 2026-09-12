"""``disseqt policy`` command family (Phase 4c).

Targets disseqt-policy-management-service via ``X-Service-API-Key`` and
BFF-injected ``X-Organization-ID`` / ``X-Project-ID`` (env-driven — see
:mod:`._http`).
"""

from __future__ import annotations

import click

from . import _http
from ._common import echo_json

BASE_ENV = "DISSEQT_POLICY_MGMT_BASE_URL"
DEFAULT_BASE = "https://api.disseqt.ai/policy"


@click.group("policy")
def policy() -> None:
    """GRC policy commands (enrollment, coverage, evidence, audit packs)."""


@policy.command("list-frameworks")
def list_frameworks() -> None:
    """List available regulatory frameworks."""
    echo_json(_http.request("GET", BASE_ENV, DEFAULT_BASE, "/v1/frameworks"))


@policy.command("enroll")
@click.option("--framework", "code", required=True, help="Framework code (e.g. OWASP_LLM_2025).")
def enroll(code: str) -> None:
    """Enroll this project into a framework."""
    echo_json(_http.request("POST", BASE_ENV, DEFAULT_BASE, f"/v1/frameworks/{code}/enroll"))


@policy.command("withdraw")
@click.option("--framework", "code", required=True)
def withdraw(code: str) -> None:
    """Withdraw this project from a framework."""
    echo_json(_http.request("POST", BASE_ENV, DEFAULT_BASE, f"/v1/frameworks/{code}/withdraw"))


@policy.command("coverage")
@click.option("--framework", "code", required=True)
def coverage(code: str) -> None:
    """Show coverage for a framework."""
    echo_json(_http.request("GET", BASE_ENV, DEFAULT_BASE, f"/v1/frameworks/{code}/coverage"))


@policy.group("evidence")
def evidence() -> None:
    """Evidence management for controls."""


@evidence.command("upload")
@click.option("--control", "control_id", required=True, help="Control id.")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True, dir_okay=False))
def evidence_upload(control_id: str, file_path: str) -> None:
    """Upload an evidence file for a control."""
    with open(file_path, "rb") as fh:
        echo_json(
            _http.request(
                "POST",
                BASE_ENV,
                DEFAULT_BASE,
                "/v1/evidence/upload",
                files={"file": (file_path, fh)},
                extra_headers={"X-Control-ID": control_id},
            )
        )


@policy.group("audit-pack")
def audit_pack() -> None:
    """Audit-pack export."""


@audit_pack.command("export")
@click.option("--framework", "code", required=True)
def audit_pack_export(code: str) -> None:
    """Kick off an audit-pack export for a framework."""
    echo_json(
        _http.request(
            "POST",
            BASE_ENV,
            DEFAULT_BASE,
            "/v1/reports/audit-pack/exports",
            json_body={"framework_code": code},
        )
    )
