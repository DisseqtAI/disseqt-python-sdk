"""``disseqt redteam`` command family (Phase 4b).

Targets the newer ``/api/v1/testing/*`` surface + the batch endpoint
``/api/v1/mr-jailbreak/batch-automate``. All routes currently require
session cookies (see plan §4a — API-key middleware still pending);
setting ``DISSEQT_REDTEAM_BASE_URL`` lets you point at a mock during
that gap.
"""

from __future__ import annotations

import time
from typing import Any

import click

from . import _http
from ._common import echo_json

BASE_ENV = "DISSEQT_REDTEAM_BASE_URL"
DEFAULT_BASE = "https://api.disseqt.ai/dataset"


@click.group("redteam")
def redteam() -> None:
    """Run red-team attacks (single-turn, multi-turn, vuln tests)."""


@redteam.command("list-attacks")
@click.option("--kind", type=click.Choice(["single", "multi", "agents", "all"]), default="all")
def list_attacks(kind: str) -> None:
    """List available attack techniques (single-turn, multi-turn, agents)."""
    out: dict[str, Any] = {}
    if kind in ("single", "all"):
        out["single_turn"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/attack-techniques"
        )
    if kind in ("multi", "all"):
        out["multi_turn"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/techniques"
        )
    if kind in ("agents", "all"):
        out["agents"] = _http.request("GET", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/agents")
    echo_json(out)


@redteam.command("attack")
@click.option("--single-turn", "single_turn", is_flag=True, help="Run a single-turn attack.")
@click.option("--multi-turn", "multi_turn", is_flag=True, help="Run a multi-turn attack.")
@click.option("--technique", required=True, help="Attack technique id or name.")
@click.option("--target", required=True, help="Target model identifier / integration id.")
@click.option("--prompt", help="Seed prompt (single-turn) or objective (multi-turn).")
@click.option("--poll-interval", default=2.0, help="Seconds between run status polls.")
@click.option("--max-wait", default=300.0, help="Max seconds to wait for the run to finish.")
def attack(
    single_turn: bool,
    multi_turn: bool,
    technique: str,
    target: str,
    prompt: str | None,
    poll_interval: float,
    max_wait: float,
) -> None:
    """Run one red-team attack end to end."""
    if single_turn == multi_turn:
        raise click.UsageError("pass exactly one of --single-turn or --multi-turn")

    if multi_turn:
        body = {"technique": technique, "target": target, "objective": prompt or ""}
        result = _http.request(
            "POST", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/batch-automate", json_body=body
        )
        echo_json(result)
        return

    # single-turn: create session, create run, poll, fetch results
    session = _http.request(
        "POST", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/sessions", json_body={"target": target}
    )
    session_id = (session or {}).get("id") or (session or {}).get("session_id")
    if not session_id:
        raise click.ClickException(f"could not resolve session id from response: {session!r}")

    run = _http.request(
        "POST",
        BASE_ENV,
        DEFAULT_BASE,
        f"/api/v1/testing/sessions/{session_id}/runs",
        json_body={"technique": technique, "prompt": prompt or ""},
    )
    run_id = (run or {}).get("id") or (run or {}).get("run_id")
    if not run_id:
        raise click.ClickException(f"could not resolve run id from response: {run!r}")

    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        status = _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{run_id}")
        state = (status or {}).get("status") or (status or {}).get("state") or ""
        if state.lower() in ("completed", "failed", "cancelled", "error", "success"):
            results = _http.request(
                "GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{run_id}/results"
            )
            echo_json({"status": status, "results": results})
            return
        time.sleep(poll_interval)
    raise click.ClickException(f"run {run_id} did not finish within {max_wait}s")


@redteam.group("session")
def session() -> None:
    """Inspect red-team sessions."""


@session.command("list")
def session_list() -> None:
    """List red-team sessions."""
    echo_json(_http.request("GET", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/sessions"))


@session.command("get")
@click.argument("session_id")
def session_get(session_id: str) -> None:
    """Fetch one session by id."""
    echo_json(
        _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/sessions/{session_id}")
    )


@redteam.command("vuln-test")
@click.option("--vulnerability", "vuln_id", required=True, help="Vulnerability id to test.")
@click.option("--target", required=True, help="Target model identifier / integration id.")
def vuln_test(vuln_id: str, target: str) -> None:
    """Run the polling vuln-test endpoint for one vulnerability."""
    body = {"target": target}
    echo_json(
        _http.request(
            "POST",
            BASE_ENV,
            DEFAULT_BASE,
            f"/api/v1/vulnerabilities/{vuln_id}/test/poll",
            json_body=body,
        )
    )
