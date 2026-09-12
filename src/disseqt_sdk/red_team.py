"""``red_team()`` one-liner and :class:`RedTeamer` class (Phase 5c).

Both defer the actual server calls to the disseqt-dataset-backend-service
red-team endpoints (``/api/v1/testing/*`` and
``/api/v1/mr-jailbreak/batch-automate``) via :class:`Client`.

This module is intentionally lean — it's a thin orchestration layer. The
CLI (:mod:`disseqt_sdk.cli.redteam`) is what most users will drive; this
Python surface exists so notebooks/eval-harness code can call the same
thing without shelling out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .extensions import BaseAttack, BaseVulnerability

ModelCallback = Callable[[str], str]


@dataclass
class RedTeamRun:
    """One end-to-end red-team run's summary."""

    total_attacks: int
    materialized: list[dict[str, Any]] = field(default_factory=list)
    mitigated: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


class RedTeamer:
    """Stateful red-team session — reuses attack results across calls.

    Args:
        client: An initialized :class:`~disseqt_sdk.Client` (needed for
            SDK-authenticated hits on the red-team endpoints once
            Phase 4a API-key auth ships).
        reuse_previous_attacks: When True, calls to :meth:`red_team` skip
            attacks whose prompts were already run in this session.
    """

    def __init__(self, client: Any, reuse_previous_attacks: bool = False) -> None:
        self.client = client
        self.reuse_previous_attacks = reuse_previous_attacks
        self._history: list[dict[str, Any]] = []

    def red_team(
        self,
        model_callback: ModelCallback,
        vulnerabilities: list[BaseVulnerability],
        attacks: list[BaseAttack],
    ) -> RedTeamRun:
        """Run attacks-x-vulnerabilities against ``model_callback``.

        For each (vulnerability, attack) pair, generates a prompt via
        the attack's :meth:`~BaseAttack.enhance`, calls ``model_callback``,
        and records the (attempted) outcome. Server-side scoring of
        materialized vs. mitigated is out of scope here — Phase 4d's
        TraceScanner + the batch-automate endpoint own that path.
        """
        prompts_seen: set[str] = (
            {e["prompt"] for e in self._history} if self.reuse_previous_attacks else set()
        )
        materialized: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        total = 0
        for vuln in vulnerabilities:
            for atk in attacks:
                base_prompt = f"[{vuln.name}] test"
                prompt = atk.enhance(base_prompt)
                if prompt in prompts_seen:
                    continue
                total += 1
                try:
                    resp = model_callback(prompt)
                    entry = {
                        "vulnerability": vuln.name,
                        "attack": atk.name,
                        "prompt": prompt,
                        "response": resp,
                    }
                    materialized.append(entry)
                    self._history.append(entry)
                except Exception as exc:  # noqa: BLE001 — surface any provider error
                    errors.append(
                        {
                            "vulnerability": vuln.name,
                            "attack": atk.name,
                            "prompt": prompt,
                            "error": str(exc),
                        }
                    )
        return RedTeamRun(total_attacks=total, materialized=materialized, errors=errors)


def red_team(
    model_callback: ModelCallback,
    vulnerabilities: list[BaseVulnerability],
    attacks: list[BaseAttack],
    client: Any = None,
    reuse_previous_attacks: bool = False,
) -> RedTeamRun:
    """One-liner for a stateless red-team run.

    Wraps :class:`RedTeamer`. When ``client`` is None, a stub client is
    passed through — the current implementation only uses the callback,
    so this is safe for offline eval loops.
    """
    return RedTeamer(client=client, reuse_previous_attacks=reuse_previous_attacks).red_team(
        model_callback, vulnerabilities, attacks
    )
