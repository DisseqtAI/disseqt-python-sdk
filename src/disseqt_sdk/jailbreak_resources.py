"""Jailbreak (``/api/v1/jailbreak``) and multi-turn MR-jailbreak (``/api/v1/mr-jailbreak``)
resource clients; re-exported from :mod:`disseqt_sdk.resources`."""

from __future__ import annotations

from typing import Any

from .resources import _Base, _one_target, _with_fields


class JailbreakResource(_Base):
    """Jailbreak technique + prompt CRUD and generated-prompt controls.

    Routes registered under ``/api/v1/jailbreak/*`` — see
    ``dataset-backend api/jailbreak_routes.go`` on stage.
    """

    _BASE = "/api/v1/jailbreak"

    # Techniques
    def create_technique(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/techniques", json_payload=payload)

    def get_technique(self, technique_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/techniques/{technique_id}")

    def list_techniques(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/techniques", params=params)

    def update_technique(self, technique_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/techniques/{technique_id}", json_payload=payload)

    def delete_technique(self, technique_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/techniques/{technique_id}")

    # Prompts
    def create_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/prompts", json_payload=payload)

    def get_prompt(self, prompt_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/prompts/{prompt_id}")

    def list_prompts(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/prompts", params=params)

    def update_prompt(self, prompt_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/prompts/{prompt_id}", json_payload=payload)

    def delete_prompt(self, prompt_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/prompts/{prompt_id}")

    # Generated prompts
    def mark_generated_prompt_successful(self, generated_prompt_id: str) -> dict[str, Any]:
        """PATCH /generated-prompts/:id/success (binds ``success`` required; admin-only)."""
        return self._req(
            "PATCH",
            f"{self._BASE}/generated-prompts/{generated_prompt_id}/success",
            json_payload={"success": True},
        )


class MrResource(_Base):
    """Multi-round (MR) jailbreak techniques, agents, jobs, and strategies.

    Mirrors Node's ``RedteamClient`` MR surface. All routes live under
    ``/api/v1/mr-jailbreak/*`` on the redteam service. Existing usages —
    the CLI (:mod:`disseqt_sdk.cli.redteam`) and the parity test suite —
    already exercise the ``techniques``, ``agents``, ``jobs/:id``,
    ``jobs/:id/interactions``, and ``batch-automate`` paths; the CRUD +
    strategy verbs added here mirror the Node SDK method set.
    """

    _BASE = "/api/v1/mr-jailbreak"

    # ---- Techniques -----------------------------------------------------
    def create_technique(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/techniques", json_payload=payload)

    def list_techniques(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/techniques", params=params)

    def get_technique(self, technique_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/techniques/{technique_id}")

    def update_technique(self, technique_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/techniques/{technique_id}", json_payload=payload)

    def delete_technique(self, technique_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/techniques/{technique_id}")

    def register_technique(self, framework: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Register a technique implementation under a framework.

        ``framework`` ∈ {race, goat, crescendo, badlikertjudge, asja, agent}.
        """
        return self._req(
            "POST",
            f"{self._BASE}/techniques/{framework}/register",
            json_payload=payload,
        )

    # ---- Agents ---------------------------------------------------------
    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/agents", json_payload=payload)

    def list_agents(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/agents", params=params)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/agents/{agent_id}")

    def update_agent(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/agents/{agent_id}", json_payload=payload)

    def delete_agent(self, agent_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/agents/{agent_id}")

    # ---- Jobs -----------------------------------------------------------
    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/jobs", json_payload=payload)

    def list_jobs(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/jobs", params=params)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/jobs/{job_id}")

    def update_job_status(self, job_id: str, status: str) -> dict[str, Any]:
        return self._req(
            "PATCH", f"{self._BASE}/jobs/{job_id}/status", json_payload={"status": status}
        )

    def get_job_interactions(
        self, job_id: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/jobs/{job_id}/interactions", params=params)

    def get_job_available_states(self, job_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/jobs/{job_id}/available-states")

    # ---- Interactions + batch ------------------------------------------
    def analyze_interaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/interactions/analyze", json_payload=payload)

    def generate_next_interaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/interactions/generate-next", json_payload=payload)

    def batch_automate(
        self,
        payload: dict[str, Any],
        *,
        application_id: str | None = None,
        app_integration_id: str | None = None,
    ) -> dict[str, Any]:
        """api/mr_jailbreak_batch_automation.go: ``app_integration_id`` is top-level,
        the AI System is ``jailbreak_config.application_id``; either replaces the template."""
        target = _one_target(application_id=application_id, app_integration_id=app_integration_id)
        body = dict(payload)
        if target and target[0] == "application_id":
            body["jailbreak_config"] = _with_fields(
                body.get("jailbreak_config"), application_id=target[1]
            )
        elif target:
            body["app_integration_id"] = target[1]
        return self._req("POST", f"{self._BASE}/batch-automate", json_payload=body)

    # ---- Strategies -----------------------------------------------------
    def strategy_successful_paths(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/strategies/successful-paths", params=params)

    def strategy_agents_in_phase(self, phase: str) -> dict[str, Any]:
        return self._req(
            "GET", f"{self._BASE}/strategies/phase/agents", params={"phase_name": phase}
        )

    def strategy_techniques_in_phase(self, phase: str) -> dict[str, Any]:
        return self._req(
            "GET", f"{self._BASE}/strategies/phase/techniques", params={"phase_name": phase}
        )
