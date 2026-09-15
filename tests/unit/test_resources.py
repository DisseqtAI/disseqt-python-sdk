"""Unit tests for the new resource clients on :class:`DisseqtAPIClient`.

Uses ``requests_mock`` at the HTTP boundary, matching the style of
``test_prompt_packs.py``. Covers the create/list/get/(update)/delete
happy-path for every resource family added in this PR.
"""

from __future__ import annotations

import pytest

from disseqt_sdk.api_client import DisseqtAPIClient

BASE_URL = "http://localhost:8000"


@pytest.fixture
def client() -> DisseqtAPIClient:
    return DisseqtAPIClient(
        project_id="test_project_123",
        api_key="test_key_xyz",
        base_url=BASE_URL,
        timeout=10,
    )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


class TestTargetsResource:
    """LLM targets (/api/v1/llm/app-integrations)."""

    PATH = f"{BASE_URL}/api/v1/llm/app-integrations"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(self.PATH, json={"id": "t1"})
        assert client.targets.create({"name": "gpt-4"}) == {"id": "t1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(self.PATH, json={"data": []})
        assert client.targets.list() == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/t1", json={"id": "t1"})
        assert client.targets.get("t1") == {"id": "t1"}

    def test_update(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.patch(f"{self.PATH}/t1", json={"id": "t1", "name": "renamed"})
        assert client.targets.update("t1", {"name": "renamed"})["name"] == "renamed"

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/t1", status_code=204)
        assert client.targets.delete("t1") == {"status": "ok"}

    def test_test(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/t1/test", json={"ok": True})
        assert client.targets.test("t1") == {"ok": True}

    def test_probe(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/test-connection", json={"ok": True})
        assert client.targets.probe({"provider": "openai"}) == {"ok": True}

    def test_parse_curl(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/parse-curl", json={"preview": {}})
        assert client.targets.parse_curl("curl https://x") == {"preview": {}}


class TestRagTargetsResource:
    PATH = f"{BASE_URL}/api/v1/rag-integrations"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(self.PATH, json={"id": "r1"})
        assert client.rag_targets.create({"kind": "pinecone"}) == {"id": "r1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(self.PATH, json={"data": []})
        assert client.rag_targets.list() == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/r1", json={"id": "r1"})
        assert client.rag_targets.get("r1") == {"id": "r1"}

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/r1", status_code=204)
        assert client.rag_targets.delete("r1") == {"status": "ok"}


class TestMcpTargetsResource:
    PATH = f"{BASE_URL}/api/v1/mcp-integrations"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(self.PATH, json={"id": "m1"})
        assert client.mcp_targets.create({"kind": "mcp"}) == {"id": "m1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(self.PATH, json={"data": []})
        assert client.mcp_targets.list() == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/m1", json={"id": "m1"})
        assert client.mcp_targets.get("m1") == {"id": "m1"}

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/m1", status_code=204)
        assert client.mcp_targets.delete("m1") == {"status": "ok"}


# ---------------------------------------------------------------------------
# Packs / Runs / Output Validations
# ---------------------------------------------------------------------------


class TestPacksResource:
    PATH = f"{BASE_URL}/api/v1/prompt-packs"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(self.PATH, json={"id": "p1"})
        assert client.packs.create({"pack_name": "P"}) == {"id": "p1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(self.PATH, json={"data": []})
        assert client.packs.list() == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/p1", json={"id": "p1"})
        assert client.packs.get("p1") == {"id": "p1"}

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/p1", status_code=204)
        assert client.packs.delete("p1") == {"status": "ok"}

    def test_add_prompts_bulk(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/prompts/bulk", json={"added": 2})
        result = client.packs.add_prompts("p1", [{"text": "a"}, {"text": "b"}])
        assert result == {"added": 2}

    def test_publish(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/publish", json={"published": True})
        assert client.packs.publish("p1") == {"published": True}

    def test_download_returns_csv_bytes(self, requests_mock, client: DisseqtAPIClient) -> None:
        csv_body = b"id,prompt\n1,hello\n2,world\n"
        requests_mock.get(
            f"{self.PATH}/p1/download",
            content=csv_body,
            headers={"Content-Type": "text/csv"},
        )
        result = client.packs.download("p1")
        assert isinstance(result, bytes)
        assert result == csv_body


class TestRunsResource:
    PATH = f"{BASE_URL}/api/v1/prompt-packs"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/runs", json={"id": "r1"})
        assert client.runs.create("p1", {"provider": "openai"}) == {"id": "r1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/p1/runs", json={"data": []})
        assert client.runs.list("p1") == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/runs/r1", json={"id": "r1"})
        assert client.runs.get("r1") == {"id": "r1"}

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/runs/r1", status_code=204)
        assert client.runs.delete("r1") == {"status": "ok"}

    def test_cancel(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/runs/r1/cancel", json={"cancelled": True})
        assert client.runs.cancel("r1") == {"cancelled": True}

    def test_reveal_output(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(
            f"{self.PATH}/runs/r1/results/o1/reveal",
            json={"revealed": True, "output_id": "o1"},
        )
        assert client.runs.reveal_output("r1", "o1") == {
            "revealed": True,
            "output_id": "o1",
        }

    def test_add_to_pack(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/prompts/add", json={"added": 2})
        result = client.runs.add_to_pack("r1", ["o1", "o2"], "p1")
        assert result == {"added": 2}
        # Body shape: {"run_id": ..., "output_ids": [...]}
        sent = requests_mock.request_history[0].json()
        assert sent == {"run_id": "r1", "output_ids": ["o1", "o2"]}


class TestOutputValidationsResource:
    PATH = f"{BASE_URL}/api/v1/prompt-packs"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/runs/r1/validate-outputs", json={"id": "v1"})
        assert client.output_validations.create("r1", {"metric_evaluations": []}) == {"id": "v1"}

    def test_list_for_pack(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/p1/output-validations", json={"data": []})
        assert client.output_validations.list_for_pack("p1") == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/output-validations/v1", json={"id": "v1"})
        assert client.output_validations.get("v1") == {"id": "v1"}

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/output-validations/v1", status_code=204)
        assert client.output_validations.delete("v1") == {"status": "ok"}


class TestRagValidationsResource:
    PATH = f"{BASE_URL}/api/v1/prompt-packs"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/runs/r1/rag-validate", json={"id": "v1"})
        assert client.rag_validations.create("r1", {}) == {"id": "v1"}

    def test_list_for_run(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/runs/r1/rag-validations", json={"data": []})
        assert client.rag_validations.list_for_run("r1") == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/rag-validations/v1", json={"id": "v1"})
        assert client.rag_validations.get("v1") == {"id": "v1"}


class TestSessionsResource:
    PATH = f"{BASE_URL}/api/v1/testing"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/sessions", json={"id": "s1"})
        assert client.sessions.create({"target": {"id": "t1"}}) == {"id": "s1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/sessions", json={"data": []})
        assert client.sessions.list() == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/sessions/s1", json={"id": "s1"})
        assert client.sessions.get("s1") == {"id": "s1"}

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/sessions/s1", status_code=204)
        assert client.sessions.delete("s1") == {"status": "ok"}

    def test_cancel_run(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/runs/r1/cancel", json={"cancelled": True})
        assert client.sessions.cancel_run("r1") == {"cancelled": True}


class TestByovValidatorsResource:
    PATH = f"{BASE_URL}/api/v1/llm/custom-validators"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(self.PATH, json={"id": "b1"})
        assert client.byov_validators.create({"name": "custom"}) == {"id": "b1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(self.PATH, json={"data": []})
        assert client.byov_validators.list() == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/b1", json={"id": "b1"})
        assert client.byov_validators.get("b1") == {"id": "b1"}

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/b1", status_code=204)
        assert client.byov_validators.delete("b1") == {"status": "ok"}

    def test_probe(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/test-connection", json={"ok": True})
        assert client.byov_validators.probe({"url": "https://x"}) == {"ok": True}


class TestVulnerabilitiesResource:
    PATH = f"{BASE_URL}/api/v1/vulnerabilities"

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(self.PATH, json={"data": [{"id": "v1"}]})
        assert client.vulnerabilities.list() == {"data": [{"id": "v1"}]}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/v1", json={"id": "v1"})
        assert client.vulnerabilities.get("v1") == {"id": "v1"}

    def test_test(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/v1/test", json={"job_id": "j1"})
        assert client.vulnerabilities.test("v1", {"target": "t1"}) == {"job_id": "j1"}


class TestTestPlansResource:
    """Test Plans T3 (/api/v1/test-plans). Endpoint list mirrors api/test_plans_routes.go."""

    PATH = f"{BASE_URL}/api/v1/test-plans"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(self.PATH, json={"id": "p1"})
        assert client.test_plans.create({"name": "n", "recipe": {}}) == {"id": "p1"}

    def test_list(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(self.PATH, json={"data": []})
        assert client.test_plans.list() == {"data": []}

    def test_gallery(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/gallery", json={"data": []})
        assert client.test_plans.gallery() == {"data": []}

    def test_list_deleted(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/deleted", json={"data": []})
        assert client.test_plans.list_deleted() == {"data": []}

    def test_options(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/options/categories", json={"data": []})
        assert client.test_plans.options("categories") == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/p1", json={"id": "p1"})
        assert client.test_plans.get("p1") == {"id": "p1"}

    def test_summary(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/p1/summary", json={"id": "p1"})
        assert client.test_plans.summary("p1") == {"id": "p1"}

    def test_update(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.patch(f"{self.PATH}/p1", json={"id": "p1", "name": "n2"})
        assert client.test_plans.update("p1", {"name": "n2"})["name"] == "n2"

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.PATH}/p1", status_code=204)
        assert client.test_plans.delete("p1") == {"status": "ok"}

    def test_restore(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/restore", json={"id": "p1"})
        assert client.test_plans.restore("p1") == {"id": "p1"}

    def test_copy(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/copy", json={"id": "p2"})
        assert client.test_plans.copy("p1") == {"id": "p2"}

    def test_list_versions(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PATH}/p1/versions", json={"data": []})
        assert client.test_plans.list_versions("p1") == {"data": []}

    def test_create_version(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/versions", json={"version": 2})
        assert client.test_plans.create_version("p1", {"recipe": {}}) == {"version": 2}

    def test_publish(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PATH}/p1/publish", json={"id": "p1"})
        body = {"sharing_scope": "PROJECT", "expected_sharing_scope": "PRIVATE"}
        assert client.test_plans.publish("p1", body) == {"id": "p1"}

    def test_generate_inputs(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(
            f"{self.PATH}/generate-inputs", json={"job_id": "j1", "status": "queued"}
        )
        body = {"app_description": "x" * 20, "subcategories": ["a"], "organization_id": "o"}
        assert client.test_plans.generate_inputs(body) == {"job_id": "j1", "status": "queued"}

    def test_get_generate_inputs_job(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(
            f"{self.PATH}/generate-inputs/j1", json={"job_id": "j1", "status": "succeeded"}
        )
        assert client.test_plans.get_generate_inputs_job("j1")["status"] == "succeeded"


class TestTestPlanRunsResource:
    """Test Plan Runs T6 (/api/v1/test-plan-runs + /api/v1/test-plans/{id}/runs)."""

    RUNS = f"{BASE_URL}/api/v1/test-plan-runs"
    PLANS = f"{BASE_URL}/api/v1/test-plans"

    def test_create(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.PLANS}/p1/runs", json={"run_id": "r1"})
        body = {"target": {"execution_mode": "app_integration", "app_integration_id": "a1"}}
        assert client.test_plan_runs.create("p1", body) == {"run_id": "r1"}

    def test_list_for_plan(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.PLANS}/p1/runs", json={"data": []})
        assert client.test_plan_runs.list_for_plan("p1") == {"data": []}

    def test_list_deleted(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.RUNS}/deleted", json={"data": []})
        assert client.test_plan_runs.list_deleted() == {"data": []}

    def test_get(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.RUNS}/r1", json={"run_id": "r1"})
        assert client.test_plan_runs.get("r1") == {"run_id": "r1"}

    def test_get_stage(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.RUNS}/r1/stages/baseline", json={"key": "baseline"})
        assert client.test_plan_runs.get_stage("r1", "baseline") == {"key": "baseline"}

    def test_trace(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.RUNS}/r1/trace", json={"prompt_ref": "pr1"})
        assert client.test_plan_runs.trace("r1", "pr1") == {"prompt_ref": "pr1"}

    def test_report(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.RUNS}/r1/report", json={"header": {}})
        assert client.test_plan_runs.report("r1", {"stage_key": "baseline"}) == {"header": {}}

    def test_prompts(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{self.RUNS}/r1/prompts", json={"items": []})
        assert client.test_plan_runs.prompts("r1") == {"items": []}

    def test_cancel(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.RUNS}/r1/cancel", json={"run_id": "r1", "status": "cancelled"})
        assert client.test_plan_runs.cancel("r1")["status"] == "cancelled"

    def test_delete(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{self.RUNS}/r1", status_code=204)
        assert client.test_plan_runs.delete("r1") == {"status": "ok"}

    def test_restore(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.RUNS}/r1/restore", json={"run_id": "r1"})
        assert client.test_plan_runs.restore("r1") == {"run_id": "r1"}

    def test_reveal(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.post(f"{self.RUNS}/r1/results/res1/reveal", json={"result_id": "res1"})
        assert client.test_plan_runs.reveal("r1", "res1") == {"result_id": "res1"}


class TestRequestAbs:
    """Cross-cutting behavior of the shared ``_request_abs`` helper."""

    def test_204_returns_ok_envelope(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.delete(f"{BASE_URL}/api/v1/llm/app-integrations/x", status_code=204)
        assert client.targets.delete("x") == {"status": "ok"}

    def test_list_response_wrapped(self, requests_mock, client: DisseqtAPIClient) -> None:
        """A bare JSON list is normalized to ``{'data': [...]}`` — dict return type."""
        requests_mock.get(f"{BASE_URL}/api/v1/vulnerabilities", json=[{"id": "v1"}])
        assert client.vulnerabilities.list() == {"data": [{"id": "v1"}]}

    def test_http_error_raises(self, requests_mock, client: DisseqtAPIClient) -> None:
        from disseqt_sdk.client import HTTPError

        requests_mock.get(f"{BASE_URL}/api/v1/vulnerabilities/nope", status_code=404, text="no")
        with pytest.raises(HTTPError) as exc:
            client.vulnerabilities.get("nope")
        assert exc.value.status_code == 404

    def test_auth_headers_sent(self, requests_mock, client: DisseqtAPIClient) -> None:
        requests_mock.get(f"{BASE_URL}/api/v1/prompt-packs", json={"data": []})
        client.packs.list()
        headers = requests_mock.request_history[0].headers
        assert headers["X-API-Key"] == "test_key_xyz"
        assert headers["X-Project-Id"] == "test_project_123"
