"""Phase 2 contract tests: identity, targets, AI Systems, wire fixes, shims.

Fixtures in ``fixtures/go_contract`` copy the Go json tags of the backend
structs; each use cites the source file in a one-line comment.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from disseqt_sdk import Client, DisseqtAPIClient
from disseqt_sdk.auth import AuthMissingError, token_store
from disseqt_sdk.cli import cli

FIX = Path(__file__).parent / "fixtures" / "go_contract"
BASE = "https://api.disseqt.ai/dataset"
POLICIES = "https://api.disseqt.ai/policy"
TESTING = f"{BASE}/api/v1/testing"
MR = f"{BASE}/api/v1/mr-jailbreak"
PACKS = f"{BASE}/api/v1/sdk/prompt-packs"
APP_ID = "55555555-5555-4555-8555-555555555555"
INTEG_ID = "88888888-8888-4888-8888-888888888888"
TECH_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
IDENTITY = {"project_id": "p-from-key", "organization_id": "o1", "user_id": "u1"}


def fx(name: str) -> Any:
    return json.loads((FIX / f"{name}.json").read_text())


def ok(data: Any) -> dict[str, Any]:
    # api/response.go RespondWithSuccess envelope.
    return {"status": "success", "data": data}


def body(req: Any) -> Any:
    return json.loads(req.text)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "DISSEQT_API_KEY",
        "DISSEQT_PROJECT_ID",
        "DISSEQT_BASE_URL",
        "DISSEQT_POLICIES_BASE_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DISSEQT_API_KEY", "sk_test_secret_key")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_api_client_omits_project_header_when_unset(self, requests_mock: Any) -> None:
        requests_mock.get(f"{TESTING}/sessions", json=ok({"items": []}))
        DisseqtAPIClient(api_key="k").sessions.list()
        assert "X-Project-Id" not in requests_mock.last_request.headers

    def test_api_client_sends_project_header_when_set(self, requests_mock: Any) -> None:
        requests_mock.get(f"{TESTING}/sessions", json=ok({"items": []}))
        DisseqtAPIClient(project_id="p1", api_key="k").sessions.list()
        assert requests_mock.last_request.headers["X-Project-Id"] == "p1"

    def test_api_client_env_fallback_and_default_base(self) -> None:
        c = DisseqtAPIClient()
        assert (c.api_key, c.project_id, c.base_url) == ("sk_test_secret_key", None, BASE)

    def test_api_client_env_beats_config(self) -> None:
        token_store.save({"api_key": "cfg", "project_id": "pcfg", "base_url": "http://cfg"})
        c = DisseqtAPIClient()
        # Env key wins and is never mixed with the config's project; base comes from config.
        assert (c.api_key, c.project_id, c.base_url) == ("sk_test_secret_key", None, "http://cfg")

    def test_api_client_without_any_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISSEQT_API_KEY")
        with pytest.raises(AuthMissingError):
            DisseqtAPIClient()

    def test_client_project_optional_and_env_beats_config(self) -> None:
        token_store.save({"api_key": "cfg", "project_id": "pcfg"})
        c = Client()
        assert (c.api_key, c.project_id) == ("sk_test_secret_key", None)
        assert "X-Project-Id" not in c._build_headers()

    def test_whoami_library(self, requests_mock: Any) -> None:
        requests_mock.get(f"{BASE}/api/v1/whoami", json=ok(IDENTITY))
        assert DisseqtAPIClient(api_key="k").whoami() == IDENTITY

    def test_cli_whoami_uses_server_identity(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.get(f"{BASE}/api/v1/whoami", json=ok(IDENTITY))
        res = runner.invoke(cli, ["whoami", "--json"])
        assert res.exit_code == 0, res.output
        out = json.loads(res.output)
        assert out["project_id"] == "p-from-key" and out["user_id"] == "u1"
        assert "sk_test_secret_key" not in res.output

    def test_cli_whoami_falls_back_to_local_on_404(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "p-env")
        requests_mock.get(f"{BASE}/api/v1/whoami", status_code=404, text="not found")
        res = runner.invoke(cli, ["whoami", "--json"])
        assert res.exit_code == 0, res.output
        assert "p-env" in res.output

    def test_cli_sends_no_project_header_without_project(
        self, runner: CliRunner, requests_mock: Any
    ) -> None:
        requests_mock.get(
            f"{TESTING}/attack-techniques", json=ok(fx("list_testing_results_response"))
        )
        res = runner.invoke(cli, ["redteam", "list-attacks", "--kind", "single"])
        assert res.exit_code == 0, res.output
        assert "X-Project-Id" not in requests_mock.last_request.headers

    def test_login_without_project_persists_base_url(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests_mock.get("http://gw/api/v1/testing/attack-techniques", json=ok({"items": []}))
        res = runner.invoke(cli, ["login", "--api-key", "sk_login", "--base-url", "http://gw"])
        assert res.exit_code == 0, res.output
        assert "X-Project-Id" not in requests_mock.last_request.headers
        stored = token_store.load() or {}
        assert stored.get("base_url") == "http://gw" and "project_id" not in stored
        monkeypatch.delenv("DISSEQT_API_KEY")
        res = runner.invoke(cli, ["redteam", "list-attacks", "--kind", "single"])
        assert res.exit_code == 0, res.output
        assert requests_mock.last_request.url.startswith("http://gw/")
        assert requests_mock.last_request.headers["X-API-Key"] == "sk_login"


# ---------------------------------------------------------------------------
# 2. Targets
# ---------------------------------------------------------------------------


def _mock_single_turn(requests_mock: Any) -> None:
    requests_mock.post(f"{TESTING}/sessions", json=ok({"id": "S1"}))
    requests_mock.post(f"{TESTING}/sessions/S1/runs", json=ok({"id": "R1"}))
    # api/testing_types.go TestingRunResponse (status "stopped" is terminal).
    requests_mock.get(f"{TESTING}/runs/R1", json=ok(fx("testing_run_response")))
    # api/testing_types.go ListTestingResultsResponse.
    requests_mock.get(f"{TESTING}/runs/R1/results", json=ok(fx("list_testing_results_response")))


_SINGLE = [
    "redteam",
    "attack",
    "--single-turn",
    "--technique",
    "t",
    "--pack",
    "P",
    "--validator",
    "v",
]
_MULTI = ["redteam", "attack", "--multi-turn", "--technique", TECH_ID, "--prompt", "hi"]


class TestTargets:
    @pytest.mark.parametrize(
        "flags,expected",
        [
            (["--app", APP_ID], {"application_id": APP_ID}),
            (["--integration", INTEG_ID], {"app_integration_id": INTEG_ID}),
            (["--endpoint", "https://t.example/chat"], {"endpoint_url": "https://t.example/chat"}),
        ],
    )
    def test_single_turn_target_config(
        self, runner: CliRunner, requests_mock: Any, flags: list[str], expected: dict[str, str]
    ) -> None:
        _mock_single_turn(requests_mock)
        res = runner.invoke(cli, [*_SINGLE, *flags, "--poll-interval", "0"])
        assert res.exit_code == 0, res.output
        assert body(requests_mock.request_history[0])["target_config"] == expected

    def test_single_turn_deprecated_target_maps_to_app(
        self, runner: CliRunner, requests_mock: Any
    ) -> None:
        _mock_single_turn(requests_mock)
        res = runner.invoke(cli, [*_SINGLE, "--target", APP_ID, "--poll-interval", "0"])
        assert res.exit_code == 0, res.output
        assert "deprecated" in res.output
        assert body(requests_mock.request_history[0])["target_config"] == {"application_id": APP_ID}

    @pytest.mark.parametrize(
        "flags",
        [[], ["--app", APP_ID, "--integration", INTEG_ID], ["--app", APP_ID, "--target", "x"]],
    )
    def test_exactly_one_target_or_usage_error(self, runner: CliRunner, flags: list[str]) -> None:
        res = runner.invoke(cli, [*_SINGLE, *flags])
        assert res.exit_code == 2, res.output

    def test_multi_turn_integration_sends_top_level_id_no_template(
        self, runner: CliRunner, requests_mock: Any
    ) -> None:
        requests_mock.get(f"{BASE}/api/v1/whoami", json=ok(IDENTITY))
        # api/mr_jailbreak_batch_automation.go BatchAutomateJailbreakResponse.
        requests_mock.post(f"{MR}/batch-automate", json=ok(fx("batch_automate_response")))
        requests_mock.get(
            f"{MR}/jobs/99999999-9999-4999-8999-999999999999", json=ok({"status": "completed"})
        )
        res = runner.invoke(cli, [*_MULTI, "--integration", INTEG_ID, "--poll-interval", "0"])
        assert res.exit_code == 0, res.output
        sent = body(requests_mock.request_history[1])
        assert sent["app_integration_id"] == INTEG_ID
        assert "app_integration_template" not in sent
        assert sent["jailbreak_config"]["project_id"] == "p-from-key"
        assert sent["jailbreak_config"]["app_type"] == "chatbot"

    def test_multi_turn_app_sets_jailbreak_config_application_id(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "p-env")
        requests_mock.post(f"{MR}/batch-automate", json=ok({"results": []}))
        res = runner.invoke(cli, [*_MULTI, "--app", APP_ID, "--app-name", "bot"])
        assert res.exit_code == 0, res.output
        sent = body(requests_mock.last_request)
        assert sent["jailbreak_config"]["application_id"] == APP_ID
        assert sent["jailbreak_config"]["app_name"] == "bot"
        assert "app_integration_template" not in sent and "app_integration_id" not in sent

    def test_multi_turn_deprecated_target_template(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "p-env")
        tpl = {
            "name": "n",
            "base_url": "http://t",
            "integration_type": "single-step",
            "send_step": {},
        }
        f = tmp_path / "t.json"
        f.write_text(json.dumps(tpl))
        requests_mock.post(f"{MR}/batch-automate", json=ok({"results": []}))
        res = runner.invoke(cli, [*_MULTI, "--target", f"@{f}"])
        assert res.exit_code == 0, res.output
        assert "deprecated" in res.output
        assert body(requests_mock.last_request)["app_integration_template"] == tpl

    def test_multi_turn_endpoint_needs_template(self, runner: CliRunner) -> None:
        res = runner.invoke(cli, [*_MULTI, "--endpoint", "https://t.example"])
        assert res.exit_code == 2, res.output

    @pytest.mark.parametrize(
        "flags,expected",
        [
            (["--integration", INTEG_ID], {"app_integration_id": INTEG_ID}),
            (["--app", APP_ID], {"application_id": APP_ID}),
            (["--target", INTEG_ID], {"app_integration_id": INTEG_ID}),
        ],
    )
    def test_vuln_test_targets(
        self,
        runner: CliRunner,
        requests_mock: Any,
        monkeypatch: pytest.MonkeyPatch,
        flags: list[str],
        expected: dict[str, str],
    ) -> None:
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "p-env")
        monkeypatch.setenv("DISSEQT_ORGANIZATION_ID", "o-env")
        requests_mock.post(f"{BASE}/api/v1/vulnerabilities/V/test/poll", json=ok({"run_id": "x"}))
        res = runner.invoke(cli, ["redteam", "vuln-test", "--vulnerability", "V", *flags])
        assert res.exit_code == 0, res.output
        # api/vulnerability_types.go VulnerabilityTestRequest.
        assert body(requests_mock.last_request) == expected

    def test_vuln_test_endpoint_is_usage_error(self, runner: CliRunner) -> None:
        res = runner.invoke(
            cli, ["redteam", "vuln-test", "--vulnerability", "V", "--endpoint", "u"]
        )
        assert res.exit_code == 2

    @pytest.mark.parametrize(
        "flags,key,value",
        [
            (["--app", APP_ID], "application_id", APP_ID),
            (["--integration", INTEG_ID], "app_integration_id", INTEG_ID),
            (["--endpoint", "https://t"], "endpoint_url", "https://t"),
        ],
    )
    def test_pp_run_create_targets(
        self, runner: CliRunner, requests_mock: Any, flags: list[str], key: str, value: str
    ) -> None:
        requests_mock.post(f"{PACKS}/P/runs", json=ok({"id": "r"}))
        res = runner.invoke(cli, ["pp-run", "create", "P", "--json", '{"max_tokens": 5}', *flags])
        assert res.exit_code == 0, res.output
        # api/prompt_pack_runs_handlers.go PromptPackRunRequest.
        assert body(requests_mock.last_request) == {"max_tokens": 5, key: value}

    def test_pp_run_create_two_targets_usage_error(self, runner: CliRunner) -> None:
        res = runner.invoke(cli, ["pp-run", "create", "P", "--app", APP_ID, "--endpoint", "u"])
        assert res.exit_code == 2

    @pytest.mark.parametrize(
        "flags,target",
        [
            (
                ["--integration", INTEG_ID],
                {"execution_mode": "app_integration", "app_integration_id": INTEG_ID},
            ),
            (["--app", APP_ID], {"execution_mode": "app_integration", "application_id": APP_ID}),
            (
                ["--endpoint", "https://t"],
                {"execution_mode": "direct", "endpoint_url": "https://t"},
            ),
        ],
    )
    def test_plan_run_create_targets(
        self, runner: CliRunner, requests_mock: Any, flags: list[str], target: dict[str, str]
    ) -> None:
        requests_mock.post(f"{BASE}/api/v1/test-plans/PL/runs", json=ok({"run_id": "r"}))
        res = runner.invoke(cli, ["plan-run", "create", "PL", *flags])
        assert res.exit_code == 0, res.output
        # api/test_plan_runs_create_types.go CreateTestPlanRunRequest.target.
        assert body(requests_mock.last_request) == {"target": target}

    def test_library_target_kwargs(self, requests_mock: Any) -> None:
        api = DisseqtAPIClient(api_key="k")
        requests_mock.post(f"{PACKS}/P/runs", json=ok({}))
        api.runs.create("P", {"max_tokens": 5}, application_id=APP_ID)
        assert body(requests_mock.last_request) == {"max_tokens": 5, "application_id": APP_ID}
        requests_mock.post(f"{BASE}/api/v1/vulnerabilities/V/test", json=ok({}))
        api.vulnerabilities.test(
            "V", app_integration_id=INTEG_ID, project_id="P1", organization_id="O1"
        )
        assert body(requests_mock.last_request) == {"app_integration_id": INTEG_ID}
        assert requests_mock.last_request.qs == {"project_id": ["p1"], "organization_id": ["o1"]}
        requests_mock.get(
            f"{BASE}/api/v1/whoami", json=ok({"project_id": "P2", "organization_id": "O2"})
        )
        api.vulnerabilities.test("V", application_id=APP_ID)
        assert requests_mock.last_request.qs == {"project_id": ["p2"], "organization_id": ["o2"]}
        requests_mock.post(f"{BASE}/api/v1/test-plans/PL/runs", json=ok({}))
        api.test_plan_runs.create("PL", endpoint_url="https://t")
        assert body(requests_mock.last_request)["target"] == {
            "execution_mode": "direct",
            "endpoint_url": "https://t",
        }
        requests_mock.post(f"{MR}/batch-automate", json=ok({}))
        api.mr.batch_automate({"jailbreak_config": {}}, application_id=APP_ID)
        assert body(requests_mock.last_request)["jailbreak_config"]["application_id"] == APP_ID
        api.mr.batch_automate({"jailbreak_config": {}}, app_integration_id=INTEG_ID)
        assert body(requests_mock.last_request)["app_integration_id"] == INTEG_ID
        with pytest.raises(ValueError):
            api.runs.create("P", {}, application_id=APP_ID, endpoint_url="u")


# ---------------------------------------------------------------------------
# 3. AI Systems
# ---------------------------------------------------------------------------


class TestApps:
    def test_app_list_hits_policy_mgmt(self, runner: CliRunner, requests_mock: Any) -> None:
        # policy-management internal/httpserver/handlers_applications.go listApplicationsResponse.
        requests_mock.get(f"{POLICIES}/api/v1/applications", json=fx("list_applications_response"))
        res = runner.invoke(cli, ["app", "list"])
        assert res.exit_code == 0, res.output
        assert "checkout-bot" in res.output

    def test_app_list_env_base(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISSEQT_POLICIES_BASE_URL", "http://pm")
        requests_mock.get("http://pm/api/v1/applications", json=fx("list_applications_response"))
        assert runner.invoke(cli, ["app", "list"]).exit_code == 0

    def test_app_link(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.post(f"{BASE}/api/v1/llm/application-links", json=ok({"id": "L"}))
        res = runner.invoke(cli, ["app", "link", APP_ID, "--integration", INTEG_ID])
        assert res.exit_code == 0, res.output
        # api/application_links.go createApplicationLinkRequest.
        assert body(requests_mock.last_request) == {
            "application_id": APP_ID,
            "integration_type": "app_integration",
            "integration_id": INTEG_ID,
        }

    def test_app_unlink(self, runner: CliRunner, requests_mock: Any) -> None:
        # api/application_links.go ListApplicationLinks bucket response.
        requests_mock.get(
            f"{BASE}/api/v1/llm/application-links", json=ok(fx("list_application_links_response"))
        )
        link_id = "66666666-6666-4666-8666-666666666666"
        requests_mock.delete(f"{BASE}/api/v1/llm/application-links/{link_id}", json=ok({}))
        res = runner.invoke(cli, ["app", "unlink", APP_ID])
        assert res.exit_code == 0, res.output
        assert requests_mock.request_history[0].qs == {"application_id": [APP_ID]}
        assert requests_mock.last_request.method == "DELETE"

    def test_app_unlink_without_link_fails(self, runner: CliRunner, requests_mock: Any) -> None:
        empty = {**fx("list_application_links_response"), "app_integrations": []}
        requests_mock.get(f"{BASE}/api/v1/llm/application-links", json=ok(empty))
        assert runner.invoke(cli, ["app", "unlink", APP_ID]).exit_code == 1


# ---------------------------------------------------------------------------
# 4. Contract fixes
# ---------------------------------------------------------------------------


class TestContractFixes:
    def test_parse_curl_sends_curl_commands(self, runner: CliRunner, requests_mock: Any) -> None:
        url = f"{BASE}/api/v1/llm/app-integrations/parse-curl"
        requests_mock.post(url, json=ok({}))
        DisseqtAPIClient(api_key="k").targets.parse_curl("curl http://x")
        # api/app_integration_types.go CurlCommands `json:"curl_commands"`.
        assert body(requests_mock.last_request) == {"curl_commands": "curl http://x"}
        assert (
            runner.invoke(cli, ["target", "parse-curl", "--curl", "curl http://y"]).exit_code == 0
        )
        assert body(requests_mock.last_request) == {"curl_commands": "curl http://y"}

    def test_add_to_pack_sends_entry_ids(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.post(f"{PACKS}/P/prompts/add", json=ok({}))
        DisseqtAPIClient(api_key="k").runs.add_to_pack("R", ["e1"], "P")
        # api/prompt_pack_entry_handlers.go AddPromptsToPackRequest entry_ids.
        assert body(requests_mock.last_request) == {"entry_ids": ["e1"]}
        res = runner.invoke(
            cli, ["pp-run", "add-to-pack", "R", "--entry-id", "e2", "--pack-id", "P"]
        )
        assert res.exit_code == 0, res.output
        assert body(requests_mock.last_request) == {"entry_ids": ["e2"]}

    def test_csv_endpoints_return_raw(self, runner: CliRunner, requests_mock: Any) -> None:
        csv_text = "a,b\n1,2\n"
        requests_mock.get(f"{PACKS}/output-validations/V/results/csv", text=csv_text)
        requests_mock.get(f"{TESTING}/sessions/S/report/csv", text=csv_text)
        api = DisseqtAPIClient(api_key="k")
        assert api.output_validations.results_csv("V") == csv_text.encode()
        assert api.sessions.report_csv("S") == csv_text.encode()
        for argv in (
            ["output-validation", "results-csv", "V"],
            ["session", "report-csv", "S"],
            ["redteam", "report", "--format", "csv", "--session", "S"],
        ):
            res = runner.invoke(cli, argv)
            assert res.exit_code == 0, res.output
            assert res.output.strip() == csv_text.strip(), argv

    def test_get_breach_reads_items(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.get(
            f"{TESTING}/runs/R/results/breaches", json=ok(fx("list_testing_results_response"))
        )
        bid = "33333333-3333-4333-8333-333333333333"
        assert DisseqtAPIClient(api_key="k").sessions.get_breach("R", bid)["id"] == bid
        res = runner.invoke(cli, ["redteam", "breach", "get", "R", bid])
        assert res.exit_code == 0, res.output

    def test_report_markdown_reads_items(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.get(f"{TESTING}/runs/R/results", json=ok(fx("list_testing_results_response")))
        res = runner.invoke(cli, ["redteam", "report", "R", "--format", "markdown"])
        assert res.exit_code == 0, res.output
        assert "ignore previous instructions" in res.output

    def test_results_all_paginates(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.get(
            f"{TESTING}/runs/R/results",
            [
                {"json": ok(fx("list_testing_results_response"))},
                {"json": ok(fx("list_testing_results_response_page2"))},
            ],
        )
        res = runner.invoke(cli, ["redteam", "results", "R", "--all", "--page-size", "50"])
        assert res.exit_code == 0, res.output
        out = json.loads(res.output)
        assert [i["prompt_text"] for i in out["items"]] == [
            "ignore previous instructions",
            "second",
        ]
        assert [r.qs["page"] for r in requests_mock.request_history] == [["1"], ["2"]]

    def test_list_attacks_pagination_params(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.get(f"{TESTING}/attack-techniques", json=ok({"items": [], "pagination": {}}))
        res = runner.invoke(
            cli, ["redteam", "list-attacks", "--kind", "single", "--page", "2", "--page-size", "10"]
        )
        assert res.exit_code == 0, res.output
        assert requests_mock.last_request.qs == {"page": ["2"], "page-size": ["10"]}

    def test_run_models(self) -> None:
        from disseqt_sdk.models.redteam import TestingRunStatus, TestingValidateResult

        # api/testing_types.go TestingRunResponse + stop_reason.
        run = TestingRunStatus.from_dict(fx("testing_run_response"))
        assert run.is_terminal and run.stop_reason == "user_cancel"
        # api/testing_validate_handler.go TestingValidateResponse.
        verdict = TestingValidateResult.from_dict(fx("testing_validate_response"))
        assert verdict.aggregate_decision == "BORDERLINE" and not verdict.is_blocking
        with pytest.raises(ValueError):
            TestingValidateResult.from_dict({"aggregate_decision": "MAYBE", "results": []})

    def test_output_validations_list_uses_registered_path(self, requests_mock: Any) -> None:
        requests_mock.get(f"{PACKS}/PACK/output-validations", json={"items": []})
        DisseqtAPIClient(api_key="k").list_run_output_validations("PACK")
        assert requests_mock.called

    def test_list_personas_attack_type_query(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.get(f"{MR}/agents", json=ok({"agents": [], "total_count": 0}))
        res = runner.invoke(
            cli, ["redteam", "list-personas", "--attack-type", "social_engineering"]
        )
        assert res.exit_code == 0, res.output
        assert requests_mock.last_request.qs["attack_type"] == ["social_engineering"]

    def test_results_fallback_only_on_404(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.get(f"{TESTING}/runs/J/results", status_code=404, text="nf")
        requests_mock.get(f"{MR}/jobs/J/interactions", json=ok([{"turn": 1}]))
        assert runner.invoke(cli, ["redteam", "results", "J"]).exit_code == 0
        requests_mock.get(f"{TESTING}/runs/J/results", status_code=500, text="boom")
        assert runner.invoke(cli, ["redteam", "results", "J"]).exit_code == 1


# ---------------------------------------------------------------------------
# 5. Back-compat shims
# ---------------------------------------------------------------------------


class TestBackCompat:
    def test_legacy_prefix_for_explicit_kong_base(self, requests_mock: Any) -> None:
        url = "http://localhost:8000/sdk/prompt-packs/api/v1/sdk/prompt-packs/runs/R"
        requests_mock.get(url, json={"id": "R"})
        DisseqtAPIClient("p", "k", base_url="http://localhost:8000").get_run("R")
        assert requests_mock.called

    def test_default_base_uses_service_mount(self, requests_mock: Any) -> None:
        requests_mock.get(f"{PACKS}/runs/R", json={"id": "R"})
        DisseqtAPIClient(api_key="k").get_run("R")
        assert requests_mock.called

    def test_policy_module_shims_warn(self) -> None:
        import disseqt_sdk
        from disseqt_sdk import policy

        assert "any_blocking" in disseqt_sdk.__all__
        with pytest.warns(DeprecationWarning):
            assert policy.is_blocking({"decision": "BLOCK"})

    def test_client_policies_kwargs_warn_and_evaluate(self, requests_mock: Any) -> None:
        from disseqt_sdk.models.input_validation import InputValidationRequest

        with pytest.warns(DeprecationWarning):
            c = Client(
                project_id="p",
                api_key="k",
                application_name="app",
                realtime_policy_base_url="http://rp",
            )
        requests_mock.post("http://rp/api/v1/sdk/policies/PID/evaluate", json={"decision": "PASS"})
        with pytest.warns(DeprecationWarning):
            out = c.validate(InputValidationRequest(prompt="hi"), policies=["PID"])
        assert out["policies"] == [{"decision": "PASS"}]


# ---------------------------------------------------------------------------
# 6. Unfinished surfaces are not exported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["red_team", "RedTeamer", "cost_accumulator"])
def test_unfinished_surface_not_exported(name: str) -> None:
    import disseqt_sdk

    assert name not in disseqt_sdk.__all__ and not hasattr(disseqt_sdk, name)


# ---------------------------------------------------------------------------
# 7. CLI polling + config
# ---------------------------------------------------------------------------


class TestCliPolling:
    def test_interrupt_cancels_run(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_single_turn(requests_mock)
        requests_mock.get(f"{TESTING}/runs/R1", json=ok({"status": "running"}))
        requests_mock.post(f"{TESTING}/runs/R1/cancel", json=ok({"status": "cancelled"}))

        def boom(_: float) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("disseqt_sdk.cli._http.time.sleep", boom)
        res = runner.invoke(cli, [*_SINGLE, "--app", APP_ID])
        assert res.exit_code == 130, res.output
        assert requests_mock.last_request.url.endswith("/runs/R1/cancel")
        assert "disseqt redteam status R1" in res.output

    def test_timeout_cancels_run(self, runner: CliRunner, requests_mock: Any) -> None:
        _mock_single_turn(requests_mock)
        requests_mock.post(f"{TESTING}/runs/R1/cancel", json=ok({}))
        res = runner.invoke(cli, [*_SINGLE, "--app", APP_ID, "--max-wait", "0"])
        assert res.exit_code == 1, res.output
        assert requests_mock.last_request.url.endswith("/runs/R1/cancel")

    def test_yaml_without_pyyaml_is_clear_error(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "run.yaml"
        cfg.write_text("name: x\n")
        monkeypatch.setitem(sys.modules, "yaml", None)
        res = runner.invoke(cli, ["redteam", "run", str(cfg)])
        assert res.exit_code == 1
        assert "pyyaml" in res.output.lower()


def test_no_stray_warnings_from_plain_client() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Client(project_id="p", api_key="k")


class TestNodeParity:
    def test_whoami_404_reports_local_source(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests_mock.get(f"{BASE}/api/v1/whoami", status_code=404, text="nf")
        res = runner.invoke(cli, ["whoami"])
        assert res.exit_code == 0, res.output
        assert "source: local" in res.output

    def test_app_link_role(self, runner: CliRunner, requests_mock: Any) -> None:
        requests_mock.post(f"{BASE}/api/v1/llm/application-links", json=ok({}))
        res = runner.invoke(
            cli, ["app", "link", APP_ID, "--integration", INTEG_ID, "--role", "staging"]
        )
        assert res.exit_code == 0, res.output
        # api/application_links.go createApplicationLinkRequest.role.
        assert body(requests_mock.last_request)["role"] == "staging"

    def test_app_unlink_deletes_every_link(self, requests_mock: Any) -> None:
        links = fx("list_application_links_response")
        extra = {**links["app_integrations"][0], "id": "L2"}
        listing = {**links, "llm_integrations": [extra]}
        requests_mock.get(f"{BASE}/api/v1/llm/application-links", json=ok(listing))
        requests_mock.delete(requests_mock_any_link(), json=ok({}))
        ids = DisseqtAPIClient(api_key="k").applications.unlink(APP_ID)
        assert ids == ["66666666-6666-4666-8666-666666666666", "L2"]

    def test_multi_turn_interrupt_only_hints(
        self, runner: CliRunner, requests_mock: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "p-env")
        requests_mock.post(f"{MR}/batch-automate", json=ok(fx("batch_automate_response")))
        job = "99999999-9999-4999-8999-999999999999"
        requests_mock.get(f"{MR}/jobs/{job}", json=ok({"status": "running"}))

        def boom(_: float) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("disseqt_sdk.cli._http.time.sleep", boom)
        res = runner.invoke(cli, [*_MULTI, "--integration", INTEG_ID])
        assert res.exit_code == 130, res.output
        assert f"disseqt redteam status {job}" in res.output
        assert all(r.method in ("GET", "POST") for r in requests_mock.request_history)


def test_multi_turn_skips_failed_zero_uuid_jobs(requests_mock: Any, monkeypatch) -> None:
    # TargetPromptResult.JobID is a non-pointer uuid: failed rows carry the nil UUID.
    monkeypatch.setenv("DISSEQT_API_KEY", "k")
    monkeypatch.setenv("DISSEQT_PROJECT_ID", "p-env")
    nil = "00000000-0000-0000-0000-000000000000"
    requests_mock.post(
        f"{MR}/batch-automate",
        json=ok({"results": [{"target_prompt": "p", "job_id": nil, "error": "x"}]}),
    )
    res = CliRunner().invoke(cli, [*_MULTI, "--integration", INTEG_ID])
    assert res.exit_code == 0, res.output
    assert not any(nil in r.url for r in requests_mock.request_history if r.method == "GET")


def requests_mock_any_link() -> Any:
    import re

    return re.compile(rf"{BASE}/api/v1/llm/application-links/.+")


def test_wait_terminal_reads_mr_job_status_nested_under_job(requests_mock, monkeypatch):
    # GET /mr-jailbreak/jobs/:id returns {job: {status}, interactions} (api/mr_jailbreak_handlers.go).
    from disseqt_sdk.cli import _http

    monkeypatch.setenv("DISSEQT_API_KEY", "k")
    monkeypatch.setenv("DISSEQT_BASE_URL", "http://x")
    requests_mock.get(
        "http://x/api/v1/mr-jailbreak/jobs/j1",
        json={
            "status": "success",
            "data": {"job": {"status": "FAILED", "stop_reason": "max_depth"}, "interactions": []},
        },
    )
    out = _http.wait_terminal("/api/v1/mr-jailbreak/jobs/j1", 0.01, 0.2)
    assert out["job"]["status"] == "FAILED"
