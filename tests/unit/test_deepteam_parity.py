"""Tests for Phase 0/1d/2a/5a/5c/6 additions from the DeepTeam parity plan.

Kept in one file so the parity surface is easy to review as a unit. CLI
end-to-end tests use click's CliRunner; no real HTTP is fired.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from disseqt_sdk import (
    DECISION_BORDERLINE,
    BaseGuard,
    BaseSingleTurnAttack,
    BaseVulnerability,
    BlockedError,
    Client,
    Guardrails,
    GuardResult,
    RedTeamer,
    cost_accumulator,
    red_team,
)
from disseqt_sdk.cli import cli
from disseqt_sdk.models import Example, SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest

BASE = "https://policies.test"
P_BLOCK = "11111111-1111-4111-8111-111111111111"
P_PASS = "22222222-2222-4222-8222-222222222222"
TOX_URL = f"{BASE}/api/v1/sdk/validators/input-validation/toxicity"
P_BLOCK_URL = f"{BASE}/api/v1/sdk/policies/{P_BLOCK}/evaluate"
P_PASS_URL = f"{BASE}/api/v1/sdk/policies/{P_PASS}/evaluate"

BLOCK_ENV = {
    "status": "success",
    "code": "DSQ-2000",
    "data": {"policy_id": P_BLOCK, "decision": "BLOCK", "enforcement": "sync"},
}
PASS_ENV = {
    "status": "success",
    "code": "DSQ-2000",
    "data": {"policy_id": P_PASS, "decision": "PASS", "enforcement": "sync"},
}
ASYNC_ENV = {
    "status": "success",
    "code": "DSQ-2000",
    "data": {"policy_id": P_PASS, "decision": "PASS", "enforcement": "async"},
}


def make_client() -> Client:
    return Client(
        project_id="proj",
        api_key="key",
        base_url=BASE,
        realtime_policy_base_url=BASE,
        application_name="parity-tests",
    )


class TestPhase0bBlockedError:
    def test_validate_sync_returns_result_on_pass(self, requests_mock):
        requests_mock.post(P_PASS_URL, json=PASS_ENV)
        result = make_client().validate_sync(InputValidationRequest(prompt="hi"), policies=[P_PASS])
        assert isinstance(result, dict) and result.get("policies")

    def test_validate_sync_raises_on_block(self, requests_mock):
        requests_mock.post(P_BLOCK_URL, json=BLOCK_ENV)
        with pytest.raises(BlockedError) as ei:
            make_client().validate_sync(InputValidationRequest(prompt="hi"), policies=[P_BLOCK])
        # Envelope is attached for downstream inspection.
        assert isinstance(ei.value.result, dict)
        assert ei.value.result["policies"][0]["data"]["decision"] == "BLOCK"

    def test_raise_on_async_flag(self, requests_mock):
        requests_mock.post(P_PASS_URL, json=ASYNC_ENV)
        with pytest.raises(BlockedError, match="async"):
            make_client().validate_sync(
                InputValidationRequest(prompt="hi"),
                policies=[P_PASS],
                raise_on_async=True,
            )


class TestPhase0cBorderline:
    def test_borderline_constant_exposed(self):
        assert DECISION_BORDERLINE == "BORDERLINE"

    def test_borderline_is_not_blocking_by_default(self):
        from disseqt_sdk import is_blocking

        env = {"decision": DECISION_BORDERLINE, "enforcement": "sync"}
        # Explicit product decision — BORDERLINE doesn't gate.
        assert is_blocking(env) is False


class TestPhase1dEvaluationExamples:
    def test_examples_serialize_into_judge_block(self):
        cfg = SDKConfigInput(
            threshold=0.5,
            llm_as_a_judge=True,
            llm_id="int-1",
            evaluation_examples=[
                Example(input="q", expected_output="a", rationale="because"),
                Example(input="q2", expected_output="a2", score=0.9),
            ],
        )
        out = cfg.to_dict()
        assert "judge" in out
        exs = out["judge"]["evaluation_examples"]
        assert exs[0] == {"input": "q", "expected_output": "a", "rationale": "because"}
        assert exs[1] == {"input": "q2", "expected_output": "a2", "score": 0.9}

    def test_empty_examples_omitted(self):
        cfg = SDKConfigInput(threshold=0.5)
        assert "judge" not in cfg.to_dict()


class TestPhase2aGuardrails:
    def test_guard_input_pass(self, requests_mock):
        requests_mock.post(P_PASS_URL, json=PASS_ENV)
        g = Guardrails(make_client(), input_guards=[P_PASS])
        result = g.guard_input(InputValidationRequest(prompt="hi"))
        assert isinstance(result, GuardResult)
        assert result.blocked is False
        assert len(result.policy_envelopes) == 1

    def test_guard_input_block(self, requests_mock):
        requests_mock.post(P_BLOCK_URL, json=BLOCK_ENV)
        g = Guardrails(make_client(), input_guards=[P_BLOCK])
        result = g.guard_input(InputValidationRequest(prompt="hi"))
        assert result.blocked is True

    def test_guard_input_no_guards_is_noop(self):
        g = Guardrails(make_client())
        assert g.guard_input(InputValidationRequest(prompt="hi")).blocked is False

    def test_guard_accepts_base_guard_object(self, requests_mock):
        requests_mock.post(P_PASS_URL, json=PASS_ENV)
        guard = BaseGuard(policy_id=P_PASS, name="allow-toxic")
        g = Guardrails(make_client(), input_guards=[guard])
        assert g.guard_input(InputValidationRequest(prompt="hi")).blocked is False


class TestPhase5aExtensions:
    def test_base_single_turn_attack_passthrough(self):
        atk = BaseSingleTurnAttack(name="noop")
        assert atk.enhance("original") == "original"
        assert atk.progress() == 1.0

    def test_base_vulnerability_stub_raises(self):
        vuln = BaseVulnerability(name="v1")
        with pytest.raises(NotImplementedError):
            vuln.assess(client=None, target=None)


class TestPhase5cRedTeam:
    def test_red_team_calls_callback_for_each_pair(self):
        seen: list[str] = []

        def cb(prompt: str) -> str:
            seen.append(prompt)
            return "ok"

        vulns = [BaseVulnerability(name="v1"), BaseVulnerability(name="v2")]
        atks = [BaseSingleTurnAttack(name="a1")]
        run = red_team(cb, vulns, atks)
        assert run.total_attacks == 2
        assert len(run.materialized) == 2
        assert len(seen) == 2

    def test_red_teamer_reuse_skips_prior_prompts(self):
        rt = RedTeamer(client=None, reuse_previous_attacks=True)
        vulns = [BaseVulnerability(name="v1")]
        atks = [BaseSingleTurnAttack(name="a1")]

        def cb(_: str) -> str:
            return "ok"

        run1 = rt.red_team(cb, vulns, atks)
        run2 = rt.red_team(cb, vulns, atks)
        assert run1.total_attacks == 1
        assert run2.total_attacks == 0  # deduped

    def test_red_team_captures_provider_errors(self):
        def cb(_: str) -> str:
            raise RuntimeError("provider down")

        run = red_team(cb, [BaseVulnerability(name="v")], [BaseSingleTurnAttack(name="a")])
        assert not run.materialized
        assert run.errors and "provider down" in run.errors[0]["error"]


class TestPhase6Telemetry:
    def test_cost_accumulator_scopes(self):
        with cost_accumulator() as bucket:
            bucket.add(simulation_cost=0.5, evaluation_cost=0.25)
            bucket.add(simulation_cost=0.5, prompt_tokens=100)
        assert bucket.total_cost == pytest.approx(1.25)
        assert bucket.prompt_tokens == 100
        assert len(bucket.entries) == 2


class TestCliValidate:
    """click CliRunner smoke tests — real HTTP intercepted by requests_mock."""

    def _env(self, monkeypatch):
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "proj")
        monkeypatch.setenv("DISSEQT_API_KEY", "key")
        monkeypatch.setenv("DISSEQT_BASE_URL", BASE)
        monkeypatch.setenv("DISSEQT_POLICY_BASE_URL", BASE)
        monkeypatch.setenv("DISSEQT_APPLICATION_NAME", "cli-tests")

    def test_validate_pass_exit_0(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.post(P_PASS_URL, json=PASS_ENV)
        r = CliRunner().invoke(cli, ["validate", "--input", "hi", "--policy", P_PASS])
        assert r.exit_code == 0, r.output

    def test_validate_block_exit_1(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.post(P_BLOCK_URL, json=BLOCK_ENV)
        r = CliRunner().invoke(cli, ["validate", "--input", "hi", "--policy", P_BLOCK])
        assert r.exit_code == 1, r.output

    def test_validate_missing_creds_exit_2(self, monkeypatch):
        # No env set
        monkeypatch.delenv("DISSEQT_PROJECT_ID", raising=False)
        monkeypatch.delenv("DISSEQT_API_KEY", raising=False)
        r = CliRunner().invoke(cli, ["validate", "--input", "hi", "--policy", P_PASS])
        assert r.exit_code == 2, r.output

    def test_run_yaml_fail_on_block(self, monkeypatch, requests_mock, tmp_path):
        pytest.importorskip("yaml")
        self._env(monkeypatch)
        requests_mock.post(P_BLOCK_URL, json=BLOCK_ENV)
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "default_policies:\n"
            f"  - {P_BLOCK}\n"
            "cases:\n"
            "  - name: c1\n"
            "    input: hello\n"
        )
        r = CliRunner().invoke(cli, ["run", str(cfg), "--fail-on-block"])
        assert r.exit_code == 1, r.output

    def test_scan_missing_binary(self, monkeypatch):
        monkeypatch.delenv("DISSEQT_SCAN_BIN", raising=False)
        # Force shutil.which to return None so the missing-binary path is
        # exercised even on machines with a coincidentally-named binary.
        import disseqt_sdk.cli.scan as scan_mod

        monkeypatch.setattr(scan_mod, "_resolve_bin", lambda: None)
        r = CliRunner().invoke(cli, ["scan"])
        assert r.exit_code == 127, r.output
