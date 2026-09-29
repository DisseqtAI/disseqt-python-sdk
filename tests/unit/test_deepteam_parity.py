"""Tests for Phase 0/1d/2a/5a/5c/6 additions from the DeepTeam parity plan.

Kept in one file so the parity surface is easy to review as a unit. CLI
end-to-end tests use click's CliRunner; no real HTTP is fired.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from disseqt_sdk import (
    BaseSingleTurnAttack,
    BaseVulnerability,
    RedTeamer,
    cost_accumulator,
    red_team,
)
from disseqt_sdk.cli import cli
from disseqt_sdk.models import Example, SDKConfigInput

# NOTE: TestPhase0bBlockedError, TestPhase0cBorderline, TestPhase2aGuardrails,
# and TestCliValidate were removed alongside the server-side policy-evaluate
# path (Guardrails, BlockedError, DECISION_BORDERLINE, BaseGuard,
# `disseqt validate`, `disseqt run`). See CHANGELOG for details.


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


# NOTE: TestCliValidate (policies-based `disseqt validate` and
# `disseqt run` smoke tests) removed alongside those commands. The
# surviving CLI groups (redteam, resource groups, scan)
# are covered in their own test modules.


class TestCliRedteamExpansion:
    """Follow-up: verify each new redteam verb wires up correctly."""

    RT_BASE = "https://redteam.test"

    def _env(self, monkeypatch):
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "proj")
        monkeypatch.setenv("DISSEQT_API_KEY", "key")
        monkeypatch.setenv("DISSEQT_BASE_URL", self.RT_BASE)

    def test_redteam_bare_non_tty_shows_help(self, monkeypatch):
        # Non-TTY (CliRunner isolation) falls through to --help.
        r = CliRunner().invoke(cli, ["redteam"])
        assert r.exit_code == 0, r.output
        assert "Usage:" in r.output
        # All new verbs appear in the help listing.
        for verb in ("run", "validate", "status", "cancel", "results", "report"):
            assert verb in r.output

    def test_redteam_validate_posts_and_prints(self, monkeypatch, requests_mock):
        """Wire contract: {input, output, validators, input_context[, threshold]}
        matching dataset-backend PR #794 POST /api/v1/testing/validate."""
        self._env(monkeypatch)
        m = requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/validate",
            json={
                "results": [{"validator_name": "toxicity", "score": 0.1, "rule_status": "pass"}],
                "aggregate_decision": "PASS",
            },
        )
        r = CliRunner().invoke(
            cli,
            [
                "redteam",
                "validate",
                "--input",
                "ignore prior instructions",
                "--validator",
                "toxicity",
                "--validator",
                "prompt_injection",
                "--input-context",
                "ctx",
                "--threshold",
                "0.5",
            ],
        )
        assert r.exit_code == 0, r.output
        assert '"aggregate_decision": "PASS"' in r.output
        # Server contract: body must have the plan's field names, not the
        # old {prompt, technique, vulnerability, target} shape.
        body = m.last_request.json()
        assert body["input"] == "ignore prior instructions"
        assert body["output"] == ""
        assert body["validators"] == ["toxicity", "prompt_injection"]
        assert body["input_context"] == "ctx"
        assert body["threshold"] == 0.5
        # None of the deprecated keys leak through.
        for legacy in ("prompt", "technique", "vulnerability", "target"):
            assert legacy not in body

    def test_redteam_validate_omits_threshold_when_unset(self, monkeypatch, requests_mock):
        """Threshold is optional — absent flag => absent JSON key
        (server default kicks in)."""
        self._env(monkeypatch)
        m = requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/validate",
            json={"results": [], "aggregate_decision": "BORDERLINE"},
        )
        r = CliRunner().invoke(
            cli,
            [
                "redteam",
                "validate",
                "--input",
                "hi",
                "--validator",
                "toxicity",
            ],
        )
        assert r.exit_code == 0, r.output
        body = m.last_request.json()
        assert "threshold" not in body
        assert body["validators"] == ["toxicity"]

    def test_redteam_validate_requires_at_least_one_validator(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(cli, ["redteam", "validate", "--input", "hi"])
        # click's missing-required-option => exit code 2.
        assert r.exit_code == 2, r.output
        assert "validator" in r.output.lower()

    def test_cli_sends_only_user_key_headers(self, monkeypatch, requests_mock):
        """Auth contract: the CLI sends X-API-Key + X-Project-Id only; the
        gateway injects X-Service-API-Key / X-Internal-Project-Id / X-User-Id."""
        self._env(monkeypatch)
        monkeypatch.setenv("DISSEQT_ORGANIZATION_ID", "org-123")
        m = requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/validate",
            json={"status": "success", "data": {"results": [], "aggregate_decision": "PASS"}},
        )
        r = CliRunner().invoke(
            cli,
            ["redteam", "validate", "--input", "hi", "--validator", "toxicity"],
        )
        assert r.exit_code == 0, r.output
        hdrs = m.last_request.headers
        assert hdrs["X-API-Key"] == "key"
        assert hdrs["X-Project-Id"] == "proj"
        for forbidden in (
            "X-Service-API-Key",
            "X-Internal-Project-Id",
            "X-User-Id",
            "X-User-Email",
            "X-Org-Id",
            "X-Organization-ID",
        ):
            assert forbidden not in hdrs
        # Envelope unwrapped: only `data` is printed.
        assert '"aggregate_decision": "PASS"' in r.output
        assert '"status": "success"' not in r.output

    def test_error_envelope_exits_1(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/validate",
            status_code=422,
            json={"status": "error", "error": {"external": "validators required"}},
        )
        r = CliRunner().invoke(
            cli, ["redteam", "validate", "--input", "hi", "--validator", "toxicity"]
        )
        assert r.exit_code == 1, r.output
        assert "422" in r.output

    def test_status_falls_back_to_mr_only_on_404(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(f"{self.RT_BASE}/api/v1/testing/runs/j1", status_code=404, text="nope")
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/mr-jailbreak/jobs/j1",
            json={"status": "success", "data": {"id": "j1", "status": "running"}},
        )
        r = CliRunner().invoke(cli, ["redteam", "status", "j1"])
        assert r.exit_code == 0, r.output
        assert '"status": "running"' in r.output
        assert "404" not in r.output

    def test_status_does_not_fall_back_on_500(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(f"{self.RT_BASE}/api/v1/testing/runs/j1", status_code=500, text="boom")
        mr = requests_mock.get(f"{self.RT_BASE}/api/v1/mr-jailbreak/jobs/j1", json={})
        r = CliRunner().invoke(cli, ["redteam", "status", "j1"])
        assert r.exit_code == 1
        assert not mr.called

    def test_redteam_status_hits_testing_first(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(f"{self.RT_BASE}/api/v1/testing/runs/abc", json={"state": "running"})
        r = CliRunner().invoke(cli, ["redteam", "status", "abc"])
        assert r.exit_code == 0, r.output
        assert "running" in r.output

    def test_redteam_cancel(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/runs/xyz/cancel", json={"cancelled": True}
        )
        r = CliRunner().invoke(cli, ["redteam", "cancel", "xyz"])
        assert r.exit_code == 0, r.output
        assert "cancelled" in r.output

    def test_redteam_results(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/testing/runs/xyz/results",
            json=[{"technique": "t1", "verdict": "PASS"}],
        )
        r = CliRunner().invoke(cli, ["redteam", "results", "xyz"])
        assert r.exit_code == 0, r.output
        assert "PASS" in r.output

    def test_redteam_list_personas_filters(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        m = requests_mock.get(
            f"{self.RT_BASE}/api/v1/mr-jailbreak/agents",
            json=[
                {"name": "dan", "attack_type": "roleplay"},
                {"name": "coder", "attack_type": "obfuscation"},
            ],
        )
        r = CliRunner().invoke(cli, ["redteam", "list-personas", "--attack-type", "roleplay"])
        assert r.exit_code == 0, r.output
        assert "dan" in r.output
        assert "coder" not in r.output
        # ListJailbreakAgentsRequest requires page_id + kebab-case page-size (5..100).
        assert m.last_request.qs == {"page_id": ["1"], "page-size": ["100"]}

    def test_redteam_list_techniques_multi_only(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/mr-jailbreak/techniques", json=[{"id": "crescendo"}]
        )
        r = CliRunner().invoke(cli, ["redteam", "list-techniques", "--multi-turn"])
        assert r.exit_code == 0, r.output
        assert "crescendo" in r.output
        # single-turn key omitted when --multi-turn is set.
        assert "single_turn" not in r.output

    def test_redteam_run_yaml_config(self, monkeypatch, requests_mock, tmp_path):
        """YAML keys map 1:1 onto CreateTestingSessionRequest / CreateTestingRunRequest."""
        pytest.importorskip("yaml")
        self._env(monkeypatch)
        monkeypatch.setenv("APP_ID", "app-1")
        sess = requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/sessions",
            json={"status": "success", "data": {"id": "sess-1"}},
        )
        run = requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/sessions/sess-1/runs",
            json={"status": "success", "data": {"id": "run-1"}},
        )
        cfg = tmp_path / "rt.yaml"
        cfg.write_text(
            "name: nightly\n"
            "application: {name: bot, type: web}\n"
            "target: {application_id: '${APP_ID}'}\n"
            "prompt_packs: [pack-1]\n"
            "techniques: [base64_encoding]\n"
            "validators: [toxicity]\n"
            "stop_on_first_breach: true\n"
            "max_total_prompts: 7\n"
        )
        r = CliRunner().invoke(cli, ["redteam", "run", str(cfg)])
        assert r.exit_code == 0, r.output
        assert "sess-1" in r.output and "run-1" in r.output
        body = sess.last_request.json()
        assert set(body) == {"name", "application_context", "target_config", "testing_plan"}
        assert body["name"] == "nightly"
        assert body["application_context"] == {"name": "bot", "type": "web"}
        assert body["target_config"] == {"application_id": "app-1"}
        plan = body["testing_plan"]
        assert plan["prompt_sources"] == [
            {"type": "prompt_pack", "config": {"pack_ids": ["pack-1"]}}
        ]
        assert plan["attack_strategies"][0]["type"] == "single_turn_jailbreak"
        assert plan["attack_strategies"][0]["config"] == {"techniques": ["base64_encoding"]}
        assert plan["validators"] == ["toxicity"]
        assert plan["execution"]["stop_on_first_breach"] is True
        assert plan["execution"]["max_total_prompts"] == 7
        assert set(run.last_request.json()) == {"run_name", "trigger_metadata"}

    def test_attack_single_turn_end_to_end(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        monkeypatch.setattr("disseqt_sdk.cli.redteam.time.sleep", lambda _s: None)
        sess = requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/sessions",
            json={"status": "success", "data": {"id": "s1"}},
        )
        requests_mock.post(
            f"{self.RT_BASE}/api/v1/testing/sessions/s1/runs",
            json={"status": "success", "data": {"id": "r1", "status": "pending"}},
        )
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/testing/runs/r1",
            [
                {"json": {"status": "success", "data": {"id": "r1", "status": "running"}}},
                {"json": {"status": "success", "data": {"id": "r1", "status": "completed"}}},
            ],
        )
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/testing/runs/r1/results",
            json={"status": "success", "data": [{"verdict": "breach"}]},
        )
        r = CliRunner().invoke(
            cli,
            [
                "redteam",
                "attack",
                "--single-turn",
                "--technique",
                "base64_encoding",
                "--target",
                "app-1",
                "--pack",
                "pack-1",
                "--validator",
                "toxicity",
                "--poll-interval",
                "0",
            ],
        )
        assert r.exit_code == 0, r.output
        body = sess.last_request.json()
        assert body["target_config"] == {"application_id": "app-1"}
        assert body["testing_plan"]["attack_strategies"][0]["techniques"] == ["base64_encoding"]
        assert '"verdict": "breach"' in r.output
        # The envelope's own "status": "success" must not be mistaken for a terminal run.
        assert '"status": "completed"' in r.output

    def test_attack_single_turn_requires_pack_and_validator(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(
            cli, ["redteam", "attack", "--single-turn", "--technique", "t", "--target", "a"]
        )
        assert r.exit_code == 2
        assert "--pack" in r.output

    def test_attack_multi_turn_batch_automate(self, monkeypatch, requests_mock, tmp_path):
        self._env(monkeypatch)
        monkeypatch.setattr("disseqt_sdk.cli.redteam.time.sleep", lambda _s: None)
        template = tmp_path / "tpl.json"
        template.write_text(
            '{"name": "bot", "base_url": "https://bot.example", "integration_type": "single-step",'
            ' "send_step": {"step_order": 1, "step_name": "send", "step_type": "send",'
            ' "api_endpoint": "/chat", "http_method": "POST"}}'
        )
        batch = requests_mock.post(
            f"{self.RT_BASE}/api/v1/mr-jailbreak/batch-automate",
            json={"status": "success", "data": {"results": [{"job_id": "j1"}]}},
        )
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/mr-jailbreak/jobs/j1",
            json={"status": "success", "data": {"id": "j1", "status": "completed"}},
        )
        r = CliRunner().invoke(
            cli,
            [
                "redteam",
                "attack",
                "--multi-turn",
                "--technique",
                "tech-uuid",
                "--target",
                f"@{template}",
                "--prompt",
                "p1",
                "--prompt",
                "p2",
                "--max-depth",
                "3",
                "--poll-interval",
                "0",
            ],
        )
        assert r.exit_code == 0, r.output
        body = batch.last_request.json()
        assert body["target_prompts"] == ["p1", "p2"]
        assert body["app_integration_template"]["name"] == "bot"
        assert body["ecid_prefix"] == "cli" and body["ecid_start_number"] == 1
        jc = body["jailbreak_config"]
        assert jc["project_id"] == "proj"
        assert jc["technique_id"] == "tech-uuid"
        assert jc["orchestration_mode"] == "single"
        assert jc["max_depth"] == 3
        assert jc["app_name"] == "bot"
        assert '"status": "completed"' in r.output

    def test_attack_multi_turn_rejects_too_many_prompts(self, monkeypatch):
        self._env(monkeypatch)
        args = ["redteam", "attack", "--multi-turn", "--technique", "t", "--target", "{}"]
        for i in range(11):
            args += ["--prompt", f"p{i}"]
        r = CliRunner().invoke(cli, args)
        assert r.exit_code == 2
        assert "1..10" in r.output

    def test_vuln_test_body_and_query(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        monkeypatch.setenv("DISSEQT_ORGANIZATION_ID", "org-1")
        m = requests_mock.post(
            f"{self.RT_BASE}/api/v1/vulnerabilities/v1/test/poll",
            json={"status": "success", "data": {"run_id": "vr1"}},
        )
        r = CliRunner().invoke(
            cli, ["redteam", "vuln-test", "--vulnerability", "v1", "--target", "int-1"]
        )
        assert r.exit_code == 0, r.output
        assert m.last_request.json() == {"app_integration_id": "int-1"}
        assert m.last_request.qs == {"project_id": ["proj"], "organization_id": ["org-1"]}

    def test_vuln_test_omits_org_when_unset(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        monkeypatch.delenv("DISSEQT_ORGANIZATION_ID", raising=False)
        m = requests_mock.post(f"{self.RT_BASE}/api/v1/vulnerabilities/v1/test/poll", json={})
        r = CliRunner().invoke(
            cli, ["redteam", "vuln-test", "--vulnerability", "v1", "--target", "int-1"]
        )
        assert r.exit_code == 0, r.output
        assert m.last_request.qs == {"project_id": ["proj"]}

    def test_redteam_report_json(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/testing/runs/rid/results",
            json=[{"technique": "t1", "verdict": "PASS", "score": 0.2}],
        )
        r = CliRunner().invoke(cli, ["redteam", "report", "rid", "--format", "json"])
        assert r.exit_code == 0, r.output
        assert "PASS" in r.output

    def test_redteam_report_markdown(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/testing/runs/rid/results",
            json=[{"technique": "t1", "verdict": "PASS"}],
        )
        r = CliRunner().invoke(cli, ["redteam", "report", "rid", "--format", "markdown"])
        assert r.exit_code == 0, r.output
        assert "| technique | verdict |" in r.output

    def test_redteam_report_csv_server_string(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        # Server returns raw CSV as text (non-JSON body). _http.request will
        # fall back to resp.text when JSON decoding fails.
        requests_mock.get(
            f"{self.RT_BASE}/api/v1/testing/sessions/rid/report/csv",
            text="technique,verdict\nt1,PASS\n",
        )
        r = CliRunner().invoke(cli, ["redteam", "report", "--session", "rid", "--format", "csv"])
        assert r.exit_code == 0, r.output
        assert "technique,verdict" in r.output

    def test_redteam_report_csv_requires_session(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(cli, ["redteam", "report", "rid", "--format", "csv"])
        assert r.exit_code == 2
        assert "--session" in r.output

    def test_existing_list_attacks_still_works(self, monkeypatch, requests_mock):
        # Regression: additive-only — pre-existing verb must keep behaving.
        self._env(monkeypatch)
        requests_mock.get(f"{self.RT_BASE}/api/v1/testing/attack-techniques", json=[{"id": "st1"}])
        requests_mock.get(f"{self.RT_BASE}/api/v1/mr-jailbreak/techniques", json=[{"id": "mt1"}])
        agents = requests_mock.get(
            f"{self.RT_BASE}/api/v1/mr-jailbreak/agents", json=[{"name": "a1"}]
        )
        r = CliRunner().invoke(cli, ["redteam", "list-attacks"])
        assert r.exit_code == 0, r.output
        assert "st1" in r.output and "mt1" in r.output and "a1" in r.output
        assert agents.last_request.qs == {"page_id": ["1"], "page-size": ["100"]}


class TestCliRedteamBatch2:
    """Batch 2: analytics, recommend, parse-curl, test-connection."""

    RT_BASE = "https://redteam.test"
    JB = "/api/v1/jailbreak"
    BOT = "/api/v1/testing/bot"

    def _env(self, monkeypatch):
        monkeypatch.setenv("DISSEQT_PROJECT_ID", "proj")
        monkeypatch.setenv("DISSEQT_API_KEY", "key")
        monkeypatch.setenv("DISSEQT_BASE_URL", self.RT_BASE)

    # ---- analytics ---------------------------------------------------------
    def test_analytics_default_hits_both(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(f"{self.RT_BASE}{self.JB}/analytics/summary", json={"total_runs": 42})
        requests_mock.get(f"{self.RT_BASE}{self.JB}/prompts-stats", json={"prompts": 100})
        r = CliRunner().invoke(cli, ["redteam", "analytics", "--format", "json"])
        assert r.exit_code == 0, r.output
        # Both endpoints wired through — JSON output contains both keys.
        assert "total_runs" in r.output
        assert "prompts" in r.output

    def test_analytics_summary_only_json(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(f"{self.RT_BASE}{self.JB}/analytics/summary", json={"total_runs": 7})
        r = CliRunner().invoke(cli, ["redteam", "analytics", "--summary", "--format", "json"])
        assert r.exit_code == 0, r.output
        assert "total_runs" in r.output
        # prompts-stats endpoint should NOT be called.
        assert "prompts" not in r.output

    def test_analytics_prompts_only_table(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.get(
            f"{self.RT_BASE}{self.JB}/prompts-stats", json={"prompts": 3, "blocked": 1}
        )
        r = CliRunner().invoke(cli, ["redteam", "analytics", "--prompts-stats"])
        assert r.exit_code == 0, r.output
        assert "prompts" in r.output
        assert "blocked" in r.output

    def test_analytics_mutex_flags_rejected(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(cli, ["redteam", "analytics", "--summary", "--prompts-stats"])
        assert r.exit_code != 0
        assert "at most one" in r.output

    # ---- recommend ---------------------------------------------------------
    def test_recommend_packs_sends_app_name_and_description(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        m = requests_mock.post(
            f"{self.RT_BASE}{self.BOT}/recommend-packs",
            json={"packs": ["owasp-top10", "prompt-injection-101"]},
        )
        r = CliRunner().invoke(
            cli,
            [
                "redteam",
                "recommend",
                "packs",
                "--app-name",
                "bank-bot",
                "--app-description",
                "banking support chatbot",
            ],
        )
        assert r.exit_code == 0, r.output
        assert "owasp-top10" in r.output
        assert m.last_request.json() == {
            "app_name": "bank-bot",
            "app_description": "banking support chatbot",
        }

    def test_recommend_attacks_sends_only_description(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        m = requests_mock.post(f"{self.RT_BASE}{self.BOT}/recommend-attacks", json={"attacks": []})
        r = CliRunner().invoke(
            cli, ["redteam", "recommend", "attacks", "--app-description", "banking support chatbot"]
        )
        assert r.exit_code == 0, r.output
        assert m.last_request.json() == {"app_description": "banking support chatbot"}

    def test_recommend_packs_requires_app_name(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(
            cli, ["redteam", "recommend", "packs", "--app-description", "banking support chatbot"]
        )
        assert r.exit_code == 2
        assert "--app-name" in r.output

    def test_recommend_rejects_short_description(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(
            cli, ["redteam", "recommend", "attacks", "--app-description", "short"]
        )
        assert r.exit_code == 2
        assert "10" in r.output

    def test_recommend_attacks_from_config(self, monkeypatch, requests_mock, tmp_path):
        self._env(monkeypatch)
        requests_mock.post(
            f"{self.RT_BASE}{self.BOT}/recommend-attacks", json={"attacks": ["dan", "aim"]}
        )
        cfg = tmp_path / "body.json"
        cfg.write_text('{"context": "custom", "focus": "roleplay"}')
        r = CliRunner().invoke(cli, ["redteam", "recommend", "attacks", "--config", str(cfg)])
        assert r.exit_code == 0, r.output
        assert "dan" in r.output

    def test_recommend_requires_description_or_config(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(cli, ["redteam", "recommend", "validators"])
        assert r.exit_code != 0
        assert "--app-description" in r.output

    def test_recommend_invalid_kind_rejected(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(cli, ["redteam", "recommend", "nope", "--app-description", "x" * 12])
        assert r.exit_code != 0

    def test_recommend_4xx_surfaces_error(self, monkeypatch, requests_mock):
        # Endpoint reality-check: optimistic route may 404; fail loudly.
        self._env(monkeypatch)
        requests_mock.post(
            f"{self.RT_BASE}{self.BOT}/recommend-packs",
            status_code=404,
            text="not implemented",
        )
        r = CliRunner().invoke(
            cli, ["redteam", "recommend", "packs", "--app-name", "a", "--app-description", "x" * 12]
        )
        assert r.exit_code == 1
        assert "404" in r.output

    # ---- parse-curl --------------------------------------------------------
    def test_parse_curl_from_file(self, monkeypatch, requests_mock, tmp_path):
        self._env(monkeypatch)
        m = requests_mock.post(
            f"{self.RT_BASE}{self.BOT}/parse-curl",
            json={"method": "POST", "url": "https://api.example.com/x"},
        )
        curl_file = tmp_path / "req.txt"
        curl_file.write_text("curl -X POST https://api.example.com/x -d 'a=1'")
        r = CliRunner().invoke(cli, ["redteam", "parse-curl", str(curl_file)])
        assert r.exit_code == 0, r.output
        # api/testing_bot_handlers.go binds {"curl_command"}.
        assert m.last_request.json() == {
            "curl_command": "curl -X POST https://api.example.com/x -d 'a=1'"
        }
        # Full URL substring — CodeQL py/incomplete-url-substring-sanitization
        # false-positives on bare-host membership checks even inside tests.
        assert "https://api.example.com/x" in r.output

    def test_parse_curl_stdin(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        requests_mock.post(f"{self.RT_BASE}{self.BOT}/parse-curl", json={"method": "GET"})
        r = CliRunner().invoke(cli, ["redteam", "parse-curl", "--stdin"], input="curl https://x/y")
        assert r.exit_code == 0, r.output
        assert "GET" in r.output

    def test_parse_curl_empty_rejected(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(cli, ["redteam", "parse-curl", "--stdin"], input="   \n")
        assert r.exit_code != 0
        assert "empty" in r.output

    # ---- test-connection ---------------------------------------------------
    def test_test_connection_flat_body(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        m = requests_mock.post(
            f"{self.RT_BASE}{self.BOT}/test-connection", json={"ok": True, "latency_ms": 42}
        )
        r = CliRunner().invoke(
            cli,
            [
                "redteam",
                "test-connection",
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
                "--api-key",
                "sk-target",
            ],
        )
        assert r.exit_code == 0, r.output
        assert "true" in r.output.lower()
        # api/testing_bot_handlers.go testTargetConnectionDirect: flat body, api_key required.
        assert m.last_request.json() == {
            "endpoint": "",
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-target",
        }

    def test_test_connection_api_key_env(self, monkeypatch, requests_mock):
        self._env(monkeypatch)
        monkeypatch.setenv("TARGET_KEY", "sk-from-env")
        m = requests_mock.post(f"{self.RT_BASE}{self.BOT}/test-connection", json={"ok": True})
        r = CliRunner().invoke(
            cli,
            [
                "redteam",
                "test-connection",
                "--endpoint",
                "https://x/y",
                "--api-key-env",
                "TARGET_KEY",
            ],
        )
        assert r.exit_code == 0, r.output
        assert m.last_request.json()["api_key"] == "sk-from-env"
        assert m.last_request.json()["endpoint"] == "https://x/y"

    def test_test_connection_missing_api_key_errors(self, monkeypatch):
        self._env(monkeypatch)
        r = CliRunner().invoke(cli, ["redteam", "test-connection", "--provider", "openai"])
        assert r.exit_code == 2
        assert "--api-key" in r.output

    # ---- help listing ------------------------------------------------------
    def test_help_lists_new_verbs(self):
        r = CliRunner().invoke(cli, ["redteam", "--help"])
        assert r.exit_code == 0, r.output
        for verb in ("analytics", "recommend", "parse-curl", "test-connection"):
            assert verb in r.output
        # Removed: both needed a Kratos browser session, unreachable with an API key.
        for gone in ("eval-csv", "eval-single-turn"):
            assert gone not in r.output
