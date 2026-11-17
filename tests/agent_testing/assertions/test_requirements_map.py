"""Every functional (FR) and non-functional (NFR) requirement of plan 03 maps to real tests.

``MAP`` names tests by function name; ``test_every_mapped_test_exists`` fails when a test is
renamed or deleted, so the map cannot rot silently.
"""

import ast
import sys
from pathlib import Path

from disseqt_agent_testing.assertions import list_metrics

TESTS = Path(__file__).resolve().parents[1]

MAP: dict[str, tuple[str, ...]] = {
    "FR-01 accepts RunResult or AgentRun": ("test_a_bare_agent_run_is_treated_as_finished",),
    "FR-02 non-finished run fails with the harness line": (
        "test_a_crashed_run_fails_with_the_exact_first_line_and_still_scores_vetoes",
        "test_t66_a_timed_out_run_is_not_a_pass",
    ),
    "FR-03 violations": ("test_violations_fail_a_finished_run_and_checks_are_still_scored",),
    "FR-04 unfinished runs skip all but forbidden/loops": (
        "test_a_crashed_run_fails_with_the_exact_first_line_and_still_scores_vetoes",
        "test_loops_still_score_a_crashed_run",
    ),
    "FR-05 calls are tool/subagent steps in seq order": ("test_model_steps_never_take_part",),
    "FR-06 planner": (
        "test_selection_is_not_enabled_by_default",
        "test_g03_thresholds_on_yes_no_checks_and_unknown_keys_are_config_errors",
    ),
    "FR-07 skipped/error carry no value": ("test_metric_result_validation_and_round_trip",),
    "FR-08 metric exception is error(metric_error)": (
        "test_g07_a_metric_that_raises_is_an_error_and_the_rest_stay_intact",
    ),
    "FR-09 plain values wrap to Eq": ("test_plain_values_wrap_to_eq",),
    "FR-10 matcher semantics": (
        "test_m01_int_is_not_bool",
        "test_m10_m11_normalized_ignores_case_and_punctuation_and_is_lossy_by_design",
        "test_m18_dict_match_modes",
    ),
    "FR-11 all mismatches reported": (
        "test_missing_required_and_all_mismatches_are_reported_sorted",
    ),
    "FR-12 Ref at scoring time": (
        "test_a08_ref_to_an_earlier_result_is_checked_against_the_bound_call",
        "test_refs_are_rebound_to_the_calls_the_assignment_actually_chose",
    ),
    "FR-13 predicates by name": (
        "test_m28_predicates_by_name",
        "test_m29_unknown_predicate_names_are_config_errors_before_scoring",
    ),
    "FR-14 Semantic skip path": (
        "test_m30_semantic_without_a_judge_is_skipped_and_the_outcome_incomplete",
    ),
    "FR-15 slots": ("test_slots_are_in_declared_order_with_required_first",),
    "FR-16 one lexicographic assignment": (
        "test_a01_the_greedy_trap_passes_under_the_optimum",
        "test_a_required_slot_wins_a_call_over_an_optional_slot_declared_earlier",
    ),
    "FR-17 backend equality": (
        "test_a11_vendored_solver_equals_brute_force",
        "test_a12_python_and_scipy_backends_agree_on_totals_and_assignments",
    ),
    "FR-18 decision_match": (
        "test_d14_decision_by_turn_evaluates_each_turn",
        "test_d11_empty_output_fails_any_expectation",
    ),
    "FR-19 tool_selection": (
        "test_selection_all_required_matched",
        "test_a05_explicit_count_cap_is_a_veto",
    ),
    "FR-20 tool_args": (
        "test_args_diff_reports_every_wrong_leaf_with_a_path",
        "test_args_no_match_is_skipped_not_a_perfect_score",
    ),
    "FR-21 order": (
        "test_o01_in_order_ignores_extra_calls",
        "test_o07_diamond_dag_accepts_both_middle_orders",
    ),
    "FR-22 forbidden_calls": (
        "test_f01_a_forbidden_call_is_a_veto_regardless_of_thresholds",
        "test_f02_unless_after_exempts_a_prior_approval",
    ),
    "FR-23 budgets": ("test_b01_inclusive_bounds", "test_b03_b04_token_budget_lower_bound"),
    "FR-24 answer": (
        "test_c06_answer_contains_strings_and_matchers",
        "test_c07_no_answer_fails_with_no_output",
    ),
    "FR-25 classifier precedence": ("test_classifier_labels", "test_d12_a_declared_decision_wins"),
    "FR-26 refusal and question detectors": (
        "test_agreement_is_at_least_95_percent_and_every_miss_is_listed",
    ),
    "FR-27 judge escalation": ("test_d13_a_judge_resolves_a_low_confidence_label",),
    "FR-28 clarification": ("test_c01_gates_in_order", "test_c04_max_questions"),
    "FR-29 refusal": ("test_refusal_gates", "test_refusal_alternative_is_judge_only_with_polarity"),
    "FR-30 scope": ("test_c05_scope_confusion_cells",),
    "FR-31 intent": ("test_c08_intent_from_call_arguments_results_answer_and_route",),
    "FR-32 routing": (
        "test_r01_routing_by_sub_agent_in_every_mode",
        "test_r04_transfer_to_agent_with_an_enum_and_an_invalid_target",
    ),
    "FR-33 loops and plan": (
        "test_l01_too_many_identical_calls",
        "test_p01_constraints_micro_and_macro",
    ),
    "FR-34 gating": ("test_g06_failed_plus_incomplete_is_failed",),
    "FR-35 thresholds": (
        "test_g02_a_threshold_lets_a_partly_wrong_call_pass",
        "test_g04_a_veto_cannot_be_cleared_by_a_threshold",
    ),
    "FR-36 strict": ("test_g05_strict_fails_a_score_below_one",),
    "FR-37 report dict is the source": (
        "test_x03_the_report_round_trips_and_ignores_unknown_keys",
    ),
    "FR-38 render rules": (
        "test_x04_line_budget_truncation_and_ci_override",
        "test_t49_i02_the_wrong_order_id_fails_with_one_precise_diff",
    ),
    "FR-39 assert_agent raises": (
        "test_assert_agent_returns_the_report_or_raises",
        "test_incomplete_raises_the_subclass_with_the_could_not_verify_line",
    ),
    "FR-40 no truth value": ("test_x01_a_report_has_no_truth_value",),
    "FR-41 picklable error": ("test_x02_errors_survive_pickling",),
    "FR-42 registry and entry points": (
        "test_duplicate_metric_names_raise_unless_overridden",
        "test_entry_point_metrics_are_loaded_once_and_planned",
    ),
    "FR-43 judge and repeat schemas": (
        "test_e07_unknown_judge_kind_passes_through_request_judge",
        "test_e08_new_dataclasses_follow_the_layout_rules",
    ),
    "FR-44 round trip": (
        "test_x03_the_report_round_trips_and_ignores_unknown_keys",
        "test_metric_result_validation_and_round_trip",
    ),
    "NFR-01 byte-identical reports": (
        "test_x06_16_threads_by_50_evaluations_are_identical",
        "test_a_golden_report_is_stable_and_loads_on_later_versions",
    ),
    "NFR-02 offline": (
        "test_outbound_connections_fail",
        "test_a_full_harness_run_needs_no_network",
    ),
    "NFR-03 stdlib only": ("test_runtime_imports_are_stdlib_only",),
    "NFR-04 performance": ("test_i09_nfr04_performance_stays_within_guard_rails",),
    "NFR-05 re-entrant": (
        "test_x06_16_threads_by_50_evaluations_are_identical",
        "test_memo_computes_once_and_caches_exceptions",
    ),
    "NFR-06 Python 3.10-3.13": (
        "test_x06_16_threads_by_50_evaluations_are_identical",
    ),  # CI matrix runs the suite
    "NFR-07 no pass from absent evidence": (
        "test_b02_llm_call_budget_needs_captured_model_steps",
        "test_g05_passed_true_is_reachable_only_from_a_gating_positive_ok_verdict",
    ),
    "NFR-08 additive schema": ("test_the_json_schema_matches_its_golden_file",),
    "NFR-09 failing checks render a code": (
        "test_every_failing_check_renders_a_registered_code",
        "test_every_reason_code_literal_in_the_source_is_registered",
    ),
    "NFR-10 bounded output": ("test_x04_line_budget_truncation_and_ci_override",),
    "NFR-11 no input mutation": ("test_inputs_are_never_mutated",),
    "NFR-12 metric versions": ("test_metric_versions_are_pinned_at_one",),
}


def test_every_mapped_test_exists() -> None:
    defined: set[str] = set()
    for path in TESTS.rglob("test_*.py"):
        tree = ast.parse(path.read_text())
        defined.update(
            n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    missing = {req: [t for t in names if t not in defined] for req, names in MAP.items()}
    assert not {k: v for k, v in missing.items() if v}, missing


def test_every_requirement_is_mapped() -> None:
    wanted = {f"FR-{n:02d}" for n in range(1, 45)} | {f"NFR-{n:02d}" for n in range(1, 13)}
    have = {key.split(" ", 1)[0] for key in MAP}
    assert wanted == have, sorted(wanted ^ have)


def test_metric_versions_are_pinned_at_one() -> None:
    """NFR-12: any change that can alter a value must bump the version, which fails this test
    and forces the author to update the goldens and the changelog on purpose."""
    versions = {m.name: m.version for m in list_metrics()}
    assert versions == {
        "answer": "1", "answer_quality": "1", "budgets": "1", "clarification": "1",
        "decision_match": "1", "forbidden_calls": "1", "groundedness": "1", "intent": "1",
        "loops": "1", "order": "1", "plan": "2", "refusal": "1", "routing": "1", "scope": "1", "tool_args": "1", "tool_selection": "1",
    }  # fmt: skip


def test_runtime_imports_are_stdlib_only() -> None:
    """NFR-03: the assertions package needs nothing outside the stdlib, the models/harness
    packages and ``typing_extensions`` (SciPy and NumPy are imported lazily, by name)."""
    root = Path(__file__).resolve().parents[3] / "src" / "disseqt_agent_testing" / "assertions"
    stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else None
    allowed_extra = {"typing_extensions", "disseqt_agent_testing"}
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                ok = name in allowed_extra or (stdlib is None or name in stdlib)
                if not ok:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders
