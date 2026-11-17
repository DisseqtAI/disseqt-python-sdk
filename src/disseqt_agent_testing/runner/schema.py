"""The JSON Schema (draft 2020-12) of ``SuiteResult.to_dict()``. Changes inside schema version 1
are additive: new optional properties only."""

from typing import Any

from .models import CASE_STATUSES, NOT_RUN_REASONS, REPEAT_STATUSES, SCHEMA_VERSION


def suite_result_schema() -> dict[str, Any]:
    nullable_str = {"type": ["string", "null"]}
    nullable_num = {"type": ["number", "null"]}
    repeat = {
        "type": "object",
        "required": ["index", "seed", "status", "failure_codes"],
        "properties": {
            "index": {"type": "integer", "minimum": 0},
            "seed": {"type": "integer"},
            "status": {"enum": list(REPEAT_STATUSES)},
            "not_run_reason": {"enum": [*NOT_RUN_REASONS, None]},
            "run_status": nullable_str,
            "decision": nullable_str,
            "failure_codes": {"type": "array", "items": {"type": "string"}},
            "trace_id": nullable_str,
            "duration_ms": nullable_num,
            "judge_calls": {"type": "integer"},
            "judge_tokens": {"type": "integer"},
            "judge_cost_usd": {"type": "number"},
            "judge_cost_unknown": {"type": "integer"},
            "judge_error": {"type": "boolean"},
            "report": {"type": ["object", "null"]},
        },
    }
    case = {
        "type": "object",
        "required": [
            "case_id",
            "status",
            "source",
            "tags",
            "n_planned",
            "k",
            "threshold",
            "repeats",
            "case_digest",
        ],
        "properties": {
            "case_id": {"type": "string"},
            "status": {"enum": list(CASE_STATUSES)},
            "source": {
                "type": "object",
                "properties": {
                    "path": nullable_str,
                    "line": {"type": ["integer", "null"]},
                    "index": {"type": "integer"},
                    "nodeid": nullable_str,
                },
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "reason_code": nullable_str,
            "reason": {"type": "string"},
            "error": {"type": ["object", "null"]},
            "n_planned": {"type": "integer", "minimum": 1},
            "k": {"type": "integer", "minimum": 1},
            "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "repeat_report": {"type": ["object", "null"]},
            "repeats": {"type": "array", "items": {"$ref": "#/$defs/repeat"}},
            "stopped_early": {"type": "boolean"},
            "comparable": {"type": "boolean"},
            "case_digest": {"type": "string"},
            "duration_ms": nullable_num,
            "extra": {"type": "object"},
        },
    }
    totals = [
        "cases",
        "passed",
        "failed",
        "incomplete",
        "skipped",
        "error",
        "repeats_run",
        "repeats_not_run",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SuiteResult",
        "type": "object",
        "required": [
            "schema_version", "suite_id", "run_label", "suite_digest", "created_at", "seed", "config",
            "cases", "totals", "aggregate", "judge_usage", "collection_errors", "interrupted", "exit_code",
        ],  # fmt: skip
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "suite_id": {"type": "string"},
            "run_label": {"type": "string"},
            "suite_digest": {"type": "string"},
            "created_at": {"type": "string"},
            "duration_ms": nullable_num,
            "seed": {"type": "integer"},
            "config": {"type": "object"},
            "environment": {"type": "object"},
            "cases": {"type": "array", "items": {"$ref": "#/$defs/case"}},
            "totals": {
                "type": "object",
                "required": totals,
                "properties": {k: {"type": "integer", "minimum": 0} for k in totals},
            },
            "aggregate": {
                "type": "object",
                "properties": {
                    "mean_pass_at_1": nullable_num,
                    "mean_pass_hat_k": nullable_num,
                    "flaky_cases": {"type": "array", "items": {"type": "string"}},
                    "consistently_failing": {"type": "array", "items": {"type": "string"}},
                    "pass_hat_k_curve": {"type": "array"},
                },
            },
            "judge_usage": {"type": "object"},
            "collection_errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["message"],
                    "properties": {
                        "path": nullable_str,
                        "line": {"type": ["integer", "null"]},
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            },
            "interrupted": {"type": "boolean"},
            "exit_code": {"type": "integer"},
            "extra": {"type": "object"},
        },
        "$defs": {"case": case, "repeat": repeat},
    }
