"""Response handling for Disseqt SDK."""

from __future__ import annotations

from typing import Any


def normalize_server_payload(server_response: dict[str, Any]) -> dict[str, Any]:
    """Normalize server response to the fixed SDK response schema.

    Args:
        server_response: Raw response from the server

    Returns:
        Normalized response with fixed schema and dynamic 'others' bag
    """
    # Handle standard validator response format
    if "score" in server_response and "category" in server_response:
        # New API format: {score, label, passed, explanation, details, category, request_id, credits_info}
        details = server_response.get("details") or {}

        # Build normalized data section
        normalized_data: dict[str, Any] = {
            "metric_name": details.get("metric_name", "unknown"),
            "actual_value": server_response.get("score", 0.0),
            "actual_value_type": "float",
            "metric_labels": [server_response.get("label", "unknown")],
            "threshold": ["Pass" if server_response.get("passed", False) else "Fail"],
            "threshold_score": details.get("threshold_score", 0.0),
        }

        # Put additional fields in 'others'
        others: dict[str, Any] = {}
        for key, value in server_response.items():
            if key not in {"score", "label", "passed"}:
                others[key] = value

        # Add details fields to others (except those we've already mapped)
        if details:  # Only if details is not None/empty
            for key, value in details.items():
                if key not in {"metric_name", "threshold_score"}:
                    others[f"details_{key}"] = value

        normalized_data["others"] = others

        # Status is always success if we got a response
        normalized_status = {"code": "200", "message": "Success"}

        return {
            "data": normalized_data,
            "status": normalized_status,
        }

    # Handle legacy format for backward compatibility
    else:
        # Extract known fields from server response
        data_section = server_response.get("data", {})

        # Build normalized data section
        normalized_data: dict[str, Any] = {}
        others: dict[str, Any] = {}

        # Known fields that should be preserved as-is
        known_fields = {
            "metric_name",
            "actual_value",
            "actual_value_type",
            "metric_labels",
            "threshold",
            "threshold_score",
        }

        # Extract known fields
        for field in known_fields:
            if field in data_section:
                normalized_data[field] = data_section[field]

        # Put unknown fields in 'others'
        for key, value in data_section.items():
            if key not in known_fields:
                others[key] = value

        # Add others bag
        normalized_data["others"] = others

        # Handle status section
        status_section = server_response.get("status", {})
        normalized_status = {
            "code": status_section.get("code", "200"),
            "message": status_section.get("message", "Success"),
        }

        return {
            "data": normalized_data,
            "status": normalized_status,
        }


def validate_actual_value_type(value: Any) -> str:
    """Validate and determine the actual_value_type.

    Args:
        value: The actual value to check

    Returns:
        String representation of the value type

    Raises:
        ValueError: If value type is not supported
    """
    # Note: bool is a subclass of int in Python, so check bool first
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, str):
        return "string"
    else:
        raise ValueError(f"Unsupported actual_value type: {type(value)}")
