"""
Cross-provider batch-job normalization.

Batch inference is not a single request/response — it's a lifecycle:
``create()`` returns immediately with a batch id and pending status, the
job runs asynchronously on the provider's side for minutes to hours, and
the user polls ``retrieve()`` until it reaches a terminal state.

Because the SDK gives us no completion callback, we can't span the whole
lifecycle. Instead we emit **one MODEL_EXEC span per SDK call** tagged with
the same ``agentic.batch.id``, so downstream consumers can ``GROUP BY``
the batch id to reconstruct the timeline.

Providers ship different batch shapes:
  * OpenAI     `Batch` with `request_counts.total/completed/failed`,
                status in {validating, in_progress, finalizing, completed,
                failed, expired, cancelling, cancelled}.
  * Anthropic  `MessageBatch` with `request_counts.processing/succeeded/
                errored`, `processing_status` in {in_progress, canceling, ended}.
  * Mistral    `BatchJob` with `total_requests` / `completed_requests`.

This module folds them into a single canonical dict + a shared
``set_batch_attrs`` writer so per-provider wrappers stay one thin adapter
call plus one attribute-writer call. Same pattern as ``_tool_calls.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from disseqt_agentic_sdk.instrumentation._utils import safe_set
from disseqt_agentic_sdk.semantics import AgenticAttributes, BatchStatus

if TYPE_CHECKING:
    from disseqt_agentic_sdk.span import DisseqtSpan


class CanonicalBatch(TypedDict, total=False):
    """
    Provider-agnostic view of a batch job.

    Fields that a given provider doesn't expose stay absent (TypedDict
    with total=False), and ``set_batch_attrs`` skips missing keys.
    """

    id: str
    status: str  # one of BatchStatus.*
    endpoint: str | None
    request_count: int | None
    completed_count: int | None
    failed_count: int | None
    input_file_id: str | None
    output_file_id: str | None
    error_file_id: str | None
    created_at: int | None
    completed_at: int | None


def _read(obj: Any, name: str) -> Any:
    """Attribute-or-key read tolerant of dicts and Pydantic models."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# OpenAI status strings → canonical BatchStatus.
_OPENAI_STATUS = {
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.RUNNING,
    "finalizing": BatchStatus.RUNNING,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "expired": BatchStatus.EXPIRED,
    "cancelling": BatchStatus.CANCELLED,
    "cancelled": BatchStatus.CANCELLED,
}


def from_openai(batch: Any) -> CanonicalBatch:
    """
    Normalize an ``openai.types.Batch`` (or dict) into ``CanonicalBatch``.

    Missing fields become None; unknown status strings map to
    ``BatchStatus.PENDING`` so downstream never sees an empty status.
    """
    rc = _read(batch, "request_counts")
    raw_status = _read(batch, "status") or ""
    completed_at = (
        _read(batch, "completed_at")
        or _read(batch, "failed_at")
        or _read(batch, "expired_at")
        or _read(batch, "cancelled_at")
    )
    return {
        "id": str(_read(batch, "id") or ""),
        "status": _OPENAI_STATUS.get(raw_status, BatchStatus.PENDING),
        "endpoint": _read(batch, "endpoint"),
        "request_count": _read(rc, "total") if rc is not None else None,
        "completed_count": _read(rc, "completed") if rc is not None else None,
        "failed_count": _read(rc, "failed") if rc is not None else None,
        "input_file_id": _read(batch, "input_file_id"),
        "output_file_id": _read(batch, "output_file_id"),
        "error_file_id": _read(batch, "error_file_id"),
        "created_at": _read(batch, "created_at"),
        "completed_at": completed_at,
    }


def set_batch_attrs(span: DisseqtSpan, batch: CanonicalBatch, operation: str) -> None:
    """
    Write canonical batch attributes onto ``span``.

    ``operation`` is one of ``AgenticOperation.BATCH_CREATE`` /
    ``BATCH_RETRIEVE`` / ``BATCH_CANCEL``. Every provider's wrapper calls
    this with the same signature; adding a new provider is (1) write a
    ``from_<provider>`` adapter, (2) write a thin wrapper that calls this.
    """
    span.set_operation(operation)
    safe_set(span, AgenticAttributes.OPERATION_NAME, operation)
    safe_set(span, AgenticAttributes.BATCH_ID, batch.get("id") or "")
    safe_set(span, AgenticAttributes.BATCH_STATUS, batch.get("status"))
    safe_set(span, AgenticAttributes.BATCH_ENDPOINT, batch.get("endpoint"))
    safe_set(span, AgenticAttributes.BATCH_REQUEST_COUNT, batch.get("request_count"))
    safe_set(span, AgenticAttributes.BATCH_COMPLETED_COUNT, batch.get("completed_count"))
    safe_set(span, AgenticAttributes.BATCH_FAILED_COUNT, batch.get("failed_count"))
    safe_set(span, AgenticAttributes.BATCH_INPUT_FILE_ID, batch.get("input_file_id"))
    safe_set(span, AgenticAttributes.BATCH_OUTPUT_FILE_ID, batch.get("output_file_id"))
    safe_set(span, AgenticAttributes.BATCH_ERROR_FILE_ID, batch.get("error_file_id"))
    safe_set(span, AgenticAttributes.BATCH_CREATED_AT, batch.get("created_at"))
    safe_set(span, AgenticAttributes.BATCH_COMPLETED_AT, batch.get("completed_at"))
