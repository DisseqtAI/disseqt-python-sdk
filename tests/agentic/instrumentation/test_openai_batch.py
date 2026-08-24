"""
End-to-end tests for the OpenAI Batch API instrumentor.

We patch the underlying HTTP seam so tests never hit OpenAI. Each SDK
call (``create``, ``retrieve``, ``cancel``) must emit its own MODEL_EXEC
span, all tagged with the same ``agentic.batch.id`` so a downstream
GROUP BY can reconstruct the lifecycle.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from openai import OpenAI  # noqa: E402
from openai.types import Batch  # noqa: E402
from openai.types.batch_request_counts import BatchRequestCounts  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import (  # noqa: E402
    AgenticAttributes,
    AgenticOperation,
    BatchStatus,
)
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_batch(status: str = "validating", **overrides) -> Batch:
    base = {
        "id": "batch_abc123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "input_file_id": "file-in",
        "status": status,
        "created_at": 1_700_000_000,
        "request_counts": BatchRequestCounts(total=10, completed=0, failed=0),
    }
    base.update(overrides)
    return Batch(**base)


class TestOpenAIBatchCreate:
    def test_records_create_span(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            fake = _fake_batch(status="validating")
            with patch.object(client.batches, "_post", return_value=fake, create=True):
                result = client.batches.create(
                    input_file_id="file-in",
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
        finally:
            uninstrument("openai")

        assert result.id == "batch_abc123"

        span = find_span(recording_client, "openai.batches.create")
        attrs = json.loads(span.attributes_json)

        assert attrs[AgenticAttributes.OPERATION_NAME] == AgenticOperation.BATCH_CREATE
        assert attrs[AgenticAttributes.BATCH_ID] == "batch_abc123"
        assert attrs[AgenticAttributes.BATCH_STATUS] == BatchStatus.PENDING  # normalized
        assert attrs[AgenticAttributes.BATCH_ENDPOINT] == "/v1/chat/completions"
        assert attrs[AgenticAttributes.BATCH_INPUT_FILE_ID] == "file-in"
        assert attrs[AgenticAttributes.BATCH_REQUEST_COUNT] == 10
        assert attrs[AgenticAttributes.BATCH_COMPLETED_COUNT] == 0
        assert attrs[AgenticAttributes.BATCH_CREATED_AT] == 1_700_000_000
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "openai"


class TestOpenAIBatchRetrieve:
    def test_maps_terminal_statuses_and_progress(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            in_progress = _fake_batch(
                status="in_progress",
                request_counts=BatchRequestCounts(total=10, completed=4, failed=1),
            )
            completed = _fake_batch(
                status="completed",
                request_counts=BatchRequestCounts(total=10, completed=9, failed=1),
                output_file_id="file-out",
                error_file_id="file-err",
                completed_at=1_700_000_500,
            )

            with patch.object(
                client.batches, "_get", side_effect=[in_progress, completed], create=True
            ):
                client.batches.retrieve("batch_abc123")
                client.batches.retrieve("batch_abc123")
        finally:
            uninstrument("openai")

        # Two spans, same batch id, different statuses.
        spans = [
            s
            for s in recording_client.buffer.spans  # type: ignore[attr-defined]
            if s.name == "openai.batches.retrieve"
        ]
        assert len(spans) == 2

        first = json.loads(spans[0].attributes_json)
        second = json.loads(spans[1].attributes_json)

        assert first[AgenticAttributes.BATCH_ID] == "batch_abc123"
        assert second[AgenticAttributes.BATCH_ID] == "batch_abc123"

        # in_progress → running
        assert first[AgenticAttributes.BATCH_STATUS] == BatchStatus.RUNNING
        assert first[AgenticAttributes.BATCH_COMPLETED_COUNT] == 4
        assert first[AgenticAttributes.BATCH_FAILED_COUNT] == 1

        # completed → completed, plus output/error files populated
        assert second[AgenticAttributes.BATCH_STATUS] == BatchStatus.COMPLETED
        assert second[AgenticAttributes.BATCH_OUTPUT_FILE_ID] == "file-out"
        assert second[AgenticAttributes.BATCH_ERROR_FILE_ID] == "file-err"
        assert second[AgenticAttributes.BATCH_COMPLETED_AT] == 1_700_000_500
        assert second[AgenticAttributes.OPERATION_NAME] == AgenticOperation.BATCH_RETRIEVE


class TestOpenAIBatchCancel:
    def test_records_cancel_span(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            cancelled = _fake_batch(status="cancelled", cancelled_at=1_700_000_200)
            # cancel goes through _post on newer openai clients.
            with patch.object(client.batches, "_post", return_value=cancelled, create=True):
                client.batches.cancel("batch_abc123")
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.batches.cancel")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.OPERATION_NAME] == AgenticOperation.BATCH_CANCEL
        assert attrs[AgenticAttributes.BATCH_STATUS] == BatchStatus.CANCELLED
        # completed_at falls back to cancelled_at when cancelled.
        assert attrs[AgenticAttributes.BATCH_COMPLETED_AT] == 1_700_000_200


class TestOpenAIBatchLifecycleLinkage:
    def test_all_spans_share_batch_id(self, recording_client):
        # Simulates the real usage pattern: create → poll → poll → complete.
        # A GROUP BY agentic.batch.id downstream should reconstruct the timeline.
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            created = _fake_batch(status="validating")
            running = _fake_batch(status="in_progress")
            done = _fake_batch(
                status="completed",
                output_file_id="file-out",
                completed_at=1_700_000_600,
            )
            with patch.object(client.batches, "_post", return_value=created, create=True):
                client.batches.create(
                    input_file_id="file-in",
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
            with patch.object(client.batches, "_get", side_effect=[running, done], create=True):
                client.batches.retrieve("batch_abc123")
                client.batches.retrieve("batch_abc123")
        finally:
            uninstrument("openai")

        ids = {
            json.loads(s.attributes_json)[AgenticAttributes.BATCH_ID]
            for s in recording_client.buffer.spans  # type: ignore[attr-defined]
            if s.name.startswith("openai.batches.")
        }
        assert ids == {"batch_abc123"}
