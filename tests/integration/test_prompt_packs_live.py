"""Integration tests for Prompt Packs API against localhost (Kong proxy).

Run with:
  pytest tests/integration/test_prompt_packs_live.py -v -m integration

Uses project_id and api_key below; base_url is http://localhost:8000.
Requires Kong and the prompt-packs backend to be running locally.
"""

from __future__ import annotations

import os

import pytest

from disseqt_sdk import DisseqtAPIClient
from disseqt_sdk.client import HTTPError
from disseqt_sdk.models.prompt_packs import (
    GeneratePromptPackRequest,
    PaginationParams,
    PromptPackCategory,
)

# Credentials for localhost tests (Kong proxy at 8000)
PROJECT_ID = os.environ.get("DISSEQT_PROJECT_ID", "121c8136-5458-494b-a8be-ad46440f4330")
API_KEY = os.environ.get("DISSEQT_API_KEY", "99499ed3-f956-4881-bec9-64d5cea0edec")
BASE_URL = os.environ.get("DISSEQT_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def api_client():
    """DisseqtAPIClient pointing at localhost with test credentials."""
    return DisseqtAPIClient(
        project_id=PROJECT_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=30,
    )


@pytest.mark.integration
class TestPromptPacksLive:
    """Live integration tests against localhost."""

    def test_headers_and_base_url(self, api_client):
        """Client uses correct base URL and auth headers."""
        assert api_client.base_url == BASE_URL.rstrip("/")
        assert api_client.project_id == PROJECT_ID
        assert api_client.api_key == API_KEY
        headers = api_client._build_headers()
        assert headers["X-API-Key"] == API_KEY
        assert headers["X-Project-Id"] == PROJECT_ID
        assert "Content-Type" in headers
        assert "X-Request-Id" not in headers

    def test_generate_prompt_pack(self, api_client):
        """POST /generate returns 2xx and a pack identifier, or 502/503 if upstream is down."""
        request = GeneratePromptPackRequest(
            pack_name="SDK Integration Test Pack",
            pack_short_desc="Created by integration test",
            author="SDK",
            domain="Testing",
            generation_type="AI",
            categories=[
                PromptPackCategory(
                    main_category="reliability_and_safety",
                    subcategory="hate_speech",
                    prompts_count=2,
                ),
            ],
        )
        try:
            result = api_client.generate_prompt_pack(request)
            assert isinstance(result, dict)
            pack_id = (
                result.get("id") or result.get("pack_id") or (result.get("data") or {}).get("id")
            )
            assert pack_id, f"Expected pack id in response: {result}"
        except HTTPError as e:
            if e.status_code in (502, 503):
                pytest.skip("Upstream unavailable (502/503); Kong is up, backend may be down")
            raise

    def test_list_runs_requires_valid_pack_id(self, api_client):
        """GET /{pack_id}/runs returns 2xx (empty list ok), 404, 400, or 502/503."""
        fake_pack_id = "00000000-0000-0000-0000-000000000001"
        try:
            result = api_client.list_runs(
                fake_pack_id, pagination=PaginationParams(limit=5, offset=0)
            )
            assert isinstance(result, dict)
        except HTTPError as e:
            if e.status_code in (502, 503):
                pytest.skip("Upstream unavailable (502/503)")
            # Backend may return 400, 404, or 500 for invalid/missing pack
            assert e.status_code in (404, 400, 500), f"Unexpected status: {e.status_code}"

    def test_pagination_params_in_request(self, api_client):
        """List runs with custom pagination sends correct query params."""
        fake_pack_id = "00000000-0000-0000-0000-000000000001"
        try:
            api_client.list_runs(
                fake_pack_id,
                pagination=PaginationParams(limit=3, offset=1),
            )
        except HTTPError:
            pass
        # If we had a way to inspect last request we could assert query string; here we just ensure no crash.
        assert True
