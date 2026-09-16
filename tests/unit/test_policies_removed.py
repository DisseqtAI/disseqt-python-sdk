"""Regression guards against re-introducing the removed server-side
policy-evaluate path.

If any of these tests turn green in a state where the removed kwargs or
methods have been added back, treat the readd as a mistake — the
transport it needed still does not exist on the backend.
"""

from __future__ import annotations

import pytest

from disseqt_sdk import Client
from disseqt_sdk.models.input_validation import InputValidationRequest


def _base_client() -> Client:
    return Client(project_id="p", api_key="k", application_name="regression-guard")


class TestClientCtorRejectsRemovedKwargs:
    def test_policies_kwarg_rejected(self) -> None:
        with pytest.raises(TypeError, match="policies"):
            Client(  # type: ignore[call-arg]
                project_id="p",
                api_key="k",
                application_name="x",
                policies=["b1f8b1f8-1111-4111-8111-111111111111"],
            )

    def test_realtime_policy_base_url_kwarg_rejected(self) -> None:
        with pytest.raises(TypeError, match="realtime_policy_base_url"):
            Client(  # type: ignore[call-arg]
                project_id="p",
                api_key="k",
                realtime_policy_base_url="http://localhost:9010",
            )


class TestValidateRejectsPoliciesKwarg:
    def test_validate_rejects_policies_kwarg(self) -> None:
        c = _base_client()
        with pytest.raises(TypeError, match="policies"):
            c.validate(  # type: ignore[call-arg]
                InputValidationRequest(prompt="hi"),
                policies=["b1f8b1f8-1111-4111-8111-111111111111"],
            )

    def test_validate_sync_rejects_policies_kwarg(self) -> None:
        c = _base_client()
        with pytest.raises(TypeError, match="policies"):
            c.validate_sync(  # type: ignore[call-arg]
                InputValidationRequest(prompt="hi"),
                policies=["b1f8b1f8-1111-4111-8111-111111111111"],
            )


class TestRemovedPublicSurfaceStaysRemoved:
    def test_guardrails_symbol_not_exported(self) -> None:
        import disseqt_sdk

        assert "Guardrails" not in disseqt_sdk.__all__
        assert not hasattr(disseqt_sdk, "Guardrails")

    def test_baseguard_symbol_not_exported(self) -> None:
        import disseqt_sdk

        assert "BaseGuard" not in disseqt_sdk.__all__
        assert not hasattr(disseqt_sdk, "BaseGuard")

    def test_policy_module_not_importable(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            __import__("disseqt_sdk.policy")

    def test_blocked_error_not_exported(self) -> None:
        import disseqt_sdk

        # SDKVersionBlockedError is a different, still-supported class.
        assert "BlockedError" not in disseqt_sdk.__all__
        assert not hasattr(disseqt_sdk, "BlockedError")
