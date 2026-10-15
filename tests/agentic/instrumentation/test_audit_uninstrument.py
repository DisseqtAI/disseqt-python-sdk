"""Regression tests for M19: uninstrument() with buried wrappers + fail-soft wrappers."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest
import wrapt

from disseqt_agentic_sdk.instrumentation import _utils as inst_utils
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor


# ---------------------------------------------------------------------------
# M19: uninstrument with a buried wrapper + fail-soft wrappers
# ---------------------------------------------------------------------------
class _FakeProvider:
    def create(self, **kwargs):
        return "sync-result"

    async def acreate(self, **kwargs):
        return "async-result"


@pytest.fixture
def fake_provider_module(monkeypatch):
    mod = types.ModuleType("fake_audit_provider")
    mod.Comp = type("Comp", (), {"create": _FakeProvider.create, "acreate": _FakeProvider.acreate})
    monkeypatch.setitem(sys.modules, "fake_audit_provider", mod)
    return mod


def _make_instrumentor():
    from disseqt_agentic_sdk.instrumentation.openai.patch import (
        async_chat_completions_create,
        chat_completions_create,
    )

    class FakeInstrumentor(DisseqtInstrumentor):
        package_name = "openai"  # installed, so the version gate passes

        def _instrument(self) -> None:
            self._wrap("fake_audit_provider", "Comp.create", chat_completions_create(self))
            self._wrap("fake_audit_provider", "Comp.acreate", async_chat_completions_create(self))

    return FakeInstrumentor()


class TestUninstrumentBuried:
    def test_buried_wrapper_keeps_client_and_calls_do_not_raise(
        self, recording_client, fake_provider_module
    ):
        inst = _make_instrumentor()
        assert inst.instrument(recording_client) is True

        def passthrough(wrapped, instance, args, kwargs):
            return wrapped(*args, **kwargs)

        async def apassthrough(wrapped, instance, args, kwargs):
            return await wrapped(*args, **kwargs)

        # Another library wraps on top of ours.
        wrapt.wrap_function_wrapper("fake_audit_provider", "Comp.create", passthrough)
        wrapt.wrap_function_wrapper("fake_audit_provider", "Comp.acreate", apassthrough)

        inst.uninstrument()

        # Our layers could not be unwound, so the client must NOT be cleared.
        assert inst._client is recording_client
        assert inst._patched  # still tracked
        comp = fake_provider_module.Comp()
        assert comp.create(model="m") == "sync-result"
        assert asyncio.run(comp.acreate(model="m")) == "async-result"

    def test_clean_uninstrument_clears_client(self, recording_client, fake_provider_module):
        inst = _make_instrumentor()
        assert inst.instrument(recording_client) is True
        inst.uninstrument()
        assert inst._client is None
        assert not inst._patched
        assert fake_provider_module.Comp().create() == "sync-result"

    def test_wrappers_fail_soft_when_client_unresolvable(
        self, recording_client, fake_provider_module
    ):
        inst = _make_instrumentor()
        assert inst.instrument(recording_client) is True
        inst._client = None  # instrumentor.client now raises RuntimeError
        comp = fake_provider_module.Comp()
        try:
            assert comp.create(model="m", stream=True) == "sync-result"
            assert asyncio.run(comp.acreate(model="m", stream=True)) == "async-result"
        finally:
            inst._client = recording_client
            inst.uninstrument()

    def test_try_open_llm_span_returns_none_on_failure(self):
        bad = SimpleNamespace()  # no .client attribute
        assert inst_utils.try_open_llm_span(bad, "x") is None

    def test_all_provider_wrappers_use_fail_soft_helper(self):
        import pathlib

        root = pathlib.Path(inst_utils.__file__).parent
        offenders = [
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if p.name != "_utils.py"
            and "open_llm_span(\n" in p.read_text()
            or (
                p.name != "_utils.py"
                and "open_llm_span(instrumentor.client" in p.read_text()
                and "try_open_llm_span(instrumentor" not in p.read_text()
            )
        ]
        assert offenders == []
