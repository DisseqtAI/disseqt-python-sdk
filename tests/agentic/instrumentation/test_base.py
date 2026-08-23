"""Tests for the DisseqtInstrumentor base class + auto module."""

from __future__ import annotations

import inspect
import threading

import openai
import pytest
import wrapt

from disseqt_agentic_sdk.instrumentation import (
    AVAILABLE_INSTRUMENTORS,
    InstrumentationError,
    get_instrumented_client,
    instrument,
    instrument_all,
    uninstrument,
    uninstrument_all,
)
from disseqt_agentic_sdk.instrumentation import auto as auto_module
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor


class BrokenInstrumentor(DisseqtInstrumentor):
    """Loadable and installed, but _instrument() always fails — used by
    strict-mode tests. `openai` is chosen as package_name because it's a
    real installed package, so we get past the version gate."""

    package_name = "openai"

    def _instrument(self) -> None:
        raise RuntimeError("boom")


class MissingPackageInstrumentor(DisseqtInstrumentor):
    """Loadable but its package_name isn't installed — used to check that
    strict mode treats package_missing as a skip, not a failure."""

    package_name = "definitely-not-a-real-package-9x8x7"

    def _instrument(self) -> None:  # pragma: no cover - shouldn't run
        raise AssertionError("must not be called")


class TestBase:
    def test_version_gate_skips_when_package_missing(self, recording_client):
        class MissingPackage(DisseqtInstrumentor):
            package_name = "definitely-not-a-real-package-9x8x7"

            def _instrument(self) -> None:  # pragma: no cover - shouldn't run
                raise AssertionError("must not be called")

        assert MissingPackage().instrument(recording_client) is False

    def test_double_instrument_is_noop(self, recording_client):
        # openai is installed in the test env
        assert instrument("openai", recording_client) is True
        assert instrument("openai", recording_client) is False
        uninstrument("openai")

    def test_unknown_provider_returns_false(self, recording_client):
        assert instrument("does-not-exist", recording_client) is False

    def test_instrument_all_lists_installed(self, recording_client):
        installed = instrument_all(recording_client)
        assert "openai" in installed
        assert set(installed).issubset(set(AVAILABLE_INSTRUMENTORS))
        uninstrument_all()

    def test_get_instrumented_client_roundtrip(self, recording_client):
        assert get_instrumented_client("openai") is None
        assert instrument("openai", recording_client) is True
        assert get_instrumented_client("openai") is recording_client
        uninstrument("openai")
        assert get_instrumented_client("openai") is None

    def test_instrument_with_different_client_refuses(self, recording_client, monkeypatch):
        # Build a second, distinct client — same fixture recipe as recording_client.
        from unittest.mock import MagicMock

        from disseqt_agentic_sdk import DisseqtAgenticClient
        from tests.agentic.instrumentation.conftest import RecordingBuffer

        monkeypatch.setattr("disseqt_agentic_sdk.client.client.HTTPTransport", MagicMock())
        monkeypatch.setattr(
            "disseqt_agentic_sdk.client.client.TraceBuffer",
            lambda **kw: RecordingBuffer(),
        )
        other_client = DisseqtAgenticClient(
            api_key="other",
            project_id="other",
            service_name="other",
            endpoint="http://localhost/v1/traces",
        )

        assert instrument("openai", recording_client) is True
        # Second call with a different client must refuse, not silently swap.
        assert instrument("openai", other_client) is False
        assert get_instrumented_client("openai") is recording_client

        uninstrument("openai")
        other_client.shutdown()

    def test_uninstrument_drops_client_reference(self, recording_client):
        # After uninstrument(), the instrumentor must not keep the client
        # alive via wrapper closures — long-lived processes that repeatedly
        # instrument/uninstrument would otherwise accumulate clients.
        assert instrument("openai", recording_client) is True
        instrumentor = auto_module._ACTIVE["openai"]
        assert instrumentor._client is recording_client
        uninstrument("openai")
        assert instrumentor._client is None

    def test_partial_instrument_failure_rolls_back(self, recording_client):
        # If _instrument() raises after patching some methods, those patches
        # must be rolled back — otherwise the SDK is left partly wrapped
        # with nothing tracking the leftovers.
        Completions = openai.resources.chat.completions.Completions
        original_create = inspect.getattr_static(Completions, "create")

        class FlakyInstrumentor(DisseqtInstrumentor):
            package_name = "openai"

            def _instrument(self) -> None:
                self._wrap(
                    "openai.resources.chat.completions",
                    "Completions.create",
                    lambda w, i, a, kw: w(*a, **kw),
                )
                raise RuntimeError("boom halfway through")

        assert FlakyInstrumentor().instrument(recording_client) is False
        # The Completions.create attribute on the class must be the pristine
        # original — the failed instrumentor's wrapper was unwound.
        assert inspect.getattr_static(Completions, "create") is original_create

    def test_nested_wrap_by_another_lib_is_not_corrupted(self, recording_client):
        # Simulate another instrumentation library wrapping the same method
        # on top of ours. uninstrument() must not tear down their wrapper.
        Completions = openai.resources.chat.completions.Completions
        original_create = inspect.getattr_static(Completions, "create")

        assert instrument("openai", recording_client) is True

        def third_party(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
            return wrapped(*args, **kwargs)

        wrapt.wrap_function_wrapper(
            "openai.resources.chat.completions",
            "Completions.create",
            third_party,
        )
        top_after_third_party = inspect.getattr_static(Completions, "create")

        uninstrument("openai")

        # The third-party wrapper must still be installed as the outermost
        # layer — we did not touch it. If we had blindly rewritten the leaf
        # to its `__wrapped__`, this identity check would fail.
        assert inspect.getattr_static(Completions, "create") is top_after_third_party
        # Restore the pristine method so we don't leak wrappers to other tests.
        Completions.create = original_create

    def test_strict_raises_on_unknown_provider(self, recording_client):
        with pytest.raises(InstrumentationError) as excinfo:
            instrument("does-not-exist", recording_client, strict=True)
        assert excinfo.value.name == "does-not-exist"
        assert excinfo.value.reason == "unknown_provider"

    def test_strict_raises_on_instrument_failure(self, recording_client):
        # Register a broken instrumentor for a real installed package.
        broken_key = "openai-broken-test"
        auto_module.INSTRUMENTOR_CLASSES[broken_key] = (
            "tests.agentic.instrumentation.test_base.BrokenInstrumentor"
        )
        try:
            with pytest.raises(InstrumentationError) as excinfo:
                instrument(broken_key, recording_client, strict=True)
            assert excinfo.value.reason == "instrument_failure"
            assert "boom" in excinfo.value.detail
        finally:
            auto_module.INSTRUMENTOR_CLASSES.pop(broken_key, None)

    def test_strict_does_not_raise_on_missing_package(self, recording_client):
        # Package not installed is a skip, not a failure — strict must not raise.
        missing_key = "missing-test-provider"
        auto_module.INSTRUMENTOR_CLASSES[missing_key] = (
            "tests.agentic.instrumentation.test_base.MissingPackageInstrumentor"
        )
        try:
            assert instrument(missing_key, recording_client, strict=True) is False
        finally:
            auto_module.INSTRUMENTOR_CLASSES.pop(missing_key, None)

    def test_non_strict_returns_false_and_records_reason(self, recording_client):
        # Non-strict path: bool return, no exception, reason captured on the
        # instrumentor for direct inspection.
        broken = BrokenInstrumentor()
        assert broken.instrument(recording_client) is False
        assert broken._last_error is not None
        reason, detail = broken._last_error
        assert reason == "instrument_failure"
        assert "boom" in detail

    def test_on_install_hook_fires_with_name_and_version(self, recording_client):
        calls: list[tuple[str, str]] = []
        assert (
            instrument("openai", recording_client, on_install=lambda n, v: calls.append((n, v)))
            is True
        )
        try:
            assert len(calls) == 1
            name, version = calls[0]
            assert name == "openai"
            # openai is installed in the test env; version comes from importlib.metadata.
            assert version and version != "unknown"
        finally:
            uninstrument("openai")

    def test_on_install_hook_not_called_on_failure(self, recording_client):
        calls: list[tuple[str, str]] = []
        # Unknown provider — hook should not fire.
        assert (
            instrument(
                "does-not-exist", recording_client, on_install=lambda n, v: calls.append((n, v))
            )
            is False
        )
        assert calls == []

    def test_on_install_hook_exceptions_are_swallowed(self, recording_client):
        # A broken user hook must not crash instrument() or corrupt _ACTIVE.
        def broken(_name: str, _version: str) -> None:
            raise RuntimeError("user hook exploded")

        assert instrument("openai", recording_client, on_install=broken) is True
        try:
            assert "openai" in auto_module._ACTIVE
        finally:
            uninstrument("openai")

    def test_on_uninstall_hook_fires(self, recording_client):
        instrument("openai", recording_client)
        calls: list[str] = []
        assert uninstrument("openai", on_uninstall=lambda n: calls.append(n)) is True
        assert calls == ["openai"]
        # Second call — nothing to uninstall, hook must not fire.
        assert uninstrument("openai", on_uninstall=lambda n: calls.append(n)) is False
        assert calls == ["openai"]

    def test_instrument_all_forwards_on_install_hook(self, recording_client):
        installed_via_hook: list[str] = []
        try:
            names = instrument_all(
                recording_client,
                on_install=lambda n, _v: installed_via_hook.append(n),
            )
            assert set(installed_via_hook) == set(names)
        finally:
            uninstrument_all()

    def test_duration_ms_recorded_on_every_span(self, recording_client):
        # Exercise the real openai wrapper; scope exit should attach
        # agentic.request.duration_ms with a plausible positive value.
        import json
        from unittest.mock import patch as mp

        from openai import OpenAI
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        from disseqt_agentic_sdk.semantics import AgenticAttributes

        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            fake = ChatCompletion(
                id="c",
                model="m",
                object="chat.completion",
                created=0,
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(role="assistant", content="ok"),
                    )
                ],
                usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
            with mp.object(client.chat.completions, "_post", return_value=fake, create=True):
                client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "x"}]
                )
        finally:
            uninstrument("openai")

        # find_span imported earlier via conftest; re-import for clarity.
        from tests.agentic.instrumentation.conftest import find_span

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)
        duration = attrs[AgenticAttributes.REQUEST_DURATION_MS]
        assert isinstance(duration, (int, float))
        assert duration >= 0.0
        # Local mocked call — must comfortably complete under a second.
        assert duration < 1000.0

    def test_slow_call_warning_fires_over_threshold(self, recording_client):
        # Lower the threshold to near-zero so a trivially fast call trips it.
        import json
        import logging
        from unittest.mock import patch as mp

        from openai import OpenAI
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        from disseqt_agentic_sdk.instrumentation import (
            _utils as utils_module,
        )
        from disseqt_agentic_sdk.instrumentation import (
            get_slow_call_threshold_ms,
            set_slow_call_threshold_ms,
        )
        from disseqt_agentic_sdk.semantics import AgenticAttributes

        original = get_slow_call_threshold_ms()
        set_slow_call_threshold_ms(0.001)

        # The SDK logger disables root propagation, so caplog can't see it.
        # Attach a local handler directly for the duration of the test.
        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.setLevel(logging.WARNING)
        handler.emit = captured.append  # type: ignore[assignment]
        utils_module._logger.addHandler(handler)
        # The SDK-wide logger factory silences the logger by default; bump
        # our logger up so WARNINGs actually get emitted for the test.
        prior_level = utils_module._logger.level
        utils_module._logger.setLevel(logging.WARNING)

        try:
            instrument("openai", recording_client)
            try:
                client = OpenAI(api_key="fake")
                fake = ChatCompletion(
                    id="c",
                    model="m",
                    object="chat.completion",
                    created=0,
                    choices=[
                        Choice(
                            index=0,
                            finish_reason="stop",
                            message=ChatCompletionMessage(role="assistant", content="ok"),
                        )
                    ],
                    usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
                with mp.object(client.chat.completions, "_post", return_value=fake, create=True):
                    client.chat.completions.create(
                        model="m", messages=[{"role": "user", "content": "x"}]
                    )
            finally:
                uninstrument("openai")

            assert any("slow LLM call" in rec.getMessage() for rec in captured)
            # Duration is still recorded regardless.
            from tests.agentic.instrumentation.conftest import find_span

            span = find_span(recording_client, "openai.chat.completions.create")
            attrs = json.loads(span.attributes_json)
            assert AgenticAttributes.REQUEST_DURATION_MS in attrs
        finally:
            set_slow_call_threshold_ms(original)
            utils_module._logger.removeHandler(handler)
            utils_module._logger.setLevel(prior_level)

    def test_threshold_is_contextvar_isolated_across_async_tasks(self):
        """
        TP-2128 P2 #2.3: _slow_threshold_ms used to be a plain module
        global — two concurrent asyncio tasks racing set/get would trample
        each other. It's now a contextvars.ContextVar so each task carries
        its own value.
        """
        import asyncio

        from disseqt_agentic_sdk.instrumentation import (
            get_slow_call_threshold_ms,
            set_slow_call_threshold_ms,
        )

        # Snapshot the outer value so we can restore.
        outer = get_slow_call_threshold_ms()

        async def one_task(value: float) -> float:
            set_slow_call_threshold_ms(value)
            # Yield control so the sibling task runs interleaved. If the
            # threshold were a plain global, the sibling's set would race
            # in here and change what we read.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return get_slow_call_threshold_ms()

        async def main() -> tuple[float, float]:
            return await asyncio.gather(one_task(100.0), one_task(9999.0))

        try:
            a, b = asyncio.run(main())
            assert a == 100.0, f"task A saw {a}, expected 100.0 — contextvar isolation broken"
            assert b == 9999.0, f"task B saw {b}, expected 9999.0 — contextvar isolation broken"
            # Outer context untouched by inner set()s (fresh async task
            # inherits at gather time; writes stay in the task's context).
            assert get_slow_call_threshold_ms() == outer
        finally:
            set_slow_call_threshold_ms(outer)

    def test_set_threshold_none_disables_warning(self, recording_client):
        # threshold=None → warning suppressed even for slow calls.
        import logging
        from unittest.mock import patch as mp

        from openai import OpenAI
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        from disseqt_agentic_sdk.instrumentation import (
            _utils as utils_module,
        )
        from disseqt_agentic_sdk.instrumentation import (
            get_slow_call_threshold_ms,
            set_slow_call_threshold_ms,
        )

        original = get_slow_call_threshold_ms()
        set_slow_call_threshold_ms(None)

        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.setLevel(logging.WARNING)
        handler.emit = captured.append  # type: ignore[assignment]
        utils_module._logger.addHandler(handler)
        # The SDK-wide logger factory silences the logger by default; bump
        # our logger up so WARNINGs actually get emitted for the test.
        prior_level = utils_module._logger.level
        utils_module._logger.setLevel(logging.WARNING)

        try:
            instrument("openai", recording_client)
            try:
                client = OpenAI(api_key="fake")
                fake = ChatCompletion(
                    id="c",
                    model="m",
                    object="chat.completion",
                    created=0,
                    choices=[
                        Choice(
                            index=0,
                            finish_reason="stop",
                            message=ChatCompletionMessage(role="assistant", content="ok"),
                        )
                    ],
                    usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
                with mp.object(client.chat.completions, "_post", return_value=fake, create=True):
                    client.chat.completions.create(
                        model="m", messages=[{"role": "user", "content": "x"}]
                    )
            finally:
                uninstrument("openai")

            assert not any("slow LLM call" in rec.getMessage() for rec in captured)
        finally:
            set_slow_call_threshold_ms(original)
            utils_module._logger.removeHandler(handler)
            utils_module._logger.setLevel(prior_level)

    def test_broken_attribute_writer_does_not_break_user_call(self, recording_client):
        # A bug in our attribute-writing must never crash the wrapped LLM
        # call. Simulate by monkey-patching an OAI-compat helper to raise
        # unconditionally, then confirm the user still gets their response.
        import json
        from unittest.mock import patch as mp

        from openai import OpenAI
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        # openai/patch.py does `from ..._oai_compat import set_common_chat_request`,
        # so we must patch the reference used by the wrapper, not the source.
        from disseqt_agentic_sdk.instrumentation.openai import patch as openai_patch

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated bug in set_common_chat_request")

        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            fake = ChatCompletion(
                id="c",
                model="m",
                object="chat.completion",
                created=0,
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(role="assistant", content="ok"),
                    )
                ],
                usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
            with (
                mp.object(openai_patch, "set_common_chat_request", _boom),
                mp.object(client.chat.completions, "_post", return_value=fake, create=True),
            ):
                # The user's call must still complete successfully even though
                # our attribute-writing path is broken.
                result = client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "x"}]
                )
        finally:
            uninstrument("openai")

        assert result.choices[0].message.content == "ok"
        # And the span was still emitted (scope exit still ran), just without
        # the attributes the broken writer would have set.
        from tests.agentic.instrumentation.conftest import find_span

        span = find_span(recording_client, "openai.chat.completions.create")
        # Response attrs may or may not exist depending on whether the
        # response-side writer also failed; the key assertion is
        # "user code kept working".
        assert json.loads(span.attributes_json) is not None

    def test_concurrent_instrument_calls_are_race_free(self, recording_client):
        # Many threads racing on the same provider must produce exactly one
        # successful instrument() and one registry entry — no duplicate
        # patching, no corrupted _ACTIVE.
        results: list[bool] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(16)

        def worker() -> None:
            barrier.wait()
            ok = instrument("openai", recording_client)
            with results_lock:
                results.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == 1, f"expected exactly one winner, got {sum(results)}"
        assert "openai" in auto_module._ACTIVE
        uninstrument("openai")
        assert "openai" not in auto_module._ACTIVE
