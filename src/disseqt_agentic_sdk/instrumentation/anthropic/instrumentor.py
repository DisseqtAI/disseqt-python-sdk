"""
Anthropic SDK instrumentor.

Patches anthropic v0.x/v1.x resource methods:
  * messages.create           (sync + async, streaming + non-streaming)
"""

from __future__ import annotations

from disseqt_agentic_sdk.instrumentation.anthropic import patch
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor


class AnthropicInstrumentor(DisseqtInstrumentor):
    package_name = "anthropic"
    min_version = "0.40.0"

    def _instrument(self) -> None:
        self._wrap(
            "anthropic.resources.messages",
            "Messages.create",
            patch.messages_create(self),
        )
        self._wrap(
            "anthropic.resources.messages",
            "AsyncMessages.create",
            patch.async_messages_create(self),
        )
