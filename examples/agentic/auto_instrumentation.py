"""
Auto-instrumentation example.

`instrument_all(client)` patches every installed LLM provider SDK
(OpenAI, Anthropic, Groq, Mistral, Cohere, Gemini, LiteLLM). After
that, every openai.chat.completions.create() / anthropic.messages.create() /
etc. call auto-emits a DisseqtSpan without any manual `trace.start_span(...)`
wrapping.

If you already opened a trace/span via `start_trace(...)`, auto-spans nest
under it. If you didn't, each LLM call creates its own single-span trace.

Install the provider SDKs you plan to call:
    pip install "disseqt-ai-sdk[openai,anthropic]"
Or install everything we auto-instrument at once:
    pip install "disseqt-ai-sdk[instrumentation]"
"""

import os

from disseqt_agentic_sdk import DisseqtAgenticClient, instrument_all, start_trace

client = DisseqtAgenticClient(
    api_key=os.environ["DISSEQT_API_KEY"],
    project_id=os.environ["DISSEQT_PROJECT_ID"],
    application_id=os.environ.get("DISSEQT_APPLICATION_ID"),  # recommended
    service_name="auto-instrumented-demo",
    endpoint=os.environ.get(
        "DISSEQT_ENDPOINT",
        "https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
    ),
)

# One call — patches every LLM SDK we can detect on this env.
instrument_all(client)

# --- Case 1: no user-managed trace. Each call becomes its own trace. ---
from openai import OpenAI  # noqa: E402

oai = OpenAI()
response = oai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print("openai:", response.choices[0].message.content)

# --- Case 2: user-managed trace. Auto-spans nest under it. ---
with start_trace(client, name="multi_llm_workflow"):
    oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi in 3 words."}],
    )

    try:
        from anthropic import Anthropic

        ant = Anthropic()
        ant.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hi in 3 words."}],
        )
    except ImportError:
        pass  # anthropic not installed — instrument_all skipped it.

client.flush()
