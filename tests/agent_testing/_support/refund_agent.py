"""Acceptance agents (plan 02 section 16): the HLD 7.1 refund flow on sync, async and FakeLLM."""

import asyncio
import re
from typing import Any

from disseqt_agent_testing.harness import spy, stub
from disseqt_agent_testing.models import FakeResponse, FakeSpec, Message, MsgToolCall

FAKES: dict[str, Any] = {
    "get_order": FakeSpec(
        "get_order",
        schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        effect="read",
        responses=(
            FakeResponse(when={"order_id": "A12"}, returns={"order_id": "A12", "total": 59.99}),
        ),
    ),
    "check_policy": spy({"eligible": True}),
    "issue_refund": stub({"amount": 59.99, "status": "ok"}),
}

ORDER_RE = re.compile(r"\b([A-Z]\d{2,})\b")


def make_refund_agent(tools: Any, *, bug: str | None = None) -> Any:
    class RefundAgent:
        def run(self, text: str) -> str:
            m = ORDER_RE.search(text)
            if not m:
                return "Which order number is this about?"  # ASK
            order = tools["get_order"](m.group(1))  # positional on purpose
            policy = tools["check_policy"](order_id=order["order_id"], reason="damaged")
            if not policy["eligible"]:
                return f"Sorry, order {order['order_id']} is not eligible."
            target = "Z999" if bug == "wrong_order" else order["order_id"]
            r = tools["issue_refund"](order_id=target, amount=order["total"])
            return f"Refunded {r['amount']} for {target}."

    return RefundAgent()


def make_async_refund_agent(tools: Any, *, bug: str | None = None) -> Any:
    """Needs ``async_tools=True``: the fakes are coroutine functions."""

    class AsyncRefundAgent:
        async def arun(self, text: str) -> str:
            m = ORDER_RE.search(text)
            if not m:
                return "Which order number is this about?"
            order = await tools["get_order"](m.group(1))
            await asyncio.sleep(0)
            policy = await tools["check_policy"](order_id=order["order_id"], reason="damaged")
            await asyncio.sleep(0)
            if not policy["eligible"]:
                return f"Sorry, order {order['order_id']} is not eligible."
            target = "Z999" if bug == "wrong_order" else order["order_id"]
            r = await tools["issue_refund"](order_id=target, amount=order["total"])
            return f"Refunded {r['amount']} for {target}."

    return AsyncRefundAgent()


def make_llm_refund_agent(tools: Any, llm: Any) -> Any:
    """Loop: ``llm.complete``, execute the requested tool calls, repeat until plain text."""

    class LLMRefundAgent:
        def run(self, text: str) -> str:
            history: list[Message] = [Message("user", text)]
            for _ in range(10):
                reply = llm.complete(history, list(tools))
                history.append(reply)
                if not reply.tool_calls:
                    return reply.content
                for call in reply.tool_calls:
                    result = tools[call.name](**dict(call.arguments))
                    history.append(Message("tool", str(result), tool_call_id=call.id))
            return "gave up"

    return LLMRefundAgent()


def refund_script() -> tuple[Message, ...]:
    """Assistant replies for ``make_llm_refund_agent``: look up, refund, then answer."""
    return (
        Message(
            "assistant", tool_calls=(MsgToolCall("c1", "get_order", arguments={"order_id": "A12"}),)
        ),
        Message(
            "assistant",
            tool_calls=(
                MsgToolCall("c2", "issue_refund", arguments={"order_id": "A12", "amount": 59.99}),
            ),
        ),
        Message("assistant", "Refunded 59.99 for A12."),
    )
