"""The refund demo from the plan's first-shippable acceptance run (importable by path)."""

import functools
from typing import Any

from .refund_agent import make_refund_agent


def make_demo_agent(tools: Any) -> Any:
    return make_refund_agent(tools)


def make_demo_wrong_order_agent(tools: Any) -> Any:
    return functools.partial(make_refund_agent, bug="wrong_order")(tools)
