"""Routing: derive hand-off events from a run (sub-agent calls, ``transfer_to_agent``, tools that
match ``route_patterns``). Targets are compared after normalization, so ``"Billing Agent"`` and
``transfer_to_billing_agent`` name the same destination."""

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase

from ..models import AgentRun, Step
from .config import EvalConfig

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_target(name: str) -> str:
    text = _NON_ALNUM.sub("_", name.casefold()).strip("_")
    return text[: -len("_agent")] if text.endswith("_agent") and len(text) > 6 else text


@dataclass(slots=True, frozen=True)
class RouteEvent:
    target: str
    raw: str
    step: Step


def _strip_pattern(name: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if fnmatchcase(name, pattern):
            prefix = pattern.split("*", 1)[0]
            return (
                name[len(prefix) :] if name.startswith(prefix) and len(name) > len(prefix) else name
            )
    return None


def route_events(run: AgentRun, cfg: EvalConfig) -> list[RouteEvent]:
    events: list[RouteEvent] = []
    for step in run.steps:
        if step.kind == "subagent":
            events.append(RouteEvent(norm_target(step.name), step.name, step))
        elif step.kind == "tool" and step.name == "transfer_to_agent":
            raw = step.args.get("agent_name")
            if isinstance(raw, str):
                events.append(RouteEvent(norm_target(raw), raw, step))
        elif step.kind == "tool":
            stripped = _strip_pattern(step.name, cfg.route_patterns)
            if stripped is not None:
                events.append(RouteEvent(norm_target(stripped), stripped, step))
    return events


def offered_targets(run: AgentRun, cfg: EvalConfig) -> set[str]:
    """Every destination the agent was offered: sub-agents, ``transfer_to_*`` tools and the
    ``agent_name`` enum of ``transfer_to_agent``."""
    out: set[str] = set()
    for tool in run.available_tools:
        if tool.kind == "subagent":
            out.add(norm_target(tool.name))
            continue
        if tool.name == "transfer_to_agent":  # generic hand-off: its enum lists the destinations
            props = tool.schema.get("properties") if isinstance(tool.schema, dict) else None
            enum = (props or {}).get("agent_name", {}).get("enum") if props else None
            out.update(norm_target(str(v)) for v in (enum or ()))
            continue
        stripped = _strip_pattern(tool.name, cfg.route_patterns)
        if stripped is not None:
            out.add(norm_target(stripped))
    return out
