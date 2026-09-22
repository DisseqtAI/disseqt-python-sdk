"""Cost + progress plumbing for the SDK (Phase 6).

Lazy — nothing here allocates a bar unless ``DISSEQT_SHOW_PROGRESS`` is set,
and cost accumulation is opt-in per call site via
:func:`cost_accumulator`. Zero overhead when unused.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostBucket:
    """Running totals attached to a :func:`cost_accumulator` scope.

    Fields are additive — callers add per-call figures with :meth:`add`.
    """

    simulation_cost: float = 0.0
    evaluation_cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return self.simulation_cost + self.evaluation_cost

    def add(
        self,
        *,
        simulation_cost: float = 0.0,
        evaluation_cost: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        label: str = "",
    ) -> None:
        self.simulation_cost += simulation_cost
        self.evaluation_cost += evaluation_cost
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.entries.append(
            {
                "label": label,
                "simulation_cost": simulation_cost,
                "evaluation_cost": evaluation_cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )


_current: ContextVar[CostBucket | None] = ContextVar("_disseqt_cost", default=None)


@contextmanager
def cost_accumulator() -> Iterator[CostBucket]:
    """Attach a :class:`CostBucket` to the current async/thread context.

    Nested scopes get their own bucket. Callers deep in the stack look up
    ``current_cost_bucket()`` to add totals; if no bucket is active, the
    add is a no-op.

    Example::

        with cost_accumulator() as costs:
            client.validate_sync(req, policies=[pid])
            print(costs.total_cost)
    """
    bucket = CostBucket()
    token = _current.set(bucket)
    try:
        yield bucket
    finally:
        _current.reset(token)


def current_cost_bucket() -> CostBucket | None:
    """Return the active :class:`CostBucket` or None."""
    return _current.get()


def progress_enabled() -> bool:
    """True when ``DISSEQT_SHOW_PROGRESS`` is a truthy env value."""
    return os.environ.get("DISSEQT_SHOW_PROGRESS", "").lower() in ("1", "true", "yes", "on")
