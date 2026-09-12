"""Extension base classes for user-defined red-team primitives (Phase 5a).

Thin, deliberately unopinionated bases. Subclasses either override the
default behavior (client-side mutation, custom guard scoring) or leave
the defaults in place and let the SDK dispatch back to the server for
the heavy lifting.

Nothing here talks to the network directly — subclass hooks receive a
:class:`~disseqt_sdk.client.Client` when they need it. That keeps tests
simple (no HTTP mocking to instantiate an extension) and keeps the
inheritance surface honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import Client


@dataclass
class BaseVulnerability:
    """Named vulnerability class. Server-driven by default.

    Override :meth:`assess` to run a bespoke check locally. The default
    delegates to the SDK's red-team endpoints via the passed client.
    """

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def assess(self, client: Client, target: Any) -> dict[str, Any]:  # noqa: D401
        """Assess ``target`` for this vulnerability.

        Default implementation is a stub — subclasses either override or
        the caller uses :func:`disseqt_sdk.red_team` which composes
        server-driven vulnerabilities.
        """
        raise NotImplementedError(
            "BaseVulnerability.assess() is a stub — subclass or use "
            "disseqt_sdk.red_team() with a server-registered vulnerability id"
        )

    def simulate_attacks(self, client: Client, count: int = 1) -> list[str]:
        """Return ``count`` attack prompts for this vulnerability."""
        raise NotImplementedError


@dataclass
class BaseAttack:
    """Base attack technique — override :meth:`enhance` for client-side mutation."""

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def enhance(self, prompt: str) -> str:
        """Return a mutated version of ``prompt``. Default: passthrough."""
        return prompt

    def progress(self) -> float:
        """Progress in [0, 1]. Multi-turn attacks override; single-turn is 1.0 after enhance."""
        return 1.0


@dataclass
class BaseSingleTurnAttack(BaseAttack):
    """Single-turn attack — one prompt in, one mutated prompt out."""

    def enhance(self, prompt: str) -> str:
        return prompt


@dataclass
class BaseMultiTurnAttack(BaseAttack):
    """Multi-turn attack — tracks turn state.

    Subclasses override :meth:`next_turn` and manage state via
    :attr:`turns`.
    """

    max_turns: int = 5
    turns: list[dict[str, Any]] = field(default_factory=list)

    def next_turn(self, target_response: str | None = None) -> str | None:
        """Return the next attacker prompt, or None when finished."""
        raise NotImplementedError

    def progress(self) -> float:
        if self.max_turns <= 0:
            return 1.0
        return min(1.0, len(self.turns) / self.max_turns)


@dataclass
class BaseGuard:
    """Custom guard — thin wrapper over a policy id.

    Composes with :class:`~disseqt_sdk.guardrails.Guardrails` — subclass
    only if you need to add local scoring or transform the response
    before returning it.
    """

    policy_id: str
    name: str = ""

    def evaluate(self, client: Client, input_data: Any) -> dict[str, Any]:
        """Evaluate ``input_data`` against this guard's policy."""
        result = client.validate(input_data, policies=[self.policy_id])
        return result if isinstance(result, dict) else {}


@dataclass
class BaseMetric:
    """Named metric — thin wrapper over criteria + threshold + labels.

    Feeds the framework-mapping contract (see plan §3c): the ``name``
    must match the server-side ``control.MetricName`` string exactly for
    control-test wiring to fire.
    """

    name: str
    threshold: float
    criteria: str = ""
    labels: list[str] = field(default_factory=list)
    lower_is_better: bool = True


class BaseLLM:
    """User-extendable LLM provider — client-side judge model plug-point.

    Subclasses implement :meth:`generate` and optionally :meth:`a_generate`.
    Kept as a plain class (not a dataclass) so subclasses are free to
    manage their own state (auth clients, connection pools, etc.).
    """

    model: str = ""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        # Default async: run sync in executor. Subclasses override for
        # true async when the underlying provider supports it.
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))
