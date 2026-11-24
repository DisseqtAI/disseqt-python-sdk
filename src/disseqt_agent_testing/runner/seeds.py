"""Deterministic per-repeat seeds and the ``current_repeat()`` context."""

import contextvars
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepeatContext:
    case_id: str
    index: int
    seed: int
    n: int


_CURRENT_REPEAT: contextvars.ContextVar[RepeatContext | None] = contextvars.ContextVar(
    "disseqt_current_repeat", default=None
)


def current_repeat() -> RepeatContext | None:
    """The repeat the calling code is running inside (works in agent code and fake handlers:
    the harness copies the context into its worker thread)."""
    return _CURRENT_REPEAT.get()


def derive_seed(suite_seed: int, case_id: str, index: int) -> int:
    """SHA-256 based, so it is identical across processes and ``PYTHONHASHSEED`` values; never
    ``hash()`` and never the global ``random`` state."""
    raw = f"disseqt-seed-v1\x1f{suite_seed}\x1f{case_id}\x1f{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & (2**63 - 1)
