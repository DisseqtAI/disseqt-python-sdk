"""Small shared literals and constants (kept free of imports to avoid cycles)."""

from typing import Literal

Effect = Literal["read", "write"]
EFFECTS: tuple[str, ...] = ("read", "write")

FakeKind = Literal["tool", "subagent"]
# One place to add a new kind of fake (for example "mcp"); validation derives from this tuple.
FAKE_KINDS: tuple[str, ...] = ("tool", "subagent")
