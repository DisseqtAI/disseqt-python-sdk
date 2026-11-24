"""Cache modes: ``replay`` (read only), ``record`` (read, live on miss, write), ``refresh``
(always live, overwrite), ``off``; ``auto`` is ``replay`` under CI and ``record`` otherwise."""

import os

from ..codes import JudgeConfigError
from ..network import is_ci

MODES = ("auto", "replay", "record", "refresh", "off")


def resolve_mode(configured: str) -> str:
    """Config, then ``DISSEQT_JUDGE_CACHE``, then ``auto``. Never returns ``auto``."""
    mode = configured
    if mode == "auto":
        mode = os.environ.get("DISSEQT_JUDGE_CACHE", "auto").strip().lower() or "auto"
    if mode not in MODES:
        raise JudgeConfigError(
            "cache_mode", f"invalid judge cache mode {mode!r}; choose one of {list(MODES)}"
        )
    if mode == "auto":
        return "replay" if is_ci() else "record"
    return mode
