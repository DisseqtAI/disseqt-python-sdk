"""Matching: score a value against a matcher; ``register_matcher_impl`` adds semantics."""

from . import impls as _impls  # noqa: F401  (registers the built-in implementations)
from . import impls3b as _impls3b  # noqa: F401
from .registry import (
    MatchContext,
    MatchResult,
    describe,
    match_dict,
    match_value,
    register_matcher_impl,
    unregister_matcher_impl,
)

__all__ = [
    "MatchContext",
    "MatchResult",
    "describe",
    "match_dict",
    "match_value",
    "register_matcher_impl",
    "unregister_matcher_impl",
]
