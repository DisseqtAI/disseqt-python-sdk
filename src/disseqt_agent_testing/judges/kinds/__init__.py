"""Judge kinds. Importing this package registers the built-in kinds."""

from . import builtin as _builtin  # noqa: F401
from . import quality as _quality  # noqa: F401
from .base import JudgeKindSpec, Precheck, get_kind, list_kinds, register_kind, unregister_kind

__all__ = [
    "JudgeKindSpec",
    "Precheck",
    "get_kind",
    "list_kinds",
    "register_kind",
    "unregister_kind",
]
