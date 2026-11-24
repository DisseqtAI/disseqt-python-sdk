"""Registry of matcher classes, keyed by their JSON tag (``{"$m": "<tag>"}``).

The serializer encodes and decodes matchers through this registry, never through a hard-coded
union, so a new matcher is one class plus one ``register_matcher`` call.
"""

import threading

from .errors import ModelValidationError

_LOCK = threading.RLock()
_BY_TAG: dict[str, type] = {}
_BY_CLASS: dict[type, str] = {}


def register_matcher(tag: str, cls: type, *, override: bool = False) -> None:
    """Register ``cls`` under ``tag``. Duplicate tags raise unless ``override=True``."""
    if not tag or not tag.replace("_", "").isalnum():
        raise ModelValidationError(f"matcher tag {tag!r} must be letters, digits and '_'")
    with _LOCK:
        if tag in _BY_TAG and _BY_TAG[tag] is not cls and not override:
            raise ModelValidationError(f"matcher tag {tag!r} is already registered")
        _BY_TAG[tag] = cls
        _BY_CLASS[cls] = tag


def unregister_matcher(tag: str) -> None:
    with _LOCK:
        cls = _BY_TAG.pop(tag, None)
        if cls is not None:
            _BY_CLASS.pop(cls, None)


def matcher_class(tag: str) -> type:
    with _LOCK:
        cls = _BY_TAG.get(tag)
        known = sorted(_BY_TAG)
    if cls is None:
        raise ModelValidationError(f"unknown matcher tag {tag!r}; known tags: {known}")
    return cls


def matcher_tag(cls: type) -> str | None:
    with _LOCK:
        return _BY_CLASS.get(cls)


def matcher_tags() -> tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_BY_TAG))
