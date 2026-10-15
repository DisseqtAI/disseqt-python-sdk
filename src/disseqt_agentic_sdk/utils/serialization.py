"""JSON-safety helpers shared by the span model and the API helpers."""

from __future__ import annotations

import json
from typing import Any

from .logging import get_logger

_logger = get_logger(__name__)


def json_safe(value: Any) -> Any:
    """
    Coerce arbitrary values into something ``json.dumps`` can handle.

    Primitives / lists / dicts that already serialise pass through;
    anything else falls back to ``str()`` (pydantic models, ORM rows,
    custom classes). Never raises — worst case yields ``repr()`` or a type placeholder.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return str(value)
        except Exception:
            try:
                return repr(value)
            except Exception:
                return f"<unserializable {type(value).__name__}>"


def dumps_attributes(attributes: dict[Any, Any]) -> str:
    """
    Serialise span attributes to a JSON string without ever raising.

    ``default=str`` covers datetime / set / bytes / arbitrary objects. If the
    whole dict still fails (circular reference, non-string dict keys, a
    ``__str__`` that raises), drop the offending *attribute* instead of the
    whole span: each value is coerced individually via :func:`json_safe`.
    """
    try:
        return json.dumps(attributes, default=str)
    except Exception:
        pass
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        skey = key if isinstance(key, str) else str(key)
        try:
            json.dumps(value, default=str)
            safe[skey] = value
            continue
        except Exception:
            pass
        coerced = json_safe(value)
        try:
            json.dumps(coerced, default=str)
            safe[skey] = coerced
        except Exception:
            _logger.warning("dropping unserialisable span attribute", extra={"attribute": skey})
    return json.dumps(safe, default=str)
