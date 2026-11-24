"""Canonical JSON, hashes and paths: the pieces that make reports byte-identical everywhere."""

import hashlib
import json
import math
from typing import Any

from ..models import thaw


def round6(value: float) -> float:
    """Round to 6 decimals; ``-0.0`` becomes ``0.0`` so equal numbers print equally."""
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded


def normalize(value: Any) -> Any:
    """Plain, sorted-friendly JSON: floats rounded, NaN/inf rejected, containers thawed."""
    value = thaw(value)
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{value!r} is not JSON-able")
        return round6(value)
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    raise TypeError(f"{type(value).__name__} is not JSON-able")


def canonical_json(value: Any) -> str:
    """Sorted keys, no whitespace, UTF-8 friendly, no NaN: identical for equal data anywhere."""
    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def short_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def args_hash(args: Any) -> str:
    """Stable hash of call arguments (``1`` and ``"1"`` differ; key order does not matter)."""
    return short_hash(args)


def path_str(segments: tuple[str | int, ...]) -> str:
    """``("items", 0, "city")`` -> ``$.items[0].city``."""
    out = "$"
    for segment in segments:
        out += f"[{segment}]" if isinstance(segment, int) else f".{segment}"
    return out


def clip(value: Any, limit: int = 80) -> str:
    """A short, single-line rendering of a value for messages (``...(len=N)`` when cut)."""
    try:
        text = canonical_json(value) if not isinstance(value, str) else repr(value)
    except (TypeError, ValueError):
        text = repr(value)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}...(len={len(text)})"
