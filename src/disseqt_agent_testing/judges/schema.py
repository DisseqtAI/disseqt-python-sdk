"""A small JSON-schema validator (about 80 lines: no pydantic, no jsonschema).

Supports ``type``, ``enum``, ``required``, ``properties``, ``additionalProperties`` (false),
``items``, ``minLength``/``maxLength``, ``minimum``/``maximum`` and ``minItems``/``maxItems``.
"""

import copy
from collections.abc import Mapping
from typing import Any

_PY: dict[str, Any] = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list,
    "object": dict, "null": type(None),
}  # fmt: skip
PROVIDER_UNSAFE = ("maxLength", "minLength", "minimum", "maximum")


def _type_ok(name: str, value: Any) -> bool:
    if name in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _PY[name])


def validate(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Return a list of problems (empty means valid)."""
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        names = list(types) if isinstance(types, (list, tuple)) else [types]
        if not any(_type_ok(n, value) for n in names):
            return [f"{path}: expected {'|'.join(names)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {list(schema['enum'])}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                errors.extend(validate(value[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected property {key!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors


def strip_keywords(schema: Any, keys: tuple[str, ...] = PROVIDER_UNSAFE) -> Any:
    """A copy of ``schema`` without keywords some providers reject; the engine still enforces
    them locally after parsing."""
    if isinstance(schema, dict):
        return {k: strip_keywords(v, keys) for k, v in schema.items() if k not in keys}
    if isinstance(schema, list):
        return [strip_keywords(v, keys) for v in schema]
    return copy.deepcopy(schema)
