"""A small JSON Schema checker used by the schema tests."""

from typing import Any


def validate(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$"
) -> list[str]:
    """A small JSON Schema checker: type, enum, const, required, properties, items, $ref,
    minimum/maximum and additionalProperties-as-schema. Enough for our own schema."""
    if "$ref" in schema:
        return validate(value, root["$defs"][schema["$ref"].rsplit("/", 1)[1]], root, path)
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        py = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list,
              "object": dict, "null": type(None)}  # fmt: skip
        ok = False
        for name in names:
            if name in ("integer", "number") and isinstance(value, bool):
                continue
            ok = ok or isinstance(value, py[name])
        if not ok:
            return [f"{path}: expected {names}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: missing {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate(value[key], sub, root, f"{path}.{key}"))
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            for key, item in value.items():
                if key not in schema.get("properties", {}):
                    errors.extend(validate(item, extra, root, f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], root, f"{path}[{i}]"))
    return errors
