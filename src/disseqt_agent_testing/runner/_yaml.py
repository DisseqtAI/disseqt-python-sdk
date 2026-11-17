"""YAML case files (needs the ``[yaml]`` extra): YAML 1.2 booleans, duplicate-key rejection and a
path -> line map built from the composed node tree."""

import importlib
import re
from typing import Any

from ._json import JsonProblem, Lines

yaml: Any
try:
    yaml = importlib.import_module("yaml")
except ImportError:  # the extra is optional
    yaml = None

BOOL = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")


def available() -> bool:
    return yaml is not None


if yaml is not None:

    class _Loader(yaml.SafeLoader):
        """``no``, ``on``, ``yes`` stay strings (YAML 1.1 would turn a country code into False)."""

    _Loader.yaml_implicit_resolvers = {
        first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    _Loader.add_implicit_resolver("tag:yaml.org,2002:bool", BOOL, list("tTfF"))


def _walk(node: Any, path: tuple[Any, ...], lines: Lines) -> None:
    if isinstance(node, yaml.MappingNode):
        seen: set[Any] = set()
        for key_node, value_node in node.value:
            key = key_node.value
            if key in seen:
                raise JsonProblem(key_node.start_mark.line + 1, f"duplicate key '{key}'")
            seen.add(key)
            lines[(*path, key)] = key_node.start_mark.line + 1
            _walk(value_node, (*path, key), lines)
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            lines[(*path, i)] = item.start_mark.line + 1
            _walk(item, (*path, i), lines)


def parse_yaml(text: str) -> tuple[Any, Lines]:
    if yaml is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("PyYAML is not installed")
    try:
        node = yaml.compose(text, Loader=_Loader)
        lines: Lines = {}
        if node is not None:
            lines[()] = node.start_mark.line + 1
            _walk(node, (), lines)
        value = yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None) or str(exc)
        raise JsonProblem((mark.line + 1) if mark else 1, str(problem)) from None
    return value, lines
