"""YAML case files (needs the ``[yaml]`` extra): YAML 1.2 booleans and numbers, duplicate-key
rejection and a path -> line map built from the composed node tree."""

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
INT = re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$")
FLOAT = re.compile(
    r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)
_TAG = "tag:yaml.org,2002:"
_OWN_RESOLVERS = (_TAG + "bool", _TAG + "int", _TAG + "float")


def available() -> bool:
    return yaml is not None


if yaml is not None:

    def _problem(node: Any, text: str) -> JsonProblem:
        return JsonProblem(node.start_mark.line + 1, text)

    class _Loader(yaml.SafeLoader):
        """Scalars resolve as in the YAML 1.2 core schema: ``no``, ``on``, ``yes`` stay strings
        (YAML 1.1 would turn a country code into False), ``010`` is ten, ``1e3`` is a float and
        ``1_000`` / ``1:30`` are strings."""

        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            """Duplicates are compared as constructed keys, so ``1`` and ``'1'`` are two keys while
            ``true`` and ``True`` are one. A ``<<`` merge is not a key."""
            seen: set[Any] = set()
            for key_node, _ in node.value if isinstance(node, yaml.MappingNode) else ():
                if key_node.tag == _TAG + "merge":
                    continue
                key = self.construct_object(key_node, deep=True)
                try:
                    duplicate = key in seen
                except TypeError:
                    raise _problem(key_node, "a mapping key must be a scalar") from None
                if duplicate:
                    shown = key_node.value if isinstance(key_node.value, str) else key
                    raise _problem(key_node, f"duplicate key '{shown}'")
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

        def _int(self, node: Any) -> int:
            text = str(self.construct_scalar(node))
            try:
                if text.startswith("0o"):
                    return int(text[2:], 8)
                if text.startswith("0x"):
                    return int(text[2:], 16)
                return int(text)
            except ValueError:
                raise _problem(node, f"invalid integer '{text}'") from None

        def _float(self, node: Any) -> float:
            text = str(self.construct_scalar(node))
            lowered = text.lower()
            if lowered.lstrip("+-") == ".inf":
                return float("-inf") if lowered.startswith("-") else float("inf")
            if lowered == ".nan":
                return float("nan")
            try:
                return float(text)
            except ValueError:
                raise _problem(node, f"invalid float '{text}'") from None

        def _timestamp(self, node: Any) -> Any:
            try:
                return self.construct_yaml_timestamp(node)
            except ValueError as exc:
                raise _problem(node, f"invalid date '{node.value}': {exc}") from None

    _Loader.yaml_implicit_resolvers = {
        first: [(tag, rx) for tag, rx in resolvers if tag not in _OWN_RESOLVERS]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    _Loader.add_implicit_resolver(_TAG + "bool", BOOL, list("tTfF"))
    _Loader.add_implicit_resolver(_TAG + "int", INT, list("-+0123456789"))
    _Loader.add_implicit_resolver(_TAG + "float", FLOAT, list("-+0123456789."))
    _Loader.add_constructor(_TAG + "int", _Loader._int)
    _Loader.add_constructor(_TAG + "float", _Loader._float)
    _Loader.add_constructor(_TAG + "timestamp", _Loader._timestamp)


def _walk(node: Any, path: tuple[Any, ...], lines: Lines, active: set[int], done: set[int]) -> None:
    """``active`` holds the collections being walked (an alias back into one is a cycle);
    ``done`` holds finished ones, so a collection aliased many times is walked once."""
    if not isinstance(node, (yaml.MappingNode, yaml.SequenceNode)):
        return
    if id(node) in active:
        raise JsonProblem(node.start_mark.line + 1, "an alias refers to its own parent")
    if id(node) in done:
        return
    active.add(id(node))
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            key = key_node.value
            if not isinstance(key, str):
                continue
            lines[(*path, key)] = key_node.start_mark.line + 1
            _walk(value_node, (*path, key), lines, active, done)
    else:
        for i, item in enumerate(node.value):
            lines[(*path, i)] = item.start_mark.line + 1
            _walk(item, (*path, i), lines, active, done)
    active.discard(id(node))
    done.add(id(node))


def parse_yaml(text: str) -> tuple[Any, Lines]:
    if yaml is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("PyYAML is not installed")
    try:
        node = yaml.compose(text, Loader=_Loader)
        lines: Lines = {}
        if node is not None:
            lines[()] = node.start_mark.line + 1
            _walk(node, (), lines, set(), set())
        value = yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None) or str(exc)
        raise JsonProblem((mark.line + 1) if mark else 1, str(problem)) from None
    except RecursionError:
        raise JsonProblem(1, "the document is nested too deeply") from None
    except ValueError as exc:
        raise JsonProblem(1, str(exc)) from None
    return value, lines
