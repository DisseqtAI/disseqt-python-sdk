"""A small JSON parser that remembers the line of every value and rejects duplicate keys.

``json.loads`` can give neither (a duplicate key silently wins; positions are lost), and a case
file with a typo'd or duplicated key must not become a silent pass.
"""

import bisect
import json
from typing import Any

Lines = dict[tuple[Any, ...], int]
_DECODER = json.JSONDecoder()


class JsonProblem(Exception):
    def __init__(self, line: int, problem: str):
        super().__init__(problem)
        self.line = line
        self.problem = problem


class _Parser:
    def __init__(self, text: str):
        self.t = text
        self.i = 0
        self.starts = [0] + [m + 1 for m, ch in enumerate(text) if ch == "\n"]
        self.lines: Lines = {}

    def line(self, pos: int) -> int:
        return bisect.bisect_right(self.starts, pos)

    def fail(self, problem: str, pos: int | None = None) -> JsonProblem:
        return JsonProblem(self.line(self.i if pos is None else pos), problem)

    def ws(self) -> None:
        while self.i < len(self.t) and self.t[self.i] in " \t\r\n":
            self.i += 1

    def value(self, path: tuple[Any, ...]) -> Any:
        self.ws()
        if self.i >= len(self.t):
            raise self.fail("Expecting value")
        self.lines[path] = self.line(self.i)
        ch = self.t[self.i]
        if ch == "{":
            return self.obj(path)
        if ch == "[":
            return self.arr(path)
        if ch == '"':
            return self.string()
        for literal, val in (("true", True), ("false", False), ("null", None)):
            if self.t.startswith(literal, self.i):
                self.i += len(literal)
                return val
        end = self.i
        while end < len(self.t) and self.t[end] in "+-0123456789.eE":
            end += 1
        if end == self.i:
            raise self.fail("Expecting value")
        try:
            number = json.loads(self.t[self.i : end])
        except ValueError:
            raise self.fail("Invalid number") from None
        self.i = end
        return number

    def string(self) -> str:
        try:
            text, end = _DECODER.raw_decode(self.t, self.i)
        except json.JSONDecodeError as exc:
            raise self.fail(exc.msg) from None
        if not isinstance(text, str):
            raise self.fail("Expecting a string")
        self.i = end
        return text

    def obj(self, path: tuple[Any, ...]) -> dict[str, Any]:
        self.i += 1
        out: dict[str, Any] = {}
        self.ws()
        if self.t[self.i : self.i + 1] == "}":
            self.i += 1
            return out
        while True:
            self.ws()
            if self.t[self.i : self.i + 1] != '"':
                raise self.fail("Expecting property name enclosed in double quotes")
            key_pos = self.i
            key = self.string()
            if key in out:
                raise self.fail(f"duplicate key '{key}'", key_pos)
            self.ws()
            if self.t[self.i : self.i + 1] != ":":
                raise self.fail("Expecting ':' delimiter")
            self.i += 1
            out[key] = self.value((*path, key))
            self.lines[(*path, key)] = self.line(key_pos)
            self.ws()
            ch = self.t[self.i : self.i + 1]
            self.i += 1
            if ch == "}":
                return out
            if ch != ",":
                self.i -= 1
                raise self.fail("Expecting ',' delimiter")

    def arr(self, path: tuple[Any, ...]) -> list[Any]:
        self.i += 1
        out: list[Any] = []
        self.ws()
        if self.t[self.i : self.i + 1] == "]":
            self.i += 1
            return out
        while True:
            out.append(self.value((*path, len(out))))
            self.ws()
            ch = self.t[self.i : self.i + 1]
            self.i += 1
            if ch == "]":
                return out
            if ch != ",":
                self.i -= 1
                raise self.fail("Expecting ',' delimiter")


def parse_json(text: str) -> tuple[Any, Lines]:
    """``(value, lines)`` where ``lines`` maps a path tuple to its 1-based line."""
    p = _Parser(text)
    value = p.value(())
    p.ws()
    if p.i != len(text):
        raise p.fail("Extra data")
    return value, p.lines
