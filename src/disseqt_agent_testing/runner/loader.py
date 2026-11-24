"""Case-file loading: JSON and YAML (the ``[yaml]`` extra), datasets, duplicate detection and
selection. ``load_cases`` never raises for file problems; it returns them as records."""

import csv
import json
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import AgentTestCase, ModelValidationError, from_dict
from ..models.test_case import ID_PATTERN, MAX_CASE_ID, RESERVED_TAGS, TAG_PATTERN
from . import _yaml
from ._json import JsonProblem, Lines, parse_json
from .errors import format_code
from .filter import CaseFilter
from .models import CaseSource, CaseSpec, CollectionErrorRecord, RunBlock

TOP_KEYS = ("version", "defaults", "cases")
CASE_FILE_GLOBS = ("*.agent.yaml", "*.agent.yml", "*.agent.json")
MAX_DATASET_ROWS = 10_000
ROW_STRIDE = 10_000
AGENT_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
PLACEHOLDER = re.compile(r"\$\$\{|\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

CaseFormat = Callable[[str], tuple[Any, Lines]]
DatasetReader = Callable[[Path], list[dict[str, Any]]]
_LOCK = threading.RLock()
_FORMATS: dict[str, CaseFormat] = {}
_DATASETS: dict[str, DatasetReader] = {}


def register_case_format(ext: str, parse: CaseFormat, *, override: bool = False) -> None:
    """``parse(text) -> (value, lines)`` for files ending in ``ext`` (for example ``.toml``)."""
    with _LOCK:
        if ext in _FORMATS and not override:
            raise ValueError(f"case format {ext!r} is already registered")
        _FORMATS[ext] = parse


def unregister_case_format(ext: str) -> None:
    with _LOCK:
        _FORMATS.pop(ext, None)


def register_dataset_format(ext: str, reader: DatasetReader, *, override: bool = False) -> None:
    with _LOCK:
        if ext in _DATASETS and not override:
            raise ValueError(f"dataset format {ext!r} is already registered")
        _DATASETS[ext] = reader


def unregister_dataset_format(ext: str) -> None:
    with _LOCK:
        _DATASETS.pop(ext, None)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"line {n}: invalid JSON ({exc})") from None
            if not isinstance(row, dict):
                raise ValueError(f"line {n}: each row must be a JSON object")
            rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


register_case_format(".json", parse_json, override=True)
register_case_format(".yaml", lambda text: _yaml.parse_yaml(text), override=True)
register_case_format(".yml", lambda text: _yaml.parse_yaml(text), override=True)
register_dataset_format(".jsonl", _read_jsonl, override=True)
register_dataset_format(".csv", _read_csv, override=True)


@dataclass(frozen=True, slots=True)
class LoadedSuite:
    cases: tuple[CaseSpec, ...] = ()
    errors: tuple[CollectionErrorRecord, ...] = ()
    deselected: tuple[str, ...] = ()
    files: tuple[str, ...] = field(default=())


def format_errors(errors: Iterable[CollectionErrorRecord], limit: int = 20) -> list[str]:
    """One line per error; at most ``limit`` per file, then ``... N more errors``."""
    by_path: dict[str | None, list[CollectionErrorRecord]] = {}
    for e in errors:
        by_path.setdefault(e.path, []).append(e)
    lines: list[str] = []
    for recs in by_path.values():
        lines.extend(e.message for e in recs[:limit])
        if len(recs) > limit:
            lines.append(f"... {len(recs) - limit} more errors")
    return lines


# ------------------------------------------------------------------------------- substitution
def _substitute(value: Any, row: Mapping[str, Any], loc: str, missing: list[str]) -> Any:
    if isinstance(value, str):
        whole = PLACEHOLDER.fullmatch(value)
        if whole and whole.group(1):
            name = whole.group(1)
            if name not in row:
                missing.append(f"unknown field ${{{name}}} in {loc}")
                return value
            return row[name]  # a whole-value placeholder keeps the row value's JSON type

        def repl(m: re.Match[str]) -> str:
            if m.group(1) is None:
                return "${"  # "$${" escapes a literal "${"
            name = m.group(1)
            if name not in row:
                missing.append(f"unknown field ${{{name}}} in {loc}")
                return m.group(0)
            return str(row[name])

        return PLACEHOLDER.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute(v, row, f"{loc}.{k}", missing) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, row, f"{loc}[{i}]", missing) for i, v in enumerate(value)]
    return value


def _has_placeholder(text: str) -> bool:
    return any(m.group(1) for m in PLACEHOLDER.finditer(text))


# ------------------------------------------------------------------------------- one file
def _line_for(lines: Lines, path: tuple[Any, ...]) -> int | None:
    for end in range(len(path), -1, -1):
        if path[:end] in lines:
            return lines[path[:end]]
    return None


def _split_loc(loc: str) -> tuple[Any, ...]:
    """``AgentTestCase.expect.tools[1].name`` -> ``("expect", "tools", 1, "name")``."""
    body = loc.split(".", 1)[1] if "." in loc else ""
    out: list[Any] = []
    for part in re.finditer(r"([A-Za-z0-9_\-]+)|\[(\d+)\]", body):
        out.append(part.group(1) if part.group(1) is not None else int(part.group(2)))
    return tuple(out)


def _merge_defaults(defaults: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(entry)
    for key, value in defaults.items():
        own = entry.get(key)
        if key == "tags":
            if "tags" not in entry or isinstance(own, list):
                both = [*value, *(own or ())]
                merged["tags"] = (
                    sorted(set(both)) if all(isinstance(t, str) for t in both) else both
                )
        elif key == "run" and isinstance(value, Mapping):
            if own is None or isinstance(own, Mapping):
                merged["run"] = {**value, **(own or {})}
        else:
            merged.setdefault(key, value)
    return merged


def _shim(data: dict[str, Any]) -> dict[str, Any]:
    """Case-file conveniences that are not part of plan 01's encoding: ``fakes`` as a mapping
    (the name comes from the key); the list form passes through unchanged."""
    fakes = data.get("fakes")
    if isinstance(fakes, Mapping):
        data["fakes"] = [
            {
                "name": name,
                **(spec if isinstance(spec, Mapping) else {"default": {"returns": spec}}),
            }
            for name, spec in fakes.items()
        ]
    return data


@dataclass(slots=True)
class _Ctx:
    path: str
    lines: Lines
    errors: list[CollectionErrorRecord]
    registry: dict[str, str]

    def err(self, code: str, line: int | None, **fields: Any) -> None:
        text = format_code(code, path=self.path, line=line or 1, **fields)
        self.errors.append(CollectionErrorRecord(self.path, text, line=line, code=code))


def _check_entry(ctx: _Ctx, data: dict[str, Any], line: int | None, at: tuple[Any, ...]) -> bool:
    """Plan 05's own validations, with the DSQ codes (the models then check everything else)."""
    ok = True
    cid = data.get("id")
    if (
        not isinstance(cid, str)
        or not ID_PATTERN.match(cid)
        or len(cid) > MAX_CASE_ID
        or cid.endswith("\n")
    ):
        ctx.err("DSQ-C04", _line_for(ctx.lines, (*at, "id")) or line, id=cid)
        ok = False
    reserved = list(RESERVED_TAGS)
    tags = data.get("tags")
    for tag in tags if isinstance(tags, (list, tuple)) else ():
        if not isinstance(tag, str) or not TAG_PATTERN.fullmatch(tag) or tag in RESERVED_TAGS:
            ctx.errors.append(
                CollectionErrorRecord(
                    ctx.path,
                    f"{ctx.path}:{line or 1}: " + format_code("DSQ-C08", tag=tag, reserved=reserved),
                    line=line, code="DSQ-C08",
                )  # fmt: skip
            )
            ok = False
    repeats, pass_k = data.get("repeats"), data.get("pass_k")
    if isinstance(repeats, int) and isinstance(pass_k, int) and pass_k > repeats:
        ctx.errors.append(
            CollectionErrorRecord(
                ctx.path,
                f"{ctx.path}:{line or 1}: " + format_code("DSQ-C06", id=cid, k=pass_k, n=repeats),
                line=line,
                code="DSQ-C06",
            )
        )
        ok = False
    t = data.get("pass_k_threshold")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t <= 1):
        ctx.errors.append(
            CollectionErrorRecord(
                ctx.path,
                f"{ctx.path}:{line or 1}: " + format_code("DSQ-C07", id=cid, t=t),
                line=line,
                code="DSQ-C07",
            )
        )
        ok = False
    return ok


def _decode_case(
    ctx: _Ctx, merged: dict[str, Any], at: tuple[Any, ...], line: int | None, source: CaseSource
) -> CaseSpec | None:
    data = _shim({k: v for k, v in merged.items() if k != "run"})
    run_raw = merged.get("run")
    if not _check_entry(ctx, data, line, at):
        return None
    run = RunBlock()
    if run_raw is not None:
        run_line = _line_for(ctx.lines, (*at, "run")) or line
        try:
            if not isinstance(run_raw, Mapping):
                raise ValueError("RunBlock: expected a mapping")
            run = RunBlock.from_dict(run_raw)
        except (ValueError, TypeError) as exc:
            ctx.err(
                "DSQ-C02", run_line, loc=f"{'.'.join(str(p) for p in at)}.run", message=str(exc)
            )
            return None
        if run.agent is not None and not AGENT_PATH.match(run.agent):
            ctx.err("DSQ-C11", run_line, value=run.agent)
            return None
    try:
        case = from_dict(AgentTestCase, data)
    except ModelValidationError as exc:
        text = str(exc)
        loc, _, message = text.partition(": ")
        path = (*at, *_split_loc(loc))
        shown = ".".join(str(p) if not isinstance(p, int) else "" for p in path).replace(".[", "[")
        shown = "".join(
            f"[{p}]" if isinstance(p, int) else (f".{p}" if i else str(p))
            for i, p in enumerate(path)
        )
        ctx.err("DSQ-C02", _line_for(ctx.lines, path) or line, loc=shown, message=message or text)
        return None
    return CaseSpec(case, source, run)


def _expand(
    ctx: _Ctx, entry: dict[str, Any], at: tuple[Any, ...], base: Path, line: int | None
) -> list[tuple[dict[str, Any], int | None, int]]:
    """A template with ``rows`` / ``dataset`` becomes one entry per row; a plain entry is itself."""
    if "rows" not in entry and "dataset" not in entry:
        return [(entry, line, 0)]
    template = {k: v for k, v in entry.items() if k not in ("rows", "dataset")}
    label = "rows"
    if "rows" in entry:
        rows = entry["rows"]
        if not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
            ctx.err("DSQ-C10", line, detail="rows must be a list of mappings")
            return []
    else:
        label = str(entry["dataset"])
        target = (base / label).resolve()
        if not Path(label).is_absolute() and not target.is_relative_to(base.resolve()):
            ctx.err(
                "DSQ-C10",
                line,
                detail=f"dataset {label}: a relative dataset path must stay inside the case file's "
                "directory (use an absolute path to read a file elsewhere)",
            )
            return []
        reader = _DATASETS.get(target.suffix)
        if reader is None:
            ctx.err(
                "DSQ-C10",
                line,
                detail=f"dataset {label}: unsupported format {target.suffix!r} (known: {sorted(_DATASETS)})",
            )
            return []
        try:
            rows = reader(target)
        except (OSError, ValueError, csv.Error) as exc:
            ctx.err("DSQ-C10", line, detail=f"dataset {label}: {exc}")
            return []
        except RecursionError:
            ctx.err("DSQ-C10", line, detail=f"dataset {label}: a row is nested too deeply")
            return []
    if len(rows) > MAX_DATASET_ROWS:
        ctx.err("DSQ-C10", line, detail=f"{label} has {len(rows)} rows (max {MAX_DATASET_ROWS})")
        return []
    if not isinstance(template.get("id"), str) or not _has_placeholder(template["id"]):
        ctx.err(
            "DSQ-C10",
            line,
            detail=f"template id '{template.get('id')}' has no ${{...}} placeholder",
        )
        return []
    out: list[tuple[dict[str, Any], int | None, int]] = []
    for r, row in enumerate(rows):
        missing: list[str] = []
        built = _substitute(template, row, "case", missing)
        if missing:
            for m in missing:
                ctx.err("DSQ-C10", line, detail=f"{label} row {r}: {m}")
            continue
        out.append((built, line, r))
    return out


def load_file(
    path: Path, registry: dict[str, str] | None = None, *, display: str | None = None
) -> tuple[list[CaseSpec], list[CollectionErrorRecord]]:
    shown = display or str(path)
    reg = registry if registry is not None else {}
    ctx = _Ctx(shown, {}, [], reg)
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml") and not _yaml.available():
        ctx.errors.append(
            CollectionErrorRecord(shown, format_code("DSQ-C01-EXTRA", path=shown), code="DSQ-C01")
        )
        return [], ctx.errors
    parse = _FORMATS.get(ext)
    if parse is None:
        ctx.err(
            "DSQ-C01",
            1,
            fmt="case file",
            problem=f"unsupported extension {ext!r} (known: {sorted(_FORMATS)})",
        )
        return [], ctx.errors
    try:
        text = path.read_text(encoding="utf-8")
        doc, ctx.lines = parse(text)
    except (OSError, UnicodeDecodeError) as exc:
        ctx.err("DSQ-C01", 1, fmt="case file", problem=str(exc))
        return [], ctx.errors
    except JsonProblem as exc:
        fmt = "YAML" if ext in (".yaml", ".yml") else "JSON"
        ctx.err("DSQ-C01", exc.line, fmt=fmt, problem=exc.problem)
        return [], ctx.errors
    except RecursionError:
        ctx.err("DSQ-C01", 1, fmt="case file", problem="the document is nested too deeply")
        return [], ctx.errors
    except ValueError as exc:
        ctx.err("DSQ-C01", 1, fmt="case file", problem=str(exc))
        return [], ctx.errors
    if not isinstance(doc, dict):
        ctx.err(
            "DSQ-C02",
            1,
            loc="<root>",
            message="a case file must be a mapping with 'version' and 'cases'",
        )
        return [], ctx.errors
    for key in doc:
        if key not in TOP_KEYS:
            ctx.err("DSQ-C05", _line_for(ctx.lines, (key,)), key=key)
    version = doc.get("version")
    if version != 1 or isinstance(version, bool):
        ctx.errors.append(
            CollectionErrorRecord(
                shown, format_code("DSQ-C09", path=shown, v=version), line=1, code="DSQ-C09"
            )
        )
        return [], ctx.errors
    defaults = doc.get("defaults") or {}
    if not isinstance(defaults, dict):
        ctx.err(
            "DSQ-C02",
            _line_for(ctx.lines, ("defaults",)),
            loc="defaults",
            message="expected a mapping",
        )
        defaults = {}
    if "tags" in defaults and not isinstance(defaults["tags"], list):
        ctx.err(
            "DSQ-C02",
            _line_for(ctx.lines, ("defaults", "tags")),
            loc="defaults.tags",
            message=f"expected a list, got {type(defaults['tags']).__name__}",
        )
        defaults = {k: v for k, v in defaults.items() if k != "tags"}
    entries = doc.get("cases")
    if not isinstance(entries, list):
        ctx.err(
            "DSQ-C02",
            _line_for(ctx.lines, ("cases",)),
            loc="cases",
            message="expected a list of cases",
        )
        return [], ctx.errors
    if not entries:
        ctx.errors.append(
            CollectionErrorRecord(shown, format_code("DSQ-W10", path=shown), code="DSQ-W10")
        )
    specs: list[CaseSpec] = []
    for i, entry in enumerate(entries):
        at: tuple[Any, ...] = ("cases", i)
        line = ctx.lines.get(at)
        if not isinstance(entry, dict):
            ctx.err("DSQ-C02", line, loc=f"cases[{i}]", message="expected a mapping")
            continue
        for built, entry_line, r in _expand(ctx, entry, at, path.parent, line):
            merged = _merge_defaults(defaults, built)
            source = CaseSource(
                shown,
                line=entry_line,
                index=i * ROW_STRIDE + r,
                nodeid=f"{shown}::{merged.get('id')}",
            )
            spec = _decode_case(ctx, merged, at, entry_line, source)
            if spec is None:
                continue
            first = reg.get(spec.case.id)
            if first is not None:
                where = first.split(":", 1)[1] if first.startswith(shown + ":") else first
                ctx.err("DSQ-C03", entry_line, id=spec.case.id, first=where)
                continue
            reg[spec.case.id] = f"{shown}:{entry_line or 1}"
            specs.append(spec)
    return specs, ctx.errors


def discover(root: Path, globs: Iterable[str] = CASE_FILE_GLOBS) -> list[Path]:
    if root.is_file():
        return [root]
    found: set[Path] = set()
    for pattern in globs:
        found.update(p for p in root.rglob(pattern) if p.is_file())
    return sorted(found)


def load_cases(
    source: Any,
    *,
    filter: CaseFilter | None = None,
    root: str | Path | None = None,
    registry: dict[str, str] | None = None,
) -> LoadedSuite:
    """Load case files, directories, ``AgentTestCase`` / ``CaseSpec`` objects or a mix of them.

    Never raises for file problems: they come back as ``LoadedSuite.errors``. The first
    definition of an id wins; a later one is the error (DSQ-C03).
    """
    items = source if isinstance(source, (list, tuple)) else [source]
    base = Path(root) if root is not None else None
    reg = registry if registry is not None else {}
    specs: list[CaseSpec] = []
    errors: list[CollectionErrorRecord] = []
    files: list[str] = []
    for n, item in enumerate(items):
        if isinstance(item, (str, Path)):
            p = Path(item)
            if base is not None and not p.is_absolute():
                p = base / p
            if not p.exists():
                errors.append(
                    CollectionErrorRecord(
                        str(item), f"{item}: no such file or directory", code="DSQ-C01"
                    )
                )
                continue
            for file in discover(p):
                shown = (
                    str(file.relative_to(base))
                    if base is not None and base in file.parents
                    else str(file)
                )
                files.append(shown)
                got, errs = load_file(file, reg, display=shown)
                specs.extend(got)
                errors.extend(errs)
        else:
            if isinstance(item, CaseSpec):
                spec = item
            elif isinstance(item, AgentTestCase):
                spec = CaseSpec(item, CaseSource(None, index=n))
            else:
                errors.append(
                    CollectionErrorRecord(
                        None,
                        f"expected an AgentTestCase or CaseSpec, got {type(item).__name__}",
                        code="DSQ-C02",
                    )
                )
                continue
            first = reg.get(spec.case.id)
            if first is not None:
                errors.append(
                    CollectionErrorRecord(
                        None,
                        format_code(
                            "DSQ-C03", path="<python>", line=0, id=spec.case.id, first=first
                        ),
                        code="DSQ-C03",
                    )
                )
                continue
            reg[spec.case.id] = f"<python>:{n}"
            specs.append(spec)
    kept: list[CaseSpec] = []
    dropped: list[str] = []
    for spec in specs:
        if (
            filter is not None
            and filter.active
            and not filter.matches(spec.case.id, spec.case.tags)
        ):
            dropped.append(spec.case.id)
        else:
            kept.append(spec)
    return LoadedSuite(tuple(kept), tuple(errors), tuple(dropped), tuple(files))
