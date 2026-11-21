"""Golden sets: JSONL files of labelled judge inputs, ``.disseqt/golden/<kind>.jsonl``.

One JSON object per line::

    {"schema_version": "1", "id": "...", "kind": "semantic_arg", "inputs": {...},
     "label": "equivalent", "source": "human", "adjudicated": false, "few_shot": false,
     "tags": [], "note": ""}

``source`` is ``human`` (counts toward gating), ``silver_code`` (labelled by a deterministic rule)
or ``draft`` (written by a tool or an assistant: a starting point for human labelling, never
evidence). Only ``human`` items can make a calibration report gate; an item without a ``source``
is a ``draft``.
"""

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any

from ..kinds import get_kind
from ..templates import get_template

SCHEMA_VERSION = "1"
SOURCES = ("human", "silver_code", "draft")
DEFAULT_SOURCE = "draft"  # an item that does not say who labelled it is never gating evidence
DEFAULT_DIR = Path(".disseqt/golden")


class GoldenError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class GoldenItem:
    id: str
    kind: str
    inputs: Mapping[str, Any]
    label: str
    _: KW_ONLY
    source: str = DEFAULT_SOURCE
    adjudicated: bool = False
    few_shot: bool = False
    tags: tuple[str, ...] = ()
    note: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "id": self.id, "kind": self.kind,
            "inputs": dict(self.inputs), "label": self.label, "source": self.source,
            "adjudicated": self.adjudicated, "few_shot": self.few_shot, "tags": list(self.tags),
            "note": self.note,
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoldenItem":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise GoldenError(f"unsupported golden schema_version {data.get('schema_version')!r}")
        try:
            return cls(
                str(data["id"]),
                str(data["kind"]),
                dict(data["inputs"]),
                str(data["label"]),
                source=str(data.get("source", DEFAULT_SOURCE)),
                adjudicated=bool(data.get("adjudicated", False)),
                few_shot=bool(data.get("few_shot", False)),
                tags=tuple(str(t) for t in data.get("tags", ())),
                note=str(data.get("note", "")),
            )
        except KeyError as exc:
            raise GoldenError(f"golden item is missing {exc.args[0]!r}") from None


def load_golden(path: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(GoldenItem.from_dict(json.loads(line)))
        except (ValueError, GoldenError) as exc:
            raise GoldenError(f"{path}:{number}: {exc}") from None
    return items


def save_golden(path: str | Path, items: Iterable[GoldenItem]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(i.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for i in items
    )
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(target)


def allowed_labels(kind: str, inputs: Mapping[str, Any]) -> tuple[str, ...]:
    spec = get_kind(kind)
    if spec is None:
        raise GoldenError(f"unknown judge kind {kind!r}")
    if spec.labels_for is not None:
        return spec.labels_for(inputs)
    return get_template(kind).label_set


def validate_items(items: Sequence[GoldenItem], kind: str | None = None) -> list[str]:
    """Problems in a golden set (empty means valid)."""
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        where = f"{item.kind}/{item.id}"
        if (item.kind, item.id) in seen:
            problems.append(f"{where}: duplicate id")
        seen.add((item.kind, item.id))
        if kind is not None and item.kind != kind:
            problems.append(f"{where}: kind is not {kind!r}")
        if item.source not in SOURCES:
            problems.append(f"{where}: source {item.source!r} is not one of {list(SOURCES)}")
        spec = get_kind(item.kind)
        if spec is None:
            problems.append(f"{where}: unknown judge kind")
            continue
        missing = [k for k in spec.required if k not in item.inputs]
        if missing:
            problems.append(f"{where}: inputs miss {missing}")
            continue
        try:
            labels = allowed_labels(item.kind, item.inputs)
        except (GoldenError, KeyError) as exc:
            problems.append(f"{where}: {exc}")
            continue
        if item.label == "unsure":
            problems.append(f"{where}: 'unsure' is not a valid truth label (drop the item)")
        elif item.label not in labels:
            problems.append(f"{where}: label {item.label!r} is not one of {list(labels)}")
    return problems


# ---------------------------------------------------------------- labelling workflow (CSV)
CSV_FIELDS = ("id", "kind", "inputs", "rater", "label", "adjudicated_label", "note")


def export_for_labeling(
    items: Sequence[GoldenItem], path: str | Path, *, blind: bool = True
) -> int:
    """Write a CSV for human raters. ``blind`` leaves the label empty so the rater does not anchor
    on an existing (draft) label. Each rater fills ``rater`` and ``label``."""
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, CSV_FIELDS)
        writer.writeheader()
        for i in items:
            writer.writerow({"id": i.id, "kind": i.kind, "inputs": json.dumps(dict(i.inputs), sort_keys=True),
                             "rater": "", "label": "" if blind else i.label,
                             "adjudicated_label": "", "note": i.note})  # fmt: skip
    return len(items)


@dataclass(slots=True, frozen=True)
class ImportResult:
    items: tuple[GoldenItem, ...]
    needs_adjudication: tuple[str, ...]
    skipped: tuple[str, ...]


def import_labels(paths: Sequence[str | Path], *, min_raters: int = 2) -> ImportResult:
    """Merge rater CSVs into human golden items. An item needs ``min_raters`` independent raters;
    if they disagree it needs an ``adjudicated_label`` (then ``adjudicated=True``); otherwise it
    is reported in ``needs_adjudication`` and not produced."""
    rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.setdefault((row["kind"], row["id"]), []).append(row)
    items: list[GoldenItem] = []
    pending: list[str] = []
    skipped: list[str] = []
    for (kind, item_id), group in sorted(rows.items()):
        by_rater = {
            r["rater"].strip(): r for r in group if r["rater"].strip() and r["label"].strip()
        }
        if len(by_rater) < min_raters:
            skipped.append(f"{kind}/{item_id}: {len(by_rater)} of {min_raters} raters")
            continue
        labels = {r["label"].strip() for r in by_rater.values()}
        adjudicated = next(
            (r["adjudicated_label"].strip() for r in group if r["adjudicated_label"].strip()), ""
        )
        if len(labels) == 1:
            label, adj = labels.pop(), False
        elif adjudicated:
            label, adj = adjudicated, True
        else:
            pending.append(f"{kind}/{item_id}: raters disagree {sorted(labels)}")
            continue
        inputs = json.loads(group[0]["inputs"])
        items.append(GoldenItem(item_id, kind, inputs, label, source="human", adjudicated=adj,
                                note=group[0].get("note", "")))  # fmt: skip
    return ImportResult(tuple(items), tuple(pending), tuple(skipped))
