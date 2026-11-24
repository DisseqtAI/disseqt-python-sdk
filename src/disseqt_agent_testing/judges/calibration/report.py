"""``CalibrationReport``: how well one exact judge identity agrees with human labels, and whether
that is good enough to let it fail a case (``gating_allowed``)."""

import json
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"

CALIBRATION_FAMILIES: dict[str, dict[str, float]] = {
    "binary": {
        "min_tpr": 0.85, "min_tnr": 0.85, "min_kappa": 0.60, "max_abstain": 0.15,
        "min_human_per_class": 30, "min_order_consistency": 0.90,
    },
    "multiclass": {
        "min_macro_f1": 0.80, "min_class_recall": 0.70, "min_support_for_recall": 15,
        "min_kappa": 0.60, "max_abstain": 0.15, "min_human_items": 100,
    },
}  # fmt: skip


def _low(value: float | None, floor: float) -> bool:
    return value is None or value < floor


def gate_failures(
    family: str,
    stats: Mapping[str, Any],
    *,
    abstain_rate: float | None,
    human_counts: Mapping[str, int],
    positive: str | None,
    order_consistency: float | None = None,
    needs_order_consistency: bool = False,
) -> list[str]:
    """Every threshold the report misses (empty means gating is allowed). Computed on the held-out
    test split and human-labelled items only; a missing statistic fails closed."""
    t = CALIBRATION_FAMILIES[family]
    out: list[str] = []
    if abstain_rate is None or abstain_rate > t["max_abstain"]:
        out.append(f"abstain rate {abstain_rate} is above {t['max_abstain']}")
    if _low(stats.get("kappa"), t["min_kappa"]):
        out.append(f"kappa {stats.get('kappa')} is below {t['min_kappa']}")
    if family == "binary":
        pos = human_counts.get(positive or "", 0)
        neg = sum(v for k, v in human_counts.items() if k != positive)
        for name, n in (("positive", pos), ("negative", neg)):
            if n < t["min_human_per_class"]:
                out.append(
                    f"only {n} human {name} test items (need {int(t['min_human_per_class'])})"
                )
        for key, floor in (("tpr", t["min_tpr"]), ("tnr", t["min_tnr"])):
            if _low(stats.get(key), floor):
                out.append(f"{key} {stats.get(key)} is below {floor}")
        if needs_order_consistency and _low(order_consistency, t["min_order_consistency"]):
            out.append(
                f"order consistency {order_consistency} is below {t['min_order_consistency']}"
            )
    else:
        total = sum(human_counts.values())
        if total < t["min_human_items"]:
            out.append(f"only {total} human test items (need {int(t['min_human_items'])})")
        if _low(stats.get("macro_f1"), t["min_macro_f1"]):
            out.append(f"macro-F1 {stats.get('macro_f1')} is below {t['min_macro_f1']}")
        for cls, row in (stats.get("per_class") or {}).items():
            if (row.get("support") or 0) >= t["min_support_for_recall"] and _low(
                row.get("recall"), t["min_class_recall"]
            ):
                out.append(
                    f"recall for {cls!r} is {row.get('recall')}, below {t['min_class_recall']}"
                )
    return out


@dataclass(slots=True, frozen=True)
class CalibrationReport:
    SCHEMA: str = field(default=SCHEMA_VERSION, init=False)
    kind: str = ""
    identity_hash: str = ""
    identity: Mapping[str, Any] = field(default_factory=dict)
    _: KW_ONLY
    family: str = "binary"
    positive: str | None = None
    samples: int = 1
    gating_allowed: bool = False
    failures: tuple[str, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)  # held-out test split, human only
    ci: Mapping[str, Any] = field(default_factory=dict)
    abstain_rate: float | None = None
    order_consistency: float | None = None
    counts: Mapping[str, Any] = field(default_factory=dict)
    silver: Mapping[str, Any] = field(default_factory=dict)  # draft/silver items: never gate
    errors: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "kind": self.kind, "identity_hash": self.identity_hash,
            "identity": dict(self.identity), "family": self.family, "positive": self.positive,
            "samples": self.samples, "gating_allowed": self.gating_allowed,
            "failures": list(self.failures), "stats": dict(self.stats), "ci": dict(self.ci),
            "abstain_rate": self.abstain_rate, "order_consistency": self.order_consistency,
            "counts": dict(self.counts), "silver": dict(self.silver), "errors": self.errors,
            "cost_usd": round(self.cost_usd, 6), "tokens": self.tokens,
            "warnings": list(self.warnings), "thresholds": CALIBRATION_FAMILIES[self.family],
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationReport":
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema_version {d.get('schema_version')!r}")
        return cls(
            d["kind"], d["identity_hash"], d.get("identity", {}), family=d["family"],
            positive=d.get("positive"), samples=int(d.get("samples", 1)),
            gating_allowed=bool(d["gating_allowed"]), failures=tuple(d.get("failures", ())),
            stats=d.get("stats", {}), ci=d.get("ci", {}), abstain_rate=d.get("abstain_rate"),
            order_consistency=d.get("order_consistency"), counts=d.get("counts", {}),
            silver=d.get("silver", {}), errors=int(d.get("errors", 0)),
            cost_usd=float(d.get("cost_usd", 0.0)), tokens=int(d.get("tokens", 0)),
            warnings=tuple(d.get("warnings", ())),
        )  # fmt: skip


def write_report(report: CalibrationReport, root: str | Path) -> Path:
    """``<root>/<kind>/<identity_hash>.json``, the exact path the engine's gate lookup reads."""
    path = Path(root) / report.kind / f"{report.identity_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_reports(root: str | Path) -> list[CalibrationReport]:
    out: list[CalibrationReport] = []
    for path in sorted(Path(root).glob("*/*.json")):
        if path.name.startswith("_"):
            continue
        try:
            out.append(CalibrationReport.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return out
