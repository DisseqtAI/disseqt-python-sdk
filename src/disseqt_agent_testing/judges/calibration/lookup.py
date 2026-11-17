"""Gating state: a judge may fail a case only when its exact identity has a calibration report
that cleared the thresholds. No report, a failing report or a disallowed source is ``advisory``."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import JudgeConfig
from ..kinds.base import JudgeKindSpec

BUILTIN_DIR = Path(__file__).with_name("builtin")


@dataclass(slots=True, frozen=True)
class CalibrationStatus:
    gating: str  # "gate" | "advisory"
    status: str  # "calibrated" | "none" | "failing" | "off" | "never_gates"
    source: str = ""
    report_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "report_id": self.report_id, "source": self.source}


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def lookup_calibration(
    spec: JudgeKindSpec, identity_hash: str, config: JudgeConfig
) -> CalibrationStatus:
    if spec.never_gates:
        return CalibrationStatus("advisory", "never_gates")
    mode = config.require_calibration
    if mode == "off":
        return CalibrationStatus("gate", "off")
    sources: list[tuple[str, Path]] = [("project", Path(d)) for d in config.calibration_dirs]
    if mode == "builtin_or_project" and spec.builtin_calibratable:
        sources.append(("builtin", BUILTIN_DIR))
    seen_failing = False
    for source, root in sources:
        report = _read(root / spec.name / f"{identity_hash}.json")
        if report is None:
            continue
        if report.get("gating_allowed") is True:
            return CalibrationStatus("gate", "calibrated", source, identity_hash)
        seen_failing = True
    return CalibrationStatus("advisory", "failing" if seen_failing else "none")
