"""``Verdict``: the one contract every check (deterministic or judge) reports with."""

from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

from ._frozen import FrozenDict, JSONObject, freeze_mapping
from ._validate import check_bool, check_choice, check_number, check_opt_int, check_str, set_field

SCORERS: tuple[str, ...] = ("deterministic", "judge", "classifier")


@dataclass(slots=True, frozen=True)
class Verdict:
    """The result of one check.

    ``reason`` is never empty: explaining *why* is the point of the framework. Step 3's richer
    ``MetricResult`` projects onto this type, and judge verdicts convert the same way.
    """

    dimension: str
    passed: bool
    score: float
    reason: str
    _: KW_ONLY
    step_index: int | None = None
    scorer: Literal["deterministic", "judge", "classifier"] = "deterministic"
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_str(self.dimension, "Verdict.dimension")
        check_bool(self.passed, "Verdict.passed")
        set_field(self, "score", check_number(self.score, "Verdict.score", minimum=0, maximum=1))
        check_str(self.reason, "Verdict.reason")
        check_opt_int(self.step_index, "Verdict.step_index", minimum=0)
        check_choice(self.scorer, "Verdict.scorer", SCORERS)
        set_field(self, "extra", freeze_mapping(self.extra, "Verdict.extra"))
