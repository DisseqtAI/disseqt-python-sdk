"""``EvalConfig``: every knob of ``evaluate`` (frozen, keyword-only, with an ``extra`` bag)."""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from typing import Any, TypedDict

from ..models import FrozenDict
from ..models._frozen import JSONObject, freeze_mapping
from ..models._validate import check_bool, check_choice, check_number, fail, set_field


class ExpectConfigError(ValueError):
    """The test itself is wrong (unknown ``min_scores`` key, bad predicate name, Ref cycle...)."""


@dataclass(slots=True, frozen=True)
class Lexicons:
    """Per-project additions to the built-in language pack."""

    _: KW_ONLY
    refusal_extra: tuple[str, ...] = ()
    refusal_exclude: tuple[str, ...] = ()
    closers_extra: tuple[str, ...] = ()
    question_cues_extra: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class EvalConfig:
    _: KW_ONLY
    use_judge: bool = False
    on_uncalibrated: str = "warn"
    triage: bool = False
    on_unavailable: str = "fail"
    strict: bool = False
    assignment_backend: str = "auto"
    min_pair_score: float = 0.0
    decision_source: str | None = None
    judge_confidence_threshold: float = 0.75
    lang: str = "en"
    lexicons: Lexicons = field(default_factory=Lexicons)
    route_patterns: tuple[str, ...] = ("transfer_to_*",)
    predicates: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    extra_metrics: tuple[str, ...] = ()
    max_report_lines: int | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_bool(self.use_judge, "EvalConfig.use_judge")
        check_bool(self.triage, "EvalConfig.triage")
        check_bool(self.strict, "EvalConfig.strict")
        check_choice(self.on_uncalibrated, "EvalConfig.on_uncalibrated", ("warn", "incomplete"))
        check_choice(self.on_unavailable, "EvalConfig.on_unavailable", ("fail", "warn", "skip"))
        if not isinstance(self.assignment_backend, str) or not self.assignment_backend:
            fail("EvalConfig.assignment_backend", "must be a backend name")
        set_field(
            self,
            "min_pair_score",
            check_number(self.min_pair_score, "EvalConfig.min_pair_score", minimum=0),
        )
        set_field(
            self,
            "judge_confidence_threshold",
            check_number(
                self.judge_confidence_threshold, "EvalConfig.judge_confidence_threshold", minimum=0
            ),
        )
        if self.decision_source is not None and not isinstance(self.decision_source, str):
            fail("EvalConfig.decision_source", "must be a string or None")
        if not isinstance(self.lexicons, Lexicons):
            fail("EvalConfig.lexicons", "must be a Lexicons")
        set_field(self, "route_patterns", tuple(self.route_patterns))
        preds = dict(self.predicates)
        for name, fn in preds.items():
            if not isinstance(name, str) or not callable(fn):
                fail("EvalConfig.predicates", "must map names to callables")
        set_field(self, "predicates", MappingProxyType(preds))
        set_field(self, "extra_metrics", tuple(self.extra_metrics))
        if self.max_report_lines is not None and (
            isinstance(self.max_report_lines, bool) or not isinstance(self.max_report_lines, int)
        ):
            fail("EvalConfig.max_report_lines", "must be an int or None")
        set_field(self, "extra", freeze_mapping(self.extra, "EvalConfig.extra"))


class EvalOptions(TypedDict, total=False):
    """Keyword overrides accepted by ``evaluate`` (same keys as ``EvalConfig``)."""

    use_judge: bool
    on_uncalibrated: str
    triage: bool
    on_unavailable: str
    strict: bool
    assignment_backend: str
    min_pair_score: float
    decision_source: str | None
    judge_confidence_threshold: float
    lang: str
    lexicons: Lexicons
    route_patterns: tuple[str, ...]
    predicates: Mapping[str, Callable[..., Any]]
    extra_metrics: tuple[str, ...]
    max_report_lines: int | None
    extra: JSONObject


def config_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(EvalConfig))


def resolve_config(config: EvalConfig | None, options: Mapping[str, Any]) -> EvalConfig:
    """``config`` overlaid with keyword ``options`` (keywords win); unknown keys are an error."""
    valid = config_field_names()
    for key in options:
        if key not in valid:
            raise TypeError(f"unknown option {key!r}; valid options: {list(valid)}")
    if config is None:
        return EvalConfig(**options)
    return dataclasses.replace(config, **options) if options else config
