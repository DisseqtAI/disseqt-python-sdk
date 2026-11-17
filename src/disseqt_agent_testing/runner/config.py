"""``SuiteConfig``: every runner knob (frozen, keyword-only, with an ``extra`` bag)."""

import dataclasses
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from ..assertions.config import EvalConfig
from ..models import FrozenDict
from ..models._frozen import JSONObject, freeze_mapping


class SuiteConfigError(ValueError):
    """The runner was configured wrongly (raised before any work starts)."""


@dataclass(slots=True, frozen=True)
class SuiteConfig:
    _: KW_ONLY
    repeats: int | None = None  # force override of every case's n
    default_repeats: int = 1
    max_workers: int = 8
    run_timeout_s: float = 60.0
    case_timeout_s: float | None = None
    suite_timeout_s: float | None = None
    early_stop: bool = False
    seed: int = 0
    incomplete_policy: str = "fail"
    judge_missing: str = "skip"
    judge_concurrency: int | None = 4  # 0 or None: no runner-side limit
    judge_rps: float | None = None
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    fail_fast: bool = False
    keep_reports: str = "failing"
    repeat_mode: str = "internal"
    only_repeat: int | None = None
    run_label: str = ""
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        def bad(name: str, why: str) -> SuiteConfigError:
            return SuiteConfigError(f"SuiteConfig.{name}: {why}")

        for name, minimum in (("repeats", 1), ("default_repeats", 1), ("max_workers", 1)):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < minimum
            ):
                raise bad(name, f"must be an integer >= {minimum}, got {value!r}")
        for name in ("run_timeout_s", "case_timeout_s", "suite_timeout_s"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            ):
                raise bad(name, f"must be > 0, got {value!r}")
        if self.incomplete_policy not in ("fail", "skip"):
            raise bad("incomplete_policy", "must be 'fail' or 'skip'")
        if self.judge_missing not in ("skip", "fail", "run"):
            raise bad("judge_missing", "must be 'skip', 'fail' or 'run'")
        if self.keep_reports not in ("worst", "failing", "all"):
            raise bad("keep_reports", "must be 'worst', 'failing' or 'all'")
        if self.repeat_mode not in ("internal", "split"):
            raise bad("repeat_mode", "must be 'internal' or 'split'")
        if self.early_stop and self.repeat_mode == "split":
            raise SuiteConfigError(
                "DSQ-U03: --disseqt-early-stop cannot be combined with --disseqt-split-repeats"
            )
        if self.judge_concurrency is not None and self.judge_concurrency < 0:
            raise bad("judge_concurrency", "must be >= 0")
        if self.only_repeat is not None and self.only_repeat < 0:
            raise bad("only_repeat", "must be >= 0")
        if not isinstance(self.eval_config, EvalConfig):
            raise bad("eval_config", "must be an EvalConfig")
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "SuiteConfig.extra"))

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: (None if f.name == "eval_config" else getattr(self, f.name))
            for f in dataclasses.fields(self)
            if f.name not in ("extra",)
        } | {
            "extra": dict(self.extra)
        }  # eval_config is not serialized (it holds callables)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuiteConfig":
        names = {f.name for f in dataclasses.fields(cls)} - {"eval_config"}
        return cls(**{k: v for k, v in data.items() if k in names})


def config_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(SuiteConfig))
