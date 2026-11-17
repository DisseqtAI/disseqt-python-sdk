"""``JudgeConfig``: every knob of the engine (frozen, keyword-only, with an ``extra`` bag)."""

import dataclasses
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from typing import Any, TypedDict

from ..models import FrozenDict
from ..models._frozen import JSONObject, freeze_mapping
from ..models._validate import (
    check_bool,
    check_choice,
    check_int,
    check_number,
    check_opt_int,
    check_opt_number,
    fail,
    set_field,
)
from .cache.modes import MODES

REQUIRE_CALIBRATION = ("builtin_or_project", "project", "off")


@dataclass(slots=True, frozen=True)
class KindConfig:
    role: str = "grader"
    prompt_version: str | None = None  # None: the latest built-in
    samples: int = 3  # per order for semantic_arg
    temperature: float | None = None  # None: 0.0 for one sample, 0.7 for several
    min_agreement: float = 2 / 3
    max_tokens: int = 1024
    template: str | None = None

    def __post_init__(self) -> None:
        check_int(self.samples, "KindConfig.samples", minimum=1)
        check_int(self.max_tokens, "KindConfig.max_tokens", minimum=1)
        check_number(self.min_agreement, "KindConfig.min_agreement", minimum=0)
        if self.temperature is not None:
            check_number(self.temperature, "KindConfig.temperature", minimum=0)

    def resolved_temperature(self) -> float:
        if self.temperature is not None:
            return float(self.temperature)
        return 0.0 if self.samples == 1 else 0.7


@dataclass(slots=True, frozen=True)
class JudgeConfig:
    _: KW_ONLY
    roles: Mapping[str, str] = field(default_factory=dict)
    model_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    kinds: Mapping[str, KindConfig] = field(default_factory=dict)
    cache_mode: str = "auto"
    cache_dir: str = ".disseqt/judge_cache"
    allow_requests: bool | None = None
    max_usd: float | None = None
    max_calls: int | None = None
    max_concurrency: int = 4
    # Accepted for forward compatibility but NOT yet enforced: setting it emits a JudgeWarning.
    rpm: int | None = None
    attempt_timeout_s: float = 60.0
    sample_timeout_s: float = 180.0
    max_transport_retries: int = 3
    max_parse_retries: int = 2
    max_judge_calls_per_eval: int = 200
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 30.0
    calibration_dirs: tuple[str, ...] = (".disseqt/calibration",)
    require_calibration: str = "builtin_or_project"
    redact_patterns: tuple[str, ...] = ()
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_choice(self.cache_mode, "JudgeConfig.cache_mode", MODES)
        check_choice(
            self.require_calibration, "JudgeConfig.require_calibration", REQUIRE_CALIBRATION
        )
        if self.allow_requests is not None:
            check_bool(self.allow_requests, "JudgeConfig.allow_requests")
        set_field(self, "max_usd", check_opt_number(self.max_usd, "JudgeConfig.max_usd", minimum=0))
        set_field(
            self, "max_calls", check_opt_int(self.max_calls, "JudgeConfig.max_calls", minimum=0)
        )
        check_int(self.max_concurrency, "JudgeConfig.max_concurrency", minimum=1)
        check_int(self.max_transport_retries, "JudgeConfig.max_transport_retries", minimum=0)
        check_int(self.max_parse_retries, "JudgeConfig.max_parse_retries", minimum=0)
        check_int(self.max_judge_calls_per_eval, "JudgeConfig.max_judge_calls_per_eval", minimum=1)
        for name in ("attempt_timeout_s", "sample_timeout_s", "backoff_base_s", "backoff_cap_s"):
            set_field(
                self, name, check_number(getattr(self, name), f"JudgeConfig.{name}", minimum=0)
            )
        for kind, kc in self.kinds.items():
            if not isinstance(kc, KindConfig):
                fail("JudgeConfig.kinds", f"{kind!r} must map to a KindConfig")
        set_field(self, "roles", MappingProxyType(dict(self.roles)))
        set_field(
            self,
            "model_options",
            MappingProxyType({k: dict(v) for k, v in self.model_options.items()}),
        )
        set_field(self, "kinds", MappingProxyType(dict(self.kinds)))
        set_field(self, "calibration_dirs", tuple(self.calibration_dirs))
        set_field(self, "redact_patterns", tuple(self.redact_patterns))
        set_field(self, "extra", freeze_mapping(self.extra, "JudgeConfig.extra"))

    def kind_config(self, kind: str) -> KindConfig:
        return self.kinds.get(kind, KindConfig())


class JudgeOptions(TypedDict, total=False):
    """Keyword overrides for ``make_judge`` (same keys as ``JudgeConfig``)."""

    roles: Mapping[str, str]
    model_options: Mapping[str, Mapping[str, Any]]
    kinds: Mapping[str, KindConfig]
    cache_mode: str
    cache_dir: str
    allow_requests: bool | None
    max_usd: float | None
    max_calls: int | None
    max_concurrency: int
    rpm: int | None
    attempt_timeout_s: float
    sample_timeout_s: float
    max_transport_retries: int
    max_parse_retries: int
    max_judge_calls_per_eval: int
    backoff_base_s: float
    backoff_cap_s: float
    calibration_dirs: tuple[str, ...]
    require_calibration: str
    redact_patterns: tuple[str, ...]
    extra: JSONObject


def config_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(JudgeConfig))


def resolve_config(config: JudgeConfig | None, options: Mapping[str, Any]) -> JudgeConfig:
    valid = config_field_names()
    for key in options:
        if key not in valid:
            raise TypeError(f"unknown option {key!r}; valid options: {list(valid)}")
    if config is None:
        return JudgeConfig(**options)
    return dataclasses.replace(config, **options) if options else config
