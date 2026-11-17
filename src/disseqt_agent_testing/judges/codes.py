"""Judge error codes (plan 04 section 13): retry / cache flags and exact messages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    retried: bool
    cached: bool
    template: str


_SPECS = [
    ErrorSpec(
        "auth_config",
        False,
        False,
        "judge {model}: authentication or configuration error ({detail})",
    ),
    ErrorSpec(
        "extra_missing",
        False,
        False,
        "judge {model}: provider package missing; install 'disseqt-agent-testing[{extra}]'",
    ),
    ErrorSpec(
        "network_forbidden",
        False,
        False,
        "judge {model}: live judge requests are disabled (DISSEQT_ALLOW_JUDGE_REQUESTS=0)",
    ),
    ErrorSpec(
        "cache_miss_offline",
        False,
        False,
        "judge {kind}: no cached response for sample {i} in replay mode (key {key12}); run with DISSEQT_JUDGE_CACHE=record",
    ),
    ErrorSpec("cache_corrupt", False, False, "judge cache entry {path} is invalid: {detail}"),
    ErrorSpec(
        "budget_exceeded",
        False,
        False,
        "judge budget exhausted: {what} (spent ${spent:.4f} of ${cap:.4f}; {calls} calls)",
    ),
    ErrorSpec("rate_limited", True, False, "judge {model}: rate limited after {n} attempts"),
    ErrorSpec("timeout", True, False, "judge {model}: timed out after {secs:.0f}s ({n} attempts)"),
    ErrorSpec("provider_error", True, False, "judge {model}: provider error {status}: {detail}"),
    ErrorSpec("refusal", False, False, "judge {model} refused to evaluate"),
    ErrorSpec("content_filter", False, False, "judge {model}: output blocked by content filter"),
    ErrorSpec("truncated", False, False, "judge {model}: output truncated at max_tokens={n}"),
    ErrorSpec(
        "parse_error", True, False, "judge {kind}: unparseable output after {n} re-asks: {detail}"
    ),
    ErrorSpec(
        "invalid_verdict", True, False, "judge {kind}: invalid label '{label}' (allowed: {labels})"
    ),
    ErrorSpec(
        "insufficient_votes",
        False,
        False,
        "judge {kind}: only {valid} of {n} samples valid (errors: {codes})",
    ),
    ErrorSpec("input_missing", False, False, "judge {kind}: invalid inputs: {detail}"),
    ErrorSpec("unknown_kind", False, False, "unknown judge kind '{kind}' (known: {kinds})"),
    ErrorSpec(
        "template_unknown", False, False, "no template version '{version}' (available: {versions})"
    ),
]
SPECS: dict[str, ErrorSpec] = {s.code: s for s in _SPECS}
ORDER: tuple[str, ...] = tuple(s.code for s in _SPECS)  # tie-break order when quorum fails


def message(code: str, **fields: object) -> str:
    """The exact message for a code (an unknown code renders as its name)."""
    spec = SPECS.get(code)
    if spec is None:
        return code
    try:
        return spec.template.format(**fields)
    except (KeyError, IndexError, ValueError):
        return f"{code}: {fields}"


class JudgeConfigError(ValueError):
    """``make_judge`` rejected the setup: ``bad_model_string``, ``unknown_provider``,
    ``price_unknown``, ``template_hash_mismatch`` or ``no_model``."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
