"""DSQ-* codes and their fixed message texts (plan 05 section 12). Consumers handle an unknown
code with a generic line."""

from dataclasses import dataclass


class DisseqtWarning(UserWarning):
    """A runner warning (under pytest the plugin re-raises it as a ``PytestWarning``)."""


@dataclass(frozen=True, slots=True)
class DsqCode:
    code: str
    template: str


_ROWS = [
    ("DSQ-C01", "{path}:{line}: invalid {fmt}: {problem}"),
    (
        "DSQ-C01-EXTRA",
        "{path}: YAML case files need the [yaml] extra: pip install 'disseqt-ai-sdk[yaml]'",
    ),
    ("DSQ-C02", "{path}:{line}: {loc}: {message}"),
    ("DSQ-C03", "{path}:{line}: duplicate case id '{id}' (first defined at {first})"),
    (
        "DSQ-C04",
        "{path}:{line}: case id '{id}' is invalid; use letters, digits, '_', '.', '-' (max 128), starting with a letter or digit",
    ),
    ("DSQ-C05", "{path}:{line}: unknown top-level key '{key}' (allowed: version, defaults, cases)"),
    ("DSQ-C06", "case '{id}': pass_k={k} exceeds repeats={n}"),
    ("DSQ-C07", "case '{id}': pass_k_threshold={t} must be in (0, 1]"),
    (
        "DSQ-C08",
        "tag '{tag}' is invalid or reserved; tags must match [A-Za-z_][A-Za-z0-9_]* and not be one of {reserved}",
    ),
    ("DSQ-C09", "{path}:1: case file version {v} is not supported (supported: 1)"),
    ("DSQ-C10", "{path}:{line}: {detail}"),
    ("DSQ-C11", "{path}:{line}: agent '{value}' is not an import path of the form 'module:attr'"),
    (
        "DSQ-E01",
        "case '{id}' has no agent factory: set 'run: {{agent: module:factory}}' in the case file, implement pytest_disseqt_agent_factory in conftest.py, or pass --disseqt-agent",
    ),
    ("DSQ-E02", "cannot import agent factory '{path}' for case '{id}': {type}: {msg}"),
    ("DSQ-E03", "harness configuration error for case '{id}': {msg}"),
    ("DSQ-E04", "agent factory failed in all {n} repeats of case '{id}': {msg}"),
    (
        "DSQ-E05",
        "case '{id}' sets use_judge=true and --disseqt-require-judge is set, but no judge is configured (implement pytest_disseqt_judge or install a judge plugin)",
    ),
    (
        "DSQ-E06",
        "disseqt internal error while running case '{id}': {type}: {msg} (please report this with the traceback below)",
    ),
    ("DSQ-E07", "expectation configuration error for case '{id}': {msg}"),
    (
        "DSQ-F01",
        "agent case '{id}' FAILED: pass^{k}={v:.2f} < threshold {t:.2f} (c={P}/{D} decided, n={n})",
    ),
    (
        "DSQ-F02",
        "agent case '{id}' FAILED: cannot reach threshold {t:.2f} even if {U} undecided repeats passed (c={P}/{D} decided, n={n})",
    ),
    (
        "DSQ-I01",
        "incomplete: case '{id}': {U} of {n} repeats undecided ({breakdown}); pass^{k} vs threshold {t:.2f} cannot be decided",
    ),
    (
        "DSQ-S01",
        "judge unavailable: case '{id}' sets use_judge=true but no judge is configured (implement pytest_disseqt_judge or pass judge=...; --disseqt-require-judge makes this an error)",
    ),
    ("DSQ-S02", "suite timeout ({s}s) reached before case '{id}' started"),
    ("DSQ-S03", "skipped by case file: {reason}"),
    ("DSQ-S04", "fail-fast: an earlier case failed"),
    ("DSQ-S05", "missing optional dependency '{module}' required by case '{id}'"),
    ("DSQ-U01", "invalid value {value!r} for {option} ({source}): {expected}"),
    ("DSQ-U02", "--disseqt-only-repeat {i} is out of range for case '{id}' with {n} repeats"),
    ("DSQ-U03", "--disseqt-early-stop cannot be combined with --disseqt-split-repeats"),
    (
        "DSQ-W01",
        "pytest-rerunfailures is active: reruns are disabled on agent items (rerun-until-pass biases pass^k upward)",
    ),
    (
        "DSQ-W02",
        "the flaky plugin is active: reruns are disabled on agent items (rerun-until-pass biases pass^k upward)",
    ),
    (
        "DSQ-W03",
        "pytest-repeat is active: use --disseqt-repeats instead (agent cases run their repeats internally)",
    ),
    ("DSQ-W04", "--disseqt-repeats {n} is smaller than pass_k={k}; using k={n}"),
    (
        "DSQ-W05",
        "case '{id}' may need {need:.0f}s but pytest-timeout is {limit:.0f}s; the thread method kills the process with no teardown",
    ),
    (
        "DSQ-W06",
        "split mode without --dist loadgroup: repeats of one case may land on different workers",
    ),
    (
        "DSQ-W07",
        "early stopping marks case '{id}' comparable=false (optional stopping biases pass^k)",
    ),
    (
        "DSQ-W08",
        "an on_event callback raised {type}: {msg}; it is disabled for the rest of the run",
    ),
    (
        "DSQ-W09",
        "{n} worker threads still running at session end (agent runs that ignored their timeout)",
    ),
    ("DSQ-W10", "case file {path} contains no cases"),
    (
        "DSQ-W11",
        "judge concurrency {limit} is below the xdist worker count {workers}; using 1 per worker",
    ),
]
CODES: dict[str, DsqCode] = {c: DsqCode(c, t) for c, t in _ROWS}


def format_code(code: str, **fields: object) -> str:
    """The message for ``code``; an unknown code renders as a generic line."""
    spec = CODES.get(code)
    if spec is None:
        return f"{code}: {fields}" if fields else code
    try:
        return spec.template.format(**fields)
    except (KeyError, IndexError, ValueError):
        return f"{code}: {fields}"
