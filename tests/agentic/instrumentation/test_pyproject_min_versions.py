"""
Regression: the floors declared in ``[project.optional-dependencies]``
must be >= each instrumentor's runtime ``min_version``.

Before TP-2128 P1 #1.5, pip extras claimed e.g. ``openai>=1.0.0`` while
``OpenAIInstrumentor.min_version == "1.50.0"``. A user installing
``pip install "disseqt-ai-sdk[openai]"`` with ``openai==1.2.0`` already
pinned would satisfy the pip constraint, follow the docs, call
``instrument_all()``, and get a silent no-op with only a log line —
because the runtime check is stricter than what pip enforced.

This test parses ``pyproject.toml`` and compares each extra's floor to
the corresponding instrumentor's ``min_version``.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # Python < 3.11
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.version import Version

from disseqt_agentic_sdk.instrumentation._registry import INSTRUMENTOR_CLASSES

# extras key in pyproject.toml → INSTRUMENTOR_CLASSES key
_EXTRA_TO_PROVIDER = {
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
    "mistral": "mistralai",
    "cohere": "cohere",
    "gemini": "google-genai",
    "litellm": "litellm",
}

# Map provider (INSTRUMENTOR_CLASSES key) → pip distribution name in the
# extras (since e.g. "mistralai" is both).
_PROVIDER_TO_DIST = {
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
    "mistralai": "mistralai",
    "cohere": "cohere",
    "google-genai": "google-genai",
    "litellm": "litellm",
}


def _repo_root() -> Path:
    # tests/agentic/instrumentation/ → repo root is 3 up.
    return Path(__file__).resolve().parents[3]


def _load_extras() -> dict[str, list[str]]:
    data = tomllib.loads((_repo_root() / "pyproject.toml").read_text())
    return data["project"]["optional-dependencies"]


def _floor_for_dist(reqs: list[str], dist: str) -> Version:
    """Return the >= floor declared for `dist` in a list of requirement strings."""
    for r in reqs:
        req = Requirement(r)
        if req.name == dist:
            for spec in req.specifier:
                if spec.operator == ">=":
                    return Version(spec.version)
    raise AssertionError(f"no >= floor for {dist!r} in {reqs}")


def _instrumentor_min_version(provider: str) -> Version:
    module_path, class_name = INSTRUMENTOR_CLASSES[provider].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    if cls.min_version is None:
        raise AssertionError(f"instrumentor {provider} has no min_version")
    return Version(cls.min_version)


class TestPyprojectMinVersions:
    def test_each_extra_floor_matches_runtime_min_version(self):
        extras = _load_extras()
        for extra_name, provider in _EXTRA_TO_PROVIDER.items():
            dist = _PROVIDER_TO_DIST[provider]
            declared = _floor_for_dist(extras[extra_name], dist)
            required = _instrumentor_min_version(provider)
            assert declared >= required, (
                f"pip extra [{extra_name}] declares {dist}>={declared}, but "
                f"{provider} instrumentor requires {dist}>={required}. "
                f"Users installing this extra can get a version pip accepts "
                f"but the runtime rejects → instrument_all() silently no-ops."
            )

    def test_bundle_extra_matches_each_standalone(self):
        """The `instrumentation` bundle should not be looser than each per-provider extra."""
        extras = _load_extras()
        bundle = extras["instrumentation"]
        for extra_name in _EXTRA_TO_PROVIDER:
            per_provider_floor = _floor_for_dist(
                extras[extra_name], _PROVIDER_TO_DIST[_EXTRA_TO_PROVIDER[extra_name]]
            )
            bundle_floor = _floor_for_dist(
                bundle, _PROVIDER_TO_DIST[_EXTRA_TO_PROVIDER[extra_name]]
            )
            assert bundle_floor >= per_provider_floor, (
                f"[instrumentation] bundle floor {bundle_floor} for "
                f"{extra_name} is looser than the [{extra_name}] extra's "
                f"floor {per_provider_floor}."
            )
