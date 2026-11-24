"""``"provider:model"`` parsing and the provider registry (lazy: nothing is imported or
constructed until the first live call)."""

import threading
from collections.abc import Callable
from typing import Any

from ..codes import JudgeConfigError
from .base import JudgeModel

ProviderFactory = Callable[..., JudgeModel]
_LOCK = threading.RLock()
_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory, *, override: bool = False) -> None:
    """``factory(model_id, **model_options) -> JudgeModel``; called lazily."""
    with _LOCK:
        if name in _PROVIDERS and not override:
            raise ValueError(f"judge provider {name!r} is already registered")
        _PROVIDERS[name] = factory


def unregister_provider(name: str) -> None:
    with _LOCK:
        _PROVIDERS.pop(name, None)


def known_providers() -> tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_PROVIDERS))


def parse_model_string(text: str) -> tuple[str, str]:
    provider, sep, model = text.partition(":")
    if not sep or not provider or not model or not provider.replace("_", "").isalnum():
        raise JudgeConfigError("bad_model_string", f"{text!r} must look like 'provider:model'")
    return provider, model


def check_provider(provider: str) -> None:
    with _LOCK:
        known = sorted(_PROVIDERS)
        ok = provider in _PROVIDERS
    if not ok:
        raise JudgeConfigError("unknown_provider", f"no provider {provider!r}; registered: {known}")


def build_model(provider: str, model: str, options: dict[str, Any]) -> JudgeModel:
    check_provider(provider)
    with _LOCK:
        factory = _PROVIDERS[provider]
    return factory(model, **options)
