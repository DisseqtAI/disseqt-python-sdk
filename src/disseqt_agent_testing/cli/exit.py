"""Exit-code policy. Step 6 registers its own policies (for example 6 for a baseline regression)."""

import threading
from collections.abc import Callable
from typing import Any

PytestToCli = {0: 0, 1: 1, 2: 3, 3: 4, 4: 3, 5: 5}
Policy = Callable[[Any], int]
PRIORITY = (130, 4, 3, 1, 2, 5, 0)  # the worse of the suite's code and pytest's wins
_LOCK = threading.RLock()
_POLICIES: dict[str, Policy] = {}


def register_exit_policy(name: str, fn: Policy, *, override: bool = False) -> None:
    with _LOCK:
        if name in _POLICIES and not override:
            raise ValueError(f"exit policy {name!r} is already registered")
        _POLICIES[name] = fn


def unregister_exit_policy(name: str) -> None:
    with _LOCK:
        _POLICIES.pop(name, None)


def map_pytest_code(pytest_code: int, *, interrupted: bool = False) -> int:
    """pytest's status as a CLI code. pytest uses 2 both for a keyboard interrupt and for a
    collection error; ``interrupted`` (seen through ``pytest_keyboard_interrupt``) tells them
    apart: a collection error is a configuration or collection problem (3), not 130."""
    if pytest_code == 2:
        return 130 if interrupted else 3
    return PytestToCli.get(pytest_code, 4)


def exit_code_for(
    result: Any,
    *,
    pytest_code: int | None = None,
    policy: str = "default",
    interrupted: bool = False,
    plain_failed: bool = False,
) -> int:
    """The default policy is ``SuiteResult.exit_code`` (130 > 4 > 3 > 1 > 2 > 5 > 0), merged
    with pytest's mapped status by the same priority, so neither hides the other: a plain failing
    test next to an incomplete or all-skipped suite exits 1, and a suite in which no case ran
    exits 5 even when plain tests passed. pytest also exits 1 for an incomplete agent case, so its
    1 outranks the suite's 2 only when a test that is not an agent case failed (``plain_failed``).
    A code from a custom policy that is not in the priority order is kept unless it is 0. Without
    a result (pytest stopped before writing one) only the pytest status counts."""
    with _LOCK:
        custom = _POLICIES.get(policy)
    if result is None:
        return map_pytest_code(
            pytest_code if pytest_code is not None else 4, interrupted=interrupted
        )
    code = custom(result) if custom is not None else int(result.exit_code)
    if pytest_code is None:
        return code
    mapped = map_pytest_code(int(pytest_code), interrupted=interrupted)
    if code not in PRIORITY:
        return code
    if mapped == 1 and code == 2 and not plain_failed:
        return code
    return min(code, mapped, key=PRIORITY.index)
