"""Schema-version upgrades for saved documents (cases, runs, results).

Version 1 has no migrations, but the hook exists now so that the first breaking change can ship a
migration instead of orphaning files that already exist.
"""

import copy
import threading
from collections.abc import Callable
from typing import Any

from .errors import ModelValidationError

Migration = Callable[[dict[str, Any]], dict[str, Any]]

_LOCK = threading.RLock()
# (document type, from_version) -> function returning the data upgraded by one version
MIGRATIONS: dict[tuple[str, int], Migration] = {}


def register_migration(doc_type: str, from_version: int, fn: Migration) -> None:
    with _LOCK:
        if (doc_type, from_version) in MIGRATIONS:
            raise ModelValidationError(
                f"a migration for {doc_type} from schema_version {from_version} already exists"
            )
        MIGRATIONS[(doc_type, from_version)] = fn


def unregister_migration(doc_type: str, from_version: int) -> None:
    with _LOCK:
        MIGRATIONS.pop((doc_type, from_version), None)


def upgrade(doc_type: str, data: dict[str, Any], current: int) -> dict[str, Any]:
    """Upgrade ``data`` step by step to ``current``. A missing version means 1."""
    version = data.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ModelValidationError(
            f"{doc_type}.schema_version: must be a non-negative integer, got {version!r}"
        )
    if version > current:
        raise ModelValidationError(
            f"{doc_type}: file is schema_version {version}, this SDK reads up to {current}; "
            "upgrade disseqt-ai-sdk"
        )
    out = copy.deepcopy(dict(data))
    while version < current:
        with _LOCK:
            migration = MIGRATIONS.get((doc_type, version))
        if migration is None:
            raise ModelValidationError(
                f"{doc_type}: no migration from schema_version {version} to {version + 1}"
            )
        out = migration(copy.deepcopy(out))
        version += 1
    out["schema_version"] = current
    return out
