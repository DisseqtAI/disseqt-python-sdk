"""``JsonDirStore``: one pretty-printed JSON file per (request, sample); atomic writes; no pickle."""

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

from ..codes import message

_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_KEY = re.compile(r"[0-9a-f]{64}")


class CacheCorrupt(Exception):
    def __init__(self, path: str, detail: str):
        super().__init__(message("cache_corrupt", path=path, detail=detail))
        self.path, self.detail = path, detail


class CacheStore(Protocol):
    def get(self, provider: str, model: str, key: str) -> dict[str, Any] | None: ...
    def put(self, provider: str, model: str, key: str, entry: dict[str, Any]) -> None: ...
    def touch(self, provider: str, model: str, key: str) -> None: ...
    def list(self) -> list[Path]: ...


def safe_segment(text: str) -> str:
    cleaned = _SAFE.sub("_", text).strip(".") or "_"
    return cleaned[:120]


class JsonDirStore:
    def __init__(self, cache_dir: str | os.PathLike[str], *, track: bool = False):
        self.root = Path(cache_dir)
        self.track = track
        self._lock = threading.Lock()

    def path_for(self, provider: str, model: str, key: str) -> Path:
        path = (
            self.root / f"{safe_segment(provider)}__{safe_segment(model)}" / key[:2] / f"{key}.json"
        )
        resolved = path.resolve()
        if self.root.resolve() not in resolved.parents:
            raise CacheCorrupt(str(path), "path escapes the cache directory")
        return path

    def get(self, provider: str, model: str, key: str) -> dict[str, Any] | None:
        path = self.path_for(provider, model, key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CacheCorrupt(str(path), str(exc)) from None
        if not isinstance(entry, dict) or entry.get("key") != key or "response" not in entry:
            raise CacheCorrupt(str(path), "missing key or response")
        return entry

    def put(self, provider: str, model: str, key: str, entry: dict[str, Any]) -> None:
        path = self.path_for(provider, model, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)  # atomic
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def touch(self, provider: str, model: str, key: str) -> None:
        if not self.track:
            return
        folder = self.root / ".touched"
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"{os.getpid()}.txt", "a", encoding="utf-8") as handle:
                handle.write(f"{safe_segment(provider)}__{safe_segment(model)}/{key[:2]}/{key}\n")

    def owns(self, path: Path) -> bool:
        """Whether ``path`` sits where ``put`` writes: ``<provider>__<model>/<key[:2]>/<key>.json``
        with a SHA-256 key. Anything else under the directory belongs to someone else."""
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            return False
        if len(parts) != 3 or "__" not in parts[0] or path.suffix != ".json":
            return False
        return bool(_KEY.fullmatch(path.stem)) and parts[1] == path.stem[:2] and path.is_file()

    def list(self) -> list[Path]:
        """The cache entries only, so ``verify`` and ``prune`` never touch another file."""
        return sorted(p for p in self.root.glob("*/*/*.json") if self.owns(p))
