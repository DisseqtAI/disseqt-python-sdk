"""Walk a source tree and yield char-bounded code chunks for scanning."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Language file-extension → language name. Add extensions here to grow support.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
}

SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "out",
        ".next",
        "target",
        "vendor",
        ".terraform",
        "coverage",
        ".turbo",
        ".cache",
        ".idea",
        ".vscode",
    }
)

SKIP_GLOBS: tuple[str, ...] = (
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "go.sum",
)

DEFAULT_MAX_FILE_BYTES = 1_500_000
DEFAULT_MAX_CHUNK_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """A slice of a source file, small enough to bundle into an LLM prompt."""

    file_path: str  # POSIX-style path relative to the scan root
    language: str
    start_line: int  # 1-indexed
    end_line: int  # inclusive
    text: str


def _skip_glob_match(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_GLOBS)


def iter_source_files(
    root: Path,
    *,
    only: Iterable[str] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Iterator[Path]:
    """Yield source files under ``root`` that match a supported language.

    ``only`` is an optional whitelist of file paths (used by --diff mode).
    Paths in ``only`` may be absolute or relative to ``root``.
    """
    whitelist: set[Path] | None = None
    if only is not None:
        whitelist = set()
        for p in only:
            candidate = Path(p)
            if not candidate.is_absolute():
                candidate = root / candidate
            whitelist.add(candidate.resolve())
        if not whitelist:
            return

    root = root.resolve()
    for path in _walk(root):
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        if _skip_glob_match(path.name):
            continue
        if whitelist is not None and path.resolve() not in whitelist:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        yield path


def _walk(root: Path) -> Iterator[Path]:
    """Recursive walker that prunes SKIP_DIRS on the way down."""
    if root.is_file():
        yield root
        return
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                if name in SKIP_DIRS or name.startswith("."):
                    # SKIP_DIRS is authoritative; also drop other dot-dirs to
                    # avoid scanning .github/.circleci/etc. If a user wants
                    # them, they can point --path at the dir directly.
                    if name in SKIP_DIRS or (name.startswith(".") and name not in {"."}):
                        continue
                stack.append(entry)
            elif entry.is_file():
                yield entry


def chunk_file(
    path: Path,
    *,
    root: Path,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> Iterator[CodeChunk]:
    """Yield line-aligned chunks of ``path`` under the ``max_chunk_chars`` cap.

    Chunks never split a line — we grow a buffer line-by-line and flush when
    adding the next line would exceed the cap, so line numbers stay honest.
    """
    lang = SUPPORTED_EXTENSIONS.get(path.suffix, "text")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if not text.strip():
        return

    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()

    lines = text.splitlines(keepends=True) or [text]
    buf: list[str] = []
    buf_size = 0
    start_line = 1
    line_no = 0

    for line_no, line in enumerate(lines, start=1):
        line_len = len(line)
        # If a single line is bigger than the cap, ship it alone anyway rather
        # than truncating — the validator will do the right thing.
        if buf and buf_size + line_len > max_chunk_chars:
            yield CodeChunk(
                file_path=rel,
                language=lang,
                start_line=start_line,
                end_line=line_no - 1,
                text="".join(buf),
            )
            buf = []
            buf_size = 0
            start_line = line_no
        buf.append(line)
        buf_size += line_len

    if buf:
        yield CodeChunk(
            file_path=rel,
            language=lang,
            start_line=start_line,
            end_line=max(line_no, start_line),
            text="".join(buf),
        )


def collect_chunks(
    root: Path,
    *,
    only: Iterable[str] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> Iterator[CodeChunk]:
    """One-shot helper: walk ``root`` and yield chunks for every source file."""
    for path in iter_source_files(root, only=only, max_file_bytes=max_file_bytes):
        yield from chunk_file(path, root=root, max_chunk_chars=max_chunk_chars)
