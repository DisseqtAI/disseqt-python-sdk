"""Case selection: id globs and tag expressions (identical for the loader, plugin and CLI)."""

from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase


@dataclass(frozen=True, slots=True)
class CaseFilter:
    """``ids`` and ``exclude_ids`` are ``fnmatchcase`` globs. Each ``tags`` entry is an expression
    ``term ("," term)*`` with ``!`` negation: commas mean OR inside one entry, and repeated entries
    mean AND (``("smoke,fast", "!slow")`` is ``(smoke or fast) and not slow``)."""

    ids: tuple[str, ...] = ()
    exclude_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        ids: Iterable[str] | str = (),
        exclude_ids: Iterable[str] | str = (),
        tags: Iterable[str] | str = (),
    ) -> "CaseFilter":
        """From CLI-style values: ``ids`` and ``exclude_ids`` may be comma-separated globs; each
        ``tags`` item stays one expression."""

        def globs(value: Iterable[str] | str) -> tuple[str, ...]:
            items = [value] if isinstance(value, str) else list(value)
            return tuple(p.strip() for item in items for p in item.split(",") if p.strip())

        tag_items = [tags] if isinstance(tags, str) else list(tags)
        return cls(globs(ids), globs(exclude_ids), tuple(t for t in tag_items if t.strip()))

    @property
    def active(self) -> bool:
        return bool(self.ids or self.exclude_ids or self.tags)

    def matches(self, case_id: str, tags: Iterable[str] = ()) -> bool:
        have = set(tags)
        if self.ids and not any(fnmatchcase(case_id, g) for g in self.ids):
            return False
        if any(fnmatchcase(case_id, g) for g in self.exclude_ids):
            return False
        for entry in self.tags:
            terms = [t.strip() for t in entry.split(",") if t.strip()]
            if not terms:
                continue
            ok = any(
                (t[1:].strip() not in have) if t.startswith("!") else (t in have) for t in terms
            )
            if not ok:
                return False
        return True
