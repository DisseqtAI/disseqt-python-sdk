"""A fixed hash split of golden items: 15% dev (few-shot / prompt tuning), 42.5% validation
(tuning), 42.5% test (the only split a gating decision may use). Stable across runs, machines and
additions: an item's split depends only on its id and kind."""

import hashlib

SPLITS = ("dev", "validation", "test")
_DEV, _VAL = 150, 575  # thresholds out of 1000


def split_of(kind: str, item_id: str) -> str:
    bucket = (
        int.from_bytes(
            hashlib.sha256(f"disseqt-split-v1\x1f{kind}\x1f{item_id}".encode()).digest()[:4], "big"
        )
        % 1000
    )
    return "dev" if bucket < _DEV else "validation" if bucket < _VAL else "test"
