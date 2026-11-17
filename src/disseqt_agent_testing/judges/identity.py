"""Identity: everything that can change what a judge says, hashed. Calibration and cache entries
are keyed by it, so any change yields a fresh, uncalibrated identity."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..assertions.canonical import canonical_json, short_hash


def sha256_hex(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(slots=True, frozen=True)
class JudgeIdentity:
    kind: str
    prompt_version: str
    prompt_hash: str
    model: str
    base_url: str | None
    params: Mapping[str, Any]
    samples: int
    min_agreement: float
    few_shot_hash: str
    output_schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "prompt_version": self.prompt_version, "prompt_hash": self.prompt_hash,
            "model": self.model, "base_url": self.base_url, "params": dict(self.params),
            "samples": self.samples, "min_agreement": self.min_agreement,
            "few_shot_hash": self.few_shot_hash, "output_schema_hash": self.output_schema_hash,
        }  # fmt: skip

    @property
    def hash(self) -> str:
        return short_hash(self.to_dict())
