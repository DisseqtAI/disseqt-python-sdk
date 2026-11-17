"""The cache key: content-addressed over everything that can change a response."""

from collections.abc import Mapping, Sequence
from typing import Any

from ...assertions.canonical import canonical_json
from ..identity import sha256_hex

CACHE_SCHEMA_VERSION = 1


def cache_key(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    messages: Sequence[Mapping[str, str]],
    output_schema: Mapping[str, Any],
    schema_name: str,
    params: Mapping[str, Any],
    sample_index: int,
    kind: str,
    prompt_hash: str,
) -> str:
    """Excluded on purpose: timeouts, retries, concurrency, caps, keys, headers, library
    version, ``min_agreement`` and ``n`` (a bigger ``n`` reuses the samples it already has)."""
    return sha256_hex(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION, "provider": provider, "model": model,
            "base_url": base_url, "messages": [dict(m) for m in messages],
            "output_schema_sha256": sha256_hex(dict(output_schema)), "schema_name": schema_name,
            "params": dict(params), "sample_index": sample_index, "kind": kind,
            "prompt_hash": prompt_hash,
        }  # fmt: skip
    )


__all__ = ["CACHE_SCHEMA_VERSION", "cache_key", "canonical_json"]
