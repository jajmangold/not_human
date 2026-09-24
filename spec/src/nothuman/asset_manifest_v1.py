"""Deterministic artifact and license manifest (P0-006).

Stable, deterministic model/dataset manifest with SHA-256 checksums, an SPDX
permissive allowlist, and canonical SBOM output. Artifacts without a
permissive SPDX identifier are rejected at record construction, matching the
ADR-0003 license-neutral provenance rule. No weights or media are embedded:
records reference revisions only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

MANIFEST_SCHEMA = "nothuman-asset-manifest/v1"
SBOM_SCHEMA = "nothuman-sbom/v1"

# Permissive SPDX identifiers only. Non-commercial research licenses
# (e.g. CC-BY-NC-4.0, SMPL-X/AMASS terms) are deliberately excluded.
SPDX_ALLOWLIST = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "Zlib",
    }
)

_KINDS = frozenset({"model", "dataset"})
_REVISION = 40


class AssetError(ValueError):
    """Raised when an asset record or manifest violates the contract."""


@dataclass(frozen=True)
class AssetRecord:
    """One model/dataset artifact with provenance and license status."""

    name: str
    kind: str
    revision: str
    spdx_id: str
    source: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AssetError("asset name must be a non-empty string")
        if self.kind not in _KINDS:
            raise AssetError(f"asset kind must be one of {sorted(_KINDS)}")
        if (
            not isinstance(self.revision, str)
            or len(self.revision) != _REVISION
            or not all(c in "0123456789abcdef" for c in self.revision)
        ):
            raise AssetError("asset revision must be a 40-char lowercase hex digest")
        if self.spdx_id not in SPDX_ALLOWLIST:
            raise AssetError(f"asset license {self.spdx_id!r} is not in the SPDX allowlist")
        if not isinstance(self.source, str) or not self.source:
            raise AssetError("asset source must be a non-empty string")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise AssetError("asset size_bytes must be a positive integer")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "revision": self.revision,
            "spdx_id": self.spdx_id,
            "source": self.source,
            "size_bytes": self.size_bytes,
        }


def checksum_sha256(data: bytes) -> str:
    """SHA-256 hex digest for deterministic artifact checksums."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> str:
    """Byte-deterministic JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class AssetManifest:
    """Ordered collection of asset records with deterministic serialization."""

    assets: tuple[AssetRecord, ...] = ()

    def __init__(self, assets: list[AssetRecord] | tuple[AssetRecord, ...] = ()) -> None:
        objects = tuple(assets)
        seen: set[str] = set()
        for record in objects:
            if not isinstance(record, AssetRecord):
                raise AssetError("manifest entries must be AssetRecord instances")
            if record.name in seen:
                raise AssetError(f"duplicate asset name: {record.name}")
            seen.add(record.name)
        object.__setattr__(self, "assets", objects)

    def __len__(self) -> int:
        return len(self.assets)

    def to_dict(self) -> dict:
        return {
            "schema": MANIFEST_SCHEMA,
            "assets": [record.to_dict() for record in self.assets],
            "checksum": checksum_sha256(
                canonical_json([record.to_dict() for record in self.assets]).encode("utf-8")
            ),
        }


def sbom(assets: list[AssetRecord] | tuple[AssetRecord, ...]) -> dict:
    """Build the deterministic SBOM document for a set of asset records."""
    manifest = AssetManifest(assets)
    if not manifest.assets:
        raise AssetError("sbom requires at least one asset")
    return {
        "schema": SBOM_SCHEMA,
        "asset_count": len(manifest.assets),
        "assets": [record.to_dict() for record in manifest.assets],
        "checksum": manifest.to_dict()["checksum"],
    }

