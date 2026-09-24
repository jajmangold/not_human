"""Contract tests for the P0-006 artifact and license manifest.

Covers normal operation, bounds, missing data, and replay determinism for the
deterministic model/dataset manifest, SPDX allowlist, and SBOM output.
"""

import hashlib

import pytest

from nothuman.asset_manifest_v1 import (
    SBOM_SCHEMA,
    SPDX_ALLOWLIST,
    AssetError,
    AssetManifest,
    AssetRecord,
    canonical_json,
    checksum_sha256,
    sbom,
)


def _record(**overrides) -> AssetRecord:
    base = dict(
        name="dense-body-pose-prior",
        kind="model",
        revision="a" * 40,
        spdx_id="Apache-2.0",
        source="internal-synthesis",
        size_bytes=1024,
    )
    base.update(overrides)
    return AssetRecord(**base)


# ---------------------------------------------------------------- normal


def test_normal_manifest_roundtrip_is_deterministic() -> None:
    manifest = AssetManifest([_record(), _record(name="pose-atlas")])
    payload = manifest.to_dict()
    assert payload["schema"] == "nothuman-asset-manifest/v1"
    assert payload["assets"][0]["name"] == "dense-body-pose-prior"
    assert payload["assets"][1]["name"] == "pose-atlas"
    # Deterministic: identical inputs produce byte-identical canonical JSON.
    replay = AssetManifest([_record(), _record(name="pose-atlas")]).to_dict()
    assert canonical_json(payload) == canonical_json(replay)


def test_manifest_checksum_covers_asset_payload() -> None:
    manifest = AssetManifest([_record()])
    expected = hashlib.sha256(
        canonical_json([_record().to_dict()]).encode("utf-8")
    ).hexdigest()
    assert manifest.to_dict()["checksum"] == expected


def test_checksum_is_sha256_and_stable() -> None:
    digest = checksum_sha256(b"nothuman")
    assert digest == hashlib.sha256(b"nothuman").hexdigest()
    assert len(digest) == 64 and digest == checksum_sha256(b"nothuman")


# ---------------------------------------------------------------- bounds


def test_spdx_allowlist_rejects_non_permissive() -> None:
    assert "Apache-2.0" in SPDX_ALLOWLIST
    assert "MIT" in SPDX_ALLOWLIST
    with pytest.raises(AssetError):
        _record(spdx_id="CC-BY-NC-4.0")
    with pytest.raises(AssetError):
        _record(spdx_id="GPL-3.0-only")


def test_record_bounds_rejected() -> None:
    with pytest.raises(AssetError):
        _record(size_bytes=0)
    with pytest.raises(AssetError):
        _record(revision="abc")
    with pytest.raises(AssetError):
        _record(revision="A" * 40)
    with pytest.raises(AssetError):
        _record(spdx_id="")
    with pytest.raises(AssetError):
        _record(name="")
    with pytest.raises(AssetError):
        _record(kind="runtime")
    with pytest.raises(AssetError):
        _record(source="")
    with pytest.raises(AssetError):
        _record(size_bytes=True)


def test_duplicate_asset_name_rejected() -> None:
    with pytest.raises(AssetError):
        AssetManifest([_record(), _record(revision="b" * 40)])


# ---------------------------------------------------------------- missing data


def test_missing_data_manifest_is_empty_and_valid() -> None:
    manifest = AssetManifest([])
    assert list(manifest.assets) == []
    payload = manifest.to_dict()
    assert payload["assets"] == []
    assert canonical_json(payload) == canonical_json(AssetManifest([]).to_dict())


def test_sbom_missing_asset_rejected() -> None:
    with pytest.raises(AssetError):
        sbom([])


# ---------------------------------------------------------------- replay


def test_replay_produces_identical_sbom_bytes() -> None:
    records = [_record(), _record(name="pose-atlas", kind="dataset", revision="b" * 40)]
    first = sbom(records)
    replay = sbom([AssetRecord(**vars(r)) for r in records])
    assert first["schema"] == SBOM_SCHEMA
    assert canonical_json(first) == canonical_json(replay)
    assert first["asset_count"] == 2
    assert all(entry["spdx_id"] in SPDX_ALLOWLIST for entry in first["assets"])
