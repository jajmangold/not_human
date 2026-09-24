"""Contract tests for the stable package boundary."""

import importlib

import nothuman


def test_package_import_and_version_are_deterministic() -> None:
    assert nothuman.__version__ == "0.1.0"


def test_version_boundary_has_expected_bounds() -> None:
    parts = nothuman.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
    assert tuple(map(int, parts)) >= (0, 1, 0)


def test_missing_optional_metadata_is_safe() -> None:
    assert getattr(nothuman, "__model__", None) is None
    assert getattr(nothuman, "__renderer__", None) is None


def test_import_replay_is_stable() -> None:
    first = nothuman.__version__
    replay = importlib.reload(nothuman)
    assert replay.__version__ == first == "0.1.0"
