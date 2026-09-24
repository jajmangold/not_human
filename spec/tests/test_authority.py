"""Authority resolver normal, bounds, missing data, and replay contract tests."""

from __future__ import annotations

import json

import pytest

from nothuman.authority import (
    AuthorityConfig,
    AuthorityError,
    AuthorityWriter,
    resolve_authority,
)


def config(**kwargs) -> AuthorityConfig:
    return AuthorityConfig(**kwargs)


def test_normal_absolute_writer_wins() -> None:
    result = resolve_authority(
        config(),
        {"a": AuthorityWriter("a", {"face": 0.5}, mode="absolute", seq=1)},
    )
    assert result.values == {"face": 0.5, "root": 0.0}
    assert result.winner == "a"
    assert result.residual is False
    assert result.dropped == {}


def test_priority_beats_higher_seq() -> None:
    result = resolve_authority(
        config(),
        {
            "low": AuthorityWriter("low", {"face": 0.9}, mode="absolute", priority=0, seq=9),
            "high": AuthorityWriter("high", {"face": 0.2}, mode="absolute", priority=1, seq=1),
        },
    )
    assert result.winner == "high"
    assert result.values["face"] == 0.2


def test_seq_tie_breaks_by_name() -> None:
    result = resolve_authority(
        config(),
        {
            "a": AuthorityWriter("a", {"face": 0.3}, mode="absolute", priority=5, seq=1),
            "b": AuthorityWriter("b", {"face": 0.7}, mode="absolute", priority=5, seq=1),
        },
    )
    assert result.winner == "b"
    assert result.values["face"] == 0.7


def test_additive_writers_apply_before_absolute() -> None:
    result = resolve_authority(
        config(),
        {
            "add": AuthorityWriter("add", {"face": 0.4}, mode="additive", seq=1),
            "abs": AuthorityWriter("abs", {"face": 0.6}, mode="absolute", seq=2),
        },
    )
    assert result.values["face"] == 0.6
    assert result.winner == "abs"


def test_additive_only_clamps_to_bounds() -> None:
    result = resolve_authority(
        config(),
        {
            "a": AuthorityWriter("a", {"face": 0.8}, mode="additive", seq=1),
            "b": AuthorityWriter("b", {"face": 0.9}, mode="additive", seq=2),
        },
    )
    assert result.values["face"] == 1.0
    assert result.winner is None


def test_mask_scales_and_disables_writer() -> None:
    result = resolve_authority(
        config(),
        {
            "a": AuthorityWriter(
                "a", {"face": 1.0, "root": 0.5}, mode="absolute", seq=1,
                mask={"face": 0.5, "root": 0.0},
            )
        },
        previous={"face": 0.25, "root": 0.75},
    )
    assert result.values["face"] == 0.5
    assert result.values["root"] == 0.75
    assert result.winner == "a"


def test_residual_gate_retains_previous() -> None:
    result = resolve_authority(
        config(residual_gate=0.2),
        {"a": AuthorityWriter("a", {"face": 0.1}, mode="absolute", seq=1)},
        previous={"face": 0.9},
    )
    assert result.values["face"] == 0.9
    assert result.residual is True
    assert result.winner == "a"


def test_residual_gate_without_previous_uses_value() -> None:
    result = resolve_authority(
        config(residual_gate=0.2),
        {"a": AuthorityWriter("a", {"face": 0.1}, mode="absolute", seq=1)},
    )
    assert result.values["face"] == 0.1
    assert result.residual is False


def test_missing_data_defaults_to_zero() -> None:
    result = resolve_authority(config(), {})
    assert result.values == {"face": 0.0, "root": 0.0}
    assert result.winner is None
    assert result.residual is False


def test_missing_data_falls_through_to_previous() -> None:
    result = resolve_authority(config(), {}, previous={"face": 0.3, "root": 0.4})
    assert result.values == {"face": 0.3, "root": 0.4}
    assert result.winner is None


def test_stale_seq_writer_is_dropped() -> None:
    result = resolve_authority(
        config(),
        {
            "a": AuthorityWriter("a", {"face": 0.5}, mode="absolute", seq=2),
            "b": AuthorityWriter("b", {"face": 0.9}, mode="absolute", seq=1),
        },
    )
    assert result.dropped == {}
    assert result.winner == "a"


def test_bounds_reject_invalid_values() -> None:
    with pytest.raises(AuthorityError):
        AuthorityWriter("a", {"face": 1.5}, seq=1)
    with pytest.raises(AuthorityError):
        AuthorityWriter("a", {"face": -0.1}, seq=1)
    with pytest.raises(AuthorityError):
        AuthorityWriter("a", {"face": float("nan")}, seq=1)
    with pytest.raises(AuthorityError):
        AuthorityWriter("a", {"face": True}, seq=1)
    with pytest.raises(AuthorityError):
        AuthorityWriter("a", {"face": 0.5}, mode="relative", seq=1)
    with pytest.raises(AuthorityError):
        AuthorityWriter("a", {"face": 0.5}, seq=-1)
    with pytest.raises(AuthorityError):
        AuthorityWriter("", {"face": 0.5}, seq=1)
    with pytest.raises(AuthorityError):
        AuthorityConfig(keys=("face", "face"))
    with pytest.raises(AuthorityError):
        AuthorityConfig(residual_gate=1.5)
    with pytest.raises(AuthorityError):
        resolve_authority(config(), {"a": AuthorityWriter("a", {"unknown": 0.5}, seq=1)})
    with pytest.raises(AuthorityError):
        resolve_authority(config(), {}, previous={"unknown": 0.5})


def test_replay_is_deterministic() -> None:
    writers = {
        "a": AuthorityWriter("a", {"face": 0.5, "root": 0.25}, mode="absolute", seq=1),
        "b": AuthorityWriter("b", {"face": 0.1}, mode="additive", seq=2),
    }
    first = resolve_authority(config(), writers, previous={"face": 0.1})
    second = resolve_authority(config(), writers, previous={"face": 0.1})
    assert first.to_dict() == second.to_dict()
    first_json = json.dumps(first.to_dict(), sort_keys=True)
    second_json = json.dumps(second.to_dict(), sort_keys=True)
    assert first_json == second_json
