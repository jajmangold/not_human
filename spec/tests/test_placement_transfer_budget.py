"""ADR-0002 placement and transfer-budget contract tests.

Covers the PCIe x1 placement boundary: renderer tensors stay GPU-resident
(0 bytes across the link), only control vectors or crops may cross the
boundary, the per-frame transfer budget is enforced, and the placement
record round-trips deterministically.
"""

import pytest

from nothuman.placement import (
    FRAME_PERIOD_NS,
    LINK_NAME,
    LINK_THROUGHPUT_BYTES_PER_SEC,
    PAYLOAD_KIND_CONTROL_VECTOR,
    PAYLOAD_KIND_CROP,
    PAYLOAD_KIND_RENDERER_TENSOR,
    PAYLOAD_KINDS,
    PAYLOAD_KIND_VALUES,
    PlacementCheck,
    PlacementRecord,
    PayloadTransfer,
    TRANSFER_BUDGET_BYTES_PER_FRAME,
    default_frame,
    transfer_budget_ok,
)


def _control_vector(name: str = "ControlFrameV1", nbytes: int = 1024) -> PayloadTransfer:
    return PayloadTransfer(kind=PAYLOAD_KIND_CONTROL_VECTOR, name=name, bytes=nbytes)


def _crop(name: str = "crop", nbytes: int = 2 * 1024 * 1024) -> PayloadTransfer:
    return PayloadTransfer(kind=PAYLOAD_KIND_CROP, name=name, bytes=nbytes)


def _tensor(name: str = "position", nbytes: int = 64 * 1024 * 1024) -> PayloadTransfer:
    return PayloadTransfer(kind=PAYLOAD_KIND_RENDERER_TENSOR, name=name, bytes=nbytes)


# ---------------------------------------------------------------- constants


def test_frozen_constants_are_pinned() -> None:
    assert LINK_NAME == "PCIe x1 Gen 4"
    assert LINK_THROUGHPUT_BYTES_PER_SEC == 312_500_000
    assert FRAME_PERIOD_NS == 33_333_333
    assert TRANSFER_BUDGET_BYTES_PER_FRAME == 8 * 1024 * 1024


def test_budget_fits_within_measured_link() -> None:
    effective = TRANSFER_BUDGET_BYTES_PER_FRAME / FRAME_PERIOD_NS * 1e9
    assert effective <= LINK_THROUGHPUT_BYTES_PER_SEC
    # The budget must leave real headroom, not saturate the link.
    assert effective < LINK_THROUGHPUT_BYTES_PER_SEC


def test_payload_kind_contract_is_closed() -> None:
    assert PAYLOAD_KIND_VALUES == ("control_vector", "crop", "renderer_tensor")
    assert set(PAYLOAD_KINDS) == {"control_vector", "crop", "renderer_tensor"}


# ---------------------------------------------------------------- transfer budget


def test_control_vector_payload_under_budget_passes() -> None:
    assert transfer_budget_ok(1024)
    assert transfer_budget_ok(TRANSFER_BUDGET_BYTES_PER_FRAME)


def test_over_budget_payload_fails() -> None:
    assert not transfer_budget_ok(TRANSFER_BUDGET_BYTES_PER_FRAME + 1)
    assert not transfer_budget_ok(64 * 1024 * 1024)


def test_transfer_budget_ok_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        transfer_budget_ok(-1)
    with pytest.raises(ValueError):
        transfer_budget_ok(True)


# ---------------------------------------------------------------- placement boundary


def test_control_vector_frame_is_admissible() -> None:
    record = default_frame(_control_vector(nbytes=1024), _crop(nbytes=2 * 1024 * 1024))
    result = record.check()
    assert result.ok
    assert result.transfer_bytes == 1024 + 2 * 1024 * 1024
    assert result.violations == ()


def test_renderer_tensor_host_transfer_is_rejected() -> None:
    record = default_frame(_control_vector(), _tensor())
    result = record.check()
    assert not result.ok
    assert any("GPU-resident" in v for v in result.violations)


def test_renderer_tensor_contributes_zero_bytes_to_transfer() -> None:
    record = default_frame(_control_vector(nbytes=1024), _tensor())
    # The tensor must never count toward the cross-boundary byte total.
    assert record.transfer_bytes == 1024


def test_over_budget_frame_is_rejected() -> None:
    # Two crops each under budget individually, but together over budget.
    record = default_frame(_crop(nbytes=6 * 1024 * 1024), _crop(nbytes=6 * 1024 * 1024))
    result = record.check()
    assert not result.ok
    assert any("budget exceeded" in v for v in result.violations)


def test_empty_frame_is_admissible() -> None:
    record = default_frame()
    result = record.check()
    assert result.ok
    assert result.transfer_bytes == 0


# ---------------------------------------------------------------- validation


def test_payload_transfer_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError):
        PayloadTransfer(kind=PAYLOAD_KIND_CONTROL_VECTOR, name="x", bytes=-1)
    with pytest.raises(ValueError):
        PayloadTransfer(kind=PAYLOAD_KIND_CONTROL_VECTOR, name="x", bytes=True)


def test_payload_transfer_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        PayloadTransfer(kind=PAYLOAD_KIND_CONTROL_VECTOR, name="", bytes=1)


def test_placement_record_rejects_invalid_frame_period() -> None:
    with pytest.raises(ValueError):
        PlacementRecord(frame_period_ns=0, budget_bytes_per_frame=1, payloads=())


def test_placement_record_rejects_non_payload_items() -> None:
    with pytest.raises(ValueError):
        PlacementRecord(
            frame_period_ns=FRAME_PERIOD_NS,
            budget_bytes_per_frame=1,
            payloads=(("control_vector", "x", 1),),  # not a PayloadTransfer
        )


# ---------------------------------------------------------------- determinism


def test_placement_record_json_round_trip_is_deterministic() -> None:
    record = default_frame(_control_vector(nbytes=1024), _crop(nbytes=2 * 1024 * 1024))
    payload = record.to_json()
    assert payload == record.to_json()
    assert PlacementRecord.from_json(payload) == record


def test_placement_check_json_is_deterministic() -> None:
    record = default_frame(_control_vector(), _tensor())
    result = record.check()
    assert isinstance(result, PlacementCheck)
    assert result.to_json() == result.to_json()
    assert result.to_dict()["ok"] is False
