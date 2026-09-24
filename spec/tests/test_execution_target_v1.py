"""Contract tests for the CMP 100-210 execution target (ADR-0001).

Enforces the hardware/runtime contract:
- the pinned target is an HMMA-disabled sm70 dp4a / CUDA-core target,
- FP16 tensor paths are optional only where the hardware proves support,
- evidence names actual hardware/resource IDs and current limitations.
"""
from __future__ import annotations

import pytest

from nothuman.execution_target_v1 import (
    CMP_100_210,
    TARGET_ID,
    ExecutionTarget,
    GpuCaps,
    assert_target,
    fp16_tensor_supported,
    resource_ids,
    limitations,
)


def _target(**over) -> ExecutionTarget:
    kw = dict(
        target_id=TARGET_ID,
        compute_capability="sm_70",
        architecture="volta",
        dp4a_enabled=True,
        hmma_enabled=False,
        cuda_core_enabled=True,
    )
    kw.update(over)
    return ExecutionTarget(**kw)


def test_target_is_sm70_dp4a_cuda_core() -> None:
    t = CMP_100_210
    assert t.target_id == TARGET_ID == "CMP-100-210"
    assert t.compute_capability == "sm_70"
    assert t.architecture == "volta"
    assert t.dp4a_enabled
    assert t.cuda_core_enabled
    assert not t.hmma_enabled


def test_hmma_disabled_is_the_contract() -> None:
    # The decision is HMMA-disabled: HMMA tensor paths must be off by default.
    assert not CMP_100_210.hmma_enabled
    assert _target(hmma_enabled=True).hmma_enabled
    with pytest.raises(ValueError):
        assert_target(_target(hmma_enabled=True))


def test_target_id_is_pinned() -> None:
    with pytest.raises(ValueError):
        assert_target(_target(target_id="CMP-999"))


def test_compute_capability_is_pinned_to_sm70() -> None:
    with pytest.raises(ValueError):
        assert_target(_target(compute_capability="sm_75"))


def test_dp4a_must_be_enabled() -> None:
    with pytest.raises(ValueError):
        assert_target(_target(dp4a_enabled=False))


def test_cuda_core_must_be_enabled() -> None:
    with pytest.raises(ValueError):
        assert_target(_target(cuda_core_enabled=False))


def test_fp16_tensor_optional_requires_hardware_proof() -> None:
    # HMMA disabled -> no FP16 tensor path regardless of reported caps.
    assert fp16_tensor_supported(CMP_100_210, GpuCaps(hmma=True)) is False
    # A different (HMMA-enabled) target may use FP16 tensor only if hardware proves it.
    hmma_target = _target(target_id="CMP-100-210-HMMA", hmma_enabled=True)
    assert fp16_tensor_supported(hmma_target, GpuCaps(hmma=True)) is True
    assert fp16_tensor_supported(hmma_target, GpuCaps(hmma=False)) is False


def test_resource_ids_name_actual_hardware() -> None:
    ids = resource_ids(CMP_100_210)
    # Evidence must name actual hardware/resource IDs.
    assert "sm_70" in ids
    assert "dp4a" in ids
    assert "cuda-core" in ids
    assert "hmma" not in ids  # HMMA is disabled for this target


def test_limitations_name_current_limitations() -> None:
    lims = limitations(CMP_100_210)
    # Evidence must name current limitations.
    assert any("HMMA" in l for l in lims)
    assert any("FP16" in l for l in lims)
    assert any("sm_70" in l for l in lims)


def test_replay_is_deterministic() -> None:
    assert resource_ids(CMP_100_210) == resource_ids(CMP_100_210)
    assert limitations(CMP_100_210) == limitations(CMP_100_210)
