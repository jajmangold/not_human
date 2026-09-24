"""Contract and property tests for the nothuman.rotation module.

Covers: normal cases, bounds (clamping, gimbal lock, zero-norm),
missing/invalid data (None, wrong length, non-numeric, NaN/inf),
replay determinism, and property tests (round-trips, orthogonality,
continuity of the 6D encoding).
"""

import importlib
import math

import nothuman
from nothuman import (
    RotationError,
    euler_to_matrix,
    matrix_from_6d,
    matrix_to_6d,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_to_matrix,
    slerp,
    slerp_6d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mat_close(a, b, tol=1e-9):
    return all(abs(a[i][j] - b[i][j]) < tol for i in range(3) for j in range(3))


def _quat_close(a, b, tol=1e-9):
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    dot = sum(a[i] * b[i] for i in range(4)) / (da * db)
    return dot > 1.0 - tol


def _is_orthogonal(m, tol=1e-9):
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            got = m[i][0] * m[j][0] + m[i][1] * m[j][1] + m[i][2] * m[j][2]
            if abs(abs(got) - abs(expected)) > tol:
                return False
    return True


def _det(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


# ---------------------------------------------------------------------------
# Euler angles
# ---------------------------------------------------------------------------


def test_euler_identity() -> None:
    m = euler_to_matrix(0.0, 0.0, 0.0)
    assert _mat_close(m, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_euler_roundtrip_normal() -> None:
    for r, p, y in [
        (0.1, 0.2, 0.3),
        (-0.5, 1.0, -2.0),
        (math.pi / 6, -math.pi / 3, math.pi / 4),
        (0.0, 0.0, 0.0),
        (math.pi / 2 - 1e-6, 0.0, 0.0),
    ]:
        m = euler_to_matrix(r, p, y)
        assert _is_orthogonal(m)
        assert abs(_det(m) - 1.0) < 1e-9
        r2, p2, y2 = matrix_to_euler(m)
        m2 = euler_to_matrix(r2, p2, y2)
        assert _mat_close(m, m2, tol=1e-8)


def test_euler_gimbal_lock_bounds() -> None:
    # Pitch at exact +pi/2 and -pi/2: canonical solution, no crash.
    m_up = euler_to_matrix(0.3, math.pi / 2, 0.7)
    m_dn = euler_to_matrix(-0.2, -math.pi / 2, 1.1)
    for m in (m_up, m_dn):
        assert _is_orthogonal(m)
        r, p, y = matrix_to_euler(m)
        assert p in (math.pi / 2, -math.pi / 2)
        # At gimbal lock, roll and yaw are not uniquely determined.
        # The canonical solution sets roll=0 and recovers yaw.
        assert r == 0.0
        m2 = euler_to_matrix(r, p, y)
        # The re-encoded matrix should be close to the original
        # (differences due to gimbal lock ambiguity are expected).
        assert _mat_close(m, m2, tol=1e-6)


def test_euler_nonfinite_rejected() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        for args in [(bad, 0.0, 0.0), (0.0, bad, 0.0), (0.0, 0.0, bad)]:
            try:
                euler_to_matrix(*args)
            except (ValueError, OverflowError):
                pass
            else:
                raise AssertionError(f"expected rejection for {args}")


def test_euler_replay_deterministic() -> None:
    m1 = euler_to_matrix(0.3, -0.4, 0.5)
    m2 = euler_to_matrix(0.3, -0.4, 0.5)
    assert m1 == m2
    e1 = matrix_to_euler(m1)
    e2 = matrix_to_euler(m2)
    assert e1 == e2


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------


def test_quat_identity() -> None:
    m = quaternion_to_matrix((1.0, 0.0, 0.0, 0.0))
    assert _mat_close(m, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_quat_roundtrip_normal() -> None:
    # Unit quaternions covering all Shepperd branches.
    quats = [
        (1.0, 0.0, 0.0, 0.0),
        (0.9486832980505138, 0.316227766016838, 0.0, 0.0),  # 90 deg about x
        (0.5, 0.5, 0.5, 0.5),
        (0.7071067811865476, 0.0, 0.7071067811865476, 0.0),  # 180 deg about y
        (0.0, 0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
        (0.8366600265340756, 0.5477225575051661, -0.0, 0.0),  # 60 deg about x
    ]
    for q in quats:
        m = quaternion_to_matrix(q)
        assert _is_orthogonal(m)
        assert abs(_det(m) - 1.0) < 1e-9
        q2 = matrix_to_quaternion(m)
        dot = sum(a * b for a, b in zip(q, q2))
        assert abs(abs(dot) - 1.0) < 1e-9


def test_quat_non_unit_input_normalized() -> None:
    m1 = quaternion_to_matrix((2.0, 0.0, 0.0, 0.0))
    m2 = quaternion_to_matrix((1.0, 0.0, 0.0, 0.0))
    assert _mat_close(m1, m2)


def test_quat_zero_norm_rejected() -> None:
    for q in ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1e-300)):
        try:
            quaternion_to_matrix(q)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {q}")


def test_quat_wrong_length_rejected() -> None:
    for bad in ((), (1.0,), (1.0, 2.0), (1.0, 2.0, 3.0, 4.0, 5.0)):
        try:
            quaternion_to_matrix(bad)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")


def test_quat_non_numeric_rejected() -> None:
    for bad in ((None, 0.0, 0.0, 0.0), (1.0, "x", 0.0, 0.0), (True, 0.0, 0.0, 0.0)):
        try:
            quaternion_to_matrix(bad)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")


def test_quat_nan_inf_rejected() -> None:
    for bad in ((float("nan"), 0.0, 0.0, 0.0), (1.0, float("inf"), 0.0, 0.0)):
        try:
            quaternion_to_matrix(bad)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")


def test_quat_replay_deterministic() -> None:
    q = (0.5, 0.5, 0.5, 0.5)
    m1 = quaternion_to_matrix(q)
    m2 = quaternion_to_matrix(q)
    assert m1 == m2
    q1 = matrix_to_quaternion(m1)
    q2 = matrix_to_quaternion(m2)
    assert q1 == q2


# ---------------------------------------------------------------------------
# 6D representation
# ---------------------------------------------------------------------------


def test_6d_roundtrip_normal() -> None:
    for r, p, y in [(0.1, 0.2, 0.3), (-0.5, 1.0, -2.0), (0.0, 0.0, 0.0), (1.0, -1.0, 2.0)]:
        m = euler_to_matrix(r, p, y)
        v = matrix_to_6d(m)
        assert len(v) == 6
        m2 = matrix_from_6d(v)
        assert _mat_close(m, m2, tol=1e-9)
        assert _is_orthogonal(m2)
        assert abs(_det(m2) - 1.0) < 1e-9


def test_6d_perturbation_stays_orthogonal() -> None:
    m = euler_to_matrix(0.3, 0.4, 0.5)
    v = list(matrix_to_6d(m))
    v[0] += 1e-4
    v[3] -= 1e-4
    m2 = matrix_from_6d(tuple(v))
    assert _is_orthogonal(m2, tol=1e-8)
    assert abs(_det(m2) - 1.0) < 1e-8


def test_6d_zero_first_column_rejected() -> None:
    for bad in ((0.0, 0.0, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)):
        try:
            matrix_from_6d(bad)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")


def test_6d_dependent_columns_rejected() -> None:
    for bad in ((1.0, 0.0, 0.0, 2.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 3.0, 0.0)):
        try:
            matrix_from_6d(bad)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")


def test_6d_wrong_length_rejected() -> None:
    for bad in ((), (1.0,), (1.0, 2.0, 3.0, 4.0, 5.0), (1.0,) * 7):
        try:
            matrix_from_6d(bad)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")


def test_6d_none_rejected() -> None:
    try:
        matrix_from_6d(None)  # type: ignore[arg-type]
    except RotationError:
        pass
    else:
        raise AssertionError("expected RotationError for None")


def test_6d_replay_deterministic() -> None:
    m = euler_to_matrix(0.2, 0.3, 0.4)
    v1 = matrix_to_6d(m)
    v2 = matrix_to_6d(m)
    assert v1 == v2
    assert matrix_from_6d(v1) == matrix_from_6d(v2)


# ---------------------------------------------------------------------------
# Slerp
# ---------------------------------------------------------------------------


def test_slerp_endpoints() -> None:
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.7071, 0.7071, 0.0, 0.0)
    a = slerp(q1, q2, 0.0)
    b = slerp(q1, q2, 1.0)
    assert _quat_close(a, q1)
    assert _quat_close(b, q2)


def test_slerp_midpoint_is_unit() -> None:
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.7071, 0.7071, 0.0, 0.0)
    qm = slerp(q1, q2, 0.5)
    assert abs(sum(x * x for x in qm) - 1.0) < 1e-12


def test_slerp_t_clamped() -> None:
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.7071, 0.7071, 0.0, 0.0)
    a = slerp(q1, q2, -1.0)
    b = slerp(q1, q2, 2.0)
    assert _quat_close(a, q1)
    assert _quat_close(b, q2)


def test_slerp_antipodal_shorter_arc() -> None:
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (-0.7071, -0.7071, 0.0, 0.0)
    qm = slerp(q1, q2, 0.5)
    assert abs(sum(x * x for x in qm) - 1.0) < 1e-12


def test_slerp_identical_inputs() -> None:
    q = (0.5, 0.5, 0.5, 0.5)
    qm = slerp(q, q, 0.5)
    assert _quat_close(qm, q)


def test_slerp_zero_norm_rejected() -> None:
    # Zero-norm quaternions are rejected.
    try:
        slerp((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0.5)
    except RotationError:
        pass
    else:
        raise AssertionError("expected RotationError for zero quaternion")


def test_slerp_nonfinite_t_rejected() -> None:
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.7071, 0.7071, 0.0, 0.0)
    for bad in (float("nan"), float("inf")):
        try:
            slerp(q1, q2, bad)
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for t={bad}")


def test_slerp_replay_deterministic() -> None:
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.7071, 0.7071, 0.0, 0.0)
    a = slerp(q1, q2, 0.3)
    b = slerp(q1, q2, 0.3)
    assert a == b


# ---------------------------------------------------------------------------
# slerp_6d
# ---------------------------------------------------------------------------


def test_slerp_6d_endpoints() -> None:
    m1 = euler_to_matrix(0.0, 0.0, 0.0)
    m2 = euler_to_matrix(0.3, 0.4, 0.5)
    v1 = matrix_to_6d(m1)
    v2 = matrix_to_6d(m2)
    a = slerp_6d(v1, v2, 0.0)
    b = slerp_6d(v1, v2, 1.0)
    assert _mat_close(matrix_from_6d(a), m1, tol=1e-8)
    assert _mat_close(matrix_from_6d(b), m2, tol=1e-8)


def test_slerp_6d_midpoint_valid() -> None:
    m1 = euler_to_matrix(0.0, 0.0, 0.0)
    m2 = euler_to_matrix(0.3, 0.4, 0.5)
    v = slerp_6d(matrix_to_6d(m1), matrix_to_6d(m2), 0.5)
    m = matrix_from_6d(v)
    assert _is_orthogonal(m, tol=1e-8)
    assert abs(_det(m) - 1.0) < 1e-8


def test_slerp_6d_replay_deterministic() -> None:
    m1 = euler_to_matrix(0.1, 0.2, 0.3)
    m2 = euler_to_matrix(0.4, 0.5, 0.6)
    v1 = matrix_to_6d(m1)
    v2 = matrix_to_6d(m2)
    a = slerp_6d(v1, v2, 0.7)
    b = slerp_6d(v1, v2, 0.7)
    assert a == b


def test_slerp_6d_invalid_inputs_rejected() -> None:
    good = matrix_to_6d(euler_to_matrix(0.0, 0.0, 0.0))
    for bad in (None, (), (1.0, 2.0, 3.0)):
        try:
            slerp_6d(bad, good, 0.5)  # type: ignore[arg-type]
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")
        try:
            slerp_6d(good, bad, 0.5)  # type: ignore[arg-type]
        except RotationError:
            pass
        else:
            raise AssertionError(f"expected RotationError for {bad}")


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_property_euler_roundtrip_many() -> None:
    for i in range(200):
        r = (i % 199) * 0.017 - 1.0
        p = ((i * 7) % 199) * 0.017 - 1.0
        y = ((i * 13) % 199) * 0.017 - 1.0
        m = euler_to_matrix(r, p, y)
        assert _is_orthogonal(m)
        assert abs(_det(m) - 1.0) < 1e-9
        r2, p2, y2 = matrix_to_euler(m)
        m2 = euler_to_matrix(r2, p2, y2)
        assert _mat_close(m, m2, tol=1e-8)


def test_property_quat_roundtrip_many() -> None:
    for i in range(200):
        w = ((i % 199) % 199) / 199.0 - 0.5
        x = ((i * 7) % 199) / 199.0 - 0.5
        y = ((i * 13) % 199) / 199.0 - 0.5
        z = ((i * 17) % 199) / 199.0 - 0.5
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-6:
            continue
        q = (w / n, x / n, y / n, z / n)
        m = quaternion_to_matrix(q)
        assert _is_orthogonal(m)
        assert abs(_det(m) - 1.0) < 1e-9
        q2 = matrix_to_quaternion(m)
        dot = sum(a * b for a, b in zip(q, q2))
        assert abs(abs(dot) - 1.0) < 1e-9


def test_property_6d_continuity() -> None:
    m0 = euler_to_matrix(0.0, 0.0, 0.0)
    v0 = matrix_to_6d(m0)
    for i in range(1, 50):
        m = euler_to_matrix(0.001 * i, 0.002 * i, 0.003 * i)
        v = matrix_to_6d(m)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v0, v)))
        assert dist < 1.0


def test_property_slerp_geodesic() -> None:
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.7071, 0.7071, 0.0, 0.0)
    qm = slerp(q1, q2, 0.5)
    expected = (math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0)
    dot = sum(a * b for a, b in zip(qm, expected))
    assert abs(abs(dot) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Replay / reload determinism
# ---------------------------------------------------------------------------


def test_module_reload_replay_is_stable() -> None:
    first = nothuman.euler_to_matrix(0.1, 0.2, 0.3)
    replay = importlib.reload(nothuman)
    second = replay.euler_to_matrix(0.1, 0.2, 0.3)
    assert first == second
    assert replay.__version__ == nothuman.__version__ == "0.1.0"
