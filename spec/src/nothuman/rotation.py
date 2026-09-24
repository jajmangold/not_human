"""Deterministic rotation representations and conversions.

Pure-Python implementation of Euler angles (ZYX / aerospace convention),
3x3 rotation matrices, unit quaternions, and minimal 6D continuous
rotation representations (Zhou et al., "On the Continuity of Rotation
Representations", CVPR 2019).

All functions are deterministic: identical inputs always produce bit-
identical outputs, with no nondeterministic iteration order or
platform-dependent reduction.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "RotationError",
    "euler_to_matrix",
    "matrix_to_euler",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "matrix_to_6d",
    "matrix_from_6d",
    "slerp",
    "slerp_6d",
]

_EPS = 1e-12
_EPS6 = 1e-12
_EPS32 = 1e-32

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
Quat = tuple[float, float, float, float]  # (w, x, y, z)
SixD = tuple[float, ...]  # length 6


class RotationError(ValueError):
    """Raised for structurally invalid rotation inputs.

    Raised cases: wrong-length vectors, non-numeric elements, NaN/inf
    elements, and quaternions whose norm is not normalizable (zero or
    subnormal).
    """


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_vec(name: str, value: Sequence[float], length: int) -> tuple[float, ...]:
    if value is None:
        raise RotationError(f"{name}: expected {length} elements, got None")
    try:
        seq = tuple(value)
    except TypeError as exc:
        raise RotationError(f"{name}: not a sequence of numbers") from exc
    if len(seq) != length:
        raise RotationError(f"{name}: expected {length} elements, got {len(seq)}")
    out: list[float] = []
    for i, element in enumerate(seq):
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise RotationError(f"{name}[{i}]: not a number: {element!r}")
        f = float(element)
        if math.isnan(f) or math.isinf(f):
            raise RotationError(f"{name}[{i}]: element must be finite")
        out.append(f)
    return tuple(out)


def _check_mat(name: str, value: Sequence[Sequence[float]]) -> Mat3:
    if value is None:
        raise RotationError(f"{name}: expected 3 rows, got None")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise RotationError(f"{name}: not a sequence of rows") from exc
    if len(rows) != 3:
        raise RotationError(f"{name}: expected 3 rows, got {len(rows)}")
    return tuple(_check_vec(f"{name}[{r}]", rows[r], 3) for r in range(3))  # type: ignore[index]


def _norm3(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _mat_mul(a: Mat3, b: Mat3) -> Mat3:
    out: list[Vec3] = []
    for i in range(3):
        row: list[float] = []
        for j in range(3):
            row.append(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        out.append(tuple(row))  # type: ignore[arg-type]
    return (out[0], out[1], out[2])  # type: ignore[call-arg]


def _mat_transpose(m: Mat3) -> Mat3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def _inv_orthogonal(m: Mat3) -> Mat3:
    """Inverse of an orthogonal matrix is its transpose."""
    return _mat_transpose(m)


def _identity() -> Mat3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Euler angles (ZYX / aerospace convention)
# ---------------------------------------------------------------------------


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> Mat3:
    """Compose R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ZYX intrinsic).

    Inputs are radians. Deterministic: pure arithmetic on the inputs.
    """
    r = float(roll)
    p = float(pitch)
    y = float(yaw)
    for name, val in (("roll", r), ("pitch", p), ("yaw", y)):
        if math.isnan(val) or math.isinf(val):
            raise RotationError(f"{name}: must be finite, got {val}")
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def matrix_to_euler(m: Mat3) -> tuple[float, float, float]:
    """Decompose R into (roll, pitch, yaw), ZYX convention, radians.

    Returns the principal-branch solution. Pitch is clamped to
    [-pi/2, pi/2]; at exact gimbal lock (|m[2][0]| == 1) the returned
    solution is the canonical one (yaw = 0, roll set by sign).
    """
    m = _check_mat("m", m)
    sp = -m[2][0]
    if sp > 1.0 - _EPS:
        # Gimbal lock: pitch = +pi/2
        pitch = math.pi / 2.0
        roll = 0.0
        yaw = math.atan2(-m[0][1], m[0][2])
    elif sp < -1.0 + _EPS:
        # Gimbal lock: pitch = -pi/2
        pitch = -math.pi / 2.0
        roll = 0.0
        yaw = math.atan2(-m[0][1], -m[0][2])
    else:
        pitch = math.asin(sp)
        cp = math.cos(pitch)
        roll = math.atan2(m[2][1] / cp, m[2][2] / cp)
        yaw = math.atan2(m[1][0] / cp, m[0][0] / cp)
    return (roll, pitch, yaw)


# ---------------------------------------------------------------------------
# Quaternions (w, x, y, z)
# ---------------------------------------------------------------------------


def _normalize_quat(q: Quat) -> Quat:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < _EPS6:
        raise RotationError("quaternion: norm is zero or subnormal")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _quat_conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    n = _norm3(axis)
    if n < _EPS6:
        raise RotationError("axis: not normalizable (zero or subnormal)")
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return (c, (axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s)


def quaternion_to_matrix(q: Quat) -> Mat3:
    """Unit quaternion (w, x, y, z) to 3x3 rotation matrix.

    The input is normalized first, so any nonzero quaternion is accepted.
    """
    w, x, y, z = _normalize_quat(_check_vec("q", q, 4))
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def matrix_to_quaternion(m: Mat3) -> Quat:
    """3x3 rotation matrix to unit quaternion (w, x, y, z), w >= 0 branch.

    Uses the Shepperd short-axis method for numerical stability.
    """
    m = _check_mat("m", m)
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > _EPS:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return _normalize_quat((w, x, y, z))


# ---------------------------------------------------------------------------
# 6D continuous representation (Zhou et al., 2019)
# ---------------------------------------------------------------------------


def matrix_to_6d(m: Mat3) -> SixD:
    """Encode R as the first two columns (6 floats, column-major).

    The 6D encoding is a continuous embedding: nearby rotations have
    nearby encodings, which makes it suitable for interpolation and
    regression targets.
    """
    m = _check_mat("m", m)
    return (
        m[0][0],
        m[1][0],
        m[2][0],
        m[0][1],
        m[1][1],
        m[2][1],
    )


def matrix_from_6d(v: SixD) -> Mat3:
    """Recover the rotation matrix from a 6D encoding via Gram-Schmidt.

    Accepts any 6-tuple whose first two vectors are linearly independent;
    the result is orthonormalized so the output is a valid rotation
    matrix even for slightly perturbed encodings.
    """
    if v is None:
        raise RotationError("v: expected 6 elements, got None")
    try:
        seq = tuple(v)
    except TypeError as exc:
        raise RotationError("v: not a sequence of numbers") from exc
    if len(seq) != 6:
        raise RotationError(f"v: expected 6 elements, got {len(seq)}")
    a = _check_vec("v[0:3]", seq[0:3], 3)
    b = _check_vec("v[3:6]", seq[3:6], 3)
    n_a = _norm3(a)
    if n_a < _EPS6:
        raise RotationError("v: first column is zero or subnormal")
    x = _scale(a, 1.0 / n_a)
    b_proj = _sub(b, _scale(x, _dot(x, b)))
    n_b = _norm3(b_proj)
    if n_b < _EPS6:
        raise RotationError("v: columns are linearly dependent")
    y = _scale(b_proj, 1.0 / n_b)
    z = _cross(x, y)
    # Return rows: each row is the dot product of the basis vectors with the standard basis
    return (
        (x[0], y[0], z[0]),
        (x[1], y[1], z[1]),
        (x[2], y[2], z[2]),
    )


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _clamp01(t: float) -> float:
    if math.isnan(t) or math.isinf(t):
        raise RotationError("t: must be finite")
    return max(0.0, min(1.0, float(t)))


def slerp(q1: Quat, q2: Quat, t: float) -> Quat:
    """Spherical linear interpolation between two quaternions.

    t is clamped to [0, 1]. The shorter arc is used (dot sign flipped
    when needed). Returns a unit quaternion; the result at t=0 is
    exactly q1 and at t=1 exactly q2 (both normalized).
    """
    a = _normalize_quat(_check_vec("q1", q1, 4))
    b = _normalize_quat(_check_vec("q2", q2, 4))
    t = _clamp01(t)
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 1.0 - _EPS32:
        # Identical rotations: linear blend, then renormalize.
        w = a[0] + t * (b[0] - a[0])
        x = a[1] + t * (b[1] - a[1])
        y = a[2] + t * (b[2] - a[2])
        z = a[3] + t * (b[3] - a[3])
        return _normalize_quat((w, x, y, z))
    theta_0 = math.acos(dot)
    sin_theta = math.sin(theta_0)
    if abs(sin_theta) < 1e-15:
        # Degenerate case: return a normalized blend
        w = a[0] + t * (b[0] - a[0])
        x = a[1] + t * (b[1] - a[1])
        y = a[2] + t * (b[2] - a[2])
        z = a[3] + t * (b[3] - a[3])
        return _normalize_quat((w, x, y, z))
    s_a = math.sin((1.0 - t) * theta_0) / sin_theta
    s_b = math.sin(t * theta_0) / sin_theta
    return _normalize_quat(
        (
            s_a * a[0] + s_b * b[0],
            s_a * a[1] + s_b * b[1],
            s_a * a[2] + s_b * b[2],
            s_a * a[3] + s_b * b[3],
        )
    )


def slerp_6d(v1: SixD, v2: SixD, t: float) -> SixD:
    """Interpolate between two 6D encodings via quaternion slerp.

    Both encodings are decoded to rotation matrices, converted to
    quaternions, slerped, and re-encoded to 6D. The output is a valid
    6D encoding of the interpolated rotation.
    """
    if v1 is None:
        raise RotationError("v1: expected 6 elements, got None")
    if v2 is None:
        raise RotationError("v2: expected 6 elements, got None")
    try:
        s1 = tuple(v1)
        s2 = tuple(v2)
    except TypeError as exc:
        raise RotationError("v1/v2: not a sequence of numbers") from exc
    if len(s1) != 6 or len(s2) != 6:
        raise RotationError("v1/v2: expected 6 elements each")
    m1 = matrix_from_6d(s1)
    m2 = matrix_from_6d(s2)
    q = slerp(matrix_to_quaternion(m1), matrix_to_quaternion(m2), t)
    return matrix_to_6d(quaternion_to_matrix(q))
