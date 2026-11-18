from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from toshot.source.models import AttitudeSample, StabMode


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conjugate(q: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_normalize(q: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def quat_to_euler_ypr(q: Sequence[float]) -> tuple[float, float, float]:
    """Return yaw, pitch, roll in degrees (ZYX)."""
    w, x, y, z = quat_normalize(q)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return yaw, pitch, roll


def euler_ypr_to_quat(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    y = math.radians(yaw) * 0.5
    p = math.radians(pitch) * 0.5
    r = math.radians(roll) * 0.5
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return quat_normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def rotate_vec(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    qv = (0.0, v[0], v[1], v[2])
    r = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return r[1], r[2], r[3]


def slerp_attitude(samples: list[AttitudeSample], t: float) -> tuple[float, float, float, float]:
    if not samples:
        return (1.0, 0.0, 0.0, 0.0)
    if t <= samples[0].t_s:
        s = samples[0]
        return (s.w, s.x, s.y, s.z)
    if t >= samples[-1].t_s:
        s = samples[-1]
        return (s.w, s.x, s.y, s.z)
    lo, hi = 0, len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid].t_s <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    if b.t_s <= a.t_s:
        return (a.w, a.x, a.y, a.z)
    u = (t - a.t_s) / (b.t_s - a.t_s)
    # Nlerp for speed
    w = a.w + (b.w - a.w) * u
    x = a.x + (b.x - a.x) * u
    y = a.y + (b.y - a.y) * u
    z = a.z + (b.z - a.z) * u
    return quat_normalize((w, x, y, z))


def stabilization_rotation(
    q_cam: Sequence[float],
    mode: StabMode,
    q0: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """
    Return yaw/pitch/roll degrees to apply on equirect via v360 so the view is stabilized.
    """
    if mode == StabMode.NONE or q_cam == (1.0, 0.0, 0.0, 0.0):
        return 0.0, 0.0, 0.0

    yaw, pitch, roll = quat_to_euler_ypr(q_cam)

    if mode == StabMode.WORLD:
        # Cancel full orientation relative to first sample if provided
        if q0 is not None:
            rel = quat_multiply(quat_conjugate(q0), q_cam)
            yaw, pitch, roll = quat_to_euler_ypr(rel)
        return -yaw, -pitch, -roll

    if mode == StabMode.LEVEL:
        # Level pitch/roll; keep world yaw locked to first frame
        yaw0 = 0.0
        if q0 is not None:
            yaw0, _, _ = quat_to_euler_ypr(q0)
        return -(yaw - yaw0), -pitch, -roll

    # HORIZON: level pitch/roll, follow yaw
    return 0.0, -pitch, -roll


def frame_rotations(
    samples: list[AttitudeSample],
    fps: float,
    n_frames: int,
    mode: StabMode,
) -> list[tuple[float, float, float]]:
    q0 = (samples[0].w, samples[0].x, samples[0].y, samples[0].z) if samples else None
    out: list[tuple[float, float, float]] = []
    for i in range(n_frames):
        t = i / fps if fps > 0 else 0.0
        q = slerp_attitude(samples, t)
        out.append(stabilization_rotation(q, mode, q0=q0))
    return out
