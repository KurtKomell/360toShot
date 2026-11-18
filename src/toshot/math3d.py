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


# equirect (x right, y up, z forward) → DJI camd/attitude frame (x forward, y left, z up)
EQ_TO_DJI = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotvec_to_matrix(v_deg: Sequence[float]) -> np.ndarray:
    """Rodrigues: rotation vector (degrees) → 3×3."""
    v = np.radians(np.asarray(v_deg, dtype=np.float64))
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = v / ang
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * kx + (1.0 - math.cos(ang)) * (kx @ kx)


def world_to_stitch_matrix(q: Sequence[float]) -> np.ndarray:
    """World direction (equirect axes, y up) → direction in the stitched lens-frame equirect."""
    return EQ_TO_DJI.T @ quat_to_matrix(q).T @ EQ_TO_DJI


def _yaw_matrix(psi: float) -> np.ndarray:
    """Rotation about world up taking forward (0,0,1) to (sin ψ, 0, cos ψ)."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


class AttitudeTrack:
    """
    Gyro attitude with outlier rejection.

    DJI djmd interleaves the lens calibration quaternions with the fused
    attitude (Osmo: first samples only; Avata 360: scattered through the clip),
    so each lookup takes the sample nearest the window's median gravity.
    """

    def __init__(self, samples: Sequence[AttitudeSample], window_s: float = 0.05) -> None:
        self.t = np.array([s.t_s for s in samples], dtype=np.float64)
        self.q = np.array([[s.w, s.x, s.y, s.z] for s in samples], dtype=np.float64)
        self.window_s = float(window_s)
        if len(self.q):
            self.q /= np.maximum(np.linalg.norm(self.q, axis=1, keepdims=True), 1e-12)
            w, x, y, z = self.q.T
            # Third row of R(q): world-up expressed in the attitude body frame
            up_body = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)
            self.down_stitch = -(up_body @ EQ_TO_DJI)
        else:
            self.down_stitch = np.zeros((0, 3))

    def __bool__(self) -> bool:
        return len(self.t) > 0

    def quat_at(self, t: float) -> tuple[float, float, float, float]:
        if not len(self.t):
            return (1.0, 0.0, 0.0, 0.0)
        lo = int(np.searchsorted(self.t, t - self.window_s, side="left"))
        hi = int(np.searchsorted(self.t, t + self.window_s, side="right"))
        if hi <= lo:
            i = int(min(max(np.searchsorted(self.t, t), 0), len(self.t) - 1))
            return tuple(self.q[i])  # type: ignore[return-value]
        g = self.down_stitch[lo:hi]
        med = np.median(g, axis=0)
        i = lo + int(np.argmax(g @ med))
        return tuple(self.q[i])  # type: ignore[return-value]


def lens_axes_vertical(samples: Sequence[AttitudeSample]) -> bool:
    """True when gravity runs along the lens axes (lenses look up/down, e.g. Avata 360)."""
    if not samples:
        return False
    step = max(1, len(samples) // 2000)
    track = AttitudeTrack(samples[::step])
    down = np.median(track.down_stitch, axis=0)
    return bool(abs(down[2]) > 0.7)


def _heading_axis(c0: np.ndarray) -> np.ndarray:
    """Stitch-frame axis whose world heading defines 'forward' (z unless it points at a pole)."""
    z_world = c0.T @ np.array([0.0, 0.0, 1.0])
    if abs(z_world[1]) < 0.7:
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def _heading(c: np.ndarray, axis: np.ndarray) -> float:
    a = c.T @ axis
    return math.atan2(a[0], a[2])


class Stabilizer:
    """
    3×3 S per time: output view direction → stitched lens-frame direction.

    Gravity always goes straight down (works for any lens mounting, e.g. the
    Avata 360's up/down lenses). HORIZON follows heading; LEVEL/WORLD lock it
    to t_ref. NONE (or no gyro) is identity.
    """

    def __init__(self, samples: Sequence[AttitudeSample], mode: StabMode, t_ref: float = 0.0) -> None:
        self.mode = mode
        self.track = AttitudeTrack(samples) if samples and mode != StabMode.NONE else None
        if self.track is None:
            return
        c_ref = world_to_stitch_matrix(self.track.quat_at(t_ref))
        self.axis = _heading_axis(c_ref)
        self.psi_ref = _heading(c_ref, self.axis)

    def matrix(self, t: float) -> np.ndarray:
        if self.track is None:
            return np.eye(3, dtype=np.float64)
        c = world_to_stitch_matrix(self.track.quat_at(float(t)))
        psi = _heading(c, self.axis) if self.mode == StabMode.HORIZON else self.psi_ref
        return c @ _yaw_matrix(psi)


def stabilization_matrices(
    samples: Sequence[AttitudeSample],
    times: Sequence[float],
    mode: StabMode,
) -> list[np.ndarray]:
    stab = Stabilizer(samples, mode, t_ref=float(times[0]) if len(times) else 0.0)
    return [stab.matrix(t) for t in times]
