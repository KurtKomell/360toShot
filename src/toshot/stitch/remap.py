from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toshot.math3d import EQ_TO_DJI, quat_to_matrix
from toshot.source.models import LensParams

# Fallback when per-lens FOV cannot be measured (legacy / incomplete calib).
LENS_FOV_DEG = 94.0
BLEND_INNER_DEG = 87.0
BLEND_OUTER_DEG = 93.0
# Front-lens yaw trim about DJI z-up; seam patch matching on Osmo 360 puts both
# seams at dx≈0 with −1° (0° left a uniform ~6 px lon offset at 2048 wide).
FRONT_YAW_TRIM_DEG = -1.0
# Keep a thin overlap ring inside the meet angle (degrees).
_BLEND_RING_DEG = 6.0
_FOV_EDGE_MARGIN_DEG = 0.5


def lens_max_fov_deg(lens: LensParams) -> float:
    """
    Largest ray angle that still projects inside the sensor (Kannala–Brandt).
    Different fx/fy ⇒ different max FOV even for the same pixel circle.
    """
    fx, fy = float(lens.fx), float(lens.fy)
    cx, cy = float(lens.cx), float(lens.cy)
    w, h = float(lens.width), float(lens.height)
    r_img = min(cx, cy, w - 1.0 - cx, h - 1.0 - cy)
    f = 0.5 * (fx + fy)
    if f < 1e-6 or r_img < 1.0:
        return float(LENS_FOV_DEG)
    k = list(lens.k) + [0.0] * 5
    k1, k2, k3, k4, k5 = k[:5]

    def theta_d(th: float) -> float:
        t2 = th * th
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        t10 = t8 * t2
        return th * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8 + k5 * t10)

    lo, hi = 0.0, math.radians(160.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f * theta_d(mid) < r_img:
            lo = mid
        else:
            hi = mid
    return float(math.degrees(lo))


@dataclass(frozen=True)
class SphereFov:
    """Per-lens reach and blend ring (degrees from each lens's own optical axis)."""

    fov_rear: float
    fov_front: float
    rear_inner: float
    rear_outer: float
    front_inner: float
    front_outer: float

    @property
    def overlap_deg(self) -> float:
        return self.fov_rear + self.fov_front - 180.0


def sphere_fov_for_pair(rear: LensParams, front: LensParams) -> SphereFov:
    """
    Sphere sized to the rear lens FOV; the front lens overlaps into it.

    Rear is used up to its own image-circle edge and fades out only in a thin
    ring at that edge. Front stays at full weight through the overlap and only
    fades at its own rim, so the front covers the rear's edge instead of both
    being clipped to a common angle.
    """
    fr = float(max(90.0, min(lens_max_fov_deg(rear) - _FOV_EDGE_MARGIN_DEG, 120.0)))
    ff = float(max(90.0, min(lens_max_fov_deg(front) - _FOV_EDGE_MARGIN_DEG, 120.0)))
    rear_outer = fr
    rear_inner = max(80.0, fr - _BLEND_RING_DEG)
    front_outer = ff
    front_inner = max(80.0, ff - 0.5)
    return SphereFov(fr, ff, rear_inner, rear_outer, front_inner, front_outer)

_EQ_TO_DJI = EQ_TO_DJI
_quat_to_mat = quat_to_matrix


def _osvplat_world_to_cam(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """OSVplat/raw360 world→lens rows (x, y, optical z); el = 90° − pitch."""
    az = math.radians(float(yaw_deg))
    el = math.radians(90.0 - float(pitch_deg))
    z = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    x = np.array([math.sin(az), -math.cos(az), 0.0], dtype=np.float64)
    n = np.linalg.norm(x)
    x = x / n if n > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y = np.cross(z, x)
    r = math.radians(float(roll_deg))
    cr, sr = math.cos(r), math.sin(r)
    x2 = cr * x + sr * y
    y2 = -sr * x + cr * y
    return np.stack([x2, y2, z], axis=0)


def _is_osmo_pitch(lens: LensParams) -> bool:
    return abs(abs(float(lens.pitch_deg)) - 90.0) < 25.0


def _world_to_lens_matrix(
    lens: LensParams,
    *,
    ref_lens: LensParams | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Return (R_w2c, use_dji_frame).
    For Osmo: OSVplat basis; front = Q1 Q0ᵀ M0 (continuous seams per raw360-formats).
    """
    from toshot.math3d import euler_ypr_to_quat, quat_conjugate, quat_multiply

    if _is_osmo_pitch(lens) or (ref_lens is not None and _is_osmo_pitch(ref_lens)):
        base = ref_lens if ref_lens is not None else lens
        R0 = _osvplat_world_to_cam(base.yaw_deg, base.pitch_deg, base.roll_deg)
        if ref_lens is not None:
            q0 = ref_lens.extrinsic_q
            q1 = lens.extrinsic_q
            q0_ok = abs(q0[0] - 1.0) > 1e-3 or abs(q0[1]) + abs(q0[2]) + abs(q0[3]) > 1e-3
            q1_ok = abs(q1[0] - 1.0) > 1e-3 or abs(q1[1]) + abs(q1[2]) + abs(q1[3]) > 1e-3
            if q0_ok and q1_ok:
                q_rel = quat_multiply(q1, quat_conjugate(q0))
                c = math.cos(math.radians(FRONT_YAW_TRIM_DEG))
                s = math.sin(math.radians(FRONT_YAW_TRIM_DEG))
                rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                return _quat_to_mat(q_rel) @ R0 @ rz, True
            return _osvplat_world_to_cam(lens.yaw_deg, lens.pitch_deg, lens.roll_deg), True
        return R0, True

    # Legacy / Insta: euler YPR as lens→world, invert for world→lens
    euler_mag = abs(lens.yaw_deg) + abs(lens.pitch_deg) + abs(lens.roll_deg)
    q = lens.extrinsic_q
    quat_is_identity = abs(q[0] - 1.0) < 1e-3 and abs(q[1]) + abs(q[2]) + abs(q[3]) < 1e-3
    if euler_mag > 0.5:
        R = _quat_to_mat(euler_ypr_to_quat(lens.yaw_deg, lens.pitch_deg, lens.roll_deg))
        return R.T, False
    if not quat_is_identity:
        return _quat_to_mat(q).T, False
    return np.eye(3, dtype=np.float64), False


def _kannala_brandt_project(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lens: LensParams,
    *,
    fov_deg: float = LENS_FOV_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project camera rays to fisheye pixels. Returns u, v, valid, theta."""
    eps = 1e-12
    r_xy = np.sqrt(x * x + y * y)
    # No clamp on z: rays past 90° (z < 0) must keep their true angle, otherwise
    # the whole 90–94° ring collapses onto the 90° circle and smears.
    theta = np.arctan2(r_xy, z)
    valid = np.ones_like(theta, dtype=bool)
    k1, k2, k3, k4 = (list(lens.k) + [0, 0, 0, 0, 0])[:4]
    k5 = lens.k[4] if len(lens.k) > 4 else 0.0
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    t10 = t8 * t2
    theta_d = theta * (1 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8 + k5 * t10)
    scale = np.where(r_xy > eps, theta_d / r_xy, 0.0)
    u = lens.fx * x * scale + lens.cx
    v = lens.fy * y * scale + lens.cy
    valid &= (u >= 0) & (u < lens.width - 1) & (v >= 0) & (v < lens.height - 1)
    valid &= theta < (fov_deg * np.pi / 180.0)
    return u.astype(np.float32), v.astype(np.float32), valid, theta


def _angular_weight(
    theta: np.ndarray,
    valid: np.ndarray,
    *,
    blend_inner_deg: float = BLEND_INNER_DEG,
    blend_outer_deg: float = BLEND_OUTER_DEG,
) -> np.ndarray:
    """1 inside blend_inner, ramp to 0 by blend_outer (degrees)."""
    inner = float(blend_inner_deg) * np.pi / 180.0
    outer = float(blend_outer_deg) * np.pi / 180.0
    span = max(outer - inner, 1e-6)
    w = np.clip((outer - theta) / span, 0.0, 1.0)
    return np.where(valid, w, 0.0).astype(np.float32)


def build_lens_maps(
    lens: LensParams,
    out_w: int,
    out_h: int,
    *,
    ref_lens: LensParams | None = None,
    fov_deg: float | None = None,
    blend_inner_deg: float | None = None,
    blend_outer_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build xmap, ymap (float32) and angular weight for equirect out_w x out_h.

    ref_lens: when set (front lens), place this lens as Q_lens Q_refᵀ M_ref (Osmo).
    fov_deg / blend_*: sphere size so both lens edges meet (see sphere_fov_for_pair).
    """
    fov = float(LENS_FOV_DEG if fov_deg is None else fov_deg)
    bi = float(BLEND_INNER_DEG if blend_inner_deg is None else blend_inner_deg)
    bo = float(BLEND_OUTER_DEG if blend_outer_deg is None else blend_outer_deg)
    ys, xs = np.meshgrid(
        np.linspace(0.5, out_h - 0.5, out_h, dtype=np.float64),
        np.linspace(0.5, out_w - 0.5, out_w, dtype=np.float64),
        indexing="ij",
    )
    lon = (xs / out_w) * 2 * np.pi - np.pi
    lat = (0.5 - ys / out_h) * np.pi
    # Equirect direction (x right, y up, z forward)
    wx = np.cos(lat) * np.sin(lon)
    wy = np.sin(lat)
    wz = np.cos(lat) * np.cos(lon)

    R_w2c, use_dji = _world_to_lens_matrix(lens, ref_lens=ref_lens)
    if use_dji:
        dx = _EQ_TO_DJI[0, 0] * wx + _EQ_TO_DJI[0, 1] * wy + _EQ_TO_DJI[0, 2] * wz
        dy = _EQ_TO_DJI[1, 0] * wx + _EQ_TO_DJI[1, 1] * wy + _EQ_TO_DJI[1, 2] * wz
        dz = _EQ_TO_DJI[2, 0] * wx + _EQ_TO_DJI[2, 1] * wy + _EQ_TO_DJI[2, 2] * wz
    else:
        dx, dy, dz = wx, wy, wz

    xl = R_w2c[0, 0] * dx + R_w2c[0, 1] * dy + R_w2c[0, 2] * dz
    yl = R_w2c[1, 0] * dx + R_w2c[1, 1] * dy + R_w2c[1, 2] * dz
    zl = R_w2c[2, 0] * dx + R_w2c[2, 1] * dy + R_w2c[2, 2] * dz

    u, v, valid, theta = _kannala_brandt_project(xl, yl, zl, lens, fov_deg=fov)
    weight = _angular_weight(theta, valid, blend_inner_deg=bi, blend_outer_deg=bo)
    return u, v, weight


def write_pgm_pair(xmap: np.ndarray, ymap: np.ndarray, path_x: Path, path_y: Path) -> None:
    """Write FFmpeg remap-compatible 16-bit PGM maps."""
    path_x.parent.mkdir(parents=True, exist_ok=True)
    _write_pgm16(path_x, np.clip(np.rint(xmap), 0, 65535).astype(np.uint16))
    _write_pgm16(path_y, np.clip(np.rint(ymap), 0, 65535).astype(np.uint16))


def _write_pgm16(path: Path, arr: np.ndarray) -> None:
    h, w = arr.shape
    header = f"P5\n{w} {h}\n65535\n".encode("ascii")
    path.write_bytes(header + arr.astype(">u2").tobytes())


# Kept for API / export scaling; angular blend is primary for Osmo.
REFERENCE_EQUIRECT_W = 2048
REFERENCE_FEATHER_PX = 56.0


def feather_px_for(out_w: int) -> float:
    """Legacy feather width helper (distance-feather fallback)."""
    return REFERENCE_FEATHER_PX * (max(int(out_w), 1) / REFERENCE_EQUIRECT_W)


def _feather_coverage(valid: np.ndarray, feather_px: float) -> np.ndarray:
    """Distance-to-edge feather: 0 at FOV border → 1 deeper inside."""
    import cv2

    mask = (valid > 0.5).astype(np.uint8)
    if mask.max() == 0:
        return np.zeros(valid.shape, dtype=np.float32)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return np.clip(dist / max(float(feather_px), 1.0), 0.0, 1.0).astype(np.float32)


def blend_weights(
    w0: np.ndarray,
    w1: np.ndarray,
    *,
    feather_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize dual-lens weights. Angular weights from build_lens_maps preferred.

    If inputs look binary, fall back to distance feather (Insta / legacy).
    """
    # Already soft angular weights (values in (0,1) beyond pure masks)
    soft = ((w0 > 0.02) & (w0 < 0.98)).mean() + ((w1 > 0.02) & (w1 < 0.98)).mean()
    if soft < 0.002 and feather_px is not None:
        f0 = _feather_coverage(w0, feather_px)
        f1 = _feather_coverage(w1, feather_px)
    else:
        f0, f1 = w0.astype(np.float32), w1.astype(np.float32)
    # Square so the more on-axis lens wins quickly — cuts double-image ghosting
    # of nearby subjects in the thin 87–93° ring.
    f0 = f0 * f0
    f1 = f1 * f1
    s = f0 + f1
    a = np.where(s > 1e-8, f0 / s, 0.0).astype(np.float32)
    b = np.where(s > 1e-8, f1 / s, 0.0).astype(np.float32)
    return a, b
