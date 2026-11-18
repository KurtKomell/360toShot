from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from toshot.math3d import quat_to_euler_ypr, slerp_attitude
from toshot.source.models import AttitudeSample
from toshot.stitch.options import StitchOptions

log = logging.getLogger(__name__)


def angular_rate_deg(
    samples: list[AttitudeSample],
    t: float,
    dt: float = 0.002,
) -> tuple[float, float, float]:
    """Approximate yaw/pitch/roll rates in deg/s around time t."""
    if not samples:
        return 0.0, 0.0, 0.0
    q0 = slerp_attitude(samples, t - dt)
    q1 = slerp_attitude(samples, t + dt)
    y0, p0, r0 = quat_to_euler_ypr(q0)
    y1, p1, r1 = quat_to_euler_ypr(q1)

    def wrap(a: float, b: float) -> float:
        d = b - a
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        return d

    inv = 1.0 / max(2 * dt, 1e-6)
    return wrap(y0, y1) * inv, wrap(p0, p1) * inv, wrap(r0, r1) * inv


def apply_rolling_shutter(
    img: np.ndarray,
    yaw_rate: float,
    pitch_rate: float,
    *,
    readout_s: float,
) -> np.ndarray:
    """
    Row-wise warp for rolling shutter on an equirect frame.
    Mid-row is the exposure reference; top/bottom rows shift by ±0.5 * readout.
    """
    if abs(readout_s) < 1e-6 or (abs(yaw_rate) < 0.5 and abs(pitch_rate) < 0.5):
        return img
    h, w = img.shape[:2]
    # Pixel shifts: yaw → horizontal, pitch → vertical
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h - 0.5
    dt = ys * readout_s
    shift_x = (-yaw_rate * dt / 360.0) * w
    shift_y = (pitch_rate * dt / 180.0) * h

    map_x = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w)).copy()
    map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w)).copy()
    map_x += shift_x[:, None]
    map_y += shift_y[:, None]
    # Horizontal wrap for equirect
    map_x %= w
    map_y = np.clip(map_y, 0, h - 1)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def selfie_stick_weights(
    rear: np.ndarray,
    front: np.ndarray,
    base_a: np.ndarray,
    base_b: np.ndarray,
    *,
    nadir_fraction: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    In the nadir band, prefer the brighter / less stick-like lens.
    Selfie sticks are typically dark and vertically elongated near the bottom.
    """
    h, w = rear.shape[:2]
    a = base_a.copy()
    b = base_b.copy()
    y0 = int(h * (1.0 - nadir_fraction))
    band = slice(y0, h)

    gray_r = cv2.cvtColor(rear[band], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_f = cv2.cvtColor(front[band], cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Stick-likeness: dark + strong vertical gradient energy
    def stick_score(g: np.ndarray) -> np.ndarray:
        dark = np.clip((80.0 - g) / 80.0, 0.0, 1.0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        vert = np.clip((np.abs(gy) - np.abs(gx)) / 40.0, 0.0, 1.0)
        score = dark * (0.55 + 0.45 * vert)
        # Soft morphology to grow thin sticks
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15))
        score = cv2.dilate(score, k)
        return cv2.GaussianBlur(score, (0, 0), 3)

    sr = stick_score(gray_r)
    sf = stick_score(gray_f)
    # Prefer the lens with lower stick score
    prefer_rear = sf > sr + 0.05
    prefer_front = sr > sf + 0.05
    strength = np.clip(np.abs(sr - sf) * 1.5, 0.0, 1.0)

    aa = a[band]
    bb = b[band]
    aa = np.where(prefer_rear, aa + strength * (1.0 - aa), aa)
    bb = np.where(prefer_rear, 1.0 - aa, bb)
    bb = np.where(prefer_front, bb + strength * (1.0 - bb), bb)
    aa = np.where(prefer_front, 1.0 - bb, aa)
    # Also fade both near pure nadir if both look like sticks
    both = (sr > 0.55) & (sf > 0.55)
    fade = both.astype(np.float32) * 0.35
    # Keep geometry; just soften conflict — no need to zero both
    a[band] = aa
    b[band] = bb
    _ = fade  # reserved for future inpaint
    s = a + b
    s = np.where(s > 1e-6, s, 1.0)
    return (a / s).astype(np.float32), (b / s).astype(np.float32)


def _seam_path(cost: np.ndarray) -> np.ndarray:
    """
    Dynamic-programming vertical seam. cost shape (h, w_band).
    Returns column index per row (0..w_band-1).
    """
    h, w = cost.shape
    dp = cost.astype(np.float64).copy()
    back = np.zeros((h, w), dtype=np.int16)
    for y in range(1, h):
        for x in range(w):
            candidates = []
            for dx in (-1, 0, 1):
                xx = x + dx
                if 0 <= xx < w:
                    candidates.append((dp[y - 1, xx], xx))
            best_val, best_x = min(candidates, key=lambda t: t[0])
            dp[y, x] += best_val
            back[y, x] = best_x
    # Trace from bottom
    x = int(np.argmin(dp[-1]))
    path = np.empty(h, dtype=np.int32)
    for y in range(h - 1, -1, -1):
        path[y] = x
        x = int(back[y, x]) if y > 0 else x
    return path


def optical_flow_align_and_seam(
    rear: np.ndarray,
    front: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    *,
    scale: float = 0.25,
    seam_blend_px: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align front→rear with DIS optical flow in the overlap, then carve a soft seam.
    Returns (aligned_front, weight_a, weight_b).
    """
    overlap = (weight_a > 0.08) & (weight_b > 0.08)
    if not np.any(overlap):
        return front, weight_a, weight_b

    h, w = rear.shape[:2]
    sc = max(0.1, min(1.0, scale))
    sw, sh = max(16, int(w * sc)), max(8, int(h * sc))

    r_s = cv2.resize(rear, (sw, sh), interpolation=cv2.INTER_AREA)
    f_s = cv2.resize(front, (sw, sh), interpolation=cv2.INTER_AREA)
    o_s = cv2.resize(overlap.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

    gray_r = cv2.cvtColor(r_s, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.cvtColor(f_s, cv2.COLOR_BGR2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(gray_f, gray_r, None)  # front → rear
    # Zero flow outside overlap (avoid wild warps)
    flow = flow.copy()
    flow[~o_s] = 0
    flow = cv2.GaussianBlur(flow, (0, 0), 1.5)

    # Upscale flow to full resolution
    flow_full = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    flow_full *= (1.0 / sc)
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = map_x + flow_full[..., 0]
    map_y = np.clip(map_y + flow_full[..., 1], 0, h - 1)
    map_x %= w
    aligned = cv2.remap(front, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # Seam cost on luminance difference in overlap
    gr = cv2.cvtColor(rear, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gf = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(gr - gf)
    diff = np.where(overlap, diff, 1e6)

    ys, xs = np.where(overlap)
    if len(ys) == 0:
        return aligned, weight_a, weight_b

    y_min, y_max = int(ys.min()), int(ys.max())
    # Osmo has two thin seams (~90° apart). Ignore polar rows — both lenses often
    # overlap there in every column, which would merge the two seams into one band.
    mid = overlap[h // 4 : (3 * h) // 4]
    col_hits = mid.sum(axis=0)
    bands = _overlap_column_bands(col_hits > max(2, mid.shape[0] // 20))
    if not bands:
        return aligned, weight_a, weight_b

    new_a = weight_a.copy()
    half = max(1, seam_blend_px)
    for x_min, x_max in bands:
        if x_max - x_min < 2:
            continue
        band = diff[y_min : y_max + 1, x_min : x_max + 1].copy()
        ov_band = overlap[y_min : y_max + 1, x_min : x_max + 1]
        band = np.where(ov_band, band, float(band[ov_band].max()) + 1000.0 if ov_band.any() else 1e6)
        path = _seam_path(band)
        # Keep the side that originally preferred rear (a) on that side of the cut
        left_pref = float(weight_a[y_min : y_max + 1, x_min : x_min + max(1, (x_max - x_min) // 3)].mean())
        right_pref = float(weight_a[y_min : y_max + 1, x_max - max(1, (x_max - x_min) // 3) : x_max + 1].mean())
        rear_on_left = left_pref >= right_pref
        for i, y in enumerate(range(y_min, y_max + 1)):
            cut = x_min + int(path[i])
            xs_row = np.arange(w, dtype=np.float32)
            t = (xs_row - (cut - half)) / max(2 * half, 1)
            t = np.clip(t, 0.0, 1.0)
            t = t * t * (3 - 2 * t)
            row_a = (1.0 - t) if rear_on_left else t
            new_a[y] = np.where(overlap[y], row_a, new_a[y])

    new_b = np.where(overlap, 1.0 - new_a, weight_b)
    new_a = np.where(overlap, new_a, weight_a)
    s = new_a + new_b
    s = np.where(s > 1e-6, s, 1.0)
    return aligned, (new_a / s).astype(np.float32), (new_b / s).astype(np.float32)


def _overlap_column_bands(cols: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive [start, end] runs of True columns (no wrap merge)."""
    bands: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(cols.tolist()):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            bands.append((start, i - 1))
            in_run = False
    if in_run:
        bands.append((start, len(cols) - 1))
    return bands


def blend_advanced(
    rear: np.ndarray,
    front: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    *,
    options: StitchOptions,
    attitude: Sequence[AttitudeSample] | list[AttitudeSample],
    t_s: float,
    fps: float,
) -> np.ndarray:
    """Apply optical flow seam, selfie mask, rolling shutter; return BGR uint8."""
    a = weight_a
    b = weight_b
    fr = front

    if options.optical_flow:
        fr, a, b = optical_flow_align_and_seam(
            rear,
            fr,
            a,
            b,
            scale=options.flow_scale,
            seam_blend_px=options.seam_blend_px,
        )

    if options.selfie_mask:
        a, b = selfie_stick_weights(
            rear,
            fr,
            a,
            b,
            nadir_fraction=options.selfie_nadir_fraction,
        )

    blended = (
        rear.astype(np.float32) * a[..., None] + fr.astype(np.float32) * b[..., None]
    ).astype(np.uint8)

    if options.rolling_shutter and attitude:
        yaw_r, pitch_r, _ = angular_rate_deg(list(attitude), t_s)
        readout = options.rs_readout_fraction / max(fps, 1e-3)
        blended = apply_rolling_shutter(blended, yaw_r, pitch_r, readout_s=readout)

    return blended
