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
    # Hard pick per pixel where stick scores disagree. Do not force the whole
    # nadir disk to one lens — that kept Osmo Front housing in BL/BR.
    aa = np.where(prefer_rear, 1.0, aa)
    bb = np.where(prefer_rear, 0.0, bb)
    bb = np.where(prefer_front, 1.0, bb)
    aa = np.where(prefer_front, 0.0, aa)
    _ = strength
    a[band] = aa
    b[band] = bb
    return a.astype(np.float32), b.astype(np.float32)


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


def hard_cut_at_lens_meet(
    rear: np.ndarray,
    front: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    *,
    scale: float = 0.25,
    seam_blend_px: int = 0,
    seam_expand_px: int = 0,
    vertical_lenses: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find where the two lens FOVs meet, join there, cut the rest.

    weight_a/b encode angular distance from each optical axis (from camd FOV).
    The meet column is where rear≈front preference in the overlap ring. Everything
    left of that cut keeps only rear; right keeps only front — no overlapping video.
    """
    _ = (scale, seam_blend_px, seam_expand_px, rear, front)  # API compat / no warp
    h, w = weight_a.shape
    # Geometric meet: prefer the lens closer to its optical axis
    hard_a = (weight_a >= weight_b).astype(np.float32)

    overlap = (weight_a > 0.08) & (weight_b > 0.08)
    bands: list[tuple[int, int]] = []
    if np.any(overlap):
        mid = overlap[h // 4 : (3 * h) // 4]
        col_hits = mid.sum(axis=0)
        bands = _overlap_column_bands(col_hits > max(2, mid.shape[0] // 20))
        # Mid-latitude row: column where |a-b| is smallest = FOV meet angle
        y_mid = h // 2
        for x_min, x_max in bands:
            if x_max <= x_min:
                continue
            diff = np.abs(weight_a[y_mid, x_min : x_max + 1] - weight_b[y_mid, x_min : x_max + 1])
            mid_col = (x_min + x_max) // 2
            rear_on_left = float(weight_a[y_mid, x_min : mid_col + 1].mean()) >= float(
                weight_a[y_mid, mid_col : x_max + 1].mean()
            )
            # Both lenses at full weight ⇒ |a−b| ties over a plateau.
            # Osmo: rear-side end → front covers the whole overlap.
            # Avata: front covers most of the plateau (per-side bias).
            tie = np.flatnonzero(diff <= float(diff.min()) + 1e-4)
            if vertical_lenses:
                bias = 0.38 if mid_col < w // 2 else 0.28
                extra_px = 0
                n_tie = len(tie) - 1
                if rear_on_left:
                    i = int(np.clip(round(n_tie * bias), 0, n_tie))
                    meet = int(x_min + int(tie[i]))
                    if extra_px:
                        meet = min(int(x_max), meet + extra_px)
                else:
                    i = int(np.clip(round(n_tie * (1.0 - bias)), 0, n_tie))
                    meet = int(x_min + int(tie[i]))
                    if extra_px:
                        meet = max(int(x_min), meet - extra_px)
            else:
                meet = int(x_min + int(tie[0] if rear_on_left else tie[-1]))
            xs = np.arange(w)
            # Cut at meet for all rows: discard the other lens past this angle
            if rear_on_left:
                # rear owns (-inf, meet], front owns (meet, +inf) — but only
                # rewrite inside a generous strip around this seam so the other
                # seam stays independent
                pad = max(32, (x_max - x_min) * 3)
                x0 = max(0, meet - pad)
                x1 = min(w - 1, meet + pad)
                strip = (xs >= x0) & (xs <= x1)
                row = (xs <= meet).astype(np.float32)
            else:
                pad = max(32, (x_max - x_min) * 3)
                x0 = max(0, meet - pad)
                x1 = min(w - 1, meet + pad)
                strip = (xs >= x0) & (xs <= x1)
                row = (xs >= meet).astype(np.float32)
            hard_a = np.where(strip[None, :], row[None, :], hard_a)

    # Nadir / zenith: both lenses see the pole with large parallax.
    # Per-longitude column at both caps — a whole-disk pick exposed Osmo
    # housing holes at BL/BR and matching zenith gaps.
    nadir_frac = 0.20
    zenith_frac = 0.12
    y_n = int(h * (1.0 - nadir_frac))
    y_z = max(1, int(h * zenith_frac))
    if not vertical_lenses:
        nadir_rear = weight_a[y_n:, :].sum(axis=0) >= weight_b[y_n:, :].sum(axis=0)
        hard_a[y_n:, :] = nadir_rear.astype(np.float32)[None, :]
        zenith_rear = weight_a[:y_z, :].sum(axis=0) >= weight_b[:y_z, :].sum(axis=0)
        hard_a[:y_z, :] = zenith_rear.astype(np.float32)[None, :]

    hard_a = (hard_a >= 0.5).astype(np.float32)
    hard_b = (1.0 - hard_a).astype(np.float32)

    return front, hard_a, hard_b


def optical_flow_align_and_seam(
    rear: np.ndarray,
    front: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    *,
    scale: float = 0.25,
    seam_blend_px: int = 0,
    seam_expand_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """API alias: geometric meet-angle hard cut (no overlap blend, no warp)."""
    return hard_cut_at_lens_meet(
        rear,
        front,
        weight_a,
        weight_b,
        scale=scale,
        seam_blend_px=seam_blend_px,
        seam_expand_px=seam_expand_px,
    )


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
        fr, a, b = hard_cut_at_lens_meet(
            rear,
            fr,
            a,
            b,
            scale=options.flow_scale,
            seam_blend_px=options.seam_blend_px,
            seam_expand_px=options.seam_expand_px,
            vertical_lenses=options.vertical_lenses,
        )

    if options.selfie_mask and not options.vertical_lenses:
        a, b = selfie_stick_weights(
            rear,
            fr,
            a,
            b,
            nadir_fraction=options.selfie_nadir_fraction,
        )

    if options.seam_blend_px > 0:
        # Osmo-style feather across the vertical meet: soft only in longitude.
        k = max(3, int(options.seam_blend_px) | 1)
        a = cv2.blur(a.astype(np.float32), (k, 1), borderType=cv2.BORDER_REPLICATE)
        b = (1.0 - a).astype(np.float32)

    blended = (
        rear.astype(np.float32) * a[..., None] + fr.astype(np.float32) * b[..., None]
    ).astype(np.uint8)

    if options.rolling_shutter and attitude and not options.vertical_lenses:
        yaw_r, pitch_r, _roll_r = angular_rate_deg(list(attitude), t_s)
        readout = options.rs_readout_fraction / max(fps, 1e-3)
        blended = apply_rolling_shutter(blended, yaw_r, pitch_r, readout_s=readout)

    return blended
