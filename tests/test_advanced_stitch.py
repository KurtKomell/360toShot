import numpy as np

from toshot.math3d import euler_ypr_to_quat
from toshot.source.models import AttitudeSample
from toshot.stitch.advanced import (
    _overlap_column_bands,
    angular_rate_deg,
    apply_rolling_shutter,
    optical_flow_align_and_seam,
    selfie_stick_weights,
)
from toshot.stitch.options import StitchOptions


def test_stitch_options_fast():
    opts = StitchOptions.fast()
    assert not opts.advanced
    assert StitchOptions().advanced


def test_angular_rate_nonzero():
    samples = [
        AttitudeSample(0.0, *euler_ypr_to_quat(0, 0, 0)),
        AttitudeSample(0.1, *euler_ypr_to_quat(10, 0, 0)),
    ]
    yaw_r, pitch_r, roll_r = angular_rate_deg(samples, 0.05, dt=0.02)
    assert yaw_r > 50  # ~100 deg/s


def test_rolling_shutter_shape():
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[:, 40:50] = 255
    out = apply_rolling_shutter(img, yaw_rate=90.0, pitch_rate=0.0, readout_s=0.02)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_selfie_prefers_brighter_nadir():
    h, w = 80, 160
    rear = np.full((h, w, 3), 200, dtype=np.uint8)
    front = np.full((h, w, 3), 200, dtype=np.uint8)
    # Dark stick only on front nadir
    front[int(h * 0.9) :, w // 2 - 3 : w // 2 + 3] = 10
    a = np.full((h, w), 0.5, dtype=np.float32)
    b = np.full((h, w), 0.5, dtype=np.float32)
    aa, bb = selfie_stick_weights(rear, front, a, b, nadir_fraction=0.15)
    # Nadir is hard single-lens (no soft mix); stick on front → prefer rear
    assert aa[-5, w // 2] >= bb[-5, w // 2]
    assert float(aa[-5:].mean()) >= 0.99 or float(bb[-5:].mean()) >= 0.99


def test_optical_flow_returns_aligned():
    h, w = 48, 96
    rear = np.zeros((h, w, 3), dtype=np.uint8)
    front = np.zeros((h, w, 3), dtype=np.uint8)
    rear[:, 40:50] = 255
    front[:, 42:52] = 255  # slight shift
    a = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)
    a[:, 20:70] = 0.5
    b[:, 20:70] = 0.5
    aligned, aa, bb = optical_flow_align_and_seam(
        rear, front, a, b, scale=0.5, seam_blend_px=2
    )
    assert aligned.shape == front.shape
    assert aa.shape == a.shape
    assert np.all(np.isfinite(aa))


def test_overlap_column_bands_two_seams():
    cols = np.zeros(100, dtype=bool)
    cols[10:20] = True
    cols[70:85] = True
    assert _overlap_column_bands(cols) == [(10, 19), (70, 84)]


def test_optical_flow_two_seam_bands():
    h, w = 64, 128
    rear = np.full((h, w, 3), 40, dtype=np.uint8)
    front = np.full((h, w, 3), 40, dtype=np.uint8)
    rear[:, 20:28] = 220
    front[:, 22:30] = 220
    rear[:, 90:98] = 200
    front[:, 92:100] = 200
    a = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)
    a[:, 15:35] = 0.5
    b[:, 15:35] = 0.5
    a[:, 85:105] = 0.5
    b[:, 85:105] = 0.5
    aligned, aa, bb = optical_flow_align_and_seam(
        rear, front, a, b, scale=0.5, seam_blend_px=2
    )
    assert aligned.shape == front.shape
    assert np.all(np.isfinite(aa + bb))
    # Both bands should leave some carved weights away from pure 0.5
    assert ((aa[:, 15:35] > 0.7) | (aa[:, 15:35] < 0.3)).any()
