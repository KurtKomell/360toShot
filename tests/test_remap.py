import numpy as np

from toshot.source.models import LensParams
from toshot.stitch.remap import blend_weights, build_lens_maps, feather_px_for


def test_build_lens_maps_shapes():
    lens = LensParams(
        fx=900,
        fy=900,
        cx=960,
        cy=960,
        k=(-0.2, 0.05, 0, 0, 0),
        width=1920,
        height=1920,
        extrinsic_q=(1, 0, 0, 0),
    )
    x, y, w = build_lens_maps(lens, 64, 32)
    assert x.shape == (32, 64)
    assert y.shape == (32, 64)
    assert w.shape == (32, 64)
    assert w.sum() > 0


def test_feather_matches_preview_at_any_width():
    assert feather_px_for(2048) == 56.0
    assert feather_px_for(7680) == 56.0 * 7680 / 2048
    assert feather_px_for(1024) == 28.0


def test_blend_weights_normalize():
    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    wa, wb = blend_weights(a, b)
    assert np.allclose(wa + wb, 1.0)
