from toshot.source.probe_streams import _is_thumbnail, _parse_rate


def test_parse_rate_fraction():
    assert _parse_rate("50/1") == 50.0
    assert _parse_rate("0/0") is None
    assert abs((_parse_rate("30000/1001") or 0) - 29.97002997) < 1e-4


def test_thumbnail_mjpeg_skipped():
    assert _is_thumbnail(
        {
            "codec_name": "mjpeg",
            "width": 688,
            "height": 344,
            "disposition": {"attached_pic": 1},
        }
    )
    assert not _is_thumbnail(
        {
            "codec_name": "hevc",
            "width": 3840,
            "height": 3840,
            "disposition": {"attached_pic": 0},
        }
    )
