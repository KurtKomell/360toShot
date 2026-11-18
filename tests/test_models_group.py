from pathlib import Path

from toshot.source.models import (
    Calibration,
    CameraFamily,
    Capture,
    LensLayout,
    LensParams,
)
from toshot.insta.reader import insta_capture_key, insta_lens_code


def test_equirect_size_from_lens_edge():
    lens = LensParams(fx=1000, fy=1000, cx=1920, cy=1920, k=(0, 0, 0, 0, 0), width=3840, height=3840)
    cap = Capture(
        family=CameraFamily.DJI,
        layout=LensLayout.DUAL_TRACK,
        master_paths=[Path("CAM.OSV")],
        calibration=Calibration(rear=lens, front=lens),
        video_streams=[],
    )
    # without streams, lens_edge falls back to 3840 default when streams empty —
    # set via video stream
    from toshot.source.models import VideoStreamInfo

    cap.video_streams = [
        VideoStreamInfo(0, 3840, 3840, 30, 1, "hevc", "yuv420p10le", 10),
        VideoStreamInfo(1, 3840, 3840, 30, 1, "hevc", "yuv420p10le", 10),
    ]
    assert cap.equirect_size() == (7680, 3840)
    assert cap.equirect_size(3840) == (3840, 1920)


def test_insta_name_parse():
    p = Path("VID_20240101_120000_00_001.insv")
    assert insta_capture_key(p) == ("20240101", "120000", "001")
    assert insta_lens_code(p) == "00"
    assert insta_lens_code(Path("VID_20240101_120000_10_001.insv")) == "10"
