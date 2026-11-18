import pytest

from toshot.cli import _encode_settings, build_parser
from toshot.ffmpeg_bin import (
    EncodeSettings,
    FFmpegError,
    choose_video_encoder,
    libx264_encoder,
    normalize_audio_bitrate,
    normalize_bitrate,
)

ENCODERS = {"libx264", "libx265", "h264_nvenc", "hevc_nvenc", "prores_ks"}


@pytest.fixture
def encoders(monkeypatch):
    monkeypatch.setattr("toshot.ffmpeg_bin.available_encoders", lambda: set(ENCODERS))


def test_normalize_bitrate_units():
    assert normalize_bitrate("80") == "80M"
    assert normalize_bitrate("80M") == "80M"
    assert normalize_bitrate(" 12.5 m ") == "12.5M"
    assert normalize_bitrate("1500k") == "1500k"
    assert normalize_bitrate(None) is None
    assert normalize_bitrate("  ") is None
    assert normalize_audio_bitrate("192") == "192k"
    assert normalize_audio_bitrate("256K") == "256k"
    with pytest.raises(ValueError):
        normalize_bitrate("0")
    with pytest.raises(ValueError):
        normalize_bitrate("schnell")


def test_default_hevc_stays_crf(encoders):
    enc = choose_video_encoder(prefer_hevc=True, bit_depth=10)
    assert enc.video == "hevc_nvenc"
    assert enc.pix_fmt == "yuv420p10le"
    assert enc.extra[enc.extra.index("-qp") + 1] == "18"
    assert "-b:v" not in enc.extra


def test_bitrate_overrides_crf_and_prefers_nvenc(encoders):
    enc = choose_video_encoder(
        prefer_hevc=True,
        bit_depth=10,
        settings=EncodeSettings(codec="hevc", crf=18, bitrate="80", preset="slow"),
    )
    assert enc.video == "hevc_nvenc"
    assert enc.extra[enc.extra.index("-b:v") + 1] == "80M"
    assert enc.extra[enc.extra.index("-maxrate") + 1] == "80M"
    assert enc.extra[enc.extra.index("-bufsize") + 1] == "160M"
    assert "-qp" not in enc.extra
    assert "-crf" not in enc.extra
    assert enc.extra[enc.extra.index("-preset") + 1] == "slow"


def test_h264_bitrate_uses_libx264_when_nvenc_missing(monkeypatch):
    monkeypatch.setattr("toshot.ffmpeg_bin.available_encoders", lambda: {"libx264"})
    enc = choose_video_encoder(
        prefer_hevc=False,
        bit_depth=8,
        settings=EncodeSettings(codec="h264", bitrate="40M", preset="fast", crf=23),
    )
    assert enc.video == "libx264"
    assert "-crf" not in enc.extra
    assert enc.extra[enc.extra.index("-b:v") + 1] == "40M"
    assert enc.extra[enc.extra.index("-preset") + 1] == "fast"


def test_ten_bit_h264_skips_nvenc(encoders):
    enc = choose_video_encoder(
        prefer_hevc=False,
        bit_depth=8,
        settings=EncodeSettings(codec="h264", bit_depth=10, crf=16),
    )
    assert enc.video == "libx264"
    assert enc.pix_fmt == "yuv420p10le"
    assert enc.extra[enc.extra.index("-crf") + 1] == "16"


def test_prores_profile_ignores_bitrate(encoders):
    enc = choose_video_encoder(
        prefer_hevc=False,
        bit_depth=8,
        settings=EncodeSettings(codec="prores4444", bitrate="200M"),
    )
    assert enc.video == "prores_ks"
    assert enc.pix_fmt == "yuv444p10le"
    assert enc.extra[enc.extra.index("-profile:v") + 1] == "4"
    assert "-b:v" not in enc.extra


def test_shot_auto_stays_h264(encoders):
    enc = choose_video_encoder(prefer_hevc=True, bit_depth=10, for_shot=True)
    assert enc.video == "h264_nvenc"
    assert enc.pix_fmt == "yuv420p"


def test_libx264_fallback_keeps_bitrate():
    enc = libx264_encoder(EncodeSettings(bitrate="60", preset="veryfast", crf=18))
    assert enc.video == "libx264"
    assert enc.extra[enc.extra.index("-b:v") + 1] == "60M"
    assert enc.extra[enc.extra.index("-preset") + 1] == "veryfast"


def test_crf_out_of_range(encoders):
    with pytest.raises(FFmpegError):
        choose_video_encoder(
            prefer_hevc=False,
            bit_depth=8,
            settings=EncodeSettings(crf=80),
        )


def test_cli_encode_flags():
    args = build_parser().parse_args(
        [
            "equirect",
            "clip.OSV",
            "--codec",
            "hevc",
            "--bitrate",
            "80",
            "--crf",
            "20",
            "--preset",
            "slow",
            "--bit-depth",
            "10",
            "--audio-bitrate",
            "256",
        ]
    )
    cfg = _encode_settings(args)
    assert cfg.codec == "hevc"
    assert cfg.bitrate == "80M"
    assert cfg.crf == 20
    assert cfg.preset == "slow"
    assert cfg.bit_depth == 10
    assert cfg.audio_bitrate == "256k"
