from __future__ import annotations

from toshot.ffmpeg_bin import probe
from toshot.source.models import VideoStreamInfo


def _parse_rate(rate: object) -> float | None:
    if rate is None or rate in ("0/0", "N/A", ""):
        return None
    if isinstance(rate, str) and "/" in rate:
        a, b = rate.split("/", 1)
        try:
            den = float(b)
            num = float(a)
        except ValueError:
            return None
        if den == 0:
            return None
        val = num / den
        return val if val > 0 else None
    try:
        val = float(rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _is_thumbnail(stream: dict) -> bool:
    disp = stream.get("disposition") or {}
    if disp.get("attached_pic"):
        return True
    codec = str(stream.get("codec_name") or "").lower()
    w = int(stream.get("width") or 0)
    h = int(stream.get("height") or 0)
    # Cover / preview MJPEG (e.g. 688x344) must not override HEVC fps
    if codec == "mjpeg" and w * h < 1_000_000:
        return True
    return False


def streams_from_probe(
    path,
) -> tuple[list[VideoStreamInfo], int | None, float, float, int, int]:
    """Return (videos, audio_idx, fps, duration_s, bit_depth, frame_count).

    Skips attached-pic / small MJPEG thumbs so they cannot overwrite real fps.
    fps and frame_count come from the largest real video stream.
    """
    info = probe(path)
    videos: list[VideoStreamInfo] = []
    audio: int | None = None
    duration = float(info.get("format", {}).get("duration") or 0.0)
    bit_depth = 8
    frame_counts: list[int] = []

    for s in info.get("streams", []) or []:
        idx = int(s.get("index", 0))
        ctype = s.get("codec_type")
        if ctype == "video":
            if _is_thumbnail(s):
                continue
            # Prefer nominal r_frame_rate (Osmo reports 50/1 here)
            fps_v = (
                _parse_rate(s.get("r_frame_rate"))
                or _parse_rate(s.get("avg_frame_rate"))
                or 0.0
            )
            pix = s.get("pix_fmt") or ""
            bd = 10 if ("10" in pix or "p010" in pix) else int(s.get("bits_per_raw_sample") or 8)
            bit_depth = max(bit_depth, bd)
            nb = s.get("nb_frames")
            try:
                nbi = int(nb) if nb not in (None, "N/A", "") else 0
            except (TypeError, ValueError):
                nbi = 0
            if nbi > 0:
                frame_counts.append(nbi)
            videos.append(
                VideoStreamInfo(
                    index=idx,
                    width=int(s.get("width") or 0),
                    height=int(s.get("height") or 0),
                    fps=fps_v,
                    duration_s=float(s.get("duration") or duration),
                    codec=str(s.get("codec_name") or ""),
                    pix_fmt=pix,
                    bit_depth=bd,
                )
            )
        elif ctype == "audio" and audio is None:
            audio = idx

    if videos:
        primary = max(videos, key=lambda v: v.width * v.height)
        fps = primary.fps if primary.fps > 0 else 30.0
    else:
        fps = 30.0

    if frame_counts:
        frame_count = max(frame_counts)
    elif duration > 0 and fps > 0:
        frame_count = max(1, int(round(duration * fps)))
    else:
        frame_count = 0

    return videos, audio, fps, duration, bit_depth, frame_count
