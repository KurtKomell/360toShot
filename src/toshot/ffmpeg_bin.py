from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    pass


def require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(
            f"'{name}' not found on PATH. Install FFmpeg with HEVC decode and "
            "filters remap/v360/sendcmd (Linux package or BtbN/Gyan build)."
        )
    return path


@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    return require_bin("ffmpeg")


@lru_cache(maxsize=1)
def ffprobe_path() -> str:
    return require_bin("ffprobe")


def run(
    cmd: list[str],
    *,
    check: bool = True,
    log_stdout: bool = True,
    log_stderr: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a process, streaming stderr/stdout live (FFmpeg -stats no longer silent until exit)."""
    import threading
    import time

    if not quiet:
        log.info("exec: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    last_progress = [0.0]

    def _pump(stream, bucket: list[str], *, label: str, enabled: bool) -> None:
        assert stream is not None
        buf = ""
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            buf += chunk
            # FFmpeg progress uses \r; normal logs use \n
            while True:
                nl = buf.find("\n")
                cr = buf.find("\r")
                if nl < 0 and cr < 0:
                    break
                if nl < 0 or (cr >= 0 and cr < nl):
                    idx, sep_len = cr, 1
                else:
                    idx, sep_len = nl, 1
                line = buf[:idx].strip()
                buf = buf[idx + sep_len :]
                if not line:
                    continue
                bucket.append(line + "\n")
                if quiet or not enabled:
                    continue
                is_progress = line.startswith("frame=") or " fps=" in line and "speed=" in line
                now = time.monotonic()
                if is_progress:
                    if now - last_progress[0] < 2.0:
                        continue
                    last_progress[0] = now
                log.info("[%s] %s", label, line)
        if buf.strip():
            bucket.append(buf.strip() + "\n")
            if enabled and not quiet:
                log.info("[%s] %s", label, buf.strip())
        stream.close()

    t_out = threading.Thread(
        target=_pump, args=(proc.stdout, stdout_parts), kwargs={"label": "stdout", "enabled": log_stdout}
    )
    t_err = threading.Thread(
        target=_pump, args=(proc.stderr, stderr_parts), kwargs={"label": "ffmpeg", "enabled": log_stderr}
    )
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    returncode = proc.wait()
    out = "".join(stdout_parts)
    err = "".join(stderr_parts)
    if not quiet:
        log.info("exit %s (code=%s)", Path(cmd[0]).name, returncode)
    result = subprocess.CompletedProcess(cmd, returncode, out, err)
    if check and returncode != 0:
        raise FFmpegError(
            f"Command failed ({returncode}): {' '.join(cmd)}\n{(err or '')[-4000:]}"
        )
    return result


def probe(path: Path) -> dict:
    cmd = [
        ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    log.info("ffprobe %s", path)
    proc = run(cmd, log_stdout=False, log_stderr=True)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    log.info(
        "ffprobe ok: %d stream(s), duration=%s",
        len(streams),
        (data.get("format") or {}).get("duration"),
    )
    for s in streams:
        log.info(
            "  stream[%s] %s %s %sx%s",
            s.get("index"),
            s.get("codec_type"),
            s.get("codec_name"),
            s.get("width"),
            s.get("height"),
        )
    return data


def probe_duration(path: Path) -> float:
    """Fast duration-only probe (no stream dump). Returns seconds or 0."""
    cmd = [
        ffprobe_path(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = run(cmd, check=False, log_stdout=False, log_stderr=False, quiet=True)
    try:
        return max(0.0, float((proc.stdout or "").strip() or 0.0))
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class EncoderChoice:
    video: str
    pix_fmt: str
    extra: list[str]
    note: str = ""


# auto follows the source (HEVC when 10-bit). ProRes ignores CRF and bitrate.
CODECS = ("auto", "h264", "hevc", "prores", "prores4444")
PRESETS = ("veryfast", "fast", "medium", "slow")


@dataclass(frozen=True)
class EncodeSettings:
    """User-facing encode knobs. Defaults match the previous fixed CRF 18 path."""

    codec: str = "auto"
    crf: int = 18
    bitrate: str | None = None
    preset: str = "medium"
    audio_bitrate: str = "320k"
    bit_depth: int | None = None

    def summary(self) -> str:
        rate = self.bitrate if self.bitrate else f"crf={self.crf}"
        depth = "auto" if self.bit_depth is None else str(self.bit_depth)
        return (
            f"codec={self.codec} {rate} preset={self.preset} "
            f"depth={depth} audio={self.audio_bitrate}"
        )


# NVENC rejects maxrate above ~2.147e9 bit/s (~2147 Mbit/s). Keep headroom.
_MAX_VIDEO_BITRATE_MBIT = 500.0


def normalize_bitrate(value: str | None, *, default_unit: str = "k") -> str | None:
    """Accept ``80000``, ``80M`` or ``1500k``. A bare number uses ``default_unit`` (k for video)."""
    if value is None:
        return None
    s = "".join(str(value).strip().split()).lower()
    if not s:
        return None
    unit = default_unit
    if s.endswith("m"):
        unit, s = "M", s[:-1]
    elif s.endswith("k"):
        unit, s = "k", s[:-1]
    try:
        num = float(s)
    except ValueError as exc:
        raise ValueError(f"Bitrate nicht lesbar: {value}") from exc
    if num <= 0:
        raise ValueError(f"Bitrate muss größer als 0 sein: {value}")
    mbit = num / 1000.0 if unit == "k" else num
    if mbit > _MAX_VIDEO_BITRATE_MBIT:
        raise ValueError(
            f"Video-Bitrate {mbit:g} Mbit/s ist zu hoch "
            f"(max {_MAX_VIDEO_BITRATE_MBIT:g} Mbit/s für NVENC). "
            f"Feld ist kbit/s — für 20 Mbit/s „20000“ eintragen."
        )
    text = str(int(num)) if num.is_integer() else f"{num:g}"
    return f"{text}{unit}"


def normalize_audio_bitrate(value: str | None) -> str:
    """Bare numbers are kbit/s (``320`` → ``320k``)."""
    return normalize_bitrate(value or "320", default_unit="k") or "320k"


def audio_encode_args(settings: EncodeSettings | None = None) -> list[str]:
    cfg = settings or EncodeSettings()
    return ["-c:a", "aac", "-b:a", normalize_audio_bitrate(cfg.audio_bitrate)]


def _double_rate(rate: str) -> str:
    unit = rate[-1]
    num = float(rate[:-1]) * 2
    text = str(int(num)) if num.is_integer() else f"{num:g}"
    return f"{text}{unit}"


def _nvenc_preset(name: str) -> str:
    return {"veryfast": "fast", "fast": "fast", "medium": "medium", "slow": "slow"}.get(
        name, "medium"
    )


def _rate_extra(*, kind: str, crf: int, bitrate: str | None, preset: str) -> list[str]:
    if bitrate:
        buf = _double_rate(bitrate)
        if kind == "nvenc":
            return [
                "-rc",
                "vbr",
                "-b:v",
                bitrate,
                "-maxrate",
                bitrate,
                "-bufsize",
                buf,
                "-preset",
                _nvenc_preset(preset),
                "-spatial-aq",
                "1",
            ]
        extra = ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", buf, "-preset", preset]
        if kind == "x265":
            extra += ["-x265-params", "log-level=error"]
        return extra
    if kind == "nvenc":
        return [
            "-rc",
            "constqp",
            "-qp",
            str(crf),
            "-preset",
            _nvenc_preset(preset),
            "-spatial-aq",
            "1",
        ]
    if kind == "x265":
        return ["-crf", str(crf), "-preset", preset, "-x265-params", "log-level=error"]
    return ["-crf", str(crf), "-preset", preset]


@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    proc = run(
        [ffmpeg_path(), "-hide_banner", "-encoders"],
        check=False,
        log_stdout=False,
        log_stderr=False,
        quiet=True,
    )
    found: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            found.add(parts[1])
    return found


def choose_video_encoder(
    *,
    prefer_hevc: bool,
    bit_depth: int,
    for_shot: bool = False,
    settings: EncodeSettings | None = None,
) -> EncoderChoice:
    cfg = settings or EncodeSettings()
    if cfg.codec not in CODECS:
        raise FFmpegError(f"Unbekannter Codec {cfg.codec!r}; erlaubt: {', '.join(CODECS)}")
    if cfg.preset not in PRESETS:
        raise FFmpegError(f"Unbekanntes Preset {cfg.preset!r}; erlaubt: {', '.join(PRESETS)}")
    if not 0 <= int(cfg.crf) <= 51:
        raise FFmpegError("CRF muss zwischen 0 und 51 liegen")
    if cfg.bit_depth not in (None, 8, 10):
        raise FFmpegError("Farbtiefe muss 8, 10 oder auto sein")

    crf = int(cfg.crf)
    bitrate = normalize_bitrate(cfg.bitrate) if cfg.bitrate else None
    encs = available_encoders()

    codec = cfg.codec
    depth = bit_depth
    if codec == "auto":
        if for_shot:
            codec = "h264"
            if cfg.bit_depth is None:
                depth = 8
        else:
            codec = "hevc" if prefer_hevc else "h264"
    if cfg.bit_depth in (8, 10):
        depth = cfg.bit_depth

    if codec in {"prores", "prores4444"}:
        if "prores_ks" not in encs:
            raise FFmpegError("ProRes (prores_ks) fehlt in diesem FFmpeg")
        if bitrate:
            log.info("ProRes ignoriert die Bitrate und nutzt das Profil")
        profile = "4" if codec == "prores4444" else "3"
        pix = "yuv444p10le" if codec == "prores4444" else "yuv422p10le"
        note = "ProRes 4444" if codec == "prores4444" else "ProRes 422 HQ"
        return EncoderChoice(
            video="prores_ks",
            pix_fmt=pix,
            extra=["-profile:v", profile, "-vendor", "apl0"],
            note=note,
        )

    ten = depth >= 10
    pix = "yuv420p10le" if ten else "yuv420p"
    if codec == "hevc":
        options = [
            ("hevc_nvenc", "nvenc", pix, "NVENC HEVC"),
            ("libx265", "x265", pix, "libx265"),
        ]
    else:
        options = []
        if not ten:
            options.append(("h264_nvenc", "nvenc", "yuv420p", "NVENC H.264"))
        options.append(("libx264", "x264", pix, "libx264"))

    for name, kind, pix_fmt, note in options:
        if name not in encs:
            continue
        extra = _rate_extra(kind=kind, crf=crf, bitrate=bitrate, preset=cfg.preset)
        return EncoderChoice(video=name, pix_fmt=pix_fmt, extra=extra, note=note)

    if codec == "hevc":
        log.warning("No HEVC encoder found; falling back to H.264")
        for name, kind, note in (
            ("h264_nvenc", "nvenc", "NVENC H.264"),
            ("libx264", "x264", "libx264"),
        ):
            if name not in encs:
                continue
            extra = _rate_extra(kind=kind, crf=crf, bitrate=bitrate, preset=cfg.preset)
            return EncoderChoice(video=name, pix_fmt="yuv420p", extra=extra, note=note)

    raise FFmpegError("No usable H.264/HEVC/ProRes encoder found in this FFmpeg build")


def libx264_encoder(settings: EncodeSettings | None = None) -> EncoderChoice:
    """Software H.264 used when NVENC or ProRes rejects the stitch command."""
    cfg = settings or EncodeSettings()
    preset = cfg.preset if cfg.preset in PRESETS else "veryfast"
    crf = int(cfg.crf) if 0 <= int(cfg.crf) <= 51 else 18
    bitrate = normalize_bitrate(cfg.bitrate) if cfg.bitrate else None
    return EncoderChoice(
        video="libx264",
        pix_fmt="yuv420p",
        extra=_rate_extra(kind="x264", crf=crf, bitrate=bitrate, preset=preset),
        note="libx264 fallback",
    )


def escape_filter_path(path: Path | str) -> str:
    """Escape a filesystem path for use inside an FFmpeg filter option value.

    Drive letters contain ``:``, which is also the filter option separator.
    Quote the value and escape ``:`` so ``C:/tmp/x`` becomes ``'C\\:/tmp/x'``.
    """
    s = Path(path).resolve().as_posix().replace("\\", "/")
    s = s.replace("'", r"\'").replace(":", r"\:")
    return f"'{s}'"


def has_filter(name: str) -> bool:
    proc = run(
        [ffmpeg_path(), "-hide_banner", "-filters"],
        check=False,
        log_stdout=False,
        log_stderr=False,
        quiet=True,
    )
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False
