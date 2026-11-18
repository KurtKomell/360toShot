from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from toshot.ffmpeg_bin import (
    EncodeSettings,
    audio_encode_args,
    choose_video_encoder,
    escape_filter_path,
    ffmpeg_path,
    run,
)
from toshot.source.models import Capture, StabMode
from toshot.stitch.options import StitchOptions
from toshot.stitch.pipeline import stitch_equirect
from toshot.track.tracker import CameraPath, make_proxy_equirect, track_subject

log = logging.getLogger(__name__)


def export_shot(
    capture: Capture,
    output: Path,
    *,
    stab: StabMode = StabMode.WORLD,
    fov: float = 90.0,
    width: int = 1920,
    height: int = 1080,
    class_name: str = "person",
    model_name: str = "yolo11n.pt",
    track_id: int | None = None,
    path: CameraPath | None = None,
    options: StitchOptions | None = None,
    encode: EncodeSettings | None = None,
) -> Path:
    """
    Track on a downscaled master equirect, then reframe full-res master to a flat shot.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opts = options or StitchOptions()

    with tempfile.TemporaryDirectory(prefix="toshot_shot_") as tmp:
        work = Path(tmp)
        if path is None:
            # Low-res stitch from OSV/INSV only (never LRF/LRV)
            track_opts = StitchOptions.fast()
            proxy_eq = make_proxy_equirect(
                capture, work, stab=stab, width=1920, options=track_opts
            )
            path = track_subject(
                proxy_eq,
                class_name=class_name,
                model_name=model_name,
                track_id=track_id,
            )
            log.info("Selected track_id=%s (%d frames)", path.track_id, len(path))

        master_eq = work / f"{capture.id}_master_equirect.mp4"
        stitch_equirect(
            capture,
            master_eq,
            stab=stab,
            width=None,
            source=capture.render_source(),
            options=opts,
        )

        cmd_file = work / "look.cmd"
        _write_look_cmd(cmd_file, path, fov=fov)

        cfg = encode or EncodeSettings()
        log.info("Shot encode %s", cfg.summary())
        enc = choose_video_encoder(
            prefer_hevc=cfg.codec == "hevc" or (cfg.codec == "auto" and cfg.bit_depth == 10),
            bit_depth=8 if cfg.bit_depth is None else cfg.bit_depth,
            for_shot=True,
            settings=cfg,
        )
        filter_graph = (
            f"[0:v]sendcmd=f={escape_filter_path(cmd_file)},"
            f"v360=input=e:output=flat:yaw=0:pitch=0:roll=0:h_fov={fov}:v_fov={fov * height / width}:"
            f"w={width}:h={height}[vout]"
        )
        # All -i inputs first, then maps (FFmpeg rejects -map before a later -i)
        cmd = [
            ffmpeg_path(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "info",
            "-stats",
            "-i",
            str(master_eq),
        ]
        has_audio = capture.audio_stream_index is not None
        if has_audio:
            cmd += ["-i", str(capture.render_source())]
        cmd += ["-filter_complex", filter_graph, "-map", "[vout]"]
        if has_audio:
            cmd += ["-map", "1:a:0?", *audio_encode_args(cfg)]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v",
            enc.video,
            "-pix_fmt",
            enc.pix_fmt,
            *enc.extra,
            "-r",
            str(capture.fps or path.fps or 30.0),
            "-movflags",
            "+faststart",
            str(output),
        ]
        log.info("Schreibe Shot → %s", output)
        run(cmd)
        log.info("Shot fertig: %s (%s)", output, output.exists())
    return output


def _write_look_cmd(path: Path, cam: CameraPath, fov: float) -> None:
    lines: list[str] = []
    for i, (yaw, pitch) in enumerate(zip(cam.yaw, cam.pitch)):
        t = i / cam.fps if cam.fps > 0 else 0.0
        lines.append(f"{t:.6f} v360 yaw {yaw:.4f};")
        lines.append(f"{t:.6f} v360 pitch {pitch:.4f};")
        lines.append(f"{t:.6f} v360 roll 0;")
    path.write_text("\n".join(lines), encoding="utf-8")

