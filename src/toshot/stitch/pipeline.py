from __future__ import annotations

import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from toshot.ffmpeg_bin import (
    EncodeSettings,
    EncoderChoice,
    audio_encode_args,
    choose_video_encoder,
    escape_filter_path,
    ffmpeg_path,
    has_filter,
)
from toshot.math3d import frame_rotations
from toshot.parallel import stitch_workers
from toshot.source.models import AttitudeSample, Capture, LensLayout, LensParams, StabMode
from toshot.stitch.advanced import blend_advanced
from toshot.stitch.options import StitchOptions
from toshot.stitch.remap import blend_weights, build_lens_maps, feather_px_for, write_pgm_pair

log = logging.getLogger(__name__)


def write_sendcmd(path: Path, rotations: list[tuple[float, float, float]], fps: float) -> None:
    lines: list[str] = []
    for i, (yaw, pitch, roll) in enumerate(rotations):
        t = i / fps if fps > 0 else 0.0
        lines.append(f"{t:.6f} v360 yaw {yaw:.4f};")
        lines.append(f"{t:.6f} v360 pitch {pitch:.4f};")
        lines.append(f"{t:.6f} v360 roll {roll:.4f};")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_weight_png(path: Path, weight: np.ndarray) -> None:
    img = np.clip(weight * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _scale_lens(lens: LensParams, src_w: int, src_h: int) -> LensParams:
    sx = src_w / max(lens.width, 1)
    sy = src_h / max(lens.height, 1)
    return LensParams(
        fx=lens.fx * sx,
        fy=lens.fy * sy,
        cx=lens.cx * sx,
        cy=lens.cy * sy,
        k=lens.k,
        width=src_w,
        height=src_h,
        extrinsic_q=lens.extrinsic_q,
        yaw_deg=lens.yaw_deg,
        pitch_deg=lens.pitch_deg,
        roll_deg=lens.roll_deg,
    )


def _lens_pixel_size(capture: Capture, source: Path | None = None) -> tuple[int, int]:
    streams = capture.video_streams
    side = capture.side_by_side
    if source is not None and source.suffix.lower() in {".lrv"}:
        side = True
    if not streams:
        edge = capture.lens_edge
        return edge, edge
    if side:
        w, h = streams[0].width, streams[0].height
        # If source is proxy SBS, re-probe
        if source is not None:
            try:
                from toshot.ffmpeg_bin import probe

                info = probe(source)
                for s in info.get("streams", []):
                    if s.get("codec_type") == "video":
                        w, h = int(s["width"]), int(s["height"])
                        break
            except Exception:
                pass
        return max(1, w // 2), h
    return streams[0].width, streams[0].height


def prepare_maps(
    capture: Capture,
    out_w: int,
    out_h: int,
    work: Path,
    source: Path | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    rear = capture.calibration.rear
    front = capture.calibration.front
    if rear is None:
        raise RuntimeError("Capture has no rear lens calibration")
    if front is None:
        front = rear

    src_w, src_h = _lens_pixel_size(capture, source)
    rear_s = _scale_lens(rear, src_w, src_h)
    front_s = _scale_lens(front, src_w, src_h)

    x0, y0, w0 = build_lens_maps(rear_s, out_w, out_h)
    x1, y1, w1 = build_lens_maps(front_s, out_w, out_h, ref_lens=rear_s)
    a, _b = blend_weights(w0, w1, feather_px=feather_px_for(out_w))

    x0p, y0p = work / "rear_x.pgm", work / "rear_y.pgm"
    x1p, y1p = work / "front_x.pgm", work / "front_y.pgm"
    wp = work / "rear_weight.png"
    write_pgm_pair(x0, y0, x0p, y0p)
    write_pgm_pair(x1, y1, x1p, y1p)
    _write_weight_png(wp, a)
    return x0p, y0p, x1p, y1p, wp


def _is_sbs_source(capture: Capture, source: Path) -> bool:
    if capture.side_by_side:
        return True
    if source.suffix.lower() == ".lrv":
        return True
    if capture.proxy_path and source.resolve() == capture.proxy_path.resolve():
        try:
            from toshot.ffmpeg_bin import probe

            info = probe(source)
            for s in info.get("streams", []):
                if s.get("codec_type") == "video":
                    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
                    return w >= h * 1.8
        except Exception:
            return True
    return False


def stitch_equirect(
    capture: Capture,
    output: Path,
    *,
    stab: StabMode = StabMode.WORLD,
    width: int | None = None,
    source: Path | None = None,
    encoder: EncoderChoice | None = None,
    sendcmd_path: Path | None = None,
    work_dir: Path | None = None,
    options: StitchOptions | None = None,
    encode: EncodeSettings | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opts = options or StitchOptions()
    src = Path(source) if source else capture.render_source()
    # Hard rule: never decode LRF/LRV — always fall back to OSV/INSV master
    if src.suffix.lower() in {".lrf", ".lrv"} or src.name.upper().startswith("LRV_"):
        master = capture.render_source()
        log.warning("Proxy-Dekode abgelehnt (%s) → Master %s", src.name, master.name)
        src = master

    out_w, out_h = capture.equirect_size(width)
    log.info("Master-Stitch %s → %dx%d", src.name, out_w, out_h)

    fps = capture.fps or 30.0
    n_frames = capture.estimated_frames()
    log.info("Capture timing: %.3f fps, %d frames, duration=%.3fs", fps, n_frames, capture.duration_s or 0.0)

    # Internal tracking stitches (no user settings, width ≤ 1920): 8-bit H.264.
    # An explicit encode request is honored at any width.
    cfg = encode or EncodeSettings()
    tracking_proxy = encode is None and width is not None and width <= 1920
    want_10 = cfg.bit_depth == 10 or (cfg.bit_depth is None and capture.bit_depth >= 10)
    prefer_hevc = cfg.codec == "hevc" or (cfg.codec == "auto" and want_10 and not tracking_proxy)
    if cfg.codec == "h264":
        prefer_hevc = False
    depth = 8 if tracking_proxy or cfg.bit_depth == 8 or not want_10 else capture.bit_depth
    if cfg.bit_depth in (8, 10):
        depth = cfg.bit_depth
    enc = encoder or choose_video_encoder(
        prefer_hevc=prefer_hevc,
        bit_depth=depth,
        for_shot=tracking_proxy,
        settings=None if tracking_proxy else cfg,
    )
    if enc.note:
        log.info("Using encoder %s (%s)", enc.video, enc.note)

    tmp_own = work_dir is None
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="toshot_"))
    work.mkdir(parents=True, exist_ok=True)

    try:
        # Advanced seam / RS / stick masking need per-frame OpenCV processing
        if opts.advanced:
            log.info(
                "Advanced stitch: optical_flow=%s selfie_mask=%s rolling_shutter=%s",
                opts.optical_flow,
                opts.selfie_mask,
                opts.rolling_shutter,
            )
            return _opencv_stitch(
                capture,
                src,
                output,
                out_w,
                out_h,
                stab,
                enc,
                fps,
                n_frames,
                options=opts,
            )

        # Same stitch as the Vulkan preview: float maps, bilinear sample, scaled feather.
        # FFmpeg's remap filter is nearest-neighbor gray16 and cannot match that blend.
        log.info(
            "Vorschau-Stitch %dx%d, Feder %.1f px",
            out_w,
            out_h,
            feather_px_for(out_w),
        )
        cmd_file = Path(sendcmd_path) if sendcmd_path else work / "stab.cmd"
        return _opencv_stitch(
            capture,
            src,
            output,
            out_w,
            out_h,
            stab,
            enc,
            fps,
            n_frames,
            options=StitchOptions.fast(),
            spherical_stab=True,
            encode=cfg,
            sendcmd_path=cmd_file,
        )
    finally:
        if tmp_own:
            import shutil

            shutil.rmtree(work, ignore_errors=True)


def _stitch_one_frame(
    i: int,
    fr: np.ndarray,
    ff: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    options: StitchOptions,
    attitude: list[AttitudeSample],
    fps: float,
    rotations: list[tuple[float, float, float]],
    apply_rotation: bool = True,
) -> np.ndarray:
    mapped0 = cv2.remap(fr, x0, y0, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mapped1 = cv2.remap(ff, x1, y1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    t_s = i / fps if fps > 0 else 0.0
    if options.advanced:
        blended = blend_advanced(
            mapped0,
            mapped1,
            a,
            b,
            options=options,
            attitude=attitude,
            t_s=t_s,
            fps=fps,
        )
    else:
        blended = (
            mapped0.astype(np.float32) * a[..., None] + mapped1.astype(np.float32) * b[..., None]
        ).astype(np.uint8)
    yaw, pitch, roll = rotations[min(i, len(rotations) - 1)]
    if not apply_rotation:
        return blended
    return _rotate_equirect_cv(blended, yaw, pitch, roll)


def _opencv_stitch(
    capture: Capture,
    source: Path,
    output: Path,
    out_w: int,
    out_h: int,
    stab: StabMode,
    enc: EncoderChoice,
    fps: float,
    n_frames: int,
    *,
    options: StitchOptions,
    spherical_stab: bool = False,
    encode: EncodeSettings | None = None,
    sendcmd_path: Path | None = None,
) -> Path:
    rear = capture.calibration.rear
    front = capture.calibration.front or rear
    if rear is None:
        raise RuntimeError("No lens calibration")

    src_w, src_h = _lens_pixel_size(capture, source)
    workers = stitch_workers()
    # Avoid OpenCV nested thread explosion when we already farm frames out
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    log.info(
        "OpenCV stitch maps %dx%d from lens %dx%d · %d threads · %d frames",
        out_w,
        out_h,
        src_w,
        src_h,
        workers,
        n_frames,
    )
    rear_s = _scale_lens(rear, src_w, src_h)
    front_s = _scale_lens(front, src_w, src_h)
    x0, y0, w0 = build_lens_maps(rear_s, out_w, out_h)
    x1, y1, w1 = build_lens_maps(front_s, out_w, out_h, ref_lens=rear_s)
    a, b = blend_weights(w0, w1, feather_px=feather_px_for(out_w))
    rotations = frame_rotations(capture.attitude, fps, n_frames, stab)
    use_v360 = spherical_stab and stab != StabMode.NONE and has_filter("v360")
    if use_v360:
        cmd_file = Path(sendcmd_path) if sendcmd_path else Path(tempfile.mkdtemp(prefix="toshot_stab_")) / "stab.cmd"
        cmd_file.parent.mkdir(parents=True, exist_ok=True)
        write_sendcmd(cmd_file, rotations, fps)
        log.info("Stabilisierung über v360 (%s)", stab.value)
    log.info("Decoding lens streams and encoding %d frames …", n_frames)

    frames_r = _iter_lens_frames(capture, source, lens=0)
    frames_f = _iter_lens_frames(capture, source, lens=1)

    has_audio = capture.audio_stream_index is not None
    cmd = [
        ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if has_audio:
        cmd += ["-i", str(source)]
    if use_v360:
        fc = (
            f"[0:v]sendcmd=f={escape_filter_path(cmd_file)},"
            f"v360=input=e:output=e:yaw=0:pitch=0:roll=0:w={out_w}:h={out_h}[vout]"
        )
        cmd += ["-filter_complex", fc, "-map", "[vout]"]
    else:
        cmd += ["-map", "0:v"]
    if has_audio:
        cmd += ["-map", "1:a:0?", *audio_encode_args(encode)]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v",
        enc.video,
        "-pix_fmt",
        enc.pix_fmt,
        *enc.extra,
        "-movflags",
        "+faststart",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    # Keep a sliding window of in-flight frames so RAM stays bounded
    window = max(workers * 2, workers)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: dict[int, object] = {}
            next_submit = 0
            next_write = 0
            it = enumerate(zip(frames_r, frames_f))

            def _submit_until_full() -> bool:
                nonlocal next_submit
                while next_submit < n_frames and len(pending) < window:
                    try:
                        i, pair = next(it)
                    except StopIteration:
                        return False
                    if i >= n_frames:
                        return False
                    fr, ff = pair
                    # Own copies — decode buffers must not be shared across threads
                    pending[i] = pool.submit(
                        _stitch_one_frame,
                        i,
                        np.copy(fr),
                        np.copy(ff),
                        x0,
                        y0,
                        x1,
                        y1,
                        a,
                        b,
                        options,
                        capture.attitude,
                        fps,
                        rotations,
                        not use_v360,
                    )
                    next_submit = i + 1
                return True

            more = _submit_until_full()
            while next_write < n_frames and (pending or more):
                if next_write not in pending:
                    more = _submit_until_full() or more
                    if next_write not in pending:
                        break
                fut = pending.pop(next_write)
                blended = fut.result()  # type: ignore[union-attr]
                if next_write == 0 or (next_write + 1) % 30 == 0 or (next_write + 1) == n_frames:
                    log.info("Stitch frame %d / %d", next_write + 1, n_frames)
                proc.stdin.write(blended.tobytes())
                next_write += 1
                more = _submit_until_full() or more
    finally:
        proc.stdin.close()
        proc.wait()
        if proc.returncode:
            raise RuntimeError(f"OpenCV stitch encode failed ({proc.returncode})")
    return output


def _rotate_equirect_cv(img: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    if abs(yaw) < 1e-3 and abs(pitch) < 1e-3 and abs(roll) < 1e-3:
        return img
    h, w = img.shape[:2]
    shift_x = int(round((-yaw / 360.0) * w))
    shift_y = int(round((pitch / 180.0) * h))
    out = np.roll(img, shift_x, axis=1)
    if shift_y:
        out = np.roll(out, shift_y, axis=0)
    return out


def _iter_lens_frames(capture: Capture, source: Path, lens: int):
    src_w, src_h = _lens_pixel_size(capture, source)
    if (
        capture.layout == LensLayout.SPLIT_FILES
        and len(capture.master_paths) >= 2
        and source.resolve() == capture.primary_path.resolve()
    ):
        path = capture.master_paths[lens]
        cmd = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={src_w}:{src_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    elif _is_sbs_source(capture, source):
        x = "0" if lens == 0 else "iw/2"
        cmd = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"crop=iw/2:ih:{x}:0,scale={src_w}:{src_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    else:
        cmd = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            f"0:v:{lens}",
            "-vf",
            f"scale={src_w}:{src_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    nbytes = src_w * src_h * 3
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((src_h, src_w, 3))
    finally:
        proc.kill()
