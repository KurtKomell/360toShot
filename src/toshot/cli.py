from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from toshot import __version__
from toshot.export.batch import run_batch
from toshot.export.equirect import export_equirect
from toshot.export.shot import export_shot
from toshot.ffmpeg_bin import CODECS, PRESETS, EncodeSettings, normalize_audio_bitrate, normalize_bitrate
from toshot.source.group import load_capture, resolve_path
from toshot.source.models import StabMode
from toshot.stitch.options import StitchOptions


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _stab(value: str) -> StabMode:
    try:
        return StabMode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid stab mode {value!r}; choose horizon|level|world|none"
        ) from exc


def _stitch_options(args: argparse.Namespace) -> StitchOptions:
    return StitchOptions.from_flags(
        optical_flow=not args.no_optical_flow,
        selfie_mask=not args.no_selfie_mask,
        rolling_shutter=not args.no_rolling_shutter,
        fast=bool(getattr(args, "fast", False)),
    )


def _bitrate_arg(value: str) -> str:
    try:
        out = normalize_bitrate(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if out is None:
        raise argparse.ArgumentTypeError("Bitrate leer")
    return out


def _audio_bitrate_arg(value: str) -> str:
    try:
        return normalize_audio_bitrate(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _crf_arg(value: str) -> int:
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CRF muss eine ganze Zahl sein") from exc
    if not 0 <= n <= 51:
        raise argparse.ArgumentTypeError("CRF muss zwischen 0 und 51 liegen")
    return n


def _encode_settings(args: argparse.Namespace) -> EncodeSettings:
    depth = getattr(args, "bit_depth", "auto")
    return EncodeSettings(
        codec=args.codec,
        crf=args.crf,
        bitrate=getattr(args, "bitrate", None),
        preset=args.preset,
        audio_bitrate=args.audio_bitrate,
        bit_depth=None if depth == "auto" else int(depth),
    )


def _add_encode_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--codec",
        default="auto",
        choices=list(CODECS),
        help="auto (HEVC bei 10-bit Quelle), h264, hevc, prores, prores4444",
    )
    p.add_argument(
        "--crf",
        type=_crf_arg,
        default=18,
        help="Qualität 0–51, kleiner ist besser (Standard 18). Gilt ohne --bitrate",
    )
    p.add_argument(
        "--bitrate",
        type=_bitrate_arg,
        default=None,
        help="Ziel-Bitrate statt CRF in kbit/s, z.B. 80000 oder 80M",
    )
    p.add_argument(
        "--preset",
        default="medium",
        choices=list(PRESETS),
        help="Encoder-Geschwindigkeit (veryfast … slow)",
    )
    p.add_argument(
        "--bit-depth",
        default="auto",
        choices=["auto", "8", "10"],
        help="Farbtiefe. auto: 10-bit wenn die Quelle 10-bit ist (Equirect)",
    )
    p.add_argument(
        "--audio-bitrate",
        type=_audio_bitrate_arg,
        default="320k",
        help="Audio-Bitrate, z.B. 320 oder 256k",
    )


def _add_stitch_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip optical flow, selfie-stick mask and rolling shutter (FFmpeg remap path)",
    )
    p.add_argument(
        "--no-optical-flow",
        action="store_true",
        help="Disable DIS optical-flow alignment and parallax seam carve",
    )
    p.add_argument(
        "--no-selfie-mask",
        action="store_true",
        help="Disable selfie-stick / nadir lens preference",
    )
    p.add_argument(
        "--no-rolling-shutter",
        action="store_true",
        help="Disable gyro-based rolling-shutter row warp",
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    cap = load_capture(resolve_path(args.input))
    print(f"id:            {cap.id}")
    print(f"family:        {cap.family.value}")
    print(f"layout:        {cap.layout.value}")
    print(f"master:        {', '.join(str(p) for p in cap.master_paths)}")
    print(f"proxy:         {cap.proxy_path or '-'}")
    print(f"fps:           {cap.fps:.3f}")
    print(f"duration_s:    {cap.duration_s:.3f}")
    print(f"bit_depth:     {cap.bit_depth}")
    print(f"side_by_side:  {cap.side_by_side}")
    print(f"audio_stream:  {cap.audio_stream_index}")
    print(f"model:         {cap.calibration.model}")
    print(f"firmware:      {cap.calibration.firmware}")
    print(f"serial:        {cap.calibration.serial}")
    eq_w, eq_h = cap.equirect_size()
    print(f"equirect_max:  {eq_w}x{eq_h}")
    print(f"attitude:      {len(cap.attitude)} samples")
    for label, lens in (("rear", cap.calibration.rear), ("front", cap.calibration.front)):
        if lens is None:
            print(f"{label}:         -")
            continue
        print(
            f"{label}:         fx={lens.fx:.1f} fy={lens.fy:.1f} "
            f"cx={lens.cx:.1f} cy={lens.cy:.1f} size={lens.width}x{lens.height} "
            f"yaw={lens.yaw_deg:.1f}"
        )
    for s in cap.video_streams:
        print(
            f"video[{s.index}]:   {s.width}x{s.height} {s.codec} {s.pix_fmt} "
            f"{s.fps:.3f}fps {s.bit_depth}-bit"
        )
    for n in cap.notes:
        print(f"note:          {n}")
    return 0


def cmd_equirect(args: argparse.Namespace) -> int:
    cap = load_capture(resolve_path(args.input))
    out = resolve_path(args.output) if args.output else Path.cwd() / f"{cap.id}_equirect.mp4"
    export_equirect(
        cap,
        out,
        stab=_stab(args.stab),
        width=args.width,
        options=_stitch_options(args),
        encode=_encode_settings(args),
    )
    print(out)
    return 0


def cmd_shot(args: argparse.Namespace) -> int:
    cap = load_capture(resolve_path(args.input))
    out = resolve_path(args.output) if args.output else Path.cwd() / f"{cap.id}_shot.mp4"
    export_shot(
        cap,
        out,
        stab=_stab(args.stab),
        fov=args.fov,
        width=args.width,
        height=args.height,
        class_name=args.class_name,
        track_id=args.track_id,
        options=_stitch_options(args),
        encode=_encode_settings(args),
    )
    print(out)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    mode = args.mode
    if args.equirect and mode == "shot":
        mode = "both"
    written = run_batch(
        resolve_path(args.inbox),
        resolve_path(args.out),
        mode=mode,
        stab=_stab(args.stab),
        fov=args.fov,
        class_name=args.class_name,
        equirect_width=args.width,
        options=_stitch_options(args),
        encode=_encode_settings(args),
        workers=args.workers,
    )
    for p in written:
        print(p)
    return 0


def cmd_gui(_args: argparse.Namespace) -> int:
    from toshot.gui.app import run_gui

    run_gui()
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from toshot.preview.viewer import run_preview_path

    run_preview_path(
        resolve_path(args.input),
        stab=_stab(args.stab),
        width=args.width,
        height=args.height,
        options=_stitch_options(args),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="toshot",
        description="DJI OSV/LRF and Insta360 INSV/LRV → equirect + AI-tracked shot (Linux/Windows)",
    )
    p.add_argument("--version", action="version", version=f"toshot {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    gui = sub.add_parser("gui", help="Open the desktop GUI")
    gui.set_defaults(func=cmd_gui)

    prev = sub.add_parser("preview", help="Vulkan live stitch preview (OSV-Studio-like)")
    prev.add_argument("input", help="Path to .osv/.insv (uses proxy when present)")
    prev.add_argument("--stab", default="world", choices=[m.value for m in StabMode])
    prev.add_argument("--width", type=int, default=1280)
    prev.add_argument("--height", type=int, default=720)
    _add_stitch_flags(prev)
    prev.set_defaults(func=cmd_preview)

    insp = sub.add_parser("inspect", help="Show capture metadata, lenses, IMU")
    insp.add_argument("input", help="Path to .osv/.lrf/.insv/.lrv")
    insp.set_defaults(func=cmd_inspect)

    eq = sub.add_parser("equirect", help="Export max-resolution equirectangular MP4")
    eq.add_argument("input")
    eq.add_argument("-o", "--output", default=None)
    eq.add_argument("--stab", default="world", choices=[m.value for m in StabMode])
    eq.add_argument("--width", type=int, default=None, help="Optional equirect width (2:1)")
    _add_encode_flags(eq)
    _add_stitch_flags(eq)
    eq.set_defaults(func=cmd_equirect)

    shot = sub.add_parser("shot", help="Track subject and export flat shot MP4")
    shot.add_argument("input")
    shot.add_argument("-o", "--output", default=None)
    shot.add_argument("--stab", default="world", choices=[m.value for m in StabMode])
    shot.add_argument("--fov", type=float, default=90.0)
    shot.add_argument("--width", type=int, default=1920)
    shot.add_argument("--height", type=int, default=1080)
    shot.add_argument("--class-name", default="person")
    shot.add_argument("--track-id", type=int, default=None)
    _add_encode_flags(shot)
    _add_stitch_flags(shot)
    shot.set_defaults(func=cmd_shot)

    batch = sub.add_parser("batch", help="Process a folder of captures")
    batch.add_argument("inbox")
    batch.add_argument("--out", required=True)
    batch.add_argument(
        "--mode",
        default="shot",
        choices=["shot", "equirect", "both"],
        help="What to write per capture",
    )
    batch.add_argument(
        "--equirect",
        action="store_true",
        help="Also write equirect when mode=shot (same as --mode both)",
    )
    batch.add_argument("--stab", default="world", choices=[m.value for m in StabMode])
    batch.add_argument("--fov", type=float, default=90.0)
    batch.add_argument("--class-name", default="person")
    batch.add_argument("--width", type=int, default=None, help="Optional equirect width")
    _add_encode_flags(batch)
    batch.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel capture jobs (default: all CPU threads)",
    )
    _add_stitch_flags(batch)
    batch.set_defaults(func=cmd_batch)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        code = args.func(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        logging.getLogger("toshot").error("%s", exc)
        if args.verbose:
            raise
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
