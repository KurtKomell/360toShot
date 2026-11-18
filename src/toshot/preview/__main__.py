from __future__ import annotations

import argparse
import logging
import sys

from toshot.preview.raw_lenses import run_raw_lenses_path
from toshot.preview.viewer import run_preview_path
from toshot.source.models import StabMode
from toshot.stitch.options import StitchOptions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="toshot-preview", description="Vulkan live stitch preview")
    p.add_argument("input", help=".OSV/.insv (uses LRF/LRV proxy when present)")
    p.add_argument("--stab", default="world", choices=[m.value for m in StabMode])
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--raw-lenses",
        action="store_true",
        help="Show rear|front fisheye side-by-side without stitching",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip optical flow / selfie mask / rolling shutter",
    )
    p.add_argument("--no-optical-flow", action="store_true")
    p.add_argument("--no-selfie-mask", action="store_true")
    p.add_argument("--no-rolling-shutter", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.raw_lenses:
        run_raw_lenses_path(args.input)
        return 0
    opts = StitchOptions.from_flags(
        optical_flow=not args.no_optical_flow,
        selfie_mask=not args.no_selfie_mask,
        rolling_shutter=not args.no_rolling_shutter,
        fast=bool(args.fast),
    )
    run_preview_path(
        args.input,
        stab=StabMode(args.stab),
        width=args.width,
        height=args.height,
        options=opts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
