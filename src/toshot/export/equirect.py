from __future__ import annotations

import logging
from pathlib import Path

from toshot.export.spherical import inject_spherical_v1
from toshot.ffmpeg_bin import EncodeSettings
from toshot.source.models import Capture, StabMode
from toshot.stitch.options import StitchOptions
from toshot.stitch.pipeline import stitch_equirect

log = logging.getLogger(__name__)


def export_equirect(
    capture: Capture,
    output: Path,
    *,
    stab: StabMode = StabMode.WORLD,
    width: int | None = None,
    source: Path | None = None,
    options: StitchOptions | None = None,
    encode: EncodeSettings | None = None,
) -> Path:
    """
    Export stabilized equirectangular MP4 at maximum lens resolution (or --width).
    Injects Google Spherical V1 metadata so players treat it as 360°.
    """
    output = Path(output)
    opts = options or StitchOptions()
    cfg = encode or EncodeSettings()
    log.info(
        "Equirect export %s -> %s (%s) advanced=%s %s",
        capture.id,
        output,
        "max" if width is None else f"width={width}",
        opts.advanced,
        cfg.summary(),
    )
    out_w, out_h = capture.equirect_size(width)
    log.info("Target resolution %dx%d, stab=%s", out_w, out_h, stab.value)

    # Always stitch from OSV/INSV master unless an explicit source is passed
    render = source or capture.render_source()
    if render.suffix.lower() in {".lrf", ".lrv"}:
        log.warning("Equirect-Quelle ist ein Proxy (%s) — erwarte Master .OSV/.insv", render.name)
    log.info("Equirect render source: %s", render)
    stitch_equirect(
        capture,
        output,
        stab=stab,
        width=width,
        source=render,
        options=opts,
        encode=cfg,
    )
    try:
        inject_spherical_v1(output)
        log.info("Injected spherical metadata into %s", output.name)
    except Exception as exc:
        log.warning("Could not inject spherical metadata: %s", exc)
    return output
