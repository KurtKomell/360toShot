"""Side-by-side raw dual-lens preview — no stitch, no remap."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from toshot.preview.decode import DualLensDecoder
from toshot.source.group import load_capture
from toshot.source.models import Capture

log = logging.getLogger(__name__)



def run_raw_lenses(
    capture: Capture,
    *,
    lens_edge: int = 768,
    window_name: str | None = None,
) -> None:
    """
    Show rear | front fisheye side-by-side (no stitching).

    Keys: Space=pause · Esc/Q=quit
    """
    edge = max(256, int(lens_edge) - (int(lens_edge) % 2))
    title = window_name or f"toshot raw lenses · {capture.id} · no stitch"
    log.info("Roh-Linsen-Vorschau (ohne Stitch): %s  %dx%d", capture.id, edge, edge)

    decoder = DualLensDecoder(capture, edge, edge)
    decoder.start()
    frame = None
    for _ in range(100):
        frame = decoder.latest()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        decoder.stop()
        raise RuntimeError(f"Kein Preview-Frame von {capture.id}")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # Rear left, front right — as in LRF SBS
    first = np.hstack([frame.rear, frame.front])
    cv2.resizeWindow(title, min(1600, first.shape[1]), min(800, first.shape[0]))

    last_idx = -1
    try:
        while True:
            fr = decoder.latest()
            if fr is not None and fr.index != last_idx:
                last_idx = fr.index
                canvas = np.hstack([fr.rear, fr.front])
                # Labels
                labeled = canvas.copy()
                cv2.putText(
                    labeled,
                    "REAR",
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    labeled,
                    "FRONT",
                    (edge + 16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    labeled,
                    "no stitch  |  Space=pause  Esc=quit",
                    (16, edge - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(title, labeled)
            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                decoder.toggle_pause()
    finally:
        decoder.stop()
        cv2.destroyWindow(title)
        log.info("Roh-Linsen-Vorschau beendet")


def run_raw_lenses_path(path: str | Path, **kwargs) -> None:
    cap = load_capture(Path(path))
    run_raw_lenses(cap, **kwargs)
