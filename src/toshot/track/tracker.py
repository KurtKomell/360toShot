from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from toshot.source.models import Capture

log = logging.getLogger(__name__)


@dataclass
class CameraPath:
    fps: float
    yaw: list[float]
    pitch: list[float]
    track_id: int | None = None

    def __len__(self) -> int:
        return len(self.yaw)


def smooth_path(path: CameraPath, alpha: float = 0.15) -> CameraPath:
    """Exponential smoothing on yaw/pitch with yaw wrap handling."""
    if not path.yaw:
        return path
    sy: list[float] = [path.yaw[0]]
    sp: list[float] = [path.pitch[0]]
    for i in range(1, len(path.yaw)):
        dy = path.yaw[i] - sy[-1]
        while dy > 180:
            dy -= 360
        while dy < -180:
            dy += 360
        sy.append(sy[-1] + alpha * dy)
        sp.append(sp[-1] + alpha * (path.pitch[i] - sp[-1]))
    return CameraPath(fps=path.fps, yaw=sy, pitch=sp, track_id=path.track_id)


def _bbox_to_ypr(cx: float, cy: float, w: int, h: int) -> tuple[float, float]:
    yaw = (cx / w) * 360.0 - 180.0
    pitch = 90.0 - (cy / h) * 180.0
    return yaw, pitch


def track_subject(
    equirect_path: Path,
    *,
    class_name: str = "person",
    track_id: int | None = None,
    model_name: str = "yolo11n.pt",
    pad_seam: bool = True,
) -> CameraPath:
    """
    Run YOLO + ByteTrack on an equirectangular video.
    Pads left/right so tracks survive the 0° meridian.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for tracking") from exc

    cap = cv2.VideoCapture(str(equirect_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {equirect_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 960)
    pad = width // 4 if pad_seam else 0

    model = YOLO(model_name)
    names = model.names
    class_ids = [i for i, n in names.items() if n == class_name]
    if not class_ids:
        log.warning("Class %s not in model; tracking all classes", class_name)
        class_ids = None  # type: ignore[assignment]

    # Collect per-track observations: id -> list of (frame, yaw, pitch, conf)
    tracks: dict[int, list[tuple[int, float, float, float]]] = {}
    frame_i = 0
    log.info("Tracking %s in %s (%dx%d @ %.2f)", class_name, equirect_path.name, width, height, fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if pad:
            frame_p = np.concatenate([frame[:, -pad:], frame, frame[:, :pad]], axis=1)
        else:
            frame_p = frame
            pad = 0

        results = model.track(
            frame_p,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=class_ids,
        )
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for tid, box, conf in zip(ids, xyxy, confs):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) * 0.5 - pad
                # Wrap into original width
                while cx < 0:
                    cx += width
                while cx >= width:
                    cx -= width
                cy = (y1 + y2) * 0.5
                yaw, pitch = _bbox_to_ypr(cx, cy, width, height)
                tracks.setdefault(int(tid), []).append((frame_i, yaw, pitch, float(conf)))
        frame_i += 1

    cap.release()
    if not tracks:
        log.warning("No tracks found; centering view")
        n = max(frame_i, 1)
        return CameraPath(fps=fps, yaw=[0.0] * n, pitch=[0.0] * n, track_id=None)

    if track_id is not None and track_id in tracks:
        chosen = track_id
    else:
        chosen = max(tracks.keys(), key=lambda k: len(tracks[k]))

    obs = {f: (y, p) for f, y, p, _ in tracks[chosen]}
    yaw_out: list[float] = []
    pitch_out: list[float] = []
    last = obs[min(obs)]
    for i in range(frame_i):
        if i in obs:
            last = obs[i]
        yaw_out.append(last[0])
        pitch_out.append(last[1])

    path = CameraPath(fps=fps, yaw=yaw_out, pitch=pitch_out, track_id=chosen)
    return smooth_path(path)


def make_proxy_equirect(
    capture: Capture,
    work: Path,
    *,
    stab,
    width: int = 1920,
    options=None,
) -> Path:
    """Low-res equirect from the **master** (.OSV/.insv) for YOLO — never from LRF/LRV."""
    from toshot.stitch.pipeline import stitch_equirect

    out = work / f"{capture.id}_track_equirect.mp4"
    src = capture.render_source()
    log.info("Tracking-Equirect aus Master %s @ width=%d", src.name, width)
    stitch_equirect(
        capture,
        out,
        stab=stab,
        width=width,
        source=src,
        options=options,
    )
    return out
