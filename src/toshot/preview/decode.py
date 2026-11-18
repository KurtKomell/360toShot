from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toshot.ffmpeg_bin import ffmpeg_path, probe
from toshot.source.models import Capture

log = logging.getLogger(__name__)

# #region agent log
_DBG = Path(r"d:\360toShot\.cursor\debug-06c6a9.log")


def _dbg(hypothesis_id: str, location: str, message: str, data: dict | None = None, run_id: str = "fps") -> None:
    try:
        payload = {
            "sessionId": "06c6a9",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with _DBG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# #endregion


@dataclass
class PreviewFrame:
    rear: np.ndarray  # HxWx3 uint8 BGR
    front: np.ndarray
    t_s: float
    index: int


def choose_preview_media(capture: Capture) -> tuple[Path, str]:
    """Prefer LRF/LRV for interactive preview; master only as fallback."""
    if capture.proxy_path is not None and capture.proxy_path.exists():
        return capture.proxy_path, "proxy"
    return capture.render_source(), "master"


def _video_size(path: Path) -> tuple[int, int, bool]:
    info = probe(path)
    best_w = best_h = 0
    sbs = False
    for s in info.get("streams") or []:
        if s.get("codec_type") != "video":
            continue
        if (s.get("disposition") or {}).get("attached_pic"):
            continue
        if str(s.get("codec_name") or "").lower() == "mjpeg":
            continue
        w, h = int(s.get("width") or 0), int(s.get("height") or 0)
        if w * h > best_w * best_h:
            best_w, best_h = w, h
            sbs = w >= h * 1.8
    return best_w, best_h, sbs


def _probe_fps(path: Path) -> float:
    """Largest non-thumb video stream fps from the preview media itself."""
    info = probe(path)
    best_area = 0
    best_fps = 0.0
    for s in info.get("streams") or []:
        if s.get("codec_type") != "video":
            continue
        if (s.get("disposition") or {}).get("attached_pic"):
            continue
        if str(s.get("codec_name") or "").lower() == "mjpeg":
            continue
        w, h = int(s.get("width") or 0), int(s.get("height") or 0)
        area = w * h
        rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
        try:
            if isinstance(rate, str) and "/" in rate:
                a, b = rate.split("/", 1)
                fps = float(a) / float(b) if float(b) else 0.0
            else:
                fps = float(rate)
        except (TypeError, ValueError, ZeroDivisionError):
            fps = 0.0
        if area > best_area and fps > 0.5:
            best_area = area
            best_fps = fps
    return best_fps


class DualLensDecoder:
    """Background dual-lens decoder for preview (proxy preferred)."""

    def __init__(self, capture: Capture, lens_w: int, lens_h: int) -> None:
        self.capture = capture
        self.lens_w = lens_w
        self.lens_h = lens_h
        self.path, self.kind = choose_preview_media(capture)
        probed = _probe_fps(self.path)
        cap_fps = float(capture.fps or 0.0)
        # Prefer probed media fps when sane; capture.fps can lag thumbs/metadata.
        if probed >= 1.0:
            self.fps = probed
        elif cap_fps >= 1.0:
            self.fps = cap_fps
        else:
            self.fps = 30.0
        self._lock = threading.Lock()
        self._frame: PreviewFrame | None = None
        self._index = 0
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._src_w, self._src_h, self._sbs = _video_size(self.path)
        log.info(
            "Preview decode %s (%s) %dx%d sbs=%s → lens %dx%d @ %.3ffps (cap=%.3f probe=%.3f)",
            self.path.name,
            self.kind,
            self._src_w,
            self._src_h,
            self._sbs,
            lens_w,
            lens_h,
            self.fps,
            cap_fps,
            probed,
        )
        # #region agent log
        _dbg(
            "P",
            "decode.py:init",
            "preview fps select",
            {
                "path": self.path.name,
                "kind": self.kind,
                "cap_fps": cap_fps,
                "probe_fps": probed,
                "use_fps": self.fps,
            },
        )
        # #endregion

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="toshot-preview-decode", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def toggle_pause(self) -> None:
        if self._pause.is_set():
            self._pause.clear()
        else:
            self._pause.set()

    def latest(self) -> PreviewFrame | None:
        with self._lock:
            return self._frame

    def _cmd_pair(self) -> tuple[list[str], list[str]]:
        lw, lh = self.lens_w, self.lens_h
        common = [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.path),
        ]
        if self._sbs or self.path.suffix.lower() in {".lrf", ".lrv"}:
            rear = common + [
                "-vf",
                f"crop=iw/2:ih:0:0,scale={lw}:{lh}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ]
            front = common + [
                "-vf",
                f"crop=iw/2:ih:iw/2:0,scale={lw}:{lh}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ]
            return rear, front
        # Dual-track OSV/INSV
        rear = common + [
            "-map",
            "0:v:0",
            "-vf",
            f"scale={lw}:{lh}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        front = common + [
            "-map",
            "0:v:1",
            "-vf",
            f"scale={lw}:{lh}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        return rear, front

    def _run(self) -> None:
        rear_cmd, front_cmd = self._cmd_pair()
        nbytes = self.lens_w * self.lens_h * 3
        pr = subprocess.Popen(rear_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        pf = subprocess.Popen(front_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert pr.stdout and pf.stdout
        frame_dt = 1.0 / self.fps if self.fps > 0 else (1.0 / 30.0)
        next_t = time.perf_counter()
        pace_log_t = next_t
        pace_frames = 0
        try:
            while not self._stop.is_set():
                while self._pause.is_set() and not self._stop.is_set():
                    self._stop.wait(0.05)
                    next_t = time.perf_counter()
                br = pr.stdout.read(nbytes)
                bf = pf.stdout.read(nbytes)
                if not br or not bf or len(br) < nbytes or len(bf) < nbytes:
                    # loop
                    pr.kill()
                    pf.kill()
                    pr = subprocess.Popen(rear_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    pf = subprocess.Popen(front_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    assert pr.stdout and pf.stdout
                    self._index = 0
                    next_t = time.perf_counter()
                    continue
                rear = np.frombuffer(br, dtype=np.uint8).reshape((self.lens_h, self.lens_w, 3)).copy()
                front = np.frombuffer(bf, dtype=np.uint8).reshape((self.lens_h, self.lens_w, 3)).copy()
                t = self._index / self.fps if self.fps > 0 else 0.0
                with self._lock:
                    self._frame = PreviewFrame(rear=rear, front=front, t_s=t, index=self._index)
                self._index += 1
                pace_frames += 1
                # Pace to source fps — without this, decode runs as fast as GPU/CPU allow.
                next_t += frame_dt
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:
                    # Fell far behind (startup decode burst): resync clock
                    next_t = time.perf_counter()
                now = time.perf_counter()
                if now - pace_log_t >= 1.0:
                    # #region agent log
                    _dbg(
                        "P",
                        "decode.py:pace",
                        "decode pacing",
                        {
                            "target_fps": self.fps,
                            "measured_fps": pace_frames / max(now - pace_log_t, 1e-6),
                            "index": self._index,
                            "behind_s": max(0.0, -delay),
                        },
                        run_id="post-fix",
                    )
                    # #endregion
                    pace_log_t = now
                    pace_frames = 0
        finally:
            pr.kill()
            pf.kill()
