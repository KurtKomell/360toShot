from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CameraFamily(str, Enum):
    DJI = "dji"
    INSTA = "insta"


class LensLayout(str, Enum):
    DUAL_TRACK = "dual_track"
    SIDE_BY_SIDE = "side_by_side"
    SPLIT_FILES = "split_files"
    SINGLE = "single"


class StabMode(str, Enum):
    HORIZON = "horizon"
    LEVEL = "level"
    WORLD = "world"
    NONE = "none"


@dataclass(frozen=True)
class VideoStreamInfo:
    index: int
    width: int
    height: int
    fps: float
    duration_s: float
    codec: str
    pix_fmt: str
    bit_depth: int = 8


@dataclass(frozen=True)
class LensParams:
    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple[float, ...]
    width: int
    height: int
    extrinsic_q: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass
class Calibration:
    rear: LensParams | None = None
    front: LensParams | None = None
    model: str = ""
    firmware: str = ""
    serial: str = ""

    @property
    def lenses(self) -> list[LensParams]:
        out: list[LensParams] = []
        if self.rear is not None:
            out.append(self.rear)
        if self.front is not None:
            out.append(self.front)
        return out


@dataclass(frozen=True)
class AttitudeSample:
    t_s: float
    w: float
    x: float
    y: float
    z: float


@dataclass
class Capture:
    family: CameraFamily
    layout: LensLayout
    master_paths: list[Path]
    proxy_path: Path | None = None
    video_streams: list[VideoStreamInfo] = field(default_factory=list)
    audio_stream_index: int | None = None
    calibration: Calibration = field(default_factory=Calibration)
    attitude: list[AttitudeSample] = field(default_factory=list)
    fps: float = 30.0
    duration_s: float = 0.0
    frame_count: int = 0
    bit_depth: int = 8
    side_by_side: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.master_paths[0].stem

    @property
    def primary_path(self) -> Path:
        return self.master_paths[0]

    @property
    def lens_edge(self) -> int:
        streams = self.video_streams
        if not streams:
            return 3840
        if self.side_by_side and streams:
            return max(streams[0].height, streams[0].width // 2)
        return max(max(s.width, s.height) for s in streams)

    def equirect_size(self, width: int | None = None) -> tuple[int, int]:
        if width is not None:
            w = width - (width % 2)
            h = (w // 2) - ((w // 2) % 2)
            return w, h
        edge = self.lens_edge
        edge = edge - (edge % 2)
        return edge * 2, edge

    def estimated_frames(self) -> int:
        if self.frame_count > 0:
            return self.frame_count
        if self.duration_s > 0 and self.fps > 0:
            return max(1, int(round(self.duration_s * self.fps)))
        return 1

    def tracking_source(self) -> Path:
        """Always master (.OSV / .insv). Proxies are never decoded."""
        return self.primary_path

    def render_source(self) -> Path:
        """Always the master (.OSV / .insv) — never a proxy."""
        return self.primary_path

