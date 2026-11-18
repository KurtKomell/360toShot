from __future__ import annotations

import logging
import math
import re
import struct
from pathlib import Path

from toshot.protobuf_lite import ProtoNode, parse_protobuf
from toshot.source.models import (
    AttitudeSample,
    Calibration,
    CameraFamily,
    Capture,
    LensLayout,
    LensParams,
    VideoStreamInfo,
)
from toshot.source.probe_streams import streams_from_probe

log = logging.getLogger(__name__)

# Insta360 trailer: records end with [type:uint16][size:uint32] before a directory / magic
INSTA_MAGIC = b"Insta360"


def _read_trailer_records(path: Path) -> list[tuple[int, bytes]]:
    """Parse Insta360 trailer records from EOF backwards."""
    data = path.read_bytes()
    if len(data) < 64:
        return []
    # Find magic near end
    magic_pos = data.rfind(INSTA_MAGIC)
    if magic_pos < 0:
        # Some files only have record footers without the string at exact end
        magic_pos = len(data)
    records: list[tuple[int, bytes]] = []
    # Walk backwards: each record ends with type(u16 LE) + size(u32 LE)
    pos = magic_pos
    # Before magic there may be a version uint32; try walking from magic_pos
    cursor = magic_pos
    # Limit search window to last 64 MiB
    window_start = max(0, len(data) - 64 * 1024 * 1024)
    while cursor - 6 >= window_start:
        typ = struct.unpack_from("<H", data, cursor - 6)[0]
        size = struct.unpack_from("<I", data, cursor - 4)[0]
        if size <= 0 or size > cursor - 6 - window_start or size > 32 * 1024 * 1024:
            break
        start = cursor - 6 - size
        payload = data[start : cursor - 6]
        records.append((typ, payload))
        cursor = start
        if len(records) > 512:
            break
    records.reverse()
    return records


def _default_lens(edge: int, yaw: float = 0.0) -> LensParams:
    q = (1.0, 0.0, 0.0, 0.0)
    if abs(yaw - 180.0) < 1:
        q = (0.0, 0.0, 1.0, 0.0)
    return LensParams(
        fx=edge * 0.30,
        fy=edge * 0.30,
        cx=edge / 2,
        cy=edge / 2,
        k=(-0.18, 0.04, 0.0, 0.0, 0.0),
        width=edge,
        height=edge,
        extrinsic_q=q,
        yaw_deg=yaw,
    )


def _lens_from_proto(nodes: list[ProtoNode], edge: int) -> list[LensParams]:
    """Try to find fx/fy/cx/cy style calibration in ExtraInfo protobuf."""
    lenses: list[LensParams] = []

    def walk(n: ProtoNode, depth: int = 0) -> None:
        if depth > 12 or len(lenses) >= 2:
            return
        # Collect consecutive float fields that look like intrinsics
        if n.children:
            floats: list[tuple[int, float]] = []
            for c in n.children:
                if c.wire == 5 and isinstance(c.value, bytes) and len(c.value) == 4:
                    floats.append((c.field, struct.unpack("<f", c.value)[0]))
            by_f = {f: v for f, v in floats}
            # Common pattern: 1=fx 2=fy 3=cx 4=cy
            if 1 in by_f and by_f[1] > 100:
                fx = by_f.get(1, 0)
                fy = by_f.get(2, fx)
                cx = by_f.get(3, edge / 2)
                cy = by_f.get(4, edge / 2)
                if fx > 50:
                    k = tuple(by_f.get(i, 0.0) for i in (5, 6, 7, 8, 9))
                    lenses.append(
                        LensParams(
                            fx=fx,
                            fy=fy,
                            cx=cx,
                            cy=cy,
                            k=k,  # type: ignore[arg-type]
                            width=edge,
                            height=edge,
                        )
                    )
            for c in n.children:
                walk(c, depth + 1)

    for n in nodes:
        walk(n)
    return lenses[:2]


def _gyro_samples_from_records(records: list[tuple[int, bytes]]) -> list[tuple[float, float, float, float]]:
    """
    Return list of (t_s, gx, gy, gz) in rad/s.
    Insta packs gyro in several formats; we try common binary layouts.
    """
    samples: list[tuple[float, float, float, float]] = []
    for typ, payload in records:
        # Gyro-related record types commonly 0x0001 / 0x0002 / protobuf envelopes
        if typ in (0x0001, 0x0002, 0x0100, 0x0200) or len(payload) > 64:
            # Try packed int16 triples with time
            if len(payload) >= 20 and len(payload) % 8 == 0:
                # time_ms u32 + gx gy gz int16? or float32 triples
                pass
            # Float32 triples
            if len(payload) >= 12 and len(payload) % 12 == 0 and len(payload) < 8_000_000:
                n = len(payload) // 12
                if 10 < n < 500_000:
                    ok = True
                    tmp: list[tuple[float, float, float, float]] = []
                    for i in range(n):
                        gx, gy, gz = struct.unpack_from("<fff", payload, i * 12)
                        if not all(math.isfinite(v) and abs(v) < 50 for v in (gx, gy, gz)):
                            ok = False
                            break
                        tmp.append((i * 0.001, gx, gy, gz))
                    if ok and len(tmp) > len(samples):
                        samples = tmp
            # int16 gyro scaled: 32768 = 0
            if len(payload) >= 8 and len(payload) % 8 == 0:
                n = len(payload) // 8
                if 10 < n < 500_000:
                    tmp = []
                    ok = True
                    for i in range(n):
                        t_raw, gx, gy, gz = struct.unpack_from("<Hhhh", payload, i * 8)
                        # scale roughly to rad/s (device-dependent); 1000 dps full scale guess
                        scale = (1000.0 * math.pi / 180.0) / 32768.0
                        tmp.append((i * 0.001, gx * scale, gy * scale, gz * scale))
                    if ok and len(tmp) > len(samples):
                        samples = tmp
            # Protobuf envelope with nested bytes
            if payload and payload[0] < 0x80:
                try:
                    nodes = parse_protobuf(payload)
                    for n in nodes:
                        raw = n.as_bytes()
                        if len(raw) >= 24 and len(raw) % 6 == 0:
                            m = len(raw) // 6
                            if m > 50:
                                tmp = []
                                for i in range(m):
                                    ax, ay, az = struct.unpack_from("<HHH", raw, i * 6)
                                    # unsigned centered at 32768
                                    def c(v: int) -> float:
                                        return (v - 32768) * (2000.0 * math.pi / 180.0) / 32768.0

                                    tmp.append((i * 0.001, c(ax), c(ay), c(az)))
                                if len(tmp) > len(samples):
                                    samples = tmp
                except Exception:
                    pass
    return samples


def integrate_gyro(
    gyro: list[tuple[float, float, float, float]],
    duration_s: float,
) -> list[AttitudeSample]:
    """Integrate body rates to quaternions; level using mean gravity estimate if accel absent."""
    if not gyro:
        return []
    # Simple quaternion integration
    w, x, y, z = 1.0, 0.0, 0.0, 0.0
    out: list[AttitudeSample] = []
    prev_t = gyro[0][0]
    for t, gx, gy, gz in gyro:
        dt = max(0.0, min(0.05, t - prev_t))
        prev_t = t
        # dq/dt = 0.5 * q * omega
        ow, ox, oy, oz = 0.0, gx, gy, gz
        dw = 0.5 * (-x * ox - y * oy - z * oz)
        dx = 0.5 * (w * ox + y * oz - z * oy)
        dy = 0.5 * (w * oy - x * oz + z * ox)
        dz = 0.5 * (w * oz + x * oy - y * ox)
        w += dw * dt
        x += dx * dt
        y += dy * dt
        z += dz * dt
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        if t <= duration_s + 1.0:
            out.append(AttitudeSample(t_s=t, w=w, x=x, y=y, z=z))
    # Subsample to ~200 Hz max for sendcmd size
    if len(out) > duration_s * 220 and duration_s > 0:
        step = max(1, int(len(out) / (duration_s * 200)))
        out = out[::step]
    return out


def parse_insta_file(path: Path) -> Capture:
    path = path.resolve()
    videos, audio, fps, duration, bit_depth, frame_count = streams_from_probe(path)
    side_by_side = False
    layout = LensLayout.SINGLE
    if len(videos) >= 2:
        layout = LensLayout.DUAL_TRACK
    elif videos:
        w, h = videos[0].width, videos[0].height
        if w >= h * 1.8:
            layout = LensLayout.SIDE_BY_SIDE
            side_by_side = True
        elif abs(w - h) < 8:
            layout = LensLayout.SINGLE

    edge = max((max(v.width, v.height) for v in videos), default=2880)
    if side_by_side and videos:
        edge = max(videos[0].height, videos[0].width // 2)

    records = _read_trailer_records(path)
    cal = Calibration(model="Insta360")
    # Metadata record 0x0101 often holds ExtraInfo
    for typ, payload in records:
        if typ in (0x0101, 0x0000, 0x0270) or (len(payload) > 100 and payload[0] < 0x40):
            try:
                nodes = parse_protobuf(payload)
                found = _lens_from_proto(nodes, edge)
                if found:
                    cal.rear = found[0]
                    if len(found) > 1:
                        cal.front = found[1]
                    break
            except Exception:
                continue

    if cal.rear is None:
        cal.rear = _default_lens(edge, 0.0)
    if cal.front is None and layout in (LensLayout.DUAL_TRACK, LensLayout.SIDE_BY_SIDE, LensLayout.SPLIT_FILES):
        cal.front = _default_lens(edge, 180.0)

    gyro = _gyro_samples_from_records(records)
    attitude = integrate_gyro(gyro, duration_s=duration or 1.0)
    notes: list[str] = []
    if not records:
        notes.append("no Insta360 trailer found; using approximate calibration")
    if not attitude:
        notes.append("no gyro samples found; stabilization will be identity")

    return Capture(
        family=CameraFamily.INSTA,
        layout=layout,
        master_paths=[path],
        video_streams=videos,
        audio_stream_index=audio,
        calibration=cal,
        attitude=attitude,
        fps=fps or 30.0,
        duration_s=duration,
        frame_count=frame_count,
        bit_depth=bit_depth,
        side_by_side=side_by_side,
        notes=notes,
    )


_SPLIT_RE = re.compile(
    r"^(?P<prefix>VID|LRV)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<lens>\d{2})_(?P<seq>\d+)\.(?P<ext>insv|lrv)$",
    re.IGNORECASE,
)


def insta_capture_key(path: Path) -> tuple[str, str, str] | None:
    m = _SPLIT_RE.match(path.name)
    if not m:
        return None
    return m.group("date"), m.group("time"), m.group("seq")


def insta_lens_code(path: Path) -> str | None:
    m = _SPLIT_RE.match(path.name)
    if not m:
        return None
    return m.group("lens")


def load_insta_capture(
    master: Path,
    second_lens: Path | None = None,
    proxy: Path | None = None,
) -> Capture:
    master = master.resolve()
    if master.suffix.lower() == ".lrv" or master.name.upper().startswith("LRV_"):
        raise ValueError(f"LRV-Proxy wird nicht geladen — bitte .insv verwenden: {master.name}")
    cap = parse_insta_file(master)
    if second_lens is not None and second_lens.exists():
        second = second_lens.resolve()
        cap.master_paths = [master, second]
        cap.layout = LensLayout.SPLIT_FILES
        # Probe second for stream info
        v2, _, _, _, _, _ = streams_from_probe(second)
        if v2:
            # Keep both video stream descriptors; encode source mapping later
            if len(cap.video_streams) == 1:
                # Reindex second as logical stream 1 from other file
                s = v2[0]
                cap.video_streams.append(
                    VideoStreamInfo(
                        index=s.index,
                        width=s.width,
                        height=s.height,
                        fps=s.fps,
                        duration_s=s.duration_s,
                        codec=s.codec,
                        pix_fmt=s.pix_fmt,
                        bit_depth=s.bit_depth,
                    )
                )
        if cap.calibration.front is None:
            edge = cap.lens_edge
            cap.calibration.front = _default_lens(edge, 180.0)

    if proxy is not None and proxy.exists():
        # Remember path for UI only — never decode the LRV
        cap.proxy_path = proxy.resolve()
        log.debug("LRV vorhanden, wird nicht dekodiert: %s", proxy.name)
    return cap
