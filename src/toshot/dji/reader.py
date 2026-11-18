from __future__ import annotations

import logging
import math
import struct
from pathlib import Path

from toshot import mp4
from toshot.protobuf_lite import ProtoNode, floats_from_node, parse_protobuf
from toshot.source.models import (
    AttitudeSample,
    Calibration,
    CameraFamily,
    Capture,
    LensLayout,
    LensParams,
)
from toshot.source.probe_streams import streams_from_probe

log = logging.getLogger(__name__)


def _float_field(node: ProtoNode | None, field: int, default: float = 0.0) -> float:
    if node is None:
        return default
    for c in node.get_all(field):
        vals = floats_from_node(c)
        if vals:
            return vals[0]
        if c.wire == 5 and isinstance(c.value, bytes) and len(c.value) == 4:
            return struct.unpack("<f", c.value)[0]
        if c.wire == 0:
            # unlikely for floats but keep safe
            continue
    return default


def _quat_from_node(node: ProtoNode | None) -> tuple[float, float, float, float]:
    if node is None:
        return (1.0, 0.0, 0.0, 0.0)
    # cam_extri_q: message of w,x,y,z as fields 1..4
    if node.children:
        w = _float_field(node, 1, 1.0)
        x = _float_field(node, 2, 0.0)
        y = _float_field(node, 3, 0.0)
        z = _float_field(node, 4, 0.0)
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        return (w / n, x / n, y / n, z / n)
    vals = floats_from_node(node)
    if len(vals) >= 4:
        w, x, y, z = vals[:4]
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        return (w / n, x / n, y / n, z / n)
    return (1.0, 0.0, 0.0, 0.0)


def _parse_dewarp_slot(slot: ProtoNode) -> LensParams | None:
    fx = _float_field(slot, 1)
    fy = _float_field(slot, 2, fx)
    if fx <= 0:
        return None
    cx = _float_field(slot, 3)
    cy = _float_field(slot, 4)
    k = (
        _float_field(slot, 5),
        _float_field(slot, 6),
        _float_field(slot, 7),
        _float_field(slot, 8),
        _float_field(slot, 15),
    )
    width = int(_float_field(slot, 10, 3840) or 3840)
    height = int(_float_field(slot, 11, 3840) or 3840)
    yaw = _float_field(slot, 12)
    pitch = _float_field(slot, 13)
    roll = _float_field(slot, 14)
    q_node = slot.get(28) or slot.get(21)
    q = _quat_from_node(q_node)
    return LensParams(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k=k,
        width=width,
        height=height,
        extrinsic_q=q,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
    )


def parse_camd_calibration(path: Path) -> Calibration:
    camd = mp4.find_box(path, "camd")
    cal = Calibration()
    if camd is None:
        log.warning("No camd box in %s", path.name)
        return cal

    # camd is itself an MP4; read its mdat
    inner_mdat = None
    for box in mp4.iter_boxes(path, start=camd.data_offset, end=camd.end):
        if box.type == "mdat":
            inner_mdat = box
            break
    if inner_mdat is None:
        # sometimes camd payload starts directly with length-delimited protobuf
        data = mp4.read_bytes(path, camd.data_offset, min(camd.size - camd.header_size, 512 * 1024))
    else:
        data = mp4.read_bytes(
            path, inner_mdat.data_offset, min(inner_mdat.size - inner_mdat.header_size, 512 * 1024)
        )

    nodes = parse_protobuf(data)
    # Records: 1 header, 2 config, 3 per-frame...
    headers = [n for n in nodes if n.field == 1]
    configs = [n for n in nodes if n.field == 2]
    if headers:
        h = headers[0]
        cal.serial = _nested_bytes_str(h, 1, 5)
        cal.firmware = _nested_bytes_str(h, 1, 6)
        cal.model = _nested_bytes_str(h, 1, 10) or "Osmo 360"

    slots: list[LensParams] = []
    for cfg in configs:
        # Prefer field 6 slots (Osmo); fall back to any nested DewarpParams-like
        candidates = cfg.get_all(6)
        if not candidates:
            candidates = cfg.get_all(5)
        for slot_parent in candidates:
            for slot in slot_parent.children or [slot_parent]:
                if slot.field == 0:
                    continue
                lens = _parse_dewarp_slot(slot if slot.children else slot_parent)
                if lens is not None:
                    slots.append(lens)
            # Also try each numbered slot under field 6
            if slot_parent.children:
                for slot in sorted(slot_parent.children, key=lambda n: n.field):
                    lens = _parse_dewarp_slot(slot)
                    if lens is not None and all(
                        abs(lens.fx - s.fx) > 1 or abs(lens.cx - s.cx) > 1 for s in slots
                    ):
                        slots.append(lens)

    # Deduplicate by fx/cx
    unique: list[LensParams] = []
    for lens in slots:
        if any(abs(lens.fx - u.fx) < 0.5 and abs(lens.cx - u.cx) < 0.5 for u in unique):
            continue
        unique.append(lens)
        if len(unique) >= 2:
            break

    if len(unique) >= 1:
        cal.rear = unique[0]
    if len(unique) >= 2:
        cal.front = unique[1]
    return cal


def _nested_bytes_str(node: ProtoNode, *path: int) -> str:
    cur: ProtoNode | None = node
    for f in path:
        if cur is None:
            return ""
        cur = cur.get(f)
    if cur is None:
        return ""
    if isinstance(cur.value, bytes):
        return cur.value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return ""


def _extract_quat_from_product_meta(data: bytes) -> list[tuple[float, float, float, float]]:
    """Best-effort extraction of fused attitude quaternions from a djmd sample."""
    nodes = parse_protobuf(data)
    quats: list[tuple[float, float, float, float]] = []

    def walk(n: ProtoNode, depth: int = 0) -> None:
        if depth > 10:
            return
        # Look for messages with 4 float children (w,x,y,z)
        if n.children:
            floats = []
            for c in sorted(n.children, key=lambda x: x.field):
                if c.wire == 5 and isinstance(c.value, bytes) and len(c.value) == 4:
                    floats.append(struct.unpack("<f", c.value)[0])
            if len(floats) >= 4 and 0.5 < math.sqrt(sum(v * v for v in floats[:4])) < 1.5:
                w, x, y, z = floats[:4]
                nrm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
                quats.append((w / nrm, x / nrm, y / nrm, z / nrm))
            for c in n.children:
                walk(c, depth + 1)

    for n in nodes:
        walk(n)
    return quats


def parse_djmd_attitude(path: Path, fps: float) -> list[AttitudeSample]:
    moov = mp4.find_box(path, "moov")
    if moov is None:
        return []
    samples: list[AttitudeSample] = []
    traks = [b for b in mp4.iter_boxes(path, start=moov.data_offset, end=moov.end) if b.type == "trak"]
    for trak in traks:
        handler = mp4.track_handler(path, trak)
        name = mp4.track_name(path, trak).lower()
        # djmd uses meta handler; prefer CAM meta
        if handler not in ("meta", "djmd") and "meta" not in name and "djmd" not in name:
            # still allow meta tracks
            if handler != "meta":
                continue
        table = mp4.load_sample_table(path, trak)
        if table is None or not table.sizes:
            continue
        # Skip tiny non-telemetry tracks
        if max(table.sizes) < 32:
            continue
        offsets = mp4.expand_sample_offsets(table)
        frame_i = 0
        for off, sz in offsets:
            if sz <= 0 or sz > 2_000_000:
                continue
            data = mp4.read_bytes(path, off, sz)
            quats = _extract_quat_from_product_meta(data)
            if not quats:
                continue
            # Spread quats across the sample's time window (~1/fps)
            t0 = frame_i / fps if fps > 0 else frame_i * 0.033
            dt = (1.0 / fps / max(len(quats), 1)) if fps > 0 else 0.001
            for j, (w, x, y, z) in enumerate(quats):
                samples.append(AttitudeSample(t_s=t0 + j * dt, w=w, x=x, y=y, z=z))
            frame_i += 1
        if samples:
            break
    samples.sort(key=lambda s: s.t_s)
    return samples


def _default_lens(edge: int = 3840) -> LensParams:
    # Approximate fisheye for Osmo when camd is missing
    return LensParams(
        fx=edge * 0.28,
        fy=edge * 0.28,
        cx=edge / 2,
        cy=edge / 2,
        k=(-0.2, 0.05, 0.0, 0.0, 0.0),
        width=edge,
        height=edge,
    )


def parse_dji_file(path: Path) -> Capture:
    path = path.resolve()
    log.info("DJI parse: %s", path)
    videos, audio, fps, duration, bit_depth, frame_count = streams_from_probe(path)
    layout = LensLayout.DUAL_TRACK if len(videos) >= 2 else LensLayout.SINGLE
    if len(videos) == 1 and videos[0].width > videos[0].height * 1.5:
        layout = LensLayout.SIDE_BY_SIDE
        side_by_side = True
    else:
        side_by_side = layout == LensLayout.SIDE_BY_SIDE

    log.info("Lese camd-Kalibrierung …")
    cal = parse_camd_calibration(path)
    edge = max((max(v.width, v.height) for v in videos), default=3840)
    if side_by_side and videos:
        edge = max(videos[0].height, videos[0].width // 2)
    used_approx = False
    if cal.rear is None:
        cal.rear = _default_lens(edge)
        used_approx = True
    if cal.front is None and (layout == LensLayout.DUAL_TRACK or side_by_side):
        front = _default_lens(edge)
        # Front lens facing opposite yaw
        cal.front = LensParams(
            fx=front.fx,
            fy=front.fy,
            cx=front.cx,
            cy=front.cy,
            k=front.k,
            width=front.width,
            height=front.height,
            extrinsic_q=(0.0, 0.0, 1.0, 0.0),  # 180 deg about Y
            yaw_deg=180.0,
        )
        used_approx = True
    if not cal.model:
        cal.model = "Osmo 360"
    eq_w, eq_h = edge * 2 - ((edge * 2) % 2), edge - (edge % 2)
    log.info(
        "Specs: equirect=%dx%d @ %.3f fps, frames=%d, model=%s",
        eq_w,
        eq_h,
        fps or 30.0,
        frame_count or int(round((duration or 0) * (fps or 30.0))),
        cal.model,
    )
    log.info(
        "Kalibrierung: model=%s rear=%s front=%s",
        cal.model,
        cal.rear is not None,
        cal.front is not None,
    )

    log.info("Lese djmd-Attitude (kann auf NAS dauern) …")
    attitude = parse_djmd_attitude(path, fps=fps or 30.0)
    log.info("Attitude-Samples: %d", len(attitude))
    notes: list[str] = []
    if used_approx:
        notes.append("camd calibration missing; using approximate lens model")
    if not attitude:
        notes.append("no djmd attitude found; stabilization will be identity")

    return Capture(
        family=CameraFamily.DJI,
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


def load_dji_capture(master: Path, proxy: Path | None = None) -> Capture:
    master = master.resolve()
    if master.suffix.lower() == ".lrf":
        raise ValueError(f"LRF-Proxy wird nicht geladen — bitte .OSV verwenden: {master.name}")
    cap = parse_dji_file(master)
    if proxy is not None and proxy.exists():
        # Remember path for UI only — never ffprobe/decode the LRF
        cap.proxy_path = proxy.resolve()
        log.debug("LRF vorhanden, wird nicht dekodiert: %s", proxy.name)
    return cap
