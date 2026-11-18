from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


@dataclass(frozen=True)
class Box:
    offset: int
    size: int
    type: str
    header_size: int

    @property
    def data_offset(self) -> int:
        return self.offset + self.header_size

    @property
    def end(self) -> int:
        return self.offset + self.size


def _read_box_header(f: BinaryIO, offset: int) -> Box | None:
    f.seek(offset)
    header = f.read(16)
    if len(header) < 8:
        return None
    size, typ = struct.unpack(">I4s", header[:8])
    header_size = 8
    if size == 1:
        if len(header) < 16:
            return None
        size = struct.unpack(">Q", header[8:16])[0]
        header_size = 16
    elif size == 0:
        f.seek(0, 2)
        size = f.tell() - offset
    type_s = typ.decode("latin1")
    if size < header_size:
        return None
    return Box(offset=offset, size=size, type=type_s, header_size=header_size)


def iter_boxes(path: Path, start: int = 0, end: int | None = None) -> Iterator[Box]:
    with path.open("rb") as f:
        f.seek(0, 2)
        file_end = f.tell() if end is None else min(end, f.tell())
        offset = start
        while offset + 8 <= file_end:
            box = _read_box_header(f, offset)
            if box is None or box.size <= 0:
                break
            yield box
            if box.end <= offset:
                break
            offset = box.end
            if offset >= file_end:
                break


def find_box(path: Path, box_type: str, start: int = 0, end: int | None = None) -> Box | None:
    for box in iter_boxes(path, start=start, end=end):
        if box.type == box_type:
            return box
    return None


def find_boxes(path: Path, box_type: str, start: int = 0, end: int | None = None) -> list[Box]:
    return [b for b in iter_boxes(path, start=start, end=end) if b.type == box_type]


def read_bytes(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(size)


def walk_children(path: Path, parent: Box, container_types: set[str] | None = None) -> Iterator[Box]:
    containers = container_types or {
        "moov",
        "trak",
        "mdia",
        "minf",
        "stbl",
        "udta",
        "meta",
        "dinf",
        "edts",
    }
    stack = [parent]
    while stack:
        cur = stack.pop()
        if cur.type not in containers:
            continue
        for child in iter_boxes(path, start=cur.data_offset, end=cur.end):
            yield child
            if child.type in containers:
                stack.append(child)


@dataclass
class SampleTable:
    sizes: list[int]
    offsets: list[int]
    chunk_samples: list[tuple[int, int]]  # first_chunk (1-based), samples_per_chunk


def _parse_stsz(data: bytes) -> list[int]:
    if len(data) < 12:
        return []
    sample_size, sample_count = struct.unpack(">II", data[4:12])
    if sample_size != 0:
        return [sample_size] * sample_count
    needed = 12 + 4 * sample_count
    if len(data) < needed:
        sample_count = max(0, (len(data) - 12) // 4)
    return list(struct.unpack(f">{sample_count}I", data[12 : 12 + 4 * sample_count]))


def _parse_stco(data: bytes, wide: bool) -> list[int]:
    if len(data) < 8:
        return []
    count = struct.unpack(">I", data[4:8])[0]
    if wide:
        need = 8 + 8 * count
        if len(data) < need:
            count = max(0, (len(data) - 8) // 8)
        return list(struct.unpack(f">{count}Q", data[8 : 8 + 8 * count]))
    need = 8 + 4 * count
    if len(data) < need:
        count = max(0, (len(data) - 8) // 4)
    return list(struct.unpack(f">{count}I", data[8 : 8 + 4 * count]))


def _parse_stsc(data: bytes) -> list[tuple[int, int]]:
    if len(data) < 8:
        return []
    count = struct.unpack(">I", data[4:8])[0]
    out: list[tuple[int, int]] = []
    pos = 8
    for _ in range(count):
        if pos + 12 > len(data):
            break
        first, spc, _desc = struct.unpack(">III", data[pos : pos + 12])
        out.append((first, spc))
        pos += 12
    return out


def load_sample_table(path: Path, trak: Box) -> SampleTable | None:
    stbl = None
    for child in walk_children(path, trak):
        if child.type == "stbl":
            stbl = child
            break
    if stbl is None:
        return None
    sizes: list[int] = []
    offsets: list[int] = []
    chunks: list[tuple[int, int]] = []
    for child in iter_boxes(path, start=stbl.data_offset, end=stbl.end):
        data = read_bytes(path, child.data_offset, child.size - child.header_size)
        if child.type == "stsz":
            sizes = _parse_stsz(data)
        elif child.type == "stco":
            offsets = _parse_stco(data, wide=False)
        elif child.type == "co64":
            offsets = _parse_stco(data, wide=True)
        elif child.type == "stsc":
            chunks = _parse_stsc(data)
    if not sizes or not offsets:
        return None
    return SampleTable(sizes=sizes, offsets=offsets, chunk_samples=chunks or [(1, 1)])


def expand_sample_offsets(table: SampleTable) -> list[tuple[int, int]]:
    """Return list of (file_offset, size) for each sample."""
    if not table.chunk_samples:
        return list(zip(table.offsets, table.sizes))
    result: list[tuple[int, int]] = []
    size_i = 0
    for chunk_i, chunk_offset in enumerate(table.offsets):
        chunk_no = chunk_i + 1
        spc = 1
        for first, samples_per in table.chunk_samples:
            if chunk_no >= first:
                spc = samples_per
            else:
                break
        off = chunk_offset
        for _ in range(spc):
            if size_i >= len(table.sizes):
                return result
            sz = table.sizes[size_i]
            result.append((off, sz))
            off += sz
            size_i += 1
    return result


def track_handler(path: Path, trak: Box) -> str:
    for child in walk_children(path, trak):
        if child.type == "hdlr":
            data = read_bytes(path, child.data_offset, child.size - child.header_size)
            if len(data) >= 12:
                return data[8:12].decode("latin1", errors="replace")
    return ""


def track_name(path: Path, trak: Box) -> str:
    for child in walk_children(path, trak):
        if child.type == "hdlr":
            data = read_bytes(path, child.data_offset, child.size - child.header_size)
            # name is null-terminated C string after component name fields
            if len(data) > 24:
                return data[24:].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return ""
