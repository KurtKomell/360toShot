from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ProtoNode:
    field: int
    wire: int
    value: int | bytes | list["ProtoNode"]
    children: list["ProtoNode"] = field(default_factory=list)

    def as_bytes(self) -> bytes:
        if isinstance(self.value, bytes):
            return self.value
        return b""

    def as_int(self) -> int:
        if isinstance(self.value, int):
            return self.value
        return 0

    def as_float(self) -> float:
        import struct

        raw = self.as_bytes()
        if len(raw) == 4:
            return struct.unpack("<f", raw)[0]
        if len(raw) == 8:
            return struct.unpack("<d", raw)[0]
        if isinstance(self.value, int):
            # fixed32 stored as int
            return struct.unpack("<f", struct.pack("<I", self.value & 0xFFFFFFFF))[0]
        return 0.0

    def get(self, *path: int) -> ProtoNode | None:
        nodes = [self]
        for f in path:
            nxt: list[ProtoNode] = []
            for n in nodes:
                kids = n.children if n.children else (
                    n.value if isinstance(n.value, list) else []
                )
                for c in kids:
                    if c.field == f:
                        nxt.append(c)
            if not nxt:
                return None
            nodes = nxt
        return nodes[0]

    def get_all(self, field: int) -> list[ProtoNode]:
        kids = self.children if self.children else (
            self.value if isinstance(self.value, list) else []
        )
        return [c for c in kids if c.field == field]


def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while i < len(data):
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
        if shift > 70:
            break
    return result, i


def parse_protobuf(data: bytes, max_depth: int = 12) -> list[ProtoNode]:
    return list(_parse(data, 0, len(data), max_depth))


def _parse(data: bytes, start: int, end: int, depth: int) -> Iterator[ProtoNode]:
    i = start
    while i < end:
        key, i = _read_varint(data, i)
        if i > end:
            break
        field = key >> 3
        wire = key & 0x7
        if field == 0:
            break
        if wire == 0:  # varint
            val, i = _read_varint(data, i)
            yield ProtoNode(field=field, wire=wire, value=val)
        elif wire == 1:  # 64-bit
            if i + 8 > end:
                break
            chunk = data[i : i + 8]
            i += 8
            yield ProtoNode(field=field, wire=wire, value=chunk)
        elif wire == 2:  # length-delimited
            length, i = _read_varint(data, i)
            if i + length > end or length < 0:
                break
            chunk = data[i : i + length]
            i += length
            children: list[ProtoNode] = []
            if depth > 0 and _looks_like_message(chunk):
                try:
                    children = list(_parse(chunk, 0, len(chunk), depth - 1))
                except Exception:
                    children = []
            yield ProtoNode(field=field, wire=wire, value=chunk, children=children)
        elif wire == 5:  # 32-bit
            if i + 4 > end:
                break
            chunk = data[i : i + 4]
            i += 4
            yield ProtoNode(field=field, wire=wire, value=chunk)
        else:
            break


def _looks_like_message(data: bytes) -> bool:
    if not data:
        return False
    # Heuristic: first key should be a plausible field/wire
    try:
        key, i = _read_varint(data, 0)
        field = key >> 3
        wire = key & 0x7
        return field > 0 and wire in (0, 1, 2, 5) and i < len(data)
    except Exception:
        return False


def floats_from_node(node: ProtoNode | None) -> list[float]:
    if node is None:
        return []
    import struct

    out: list[float] = []
    if node.wire == 5 and isinstance(node.value, bytes) and len(node.value) == 4:
        out.append(struct.unpack("<f", node.value)[0])
        return out
    if node.wire == 1 and isinstance(node.value, bytes) and len(node.value) == 8:
        out.append(struct.unpack("<d", node.value)[0])
        return out
    if node.wire == 2 and isinstance(node.value, bytes):
        raw = node.value
        if len(raw) % 4 == 0 and len(raw) <= 64:
            for j in range(0, len(raw), 4):
                out.append(struct.unpack("<f", raw[j : j + 4])[0])
            return out
    if node.children:
        for c in sorted(node.children, key=lambda n: n.field):
            out.extend(floats_from_node(c))
    return out
