import struct
from pathlib import Path

from toshot import mp4
from toshot.protobuf_lite import parse_protobuf


def _box(typ: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    return struct.pack(">I4s", size, typ) + payload


def test_iter_boxes_and_extended_size(tmp_path: Path):
    # large-size header size==1
    payload = b"hello-camd"
    # size=1 means 64-bit size follows
    total = 16 + len(payload)
    data = struct.pack(">I4sQ", 1, b"camd", total) + payload
    data = _box(b"ftyp", b"isom") + data
    p = tmp_path / "x.bin"
    p.write_bytes(data)
    boxes = list(mp4.iter_boxes(p))
    assert boxes[0].type == "ftyp"
    assert boxes[1].type == "camd"
    assert boxes[1].header_size == 16
    assert mp4.read_bytes(p, boxes[1].data_offset, len(payload)) == payload


def test_parse_protobuf_varint_and_bytes():
    # field 1 wire 2 length-delimited "Osmo"
    msg = b"\x0a\x04Osmo"
    # field 2 wire 0 varint 7
    msg += b"\x10\x07"
    nodes = parse_protobuf(msg)
    assert nodes[0].field == 1
    assert nodes[0].as_bytes() == b"Osmo"
    assert nodes[1].field == 2
    assert nodes[1].as_int() == 7
