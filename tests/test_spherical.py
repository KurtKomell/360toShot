from pathlib import Path
import struct

from toshot.export.spherical import inject_spherical_v1, SPHERICAL_UUID


def test_inject_spherical_v1(tmp_path: Path):
    # minimal fake mp4: ftyp + moov + mdat
    def box(typ: bytes, payload: bytes) -> bytes:
        return struct.pack(">I4s", 8 + len(payload), typ) + payload

    data = box(b"ftyp", b"isom") + box(b"moov", b"\x00" * 16) + box(b"mdat", b"\x00" * 8)
    p = tmp_path / "a.mp4"
    p.write_bytes(data)
    inject_spherical_v1(p)
    out = p.read_bytes()
    assert b"uuid" in out
    assert SPHERICAL_UUID.bytes in out
    assert b"equirectangular" in out
