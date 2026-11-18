from pathlib import Path
import struct

from toshot.export.spherical import (
    SPHERICAL_UUID,
    inject_spherical_v1,
    strip_spherical_uuid,
)


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def test_inject_spherical_appends_after_mdat(tmp_path: Path):
    # faststart layout: ftyp + moov + mdat
    data = _box(b"ftyp", b"isom") + _box(b"moov", b"\x00" * 16) + _box(b"mdat", b"MEDIA!!")
    p = tmp_path / "a.mp4"
    p.write_bytes(data)
    inject_spherical_v1(p)
    out = p.read_bytes()
    assert SPHERICAL_UUID.bytes in out
    assert b"equirectangular" in out
    # mdat must stay at the same offset; uuid only after it
    assert out.index(b"mdat") == data.index(b"mdat")
    assert out.index(b"uuid") > out.index(b"mdat")
    # idempotent
    inject_spherical_v1(p)
    assert p.read_bytes().count(SPHERICAL_UUID.bytes) == 1


def test_repair_uuid_inserted_before_mdat(tmp_path: Path):
    xml = b"<rdf:SphericalVideo/>"
    bad_uuid = struct.pack(">I4s", 8 + 16 + len(xml), b"uuid") + SPHERICAL_UUID.bytes + xml
    data = (
        _box(b"ftyp", b"isom")
        + _box(b"moov", b"\x00" * 16)
        + bad_uuid
        + _box(b"mdat", b"MEDIA!!")
    )
    p = tmp_path / "broken.mp4"
    p.write_bytes(data)
    assert strip_spherical_uuid(p)
    fixed = p.read_bytes()
    assert SPHERICAL_UUID.bytes not in fixed
    assert fixed == (
        _box(b"ftyp", b"isom") + _box(b"moov", b"\x00" * 16) + _box(b"mdat", b"MEDIA!!")
    )
    inject_spherical_v1(p)
    out = p.read_bytes()
    assert out.index(b"uuid") > out.index(b"mdat")
