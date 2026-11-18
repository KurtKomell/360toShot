from __future__ import annotations

import logging
import struct
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

SPHERICAL_XML = """<?xml version="1.0"?>
<rdf:SphericalVideo
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
  <GSpherical:Spherical>true</GSpherical:Spherical>
  <GSpherical:Stitched>true</GSpherical:Stitched>
  <GSpherical:StitchingSoftware>toshot</GSpherical:StitchingSoftware>
  <GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
</rdf:SphericalVideo>
"""

# UUID for Google spherical metadata (XMP)
SPHERICAL_UUID = uuid.UUID("fffcc831-3fcc-4ef8-bb65-f2223cd08d9d")


def _iter_top_boxes(data: bytes | bytearray):
    offset = 0
    n = len(data)
    while offset + 8 <= n:
        size32 = struct.unpack_from(">I", data, offset)[0]
        typ = bytes(data[offset + 4 : offset + 8])
        header = 8
        if size32 == 1:
            if offset + 16 > n:
                break
            total = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size32 == 0:
            total = n - offset
        else:
            total = size32
        if total < header:
            break
        yield offset, typ, total
        offset += total


def _has_spherical_uuid(data: bytes | bytearray) -> bool:
    needle = SPHERICAL_UUID.bytes
    for offset, typ, total in _iter_top_boxes(data):
        if typ != b"uuid" or total < 24:
            continue
        if bytes(data[offset + 8 : offset + 24]) == needle:
            return True
    return False


def _make_uuid_box() -> bytes:
    xml = SPHERICAL_XML.encode("utf-8")
    payload = SPHERICAL_UUID.bytes + xml
    return struct.pack(">I4s", 8 + len(payload), b"uuid") + payload


def strip_spherical_uuid(path: Path) -> bool:
    """
    Remove a top-level Google spherical uuid box inserted between moov and mdat
    (or elsewhere). Restores chunk offsets when the box sat before mdat.
    Returns True if a box was removed.
    """
    data = bytearray(path.read_bytes())
    needle = SPHERICAL_UUID.bytes
    for offset, typ, total in _iter_top_boxes(data):
        if typ != b"uuid" or total < 24:
            continue
        if bytes(data[offset + 8 : offset + 24]) != needle:
            continue
        del data[offset : offset + total]
        path.write_bytes(data)
        log.info("Removed spherical uuid box (%d bytes) from %s", total, path.name)
        return True
    return False


def _uuid_before_mdat(data: bytes | bytearray) -> bool:
    """True if spherical uuid sits ahead of mdat (breaks chunk offsets)."""
    needle = SPHERICAL_UUID.bytes
    seen = False
    for offset, typ, total in _iter_top_boxes(data):
        if typ == b"uuid" and total >= 24 and bytes(data[offset + 8 : offset + 24]) == needle:
            seen = True
        elif typ == b"mdat":
            return seen
    return False


def inject_spherical_v1(path: Path) -> None:
    """
    Inject Google Spherical V1 uuid metadata into an MP4.

    Always appends at the end of the file. Inserting after moov (before mdat)
    with +faststart shifts media without updating stco/co64 and corrupts playback.
    """
    data = path.read_bytes()
    if _has_spherical_uuid(data):
        if _uuid_before_mdat(data):
            log.warning(
                "Spherical uuid before mdat in %s — repairing offsets",
                path.name,
            )
            strip_spherical_uuid(path)
            data = path.read_bytes()
        else:
            log.info("Spherical metadata already present in %s", path.name)
            return
    box = _make_uuid_box()
    path.write_bytes(data + box)
