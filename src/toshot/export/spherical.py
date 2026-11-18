from __future__ import annotations

import struct
import uuid
from pathlib import Path


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


def inject_spherical_v1(path: Path) -> None:
    """
    Append/inject a uuid box with Google Spherical V1 XML into an MP4.
    Best-effort: inserts a top-level uuid box before the final free space if needed
    by rewriting with a sidecar approach — here we append after moov if possible.
    """
    data = bytearray(path.read_bytes())
    xml = SPHERICAL_XML.encode("utf-8")
    # uuid box: size(4) + 'uuid'(4) + usertype(16) + xml
    payload = SPHERICAL_UUID.bytes + xml
    size = 8 + len(payload)
    box = struct.pack(">I4s", size, b"uuid") + payload

    # Find moov; insert uuid after moov (compatible with many players)
    offset = 0
    moov_end = None
    while offset + 8 <= len(data):
        size32 = struct.unpack_from(">I", data, offset)[0]
        typ = data[offset + 4 : offset + 8]
        header = 8
        if size32 == 1:
            if offset + 16 > len(data):
                break
            size64 = struct.unpack_from(">Q", data, offset + 8)[0]
            total = size64
            header = 16
        elif size32 == 0:
            total = len(data) - offset
        else:
            total = size32
        if typ == b"moov":
            moov_end = offset + total
            break
        if total < header:
            break
        offset += total

    if moov_end is None:
        # append at end
        path.write_bytes(bytes(data) + box)
        return

    new_data = bytes(data[:moov_end]) + box + bytes(data[moov_end:])
    path.write_bytes(new_data)
