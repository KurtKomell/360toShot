from pathlib import Path

from toshot.ffmpeg_bin import escape_filter_path


def test_escape_filter_path_escapes_drive_colon():
    # Use a synthetic Windows-style path via as_posix behaviour on whatever OS.
    # On Windows Path('C:/tmp/x.cmd') keeps the drive; escape must quote + \:
    p = Path("C:/Users/manue/AppData/Local/Temp/stab.cmd")
    out = escape_filter_path(p)
    assert out.startswith("'")
    assert out.endswith("'")
    assert r"\:" in out or ":" not in out[1:-1]
    # colon after drive letter must be escaped
    assert "C\\:/" in out or "C\\:" in out
