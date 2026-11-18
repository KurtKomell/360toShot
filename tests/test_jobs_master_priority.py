from pathlib import Path

from toshot.source.jobs import group_jobs, is_master_media


def test_is_master_media():
    assert is_master_media(Path("CAM_x.OSV"))
    assert is_master_media(Path("VID_20240101_120000_00_001.insv"))
    assert not is_master_media(Path("CAM_x.LRF"))
    assert not is_master_media(Path("VID_20240101_120000_01_001.lrv"))
    assert not is_master_media(Path("LRV_20240101_120000_11_001.insv"))


def test_group_jobs_skips_lrf_without_osv(tmp_path: Path):
    lrf = tmp_path / "CAM_only.LRF"
    lrf.write_bytes(b"x")
    jobs = group_jobs([lrf])
    assert jobs == []


def test_group_jobs_osv_with_lrf_proxy(tmp_path: Path):
    osv = tmp_path / "CAM_pair.OSV"
    lrf = tmp_path / "CAM_pair.LRF"
    osv.write_bytes(b"x")
    lrf.write_bytes(b"x")
    jobs = group_jobs([osv, lrf])
    assert len(jobs) == 1
    assert jobs[0].master == osv.resolve()
    assert jobs[0].proxy == lrf.resolve()
