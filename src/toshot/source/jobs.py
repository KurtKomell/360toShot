from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from toshot.dji.reader import load_dji_capture
from toshot.insta.reader import (
    insta_capture_key,
    insta_lens_code,
    load_insta_capture,
)
from toshot.source.models import Capture

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".osv", ".lrf", ".insv", ".lrv"}
MASTER_EXTS = {".osv", ".insv"}
PROXY_EXTS = {".lrf", ".lrv"}


def _ext(p: Path) -> str:
    return p.suffix.lower()


def is_master_media(path: Path) -> bool:
    """True for conversion sources (.OSV / .insv). Proxies (.LRF / .lrv) are never masters."""
    e = _ext(path)
    if e in MASTER_EXTS:
        # Insta legacy proxy sometimes named LRV_…_11_….insv
        if e == ".insv" and (insta_lens_code(path) == "11" or path.name.upper().startswith("LRV_")):
            return False
        return True
    return False


def normalize_path(path: str | Path) -> Path:
    """Expand user; resolve carefully so UNC paths with spaces stay intact."""
    p = Path(path).expanduser()
    try:
        return p.resolve()
    except OSError:
        return p


@dataclass(frozen=True)
class CaptureJob:
    """Path-only grouping — no ffprobe / metadata until load()."""

    kind: str  # dji | insta
    master: Path
    second: Path | None = None
    proxy: Path | None = None

    @property
    def id(self) -> str:
        return self.master.stem

    def load(self) -> Capture:
        log.info("Lade Capture %s …", self.master)
        if self.kind == "dji":
            return load_dji_capture(self.master, proxy=self.proxy)
        return load_insta_capture(self.master, second_lens=self.second, proxy=self.proxy)


def list_media_files(folder: Path) -> list[Path]:
    folder = normalize_path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and _ext(p) in SUPPORTED_EXTS]
    log.info("Ordner %s: %d Medien-Datei(en)", folder, len(files))
    return files


def discover_jobs(folder: Path) -> list[CaptureJob]:
    """Fast: only filesystem pairing, no probing."""
    files = list_media_files(folder)
    return group_jobs(files)


def jobs_total_duration(jobs: list[CaptureJob]) -> tuple[float, list[tuple[str, float]]]:
    """
    Sum media durations via ffprobe. Prefers proxy (.LRF/.lrv) when present
    (faster on NAS); falls back to master. Returns (total_s, [(id, dur_s), …]).
    """
    from toshot.ffmpeg_bin import probe_duration

    rows: list[tuple[str, float]] = []
    total = 0.0
    for job in jobs:
        src = job.proxy if job.proxy is not None else job.master
        try:
            dur = probe_duration(src)
        except Exception as exc:
            log.warning("Dauer nicht lesbar (%s): %s", src.name, exc)
            dur = 0.0
        rows.append((job.id, dur))
        total += dur
    return total, rows


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS (or M:SS if under one hour)."""
    s = int(round(max(0.0, seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def group_jobs(files: list[Path]) -> list[CaptureJob]:
    files = [normalize_path(f) for f in files]
    used: set[Path] = set()
    jobs: list[CaptureJob] = []

    by_stem: dict[str, dict[str, Path]] = {}
    for f in files:
        by_stem.setdefault(f.stem.lower(), {})[_ext(f)] = f

    for _stem, exts in sorted(by_stem.items()):
        osv = exts.get(".osv")
        lrf = exts.get(".lrf")
        if osv:
            # Always convert OSV; LRF is tracking proxy only
            jobs.append(CaptureJob(kind="dji", master=osv, proxy=lrf))
            used.add(osv)
            if lrf:
                used.add(lrf)
                log.debug("LRF nur als Pfad vermerkt: %s", lrf.name)
        elif lrf and lrf not in used:
            log.warning("Überspringe LRF ohne OSV: %s", lrf.name)
            used.add(lrf)

    insta_files = [f for f in files if f not in used and _ext(f) in {".insv", ".lrv"}]
    buckets: dict[tuple[str, str, str], dict[str, Path]] = {}
    loose: list[Path] = []
    for f in insta_files:
        key = insta_capture_key(f)
        if key is None:
            loose.append(f)
            continue
        lens = insta_lens_code(f) or "00"
        buckets.setdefault(key, {})[f"{lens}:{_ext(f)}"] = f

    for _key, roles in sorted(buckets.items()):
        master = second = proxy = None
        for role, path in roles.items():
            lens, ext = role.split(":")
            if ext == ".lrv" or (ext == ".insv" and lens == "11"):
                proxy = path
            elif lens == "00" and ext == ".insv":
                master = path
            elif lens == "10" and ext == ".insv":
                second = path
            elif master is None and ext == ".insv" and is_master_media(path):
                master = path
        # Never promote LRV/proxy to master
        if master is None:
            if proxy is not None:
                log.warning(
                    "Überspringe Proxy ohne INSV: %s",
                    proxy.name,
                )
            for p in roles.values():
                used.add(p)
            continue
        if master in used:
            continue
        lens = insta_lens_code(master)
        if lens == "10" and second is None and not any(
            insta_lens_code(p) == "00" for p in roles.values()
        ):
            log.warning("Skipping orphan second-lens file %s", master.name)
            continue
        jobs.append(CaptureJob(kind="insta", master=master, second=second, proxy=proxy))
        if proxy is not None:
            log.info("Proxy übersprungen (nur Tracking): %s", proxy.name)
        for p in roles.values():
            used.add(p)

    for f in loose:
        if f in used:
            continue
        if not is_master_media(f):
            log.warning("Überspringe Proxy/lose LRV: %s", f.name)
            used.add(f)
            continue
        jobs.append(CaptureJob(kind="insta", master=f))
        used.add(f)
    log.info("Gruppiert: %d Capture-Job(s)", len(jobs))
    for i, job in enumerate(jobs, 1):
        log.info(
            "  [%d/%d] %s  master=%s  proxy=%s",
            i,
            len(jobs),
            job.kind,
            job.master.name,
            job.proxy.name if job.proxy else "-",
        )
    return jobs


def job_for_path(path: Path) -> CaptureJob:
    """Build a single job for one file + siblings. Always prefers OSV/INSV over LRF/LRV."""
    path = normalize_path(path)
    ext = _ext(path)
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension: {path.suffix}")

    folder = path.parent
    stem = path.stem.lower()

    if ext in {".osv", ".lrf"}:
        osv = path if ext == ".osv" else None
        lrf = path if ext == ".lrf" else None
        for sib in folder.iterdir():
            if not sib.is_file():
                continue
            if sib.stem.lower() != stem:
                continue
            e = _ext(sib)
            if e == ".osv":
                osv = sib
            elif e == ".lrf":
                lrf = sib
        if osv is None:
            raise FileNotFoundError(
                f"Kein .OSV neben Proxy gefunden — Umwandlung braucht den Master: {path.name}"
            )
        if lrf is not None and ext == ".lrf":
            log.info("LRF gewählt → verwende OSV-Master %s", osv.name)
        return CaptureJob(
            kind="dji",
            master=normalize_path(osv),
            proxy=normalize_path(lrf) if lrf else None,
        )

    # Insta
    key = insta_capture_key(path)
    master = path if is_master_media(path) else None
    second = proxy = None
    if key is not None:
        for sib in folder.iterdir():
            if not sib.is_file() or _ext(sib) not in {".insv", ".lrv"}:
                continue
            if insta_capture_key(sib) != key:
                continue
            lens = insta_lens_code(sib) or "00"
            e = _ext(sib)
            if e == ".lrv" or lens == "11" or sib.name.upper().startswith("LRV_"):
                proxy = sib
            elif lens == "00" and e == ".insv":
                master = sib
            elif lens == "10" and e == ".insv":
                second = sib
            elif master is None and is_master_media(sib):
                master = sib
    elif is_master_media(path):
        master = path
    else:
        proxy = path

    if master is None or not is_master_media(master):
        raise FileNotFoundError(
            f"Kein .insv-Master neben Proxy gefunden — Umwandlung braucht den Master: {path.name}"
        )
    if ext in PROXY_EXTS or not is_master_media(path):
        log.info("Proxy gewählt → verwende INSV-Master %s", master.name)
    return CaptureJob(
        kind="insta",
        master=normalize_path(master),
        second=normalize_path(second) if second else None,
        proxy=normalize_path(proxy) if proxy else None,
    )
