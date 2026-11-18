from __future__ import annotations

import logging
from pathlib import Path

from toshot.source.jobs import (
    CaptureJob,
    discover_jobs,
    group_jobs,
    job_for_path,
    list_media_files,
    normalize_path,
)
from toshot.source.models import Capture

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".osv", ".lrf", ".insv", ".lrv"}


def resolve_path(path: str | Path) -> Path:
    return normalize_path(path)


def discover_files(folder: Path) -> list[Path]:
    return list_media_files(folder)


def discover_captures(folder: Path) -> list[Capture]:
    """Load every capture in the folder (slow on NAS). Prefer discover_jobs + lazy load."""
    jobs = discover_jobs(folder)
    out: list[Capture] = []
    for i, job in enumerate(jobs, 1):
        log.info("Parse %d/%d: %s", i, len(jobs), job.master.name)
        out.append(job.load())
    return out


def group_files(files: list[Path]) -> list[Capture]:
    jobs = group_jobs(files)
    return [j.load() for j in jobs]


def load_capture(path: Path) -> Capture:
    """Load one capture; only looks up sibling proxy/pair by name — not the whole folder."""
    job = job_for_path(path)
    log.info(
        "Einzel-Capture: %s (proxy=%s)",
        job.master,
        job.proxy.name if job.proxy else "-",
    )
    return job.load()


__all__ = [
    "CaptureJob",
    "SUPPORTED_EXTS",
    "resolve_path",
    "discover_files",
    "discover_captures",
    "discover_jobs",
    "group_files",
    "group_jobs",
    "job_for_path",
    "load_capture",
]
