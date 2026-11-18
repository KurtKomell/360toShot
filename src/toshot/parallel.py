from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)

_stitch_share = 1
_share_lock = threading.Lock()


def cpu_workers(cap: int | None = None) -> int:
    """Logical CPU count (override via TOSHOT_WORKERS)."""
    env = os.environ.get("TOSHOT_WORKERS", "").strip()
    if env:
        try:
            n = max(1, int(env))
            return min(n, cap) if cap else n
        except ValueError:
            pass
    n = max(1, os.cpu_count() or 4)
    return min(n, cap) if cap else n


def set_stitch_share(n: int) -> None:
    """When N batch jobs run at once, each stitch uses CPUs/N threads."""
    global _stitch_share
    with _share_lock:
        _stitch_share = max(1, int(n))


def batch_workers() -> int:
    """Parallel capture jobs.

    Default: min(8, CPUs) so NAS/NVENC are not flooded with dozens of 8K
    decodes. Override with TOSHOT_BATCH_WORKERS (e.g. 24 for max).
    Frame stitch still uses all remaining cores via stitch_workers().
    """
    env = os.environ.get("TOSHOT_BATCH_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(8, cpu_workers()))


def stitch_workers() -> int:
    """Frame-stitch threads — all CPUs, divided across active batch jobs."""
    env = os.environ.get("TOSHOT_STITCH_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    with _share_lock:
        share = _stitch_share
    return max(1, cpu_workers() // share)
