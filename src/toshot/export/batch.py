from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from toshot.export.equirect import export_equirect
from toshot.export.shot import export_shot
from toshot.ffmpeg_bin import EncodeSettings
from toshot.parallel import batch_workers, set_stitch_share
from toshot.source.jobs import CaptureJob, discover_jobs
from toshot.source.models import StabMode
from toshot.stitch.options import StitchOptions

log = logging.getLogger(__name__)


def _process_one(
    i: int,
    total: int,
    job: CaptureJob,
    out_dir: Path,
    *,
    do_shot: bool,
    do_eq: bool,
    stab: StabMode,
    fov: float,
    class_name: str,
    model_name: str,
    equirect_width: int | None,
    opts: StitchOptions,
    encode: EncodeSettings | None,
) -> list[Path]:
    log_path = out_dir / f"{job.id}.log"
    log.info("===== [%d/%d] %s =====", i, total, job.master)
    written: list[Path] = []
    try:
        cap = job.load()
        messages = [
            f"id={cap.id}",
            f"family={cap.family.value}",
            f"layout={cap.layout.value}",
            f"master={[str(p) for p in cap.master_paths]}",
            f"proxy={cap.proxy_path}",
            f"fps={cap.fps}",
            f"duration={cap.duration_s}",
            f"frames={cap.estimated_frames()}",
            f"equirect={cap.equirect_size()}",
            f"attitude_samples={len(cap.attitude)}",
            f"optical_flow={opts.optical_flow}",
            f"selfie_mask={opts.selfie_mask}",
            f"rolling_shutter={opts.rolling_shutter}",
            *cap.notes,
        ]
        if do_eq:
            eq_out = out_dir / f"{cap.id}_equirect.mp4"
            log.info("Equirect → %s", eq_out)
            export_equirect(
                cap, eq_out, stab=stab, width=equirect_width, options=opts, encode=encode
            )
            written.append(eq_out)
            messages.append(f"equirect={eq_out}")
        if do_shot:
            shot_out = out_dir / f"{cap.id}_shot.mp4"
            log.info("Shot → %s", shot_out)
            export_shot(
                cap,
                shot_out,
                stab=stab,
                fov=fov,
                class_name=class_name,
                model_name=model_name,
                options=opts,
                encode=encode,
            )
            written.append(shot_out)
            messages.append(f"shot={shot_out}")
        log_path.write_text("\n".join(messages) + "\n", encoding="utf-8")
        log.info("Fertig %s", cap.id)
    except Exception:
        err = traceback.format_exc()
        log.error("Failed on %s:\n%s", job.id, err)
        log_path.write_text(err, encoding="utf-8")
    return written


def run_batch(
    inbox: Path,
    out_dir: Path,
    *,
    mode: str = "shot",
    stab: StabMode = StabMode.WORLD,
    fov: float = 90.0,
    class_name: str = "person",
    model_name: str = "yolo11n.pt",
    equirect_width: int | None = None,
    options: StitchOptions | None = None,
    encode: EncodeSettings | None = None,
    workers: int | None = None,
) -> list[Path]:
    """mode: 'shot' | 'equirect' | 'both'. Parallel jobs = all CPUs by default."""
    inbox = Path(inbox)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = options or StitchOptions()
    jobs = discover_jobs(inbox)
    n_workers = max(1, workers if workers is not None else batch_workers())
    n_workers = min(n_workers, max(1, len(jobs)))
    log.info("Batch: %d Job(s) in %s · %d parallele Jobs", len(jobs), inbox, n_workers)
    written: list[Path] = []

    do_shot = mode in ("shot", "both")
    do_eq = mode in ("equirect", "both")
    total = len(jobs)

    set_stitch_share(n_workers)
    try:
        if n_workers == 1 or total <= 1:
            for i, job in enumerate(jobs, 1):
                written.extend(
                    _process_one(
                        i,
                        total,
                        job,
                        out_dir,
                        do_shot=do_shot,
                        do_eq=do_eq,
                        stab=stab,
                        fov=fov,
                        class_name=class_name,
                        model_name=model_name,
                        equirect_width=equirect_width,
                        opts=opts,
                        encode=encode,
                    )
                )
            return written

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [
                pool.submit(
                    _process_one,
                    i,
                    total,
                    job,
                    out_dir,
                    do_shot=do_shot,
                    do_eq=do_eq,
                    stab=stab,
                    fov=fov,
                    class_name=class_name,
                    model_name=model_name,
                    equirect_width=equirect_width,
                    opts=opts,
                    encode=encode,
                )
                for i, job in enumerate(jobs, 1)
            ]
            for fut in as_completed(futs):
                written.extend(fut.result())
    finally:
        set_stitch_share(1)
    return written
