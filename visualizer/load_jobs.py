"""Background CSV load: preview → index → 1 kHz summary → Part A cache (no Streamlit)."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_mod = sys.modules[__name__]
if not hasattr(_mod, "_jobs"):
    _mod._jobs = {}  # type: ignore[attr-defined]
    _mod._lock = threading.Lock()  # type: ignore[attr-defined]
    _mod._recordings = {}  # type: ignore[attr-defined]
_jobs: dict[str, "LoadJob"] = _mod._jobs  # type: ignore[attr-defined]
_lock: threading.Lock = _mod._lock  # type: ignore[attr-defined]
_recordings: dict[str, Any] = _mod._recordings  # type: ignore[attr-defined]


@dataclass
class LoadJob:
    key: str
    source: str
    # queued | preview | indexing | summarizing | part_a | ready | error
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued…"
    error: str | None = None
    from_cache: bool = False
    has_preview: bool = False
    # Bumped when UI should full-rerun (preview / index / ready / error).
    milestone: str | None = None


def _job_key(path: Path) -> str:
    return str(path.resolve())


def get_job(path: Path | str) -> LoadJob | None:
    key = _job_key(Path(path))
    with _lock:
        return _jobs.get(key)


def all_jobs() -> list[LoadJob]:
    with _lock:
        return list(_jobs.values())


def active_jobs() -> list[LoadJob]:
    with _lock:
        return [
            j for j in _jobs.values()
            if j.status in ("queued", "preview", "indexing", "summarizing", "part_a")
        ]


def clear_job(path: Path | str) -> None:
    key = _job_key(Path(path))
    with _lock:
        _jobs.pop(key, None)


def get_recording(path: Path | str):
    key = _job_key(Path(path))
    with _lock:
        return _recordings.get(key)


def set_recording(path: Path | str, recording) -> None:
    key = _job_key(Path(path))
    with _lock:
        _recordings[key] = recording


def start_load(path: Path) -> LoadJob:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Method B load expects a .csv file, got {path.suffix}")

    key = _job_key(path)
    with _lock:
        existing = _jobs.get(key)
        if existing and existing.status in (
            "queued", "preview", "indexing", "summarizing", "part_a",
        ):
            return existing
        if key in _recordings and _recordings[key].phase == "ready":
            job = LoadJob(
                key=key, source=str(path), status="ready", progress=1.0,
                message="Ready (cached in memory)", has_preview=True, from_cache=True,
                milestone="ready",
            )
            _jobs[key] = job
            return job
        job = LoadJob(key=key, source=str(path), status="queued", message="Queued…")
        _jobs[key] = job

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def _core_imports():
    import signal_io
    signal_io._ensure_core_on_path()
    from core import fast_io, pipeline, preprocessing
    return fast_io, pipeline, preprocessing


def _publish(job: LoadJob, rec, *, status: str, progress: float, message: str,
             milestone: str | None = None) -> None:
    set_recording(Path(job.source), rec)
    job.status = status
    job.progress = progress
    job.message = message
    if milestone:
        job.milestone = milestone
        job.has_preview = True


def _run_job(job: LoadJob) -> None:
    import signal_io

    csv_path = Path(job.source)
    try:
        fast_io, pipeline, preprocessing = _core_imports()
        cache_root = pipeline.DEFAULT_CACHE
        folder = pipeline._cache_dir(csv_path, cache_root) if cache_root else None

        hit = pipeline._load(folder, csv_path) if folder else None
        if hit is not None:
            index, summary = hit
            clean = pipeline.load_part_a(folder) if folder else None
            if clean is None:
                clean = preprocessing.preprocess(summary["mean"], fs=float(summary["fs_hz"]))
                if folder:
                    pipeline.save_part_a(folder, clean)
            rec = signal_io.Recording(
                path=csv_path, index=index, summary=summary, preview=None,
                phase="ready", from_cache=True, clean=clean,
            )
            job.from_cache = True
            _publish(job, rec, status="ready", progress=1.0,
                     message="Ready (disk cache)", milestone="ready")
            return

        # #21: rough preview first (no index) so the UI can paint in ~1 s.
        job.status = "preview"
        job.message = "Rough preview…"
        job.progress = 0.05
        preview = fast_io.preview(csv_path)
        est_rows = int(float(preview["t"][-1]) * fast_io.DEFAULT_FS_HZ) if len(preview["t"]) else None
        rec = signal_io.Recording(
            path=csv_path, index=None, summary=None, preview=preview,
            phase="preview", estimated_n_rows=est_rows,
        )
        _publish(job, rec, status="preview", progress=0.12,
                 message="Building index…", milestone="preview")

        job.status = "indexing"
        index = fast_io.build_index(csv_path)
        rec = signal_io.Recording(
            path=csv_path, index=index, summary=None, preview=preview,
            phase="preview",
        )
        # Duration may jump from estimate → exact; UI should reset the window (#21).
        _publish(job, rec, status="indexing", progress=0.20,
                 message="Building 1 kHz summary…", milestone="index")

        job.status = "summarizing"

        def on_progress(fraction: float) -> None:
            job.progress = 0.20 + 0.65 * float(fraction)
            job.message = f"Building 1 kHz summary… {fraction:.0%}"

        def on_partial(partial: dict) -> None:
            # Copy arrays so the worker can keep filling the live buffers.
            snap = {
                "mean": np.array(partial["mean"], copy=True),
                "min": np.array(partial["min"], copy=True),
                "max": np.array(partial["max"], copy=True),
                "fs_hz": float(partial["fs_hz"]),
            }
            cur = signal_io.Recording(
                path=csv_path, index=index, summary=snap, preview=preview,
                phase="preview",
            )
            set_recording(csv_path, cur)

        summary = fast_io.summarize(
            index,
            progress_callback=on_progress,
            partial_callback=on_partial,
        )
        if folder:
            pipeline._save(folder, index, summary)

        rec = signal_io.Recording(
            path=csv_path, index=index, summary=summary, preview=preview,
            phase="ready",
        )
        _publish(job, rec, status="part_a", progress=0.90,
                 message="Running Part A (clean)…", milestone="summary")

        # #17: Part A in the background once summary is ready.
        clean = preprocessing.preprocess(summary["mean"], fs=float(summary["fs_hz"]))
        if folder:
            pipeline.save_part_a(folder, clean)
        rec = signal_io.Recording(
            path=csv_path, index=index, summary=summary, preview=preview,
            phase="ready", clean=clean,
        )
        _publish(job, rec, status="ready", progress=1.0,
                 message="Ready", milestone="ready")
    except Exception as exc:
        job.error = str(exc)
        job.message = f"Failed: {exc}"
        job.status = "error"
        job.milestone = "error"
