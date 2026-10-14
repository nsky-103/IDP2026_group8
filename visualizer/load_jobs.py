"""Background CSV load: index → rough preview → 1 kHz summary cache (no Streamlit)."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keep the registry on sys.modules so it survives accidental reload of this file.
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
    """Shared status object updated from a worker thread."""

    key: str
    source: str
    status: str = "queued"  # queued | indexing | preview | summarizing | ready | error
    progress: float = 0.0  # 0..1
    message: str = "Queued…"
    error: str | None = None
    from_cache: bool = False
    # Set when a rough overview is available (status may still be summarizing).
    has_preview: bool = False


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
            j
            for j in _jobs.values()
            if j.status in ("queued", "indexing", "preview", "summarizing")
        ]


def clear_job(path: Path | str) -> None:
    key = _job_key(Path(path))
    with _lock:
        _jobs.pop(key, None)


def get_recording(path: Path | str):
    """Return the ``Recording`` published by the worker, if any."""
    key = _job_key(Path(path))
    with _lock:
        return _recordings.get(key)


def set_recording(path: Path | str, recording) -> None:
    key = _job_key(Path(path))
    with _lock:
        _recordings[key] = recording


def start_load(path: Path) -> LoadJob:
    """Start (or reuse) a background coarse-first load for a CSV path."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Method B load expects a .csv file, got {path.suffix}")

    key = _job_key(path)
    with _lock:
        existing = _jobs.get(key)
        if existing and existing.status in (
            "queued",
            "indexing",
            "preview",
            "summarizing",
        ):
            return existing
        # Already ready in this process — return a done job without re-work.
        if key in _recordings and _recordings[key].phase == "ready":
            job = LoadJob(
                key=key,
                source=str(path),
                status="ready",
                progress=1.0,
                message="Ready (cached in memory)",
                has_preview=True,
                from_cache=True,
            )
            _jobs[key] = job
            return job

        job = LoadJob(key=key, source=str(path), status="queued", message="Queued…")
        _jobs[key] = job

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def _core_imports():
    """Import core (works from IDP2026_group8/visualizer or sibling clone)."""
    import signal_io

    signal_io._ensure_core_on_path()
    from core import fast_io, pipeline

    return fast_io, pipeline


def _run_job(job: LoadJob) -> None:
    import signal_io

    csv_path = Path(job.source)
    job.status = "indexing"
    job.message = "Building index…"
    job.progress = 0.02

    try:
        fast_io, pipeline = _core_imports()
        cache_root = pipeline.DEFAULT_CACHE
        folder = pipeline._cache_dir(csv_path, cache_root) if cache_root else None

        # Fast path: disk cache already has index + 1 kHz summary.
        hit = pipeline._load(folder, csv_path) if folder else None
        if hit is not None:
            index, summary = hit
            rec = signal_io.Recording(
                path=csv_path,
                index=index,
                summary=summary,
                preview=None,
                phase="ready",
                from_cache=True,
            )
            set_recording(csv_path, rec)
            job.from_cache = True
            job.has_preview = True
            job.progress = 1.0
            job.message = "Ready (disk cache)"
            job.status = "ready"
            return

        index = fast_io.build_index(csv_path)
        job.progress = 0.12
        job.message = "Rough preview…"

        preview = fast_io.preview(csv_path, index)
        rec = signal_io.Recording(
            path=csv_path,
            index=index,
            summary=None,
            preview=preview,
            phase="preview",
            from_cache=False,
        )
        set_recording(csv_path, rec)
        job.has_preview = True
        job.status = "preview"
        job.progress = 0.18
        job.message = "Building 1 kHz summary…"

        job.status = "summarizing"

        def on_progress(fraction: float) -> None:
            job.progress = 0.18 + 0.80 * float(fraction)
            job.message = f"Building 1 kHz summary… {fraction:.0%}"

        summary = fast_io.summarize(index, progress_callback=on_progress)
        if folder:
            pipeline._save(folder, index, summary)

        rec = signal_io.Recording(
            path=csv_path,
            index=index,
            summary=summary,
            preview=preview,
            phase="ready",
            from_cache=False,
        )
        set_recording(csv_path, rec)
        job.progress = 1.0
        job.message = "Ready"
        job.status = "ready"
    except Exception as exc:
        job.error = str(exc)
        job.message = f"Failed: {exc}"
        job.status = "error"
