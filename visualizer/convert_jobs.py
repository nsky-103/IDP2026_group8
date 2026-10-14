"""Background CSV → NPY conversion jobs (no Streamlit imports)."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import signal_io


@dataclass
class ConvertJob:
    """Shared status object updated from a worker thread."""

    key: str
    source: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0  # 0..1
    message: str = "Queued…"
    npy_path: str | None = None
    error: str | None = None


# Keep the registry on sys.modules so it survives accidental reload of this file.
_mod = sys.modules[__name__]
if not hasattr(_mod, "_jobs"):
    _mod._jobs = {}  # type: ignore[attr-defined]
    _mod._lock = threading.Lock()  # type: ignore[attr-defined]
_jobs: dict[str, ConvertJob] = _mod._jobs  # type: ignore[attr-defined]
_lock: threading.Lock = _mod._lock  # type: ignore[attr-defined]


def _job_key(path: Path) -> str:
    return str(path.resolve())


def get_job(path: Path | str) -> ConvertJob | None:
    key = _job_key(Path(path))
    with _lock:
        return _jobs.get(key)


def active_jobs() -> list[ConvertJob]:
    with _lock:
        return [j for j in _jobs.values() if j.status in ("queued", "running")]


def all_jobs() -> list[ConvertJob]:
    with _lock:
        return list(_jobs.values())


def clear_job(path: Path | str) -> None:
    key = _job_key(Path(path))
    with _lock:
        _jobs.pop(key, None)


def start_convert(path: Path) -> ConvertJob:
    """Start (or reuse) a background conversion for ``path``.

    Cached / already-``npy`` inputs finish almost immediately on the worker.
    """
    path = Path(path)
    key = _job_key(path)

    with _lock:
        existing = _jobs.get(key)
        if existing and existing.status in ("queued", "running"):
            return existing
        job = ConvertJob(key=key, source=str(path), status="queued", message="Queued…")
        _jobs[key] = job

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def _run_job(job: ConvertJob) -> None:
    path = Path(job.source)
    job.status = "running"
    job.message = "Starting…"
    job.progress = 0.0

    def on_progress(p: float, message: str | None = None) -> None:
        job.progress = float(min(max(p, 0.0), 1.0))
        if message:
            job.message = message

    try:
        if path.suffix.lower() == ".npy":
            job.npy_path = str(path)
            job.progress = 1.0
            job.message = "Ready (npy)"
            job.status = "done"
            return

        out = signal_io.npy_path_for(path)
        if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
            job.npy_path = str(out)
            job.progress = 1.0
            job.message = "Ready (cached npy)"
            job.status = "done"
            return

        npy = signal_io.convert_csv_to_npy(path, out, progress_callback=on_progress)
        job.npy_path = str(npy)
        job.progress = 1.0
        job.message = "Done"
        job.status = "done"
    except Exception as exc:
        job.error = str(exc)
        job.message = f"Failed: {exc}"
        job.status = "error"
