"""Left panel, top: import recordings (upload or local path) and pick a run."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

import convert_jobs
import state
from style import section

_PROJECT = Path(__file__).resolve().parent.parent.parent
# Prefer local course data if present; otherwise the shared OneDrive-style folder.
_SUCTION = Path("2025-10-29 Suction data for tumor size estimation") / "Suction data" / "Original"
_DATA_CANDIDATES = [
    _PROJECT / "idp2023" / "Test signals from previous years" / "Suction data" / "Original",
    _PROJECT.parent / _SUCTION,          # next to the repository clone
    _PROJECT.parent.parent / _SUCTION,
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])
DEFAULT_CSV = DATA_DIR / "5. suction_HIGH_CONC_1ml_and_no_resuction.csv"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "marginum_uploads"


def render() -> None:
    render_import()
    render_runs()


def render_import() -> None:
    section("Import")
    _poll_convert_jobs()

    started = False
    uploads = st.file_uploader(
        "Drop CSV files here",
        type=["csv", "npy"],
        accept_multiple_files=True,
        help="Browser uploads are size-limited. For multi-GB files use a local path below.",
    )
    # The uploader keeps its files on every rerun, so remember which ones were added.
    added = st.session_state.setdefault("added_uploads", set())
    for file in uploads or []:
        if (file.name, file.size) in added:
            continue
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved = UPLOAD_DIR / file.name
        saved.write_bytes(file.getbuffer())
        convert_jobs.start_convert(saved)
        added.add((file.name, file.size))
        started = True

    with st.expander("…or load from a local path", expanded=not state.runs()):
        path_text = st.text_input(
            "Path to CSV or NPY",
            value=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else "",
        ).strip()
        if st.button("Load", type="primary", use_container_width=True):
            if path_text and Path(path_text).exists():
                convert_jobs.start_convert(Path(path_text))
                started = True
            else:
                st.error("File not found. Check the path.")

    if started:
        st.rerun()  # start polling the new job


def _poll_convert_jobs() -> None:
    """Show progress of background CSV -> NPY jobs; register runs when they finish."""
    jobs = convert_jobs.all_jobs()
    running = finished = False
    for job in jobs:
        source = Path(job.source)
        if job.status in ("queued", "running"):
            running = True
            st.progress(min(max(job.progress, 0.0), 1.0),
                        text=f"{source.name}: {job.message} ({job.progress:.0%})")
        elif job.status == "done" and job.npy_path:
            state.register_run(source, Path(job.npy_path))
            convert_jobs.clear_job(source)
            st.toast(f"Loaded {source.name}", icon="✅")
            finished = True
        elif job.status == "error":
            st.error(f"{source.name}: {job.error or job.message}")
            convert_jobs.clear_job(source)
            finished = True

    if running:
        st.caption("Conversion runs in the background — you can keep using the UI.")
        time.sleep(0.4)
        st.rerun()
    elif finished:
        st.rerun()


def render_runs() -> None:
    runs = state.runs()
    section("Runs", f"{len(runs)} loaded")
    if not runs:
        st.caption("No runs loaded yet.")
        return

    names = list(runs)
    st.radio(
        "Runs",
        options=names,
        captions=[
            f"{len(state.accepted_events(n))} events" if state.detection(n) else "" for n in names
        ],
        key="active_run",
        on_change=state.reset_view,
        label_visibility="collapsed",
    )
