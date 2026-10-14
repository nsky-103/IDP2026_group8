"""Left panel, top: import recordings (upload or local path) and pick a run.

Loading is coarse-first (``load_jobs``): row index -> rough preview -> 1 kHz summary,
in a background thread. A run appears as soon as its preview exists ("preview…") and
becomes "ready" when the summary is done.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

import load_jobs
import state
from style import section

_VISUALIZER = Path(__file__).resolve().parent.parent
_PROJECT = _VISUALIZER.parent
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
ACTIVE = ("queued", "indexing", "preview", "summarizing")


def render() -> None:
    render_import()
    render_runs()


def resolve_data_path(raw: str) -> tuple[Path | None, list[Path]]:
    """Path typed by the user -> existing file (or None) and the places tried.

    Accepts quotes, a leading ``@`` (editor-style reference), and paths relative to the
    working directory, the project folder or this folder.
    """
    text = (raw or "").strip().strip('"').strip("'").removeprefix("@").strip()
    if not text:
        return None, []
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        tried = [candidate]
    else:
        tried = [(base / candidate).resolve() for base in (Path.cwd(), _PROJECT, _VISUALIZER)]
    return next((p for p in tried if p.is_file()), None), tried


def render_import() -> None:
    section("Import")
    _poll_load_jobs()

    started = False
    uploads = st.file_uploader(
        "Drop CSV files here",
        type=["csv"],
        accept_multiple_files=True,
        help=("Upload limit is 16 GB (.streamlit/config.toml). For multi-GB samples (~8 GB), "
              "prefer a local path below — browser upload is slow and memory-heavy."),
    )
    # The uploader keeps its files on every rerun, so remember which ones were added.
    added = st.session_state.setdefault("added_uploads", set())
    for file in uploads or []:
        if (file.name, file.size) in added:
            continue
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved = UPLOAD_DIR / file.name
        saved.write_bytes(file.getbuffer())
        load_jobs.start_load(saved)
        added.add((file.name, file.size))
        started = True

    with st.expander("…or load from a local path", expanded=not state.runs()):
        path_text = st.text_input(
            "Path to CSV",
            value=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else "",
            help=("Absolute path, or relative to the project folder. `@folder/...` prefixes are OK. "
                  "Load is coarse-first: rough overview, then 1 kHz summary (cached)."),
        )
        if st.button("Load", type="primary", use_container_width=True):
            path, tried = resolve_data_path(path_text)
            if path is None:
                st.error(f"File not found. Paste the full path, or a path relative to `{_PROJECT}`.")
                if tried:
                    st.caption("Tried:\n" + "\n".join(f"- `{p}`" for p in tried))
            elif path.suffix.lower() != ".csv":
                st.error("Only `.csv` files are supported (Method B uses fast_io).")
            else:
                st.info(f"Loading `{path.name}` ({path.stat().st_size / 1024 ** 3:.2f} GB). "
                        "Rough overview appears first; 1 kHz summary continues in the background.")
                load_jobs.start_load(path)
                started = True

    if started:
        st.rerun()  # start polling the new job


def _poll_load_jobs() -> None:
    """Show progress of background loads; add runs at preview, mark them ready at the end."""
    active = changed = False
    for job in load_jobs.all_jobs():
        source = Path(job.source)
        if job.status in ACTIVE:
            active = True
            st.progress(min(max(job.progress, 0.0), 1.0),
                        text=f"{source.name}: {job.message} ({job.progress:.0%})")
            if job.has_preview:
                state.register_run(source, "preview")
        elif job.status == "ready":
            state.register_run(source, "ready")
            load_jobs.clear_job(source)
            st.toast(f"Loaded {source.name}", icon="✅")
            changed = True
        elif job.status == "error":
            st.error(f"{source.name}: {job.error or job.message}")
            load_jobs.clear_job(source)
            changed = True

    if active:
        st.caption("Coarse-first load: rough overview first, then 1 kHz summary in the "
                   "background. You can keep using the UI.")
        time.sleep(0.4)
        st.rerun()
    elif changed:
        st.rerun()


def render_runs() -> None:
    runs = state.runs()
    section("Runs", f"{len(runs)} loaded")
    if not runs:
        st.caption("No runs loaded yet.")
        return

    def caption(name: str) -> str:
        tag = "ready" if state.is_ready(name) else "preview…"
        n_events = len(state.accepted_events(name)) if state.detection(name) else 0
        return tag + (f" · {n_events} events" if n_events else "")

    names = list(runs)
    st.radio(
        "Runs",
        options=names,
        captions=[caption(n) for n in names],
        key="active_run",
        on_change=state.reset_view,
        label_visibility="collapsed",
    )
