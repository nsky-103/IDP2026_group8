"""Left panel, top: import recordings (Browse… path or typed path) and pick a run.

Loading is coarse-first (``load_jobs``): rough preview → index → 1 kHz summary → Part A,
in a background thread. Progress updates via ``st.fragment`` so the whole page is not
redrawn every 0.4 s (GitHub #20, #21, #22).
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import streamlit as st

import load_jobs
import state
from style import section

_VISUALIZER = Path(__file__).resolve().parent.parent
_PROJECT = _VISUALIZER.parent
_SUCTION = Path("2025-10-29 Suction data for tumor size estimation") / "Suction data" / "Original"
_DATA_CANDIDATES = [
    _PROJECT / "idp2023" / "Test signals from previous years" / "Suction data" / "Original",
    _PROJECT.parent / _SUCTION,
    _PROJECT.parent.parent / _SUCTION,
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])
DEFAULT_CSV = DATA_DIR / "5. suction_HIGH_CONC_1ml_and_no_resuction.csv"
ACTIVE = ("queued", "preview", "indexing", "summarizing", "part_a")


def render() -> None:
    render_import()
    render_runs()


def resolve_data_path(raw: str) -> tuple[Path | None, list[Path]]:
    text = (raw or "").strip().strip('"').strip("'").removeprefix("@").strip()
    if not text:
        return None, []
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        tried = [candidate]
    else:
        tried = [(base / candidate).resolve() for base in (Path.cwd(), _PROJECT, _VISUALIZER)]
    return next((p for p in tried if p.is_file()), None), tried


def browse_csv_path() -> Path | None:
    """OS file dialog in a separate process; returns the chosen path (GitHub #20).

    Streamlit is not on the main thread, so in-process tkinter freezes on macOS.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            script = (
                'set f to choose file with prompt "Choose a Marginum CSV" '
                'of type {"csv", "public.comma-separated-values-text"}\n'
                "POSIX path of f"
            )
            out = subprocess.check_output(["osascript", "-e", script], text=True, timeout=120)
            path = Path(out.strip())
            return path if path.is_file() else None
        if system == "Windows":
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$d = New-Object System.Windows.Forms.OpenFileDialog; "
                "$d.Filter = 'CSV (*.csv)|*.csv'; "
                "if ($d.ShowDialog() -eq 'OK') { $d.FileName }"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps], text=True, timeout=120,
            )
            path = Path(out.strip())
            return path if path.is_file() else None
        # Linux / other: tiny helper process so tk runs on its own main thread
        helper = (
            "import tkinter as tk\n"
            "from tkinter import filedialog\n"
            "r = tk.Tk(); r.withdraw()\n"
            "p = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])\n"
            "print(p or '')\n"
        )
        out = subprocess.check_output(["python3", "-c", helper], text=True, timeout=120)
        path = Path(out.strip())
        return path if path.is_file() else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def render_import() -> None:
    section("Import")
    _load_progress_fragment()

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Browse…", width="stretch", help="Opens the OS file dialog; only the path is used (no file copy)."):
            chosen = browse_csv_path()
            if chosen is None:
                st.warning("No file chosen.")
            elif chosen.suffix.lower() != ".csv":
                st.error("Only `.csv` files are supported.")
            else:
                st.session_state["path_input"] = str(chosen)
                load_jobs.start_load(chosen)
                st.rerun()
    with c2:
        st.caption("Browse passes the path only — no multi-GB upload copy.")

    with st.expander("…or paste a local path", expanded=not state.runs()):
        path_text = st.text_input(
            "Path to CSV",
            value=st.session_state.get(
                "path_input",
                str(DEFAULT_CSV) if DEFAULT_CSV.exists() else "",
            ),
            key="path_input",
            help=("Absolute path, or relative to the project folder. "
                  "Load is coarse-first: preview → index → 1 kHz summary (cached)."),
        )
        if st.button("Load", type="primary", width="stretch"):
            path, tried = resolve_data_path(path_text)
            if path is None:
                st.error(f"File not found. Paste the full path, or a path relative to `{_PROJECT}`.")
                if tried:
                    st.caption("Tried:\n" + "\n".join(f"- `{p}`" for p in tried))
            elif path.suffix.lower() != ".csv":
                st.error("Only `.csv` files are supported (Method B uses fast_io).")
            else:
                st.info(
                    f"Loading `{path.name}` ({path.stat().st_size / 1024 ** 3:.2f} GB). "
                    "Rough overview first; 1 kHz summary continues in the background."
                )
                load_jobs.start_load(path)
                st.rerun()


@st.fragment(run_every=0.5)
def _load_progress_fragment() -> None:
    """Progress only — full app rerun at milestones (#22)."""
    need_rerun = False
    for job in load_jobs.all_jobs():
        source = Path(job.source)
        if job.status in ACTIVE:
            st.progress(
                min(max(job.progress, 0.0), 1.0),
                text=f"{source.name}: {job.message} ({job.progress:.0%})",
            )
            if job.has_preview:
                prev = state.runs().get(source.stem, {}).get("phase")
                phase = "ready" if job.status == "ready" else "preview"
                state.register_run(source, phase)
                # Index milestone: duration may jump — reset viewing window (#21).
                if job.milestone == "index":
                    state.reset_view()
                    job.milestone = None
                    need_rerun = True
                elif job.milestone == "preview":
                    job.milestone = None
                    need_rerun = True
                elif job.milestone == "summary":
                    # Summary ready enough to upgrade charts once; still may run Part A.
                    state.register_run(source, "ready" if job.status == "ready" else "preview")
                    job.milestone = None
                    need_rerun = True
                elif prev != phase:
                    need_rerun = True
        elif job.status == "ready":
            state.register_run(source, "ready")
            load_jobs.clear_job(source)
            st.toast(f"Loaded {source.name}", icon="✅")
            need_rerun = True
        elif job.status == "error":
            st.error(f"{source.name}: {job.error or job.message}")
            load_jobs.clear_job(source)
            need_rerun = True

    if any(j.status in ACTIVE for j in load_jobs.all_jobs()):
        st.caption(
            "Coarse-first load (progress updates here without redrawing the whole page)."
        )
    if need_rerun:
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
