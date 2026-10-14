"""
Marginum signal visualizer (Streamlit app).

Run with:  streamlit run visualizer/app.py

Page layout
-----------
    header   title · file info
    left     import · runs · channels · detection settings
    center   full-recording overview · detail zoom
    right    selected event · events list

Files
-----
    app.py         this file: page layout only (all Streamlit code lives here)
    processing.py  detection pipeline wrappers (no Streamlit)
    plots.py       Plotly figures (no Streamlit)
    signal_io.py   CSV → NPY conversion + loading a time window
    convert_jobs.py background CSV→NPY worker (keeps the UI responsive)
    lttb.py        LTTB downsampling for plots

How to add a feature
--------------------
    1. New input?      add a widget in render_left_panel() and a field in Settings.
    2. New chart?      add a function in plots.py that returns a go.Figure.
    3. New calculation? add a function in processing.py.
    4. Show it:        write a render_xxx() function and call it from main().
                       The order of calls in main() is the order on the page.

Streamlit reruns this whole file on every click, so anything that must be
remembered between clicks is stored in st.session_state:
    runs          {run name: {"source": csv path, "npy": npy path}}
    active_run    name of the selected run
    detections    {run name: detection result}
    window        (start_s, end_s) viewing range (overview box-drag or range bar)
    selected_event  event ID, e.g. "E3"
"""

from __future__ import annotations

import html
import importlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import convert_jobs
import lttb
import plots
import processing
import signal_io

# Reload UI/plot helpers so edits show up without restarting Streamlit.
# Do NOT reload convert_jobs: that would wipe the in-memory job registry
# while a background conversion thread is still running.
importlib.reload(lttb)
importlib.reload(signal_io)
importlib.reload(processing)
importlib.reload(plots)


# Prefer local course data if present; otherwise the friend's OneDrive-style path.
_PROJECT = Path(__file__).resolve().parent.parent
_DATA_CANDIDATES = [
    _PROJECT
    / "idp2023"
    / "Test signals from previous years"
    / "Suction data"
    / "Original",
    _PROJECT.parent.parent
    / "2025-10-29 Suction data for tumor size estimation"
    / "Suction data"
    / "Original",
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])
DEFAULT_CSV = DATA_DIR / "5. suction_HIGH_CONC_1ml_and_no_resuction.csv"
DEFAULT_BLANK = (
    _PROJECT / "idp2023" / "Test signals from previous years" / "just water.csv"
)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "marginum_uploads"

DETECTION_ALGORITHMS = ["Method A · Threshold"]  # add new algorithms here
MAX_EVENTS_LISTED = 25
DEFAULT_WINDOW_S = 30.0
EVENT_ZOOM_S = 5.0  # "Show in plot" shows ± this many seconds around the event

CSS = """
<style>
.block-container { padding-top: 3.2rem; max-width: 100%; }
.sv-title { font-size: 1.25rem; font-weight: 700; }
.sv-chip {
    float: right; font-family: monospace; font-size: 0.8rem;
    padding: 0.3rem 0.7rem; border: 1px solid rgba(128,128,128,0.35); border-radius: 6px;
}
.sv-section {
    display: flex; justify-content: space-between; margin: 0.9rem 0 0.3rem;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.65;
}
.sv-card { background: #18222d; color: #eef2f6; border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem; }
.sv-card-label { font-size: 0.75rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em; }
.sv-card-value { font-size: 1.8rem; font-weight: 700; }
.sv-card-note { font-size: 0.75rem; opacity: 0.65; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str, note: str = "") -> None:
    """Small uppercase section heading with an optional note on the right."""
    st.markdown(
        f'<div class="sv-section"><span>{html.escape(title)}</span>'
        f"<span>{html.escape(note)}</span></div>",
        unsafe_allow_html=True,
    )


def fmt_time(seconds: float) -> str:
    """65.2 → '01:05.20'"""
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - 60 * minutes:05.2f}"


def get_runs() -> dict:
    return st.session_state.setdefault("runs", {})


def get_detections() -> dict:
    return st.session_state.setdefault("detections", {})


def get_events(run_name: str) -> list[tuple[str, dict]]:
    """Accepted events of a run as (ID, event), numbered in time order: E1, E2, …"""
    detection = get_detections().get(run_name)
    if detection is None:
        return []
    accepted = [e for e in detection["events"] if e["valid"]]
    accepted.sort(key=lambda e: e["t_peak"])
    return [(f"E{i + 1}", e) for i, e in enumerate(accepted)]


def reset_view() -> None:
    """Forget the time range and selected event (e.g. when switching runs)."""
    st.session_state.pop("window", None)
    st.session_state.pop("selected_event", None)
    st.session_state.pop("overview_range_slider", None)
    st.session_state.pop("_overview_sel_sig", None)
    st.session_state.pop("_pending_slider_from_window", None)
    st.session_state.pop("_overview_slider_live", None)


def select_event(event_id: str, t_peak: float, total_s: float) -> None:
    """Button callback: select an event and zoom the time range to it."""
    st.session_state["selected_event"] = event_id
    _set_window(
        max(0.0, t_peak - EVENT_ZOOM_S),
        min(total_s, t_peak + EVENT_ZOOM_S),
        total_s,
        bump_overview=True,
    )


# ---------------------------------------------------------------------------
# Left panel
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """Everything the user chose in the left panel."""

    channels: list[str]
    algorithm: str
    sigma_mult: float
    channel_ratio: float
    show_rejected: bool
    blank_path: str
    fs_hz: float
    max_points: int
    downsample_method: str
    detect_clicked: bool


def _register_run(source: Path, npy: Path) -> None:
    """Add a finished conversion to the runs list and select it."""
    name = source.stem
    get_runs()[name] = {"source": str(source), "npy": str(npy)}
    get_detections().pop(name, None)
    st.session_state["active_run"] = name
    reset_view()


def queue_convert(path: Path) -> None:
    """Kick off background CSV→NPY (or instant npy/cache). Does not block the UI."""
    convert_jobs.start_convert(Path(path))


def poll_convert_jobs() -> None:
    """Show progress for background jobs; register runs when they finish."""
    jobs = convert_jobs.all_jobs()
    if not jobs:
        return

    any_running = False
    just_finished = False
    for job in jobs:
        source = Path(job.source)
        if job.status in ("queued", "running"):
            any_running = True
            st.progress(
                min(max(job.progress, 0.0), 1.0),
                text=f"{source.name}: {job.message} ({job.progress:.0%})",
            )
        elif job.status == "done" and job.npy_path:
            _register_run(source, Path(job.npy_path))
            convert_jobs.clear_job(source)
            st.toast(f"Loaded {source.name}", icon="✅")
            just_finished = True
        elif job.status == "error":
            st.error(f"{source.name}: {job.error or job.message}")
            convert_jobs.clear_job(source)
            just_finished = True

    if any_running:
        st.caption("Conversion runs in the background — you can keep using the UI.")
        time.sleep(0.4)
        st.rerun()
    elif just_finished:
        st.rerun()


def render_import() -> None:
    section("Import")
    poll_convert_jobs()

    uploads = st.file_uploader(
        "Drop CSV files here",
        type=["csv", "npy"],
        accept_multiple_files=True,
        help="Browser uploads are size-limited. For multi-GB files use a local path below.",
    )
    # The uploader keeps its files on every rerun, so remember which ones we already added.
    added = st.session_state.setdefault("added_uploads", set())
    started = False
    for file in uploads or []:
        if (file.name, file.size) in added:
            continue
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = UPLOAD_DIR / file.name
        saved_path.write_bytes(file.getbuffer())
        queue_convert(saved_path)
        added.add((file.name, file.size))
        started = True

    with st.expander("…or load from a local path", expanded=not get_runs()):
        path_text = st.text_input(
            "Path to CSV or NPY",
            value=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else "",
        ).strip()
        if st.button("Load", type="primary", use_container_width=True):
            if path_text and Path(path_text).exists():
                queue_convert(Path(path_text))
                started = True
            else:
                st.error("File not found. Check the path.")

    # Load click runs after poll_convert_jobs(); force a rerun so polling starts.
    if started:
        st.rerun()


def render_runs() -> None:
    runs = get_runs()
    section("Runs", f"{len(runs)} loaded")
    if not runs:
        st.caption("No runs loaded yet.")
        return

    names = list(runs)
    captions = [
        f"{len(get_events(name))} events" if name in get_detections() else ""
        for name in names
    ]
    st.radio(
        "Runs",
        options=names,
        captions=captions,
        key="active_run",
        on_change=reset_view,
        label_visibility="collapsed",
    )


def render_left_panel() -> Settings:
    render_import()
    render_runs()

    section("Channels")
    channels = []
    if st.checkbox(f"Ch1 · {plots.SENSOR1}", value=True):
        channels.append(plots.SENSOR1)
    if st.checkbox(f"Ch2 · {plots.SENSOR2}", value=True):
        channels.append(plots.SENSOR2)

    section("Detection")
    algorithm = st.selectbox("Algorithm", DETECTION_ALGORITHMS)
    with st.expander("Detection parameters"):
        sigma_mult = st.slider("Noise σ multiplier", 3.0, 20.0, 5.0, 0.5)
        channel_ratio = st.slider("Min adc2/adc1 ratio", 0.5, 3.0, 1.3, 0.1)
        show_rejected = st.checkbox("Show rejected candidates (gray)", value=True)
        blank_path = st.text_input(
            "Blank (just water) CSV — optional noise floor",
            value=str(DEFAULT_BLANK) if DEFAULT_BLANK.exists() else "",
        )
    detect_clicked = st.button(
        "Detect peaks",
        type="primary",
        disabled=not get_runs(),
        use_container_width=True,
    )

    with st.expander("Display settings"):
        fs_hz = st.number_input(
            "Sample rate (Hz)", value=float(signal_io.DEFAULT_FS_HZ), min_value=1.0, step=1000.0
        )
        max_points = st.slider("Max plot points", 2_000, 100_000, 20_000, 1_000)
        downsample_method = st.selectbox(
            "Plot downsample",
            ["lttb", "stride"],
            format_func=lambda m: "LTTB (keeps peaks)" if m == "lttb" else "Every-nth sample",
        )

    return Settings(
        channels=channels,
        algorithm=algorithm,
        sigma_mult=sigma_mult,
        channel_ratio=channel_ratio,
        show_rejected=show_rejected,
        blank_path=blank_path,
        fs_hz=fs_hz,
        max_points=max_points,
        downsample_method=downsample_method,
        detect_clicked=detect_clicked,
    )


# ---------------------------------------------------------------------------
# Center column
# ---------------------------------------------------------------------------

def run_detection(run_name: str, settings: Settings) -> None:
    """Run peak detection on a run and store the result."""
    with st.spinner("Running peak detection…"):
        try:
            get_detections()[run_name] = processing.run_detection(
                get_runs()[run_name]["source"],
                settings.fs_hz,
                settings.blank_path,
                settings.sigma_mult,
                settings.channel_ratio,
            )
            st.session_state.pop("selected_event", None)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            return
    st.rerun()  # redraw the page so the left panel shows the new event count


def _ensure_window(total_s: float) -> tuple[float, float]:
    window = st.session_state.get("window")
    if window is None or window[1] > total_s or window[0] < 0:
        st.session_state["window"] = (0.0, min(float(total_s), DEFAULT_WINDOW_S))
    return st.session_state["window"]


def _set_window(
    start_s: float,
    end_s: float,
    total_s: float,
    *,
    bump_overview: bool = False,
    sync_slider: bool = True,
) -> None:
    """Store viewing window; optionally sync the range bar and refresh the overview chart.

    Streamlit forbids writing a widget key after that widget is instantiated in the
    same run. Zoom/pan buttons run after the overview slider, so in that case we
    defer the slider sync to the next run via ``_pending_slider_from_window``.
    """
    a = max(0.0, min(float(start_s), float(total_s)))
    b = max(0.0, min(float(end_s), float(total_s)))
    if b < a:
        a, b = b, a
    if b - a < 0.5:
        b = min(float(total_s), a + 0.5)
    st.session_state["window"] = (float(a), float(b))
    if sync_slider:
        if st.session_state.get("_overview_slider_live"):
            st.session_state["_pending_slider_from_window"] = True
        else:
            st.session_state["overview_range_slider"] = (float(a), float(b))
            st.session_state.pop("_pending_slider_from_window", None)
    if bump_overview:
        st.session_state["_overview_chart_gen"] = (
            st.session_state.get("_overview_chart_gen", 0) + 1
        )


def _window_from_overview_event(event, total_s: float) -> tuple[float, float] | None:
    """Parse a horizontal box selection on the overview into (start, end)."""
    try:
        boxes = event.selection.box  # type: ignore[attr-defined]
    except Exception:
        boxes = []
    if not boxes:
        return None
    xs = boxes[0].get("x") or []
    if len(xs) < 2:
        return None
    a, b = float(min(xs)), float(max(xs))
    a = max(0.0, min(a, float(total_s)))
    b = max(0.0, min(b, float(total_s)))
    if b - a < 0.5:
        b = min(float(total_s), a + 0.5)
    return float(a), float(b)


def _sync_slider_into_window(total_s: float) -> tuple[float, float]:
    """Reconcile range bar and `window` before the overview slider is created."""
    start_s, end_s = _ensure_window(total_s)

    # Zoom/pan/etc. updated `window` after the slider was live — push into slider now.
    if st.session_state.pop("_pending_slider_from_window", False):
        st.session_state["overview_range_slider"] = (start_s, end_s)
        return start_s, end_s

    if "overview_range_slider" not in st.session_state:
        st.session_state["overview_range_slider"] = (start_s, end_s)
        return start_s, end_s

    sa, sb = st.session_state["overview_range_slider"]
    if (round(float(sa), 4), round(float(sb), 4)) != (round(start_s, 4), round(end_s, 4)):
        # User moved the range bar — apply to window (slider not live yet this run).
        _set_window(float(sa), float(sb), total_s, bump_overview=True, sync_slider=True)
        return st.session_state["window"]
    return start_s, end_s


def render_full_recording(
    signal,
    run_name: str,
    total_s: float,
    settings: Settings,
) -> tuple[float, float]:
    """Top strip: whole-recording waveform; box-drag or range bar sets the window."""
    st.session_state["_overview_slider_live"] = False
    start_s, end_s = _sync_slider_into_window(total_s)

    with st.container(border=True):
        section(
            f"Full recording · {fmt_time(total_s)}",
            "drag on the chart or move the range bar to set the viewing window",
        )
        st.caption(
            f"Viewing `{fmt_time(start_s)}` – `{fmt_time(end_s)}` · "
            "black frame = current range · chart box-drag or range bar below"
        )

        if settings.channels:
            t, s1, s2 = signal_io.load_window(
                signal, 0.0, total_s,
                fs_hz=settings.fs_hz,
                max_points=min(4_000, settings.max_points),
                method="stride",
            )
            detection = get_detections().get(run_name)
            fig = plots.overview_figure(
                t, s1, s2,
                channels=settings.channels,
                window=(start_s, end_s),
                events=detection["events"] if detection else None,
                height=180,
            )
            gen = st.session_state.get("_overview_chart_gen", 0)
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"overview_chart_{gen}",
                on_select="rerun",
                selection_mode="box",
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                },
            )
            picked = _window_from_overview_event(event, total_s)
            if picked is not None:
                sig = (round(picked[0], 4), round(picked[1], 4))
                prev = st.session_state.get("_overview_sel_sig")
                if sig != prev:
                    st.session_state["_overview_sel_sig"] = sig
                    _set_window(picked[0], picked[1], total_s, bump_overview=True)
                    st.rerun()

            step = max(float(total_s) / 2000.0, 0.01)
            st.slider(
                "Viewing range (s)",
                min_value=0.0,
                max_value=float(total_s),
                step=step,
                key="overview_range_slider",
                format="%.2f",
                help="Drag either handle to adjust the black frame (same as box-select on the chart).",
            )
            # After this, writing overview_range_slider in the same run is illegal.
            st.session_state["_overview_slider_live"] = True
        else:
            st.caption("Select at least one channel to show the overview.")

        return st.session_state["window"]


def _adjust_window(total_s: float, factor: float) -> None:
    """Zoom the detail window around its center (factor < 1 zooms in).

    When the window hits an edge, expand/shrink the free side so Zoom out
    always widens the visible range (trackpad scroll alone is unreliable).
    """
    start, end = _ensure_window(total_s)
    center = 0.5 * (start + end)
    width = max(0.5, min((end - start) * factor, float(total_s)))
    new_start = center - 0.5 * width
    new_end = center + 0.5 * width
    if new_start < 0.0:
        new_end = min(float(total_s), new_end - new_start)
        new_start = 0.0
    if new_end > total_s:
        new_start = max(0.0, new_start - (new_end - total_s))
        new_end = float(total_s)
    _set_window(new_start, new_end, total_s, bump_overview=True)


def _nudge_window(total_s: float, direction: float) -> None:
    """Pan the detail window left (−1) or right (+1) by 50% of its width."""
    start, end = _ensure_window(total_s)
    width = end - start
    shift = direction * 0.5 * width
    new_start = start + shift
    new_end = end + shift
    if new_start < 0:
        new_end -= new_start
        new_start = 0.0
    if new_end > total_s:
        new_start -= new_end - total_s
        new_end = total_s
    _set_window(max(0.0, new_start), new_end, total_s, bump_overview=True)


def render_window(
    signal,
    run_name: str,
    start_s: float,
    end_s: float,
    settings: Settings,
    total_s: float,
) -> None:
    """Middle: zoomed detail of the selected window."""
    with st.container(border=True):
        event_id = st.session_state.get("selected_event")
        title = f"Event {event_id}" if event_id else "Detail window"
        st.markdown(
            f"**{title}** &nbsp; `{fmt_time(start_s)} – {fmt_time(end_s)}`"
        )

        # Explicit zoom / pan — Streamlit trackpad scrollZoom is unreliable
        z1, z2, z3, z4, z5, z6 = st.columns(6)
        if z1.button("➖ Zoom out", use_container_width=True, key="zoom_out"):
            _adjust_window(total_s, factor=2.5)
            st.rerun()
        if z2.button("➕ Zoom in", use_container_width=True, key="zoom_in"):
            _adjust_window(total_s, factor=0.5)
            st.rerun()
        if z3.button("◀ Pan left", use_container_width=True, key="pan_left"):
            _nudge_window(total_s, direction=-1.0)
            st.rerun()
        if z4.button("▶ Pan right", use_container_width=True, key="pan_right"):
            _nudge_window(total_s, direction=1.0)
            st.rerun()
        if z5.button("↺ Default", use_container_width=True, key="zoom_reset"):
            _set_window(0.0, min(total_s, DEFAULT_WINDOW_S), total_s, bump_overview=True)
            st.rerun()
        if z6.button("⛶ Fit all", use_container_width=True, key="zoom_fit_all"):
            _set_window(0.0, float(total_s), total_s, bump_overview=True)
            st.rerun()

        if not settings.channels:
            st.warning("Select at least one channel.")
            return

        # Re-read window after possible button updates (same run; buttons call rerun)
        start_s, end_s = st.session_state["window"]

        t, s1, s2 = signal_io.load_window(
            signal, start_s, end_s,
            fs_hz=settings.fs_hz,
            max_points=settings.max_points,
            method=settings.downsample_method,
        )
        detection = get_detections().get(run_name)
        fig = plots.signal_figure(
            t, s1, s2,
            channels=settings.channels,
            start_s=start_s,
            end_s=end_s,
            events=detection["events"] if detection else None,
            show_rejected=settings.show_rejected,
            height=420,
        )
        # Pan with drag is easier than box-zoom for trackpads
        fig.update_layout(dragmode="pan")
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="detail_chart",
            config={
                **plots.PLOTLY_ZOOM_CONFIG,
                "scrollZoom": True,
                "modeBarButtonsToAdd": ["zoom2d", "pan2d", "autoScale2d", "resetScale2d"],
            },
        )
        st.caption(
            "Zoom out / Fit all change the black frame on the top graph. "
            "Drag the chart to pan; use toolbar 🔍 for box-zoom."
        )

        # Optional: peak stem overview kept out of the main 3-stack
        if detection is not None:
            with st.expander("Peak stems (optional)", expanded=False):
                proc2 = detection.get("proc2") or {}
                stem = plots.peaks_overview_figure(
                    events=detection["events"],
                    corrected_t=proc2.get("t"),
                    corrected=proc2.get("corrected"),
                    threshold=(detection.get("noise") or {}).get("threshold"),
                    show_rejected=settings.show_rejected,
                )
                st.plotly_chart(
                    stem,
                    use_container_width=True,
                    key="peaks_stem_chart",
                    config=plots.PLOTLY_ZOOM_CONFIG,
                )


# ---------------------------------------------------------------------------
# Right column
# ---------------------------------------------------------------------------

def render_selected_event(run_name: str) -> None:
    event_id = st.session_state.get("selected_event")
    event = dict(get_events(run_name)).get(event_id)

    if event is None:
        st.markdown(
            '<div class="sv-card"><div class="sv-card-label">Selected event</div>'
            '<div class="sv-card-note">None yet. Run <b>Detect peaks</b>, then pick an event below.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""<div class="sv-card">
<div class="sv-card-label">Selected event · {html.escape(event_id)}</div>
<div class="sv-card-value">{html.escape(str(event['class']))}</div>
<div class="sv-card-note">amplitude {event['amplitude']:.1f} · {html.escape(event['polarity'])}
· at {fmt_time(event['t_peak'])}</div>
<div class="sv-card-note">estimated size: — mm (TODO: calibration model)</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_events(run_name: str, total_s: float) -> None:
    events = get_events(run_name)
    section(f"Events · {len(events)}", "largest first")

    detection = get_detections().get(run_name)
    if detection is None:
        st.caption("Run **Detect peaks** to list events.")
        return

    largest_first = sorted(events, key=lambda item: item[1]["amplitude"], reverse=True)
    selected = st.session_state.get("selected_event")

    for event_id, event in largest_first[:MAX_EVENTS_LISTED]:
        label = f"{event_id} · {fmt_time(event['t_peak'])} · {event['class']} · amp {event['amplitude']:.1f}"
        with st.expander(label, expanded=event_id == selected):
            st.caption(f"polarity: {event['polarity']}")
            st.button(
                "Show in plot",
                key=f"show_{event_id}",
                on_click=select_event,
                args=(event_id, event["t_peak"], total_s),
                use_container_width=True,
            )

    hidden = len(largest_first) - MAX_EVENTS_LISTED
    if hidden > 0:
        st.caption(f"+ {hidden} more events")

    with st.expander("Detection summary"):
        st.metric("Detected peaks", detection["n_valid"])
        st.metric("Clusters", len(detection["clusters"]))
        st.dataframe(pd.DataFrame(detection["events_table"]), use_container_width=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Marginum Signal Viewer", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    header_left, header_right = st.columns([1, 2])
    header_left.markdown('<div class="sv-title">Signal Viewer</div>', unsafe_allow_html=True)
    file_info = header_right.empty()  # filled in once we know the run

    left, center, right = st.columns([1.15, 3.4, 1.6], gap="medium")

    with left:
        settings = render_left_panel()

    run_name = st.session_state.get("active_run")
    if run_name not in get_runs():
        center.info("Drop a Marginum CSV into **Import**, or load one from a local path.")
        return

    signal = signal_io.open_signal(Path(get_runs()[run_name]["npy"]))
    total_s = signal_io.duration_s(signal.shape[0], settings.fs_hz)

    with center:
        if settings.detect_clicked:
            run_detection(run_name, settings)
        # Overview (top) → detail zoom (middle).
        start_s, end_s = render_full_recording(signal, run_name, total_s, settings)
        render_window(signal, run_name, start_s, end_s, settings, total_s=total_s)

    with right:
        render_selected_event(run_name)
        render_events(run_name, total_s)

    source_name = Path(get_runs()[run_name]["source"]).name
    file_info.markdown(
        f'<div class="sv-chip">{html.escape(source_name)} · {fmt_time(total_s)}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
