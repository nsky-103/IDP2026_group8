"""
Marginum signal visualizer (Streamlit app).

Run with:  streamlit run visualizer/app.py

Page layout                                  file
-----------
    header   title · file info               app.py
    left     import · runs                   panels/data_import.py
             channels · detection · display  panels/controls.py
    center   full-recording overview         panels/overview.py
             detail zoom                     panels/detail.py
             activity heatmap                panels/heatmap.py
    right    selected event · events list    panels/events.py

Other files
-----------
    state.py         everything kept between clicks (st.session_state)
    viewport.py      viewing-window math (zoom, pan, clamp)
    style.py         CSS, section headings, cards, time format
    plots.py         Plotly figures (no Streamlit)
    signal_io.py     Recording: row index + 1 kHz summary + exact windows (core.fast_io)
    load_jobs.py     background coarse-first load: index → preview → 1 kHz summary
    lttb.py          LTTB downsampling for the detail plot
    detectors.py     detection algorithms and their result format (no UI)

How to add a feature
--------------------
    New algorithm?  add a function + a Detector entry in detectors.py.
                    Its parameters appear in the left panel automatically.
    New chart?      add a function in plots.py that returns a go.Figure,
                    and show it from a panel (or a new panels/xxx.py).
    New setting?    add a widget in panels/controls.py and a field in Settings.
"""

from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st

import load_jobs
import signal_io
import state
import style
from panels import controls, data_import, detail, events, heatmap, overview

signal_io._ensure_core_on_path()  # `core` lives next to this folder


def main() -> None:
    st.set_page_config(page_title="Marginum Signal Viewer", layout="wide")
    style.inject_css()

    header_left, header_right = st.columns([1, 2])
    header_left.markdown('<div class="sv-title">Signal Viewer</div>', unsafe_allow_html=True)
    left, center, right = st.columns([1.15, 3.4, 1.6], gap="medium")

    with left:
        data_import.render()
        settings = controls.render()

    run = state.active_run()
    if run is None:
        center.info("Drop a Marginum CSV into **Import**, or load one from a local path.")
        return

    csv_path = Path(state.runs()[run]["csv"])
    recording = load_jobs.get_recording(csv_path)
    if recording is None:
        center.info("Loading recording…")
        time.sleep(0.3)
        st.rerun()
    state.runs()[run]["phase"] = recording.phase  # keep in sync with the live Recording
    total_s = recording.duration_s(settings.fs_hz)
    header_right.markdown(
        f'<div class="sv-chip">{html.escape(csv_path.name)} · {style.fmt_time(total_s)}'
        f' · {recording.phase}</div>',
        unsafe_allow_html=True,
    )

    with center:
        if settings.detect_clicked:
            if recording.phase != "ready":
                st.warning("Wait until the 1 kHz summary is ready before detecting.")
            else:
                controls.run_detection(run, recording, settings)
        overview.render(recording, run, total_s, settings)
        detail.render(recording, run, total_s, settings)
        heatmap.render(recording, run, total_s, settings)

    with right:
        events.render(run, total_s)


if __name__ == "__main__":
    main()
