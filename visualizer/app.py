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
    signal_io.py     CSV → NPY conversion + loading a time window
    convert_jobs.py  background CSV → NPY worker (keeps the UI responsive)
    lttb.py          LTTB downsampling for plots
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
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for `core` (used by detectors.py)

import signal_io  # noqa: E402
import state  # noqa: E402
import style  # noqa: E402
from panels import controls, data_import, detail, events, heatmap, overview  # noqa: E402


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

    signal = signal_io.open_signal(Path(state.runs()[run]["npy"]))
    total_s = signal_io.duration_s(signal.shape[0], settings.fs_hz)
    source_name = Path(state.runs()[run]["source"]).name
    header_right.markdown(
        f'<div class="sv-chip">{html.escape(source_name)} · {style.fmt_time(total_s)}</div>',
        unsafe_allow_html=True,
    )

    with center:
        if settings.detect_clicked:
            controls.run_detection(run, settings)
        overview.render(signal, run, total_s, settings)
        detail.render(signal, run, total_s, settings)
        heatmap.render(run, total_s, settings)

    with right:
        events.render(run, total_s)


if __name__ == "__main__":
    main()
