"""Centre, top: whole recording; box-drag or the range slider sets the viewing window."""

from __future__ import annotations

import streamlit as st

import plots
import signal_io
import state
from panels.controls import Settings
from style import fmt_time, section

OVERVIEW_POINTS = 4_000


def render(signal, run: str, total_s: float, settings: Settings) -> None:
    start_s, end_s = state.sync_slider(total_s)

    with st.container(border=True):
        section(f"Full recording · {fmt_time(total_s)}",
                "drag on the chart or move the range bar to set the viewing window")
        st.caption(f"Viewing `{fmt_time(start_s)}` – `{fmt_time(end_s)}` · "
                   "black frame = current range · chart box-drag or range bar below")
        if not settings.channels:
            st.caption("Select at least one channel to show the overview.")
            return

        t, s1, s2 = signal_io.load_window(
            signal, 0.0, total_s, fs_hz=settings.fs_hz,
            max_points=min(OVERVIEW_POINTS, settings.max_points), method="stride",
        )
        detection = state.detection(run)
        fig = plots.overview_figure(
            t, s1, s2, channels=settings.channels, window=(start_s, end_s),
            events=detection.events if detection else None, height=180,
        )
        chart = st.plotly_chart(
            fig, use_container_width=True,
            key=state.chart_key("overview_chart", "_overview_chart_gen"),
            on_select="rerun", selection_mode="box",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )
        _apply_box_selection(chart, total_s)

        st.slider(
            "Viewing range (s)", min_value=0.0, max_value=float(total_s),
            step=max(float(total_s) / 2000.0, 0.01), key=state.SLIDER_KEY, format="%.2f",
            help="Drag either handle to adjust the black frame (same as box-select on the chart).",
        )
        state.slider_drawn()


def _apply_box_selection(chart, total_s: float) -> None:
    """A new horizontal box selection on the chart becomes the viewing window."""
    try:
        boxes = chart.selection.box
    except AttributeError:
        return
    xs = (boxes[0].get("x") or []) if boxes else []
    if len(xs) < 2:
        return
    picked = (round(float(min(xs)), 4), round(float(max(xs)), 4))
    if picked != st.session_state.get("_overview_sel_sig"):
        st.session_state["_overview_sel_sig"] = picked
        state.set_window(*picked, total_s)
        st.rerun()
