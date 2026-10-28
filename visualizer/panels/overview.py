"""Centre, top: whole recording; drag the frame or the range slider to set the viewing window."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import streamlit as st

import plots
import signal_io
import state
import viewport
from panels.controls import Settings
from style import fmt_time, section

OVERVIEW_POINTS = 4_000
OVERVIEW_HEIGHT = 120  # px, including the time axis and the start/end labels
CHART_KEY = "overview_window"

# Canvas strip with a movable, resizable frame; sends the new window when the drag ends.
_overview_window = st.components.v2.component(
    "overview_window", js=(Path(__file__).with_name("overview_window.js")).read_text(encoding="utf-8"),
)


def render(recording: signal_io.Recording, run: str, total_s: float, settings: Settings) -> None:
    start_s, end_s = state.sync_slider(total_s)

    with st.container(border=True):
        # note = ("approximate preview — 1 kHz summary still building" if recording.phase != "ready"
        #         else "1 kHz summary · drag the frame or its sides, or move the range bar")
        # section(f"Full recording · {fmt_time(total_s)}", note)
        # st.caption(f"Viewing `{fmt_time(start_s)}` – `{fmt_time(end_s)}` · "
        #            "frame = current range")
        if not settings.channels:
            st.caption("Select at least one channel to show the overview.")
            return

        t, s1, s2, s1_min, s1_max, s2_min, s2_max, approximate = recording.overview_series(
            max_points=min(OVERVIEW_POINTS, settings.max_points))
        detection = state.detection(run)
        series = {plots.SENSOR1: (s1, s1_min, s1_max), plots.SENSOR2: (s2, s2_min, s2_max)}
        events = plots.overview_event_times(detection.events if detection else None)
        _overview_window(
            key=CHART_KEY,
            data={
                "t": _list(t),
                "series": [
                    {"color": plots.SENSOR_COLORS[name], "y": _list(y), "lo": _list(lo), "hi": _list(hi)}
                    for name, (y, lo, hi) in series.items() if name in settings.channels
                ],
                "events": _list(events),
                "window": [start_s, end_s],
                "total": float(total_s),
                "min_width": viewport.MIN_WIDTH_S,
                "approximate": bool(approximate),
                "height": OVERVIEW_HEIGHT,
            },
            # A callback (not a rerun) runs before the size slider is drawn,
            # so the window can still be changed.
            on_window_change=functools.partial(_on_drag, total_s),
        )

        # Label beside the slider (not above it) keeps the overview box short.
        row = st.container(horizontal=True, vertical_alignment="center")
        row.markdown("Window size (s)", width="content")
        row.slider(
            "Window size (s)", min_value=viewport.MIN_WIDTH_S, max_value=float(total_s),
            step=max(float(total_s) / 2000.0, 0.01), key=state.SLIDER_KEY, format="%.2f",
            label_visibility="collapsed",
            # help="Width of the frame, kept centred. Drag the frame on the chart to move it.",
        )
        state.slider_drawn()


def _list(values) -> list[float] | None:
    """Array -> JSON list (rounded to keep the message small); None stays None."""
    if values is None:
        return None
    return np.round(np.asarray(values, dtype=float), 5).tolist()


def _on_drag(total_s: float) -> None:
    """The frame was moved or resized on the chart: that becomes the viewing window."""
    try:
        picked = st.session_state[CHART_KEY]["window"]
    except (KeyError, TypeError):
        return
    if picked and len(picked) == 2:
        state.set_window(float(picked[0]), float(picked[1]), total_s)
