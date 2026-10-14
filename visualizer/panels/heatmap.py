"""Centre, bottom: whole recording folded into rows; click a cell to view that time."""

from __future__ import annotations

import functools

import numpy as np
import streamlit as st

import plots
import signal_io
import state
import viewport
from style import section

ROW_CHOICES = {"10 s": 10.0, "30 s": 30.0, "1 min": 60.0, "2 min": 120.0, "5 min": 300.0}
EVENTS_METRIC = "Detected events (count)"
METRICS = {  # name: (kind, channel)
    "Sensor 2 activity (peak-to-peak)": ("ptp", 1),
    "Sensor 1 activity (peak-to-peak)": ("ptp", 0),
    EVENTS_METRIC: ("events", None),
}
MAX_HEIGHT = 480  # px; taller heatmaps scroll


def _event_counts(run: str, bin_s: float, n_bins: int) -> np.ndarray:
    bins = (state.accepted_events(run)["t_peak"].to_numpy(dtype=float) // bin_s).astype(int)
    bins = bins[(bins >= 0) & (bins < n_bins)]
    return np.bincount(bins, minlength=n_bins).astype(float)


def render(recording: signal_io.Recording, run: str, total_s: float, settings) -> None:
    """Peak-to-peak comes from the 1 kHz summary's min/max envelope (not full-rate samples)."""
    with st.container(border=True, key="heatmap_box"):
        section("Activity heatmap")
        if recording.phase != "ready" or recording.summary is None:
            st.caption("Heatmap needs the 1 kHz summary (still building…).")
            return

        row_s = settings.heatmap_row_s
        bin_s = row_s / settings.heatmap_cells
        n_bins = max(1, int(np.ceil(total_s / bin_s - 1e-9)))
        kind, channel = METRICS[settings.heatmap_metric]
        if kind == "events":
            values = _event_counts(run, bin_s, n_bins)
        else:
            values = recording.binned_peak_to_peak(bin_s, channel=channel)

        fig = plots.time_heatmap_figure(values, bin_s=bin_s, row_s=row_s, total_s=total_s,
                                        window=st.session_state.get("window"))
        key = state.chart_key("heatmap_chart", "_heatmap_chart_gen")
        box = {"height": MAX_HEIGHT} if fig.layout.height > MAX_HEIGHT else {}
        with st.container(border=False, **box):
            st.plotly_chart(
                fig, use_container_width=True, key=key,
                # A callback (not "rerun") runs before the overview slider is drawn,
                # so the window can still be changed.
                on_select=functools.partial(_on_click, key, row_s, bin_s, total_s),
                selection_mode="points", config={"displayModeBar": False},
            )


def _on_click(key: str, row_s: float, bin_s: float, total_s: float) -> None:
    """Centre the viewing window (same width) on the clicked cell."""
    try:
        points = st.session_state[key].selection.points
    except (KeyError, AttributeError):
        return
    if not points:
        return
    centre = float(points[0]["y"]) * row_s + float(points[0]["x"])  # y = row, x = s into row
    w0, w1 = state.window(total_s)
    state.set_window(*viewport.centred(centre, max(w1 - w0, bin_s), total_s), total_s)
    state.bump("_heatmap_chart_gen")  # new key clears the selection so the cell can be re-clicked
