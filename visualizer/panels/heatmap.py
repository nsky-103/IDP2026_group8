"""Centre, bottom: whole recording folded into rows; click a cell to view that time."""

from __future__ import annotations

import functools
from pathlib import Path

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


@st.cache_data(show_spinner="Computing heatmap…", max_entries=16)
def _peak_to_peak(npy_path: str, mtime: float, bin_s: float, fs_hz: float, channel: int):
    """Full-resolution binned stats; cached per file/settings (mtime busts stale caches)."""
    signal = signal_io.open_signal(Path(npy_path))
    return signal_io.binned_peak_to_peak(signal, bin_s, fs_hz=fs_hz, channel=channel)


def _event_counts(run: str, bin_s: float, n_bins: int) -> np.ndarray:
    bins = (state.accepted_events(run)["t_peak"].to_numpy(dtype=float) // bin_s).astype(int)
    bins = bins[(bins >= 0) & (bins < n_bins)]
    return np.bincount(bins, minlength=n_bins).astype(float)


def render(run: str, total_s: float, settings) -> None:
    with st.container(border=True, key="heatmap_box"):
        section("Activity heatmap")

        row_s = settings.heatmap_row_s
        bin_s = row_s / settings.heatmap_cells
        n_bins = max(1, int(np.ceil(total_s / bin_s - 1e-9)))
        kind, channel = METRICS[settings.heatmap_metric]
        if kind == "events":
            values = _event_counts(run, bin_s, n_bins)
        else:
            npy = state.runs()[run]["npy"]
            values = _peak_to_peak(npy, Path(npy).stat().st_mtime, bin_s, settings.fs_hz, channel)

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
