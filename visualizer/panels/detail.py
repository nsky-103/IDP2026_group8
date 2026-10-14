"""Centre, middle: zoomed detail of the viewing window, with zoom / pan buttons."""

from __future__ import annotations

import streamlit as st

import plots
import signal_io
import state
import viewport
from panels.controls import Settings
from style import fmt_time


def render(recording: signal_io.Recording, run: str, total_s: float, settings: Settings) -> None:
    with st.container(border=True):
        start_s, end_s = state.window(total_s)
        event_id = st.session_state.get("selected_event")
        title = f"Event {event_id}" if event_id else "Detail window"
        st.markdown(f"**{title}** &nbsp; `{fmt_time(start_s)} – {fmt_time(end_s)}`")

        _zoom_buttons(total_s)
        if not settings.channels:
            st.warning("Select at least one channel.")
            return

        try:  # exact samples, read from the CSV on demand (core.fast_io.read_window)
            t, s1, s2 = recording.load_window(
                start_s, end_s, fs_hz=settings.fs_hz,
                max_points=settings.max_points, method=settings.downsample_method,
            )
        except Exception as exc:
            st.error(f"Could not read detail window: {exc}")
            return
        detection = state.detection(run)
        fig = plots.signal_figure(
            t, s1, s2, channels=settings.channels, start_s=start_s, end_s=end_s,
            events=detection.events if detection else None,
            show_rejected=settings.show_rejected, height=420,
        )
        fig.update_layout(dragmode="pan")  # pan is easier than box-zoom on trackpads
        st.plotly_chart(fig, width="stretch", key="detail_chart", config={
            **plots.PLOTLY_ZOOM_CONFIG,
            "modeBarButtonsToAdd": ["zoom2d", "pan2d", "autoScale2d", "resetScale2d"],
        })
        st.caption("Detail uses on-demand CSV reads (read_window), not a full-rate NPY. "
                   "Zoom out / Fit all update the black frame on the overview.")

        if detection is not None:
            with st.expander("Peak stems (optional)", expanded=False):
                st.plotly_chart(
                    plots.peaks_overview_figure(detection.events, trace=detection.trace,
                                                threshold=detection.threshold,
                                                show_rejected=settings.show_rejected),
                    width="stretch", key="peaks_stem_chart", config=plots.PLOTLY_ZOOM_CONFIG,
                )


def _zoom_buttons(total_s: float) -> None:
    """Explicit zoom / pan (Streamlit's trackpad scroll-zoom is unreliable)."""
    w = state.window(total_s)
    actions = {
        "zoom_out": ("➖ Zoom out", lambda: viewport.zoom(w, total_s, 2.5)),
        "zoom_in": ("➕ Zoom in", lambda: viewport.zoom(w, total_s, 0.5)),
        "pan_left": ("◀ Pan left", lambda: viewport.pan(w, total_s, -1.0)),
        "pan_right": ("▶ Pan right", lambda: viewport.pan(w, total_s, 1.0)),
        "zoom_reset": ("↺ Default", lambda: (0.0, min(total_s, state.DEFAULT_WINDOW_S))),
        "zoom_fit_all": ("⛶ Fit all", lambda: (0.0, float(total_s))),
    }
    for col, (key, (label, new_window)) in zip(st.columns(len(actions)), actions.items()):
        if col.button(label, width="stretch", key=key):
            state.set_window(*new_window(), total_s)
            st.rerun()
