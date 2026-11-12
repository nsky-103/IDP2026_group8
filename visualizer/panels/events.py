"""Right column: selected event card, event list (largest first) and detection summary."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import signal_io
import state
from style import card, fmt_time, section

MAX_EVENTS_LISTED = 25
MAX_TABLE_ROWS = 500
EVENT_ZOOM_S = 5.0  # "Show in plot" shows ± this many seconds around the event


def _amount_in_window(events: pd.DataFrame, window: tuple[float, float] | None) -> dict | None:
    """Relative amount for pieces that start inside the viewing window (issue #38)."""
    if window is None or events is None or events.empty:
        return None
    if "area" not in events.columns or "start_s" not in events.columns:
        return None
    signal_io._ensure_core_on_path()
    from core import detection as detection_core

    t0, t1 = float(window[0]), float(window[1])
    return detection_core.amount(events, t0, t1)


def render(run: str, total_s: float) -> None:
    render_selected(run)
    render_list(run, total_s)


def render_selected(run: str) -> None:
    events = state.accepted_events(run)
    event_id = st.session_state.get("selected_event")
    if event_id not in events.index:
        card("Selected event", notes=["None yet. Run Detect peaks, then pick an event below."])
        return

    e = events.loc[event_id]
    card(
        f"Selected event · {event_id}",
        value=str(e["label"]),
        notes=[
            f"amplitude {e['amplitude']:.1f} · {e['polarity']} · at {fmt_time(e['t_peak'])}",
        ],
    )


def render_list(run: str, total_s: float) -> None:
    events = state.accepted_events(run)
    section(f"Events · {len(events)}", "largest first")
    detection = state.detection(run)
    if detection is None:
        st.caption("Run **Detect peaks** to list events (after the 1 kHz summary is ready).")
        return

    window = st.session_state.get("window")
    amt_view = _amount_in_window(detection.events, window)
    if amt_view is not None:
        st.metric("Amount in view: total area (counts·ms)", round(amt_view["area"]))
        st.caption(
            "Relative fluorescence·time over valid pieces that start in the current window "
            "(not ml/g — concentration-dependent)."
        )

    selected = st.session_state.get("selected_event")
    largest = events.sort_values("amplitude", ascending=False)
    for event_id, e in largest.head(MAX_EVENTS_LISTED).iterrows():
        label = f"{event_id} · {fmt_time(e['t_peak'])} · {e['label']} · amp {e['amplitude']:.1f}"
        with st.expander(label, expanded=event_id == selected):
            st.caption(f"polarity: {e['polarity']}")
            st.button("Show in plot", key=f"show_{event_id}", width="stretch",
                      on_click=state.select_event,
                      args=(event_id, float(e["t_peak"]), total_s, EVENT_ZOOM_S))
    if len(largest) > MAX_EVENTS_LISTED:
        st.caption(f"+ {len(largest) - MAX_EVENTS_LISTED} more events")

    with st.expander("Detection summary"):
        for name, value in detection.summary.items():
            st.metric(name, value)
        if detection.timings:
            st.json(detection.timings)
        table = detection.events
        if "size_mm" in table.columns:
            table = table.drop(columns=["size_mm"])
        if len(table) > MAX_TABLE_ROWS:  # a huge table freezes the page
            st.caption(f"Showing {MAX_TABLE_ROWS} / {len(table)} rows (largest amplitudes).")
            table = table.nlargest(MAX_TABLE_ROWS, "amplitude")
        st.dataframe(table, width="stretch")
